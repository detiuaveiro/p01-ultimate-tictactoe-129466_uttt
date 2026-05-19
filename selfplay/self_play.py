"""
Self-play game generation for AlphaZero-lite training.

Provides :class:`TrainingExample` (a single state->policy->outcome record)
and :func:`generate_self_play_games` which uses an MCTS guided by a neural
network to produce a batch of training examples from full self-play games.
"""

import logging
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch

from engine.game_state import UTTTState
from engine.mcts_core import MCTS
from engine.nn_mcts_bridge import create_nn_mcts_functions
from engine.policy_value_network import PolicyValueNetwork, encode_state
from selfplay.config import SelfPlayConfig
from utils.progress import WorkerGameBar

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TrainingExample
# ---------------------------------------------------------------------------


@dataclass
class TrainingExample:
    """A single training example generated during self-play.

    Attributes:
        state_features: Encoded board state tensor of shape (3, 9, 9).
        search_policy: MCTS search policy as an 81-dim probability vector
            (probabilities sum to 1 over the entire board, with zeros at
            illegal positions).
        outcome: The game outcome from the current player's perspective:
            +1 (win), -1 (loss), or 0 (draw).
    """

    state_features: np.ndarray  # shape (3, 9, 9)
    search_policy: np.ndarray  # shape (81,)
    outcome: float  # +1, -1, or 0


# ---------------------------------------------------------------------------
# Temperature schedule helper
# ---------------------------------------------------------------------------


def get_temperature(
    move_number: int,
    schedule: List[Tuple[int, float]],
) -> float:
    """Return the temperature for a given move number using linear
    interpolation between schedule breakpoints.

    Args:
        move_number: The current move number (0-indexed).
        schedule: List of ``(move_count, temperature)`` pairs sorted by
            move_count.  The first pair should have move_count = 0.

    Returns:
        The temperature at the given move number.
    """
    if not schedule:
        return 1.0

    # If past the last breakpoint, use the last temperature value
    last_move, last_temp = schedule[-1]
    if move_number >= last_move:
        return last_temp

    # Find the interval containing move_number and interpolate
    for i in range(len(schedule) - 1):
        m1, t1 = schedule[i]
        m2, t2 = schedule[i + 1]
        if m1 <= move_number < m2:
            if m2 == m1:
                return t1
            fraction = (move_number - m1) / (m2 - m1)
            return t1 + fraction * (t2 - t1)

    return schedule[0][1]


# ---------------------------------------------------------------------------
# Search-policy computation
# ---------------------------------------------------------------------------


def compute_search_policy(
    root_visits: Dict[Tuple[int, int], float],
    temperature: float,
    valid_actions: List[List[int]],
) -> np.ndarray:
    """Compute an 81-dimensional search-policy vector from MCTS root visit
    counts using the formula::

        π(a) = N(a)^(1/τ) / Σ_b N(b)^(1/τ)

    where N(a) is the visit count for action a and τ is the temperature.
    Illegal actions receive probability 0.

    Args:
        root_visits: Mapping from ``(x, y)`` action to visit count (float).
        temperature: Temperature parameter (τ).  If 0, a one-hot vector
            is returned for the most-visited action.
        valid_actions: List of ``[x, y]`` legal actions.

    Returns:
        A float32 numpy array of shape ``(81,)`` whose entries sum to 1.
    """
    policy = np.zeros(81, dtype=np.float32)

    if not root_visits:
        # Fallback: uniform over valid actions
        for x, y in valid_actions:
            idx = y * 9 + x
            policy[idx] = 1.0
        policy /= policy.sum()
        return policy

    if temperature == 0.0:
        # Deterministic: pick the most-visited action
        best_action = max(root_visits, key=root_visits.__getitem__)  # type: ignore[arg-type]
        idx = best_action[1] * 9 + best_action[0]
        policy[idx] = 1.0
        return policy

    inv_temp = 1.0 / temperature
    total = 0.0
    for (x, y), visits in root_visits.items():
        idx = y * 9 + x
        weight = visits ** inv_temp
        policy[idx] = weight
        total += weight

    if total > 0:
        policy /= total
    else:
        # Fallback uniform
        for x, y in valid_actions:
            idx = y * 9 + x
            policy[idx] = 1.0
        policy /= policy.sum()

    return policy


# ---------------------------------------------------------------------------
# Action sampling
# ---------------------------------------------------------------------------


def sample_action(
    search_policy: np.ndarray,
    valid_actions: List[List[int]],
    temperature: float,
    rng: random.Random,
) -> List[int]:
    """Sample an action from the search-policy distribution.

    Args:
        search_policy: 81-dim probability vector.
        valid_actions: List of ``[x, y]`` legal actions.
        temperature: Temperature used for sampling (τ).  If 0, the most
            probable action is chosen deterministically.
        rng: Seeded ``random.Random`` instance.

    Returns:
        The chosen action ``[x, y]``.
    """
    if temperature == 0.0:
        # Deterministic: argmax
        best_idx = int(np.argmax(search_policy))
        return [best_idx % 9, best_idx // 9]

    # Build list of (action, probability) for valid actions
    actions_and_probs = []
    for x, y in valid_actions:
        idx = y * 9 + x
        prob = float(search_policy[idx])
        if prob > 0:
            actions_and_probs.append(([x, y], prob))

    if not actions_and_probs:
        # Fallback: first valid action
        return valid_actions[0]

    # Weighted random choice
    actions, probs = zip(*actions_and_probs)
    total = sum(probs)
    normalized = [p / total for p in probs]
    chosen_idx = rng.choices(range(len(actions)), weights=normalized, k=1)[0]
    return list(actions[chosen_idx])


# ---------------------------------------------------------------------------
# Self-play game generation
# ---------------------------------------------------------------------------


def generate_self_play_games(
    network: PolicyValueNetwork,
    config: SelfPlayConfig,
    rng: random.Random,
    device: str = "cpu",
    worker_index: int = 0,
) -> List[TrainingExample]:
    """Play a batch of self-play games using *network* to guide MCTS and
    collect training examples.

    For each move in each game:
      1. Determine temperature from the schedule.
      2. Create an MCTS instance with NN-based prior_fn/value_fn (with
         Dirichlet noise for exploration).
      3. Run MCTS search.
      4. Compute the search policy from root visit counts.
      5. Record a :class:`TrainingExample` for the current state.
      6. Sample an action from the search policy and apply it.

    After each game, outcomes (+1 / -1 / 0) are assigned to all recorded
    examples based on which player won.

    Args:
        network: The current PolicyValueNetwork (will be set to eval mode).
        config: Self-play configuration.
        rng: Seeded ``random.Random`` instance for reproducibility.
        device: Device for network inference (``'cpu'`` or ``'cuda'``).
        worker_index: Worker index for tqdm display positioning.
            Worker i shows a move counter at terminal line (1 + i),
            displaying "⠋ Worker N: Game X/Y: Z moves [elapsed, rate]".
            Resets each game. Default 0.

    Returns:
        A list of :class:`TrainingExample` instances collected from all
        completed games.
    """
    network.eval()
    examples: List[TrainingExample] = []

    for game_idx in range(config.games_per_iteration):
        game_examples: List[TrainingExample] = []
        state = UTTTState()

        progress = WorkerGameBar(
            worker_index,
            config.games_per_iteration,
        )
        progress.start_game(game_idx)

        while not state.is_terminal():
            valid = state.get_valid_actions()
            if not valid:
                break

            move_number = state.move_count
            temperature = get_temperature(
                move_number, config.temperature_schedule
            )

            # Create MCTS with NN functions (Dirichlet noise for exploration)
            prior_fn, value_fn = create_nn_mcts_functions(
                network,
                device=device,
                add_dirichlet_noise=True,
                dirichlet_alpha=config.dirichlet_alpha,
                dirichlet_epsilon=config.dirichlet_epsilon,
                rng=rng,
            )
            mcts = MCTS(
                iterations=config.mcts_iterations,
                exploration_constant=config.c_puct,
                random_seed=rng.randint(0, 2**31 - 1),
                prior_fn=prior_fn,
                value_fn=value_fn,
                use_puct=True,
            )

            try:
                progress.set_status("searching...")
                mcts.search(state)
                progress.set_status("")
            except RuntimeError:
                progress.set_status("")
                # Fallback: pick a random valid action
                action = rng.choice(valid)
                _record_example(game_examples, state, _uniform_policy(valid), 0.0)
                state = state.apply_action(action[0], action[1])
                progress.on_move()
                continue

            # Compute search policy from root visit distribution
            root_visits = mcts.get_root_visit_distribution()
            search_policy = compute_search_policy(
                root_visits, temperature, valid
            )

            # Record training example BEFORE the move
            _record_example(game_examples, state, search_policy, 0.0)

            # Sample action from search policy
            action = sample_action(search_policy, valid, temperature, rng)

            # Apply action
            state = state.apply_action(action[0], action[1])

            progress.on_move()

        progress.close()

        # Game finished: assign outcomes
        winner = state.get_winner()
        _assign_outcomes(game_examples, winner)

        examples.extend(game_examples)

    return examples


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _record_example(
    examples: List[TrainingExample],
    state: UTTTState,
    search_policy: np.ndarray,
    outcome: float,
) -> None:
    """Encode *state* and append a TrainingExample to *examples*."""
    encoded = encode_state(state).numpy()  # (3, 9, 9)
    examples.append(
        TrainingExample(
            state_features=encoded,
            search_policy=search_policy.copy(),
            outcome=outcome,
        )
    )


def _assign_outcomes(
    examples: List[TrainingExample],
    winner: int,
) -> None:
    """Assign final outcomes to each example based on *winner*.

    Each example stores which player was to move at the time of recording
    (the ``current_player`` at that state).  We infer the player from the
    state_features encoding (channel 2 offset: +0.25 = P1, -0.25 = P2).

    For each example:
      - If the player won → outcome = +1
      - If the player lost → outcome = -1
      - If draw            → outcome =  0

    Args:
        examples: List of examples to update in place.
        winner: Game winner (1, 2, or 3 for draw).
    """
    if winner == 3:
        # Draw — all examples get 0
        for ex in examples:
            ex.outcome = 0.0
        return

    for ex in examples:
        # Determine which player was to move: P1 if offset > 0, else P2
        offset = ex.state_features[2].mean()  # rough but reliable
        current_player = 1 if offset >= 0 else 2

        if winner == current_player:
            ex.outcome = 1.0
        else:
            ex.outcome = -1.0


def _uniform_policy(valid_actions: List[List[int]]) -> np.ndarray:
    """Return a uniform policy over valid actions (fallback)."""
    policy = np.zeros(81, dtype=np.float32)
    for x, y in valid_actions:
        idx = y * 9 + x
        policy[idx] = 1.0
    policy /= policy.sum()
    return policy
