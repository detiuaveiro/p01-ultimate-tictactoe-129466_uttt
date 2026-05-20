"""
AlphaZero-style agent for Ultimate Tic-Tac-Toe.

Uses a neural network (PolicyValueNetwork) to guide MCTS search via PUCT.
Supports both server-based (async deliberate) and headless (deliberate_from_state)
operation, as well as temperature-based move selection.

The network is loaded lazily to support pickling for multiprocessing.
"""

import argparse
import asyncio
import logging
import math
import random
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from agents.base_agent import BaseUTTTAgent
from engine.game_state import UTTTState
from engine.mcts_core import MCTS
from engine.nn_mcts_bridge import create_nn_mcts_functions
from engine.policy_value_network import (
    PolicyValueNetwork,
    load_network,
)

# Module-level cache: maps (checkpoint_path, device) -> PolicyValueNetwork
# This avoids reloading the network from disk every time a new agent
# instance is created during tournament play.
_NETWORK_CACHE: Dict[Tuple[str, str], PolicyValueNetwork] = {}

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - ALPHAZERO - %(message)s"
)


class AlphaZeroUTTTAgent(BaseUTTTAgent):
    """
    An Ultimate Tic-Tac-Toe agent that uses MCTS guided by a neural network
    to select moves, following the AlphaZero paradigm.

    The network is loaded lazily (on first access) to avoid pickling issues
    when used with ``multiprocessing`` in the tournament runner.

    Attributes:
        mcts_iterations: Number of MCTS simulations per move.
        mcts_exploration_constant: PUCT exploration constant (C_puct).
        mcts_time_limit: Optional time limit per move (seconds).
        random_seed: Seed for deterministic MCTS.
        checkpoint_path: Path to the network checkpoint (``.pt`` file).
        temperature: Temperature for move selection.
            0 = deterministic (most-visited child).
            > 0 = sample from softmax(visits / temperature).
    """

    def __init__(
        self,
        server_uri: str = "ws://localhost:8765",
        mcts_iterations: int = 800,
        mcts_exploration_constant: float = 1.414,
        mcts_time_limit: Optional[float] = None,
        random_seed: Optional[int] = None,
        checkpoint_path: Optional[str] = None,
        temperature: float = 0.0,
        device: str = "cpu",
    ) -> None:
        """
        Initializes the AlphaZeroUTTTAgent.

        Args:
            server_uri: WebSocket server URI.
            mcts_iterations: Maximum MCTS iterations per move.
            mcts_exploration_constant: PUCT exploration constant.
            mcts_time_limit: Time limit per move (optional).
            random_seed: Random seed (optional).
            checkpoint_path: Path to the network ``.pt`` checkpoint.
            temperature: Temperature for move selection (0 = deterministic).
            device: Device for network inference (``'cpu'`` or ``'cuda'``).
        """
        super().__init__(server_uri)
        self.mcts_iterations = mcts_iterations
        self.mcts_exploration_constant = mcts_exploration_constant
        self.mcts_time_limit = mcts_time_limit
        self.random_seed = random_seed
        self.checkpoint_path = checkpoint_path
        self.temperature = temperature
        self.device = device

        # Lazy network init for pickling support
        self._network: Optional[PolicyValueNetwork] = None
        self._last_stats: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Lazy network initialisation
    # ------------------------------------------------------------------

    def _get_network(self) -> PolicyValueNetwork:
        """Return the neural network, loading it lazily if needed.

        Uses a module-level cache keyed by ``(checkpoint_path, device)``
        so that multiple agent instances created during a tournament
        reuse the same in-memory network.

        Returns:
            The PolicyValueNetwork instance (in eval mode) on ``self.device``.

        Raises:
            RuntimeError: If no checkpoint path was provided.
        """
        if self._network is None:
            if self.checkpoint_path is None:
                raise RuntimeError(
                    "No checkpoint path specified. "
                    "Provide checkpoint_path to load the network."
                )
            cache_key = (self.checkpoint_path, self.device)
            if cache_key in _NETWORK_CACHE:
                self._network = _NETWORK_CACHE[cache_key]
                logging.debug("Reusing cached network for %s", cache_key)
            else:
                self._network = load_network(
                    self.checkpoint_path, device=self.device
                )
                _NETWORK_CACHE[cache_key] = self._network
            self._network.eval()
        return self._network

    # ------------------------------------------------------------------
    # Bridge function factory
    # ------------------------------------------------------------------

    def _make_bridge_functions(self) -> Tuple[Any, Any]:
        """Create ``(prior_fn, value_fn)`` closures from the loaded network.

        Returns:
            Tuple of (prior_fn, value_fn) callables.
        """
        network = self._get_network()
        return create_nn_mcts_functions(
            network,
            add_dirichlet_noise=False,  # no exploration noise during play
        )

    # ------------------------------------------------------------------
    # Deliberation
    # ------------------------------------------------------------------

    async def deliberate(
        self,
        board: List[List[int]],
        macro_board: List[List[int]],
        active_macro: Optional[List[int]],
        valid_actions: List[List[int]],
    ) -> Optional[Union[List[int], Tuple[int, int]]]:
        """Deliberate on the next move using NN-guided MCTS.

        Args:
            board: The current 9x9 board state.
            macro_board: The current 3x3 macro board.
            active_macro: The active macro-board [my, mx] or None.
            valid_actions: List of valid moves [x, y].

        Returns:
            The chosen move [x, y], or None if the game is over.
        """
        if not valid_actions:
            return None

        # Construct state from server data
        state = UTTTState(
            board=board,
            macro_board=macro_board,
            active_macro=active_macro,
            current_player=self.player_id,  # type: ignore[arg-type]
        )

        if state.is_terminal():
            return None

        return self._deliberate(state, valid_actions)

    def deliberate_from_state(
        self, state: UTTTState
    ) -> Optional[List[int]]:
        """Headless deliberation from a UTTTState directly.

        Args:
            state: The game state to search from.

        Returns:
            The best action [x, y], or None.
        """
        if state.is_terminal() or not state.get_valid_actions():
            return None

        valid = state.get_valid_actions()
        return self._deliberate(state, valid)

    # ------------------------------------------------------------------
    # Core deliberation (shared by both paths)
    # ------------------------------------------------------------------

    def _deliberate(
        self,
        state: UTTTState,
        valid_actions: List[List[int]],
    ) -> Optional[List[int]]:
        """Run NN-guided MCTS and select a move.

        Args:
            state: The game state.
            valid_actions: Valid actions from this state.

        Returns:
            The chosen action [x, y], or None on error.
        """
        try:
            prior_fn, value_fn = self._make_bridge_functions()
        except RuntimeError as exc:
            logging.error(
                "AlphaZero agent cannot deliberate without a network. "
                "Pass --checkpoint-path / -p with a path to a trained .pt file. "
                f"Error: {exc}"
            )
            if valid_actions:
                logging.warning("Falling back to first valid action.")
                return valid_actions[0]
            return None

        start_time = time.monotonic()

        mcts = MCTS(
            iterations=self.mcts_iterations,
            exploration_constant=self.mcts_exploration_constant,
            time_limit=self.mcts_time_limit,
            random_seed=self.random_seed,
            prior_fn=prior_fn,
            value_fn=value_fn,
            use_puct=True,
        )

        try:
            best_action = mcts.search(state)
        except RuntimeError as exc:
            logging.warning(f"MCTS search failed: {exc}")
            return valid_actions[0] if valid_actions else None

        elapsed = time.monotonic() - start_time

        # ---- Temperature-based move selection ----
        if self.temperature == 0.0:
            # Deterministic: most-visited child (already returned by search)
            selected_action = best_action
        else:
            # Stochastic: sample from softmax of visit counts / temperature
            visit_dist = mcts.get_root_visit_distribution()
            selected_action = self._sample_from_distribution(
                visit_dist, valid_actions
            )

        # Collect stats
        stats = mcts.get_stats()
        stats["temperature"] = self.temperature
        stats["inference_elapsed"] = elapsed
        self._last_stats = stats

        logging.debug(
            f"AlphaZero search completed in {elapsed:.3f}s: "
            f"action={selected_action}, "
            f"iterations={stats.get('total_iterations', 0)}, "
            f"tree_size={stats.get('tree_size', 0)}, "
            f"win_rate={stats.get('best_action_win_rate', 0.0):.3f}"
        )

        # Safety: validate the returned action
        selected_list = (
            list(selected_action) if selected_action else None
        )
        if selected_list is not None and selected_list not in valid_actions:
            logging.warning(
                f"Selected invalid action {selected_list}, "
                f"falling back to {valid_actions[0]}"
            )
            return valid_actions[0]

        return selected_list

    # ------------------------------------------------------------------
    # Stochastic move selection
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_from_distribution(
        visit_dist: Dict[Tuple[int, int], float],
        valid_actions: List[List[int]],
    ) -> Optional[List[int]]:
        """Sample an action from the softmax of visit counts.

        Args:
            visit_dist: Mapping from (x, y) to visit count.
            valid_actions: List of valid [x, y] actions.

        Returns:
            A sampled action [x, y], or None if no valid actions.
        """
        if not valid_actions:
            return None

        # Build (action, weight) list
        actions_weights = []
        for x, y in valid_actions:
            weight = visit_dist.get((x, y), 0.0)
            actions_weights.append(([x, y], weight))

        # If all weights are zero, fall back to uniform
        total_weight = sum(w for _, w in actions_weights)
        if total_weight <= 0:
            return valid_actions[0]

        # Weighted choice using Python's random.choices
        actions, weights = zip(*actions_weights)
        rng = random.Random()
        chosen_idx = rng.choices(range(len(actions)), weights=weights, k=1)[0]
        return list(actions[chosen_idx])

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_last_stats(self) -> Dict[str, Any]:
        """Return statistics from the last deliberation.

        Returns:
            Dict with keys such as:
              - total_iterations
              - tree_size
              - root_visits
              - best_action
              - best_action_visits
              - best_action_win_rate
              - elapsed_seconds
              - temperature
              - inference_elapsed
        """
        return dict(self._last_stats)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "AlphaZero-style agent for Ultimate Tic-Tac-Toe. "
            "Uses a neural network to guide MCTS with PUCT."
        )
    )
    parser.add_argument(
        "--server-uri",
        type=str,
        default="ws://localhost:8765",
        help="WebSocket server URI (default: ws://localhost:8765)",
    )
    parser.add_argument(
        "--iterations",
        "-i",
        type=int,
        default=800,
        help="MCTS iterations per move (default: 800)",
    )
    parser.add_argument(
        "--exploration-constant",
        "-c",
        type=float,
        default=1.414,
        help="PUCT exploration constant C (default: 1.414)",
    )
    parser.add_argument(
        "--time-limit",
        "-t",
        type=float,
        default=None,
        help="Time limit per move in seconds (default: no limit)",
    )
    parser.add_argument(
        "--random-seed",
        "-s",
        type=int,
        default=None,
        help="Random seed for deterministic MCTS (default: no seed)",
    )
    parser.add_argument(
        "--checkpoint-path",
        "-p",
        type=str,
        required=True,
        help="Path to network checkpoint (.pt file)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for move selection (0=deterministic, default: 0.0)",
    )

    args = parser.parse_args()

    logging.debug(
        "Starting AlphaZeroUTTTAgent with config: "
        f"server_uri={args.server_uri}, "
        f"iterations={args.iterations}, "
        f"exploration_constant={args.exploration_constant}, "
        f"time_limit={args.time_limit}, "
        f"random_seed={args.random_seed}, "
        f"checkpoint_path={args.checkpoint_path}, "
        f"temperature={args.temperature}"
    )

    agent = AlphaZeroUTTTAgent(
        server_uri=args.server_uri,
        mcts_iterations=args.iterations,
        mcts_exploration_constant=args.exploration_constant,
        mcts_time_limit=args.time_limit,
        random_seed=args.random_seed,
        checkpoint_path=args.checkpoint_path,
        temperature=args.temperature,
    )
    asyncio.run(agent.run())
