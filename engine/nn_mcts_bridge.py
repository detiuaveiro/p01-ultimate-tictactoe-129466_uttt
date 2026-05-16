"""
Bridge between the Policy-Value Network and MCTS.

Provides factory functions that create ``prior_fn`` and ``value_fn`` closures
compatible with ``MCTS.__init__`` (``use_puct=True`` mode).

Usage::

    from engine.nn_mcts_bridge import create_nn_mcts_functions

    network = load_network("checkpoint.pt")
    prior_fn, value_fn = create_nn_mcts_functions(network, add_dirichlet_noise=True)

    mcts = MCTS(iterations=800, use_puct=True,
                prior_fn=prior_fn, value_fn=value_fn)
    best_action = mcts.search(state)
"""

import math
import random
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch

from engine.game_state import UTTTState
from engine.policy_value_network import (
    PolicyValueNetwork,
    encode_state,
    get_masked_policy,
)


def create_nn_mcts_functions(
    network: PolicyValueNetwork,
    device: str = "cpu",
    add_dirichlet_noise: bool = False,
    dirichlet_alpha: float = 0.3,
    dirichlet_epsilon: float = 0.25,
    rng: Optional[random.Random] = None,
) -> Tuple[Callable, Callable]:
    """Create ``(prior_fn, value_fn)`` closures for MCTS from a neural network.

    Args:
        network: A trained PolicyValueNetwork (will be set to eval mode).
        device: Device to run inference on (``'cpu'`` or ``'cuda'``).
        add_dirichlet_noise: Whether to add Dirichlet noise to the policy
            at the root node (standard AlphaZero exploration technique).
        dirichlet_alpha: Concentration parameter for the Dirichlet
            distribution (default ``0.3``, suitable for 81-action space).
        dirichlet_epsilon: Mixing weight ``(1-ε) * policy + ε * noise``.
        rng: Optional ``random.Random`` instance for reproducibility.
            If ``None``, a new ``Random`` instance is created.

    Returns:
        A tuple of ``(prior_fn, value_fn)`` callables, each accepting a
        :class:`UTTTState` and returning:
          - ``prior_fn``: ``Dict[Tuple[int,int], float]`` mapping each legal
            ``(x, y)`` to a prior probability (summing to 1).
          - ``value_fn``: ``float`` in ``[-1, 1]`` from the current player's
            perspective.
    """
    network.eval()
    rng = rng if rng is not None else random.Random()

    # ---- prior_fn closure ----

    def prior_fn(state: UTTTState) -> Dict[Tuple[int, int], float]:
        """Compute prior probabilities for all legal actions from *state*.

        The state is encoded as a 3 x 9 x 9 tensor and passed through the
        network.  Illegal actions are masked out and the result is
        normalised via softmax.

        If ``add_dirichlet_noise`` was set at construction time, Dirichlet
        noise is mixed into the policy::

            final = (1 - ε) * policy + ε * dirichlet_sample

        Args:
            state: The game state for which to compute priors.

        Returns:
            A dict mapping each legal ``(x, y)`` action to its prior
            probability.  Probabilities sum to 1 over legal actions.
        """
        # Encode and add batch dimension
        encoded = encode_state(state).unsqueeze(0).to(device)  # (1, 3, 9, 9)

        with torch.no_grad():
            policy_logits, _ = network(encoded)

        # policy_logits: (1, 81) -> squeeze to (81,)
        logits = policy_logits.squeeze(0)

        # Valid actions
        valid_actions = state.get_valid_actions()

        # Mask and softmax
        probs = get_masked_policy(logits, valid_actions)  # (81,)

        # Dirichlet noise at root
        if add_dirichlet_noise:
            dirichlet = torch.from_numpy(
                _dirichlet_sample(dirichlet_alpha, 81, rng)
            ).float()
            # Only mix noise into legal positions
            mask = torch.zeros(81, dtype=torch.bool)
            for x, y in valid_actions:
                mask[y * 9 + x] = True
            # Apply mixing only to legal entries (keep illegal as 0)
            noisy = (
                1 - dirichlet_epsilon
            ) * probs + dirichlet_epsilon * dirichlet
            # Re-normalise over legal actions
            noisy = noisy * mask.float()
            noisy = noisy / (noisy.sum() + 1e-12)
            probs = noisy

        # Build result dict
        result: Dict[Tuple[int, int], float] = {}
        for x, y in valid_actions:
            idx = y * 9 + x
            result[(x, y)] = float(probs[idx].item())

        return result

    # ---- value_fn closure ----

    def value_fn(state: UTTTState) -> float:
        """Evaluate *state* using the network's value head.

        Args:
            state: The game state to evaluate.

        Returns:
            A scalar in ``[-1, 1]`` representing the expected outcome from
            the current player's perspective (positive = favourable).
        """
        encoded = encode_state(state).unsqueeze(0).to(device)  # (1, 3, 9, 9)

        with torch.no_grad():
            _, value = network(encoded)

        return float(value.item())

    return prior_fn, value_fn


# ---------------------------------------------------------------------------
# Dirichlet noise helper (pure Python, no np.random dependency)
# ---------------------------------------------------------------------------


def _dirichlet_sample(
    alpha: float, size: int, rng: random.Random
) -> np.ndarray:
    """Draw a single Dirichlet sample with equal concentration parameters.

    Uses Gamma(alpha, 1) samples normalised to sum to 1.

    Args:
        alpha: Concentration parameter (same for all categories).
        size: Number of categories.
        rng: A ``random.Random`` instance for reproducibility.

    Returns:
        A 1-D NumPy array of shape ``(size,)`` summing to 1.
    """
    gamma_samples = np.array(
        [_gamma_sample(alpha, rng) for _ in range(size)]
    )
    return gamma_samples / (gamma_samples.sum() + 1e-12)


def _gamma_sample(alpha: float, rng: random.Random) -> float:
    """Sample from Gamma(alpha, 1) using Marsaglia-Tsang (alpha >= 1).

    For alpha < 1, uses the alpha + 1 transformation.

    Args:
        alpha: Shape parameter (> 0).
        rng: A ``random.Random`` instance.

    Returns:
        A single Gamma(alpha, 1) sample.
    """
    if alpha >= 1:
        # Marsaglia-Tsang method
        d = alpha - 1.0 / 3.0
        c = 1.0 / (3.0 * math.sqrt(d))
        while True:
            u = rng.random()
            # Box-Muller for Gaussian
            g = math.sqrt(-2.0 * math.log(u)) * math.cos(
                2.0 * math.pi * rng.random()
            )
            v = (1.0 + c * g) ** 3
            if v <= 0:
                continue
            u2 = rng.random()
            if u2 < 1.0 - 0.0331 * (v * v) / (g * g * g * g + 1e-12):
                return d * v
            if math.log(u2) < 0.5 * g * g + d * (1.0 - v + math.log(v)):
                return d * v
    else:
        # alpha < 1: use alpha+1 transformation
        correction = _gamma_sample(alpha + 1.0, rng)
        return correction * (rng.random() ** (1.0 / alpha))
