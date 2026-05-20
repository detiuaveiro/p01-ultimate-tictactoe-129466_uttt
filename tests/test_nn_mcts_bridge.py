"""
Tests for engine.nn_mcts_bridge — create_nn_mcts_functions, prior_fn, value_fn.
"""

import random
from typing import Callable, Dict, Tuple

import pytest
import torch

from engine.game_state import UTTTState
from engine.nn_mcts_bridge import (
    _dirichlet_sample,
    create_nn_mcts_functions,
)
from engine.policy_value_network import PolicyValueNetwork


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_network() -> PolicyValueNetwork:
    """Create a small network for testing (faster inference)."""
    return PolicyValueNetwork(channels=32, num_res_blocks=3)


# ---------------------------------------------------------------------------
# create_nn_mcts_functions
# ---------------------------------------------------------------------------


class TestCreateNNMCTSFunctions:
    """Tests for create_nn_mcts_functions factory."""

    def test_returns_two_callables(self) -> None:
        """Factory returns a tuple of two callables."""
        network = _make_network()
        prior_fn, value_fn = create_nn_mcts_functions(network)
        assert callable(prior_fn), "prior_fn should be callable"
        assert callable(value_fn), "value_fn should be callable"

    def test_returns_prior_fn_and_value_fn(self) -> None:
        """Return values have correct names."""
        network = _make_network()
        prior_fn, value_fn = create_nn_mcts_functions(network)
        assert prior_fn.__name__ == "prior_fn"
        assert value_fn.__name__ == "value_fn"

    def test_network_set_to_eval(self) -> None:
        """Network is set to eval mode."""
        network = _make_network()
        network.train()  # ensure it starts in train mode
        assert network.training
        create_nn_mcts_functions(network)
        assert not network.training, (
            "Network should be in eval mode after factory call"
        )

    def test_without_dirichlet_noise(self) -> None:
        """prior_fn works without Dirichlet noise."""
        network = _make_network()
        prior_fn, value_fn = create_nn_mcts_functions(
            network, add_dirichlet_noise=False
        )
        state = UTTTState()
        priors = prior_fn(state)
        assert isinstance(priors, dict)

    def test_with_dirichlet_noise(self) -> None:
        """prior_fn works with Dirichlet noise enabled."""
        network = _make_network()
        rng = random.Random(42)
        prior_fn, value_fn = create_nn_mcts_functions(
            network,
            add_dirichlet_noise=True,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.25,
            rng=rng,
        )
        state = UTTTState()
        priors = prior_fn(state)
        assert isinstance(priors, dict)


# ---------------------------------------------------------------------------
# prior_fn
# ---------------------------------------------------------------------------


class TestPriorFn:
    """Tests for the prior_fn closure."""

    def test_returns_dict(self) -> None:
        """prior_fn returns a dict."""
        network = _make_network()
        prior_fn, _ = create_nn_mcts_functions(network)
        state = UTTTState()
        priors = prior_fn(state)
        assert isinstance(priors, dict)

    def test_keys_are_tuples(self) -> None:
        """All keys in returned dict are (int, int) tuples."""
        network = _make_network()
        prior_fn, _ = create_nn_mcts_functions(network)
        state = UTTTState()
        priors = prior_fn(state)
        for key in priors:
            assert isinstance(key, tuple)
            assert len(key) == 2
            assert isinstance(key[0], int)
            assert isinstance(key[1], int)

    def test_values_are_floats(self) -> None:
        """All values in returned dict are floats."""
        network = _make_network()
        prior_fn, _ = create_nn_mcts_functions(network)
        state = UTTTState()
        priors = prior_fn(state)
        for val in priors.values():
            assert isinstance(val, float)

    def test_only_legal_actions(self) -> None:
        """Only legal action keys are present."""
        network = _make_network()
        prior_fn, _ = create_nn_mcts_functions(network)
        state = UTTTState()
        priors = prior_fn(state)
        legal = state.get_valid_actions()
        legal_set = {(x, y) for x, y in legal}
        for key in priors:
            assert key in legal_set, (
                f"Key {key} is not a legal action"
            )
        # All legal actions are present
        for x, y in legal:
            assert (x, y) in priors, (
                f"Legal action ({x},{y}) missing from priors"
            )

    def test_probs_sum_to_one(self) -> None:
        """Probabilities sum to 1.0 over legal actions."""
        network = _make_network()
        prior_fn, _ = create_nn_mcts_functions(network)
        state = UTTTState()
        priors = prior_fn(state)
        total = sum(priors.values())
        assert total == pytest.approx(1.0, abs=1e-5), (
            f"Probabilities sum to {total}, expected 1.0"
        )

    def test_all_positive(self) -> None:
        """All probabilities are positive (> 0)."""
        network = _make_network()
        prior_fn, _ = create_nn_mcts_functions(network)
        state = UTTTState()
        priors = prior_fn(state)
        for key, val in priors.items():
            assert val > 0.0, (
                f"Probability for {key} is {val}, expected > 0"
            )

    def test_mid_game_state(self) -> None:
        """prior_fn works on a mid-game state with fewer legal actions."""
        network = _make_network()
        prior_fn, _ = create_nn_mcts_functions(network)

        # Build a state with some occupied cells
        state = UTTTState()
        state = state.apply_action(0, 0)  # P1
        state = state.apply_action(1, 1)  # P2
        state = state.apply_action(4, 4)  # P1
        state = state.apply_action(3, 3)  # P2

        priors = prior_fn(state)
        total = sum(priors.values())
        assert total == pytest.approx(1.0, abs=1e-5)
        # Ensure only legal actions are present
        legal = state.get_valid_actions()
        assert len(priors) == len(legal)

    def test_deterministic_without_noise(self) -> None:
        """Without noise, same state produces same priors."""
        network = _make_network()
        prior_fn_a, _ = create_nn_mcts_functions(network)
        prior_fn_b, _ = create_nn_mcts_functions(network)
        state = UTTTState()
        priors_a = prior_fn_a(state)
        priors_b = prior_fn_b(state)
        # Same network, same state -> same outputs (tiny floating point diff OK)
        for key in priors_a:
            assert priors_a[key] == pytest.approx(priors_b[key], abs=1e-6)

    def test_with_noise_still_sums_to_one(self) -> None:
        """Dirichlet noise still produces probabilities summing to 1."""
        network = _make_network()
        rng = random.Random(12345)
        prior_fn, _ = create_nn_mcts_functions(
            network,
            add_dirichlet_noise=True,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.25,
            rng=rng,
        )
        state = UTTTState()
        priors = prior_fn(state)
        total = sum(priors.values())
        assert total == pytest.approx(1.0, abs=1e-5)

    def test_with_noise_modifies_distribution(self) -> None:
        """Dirichlet noise changes the distribution."""
        network = _make_network()

        # Without noise
        prior_fn_clean, _ = create_nn_mcts_functions(
            network, add_dirichlet_noise=False
        )
        # With noise
        rng = random.Random(999)
        prior_fn_noisy, _ = create_nn_mcts_functions(
            network,
            add_dirichlet_noise=True,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.25,
            rng=rng,
        )

        state = UTTTState()
        clean = prior_fn_clean(state)
        noisy = prior_fn_noisy(state)

        # The two distributions should differ
        clean_vals = [clean[k] for k in sorted(clean.keys())]
        noisy_vals = [noisy[k] for k in sorted(noisy.keys())]
        assert clean_vals != pytest.approx(noisy_vals, abs=1e-6), (
            "Noise should modify the distribution"
        )


# ---------------------------------------------------------------------------
# value_fn
# ---------------------------------------------------------------------------


class TestValueFn:
    """Tests for the value_fn closure."""

    def test_returns_float(self) -> None:
        """value_fn returns a float."""
        network = _make_network()
        _, value_fn = create_nn_mcts_functions(network)
        state = UTTTState()
        val = value_fn(state)
        assert isinstance(val, float)

    def test_in_range_neg_one_to_one(self) -> None:
        """Value is in [-1, 1]."""
        network = _make_network()
        _, value_fn = create_nn_mcts_functions(network)

        # Test several random-ish states
        for _ in range(5):
            state = UTTTState()
            val = value_fn(state)
            assert -1.0 <= val <= 1.0, (
                f"Value {val} outside [-1, 1]"
            )

    def test_mid_game_range(self) -> None:
        """Value for a mid-game state is in [-1, 1]."""
        network = _make_network()
        _, value_fn = create_nn_mcts_functions(network)

        state = UTTTState()
        state = state.apply_action(0, 0)
        state = state.apply_action(1, 1)
        state = state.apply_action(4, 4)
        state = state.apply_action(3, 3)

        val = value_fn(state)
        assert -1.0 <= val <= 1.0

    def test_deterministic(self) -> None:
        """value_fn returns same value for same state + network."""
        network = _make_network()
        _, value_fn_a = create_nn_mcts_functions(network)
        _, value_fn_b = create_nn_mcts_functions(network)
        state = UTTTState()
        val_a = value_fn_a(state)
        val_b = value_fn_b(state)
        assert val_a == pytest.approx(val_b, abs=1e-6)

    def test_empty_board(self) -> None:
        """value_fn returns a finite value for an empty board."""
        network = _make_network()
        _, value_fn = create_nn_mcts_functions(network)
        state = UTTTState()
        val = value_fn(state)
        assert val == val  # not NaN
        assert float("-inf") < val < float("inf")


# ---------------------------------------------------------------------------
# Integration: prior_fn + value_fn together
# ---------------------------------------------------------------------------


class TestPriorAndValueIntegration:
    """Integration tests using both prior_fn and value_fn together."""

    def test_same_state_consistent(self) -> None:
        """prior_fn and value_fn operate consistently on same state."""
        network = _make_network()
        prior_fn, value_fn = create_nn_mcts_functions(network)

        state = UTTTState()
        priors = prior_fn(state)
        val = value_fn(state)

        assert sum(priors.values()) == pytest.approx(1.0, abs=1e-5)
        assert -1.0 <= val <= 1.0

    def test_multiple_states(self) -> None:
        """Both functions work on multiple different states."""
        network = _make_network()
        prior_fn, value_fn = create_nn_mcts_functions(network)

        states = [
            UTTTState(),  # empty
            UTTTState(active_macro=[0, 0]),  # restricted to top-left
            UTTTState(active_macro=None),  # free move
        ]

        for state in states:
            priors = prior_fn(state)
            val = value_fn(state)
            assert sum(priors.values()) == pytest.approx(1.0, abs=1e-5)
            assert -1.0 <= val <= 1.0


# ---------------------------------------------------------------------------
# _dirichlet_sample
# ---------------------------------------------------------------------------


class TestDirichletSample:
    """Tests for the internal _dirichlet_sample helper."""

    def test_returns_numpy_array(self) -> None:
        """_dirichlet_sample returns a numpy array."""
        rng = random.Random(42)
        sample = _dirichlet_sample(0.3, 81, rng)
        import numpy as np

        assert isinstance(sample, np.ndarray)

    def test_shape(self) -> None:
        """_dirichlet_sample returns correct shape."""
        rng = random.Random(42)
        sample = _dirichlet_sample(0.3, 81, rng)
        assert sample.shape == (81,)

    def test_sums_to_one(self) -> None:
        """Dirichlet sample sums to 1."""
        rng = random.Random(42)
        sample = _dirichlet_sample(0.3, 81, rng)
        assert sample.sum() == pytest.approx(1.0, abs=1e-5)

    def test_all_positive(self) -> None:
        """All Dirichlet values are positive."""
        rng = random.Random(42)
        sample = _dirichlet_sample(0.3, 81, rng)
        assert (sample > 0).all()

    def test_deterministic_seed(self) -> None:
        """Same seed produces same sample."""
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        s1 = _dirichlet_sample(0.3, 81, rng1)
        s2 = _dirichlet_sample(0.3, 81, rng2)
        import numpy as np

        assert np.allclose(s1, s2)

    def test_different_seed_different(self) -> None:
        """Different seeds produce different samples."""
        rng1 = random.Random(1)
        rng2 = random.Random(2)
        s1 = _dirichlet_sample(0.3, 81, rng1)
        s2 = _dirichlet_sample(0.3, 81, rng2)
        import numpy as np

        assert not np.allclose(s1, s2)

    def test_diff_alpha_zero(self) -> None:
        """Small alpha near 0 still produces valid sample."""
        rng = random.Random(42)
        sample = _dirichlet_sample(0.001, 81, rng)
        assert sample.sum() == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Network caching
# ---------------------------------------------------------------------------


class TestNNEvaluatorCache:
    """Tests that the internal evaluator caches network evaluations."""

    def test_same_state_reuses_cache(self):
        """Calling prior_fn then value_fn on the same state uses one forward pass."""
        from unittest.mock import patch

        network = _make_network()
        prior_fn, value_fn = create_nn_mcts_functions(network, device="cpu")
        state = UTTTState()

        with patch.object(
            network, "forward", wraps=network.forward
        ) as mock_forward:
            prior = prior_fn(state)
            val = value_fn(state)
            # Only one forward pass should have occurred
            assert mock_forward.call_count == 1

        assert isinstance(prior, dict)
        assert isinstance(val, float)

    def test_cache_returns_consistent_results(self):
        """Cached results are identical across repeated calls."""
        network = _make_network()
        prior_fn, value_fn = create_nn_mcts_functions(network, device="cpu")
        state = UTTTState()
        p1 = prior_fn(state)
        v1 = value_fn(state)
        p2 = prior_fn(state)
        v2 = value_fn(state)
        assert p1 == p2
        assert v1 == v2
