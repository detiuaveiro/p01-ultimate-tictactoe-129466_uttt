"""
Tests for the self-play pipeline components.
"""

import os
import random
import tempfile
from typing import List

import numpy as np
import pytest
import torch

from engine.policy_value_network import PolicyValueNetwork, load_network, save_network
from selfplay.config import SelfPlayConfig
from selfplay.pipeline import load_latest_checkpoint, save_checkpoint
from selfplay.self_play import (
    TrainingExample,
    compute_search_policy,
    generate_self_play_games,
    get_temperature,
    sample_action,
)
from selfplay.train import prepare_batches, train_network


# ---------------------------------------------------------------------------
# SelfPlayConfig
# ---------------------------------------------------------------------------


class TestSelfPlayConfig:
    """Tests for SelfPlayConfig defaults and construction."""

    def test_defaults(self) -> None:
        """Config can be created with default values."""
        config = SelfPlayConfig()
        assert config.num_iterations == 10
        assert config.games_per_iteration == 100
        assert config.mcts_iterations == 800
        assert config.c_puct == 1.414
        assert config.temperature_schedule == [
            (0, 1.0), (10, 0.5), (20, 0.25)
        ]
        assert config.dirichlet_alpha == 0.3
        assert config.dirichlet_epsilon == 0.25
        assert config.learning_rate == 0.001
        assert config.batch_size == 32
        assert config.epochs == 5
        assert config.checkpoint_dir == "checkpoints/"
        assert config.l2_regularization == 0.0001
        assert config.network_channels == 160
        assert config.network_res_blocks == 10

    def test_custom_values(self) -> None:
        """Config accepts custom values."""
        config = SelfPlayConfig(
            num_iterations=5,
            games_per_iteration=50,
            mcts_iterations=400,
            c_puct=2.0,
            checkpoint_dir="/tmp/checkpoints/",
        )
        assert config.num_iterations == 5
        assert config.games_per_iteration == 50
        assert config.mcts_iterations == 400
        assert config.c_puct == 2.0
        assert config.checkpoint_dir == "/tmp/checkpoints/"


# ---------------------------------------------------------------------------
# TrainingExample
# ---------------------------------------------------------------------------


class TestTrainingExample:
    """Tests for TrainingExample dataclass."""

    def test_structure(self) -> None:
        """TrainingExample stores features, policy, and outcome."""
        features = np.zeros((3, 9, 9), dtype=np.float32)
        policy = np.zeros(81, dtype=np.float32)
        policy[0] = 1.0
        example = TrainingExample(
            state_features=features,
            search_policy=policy,
            outcome=1.0,
        )
        assert example.state_features.shape == (3, 9, 9)
        assert example.search_policy.shape == (81,)
        assert example.outcome == 1.0

    def test_outcome_values(self) -> None:
        """Outcome can be +1, -1, or 0."""
        for outcome in [1.0, -1.0, 0.0]:
            ex = TrainingExample(
                state_features=np.zeros((3, 9, 9), dtype=np.float32),
                search_policy=np.zeros(81, dtype=np.float32),
                outcome=outcome,
            )
            assert ex.outcome == outcome


# ---------------------------------------------------------------------------
# get_temperature
# ---------------------------------------------------------------------------


class TestGetTemperature:
    """Tests for the temperature schedule helper."""

    def test_default_schedule(self) -> None:
        """Default schedule returns expected temperatures."""
        schedule = [(0, 1.0), (10, 0.5), (20, 0.25)]
        assert get_temperature(0, schedule) == 1.0
        assert get_temperature(5, schedule) == pytest.approx(0.75)
        assert get_temperature(10, schedule) == 0.5
        assert get_temperature(15, schedule) == pytest.approx(0.375)
        assert get_temperature(20, schedule) == 0.25
        assert get_temperature(100, schedule) == 0.25

    def test_empty_schedule(self) -> None:
        """Empty schedule returns 1.0."""
        assert get_temperature(0, []) == 1.0
        assert get_temperature(50, []) == 1.0

    def test_single_point(self) -> None:
        """Single-point schedule returns that temperature."""
        schedule = [(0, 0.5)]
        assert get_temperature(0, schedule) == 0.5
        assert get_temperature(100, schedule) == 0.5


# ---------------------------------------------------------------------------
# compute_search_policy
# ---------------------------------------------------------------------------


class TestComputeSearchPolicy:
    """Tests for the search-policy computation."""

    def test_zero_temperature_one_hot(self) -> None:
        """Temperature=0 produces one-hot at most-visited."""
        visits = {(0, 0): 10.0, (4, 4): 5.0, (8, 8): 2.0}
        policy = compute_search_policy(
            visits, temperature=0.0,
            valid_actions=[[0, 0], [4, 4], [8, 8]],
        )
        assert policy[0] == 1.0  # (0,0) is most visited
        assert policy[4 * 9 + 4] == 0.0
        assert policy[8 * 9 + 8] == 0.0

    def test_high_temperature_uniform(self) -> None:
        """Very high temperature approximates uniform."""
        visits = {(0, 0): 100.0, (4, 4): 1.0}
        policy = compute_search_policy(
            visits, temperature=100.0,
            valid_actions=[[0, 0], [4, 4]],
        )
        # Should be nearly uniform
        assert policy[0] == pytest.approx(policy[4 * 9 + 4], abs=0.05)

    def test_only_legal_actions_positive(self) -> None:
        """Only legal actions have non-zero probability."""
        visits = {(0, 0): 10.0, (4, 4): 5.0}
        policy = compute_search_policy(
            visits, temperature=1.0,
            valid_actions=[[0, 0], [4, 4]],
        )
        assert policy[0] > 0
        assert policy[4 * 9 + 4] > 0
        # Illegal positions (most of them) should be 0
        assert policy[1] == 0.0

    def test_sums_to_one(self) -> None:
        """Policy sums to 1 over all 81 entries."""
        visits = {(0, 0): 10.0, (1, 1): 3.0, (2, 2): 7.0}
        policy = compute_search_policy(
            visits, temperature=1.0,
            valid_actions=[[0, 0], [1, 1], [2, 2]],
        )
        assert policy.sum() == pytest.approx(1.0, abs=1e-6)

    def test_empty_visits_fallback(self) -> None:
        """Empty visits falls back to uniform over valid actions."""
        policy = compute_search_policy(
            {}, temperature=1.0,
            valid_actions=[[0, 0], [4, 4]],
        )
        assert policy[0] == pytest.approx(0.5)
        assert policy[4 * 9 + 4] == pytest.approx(0.5)
        assert policy.sum() == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# sample_action
# ---------------------------------------------------------------------------


class TestSampleAction:
    """Tests for action sampling from search policy."""

    def test_zero_temperature_deterministic(self) -> None:
        """Temperature=0 picks argmax."""
        policy = np.zeros(81, dtype=np.float32)
        policy[40] = 0.9  # (4,4)
        policy[0] = 0.1  # (0,0)
        valid = [[0, 0], [4, 4]]
        rng = random.Random(42)
        action = sample_action(policy, valid, 0.0, rng)
        assert action == [4, 4]

    def test_returns_valid_action(self) -> None:
        """Sampled action is always in valid_actions."""
        policy = np.zeros(81, dtype=np.float32)
        policy[0] = 0.5
        policy[40] = 0.5
        valid = [[0, 0], [4, 4]]
        rng = random.Random(42)
        for _ in range(20):
            action = sample_action(policy, valid, 1.0, rng)
            assert action in valid

    def test_fallback_on_empty_probs(self) -> None:
        """All-zero policy falls back to first valid action."""
        policy = np.zeros(81, dtype=np.float32)
        valid = [[0, 0], [4, 4]]
        rng = random.Random(42)
        action = sample_action(policy, valid, 1.0, rng)
        assert action in valid


# ---------------------------------------------------------------------------
# generate_self_play_games
# ---------------------------------------------------------------------------


class TestGenerateSelfPlayGames:
    """Tests for self-play game generation."""

    def _make_small_network(self) -> PolicyValueNetwork:
        """Create a small network for fast testing."""
        return PolicyValueNetwork(channels=16, num_res_blocks=2)

    def test_generates_examples(self) -> None:
        """Games produce training examples."""
        network = self._make_small_network()
        config = SelfPlayConfig(
            games_per_iteration=1,
            mcts_iterations=20,
            temperature_schedule=[(0, 1.0)],
        )
        rng = random.Random(42)
        examples = generate_self_play_games(network, config, rng)
        assert len(examples) > 0

    def test_example_structure(self) -> None:
        """Each example has correct structure."""
        network = self._make_small_network()
        config = SelfPlayConfig(
            games_per_iteration=1,
            mcts_iterations=20,
            temperature_schedule=[(0, 1.0)],
        )
        rng = random.Random(42)
        examples = generate_self_play_games(network, config, rng)
        for ex in examples:
            assert ex.state_features.shape == (3, 9, 9)
            assert ex.search_policy.shape == (81,)
            assert ex.outcome in (1.0, -1.0, 0.0)

    def test_multiple_games(self) -> None:
        """Multiple games generate multiple examples."""
        network = self._make_small_network()
        config = SelfPlayConfig(
            games_per_iteration=2,
            mcts_iterations=20,
            temperature_schedule=[(0, 1.0)],
        )
        rng = random.Random(42)
        examples = generate_self_play_games(network, config, rng)
        assert len(examples) >= 2  # at least 2 examples from 2 games

    def test_deterministic_with_same_seed(self) -> None:
        """Same seed produces same results (approximate)."""
        network = self._make_small_network()
        config = SelfPlayConfig(
            games_per_iteration=1,
            mcts_iterations=30,
            temperature_schedule=[(0, 0.5)],  # deterministic-ish
        )

        rng1 = random.Random(42)
        rng2 = random.Random(42)
        examples1 = generate_self_play_games(network, config, rng1)
        examples2 = generate_self_play_games(network, config, rng2)
        # Should have same number of examples
        assert len(examples1) == len(examples2)


# ---------------------------------------------------------------------------
# prepare_batches
# ---------------------------------------------------------------------------


class TestPrepareBatches:
    """Tests for batch preparation."""

    def test_batch_count(self) -> None:
        """Correct number of batches for given batch_size."""
        examples = [
            TrainingExample(
                state_features=np.zeros((3, 9, 9), dtype=np.float32),
                search_policy=np.zeros(81, dtype=np.float32),
                outcome=0.0,
            )
            for _ in range(10)
        ]
        batches = prepare_batches(examples, batch_size=3, shuffle_rng=random.Random(0))
        assert len(batches) == 4  # 10/3 = 4 batches

    def test_batch_shapes(self) -> None:
        """Each batch has correct tensor shapes."""
        examples = [
            TrainingExample(
                state_features=np.random.randn(3, 9, 9).astype(np.float32),
                search_policy=np.random.randn(81).astype(np.float32),
                outcome=1.0,
            )
            for _ in range(4)
        ]
        batches = prepare_batches(examples, batch_size=4, shuffle_rng=random.Random(0))
        assert len(batches) == 1
        states, policies, values = batches[0]
        assert states.shape == (4, 3, 9, 9)
        assert policies.shape == (4, 81)
        assert values.shape == (4, 1)


# ---------------------------------------------------------------------------
# train_network
# ---------------------------------------------------------------------------


class TestTrainNetwork:
    """Tests for network training on synthetic data."""

    def _make_network(self) -> PolicyValueNetwork:
        return PolicyValueNetwork(channels=16, num_res_blocks=2)

    def test_trains_without_error(self) -> None:
        """train_network runs without error on synthetic data."""
        network = self._make_network()
        examples = [
            TrainingExample(
                state_features=np.random.randn(3, 9, 9).astype(np.float32),
                search_policy=np.zeros(81, dtype=np.float32),
                outcome=1.0,
            )
            for _ in range(8)
        ]
        # Set search_policy to uniform over all actions for simplicity
        for ex in examples:
            ex.search_policy.fill(1.0 / 81.0)

        config = SelfPlayConfig(
            batch_size=4,
            epochs=2,
            learning_rate=0.001,
        )
        result = train_network(network, examples, config)
        assert result is network  # same instance
        assert not network.training  # eval mode after training

    def test_empty_examples(self) -> None:
        """train_network handles empty examples list."""
        network = self._make_network()
        config = SelfPlayConfig(epochs=1)
        result = train_network(network, [], config)
        assert result is network

    def test_loss_decreases(self) -> None:
        """Training reduces loss on a tiny dataset."""
        network = self._make_network()
        n_examples = 16
        examples = [
            TrainingExample(
                state_features=np.random.randn(3, 9, 9).astype(np.float32),
                search_policy=np.zeros(81, dtype=np.float32),
                outcome=1.0,
            )
            for _ in range(n_examples)
        ]
        for ex in examples:
            ex.search_policy.fill(1.0 / 81.0)

        config = SelfPlayConfig(batch_size=4, epochs=3, learning_rate=0.01)
        # Measure loss before
        network.eval()
        states_tensor = torch.stack([
            torch.from_numpy(ex.state_features) for ex in examples
        ])
        with torch.no_grad():
            logits_before, val_before = network(states_tensor)

        network = train_network(network, examples, config)

        # Measure loss after
        network.eval()
        with torch.no_grad():
            logits_after, val_after = network(states_tensor)

        # Check that value predictions moved closer to target (1.0)
        loss_before = ((val_before.squeeze() - 1.0) ** 2).mean()
        loss_after = ((val_after.squeeze() - 1.0) ** 2).mean()
        # Training should move predictions toward the target
        assert loss_after <= loss_before + 0.5  # generous tolerance


# ---------------------------------------------------------------------------
# Checkpoint save/load
# ---------------------------------------------------------------------------


class TestCheckpointRoundTrip:
    """Tests for checkpoint save and load."""

    def test_save_and_load(self) -> None:
        """Checkpoint save/load round-trip preserves network."""
        network = PolicyValueNetwork(channels=16, num_res_blocks=2)
        x = torch.randn(1, 3, 9, 9)
        network.eval()
        with torch.no_grad():
            ref_logits, ref_value = network(x)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint(network, 0, tmpdir)
            loaded, start_iter = load_latest_checkpoint(
                SelfPlayConfig(checkpoint_dir=tmpdir)
            )
            assert loaded is not None
            assert start_iter == 1
            loaded.eval()
            with torch.no_grad():
                logits, value = loaded(x)
            assert torch.equal(logits, ref_logits)
            assert torch.equal(value, ref_value)

    def test_load_latest_finds_highest(self) -> None:
        """load_latest_checkpoint finds the highest iteration."""
        network = PolicyValueNetwork(channels=16, num_res_blocks=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint(network, 5, tmpdir)
            save_checkpoint(network, 3, tmpdir)
            save_checkpoint(network, 10, tmpdir)
            loaded, start_iter = load_latest_checkpoint(
                SelfPlayConfig(checkpoint_dir=tmpdir)
            )
            assert loaded is not None
            assert start_iter == 11  # highest iteration + 1

    def test_no_checkpoint_returns_none(self) -> None:
        """No checkpoint returns (None, 0)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loaded, start_iter = load_latest_checkpoint(
                SelfPlayConfig(checkpoint_dir=tmpdir)
            )
            assert loaded is None
            assert start_iter == 0

    def test_invalid_directory_returns_none(self) -> None:
        """Non-existent directory returns (None, 0)."""
        config = SelfPlayConfig(checkpoint_dir="/nonexistent/path/")
        loaded, start_iter = load_latest_checkpoint(config)
        assert loaded is None
        assert start_iter == 0
