"""
Tests for agents.alphazero_agent.AlphaZeroUTTTAgent.

Some tests depend on PyTorch (for network-related functionality). Those tests
are guarded with ``pytest.importorskip("torch")`` so they gracefully skip when
torch is unavailable (e.g. when not running inside ``nix develop``).
"""

from typing import Any, Dict, List, Optional, Tuple

import pytest

from agents.alphazero_agent import AlphaZeroUTTTAgent
from engine.game_state import UTTTState


# ============================================================================
# Test helpers
# ============================================================================


def _create_temp_checkpoint(tmp_path: Any) -> str:
    """Create a ``PolicyValueNetwork`` checkpoint at a temporary path.

    Uses the default architecture (``channels=160, num_res_blocks=10``) because
    the engine's ``load_network`` always instantiates ``PolicyValueNetwork()``
    with default parameters — a mismatched architecture would cause a
    ``RuntimeError`` on ``load_state_dict``.

    Note:
        For a smaller/faster alternative, consider injecting a mock network
        or refactoring ``load_network`` to accept architecture kwargs.

    Args:
        tmp_path: A ``pathlib.Path`` provided by the ``tmp_path`` fixture.

    Returns:
        The string path to the saved ``.pt`` file.
    """
    from engine.policy_value_network import PolicyValueNetwork, save_network

    network = PolicyValueNetwork()  # default: channels=160, num_res_blocks=10
    path = str(tmp_path / "test_checkpoint.pt")
    save_network(network, path)
    return path


# ============================================================================
# Constructor tests (no torch required)
# ============================================================================


class TestAlphaZeroUTTTAgentConstructor:
    """Tests for agent construction and parameter storage."""

    def test_constructor_defaults(self) -> None:
        """Agent can be constructed with default parameters."""
        agent = AlphaZeroUTTTAgent()
        assert agent.server_uri == "ws://localhost:8765"
        assert agent.mcts_iterations == 800
        assert agent.mcts_exploration_constant == 1.414
        assert agent.mcts_time_limit is None
        assert agent.random_seed is None
        assert agent.checkpoint_path is None
        assert agent.temperature == 0.0

    def test_constructor_custom(self) -> None:
        """Agent can be constructed with custom parameters."""
        agent = AlphaZeroUTTTAgent(
            server_uri="ws://example.com:9000",
            mcts_iterations=500,
            mcts_exploration_constant=2.5,
            mcts_time_limit=0.5,
            random_seed=42,
            checkpoint_path="/some/path.pt",
            temperature=1.0,
        )
        assert agent.server_uri == "ws://example.com:9000"
        assert agent.mcts_iterations == 500
        assert agent.mcts_exploration_constant == 2.5
        assert agent.mcts_time_limit == 0.5
        assert agent.random_seed == 42
        assert agent.checkpoint_path == "/some/path.pt"
        assert agent.temperature == 1.0

    def test_network_is_none_after_init(self) -> None:
        """_network is None after construction (lazy init)."""
        agent = AlphaZeroUTTTAgent()
        assert agent._network is None

    def test_checkpoint_path_none_by_default(self) -> None:
        """checkpoint_path is None when not provided."""
        agent = AlphaZeroUTTTAgent()
        assert agent.checkpoint_path is None

    def test_temperature_zero_by_default(self) -> None:
        """temperature defaults to 0 (deterministic)."""
        agent = AlphaZeroUTTTAgent()
        assert agent.temperature == 0.0

    def test_last_stats_empty_after_init(self) -> None:
        """_last_stats is an empty dict after construction."""
        agent = AlphaZeroUTTTAgent()
        assert agent._last_stats == {}


# ============================================================================
# Lazy network tests (no torch required for the RuntimeError path)
# ============================================================================


class TestAlphaZeroUTTTAgentLazyNetwork:
    """Tests for lazy network initialisation behaviour."""

    def test_get_network_raises_without_checkpoint(self) -> None:
        """_get_network raises RuntimeError when checkpoint_path is None."""
        agent = AlphaZeroUTTTAgent()
        with pytest.raises(RuntimeError, match="No checkpoint path specified"):
            agent._get_network()

    def test_make_bridge_functions_raises_without_checkpoint(self) -> None:
        """_make_bridge_functions raises RuntimeError when no network loaded."""
        agent = AlphaZeroUTTTAgent()
        with pytest.raises(RuntimeError, match="No checkpoint path specified"):
            agent._make_bridge_functions()

    def test_get_network_returns_network_after_load(self, tmp_path: Any) -> None:
        """_get_network returns a loaded network when checkpoint_path is set."""
        pytest.importorskip("torch")
        from engine.policy_value_network import PolicyValueNetwork

        checkpoint = _create_temp_checkpoint(tmp_path)
        agent = AlphaZeroUTTTAgent(checkpoint_path=checkpoint)
        network = agent._get_network()
        assert network is not None
        assert isinstance(network, PolicyValueNetwork)
        # Verify it is in eval mode
        assert not network.training

    def test_lazy_network_singleton(self, tmp_path: Any) -> None:
        """_get_network always returns the same cached instance."""
        pytest.importorskip("torch")
        checkpoint = _create_temp_checkpoint(tmp_path)
        agent = AlphaZeroUTTTAgent(checkpoint_path=checkpoint)
        n1 = agent._get_network()
        n2 = agent._get_network()
        assert n1 is n2


# ============================================================================
# Deliberation tests (headless, no torch required for early-return paths)
# ============================================================================


class TestAlphaZeroUTTTAgentDeliberateFromState:
    """Tests for ``deliberate_from_state`` (headless)."""

    def test_deliberate_from_state_terminal(self) -> None:
        """Returns None for a terminal state."""
        macro_board = [[1, 1, 1], [0, 0, 0], [0, 0, 0]]
        state = UTTTState(macro_board=macro_board)
        assert state.is_terminal()
        agent = AlphaZeroUTTTAgent()
        action = agent.deliberate_from_state(state)
        assert action is None

    def test_deliberate_from_state_no_valid_actions(self) -> None:
        """Returns None when no valid actions remain."""
        board = [[0] * 9 for _ in range(9)]
        macro_board = [[1, 0, 0], [0, 0, 0], [0, 0, 0]]
        # Fill all cells in macro (0,0)
        for y in range(3):
            for x in range(3):
                board[y][x] = 1
        state = UTTTState(
            board=board, macro_board=macro_board, active_macro=[0, 0]
        )
        assert len(state.get_valid_actions()) == 0
        agent = AlphaZeroUTTTAgent()
        action = agent.deliberate_from_state(state)
        assert action is None

    def test_deliberate_from_state_falls_back_on_no_network(self) -> None:
        """Without a checkpoint, falls back to first valid action."""
        state = UTTTState()
        agent = AlphaZeroUTTTAgent()
        action = agent.deliberate_from_state(state)
        # The _deliberate method catches RuntimeError from _make_bridge_functions
        # and returns the first valid action as fallback
        assert action is not None
        assert isinstance(action, list)
        assert len(action) == 2
        valid = state.get_valid_actions()
        assert action in valid

    def test_deliberate_from_state_with_network(self, tmp_path: Any) -> None:
        """With a loaded checkpoint, returns a valid action."""
        pytest.importorskip("torch")
        checkpoint = _create_temp_checkpoint(tmp_path)
        state = UTTTState()
        agent = AlphaZeroUTTTAgent(
            checkpoint_path=checkpoint,
            mcts_iterations=50,
            random_seed=42,
        )
        action = agent.deliberate_from_state(state)
        assert action is not None
        assert isinstance(action, list)
        assert len(action) == 2
        valid = state.get_valid_actions()
        assert action in valid

    def test_deliberate_from_state_deterministic_with_network(
        self, tmp_path: Any
    ) -> None:
        """Same checkpoint + same seed produces the same action."""
        pytest.importorskip("torch")
        checkpoint = _create_temp_checkpoint(tmp_path)
        state = UTTTState()
        agent1 = AlphaZeroUTTTAgent(
            checkpoint_path=checkpoint,
            mcts_iterations=50,
            random_seed=42,
        )
        agent2 = AlphaZeroUTTTAgent(
            checkpoint_path=checkpoint,
            mcts_iterations=50,
            random_seed=42,
        )
        action1 = agent1.deliberate_from_state(state)
        action2 = agent2.deliberate_from_state(state)
        assert action1 == action2

    def test_deliberate_from_state_terminal_with_network(
        self, tmp_path: Any
    ) -> None:
        """Returns None for terminal state even when network is loaded."""
        pytest.importorskip("torch")
        checkpoint = _create_temp_checkpoint(tmp_path)
        macro_board = [[1, 1, 1], [0, 0, 0], [0, 0, 0]]
        state = UTTTState(macro_board=macro_board)
        assert state.is_terminal()
        agent = AlphaZeroUTTTAgent(checkpoint_path=checkpoint)
        action = agent.deliberate_from_state(state)
        assert action is None


# ============================================================================
# Stats tests (no torch required)
# ============================================================================


class TestAlphaZeroUTTTAgentStats:
    """Tests for ``get_last_stats``."""

    def test_get_last_stats_empty_initially(self) -> None:
        """Returns empty dict before any deliberation."""
        agent = AlphaZeroUTTTAgent()
        stats = agent.get_last_stats()
        assert stats == {}

    def test_get_last_stats_returns_dict(self) -> None:
        """Returns a dict after deliberation (fallback path)."""
        state = UTTTState()
        agent = AlphaZeroUTTTAgent()
        agent.deliberate_from_state(state)
        stats = agent.get_last_stats()
        assert isinstance(stats, dict)
        # When falling back (no network), stats may still be empty or contain
        # fallback info; at minimum it should be a dict
        assert "inference_elapsed" in stats or stats == {}

    def test_get_last_stats_contains_expected_keys_with_network(
        self, tmp_path: Any
    ) -> None:
        """With a loaded network, stats contain expected keys."""
        pytest.importorskip("torch")
        checkpoint = _create_temp_checkpoint(tmp_path)
        state = UTTTState()
        agent = AlphaZeroUTTTAgent(
            checkpoint_path=checkpoint,
            mcts_iterations=50,
            random_seed=42,
        )
        agent.deliberate_from_state(state)
        stats = agent.get_last_stats()
        assert "total_iterations" in stats
        assert stats["total_iterations"] > 0
        assert "tree_size" in stats
        assert "best_action_win_rate" in stats
        assert "temperature" in stats
        assert "inference_elapsed" in stats

    def test_get_last_stats_isolated_copy(self) -> None:
        """Returned dict is a copy (modifying it does not affect internal state)."""
        agent = AlphaZeroUTTTAgent()
        stats = agent.get_last_stats()
        stats["foo"] = "bar"
        assert "foo" not in agent.get_last_stats()


# ============================================================================
# Deliberate (async wrapper) - no torch required
# ============================================================================


class TestAlphaZeroUTTTAgentDeliberate:
    """Tests for the async ``deliberate`` method."""

    def test_deliberate_returns_none_for_empty_actions(self) -> None:
        """Returns None when valid_actions is empty."""
        import asyncio

        agent = AlphaZeroUTTTAgent()
        agent.player_id = 1

        async def run_test() -> None:
            result = await agent.deliberate(
                board=[[0] * 9 for _ in range(9)],
                macro_board=[[0] * 3 for _ in range(3)],
                active_macro=None,
                valid_actions=[],
            )
            assert result is None

        asyncio.run(run_test())

    def test_deliberate_returns_fallback_for_no_network(self) -> None:
        """Without a network, falls back to first valid action."""
        import asyncio

        agent = AlphaZeroUTTTAgent()
        agent.player_id = 1
        valid = [[0, 0], [1, 1]]

        async def run_test() -> None:
            result = await agent.deliberate(
                board=[[0] * 9 for _ in range(9)],
                macro_board=[[0] * 3 for _ in range(3)],
                active_macro=None,
                valid_actions=valid,
            )
            # Falls back to first valid action
            assert result == [0, 0]

        asyncio.run(run_test())


# ============================================================================
# Temperature / stochastic selection tests (no torch required)
# ============================================================================


class TestAlphaZeroUTTTAgentTemperature:
    """Tests for temperature-based move selection."""

    def test_sample_from_distribution_empty_actions(self) -> None:
        """_sample_from_distribution returns None for empty actions."""
        result = AlphaZeroUTTTAgent._sample_from_distribution({}, [])
        assert result is None

    def test_sample_from_distribution_all_zero_weights(self) -> None:
        """_sample_from_distribution returns first action when all weights zero."""
        valid = [[0, 0], [1, 1], [2, 2]]
        visit_dist = {(0, 0): 0.0, (1, 1): 0.0, (2, 2): 0.0}
        result = AlphaZeroUTTTAgent._sample_from_distribution(visit_dist, valid)
        assert result == [0, 0]

    def test_sample_from_distribution_deterministic(self) -> None:
        """With only one non-zero weight, returns that action."""
        valid = [[0, 0], [1, 1], [2, 2]]
        visit_dist = {(0, 0): 0.0, (1, 1): 10.0, (2, 2): 0.0}
        result = AlphaZeroUTTTAgent._sample_from_distribution(visit_dist, valid)
        assert result == [1, 1]

    def test_sample_from_distribution_missing_action(self) -> None:
        """Missing actions in visit_dist are treated as zero weight."""
        valid = [[0, 0], [5, 5]]
        visit_dist = {(0, 0): 10.0}  # (5,5) not present
        result = AlphaZeroUTTTAgent._sample_from_distribution(visit_dist, valid)
        assert result == [0, 0]  # the only non-zero weight


# ============================================================================
# Checkpoint creation helper (torch required)
# ============================================================================


class TestAlphaZeroUTTTAgentCheckpoint:
    """Tests for checkpoint loading functionality."""

    def test_load_checkpoint_creates_network(self, tmp_path: Any) -> None:
        """Loading a checkpoint produces a usable PolicyValueNetwork."""
        pytest.importorskip("torch")
        from engine.policy_value_network import PolicyValueNetwork

        checkpoint = _create_temp_checkpoint(tmp_path)
        agent = AlphaZeroUTTTAgent(checkpoint_path=checkpoint)
        network = agent._get_network()
        assert isinstance(network, PolicyValueNetwork)
        # Check the network can do a forward pass
        import torch

        dummy_input = torch.zeros(1, 3, 9, 9)
        policy, value = network(dummy_input)
        assert policy.shape == (1, 81)
        assert value.shape == (1, 1)

    def test_load_checkpoint_from_nonexistent_file_raises(self) -> None:
        """Loading a non-existent checkpoint raises ``FileNotFoundError``.

        ``torch.load`` raises ``FileNotFoundError`` directly when the file
        does not exist — the agent does not wrap this into ``RuntimeError``.
        """
        agent = AlphaZeroUTTTAgent(checkpoint_path="/nonexistent/path.pt")
        with pytest.raises(FileNotFoundError):
            agent._get_network()

    def test_make_bridge_functions_with_checkpoint(self, tmp_path: Any) -> None:
        """_make_bridge_functions returns callables when checkpoint is loaded."""
        pytest.importorskip("torch")
        checkpoint = _create_temp_checkpoint(tmp_path)
        agent = AlphaZeroUTTTAgent(checkpoint_path=checkpoint)
        prior_fn, value_fn = agent._make_bridge_functions()
        assert callable(prior_fn)
        assert callable(value_fn)

        # The functions should work with a state
        state = UTTTState()
        prior = prior_fn(state)
        assert isinstance(prior, dict)
        # All valid actions should have a prior
        valid = state.get_valid_actions()
        for x, y in valid:
            assert (x, y) in prior
            assert 0.0 <= prior[(x, y)] <= 1.0
        # Priors should sum to approximately 1
        total = sum(prior.values())
        assert abs(total - 1.0) < 1e-5

        value = value_fn(state)
        assert isinstance(value, float)
        assert -1.0 <= value <= 1.0
