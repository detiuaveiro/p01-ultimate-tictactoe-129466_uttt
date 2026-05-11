"""
Tests for agents.mcts_heuristic_agent.MCTSHeuristicAgent.
"""

from typing import Any, Dict, Optional

import pytest

from agents.mcts_heuristic_agent import MCTSHeuristicAgent
from engine.game_state import UTTTState
from engine.heuristics import HeuristicEvaluator


class TestMCTSHeuristicAgent:
    """Tests for the MCTSHeuristicAgent class."""

    # -- Constructor -------------------------------------------------------

    def test_constructor_defaults(self) -> None:
        """MCTSHeuristicAgent can be constructed with default parameters."""
        agent = MCTSHeuristicAgent()
        assert agent.server_uri == "ws://localhost:8765"
        assert agent.mcts_iterations == 10000
        assert agent.mcts_exploration_constant == 1.414
        assert agent.mcts_time_limit is None
        assert agent.random_seed is None
        assert agent.heuristic_weights is None
        assert agent.heuristic_playout_bias == 0.8
        assert agent.heuristic_max_depth == 50
        assert agent._evaluator is None  # lazy init

    def test_constructor_custom(self) -> None:
        """MCTSHeuristicAgent can be constructed with custom parameters."""
        agent = MCTSHeuristicAgent(
            server_uri="ws://example.com:9000",
            mcts_iterations=500,
            mcts_exploration_constant=2.0,
            mcts_time_limit=0.5,
            random_seed=123,
            heuristic_weights={"micro_win": 150.0},
            heuristic_playout_bias=0.9,
            heuristic_max_depth=30,
        )
        assert agent.server_uri == "ws://example.com:9000"
        assert agent.mcts_iterations == 500
        assert agent.mcts_exploration_constant == 2.0
        assert agent.mcts_time_limit == 0.5
        assert agent.random_seed == 123
        assert agent.heuristic_weights == {"micro_win": 150.0}
        assert agent.heuristic_playout_bias == 0.9
        assert agent.heuristic_max_depth == 30

    def test_constructor_invalid_bias_raises(self) -> None:
        """Invalid heuristic_playout_bias raises AssertionError."""
        with pytest.raises(AssertionError):
            MCTSHeuristicAgent(heuristic_playout_bias=-0.1)
        with pytest.raises(AssertionError):
            MCTSHeuristicAgent(heuristic_playout_bias=1.1)

    def test_constructor_invalid_depth_raises(self) -> None:
        """Negative heuristic_max_depth raises AssertionError."""
        with pytest.raises(AssertionError):
            MCTSHeuristicAgent(heuristic_max_depth=-1)

    def test_constructor_zero_depth_ok(self) -> None:
        """heuristic_max_depth=0 is valid (immediate leaf eval)."""
        agent = MCTSHeuristicAgent(heuristic_max_depth=0)
        assert agent.heuristic_max_depth == 0

    # -- Lazy evaluator ----------------------------------------------------

    def test_lazy_evaluator_init(self) -> None:
        """_get_evaluator creates evaluator on first access."""
        agent = MCTSHeuristicAgent()
        assert agent._evaluator is None
        evaluator = agent._get_evaluator()
        assert evaluator is not None
        assert isinstance(evaluator, HeuristicEvaluator)
        assert agent._evaluator is evaluator  # cached

    def test_lazy_evaluator_custom_weights(self) -> None:
        """_get_evaluator uses custom weights when provided."""
        weights = {"micro_win": 200.0}
        agent = MCTSHeuristicAgent(heuristic_weights=weights)
        evaluator = agent._get_evaluator()
        assert evaluator._weights["micro_win"] == 200.0

    def test_lazy_evaluator_singleton(self) -> None:
        """_get_evaluator always returns the same instance."""
        agent = MCTSHeuristicAgent()
        e1 = agent._get_evaluator()
        e2 = agent._get_evaluator()
        assert e1 is e2

    # -- Deliberation ------------------------------------------------------

    def test_deliberate_from_state_returns_valid_action(self) -> None:
        """deliberate_from_state returns a valid action from a non-terminal state."""
        state = UTTTState()
        agent = MCTSHeuristicAgent(mcts_iterations=100, random_seed=42)
        action = agent.deliberate_from_state(state)
        assert action is not None
        assert isinstance(action, list)
        assert len(action) == 2
        valid = state.get_valid_actions()
        assert action in valid

    def test_deliberate_from_state_terminal(self) -> None:
        """deliberate_from_state returns None for a terminal state."""
        macro_board = [[1, 1, 1], [0, 0, 0], [0, 0, 0]]
        state = UTTTState(macro_board=macro_board)
        assert state.is_terminal()
        agent = MCTSHeuristicAgent(mcts_iterations=100, random_seed=42)
        action = agent.deliberate_from_state(state)
        assert action is None

    def test_deliberate_from_state_no_valid_actions(self) -> None:
        """deliberate_from_state returns None when no valid actions."""
        board = [[0] * 9 for _ in range(9)]
        macro_board = [[1, 0, 0], [0, 0, 0], [0, 0, 0]]
        for y in range(3):
            for x in range(3):
                board[y][x] = 1
        state = UTTTState(board=board, macro_board=macro_board, active_macro=[0, 0])
        assert len(state.get_valid_actions()) == 0
        agent = MCTSHeuristicAgent(mcts_iterations=100, random_seed=42)
        action = agent.deliberate_from_state(state)
        assert action is None

    def test_deterministic_with_fixed_seed(self) -> None:
        """Same seed produces the same result."""
        state = UTTTState()
        agent1 = MCTSHeuristicAgent(mcts_iterations=200, random_seed=42)
        agent2 = MCTSHeuristicAgent(mcts_iterations=200, random_seed=42)
        action1 = agent1.deliberate_from_state(state)
        action2 = agent2.deliberate_from_state(state)
        assert action1 == action2

    def test_get_last_stats(self) -> None:
        """get_last_stats returns stats after deliberation."""
        state = UTTTState()
        agent = MCTSHeuristicAgent(mcts_iterations=100, random_seed=42)
        agent.deliberate_from_state(state)
        stats = agent.get_last_stats()
        assert "total_iterations" in stats
        assert stats["total_iterations"] > 0

    def test_get_last_stats_empty_initially(self) -> None:
        """get_last_stats returns empty dict before first deliberation."""
        agent = MCTSHeuristicAgent()
        stats = agent.get_last_stats()
        assert stats == {}

    # -- Playout factory ---------------------------------------------------

    def test_make_playout_fn_returns_callable(self) -> None:
        """_make_playout_fn returns a callable."""
        agent = MCTSHeuristicAgent()
        playout_fn = agent._make_playout_fn()
        assert callable(playout_fn)

    def test_make_playout_fn_returns_winner(self) -> None:
        """The playout function returns a valid winner (1, 2, or 3)."""
        import random

        agent = MCTSHeuristicAgent()
        playout_fn = agent._make_playout_fn()
        state = UTTTState()
        rng = random.Random(42)
        winner = playout_fn(state, rng)
        assert winner in (1, 2, 3)

    # -- Config propagation ------------------------------------------------

    def test_config_propagation_mcts(self) -> None:
        """MCTS config values are passed through to the MCTS instance."""
        agent = MCTSHeuristicAgent(
            mcts_iterations=500,
            mcts_exploration_constant=2.0,
            mcts_time_limit=0.1,
            random_seed=42,
        )
        state = UTTTState()
        # We can verify by checking that deliberation uses these values
        # (indirectly: the search should complete without error)
        action = agent.deliberate_from_state(state)
        assert action is not None
        assert len(action) == 2

    # -- Deliberate (async wrapper) ----------------------------------------

    def test_deliberate_returns_none_for_empty_actions(self) -> None:
        """deliberate returns None when valid_actions is empty."""
        import asyncio

        agent = MCTSHeuristicAgent(mcts_iterations=100, random_seed=42)
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
