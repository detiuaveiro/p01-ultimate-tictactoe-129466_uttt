"""
Tests for agents.mcts_agent.MCTSAgent.
"""

from typing import List

import pytest

from agents.mcts_agent import MCTSAgent
from engine.game_state import UTTTState


class TestMCTSAgent:
    """Tests for the MCTSAgent class."""

    def test_construction_defaults(self) -> None:
        """MCTSAgent can be constructed with default parameters."""
        agent = MCTSAgent()
        assert agent.server_uri == "ws://localhost:8765"
        assert agent.mcts_iterations == 10000
        assert agent.mcts_exploration_constant == 1.414
        assert agent.mcts_time_limit is None
        assert agent.random_seed is None

    def test_construction_custom(self) -> None:
        """MCTSAgent can be constructed with custom parameters."""
        agent = MCTSAgent(
            server_uri="ws://example.com:9000",
            mcts_iterations=500,
            mcts_exploration_constant=2.0,
            mcts_time_limit=0.5,
            random_seed=123,
        )
        assert agent.server_uri == "ws://example.com:9000"
        assert agent.mcts_iterations == 500
        assert agent.mcts_exploration_constant == 2.0
        assert agent.mcts_time_limit == 0.5
        assert agent.random_seed == 123

    def test_deliberate_from_state_returns_valid_action(self) -> None:
        """deliberate_from_state returns a valid action from a non-terminal state."""
        state = UTTTState()
        agent = MCTSAgent(mcts_iterations=100, random_seed=42)
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
        agent = MCTSAgent(mcts_iterations=100, random_seed=42)
        action = agent.deliberate_from_state(state)
        assert action is None

    def test_deliberate_from_state_no_valid_actions(self) -> None:
        """deliberate_from_state returns None when no valid actions."""
        board = [[0] * 9 for _ in range(9)]
        macro_board = [[1, 0, 0], [0, 0, 0], [0, 0, 0]]
        # Fill all cells in macro (0,0)
        for y in range(3):
            for x in range(3):
                board[y][x] = 1
        state = UTTTState(board=board, macro_board=macro_board, active_macro=[0, 0])
        assert len(state.get_valid_actions()) == 0
        agent = MCTSAgent(mcts_iterations=100, random_seed=42)
        action = agent.deliberate_from_state(state)
        assert action is None

    def test_deterministic_with_fixed_seed(self) -> None:
        """MCTSAgent with the same seed produces the same result."""
        state = UTTTState()
        agent1 = MCTSAgent(mcts_iterations=200, random_seed=42)
        agent2 = MCTSAgent(mcts_iterations=200, random_seed=42)
        action1 = agent1.deliberate_from_state(state)
        action2 = agent2.deliberate_from_state(state)
        assert action1 == action2

    def test_get_last_stats(self) -> None:
        """get_last_stats returns stats after deliberation."""
        state = UTTTState()
        agent = MCTSAgent(mcts_iterations=100, random_seed=42)
        agent.deliberate_from_state(state)
        stats = agent.get_last_stats()
        assert "total_iterations" in stats
        assert stats["total_iterations"] > 0

    def test_deliberate_returns_none_for_empty_actions(self) -> None:
        """deliberate returns None when valid_actions is empty."""
        import asyncio

        agent = MCTSAgent(mcts_iterations=100, random_seed=42)
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
