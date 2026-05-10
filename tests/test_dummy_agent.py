"""Tests for agents.dummy_agent.DummyUTTTAgent."""
from typing import List

import pytest

from agents.dummy_agent import DummyUTTTAgent
from engine.game_state import UTTTState


class TestDummyUTTTAgent:
    """Tests for the DummyUTTTAgent class."""

    def test_deliberate_from_state_returns_valid_action(self, empty_state: UTTTState) -> None:
        """deliberate_from_state returns a valid action from an empty state."""
        agent = DummyUTTTAgent()
        action = agent.deliberate_from_state(empty_state)
        assert action is not None
        assert isinstance(action, list)
        assert len(action) == 2
        valid = empty_state.get_valid_actions()
        assert action in valid

    def test_deliberate_from_state_returns_none_for_terminal(self) -> None:
        """deliberate_from_state returns None for a terminal state."""
        macro_board = [[1, 1, 1], [0, 0, 0], [0, 0, 0]]
        state = UTTTState(macro_board=macro_board)
        assert state.is_terminal()
        agent = DummyUTTTAgent()
        action = agent.deliberate_from_state(state)
        assert action is None

    def test_deliberate_from_state_no_valid_actions(self) -> None:
        """deliberate_from_state returns None when no valid actions remain."""
        board = [[0] * 9 for _ in range(9)]
        macro_board = [[1, 0, 0], [0, 0, 0], [0, 0, 0]]
        # Fill all cells in macro (0,0) so no valid actions when active_macro=[0,0]
        for y in range(3):
            for x in range(3):
                board[y][x] = 1
        state = UTTTState(board=board, macro_board=macro_board, active_macro=[0, 0])
        assert len(state.get_valid_actions()) == 0
        agent = DummyUTTTAgent()
        action = agent.deliberate_from_state(state)
        assert action is None

    def test_random_behavior(self) -> None:
        """Multiple calls may return different actions (non-deterministic)."""
        agent = DummyUTTTAgent()
        state = UTTTState()
        actions: set = set()
        # Run several times; with 81 possible actions, it's extremely unlikely
        # that all calls return the same action if randomization works
        for _ in range(20):
            action = agent.deliberate_from_state(state)
            assert action is not None
            actions.add(tuple(action))
        # With 81 actions and 20 samples, we almost certainly see >1 unique action
        assert len(actions) > 1, (
            f"Expected random behavior but only got {len(actions)} unique action(s)"
        )

    def test_deliberate_from_state_mid_game(self, mid_game_state: UTTTState) -> None:
        """deliberate_from_state returns a valid action from a mid-game state."""
        agent = DummyUTTTAgent()
        action = agent.deliberate_from_state(mid_game_state)
        assert action is not None
        assert isinstance(action, list)
        assert len(action) == 2
        valid = mid_game_state.get_valid_actions()
        assert action in valid
