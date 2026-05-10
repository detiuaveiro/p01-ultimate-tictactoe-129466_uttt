"""
Tests for engine.game_state.UTTTState.
"""

from typing import List

import pytest

from engine.game_state import UTTTState


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_empty_state_construction() -> None:
    """An empty state has all zeros and player 1 to move."""
    state = UTTTState()
    assert state.board == [[0] * 9 for _ in range(9)]
    assert state.macro_board == [[0] * 3 for _ in range(3)]
    assert state.active_macro is None
    assert state.current_player == 1
    assert state.last_move is None
    assert state.move_count == 0
    assert state.is_terminal() is False
    assert state.get_winner() == 0


def test_custom_construction() -> None:
    """Construction with custom parameters works."""
    board = [[0] * 9 for _ in range(9)]
    board[0][0] = 1
    macro = [[0] * 3 for _ in range(3)]
    macro[0][0] = 1
    state = UTTTState(
        board=board,
        macro_board=macro,
        active_macro=[1, 1],
        current_player=2,
        last_move=(4, 4),
        move_count=5,
    )
    assert state.board[0][0] == 1
    assert state.macro_board[0][0] == 1
    assert state.active_macro == [1, 1]
    assert state.current_player == 2
    assert state.last_move == (4, 4)
    assert state.move_count == 5


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------

def test_board_property_returns_copy(empty_state: UTTTState) -> None:
    """The board property returns a copy, not the internal list."""
    board_copy = empty_state.board
    board_copy[0][0] = 99
    # Original should be unchanged
    assert empty_state.board[0][0] == 0


def test_macro_board_property_returns_copy(empty_state: UTTTState) -> None:
    """The macro_board property returns a copy."""
    macro_copy = empty_state.macro_board
    macro_copy[0][0] = 99
    assert empty_state.macro_board[0][0] == 0


def test_active_macro_property_returns_copy(empty_state: UTTTState) -> None:
    """The active_macro property returns a copy."""
    # Create state with active_macro set
    state = UTTTState(active_macro=[1, 2])
    am = state.active_macro
    assert am is not None
    am[0] = 99
    assert state.active_macro == [1, 2]


# ---------------------------------------------------------------------------
# Hash and equality
# ---------------------------------------------------------------------------

def test_hash_consistency() -> None:
    """Equal states produce the same hash."""
    s1 = UTTTState()
    s2 = UTTTState()
    assert hash(s1) == hash(s2)


def test_equality() -> None:
    """Structurally equal states are equal."""
    s1 = UTTTState()
    s2 = UTTTState()
    assert s1 == s2
    assert not (s1 != s2)


def test_inequality() -> None:
    """Different states are not equal."""
    s1 = UTTTState()
    s2 = s1.apply_action(0, 0)
    assert s1 != s2


def test_equality_with_non_state() -> None:
    """Equality with non-UTTTState returns NotImplemented."""
    s = UTTTState()
    assert s.__eq__("not_a_state") is NotImplemented


# ---------------------------------------------------------------------------
# apply_action
# ---------------------------------------------------------------------------

def test_apply_action_basic() -> None:
    """Applying an action returns a new state with updated board and toggled player."""
    state = UTTTState()
    new_state = state.apply_action(4, 4)
    # Original unchanged
    assert state.board[4][4] == 0
    assert state.current_player == 1
    # New state updated
    assert new_state.board[4][4] == 1
    assert new_state.current_player == 2
    assert new_state.last_move == (4, 4)
    assert new_state.move_count == 1
    assert new_state.active_macro == [1, 1]  # micro (1,1) maps to macro (1,1)


def test_apply_action_invalid_raises() -> None:
    """Applying an invalid action raises ValueError."""
    state = UTTTState()
    with pytest.raises(ValueError):
        state.apply_action(-1, -1)


def test_apply_action_occupied_raises() -> None:
    """Applying an action to an occupied cell raises ValueError."""
    state = UTTTState()
    state = state.apply_action(0, 0)
    with pytest.raises(ValueError):
        state.apply_action(0, 0)


# ---------------------------------------------------------------------------
# get_valid_actions
# ---------------------------------------------------------------------------

def test_valid_actions_empty() -> None:
    """An empty board has 81 valid actions."""
    state = UTTTState()
    actions = state.get_valid_actions()
    assert len(actions) == 81


def test_valid_actions_delegates_to_rules(empty_state: UTTTState) -> None:
    """get_valid_actions delegates to game_rules.get_valid_actions."""
    actions = empty_state.get_valid_actions()
    assert isinstance(actions, list)
    assert all(isinstance(a, list) and len(a) == 2 for a in actions)


# ---------------------------------------------------------------------------
# Terminal / Winner
# ---------------------------------------------------------------------------

def test_terminal_state() -> None:
    """A terminal state is detected correctly."""
    state = UTTTState()
    assert state.is_terminal() is False
    assert state.get_winner() == 0


def test_clone(mid_game_state: UTTTState) -> None:
    """clone creates an independent deep copy."""
    clone = mid_game_state.clone()
    assert clone == mid_game_state
    # Apply a valid action on clone should not affect original
    valid = clone.get_valid_actions()
    assert len(valid) > 0
    clone2 = clone.apply_action(valid[0][0], valid[0][1])
    assert clone2 != mid_game_state
    # Original should have same move count
    assert mid_game_state.move_count == 4


# ---------------------------------------------------------------------------
# __str__ and __repr__
# ---------------------------------------------------------------------------

def test_str_representation(empty_state: UTTTState) -> None:
    """__str__ produces a readable board."""
    s = str(empty_state)
    assert "." in s  # empty cells
    assert "|" in s  # macro separators


def test_repr_representation(empty_state: UTTTState) -> None:
    """__repr__ produces a concise representation."""
    r = repr(empty_state)
    assert "UTTTState" in r
    assert "player=1" in r
