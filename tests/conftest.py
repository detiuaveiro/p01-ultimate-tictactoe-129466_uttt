"""
Shared test fixtures for UTTT tests.
"""

from typing import List, Optional

import pytest

from engine.game_state import UTTTState


@pytest.fixture
def empty_state() -> UTTTState:
    """Returns an empty initial game state."""
    return UTTTState()


@pytest.fixture
def mid_game_state() -> UTTTState:
    """
    Returns a mid-game state with some moves already played.
    Player 1 has played at (0,0), (4,4); Player 2 at (1,1), (3,3).
    """
    state = UTTTState()
    state = state.apply_action(0, 0)  # P1 at top-left
    state = state.apply_action(1, 1)  # P2
    state = state.apply_action(4, 4)  # P1 at center of center macro
    state = state.apply_action(3, 3)  # P2
    return state
