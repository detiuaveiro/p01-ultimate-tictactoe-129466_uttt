"""
Tests for engine.game_rules pure functions.
"""

from typing import List, Optional

from engine.game_rules import (
    apply_move,
    check_3x3_win,
    get_global_winner,
    get_valid_actions,
    is_3x3_full,
)


# ---------------------------------------------------------------------------
# check_3x3_win
# ---------------------------------------------------------------------------

def test_check_3x3_win_row() -> None:
    """A complete row in a 3x3 grid detects the winner."""
    grid = [[0] * 9 for _ in range(9)]
    # Fill row 0 in macro (0,0)
    grid[0][0] = 1
    grid[0][1] = 1
    grid[0][2] = 1
    assert check_3x3_win(grid, 0, 0) == 1


def test_check_3x3_win_col() -> None:
    """A complete column in a 3x3 grid detects the winner."""
    grid = [[0] * 9 for _ in range(9)]
    grid[0][0] = 2
    grid[1][0] = 2
    grid[2][0] = 2
    assert check_3x3_win(grid, 0, 0) == 2


def test_check_3x3_win_diag() -> None:
    """A complete diagonal in a 3x3 grid detects the winner."""
    grid = [[0] * 9 for _ in range(9)]
    # Main diagonal macro (1,1) starting at (3,3)
    grid[3][3] = 1
    grid[4][4] = 1
    grid[5][5] = 1
    assert check_3x3_win(grid, 3, 3) == 1

    # Anti-diagonal
    grid2 = [[0] * 9 for _ in range(9)]
    grid2[5][3] = 2
    grid2[4][4] = 2
    grid2[3][5] = 2
    assert check_3x3_win(grid2, 3, 3) == 2


def test_check_3x3_win_no_winner() -> None:
    """No winning line returns 0."""
    grid = [[0] * 9 for _ in range(9)]
    grid[0][0] = 1
    grid[0][1] = 2
    grid[0][2] = 1
    assert check_3x3_win(grid, 0, 0) == 0


# ---------------------------------------------------------------------------
# is_3x3_full
# ---------------------------------------------------------------------------

def test_is_3x3_full_filled() -> None:
    """A completely filled 3x3 grid returns True."""
    grid = [[0] * 9 for _ in range(9)]
    for dy in range(3):
        for dx in range(3):
            grid[dy][dx] = 1
    assert is_3x3_full(grid, 0, 0) is True


def test_is_3x3_not_full() -> None:
    """An incomplete 3x3 grid returns False."""
    grid = [[0] * 9 for _ in range(9)]
    grid[0][0] = 1
    grid[1][1] = 2
    assert is_3x3_full(grid, 0, 0) is False


# ---------------------------------------------------------------------------
# get_global_winner
# ---------------------------------------------------------------------------

def test_get_global_winner_row() -> None:
    """A row on the macro board detects the global winner."""
    macro = [
        [1, 1, 1],
        [0, 0, 0],
        [0, 0, 0],
    ]
    assert get_global_winner(macro) == 1


def test_get_global_winner_draw() -> None:
    """A full macro board with no winner returns 3 (draw)."""
    macro = [
        [1, 2, 1],
        [1, 1, 2],
        [2, 1, 2],
    ]
    # check_3x3_win returns 0, but is_3x3_full returns True -> draw
    assert get_global_winner(macro) == 3


def test_get_global_winner_no_winner() -> None:
    """An ongoing game with empty cells returns 0."""
    macro = [
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]
    assert get_global_winner(macro) == 0


def test_get_global_winner_ongoing_full() -> None:
    """A full macro board with a winner returns the winner (1 or 2), not 3."""
    macro = [
        [1, 2, 2],
        [1, 1, 2],
        [2, 1, 1],
    ]
    # Player 1 has anti-diagonal
    assert get_global_winner(macro) == 1


# ---------------------------------------------------------------------------
# get_valid_actions
# ---------------------------------------------------------------------------

def test_get_valid_actions_free_move() -> None:
    """With no active macro, all empty cells in unresolved macro-boards are valid."""
    board = [[0] * 9 for _ in range(9)]
    macro = [[0] * 3 for _ in range(3)]
    actions = get_valid_actions(board, macro, None)
    # All 81 cells should be valid
    assert len(actions) == 81
    assert [0, 0] in actions
    assert [8, 8] in actions


def test_get_valid_actions_constrained() -> None:
    """With an active macro, only cells in that macro are valid."""
    board = [[0] * 9 for _ in range(9)]
    macro = [[0] * 3 for _ in range(3)]
    actions = get_valid_actions(board, macro, [1, 1])
    # Only the 9 cells in macro (1,1) starting at (3,3)
    assert len(actions) == 9
    assert [3, 3] in actions
    assert [5, 5] in actions
    assert [0, 0] not in actions


def test_get_valid_actions_blocked_macro() -> None:
    """Resolved macro-boards are excluded even in free move."""
    board = [[0] * 9 for _ in range(9)]
    macro = [
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]
    actions = get_valid_actions(board, macro, None)
    # Top-left macro (0,0) is resolved (value 1) — its 9 cells excluded
    assert len(actions) == 81 - 9
    for y in range(3):
        for x in range(3):
            assert [x, y] not in actions
    assert [3, 0] in actions  # first cell outside blocked macro


def test_get_valid_actions_blocked_cells() -> None:
    """Occupied cells are excluded."""
    board = [[0] * 9 for _ in range(9)]
    board[0][0] = 1  # occupied
    macro = [[0] * 3 for _ in range(3)]
    actions = get_valid_actions(board, macro, None)
    assert len(actions) == 80
    assert [0, 0] not in actions


# ---------------------------------------------------------------------------
# apply_move
# ---------------------------------------------------------------------------

def test_apply_move_basic() -> None:
    """A basic move places the piece and sets the next active macro."""
    board = [[0] * 9 for _ in range(9)]
    macro = [[0] * 3 for _ in range(3)]
    new_board, new_macro, new_active = apply_move(board, macro, 1, 0, 0)

    # Piece placed
    assert new_board[0][0] == 1
    # Original unchanged
    assert board[0][0] == 0
    # Active macro set to [0, 0] (where the micro moved to)
    assert new_active == [0, 0]


def test_apply_move_local_win() -> None:
    """A move that wins a macro-board updates the macro board."""
    board = [[0] * 9 for _ in range(9)]
    # Fill row 0 of macro (0,0) with player 1 except last cell
    board[0][0] = 1
    board[0][1] = 1
    # board[0][2] is empty — we'll play there
    macro = [[0] * 3 for _ in range(3)]

    new_board, new_macro, new_active = apply_move(board, macro, 1, 2, 0)

    assert new_board[0][2] == 1
    # Macro (0,0) should now show player 1 win
    assert new_macro[0][0] == 1
    # The micro position is (0, 2) which maps to macro (0, 2).
    # Macro (0, 2) is still unresolved, so active_macro = [0, 2].
    assert new_active == [0, 2]


def test_apply_move_free_move() -> None:
    """A move to a cell whose target macro is resolved grants a free move."""
    board = [[0] * 9 for _ in range(9)]
    macro = [[0] * 3 for _ in range(3)]
    # Resolve macro (0, 0)
    macro[0][0] = 1

    # Play in macro (1,1) at global (3,3). micro is (0,0) which maps to macro (0,0) which is resolved
    new_board, new_macro, new_active = apply_move(board, macro, 1, 3, 3)

    assert new_board[3][3] == 1
    # Active macro should be None because the target macro (0,0) is resolved
    assert new_active is None


def test_apply_move_turn_unchanged() -> None:
    """apply_move is stateless — it doesn't track whose turn it is."""
    board = [[0] * 9 for _ in range(9)]
    macro = [[0] * 3 for _ in range(3)]
    _, _, new_active = apply_move(board, macro, 2, 5, 5)
    # Only check that the move was applied and active_macro is correct
    # The caller (UTTTState) handles turn switching
    # micro (5%3, 5%3) = (2, 2) which maps to macro (2, 2)
    assert new_active == [2, 2]

def test_apply_move_immutability() -> None:
    """Original board and macro_board are not mutated."""
    board = [[0] * 9 for _ in range(9)]
    macro = [[0] * 3 for _ in range(3)]
    board_copy = [row[:] for row in board]
    macro_copy = [row[:] for row in macro]

    apply_move(board, macro, 1, 4, 4)

    assert board == board_copy
    assert macro == macro_copy
