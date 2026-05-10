"""
Shared stateless game rules for Ultimate Tic-Tac-Toe.

All functions are pure functions that take board state as parameters
and return results without side effects or mutations.
"""

from typing import List, Optional, Tuple


def check_3x3_win(grid: List[List[int]], start_x: int, start_y: int) -> int:
    """
    Checks for a win in a specific 3x3 subset of a grid.

    Args:
        grid (List[List[int]]): The grid to check (9x9 or 3x3).
        start_x (int): The starting x-coordinate of the 3x3 subset.
        start_y (int): The starting y-coordinate of the 3x3 subset.

    Returns:
        int: The ID of the winning player (1 or 2), 0 if no winner.
    """
    for i in range(3):
        # Rows
        if (
            grid[start_y + i][start_x] != 0
            and grid[start_y + i][start_x]
            == grid[start_y + i][start_x + 1]
            == grid[start_y + i][start_x + 2]
        ):
            return grid[start_y + i][start_x]
        # Cols
        if (
            grid[start_y][start_x + i] != 0
            and grid[start_y][start_x + i]
            == grid[start_y + 1][start_x + i]
            == grid[start_y + 2][start_x + i]
        ):
            return grid[start_y][start_x + i]
    # Diagonals
    if (
        grid[start_y][start_x] != 0
        and grid[start_y][start_x]
        == grid[start_y + 1][start_x + 1]
        == grid[start_y + 2][start_x + 2]
    ):
        return grid[start_y][start_x]
    if (
        grid[start_y + 2][start_x] != 0
        and grid[start_y + 2][start_x]
        == grid[start_y + 1][start_x + 1]
        == grid[start_y][start_x + 2]
    ):
        return grid[start_y + 2][start_x]
    return 0


def is_3x3_full(grid: List[List[int]], start_x: int, start_y: int) -> bool:
    """
    Checks if a 3x3 subset of a grid is full.

    Args:
        grid (List[List[int]]): The grid to check.
        start_x (int): The starting x-coordinate of the 3x3 subset.
        start_y (int): The starting y-coordinate of the 3x3 subset.

    Returns:
        bool: True if the 3x3 subset is full, False otherwise.
    """
    for y in range(3):
        for x in range(3):
            if grid[start_y + y][start_x + x] == 0:
                return False
    return True


def get_global_winner(macro_board: List[List[int]]) -> int:
    """
    Determines the global winner of the game based on the macro board.

    Args:
        macro_board (List[List[int]]): 3x3 macro board.
            0=ongoing, 1=P1 win, 2=P2 win, 3=draw.

    Returns:
        int: 0 (ongoing), 1 (P1 wins), 2 (P2 wins), 3 (draw).
    """
    winner = check_3x3_win(macro_board, 0, 0)
    if winner in [1, 2]:
        return winner
    if is_3x3_full(macro_board, 0, 0):
        return 3
    return 0


def get_valid_actions(
    board: List[List[int]],
    macro_board: List[List[int]],
    active_macro: Optional[List[int]],
) -> List[List[int]]:
    """
    Returns a list of all currently valid actions.

    Args:
        board (List[List[int]]): 9x9 board state (0=empty, 1=P1, 2=P2).
        macro_board (List[List[int]]): 3x3 macro board (0=ongoing, 1=P1, 2=P2, 3=draw).
        active_macro (Optional[List[int]]): [my, mx] of active macro-board, or None for free move.

    Returns:
        List[List[int]]: A list of [x, y] coordinates representing valid moves.
    """
    actions: List[List[int]] = []
    for y in range(9):
        for x in range(9):
            my, mx = y // 3, x // 3
            # Cannot play in a resolved macro-board
            if macro_board[my][mx] != 0:
                continue
            # Cannot play in an occupied cell
            if board[y][x] != 0:
                continue
            # Must play in the active macro-board, unless free move is granted
            if active_macro is not None and active_macro != [my, mx]:
                continue

            actions.append([x, y])
    return actions


def apply_move(
    board: List[List[int]],
    macro_board: List[List[int]],
    player_id: int,
    x: int,
    y: int,
) -> Tuple[List[List[int]], List[List[int]], Optional[List[int]]]:
    """
    Applies a move to the board and returns the new state.

    This is a pure function — it does NOT mutate the inputs.

    Args:
        board (List[List[int]]): 9x9 board state.
        macro_board (List[List[int]]): 3x3 macro board state.
        player_id (int): The ID of the player making the move (1 or 2).
        x (int): The global x-coordinate of the move.
        y (int): The global y-coordinate of the move.

    Returns:
        Tuple[List[List[int]], List[List[int]], Optional[List[int]]]:
            (new_board, new_macro_board, new_active_macro).
    """
    # Deep copy inputs
    new_board = [row[:] for row in board]
    new_macro_board = [row[:] for row in macro_board]

    # Place the piece
    new_board[y][x] = player_id

    # Compute macro and micro positions
    my, mx = y // 3, x // 3
    micro_y, micro_x = y % 3, x % 3

    # Check if this move won the local macro-board
    local_winner = check_3x3_win(new_board, mx * 3, my * 3)
    if local_winner:
        new_macro_board[my][mx] = local_winner
    elif is_3x3_full(new_board, mx * 3, my * 3):
        new_macro_board[my][mx] = 3  # Draw

    # Determine next active macro-board
    next_my, next_mx = micro_y, micro_x
    if new_macro_board[next_my][next_mx] != 0:
        new_active_macro: Optional[List[int]] = None  # Free move!
    else:
        new_active_macro = [next_my, next_mx]

    return new_board, new_macro_board, new_active_macro
