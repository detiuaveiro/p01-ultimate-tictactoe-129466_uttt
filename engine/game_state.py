"""
Immutable game state class for Ultimate Tic-Tac-Toe MCTS simulation.
"""

from typing import Any, List, Optional, Tuple

from engine.game_rules import (
    apply_move,
    get_valid_actions,
    get_global_winner,
)


class UTTTState:
    """
    Immutable representation of a UTTT game state for MCTS simulation.
    """

    __slots__ = (
        "_board",
        "_macro_board",
        "_active_macro",
        "_current_player",
        "_last_move",
        "_move_count",
        "_hash_cache",
    )

    def __init__(
        self,
        board: Optional[List[List[int]]] = None,
        macro_board: Optional[List[List[int]]] = None,
        active_macro: Optional[List[int]] = None,
        current_player: int = 1,
        last_move: Optional[Tuple[int, int]] = None,
        move_count: int = 0,
    ) -> None:
        """
        Initializes a new UTTTState.

        Args:
            board: 9x9 list of lists, 0=empty, 1=P1, 2=P2 (default: all zeros).
            macro_board: 3x3 list of lists, 0=ongoing, 1=P1, 2=P2, 3=draw (default: all zeros).
            active_macro: [my, mx] or None (default: None).
            current_player: 1 or 2 (default: 1).
            last_move: (x, y) or None (default: None).
            move_count: int (default: 0).
        """
        if board is not None:
            self._board = [row[:] for row in board]
        else:
            self._board = [[0] * 9 for _ in range(9)]

        if macro_board is not None:
            self._macro_board = [row[:] for row in macro_board]
        else:
            self._macro_board = [[0] * 3 for _ in range(3)]

        if active_macro is not None:
            self._active_macro = active_macro[:]
        else:
            self._active_macro = None

        self._current_player = current_player
        self._last_move = last_move
        self._move_count = move_count
        self._hash_cache: Optional[int] = None

    @property
    def board(self) -> List[List[int]]:
        """Returns a copy of the 9x9 board."""
        return [row[:] for row in self._board]

    @property
    def macro_board(self) -> List[List[int]]:
        """Returns a copy of the 3x3 macro board."""
        return [row[:] for row in self._macro_board]

    @property
    def active_macro(self) -> Optional[List[int]]:
        """Returns a copy of the active macro coordinates, or None."""
        if self._active_macro is not None:
            return self._active_macro[:]
        return None

    @property
    def current_player(self) -> int:
        """Returns the current player ID (1 or 2)."""
        return self._current_player

    @property
    def last_move(self) -> Optional[Tuple[int, int]]:
        """Returns the last move played, or None."""
        return self._last_move

    @property
    def move_count(self) -> int:
        """Returns the number of moves played so far."""
        return self._move_count

    def clone(self) -> "UTTTState":
        """
        Creates a deep copy of this state.

        Returns:
            UTTTState: A new state with copied data.
        """
        return UTTTState(
            board=self._board,
            macro_board=self._macro_board,
            active_macro=self._active_macro,
            current_player=self._current_player,
            last_move=self._last_move,
            move_count=self._move_count,
        )

    def get_valid_actions(self) -> List[List[int]]:
        """
        Returns valid actions from this state.

        Returns:
            List[List[int]]: A list of [x, y] coordinates.
        """
        return get_valid_actions(
            self._board, self._macro_board, self._active_macro
        )

    def apply_action(self, x: int, y: int) -> "UTTTState":
        """
        Applies an action and returns a new state.

        Args:
            x (int): The x-coordinate of the move.
            y (int): The y-coordinate of the move.

        Returns:
            UTTTState: The new state after applying the move.

        Raises:
            ValueError: If the action is not valid.
        """
        valid = self.get_valid_actions()
        if [x, y] not in valid:
            raise ValueError(
                f"Invalid action ({x}, {y}). Valid actions: {valid}"
            )

        new_board, new_macro_board, new_active_macro = apply_move(
            self._board, self._macro_board, self._current_player, x, y
        )

        return UTTTState(
            board=new_board,
            macro_board=new_macro_board,
            active_macro=new_active_macro,
            current_player=3 - self._current_player,
            last_move=(x, y),
            move_count=self._move_count + 1,
        )

    def is_terminal(self) -> bool:
        """
        Checks if the game has reached a terminal state.

        Returns:
            bool: True if the game is over.
        """
        return get_global_winner(self._macro_board) != 0

    def get_winner(self) -> int:
        """
        Returns the winner of the game.

        Returns:
            int: 0 (ongoing), 1 (P1 wins), 2 (P2 wins), 3 (draw).
        """
        return get_global_winner(self._macro_board)

    def __hash__(self) -> int:
        """
        Computes a hash based on the board and player state.

        Returns:
            int: The hash value (cached after first computation).
        """
        if self._hash_cache is not None:
            return self._hash_cache

        board_tuple = tuple(tuple(row) for row in self._board)
        macro_tuple = tuple(tuple(row) for row in self._macro_board)
        active_tuple = (
            tuple(self._active_macro) if self._active_macro is not None else None
        )
        self._hash_cache = hash(
            (board_tuple, macro_tuple, active_tuple, self._current_player)
        )
        return self._hash_cache

    def __eq__(self, other: Any) -> bool:
        """
        Structural equality check.

        Args:
            other (Any): The object to compare against.

        Returns:
            bool: True if the states are structurally equal.
        """
        if not isinstance(other, UTTTState):
            return NotImplemented
        return (
            self._board == other._board
            and self._macro_board == other._macro_board
            and self._active_macro == other._active_macro
            and self._current_player == other._current_player
        )

    def __str__(self) -> str:
        """
        Returns a human-readable board display.

        Returns:
            str: A string representation of the board.
        """
        lines: List[str] = []
        for row_idx, row in enumerate(self._board):
            if row_idx > 0 and row_idx % 3 == 0:
                lines.append("-" * 21)
            line_parts: List[str] = []
            for col_idx, cell in enumerate(row):
                if col_idx > 0 and col_idx % 3 == 0:
                    line_parts.append("|")
                if cell == 0:
                    line_parts.append(".")
                elif cell == 1:
                    line_parts.append("X")
                else:
                    line_parts.append("O")
            lines.append(" ".join(line_parts))
        return "\n".join(lines)

    def __repr__(self) -> str:
        """
        Returns a concise representation of the state.

        Returns:
            str: A concise string representation.
        """
        return (
            f"UTTTState(player={self._current_player}, "
            f"moves={self._move_count}, "
            f"active_macro={self._active_macro}, "
            f"terminal={self.is_terminal()})"
        )
