"""
Heuristic evaluation functions for Ultimate Tic-Tac-Toe.

Provides weighted board-state evaluation for guiding MCTS rollouts,
heuristic-biased move selection, and early-cutoff leaf evaluation.
"""

import math
import random
from typing import Any, Callable, Dict, List, Optional

from engine.game_state import UTTTState

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The 8 win-lines on a 3x3 grid (rows, columns, diagonals)
WIN_LINES: List[List[tuple[int, int]]] = [
    [(0, 0), (0, 1), (0, 2)],  # row 0
    [(1, 0), (1, 1), (1, 2)],  # row 1
    [(2, 0), (2, 1), (2, 2)],  # row 2
    [(0, 0), (1, 0), (2, 0)],  # col 0
    [(0, 1), (1, 1), (2, 1)],  # col 1
    [(0, 2), (1, 2), (2, 2)],  # col 2
    [(0, 0), (1, 1), (2, 2)],  # diagonal
    [(0, 2), (1, 1), (2, 0)],  # anti-diagonal
]

# Threshold for considering a heuristic score a "win" for leaf evaluation
LEAF_WIN_THRESHOLD: float = 50.0


# ---------------------------------------------------------------------------
# HeuristicEvaluator
# ---------------------------------------------------------------------------


class HeuristicEvaluator:
    """
    Weighted heuristic evaluation for Ultimate Tic-Tac-Toe board positions.

    Scores board states from the perspective of a given player using a
    configurable set of strategic features and weights.  Designed for use
    as a drop-in leaf evaluator / move scorer in MCTS rollouts.

    The evaluator is **stateless** during ``evaluate()`` — no instance state
    is modified during evaluation, making it safe for concurrent use.
    """

    DEFAULT_WEIGHTS: Dict[str, float] = {
        "micro_win": 100.0,
        "macro_threat": 200.0,
        "block_macro_threat": 150.0,
        "center_macro": 10.0,
        "corner_macro": 3.0,
        "center_micro": 3.0,
        "free_move": 2.0,
        "micro_threat": 5.0,
        "block_micro_threat": 20.0,
    }

    def __init__(
        self, weights: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Initializes the HeuristicEvaluator with optional custom weights.

        Provided weights are merged with defaults: missing keys use the
        default value; unknown keys are silently ignored.

        Args:
            weights: Optional dictionary of feature weights to override
                defaults.  If ``None``, all defaults are used.
        """
        self._weights: Dict[str, float] = dict(self.DEFAULT_WEIGHTS)
        if weights is not None:
            for key in self._weights:
                if key in weights:
                    self._weights[key] = weights[key]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, state: UTTTState, player_id: int) -> float:
        """
        Evaluates a board state from the perspective of *player_id*.

        Terminal states return ``+inf`` (won), ``-inf`` (lost), or ``0``
        (draw).  Non-terminal states return a weighted sum of strategic
        features.  The score is positive when *player_id* is ahead and
        negative when behind.

        Args:
            state: The game state to evaluate.
            player_id: The player to evaluate from (1 or 2).

        Returns:
            float: The heuristic score.

        Raises:
            ValueError: If *player_id* is not 1 or 2.
        """
        if player_id not in (1, 2):
            raise ValueError(f"player_id must be 1 or 2, got {player_id}")

        if state.is_terminal():
            winner = state.get_winner()
            if winner == 3:
                return 0.0  # draw
            if winner == player_id:
                return float("inf")  # win
            return float("-inf")  # loss

        # Weighted sum of features
        score: float = 0.0
        score += self._weights["micro_win"] * self._score_micro_wins(state, player_id)
        score += self._weights["macro_threat"] * self._score_macro_threats(state, player_id)
        score += self._weights["block_macro_threat"] * self._score_block_macro_threats(
            state, player_id
        )
        score += self._weights["center_macro"] * self._score_center_macro(state, player_id)
        score += self._weights["corner_macro"] * self._score_corner_macros(state, player_id)
        score += self._weights["center_micro"] * self._score_center_micros(state, player_id)
        score += self._weights["free_move"] * self._score_free_move(state, player_id)
        score += self._weights["micro_threat"] * self._score_micro_threats(state, player_id)
        score += self._weights["block_micro_threat"] * self._score_block_micro_threats(
            state, player_id
        )
        return score

    def score_move(
        self,
        board: List[List[int]],
        macro_board: List[List[int]],
        active_macro: Optional[List[int]],
        current_player: int,
        x: int,
        y: int,
        player_id: int,
    ) -> float:
        """
        Lightweight scoring of a single candidate move (~5μs).

        Evaluates the quality of a move from the perspective of *player_id*
        by examining immediate tactical consequences *before* the move is
        actually applied.  This is much faster than cloning the state and
        calling ``evaluate()``.

        Args:
            board: The 9x9 board state before the move.
            macro_board: The 3x3 macro board before the move.
            active_macro: The active macro-board (or ``None`` for free move).
            current_player: The player making the move (1 or 2).
            x: Global x-coordinate of the candidate move.
            y: Global y-coordinate of the candidate move.
            player_id: The evaluating player (1 or 2).

        Returns:
            float: A heuristic score for the candidate move.
        """
        my, mx = y // 3, x // 3
        micro_y, micro_x = y % 3, x % 3

        score: float = 0.0

        # --- Immediate micro-board win? ---
        # Simulate the move on a local copy to check for a win
        # We only check the 3x3 subgrid; full state cloning is expensive.
        subgrid = [
            row[mx * 3 : mx * 3 + 3]
            for row in board[my * 3 : my * 3 + 3]
        ]
        subgrid[micro_y][micro_x] = current_player
        if self._check_3x3_winner(subgrid) == current_player:
            score += self._weights["micro_win"]

        # --- Blocks opponent micro-board win? ---
        # Check if opponent would have won in this cell
        subgrid_opp = [
            row[mx * 3 : mx * 3 + 3]
            for row in board[my * 3 : my * 3 + 3]
        ]
        opponent = 3 - current_player
        subgrid_opp[micro_y][micro_x] = opponent
        if self._check_3x3_winner(subgrid_opp) == opponent:
            score += self._weights["block_micro_threat"]

        # --- Creates a 2-in-a-row in micro-board? ---
        # Check if the move creates a threat for the current player
        player_count = 0
        empty_count = 0
        for line in WIN_LINES:
            cells_in_line = [subgrid[dy][dx] for dy, dx in line]
            pc = sum(1 for c in cells_in_line if c == current_player)
            ec = sum(1 for c in cells_in_line if c == 0)
            if pc == 2 and ec == 1:
                player_count += 1
            # Also break opponent's 2-in-a-row
            oc = sum(1 for c in cells_in_line if c == opponent)
            if oc == 2 and ec == 1:
                empty_count += 1

        # Each new 2-in-a-row (before the move there were none at this position)
        # is a rough heuristic
        if player_count > 0:
            score += self._weights["micro_threat"] * player_count
        if empty_count > 0:
            # Opponent was about to get 2-in-a-row; blocking it is good
            score += self._weights["block_micro_threat"] * empty_count  # type: ignore[operator]

        # --- Center cell control ---
        if [micro_y, micro_x] == [1, 1]:
            score += self._weights["center_micro"]

        # --- Sends opponent to a resolved board? ---
        next_my, next_mx = micro_y, micro_x
        if macro_board[next_my][next_mx] != 0:
            score += self._weights["free_move"]

        # --- Target board already owned? ---
        if macro_board[my][mx] != 0:
            # Already resolved — playing here shouldn't happen in normal play.
            # Subtle: this could be a "waste" move, so penalize slightly.
            score -= 1.0

        return score

    def heuristic_playout(
        self,
        state: UTTTState,
        rng: random.Random,
        max_depth: int = 50,
        playout_bias: float = 0.8,
    ) -> int:
        """
        Epsilon-greedy heuristic playout from a given state.

        At each step, with probability *playout_bias*, the best move
        according to ``score_move()`` is selected; otherwise a random
        move is played.  The playout terminates at a terminal state or
        when *max_depth* is reached (in which case leaf evaluation is
        used to determine the winner via ``_leaf_to_winner()``).

        Args:
            state: The game state to simulate from.
            rng: A seeded ``random.Random`` instance for reproducibility.
            max_depth: Maximum number of moves to simulate (default 50).
                Must be >= 0.
            playout_bias: Probability of selecting the heuristic-best
                move vs. a random move (default 0.8).  Must be in [0, 1].

        Returns:
            int: The winner (1, 2, or 3 for draw).
        """
        current_state = state
        depth = 0

        while not current_state.is_terminal() and depth < max_depth:
            actions = current_state.get_valid_actions()
            if not actions:
                break

            if rng.random() < playout_bias and len(actions) > 1:
                # Pick the best heuristic move
                best_action = max(
                    actions,
                    key=lambda a: self.score_move(
                        board=current_state._board,
                        macro_board=current_state._macro_board,
                        active_macro=current_state._active_macro,
                        current_player=current_state._current_player,
                        x=a[0],
                        y=a[1],
                        player_id=current_state._current_player,
                    ),
                )
                action = best_action
            else:
                action = rng.choice(actions)

            current_state = current_state.apply_action(action[0], action[1])
            depth += 1

        # Reached terminal or max_depth
        if current_state.is_terminal():
            return current_state.get_winner()

        # Leaf evaluation
        last_mover = 3 - current_state._current_player
        score = self.evaluate(current_state, last_mover)
        return self._leaf_to_winner(score, last_mover)

    def get_feature_breakdown(
        self, state: UTTTState, player_id: int
    ) -> Dict[str, float]:
        """
        Returns the raw contribution of each feature (before weighting).

        This is useful for debugging, tuning weights, or understanding
        why the evaluator assigned a particular score.

        Args:
            state: The game state to evaluate.
            player_id: The player to evaluate from (1 or 2).

        Returns:
            Dict[str, float]: Feature name → raw score contribution
            (pre-weight multiplication).
        """
        return {
            "micro_win": self._score_micro_wins(state, player_id),
            "macro_threat": self._score_macro_threats(state, player_id),
            "block_macro_threat": self._score_block_macro_threats(state, player_id),
            "center_macro": self._score_center_macro(state, player_id),
            "corner_macro": self._score_corner_macros(state, player_id),
            "center_micro": self._score_center_micros(state, player_id),
            "free_move": self._score_free_move(state, player_id),
            "micro_threat": self._score_micro_threats(state, player_id),
            "block_micro_threat": self._score_block_micro_threats(state, player_id),
        }

    # ------------------------------------------------------------------
    # Leaf evaluation helper
    # ------------------------------------------------------------------

    @staticmethod
    def _leaf_to_winner(score: float, last_mover: int) -> int:
        """
        Converts a heuristic score to a pseudo-winner for MCTS backpropagation.

        If the absolute score exceeds ``LEAF_WIN_THRESHOLD``, it is treated
        as a decisive win/loss for the *last_mover*.  Otherwise, the score
        is mapped to a probability-like outcome.

        Args:
            score: The heuristic evaluation score.
            last_mover: The player who made the last move (1 or 2).

        Returns:
            int: 1 if last_mover wins, 2 if opponent wins, 3 for draw.
        """
        if score >= LEAF_WIN_THRESHOLD:
            return last_mover
        if score <= -LEAF_WIN_THRESHOLD:
            return 3 - last_mover
        # Map score to a probability-like outcome
        prob = (math.tanh(score / 25.0) + 1.0) / 2.0  # sigmoid-ish in [-1,1] range
        if prob > 0.6:
            return last_mover
        if prob < 0.4:
            return 3 - last_mover
        return 3  # draw / uncertain

    # ------------------------------------------------------------------
    # Feature-scoring methods
    # ------------------------------------------------------------------

    @staticmethod
    def _score_micro_wins(state: UTTTState, player_id: int) -> float:
        """
        Count micro-boards owned by *player_id* minus those owned by opponent.

        Returns:
            float: +N for player-owned, -N for opponent-owned micro-boards.
        """
        opponent = 3 - player_id
        player_count = 0
        opponent_count = 0
        for row in state._macro_board:
            for cell in row:
                if cell == player_id:
                    player_count += 1
                elif cell == opponent:
                    opponent_count += 1
        return float(player_count - opponent_count)

    @staticmethod
    def _score_macro_threats(state: UTTTState, player_id: int) -> float:
        """
        Count two-macro-boards-in-a-row (unblocked third) for *player_id*.

        Returns:
            float: Number of macro-level two-in-a-row threats for the player.
        """
        return float(
            _count_macro_two_in_row(state._macro_board, player_id)
        )

    @staticmethod
    def _score_block_macro_threats(state: UTTTState, player_id: int) -> float:
        """
        Count opponent two-macro-boards-in-a-row that *player_id* can block.

        Returns:
            float: Number of opponent macro-level two-in-a-row threats.
        """
        opponent = 3 - player_id
        return float(
            _count_macro_two_in_row(state._macro_board, opponent)
        )

    @staticmethod
    def _score_center_macro(state: UTTTState, player_id: int) -> float:
        """
        Score the center macro-board (1,1) ownership.

        Returns:
            float: +1 if owned by player, -1 if owned by opponent, 0 otherwise.
        """
        cell = state._macro_board[1][1]
        if cell == player_id:
            return 1.0
        if cell == 3 - player_id:
            return -1.0
        return 0.0

    @staticmethod
    def _score_corner_macros(state: UTTTState, player_id: int) -> float:
        """
        Score corner macro-board ownership.

        Corners are (0,0), (0,2), (2,0), (2,2).

        Returns:
            float: +1 per corner owned by player, -1 per corner owned by opponent.
        """
        corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
        opponent = 3 - player_id
        score = 0
        for my, mx in corners:
            cell = state._macro_board[my][mx]
            if cell == player_id:
                score += 1
            elif cell == opponent:
                score -= 1
        return float(score)

    @staticmethod
    def _score_center_micros(state: UTTTState, player_id: int) -> float:
        """
        Score control of center cells within unresolved micro-boards.

        For each unresolved micro-board, checks if the center cell
        (1,1 within the 3x3 subgrid) is occupied.

        Returns:
            float: +1 per center owned by player, -1 per center owned by opponent.
        """
        opponent = 3 - player_id
        score = 0
        for my in range(3):
            for mx in range(3):
                if state._macro_board[my][mx] != 0:
                    continue  # resolved, skip
                center_y = my * 3 + 1
                center_x = mx * 3 + 1
                cell = state._board[center_y][center_x]
                if cell == player_id:
                    score += 1
                elif cell == opponent:
                    score -= 1
        return float(score)

    @staticmethod
    def _score_free_move(state: UTTTState, player_id: int) -> float:
        """
        Score whether the current player has a free-move opportunity.

        A free-move opportunity exists when the current player can make a
        move that sends the opponent to a resolved macro-board, giving the
        opponent no restriction.

        Returns:
            float: +1 if current player has this opportunity (and is
                *player_id*), -1 if the opponent has it, 0 otherwise.
        """
        # Quick check: if active_macro is None, every move sends opponent
        # to whatever board is indicated by the micro-position.
        if state._active_macro is not None:
            return 0.0

        # Check if there exists at least one move that sends opponent
        # to a resolved board.
        for y in range(9):
            for x in range(9):
                my, mx = y // 3, x // 3
                if state._macro_board[my][mx] != 0:
                    continue  # can't play in resolved board
                if state._board[y][x] != 0:
                    continue  # occupied
                if (
                    state._active_macro is not None
                    and state._active_macro != [my, mx]
                ):
                    continue

                # Check where this move would send opponent
                next_my, next_mx = y % 3, x % 3
                if state._macro_board[next_my][next_mx] != 0:
                    # Found a move that sends opponent to a resolved board
                    if state._current_player == player_id:
                        return 1.0
                    else:
                        return -1.0

        return 0.0

    @staticmethod
    def _score_micro_threats(state: UTTTState, player_id: int) -> float:
        """
        Count two-in-a-row threats for *player_id* within micro-boards.

        Iterates each unresolved micro-board and checks all 8 win lines
        for a two-in-a-row (exactly 2 cells owned by player, 1 empty).

        Returns:
            float: Number of micro-level two-in-a-row threats for the player.
        """
        opponent = 3 - player_id
        count = 0
        for my in range(3):
            for mx in range(3):
                if state._macro_board[my][mx] != 0:
                    continue  # resolved
                start_y = my * 3
                start_x = mx * 3
                for line in WIN_LINES:
                    player_cells = 0
                    empty_cells = 0
                    for dy, dx in line:
                        cell = state._board[start_y + dy][start_x + dx]
                        if cell == player_id:
                            player_cells += 1
                        elif cell == 0:
                            empty_cells += 1
                    if player_cells == 2 and empty_cells == 1:
                        count += 1
        return float(count)

    @staticmethod
    def _score_block_micro_threats(state: UTTTState, player_id: int) -> float:
        """
        Count opponent two-in-a-row threats within micro-boards.

        This identifies threats that *player_id* may need to block.

        Returns:
            float: Number of opponent micro-level two-in-a-row threats.
        """
        opponent = 3 - player_id
        count = 0
        for my in range(3):
            for mx in range(3):
                if state._macro_board[my][mx] != 0:
                    continue  # resolved
                start_y = my * 3
                start_x = mx * 3
                for line in WIN_LINES:
                    player_cells = 0
                    empty_cells = 0
                    for dy, dx in line:
                        cell = state._board[start_y + dy][start_x + dx]
                        if cell == opponent:
                            player_cells += 1
                        elif cell == 0:
                            empty_cells += 1
                    if player_cells == 2 and empty_cells == 1:
                        count += 1
        return float(count)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_3x3_winner(grid: List[List[int]]) -> int:
        """
        Checks if a 3x3 grid has a winner.

        Args:
            grid: A 3x3 list of lists (0=empty, 1=P1, 2=P2).

        Returns:
            int: Player ID (1 or 2) if winner found, else 0.
        """
        for line in WIN_LINES:
            vals = [grid[dy][dx] for dy, dx in line]
            if vals[0] != 0 and vals[0] == vals[1] == vals[2]:
                return vals[0]
        return 0


# ---------------------------------------------------------------------------
# Module-level helpers (also used by feature methods)
# ---------------------------------------------------------------------------


def _count_macro_two_in_row(
    macro_board: List[List[int]], player: int
) -> int:
    """
    Count the number of two-in-a-row threats for *player* on the macro board.

    A threat is defined as exactly 2 cells owned by *player* in a win line,
    with the third cell still empty (0).

    Args:
        macro_board: 3x3 macro board.
        player: Player ID (1 or 2).

    Returns:
        int: Number of macro-level two-in-a-row threats.
    """
    count = 0
    for line in WIN_LINES:
        player_cells = 0
        empty_cells = 0
        for dy, dx in line:
            cell = macro_board[dy][dx]
            if cell == player:
                player_cells += 1
            elif cell == 0:
                empty_cells += 1
        if player_cells == 2 and empty_cells == 1:
            count += 1
    return count
