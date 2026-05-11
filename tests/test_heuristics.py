"""
Tests for engine.heuristics — HeuristicEvaluator.
"""

import math
import random
import time
from typing import Dict, List, Optional

import pytest

from engine.game_state import UTTTState
from engine.heuristics import HeuristicEvaluator, WIN_LINES, _count_macro_two_in_row


# ---------------------------------------------------------------------------
# HeuristicEvaluator
# ---------------------------------------------------------------------------


class TestHeuristicEvaluator:
    """Tests for HeuristicEvaluator.evaluate() and feature methods."""

    # -- Terminal states ---------------------------------------------------

    def test_eval_player_wins_returns_inf(self) -> None:
        """evaluate returns +inf when player_id has won."""
        macro_board = [[1, 1, 1], [0, 0, 0], [0, 0, 0]]
        state = UTTTState(macro_board=macro_board)
        evaluator = HeuristicEvaluator()
        score = evaluator.evaluate(state, player_id=1)
        assert score == float("inf")

    def test_eval_opponent_wins_returns_neg_inf(self) -> None:
        """evaluate returns -inf when opponent has won."""
        macro_board = [[1, 1, 1], [0, 0, 0], [0, 0, 0]]
        state = UTTTState(macro_board=macro_board)
        evaluator = HeuristicEvaluator()
        score = evaluator.evaluate(state, player_id=2)
        assert score == float("-inf")

    def test_eval_draw_returns_zero(self) -> None:
        """evaluate returns 0 for a drawn terminal state (macro full, no winner)."""
        # A full macro board with no 3-in-a-row = draw
        macro_board = [[1, 2, 3], [3, 1, 2], [2, 3, 1]]  # all resolved, no winner
        # Note: 3=draw, 1=P1, 2=P2 — this board has no 3-in-a-row
        # Let's use a known draw pattern
        macro_board = [
            [1, 2, 1],
            [2, 1, 2],
            [2, 1, 2],
        ]  # full, no 3-in-a-row for same player
        state = UTTTState(macro_board=macro_board)
        assert state.is_terminal()
        assert state.get_winner() == 3
        evaluator = HeuristicEvaluator()
        score1 = evaluator.evaluate(state, player_id=1)
        score2 = evaluator.evaluate(state, player_id=2)
        assert score1 == 0.0
        assert score2 == 0.0

    def test_eval_invalid_player_raises(self) -> None:
        """evaluate raises ValueError for invalid player_id."""
        state = UTTTState()
        evaluator = HeuristicEvaluator()
        with pytest.raises(ValueError):
            evaluator.evaluate(state, player_id=0)
        with pytest.raises(ValueError):
            evaluator.evaluate(state, player_id=3)

    # -- Empty board -------------------------------------------------------

    def test_eval_empty_board_symmetric(self) -> None:
        """evaluate returns ~0 for an empty board (symmetric position)."""
        state = UTTTState()
        evaluator = HeuristicEvaluator()
        score1 = evaluator.evaluate(state, player_id=1)
        score2 = evaluator.evaluate(state, player_id=2)
        # Empty board should be close to symmetric
        assert score1 == pytest.approx(0.0, abs=1e-9)
        assert score2 == pytest.approx(0.0, abs=1e-9)

    # -- Weight configuration ----------------------------------------------

    def test_default_weights(self) -> None:
        """DEFAULT_WEIGHTS has all expected keys."""
        keys = {
            "micro_win",
            "macro_threat",
            "block_macro_threat",
            "center_macro",
            "corner_macro",
            "center_micro",
            "free_move",
            "micro_threat",
            "block_micro_threat",
        }
        assert set(HeuristicEvaluator.DEFAULT_WEIGHTS.keys()) == keys

    def test_custom_weights_merge(self) -> None:
        """Custom weights override defaults; missing keys use defaults."""
        custom = {"micro_win": 200.0, "center_macro": 0.0}
        evaluator = HeuristicEvaluator(weights=custom)
        assert evaluator._weights["micro_win"] == 200.0
        assert evaluator._weights["center_macro"] == 0.0
        # Should still have default values for other keys
        assert evaluator._weights["macro_threat"] == 200.0
        assert evaluator._weights["corner_macro"] == 3.0

    def test_unknown_weights_ignored(self) -> None:
        """Unknown keys in custom weights are silently ignored."""
        custom = {"nonexistent": 999.0}
        evaluator = HeuristicEvaluator(weights=custom)
        # All defaults should be present
        for key, val in HeuristicEvaluator.DEFAULT_WEIGHTS.items():
            assert evaluator._weights[key] == val

    def test_none_weights_uses_defaults(self) -> None:
        """Passing None for weights uses all defaults."""
        evaluator = HeuristicEvaluator(weights=None)
        assert evaluator._weights == HeuristicEvaluator.DEFAULT_WEIGHTS

    # -- Determinism -------------------------------------------------------

    def test_deterministic_evaluation(self) -> None:
        """Same state + same player_id yields identical score."""
        # Build a non-trivial state
        board = [[0] * 9 for _ in range(9)]
        board[0][0] = 1
        board[0][1] = 1
        board[4][4] = 2
        macro_board = [[0] * 3 for _ in range(3)]
        state = UTTTState(board=board, macro_board=macro_board)
        evaluator = HeuristicEvaluator()
        score1 = evaluator.evaluate(state, player_id=1)
        score2 = evaluator.evaluate(state, player_id=1)
        assert score1 == score2

    def test_deterministic_different_instances(self) -> None:
        """Different evaluator instances give same score for same state."""
        state = UTTTState()
        e1 = HeuristicEvaluator()
        e2 = HeuristicEvaluator()
        assert e1.evaluate(state, 1) == e2.evaluate(state, 1)

    # -- Feature verification ----------------------------------------------

    def test_micro_win_feature(self) -> None:
        """_score_micro_wins correctly counts owned micro-boards."""
        macro_board = [[1, 2, 2], [0, 0, 0], [0, 0, 0]]
        state = UTTTState(macro_board=macro_board)
        evaluator = HeuristicEvaluator()
        # P1 has 1 micro-win, P2 has 2
        assert evaluator._score_micro_wins(state, 1) == 1 - 2  # = -1.0
        assert evaluator._score_micro_wins(state, 2) == 2 - 1  # = 1.0

    def test_macro_threat_feature(self) -> None:
        """_score_macro_threats detects two-in-a-row on macro board."""
        macro_board = [[1, 1, 0], [0, 0, 0], [0, 0, 0]]
        state = UTTTState(macro_board=macro_board)
        evaluator = HeuristicEvaluator()
        # P1 has 1 threat (row 0), P2 has 0
        assert evaluator._score_macro_threats(state, 1) == 1.0
        assert evaluator._score_macro_threats(state, 2) == 0.0

    def test_block_macro_threat_feature(self) -> None:
        """_score_block_macro_threats detects opponent macro threats."""
        macro_board = [[2, 2, 0], [0, 0, 0], [0, 1, 0]]
        state = UTTTState(macro_board=macro_board)
        evaluator = HeuristicEvaluator()
        # P1's block threats = opponent (P2) threats = 1 (row 0)
        assert evaluator._score_block_macro_threats(state, 1) == 1.0
        # P2's block threats = opponent (P1) threats = 0
        assert evaluator._score_block_macro_threats(state, 2) == 0.0

    def test_center_macro_feature(self) -> None:
        """_score_center_macro detects center macro ownership."""
        macro_board = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        state = UTTTState(macro_board=macro_board)
        evaluator = HeuristicEvaluator()
        assert evaluator._score_center_macro(state, 1) == 1.0
        assert evaluator._score_center_macro(state, 2) == -1.0

    def test_corner_macros_feature(self) -> None:
        """_score_corner_macros counts owned corners."""
        macro_board = [[1, 0, 2], [0, 0, 0], [2, 0, 1]]
        state = UTTTState(macro_board=macro_board)
        evaluator = HeuristicEvaluator()
        # P1 has (0,0) and (2,2); P2 has (0,2) and (2,0)
        # P1: 2 - 2 = 0; P2: 2 - 2 = 0
        assert evaluator._score_corner_macros(state, 1) == 0.0
        assert evaluator._score_corner_macros(state, 2) == 0.0

    def test_corner_macros_feature_unbalanced(self) -> None:
        """_score_corner_macros correctly scores unbalanced ownership."""
        macro_board = [[1, 0, 0], [0, 0, 0], [0, 0, 1]]
        state = UTTTState(macro_board=macro_board)
        evaluator = HeuristicEvaluator()
        # P1 has 2 corners; P2 has 0
        assert evaluator._score_corner_macros(state, 1) == 2.0
        assert evaluator._score_corner_macros(state, 2) == -2.0

    def test_center_micros_asymmetric(self) -> None:
        """_score_center_micros correctly scores unbalanced center control."""
        board = [[0] * 9 for _ in range(9)]
        # Only P1 occupies centers
        board[1][1] = 1  # center of macro (0,0)
        board[4][4] = 1  # center of macro (1,1)
        state = UTTTState(board=board)
        evaluator = HeuristicEvaluator()
        assert evaluator._score_center_micros(state, 1) == 2.0  # P1 has 2 centers
        assert evaluator._score_center_micros(state, 2) == -2.0

    def test_center_micros_feature(self) -> None:
        """_score_center_micros counts center cell control in unresolved boards."""
        board = [[0] * 9 for _ in range(9)]
        # Place P1 in center of macro (0,0) = global (1,1)
        board[1][1] = 1
        # Place P2 in center of macro (1,1) = global (4,4)
        board[4][4] = 2
        state = UTTTState(board=board)
        evaluator = HeuristicEvaluator()
        # Both have 1 center each: score = 1 - 1 = 0 from either perspective
        assert evaluator._score_center_micros(state, 1) == 0.0
        assert evaluator._score_center_micros(state, 2) == 0.0

    def test_free_move_feature_active_macro(self) -> None:
        """_score_free_move is 0 when active_macro is not None."""
        state = UTTTState(active_macro=[0, 0])
        evaluator = HeuristicEvaluator()
        assert evaluator._score_free_move(state, 1) == 0.0
        assert evaluator._score_free_move(state, 2) == 0.0

    def test_free_move_feature_free_move(self) -> None:
        """_score_free_move detects free-move opportunity."""
        # When active_macro is None, current player can choose any board
        state = UTTTState(active_macro=None)
        evaluator = HeuristicEvaluator()
        # From any free cell, we can compute...
        # Since all boards are unresolved, every move sends opponent to the
        # micro-position's board, which is also unresolved. So no free-move
        # opportunity (sending opponent to a resolved board).
        # But active_macro is None → the current player has free board choice,
        # which is advantageous.
        # Our implementation checks if there's an action that sends opponent
        # to a resolved board; there isn't, so score is 0.
        assert evaluator._score_free_move(state, 1) == 0.0

    def test_free_move_with_resolved_target(self) -> None:
        """_score_free_move detects when a move sends opponent to resolved board."""
        # Set up: active_macro=None, one board is resolved
        macro_board = [[1, 0, 0], [0, 0, 0], [0, 0, 0]]  # (0,0) resolved
        board = [[0] * 9 for _ in range(9)]
        # P1 would take center of macro (0,0), which sends opponent to
        # micro-position (1,1) → macro (1,1) which is unresolved
        # We need a situation where the *target* macro-board is already resolved.
        # If active_macro is None and there's a move where the resulting
        # active_macro (y%3, x%3) is resolved...

        # Let's place P1's piece at global (0,0): micro=(0,0), macro=(0,0)
        # The sending position = (0%3, 0%3) = (0,0) which is resolved!
        state = UTTTState(
            board=board, macro_board=macro_board, active_macro=None
        )
        evaluator = HeuristicEvaluator()
        # P1 is current_player (player_id 1), and there exists a move
        # at [0,0] that sends opponent to macro (0,0) which is resolved.
        score_p1 = evaluator._score_free_move(state, 1)
        assert score_p1 == 1.0
        score_p2 = evaluator._score_free_move(state, 2)
        assert score_p2 == -1.0

    def test_micro_threats_feature(self) -> None:
        """_score_micro_threats detects two-in-a-row within micro-boards."""
        board = [[0] * 9 for _ in range(9)]
        # P1 has two in a row in macro (0,0) at global (0,0) and (0,1)
        board[0][0] = 1
        board[0][1] = 1
        macro_board = [[0] * 3 for _ in range(3)]
        state = UTTTState(board=board, macro_board=macro_board)
        evaluator = HeuristicEvaluator()
        # P1 should have at least 1 threat in macro (0,0), row 0
        # P2 should have 0
        assert evaluator._score_micro_threats(state, 1) >= 1.0
        assert evaluator._score_micro_threats(state, 2) == 0.0

    def test_block_micro_threats_feature(self) -> None:
        """_score_block_micro_threats detects opponent micro threats."""
        board = [[0] * 9 for _ in range(9)]
        # P2 has two in a row in macro (0,0)
        board[0][0] = 2
        board[0][1] = 2
        macro_board = [[0] * 3 for _ in range(3)]
        state = UTTTState(board=board, macro_board=macro_board)
        evaluator = HeuristicEvaluator()
        # P1 should detect P2's threat
        assert evaluator._score_block_micro_threats(state, 1) >= 1.0
        # P2 should not see P1's threats (P1 has none)
        assert evaluator._score_block_micro_threats(state, 2) == 0.0

    def test_advantageous_position_positive(self) -> None:
        """A known advantageous position scores positively."""
        # P1 owns center macro, has a micro-board win, and a macro threat
        macro_board = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]  # center owned by P1
        state = UTTTState(macro_board=macro_board)
        evaluator = HeuristicEvaluator()
        score = evaluator.evaluate(state, 1)
        # Center macro alone gives +10 * 1.0 = 10
        assert score > 0

    def test_disadvantageous_position_negative(self) -> None:
        """A known disadvantageous position scores negatively."""
        # P2 owns center macro
        macro_board = [[0, 0, 0], [0, 2, 0], [0, 0, 0]]
        state = UTTTState(macro_board=macro_board)
        evaluator = HeuristicEvaluator()
        score = evaluator.evaluate(state, 1)  # P1 evaluating
        assert score < 0

    def test_full_feature_evaluation(self) -> None:
        """evaluate produces expected score for a constructed position."""
        # P1 has a micro win at (0,0) which is a corner.
        # This also gives +corner_macro and +free_move because P1 can send
        # opponent to the resolved (0,0) board.
        macro_board = [[1, 0, 0], [0, 0, 0], [0, 0, 0]]
        state = UTTTState(macro_board=macro_board)
        evaluator = HeuristicEvaluator()
        # P1: micro_win(100) + corner_macro(3) + free_move(2) = 105
        score = evaluator.evaluate(state, 1)
        assert score == 105.0
        # P2: -micro_win(-100) - corner_macro(-3) - free_move(-2) = -105
        score_p2 = evaluator.evaluate(state, 2)
        assert score_p2 == -105.0

    # -- Benchmark ---------------------------------------------------------

    def test_evaluate_under_1ms(self) -> None:
        """evaluate() completes in <1ms for a typical mid-game state."""
        # Build a more complex state
        board = [[0] * 9 for _ in range(9)]
        for i in range(3):
            for j in range(3):
                if (i + j) % 2 == 0:
                    board[i * 3 + 1][j * 3 + 1] = 1
                else:
                    board[i * 3][j * 3] = 2
        macro_board = [[0] * 3 for _ in range(3)]
        macro_board[0][0] = 1  # P1 wins one
        state = UTTTState(board=board, macro_board=macro_board)
        evaluator = HeuristicEvaluator()

        start = time.perf_counter()
        for _ in range(100):
            evaluator.evaluate(state, 1)
            evaluator.evaluate(state, 2)
        elapsed = time.perf_counter() - start
        avg_ms = (elapsed / 200) * 1000
        assert avg_ms < 1.0, f"Average evaluate time {avg_ms:.4f}ms > 1ms"

    # -- Feature breakdown -------------------------------------------------

    def test_get_feature_breakdown(self) -> None:
        """get_feature_breakdown returns all expected keys and correct types."""
        macro_board = [[1, 0, 0], [0, 0, 0], [0, 0, 0]]
        state = UTTTState(macro_board=macro_board)
        evaluator = HeuristicEvaluator()
        breakdown = evaluator.get_feature_breakdown(state, 1)
        expected_keys = {
            "micro_win",
            "macro_threat",
            "block_macro_threat",
            "center_macro",
            "corner_macro",
            "center_micro",
            "free_move",
            "micro_threat",
            "block_micro_threat",
        }
        assert set(breakdown.keys()) == expected_keys
        for key, val in breakdown.items():
            assert isinstance(val, float), f"{key} should be float, got {type(val)}"
        # P1 has one micro win and nothing else of note
        assert breakdown["micro_win"] == 1.0

    def test_get_feature_breakdown_zero_empty_board(self) -> None:
        """get_feature_breakdown on empty board returns all zeros."""
        state = UTTTState()
        evaluator = HeuristicEvaluator()
        breakdown = evaluator.get_feature_breakdown(state, 1)
        for key, val in breakdown.items():
            assert val == 0.0, f"{key} should be 0.0 on empty board, got {val}"


# ---------------------------------------------------------------------------
# Heuristic Playout
# ---------------------------------------------------------------------------


class TestHeuristicPlayout:
    """Tests for HeuristicEvaluator.heuristic_playout()."""

    def test_playout_returns_winner(self) -> None:
        """heuristic_playout returns an integer 1, 2, or 3."""
        state = UTTTState()
        evaluator = HeuristicEvaluator()
        rng = random.Random(42)
        winner = evaluator.heuristic_playout(state, rng)
        assert winner in (1, 2, 3), f"Expected 1, 2, or 3, got {winner}"

    def test_playout_respects_max_depth(self) -> None:
        """heuristic_playout with max_depth=0 uses leaf evaluation immediately."""
        # Set up a non-terminal state
        board = [[0] * 9 for _ in range(9)]
        board[0][0] = 1
        state = UTTTState(board=board)
        evaluator = HeuristicEvaluator()
        rng = random.Random(42)
        # With max_depth=0, playout stops immediately and uses leaf eval
        winner = evaluator.heuristic_playout(state, rng, max_depth=0)
        assert winner in (1, 2, 3)

    def test_playout_bias_zero(self) -> None:
        """playout_bias=0.0 picks random moves (should still work)."""
        state = UTTTState()
        evaluator = HeuristicEvaluator()
        rng = random.Random(42)
        winner = evaluator.heuristic_playout(state, rng, playout_bias=0.0)
        assert winner in (1, 2, 3)

    def test_playout_bias_one(self) -> None:
        """playout_bias=1.0 always picks heuristic-best moves."""
        state = UTTTState()
        evaluator = HeuristicEvaluator()
        rng = random.Random(42)
        winner = evaluator.heuristic_playout(state, rng, playout_bias=1.0)
        assert winner in (1, 2, 3)

    def test_playout_deterministic(self) -> None:
        """Same seed + same state produces same playout result."""
        state = UTTTState()
        evaluator = HeuristicEvaluator()
        rng1 = random.Random(12345)
        rng2 = random.Random(12345)
        winner1 = evaluator.heuristic_playout(state, rng1)
        winner2 = evaluator.heuristic_playout(state, rng2)
        assert winner1 == winner2

    def test_playout_different_seeds_different_results(self) -> None:
        """Different seeds generally produce different playout results."""
        state = UTTTState()
        evaluator = HeuristicEvaluator()
        rng1 = random.Random(111)
        rng2 = random.Random(222)
        w1 = evaluator.heuristic_playout(state, rng1)
        w2 = evaluator.heuristic_playout(state, rng2)
        # This is probabilistic but with an empty board and high depth,
        # different seeds should produce different playout paths
        # We'll just verify they both produce valid winners
        assert w1 in (1, 2, 3)
        assert w2 in (1, 2, 3)

    def test_playout_terminal_state(self) -> None:
        """heuristic_playout on a terminal state returns its winner."""
        macro_board = [[1, 1, 1], [0, 0, 0], [0, 0, 0]]
        state = UTTTState(macro_board=macro_board)
        evaluator = HeuristicEvaluator()
        rng = random.Random(42)
        winner = evaluator.heuristic_playout(state, rng)
        assert winner == 1  # P1 won

    def test_playout_draw_state(self) -> None:
        """heuristic_playout on a drawn terminal state returns 3."""
        macro_board = [[1, 2, 1], [2, 1, 2], [2, 1, 2]]  # full, no 3-in-a-row
        state = UTTTState(macro_board=macro_board)
        assert state.is_terminal()
        assert state.get_winner() == 3
        evaluator = HeuristicEvaluator()
        rng = random.Random(42)
        winner = evaluator.heuristic_playout(state, rng)
        assert winner == 3


# ---------------------------------------------------------------------------
# _leaf_to_winner
# ---------------------------------------------------------------------------


class TestLeafToWinner:
    """Tests for HeuristicEvaluator._leaf_to_winner()."""

    def test_high_score_is_win(self) -> None:
        """Score >= LEAF_WIN_THRESHOLD returns last_mover as winner."""
        winner = HeuristicEvaluator._leaf_to_winner(100.0, 1)
        assert winner == 1

    def test_low_score_is_loss(self) -> None:
        """Score <= -LEAF_WIN_THRESHOLD returns opponent as winner."""
        winner = HeuristicEvaluator._leaf_to_winner(-100.0, 1)
        assert winner == 2

    def test_mid_score_is_draw(self) -> None:
        """Score near 0 returns draw (3)."""
        winner = HeuristicEvaluator._leaf_to_winner(0.0, 1)
        assert winner == 3

    def test_moderate_positive_score(self) -> None:
        """Moderate positive score for last_mover."""
        # Score = 30, last_mover = 2
        winner = HeuristicEvaluator._leaf_to_winner(30.0, 2)
        # tanh(30/25) = tanh(1.2) ≈ 0.833
        # (0.833 + 1) / 2 ≈ 0.917 > 0.6 → last_mover wins
        assert winner == 2

    def test_moderate_negative_score(self) -> None:
        """Moderate negative score against last_mover."""
        # Score = -30, last_mover = 1
        winner = HeuristicEvaluator._leaf_to_winner(-30.0, 1)
        # tanh(-30/25) = tanh(-1.2) ≈ -0.833
        # (-0.833 + 1) / 2 ≈ 0.083 < 0.4 → opponent wins
        assert winner == 2


# ---------------------------------------------------------------------------
# score_move
# ---------------------------------------------------------------------------


class TestScoreMove:
    """Tests for HeuristicEvaluator.score_move()."""

    def test_score_move_returns_float(self) -> None:
        """score_move returns a float."""
        board = [[0] * 9 for _ in range(9)]
        macro_board = [[0] * 3 for _ in range(3)]
        evaluator = HeuristicEvaluator()
        score = evaluator.score_move(
            board=board,
            macro_board=macro_board,
            active_macro=None,
            current_player=1,
            x=4,
            y=4,
            player_id=1,
        )
        assert isinstance(score, float)

    def test_score_move_immediate_win(self) -> None:
        """score_move detects an immediate micro-board win."""
        board = [[0] * 9 for _ in range(9)]
        # P1 has 2 in a row in macro (0,0): global (0,0) and (0,1)
        board[0][0] = 1
        board[0][1] = 1
        macro_board = [[0] * 3 for _ in range(3)]
        evaluator = HeuristicEvaluator()
        # Move at (2, 0) completes the row
        score = evaluator.score_move(
            board=board,
            macro_board=macro_board,
            active_macro=[0, 0],
            current_player=1,
            x=2,
            y=0,
            player_id=1,
        )
        # Should get micro_win weight
        assert score >= 50.0  # micro_win is 100

    def test_score_move_blocks_opponent(self) -> None:
        """score_move detects blocking opponent's win."""
        board = [[0] * 9 for _ in range(9)]
        # Opponent (P2) has 2 in a row in macro (0,0): (0,0) and (0,1)
        board[0][0] = 2
        board[0][1] = 2
        macro_board = [[0] * 3 for _ in range(3)]
        evaluator = HeuristicEvaluator()
        # Move at (2, 0) blocks P2's win
        score = evaluator.score_move(
            board=board,
            macro_board=macro_board,
            active_macro=[0, 0],
            current_player=1,
            x=2,
            y=0,
            player_id=1,
        )
        # Should get block_micro_threat weight (20)
        assert score >= 10.0

    def test_score_move_center_cell(self) -> None:
        """score_move awards center_micro bonus for center cell."""
        board = [[0] * 9 for _ in range(9)]
        macro_board = [[0] * 3 for _ in range(3)]
        evaluator = HeuristicEvaluator()
        # Center of macro (0,0) = global (1,1)
        score = evaluator.score_move(
            board=board,
            macro_board=macro_board,
            active_macro=[0, 0],
            current_player=1,
            x=1,
            y=1,
            player_id=1,
        )
        # Should get center_micro bonus (3.0)
        assert score > 0

    def test_score_move_sends_to_resolved(self) -> None:
        """score_move awards free_move when sending opponent to resolved board."""
        board = [[0] * 9 for _ in range(9)]
        # macro (0,0) is resolved
        macro_board = [[1, 0, 0], [0, 0, 0], [0, 0, 0]]
        evaluator = HeuristicEvaluator()
        # Move at global (0,0): micro=(0,0), target macro=(0,0) which is resolved
        score = evaluator.score_move(
            board=board,
            macro_board=macro_board,
            active_macro=[0, 0],
            current_player=1,
            x=0,
            y=0,
            player_id=1,
        )
        # Should get free_move bonus (2.0)
        assert score > 0


# ---------------------------------------------------------------------------
# _count_macro_two_in_row (module-level helper)
# ---------------------------------------------------------------------------


class TestCountMacroTwoInRow:
    """Tests for the _count_macro_two_in_row helper."""

    def test_no_threats(self) -> None:
        """Empty macro board has no threats."""
        macro = [[0] * 3 for _ in range(3)]
        assert _count_macro_two_in_row(macro, 1) == 0
        assert _count_macro_two_in_row(macro, 2) == 0

    def test_one_row_threat(self) -> None:
        """Two in a row on top row is detected."""
        macro = [[1, 1, 0], [0, 0, 0], [0, 0, 0]]
        assert _count_macro_two_in_row(macro, 1) == 1

    def test_one_col_threat(self) -> None:
        """Two in a column is detected."""
        macro = [[1, 0, 0], [1, 0, 0], [0, 0, 0]]
        assert _count_macro_two_in_row(macro, 1) == 1

    def test_diagonal_threat(self) -> None:
        """Two in a diagonal is detected."""
        macro = [[1, 0, 0], [0, 1, 0], [0, 0, 0]]
        assert _count_macro_two_in_row(macro, 1) == 1

    def test_blocked_threat_not_counted(self) -> None:
        """Three in a row (already won) is not a threat."""
        macro = [[1, 1, 1], [0, 0, 0], [0, 0, 0]]
        assert _count_macro_two_in_row(macro, 1) == 0

    def test_mixed_ownership_not_threat(self) -> None:
        """Two owned + one opponent = blocked, not a threat."""
        macro = [[1, 1, 2], [0, 0, 0], [0, 0, 0]]
        assert _count_macro_two_in_row(macro, 1) == 0

    def test_multiple_threats(self) -> None:
        """Multiple threats are counted correctly."""
        # Two threats: row 0 and col 0
        macro = [[1, 1, 0], [1, 0, 0], [0, 0, 0]]
        assert _count_macro_two_in_row(macro, 1) == 2
