"""
Engine package for Ultimate Tic-Tac-Toe.

Provides shared game logic, immutable game state,
MCTS algorithm implementation, and heuristic evaluation.
"""

from engine.game_rules import (
    apply_move,
    check_3x3_win,
    get_global_winner,
    get_valid_actions,
    is_3x3_full,
)
from engine.game_state import UTTTState
from engine.heuristics import HeuristicEvaluator
from engine.mcts_core import MCTS, MCTSNode

__all__ = [
    "apply_move",
    "check_3x3_win",
    "get_global_winner",
    "get_valid_actions",
    "is_3x3_full",
    "UTTTState",
    "HeuristicEvaluator",
    "MCTS",
    "MCTSNode",
]
