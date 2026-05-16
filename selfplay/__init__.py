"""
Self-play training pipeline for Ultimate Tic-Tac-Toe.

Provides the components needed for AlphaZero-lite self-play:
configuration, game generation, training, and the full pipeline runner.
"""

from selfplay.config import SelfPlayConfig
from selfplay.self_play import TrainingExample, generate_self_play_games
from selfplay.train import train_network

__all__ = [
    "SelfPlayConfig",
    "TrainingExample",
    "generate_self_play_games",
    "train_network",
]
