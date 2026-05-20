"""
Configuration dataclass for the AlphaZero-lite self-play training pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class SelfPlayConfig:
    """Configuration for the self-play training pipeline.

    Attributes:
        num_iterations: Number of self-play/training iterations.
        games_per_iteration: Number of self-play games per iteration.
        mcts_iterations: Number of MCTS simulations per move.
        c_puct: PUCT exploration constant.
        temperature_schedule: List of (move_count, temperature) pairs defining
            the temperature schedule. The temperature is interpolated linearly
            between schedule points.
        dirichlet_alpha: Alpha parameter for Dirichlet noise added to root
            priors during self-play (exploration).
        dirichlet_epsilon: Mixing weight for Dirichlet noise:
            (1 - epsilon) * prior + epsilon * noise.
        learning_rate: Learning rate for the Adam optimizer.
        batch_size: Number of training examples per batch.
        epochs: Number of training epochs per iteration.
        checkpoint_dir: Directory to save/load network checkpoints.
        l2_regularization: L2 weight decay coefficient (used via Adam's
            weight_decay parameter).
        network_channels: Number of convolutional channels in the residual tower.
        network_res_blocks: Number of residual blocks in the tower.
        workers: Number of parallel workers for self-play game generation.
            When set to 1 (default), games are generated sequentially.
            When > 1, games are distributed across worker processes.
    """

    num_iterations: int = 30
    games_per_iteration: int = 100
    mcts_iterations: int = 100
    c_puct: float = 1.0
    temperature_schedule: List[Tuple[int, float]] = field(
        default_factory=lambda: [(0, 1.0), (10, 0.5), (20, 0.25)]
    )
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    learning_rate: float = 0.002
    batch_size: int = 256
    epochs: int = 3
    checkpoint_dir: str = "checkpoints/"
    l2_regularization: float = 0.0001
    network_channels: int = 160
    network_res_blocks: int = 10
    workers: int = 1  # Parallel workers for self-play (1 = sequential)
