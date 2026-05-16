"""
Self-play training pipeline runner.

Orchestrates the iterative process of:
  1. Generating self-play games with the current network.
  2. Training the network on the collected data.
  3. Saving checkpoints for later use or evaluation.

Supports resuming from the latest checkpoint if interrupted.
"""

import argparse
import logging
import os
import random
import sys
from typing import List, Optional, Tuple

import torch

from engine.policy_value_network import (
    PolicyValueNetwork,
    load_network,
    save_network,
)
from selfplay.config import SelfPlayConfig
from selfplay.self_play import TrainingExample, generate_self_play_games
from selfplay.train import train_network

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - PIPELINE - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def save_checkpoint(
    network: PolicyValueNetwork,
    iteration: int,
    path: str,
) -> str:
    """Save a network checkpoint to *path*.

    Stores architecture parameters (channels, num_res_blocks) alongside
    the state dict so that the network can be correctly reconstructed
    when loaded.

    Args:
        network: The network to save.
        iteration: The current iteration number (used in the filename).
        path: Directory path where the checkpoint will be saved.

    Returns:
        The full path to the saved checkpoint file.
    """
    os.makedirs(path, exist_ok=True)
    filename = f"checkpoint_iter_{iteration:04d}.pt"
    full_path = os.path.join(path, filename)

    # Save state dict + architecture metadata
    checkpoint = {
        "state_dict": network.state_dict(),
        "channels": network.channels,
        "num_res_blocks": network.num_res_blocks,
    }
    torch.save(checkpoint, full_path)
    logger.info(f"Saved checkpoint: {full_path}")
    return full_path


def load_latest_checkpoint(
    config: SelfPlayConfig,
    device: str = "cpu",
) -> Tuple[Optional[PolicyValueNetwork], int]:
    """Load the latest checkpoint from the checkpoint directory.

    Scans for files matching ``checkpoint_iter_*.pt`` and loads the one
    with the highest iteration number.  Supports both the new format
    (with architecture metadata) and legacy format (state_dict only).

    Args:
        config: Self-play configuration (provides ``checkpoint_dir``).
        device: Device to load the network onto.

    Returns:
        A tuple of ``(network, start_iteration)`` where:
          - ``network`` is the loaded network, or ``None`` if no checkpoint
            was found.
          - ``start_iteration`` is the next iteration to run (one past the
            loaded checkpoint), or 0 if no checkpoint was found.
    """
    checkpoint_dir = config.checkpoint_dir
    if not os.path.isdir(checkpoint_dir):
        return None, 0

    pattern = "checkpoint_iter_"
    best_iter = -1
    best_path: Optional[str] = None

    for fname in os.listdir(checkpoint_dir):
        if fname.startswith(pattern) and fname.endswith(".pt"):
            try:
                # Extract iteration number from filename
                rest = fname[len(pattern) : -3]  # remove prefix and .pt
                iter_num = int(rest)
                if iter_num > best_iter:
                    best_iter = iter_num
                    best_path = os.path.join(checkpoint_dir, fname)
            except (ValueError, IndexError):
                continue

    if best_path is not None:
        logger.info(f"Loading checkpoint: {best_path}")
        data = torch.load(best_path, map_location=device, weights_only=True)

        # Check if it's the new format (dict with metadata)
        if isinstance(data, dict) and "state_dict" in data:
            channels = data.get("channels", 160)
            num_res_blocks = data.get("num_res_blocks", 10)
            network = PolicyValueNetwork(
                channels=channels, num_res_blocks=num_res_blocks
            )
            network.load_state_dict(data["state_dict"])
        else:
            # Legacy format: raw state_dict
            network = PolicyValueNetwork()
            network.load_state_dict(data)

        network.eval()
        network.to(device)
        return network, best_iter + 1

    return None, 0


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


def run_pipeline(
    config: Optional[SelfPlayConfig] = None,
    resume: bool = True,
    device: str = "cpu",
) -> None:
    """Run the self-play training loop.

    The pipeline:
      1. Creates a new network or loads the latest checkpoint.
      2. For each iteration:
         a. Generates self-play games using the current network.
         b. Trains the network on the generated data.
         c. Saves a checkpoint.
         d. Logs progress.

    Args:
        config: Self-play configuration.  Uses defaults if ``None``.
        resume: If ``True``, attempts to resume from the latest checkpoint.
        device: Device for network training and inference
            (``'cpu'`` or ``'cuda'``).
    """
    if config is None:
        config = SelfPlayConfig()

    # ------------------------------------------------------------------
    # 1. Load or create network
    # ------------------------------------------------------------------
    start_iteration = 0
    network: PolicyValueNetwork

    if resume:
        loaded_network, start_iteration = load_latest_checkpoint(
            config, device=device
        )
        if loaded_network is not None:
            network = loaded_network
            logger.info(
                f"Resuming from iteration {start_iteration}"
            )
        else:
            logger.info("No checkpoint found; starting from scratch.")
            network = PolicyValueNetwork(
                channels=config.network_channels,
                num_res_blocks=config.network_res_blocks,
            )
    else:
        logger.info("Starting fresh (resume=False).")
        network = PolicyValueNetwork(
            channels=config.network_channels,
            num_res_blocks=config.network_res_blocks,
        )

    network.to(device)

    # ------------------------------------------------------------------
    # 2. Main loop
    # ------------------------------------------------------------------
    rng = random.Random(42)  # fixed seed for reproducibility

    for iteration in range(start_iteration, config.num_iterations):
        logger.info(
            f"Iteration {iteration + 1}/{config.num_iterations} "
            f"({config.games_per_iteration} games)"
        )

        # a) Generate self-play games
        logger.info("Generating self-play games...")
        examples: List[TrainingExample] = generate_self_play_games(
            network=network,
            config=config,
            rng=rng,
            device=device,
        )
        logger.info(f"Generated {len(examples)} training examples.")

        # b) Train network
        logger.info("Training network...")
        network = train_network(
            network=network,
            examples=examples,
            config=config,
            device=device,
        )

        # c) Save checkpoint
        save_checkpoint(
            network=network,
            iteration=iteration,
            path=config.checkpoint_dir,
        )

        logger.info(
            f"Completed iteration {iteration + 1}/{config.num_iterations}"
        )

    logger.info("Pipeline complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point for the self-play pipeline.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).
    """
    parser = argparse.ArgumentParser(
        description="AlphaZero-lite self-play training pipeline for UTTT."
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Number of self-play/training iterations (overrides config default).",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=None,
        help="Number of self-play games per iteration (overrides config default).",
    )
    parser.add_argument(
        "--mcts-iterations",
        type=int,
        default=None,
        help="MCTS simulations per move (overrides config default).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory for network checkpoints (overrides config default).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Training batch size (overrides config default).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs per iteration (overrides config default).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate (overrides config default).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh, ignoring any existing checkpoints.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for training/inference (default: cpu).",
    )

    args = parser.parse_args(argv)

    config = SelfPlayConfig()
    if args.iterations is not None:
        config.num_iterations = args.iterations
    if args.games is not None:
        config.games_per_iteration = args.games
    if args.mcts_iterations is not None:
        config.mcts_iterations = args.mcts_iterations
    if args.checkpoint_dir is not None:
        config.checkpoint_dir = args.checkpoint_dir
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate

    run_pipeline(
        config=config,
        resume=not args.no_resume,
        device=args.device,
    )


if __name__ == "__main__":
    main()
