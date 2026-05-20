"""
Self-play training pipeline runner.

Orchestrates the iterative process of:
  1. Generating self-play games with the current network.
  2. Training the network on the collected data.
  3. Saving checkpoints for later use or evaluation.

Supports resuming from the latest checkpoint if interrupted.
"""

import argparse
import copy
import logging
import os
import random
import shutil
import signal
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional, Tuple

import torch
from tqdm import tqdm

from engine.policy_value_network import (
    PolicyValueNetwork,
    load_network,
    save_network,
)
from selfplay.config import SelfPlayConfig
from selfplay.self_play import TrainingExample, generate_self_play_games
from selfplay.train import train_network
from utils.progress import setup_mp_lock

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
# Parallel worker
# ---------------------------------------------------------------------------


def _generate_games_worker(
    checkpoint_path: str,
    config_dict: dict,
    device: str,
    rng_seed: int,
    num_games: int,
    worker_index: int = 0,
) -> List[TrainingExample]:
    """Worker function for parallel self-play.

    Each worker loads the network independently from a checkpoint,
    plays *num_games* games, and returns the training examples.

    This must be a module-level function for multiprocessing pickling.

    Args:
        checkpoint_path: Path to a saved ``.pt`` checkpoint file.
        config_dict: Serialized ``SelfPlayConfig`` as a dictionary.
        device: Device for network inference (``'cpu'`` or ``'cuda'``).
        rng_seed: Seed for the local ``random.Random`` instance.
        num_games: Number of self-play games to generate in this worker.
        worker_index: Worker index for tqdm display positioning.

    Returns:
        A list of :class:`TrainingExample` instances.
    """
    from selfplay.self_play import generate_self_play_games
    from selfplay.config import SelfPlayConfig
    from engine.policy_value_network import load_network
    import random

    config = SelfPlayConfig(**config_dict)
    config.games_per_iteration = num_games
    network = load_network(checkpoint_path, device=device)
    rng = random.Random(rng_seed)
    return generate_self_play_games(
        network=network, config=config, rng=rng, device=device,
        worker_index=worker_index,
    )


def _load_network_from_path(path: str, device: str) -> PolicyValueNetwork:
    """Load a PolicyValueNetwork from a checkpoint file.

    Args:
        path: Path to the checkpoint (.pt file).
        device: Device to load the network onto.

    Returns:
        The loaded PolicyValueNetwork in eval mode on the requested device.
    """
    data = torch.load(path, map_location=device, weights_only=True)
    if isinstance(data, dict) and "state_dict" in data:
        network = PolicyValueNetwork(
            channels=data.get("channels", 160),
            num_res_blocks=data.get("num_res_blocks", 10),
        )
        network.load_state_dict(data["state_dict"])
    else:
        network = PolicyValueNetwork()
        network.load_state_dict(data)
    network.to(device)
    network.eval()
    return network


def _evaluate_arena(
    candidate_path: str,
    best_path: str,
    num_games: int,
    device: str,
    mcts_iterations: int = 200,
    update_threshold: float = 0.55,
) -> Tuple[float, bool]:
    """Pit a candidate network against the previous best network (Arena).

    Follows the standard AlphaZero acceptance criteria: the candidate is
    accepted if its score (wins + 0.5 * draws) / num_games >= threshold.

    Args:
        candidate_path: Path to the candidate checkpoint.
        best_path: Path to the current best checkpoint.
        num_games: Number of evaluation games to play (recommended even).
        device: Device for inference.
        mcts_iterations: MCTS iterations for both agents during evaluation.
        update_threshold: Minimum win-rate (including draws as 0.5) to accept
            the candidate.

    Returns:
        Tuple of (candidate_score, accepted) where *candidate_score* is
        ``(wins + 0.5 * draws) / num_games``.
    """
    from agents.alphazero_agent import AlphaZeroUTTTAgent
    from tournament.runner import _play_single_game

    candidate_agent = AlphaZeroUTTTAgent(
        checkpoint_path=candidate_path,
        mcts_iterations=mcts_iterations,
        temperature=0.0,
        device=device,
    )
    best_agent = AlphaZeroUTTTAgent(
        checkpoint_path=best_path,
        mcts_iterations=mcts_iterations,
        temperature=0.0,
        device=device,
    )

    score = 0.0
    for i in tqdm(
        range(num_games),
        desc="Arena: candidate vs best",
        unit="game",
        leave=False,
        position=0,
    ):
        candidate_is_p1 = i % 2 == 0
        result = _play_single_game(
            game_idx=i,
            agent1=candidate_agent if candidate_is_p1 else best_agent,
            agent2=best_agent if candidate_is_p1 else candidate_agent,
            agent1_name="Candidate" if candidate_is_p1 else "Best",
            agent2_name="Best" if candidate_is_p1 else "Candidate",
            agent1_config="",
            agent2_config="",
            seed=42 + i,
            verbose=False,
            num_games=num_games,
        )
        if result["winner"] == 3:
            score += 0.5
        elif (candidate_is_p1 and result["winner"] == 1) or (
            not candidate_is_p1 and result["winner"] == 2
        ):
            score += 1.0

    candidate_score = score / num_games if num_games > 0 else 0.0
    accepted = candidate_score >= update_threshold
    logger.info(
        f"Arena result: {candidate_score:.1%} "
        f"(threshold={update_threshold:.0%}) "
        f"- {'ACCEPTED' if accepted else 'REJECTED'}"
    )
    return candidate_score, accepted


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


def run_pipeline(
    config: Optional[SelfPlayConfig] = None,
    resume: bool = True,
    device: str = "cpu",
    verbose: bool = False,
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
        verbose: If ``True``, sets root logger to DEBUG level.
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if config is None:
        config = SelfPlayConfig()

    # Use spawn for multiprocessing (compatible with PyTorch; fork can deadlock)
    import multiprocessing as _mp
    try:
        _mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # already set

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

    # Replay buffer: keep examples from recent iterations, capped by total count
    _MAX_REPLAY_EXAMPLES = 500_000
    replay_buffer: List[List[TrainingExample]] = []
    best_path: Optional[str] = None

    # Create shared executor once for all iterations (avoids tqdm lock issues)
    executor: Optional[ProcessPoolExecutor] = None
    if config.workers > 1:
        mp_lock = setup_mp_lock()
        executor = ProcessPoolExecutor(
            max_workers=config.workers,
            initializer=tqdm.set_lock,
            initargs=(mp_lock,),
        )

    for iteration in range(start_iteration, config.num_iterations):
        logger.info(
            f"Iteration {iteration + 1}/{config.num_iterations} "
            f"({config.games_per_iteration} games)"
        )

        # a) Generate self-play games (optionally in parallel)
        logger.info(
            f"Generating self-play games ({config.workers} worker(s))..."
        )

        if config.workers <= 1:
            # Sequential (original behavior)
            examples: List[TrainingExample] = generate_self_play_games(
                network=network,
                config=config,
                rng=rng,
                device=device,
            )
        else:
            # Parallel: save a temp checkpoint for workers to load
            tmp_checkpoint = os.path.join(
                config.checkpoint_dir,
                f".worker_checkpoint_{iteration}.pt",
            )
            os.makedirs(config.checkpoint_dir, exist_ok=True)
            torch.save(network.state_dict(), tmp_checkpoint)

            # Split games across workers
            total_games = config.games_per_iteration
            games_per_worker = [total_games // config.workers] * config.workers
            for i in range(total_games % config.workers):
                games_per_worker[i] += 1

            # Serialize config for workers
            import dataclasses

            config_dict = dataclasses.asdict(config)
            config_dict.pop("workers")  # Workers pick workers internally

            # Submit tasks to the shared executor
            futures = []
            for w_idx in range(config.workers):
                seed = rng.randint(0, 2**31 - 1)
                future = executor.submit(
                    _generate_games_worker,
                    tmp_checkpoint,
                    config_dict,
                    device,
                    seed,
                    games_per_worker[w_idx],
                    w_idx,  # worker_index
                )
                futures.append(future)

            # Parent "Games" bar at position 0 (batch updates as workers finish)
            games_pbar = tqdm(
                total=config.games_per_iteration,
                desc=f"Iteration {iteration + 1}/{config.num_iterations} ({config.games_per_iteration} games)",
                unit="game",
                position=0,
            )
            try:
                examples = []
                for future in as_completed(futures):
                    worker_examples = future.result()
                    examples.extend(worker_examples)
                    # Find how many games this worker did
                    w_idx = next(
                        i for i, f in enumerate(futures) if f is future
                    )
                    games_pbar.update(games_per_worker[w_idx])
            except KeyboardInterrupt:
                print(
                    "\nInterrupted during self-play. Shutting down workers...",
                    file=sys.stderr,
                )
                executor.shutdown(wait=False, cancel_futures=True)
                # Force-kill any remaining worker processes
                import multiprocessing as _mp
                for p in _mp.active_children():
                    try:
                        os.kill(p.pid, signal.SIGKILL)
                    except (ProcessLookupError, OSError):
                        pass
                # Clean up temp checkpoint
                try:
                    os.remove(tmp_checkpoint)
                except OSError:
                    pass
                try:
                    games_pbar.close()
                except NameError:
                    pass
                print("Self-play interrupted.", file=sys.stderr, flush=True)
                sys.exit(1)
            finally:
                games_pbar.close()

            # Clean up temp checkpoint
            try:
                os.remove(tmp_checkpoint)
            except OSError:
                pass

        logger.info(f"Generated {len(examples)} training examples.")

        # b) Add to replay buffer and cap by total examples
        replay_buffer.append(examples)
        _total = sum(len(b) for b in replay_buffer)
        while _total > _MAX_REPLAY_EXAMPLES and len(replay_buffer) > 1:
            _total -= len(replay_buffer.pop(0))
        flat_examples: List[TrainingExample] = [
            ex for batch in replay_buffer for ex in batch
        ]
        logger.info(
            f"Replay buffer: {len(flat_examples)} examples from "
            f"{len(replay_buffer)} iteration(s)."
        )

        # c) Train network
        logger.info("Training network...")
        network = train_network(
            network=network,
            examples=flat_examples,
            config=config,
            device=device,
        )

        # d) Save iteration checkpoint
        iter_path = save_checkpoint(
            network=network,
            iteration=iteration,
            path=config.checkpoint_dir,
        )

        # e) Evaluation gate
        best_pt = os.path.join(config.checkpoint_dir, "best.pt")
        global_iter = iteration + 1
        if best_path is None:
            # First iteration: always promote
            best_path = iter_path
            shutil.copy(iter_path, best_pt)
            logger.info("First iteration promoted to best.")
        elif global_iter <= 10:
            # Phase 1 (iterations 2–10): auto-promote to bootstrap
            best_path = iter_path
            shutil.copy(iter_path, best_pt)
            logger.info(
                f"Phase 1: iteration {global_iter}/10 auto-promoted."
            )
        else:
            # Phase 2 (iteration 11+): Arena evaluation against previous best
            logger.info(
                f"Phase 2: arena candidate vs best ({best_path})"
            )
            candidate_score, accepted = _evaluate_arena(
                iter_path,
                best_path,
                num_games=20,
                device=device,
                mcts_iterations=200,
            )
            if accepted:
                best_path = iter_path
                shutil.copy(iter_path, best_pt)
                logger.info(
                    f"Candidate accepted! Score={candidate_score:.1%}"
                )
            else:
                logger.info(
                    f"Candidate rejected. Score={candidate_score:.1%}. "
                    f"Restoring best network."
                )
                network = _load_network_from_path(best_path, device)

        logger.info(
            f"Completed iteration {global_iter}/{config.num_iterations}"
        )

    logger.info("Pipeline complete.")

    # Shutdown the shared executor if it was created
    if executor is not None:
        executor.shutdown(wait=True)


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
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=None,
        help="Number of parallel workers for self-play (default: 1, sequential).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging.",
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
    if args.workers is not None:
        config.workers = args.workers
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    run_pipeline(
        config=config,
        resume=not args.no_resume,
        device=args.device,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
