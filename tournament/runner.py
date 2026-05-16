"""Tournament runner for head-to-head UTTT agent competition.

Provides a headless game loop that pits two agents against each other
over a configurable number of games, logging statistics and handling
crashes/illegal moves gracefully.
"""

import argparse
import json
import logging
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from concurrent.futures import as_completed, ProcessPoolExecutor
from tqdm import tqdm

from agents.dummy_agent import DummyUTTTAgent
from agents.mcts_agent import MCTSAgent
from agents.mcts_heuristic_agent import MCTSHeuristicAgent
from engine.game_state import UTTTState
from logger.stats_logger import StatsLogger

# ---------------------------------------------------------------------------
# Agent Registry
# ---------------------------------------------------------------------------

AGENT_REGISTRY: Dict[str, Dict[str, Any]] = {
    "dummy": {"class": DummyUTTTAgent, "display_name": "Dummy"},
    "mcts": {"class": MCTSAgent, "display_name": "MCTS"},
    "mcts_heuristic": {
        "class": MCTSHeuristicAgent,
        "display_name": "MCTS+Heuristic",
    },
    "alphazero": {
        "class": None,  # Resolved lazily on first use
        "display_name": "AlphaZero",
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _serialize_config(agent: Any) -> str:
    """Serialize agent configuration to a JSON string for CSV logging.

    Args:
        agent: An agent instance whose configuration attributes to extract.

    Returns:
        str: JSON-encoded configuration dictionary.
    """
    config: Dict[str, Any] = {}
    if hasattr(agent, "mcts_iterations"):
        config["mcts_iterations"] = agent.mcts_iterations
    if hasattr(agent, "mcts_exploration_constant"):
        config["mcts_exploration_constant"] = agent.mcts_exploration_constant
    if hasattr(agent, "mcts_time_limit"):
        config["mcts_time_limit"] = agent.mcts_time_limit
    if hasattr(agent, "random_seed"):
        config["random_seed"] = agent.random_seed
    if hasattr(agent, "heuristic_playout_bias"):
        config["heuristic_playout_bias"] = agent.heuristic_playout_bias
    if hasattr(agent, "heuristic_max_depth"):
        config["heuristic_max_depth"] = agent.heuristic_max_depth
    if hasattr(agent, "heuristic_weights") and agent.heuristic_weights is not None:
        config["heuristic_weights"] = agent.heuristic_weights
    if hasattr(agent, "checkpoint_path"):
        config["checkpoint_path"] = agent.checkpoint_path
    if hasattr(agent, "temperature"):
        config["temperature"] = agent.temperature
    return json.dumps(config)


def _create_agent_instance(prototype: Any, seed: Optional[int]) -> Any:
    """Create a fresh agent instance of the same type as *prototype*.

    Copies MCTS configuration from the prototype and applies the given seed.

    Args:
        prototype: The agent instance to copy configuration from.
        seed: Random seed for the new instance (may be None).

    Returns:
        A new agent instance of the same class as *prototype*.
    """
    agent_class = type(prototype)
    kwargs: Dict[str, Any] = {"random_seed": seed}
    if hasattr(prototype, "mcts_iterations"):
        kwargs["mcts_iterations"] = prototype.mcts_iterations
    if hasattr(prototype, "mcts_exploration_constant"):
        kwargs["mcts_exploration_constant"] = prototype.mcts_exploration_constant
    if hasattr(prototype, "mcts_time_limit"):
        kwargs["mcts_time_limit"] = prototype.mcts_time_limit
    if hasattr(prototype, "heuristic_playout_bias"):
        kwargs["heuristic_playout_bias"] = prototype.heuristic_playout_bias
    if hasattr(prototype, "heuristic_max_depth"):
        kwargs["heuristic_max_depth"] = prototype.heuristic_max_depth
    if hasattr(prototype, "heuristic_weights"):
        kwargs["heuristic_weights"] = prototype.heuristic_weights
    if hasattr(prototype, "checkpoint_path"):
        kwargs["checkpoint_path"] = prototype.checkpoint_path
    if hasattr(prototype, "temperature"):
        kwargs["temperature"] = prototype.temperature

    return agent_class(**kwargs)


def _agent_display_name(agent: Any) -> str:
    """Resolve a human-readable display name for an agent instance.

    Checks the registry first, then falls back to the class name.

    Args:
        agent: An agent instance.

    Returns:
        str: A display name (e.g. "MCTS", "Dummy").
    """
    agent_class = type(agent)
    for entry in AGENT_REGISTRY.values():
        if entry["class"] is agent_class:
            return entry["display_name"]
    name = agent_class.__name__
    for suffix in ("UTTTAgent", "Agent"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name or agent_class.__name__


# ---------------------------------------------------------------------------
# Single Game
# ---------------------------------------------------------------------------


def _play_single_game(
    game_idx: int,
    agent1: Any,
    agent2: Any,
    agent1_name: str,
    agent2_name: str,
    agent1_config: str,
    agent2_config: str,
    seed: Optional[int],
    verbose: bool,
    num_games: int = 0,
) -> Dict[str, Any]:
    """Play a single head-to-head game between two agents.

    Each call is completely independent — creates fresh agent instances,
    plays a full game, returns a result dict.  Does NOT perform any
    I/O; local events are accumulated in a list that the caller logs.

    Args:
        num_games: Total number of games (for verbose display only).

    Returns:
        Dict with: game_idx, winner (1/2/3), game_length, game_time,
        p1_label, p2_label, p1_config_str, p2_config_str, local_events
    """
    # --- Alternate first player ---
    if game_idx % 2 == 0:
        p1_proto, p2_proto = agent1, agent2
        p1_label, p2_label = agent1_name, agent2_name
        p1_config_str, p2_config_str = agent1_config, agent2_config
    else:
        p1_proto, p2_proto = agent2, agent1
        p1_label, p2_label = agent2_name, agent1_name
        p1_config_str, p2_config_str = agent2_config, agent1_config

    # --- Derive per-game seeds ---
    if seed is not None:
        p1_seed = seed + game_idx * 2
        p2_seed = seed + game_idx * 2 + 1
    else:
        p1_seed = None
        p2_seed = None

    # --- Create fresh agent instances ---
    p1 = _create_agent_instance(p1_proto, p1_seed)
    p2 = _create_agent_instance(p2_proto, p2_seed)

    if verbose:
        print(
            f"Game {game_idx + 1}/{num_games}: "
            f"{p1_label} (P1) vs {p2_label} (P2)"
        )

    # --- Play the game ---
    game_start = time.monotonic()
    state = UTTTState()
    local_events: List[Dict[str, Any]] = []
    prev_macro = [[0] * 3 for _ in range(3)]
    local_games_seen: set = set()
    game_length = 0
    game_result: Optional[int] = None  # 1=P1 wins, 2=P2 wins, 3=draw
    crash_or_illegal = False

    while not state.is_terminal():
        valid = state.get_valid_actions()
        if not valid:
            # No valid actions but state is not terminal → draw
            game_result = 3
            break

        current_agent = p1 if state.current_player == 1 else p2

        # --- Get action from agent ---
        try:
            action = current_agent.deliberate_from_state(state)
        except Exception as exc:
            msg = (
                f"  Player {state.current_player} ({current_agent.__class__.__name__}) "
                f"crashed: {exc}"
            )
            if verbose:
                print(msg)
            else:
                tqdm.write(msg)
            crash_or_illegal = True
            game_result = 2 if state.current_player == 1 else 1  # opponent wins
            break

        # --- Validate action ---
        if action is None or action not in valid:
            msg = (
                f"  Player {state.current_player} ({current_agent.__class__.__name__}) "
                f"illegal move: {action}"
            )
            if verbose:
                print(msg)
            else:
                tqdm.write(msg)
            crash_or_illegal = True
            game_result = 2 if state.current_player == 1 else 1  # opponent wins
            break

        # --- Apply move ---
        state = state.apply_action(action[0], action[1])
        game_length += 1

        # --- Detect local macro-board completions ---
        for my in range(3):
            for mx in range(3):
                new_val = state.macro_board[my][mx]
                old_val = prev_macro[my][mx]
                if new_val != 0 and old_val == 0 and (my, mx) not in local_games_seen:
                    local_games_seen.add((my, mx))
                    # Count occupied cells in this macro-board
                    count = 0
                    for by in range(my * 3, my * 3 + 3):
                        for bx in range(mx * 3, mx * 3 + 3):
                            if state.board[by][bx] != 0:
                                count += 1
                    local_events.append({
                        "macro_pos": (my, mx),
                        "winner": new_val,
                        "moves_played": count,
                        "p1_agent": p1_label,
                        "p2_agent": p2_label,
                    })

        prev_macro = [row[:] for row in state.macro_board]

    # --- Determine result if not already set ---
    if not crash_or_illegal and game_result is None:
        game_result = state.get_winner()

    game_elapsed = time.monotonic() - game_start

    return {
        "game_idx": game_idx,
        "winner": game_result if game_result is not None else 0,
        "game_length": game_length,
        "game_time": game_elapsed,
        "p1_label": p1_label,
        "p2_label": p2_label,
        "p1_config_str": p1_config_str,
        "p2_config_str": p2_config_str,
        "local_events": local_events,
    }


# ---------------------------------------------------------------------------
# Tournament Runner
# ---------------------------------------------------------------------------


def run_tournament(
    agent1: Any,
    agent2: Any,
    num_games: int = 100,
    seed: Optional[int] = None,
    log_dir: str = "stats/",
    verbose: bool = False,
    workers: int = 1,
) -> Dict[str, Any]:
    """Run a head-to-head tournament between two agents.

    Each game alternates the first player (even-indexed games → agent1=P1).
    Fresh agent instances are created per game with derived seeds for
    reproducibility.  Local (macro-board) and global game outcomes are
    logged to CSV via :class:`StatsLogger`.  Agent crashes and illegal
    moves are caught and result in a win for the opponent.

    When *workers* > 1, games are executed in parallel using a
    :class:`~concurrent.futures.ProcessPoolExecutor`.  Verbose mode forces
    sequential execution regardless of *workers*.

    Args:
        agent1: First agent prototype instance.
        agent2: Second agent prototype instance.
        num_games: Number of games to play (default 100).
        seed: Base random seed for reproducibility.  Per-game seeds are
            derived as ``seed + game_idx * 2 + player_offset``.
        log_dir: Directory for CSV statistics output (default ``stats/``).
        verbose: If True, print per-game progress to stdout.
        workers: Number of parallel worker threads (default 1).  Set > 1
            to enable parallel game execution.

    Returns:
        Dict with keys: ``agent1_wins``, ``agent2_wins``, ``draws``,
        ``total_games``, ``agent1_name``, ``agent2_name``,
        ``avg_game_length``, ``avg_game_time``, ``max_game_time``,
        ``min_game_time``, ``total_time``.
    """
    logger = StatsLogger(log_dir=log_dir)

    agent1_name = _agent_display_name(agent1)
    agent2_name = _agent_display_name(agent2)

    agent1_config = _serialize_config(agent1)
    agent2_config = _serialize_config(agent2)

    results: List[Dict[str, Any]] = []

    tournament_start = time.monotonic()

    if num_games > 0:
        if verbose and workers > 1:
            print(
                "Warning: verbose mode forces sequential execution (workers=1).",
                file=sys.stderr,
            )
            workers = 1

        executor = ProcessPoolExecutor(max_workers=workers)
        try:
            futures = [
                executor.submit(
                    _play_single_game,
                    game_idx=i,
                    agent1=agent1,
                    agent2=agent2,
                    agent1_name=agent1_name,
                    agent2_name=agent2_name,
                    agent1_config=agent1_config,
                    agent2_config=agent2_config,
                    seed=seed,
                    verbose=verbose,
                    num_games=num_games,
                )
                for i in range(num_games)
            ]
            # Running win-rate counters (mapped to agent1/agent2 perspective)
            _w1 = 0
            _w2 = 0
            _draws = 0

            pbar = tqdm(
                as_completed(futures),
                total=num_games,
                desc="Tournament",
                unit="game",
                disable=verbose,
            )
            for future in pbar:
                result = future.result()
                results.append(result)

                # Map P1/P2 result to agent1/agent2 using game parity
                if result["game_idx"] % 2 == 0:
                    # game_idx even: P1=agent1, P2=agent2
                    if result["winner"] == 1:
                        _w1 += 1
                    elif result["winner"] == 2:
                        _w2 += 1
                    else:
                        _draws += 1
                else:
                    # game_idx odd: P1=agent2, P2=agent1
                    if result["winner"] == 1:
                        _w2 += 1
                    elif result["winner"] == 2:
                        _w1 += 1
                    else:
                        _draws += 1

                games_done = len(results)
                pbar.set_description(
                    f"{agent1_name} {_w1/games_done*100:.0f}% | "
                    f"{agent2_name} {_w2/games_done*100:.0f}%"
                )
        except KeyboardInterrupt:
            print("\nInterrupted by user. Shutting down...", file=sys.stderr)
            # Cancel all pending futures and stop workers immediately
            executor.shutdown(wait=False, cancel_futures=True)
            # Force-kill any remaining worker processes
            import multiprocessing, os, signal
            for p in multiprocessing.active_children():
                try:
                    os.kill(p.pid, signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass
            pbar.close()
            if results:
                print(
                    f"Completed {len(results)}/{num_games} games before interrupt.",
                    file=sys.stderr,
                )
            else:
                print("No games completed.", file=sys.stderr)
            # Exit immediately instead of re-raising (avoids atexit cleanup hangs)
            sys.exit(1)
        finally:
            executor.shutdown(wait=False)

    # --- Log all results ---
    for result in results:
        # Log local events
        for event in result["local_events"]:
            logger.log_local_game(
                macro_pos=event["macro_pos"],
                winner=event["winner"],
                moves_played=event["moves_played"],
                p1_agent=event["p1_agent"],
                p2_agent=event["p2_agent"],
            )
        # Log global game
        logger.log_global_game(
            winner=result["winner"],
            total_moves=result["game_length"],
            p1_name=result["p1_label"],
            p2_name=result["p2_label"],
            p1_config=result["p1_config_str"],
            p2_config=result["p2_config_str"],
            round_number=result["game_idx"] + 1,
        )

    # --- Aggregate results ---
    agent1_wins = sum(
        1 for r in results
        if (r["game_idx"] % 2 == 0 and r["winner"] == 1)
        or (r["game_idx"] % 2 == 1 and r["winner"] == 2)
    )
    agent2_wins = sum(
        1 for r in results
        if (r["game_idx"] % 2 == 0 and r["winner"] == 2)
        or (r["game_idx"] % 2 == 1 and r["winner"] == 1)
    )
    draws = sum(1 for r in results if r["winner"] == 3)
    total_game_lengths = [r["game_length"] for r in results]
    total_game_times = [r["game_time"] for r in results]

    avg_length = (
        sum(total_game_lengths) / len(total_game_lengths)
        if total_game_lengths
        else 0.0
    )

    return {
        "agent1_wins": agent1_wins,
        "agent2_wins": agent2_wins,
        "draws": draws,
        "total_games": num_games,
        "agent1_name": agent1_name,
        "agent2_name": agent2_name,
        "avg_game_length": avg_length,
        "avg_game_time": sum(total_game_times) / len(total_game_times) if total_game_times else 0.0,
        "max_game_time": max(total_game_times) if total_game_times else 0.0,
        "min_game_time": min(total_game_times) if total_game_times else 0.0,
        "total_time": time.monotonic() - tournament_start,
    }


# ---------------------------------------------------------------------------
# Summary Printer
# ---------------------------------------------------------------------------


def print_summary(results: Dict[str, Any]) -> None:
    """Print a formatted tournament results summary to stdout.

    Args:
        results: The dictionary returned by :func:`run_tournament`.
    """
    name1 = results["agent1_name"]
    name2 = results["agent2_name"]
    total = results["total_games"]
    w1 = results["agent1_wins"]
    w2 = results["agent2_wins"]
    draws = results["draws"]
    avg_len = results["avg_game_length"]

    label1 = f"Agent 1 ({name1})"
    label2 = f"Agent 2 ({name2})"

    print("=" * 60)
    print(f"TOURNAMENT RESULTS: {name1} vs {name2}")
    print("=" * 60)
    print(f"{'Total games:':<20}{total:>8}")
    print(f"{label1 + ' wins:':<20}{w1:>8} ({w1 / total * 100:.1f}%)" if total > 0 else f"{label1 + ' wins:':<20}{w1:>8}")
    print(f"{label2 + ' wins:':<20}{w2:>8} ({w2 / total * 100:.1f}%)" if total > 0 else f"{label2 + ' wins:':<20}{w2:>8}")
    print(f"{'Draws:':<20}{draws:>8} ({draws / total * 100:.1f}%)" if total > 0 else f"{'Draws:':<20}{draws:>8}")
    print(f"{'Avg game length:':<20}{avg_len:>8.1f} moves")
    print(f"{'Avg game time:':<20}{results['avg_game_time']:>8.2f}s")
    print(f"{'Max game time:':<20}{results['max_game_time']:>8.2f}s")
    print(f"{'Min game time:':<20}{results['min_game_time']:>8.2f}s")
    print(f"{'Total tournament time:':<20}{results['total_time']:>8.2f}s")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _resolve_agent(name: str, args: argparse.Namespace) -> Any:
    """Resolve an agent name from the registry and apply CLI overrides.

    Args:
        name: Agent key from the registry (e.g. ``"mcts"``, ``"dummy"``).
        args: Parsed CLI arguments.

    Returns:
        A configured agent instance.

    Raises:
        SystemExit: If the agent name is not recognised.
    """
    if name not in AGENT_REGISTRY:
        print(
            f"Unknown agent {name!r}. Available: {', '.join(AGENT_REGISTRY)}",
            file=sys.stderr,
        )
        sys.exit(1)

    entry = AGENT_REGISTRY[name]

    # Lazy import for alphazero (avoids torch/numpy dependency at module load)
    if entry["class"] is None:
        if name == "alphazero":
            from agents.alphazero_agent import AlphaZeroUTTTAgent

            entry["class"] = AlphaZeroUTTTAgent  # Cache for next use
        else:
            print(f"Agent {name!r} not properly initialized.", file=sys.stderr)
            sys.exit(1)

    agent_class = entry["class"]
    kwargs: Dict[str, Any] = {}

    if agent_class is MCTSAgent or agent_class is MCTSHeuristicAgent:
        if args.iterations is not None:
            kwargs["mcts_iterations"] = args.iterations
        if args.exploration_constant is not None:
            kwargs["mcts_exploration_constant"] = args.exploration_constant
        if args.time_limit is not None:
            kwargs["mcts_time_limit"] = args.time_limit
        kwargs["random_seed"] = args.seed

    if agent_class is MCTSHeuristicAgent:
        if args.playout_bias is not None:
            kwargs["heuristic_playout_bias"] = args.playout_bias
        if args.max_depth is not None:
            kwargs["heuristic_max_depth"] = args.max_depth
        if args.heuristic_weights is not None:
            import json as _json
            kwargs["heuristic_weights"] = _json.loads(args.heuristic_weights)

    if agent_class.__name__ == "AlphaZeroUTTTAgent":
        if args.checkpoint_path is not None:
            kwargs["checkpoint_path"] = args.checkpoint_path
        else:
            print(
                "ERROR: --checkpoint-path/-p is required when using the alphazero agent.",
                file=sys.stderr,
            )
            sys.exit(1)
        if args.temperature is not None:
            kwargs["temperature"] = args.temperature
        kwargs["random_seed"] = args.seed

    return agent_class(**kwargs)


def _build_cli_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Run a UTTT tournament between two agents."
    )
    parser.add_argument(
        "--agent1",
        type=str,
        default="mcts",
        choices=list(AGENT_REGISTRY),
        help="First agent key (default: mcts)",
    )
    parser.add_argument(
        "--agent2",
        type=str,
        default="dummy",
        choices=list(AGENT_REGISTRY),
        help="Second agent key (default: dummy)",
    )
    parser.add_argument(
        "--games",
        "-n",
        type=int,
        default=100,
        help="Number of games to play (default: 100)",
    )
    parser.add_argument(
        "--iterations",
        "-i",
        type=int,
        default=None,
        help="MCTS iterations per move (overrides agent default)",
    )
    parser.add_argument(
        "--exploration-constant",
        "-c",
        type=float,
        default=None,
        help="MCTS UCB1 exploration constant (overrides agent default)",
    )
    parser.add_argument(
        "--time-limit",
        "-t",
        type=float,
        default=None,
        help="MCTS time limit per move in seconds",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=None,
        help="Base random seed for reproducibility",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="stats/",
        help="Directory for CSV statistics (default: stats/)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print per-game progress",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1; verbose mode forces 1)",
    )
    parser.add_argument(
        "--playout-bias",
        "-b",
        type=float,
        default=None,
        help="Heuristic playout bias 0-1 (for mcts_heuristic agent)",
    )
    parser.add_argument(
        "--max-depth",
        "-d",
        type=int,
        default=None,
        help="Heuristic playout max depth (for mcts_heuristic agent)",
    )
    parser.add_argument(
        "--heuristic-weights",
        type=str,
        default=None,
        help=(
            "JSON string of heuristic weight overrides "
            "(for mcts_heuristic agent, e.g. '{\"micro_win\": 150.0}')"
        ),
    )
    parser.add_argument(
        "--checkpoint-path",
        "-p",
        type=str,
        default=None,
        help="Path to network checkpoint (.pt file) for alphazero agent",
    )
    parser.add_argument(
        "--temperature",
        "--temp",
        type=float,
        default=None,
        help=(
            "Temperature for move selection (0=deterministic, "
            ">0=stochastic; for alphazero agent, default: 0.0)"
        ),
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point for the tournament runner.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).
    """
    parser = _build_cli_parser()
    args = parser.parse_args(argv)

    agent1 = _resolve_agent(args.agent1, args)
    agent2 = _resolve_agent(args.agent2, args)

    workers = args.workers
    if args.verbose and args.workers > 1:
        print(
            "Warning: verbose mode forces sequential execution "
            "(workers reset to 1).",
            file=sys.stderr,
        )
        workers = 1

    if args.verbose:
        print(f"Agent 1: {args.agent1} ({type(agent1).__name__})")
        print(f"Agent 2: {args.agent2} ({type(agent2).__name__})")
        print(f"Games: {args.games}, Seed: {args.seed}")
        if args.workers > 1:
            print(f"Workers: {args.workers} (forced sequential due to -v)")
        print()
    elif args.workers > 1:
        print(f"Using {args.workers} parallel workers.")

    results = run_tournament(
        agent1=agent1,
        agent2=agent2,
        num_games=args.games,
        seed=args.seed,
        log_dir=args.log_dir,
        verbose=args.verbose,
        workers=workers,
    )

    print_summary(results)


if __name__ == "__main__":
    main()
