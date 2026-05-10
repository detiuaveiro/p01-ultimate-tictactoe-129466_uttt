"""Tournament runner for head-to-head UTTT agent competition.

Provides a headless game loop that pits two agents against each other
over a configurable number of games, logging statistics and handling
crashes/illegal moves gracefully.
"""

import argparse
import json
import logging
import sys
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from agents.dummy_agent import DummyUTTTAgent
from agents.mcts_agent import MCTSAgent
from engine.game_state import UTTTState
from logger.stats_logger import StatsLogger

# ---------------------------------------------------------------------------
# Agent Registry
# ---------------------------------------------------------------------------

AGENT_REGISTRY: Dict[str, Dict[str, Any]] = {
    "dummy": {"class": DummyUTTTAgent, "display_name": "Dummy"},
    "mcts": {"class": MCTSAgent, "display_name": "MCTS"},
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
# Tournament Runner
# ---------------------------------------------------------------------------


def run_tournament(
    agent1: Any,
    agent2: Any,
    num_games: int = 100,
    seed: Optional[int] = None,
    log_dir: str = "stats/",
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run a head-to-head tournament between two agents.

    Each game alternates the first player (even-indexed games → agent1=P1).
    Fresh agent instances are created per game with derived seeds for
    reproducibility.  Local (macro-board) and global game outcomes are
    logged to CSV via :class:`StatsLogger`.  Agent crashes and illegal
    moves are caught and result in a win for the opponent.

    Args:
        agent1: First agent prototype instance.
        agent2: Second agent prototype instance.
        num_games: Number of games to play (default 100).
        seed: Base random seed for reproducibility.  Per-game seeds are
            derived as ``seed + game_idx * 2 + player_offset``.
        log_dir: Directory for CSV statistics output (default ``stats/``).
        verbose: If True, print per-game progress to stdout.

    Returns:
        Dict with keys: ``agent1_wins``, ``agent2_wins``, ``draws``,
        ``total_games``, ``agent1_name``, ``agent2_name``,
        ``avg_game_length``.
    """
    logger = StatsLogger(log_dir=log_dir)

    agent1_name = _agent_display_name(agent1)
    agent2_name = _agent_display_name(agent2)

    agent1_config = _serialize_config(agent1)
    agent2_config = _serialize_config(agent2)

    agent1_wins = 0
    agent2_wins = 0
    draws = 0
    total_game_lengths: List[int] = []

    # --- Build game iterator (tqdm when not verbose) ---
    game_range = range(num_games)
    if verbose or num_games == 0:
        game_iter: Any = game_range
    else:
        game_iter = tqdm(game_range, desc="Tournament", unit="game")

    for game_idx in game_iter:
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
        state = UTTTState()
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
                        logger.log_local_game(
                            macro_pos=(my, mx),
                            winner=new_val,
                            moves_played=count,
                            p1_agent=p1_label,
                            p2_agent=p2_label,
                        )

            prev_macro = [row[:] for row in state.macro_board]

        # --- Determine result if not already set ---
        if not crash_or_illegal and game_result is None:
            game_result = state.get_winner()

        # --- Update win/draw counters ---
        # Map P1/P2 result to agent1/agent2
        if game_idx % 2 == 0:
            # agent1 = P1, agent2 = P2
            if game_result == 1:
                agent1_wins += 1
            elif game_result == 2:
                agent2_wins += 1
            elif game_result == 3:
                draws += 1
        else:
            # agent2 = P1, agent1 = P2
            if game_result == 1:
                agent2_wins += 1
            elif game_result == 2:
                agent1_wins += 1
            elif game_result == 3:
                draws += 1

        total_game_lengths.append(game_length)

        # --- Log global game ---
        logger.log_global_game(
            winner=game_result if game_result is not None else 0,
            total_moves=game_length,
            p1_name=p1_label,
            p2_name=p2_label,
            p1_config=p1_config_str,
            p2_config=p2_config_str,
            round_number=game_idx + 1,
        )

        # --- Update tqdm description with running win rates ---
        if not verbose and num_games > 0:
            games_done = game_idx + 1
            w1_pct = agent1_wins / games_done * 100
            w2_pct = agent2_wins / games_done * 100
            game_iter.set_description(
                f"{agent1_name} {w1_pct:.0f}% | {agent2_name} {w2_pct:.0f}%"
            )

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
    agent_class = entry["class"]
    kwargs: Dict[str, Any] = {}

    if agent_class is MCTSAgent:
        if args.iterations is not None:
            kwargs["mcts_iterations"] = args.iterations
        if args.exploration_constant is not None:
            kwargs["mcts_exploration_constant"] = args.exploration_constant
        if args.time_limit is not None:
            kwargs["mcts_time_limit"] = args.time_limit
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

    if args.verbose:
        print(f"Agent 1: {args.agent1} ({type(agent1).__name__})")
        print(f"Agent 2: {args.agent2} ({type(agent2).__name__})")
        print(f"Games: {args.games}, Seed: {args.seed}")
        print()

    results = run_tournament(
        agent1=agent1,
        agent2=agent2,
        num_games=args.games,
        seed=args.seed,
        log_dir=args.log_dir,
        verbose=args.verbose,
    )

    print_summary(results)


if __name__ == "__main__":
    main()
