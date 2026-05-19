"""
MCTS + Heuristics agent for Ultimate Tic-Tac-Toe.

Combines Monte Carlo Tree Search with heuristic-guided rollouts
and leaf evaluation for stronger play than pure MCTS.
"""

import argparse
import asyncio
import logging
import random
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from agents.base_agent import BaseUTTTAgent
from engine.game_state import UTTTState
from engine.heuristics import HeuristicEvaluator
from engine.mcts_core import MCTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - AGENT - %(message)s")


class MCTSHeuristicAgent(BaseUTTTAgent):
    """
    An Ultimate Tic-Tac-Toe agent that uses MCTS with heuristic-guided
    rollouts and leaf evaluation to decide moves.

    The agent combines the strategic lookahead of MCTS with the tactical
    knowledge encoded in a heuristic evaluator.  Heuristic parameters
    (weights, playout bias, max depth) are fully configurable.

    The heuristic evaluator is lazily initialised on first access to
    avoid pickling issues when used with ``multiprocessing``.
    """

    def __init__(
        self,
        server_uri: str = "ws://localhost:8765",
        mcts_iterations: int = 100,
        mcts_exploration_constant: float = 1.414,
        mcts_time_limit: Optional[float] = None,
        random_seed: Optional[int] = None,
        heuristic_weights: Optional[Dict[str, float]] = None,
        heuristic_playout_bias: float = 0.8,
        heuristic_max_depth: int = 50,
    ) -> None:
        """
        Initializes the MCTSHeuristicAgent.

        Args:
            server_uri: WebSocket server URI.
            mcts_iterations: Maximum MCTS iterations per move.
            mcts_exploration_constant: UCB1 exploration constant.
            mcts_time_limit: Time limit per move in seconds (optional).
            random_seed: Random seed for deterministic playouts.
            heuristic_weights: Optional custom heuristic weights dict.
            heuristic_playout_bias: Probability of selecting the heuristic-best
                move vs random (must be in [0, 1]).
            heuristic_max_depth: Maximum depth for heuristic playouts
                (must be >= 0).

        Raises:
            AssertionError: If *heuristic_playout_bias* is outside [0, 1]
                or *heuristic_max_depth* is negative.
        """
        super().__init__(server_uri)
        self.mcts_iterations = mcts_iterations
        self.mcts_exploration_constant = mcts_exploration_constant
        self.mcts_time_limit = mcts_time_limit
        self.random_seed = random_seed
        self.heuristic_weights = heuristic_weights
        self.heuristic_playout_bias = heuristic_playout_bias
        self.heuristic_max_depth = heuristic_max_depth

        # Validate heuristic parameters
        assert 0.0 <= self.heuristic_playout_bias <= 1.0, (
            f"heuristic_playout_bias must be in [0, 1], got {self.heuristic_playout_bias}"
        )
        assert self.heuristic_max_depth >= 0, (
            f"heuristic_max_depth must be >= 0, got {self.heuristic_max_depth}"
        )

        # Lazy evaluator — created on first access (not in __init__) to
        # avoid pickling issues with multiprocessing.
        self._evaluator: Optional[HeuristicEvaluator] = None
        self._last_stats: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _get_evaluator(self) -> HeuristicEvaluator:
        """
        Returns the heuristic evaluator, creating it lazily if needed.

        Returns:
            HeuristicEvaluator: The heuristic evaluator instance.
        """
        if self._evaluator is None:
            self._evaluator = HeuristicEvaluator(weights=self.heuristic_weights)
        return self._evaluator

    # ------------------------------------------------------------------
    # Playout factory
    # ------------------------------------------------------------------

    def _make_playout_fn(self) -> Callable[..., int]:
        """
        Returns a closure wrapping ``HeuristicEvaluator.heuristic_playout``
        with the agent's configuration bound in.

        The closure captures the evaluator, bias, and max_depth so that
        it can be passed to the MCTS constructor with no per-call overhead
        for configuration lookups.

        Returns:
            Callable: A function ``(state, rng) -> winner``.
        """
        evaluator = self._get_evaluator()
        bias = self.heuristic_playout_bias
        max_depth = self.heuristic_max_depth

        def _playout(state: UTTTState, rng: random.Random) -> int:
            # state is always UTTTState at runtime
            return evaluator.heuristic_playout(
                state, rng, max_depth=max_depth, playout_bias=bias
            )

        return _playout

    # ------------------------------------------------------------------
    # Deliberation
    # ------------------------------------------------------------------

    async def deliberate(
        self,
        board: List[List[int]],
        macro_board: List[List[int]],
        active_macro: Optional[List[int]],
        valid_actions: List[List[int]],
    ) -> Optional[Union[List[int], Tuple[int, int]]]:
        """
        Deliberates on the next move using MCTS + heuristic rollouts.

        Args:
            board: The current 9x9 board state.
            macro_board: The current 3x3 macro board.
            active_macro: The active macro-board [my, mx] or None.
            valid_actions: List of valid moves [x, y].

        Returns:
            Optional[Union[List[int], Tuple[int, int]]]: The chosen move
            [x, y] or None.
        """
        if not valid_actions:
            return None

        # Construct state from server data
        state = UTTTState(
            board=board,
            macro_board=macro_board,
            active_macro=active_macro,
            current_player=self.player_id,  # type: ignore[arg-type]
        )

        if state.is_terminal():
            return None

        start_time = time.monotonic()
        mcts = MCTS(
            iterations=self.mcts_iterations,
            exploration_constant=self.mcts_exploration_constant,
            time_limit=self.mcts_time_limit,
            random_seed=self.random_seed,
            playout_fn=self._make_playout_fn(),
        )

        try:
            action = mcts.search(state)
        except RuntimeError:
            return valid_actions[0] if valid_actions else None

        elapsed = time.monotonic() - start_time
        stats = mcts.get_stats()
        self._last_stats = stats

        logging.info(
            f"MCTS+Heuristic search completed in {elapsed:.3f}s: "
            f"action={action}, "
            f"iterations={stats['total_iterations']}, "
            f"tree_size={stats['tree_size']}, "
            f"win_rate={stats['best_action_win_rate']:.3f}"
        )

        # Safety: ensure the returned action is valid
        action_list = list(action) if action else None
        if action_list is not None and action_list not in valid_actions:
            logging.warning(
                f"MCTS+Heuristic returned invalid action {action_list}, "
                f"falling back to {valid_actions[0]}"
            )
            return valid_actions[0]

        return action_list

    def deliberate_from_state(
        self, state: UTTTState
    ) -> Optional[List[int]]:
        """
        Deliberates from a UTTTState directly (for headless testing).

        Args:
            state: The game state to search from.

        Returns:
            Optional[List[int]]: The best action [x, y], or None.
        """
        if state.is_terminal() or not state.get_valid_actions():
            return None

        mcts = MCTS(
            iterations=self.mcts_iterations,
            exploration_constant=self.mcts_exploration_constant,
            time_limit=self.mcts_time_limit,
            random_seed=self.random_seed,
            playout_fn=self._make_playout_fn(),
        )

        try:
            action = mcts.search(state)
        except RuntimeError:
            return None

        self._last_stats = mcts.get_stats()
        return action

    def get_last_stats(self) -> Dict[str, Any]:
        """
        Returns statistics from the last deliberation.

        Returns:
            Dict[str, Any]: The stats dictionary.
        """
        return dict(self._last_stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "MCTS + Heuristics agent for Ultimate Tic-Tac-Toe. "
            "Combines MCTS with heuristic-guided rollouts and leaf evaluation."
        )
    )
    parser.add_argument(
        "--server-uri",
        type=str,
        default="ws://localhost:8765",
        help="WebSocket server URI (default: ws://localhost:8765)",
    )
    parser.add_argument(
        "--iterations",
        "-i",
        type=int,
        default=10000,
        help="MCTS iterations per move (default: 10000)",
    )
    parser.add_argument(
        "--exploration-constant",
        "-c",
        type=float,
        default=1.414,
        help="UCB1 exploration constant C (default: 1.414)",
    )
    parser.add_argument(
        "--time-limit",
        "-t",
        type=float,
        default=None,
        help="Time limit per move in seconds (default: no limit)",
    )
    parser.add_argument(
        "--random-seed",
        "-s",
        type=int,
        default=None,
        help="Random seed for reproducible playouts (default: no seed)",
    )
    parser.add_argument(
        "--playout-bias",
        "-b",
        type=float,
        default=0.8,
        help="Heuristic playout bias (0-1, default: 0.8)",
    )
    parser.add_argument(
        "--max-depth",
        "-d",
        type=int,
        default=50,
        help="Heuristic playout max depth (default: 50)",
    )
    parser.add_argument(
        "--heuristic-weights",
        type=str,
        default=None,
        help=(
            "JSON string of heuristic weight overrides "
            "(e.g. '{\"micro_win\": 150.0}')"
        ),
    )

    args = parser.parse_args()

    # Parse heuristic weights JSON if provided
    weights = None
    if args.heuristic_weights:
        import json

        weights = json.loads(args.heuristic_weights)

    logging.info(
        "Starting MCTSHeuristicAgent with config: "
        f"server_uri={args.server_uri}, "
        f"iterations={args.iterations}, "
        f"exploration_constant={args.exploration_constant}, "
        f"time_limit={args.time_limit}, "
        f"random_seed={args.random_seed}, "
        f"playout_bias={args.playout_bias}, "
        f"max_depth={args.max_depth}, "
        f"heuristic_weights={weights}"
    )

    agent = MCTSHeuristicAgent(
        server_uri=args.server_uri,
        mcts_iterations=args.iterations,
        mcts_exploration_constant=args.exploration_constant,
        mcts_time_limit=args.time_limit,
        random_seed=args.random_seed,
        heuristic_weights=weights,
        heuristic_playout_bias=args.playout_bias,
        heuristic_max_depth=args.max_depth,
    )
    asyncio.run(agent.run())
