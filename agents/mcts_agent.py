"""
MCTS-based agent for Ultimate Tic-Tac-Toe.
"""

import argparse
import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from agents.base_agent import BaseUTTTAgent
from engine.game_state import UTTTState
from engine.mcts_core import MCTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - AGENT - %(message)s")


class MCTSAgent(BaseUTTTAgent):
    """
    An Ultimate Tic-Tac-Toe agent that uses MCTS to decide moves.
    """

    def __init__(
        self,
        server_uri: str = "ws://localhost:8765",
        mcts_iterations: int = 10000,
        mcts_exploration_constant: float = 1.414,
        mcts_time_limit: Optional[float] = None,
        random_seed: Optional[int] = None,
    ) -> None:
        """
        Initializes the MCTSAgent.

        Args:
            server_uri (str): The URI of the UTTT server.
            mcts_iterations (int): Maximum MCTS iterations per move.
            mcts_exploration_constant (float): UCB1 exploration constant.
            mcts_time_limit (Optional[float]): Time limit in seconds per move.
            random_seed (Optional[int]): Random seed for deterministic playouts.
        """
        super().__init__(server_uri)
        self.mcts_iterations = mcts_iterations
        self.mcts_exploration_constant = mcts_exploration_constant
        self.mcts_time_limit = mcts_time_limit
        self.random_seed = random_seed
        self._last_stats: Dict[str, Any] = {}

    async def deliberate(
        self,
        board: List[List[int]],
        macro_board: List[List[int]],
        active_macro: Optional[List[int]],
        valid_actions: List[List[int]],
    ) -> Optional[Union[List[int], Tuple[int, int]]]:
        """
        Deliberates on the next move using MCTS.

        Args:
            board (List[List[int]]): The current 9x9 board state.
            macro_board (List[List[int]]): The current 3x3 macro board state.
            active_macro (Optional[List[int]]): The active macro board [my, mx].
            valid_actions (List[List[int]]): A list of valid moves [x, y].

        Returns:
            Optional[Union[List[int], Tuple[int, int]]]: The chosen move [x, y] or None.
        """
        if not valid_actions:
            return None

        # Construct the state from server data
        state = UTTTState(
            board=board,
            macro_board=macro_board,
            active_macro=active_macro,
            current_player=self.player_id,  # type: ignore[arg-type]
        )

        # Safety: if state has no valid actions (shouldn't happen if valid_actions is non-empty)
        if state.is_terminal():
            return None

        start_time = time.monotonic()
        mcts = MCTS(
            iterations=self.mcts_iterations,
            exploration_constant=self.mcts_exploration_constant,
            time_limit=self.mcts_time_limit,
            random_seed=self.random_seed,
        )

        try:
            action = mcts.search(state)
        except RuntimeError:
            return valid_actions[0] if valid_actions else None

        elapsed = time.monotonic() - start_time
        stats = mcts.get_stats()
        self._last_stats = stats

        logging.info(
            f"MCTS search completed in {elapsed:.3f}s: "
            f"action={action}, "
            f"iterations={stats['total_iterations']}, "
            f"tree_size={stats['tree_size']}, "
            f"win_rate={stats['best_action_win_rate']:.3f}"
        )

        # Safety: ensure the returned action is valid
        action_list = list(action) if action else None
        if action_list is not None and action_list not in valid_actions:
            logging.warning(
                f"MCTS returned invalid action {action_list}, "
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
            state (UTTTState): The game state to search from.

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
        description="MCTS-based agent for Ultimate Tic-Tac-Toe."
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

    args = parser.parse_args()

    logging.info(
        "Starting MCTSAgent with config: "
        f"server_uri={args.server_uri}, "
        f"iterations={args.iterations}, "
        f"exploration_constant={args.exploration_constant}, "
        f"time_limit={args.time_limit}, "
        f"random_seed={args.random_seed}"
    )

    agent = MCTSAgent(
        server_uri=args.server_uri,
        mcts_iterations=args.iterations,
        mcts_exploration_constant=args.exploration_constant,
        mcts_time_limit=args.time_limit,
        random_seed=args.random_seed,
    )
    asyncio.run(agent.run())
