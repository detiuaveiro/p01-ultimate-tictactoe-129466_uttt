import asyncio
import random
from typing import List, Optional, Tuple, Union

from agents.base_agent import BaseUTTTAgent
from engine.game_state import UTTTState


class DummyUTTTAgent(BaseUTTTAgent):
    """
    A simple Ultimate Tic-Tac-Toe agent that picks moves randomly.
    """

    def __init__(
        self,
        server_uri: str = "ws://localhost:8765",
        random_seed: Optional[int] = None,
    ) -> None:
        """
        Initializes the DummyUTTTAgent.

        Args:
            server_uri (str): The URI of the UTTT server.
            random_seed (Optional[int]): Random seed for deterministic behavior.
        """
        super().__init__(server_uri)
        self.random_seed = random_seed
        self._rng = random.Random(random_seed)

    async def deliberate(
        self,
        board: List[List[int]],
        macro_board: List[List[int]],
        active_macro: Optional[List[int]],
        valid_actions: List[List[int]],
    ) -> Optional[Union[List[int], Tuple[int, int]]]:
        """
        Randomly selects a move from the available valid actions.

        Args:
            board (List[List[int]]): The current 9x9 board state.
            macro_board (List[List[int]]): The current 3x3 macro board state.
            active_macro (Optional[List[int]]): The active macro board coordinates [my, mx].
            valid_actions (List[List[int]]): A list of valid moves [x, y].

        Returns:
            Optional[Union[List[int], Tuple[int, int]]]: The chosen move [x, y] or None.
        """
        await asyncio.sleep(0.5)
        if not valid_actions:
            return None
        return self._rng.choice(valid_actions)

    def deliberate_from_state(
        self, state: "UTTTState"
    ) -> Optional[List[int]]:
        """Synchronous deliberation from a UTTTState for headless play.

        Args:
            state (UTTTState): The game state to deliberate from.

        Returns:
            Optional[List[int]]: The chosen action [x, y], or None if no valid actions
                or the state is terminal.
        """
        if state.is_terminal() or not state.get_valid_actions():
            return None
        return self._rng.choice(state.get_valid_actions())


if __name__ == "__main__":
    agent = DummyUTTTAgent()
    asyncio.run(agent.run())
