import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

import websockets
from websockets.server import WebSocketServerProtocol

from engine.game_rules import (
    check_3x3_win as rules_check_3x3_win,
    get_global_winner,
    get_valid_actions as rules_get_valid_actions,
    is_3x3_full as rules_is_3x3_full,
    apply_move as rules_apply_move,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class UTTTServer:
    """
    Ultimate Tic-Tac-Toe (UTTT) Server.

    Handles map loading, agent movement, and state broadcasting.
    The game is played on a $9 \times 9$ micro-board, divided into $3 \times 3$ macro-boards.
    """

    def __init__(self) -> None:
        """
        Initializes the UTTTServer.
        """
        self.frontend_ws: Optional[WebSocketServerProtocol] = None
        self.agent1_ws: Optional[WebSocketServerProtocol] = None
        self.agent2_ws: Optional[WebSocketServerProtocol] = None

        # 9x9 Micro Board (0=Empty, 1=P1, 2=P2)
        self.board: List[List[int]] = [[0] * 9 for _ in range(9)]
        # 3x3 Macro Board (0=Ongoing, 1=P1 Win, 2=P2 Win, 3=Draw)
        self.macro_board: List[List[int]] = [[0] * 3 for _ in range(3)]

        # (my, mx) indicating which macro-board the current player MUST play in.
        # None means the player can play in ANY available macro-board.
        self.active_macro: Optional[List[int]] = None

        self.first_player_this_round: int = 1
        self.current_turn: int = 1
        self.running: bool = False
        self.match_scores: Dict[int, int] = {1: 0, 2: 0}

    async def start(self, host: str = "0.0.0.0", port: int = 8765) -> None:
        """
        Starts the UTTT server.

        Args:
            host (str): The host address to bind to.
            port (int): The port to listen on.
        """
        logging.info(f"UTTT Server started on ws://{host}:{port}")
        async with websockets.serve(self.handle_client, host, port):
            await asyncio.Future()

    async def handle_client(self, websocket: WebSocketServerProtocol) -> None:
        """
        Handles incoming WebSocket connections.

        Args:
            websocket (WebSocketServerProtocol): The connected WebSocket client.
        """
        client_type = "Unknown"
        try:
            init_msg = await websocket.recv()
            if isinstance(init_msg, bytes):
                init_msg = init_msg.decode("utf-8")
            data: Dict[str, Any] = json.loads(init_msg)
            client_type = data.get("client", "Unknown")

            if client_type == "frontend":
                logging.info("Frontend connected.")
                self.frontend_ws = websocket
                await self.update_frontend()
                await self.frontend_loop(websocket)
            elif client_type == "agent":
                if not self.agent1_ws:
                    self.agent1_ws = websocket
                    logging.info("Player 1 (X) connected.")
                    await websocket.send(json.dumps({"type": "setup", "player_id": 1}))
                    # Start the agent loop and check conditions in parallel
                    await asyncio.gather(
                        self.agent_loop(websocket, 1),
                        self.check_start_conditions()
                    )
                elif not self.agent2_ws:
                    self.agent2_ws = websocket
                    logging.info("Player 2 (O) connected.")
                    await websocket.send(json.dumps({"type": "setup", "player_id": 2}))
                    # Start the agent loop and check conditions in parallel
                    await asyncio.gather(
                        self.agent_loop(websocket, 2),
                        self.check_start_conditions()
                    )
                else:
                    await websocket.close()
        except Exception as e:
            logging.error(f"Error: {e}")
        finally:
            if websocket == self.frontend_ws:
                self.frontend_ws = None
            elif websocket == self.agent1_ws:
                self.agent1_ws = None
                self.running = False
                await self.update_frontend()
            elif websocket == self.agent2_ws:
                self.agent2_ws = None
                self.running = False
                await self.update_frontend()

    async def frontend_loop(self, websocket: WebSocketServerProtocol) -> None:
        """
        Main loop for handling frontend communication.

        Args:
            websocket (WebSocketServerProtocol): The connected frontend client.
        """
        async for _ in websocket:
            pass

    async def agent_loop(self, websocket: WebSocketServerProtocol, player_id: int) -> None:
        """
        Main loop for handling agent communication.

        Args:
            websocket (WebSocketServerProtocol): The connected agent client.
            player_id (int): The ID of the player (1 or 2).
        """
        async for message in websocket:
            if not self.running or self.current_turn != player_id:
                continue
            try:
                if isinstance(message, bytes):
                    message = message.decode("utf-8")
                data: Dict[str, Any] = json.loads(message)
                if data.get("action") == "move":
                    x, y = data.get("x"), data.get("y")
                    if x is not None and y is not None and self.process_move(player_id, x, y):
                        await self.check_game_over()
                        if self.running:
                            self.current_turn = 3 - self.current_turn
                            await self.broadcast_state()
                            await self.update_frontend()
            except Exception as e:
                logging.error(f"Error processing move: {e}")

    async def check_start_conditions(self) -> None:
        """
        Checks if both agents are connected and starts the game if so.
        """
        if self.agent1_ws and self.agent2_ws and not self.running:
            self.running = True
            self.board = [[0] * 9 for _ in range(9)]
            self.macro_board = [[0] * 3 for _ in range(3)]
            self.active_macro = None
            self.current_turn = self.first_player_this_round
            await self.update_frontend()
            # Give enough time for the loops to start and agents to be ready
            await asyncio.sleep(1.0)
            await self.broadcast_state()

    def get_valid_actions(self) -> List[List[int]]:
        """
        Returns a list of all currently valid actions for the current player.

        Returns:
            List[List[int]]: A list of [x, y] coordinates representing valid moves.
        """
        return rules_get_valid_actions(self.board, self.macro_board, self.active_macro)

    def process_move(self, player_id: int, x: int, y: int) -> bool:
        """
        Processes a move from a player.

        Args:
            player_id (int): The ID of the player making the move.
            x (int): The x-coordinate of the move.
            y (int): The y-coordinate of the move.

        Returns:
            bool: True if the move was processed successfully, False otherwise.
        """
        if [x, y] not in self.get_valid_actions():
            return False

        self.board, self.macro_board, self.active_macro = rules_apply_move(
            self.board, self.macro_board, player_id, x, y
        )

        return True

    def check_3x3_win(self, grid: List[List[int]], start_x: int, start_y: int) -> int:
        """
        Checks for a win in a specific 3x3 subset of a grid.

        Delegates to engine.game_rules.check_3x3_win.

        Args:
            grid (List[List[int]]): The grid to check (9x9 or 3x3).
            start_x (int): The starting x-coordinate of the 3x3 subset.
            start_y (int): The starting y-coordinate of the 3x3 subset.

        Returns:
            int: The ID of the winning player (1 or 2), 0 if no winner.
        """
        return rules_check_3x3_win(grid, start_x, start_y)

    def is_3x3_full(self, grid: List[List[int]], start_x: int, start_y: int) -> bool:
        """
        Checks if a 3x3 subset of a grid is full.

        Delegates to engine.game_rules.is_3x3_full.

        Args:
            grid (List[List[int]]): The grid to check.
            start_x (int): The starting x-coordinate of the 3x3 subset.
            start_y (int): The starting y-coordinate of the 3x3 subset.

        Returns:
            bool: True if the 3x3 subset is full, False otherwise.
        """
        return rules_is_3x3_full(grid, start_x, start_y)

    async def check_game_over(self) -> None:
        """
        Checks if the game is over and handles the results.
        """
        winner = get_global_winner(self.macro_board)

        if winner in [1, 2]:
            self.match_scores[winner] += 1
            await self.end_round(f"Player {winner} Wins!")
        elif winner == 3:
            await self.end_round("Global Draw!")

    async def end_round(self, message: str) -> None:
        """
        Ends the current round.

        Args:
            message (str): The message to display at the end of the round.
        """
        self.running = False
        payload = {"type": "game_over", "message": message}
        if self.agent1_ws:
            await self.agent1_ws.send(json.dumps(payload))
        if self.agent2_ws:
            await self.agent2_ws.send(json.dumps(payload))
        await self.update_frontend()

        await asyncio.sleep(3.0)
        self.first_player_this_round = 3 - self.first_player_this_round
        await self.check_start_conditions()

    async def broadcast_state(self) -> None:
        """
        Broadcasts the current game state to both agents.
        """
        payload = {
            "type": "state",
            "current_turn": self.current_turn,
            "board": self.board,
            "macro_board": self.macro_board,
            "active_macro": self.active_macro,
            "valid_actions": self.get_valid_actions(),
        }
        msg = json.dumps(payload)
        if self.agent1_ws:
            await self.agent1_ws.send(msg)
        if self.agent2_ws:
            await self.agent2_ws.send(msg)

    async def update_frontend(self) -> None:
        """
        Sends an update to the frontend.
        """
        if self.frontend_ws:
            await self.frontend_ws.send(
                json.dumps(
                    {
                        "type": "update",
                        "current_turn": self.current_turn,
                        "board": self.board,
                        "macro_board": self.macro_board,
                        "active_macro": self.active_macro,
                        "match_scores": self.match_scores,
                        "p1_connected": self.agent1_ws is not None,
                        "p2_connected": self.agent2_ws is not None,
                    }
                )
            )


if __name__ == "__main__":
    server = UTTTServer()
    asyncio.run(server.start())
