"""
Generic UCT-based Monte Carlo Tree Search algorithm for Ultimate Tic-Tac-Toe.

Uses a Protocol for the game state interface, making it reusable
for any game that implements the required interface.
"""

import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple


class GameStateProtocol(Protocol):
    """
    Protocol defining the interface required for MCTS to work with a game state.
    """

    @property
    def current_player(self) -> int:
        ...

    def get_valid_actions(self) -> List[List[int]]:
        ...

    def apply_action(self, x: int, y: int) -> Any:
        ...

    def is_terminal(self) -> bool:
        ...

    def get_winner(self) -> int:
        ...


class MCTSNode:
    """
    A node in the MCTS tree representing a game state.
    """

    __slots__ = (
        "state",
        "parent",
        "children",
        "visits",
        "wins",
        "untried_actions",
        "action_taken",
    )

    def __init__(
        self,
        state: GameStateProtocol,
        parent: Optional["MCTSNode"] = None,
        action_taken: Optional[List[int]] = None,
    ) -> None:
        """
        Initializes an MCTSNode.

        Args:
            state: The game state at this node.
            parent: The parent node, or None for the root.
            action_taken: The action [x, y] that led to this state, or None for root.
        """
        self.state = state
        self.parent = parent
        self.children: List["MCTSNode"] = []
        self.visits: int = 0
        self.wins: float = 0.0
        self.untried_actions: List[List[int]] = state.get_valid_actions()
        self.action_taken = action_taken

    @property
    def is_fully_expanded(self) -> bool:
        """Returns True if all actions have been tried."""
        return len(self.untried_actions) == 0

    @property
    def is_terminal_node(self) -> bool:
        """Returns True if this node represents a terminal state."""
        return self.state.is_terminal()

    def ucb1_value(self, exploration_constant: float = 1.414) -> float:
        """
        Calculates the UCB1 value for this node.

        Args:
            exploration_constant (float): The exploration constant (C).

        Returns:
            float: The UCB1 value.
        """
        if self.visits == 0:
            return float("inf")

        if self.parent is None:
            return self.wins / self.visits

        exploitation = self.wins / self.visits
        exploration = exploration_constant * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        return exploitation + exploration

    def best_child(self, exploration_constant: float = 1.414) -> "MCTSNode":
        """
        Selects the child with the highest UCB1 value.

        Args:
            exploration_constant (float): The exploration constant.

        Returns:
            MCTSNode: The child with the highest UCB1 value.

        Raises:
            ValueError: If the node has no children.
        """
        if not self.children:
            raise ValueError("No children to select from.")

        return max(
            self.children,
            key=lambda child: child.ucb1_value(exploration_constant),
        )

    def most_visited_child(self) -> Tuple["MCTSNode", List[int]]:
        """
        Selects the child with the highest visit count.

        Returns:
            Tuple[MCTSNode, List[int]]: The most visited child and its action.
        """
        if not self.children:
            raise ValueError("No children to select from.")

        best = max(self.children, key=lambda child: child.visits)
        return best, best.action_taken  # type: ignore[return-value]


class MCTS:
    """
    Generic UCT-based Monte Carlo Tree Search algorithm.
    """

    def __init__(
        self,
        iterations: int = 10000,
        exploration_constant: float = 1.414,
        time_limit: Optional[float] = None,
        random_seed: Optional[int] = None,
    ) -> None:
        """
        Initializes the MCTS solver.

        Args:
            iterations (int): Maximum number of MCTS iterations.
            exploration_constant (float): UCB1 exploration constant (C).
            time_limit (Optional[float]): Time limit in seconds. If set,
                this takes priority over iterations.
            random_seed (Optional[int]): Random seed for deterministic playouts.
        """
        self.iterations = iterations
        self.exploration_constant = exploration_constant
        self.time_limit = time_limit
        self.rng = random.Random(random_seed)
        self._total_iterations: int = 0
        self._tree_size: int = 0
        self._best_action: Optional[List[int]] = None
        self._best_action_visits: int = 0
        self._best_action_win_rate: float = 0.0
        self._last_stats: Dict[str, Any] = {}

    def search(self, state: GameStateProtocol) -> List[int]:
        """
        Runs the MCTS search from the given state.

        Args:
            state: The game state to search from.

        Returns:
            List[int]: The best action [x, y].

        Raises:
            RuntimeError: If there are no valid actions or the state is terminal.
        """
        valid_actions = state.get_valid_actions()
        if not valid_actions:
            raise RuntimeError("No valid actions available.")
        if state.is_terminal():
            raise RuntimeError("Cannot search from a terminal state.")

        # Optimization: if only one action, return immediately
        if len(valid_actions) == 1:
            return valid_actions[0]

        root = MCTSNode(state)
        self._total_iterations = 0
        start_time = time.monotonic()

        if self.time_limit is not None:
            # Time-limited search
            while True:
                elapsed = time.monotonic() - start_time
                if elapsed >= self.time_limit:
                    break
                self._run_iteration(root)
                self._total_iterations += 1
        else:
            # Iteration-limited search
            for _ in range(self.iterations):
                self._run_iteration(root)
                self._total_iterations += 1

        # Select the most visited child's action
        best_child, best_action = root.most_visited_child()
        self._tree_size = self._count_nodes(root)
        self._best_action = best_action
        self._best_action_visits = best_child.visits
        self._best_action_win_rate = (
            best_child.wins / best_child.visits if best_child.visits > 0 else 0.0
        )

        elapsed = time.monotonic() - start_time
        self._last_stats = {
            "total_iterations": self._total_iterations,
            "tree_size": self._tree_size,
            "root_visits": root.visits,
            "best_action": best_action,
            "best_action_visits": best_child.visits,
            "best_action_win_rate": self._best_action_win_rate,
            "elapsed_seconds": elapsed,
        }

        return best_action

    def _run_iteration(self, root: MCTSNode) -> None:
        """
        Runs one iteration of MCTS (SELECT, EXPAND, SIMULATE, BACKPROPAGATE).

        Args:
            root: The root node of the search tree.
        """
        # SELECT: traverse tree using UCB1 until non-terminal, non-fully-expanded node
        node = root
        while not node.is_terminal_node and node.is_fully_expanded and node.children:
            node = node.best_child(self.exploration_constant)

        # EXPAND: if node is not terminal and has untried actions, expand one
        if not node.is_terminal_node and not node.is_fully_expanded:
            action = node.untried_actions.pop()
            new_state = node.state.apply_action(action[0], action[1])
            child = MCTSNode(new_state, parent=node, action_taken=action)
            node.children.append(child)
            node = child

        # SIMULATE: random playout from the expanded/selected node
        winner = self._simulate(node.state)

        # BACKPROPAGATE: propagate the result up the tree
        self._backpropagate(node, winner)

    def _simulate(self, state: GameStateProtocol) -> int:
        """
        Runs a random playout from the given state to a terminal state.

        Args:
            state: The game state to simulate from.

        Returns:
            int: The winner (0=ongoing, 1=P1, 2=P2, 3=draw).
        """
        current_state = state
        while not current_state.is_terminal():
            actions = current_state.get_valid_actions()
            if not actions:
                break
            action = self.rng.choice(actions)
            current_state = current_state.apply_action(action[0], action[1])
        return current_state.get_winner()

    def _backpropagate(self, node: MCTSNode, winner: int) -> None:
        """
        Propagates the simulation result up the tree.

        The score is tracked from the perspective of the player who made the
        move TO this node (i.e., the moving player).

        Args:
            node: The leaf node to start backpropagation from.
            winner: The winner of the simulation (1, 2, 3, or 0).
        """
        current = node
        while current is not None:
            current.visits += 1
            if current.parent is not None:
                # current.state.current_player is the player to move NEXT.
                # The move that led to this node was made by the OTHER player.
                moving_player = 3 - current.state.current_player
                if winner == 3:
                    current.wins += 0.5
                elif winner == moving_player:
                    current.wins += 1.0
            current = current.parent

    def get_stats(self) -> Dict[str, Any]:
        """
        Returns statistics about the last search.

        Returns:
            Dict: Statistics including total_iterations, tree_size, root_visits,
                  best_action, best_action_visits, best_action_win_rate.
        """
        return dict(self._last_stats)

    @staticmethod
    def _count_nodes(node: MCTSNode) -> int:
        """
        Counts the total number of nodes in the tree.

        Args:
            node: The root node.

        Returns:
            int: The total number of nodes.
        """
        count = 1
        for child in node.children:
            count += MCTS._count_nodes(child)
        return count
