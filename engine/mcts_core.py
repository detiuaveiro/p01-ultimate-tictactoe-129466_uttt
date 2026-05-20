"""
Generic UCT-based Monte Carlo Tree Search algorithm for Ultimate Tic-Tac-Toe.

Uses a Protocol for the game state interface, making it reusable
for any game that implements the required interface.
"""

import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, cast


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
        "prior",
        "is_expanded",
        "_pending_priors",
    )

    def __init__(
        self,
        state: GameStateProtocol,
        parent: Optional["MCTSNode"] = None,
        action_taken: Optional[List[int]] = None,
        prior: float = 0.0,
    ) -> None:
        """
        Initializes an MCTSNode.

        Args:
            state: The game state at this node.
            parent: The parent node, or None for the root.
            action_taken: The action [x, y] that led to this state, or None for root.
            prior: The prior probability for this action (used in PUCT selection).
        """
        self.state = state
        self.parent = parent
        self.children: List["MCTSNode"] = []
        self.visits: int = 0
        self.wins: float = 0.0
        self.untried_actions: List[List[int]] = state.get_valid_actions()
        self.action_taken = action_taken
        self.prior: float = prior
        self.is_expanded: bool = False
        self._pending_priors: Optional[Dict[Tuple[int, int], float]] = None

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

    def detach_subtree(self) -> "MCTSNode":
        """Detach this node from its parent for subtree reuse.

        Removes this node from its parent's children list and clears the
        parent reference so the subtree can be used as a new root.

        Returns:
            The detached node (self).
        """
        if self.parent is not None:
            if self in self.parent.children:
                self.parent.children.remove(self)
            self.parent = None
        return self


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
        playout_fn: Optional[Callable[[GameStateProtocol, random.Random], int]] = None,
        prior_fn: Optional[Callable[[GameStateProtocol], Dict[Tuple[int, int], float]]] = None,
        value_fn: Optional[Callable[[GameStateProtocol], float]] = None,
        use_puct: bool = False,
    ) -> None:
        """
        Initializes the MCTS solver.

        Args:
            iterations (int): Maximum number of MCTS iterations.
            exploration_constant (float): UCB1 exploration constant (C).
            time_limit (Optional[float]): Time limit in seconds. If set,
                this takes priority over iterations.
            random_seed (Optional[int]): Random seed for deterministic playouts.
            playout_fn (Optional[Callable]): Optional function to replace the
                random playout phase.  Receives the state and the MCTS internal
                RNG, returns a winner (1, 2, or 3).  Default ``None`` uses
                random playouts (existing behaviour).
            prior_fn (Optional[Callable]): Optional function that takes a state
                and returns a dict of action -> prior probability. Used in PUCT
                selection to guide tree expansion.
            value_fn (Optional[Callable]): Optional function that takes a state
                and returns a value in [-1, 1] from the current player's
                perspective. Used to replace random simulation when use_puct
                is True.
            use_puct (bool): Whether to use PUCT selection (AlphaZero-style)
                instead of UCB1. Default False preserves existing UCB1 behaviour.
        """
        self.iterations = iterations
        self.exploration_constant = exploration_constant
        self.time_limit = time_limit
        self.rng = random.Random(random_seed)
        self.playout_fn = playout_fn
        self.prior_fn = prior_fn
        self.value_fn = value_fn
        self.use_puct = use_puct
        self._total_iterations: int = 0
        self._tree_size: int = 0
        self._best_action: Optional[List[int]] = None
        self._best_action_visits: int = 0
        self._best_action_win_rate: float = 0.0
        self._last_stats: Dict[str, Any] = {}
        self._last_root: Optional[MCTSNode] = None

    def search(
        self, state: GameStateProtocol, reuse_root: Optional["MCTSNode"] = None
    ) -> List[int]:
        """
        Runs the MCTS search from the given state.

        Args:
            state: The game state to search from.
            reuse_root: An optional previously-computed subtree root to reuse.
                The caller is responsible for ensuring ``reuse_root.state``
                matches *state* (typically by detaching the chosen child from
                a prior search).

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
            self._last_stats = {
                "total_iterations": 0,
                "tree_size": 1,
                "root_visits": 0,
                "best_action": valid_actions[0],
                "best_action_visits": 0,
                "best_action_win_rate": 0.0,
                "elapsed_seconds": 0.0,
            }
            self._last_root = None
            return valid_actions[0]

        if reuse_root is not None:
            root = reuse_root
        else:
            root = MCTSNode(state)
        self._total_iterations = 0
        start_time = time.monotonic()

        if self.time_limit is not None:
            # Time-limited search
            while True:
                # print('Running iteration')
                elapsed = time.monotonic() - start_time
                if elapsed >= self.time_limit:
                    break
                self._run_iteration(root)
                self._total_iterations += 1
        else:
            # Iteration-limited search
            for i in range(self.iterations):
                # print(f"Running iteration {i}/{self.iterations}")
                self._run_iteration(root)
                self._total_iterations += 1

        # Save root for visit distribution queries
        self._last_root = root

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

    def _puct_score(self, child: MCTSNode) -> float:
        """
        Computes the PUCT score for a child node, used in AlphaZero-style
        tree selection.  The formula is:

            Q(s,a) + C_puct * P(s,a) * sqrt(N_parent) / (1 + N_child)

        where Q(s,a) is the mean action value, P(s,a) is the prior
        probability, and C_puct is the exploration constant.

        Args:
            child: The child MCTSNode to score.

        Returns:
            float: The PUCT score. Returns infinitiy for unvisited children.
        """
        c_puct = self.exploration_constant
        parent_visits = child.parent.visits if child.parent else 1
        # AlphaZero-style: always use prior, even for unvisited children
        q_value = child.wins / child.visits if child.visits > 0 else 0.0
        exploration = (
            c_puct * child.prior * math.sqrt(parent_visits) / (1 + child.visits)
        )
        return q_value + exploration

    def _run_iteration(self, root: MCTSNode) -> None:
        """
        Runs one iteration of MCTS (SELECT, EXPAND, SIMULATE, BACKPROPAGATE).

        Args:
            root: The root node of the search tree.
        """
        # SELECT: traverse tree using UCB1 or PUCT until non-terminal,
        # non-fully-expanded node
        node = root
        while not node.is_terminal_node and node.is_fully_expanded and node.children:
            if self.use_puct:
                node = max(node.children, key=lambda c: self._puct_score(c))
            else:
                node = node.best_child(self.exploration_constant)

        # EXPAND: if node is not terminal and has untried actions, expand one
        if not node.is_terminal_node and not node.is_fully_expanded:
            # Call prior_fn if this node hasn't been expanded yet
            if self.prior_fn is not None and not node.is_expanded:
                node._pending_priors = self.prior_fn(node.state)
                node.is_expanded = True

            action = node.untried_actions.pop()
            new_state = node.state.apply_action(action[0], action[1])

            # Look up prior from pending_priors if available
            prior = 0.0
            if node._pending_priors is not None:
                prior = node._pending_priors.get(
                    cast(Tuple[int, int], tuple(action)), 0.0
                )

            child = MCTSNode(
                new_state, parent=node, action_taken=action, prior=prior
            )
            node.children.append(child)
            node = child

        # SIMULATE: random playout from the expanded/selected node
        winner = self._simulate(node.state)

        # BACKPROPAGATE: propagate the result up the tree
        self._backpropagate(node, winner)

    def _simulate(self, state: GameStateProtocol) -> float:
        """
        Runs a playout from the given state to a terminal state and returns
        the result as a continuous value in ``[-1, 1]`` from the perspective
        of ``state.current_player``.

        If ``use_puct`` is True and a ``value_fn`` was provided at
        construction, the value function is used to directly evaluate the
        leaf state (no random playout).  Otherwise, if a ``playout_fn``
        was provided, it is used instead of the default random playout.
        The playout function receives the state and the MCTS internal RNG
        for reproducibility.

        Args:
            state: The game state to simulate from.

        Returns:
            float: A value in ``[-1, 1]`` from the current player's
            perspective.  ``+1`` = win, ``0`` = draw, ``-1`` = loss.
        """
        # Terminal state shortcut: return exact outcome
        if state.is_terminal():
            winner = state.get_winner()
            if winner == 3:
                return 0.0
            elif winner == state.current_player:
                return 1.0
            else:
                return -1.0

        # AlphaZero-lite: value_fn directly evaluates the leaf node
        if self.use_puct and self.value_fn is not None:
            return self.value_fn(state)

        # Custom playout function (e.g., heuristic-guided rollouts)
        if self.playout_fn is not None:
            winner = self.playout_fn(state, self.rng)
        else:
            # Default random playout (existing behaviour)
            current_state = state
            while not current_state.is_terminal():
                actions = current_state.get_valid_actions()
                if not actions:
                    break
                action = self.rng.choice(actions)
                current_state = current_state.apply_action(action[0], action[1])
            winner = current_state.get_winner()

        # Convert winner to value from current player's perspective
        if winner == 3:
            return 0.0
        elif winner == state.current_player:
            return 1.0
        else:
            return -1.0

    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        """
        Propagates the simulation result up the tree.

        The *value* is the leaf evaluation from the perspective of
        ``node.state.current_player``.  As we move up the tree, the sign is
        flipped at each level so that the value stored at each edge/node
        reflects the perspective of the player who moved TO reach it.

        This matches the standard AlphaZero approach where ``v`` from the
        leaf is returned upward with ``return -v`` at each ply.

        Args:
            node: The leaf node to start backpropagation from.
            value: Leaf evaluation in ``[-1, 1]`` from the leaf state's
                current-player perspective.
        """
        current = node
        while current is not None:
            current.visits += 1
            if current.parent is not None:
                # The move that led to *current* was made by the OTHER player
                # (3 - current.state.current_player).  *value* is from
                # current.state.current_player perspective, so negate to get
                # the moving player's perspective.
                current.wins += -value
            current = current.parent
            value = -value

    def get_stats(self) -> Dict[str, Any]:
        """
        Returns statistics about the last search.

        Returns:
            Dict: Statistics including total_iterations, tree_size, root_visits,
                  best_action, best_action_visits, best_action_win_rate.
        """
        return dict(self._last_stats)

    def get_root_visit_distribution(self) -> Dict[Tuple[int, int], float]:
        """
        Returns the visit counts for each child of the root node from the
        most recent search.  This provides the search policy distribution
        used in AlphaZero-style training.

        Returns:
            Dict[Tuple[int, int], float]: Mapping from ``(x, y)`` actions to
            visit counts.  Returns an empty dict if no search has been run.
        """
        if self._last_root is None:
            return {}
        return {
            cast(Tuple[int, int], tuple(c.action_taken)): float(c.visits)
            for c in self._last_root.children
            if c.action_taken is not None
        }

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
