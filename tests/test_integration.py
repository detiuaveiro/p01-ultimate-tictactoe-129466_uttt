"""
Integration tests for the UTTT engine and MCTS agent.
"""

import random
from typing import List

from agents.mcts_agent import MCTSAgent
from engine.game_rules import (
    apply_move,
    get_valid_actions,
)
from engine.game_state import UTTTState


def test_game_rules_to_state_consistency() -> None:
    """game_rules functions and UTTTState agree on valid actions."""
    board = [[0] * 9 for _ in range(9)]
    macro_board = [[0] * 3 for _ in range(3)]

    # Make some moves directly with game_rules
    board, macro_board, active_macro = apply_move(board, macro_board, 1, 0, 0)
    board, macro_board, active_macro = apply_move(board, macro_board, 2, 3, 3)
    board, macro_board, active_macro = apply_move(board, macro_board, 1, 1, 1)

    # Create equivalent state
    state = UTTTState(
        board=board,
        macro_board=macro_board,
        active_macro=active_macro,
        current_player=2,
    )

    # Both should agree on valid actions
    rules_actions = get_valid_actions(board, macro_board, active_macro)
    state_actions = state.get_valid_actions()

    # Sort for comparison
    assert sorted(rules_actions) == sorted(state_actions)


def test_full_game_simulation_with_random_moves() -> None:
    """A full game can be simulated with random moves via UTTTState."""
    state = UTTTState()
    move_count = 0
    max_moves = 81  # maximum possible moves

    while not state.is_terminal() and move_count < max_moves:
        actions = state.get_valid_actions()
        assert len(actions) > 0, "No valid actions in non-terminal state"

        # Pick a random action
        action = random.choice(actions)
        state = state.apply_action(action[0], action[1])
        move_count += 1

    # Game should have ended
    winner = state.get_winner()
    assert winner in [0, 1, 2, 3]
    if winner in [1, 2]:
        assert state.is_terminal()
    elif winner == 3:
        assert state.is_terminal()


def test_full_game_simulation_alternative_moves() -> None:
    """A full game can reach a terminal state with different move patterns."""
    state = UTTTState()
    rng = random.Random(42)
    move_count = 0
    max_moves = 81

    while not state.is_terminal() and move_count < max_moves:
        actions = state.get_valid_actions()
        assert len(actions) > 0
        action = rng.choice(actions)
        state = state.apply_action(action[0], action[1])
        move_count += 1

    winner = state.get_winner()
    assert winner in [0, 1, 2, 3]
    # With 81 random moves and the random seed 42, the game should finish
    if winner == 0:
        # If no winner, the board must be completely full (draw)
        assert all(state.board[y][x] != 0 for y in range(9) for x in range(9))


def test_mcts_agent_makes_valid_moves_during_game() -> None:
    """MCTSAgent makes valid moves throughout a simulated game."""
    state = UTTTState()
    agent = MCTSAgent(mcts_iterations=50, random_seed=42)
    move_count = 0
    max_moves = 50  # Limit to keep test fast

    while not state.is_terminal() and move_count < max_moves:
        action = agent.deliberate_from_state(state)
        assert action is not None, "Agent returned None for non-terminal state"

        valid = state.get_valid_actions()
        assert action in valid, (
            f"Agent returned invalid action {action}. "
            f"Valid: {valid}"
        )

        state = state.apply_action(action[0], action[1])
        move_count += 1


def test_game_rules_and_state_track_active_macro() -> None:
    """Both game_rules and UTTTState track active_macro consistently."""
    # Start with a free move
    board = [[0] * 9 for _ in range(9)]
    macro_board = [[0] * 3 for _ in range(3)]
    active_macro = None

    # Play at (4, 4) — center of center macro
    board, macro_board, active_macro = apply_move(board, macro_board, 1, 4, 4)

    # The active macro should be [1, 1] (since micro is at (1, 1))
    assert active_macro == [1, 1], f"Expected [1, 1], got {active_macro}"

    # Create state and verify
    state = UTTTState(
        board=board,
        macro_board=macro_board,
        active_macro=active_macro,
        current_player=2,
    )
    assert state.active_macro == [1, 1]

    # Both should agree on valid actions (only macro (1, 1) cells)
    rules_actions = get_valid_actions(board, macro_board, active_macro)
    state_actions = state.get_valid_actions()
    assert sorted(rules_actions) == sorted(state_actions)
    assert all(3 <= a[0] <= 5 and 3 <= a[1] <= 5 for a in rules_actions)


def test_mcts_with_limited_time() -> None:
    """MCTS with time limit works correctly."""
    state = UTTTState()
    agent = MCTSAgent(mcts_time_limit=0.05, random_seed=42)
    action = agent.deliberate_from_state(state)
    assert action is not None
    valid = state.get_valid_actions()
    assert action in valid
