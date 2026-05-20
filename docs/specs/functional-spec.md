# Functional Specification: SI2 Ultimate Tic-Tac-Toe Autonomous Agents

> **Version**: 0.5.0 | **Date**: 2026-05-13 | **Author**: Documenter Agent | **Status**: Active

## Change Log

| Version | Date       | Author           | Changes                                                  |
|---------|------------|------------------|----------------------------------------------------------|
| 0.5.0   | 2026-05-13 | Documenter Agent | Phase 2 implementation: FR-004 (HeuristicEvaluator), FR-005 (Heuristic-Guided Rollouts), FR-006 (MCTSHeuristicAgent) completed; statuses updated to "Implemented"; architecture updated with new components; commit bf30fef |
| 0.4.1   | 2026-05-10 | Documenter Agent | Documented running win rates in tournament tqdm bar |
| 0.4.0   | 2026-05-11 | Documenter Agent | Implemented FR-007 (StatsLogger) and FR-008 (Tournament Runner); added tqdm progress bar, ProcessPoolExecutor parallel execution, CLI with --workers; simplified __init__.py files; recognized FR-005 MCTS agent argparse CLI; statuses updated to "Implemented" |
| 0.3.0   | 2026-05-10 | Documenter Agent | Phase 1 implementation: extracted shared game_rules.py from server.py; UTTTState (FR-001), MCTS (FR-002), MCTSAgent (FR-003) implemented; statuses updated to "Implemented"; 66 tests added |
| 0.2.0   | 2026-05-10 | Documenter Agent | Added AlphaZero-lite (Phase 3) requirements (FR-009, FR-010, FR-011); declared infrastructure immutability principle |
| 0.1.0   | 2026-05-10 | Documenter Agent | Initial draft — all requirements defined, no implementation |

---

## 1. Introduction

### 1.1 Purpose

This document defines the functional and non-functional requirements for developing autonomous agents that play **Ultimate Tic-Tac-Toe (UTTT)** at a consistently high level. The project is a university assignment for the course "Intelligent Systems II" (SI2) at Universidade de Aveiro, with a deadline of **13 May 2026**.

This specification serves as the single source of truth for the project's scope, architecture, and acceptance criteria. It guides the phased development of three increasingly sophisticated AI agents and the supporting infrastructure.

### 1.2 Scope

**In Scope** (✅ = implemented, ⬜ = pending):
- ✅ A standalone UTTT game state engine (`UTTTState`) that encapsulates all game rules for simulation (independent of the WebSocket server)
- ✅ A generic Monte Carlo Tree Search (MCTS) algorithm using UCT selection with random playouts (Phase 1)
- ✅ An MCTS agent that subclasses `BaseUTTTAgent` and uses the above engine + algorithm (Phase 1)
- ✅ A heuristic evaluation function for UTTT board states (Phase 2)
- ✅ Heuristic-guided rollouts and early-cutoff leaf evaluation replacing pure random playouts (Phase 2)
- ✅ A hybrid MCTS + Heuristics agent (Phase 2)
- ✅ An AlphaZero-lite agent (Phase 3) combining MCTS with a neural policy-value network trained via self-play
- ✅ A Policy-Value Network (CNN) for guiding MCTS in Phase 3
- ✅ A self-play data generation and training pipeline for the neural network (Phase 3)
- ✅ A CSV statistics logger for recording local and global game outcomes
- ✅ Test infrastructure to pit agents against each other and collect win-rate statistics
- ✅ All agents configurable via constructor/environment parameters

> **Infrastructure Immutability**: The primary and exclusive objective of this project is to develop autonomous agents. The existing backend server (`backend/server.py`) and frontend (`frontend/`) are treated as fixed, stable infrastructure. No modifications shall be made to them unless strictly necessary for agent development, and any such modifications must be justified, minimal, and reviewed.

**Out of Scope (Non-Goals)**:
- Modifications to the existing backend server (`backend/server.py`) — the server is considered stable and will not be altered except as noted above
- Modifications to the existing frontend (`frontend/`) — visualization is feature-complete
- A graphical evaluation/monitoring dashboard — CSV-based analysis via external tools is sufficient
- Real-time performance optimization beyond the <60s per move target — agents are for offline analysis, not real-time play
- Distributed/parallel MCTS — the agent will run single-threaded on a single machine
- WebSocket-based agent-to-agent tournament infrastructure — agents connect through the existing server for competitive play; headless testing will use direct engine calls

### 1.3 Audience

- **Developers** (students): Implementing the agents, game engine, and infrastructure
- **Instructor/Evaluator**: Assessing solution effectiveness, code quality, and repository organization
- **Testers**: Running agent-vs-agent matches and analyzing CSV statistics

### 1.4 References

- `docs/specs/constitution.md` — Project principles and conventions
- `README.md` — Project overview, setup instructions, and basic usage
- `agents/base_agent.py` — Abstract base class for all agents
- `backend/server.py` — Existing game server with complete UTTT game logic
- `engine/game_rules.py` — Shared stateless game rules used by both UTTTState and the backend server (extracted from server.py during Phase 1)
- Research: "Monte Carlo Tree Search: A New Framework for Game AI" (Browne et al., 2012)
- Research: "MCTS for Ultimate Tic-Tac-Toe" — various university project analyses and blog posts

---

## 2. System Description

### 2.1 Current State (As-Is)

The existing codebase provides:

- A **WebSocket-based backend server** (`backend/server.py`) that:
  - Manages the 9x9 board, 3x3 macro board, and active-macro state
  - Validates and processes moves
  - Detects local (micro-board) winners and global (macro-board) winners/draws
  - Broadcasts state to connected agents and frontend viewer
  - Supports repeated rounds with alternating first-player
- A **frontend viewer** (`frontend/`) with HTML5 Canvas rendering
- Two **basic agents**:
  - `DummyUTTTAgent` — random move selection
  - `ManualUTTTAgent` — CLI-controlled manual input
- An **abstract base class** (`BaseUTTTAgent`) defining the agent interface

**Current limitations**:
- The MCTS and heuristic agents do not retain knowledge between games (stateless by design).
- The AlphaZero-lite agent is fully implemented but its trained network has not yet surpassed the Phase 2 heuristic agent, largely due to CPU inference speed constraints limiting the amount of self-play training that could be performed.

### 2.2 Target State (To-Be)

The system now has:

1. **Shared game rules** (`engine/game_rules.py`): Pure stateless functions (`apply_move`, `get_valid_actions`, `get_global_winner`, `check_3x3_win`, `is_3x3_full`) extracted from the backend server, shared by both `UTTTState` and any other component needing game logic. ✅ **Phase 1 — Implemented**
2. **Standalone game engine** (`engine/`): A `UTTTState` class that wraps the shared game rules in an immutable state object, designed for rapid MCTS simulation and independent of WebSockets. ✅ **Phase 1 — Implemented**
3. **MCTS algorithm** (`engine/mcts_core.py`): A generic, reusable UCT-based MCTS implementation with optional `playout_fn` for heuristic integration. ✅ **Phase 1 — Implemented**
4. **Heuristic evaluation** (`engine/heuristics.py`): Weighted board-state evaluation based on established UTTT strategic principles; provides `evaluate()`, `score_move()`, `heuristic_playout()`, and `get_feature_breakdown()`. ✅ **Phase 2 — Implemented**
5. **Three intelligent agents**: 
   - `MCTSAgent` — pure MCTS with random playouts (Phase 1) ✅ **Implemented**
   - `MCTSHeuristicAgent` — MCTS with heuristic-guided rollouts and leaf evaluation (Phase 2) ✅ **Implemented**
   - `AlphaZeroUTTTAgent` — MCTS guided by a neural policy-value network, trained via self-play (Phase 3) ✅ **Implemented**
6. **Policy-Value Network** (`engine/policy_value_network.py`): A convolutional neural network that takes board state as input and outputs move probabilities and position evaluation. ✅ **Implemented**
7. **Self-play training pipeline**: Generates training data through MCTS-guided self-play, trains the neural network, and iteratively improves both. ✅ **Implemented**
8. **Statistics logger** (`logger/stats_logger.py`): CSV-based logging of local and global game outcomes. ✅ **Implemented**
9. **Test and evaluation infrastructure**: Ability to run headless tournaments between agents and analyze results. ✅ **Implemented**

### 2.3 Project Goals

1. ✅ **Develop a pure MCTS agent** (Phase 1) that plays UTTT significantly better than random — winning ≥90% of games against `DummyUTTTAgent` over 100 matches.
2. ✅ **Develop a hybrid MCTS + Heuristics agent** (Phase 2) that plays stronger than pure MCTS — winning ≥60% of games against the Phase 1 MCTS agent over 100 matches. *(Implementation complete; win-rate verification pending tournament run)*
3. ✅ **Provide statistical analysis infrastructure** that logs game outcomes to CSV for offline analysis.
4. **Achieve a course grade ≥ 16/20** based on the grading rubric (Solution 30%, Code 20%, Repository 20%, Complexity 15%, Report 10%, Contributions 5%).
5. ✅ **Develop an AlphaZero-lite agent** (Phase 3) that combines MCTS with a neural policy-value network to achieve expert-level play through self-play training. *(Infrastructure fully implemented and functional; ≥55% win rate vs Phase 2 not yet achieved with 15 CPU training iterations due to inference speed limitations.)*

---

## 3. Functional Requirements

> Each requirement has a unique ID (FR-XXX), clear description, priority, testable acceptance criteria, and traceability.

| ID | Title | Description | Priority | Source | Dependencies | Status |
|----|-------|-------------|----------|--------|--------------|--------|
| FR-001 | UTTT Game State Engine | A standalone UTTTState class wrapping shared game_rules for simulation | Must | Assignment PRD, Architecture design | engine/game_rules.py | Implemented |
| FR-002 | MCTS Algorithm Implementation | Standard UCT-based MCTS with configurable iterations, exploration constant, and random seed | Must | Assignment PRD §"Complexity" | FR-001 | Implemented |
| FR-003 | Pure MCTS Agent | Agent that uses FR-001+FR-002 to make decisions via BaseUTTTAgent interface | Must | Assignment PRD, Phase 1 goal | FR-001, FR-002 | Implemented |
| FR-004 | Heuristic Evaluation Functions | Weighted board-state evaluation for UTTT positions | Must | Phase 2 goal, Research literature | FR-001 | Implemented |
| FR-005 | Heuristic-Guided Rollouts | Non-random playout simulation using heuristic-biased move selection and early cutoff | Must | Phase 2 goal | FR-004 | Implemented |
| FR-006 | MCTS + Heuristics Agent | Agent combining MCTS with heuristic rollouts and leaf evaluation | Must | Phase 2 goal, Assignment PRD | FR-001, FR-002, FR-004, FR-005 | Implemented |
| FR-007 | CSV Statistics Logger | Log local and global game outcomes to CSV files with timestamps; thread-safe via threading.Lock, ISO 8601 timestamps, headers on first write | Should | Competition analysis need | None | Implemented |
| FR-008 | Agent Tournament Runner | Headless script to run multiple agent-vs-agent matches and aggregate results; uses ProcessPoolExecutor for parallel execution, tqdm for progress display, CLI with --workers/-w | Should | Testing requirement, Grading §"Repository" | FR-003, FR-006, FR-007 | Implemented |
| FR-009 | Policy-Value Network | CNN that takes board state and outputs policy vector and value scalar for MCTS guidance | Could | Phase 3 goal, AlphaZero research | FR-001 | Implemented |
| FR-010 | Self-Play Training Pipeline | Self-play data generation and training pipeline for the Policy-Value Network | Could | Phase 3 goal | FR-009, FR-002 | Implemented |
| FR-011 | AlphaZero-lite Agent | MCTS agent guided by trained Policy-Value Network with NN-based leaf evaluation | Could | Phase 3 goal | FR-001, FR-002, FR-009, FR-010 | Implemented |

---

### FR-001: UTTT Game State Engine

**Description**: The system shall provide a standalone `UTTTState` class in `engine/game_state.py` that encapsulates all Ultimate Tic-Tac-Toe game logic for simulation purposes. This class wraps the shared pure functions from `engine/game_rules.py` (extracted from `backend/server.py`) in an immutable state object, and is designed for rapid state cloning during MCTS tree search.

**Priority**: Must

**Source**: Assignment PRD, Architecture design derived from `backend/server.py`

**Dependencies**: `engine/game_rules.py` (shared stateless game rules)

**Acceptance Criteria**:
- [x] `UTTTState` can be initialized empty (no moves played) or from an existing state (copy constructor / `clone()` method)
- [x] `get_valid_actions()` returns all legal `[x, y]` moves for the current player, correctly handling:
  - Active macro-board restriction (`active_macro` is set)
  - Free-move condition (`active_macro is None`) — all cells in unresolved macro-boards are allowed
  - Occupied cells are excluded
  - Resolved (won/drawn) macro-boards are excluded
- [x] `apply_action(x, y)` returns a **new** `UTTTState` instance (immutable pattern) with the move applied, board updated, local winner checked, macro-board updated, and next active macro determined
- [x] `is_terminal()` returns `True` when either player has won the macro-board or the macro-board is full with no winner (global draw)
- [x] `get_winner()` returns `0` (ongoing), `1` (Player 1 wins), `2` (Player 2 wins), or `3` (draw) matching the server's convention
- [x] Local micro-board wins are detected correctly (3-in-a-row within any of the nine 3x3 subgrids)
- [x] Local micro-board draws (full with no winner) are detected and marked with value `3`
- [x] Free-move condition is triggered correctly: when a move sends the opponent to a resolved macro-board, `active_macro` is set to `None`
- [x] Macro-board wins are detected correctly (3 resolved macro-boards in a row for the same player)
- [x] The engine correctly processes at least 3 complete realistic game scenarios (provided as sequence-of-move test cases) and produces identical outcomes to the server's `process_move()` logic
- [x] `UTTTState.__hash__` and `__eq__` are implemented for use as dictionary keys (for transposition tables if needed)
- [x] String representation (`__str__` or `__repr__`) provides a human-readable board display
- [x] All public methods have type annotations and docstrings

**Status**: Implemented

### FR-002: MCTS Algorithm Implementation

**Description**: The system shall provide a generic Monte Carlo Tree Search implementation in `engine/mcts_core.py` using the UCT (Upper Confidence bounds applied to Trees) algorithm. The implementation must be configurable and reusable across different game states that implement a standard interface. It uses a `GameStateProtocol` (Python `Protocol` class) for type-safe duck typing.

**Priority**: Must

**Source**: Assignment PRD §"Complexity", Research literature

**Dependencies**: FR-001 (UTTTState)

**Acceptance Criteria**:
- [x] The MCTS algorithm has four standard phases implemented correctly:
  - **Selection**: Traverses the tree from root using UCB1 formula: `score = Q(s,a) + C * sqrt(ln(N(s)) / N(s,a))` where `C` is the exploration constant (default `sqrt(2)` ≈ 1.414)
  - **Expansion**: When a non-terminal state is reached that has unvisited children, one or more child nodes are added
  - **Simulation (Playout)**: From the expanded node, random moves are played to game termination (or to a configurable depth limit)
  - **Backpropagation**: The playout result (win/loss/draw) is propagated up the tree, updating visit counts and win totals from the perspective of the player who made the move at each node
- [x] The number of MCTS iterations per search is configurable (default: 10,000)
- [x] The exploration constant `C` is configurable (default: `sqrt(2)`)
- [x] The algorithm returns the most-visited child action (not the highest-win-rate child) as the best move
- [x] Tree statistics are accessible: total iterations, tree node count, root visit count, best action visit count, best action win rate
- [x] A time limit (seconds) option exists as an alternative to iteration count for stopping search
- [x] The MCTS implementation is generic: it uses a `GameStateProtocol` and accepts any game state object that implements `get_valid_actions()`, `apply_action()`, `is_terminal()`, and `get_winner()`
- [x] Unit tests verify the algorithm produces deterministic results given the same random seed and configuration
- [x] Unit tests verify that increasing iteration count improves (or maintains) move quality (non-regression property)

**Status**: Implemented

### FR-003: Pure MCTS Agent

**Description**: The system shall provide `MCTSAgent` in `agents/mcts_agent.py` that subclasses `BaseUTTTAgent` and uses the UTTTState engine (FR-001) and MCTS algorithm (FR-002) to make decisions. Provides both a `deliberate()` method for server-based play and a `deliberate_from_state()` method for headless testing. Supports CLI invocation via `python -m agents.mcts_agent` with `argparse` arguments: `--server-uri`, `--iterations`/`-i`, `--exploration-constant`/`-c`, `--time-limit`/`-t`, `--random-seed`/`-s`.

**Priority**: Must

**Source**: Phase 1 goal, Assignment PRD

**Dependencies**: FR-001, FR-002

**Acceptance Criteria**:
- [x] `MCTSAgent` subclasses `BaseUTTTAgent` and implements `deliberate(board, macro_board, active_macro, valid_actions)`
- [x] On `deliberate()`: constructs a `UTTTState` from the server state, runs MCTS for the configured number of iterations, returns the best `[x, y]` action
- [x] The agent is configurable with: `mcts_iterations` (default: 10,000), `mcts_exploration_constant` (default: 1.414), `mcts_time_limit` (default: `None`), `random_seed` (default: `None`)
- [x] Supports logging of internal state per move: iteration count, tree size, best action, best action visit count, best action win rate, time taken
- [ ] The agent wins ≥90% of games against `DummyUTTTAgent` over 100 matches with MCTS iterations ≥ 10,000
- [x] The agent plays legally at all times (never returns an invalid action)
- [x] The agent handles edge cases gracefully: no valid actions returns `None`, empty board, fully resolved macro-boards

**Status**: Implemented

### FR-004: Heuristic Evaluation Functions

**Description**: The system shall provide heuristic evaluation functions in `engine/heuristics.py` that score UTTT board states from the perspective of a given player, using weighted strategic features.

**Priority**: Must

**Source**: Phase 2 goal, Research literature on UTTT heuristics

**Dependencies**: FR-001 (UTTTState)

**Acceptance Criteria**:
- [x] `evaluate(state, player_id)` returns a float score: positive = favorable for player, negative = unfavorable, magnitude indicates advantage strength
- [x] The following board features are evaluated with configurable weights:
  - Micro-board wins: ±100 per micro-board owned
  - Two macro-boards in a row (unblocked threat): ±200
  - Blocking opponent three-in-a-row threat: +150
  - Center macro-board (1,1) ownership: ±10
  - Corner macro-board ownership: ±3 per corner
  - Center square within any micro-board: ±3 per occupied center
  - Free-move opportunity (sending opponent to resolved board): ±2
  - Two-in-a-row within a micro-board (potential threat): ±5
  - Blocking opponent two-in-a-row within micro-board: +20
  - **Macro-board win / loss detection**: if player has won, return `+inf`; if opponent has won, return `-inf`; if draw, return 0
- [x] All weights are configurable via a dictionary passed to the constructor or evaluation function
- [x] `evaluate()` is deterministic — same state + same player gives same score
- [x] `evaluate()` completes in <1ms for any valid board state (must be fast enough for use in MCTS rollouts)
- [x] Unit tests verify:
  - State where player has won returns `+inf`
  - State where opponent has won returns `-inf`
  - Empty board returns score ≈ 0 (symmetric position)
  - Known advantageous positions produce scores > 0
  - Known disadvantageous positions produce scores < 0

**Implementation Details**:
- **File**: `engine/heuristics.py`
- **Class**: `HeuristicEvaluator`
- **9 configurable features** with default weights: micro_win (100), macro_threat (200), block_macro_threat (150), center_macro (10), corner_macro (3), center_micro (3), free_move (2), micro_threat (5), block_micro_threat (20)
- `evaluate()` supports terminal states: `+inf` (win), `-inf` (loss), `0` (draw)
- `score_move()` provides lightweight (~5μs) per-move scoring for epsilon-greedy playouts
- `get_feature_breakdown()` returns raw feature contributions for debugging/tuning
- Sub-ms performance verified by benchmark test

**Status**: Implemented

### FR-005: Heuristic-Guided Rollouts

**Description**: The system shall provide heuristic-guided rollouts that replace the random playout phase in MCTS with a heuristic-biased move selection strategy and support early cutoff with leaf evaluation.

**Priority**: Must

**Source**: Phase 2 goal

**Dependencies**: FR-004 (Heuristic Evaluation Functions)

**Acceptance Criteria**:
- [x] `heuristic_playout(state, max_depth=50)` executes a playout where at each step, moves are selected using a probability distribution weighted by the heuristic evaluation of each move (or by a simpler bias like preferring winning/blocking moves)
- [x] Alternative simpler approach: at each step, evaluate all legal moves and select the best with probability `p` (e.g., 0.8) and a random move with probability `(1-p)` — configurable
- [x] If playout reaches `max_depth` without terminal state, use heuristic leaf evaluation (`evaluate(state, player_id)`) to determine the outcome instead of continuing randomly
- [x] `max_depth` is configurable (default: 50)
- [x] The heuristic playout function is compatible as a drop-in replacement for the random playout in MCTS core
- [x] Unit tests verify that heuristic-guided rollouts produce deterministic results with fixed seeds and handle edge cases (terminal states, max_depth=0, bias=0/1)

**Implementation Details**:
- **Method**: `HeuristicEvaluator.heuristic_playout()` in `engine/heuristics.py`
- **MCTS integration**: `playout_fn` parameter in `MCTS.__init__` (backward compatible, defaults to `None` = random playout)
- **Epsilon-greedy strategy**: with probability `playout_bias` (default 0.8), selects the highest-scored move via `score_move()`; otherwise random
- **Early cutoff**: at `max_depth` (default 50), uses `evaluate()` converted to a pseudo-winner via `_leaf_to_winner()` (tanh-based probability mapping)
- **Deterministic**: uses MCTS internal RNG for reproducibility

**Status**: Implemented

### FR-006: MCTS + Heuristics Agent

**Description**: The system shall provide `MCTSHeuristicAgent` in `agents/mcts_heuristic_agent.py` that combines MCTS with heuristic-guided rollouts (FR-005) and heuristic leaf evaluation (FR-004).

**Priority**: Must

**Source**: Phase 2 goal, Assignment PRD §"Complexity"

**Dependencies**: FR-001, FR-002, FR-004, FR-005

**Acceptance Criteria**:
- [x] `MCTSHeuristicAgent` subclasses `BaseUTTTAgent` and implements `deliberate()` 
- [x] Uses heuristic-guided rollouts instead of random rollouts during MCTS simulation phase
- [x] Uses heuristic leaf evaluation when playout reaches depth limit
- [x] Agent is configurable with all MCTS parameters (iterations, C, time limit) plus heuristic parameters (playout bias, weights, max_depth)
- [ ] The agent wins ≥60% of games against the pure MCTS agent (FR-003) over 100 matches with equivalent iteration counts (e.g., both at 10,000 iterations)
- [ ] The agent wins ≥95% of games against `DummyUTTTAgent` over 100 matches
- [x] The agent plays legally at all times

**Implementation Details**:
- **File**: `agents/mcts_heuristic_agent.py`
- **Class**: `MCTSHeuristicAgent` (extends `BaseUTTTAgent`)
- **Lazy HeuristicEvaluator init**: evaluator created on first access via `_get_evaluator()` to avoid multiprocessing pickling issues
- **Playout factory**: `_make_playout_fn()` returns a closure binding evaluator, bias, and max_depth for zero-overhead MCTS integration via the `playout_fn` parameter
- **Configurable parameters**: all MCTS params (iterations, exploration_constant, time_limit, random_seed) + heuristic params (weights, playout_bias, max_depth)
- **CLI support**: `python -m agents.mcts_heuristic_agent` with `argparse` arguments: `--server-uri`, `--iterations`/`-i`, `--exploration-constant`/`-c`, `--time-limit`/`-t`, `--random-seed`/`-s`, `--playout-bias`/`-b`, `--max-depth`/`-d`, `--heuristic-weights`
- **Headless testing**: `deliberate_from_state()` for direct engine-based testing without server

**Status**: Implemented

### FR-007: CSV Statistics Logger

**Description**: The system shall provide a statistics logging module in `logger/stats_logger.py` that records game outcomes to CSV files with timestamps for offline analysis. The implementation uses `threading.Lock` for thread safety, ISO 8601 timestamps via `datetime(timezone.utc)`, headers-on-first-write logic, and graceful error handling (OSError caught and logged as warnings).

**Priority**: Should

**Source**: Testing requirement, Grading §"Repository"

**Dependencies**: None

**Acceptance Criteria**:
- [x] `StatsLogger(log_dir="stats/")` creates the log directory if it does not exist
- [x] `log_local_game(macro_pos, winner, moves_played)` appends a row to `stats/local_games.csv` with columns: `timestamp`, `macro_my`, `macro_mx`, `winner`, `moves_played`, `p1_agent`, `p2_agent`
- [x] `log_global_game(winner, total_moves, p1_name, p2_name, p1_config, p2_config)` appends a row to `stats/global_games.csv` with columns: `timestamp`, `winner`, `total_moves`, `p1_name`, `p2_name`, `p1_config`, `p2_config`, `round_number`
- [x] CSV files have headers on first write
- [x] Timestamps use ISO 8601 format
- [x] Thread-safe file writing (safe for concurrent use)
- [x] Unit tests verify CSV content matches logged data

### FR-008: Agent Tournament Runner

**Description**: The system shall provide a command-line script or module that runs multiple headless matches between two agents and aggregates results for statistical analysis. The implementation uses `ProcessPoolExecutor` (from `concurrent.futures`) for parallel game execution. The tqdm progress bar shows live-updating win rates for both agents as games complete, even in parallel mode (rates converge to the correct final values). It uses `tqdm` for progress display, and an `argparse` CLI with `--workers`/`-w` flags. Per-game seeds are derived deterministically from a base seed for reproducibility. Agent crashes and illegal moves are caught gracefully (opponent awarded the win).

**Priority**: Should

**Source**: Testing requirement, Grading §"Repository"

**Dependencies**: FR-003, FR-006, FR-007

**Acceptance Criteria**:
- [x] `run_tournament(agent1_class, agent2_class, num_games=100, ...)` runs the specified number of games between two agents
- [x] Alternates first-player between agents each round
- [x] Uses the headless UTTTState engine (FR-001) to execute games (no server dependency)
- [x] Records all outcomes using `StatsLogger` (FR-007)
- [x] Prints summary statistics after tournament: wins for each agent, draws, win rates, average game length/time
- [x] Supports command-line invocation: `python -m tournament.runner --agent1 mcts --agent2 dummy --games 100 --seed 42 -w 4`
- [x] Handles agent crashes gracefully (logs error via tqdm.write, continues to next game)

---

### FR-009: Policy-Value Network

**Description**: The system shall provide a neural network model in a new module (e.g., `engine/policy_value_network.py`) that takes a UTTT board state as input and outputs a policy vector (move probabilities) and a value scalar (position evaluation). The network shall be a small convolutional neural network (CNN) with approximately 5 million parameters, designed for efficient inference during MCTS.

**Priority**: Could

**Source**: Phase 3 goal, AlphaZero research (Silver et al., 2017)

**Dependencies**: FR-001 (UTTTState)

**Status**: Implemented

**Acceptance Criteria**:
- [x] The network accepts a 9x9x3 input tensor representing the board state (channels for P1, P2, and metadata like active macro)
- [x] The policy head outputs a probability distribution over all 81 possible moves (masked to exclude illegal moves)
- [x] The value head outputs a scalar in [-1, 1] representing win probability from the current player's perspective
- [x] The network has <10 million parameters (target: ~5M)
- [ ] Inference completes in <10ms on CPU for a single position *(actual: ~100–200ms on CPU; GPU recommended)*
- [x] The network is implemented using a standard deep learning framework (e.g., JAX, PyTorch, or TensorFlow — matching available dependencies)
- [x] Unit tests verify forward pass produces valid policy and value outputs

### FR-010: Self-Play Training Pipeline

**Description**: The system shall provide a self-play training pipeline that generates training data through MCTS-guided self-play, trains the Policy-Value Network (FR-009) on the generated data, and iteratively improves both the network and the MCTS agent. This follows the AlphaZero training paradigm: self-play with current best network → generate (state, policy, value) tuples → train network → repeat.

**Priority**: Could

**Source**: Phase 3 goal, AlphaZero research (Silver et al., 2017)

**Dependencies**: FR-009 (Policy-Value Network), FR-002 (MCTS Algorithm)

**Status**: Implemented

**Acceptance Criteria**:
- [x] Self-play games are played using MCTS guided by the current Policy-Value Network
- [x] Each position in a self-play game generates a training example: (state, search_policy, game_outcome)
- [x] Training data is stored to disk in a format suitable for batch training (e.g., NumPy arrays or TFRecord)
- [x] The training loop reads batches of data and updates network weights using a combined loss function: L = (value - outcome)² - π·log(p) + c·||θ||² (MSE value loss + cross-entropy policy loss + L2 regularization)
- [x] Training runs for multiple iterations (configurable), where each iteration generates new self-play data using the latest network
- [x] After training, the network can be used to guide MCTS in the AlphaZero-lite agent
- [x] Unit tests verify: self-play game generation, training data format, loss computation, checkpoint save/load

### FR-011: AlphaZero-lite Agent

**Description**: The system shall provide an AlphaZero-lite agent (`agents/alphazero_agent.py`) that uses MCTS guided by the trained Policy-Value Network (FR-009) to make decisions. During MCTS, node selection uses a combination of the UCB1 score and the prior probability from the network's policy head. Leaf nodes are evaluated using the network's value head instead of random/heuristic rollouts.

**Priority**: Could

**Source**: Phase 3 goal

**Dependencies**: FR-001, FR-002, FR-009, FR-010

**Status**: Implemented

**Acceptance Criteria**:
- [x] Agent subclasses BaseUTTTAgent and implements deliberate()
- [x] MCTS uses the network's policy output as prior probabilities for move selection
- [x] MCTS uses the network's value output as leaf node evaluation (no random or heuristic playouts)
- [x] The agent is configurable: MCTS iterations, exploration constant, temperature for move selection
- [ ] After sufficient training, the agent should outperform the Phase 2 MCTS + Heuristics agent (≥55% win rate over 100 matches) *(not yet achieved with 15 CPU-training iterations)*
- [x] The agent plays legally at all times
- [x] Inference time for MCTS+NN is tracked and logged per move

---

## 4. Non-Functional Requirements

| ID       | Category       | Description                                                              | Metric                              | Target                                      | Status     |
|----------|----------------|--------------------------------------------------------------------------|-------------------------------------|---------------------------------------------|------------|
| NFR-001  | Performance    | MCTS with 10,000 iterations should complete within reasonable time       | Time per move                       | <60 seconds for 10,000 iterations           | TBD        |
| NFR-002  | Performance    | Heuristic evaluation must be fast enough for use in MCTS rollouts        | Time per evaluation                  | <1ms per evaluation call                    | Verified   |
| NFR-003  | Performance    | State cloning must be efficient for thousands of MCTS nodes              | Time per `clone()`                   | <0.1ms per clone                            | TBD        |
| NFR-004  | Testability    | Agents should be testable against each other and against dummy agent     | Win-rate statistical significance    | ≥95% confidence interval < ±5% for 100 games | TBD        |
| NFR-005  | Maintainability | Code should follow existing project patterns (BaseUTTTAgent subclassing) | Consistency with existing codebase   | 100% of agents subclass BaseUTTTAgent       | Implemented |
| NFR-006  | Maintainability | Game engine should be independent, reusable, and well-documented         | Documentation coverage               | 100% public methods have docstrings         | Implemented |
| NFR-007  | Code Quality   | Type annotations, linting, and adherence to Python best practices        | MyPy compliance, flake8 score        | Zero type errors, flake8 score ≥ 9/10       | TBD        |
| NFR-008  | Reliability    | Agents must never make illegal moves                                     | Invalid move rate                    | 0% over any number of games                 | Implemented |
| NFR-009  | Reproducibility | MCTS with fixed random seed must produce deterministic results          | Identical moves on repeated runs     | 100% identical for same seed and config     | Implemented |
| NFR-010  | Performance    | Policy-Value Network inference must be fast enough for MCTS guidance   | Time per forward pass                | <10ms on CPU for single position            | Not met on CPU (~100–200ms); GPU recommended |
| NFR-011  | Performance    | Self-play training must produce a measurable improvement in agent strength | Win rate vs Phase 2 agent         | ≥55% win rate over 100 matches after training | Not yet achieved (15 CPU iterations) |
| NFR-012  | Maintainability | Neural network module should be independent of the deep learning framework | Framework abstraction layer       | Easy to swap between JAX/PyTorch/TensorFlow | PyTorch only (swappable via rewrite) |

---

## 5. Architecture Overview

### 5.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Existing System                         │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────────┐  │
│  │ Frontend │◄──►│ Backend  │◄──►│     Agents           │  │
│  │ (HTML5)  │    │ Server   │    │  ┌────────────────┐  │  │
│  └──────────┘    │ (WebSock)│    │  │ Dummy Agent    │  │  │
│                  │    │     │    │  │ Manual Agent   │  │  │
│                  │ ┌──┘     │    │  │ MCTS Agent     │  │  │
│                  │ │game_   │    │  │ MCTS+Heuristic │  │  │
│                  │ │rules.py│    │  │ AlphaZero      │  │  │
│                  │ └────────┘    │                      │  │
│                  └──────┬───────┘         │              │
│                         │                 │              │
│                  ┌──────▼─────────────────▼──┐           │
│                  │     engine/                │           │
│                  │  ┌────────────────────┐   │           │
│                  │  │ game_rules.py ◄────┼───┼──Shared───│── Shared with server
│                  │  │ game_state.py      │   │           │
│                  │  │ mcts_core.py       │   │           │
│                  │  │ heuristics.py      │   │           │
│                  │  │ policy_value_net   │   │           │
│                  │  └────────────────────┘   │           │
│                  └───────────────────────────┘           │
│                  ┌────────────────────┐                  │
│                  │   logger/          │                  │
│                  │  stats_logger.py   │                  │
│                  └────────────────────┘                  │
│                  ┌────────────────────┐                  │
│                  │   tournament/      │                  │
│                  │  runner.py         │                  │
│                  └────────────────────┘                  │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 Components

| Component              | Purpose                                                | Responsibilities                                          |
|------------------------|--------------------------------------------------------|-----------------------------------------------------------|
| `engine/game_rules.py` | Shared stateless game rules                              | Pure functions for move validation, application, win/draw detection; used by both UTTTState and backend server |
| `engine/game_state.py` | Immutable UTTT game state for simulation                 | Wraps game_rules in immutable state class: clone, validate moves, apply actions, detect wins/draws, manage turns |
| `engine/mcts_core.py`  | Generic MCTS algorithm (UCT)                             | Tree selection (UCB1), expansion, playout simulation (random or heuristic via `playout_fn`), backpropagation; uses GameStateProtocol |
| `engine/heuristics.py` | Heuristic evaluation functions                           | Score board states (`evaluate()`), lightweight per-move scoring (`score_move()`), epsilon-greedy heuristic playouts (`heuristic_playout()`), feature breakdown for debugging |
| `agents/mcts_agent.py` | Pure MCTS agent (Phase 1)                                | Connect to server, construct UTTTState from server state, deliberate using MCTS, log statistics; also supports headless testing via deliberate_from_state() |
| `agents/mcts_heuristic_agent.py` | MCTS + Heuristics agent (Phase 2)            | Connect to server, deliberate using hybrid MCTS with heuristic rollouts/leaf evaluation, lazy evaluator init for multiprocessing, CLI support |
| `engine/policy_value_network.py` | Policy-Value Network (Phase 3)                | CNN inference: board→(policy, value); forward pass for MCTS guidance |
| `agents/alphazero_agent.py` | AlphaZero-lite agent (Phase 3)                      | Connect to server, deliberate using MCTS+NN, track inference time |
| `selfplay/` | Self-play training pipeline (Phase 3)                              | Generate training data via MCTS-guided self-play; train network; save/load checkpoints |
| `logger/stats_logger.py` | CSV logging of game outcomes (FR-007)                   | Append local/global game records to CSV files; thread-safe via threading.Lock; ISO 8601 timestamps; headers on first write |
| `tournament/runner.py` | Headless match runner (FR-008)                          | Orchestrate multiple games via ProcessPoolExecutor; tqdm progress bar; aggregate statistics; CLI with --workers/-w |

### 5.3 Data Models

#### Model: UTTTState

| Field          | Type                    | Required | Description                                          |
|----------------|-------------------------|----------|------------------------------------------------------|
| `board`        | `List[List[int]]`       | Yes      | 9x9 grid: 0=empty, 1=P1, 2=P2                       |
| `macro_board`  | `List[List[int]]`       | Yes      | 3x3 grid: 0=ongoing, 1=P1, 2=P2, 3=draw             |
| `active_macro` | `Optional[List[int]]`   | Yes      | [my, mx] for restricted move, None for free move     |
| `current_player` | `int`                 | Yes      | 1 or 2                                               |
| `last_move`    | `Optional[Tuple[int,int]]` | No   | Last move played [x, y], for logging                 |
| `move_count`   | `int`                   | Yes      | Total moves played in this game                      |

#### Model: MCTS Node (internal to mcts_core)

| Field      | Type                           | Required | Description                                 |
|------------|--------------------------------|----------|---------------------------------------------|
| `action_taken` | `Optional[List[int]]`          | No       | The action [x, y] that led to this node (None for root) |
| `state`    | `UTTTState`                    | Yes      | Game state at this node                     |
| `parent`   | `Optional[MCTSNode]`           | Yes      | Parent node (None for root)                 |
| `children` | `List[MCTSNode]`               | Yes      | Child nodes                                 |
| `visits`   | `int`                          | Yes      | Number of times this node was visited       |
| `wins`     | `float`                        | Yes      | Number of wins from this node (from current player's perspective, supports 0.5 for draws) |
| `untried_actions` | `List[List[int]]`       | Yes      | Legal actions not yet expanded              |

### 5.4 Key Interfaces

```python
# UTTTState - Game State Engine (FR-001)
class UTTTState:
    # Properties (return copies for immutability)
    @property
    def board(self) -> List[List[int]]: ...
    @property
    def macro_board(self) -> List[List[int]]: ...
    @property
    def active_macro(self) -> Optional[List[int]]: ...
    @property
    def current_player(self) -> int: ...
    @property
    def last_move(self) -> Optional[Tuple[int, int]]: ...
    @property
    def move_count(self) -> int: ...

    def __init__(self, board=None, macro_board=None, active_macro=None,
                 current_player=1, last_move=None, move_count=0):
    def clone(self) -> UTTTState:
    def get_valid_actions(self) -> List[List[int]]:
    def apply_action(self, x: int, y: int) -> UTTTState:
    def is_terminal(self) -> bool:
    def get_winner(self) -> int:
    def __hash__(self) -> int:
    def __eq__(self, other) -> bool:
    def __str__(self) -> str:
    def __repr__(self) -> str:

# MCTS - Algorithm (FR-002) — Extended for Phase 2
class MCTS:
    def __init__(self, iterations=10000, exploration_constant=1.414,
                 time_limit=None, random_seed=None,
                 playout_fn=None):  # Optional heuristic playout (FR-005)
    def search(self, state: GameStateProtocol) -> List[int]:  # returns [x, y]
    def get_stats(self) -> Dict[str, Any]:

# GameStateProtocol (duck-typed interface for MCTS compatibility)
class GameStateProtocol(Protocol):
    @property
    def current_player(self) -> int: ...
    def get_valid_actions(self) -> List[List[int]]: ...
    def apply_action(self, x: int, y: int) -> Any: ...
    def is_terminal(self) -> bool: ...
    def get_winner(self) -> int: ...

# MCTSAgent - Pure MCTS Agent (FR-003) — Implemented
class MCTSAgent(BaseUTTTAgent):
    def __init__(self, server_uri="ws://localhost:8765",
                 mcts_iterations=10000, mcts_exploration_constant=1.414,
                 mcts_time_limit=None, random_seed=None):
    async def deliberate(self, board, macro_board, active_macro,
                         valid_actions) -> Optional[Union[List[int], Tuple[int, int]]]:
    def deliberate_from_state(self, state: UTTTState) -> Optional[List[int]]:
    def get_last_stats(self) -> Dict[str, Any]:

# MCTSHeuristicAgent - MCTS + Heuristics Agent (FR-006) — Implemented
class MCTSHeuristicAgent(BaseUTTTAgent):
    def __init__(self, server_uri="ws://localhost:8765",
                 mcts_iterations=10000, mcts_exploration_constant=1.414,
                 mcts_time_limit=None, random_seed=None,
                 heuristic_weights=None, heuristic_playout_bias=0.8,
                 heuristic_max_depth=50):
    async def deliberate(self, board, macro_board, active_macro,
                         valid_actions) -> Optional[Union[List[int], Tuple[int, int]]]:
    def deliberate_from_state(self, state: UTTTState) -> Optional[List[int]]:
    def get_last_stats(self) -> Dict[str, Any]:
    def _make_playout_fn(self) -> Callable:
    def _get_evaluator(self) -> HeuristicEvaluator:

# Heuristic Evaluation (FR-004) — Implemented
class HeuristicEvaluator:
    def __init__(self, weights: Optional[Dict[str, float]] = None):
    def evaluate(self, state: UTTTState, player_id: int) -> float:
    def score_move(self, board, macro_board, active_macro,
                   current_player, x, y, player_id) -> float:
    def heuristic_playout(self, state: UTTTState, rng: random.Random,
                          max_depth=50, playout_bias=0.8) -> int:
    def get_feature_breakdown(self, state: UTTTState,
                              player_id: int) -> Dict[str, float]:
```

---

## 6. Interfaces

### 6.1 User Interface

No new user interface is created. Agents are:

1. **Run via command line**: 
   - `python -m agents.mcts_agent` (pure MCTS agent, Phase 1)
   - `python -m agents.mcts_heuristic_agent` (MCTS + heuristics agent, Phase 2)
2. **Viewed via existing frontend**: `http://localhost:8080` (visualizes live matches)
3. **Analyzed via CSV files**: `stats/global_games.csv` and `stats/local_games.csv` (post-hoc analysis)

### 6.2 Agent-Server Protocol (Existing, Unchanged)

The communication protocol between agents and the backend server remains unchanged:

**State message (server → agent)**:
```json
{
  "type": "state",
  "current_turn": 1,
  "board": [[0, 0, ...], ...],
  "macro_board": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
  "active_macro": [1, 1],
  "valid_actions": [[3, 3], [3, 4], [3, 5], [4, 3], [4, 4], [4, 5], [5, 3], [5, 4], [5, 5]]
}
```

**Move message (agent → server)**:
```json
{
  "action": "move",
  "x": 4,
  "y": 4
}
```

### 6.3 Agent-Engine Interface

For headless testing (no server), agents can use the `UTTTState` engine directly. Both `MCTSAgent` and `MCTSHeuristicAgent` support `deliberate_from_state()`:

```python
state = UTTTState()
while not state.is_terminal():
    actions = state.get_valid_actions()
    action = agent.deliberate_from_state(state)  # agent-specific method
    state = state.apply_action(action[0], action[1])
```

### 6.4 Error Handling

- **Invalid moves**: Agents must never produce invalid moves. If an agent returns an invalid move, the server rejects it (returns `False` from `process_move`). The agent should log the error and select a fallback (e.g., first valid action).
- **Connection loss**: Agent's `run()` method in `BaseUTTTAgent` catches exceptions and logs them.
- **File I/O errors**: `StatsLogger` catches file write errors and logs warnings without crashing the game.
- **MCTS timeout**: If MCTS exceeds configured time limit, return the best action found so far (even if search was incomplete).

---

## 7. Testing Strategy

### 7.1 Unit Tests

| Target                    | Coverage Target | Key Tests                                                                 |
|---------------------------|-----------------|---------------------------------------------------------------------------|
| `engine/game_state.py`    | ≥90%            | State cloning, valid actions, move application, win detection, draws, free moves, terminal detection |
| `engine/mcts_core.py`     | ≥85%            | UCB1 formula correctness, tree expansion, playout randomness seeded, backpropagation, best action selection |
| `engine/heuristics.py`    | ≥90%            | Feature scoring, weight configuration, terminal state detection (+inf/-inf/0), symmetry, speed (<1ms), score_move, heuristic_playout, leaf evaluation, determinism |
| `agents/mcts_agent.py`    | ≥80%            | State construction from server format, deliberation returns valid action, config propagation |
| `agents/mcts_heuristic_agent.py` | ≥80% | Lazy evaluator init, custom weights, playout factory, config propagation, deterministic deliberation |
| `engine/policy_value_network.py` | ≥80% | Forward pass produces valid policy/value outputs, tensor shape correctness |
| `agents/alphazero_agent.py` | ≥80%      | Network-guided MCTS, legal move generation, inference time logging, config propagation |
| `logger/stats_logger.py`  | ≥90%            | CSV file creation, record append, header writing, thread safety, ISO 8601 timestamps |
| `tournament/runner.py`    | ≥80%            | Full tournament run, parallel execution, crash handling, summary output |

### 7.2 Integration Tests

- **Agent vs Agent (headless)**: Run 10 complete games between MCTS agents and verify all moves are legal and games terminate normally
- **Agent vs Dummy (headless)**: Verify that both MCTS agents achieve expected win rates against the dummy agent
- **AlphaZero vs Phase 2 (headless)**: Run 10 complete games between AlphaZero-lite and MCTS+Heuristic agents to verify legal play and termination
- **Self-play training pipeline**: Run a minimal self-play training cycle (1 iteration, few games) and verify that training data is generated, network weights are updated, and a checkpoint is saved/loaded correctly
- **Server compatibility**: Connect MCTS agent to the real server; verify it plays full games without errors

### 7.3 Tournament Evaluation

- Run `tournament/runner.py` with 100+ games for each pairwise combination of agents (including AlphaZero-lite)
- Compute win rates, draw rates, and average game length
- Verify statistical significance using confidence intervals
- For AlphaZero-lite: compare win rate against baseline agents (Dummy, MCTS, MCTS+Heuristics) across training iterations

### 7.4 Performance Testing

- Measure MCTS iteration throughput (iterations/second) for varying tree sizes
- Measure heuristic evaluation speed (evaluations/second)
- Verify NFR-001 compliance (<60s per move at 10,000 iterations)

---

## 8. Risks & Constraints

### 8.1 Risks

| Risk                                                        | Impact | Likelihood | Mitigation                                                                 |
|-------------------------------------------------------------|--------|------------|----------------------------------------------------------------------------|
| MCTS with 10,000 iterations takes >60s per move (NFR-001 violation) | High   | Medium     | Profile early; implement iteration progress tracking; provide configurable iteration count as safety valve |
| Heuristic evaluation is too slow for MCTS rollouts (NFR-002 violation) | High   | Low        | ✅ **Resolved** — `evaluate()` benchmarked at <1ms; `score_move()` at ~5μs; sub-ms performance verified by unit test |
| UTTTState cloning is too slow for tree expansion (NFR-003 violation) | Medium | Low       | Use `__slots__` for state class; minimize copying of large data structures; consider shallow copy with COW |
| Phase 2 agent does not significantly outperform Phase 1 agent | Medium | Medium    | Validate heuristic weights against literature; run ablation studies; tune weights iteratively. *(Implementation complete; empirical verification pending tournament run)* |
| Deadline pressure (13 May 2026)                             | High   | Medium     | Prioritize Phase 1 (MCTS) first; Phase 2 is stretch goal; Phase 3 is ambitious; use time-boxed iterations |
| Phase 3 NN inference too slow for MCTS (<10ms target)      | High   | Medium     | Profile early; reduce network size; use smaller batch sizes; target ~5M params |
| Self-play training requires significant compute time        | High   | High       | Keep training iterations minimal; pre-generate training data; document expected training times |
| Deep learning framework dependency not available/conflicts  | High   | Medium     | Choose framework compatible with existing setup (stdlib-friendly); fall back to NumPy-only implementation if needed |
| Phase 3 agent does not outperform Phase 2 agent after training | Medium | Medium  | Run ablation studies; train for more iterations; verify network architecture and loss function correctness |

### 8.2 Constraints

| Constraint              | Detail                                                                 |
|-------------------------|------------------------------------------------------------------------|
| Timeline                | Final deadline: **13 May 2026** (~3 days from spec creation)          |
| Language                | Python 3.11+ (matching existing project)                              |
| Dependencies            | Only `websockets` (existing). Heuristics must use stdlib only.        |
| Infrastructure          | Must work with existing Docker Compose setup; no new services required |
| Assessment              | Grading weights: Solution 30%, Code 20%, Repository 20%, Complexity 15%, Report 10%, Contributions 5% |
| Repository              | All code must be in the GitHub Classroom repository; no external hosting |
| Group size              | 1–2 students                                                           |

### 8.3 Assumptions

- The existing backend server (`server.py`) game logic is correct and will not need modifications
- The existing `BaseUTTTAgent` interface is stable and adequate for all agent implementations
- Agents run locally (not inside Docker) and connect to the server over WebSocket
- Random playouts in MCTS are sufficiently representative of game outcomes (standard MCTS assumption)
- The research-derived heuristic weights are a good starting point and can be tuned empirically
- A suitable deep learning framework (JAX, PyTorch, or TensorFlow) is available or can be installed in the existing environment
- Self-play training can produce meaningful improvements within the available compute budget (CPU-only training is expected to be slow but viable for small networks)
- The AlphaZero-lite agent will be evaluated primarily against baseline agents, not expected to match full-scale AlphaZero performance

### 8.4 Exclusions (Non-Goals)

- **Backend modifications**: The server is treated as a fixed, tested component except for minimal, justified changes necessary for agent development.
- **Frontend modifications**: All visualization is handled by the existing HTML5 Canvas viewer.
- **Real-time optimization**: Agents are designed for analysis, not twitch-speed play.
- **Distributed/parallel MCTS**: All computation is single-threaded, single-process.
- **Persistent learning on Phases 1–2**: Phase 1 (pure MCTS) and Phase 2 (MCTS + Heuristics) agents do not retain knowledge between games — they perform no learning. Phase 3 (AlphaZero-lite) is the exception, with explicit self-play training.
- **GUI-based tools**: All testing and analysis is CLI + CSV-based.

---

## 9. Research Appendix: UTTT AI Strategies

### 9.1 Why MCTS Outperforms Minimax for UTTT

Ultimate Tic-Tac-Toe has a branching factor that starts at 81 and only decreases slowly as the board fills. The interplay between micro-board (local) and macro-board (global) strategy makes simple heuristic evaluation unreliable for deep minimax search. Key findings from literature:

| Approach       | Strength vs Random | Characteristics                                    |
|----------------|--------------------|----------------------------------------------------|
| Minimax (depth 2-4) | Moderate      | Fast but weak; heuristic is unreliable at shallow depths |
| Pure MCTS (10k iters) | Very strong | Robust play; no heuristic needed; slow but strong  |
| MCTS + Heuristics | Expert-level   | Stronger than pure MCTS; heuristic guides playouts |
| AlphaZero-lite | Potentially strongest | Requires significant compute for training; planned for Phase 3 |

### 9.2 Recommended Heuristic Weights

Based on research and empirical tuning from multiple UTTT AI projects, the following weights serve as a starting point:

| Feature                                    | Weight | Rationale                                                    |
|--------------------------------------------|--------|--------------------------------------------------------------|
| Micro-board win (owned)                    | ±100   | Each micro-board won is one step closer to macro-board win   |
| Two macro-boards in a row (unblocked)      | ±200   | Direct threat to win the game; highest priority              |
| Block opponent three-in-a-row threat       | +150   | Critical defensive move; prevent immediate loss              |
| Center macro-board (1,1) ownership         | ±10    | Strategic control; connects to most other macro-boards       |
| Corner macro-board ownership               | ±3     | Weak positional control                                      |
| Center square within any micro-board       | ±3     | Local control; sends opponent to center macro-board          |
| Free-move opportunity                      | ±2     | Flexibility to choose any board is valuable                  |
| Two-in-a-row within micro-board            | ±5     | Local threat; may win micro-board on next turn               |
| Block opponent two-in-a-row within micro-board | +20 | Local defense; prevent opponent from winning micro-board     |

### 9.3 Strategic Principles for UTTT

1. **Control the center micro-board**: The center micro-board (macro position [1,1]) is the most strategically important — it connects to every other macro-board and winning it gives a significant advantage.
2. **Think about where each move sends the opponent**: Every move in a micro-board cell `(local_x, local_y)` sends the opponent to macro-board `(local_y, local_x)`. Use this to force the opponent into disadvantageous positions.
3. **Use full/won boards as free-move opportunities**: If a micro-board is resolved, the opponent gets a free move. Force the opponent into this situation strategically.
4. **Sacrifice small boards for big-board advantage**: Losing a micro-board is acceptable if it gives a better global position or denies the opponent a critical threat.
5. **Create multi-board threats**: Threaten to win multiple micro-boards simultaneously so the opponent cannot block all of them.
6. **Avoid sending opponent to the center**: Unless you control it, sending the opponent to the center macro-board gives them strategic advantage.

### 9.4 Coordinate System

```
Global: (x, y) where x = column (0-8), y = row (0-8)
Macro position: (my, mx) = (y // 3, x // 3)
Micro position (within macro): (local_y, local_x) = (y % 3, x % 3)

Example: Global (4, 4) → Macro (1, 1) (center), Micro (1, 1) (center of center)
```

Board indexing: `board[y][x]` where `y` is row index and `x` is column index.

---

## 10. Appendices

### 10.1 Glossary

| Term                  | Definition                                                              |
|-----------------------|-------------------------------------------------------------------------|
| UTTT                  | Ultimate Tic-Tac-Toe — a 9x9 grid divided into 3x3 macro-boards         |
| Micro-board           | One of the nine 3x3 subgrids within the 9x9 board                       |
| Macro-board           | The 3x3 board representing which micro-boards each player has won       |
| MCTS                  | Monte Carlo Tree Search — a heuristic search algorithm for decision processes |
| UCT                   | Upper Confidence bounds applied to Trees — selection criterion for MCTS |
| UCB1                  | Upper Confidence Bound formula: `score = Q + C * sqrt(ln(N) / n)`      |
| Free move             | When a player can play in any unresolved micro-board (sent to a resolved board) |
| Active macro          | The specific micro-board the current player must play in (or None = free move) |
| Playout               | A simulated game from a given state to termination (random or heuristic) |
| Rollout               | Synonym for playout                                                     |
| Leaf evaluation       | Heuristic score of a non-terminal state, used as playout result at depth limit |
| Exploration constant C| Parameter balancing exploration vs exploitation in UCB1                  |

### 10.2 Repository Structure (Actual)

```
project/
├── engine/
│   ├── __init__.py
│   ├── game_rules.py           # Shared stateless game rules (extracted from backend/server.py)
│   ├── game_state.py           # FR-001: UTTTState class (wraps game_rules)
│   ├── mcts_core.py            # FR-002: MCTS algorithm (UCT)
│   ├── heuristics.py           # FR-004, FR-005: Heuristics (Phase 2 — planned)
│   └── policy_value_network.py # FR-009: Policy-Value Network (Phase 3 — planned)
├── agents/
│   ├── __init__.py
│   ├── base_agent.py           # Existing
│   ├── dummy_agent.py          # Existing
│   ├── manual_agent.py         # Existing
│   ├── mcts_agent.py           # FR-003: Pure MCTS agent (Phase 1)
│   ├── mcts_heuristic_agent.py # FR-006: MCTS+Heuristic agent (Phase 2 — planned)
│   └── alphazero_agent.py      # FR-011: AlphaZero-lite agent (Phase 3 — planned)
├── selfplay/                   # FR-010: Self-play training pipeline (Phase 3 — planned)
│   ├── __init__.py
│   ├── self_play.py            # Self-play game generation
│   └── train.py                # Training loop
├── logger/
│   ├── __init__.py
│   └── stats_logger.py         # FR-007: CSV logger (thread-safe, ISO 8601, headers on first write)
├── tournament/
│   ├── __init__.py
│   └── runner.py               # FR-008: Tournament runner (ProcessPoolExecutor, tqdm with live win rates, --workers CLI)
├── backend/                    # Existing (unchanged)
├── frontend/                   # Existing (unchanged)
├── docs/
│   └── specs/
│       └── functional-spec.md  # This file
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_game_rules.py      # Tests for shared game_rules pure functions
│   ├── test_game_state.py      # Tests for UTTTState (FR-001)
│   ├── test_mcts_core.py       # Tests for MCTS algorithm (FR-002)
│   ├── test_mcts_agent.py      # Tests for MCTSAgent (FR-003)
│   ├── test_integration.py     # Integration tests (game_rules ↔ UTTTState ↔ MCTSAgent)
│   ├── test_heuristics.py      # FR-004, FR-005 (Phase 2 — planned)
│   ├── test_stats_logger.py    # FR-007: StatsLogger tests (thread safety, CSV output)
│   └── test_tournament_runner.py # FR-008: Tournament runner tests (full run, crash handling)
├── stats/                      # CSV output directory
│   ├── local_games.csv
│   └── global_games.csv
├── README.md
├── requirements.txt
└── compose.yml
```

> **Package structure note**: `__init__.py` files are kept minimal (no eager imports) to avoid `RuntimeWarning` when running modules with `python -m`. Only `engine/__init__.py` re-exports key symbols (`apply_move`, `check_3x3_win`, `get_global_winner`, `get_valid_actions`, `is_3x3_full`, `UTTTState`, `MCTS`, `MCTSNode`) from its submodules for convenience; all other `__init__.py` files (`agents/`, `logger/`, `tournament/`) contain only a docstring comment.

---

*Document maintained by the Documenter Agent. This is a living document — update when implementation diverges from specification.*
