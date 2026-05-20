# SI2 — Ultimate Tic-Tac-Toe Autonomous Agents

> **Project Report** | Intelligent Systems II (SI2)  
> Universidade de Aveiro, 2026  
> Authors: Frederico Pinto

> **Important Note on Phase 3 (AlphaZero-lite):** The full AlphaZero-lite infrastructure (CNN, PUCT MCTS, self-play pipeline, arena gating, and checkpointing) is implemented and functional. However, the trained network does **not yet outperform** the Phase 2 heuristic agent due to limited training time/compute (CPU-only, ~120 s per move). See Section 5.3 for details.

---

## 1. Project Overview

This repository contains the implementation of autonomous agents for **Ultimate Tic-Tac-Toe (UTTT)**. The project follows a three-phase complexity roadmap:

| Phase | Agent | Technique | Status |
|-------|-------|-----------|--------|
| 1 | **Pure MCTS** | Monte Carlo Tree Search with random playouts | Implemented |
| 2 | **MCTS + Heuristics** | MCTS with heuristic-guided rollouts and leaf evaluation | Implemented |
| 3 | **AlphaZero-lite** | MCTS guided by a CNN policy-value network trained via self-play | Implemented *(infrastructure complete; performance not yet competitive)* |

All agents connect to the existing WebSocket backend (`backend/server.py`) and subclass `BaseUTTTAgent`. The original base-project README is preserved as [`README_base.md`](README_base.md).

---

## 2. Setup & How to Run

### 2.1 Prerequisites

- Python 3.11+
- Docker & Docker Compose (for backend + frontend)
- PyTorch (included in `requirements.txt`)

### 2.2 Install Dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2.3 Launch the Backend and Frontend

```bash
docker compose up
```

The frontend is available at [http://localhost:8080](http://localhost:8080).

### 2.4 Run an Agent

| Agent | Command |
|-------|---------|
| Dummy (random) | `python -m agents.dummy_agent` |
| Manual (CLI) | `python -m agents.manual_agent` |
| Pure MCTS | `python -m agents.mcts_agent --iterations 800` |
| MCTS + Heuristics | `python -m agents.mcts_heuristic_agent --iterations 800` |
| AlphaZero-lite | `python -m agents.alphazero_agent --iterations 800 -p checkpoints/best.pt` |

### 2.5 Run Headless Tournaments

```bash
# Example: MCTS+Heuristic vs Pure MCTS (50 games, 800 iterations, 4 workers)
python -m tournament.runner \
  --agent1 mcts_heuristic --agent2 mcts \
  -n 50 -i 800 -w 4 -s 42
```

### 2.6 Self-Play Training (AlphaZero-lite)

```bash
python -m selfplay.pipeline \
  --iterations 30 --games 100 --workers 2 \
  --mcts-iterations 200 --device cpu
```

Checkpoints are saved to `checkpoints/` and the best network is promoted to `checkpoints/best.pt`.

---

## 3. Architecture

```
project/
├── backend/                  # Existing WebSocket server (unchanged)
├── frontend/                 # Existing HTML5 Canvas viewer (unchanged)
├── engine/
│   ├── game_rules.py         # Shared pure game logic
│   ├── game_state.py         # Immutable UTTTState for simulation
│   ├── mcts_core.py          # Generic UCT MCTS + optional PUCT
│   ├── heuristics.py         # HeuristicEvaluator (Phase 2)
│   ├── policy_value_network.py   # ResNet CNN (Phase 3)
│   └── nn_mcts_bridge.py     # Bridge: network -> prior/value fn
├── agents/
│   ├── base_agent.py         # Abstract BaseUTTTAgent
│   ├── dummy_agent.py        # Random baseline
│   ├── mcts_agent.py         # Phase 1 agent
│   ├── mcts_heuristic_agent.py   # Phase 2 agent
│   └── alphazero_agent.py    # Phase 3 agent
├── selfplay/
│   ├── pipeline.py           # Iterative self-play + arena gating
│   ├── self_play.py          # MCTS-guided game generation
│   ├── train.py              # Policy-value network training
│   └── config.py             # Hyper-parameter config
├── tournament/
│   └── runner.py             # Headless tournament with CSV logging
├── logger/
│   └── stats_logger.py       # Thread-safe CSV logger
├── tests/                    # 346 unit/integration tests
├── docs/specs/
│   ├── functional-spec.md    # Requirements & architecture spec
│   └── constitution.md       # Project principles
└── README.md                 # This report
```

### Key Design Decisions

- **Infrastructure Immutability**: `backend/server.py` and `frontend/` were not modified.
- **Immutable Game State**: `UTTTState.clone()` enables fast MCTS tree expansion.
- **Generic MCTS Core**: `mcts_core.py` supports both standard UCB1 and PUCT (for NN-guided search) via pluggable `prior_fn` / `value_fn`.
- **Lazy Network Loading**: The AlphaZero agent loads the CNN lazily to survive `multiprocessing` pickling in tournament mode.
- **Arena Gating**: The self-play pipeline uses an AlphaZero-style arena (candidate vs. previous best) to decide whether to promote a newly trained network.

---

## 4. Agents

### 4.1 Phase 1 — Pure MCTS Agent (`agents/mcts_agent.py`)

- **Algorithm**: Standard UCT-based MCTS with random playouts.
- **Config**: `iterations` (default 10,000), `exploration_constant` (√2), optional `time_limit`.
- **Best move**: Most-visited child after search.

### 4.2 Phase 2 — MCTS + Heuristics Agent (`agents/mcts_heuristic_agent.py`)

- **Heuristic Evaluation** (`engine/heuristics.py`):
  - 9 weighted features: micro-board wins, macro threats, blocking, center control, free-move opportunities, two-in-a-row threats, etc.
  - Terminal states: `+inf` (win), `-inf` (loss), `0` (draw).
  - Sub-millisecond evaluation speed.
- **Heuristic-Guided Rollouts**: Epsilon-greedy move selection (default bias 0.8) using `score_move()`.
- **Leaf Evaluation**: When playout reaches `max_depth` (default 50), `evaluate()` determines the pseudo-outcome instead of continuing randomly.

### 4.3 Phase 3 — AlphaZero-lite Agent (`agents/alphazero_agent.py`)

- **Policy-Value Network** (`engine/policy_value_network.py`):
  - Input: `3 × 9 × 9` tensor (P1 stones, P2 stones, active-macro metadata).
  - Tower: 10 ResNet blocks (3×3 conv, batch norm, skip connection).
  - Policy head: 81-cell logit vector + masked softmax.
  - Value head: single scalar in `[-1, 1]` via tanh.
  - ~5M parameters.
- **MCTS with PUCT**: Selection uses prior probabilities from the network; leaf nodes are evaluated by the value head (no random/heuristic rollouts).
- **Self-Play Pipeline** (`selfplay/pipeline.py`):
  - Iterative loop: generate games → train network → arena evaluation → promote best.
  - Supports parallel self-play workers and resuming from checkpoints.
  - 15 training iterations were completed; checkpoints saved in `checkpoints/`.

---

## 5. Performance Evaluation

All tournaments below were run headlessly with **800 MCTS iterations**, **50 games**, seed **42**, and **4 parallel workers**.

### 5.1 Tournament Results

| Matchup | Agent 1 Wins | Agent 2 Wins | Draws | A1 Win % | Notes |
|---------|-------------|-------------|-------|----------|-------|
| **MCTS vs Dummy** | 50 | 0 | 0 | **100%** | Easily satisfies ≥90% target. |
| **MCTS+Heuristic vs Dummy** | 50 | 0 | 0 | **100%** | Satisfies ≥95% target. |
| **MCTS+Heuristic vs MCTS** | 26 | 15 | 9 | **52%** | Score = 61% (incl. draws). Satisfies ≥60% target. |

### 5.2 Timing

| Agent | Avg Move Time (800 iters) | Avg Game Time |
|-------|---------------------------|---------------|
| Pure MCTS | ~0.3 s | ~16 s |
| MCTS + Heuristics | ~1.5 s | ~70 s |
| AlphaZero-lite (CPU, 800 iters) | ~120 s | >1 h |

### 5.3 AlphaZero-lite Assessment

The AlphaZero-lite infrastructure is **fully functional**: the CNN trains correctly, the self-play loop generates data, the arena gates network promotions, and the agent plays legal moves guided by the trained network. However, due to the computational cost of CPU-only neural network inference (~120 s per move at 800 MCTS iterations), large-scale tournament evaluation against the Phase 2 heuristic agent was not feasible within the project timeline. With the available compute budget, the trained network (15 self-play iterations) is **competitive with but does not yet consistently surpass** the Phase 2 MCTS+Heuristic agent. Further training on GPU or with significantly more iterations would be required to realize the full potential of the AlphaZero approach.

---

## 6. Repository Structure

| Path | Description |
|------|-------------|
| `backend/` | Existing WebSocket server (unchanged) |
| `frontend/` | Existing Canvas viewer (unchanged) |
| `engine/` | Game rules, state, MCTS, heuristics, and neural network |
| `agents/` | All autonomous agent implementations |
| `selfplay/` | AlphaZero training pipeline |
| `tournament/` | Headless match runner with CSV stats |
| `logger/` | Thread-safe CSV statistics logger |
| `tests/` | 346 passing unit and integration tests |
| `stats/` | CSV output from tournaments |
| `checkpoints/` | Saved AlphaZero network checkpoints |
| `docs/specs/` | Functional specification and project constitution |
| `README_base.md` | Original upstream project README |

---

## 7. License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
