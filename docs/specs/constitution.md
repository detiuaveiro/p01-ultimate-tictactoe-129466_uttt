# Project Constitution

> **Version**: 1.0 | **Date**: 2026-05-10 | **Status**: Ratified

## Preamble

This document establishes the foundational principles and inviolable rules that govern the development of the SI2 Ultimate Tic-Tac-Toe project. All decisions, implementations, and evaluations shall be guided by these principles.

## Core Principles

### 1. Agent-First Development
The **sole objective** of this project is to develop autonomous agents that play Ultimate Tic-Tac-Toe at a consistently high level. Every line of code written shall serve this purpose. Features that do not directly contribute to agent performance, analysis, or development workflow are out of scope.

### 2. Infrastructure Immutability
The existing backend server (`backend/server.py`) and frontend viewer (`frontend/`) are **fixed, stable components**. They shall not be modified unless:

- A bug in the infrastructure prevents agents from functioning correctly, AND
- The bug cannot be worked around from the agent side.

Any infrastructure modification must be: (a) minimal, (b) justified in writing, and (c) reviewed before implementation.

### 3. Interface Stability
The agent-server communication protocol and the `BaseUTTTAgent` interface shall remain unchanged. All agents must work with the existing server without protocol modifications.

### 4. Incremental Complexity
Agents shall be developed in phases of increasing complexity:

| Phase | Agent | Technique |
|-------|-------|-----------|
| 1 | Pure MCTS | Monte Carlo Tree Search with random playouts |
| 2 | MCTS + Heuristics | MCTS with heuristic-guided rollouts and leaf evaluation |
| 3 | AlphaZero-lite | MCTS guided by a neural policy-value network, trained via self-play |

Each phase must be fully functional before progressing to the next. Earlier phases serve as baselines for evaluating later phases.

### 5. Empirical Validation
All claims about agent performance must be supported by empirical evidence. Win rates shall be established over a statistically significant number of games (minimum 100 per matchup). Random seeds shall be controlled for reproducibility.

### 6. Code Quality
All code shall be:
- **Type-annotated** (Python type hints)
- **Documented** (docstrings for all public methods)
- **Tested** (unit tests with ≥80% coverage)
- **Readable** (follow PEP 8, meaningful names, appropriate decomposition)

### 7. Dependency Minimalism
External dependencies beyond those already in `requirements.txt` (i.e., `websockets`) shall be avoided unless absolutely necessary. Phase 3 (AlphaZero-lite) may introduce a deep learning framework, which must be justified and documented.

### 8. Reproducibility
All experiments (tournaments, self-play training runs) must be reproducible. Random seeds must be configurable. Configuration parameters must be logged alongside results.

### 9. Single Source of Truth
The `docs/specs/functional-spec.md` is the authoritative specification for the project's requirements and architecture. The `README.md` is the authoritative guide for setup and usage. These documents shall be kept in sync with the codebase.

### 10. Open Science
All code, data, and results shall be committed to the repository. The repository shall be organized, documented, and self-contained. External evaluators must be able to reproduce results with only the instructions in the README and the code in the repository.

## Governance

- Violations of this constitution should be raised as issues in the repository.
- Amendments require consensus among team members.
- The constitution itself should be reviewed and updated as the project evolves.

## Signatories

- [Frederico Pinto] — 2026-05-10
