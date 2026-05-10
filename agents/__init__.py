"""
Agents package for Ultimate Tic-Tac-Toe.
"""

__all__ = [
    "BaseUTTTAgent",
    "DummyUTTTAgent",
    "ManualUTTTAgent",
    "MCTSAgent",
]


def __getattr__(name):
    if name == "BaseUTTTAgent":
        from agents.base_agent import BaseUTTTAgent

        return BaseUTTTAgent
    elif name == "DummyUTTTAgent":
        from agents.dummy_agent import DummyUTTTAgent

        return DummyUTTTAgent
    elif name == "ManualUTTTAgent":
        from agents.manual_agent import ManualUTTTAgent

        return ManualUTTTAgent
    elif name == "MCTSAgent":
        from agents.mcts_agent import MCTSAgent

        return MCTSAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
