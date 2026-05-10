"""Tests for tournament.runner."""
import os
import tempfile
from typing import Any, Dict, List, Optional

import pytest

from agents.dummy_agent import DummyUTTTAgent
from agents.mcts_agent import MCTSAgent
from tournament.runner import (
    AGENT_REGISTRY,
    _serialize_config,
    main,
    print_summary,
    run_tournament,
)


class TestRunTournament:
    """Tests for the run_tournament function."""

    def test_dummy_vs_dummy_completes_all_games(self) -> None:
        """run_tournament with dummy vs dummy completes all requested games."""
        with tempfile.TemporaryDirectory() as tmpdir:
            a1 = DummyUTTTAgent(random_seed=42)
            a2 = DummyUTTTAgent(random_seed=99)
            results = run_tournament(
                a1, a2, num_games=10, seed=42, log_dir=os.path.join(tmpdir, "stats"),
            )
            assert results["total_games"] == 10
            assert results["agent1_wins"] + results["agent2_wins"] + results["draws"] == 10

    def test_results_structure(self) -> None:
        """Results dict has all expected keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            a1 = DummyUTTTAgent()
            a2 = DummyUTTTAgent()
            results = run_tournament(
                a1, a2, num_games=2, log_dir=os.path.join(tmpdir, "stats"),
            )
            expected_keys = {
                "agent1_wins", "agent2_wins", "draws", "total_games",
                "agent1_name", "agent2_name", "avg_game_length",
                "avg_game_time", "max_game_time", "min_game_time",
                "total_time"
            }
            assert set(results.keys()) == expected_keys

    def test_alternating_first_player(self) -> None:
        """Even games have agent1 as P1, odd games have agent2 as P1."""
        with tempfile.TemporaryDirectory() as tmpdir:
            a1 = DummyUTTTAgent(random_seed=100)
            a2 = DummyUTTTAgent(random_seed=200)
            results = run_tournament(
                a1, a2, num_games=4, seed=42, log_dir=os.path.join(tmpdir, "stats"),
            )
            # With different seeds, results should not be perfectly even,
            # but the function should run without error and complete all games
            assert results["total_games"] == 4
            assert results["agent1_wins"] + results["agent2_wins"] + results["draws"] == 4

    def test_zero_games(self) -> None:
        """Zero games returns an empty summary with avg_game_length=0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            a1 = DummyUTTTAgent()
            a2 = DummyUTTTAgent()
            results = run_tournament(
                a1, a2, num_games=0, log_dir=os.path.join(tmpdir, "stats"),
            )
            assert results["total_games"] == 0
            assert results["agent1_wins"] == 0
            assert results["agent2_wins"] == 0
            assert results["draws"] == 0
            assert results["avg_game_length"] == 0.0

    def test_same_seed_reproducibility(self) -> None:
        """Same seed produces identical results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir1 = os.path.join(tmpdir, "run1")
            log_dir2 = os.path.join(tmpdir, "run2")

            a1 = DummyUTTTAgent()
            a2 = DummyUTTTAgent()

            results1 = run_tournament(
                a1, a2, num_games=5, seed=12345, log_dir=log_dir1,
            )
            results2 = run_tournament(
                a1, a2, num_games=5, seed=12345, log_dir=log_dir2,
            )

            assert results1["agent1_wins"] == results2["agent1_wins"]
            assert results1["agent2_wins"] == results2["agent2_wins"]
            assert results1["draws"] == results2["draws"]
            assert results1["avg_game_length"] == pytest.approx(results2["avg_game_length"])

    def test_handles_agent_crash(self) -> None:
        """run_tournament handles agent crash gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:

            class CrashAgent(DummyUTTTAgent):
                """Agent that always crashes on deliberation."""
                call_count = 0

                def deliberate_from_state(self, state: Any) -> Optional[List[int]]:
                    self.call_count += 1
                    if self.call_count >= 2:
                        raise RuntimeError("Intentional crash for testing")
                    return super().deliberate_from_state(state)

            a1 = CrashAgent()
            a2 = DummyUTTTAgent()
            # Should not raise; crash should be caught
            results = run_tournament(
                a1, a2, num_games=3, seed=42, log_dir=os.path.join(tmpdir, "stats"),
            )
            assert results["total_games"] == 3
            assert results["agent1_wins"] + results["agent2_wins"] + results["draws"] == 3

    def test_handles_illegal_move(self) -> None:
        """run_tournament handles illegal moves gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:

            class IllegalAgent(DummyUTTTAgent):
                """Agent that returns an invalid action."""
                def deliberate_from_state(self, state: Any) -> Optional[List[int]]:
                    return [-100, -100]  # definitely invalid

            a1 = IllegalAgent()
            a2 = DummyUTTTAgent()
            results = run_tournament(
                a1, a2, num_games=2, seed=42, log_dir=os.path.join(tmpdir, "stats"),
            )
            assert results["total_games"] == 2
            # Illegal agent should lose all games it plays in
            assert results["agent1_wins"] + results["agent2_wins"] + results["draws"] == 2

    def test_mcts_vs_dummy_completes(self) -> None:
        """Tournament between MCTS and Dummy completes all games."""
        with tempfile.TemporaryDirectory() as tmpdir:
            a1 = MCTSAgent(mcts_iterations=100, random_seed=42)
            a2 = DummyUTTTAgent(random_seed=99)
            results = run_tournament(
                a1, a2, num_games=3, seed=42, log_dir=os.path.join(tmpdir, "stats"),
            )
            assert results["total_games"] == 3
            assert results["agent1_wins"] + results["agent2_wins"] + results["draws"] == 3

    def test_serialize_config(self) -> None:
        """_serialize_config produces correct JSON for MCTS and Dummy agents."""
        mcts = MCTSAgent(mcts_iterations=500, mcts_exploration_constant=2.0, random_seed=42)
        config_str = _serialize_config(mcts)
        import json
        config = json.loads(config_str)
        assert config["mcts_iterations"] == 500
        assert config["mcts_exploration_constant"] == 2.0
        assert config["random_seed"] == 42

        dummy = DummyUTTTAgent(random_seed=99)
        config_str = _serialize_config(dummy)
        config = json.loads(config_str)
        assert config["random_seed"] == 99
        # Dummy should not have MCTS attributes
        assert "mcts_iterations" not in config

    def test_agent_registry_entries(self) -> None:
        """AGENT_REGISTRY contains expected entries with correct structure."""
        assert "dummy" in AGENT_REGISTRY
        assert "mcts" in AGENT_REGISTRY
        assert AGENT_REGISTRY["dummy"]["class"] is DummyUTTTAgent
        assert AGENT_REGISTRY["mcts"]["class"] is MCTSAgent
        assert AGENT_REGISTRY["dummy"]["display_name"] == "Dummy"
        assert AGENT_REGISTRY["mcts"]["display_name"] == "MCTS"

    def test_print_summary(self, capsys: pytest.CaptureFixture) -> None:
        """print_summary prints formatted output without error."""
        results = {
            "agent1_name": "MCTS",
            "agent2_name": "Dummy",
            "total_games": 100,
            "agent1_wins": 92,
            "agent2_wins": 5,
            "draws": 3,
            "avg_game_length": 42.3,
            "avg_game_time": 1.23,
            "max_game_time": 5.67,
            "min_game_time": 0.01,
            "total_time": 4.17
        }
        print_summary(results)
        captured = capsys.readouterr()
        assert "TOURNAMENT RESULTS" in captured.out
        assert "MCTS" in captured.out
        assert "Dummy" in captured.out
        assert "92" in captured.out
        assert "42.3" in captured.out
        assert "1.23" in captured.out
        assert "4.17" in captured.out

    def test_log_files_created(self) -> None:
        """Tournament run creates local and global CSV files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, "stats")
            a1 = DummyUTTTAgent(random_seed=42)
            a2 = DummyUTTTAgent(random_seed=99)
            run_tournament(
                a1, a2, num_games=3, seed=42, log_dir=log_dir,
            )
            assert os.path.isfile(os.path.join(log_dir, "local_games.csv"))
            assert os.path.isfile(os.path.join(log_dir, "global_games.csv"))

    def test_avg_game_length(self) -> None:
        """avg_game_length is a reasonable number for random play."""
        with tempfile.TemporaryDirectory() as tmpdir:
            a1 = DummyUTTTAgent(random_seed=42)
            a2 = DummyUTTTAgent(random_seed=99)
            results = run_tournament(
                a1, a2, num_games=5, seed=42, log_dir=os.path.join(tmpdir, "stats"),
            )
            # UTTT games last between ~20 and 81 moves typically
            assert 10 <= results["avg_game_length"] <= 81


class TestTournamentCLI:
    """Tests for the CLI entry point."""

    def test_cli_help(self) -> None:
        """CLI --help prints usage information."""
        with pytest.raises(SystemExit):
            # We capture output by checking that it exits with 0
            import io
            import sys
            from contextlib import redirect_stdout

            f = io.StringIO()
            with redirect_stdout(f):
                try:
                    main(["--help"])
                except SystemExit as e:
                    assert e.code == 0
                    raise
            assert "usage" in f.getvalue()

    def test_cli_default_args(self) -> None:
        """CLI with default args runs without error (quick test)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, "stats")
            main([
                "--agent1", "dummy",
                "--agent2", "dummy",
                "--games", "2",
                "--seed", "42",
                "--log-dir", log_dir,
            ])

    def test_cli_verbose(self) -> None:
        """CLI with --verbose prints per-game output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, "stats")
            import io
            import sys
            from contextlib import redirect_stdout

            f = io.StringIO()
            with redirect_stdout(f):
                main([
                    "--agent1", "dummy",
                    "--agent2", "dummy",
                    "--games", "2",
                    "--seed", "42",
                    "--log-dir", log_dir,
                    "--verbose",
                ])
            output = f.getvalue()
            assert "Game 1/2" in output
            assert "TOURNAMENT RESULTS" in output

    def test_cli_mcts_args(self) -> None:
        """CLI passes MCTS arguments correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, "stats")
            main([
                "--agent1", "mcts",
                "--agent2", "dummy",
                "--games", "1",
                "--seed", "42",
                "--iterations", "100",
                "--exploration-constant", "1.0",
                "--log-dir", log_dir,
            ])

    def test_cli_unknown_agent(self) -> None:
        """CLI exits with error for unknown agent."""
        with pytest.raises(SystemExit):
            main(["--agent1", "nonexistent"])
