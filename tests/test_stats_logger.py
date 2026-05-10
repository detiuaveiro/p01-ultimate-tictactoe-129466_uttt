"""Tests for logger.stats_logger.StatsLogger."""
import csv
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import List

import pytest

from logger.stats_logger import StatsLogger


class TestStatsLogger:
    """Tests for the StatsLogger class."""

    def test_init_creates_directory(self) -> None:
        """StatsLogger creates the log directory on init."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, "mystats")
            logger = StatsLogger(log_dir=log_dir)
            assert os.path.isdir(log_dir)

    def test_log_local_game_creates_file_with_headers(self) -> None:
        """log_local_game creates local_games.csv with correct headers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = StatsLogger(log_dir=tmpdir)
            logger.log_local_game(
                macro_pos=(0, 1), winner=1, moves_played=9,
                p1_agent="dummy", p2_agent="mcts",
            )
            filepath = os.path.join(tmpdir, "local_games.csv")
            assert os.path.isfile(filepath)
            with open(filepath, "r", newline="") as f:
                reader = csv.reader(f)
                rows = list(reader)
            # First row should be headers
            assert rows[0] == StatsLogger.LOCAL_COLUMNS

    def test_log_global_game_creates_file_with_headers(self) -> None:
        """log_global_game creates global_games.csv with correct headers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = StatsLogger(log_dir=tmpdir)
            logger.log_global_game(
                winner=1, total_moves=42,
                p1_name="MCTS", p2_name="Dummy",
                p1_config="{}", p2_config="{}",
                round_number=1,
            )
            filepath = os.path.join(tmpdir, "global_games.csv")
            assert os.path.isfile(filepath)
            with open(filepath, "r", newline="") as f:
                reader = csv.reader(f)
                rows = list(reader)
            assert rows[0] == StatsLogger.GLOBAL_COLUMNS

    def test_log_local_game_appends_rows(self) -> None:
        """Multiple local games are appended as new rows."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = StatsLogger(log_dir=tmpdir)
            logger.log_local_game((0, 0), 1, 9, "a", "b")
            logger.log_local_game((1, 1), 2, 15, "c", "d")
            logger.log_local_game((2, 2), 0, 5, "e", "f")

            filepath = os.path.join(tmpdir, "local_games.csv")
            with open(filepath, "r", newline="") as f:
                reader = csv.reader(f)
                rows = list(reader)

            # Headers + 3 data rows
            assert len(rows) == 4
            assert rows[0] == StatsLogger.LOCAL_COLUMNS
            # Check data rows (skip header)
            assert int(rows[1][1]) == 0  # macro_my
            assert int(rows[1][2]) == 0  # macro_mx
            assert int(rows[1][3]) == 1  # winner
            assert int(rows[2][3]) == 2  # winner
            assert int(rows[3][3]) == 0  # winner

    def test_log_global_game_appends_rows(self) -> None:
        """Multiple global games are appended as new rows."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = StatsLogger(log_dir=tmpdir)
            logger.log_global_game(winner=1, total_moves=30, round_number=1)
            logger.log_global_game(winner=2, total_moves=45, round_number=2)
            logger.log_global_game(winner=3, total_moves=81, round_number=3)

            filepath = os.path.join(tmpdir, "global_games.csv")
            with open(filepath, "r", newline="") as f:
                reader = csv.reader(f)
                rows = list(reader)

            assert len(rows) == 4
            assert rows[0] == StatsLogger.GLOBAL_COLUMNS
            assert int(rows[1][1]) == 1   # winner
            assert int(rows[1][2]) == 30  # total_moves
            assert int(rows[2][1]) == 2
            assert int(rows[3][1]) == 3

    def test_timestamps_are_iso8601(self) -> None:
        """Timestamps logged are valid ISO 8601 format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = StatsLogger(log_dir=tmpdir)
            logger.log_local_game((0, 0), 1, 9)
            logger.log_global_game(winner=1, total_moves=30)

            for fname in ["local_games.csv", "global_games.csv"]:
                filepath = os.path.join(tmpdir, fname)
                with open(filepath, "r", newline="") as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                timestamp = rows[1][0]
                # ISO 8601 should contain 'T' and end with timezone info
                assert "T" in timestamp
                assert "+" in timestamp or timestamp.endswith("Z")

    def test_thread_safety(self) -> None:
        """Multiple threads can log simultaneously without corruption."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = StatsLogger(log_dir=tmpdir)
            n_threads = 10

            def log_local(i: int) -> None:
                logger.log_local_game(
                    macro_pos=(i % 3, i // 3), winner=(i % 3), moves_played=i,
                )

            with ThreadPoolExecutor(max_workers=4) as executor:
                list(executor.map(log_local, range(n_threads)))

            filepath = os.path.join(tmpdir, "local_games.csv")
            with open(filepath, "r", newline="") as f:
                reader = csv.reader(f)
                rows = list(reader)

            # Headers + n_threads data rows
            assert len(rows) == n_threads + 1
            assert rows[0] == StatsLogger.LOCAL_COLUMNS

    def test_error_handling_unwritable_dir(self) -> None:
        """StatsLogger handles unwritable directories gracefully (logs warning, no crash)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a subdirectory and make it read-only
            protected_dir = os.path.join(tmpdir, "protected")
            os.makedirs(protected_dir, exist_ok=True)
            os.chmod(protected_dir, 0o444)  # Read-only

            logger = StatsLogger(log_dir=protected_dir)
            # Should not raise, just log a warning
            logger.log_local_game((0, 0), 1, 9)
            logger.log_global_game(winner=1, total_moves=30)
            # No assertion needed - the test is that no exception propagates

            # Restore permissions for cleanup
            os.chmod(protected_dir, 0o755)

    def test_header_not_rewritten_on_append(self) -> None:
        """Headers are only written once, not on every append."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = StatsLogger(log_dir=tmpdir)
            logger.log_local_game((0, 0), 1, 9)
            logger.log_local_game((1, 1), 2, 15)

            filepath = os.path.join(tmpdir, "local_games.csv")
            with open(filepath, "r", newline="") as f:
                reader = csv.reader(f)
                rows = list(reader)

            assert len(rows) == 3  # Header + 2 data rows
            # Verify no duplicate headers
            assert rows[0] == StatsLogger.LOCAL_COLUMNS
            assert rows[1] != StatsLogger.LOCAL_COLUMNS
            assert rows[2] != StatsLogger.LOCAL_COLUMNS

    def test_local_and_global_separate_files(self) -> None:
        """Local and global logs go to separate files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = StatsLogger(log_dir=tmpdir)
            logger.log_local_game((0, 0), 1, 9)
            logger.log_global_game(winner=2, total_moves=42)

            local_path = os.path.join(tmpdir, "local_games.csv")
            global_path = os.path.join(tmpdir, "global_games.csv")
            assert os.path.isfile(local_path)
            assert os.path.isfile(global_path)

            # Local file has LOCAL_COLUMNS
            with open(local_path, "r", newline="") as f:
                assert f.readline().strip().split(",") == StatsLogger.LOCAL_COLUMNS
            # Global file has GLOBAL_COLUMNS
            with open(global_path, "r", newline="") as f:
                assert f.readline().strip().split(",") == StatsLogger.GLOBAL_COLUMNS
