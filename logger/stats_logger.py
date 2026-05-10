"""Thread-safe CSV logger for game statistics."""
import csv
import logging
import os
import threading
from datetime import datetime, timezone
from typing import List, Optional, Tuple


class StatsLogger:
    """Thread-safe CSV logger for UTTT game statistics.

    Logs both local (micro-board) and global game outcomes to CSV files.
    Thread-safe via threading.Lock.
    """

    LOCAL_COLUMNS = [
        "timestamp", "macro_my", "macro_mx", "winner",
        "moves_played", "p1_agent", "p2_agent",
    ]
    GLOBAL_COLUMNS = [
        "timestamp", "winner", "total_moves",
        "p1_name", "p2_name", "p1_config", "p2_config",
        "round_number",
    ]
    LOCAL_FILENAME = "local_games.csv"
    GLOBAL_FILENAME = "global_games.csv"

    def __init__(self, log_dir: str = "stats/") -> None:
        self._log_dir = log_dir
        os.makedirs(self._log_dir, exist_ok=True)
        self._lock = threading.Lock()
        self._local_headers_written: bool = False
        self._global_headers_written: bool = False

    def log_local_game(
        self,
        macro_pos: Tuple[int, int],
        winner: int,
        moves_played: int,
        p1_agent: str = "",
        p2_agent: str = "",
    ) -> None:
        filepath = os.path.join(self._log_dir, self.LOCAL_FILENAME)
        macro_my, macro_mx = macro_pos
        row = [
            self._timestamp(), macro_my, macro_mx, winner,
            moves_played, p1_agent, p2_agent,
        ]
        try:
            with self._lock:
                self._ensure_headers(filepath, self.LOCAL_COLUMNS, "_local_headers_written")
                with open(filepath, "a", newline="") as f:
                    csv.writer(f).writerow(row)
        except OSError as e:
            logging.warning(f"StatsLogger: failed to write local game: {e}")

    def log_global_game(
        self,
        winner: int,
        total_moves: int,
        p1_name: str = "",
        p2_name: str = "",
        p1_config: str = "",
        p2_config: str = "",
        round_number: int = 0,
    ) -> None:
        filepath = os.path.join(self._log_dir, self.GLOBAL_FILENAME)
        row = [
            self._timestamp(), winner, total_moves,
            p1_name, p2_name, p1_config, p2_config,
            round_number,
        ]
        try:
            with self._lock:
                self._ensure_headers(filepath, self.GLOBAL_COLUMNS, "_global_headers_written")
                with open(filepath, "a", newline="") as f:
                    csv.writer(f).writerow(row)
        except OSError as e:
            logging.warning(f"StatsLogger: failed to write global game: {e}")

    def _ensure_headers(self, filepath: str, columns: List[str], flag_attr: str) -> None:
        if getattr(self, flag_attr):
            return
        write_header = not (os.path.exists(filepath) and os.path.getsize(filepath) > 0)
        if write_header:
            with open(filepath, "a", newline="") as f:
                csv.writer(f).writerow(columns)
        setattr(self, flag_attr, True)

    def _timestamp(self) -> str:
        return datetime.now(timezone.utc).isoformat()
