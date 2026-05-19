"""Shared progress bar utilities for game-generation pipelines.

Provides :class:`WorkerGameBar` for per-worker move/game progress display,
and :func:`setup_mp_lock` for coordinating tqdm across multiple processes.
"""

from tqdm import tqdm

_SPINNER_CHARS = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


class WorkerGameBar:
    """Per-worker progress bar showing move count within the current game.

    Creates a ``tqdm`` bar at ``position=1+worker_index`` that displays a
    cycling spinner, the worker label, the current game number, and a running
    move count with elapsed time and rate.

    Usage::

        bar = WorkerGameBar(worker_index=0, total_games=50)
        for game_idx in range(total_games):
            bar.start_game(game_idx)
            while not done:
                ...
                bar.on_move()
        bar.close()

    Args:
        worker_index: Zero-based worker index used for display positioning.
        total_games: Total number of games this worker will play (for X/Y).
        label: Optional label prefix (default ``"Worker"``).
        position: Optional explicit tqdm position. Defaults to
            ``1 + worker_index`` (below the parent bar at position 0).
    """

    def __init__(
        self,
        worker_index: int,
        total_games: int,
        label: str = "Worker",
        position: int | None = None,
    ) -> None:
        self.worker_index = worker_index
        self.total_games = total_games
        self.label = label
        self._game_idx = 0
        self._status = ""

        self._tqdm = tqdm(
            total=float("inf"),
            desc=f"{label} {worker_index}",
            bar_format="{desc}: {n_fmt} moves [{elapsed}, {rate_fmt}]",
            unit="move",
            position=position if position is not None else (1 + worker_index),
            leave=False,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_game(self, game_idx: int) -> None:
        """Call at the start of each new game.

        Resets the move counter and timer so the bar shows per-game progress,
        then updates the description to reflect the new game number.
        """
        self._game_idx = game_idx
        self._status = ""
        self._tqdm.reset()
        self._refresh()

    def on_move(self) -> None:
        """Call once per move within the current game.

        Advances the spinner animation and increments the move counter.
        """
        self._refresh()
        self._tqdm.update(1)

    def set_status(self, status: str) -> None:
        """Set a status note appended to the bar description.

        Also forces a tqdm refresh so elapsed time shows wall-clock time.
        """
        self._status = status
        self._refresh()
        self._tqdm.refresh()

    def refresh(self) -> None:
        """Force display refresh (updates elapsed time without changing status)."""
        self._tqdm.refresh()

    def close(self) -> None:
        """Clean up the underlying tqdm bar."""
        self._tqdm.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _refresh(self) -> None:
        spinner = _SPINNER_CHARS[int(self._tqdm.n) % len(_SPINNER_CHARS)]
        desc = (
            f"{spinner} {self.label} {self.worker_index} "
            f"Game {self._game_idx + 1}/{self.total_games}"
        )
        if self._status:
            desc += f" — {self._status}"
        self._tqdm.set_description_str(desc)


def setup_mp_lock() -> object:
    """Set up a shared tqdm lock for multiprocessing workers.

    Must be called in the parent process **before** creating any worker
    processes.  Returns the lock, which should be passed to
    :class:`~concurrent.futures.ProcessPoolExecutor` via ``initializer``
    and ``initargs``.

    Example::

        mp_lock = setup_mp_lock()
        executor = ProcessPoolExecutor(
            max_workers=N,
            initializer=tqdm.set_lock,
            initargs=(mp_lock,),
        )
    """
    mp_lock = tqdm.get_lock().mp_lock
    tqdm.set_lock(mp_lock)
    return mp_lock
