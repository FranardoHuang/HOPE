"""Small dependency-free atomic file replacement helper for export artifacts."""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import tempfile
from typing import Iterator


@contextmanager
def atomic_output_path(final_path: str | os.PathLike[str]) -> Iterator[Path]:
    """Yield a same-directory temporary path and atomically install it on success.

    The existing destination is never opened or truncated. Any exception, missing output, or
    empty output removes only the owned temporary file and leaves the destination byte-identical.
    """

    final = Path(final_path).expanduser().absolute()
    parent = final.parent
    if not parent.is_dir():
        raise FileNotFoundError(f"atomic output parent does not exist: {parent}")
    fd, raw_temp = tempfile.mkstemp(prefix=f".{final.name}.", suffix=".tmp", dir=parent)
    os.close(fd)
    temp = Path(raw_temp)
    try:
        yield temp
        if not temp.is_file() or temp.stat().st_size <= 0:
            raise RuntimeError(f"atomic output producer left a missing/empty temporary file: {temp}")
        os.chmod(temp, 0o644)
        with temp.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temp, final)
        # Persist the directory entry where supported. Replacement is already atomic if this
        # best-effort durability fence is unavailable on a platform.
        directory_fd = None
        try:
            directory_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            os.fsync(directory_fd)
        except OSError:
            pass
        finally:
            if directory_fd is not None:
                os.close(directory_fd)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass
