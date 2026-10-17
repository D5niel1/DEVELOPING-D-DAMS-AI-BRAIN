"""Streaming snapshots that bind content hashes to the exact bytes consumed."""
from __future__ import annotations

import hashlib
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


@contextmanager
def snapshot_file(path: str | Path) -> Iterator[tuple[Path, str]]:
    """Copy a source to a temporary file while hashing, then yield that immutable snapshot."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"file does not exist: {source}")
    with tempfile.TemporaryDirectory(prefix="ddams-snapshot-") as directory:
        snapshot = Path(directory) / (source.name or "snapshot.bin")
        digest = hashlib.sha256()
        try:
            with source.open("rb") as incoming, snapshot.open("wb") as outgoing:
                for chunk in iter(lambda: incoming.read(1024 * 1024), b""):
                    digest.update(chunk)
                    outgoing.write(chunk)
                outgoing.flush()
                os.fsync(outgoing.fileno())
        except OSError:
            raise
        yield snapshot, digest.hexdigest()
