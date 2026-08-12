"""Content hashing — the backbone of "repeatable".

A corpus is identified by what is *in* it, never by its path or mtime. Copy a folder
somewhere else and it hashes identically, so ingesting it again is a cache hit. Change
one byte in one file and everything downstream correctly invalidates.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

# Files below this get hashed whole. Above it we sample head and tail, which keeps
# ingest of a 100 GB corpus from being dominated by hashing.
_FULL_HASH_LIMIT = 8 * 1024 * 1024
_SAMPLE_BYTES = 4 * 1024 * 1024


def hash_file(path: Path) -> str:
    """Content hash of one file. Large files are sampled, not read whole."""
    h = hashlib.sha256()
    size = path.stat().st_size
    h.update(str(size).encode())

    with path.open("rb") as f:
        if size <= _FULL_HASH_LIMIT:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        else:
            h.update(f.read(_SAMPLE_BYTES))
            f.seek(-_SAMPLE_BYTES, 2)
            h.update(f.read(_SAMPLE_BYTES))

    return h.hexdigest()


def hash_files(root: Path, files: Iterable[Path]) -> str:
    """Stable hash over a set of files, keyed by path-relative-to-root plus content.

    Sorted so filesystem walk order can never change the result.
    """
    h = hashlib.sha256()
    for path in sorted(files):
        try:
            rel = path.relative_to(root)
        except ValueError:
            rel = path
        h.update(str(rel).encode())
        h.update(b"\0")
        h.update(hash_file(path).encode())
        h.update(b"\0")
    return h.hexdigest()


def hash_obj(obj: Any) -> str:
    """Hash of any JSON-serialisable object, stable across dict ordering.

    Used to fingerprint configs so two runs with identical settings collide.
    """
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def short(digest: str, n: int = 12) -> str:
    """Human-facing prefix. Long enough to not collide in a workspace."""
    return digest[:n]
