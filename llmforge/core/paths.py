"""Where LLMForge keeps its stuff.

One workspace directory holds everything derived: ingested corpora, trained
tokenizers, run artifacts, and the run registry. It is deliberately separate from
the user's source corpus folders, which we only ever read.

Override with LLMFORGE_HOME.
"""

from __future__ import annotations

import os
from pathlib import Path


def workspace() -> Path:
    """Root of all derived state. Created on demand."""
    env = os.environ.get("LLMFORGE_HOME")
    root = Path(env).expanduser() if env else Path.cwd() / "workspace"
    return root.resolve()


def _sub(name: str) -> Path:
    p = workspace() / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def corpora_dir() -> Path:
    """Ingested + packed corpora, keyed by content hash."""
    return _sub("corpora")


def tokenizers_dir() -> Path:
    """Trained tokenizers, keyed by content hash."""
    return _sub("tokenizers")


def runs_dir() -> Path:
    """One subdirectory per training run."""
    return _sub("runs")


def models_dir() -> Path:
    """Exported models, ready to load or ship."""
    return _sub("models")


def cache_dir() -> Path:
    """Scratch space for extraction and downloads."""
    return _sub("cache")


def registry_path() -> Path:
    """SQLite database tracking runs and their lineage."""
    return workspace() / "registry.db"


def run_dir(run_id: str) -> Path:
    """Artifact directory for a single run."""
    p = runs_dir() / run_id
    p.mkdir(parents=True, exist_ok=True)
    return p
