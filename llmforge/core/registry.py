"""The run registry.

Every training run is recorded here with enough provenance to find it again, compare
it, resume it, or reproduce it. SQLite because it is a single file, needs no server,
and survives a crashed training process without corrupting.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from llmforge.core import paths

RunStatus = Literal["pending", "running", "completed", "failed", "cancelled"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id             TEXT PRIMARY KEY,
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL,
    status         TEXT NOT NULL,
    mode           TEXT NOT NULL,
    name           TEXT,
    corpus_hash    TEXT,
    corpus_root    TEXT,
    tokenizer_id   TEXT,
    base_model     TEXT,
    plan_json      TEXT,
    parent_run_id  TEXT,
    step           INTEGER NOT NULL DEFAULT 0,
    total_steps    INTEGER NOT NULL DEFAULT 0,
    train_loss     REAL,
    val_loss       REAL,
    best_val_loss  REAL,
    tokens_seen    INTEGER NOT NULL DEFAULT 0,
    elapsed_s      REAL NOT NULL DEFAULT 0,
    error          TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_created ON runs (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_runs_corpus  ON runs (corpus_hash);
"""


class RunRecord(BaseModel):
    id: str
    created_at: str
    updated_at: str
    status: RunStatus
    mode: str
    name: str | None = None
    corpus_hash: str | None = None
    corpus_root: str | None = None
    tokenizer_id: str | None = None
    base_model: str | None = None
    plan: dict = Field(default_factory=dict)
    parent_run_id: str | None = None
    step: int = 0
    total_steps: int = 0
    train_loss: float | None = None
    val_loss: float | None = None
    best_val_loss: float | None = None
    tokens_seen: int = 0
    elapsed_s: float = 0.0
    error: str | None = None

    @property
    def dir(self):
        return paths.runs_dir() / self.id

    @property
    def progress(self) -> float:
        return self.step / self.total_steps if self.total_steps else 0.0


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    paths.workspace().mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(paths.registry_path(), timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        # A training process holds this open across long steps; WAL keeps the API
        # able to read progress concurrently without blocking.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def _to_record(row: sqlite3.Row) -> RunRecord:
    data = dict(row)
    data["plan"] = json.loads(data.pop("plan_json") or "{}")
    return RunRecord(**data)


def new_run_id(mode: str, label: str = "") -> str:
    """Sortable, readable, and unique within a workspace.

    `label` is the tier or method where it is already known. A worker-launched run has
    not derived its plan yet, so it is left off rather than filled with a placeholder.
    """
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    base = f"{stamp}-{mode}-{label}" if label else f"{stamp}-{mode}"

    with connect() as conn:
        candidate, n = base, 2
        while conn.execute("SELECT 1 FROM runs WHERE id = ?", (candidate,)).fetchone():
            candidate = f"{base}-{n}"
            n += 1
        return candidate


def create(
    *,
    run_id: str,
    mode: str,
    plan: dict,
    name: str | None = None,
    corpus_hash: str | None = None,
    corpus_root: str | None = None,
    tokenizer_id: str | None = None,
    base_model: str | None = None,
    parent_run_id: str | None = None,
) -> RunRecord:
    now = _now()
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO runs (id, created_at, updated_at, status, mode, name,
                              corpus_hash, corpus_root, tokenizer_id, base_model,
                              plan_json, parent_run_id, total_steps)
            VALUES (?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id, now, now, mode, name, corpus_hash, corpus_root, tokenizer_id,
                base_model, json.dumps(plan), parent_run_id, plan.get("total_steps", 0),
            ),
        )
    return get(run_id)


def update(run_id: str, **fields: Any) -> None:
    """Patch a run's mutable fields. Unknown keys are rejected loudly."""
    if not fields:
        return

    allowed = {
        "status", "step", "total_steps", "train_loss", "val_loss",
        "best_val_loss", "tokens_seen", "elapsed_s", "error", "name",
    }
    unknown = set(fields) - allowed
    if unknown:
        raise ValueError(f"cannot update unknown run fields: {sorted(unknown)}")

    fields["updated_at"] = _now()
    assignments = ", ".join(f"{k} = ?" for k in fields)

    with connect() as conn:
        conn.execute(
            f"UPDATE runs SET {assignments} WHERE id = ?", (*fields.values(), run_id)
        )


def get(run_id: str) -> RunRecord:
    with connect() as conn:
        row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    if row is None:
        raise KeyError(f"no such run: {run_id}")
    return _to_record(row)


def find(run_id: str) -> RunRecord | None:
    try:
        return get(run_id)
    except KeyError:
        return None


def list_runs(limit: int = 50, status: RunStatus | None = None) -> list[RunRecord]:
    query = "SELECT * FROM runs"
    params: list[Any] = []
    if status:
        query += " WHERE status = ?"
        params.append(status)
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)

    with connect() as conn:
        return [_to_record(r) for r in conn.execute(query, params).fetchall()]


def resolve(prefix: str) -> RunRecord:
    """Look up a run by id or unambiguous prefix, plus the alias 'last'."""
    if prefix == "last":
        runs = list_runs(limit=1)
        if not runs:
            raise KeyError("no runs yet")
        return runs[0]

    exact = find(prefix)
    if exact:
        return exact

    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM runs WHERE id LIKE ? ORDER BY created_at DESC", (f"{prefix}%",)
        ).fetchall()

    if not rows:
        raise KeyError(f"no such run: {prefix}")
    if len(rows) > 1:
        ids = ", ".join(r["id"] for r in rows[:5])
        raise KeyError(f"'{prefix}' is ambiguous: {ids}")
    return _to_record(rows[0])
