"""Schemas that cross boundaries.

Everything here is a pydantic model because it travels: CLI to disk, disk to the API,
API to the GUI, and back into a run lockfile. Nothing in this file knows about torch.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

CorpusKind = Literal["raw", "instruction"]


def _utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class RunPlan(BaseModel):
    """Fields the training loop needs, whichever kind of run it is driving.

    Pretraining and fine-tuning differ entirely in what they build and how they feed
    it, and not at all in how the loop schedules, checkpoints, evaluates, and stops.
    Everything shared lives here so there is one implementation of the tricky parts.
    """

    mode: str

    # batch shape
    micro_batch: int
    grad_accum: int
    tokens_per_step: int
    seq_len: int

    # schedule
    total_steps: int
    lr: float
    min_lr: float
    warmup_steps: int
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95

    # loop behaviour
    eval_every: int
    eval_batches: int = 20
    sample_every: int
    checkpoint_every: int
    seed: int = 1337
    compile: bool = True

    # How the run spreads across devices. "single" is one GPU; "ddp" replicates the
    # model per GPU; "fsdp" shards it when one GPU cannot hold it.
    strategy: str = "single"
    n_gpus: int = 1

    # projections shown before the run starts
    estimated_hours: float = 0.0
    estimated_memory_gb: float = 0.0
    notes: list[str] = Field(default_factory=list)


class ExtensionStat(BaseModel):
    files: int = 0
    documents: int = 0
    chars: int = 0


class CorpusAnalysis(BaseModel):
    """What we learned by reading a folder. The planner's only input about data."""

    root: str
    content_hash: str
    created_at: str = Field(default_factory=_utcnow)

    kind: CorpusKind
    # Share of surviving documents that are conversations rather than prose. Drives
    # `kind`, and is worth surfacing because mixed corpora are common and ambiguous.
    chat_fraction: float = 0.0

    n_files_scanned: int = 0
    n_files_used: int = 0
    n_files_skipped: int = 0
    skip_reasons: dict[str, int] = Field(default_factory=dict)
    by_extension: dict[str, ExtensionStat] = Field(default_factory=dict)

    n_documents: int = 0
    n_chars: int = 0
    n_dropped_quality: int = 0
    n_dropped_duplicate: int = 0
    drop_reasons: dict[str, int] = Field(default_factory=dict)

    # Pre-tokenizer estimate. Replaced with the true count after packing.
    est_tokens: int = 0
    exact_tokens: int | None = None

    warnings: list[str] = Field(default_factory=list)

    @property
    def tokens(self) -> int:
        """Best available token count — measured if we have it, estimated otherwise."""
        return self.exact_tokens if self.exact_tokens is not None else self.est_tokens
