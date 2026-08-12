"""The description of a run, as the user requested it.

A plan says what will be trained; a spec says what was *asked for*. Keeping it lets a
worker process re-derive the plan from scratch instead of receiving a live object,
which is what allows training to happen in another process — and it means a run
directory on disk fully describes its own origin.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class RunSpec(BaseModel):
    """Everything needed to reconstruct a proposal."""

    folder: str
    base: str | None = None
    teacher: str | None = None

    # from-scratch overrides
    tier: str | None = None
    vocab_size: int | None = None

    # fine-tuning overrides
    method: str | None = None
    epochs: float | None = None

    # shared
    seq_len: int | None = None
    seed: int = 1337
    name: str | None = None

    @property
    def is_finetune(self) -> bool:
        return self.base is not None

    @property
    def is_distill(self) -> bool:
        return self.teacher is not None

    def write(self, run_dir: Path) -> Path:
        path = run_dir / "spec.json"
        path.write_text(self.model_dump_json(indent=2))
        return path

    @classmethod
    def read(cls, run_dir: Path) -> RunSpec:
        return cls.model_validate_json((run_dir / "spec.json").read_text())


class AnalyzeRequest(BaseModel):
    """What the GUI sends to get a plan without committing to it."""

    folder: str
    base: str | None = None
    teacher: str | None = None
    tier: str | None = None
    method: str | None = None
    epochs: float | None = None
    seq_len: int | None = None
    vocab_size: int | None = None
    seed: int = 1337
    force: bool = False

    def to_spec(self, name: str | None = None) -> RunSpec:
        return RunSpec(
            folder=self.folder,
            base=self.base,
            teacher=self.teacher,
            tier=self.tier,
            method=self.method,
            epochs=self.epochs,
            seq_len=self.seq_len,
            vocab_size=self.vocab_size,
            seed=self.seed,
            name=name,
        )


class StartRequest(BaseModel):
    """Start a run. Carries the spec the user reviewed, not a plan — the worker
    re-derives the plan so what runs is always what the spec says."""

    spec: RunSpec
    name: str | None = None


class ChatRequest(BaseModel):
    prompt: str = ""
    max_tokens: int = 200
    temperature: float = 0.8
    checkpoint: str = "best"


class ExportRequest(BaseModel):
    format: str = "gguf"
    quantization: str | None = None
    checkpoint: str = "best"


class EvalRequest(BaseModel):
    examples: int = 64
    prompts: int = 5
    checkpoint: str = "best"


class BrowseEntry(BaseModel):
    name: str
    path: str
    is_dir: bool
    # Populated for directories, so the picker can show what is worth selecting.
    n_files: int | None = None
    # A directory holding config.json — i.e. something loadable as a base model.
    is_model: bool = False


class BrowseResponse(BaseModel):
    path: str
    parent: str | None
    entries: list[BrowseEntry] = Field(default_factory=list)
