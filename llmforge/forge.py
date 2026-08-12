"""The top-level "make me a model" entry point.

Both the CLI and the API call in here, so a run started by clicking and a run started
by typing are the same run, described by the same plan and recorded the same way.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from llmforge.core import hardware, lock, paths, planner, registry
from llmforge.core.config import CorpusAnalysis
from llmforge.core.planner import TrainPlan
from llmforge.data import prepare as prep

if TYPE_CHECKING:
    from llmforge.data.ingest import Corpus
    from llmforge.finetune.base import BaseModelInfo
    from llmforge.finetune.plan import FinetunePlan
    from llmforge.finetune.sft import Dataset

ProgressFn = Callable[[str, int, int], None]
EventFn = Callable[[dict], None]


@dataclass
class Proposal:
    """What we intend to do, before committing any compute to it."""

    analysis: CorpusAnalysis
    plan: TrainPlan
    prepared: prep.Prepared
    hardware: hardware.Hardware


@dataclass
class FinetuneProposal:
    """The fine-tuning counterpart: a base model plus a tokenized dataset."""

    analysis: CorpusAnalysis
    plan: FinetunePlan
    corpus: Corpus
    dataset: Dataset
    tokenizer: object
    info: BaseModelInfo
    hardware: hardware.Hardware


def propose(
    folder: Path,
    *,
    tier: str | None = None,
    seq_len: int | None = None,
    vocab_size: int | None = None,
    seed: int = 1337,
    force: bool = False,
    progress: ProgressFn | None = None,
) -> Proposal:
    """Prepare the corpus and derive a training plan, without starting training.

    This is what the GUI shows on the review screen, and what `--dry-run` prints.
    """
    prepared = prep.prepare(folder, vocab_size=vocab_size, force=force, progress=progress)
    hw = hardware.profile()

    plan = planner.plan_pretrain(
        prepared.analysis,
        vocab_size=prepared.packed.vocab_size,
        hw=hw,
        tier_name=tier,
        seq_len=seq_len,
        seed=seed,
    )
    return Proposal(
        analysis=prepared.analysis, plan=plan, prepared=prepared, hardware=hw
    )


def start(
    proposal: Proposal,
    *,
    name: str | None = None,
    emit: EventFn | None = None,
) -> dict:
    """Register a run for `proposal` and train it to completion.

    Serves both from-scratch training and distillation: they produce the same kind of
    artifact from the same packed corpus, differing only in where the training signal
    comes from.
    """
    plan = proposal.plan
    analysis = proposal.analysis

    run_id = registry.new_run_id(plan.mode, plan.tier)
    registry.create(
        run_id=run_id,
        mode=plan.mode,
        plan=plan.model_dump(),
        name=name,
        corpus_hash=analysis.content_hash,
        corpus_root=analysis.root,
        tokenizer_id=proposal.prepared.tokenizer_id,
        base_model=getattr(plan, "teacher", None),
    )
    return attach_and_train(run_id, proposal, emit=emit)


def _guarded(run_id: str, call) -> dict:
    """Run training so that *any* failure leaves the registry consistent.

    The loop marks its own status once it is under way, but a crash during setup —
    a bad shard, an unloadable checkpoint — would otherwise strand the run as
    'running' forever.
    """
    try:
        return call()
    except KeyboardInterrupt:
        registry.update(run_id, status="cancelled")
        raise
    except BaseException as e:
        registry.update(run_id, status="failed", error=f"{type(e).__name__}: {e}"[:2000])
        raise


def propose_finetune(
    folder: Path,
    base: str,
    *,
    method: str | None = None,
    seq_len: int | None = None,
    epochs: float | None = None,
    seed: int = 1337,
    force: bool = False,
    progress: ProgressFn | None = None,
) -> FinetuneProposal:
    """Prepare a fine-tune of `base` on `folder`, without starting training.

    Note the ordering: the base model's metadata is read first (cheap, no weights) so
    an unusable model is reported before spending minutes tokenizing a corpus.
    """
    from llmforge.data import ingest as ing
    from llmforge.finetune import base as base_mod
    from llmforge.finetune import sft
    from llmforge.finetune.plan import plan_finetune

    if progress:
        progress("resolving base model", 0, 0)
    info = base_mod.resolve(base)

    corpus = ing.ingest(folder, force=force, progress=progress)
    if corpus.analysis.n_documents == 0:
        raise ValueError(f"No usable documents in {folder}.")

    if progress:
        progress("loading tokenizer", 0, 0)
    tokenizer = sft.load_tokenizer(info.ref)

    supervised = corpus.analysis.kind == "instruction"
    # Tokenize at the model's ceiling first, then let the plan choose a working
    # length from the resulting distribution rather than guessing beforehand.
    ceiling = min(info.max_position, 4096)
    dataset = sft.build_dataset(
        corpus, tokenizer, supervised=supervised, max_len=ceiling, progress=progress
    )

    hw = hardware.profile()
    plan = plan_finetune(
        corpus.analysis,
        info,
        hw=hw,
        n_examples=len(dataset.train),
        total_tokens=dataset.n_tokens,
        p95_length=dataset.p95_length,
        method=method,
        seq_len=seq_len,
        epochs=epochs,
        seed=seed,
    )

    # The plan chose a context from the length distribution; hold the data to it so
    # batches cannot exceed the memory the plan was budgeted against.
    dataset.truncate(plan.seq_len)

    return FinetuneProposal(
        analysis=corpus.analysis,
        plan=plan,
        corpus=corpus,
        dataset=dataset,
        tokenizer=tokenizer,
        info=info,
        hardware=hw,
    )


def start_finetune(
    proposal: FinetuneProposal,
    *,
    name: str | None = None,
    emit: EventFn | None = None,
) -> dict:
    """Register a fine-tuning run and train it to completion."""
    from llmforge.finetune.sft import train

    plan = proposal.plan
    analysis = proposal.analysis

    run_id = registry.new_run_id("finetune", plan.method)
    run_dir = paths.run_dir(run_id)

    registry.create(
        run_id=run_id,
        mode="finetune",
        plan=plan.model_dump(),
        name=name,
        corpus_hash=analysis.content_hash,
        corpus_root=analysis.root,
        base_model=plan.base_model,
    )

    lock.write(
        run_dir,
        lock.build(
            run_id=run_id,
            mode="finetune",
            plan=plan.model_dump(),
            corpus_hash=analysis.content_hash,
            corpus_root=analysis.root,
            base_model=plan.base_model,
            hardware=proposal.hardware.to_dict(),
        ),
    )

    registry.update(run_id, status="running")

    summary = _guarded(
        run_id,
        lambda: train(
            run_id=run_id,
            plan=plan,
            dataset=proposal.dataset,
            tokenizer=proposal.tokenizer,
            run_dir=run_dir,
            emit=emit,
        ),
    )
    summary["run_id"] = run_id
    summary["run_dir"] = str(run_dir)
    return summary


def propose_distill(
    folder: Path,
    teacher: str,
    *,
    tier: str | None = None,
    seq_len: int | None = None,
    temperature: float | None = None,
    alpha: float | None = None,
    seed: int = 1337,
    force: bool = False,
    progress: ProgressFn | None = None,
) -> Proposal:
    """Prepare a distillation of `teacher` on `folder`, without starting training.

    The corpus is packed with the *teacher's* tokenizer rather than one trained on the
    data: the student has to predict over the same vocabulary the teacher scores, or
    the two distributions are not comparable.
    """
    from llmforge.data import ingest as ing
    from llmforge.data import pack as pk
    from llmforge.distill.plan import DEFAULT_ALPHA, DEFAULT_TEMPERATURE, plan_distill
    from llmforge.finetune import base as base_mod
    from llmforge.finetune.sft import load_tokenizer

    if progress:
        progress("resolving teacher", 0, 0)
    info = base_mod.resolve(teacher)

    corpus = ing.ingest(folder, force=force, progress=progress)
    if corpus.analysis.n_documents == 0:
        raise ValueError(f"No usable documents in {folder}.")

    if progress:
        progress("loading teacher tokenizer", 0, 0)
    hf_tokenizer = load_tokenizer(info.ref)
    backend = getattr(hf_tokenizer, "backend_tokenizer", None)
    if backend is None:
        raise ValueError(
            f"{info.ref} has no fast tokenizer, which distillation needs for packing."
        )

    eos_id = hf_tokenizer.eos_token_id
    if eos_id is None:
        eos_id = hf_tokenizer.pad_token_id
    if eos_id is None:
        raise ValueError(f"{info.ref}'s tokenizer defines no end-of-text token.")

    # Keyed by the teacher so two teachers over one corpus do not collide.
    tokenizer_id = f"teacher-{info.ref.replace('/', '_')}"

    packed = pk.pack(
        corpus,
        backend,
        tokenizer_id,
        force=force,
        eos_id=eos_id,
        vocab_size=info.vocab_size,
        progress=progress,
    )

    analysis = corpus.analysis
    available = packed.split_tokens("train") + packed.split_tokens("val")
    if analysis.exact_tokens != available:
        analysis.exact_tokens = available
        ing.refresh_warnings(analysis)
        (corpus.dir / "analysis.json").write_text(analysis.model_dump_json(indent=2))

    hw = hardware.profile()
    plan = plan_distill(
        analysis,
        info,
        hw=hw,
        available_tokens=available,
        tier_name=tier,
        seq_len=seq_len,
        temperature=temperature if temperature is not None else DEFAULT_TEMPERATURE,
        alpha=alpha if alpha is not None else DEFAULT_ALPHA,
        seed=seed,
    )

    prepared = prep.Prepared(
        corpus=corpus,
        tokenizer=backend,
        tokenizer_dir=Path(info.ref),
        tokenizer_id=tokenizer_id,
        packed=packed,
    )
    return Proposal(analysis=analysis, plan=plan, prepared=prepared, hardware=hw)


def attach_and_train(
    run_id: str, proposal: Proposal | FinetuneProposal, *, emit: EventFn | None = None
) -> dict:
    """Train into a run record that already exists.

    `start` and `start_finetune` create their own record, which suits a CLI call that
    owns the whole operation. A worker subprocess is the other way round: the record
    was created before the worker launched, so that the GUI had something to show
    while the corpus was still being prepared.
    """
    plan = proposal.plan
    analysis = proposal.analysis
    run_dir = paths.run_dir(run_id)
    finetuning = isinstance(proposal, FinetuneProposal)

    lock.write(
        run_dir,
        lock.build(
            run_id=run_id,
            mode=plan.mode,
            plan=plan.model_dump(),
            corpus_hash=analysis.content_hash,
            corpus_root=analysis.root,
            tokenizer_id=None if finetuning else proposal.prepared.tokenizer_id,
            base_model=(
                plan.base_model if finetuning
                else getattr(plan, 'teacher', None)
            ),
            hardware=proposal.hardware.to_dict(),
        ),
    )

    registry.update(run_id, status="running", total_steps=plan.total_steps)

    if finetuning:
        from llmforge.finetune.sft import train as train_fn

        call = lambda: train_fn(  # noqa: E731
            run_id=run_id, plan=plan, dataset=proposal.dataset,
            tokenizer=proposal.tokenizer, run_dir=run_dir, emit=emit,
        )
    elif plan.mode == "distill":
        from llmforge.distill.train import train as train_fn

        call = lambda: train_fn(  # noqa: E731
            run_id=run_id, plan=plan, prepared=proposal.prepared,
            run_dir=run_dir, emit=emit,
        )
    else:
        from llmforge.pretrain.train import train as train_fn

        call = lambda: train_fn(  # noqa: E731
            run_id=run_id, plan=plan, prepared=proposal.prepared,
            run_dir=run_dir, emit=emit,
        )

    summary = _guarded(run_id, call)
    summary["run_id"] = run_id
    summary["run_dir"] = str(run_dir)
    return summary


def resume(run_id: str, *, emit: EventFn | None = None) -> dict:
    """Continue an interrupted run from its last checkpoint, of either kind."""
    record = registry.resolve(run_id)
    run_dir = paths.run_dir(record.id)

    if not record.plan:
        raise ValueError(
            f"run {record.id} never got as far as a plan; start it again rather than "
            f"resuming it"
        )

    registry.update(record.id, status="running", error=None)

    if record.mode == "finetune":
        call = _resume_finetune(record, run_dir, emit)
    elif record.mode == "distill":
        call = _resume_distill(record, run_dir, emit)
    else:
        call = _resume_pretrain(record, run_dir, emit)

    summary = _guarded(record.id, call)
    summary["run_id"] = record.id
    summary["run_dir"] = str(run_dir)
    return summary


def _resume_pretrain(record, run_dir: Path, emit: EventFn | None):
    from llmforge.pretrain.train import train

    if not record.corpus_hash or not record.tokenizer_id:
        raise ValueError(f"run {record.id} has no corpus recorded")

    plan = TrainPlan(**record.plan)
    prepared = prep.load_prepared(record.corpus_hash, record.tokenizer_id)

    return lambda: train(
        run_id=record.id, plan=plan, prepared=prepared,
        run_dir=run_dir, emit=emit, resume=True,
    )


def _resume_distill(record, run_dir: Path, emit: EventFn | None):
    """Reopen the teacher-packed corpus and continue from the checkpoint."""
    from llmforge.distill.plan import DistillPlan
    from llmforge.distill.train import train

    if not record.corpus_hash or not record.tokenizer_id:
        raise ValueError(f"run {record.id} has no corpus recorded")

    plan = DistillPlan(**record.plan)
    prepared = prep.load_prepared(record.corpus_hash, record.tokenizer_id)

    return lambda: train(
        run_id=record.id, plan=plan, prepared=prepared,
        run_dir=run_dir, emit=emit, resume=True,
    )


def _resume_finetune(record, run_dir: Path, emit: EventFn | None):
    """Rebuild the tokenized dataset, then continue from the checkpoint.

    Fine-tuning datasets are not cached the way packed shards are, so resuming means
    re-tokenizing. That is seconds of work on data this size, and it keeps the corpus
    the single source of truth rather than a second on-disk format to keep in sync.
    """
    from llmforge.data import ingest as ing
    from llmforge.finetune.plan import FinetunePlan
    from llmforge.finetune.sft import build_dataset, load_tokenizer, train

    plan = FinetunePlan(**record.plan)
    corpus = ing.load_cached(record.corpus_hash) if record.corpus_hash else None
    if corpus is None:
        corpus = ing.ingest(Path(record.corpus_root))

    tokenizer = load_tokenizer(plan.base_model)
    dataset = build_dataset(
        corpus, tokenizer, supervised=plan.supervised, max_len=plan.seq_len
    )

    return lambda: train(
        run_id=record.id, plan=plan, dataset=dataset, tokenizer=tokenizer,
        run_dir=run_dir, emit=emit, resume=True,
    )


def reproduce(run_id: str, *, emit: EventFn | None = None) -> dict:
    """Re-run a past run from its lockfile, reporting anything that has since drifted."""
    record = registry.resolve(run_id)
    run_dir = paths.run_dir(record.id)

    original = lock.read(run_dir)
    plan = TrainPlan(**original["plan"])
    inputs = original["inputs"]

    prepared = prep.load_prepared(inputs["corpus_hash"], inputs["tokenizer_id"])
    hw = hardware.profile()

    current = lock.build(
        run_id="(pending)",
        mode=record.mode,
        plan=plan.model_dump(),
        corpus_hash=inputs["corpus_hash"],
        corpus_root=inputs["corpus_root"],
        tokenizer_id=inputs["tokenizer_id"],
        hardware=hw.to_dict(),
    )
    drift = lock.diff(original, current)

    proposal = Proposal(
        analysis=prepared.analysis, plan=plan, prepared=prepared, hardware=hw
    )
    summary = start(proposal, name=f"repro of {record.id}", emit=emit)
    summary["reproduced"] = record.id
    summary["drift"] = drift
    return summary
