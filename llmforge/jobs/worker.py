"""The training subprocess.

Invoked as `python -m llmforge.jobs.worker <run_id>`. It re-derives the plan from the
run's spec rather than receiving one, so what actually trains is always what the spec
describes — and a run directory stays a complete, self-contained record.

Everything it prints goes to the run's log; progress reaches the GUI through
`metrics.jsonl` and the registry, both of which the training loop already maintains.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback

from llmforge.core import paths, registry
from llmforge.jobs.spec import RunSpec


def _log(event: str, **fields) -> None:
    """One JSON object per line, so the log is both readable and parseable."""
    print(json.dumps({"event": event, **fields}), flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="llmforge-worker")
    parser.add_argument("run_id")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    run_id = args.run_id
    run_dir = paths.run_dir(run_id)

    try:
        spec = RunSpec.read(run_dir)
    except FileNotFoundError:
        _log("failed", message="no spec.json for this run")
        registry.update(run_id, status="failed", error="no spec.json")
        return 2

    try:
        if args.resume:
            return _resume(run_id)
        return _start(run_id, spec)
    except KeyboardInterrupt:
        registry.update(run_id, status="cancelled")
        return 130
    except BaseException as e:
        # The registry is the GUI's source of truth, so a crash must be recorded there
        # and not only in a log file nobody is watching.
        detail = f"{type(e).__name__}: {e}"
        _log("failed", message=detail, traceback=traceback.format_exc()[-4000:])
        registry.update(run_id, status="failed", error=detail[:2000])
        return 1


def _progress(stage: str, done: int, total: int) -> None:
    _log("progress", stage=stage, done=done, total=total)


def _emit(event: dict) -> None:
    _log(event.pop("event", "step"), **event)


def _start(run_id: str, spec: RunSpec) -> int:
    from llmforge import forge

    _log("preparing", folder=spec.folder, base=spec.base)

    if spec.is_generate:
        return _generate(run_id, spec)

    if spec.is_distill:
        proposal = forge.propose_distill(
            spec.folder,
            spec.teacher,
            tier=spec.tier,
            seq_len=spec.seq_len,
            seed=spec.seed,
            progress=_progress,
        )
    elif spec.is_finetune:
        proposal = forge.propose_finetune(
            spec.folder,
            spec.base,
            method=spec.method,
            seq_len=spec.seq_len,
            epochs=spec.epochs,
            seed=spec.seed,
            progress=_progress,
        )
    else:
        proposal = forge.propose(
            spec.folder,
            tier=spec.tier,
            seq_len=spec.seq_len,
            vocab_size=spec.vocab_size,
            seed=spec.seed,
            progress=_progress,
        )

    plan = proposal.plan
    _log("planned", plan=plan.model_dump())

    # Multi-GPU needs a process per device, and the plan is what decides that. Since
    # the plan is only known now, relaunch under torchrun and let that run the work.
    if getattr(plan, "n_gpus", 1) > 1 and not _under_torchrun():
        return _relaunch_with_torchrun(run_id, plan.n_gpus)

    # The registry row was created before the plan existed; fill it in now so the GUI
    # can render progress bars and totals.
    registry.update(run_id, total_steps=plan.total_steps, status="running")
    _adopt_plan(run_id, proposal)

    summary = forge.attach_and_train(run_id, proposal, emit=_emit)
    _log("finished", **{k: v for k, v in summary.items() if k != "drift"})
    return 0


def _under_torchrun() -> bool:
    return "WORLD_SIZE" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1


def _relaunch_with_torchrun(run_id: str, n_gpus: int) -> int:
    """Re-exec this worker as `n_gpus` ranks.

    Everything the plan depended on is content-addressed and cached, so each rank
    re-deriving it is fast and lands on the identical plan.
    """
    _log("relaunching", n_gpus=n_gpus, reason="plan requires multiple GPUs")

    command = [
        sys.executable, "-m", "torch.distributed.run",
        f"--nproc_per_node={n_gpus}",
        "--standalone",
        "-m", "llmforge.jobs.worker", run_id,
    ]
    return subprocess.call(command, env={**os.environ, "LLMFORGE_LAUNCHED": "1"})


def _generate(run_id: str, spec: RunSpec) -> int:
    """Have a teacher write a corpus, reporting progress like a training run.

    Progress maps onto the same step/total_steps the registry already tracks, so the
    GUI's progress bar, cancellation and log streaming all work without special cases.
    """
    import signal

    from llmforge.core import paths
    from llmforge.distill import generate as gen

    stopping = {"requested": False}

    def on_signal(signum, frame):
        stopping["requested"] = True

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    _log("collecting prompts", source=spec.generate_from)
    prompts = gen.prompts_from(spec.generate_from, limit=spec.prompt_limit)
    total = len(prompts) * spec.samples_per_prompt

    out_dir = paths.workspace() / "generated" / run_id
    registry.update(run_id, status="running", total_steps=total)
    _log("generating", teacher=spec.teacher, prompts=len(prompts), answers=total)

    metrics = paths.run_dir(run_id) / "metrics.jsonl"

    def on_progress(stage: str, done: int, all_: int) -> None:
        registry.update(run_id, step=done)
        # The GUI reads progress from metrics.jsonl exactly as it does for training.
        with metrics.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"step": done, "generated": done}) + "\n")

    stats = gen.generate(
        spec.teacher,
        prompts,
        out_dir,
        samples_per_prompt=spec.samples_per_prompt,
        max_new_tokens=spec.max_new_tokens,
        temperature=spec.temperature if spec.temperature is not None else 0.8,
        top_p=spec.top_p,
        batch_size=spec.batch_size,
        system=spec.system,
        progress=on_progress,
        should_stop=lambda: stopping["requested"],
    )

    status = "cancelled" if stats.stopped else "completed"
    registry.update(run_id, status=status, step=stats.generated)
    _log(
        "finished",
        status=status,
        generated=stats.generated,
        rejected=stats.rejected,
        out_dir=str(out_dir),
    )
    return 0


def _adopt_plan(run_id: str, proposal) -> None:
    """Write the derived plan into the registry row the runner created."""
    from llmforge.core import registry as reg

    with reg.connect() as conn:
        conn.execute(
            "UPDATE runs SET plan_json = ?, corpus_hash = ?, tokenizer_id = ? WHERE id = ?",
            (
                json.dumps(proposal.plan.model_dump()),
                proposal.analysis.content_hash,
                getattr(proposal, "prepared", None)
                and proposal.prepared.tokenizer_id,
                run_id,
            ),
        )


def _resume(run_id: str) -> int:
    from llmforge import forge

    record = registry.get(run_id)
    _log("resuming", step=record.step)
    summary = forge.resume(run_id, emit=_emit)
    _log("finished", **summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
