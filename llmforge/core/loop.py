"""The training loop, shared by pretraining and fine-tuning.

What differs between the two is what gets built and how batches are produced. What
does not differ is scheduling, gradient accumulation, evaluation, checkpointing,
resumption, and stopping cleanly — and those are exactly the parts where mistakes are
subtle and expensive. So they live here once, behind a small task interface.
"""

from __future__ import annotations

import json
import math
import os
import signal
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import torch

from llmforge.core import distributed as dist
from llmforge.core import registry
from llmforge.core.config import RunPlan

# How often to append a metrics line. Every step would bloat the file and make the
# curve noisier without telling the user anything more.
LOG_EVERY = 10

EventFn = Callable[[dict], None]


class Cancelled(Exception):
    """Raised when a stop was requested. Not an error — the checkpoint is kept."""


class TrainingTask(ABC):
    """What the loop needs from a thing being trained."""

    @abstractmethod
    def train_loss(self, step: int, micro: int) -> torch.Tensor:
        """Loss for one micro-batch. The loop handles scaling and backward."""

    @abstractmethod
    def eval_loss(self) -> float:
        """Mean loss over a fixed held-out set. Must be comparable across steps."""

    @abstractmethod
    def trainable_parameters(self):
        """Parameters to clip. For LoRA this is the adapter, not the frozen base."""

    @abstractmethod
    def checkpoint_state(self) -> dict:
        """Whatever `restore` needs to continue. Weights, and only the trainable ones
        where the rest can be reloaded from the base model."""

    @abstractmethod
    def restore(self, state: dict) -> None:
        """Inverse of `checkpoint_state`."""

    def clip_gradients(self, max_norm: float) -> float:
        """Clip and report the gradient norm.

        Overridable because FSDP shards gradients: clipping them needs the sharded
        collective, not the plain utility, or each rank clips against its own slice.
        """
        return float(
            torch.nn.utils.clip_grad_norm_(self.trainable_parameters(), max_norm)
        )

    def sample(self) -> str | None:
        """Optional generation for eyeballing progress."""
        return None

    def on_train_begin(self) -> None:
        """Hook for warmup work that must happen before stopping is armed.

        Optional: tasks with nothing to warm up need not override it.
        """
        return None


@dataclass
class _Stopper:
    """Turns SIGINT/SIGTERM into a clean stop at the next step boundary.

    Killing a run mid-step would leave a half-written checkpoint; this lets the loop
    finish its step, save, and exit consistent.
    """

    requested: bool = False
    _previous: dict = field(default_factory=dict)

    def install(self) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                self._previous[sig] = signal.signal(sig, self._handle)
            except ValueError:
                pass  # not on the main thread; the caller drives cancellation instead

    def restore(self) -> None:
        for sig, handler in self._previous.items():
            signal.signal(sig, handler)

    def _handle(self, signum, frame) -> None:
        self.requested = True


def lr_at(step: int, plan: RunPlan) -> float:
    """Linear warmup into a cosine decay — the schedule that works for this shape of
    run without needing tuning."""
    if step < plan.warmup_steps:
        return plan.lr * (step + 1) / plan.warmup_steps

    progress = (step - plan.warmup_steps) / max(1, plan.total_steps - plan.warmup_steps)
    progress = min(1.0, progress)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return plan.min_lr + (plan.lr - plan.min_lr) * cosine


def build_optimizer(task: TrainingTask, plan: RunPlan, param_groups=None):
    """AdamW, or its 8-bit variant when the plan needed the memory."""
    groups = param_groups if param_groups is not None else list(task.trainable_parameters())
    decay = plan.weight_decay if param_groups is None else 0.0

    if getattr(plan, "optimizer", "adamw") == "adamw8bit":
        try:
            import bitsandbytes as bnb

            return bnb.optim.AdamW8bit(
                groups, lr=plan.lr, betas=(plan.beta1, plan.beta2), weight_decay=decay
            )
        except ImportError:
            # The plan chose 8-bit to make the model fit, so falling back silently
            # would OOM at an unpredictable point instead of failing here.
            raise RuntimeError(
                "This plan needs an 8-bit optimizer to fit in memory, but bitsandbytes "
                "is not installed. Run `uv sync --extra finetune`."
            ) from None

    return torch.optim.AdamW(
        groups,
        lr=plan.lr,
        betas=(plan.beta1, plan.beta2),
        weight_decay=decay,
        # One kernel launch for the whole update, which matters when the step is
        # bandwidth-bound rather than compute-bound.
        fused=torch.cuda.is_available(),
    )


def save_checkpoint(path: Path, payload: dict) -> None:
    """Atomic write: a crash during checkpointing must not destroy the previous one."""
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def prune_checkpoints(ckpt_dir: Path) -> list[Path]:
    """Keep `best` and `last`, delete anything else.

    Checkpoints are large and runs are long; without this a workspace fills a 1 TB
    disk with intermediate states nobody will load. Both survivors are what resume
    and export actually use.
    """
    keep = {"best.pt", "last.pt"}
    removed = []
    for path in ckpt_dir.glob("*.pt"):
        if path.name not in keep:
            path.unlink(missing_ok=True)
            removed.append(path)
    # Interrupted atomic writes leave these behind.
    for path in ckpt_dir.glob("*.tmp"):
        path.unlink(missing_ok=True)
        removed.append(path)
    return removed


def run(
    *,
    run_id: str,
    plan: RunPlan,
    task: TrainingTask,
    optimizer: torch.optim.Optimizer,
    run_dir: Path,
    emit: EventFn | None = None,
    resume: bool = False,
    should_stop: Callable[[], bool] | None = None,
) -> dict:
    """Drive `task` to completion under `plan`. Returns a summary dict.

    Under multi-GPU every rank runs this loop and trains; side effects — metrics,
    checkpoints, registry updates, sampling — happen only on rank 0, which is what
    keeps the ranks from racing each other over the same files.
    """
    emit = emit or (lambda _: None)
    writer = dist.is_main()
    if not writer:
        emit = lambda _: None  # noqa: E731  — non-writing ranks stay silent

    metrics_path = run_dir / "metrics.jsonl"
    ckpt_dir = run_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt = ckpt_dir / "last.pt"
    best_ckpt = ckpt_dir / "best.pt"

    start_step = 0
    tokens_seen = 0
    best_val_loss: float | None = None
    elapsed_offset = 0.0

    if resume and last_ckpt.exists():
        state = torch.load(last_ckpt, map_location="cuda", weights_only=False)
        task.restore(state["task"])
        optimizer.load_state_dict(state["optimizer"])
        start_step = state["step"]
        tokens_seen = state["tokens_seen"]
        best_val_loss = state["best_val_loss"]
        elapsed_offset = state.get("elapsed_s", 0.0)
        emit({"event": "resumed", "step": start_step})

    # Any compilation or warmup happens here, before stopping is armed: torch.compile
    # farms work out to worker processes, and a signal arriving while the main thread
    # waits on that pool blocks it on a futex with no way to interrupt the run.
    task.on_train_begin()

    stopper = _Stopper()
    stopper.install()

    def snapshot(step: int, elapsed: float) -> dict:
        return {
            "task": task.checkpoint_state(),
            "optimizer": optimizer.state_dict(),
            "plan": plan.model_dump(),
            "step": step,
            "tokens_seen": tokens_seen,
            "best_val_loss": best_val_loss,
            "elapsed_s": elapsed,
        }

    started = time.perf_counter()
    window_start = started
    window_tokens = 0
    status = "completed"
    error: str | None = None
    # Steps actually finished, as opposed to planned. These diverge on cancellation
    # and the summary must report the truth.
    completed = start_step

    try:
        with metrics_path.open("a", encoding="utf-8") as metrics_file:
            for step in range(start_step, plan.total_steps):
                if stopper.requested or (should_stop and should_stop()):
                    raise Cancelled

                lr = lr_at(step, plan)
                for group in optimizer.param_groups:
                    group["lr"] = lr

                # --- one optimizer step, over grad_accum micro-batches ------------
                accumulated = 0.0
                for micro in range(plan.grad_accum):
                    loss = task.train_loss(step, micro)
                    # Scale so the accumulated gradient is the mean over the full
                    # batch, not the sum of micro-batch means.
                    (loss / plan.grad_accum).backward()
                    accumulated += loss.item() / plan.grad_accum

                grad_norm = task.clip_gradients(plan.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                tokens_seen += plan.tokens_per_step
                window_tokens += plan.tokens_per_step
                completed = step + 1
                elapsed = elapsed_offset + time.perf_counter() - started

                # --- logging -------------------------------------------------------
                if (completed % LOG_EVERY == 0 or completed == plan.total_steps) and writer:
                    torch.cuda.synchronize()
                    now = time.perf_counter()
                    tok_per_s = window_tokens / max(1e-9, now - window_start)
                    window_start, window_tokens = now, 0

                    elapsed = elapsed_offset + (now - started)
                    remaining = (plan.total_steps - completed) * plan.tokens_per_step
                    record = {
                        "step": completed,
                        "loss": round(accumulated, 5),
                        "lr": lr,
                        "grad_norm": round(float(grad_norm), 4),
                        "tokens": tokens_seen,
                        "tokens_per_sec": round(tok_per_s),
                        "elapsed_s": round(elapsed, 1),
                        # Measured, not projected — this replaces the planner's guess
                        # as soon as there is real throughput to divide by.
                        "eta_s": round(remaining / max(1.0, tok_per_s)),
                        "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                    }
                    metrics_file.write(json.dumps(record) + "\n")
                    metrics_file.flush()
                    emit({"event": "step", **record})

                    registry.update(
                        run_id,
                        step=completed,
                        train_loss=accumulated,
                        tokens_seen=tokens_seen,
                        elapsed_s=elapsed,
                    )

                # --- evaluation ----------------------------------------------------
                if completed % plan.eval_every == 0 or completed == plan.total_steps:
                    val_loss = task.eval_loss()
                    if not writer:
                        continue
                    record = {
                        "step": completed,
                        "val_loss": round(val_loss, 5),
                        "val_ppl": round(math.exp(min(val_loss, 20)), 3),
                    }
                    metrics_file.write(json.dumps(record) + "\n")
                    metrics_file.flush()
                    emit({"event": "eval", **record})

                    if best_val_loss is None or val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_checkpoint(best_ckpt, snapshot(completed, elapsed))

                    registry.update(run_id, val_loss=val_loss, best_val_loss=best_val_loss)

                # --- sampling ------------------------------------------------------
                if plan.sample_every and completed % plan.sample_every == 0 and writer:
                    text = task.sample()
                    if text is not None:
                        (run_dir / "samples").mkdir(exist_ok=True)
                        (run_dir / "samples" / f"step_{completed:06d}.txt").write_text(text)
                        emit({"event": "sample", "step": completed, "text": text})

                # --- checkpointing -------------------------------------------------
                if (
                    completed % plan.checkpoint_every == 0 or completed == plan.total_steps
                ) and writer:
                    save_checkpoint(last_ckpt, snapshot(completed, elapsed))

    except Cancelled:
        status = "cancelled"
        elapsed = elapsed_offset + time.perf_counter() - started
        if writer:
            save_checkpoint(last_ckpt, snapshot(completed, elapsed))
            registry.update(run_id, step=completed, tokens_seen=tokens_seen)
            emit({"event": "cancelled", "step": completed})

    except Exception as e:
        status = "failed"
        error = f"{type(e).__name__}: {e}"
        emit({"event": "failed", "message": error})
        raise

    finally:
        stopper.restore()
        if writer:
            prune_checkpoints(ckpt_dir)
        elapsed = elapsed_offset + time.perf_counter() - started
        if writer:
            registry.update(run_id, status=status, elapsed_s=elapsed, error=error)

    return {
        "status": status,
        "steps": completed,
        "total_steps": plan.total_steps,
        "tokens_seen": tokens_seen,
        "best_val_loss": best_val_loss,
        "elapsed_s": elapsed,
        "checkpoint": str(best_ckpt if best_ckpt.exists() else last_ckpt),
    }
