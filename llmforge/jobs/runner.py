"""Running training as a separate process.

Training holds a GPU for hours and blocks on CUDA calls that no amount of async will
make interruptible. So it runs in its own process: the API stays responsive, a stop
request is a real signal rather than a cooperative flag, and a crash in training takes
down training rather than the server.

Progress does not need an IPC channel. The training loop already writes `metrics.jsonl`
and updates the registry, so a reader can reconstruct the whole state of a run from
disk — which also means progress survives restarting the server.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from llmforge.core import paths, registry
from llmforge.jobs.spec import RunSpec

# How long a stopped worker gets to save its checkpoint before we stop being polite.
GRACE_SECONDS = 90


def _pid_file(run_dir: Path) -> Path:
    return run_dir / "worker.pid"


def _worker_command(run_id: str, resume: bool = False) -> list[str]:
    """How to launch the worker: directly, or under torchrun for multiple GPUs.

    The decision comes from the run's plan, which the worker itself derives — so on
    the first launch there is no plan yet and we start single-process. The worker
    re-execs itself under torchrun once it knows it needs to.
    """
    command = [sys.executable, "-m", "llmforge.jobs.worker", run_id]
    if resume:
        command.append("--resume")
    return command


def _log_file(run_dir: Path) -> Path:
    logs = run_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    return logs / "worker.log"


def start(spec: RunSpec) -> str:
    """Register a run and launch a worker for it. Returns the run id."""
    mode = "distill" if spec.is_distill else "finetune" if spec.is_finetune else "pretrain"

    # The plan is not known until the worker derives it, so the id carries no tier
    # and the registry row is filled in once training begins.
    run_id = registry.new_run_id(mode)
    run_dir = paths.run_dir(run_id)
    spec.write(run_dir)

    registry.create(
        run_id=run_id,
        mode=mode,
        plan={},
        name=spec.name,
        corpus_root=spec.folder,
        base_model=spec.base or spec.teacher,
    )

    log = _log_file(run_dir)
    with log.open("ab") as handle:
        process = subprocess.Popen(
            _worker_command(run_id),
            stdout=handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            # Its own process group, so a signal aimed at the worker cannot reach the
            # API server and vice versa.
            start_new_session=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )

    _pid_file(run_dir).write_text(str(process.pid))
    return run_id


def resume(run_id: str) -> str:
    """Relaunch a worker to continue an existing run."""
    record = registry.resolve(run_id)
    run_dir = paths.run_dir(record.id)

    if not (run_dir / "spec.json").exists():
        raise FileNotFoundError(f"run {record.id} has no spec to resume from")
    if is_alive(record.id):
        raise RuntimeError(f"run {record.id} is already running")

    registry.update(record.id, status="running", error=None)

    log = _log_file(run_dir)
    with log.open("ab") as handle:
        process = subprocess.Popen(
            _worker_command(record.id, resume=True),
            stdout=handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )

    _pid_file(run_dir).write_text(str(process.pid))
    return record.id


def worker_pid(run_id: str) -> int | None:
    path = _pid_file(paths.run_dir(run_id))
    if not path.exists():
        return None
    try:
        return int(path.read_text().strip())
    except ValueError:
        return None


def is_alive(run_id: str) -> bool:
    pid = worker_pid(run_id)
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def cancel(run_id: str) -> bool:
    """Ask a worker to stop. It saves a checkpoint and exits at the next step."""
    pid = worker_pid(run_id)
    if pid is None or not is_alive(run_id):
        return False

    os.kill(pid, signal.SIGTERM)
    return True


def kill(run_id: str) -> bool:
    """Force-stop a worker that ignored SIGTERM. Loses progress since the last
    checkpoint, which is why it is separate from `cancel`."""
    pid = worker_pid(run_id)
    if pid is None:
        return False
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return False
    registry.update(run_id, status="cancelled", error="force-stopped")
    return True


def wait(run_id: str, timeout: float = GRACE_SECONDS) -> bool:
    """Block until a worker exits. Returns False on timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not is_alive(run_id):
            return True
        time.sleep(0.25)
    return False


def reconcile() -> list[str]:
    """Mark runs whose worker is gone but which still claim to be running.

    A machine that loses power mid-run leaves the registry claiming otherwise; without
    this the GUI would show a run making progress forever.
    """
    fixed = []
    for record in registry.list_runs(limit=200, status="running"):
        if not is_alive(record.id):
            registry.update(
                record.id,
                status="failed",
                error="worker process disappeared (machine restarted, or it was killed)",
            )
            fixed.append(record.id)
    return fixed


def read_metrics(run_id: str, since_line: int = 0) -> tuple[list[dict], int]:
    """Read metric records appended since `since_line`. Returns (records, new offset)."""
    path = paths.run_dir(run_id) / "metrics.jsonl"
    if not path.exists():
        return [], since_line

    records: list[dict] = []
    # Track lines consumed, not records parsed. Advancing by the record count would
    # drift whenever a line is skipped, and the next poll would re-emit a duplicate.
    consumed = since_line

    with path.open("r", encoding="utf-8") as handle:
        for i, raw in enumerate(handle):
            if i < since_line:
                continue

            line = raw.strip()
            if not line:
                consumed = i + 1
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # Almost certainly the worker mid-write on the last line. Stop here
                # without consuming it so the next poll picks it up complete.
                break

            consumed = i + 1

    return records, consumed


def read_log(run_id: str, tail_bytes: int = 16_384) -> str:
    path = _log_file(paths.run_dir(run_id))
    if not path.exists():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        handle.seek(max(0, size - tail_bytes))
        return handle.read().decode("utf-8", errors="replace")


def samples(run_id: str) -> list[dict]:
    """Generated samples, newest first."""
    directory = paths.run_dir(run_id) / "samples"
    if not directory.exists():
        return []
    out = []
    for path in sorted(directory.glob("step_*.txt"), reverse=True):
        step = int(path.stem.split("_")[1])
        out.append({"step": step, "text": path.read_text(errors="replace")})
    return out
