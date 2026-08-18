"""HTTP and WebSocket API.

A thin layer. Every endpoint here maps onto something the CLI can also do, and the
plan a run executes is derived by the same code either way — that equivalence is what
keeps a clicked run as reproducible as a typed one.

Long work never happens on the event loop: corpus analysis runs in a thread, and
training runs in a separate process entirely.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from llmforge import __version__
from llmforge.core import hardware, paths, registry
from llmforge.jobs import runner
from llmforge.jobs.spec import (
    AnalyzeRequest,
    BrowseEntry,
    BrowseResponse,
    ChatRequest,
    EvalRequest,
    ExportRequest,
    RenameRequest,
    StartRequest,
)

# How often the WebSocket checks for new metrics. Training steps are far slower than
# this, so it reads as live without hammering the disk.
POLL_SECONDS = 0.5

@asynccontextmanager
async def lifespan(_: FastAPI):
    # A machine that lost power mid-run leaves rows claiming to be running.
    runner.reconcile()
    yield


app = FastAPI(title="LLMForge", version=__version__, lifespan=lifespan)


# ---------------------------------------------------------------------------
# system
# ---------------------------------------------------------------------------


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "version": __version__, "workspace": str(paths.workspace())}


@app.get("/api/hardware")
def get_hardware(refresh: bool = False) -> dict:
    return hardware.profile(refresh=refresh).to_dict()


@app.get("/api/doctor")
async def doctor(skip_compile: bool = True) -> dict:
    """Run the preflight checks. Off the event loop — the probes are benchmarks."""
    from llmforge import doctor as doc

    def run() -> dict:
        checks = doc.run_checks(skip_compile=skip_compile)
        return {
            "checks": [
                {"name": c.name, "status": c.status, "detail": c.detail} for c in checks
            ],
            "environment": doc.collect_environment(checks),
            "ok": not any(c.status == "fail" for c in checks),
        }

    return await asyncio.to_thread(run)


@app.post("/api/runs/{run_id}/rename")
def rename_run(run_id: str, request: RenameRequest) -> dict:
    """Name a run. Exports are filed under this name."""
    from llmforge.export.exporter import slugify

    record = _resolve(run_id)
    registry.update(record.id, name=request.name)
    return {"ok": True, "name": request.name, "slug": slugify(request.name)}


@app.get("/api/browse", response_model=BrowseResponse)
def browse(path: str | None = None, mode: str = "corpus") -> BrowseResponse:
    """Server-side directory listing for the folder picker.

    The corpus lives on the machine doing the training, which is not necessarily the
    machine running the browser, so the picker cannot use the local filesystem.

    `mode` changes only what is *reported* about each directory: a corpus is judged by
    how many readable files it holds, a model by whether it has a config.json.
    """
    target = Path(path).expanduser() if path else Path.home()
    try:
        target = target.resolve()
    except OSError as e:
        raise HTTPException(400, str(e)) from e

    if not target.is_dir():
        raise HTTPException(404, f"not a directory: {target}")

    entries: list[BrowseEntry] = []
    try:
        for child in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
            if child.name.startswith("."):
                continue
            if not child.is_dir():
                continue  # only directories are selectable
            entries.append(
                BrowseEntry(
                    name=child.name,
                    path=str(child),
                    is_dir=True,
                    # Counting files is the expensive part, so skip it when the user
                    # is looking for a model rather than a corpus.
                    n_files=None if mode == "model" else _count(child),
                    is_model=(child / "config.json").exists(),
                )
            )
    except PermissionError as e:
        raise HTTPException(403, f"permission denied: {target}") from e

    return BrowseResponse(
        path=str(target),
        parent=str(target.parent) if target.parent != target else None,
        entries=entries,
    )


def _count(directory: Path, limit: int = 2000) -> int | None:
    """Rough count of usable files, so the picker can show what is worth choosing.

    Bounded: a directory of a million files should not stall the listing.
    """
    from llmforge.data.extract import SKIP_DIRS, supported_extensions

    supported = supported_extensions()
    total = 0
    try:
        for child in directory.rglob("*"):
            if any(part in SKIP_DIRS for part in child.parts):
                continue
            if child.is_file() and child.suffix.lower() in supported:
                total += 1
                if total >= limit:
                    return limit
    except (PermissionError, OSError):
        return None
    return total


# ---------------------------------------------------------------------------
# planning
# ---------------------------------------------------------------------------


@app.post("/api/analyze")
async def analyze(request: AnalyzeRequest) -> dict:
    """Ingest the corpus and derive a plan, without training anything.

    This is the GUI's review screen. It can take minutes on a large corpus, so it runs
    off the event loop.
    """
    try:
        return await asyncio.to_thread(_analyze_sync, request)
    except (NotADirectoryError, FileNotFoundError) as e:
        raise HTTPException(404, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


def _analyze_sync(request: AnalyzeRequest) -> dict:
    from llmforge import forge

    if request.base and request.teacher:
        raise ValueError("a run is either a fine-tune or a distillation, not both")

    if request.teacher:
        proposal = forge.propose_distill(
            Path(request.folder),
            request.teacher,
            tier=request.tier,
            seq_len=request.seq_len,
            temperature=request.temperature,
            alpha=request.alpha,
            seed=request.seed,
            force=request.force,
        )
        extra = {"base_model": None}
    elif request.base:
        proposal = forge.propose_finetune(
            Path(request.folder),
            request.base,
            method=request.method,
            seq_len=request.seq_len,
            epochs=request.epochs,
            seed=request.seed,
            force=request.force,
        )
        extra = {"base_model": proposal.info.__dict__}
    else:
        proposal = forge.propose(
            Path(request.folder),
            tier=request.tier,
            seq_len=request.seq_len,
            vocab_size=request.vocab_size,
            seed=request.seed,
            force=request.force,
        )
        extra = {"base_model": None}

    return {
        "analysis": proposal.analysis.model_dump(),
        "plan": proposal.plan.model_dump(),
        "hardware": proposal.hardware.to_dict(),
        "spec": request.to_spec().model_dump(),
        **extra,
    }


# ---------------------------------------------------------------------------
# runs
# ---------------------------------------------------------------------------


@app.get("/api/runs")
def list_runs(limit: int = 50) -> list[dict]:
    return [_run_summary(r) for r in registry.list_runs(limit=limit)]


@app.post("/api/runs")
def create_run(request: StartRequest) -> dict:
    spec = request.spec
    if request.name:
        spec.name = request.name

    if not Path(spec.folder).expanduser().is_dir():
        raise HTTPException(400, f"not a directory: {spec.folder}")

    run_id = runner.start(spec)
    return {"run_id": run_id}


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict:
    record = _resolve(run_id)
    metrics, _ = runner.read_metrics(record.id)
    return {
        **_run_summary(record),
        "plan": record.plan,
        "metrics": metrics,
        "samples": runner.samples(record.id)[:5],
        "log": runner.read_log(record.id, tail_bytes=8192),
    }


@app.post("/api/runs/{run_id}/cancel")
def cancel_run(run_id: str) -> dict:
    record = _resolve(run_id)
    if not runner.cancel(record.id):
        raise HTTPException(409, "run is not active")
    return {"ok": True, "run_id": record.id}


@app.post("/api/runs/{run_id}/resume")
def resume_run(run_id: str) -> dict:
    record = _resolve(run_id)
    try:
        runner.resume(record.id)
    except (FileNotFoundError, RuntimeError) as e:
        raise HTTPException(409, str(e)) from e
    return {"ok": True, "run_id": record.id}


@app.post("/api/runs/{run_id}/chat")
async def chat(run_id: str, request: ChatRequest) -> dict:
    """Generate from a finished run. Loading a model is slow, so this runs in a thread."""
    record = _resolve(run_id)
    if record.status == "running":
        raise HTTPException(409, "cannot sample from a run that is still training")

    try:
        text = await asyncio.to_thread(_generate_sync, record, request)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e
    return {"run_id": record.id, "prompt": request.prompt, "text": text}


def _generate_sync(record, request: ChatRequest) -> str:
    import torch

    if record.mode == "finetune":
        from llmforge.finetune import infer
        from llmforge.finetune.plan import FinetunePlan

        plan = FinetunePlan(**record.plan)
        model, tokenizer = infer.load(record, checkpoint=request.checkpoint)
        return infer.generate(
            model, tokenizer, request.prompt,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            supervised=plan.supervised,
        )

    from llmforge.core.planner import TrainPlan
    from llmforge.data import prepare as prep
    from llmforge.pretrain.train import build_model, sample_text

    path = paths.run_dir(record.id) / "ckpt" / f"{request.checkpoint}.pt"
    if not path.exists():
        raise FileNotFoundError(f"no {request.checkpoint} checkpoint for {record.id}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(path, map_location=device, weights_only=False)
    plan = TrainPlan(**state["plan"])

    model = build_model(plan, device)
    model.load_state_dict(state["task"]["model"])
    model.eval()

    prepared = prep.load_prepared(record.corpus_hash, record.tokenizer_id)
    return sample_text(
        model, prepared, device,
        prompt=request.prompt,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
    )


@app.get("/api/export/levels")
def export_levels() -> dict:
    """Which formats and quantizations this machine can actually produce."""
    from llmforge.export.formats import QUANT_LEVELS, default_level

    return {
        "formats": ["gguf", "safetensors"],
        "defaults": {f: default_level(f) for f in ("gguf", "safetensors")},
        "levels": [
            {
                "name": q.name,
                "format": q.format,
                "bits": q.bits,
                "summary": q.summary,
                "available": q.available,
                "needs_llama_cpp": q.needs_llama_cpp,
            }
            for q in QUANT_LEVELS
        ],
    }


@app.post("/api/runs/{run_id}/export")
async def export_run(run_id: str, request: ExportRequest) -> dict:
    """Write a run out as a portable model. Slow, so it runs off the event loop."""
    record = _resolve(run_id)
    if record.status == "running":
        raise HTTPException(409, "cannot export a run that is still training")

    from llmforge.export import exporter

    try:
        result = await asyncio.to_thread(
            exporter.export_run,
            record.id,
            fmt=request.format,
            quantization=request.quantization,
            checkpoint=request.checkpoint,
            name=request.name,
        )
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(400, str(e)) from e

    return {
        "path": str(result.path),
        "format": result.format,
        "quantization": result.quantization,
        "megabytes": round(result.megabytes, 1),
    }


@app.post("/api/runs/{run_id}/eval")
async def eval_run(run_id: str, request: EvalRequest) -> dict:
    """Measure whether the run changed anything. Loads two models, so: a thread."""
    record = _resolve(run_id)
    if record.status == "running":
        raise HTTPException(409, "cannot evaluate a run that is still training")

    from llmforge.eval import harness

    try:
        report = await asyncio.to_thread(
            harness.evaluate,
            record.id,
            n_examples=request.examples,
            n_prompts=request.prompts,
            checkpoint=request.checkpoint,
        )
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(400, str(e)) from e

    return report.to_dict()


@app.websocket("/api/runs/{run_id}/stream")
async def stream(websocket: WebSocket, run_id: str) -> None:
    """Push metrics and status as they appear.

    State is reconstructed from disk rather than from an in-memory channel, so a
    browser that reconnects — or connects for the first time to a run already in
    progress — gets the full history and then live updates.
    """
    await websocket.accept()

    record = registry.find(run_id)
    if record is None:
        await websocket.close(code=4004, reason="no such run")
        return

    offset = 0
    last_status = None

    try:
        while True:
            record = registry.find(run_id)
            if record is None:
                break

            records, offset = runner.read_metrics(run_id, offset)
            if records:
                await websocket.send_json({"type": "metrics", "records": records})

            if record.status != last_status:
                last_status = record.status
                await websocket.send_json({"type": "status", "run": _run_summary(record)})
            else:
                await websocket.send_json({"type": "progress", "run": _run_summary(record)})

            if record.status in ("completed", "failed", "cancelled"):
                # Send any samples produced late, then stop.
                await websocket.send_json(
                    {"type": "done", "samples": runner.samples(run_id)[:5]}
                )
                break

            await asyncio.sleep(POLL_SECONDS)
    except WebSocketDisconnect:
        return
    except RuntimeError:
        # The socket closed underneath us mid-send; nothing to clean up.
        return

    try:
        await websocket.close()
    except RuntimeError:
        pass


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _resolve(run_id: str):
    try:
        return registry.resolve(run_id)
    except KeyError as e:
        raise HTTPException(404, str(e)) from e


def _run_summary(record) -> dict:
    return {
        "id": record.id,
        "name": record.name,
        "mode": record.mode,
        "status": record.status,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "step": record.step,
        "total_steps": record.total_steps,
        "progress": record.progress,
        "train_loss": record.train_loss,
        "val_loss": record.val_loss,
        "best_val_loss": record.best_val_loss,
        "tokens_seen": record.tokens_seen,
        "elapsed_s": record.elapsed_s,
        "error": record.error,
        "corpus_root": record.corpus_root,
        "base_model": record.base_model,
        "alive": runner.is_alive(record.id),
        # Enough of the plan for a list row, without shipping the whole thing.
        "tier": record.plan.get("tier"),
        "method": record.plan.get("method"),
        "n_params": record.plan.get("n_params") or record.plan.get("base_params"),
    }


# ---------------------------------------------------------------------------
# frontend
# ---------------------------------------------------------------------------


def mount_frontend() -> None:
    """Serve the built GUI, if it has been built.

    Mounted last so it cannot shadow /api routes.
    """
    dist = Path(__file__).resolve().parent.parent.parent / "web" / "dist"
    if not (dist / "index.html").exists():
        return

    app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

    @app.get("/{full_path:path}")
    def spa(full_path: str) -> FileResponse:
        # Client-side routing: unknown paths return the app, not a 404.
        return FileResponse(dist / "index.html")


mount_frontend()
