"""LLMForge command line.

Everything the GUI can do is available here, and both go through the same config
objects — that equivalence is what makes runs reproducible rather than clicked.
"""

from __future__ import annotations

import math
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="llmforge",
    help="Point at a folder, get a model.",
    no_args_is_help=True,
    add_completion=False,
)

console = Console()


@app.callback()
def main() -> None:
    """Keeps typer in multi-command mode regardless of how many commands exist."""


_GLYPH = {
    "pass": ("[green]OK[/green]", "green"),
    "warn": ("[yellow]WARN[/yellow]", "yellow"),
    "fail": ("[red]FAIL[/red]", "red"),
    "skip": ("[dim]--[/dim]", "dim"),
}


@app.command()
def doctor(
    skip_compile: bool = typer.Option(
        False, "--skip-compile", help="Skip the torch.compile probe (it is the slow one)."
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit machine-readable results."),
) -> None:
    """Verify this machine can actually train — before a run finds out the hard way."""
    from llmforge import doctor as doc

    with console.status("[dim]probing hardware...[/dim]", spinner="dots"):
        results = doc.run_checks(skip_compile=skip_compile)

    if json_out:
        import json

        payload = {
            "checks": [{"name": c.name, "status": c.status, "detail": c.detail} for c in results],
            "environment": doc.collect_environment(results),
        }
        console.print_json(json.dumps(payload))
    else:
        table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
        table.add_column("", width=6)
        table.add_column("check", style="bold")
        table.add_column("detail", overflow="fold")

        for c in results:
            glyph, style = _GLYPH[c.status]
            table.add_row(glyph, c.name, f"[{style}]{c.detail}[/{style}]")

        console.print()
        console.print(table)
        console.print()

    failures = [c for c in results if c.status == "fail"]
    warnings = [c for c in results if c.status == "warn"]

    if failures:
        if not json_out:
            console.print(f"[red]{len(failures)} blocking problem(s).[/red] Training will not work.")
        raise typer.Exit(1)

    if not json_out:
        summary = "Ready to train."
        if warnings:
            summary += f" [yellow]{len(warnings)} warning(s)[/yellow] — usable, but degraded."
        console.print(summary)


def _fmt(n: int) -> str:
    """Compact human counts — corpora span six orders of magnitude."""
    for limit, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if n >= limit:
            return f"{n / limit:.1f}{suffix}"
    return str(n)


def print_analysis(analysis) -> None:
    """Shared rendering for `ingest` and the analysis step of `create`."""
    from llmforge.core.config import CorpusAnalysis

    assert isinstance(analysis, CorpusAnalysis)

    summary = Table(show_header=False, box=None, pad_edge=False)
    summary.add_column(style="dim", width=16)
    summary.add_column()

    summary.add_row("corpus", f"[bold]{analysis.root}[/bold]")
    summary.add_row("fingerprint", analysis.content_hash[:12])
    kind_label = (
        "instruction / chat data" if analysis.kind == "instruction" else "raw text"
    )
    summary.add_row("detected as", f"[bold]{kind_label}[/bold]")
    summary.add_row(
        "files", f"{analysis.n_files_used:,} used, {analysis.n_files_skipped:,} skipped"
    )
    summary.add_row("documents", f"{analysis.n_documents:,}")
    summary.add_row("characters", f"{_fmt(analysis.n_chars)}")
    summary.add_row(
        "tokens",
        f"[bold]~{_fmt(analysis.tokens)}[/bold]"
        + ("" if analysis.exact_tokens is not None else " (estimated)"),
    )
    summary.add_row(
        "dropped",
        f"{analysis.n_dropped_duplicate:,} duplicate, {analysis.n_dropped_quality:,} low quality",
    )

    console.print()
    console.print(summary)

    if analysis.by_extension:
        console.print()
        breakdown = Table(title="by format", title_justify="left", box=None, pad_edge=False)
        breakdown.add_column("ext", style="cyan")
        breakdown.add_column("files", justify="right")
        breakdown.add_column("docs", justify="right")
        breakdown.add_column("chars", justify="right")
        for ext, stat in sorted(
            analysis.by_extension.items(), key=lambda kv: -kv[1].chars
        ):
            breakdown.add_row(ext, f"{stat.files:,}", f"{stat.documents:,}", _fmt(stat.chars))
        console.print(breakdown)

    for warning in analysis.warnings:
        console.print(f"\n[yellow]![/yellow] {warning}")


@app.command()
def ingest(
    folder: Path = typer.Argument(..., help="Folder of corpus data to ingest."),
    force: bool = typer.Option(False, "--force", help="Re-ingest even if cached."),
    no_dedupe: bool = typer.Option(
        False, "--no-dedupe", help="Skip near-duplicate detection (faster, lower quality)."
    ),
) -> None:
    """Read a folder of corpus data and report what is in it."""
    from llmforge.data import ingest as ing

    with console.status("[dim]reading corpus...[/dim]", spinner="dots") as status:

        def on_progress(stage: str, done: int, total: int) -> None:
            suffix = f" {done:,}/{total:,}" if total else ""
            status.update(f"[dim]{stage}{suffix}[/dim]")

        try:
            corpus = ing.ingest(
                folder, force=force, near_dupes=not no_dedupe, progress=on_progress
            )
        except (NotADirectoryError, ValueError) as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1) from e

    print_analysis(corpus.analysis)
    console.print(f"\n[dim]stored at {corpus.dir}[/dim]")


@app.command()
def prepare(
    folder: Path = typer.Argument(..., help="Folder of corpus data."),
    vocab_size: int = typer.Option(
        0, "--vocab-size", help="Tokenizer vocabulary size. 0 derives it from corpus size."
    ),
    val_fraction: float = typer.Option(0.005, "--val-fraction", help="Held-out fraction."),
    force: bool = typer.Option(False, "--force", help="Rebuild even if cached."),
) -> None:
    """Ingest a folder, train a tokenizer on it, and pack it into training shards."""
    from llmforge.data import prepare as prep

    with console.status("[dim]preparing...[/dim]", spinner="dots") as status:

        def on_progress(stage: str, done: int, total: int) -> None:
            suffix = f" {done:,}/{total:,}" if total else ""
            status.update(f"[dim]{stage}{suffix}[/dim]")

        try:
            result = prep.prepare(
                folder,
                vocab_size=vocab_size or None,
                val_fraction=val_fraction,
                force=force,
                progress=on_progress,
            )
        except (NotADirectoryError, ValueError, FileNotFoundError) as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1) from e

    print_analysis(result.analysis)

    packed = result.packed
    train_tokens = packed.split_tokens("train")
    val_tokens = packed.split_tokens("val")
    # How many characters one token buys. Higher is better: it means the tokenizer
    # fits the corpus, so the same context window holds more text.
    ratio = result.analysis.n_chars / max(train_tokens + val_tokens, 1)

    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="dim", width=16)
    table.add_column()
    table.add_row("tokenizer", f"{packed.vocab_size:,} vocab  [dim]({result.tokenizer_id})[/dim]")
    table.add_row("compression", f"{ratio:.2f} chars/token")
    table.add_row("train", f"{train_tokens:,} tokens in {len(packed.shard_paths('train'))} shard(s)")
    table.add_row("val", f"{val_tokens:,} tokens")
    table.add_row("dtype", packed.index["dtype"])

    console.print()
    console.print(table)
    console.print(f"\n[dim]packed at {packed.dir}[/dim]")


def _fmt_duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    if seconds < 172800:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


def print_plan(proposal) -> None:
    """The review screen: what will be trained, how long, and how good to expect."""
    plan = proposal.plan

    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="dim", width=16)
    table.add_column()

    table.add_row("mode", "[bold]train from scratch[/bold]")
    table.add_row("size", f"[bold]{plan.tier}[/bold] — {_fmt(plan.n_params)} parameters")
    table.add_row(
        "architecture",
        f"{plan.n_layer} layers, {plan.n_head} heads "
        f"({plan.n_kv_head} kv), d_model {plan.d_model}, ffn {plan.d_ff}",
    )
    table.add_row("context", f"{plan.seq_len:,} tokens")
    table.add_row("vocabulary", f"{plan.vocab_size:,}")
    table.add_row(
        "batch",
        f"{plan.micro_batch} x {plan.grad_accum} accum = "
        f"{_fmt(plan.tokens_per_step)} tokens/step",
    )
    table.add_row(
        "training",
        f"{plan.total_steps:,} steps, {_fmt(plan.total_tokens)} tokens "
        f"({plan.epochs:.1f} passes over the corpus)",
    )
    table.add_row("learning rate", f"{plan.lr:.1e} → {plan.min_lr:.1e} cosine, {plan.warmup_steps} warmup")
    table.add_row("memory", f"~{plan.estimated_memory_gb:.1f} GB of {proposal.hardware.memory_gb:.0f} GB")
    table.add_row(
        "estimated time",
        f"[bold]{_fmt_duration(plan.estimated_hours * 3600)}[/bold] [dim](refined once training starts)[/dim]",
    )

    console.print()
    console.print(table)

    for note in plan.notes:
        console.print(f"\n[yellow]![/yellow] {note}")


def print_finetune_plan(proposal) -> None:
    """The review screen for a fine-tune."""
    plan = proposal.plan
    info = proposal.info

    method_label = {
        "full": "full fine-tune (every weight updated)",
        "lora": "LoRA adapter (base frozen)",
        "qlora": "QLoRA adapter (base quantized to 4-bit, then frozen)",
    }[plan.method]

    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="dim", width=16)
    table.add_column()

    table.add_row("mode", "[bold]fine-tune an existing model[/bold]")
    table.add_row("base model", f"[bold]{plan.base_model}[/bold] — {info.label} ({plan.architecture})")
    table.add_row("method", f"[bold]{method_label}[/bold]")
    table.add_row(
        "trainable",
        f"{_fmt(plan.trainable_params)} of {_fmt(plan.base_params)} parameters "
        f"({plan.trainable_params / plan.base_params:.2%})",
    )
    table.add_row(
        "data",
        f"{plan.n_examples:,} examples, "
        + ("supervised conversations" if plan.supervised else "raw text (continued pretraining)"),
    )
    table.add_row("context", f"{plan.seq_len:,} tokens")
    table.add_row(
        "batch", f"{plan.micro_batch} x {plan.grad_accum} accum = {plan.micro_batch * plan.grad_accum} sequences/step"
    )
    table.add_row("training", f"{plan.total_steps:,} steps over {plan.epochs:.1f} epochs")
    table.add_row("learning rate", f"{plan.lr:.1e} → {plan.min_lr:.1e} cosine, {plan.warmup_steps} warmup")
    table.add_row("memory", f"~{plan.estimated_memory_gb:.1f} GB of {proposal.hardware.memory_gb:.0f} GB")
    table.add_row(
        "estimated time",
        f"[bold]{_fmt_duration(plan.estimated_hours * 3600)}[/bold] [dim](refined once training starts)[/dim]",
    )

    console.print()
    console.print(table)

    for note in plan.notes:
        console.print(f"\n[yellow]![/yellow] {note}")


def print_distill_plan(proposal) -> None:
    """The review screen for a distillation."""
    plan = proposal.plan

    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="dim", width=16)
    table.add_column()

    table.add_row("mode", "[bold]distil a teacher into a smaller model[/bold]")
    table.add_row(
        "teacher",
        f"[bold]{plan.teacher}[/bold] — {plan.teacher_label}"
        + ("  [yellow](loaded 4-bit)[/yellow]" if plan.teacher_load_4bit else ""),
    )
    table.add_row("student", f"[bold]{plan.tier}[/bold] — {_fmt(plan.n_params)} parameters")
    table.add_row(
        "architecture",
        f"{plan.n_layer} layers, {plan.n_head} heads ({plan.n_kv_head} kv), "
        f"d_model {plan.d_model}, ffn {plan.d_ff}",
    )
    table.add_row("vocabulary", f"{plan.vocab_size:,} [dim](the teacher's)[/dim]")
    table.add_row("context", f"{plan.seq_len:,} tokens")
    table.add_row(
        "objective",
        f"{plan.alpha:.0%} teacher KL at T={plan.temperature:g}, "
        f"{1 - plan.alpha:.0%} true-token cross-entropy",
    )
    table.add_row(
        "batch",
        f"{plan.micro_batch} x {plan.grad_accum} accum = {_fmt(plan.tokens_per_step)} tokens/step",
    )
    table.add_row(
        "training",
        f"{plan.total_steps:,} steps, {_fmt(plan.total_tokens)} tokens "
        f"({plan.epochs:.1f} passes)",
    )
    table.add_row(
        "learning rate",
        f"{plan.lr:.1e} → {plan.min_lr:.1e} cosine, {plan.warmup_steps} warmup",
    )
    table.add_row(
        "memory",
        f"~{plan.estimated_memory_gb:.1f} GB of {proposal.hardware.memory_gb:.0f} GB "
        f"[dim](teacher + student)[/dim]",
    )
    table.add_row(
        "estimated time",
        f"[bold]{_fmt_duration(plan.estimated_hours * 3600)}[/bold] "
        f"[dim](refined once training starts)[/dim]",
    )

    console.print()
    console.print(table)

    for note in plan.notes:
        console.print(f"\n[yellow]![/yellow] {note}")


@app.command()
def create(
    folder: Path = typer.Argument(..., help="Folder of corpus data."),
    base: str = typer.Option(
        None,
        "--base",
        "-b",
        help="Fine-tune this model instead of training from scratch "
        "(Hugging Face id or local path).",
    ),
    teacher: str = typer.Option(
        None,
        "--teacher",
        "-t",
        help="Distil this model into a smaller one you build (Hugging Face id or path).",
    ),
    name: str = typer.Option(None, "--name", help="A label for this run."),
    tier: str = typer.Option(
        None, "--tier", help="Force a model size: nano, micro, small, medium, large."
    ),
    seq_len: int = typer.Option(None, "--seq-len", help="Override the context length."),
    vocab_size: int = typer.Option(None, "--vocab-size", help="Override tokenizer vocabulary."),
    method: str = typer.Option(
        None, "--method", help="Force full, lora, or qlora (fine-tuning only)."
    ),
    epochs: float = typer.Option(
        None, "--epochs", help="Passes over the data (fine-tuning only). Derived if unset."
    ),
    seed: int = typer.Option(1337, "--seed"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show the plan and stop without training."
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Start without confirming."),
    force: bool = typer.Option(False, "--force", help="Re-ingest the corpus from scratch."),
) -> None:
    """Build a model from a folder of corpus data, or fine-tune one with --base."""
    from llmforge import forge

    if base and teacher:
        console.print("[red]--base and --teacher are different modes; pick one.[/red]")
        raise typer.Exit(1)

    finetuning = base is not None
    distilling = teacher is not None

    with console.status("[dim]preparing...[/dim]", spinner="dots") as status:

        def on_progress(stage: str, done: int, total: int) -> None:
            suffix = f" {done:,}/{total:,}" if total else ""
            status.update(f"[dim]{stage}{suffix}[/dim]")

        try:
            if distilling:
                proposal = forge.propose_distill(
                    folder,
                    teacher,
                    tier=tier,
                    seq_len=seq_len,
                    seed=seed,
                    force=force,
                    progress=on_progress,
                )
            elif finetuning:
                proposal = forge.propose_finetune(
                    folder,
                    base,
                    method=method,
                    seq_len=seq_len,
                    epochs=epochs,
                    seed=seed,
                    force=force,
                    progress=on_progress,
                )
            else:
                proposal = forge.propose(
                    folder,
                    tier=tier,
                    seq_len=seq_len,
                    vocab_size=vocab_size,
                    seed=seed,
                    force=force,
                    progress=on_progress,
                )
        except (NotADirectoryError, ValueError, FileNotFoundError) as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1) from e

    print_analysis(proposal.analysis)
    if finetuning:
        print_finetune_plan(proposal)
    elif distilling:
        print_distill_plan(proposal)
    else:
        print_plan(proposal)

    if dry_run:
        console.print("\n[dim]--dry-run: stopping before training.[/dim]")
        return

    if not yes:
        console.print()
        if not typer.confirm("Start training?", default=True):
            raise typer.Exit(0)

    runner = forge.start_finetune if finetuning else forge.start
    _train_with_progress(lambda emit: runner(proposal, name=name, emit=emit), proposal.plan)


def _train_with_progress(runner, plan) -> None:
    """Drive a training call, rendering live progress."""
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeRemainingColumn,
    )

    console.print()
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed:,}/{task.total:,}"),
        TextColumn("{task.fields[detail]}"),
        TimeRemainingColumn(),
        console=console,
    )

    with progress:
        task = progress.add_task("training", total=plan.total_steps, detail="")

        def emit(event: dict) -> None:
            kind = event.get("event")
            if kind == "step":
                progress.update(
                    task,
                    completed=event["step"],
                    detail=(
                        f"loss {event['loss']:.3f}  "
                        f"{_fmt(event['tokens_per_sec'])} tok/s  "
                        f"{event['peak_gb']:.1f}GB"
                    ),
                )
            elif kind == "eval":
                progress.console.print(
                    f"  [dim]step {event['step']:>7,}[/dim]  "
                    f"val loss [bold]{event['val_loss']:.4f}[/bold]  "
                    f"ppl {event['val_ppl']:.1f}"
                )
            elif kind == "sample":
                snippet = event["text"].strip().replace("\n", " ")[:160]
                progress.console.print(f"  [dim]sample:[/dim] [italic]{snippet}[/italic]")
            elif kind in ("plan_adjusted", "fit_check", "compile_failed", "resumed"):
                progress.console.print(f"  [yellow]{kind}:[/yellow] [dim]{event}[/dim]")

        try:
            summary = runner(emit)
        except KeyboardInterrupt:
            console.print("\n[yellow]interrupted — checkpoint saved[/yellow]")
            raise typer.Exit(130) from None

    console.print()
    if summary["status"] == "cancelled":
        console.print(
            f"[yellow]Stopped at step {summary['steps']:,} of "
            f"{summary['total_steps']:,}.[/yellow] "
            f"Resume with [bold]llmforge resume {summary['run_id']}[/bold]"
        )
        return

    best = summary.get("best_val_loss")
    console.print(f"[green]Done.[/green] run [bold]{summary['run_id']}[/bold]")
    if best is not None:
        console.print(f"  best validation loss {best:.4f}  (perplexity {math.exp(min(best, 20)):.1f})")
    console.print(f"  {_fmt(summary['tokens_seen'])} tokens in {_fmt_duration(summary['elapsed_s'])}")
    console.print(f"\nTry it:  [bold]llmforge sample {summary['run_id']}[/bold]")


@app.command()
def resume(
    run_id: str = typer.Argument("last", help="Run id, prefix, or 'last'."),
) -> None:
    """Continue an interrupted run from its last checkpoint."""
    from llmforge import forge
    from llmforge.core import registry
    from llmforge.core.planner import TrainPlan

    try:
        record = registry.resolve(run_id)
    except KeyError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e

    console.print(f"Resuming [bold]{record.id}[/bold] from step {record.step:,}")
    _train_with_progress(lambda emit: forge.resume(record.id, emit=emit), TrainPlan(**record.plan))


@app.command()
def rename(
    run_id: str = typer.Argument(..., help="Run id, prefix, or 'last'."),
    name: str = typer.Argument(..., help="The new name."),
) -> None:
    """Name a run, or rename one. Exports use this name."""
    from llmforge.core import registry
    from llmforge.export.exporter import slugify

    try:
        record = registry.resolve(run_id)
    except KeyError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e

    registry.update(record.id, name=name)
    console.print(f"[green]Renamed.[/green] {record.id} is now [bold]{name}[/bold]")
    console.print(f"  [dim]exports will be filed as {slugify(name)}-<quant>.gguf[/dim]")


@app.command(name="runs")
def list_runs(
    limit: int = typer.Option(20, "--limit", "-n"),
) -> None:
    """List training runs."""
    from llmforge.core import registry

    records = registry.list_runs(limit=limit)
    if not records:
        console.print("[dim]No runs yet. Start one with `llmforge create <folder>`.[/dim]")
        return

    colours = {
        "completed": "green",
        "running": "cyan",
        "failed": "red",
        "cancelled": "yellow",
        "pending": "dim",
    }

    table = Table(box=None, pad_edge=False, header_style="bold")
    table.add_column("run")
    table.add_column("name", style="cyan")
    table.add_column("status")
    table.add_column("progress", justify="right")
    table.add_column("val loss", justify="right")
    table.add_column("tokens", justify="right")
    table.add_column("time", justify="right")
    table.add_column("corpus", overflow="ellipsis", max_width=28)

    for r in records:
        colour = colours.get(r.status, "white")
        table.add_row(
            r.id,
            r.name or "[dim]—[/dim]",
            f"[{colour}]{r.status}[/{colour}]",
            f"{r.progress:.0%}" if r.total_steps else "-",
            f"{r.best_val_loss:.4f}" if r.best_val_loss is not None else "-",
            _fmt(r.tokens_seen),
            _fmt_duration(r.elapsed_s),
            Path(r.corpus_root).name if r.corpus_root else "-",
        )

    console.print()
    console.print(table)


@app.command()
def sample(
    run_id: str = typer.Argument("last", help="Run id, prefix, or 'last'."),
    prompt: str = typer.Option("", "--prompt", "-p", help="Text to continue."),
    tokens: int = typer.Option(200, "--tokens", "-n", help="How many tokens to generate."),
    temperature: float = typer.Option(0.8, "--temperature", "-t"),
    checkpoint: str = typer.Option("best", "--checkpoint", help="'best' or 'last'."),
) -> None:
    """Generate text from a trained model, from either kind of run."""
    import torch

    from llmforge.core import paths, registry
    from llmforge.core.planner import TrainPlan
    from llmforge.data import prepare as prep
    from llmforge.pretrain.train import build_model, sample_text

    try:
        record = registry.resolve(run_id)
    except KeyError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e

    if record.mode == "finetune":
        _sample_finetuned(record, prompt, tokens, temperature, checkpoint)
        return

    ckpt_path = paths.run_dir(record.id) / "ckpt" / f"{checkpoint}.pt"
    if not ckpt_path.exists():
        console.print(f"[red]no {checkpoint} checkpoint for {record.id}[/red]")
        raise typer.Exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    plan = TrainPlan(**state["plan"])

    model = build_model(plan, device)
    # Checkpoints nest task state under "task" so the loop can serve pretraining,
    # fine-tuning and distillation through one format.
    model.load_state_dict(state["task"]["model"])
    model.eval()

    prepared = prep.load_prepared(record.corpus_hash, record.tokenizer_id)

    text = sample_text(
        model, prepared, device, prompt=prompt, max_new_tokens=tokens, temperature=temperature
    )

    console.print()
    if prompt:
        console.print(f"[dim]{prompt}[/dim]", end="")
    console.print(text)


def _sample_finetuned(record, prompt: str, tokens: int, temperature: float, checkpoint: str) -> None:
    """Generation for fine-tuning runs, which reload a base model plus an adapter."""
    from llmforge.finetune import infer
    from llmforge.finetune.plan import FinetunePlan

    plan = FinetunePlan(**record.plan)
    if not prompt:
        prompt = "Hello" if plan.supervised else ""

    with console.status(f"[dim]loading {plan.base_model}...[/dim]", spinner="dots"):
        try:
            model, tokenizer = infer.load(record, checkpoint=checkpoint)
        except FileNotFoundError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1) from e

    text = infer.generate(
        model,
        tokenizer,
        prompt,
        max_new_tokens=tokens,
        temperature=temperature,
        supervised=plan.supervised,
    )

    console.print()
    if plan.supervised:
        console.print(f"[dim]user:[/dim] {prompt}")
        console.print(f"[dim]{plan.base_label} + your data:[/dim] {text.strip()}")
    else:
        console.print(f"[dim]{prompt}[/dim]{text}")


@app.command()
def export(
    run_id: str = typer.Argument("last", help="Run id, prefix, or 'last'."),
    fmt: str = typer.Option("gguf", "--format", "-f", help="gguf or safetensors."),
    quantize: str = typer.Option(
        None, "--quantize", "-q", help="Quantization level. Derived if unset."
    ),
    checkpoint: str = typer.Option("best", "--checkpoint", help="'best' or 'last'."),
    out: Path = typer.Option(None, "--out", "-o", help="Where to write it."),
    name: str = typer.Option(
        None, "--name", help="Name the exported model. Defaults to the run's name."
    ),
    list_levels: bool = typer.Option(
        False, "--list", help="Show the levels available and stop."
    ),
) -> None:
    """Export a trained model so it runs outside LLMForge."""
    from llmforge.core import registry
    from llmforge.export import exporter
    from llmforge.export.formats import default_level, estimate_bytes, levels_for

    if fmt not in ("gguf", "safetensors"):
        console.print(f"[red]unknown format '{fmt}' — use gguf or safetensors[/red]")
        raise typer.Exit(1)

    if list_levels:
        table = Table(box=None, pad_edge=False, header_style="bold")
        table.add_column("level")
        table.add_column("bits", justify="right")
        table.add_column("")
        table.add_column("notes", overflow="fold")
        for level in levels_for(fmt):
            available = "" if level.available else "[dim](needs llama.cpp)[/dim]"
            table.add_row(level.name, f"{level.bits:g}", available, level.summary)
        console.print()
        console.print(table)
        console.print(f"\n[dim]default: {default_level(fmt)}[/dim]")
        return

    try:
        record = registry.resolve(run_id)
    except KeyError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e

    level_name = quantize or default_level(fmt)
    n_params = record.plan.get("n_params") or record.plan.get("base_params") or 0
    if n_params:
        from llmforge.export.formats import find_level

        try:
            estimate = estimate_bytes(n_params, find_level(fmt, level_name))
            console.print(f"[dim]roughly {estimate / 1e6:.0f} MB at {level_name}[/dim]")
        except ValueError:
            pass

    with console.status("[dim]exporting...[/dim]", spinner="dots") as status:

        def on_progress(stage: str, done: int, total: int) -> None:
            status.update(f"[dim]{stage}[/dim]")

        try:
            result = exporter.export_run(
                record.id,
                fmt=fmt,
                quantization=quantize,
                checkpoint=checkpoint,
                out_dir=out,
                name=name,
                progress=on_progress,
            )
        except (ValueError, FileNotFoundError) as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1) from e

    console.print(f"\n[green]Exported.[/green] {result.megabytes:.1f} MB, {result.quantization}")
    console.print(f"  [bold]{result.path}[/bold]")
    if result.format == "gguf":
        console.print(f"\n[dim]Run it:  ollama create mymodel -f Modelfile  # FROM {result.path}[/dim]")


@app.command()
def generate(
    teacher: str = typer.Argument(..., help="Run id, Hugging Face id, or local model path."),
    source: Path = typer.Option(
        ..., "--from", "-f", help="Corpus folder, or a text file with one prompt per line."
    ),
    out: Path = typer.Option(..., "--out", "-o", help="Where to write the generated corpus."),
    limit: int = typer.Option(None, "--limit", "-n", help="Use at most this many prompts."),
    samples: int = typer.Option(
        1, "--samples", "-s", help="Answers to generate per prompt."
    ),
    max_tokens: int = typer.Option(512, "--max-tokens"),
    temperature: float = typer.Option(0.8, "--temperature", "-t"),
    batch: int = typer.Option(8, "--batch", help="Prompts generated at once."),
    system: str = typer.Option(None, "--system", help="System prompt for the teacher."),
) -> None:
    """Have a teacher write a training corpus, for distilling into a smaller model.

    The teacher answers each prompt once; the result is a folder `create` can train on.
    Unlike scoring the teacher on every token of every epoch, this pays for the teacher
    a single time — and leaves the student free to use its own tokenizer.
    """
    from llmforge.distill import generate as gen

    with console.status("[dim]collecting prompts...[/dim]", spinner="dots") as status:
        try:
            prompts = gen.prompts_from(source, limit=limit)
        except (ValueError, NotADirectoryError, FileNotFoundError) as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1) from e

        total = len(prompts) * samples
        status.update(f"[dim]loading {teacher}...[/dim]")

        def on_progress(stage: str, done: int, all_: int) -> None:
            status.update(f"[dim]{stage} {done:,}/{all_:,}[/dim]")

        console.print(
            f"{len(prompts):,} prompts x {samples} = [bold]{total:,}[/bold] answers to write"
        )
        try:
            stats = gen.generate(
                teacher, prompts, out,
                samples_per_prompt=samples,
                max_new_tokens=max_tokens,
                temperature=temperature,
                batch_size=batch,
                system=system,
                progress=on_progress,
            )
        except (ValueError, FileNotFoundError) as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1) from e

    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="dim", width=14)
    table.add_column()
    table.add_row("prompts", f"{stats.prompts:,}")
    table.add_row("generated", f"[bold]{stats.generated:,}[/bold]")
    table.add_row("rejected", f"{stats.rejected:,}")
    for reason, count in stats.reasons.items():
        table.add_row("", f"[dim]{count:,} {reason}[/dim]")

    console.print()
    console.print(table)
    console.print(f"\n[dim]written to {out}[/dim]")
    console.print(f"\nTrain on it:  [bold]llmforge create {out} --base <small model>[/bold]")


@app.command(name="eval")
def evaluate_run(
    run_id: str = typer.Argument("last", help="Run id, prefix, or 'last'."),
    examples: int = typer.Option(64, "--examples", "-n", help="Held-out examples to score."),
    prompts: int = typer.Option(5, "--prompts", "-p", help="Prompts to compare on."),
    checkpoint: str = typer.Option("best", "--checkpoint"),
) -> None:
    """Measure whether a run actually changed anything."""
    from llmforge.core import registry
    from llmforge.eval import harness

    try:
        record = registry.resolve(run_id)
    except KeyError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e

    with console.status("[dim]evaluating...[/dim]", spinner="dots") as status:

        def on_progress(stage: str, done: int, total: int) -> None:
            status.update(f"[dim]{stage}[/dim]")

        try:
            report = harness.evaluate(
                record.id,
                n_examples=examples,
                n_prompts=prompts,
                checkpoint=checkpoint,
                progress=on_progress,
            )
        except (ValueError, FileNotFoundError) as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1) from e

    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="dim", width=18)
    table.add_column()
    table.add_row("run", report.run_id)
    table.add_row("held-out examples", f"{report.n_examples:,}")
    if report.before_ppl is not None:
        label = "teacher" if report.mode == "distill" else "before"
        table.add_row(f"perplexity {label}", f"{report.before_ppl:.2f}")
    table.add_row(
        "perplexity yours",
        f"[bold]{report.after_ppl:.2f}[/bold]" if report.after_ppl else "—",
    )
    if report.improvement is not None:
        delta = report.improvement
        colour = "green" if delta > 0.05 else "yellow" if delta > -0.05 else "red"
        table.add_row("change", f"[{colour}]{delta:+.1%}[/{colour}]")

    console.print()
    console.print(table)

    for note in report.notes:
        console.print(f"\n[yellow]![/yellow] {note}")

    if report.comparisons:
        console.print("\n[bold]Generations[/bold]")
        for comparison in report.comparisons:
            console.print(f"\n  [dim]prompt:[/dim] {comparison.prompt[:120]}")
            if comparison.before:
                console.print(f"  [dim]before:[/dim] {comparison.before[:200]}")
            console.print(f"  [dim]after: [/dim] [bold]{comparison.after[:200]}[/bold]")


@app.command(name="app")
def desktop_app(
    port: int = typer.Option(None, "--port", help="Preferred port. Steps along if taken."),
    no_window: bool = typer.Option(False, "--no-window", help="Start the server only."),
) -> None:
    """Launch LLMForge as a desktop application."""
    import time

    from llmforge import app as launcher

    preferred = port or launcher.DEFAULT_PORT
    try:
        chosen, already = launcher.choose_port(preferred)
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e

    url = f"http://127.0.0.1:{chosen}"
    server = None

    if already:
        console.print(f"[dim]Already running at {url} — opening it.[/dim]")
    else:
        console.print(f"  [bold]LLMForge[/bold]  starting at {url}")
        server = launcher.start_server(chosen)
        if not launcher.wait_for_server(chosen):
            server.terminate()
            console.print(
                f"[red]Server did not start within "
                f"{launcher.STARTUP_TIMEOUT:.0f}s.[/red] See "
                f"{paths_workspace_log()}"
            )
            raise typer.Exit(1)

    if no_window:
        console.print(f"  serving at {url} — Ctrl-C to stop")
        try:
            while server and server.poll() is None:
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            _shutdown(server)
        return

    window = launcher.open_window(url)

    if window is None:
        # We opened a tab in a shared browser and cannot tell when it closes, so the
        # server has to outlive this command.
        console.print(f"  opened {url} in your browser")
        console.print("  [dim]the server keeps running; stop it with: pkill -f 'llmforge app'[/dim]")
        try:
            while server and server.poll() is None:
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            _shutdown(server)
        return

    # Own the window: when the user closes it, put the server away too.
    try:
        window.wait()
    except KeyboardInterrupt:
        window.terminate()
    finally:
        _shutdown(server)
        console.print(
            "\n[dim]Closed. Any training already started keeps running — "
            "reopen to watch it.[/dim]"
        )


def paths_workspace_log() -> str:
    from llmforge.core import paths

    return str(paths.workspace() / "server.log")


def _shutdown(server) -> None:
    """Stop the server we started, if we started one. Training workers are in their
    own sessions and are deliberately left alone."""
    if server is None or server.poll() is not None:
        return
    server.terminate()
    try:
        server.wait(timeout=10)
    except Exception:
        server.kill()


@app.command(name="install-app")
def install_app(
    uninstall: bool = typer.Option(False, "--uninstall", help="Remove the launcher."),
) -> None:
    """Add LLMForge to the desktop application menu."""
    from llmforge import app as launcher

    if uninstall:
        removed = launcher.uninstall_desktop_entry()
        if removed:
            console.print("Removed:")
            for path in removed:
                console.print(f"  [dim]{path}[/dim]")
        else:
            console.print("[dim]Nothing installed.[/dim]")
        return

    entry, icon = launcher.install_desktop_entry()
    console.print("\n[green]Installed.[/green] LLMForge is now in your applications menu.")
    console.print(f"  [dim]{entry}[/dim]")
    console.print(f"  [dim]{icon}[/dim]")
    console.print("\nSearch for [bold]LLMForge[/bold], or run [bold]llmforge app[/bold].")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address. 0.0.0.0 exposes it on the LAN."),
    port: int = typer.Option(8000, "--port"),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code changes."),
) -> None:
    """Start the web interface."""
    import uvicorn

    from llmforge.core import paths

    dist = Path(__file__).resolve().parent.parent / "web" / "dist"
    if not (dist / "index.html").exists():
        console.print(
            "[yellow]The GUI has not been built.[/yellow] Run [bold]npm install && "
            "npm run build[/bold] in web/, or use the API directly."
        )

    console.print(f"\n  [bold]LLMForge[/bold]  http://{host}:{port}")
    console.print(f"  [dim]workspace {paths.workspace()}[/dim]\n")

    uvicorn.run(
        "llmforge.api.main:app", host=host, port=port, reload=reload, log_level="warning"
    )


if __name__ == "__main__":
    app()
