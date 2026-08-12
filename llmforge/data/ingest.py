"""Folder in, corpus out.

This is what "select a folder" actually does: walk it, extract text from every format
we understand, drop junk and duplicates, work out whether we are looking at raw prose
or an instruction dataset, and write a normalised document stream to the workspace.

Results are keyed by content hash, so re-ingesting the same folder is a no-op and two
folders with identical contents share one corpus.
"""

from __future__ import annotations

import gzip
import json
import shutil
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from llmforge.core import paths
from llmforge.core.config import CorpusAnalysis, ExtensionStat
from llmforge.core.hashing import hash_files
from llmforge.data import extract as ex
from llmforge.data.clean import Deduper, QualityConfig, quality_reject

# Files larger than this are skipped: they are almost always data dumps or archives
# that slipped past the extension filter, and one of them can dominate a corpus.
MAX_FILE_BYTES = 2 * 1024 * 1024 * 1024

# Prose averages ~4 characters per token for English BPE. Refined to an exact count
# once the tokenizer actually runs.
CHARS_PER_TOKEN = 4.0

# A folder is called an instruction dataset when most of its content is conversational.
INSTRUCTION_THRESHOLD = 0.5

ProgressFn = Callable[[str, int, int], None]


@dataclass
class Corpus:
    """An ingested corpus on disk."""

    dir: Path
    analysis: CorpusAnalysis

    @property
    def documents_path(self) -> Path:
        return self.dir / "documents.jsonl.gz"

    def iter_records(self) -> Iterator[dict]:
        """Stream the normalised documents back out, in ingest order."""
        with gzip.open(self.documents_path, "rt", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)


def scan(root: Path) -> list[Path]:
    """Every candidate file under `root`, sorted for determinism."""
    supported = ex.supported_extensions()
    found: list[Path] = []

    for path in root.rglob("*"):
        # Skip anything inside an ignored directory, at any depth.
        if any(part in ex.SKIP_DIRS for part in path.parts):
            continue
        if path.is_symlink() or not path.is_file():
            continue
        if path.name.startswith("."):
            continue
        if path.suffix.lower() in supported:
            found.append(path)

    return sorted(found)


def _analysis_path(corpus_dir: Path) -> Path:
    return corpus_dir / "analysis.json"


def load_cached(content_hash: str) -> Corpus | None:
    corpus_dir = paths.corpora_dir() / content_hash
    analysis_file = _analysis_path(corpus_dir)
    if not analysis_file.exists() or not (corpus_dir / "documents.jsonl.gz").exists():
        return None
    return Corpus(corpus_dir, CorpusAnalysis.model_validate_json(analysis_file.read_text()))


def ingest(
    root: Path,
    *,
    force: bool = False,
    near_dupes: bool = True,
    quality: QualityConfig | None = None,
    progress: ProgressFn | None = None,
) -> Corpus:
    """Ingest a folder into the workspace, or return the cached result."""
    # Callers reach this from the CLI, the API, and user scripts; accept a string.
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"{root} is not a directory")

    quality = quality or QualityConfig()

    if progress:
        progress("scanning", 0, 0)
    files = scan(root)
    if not files:
        raise ValueError(
            f"No readable files under {root}. "
            f"Supported: {', '.join(sorted(ex.supported_extensions()))}"
        )

    if progress:
        progress("hashing", 0, len(files))
    content_hash = hash_files(root, files)

    if not force:
        cached = load_cached(content_hash)
        if cached is not None:
            if progress:
                progress("cached", len(files), len(files))
            return cached

    corpus_dir = paths.corpora_dir() / content_hash
    if corpus_dir.exists():
        shutil.rmtree(corpus_dir)  # a forced re-ingest must not merge with stale output
    corpus_dir.mkdir(parents=True)

    analysis = _run(
        root=root,
        files=files,
        content_hash=content_hash,
        corpus_dir=corpus_dir,
        quality=quality,
        near_dupes=near_dupes,
        progress=progress,
    )

    _analysis_path(corpus_dir).write_text(analysis.model_dump_json(indent=2))
    return Corpus(corpus_dir, analysis)


def _run(
    *,
    root: Path,
    files: list[Path],
    content_hash: str,
    corpus_dir: Path,
    quality: QualityConfig,
    near_dupes: bool,
    progress: ProgressFn | None,
) -> CorpusAnalysis:
    deduper = Deduper(near_dupes=near_dupes)
    by_ext: dict[str, ExtensionStat] = {}
    skip_reasons: Counter[str] = Counter()
    drop_reasons: Counter[str] = Counter()

    n_used = n_skipped = 0
    n_docs = n_chars = n_chat = 0
    n_quality = n_dupe = 0

    out_path = corpus_dir / "documents.jsonl.gz"
    manifest: list[dict] = []

    with gzip.open(out_path, "wt", encoding="utf-8") as out:
        for i, path in enumerate(files):
            if progress and i % 25 == 0:
                progress("extracting", i, len(files))

            ext = path.suffix.lower()
            stat = by_ext.setdefault(ext, ExtensionStat())
            rel = str(path.relative_to(root))

            try:
                if path.stat().st_size > MAX_FILE_BYTES:
                    raise ex.SkipFile("file too large")
                records = list(ex.extract(path, rel))
            except ex.SkipFile as e:
                skip_reasons[str(e)] += 1
                n_skipped += 1
                continue
            except Exception as e:
                # Extractors are best-effort; a malformed file loses that file only.
                skip_reasons[f"{type(e).__name__}"] += 1
                n_skipped += 1
                continue

            if not records:
                skip_reasons["no content"] += 1
                n_skipped += 1
                continue

            kept_here = 0
            for rec in records:
                text = rec.text if rec.text is not None else _render_chat(rec.messages or [])

                reason = quality_reject(text, quality)
                if reason:
                    drop_reasons[reason] += 1
                    n_quality += 1
                    continue

                # Conversations get exact-match dedup only; see Deduper.is_duplicate.
                if deduper.is_duplicate(text, near=rec.messages is None):
                    n_dupe += 1
                    continue

                payload: dict = {"source": rec.source}
                if rec.messages is not None:
                    payload["messages"] = rec.messages
                    n_chat += 1
                else:
                    payload["text"] = rec.text
                out.write(json.dumps(payload, ensure_ascii=False) + "\n")

                chars = rec.char_len()
                n_docs += 1
                n_chars += chars
                stat.documents += 1
                stat.chars += chars
                kept_here += 1

            if kept_here:
                n_used += 1
                stat.files += 1
                manifest.append({"path": rel, "documents": kept_here})
            else:
                n_skipped += 1
                skip_reasons["all documents filtered"] += 1

    (corpus_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    if progress:
        progress("done", len(files), len(files))

    chat_fraction = (n_chat / n_docs) if n_docs else 0.0
    kind = "instruction" if chat_fraction >= INSTRUCTION_THRESHOLD else "raw"
    est_tokens = int(n_chars / CHARS_PER_TOKEN)

    analysis = CorpusAnalysis(
        root=str(root),
        content_hash=content_hash,
        kind=kind,
        chat_fraction=round(chat_fraction, 4),
        n_files_scanned=len(files),
        n_files_used=n_used,
        n_files_skipped=n_skipped,
        skip_reasons=dict(skip_reasons),
        by_extension={k: v for k, v in sorted(by_ext.items()) if v.documents},
        n_documents=n_docs,
        n_chars=n_chars,
        n_dropped_quality=n_quality,
        n_dropped_duplicate=n_dupe,
        drop_reasons=dict(drop_reasons),
        est_tokens=est_tokens,
    )
    refresh_warnings(analysis)
    return analysis


def _render_chat(messages: list[dict[str, str]]) -> str:
    """Flat text view of a conversation, used only for filtering and deduplication.

    Real chat templating happens at pack time against the target tokenizer.
    """
    return "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)


def refresh_warnings(analysis: CorpusAnalysis) -> CorpusAnalysis:
    """Recompute the user-facing warnings in place.

    Called once at ingest against the character-based token estimate, and again after
    packing once the exact count is known — so the numbers quoted here never
    contradict the numbers in the summary table.
    """
    out: list[str] = []
    tokens = analysis.tokens
    n_docs = analysis.n_documents

    if n_docs == 0:
        analysis.warnings = ["No documents survived filtering — nothing to train on."]
        return analysis

    qualifier = "" if analysis.exact_tokens is not None else "~"

    if tokens < 1_000_000:
        out.append(
            f"Only {qualifier}{tokens:,} tokens. Far too little to train a model from "
            "scratch; fine-tuning an existing model is the realistic option here."
        )
    elif tokens < 50_000_000:
        out.append(
            f"{qualifier}{tokens:,} tokens is a small corpus. A from-scratch model will "
            "imitate your corpus's style but will not be generally capable."
        )

    dupe_ratio = analysis.n_dropped_duplicate / max(n_docs + analysis.n_dropped_duplicate, 1)
    if dupe_ratio > 0.3:
        out.append(f"{dupe_ratio:.0%} of documents were duplicates and were removed.")

    # The genuinely ambiguous middle: neither treatment is clearly right.
    if 0.15 < analysis.chat_fraction < 0.85:
        out.append(
            f"Mixed corpus ({analysis.chat_fraction:.0%} conversational) — treating it "
            f"as '{analysis.kind}'. Override with --kind if that is wrong."
        )

    analysis.warnings = out
    return analysis
