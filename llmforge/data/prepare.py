"""Folder → trained tokenizer → packed token shards.

The single call that turns "the user picked a directory" into something a training
loop can consume. Every stage is content-addressed and cached independently, so
re-running after a crash resumes rather than restarts.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from tokenizers import Tokenizer

from llmforge.data import ingest as ing
from llmforge.data import pack as pk
from llmforge.data.clean import QualityConfig
from llmforge.tokenizer import train as tok_train


@dataclass
class Prepared:
    corpus: ing.Corpus
    tokenizer: Tokenizer
    tokenizer_dir: Path
    tokenizer_id: str
    packed: pk.PackedDataset

    @property
    def analysis(self):
        return self.corpus.analysis


def _training_texts(corpus: ing.Corpus) -> Iterator[str]:
    """Text as the tokenizer should see it — chat markup included, since the model
    will have to tokenize that markup too."""
    for record in corpus.iter_records():
        text = pk.record_text(record)
        if text:
            yield text


def prepare(
    root: Path,
    *,
    vocab_size: int | None = None,
    val_fraction: float = 0.005,
    near_dupes: bool = True,
    quality: QualityConfig | None = None,
    force: bool = False,
    progress=None,
) -> Prepared:
    """Ingest, train a tokenizer, and pack. Returns everything training needs."""
    corpus = ing.ingest(
        root, force=force, near_dupes=near_dupes, quality=quality, progress=progress
    )
    analysis = corpus.analysis

    if analysis.n_documents == 0:
        raise ValueError(
            f"No usable documents in {root} — everything was filtered out. "
            "See the ingest report for why."
        )

    if vocab_size is None:
        vocab_size = tok_train.choose_vocab_size(analysis.est_tokens)

    if progress:
        progress("training tokenizer", 0, 0)
    tokenizer, tok_dir = tok_train.train(
        _training_texts(corpus),
        vocab_size=vocab_size,
        corpus_hash=analysis.content_hash,
        force=force,
    )
    tokenizer_id = tok_dir.name

    packed = pk.pack(
        corpus,
        tokenizer,
        tokenizer_id,
        val_fraction=val_fraction,
        force=force,
        progress=progress,
    )

    # Now that tokenization has happened, replace the character-based estimate with
    # the real number. Everything downstream — model sizing, step counts, time
    # estimates — depends on this being accurate.
    total = packed.split_tokens("train") + packed.split_tokens("val")
    changed = analysis.exact_tokens != total
    analysis.exact_tokens = total

    # Recompute unconditionally rather than only when the count moved: a cached
    # analysis may carry warnings written by an older version of this code, and a
    # warning that contradicts the numbers beside it is worse than no warning.
    before = list(analysis.warnings)
    ing.refresh_warnings(analysis)

    if changed or analysis.warnings != before:
        (corpus.dir / "analysis.json").write_text(analysis.model_dump_json(indent=2))

    return Prepared(
        corpus=corpus,
        tokenizer=tokenizer,
        tokenizer_dir=tok_dir,
        tokenizer_id=tokenizer_id,
        packed=packed,
    )


def load_prepared(corpus_hash: str, tokenizer_id: str) -> Prepared:
    """Reopen a previously prepared dataset, for resuming or reproducing a run."""
    from llmforge.core import paths

    corpus = ing.load_cached(corpus_hash)
    if corpus is None:
        raise FileNotFoundError(f"no ingested corpus with hash {corpus_hash}")

    if tokenizer_id.startswith("teacher-"):
        # A distillation corpus was packed with a teacher's tokenizer, which lives in
        # the model repo rather than in our tokenizer store.
        from llmforge.finetune.sft import load_tokenizer

        ref = tokenizer_id[len("teacher-") :].replace("_", "/", 1)
        tok_dir = Path(ref)
        tokenizer = load_tokenizer(ref).backend_tokenizer
    else:
        tok_dir = paths.tokenizers_dir() / tokenizer_id
        tokenizer = tok_train.load(tok_dir)

    packed_dir = corpus.dir / "packed" / tokenizer_id
    index = json.loads((packed_dir / "index.json").read_text())

    return Prepared(
        corpus=corpus,
        tokenizer=tokenizer,
        tokenizer_dir=tok_dir,
        tokenizer_id=tokenizer_id,
        packed=pk.PackedDataset(packed_dir, index),
    )
