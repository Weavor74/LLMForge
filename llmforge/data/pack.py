"""Tokenize a corpus into flat binary shards.

Training reads a contiguous stream of token ids, not documents. Packing converts the
document stream into `.bin` files of raw ids that can be memory-mapped, so the
training loop never parses anything and the OS page cache does the work.

Documents are separated by the end-of-text token, which is how the model learns that
documents end.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from llmforge.core.hashing import hash_text
from llmforge.data.ingest import Corpus
from llmforge.tokenizer.train import CHAT_END, CHAT_START, END_OF_TEXT

# Tokenizer batch size. The Rust tokenizer parallelises internally across a batch,
# so larger is faster up to the point where the batch stops fitting comfortably.
BATCH_DOCS = 512

# ~256M tokens per shard: big enough that per-file overhead vanishes, small enough
# that a shard is a convenient unit to copy or delete.
SHARD_TOKENS = 256 * 1024 * 1024

# Validation sizing. A held-out loss computed on one or two documents is noise, so
# small corpora get a proportionally larger split — but never so large that it
# meaningfully starves training.
MIN_VAL_DOCS = 8
MAX_VAL_FRACTION = 0.1


@dataclass
class PackedDataset:
    """Tokenized shards on disk, ready to memory-map."""

    dir: Path
    index: dict

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.index["dtype"])

    @property
    def vocab_size(self) -> int:
        return self.index["vocab_size"]

    def split_tokens(self, split: str) -> int:
        return self.index["splits"][split]["tokens"]

    def shard_paths(self, split: str) -> list[Path]:
        return [self.dir / s["file"] for s in self.index["splits"][split]["shards"]]

    def open_split(self, split: str) -> list[np.ndarray]:
        """Memory-map every shard of a split. No data is read until it is indexed."""
        return [np.memmap(p, dtype=self.dtype, mode="r") for p in self.shard_paths(split)]


def render_chat(messages: list[dict[str, str]]) -> str:
    """ChatML rendering, used when pretraining over conversational data.

    Fine-tuning does not come through here — it uses the base model's own template
    with proper loss masking (see `finetune`).
    """
    parts = [
        f"{CHAT_START}{m.get('role', 'user')}\n{m.get('content', '')}{CHAT_END}"
        for m in messages
    ]
    return "\n".join(parts)


def record_text(record: dict) -> str:
    if "messages" in record:
        return render_chat(record["messages"])
    return record.get("text", "")


def _is_val(record: dict, val_fraction: float) -> bool:
    """Deterministic per-document split.

    Hashing the source id rather than slicing the tail means the validation set is
    drawn from across the whole corpus, and stays identical if shard sizes change.
    """
    if val_fraction <= 0:
        return False
    bucket = int(hash_text(record.get("source", "")), 16) % 10_000
    return bucket < val_fraction * 10_000


class _ShardWriter:
    """Buffers token ids and flushes fixed-size shards to disk."""

    def __init__(self, out_dir: Path, split: str, dtype: np.dtype):
        self.out_dir = out_dir
        self.split = split
        self.dtype = dtype
        self.shards: list[dict] = []
        self._buf: list[np.ndarray] = []
        self._buffered = 0
        self.total = 0

    def add(self, ids: np.ndarray) -> None:
        self._buf.append(ids)
        self._buffered += ids.size
        self.total += ids.size
        if self._buffered >= SHARD_TOKENS:
            self.flush()

    def flush(self) -> None:
        if not self._buffered:
            return
        name = f"{self.split}_{len(self.shards):06d}.bin"
        data = np.concatenate(self._buf).astype(self.dtype, copy=False)
        data.tofile(self.out_dir / name)
        self.shards.append({"file": name, "tokens": int(data.size)})
        self._buf.clear()
        self._buffered = 0

    def manifest(self) -> dict:
        return {"shards": self.shards, "tokens": self.total}


def pack(
    corpus: Corpus,
    tokenizer: Tokenizer,
    tokenizer_id: str,
    *,
    val_fraction: float = 0.005,
    force: bool = False,
    eos_id: int | None = None,
    vocab_size: int | None = None,
    progress=None,
) -> PackedDataset:
    """Tokenize `corpus` into shards under its packed/ directory.

    `eos_id` and `vocab_size` override what the tokenizer reports. Distillation packs
    the corpus with a teacher's tokenizer, which has its own end-of-text token and
    whose reported vocabulary can be smaller than the model's embedding matrix.
    """
    out_dir = corpus.dir / "packed" / tokenizer_id
    index_path = out_dir / "index.json"

    if index_path.exists() and not force:
        return PackedDataset(out_dir, json.loads(index_path.read_text()))

    out_dir.mkdir(parents=True, exist_ok=True)

    vocab_size = vocab_size or tokenizer.get_vocab_size()
    if eos_id is None:
        eos_id = tokenizer.token_to_id(END_OF_TEXT)
    if eos_id is None:
        raise ValueError(f"tokenizer is missing the {END_OF_TEXT} token")

    # A fixed percentage rounds down to zero documents on small corpora. Raise it
    # just enough to land a handful of documents in the validation set.
    n_docs_total = corpus.analysis.n_documents
    effective_val = val_fraction
    if val_fraction > 0 and n_docs_total:
        if n_docs_total * val_fraction < MIN_VAL_DOCS:
            effective_val = min(MAX_VAL_FRACTION, MIN_VAL_DOCS / n_docs_total)

    # uint16 halves both disk and memory-bandwidth cost, which matters a great deal
    # on a machine whose binding constraint is bandwidth.
    dtype = np.dtype(np.uint16 if vocab_size <= 65_536 else np.uint32)

    writers = {
        "train": _ShardWriter(out_dir, "train", dtype),
        "val": _ShardWriter(out_dir, "val", dtype),
    }

    n_docs = 0
    for batch in _batched(corpus.iter_records(), BATCH_DOCS):
        texts = [record_text(r) for r in batch]
        encodings = tokenizer.encode_batch(texts, add_special_tokens=False)

        for record, encoding in zip(batch, encodings, strict=True):
            if not encoding.ids:
                continue
            ids = np.fromiter(encoding.ids, dtype=dtype, count=len(encoding.ids))
            # Append the boundary marker so documents do not bleed into each other.
            ids = np.append(ids, dtype.type(eos_id))
            writers["val" if _is_val(record, effective_val) else "train"].add(ids)
            n_docs += 1

        if progress:
            progress("tokenizing", n_docs, corpus.analysis.n_documents)

    for writer in writers.values():
        writer.flush()

    # A validation split that never received a document is worse than no split at all,
    # because the training loop would silently evaluate on nothing.
    if writers["val"].total == 0 and writers["train"].total > 0:
        writers["val"] = _steal_val_shard(writers["train"], out_dir, dtype)

    index = {
        "corpus_hash": corpus.analysis.content_hash,
        "tokenizer_id": tokenizer_id,
        "dtype": dtype.name,
        "vocab_size": vocab_size,
        "eos_id": eos_id,
        "n_documents": n_docs,
        "val_fraction": effective_val,
        "splits": {name: w.manifest() for name, w in writers.items()},
    }
    index_path.write_text(json.dumps(index, indent=2))

    return PackedDataset(out_dir, index)


def _steal_val_shard(
    train: _ShardWriter, out_dir: Path, dtype: np.dtype, n_tokens: int = 65_536
) -> _ShardWriter:
    """Carve a validation set off the tail of training data.

    Only reached when the corpus has too few documents for a document-aligned split
    to produce anything. Capped at a tenth of the data — a validation set large
    enough to starve training is worse than a noisy one.
    """
    val = _ShardWriter(out_dir, "val", dtype)
    if not train.shards:
        return val

    last = train.shards[-1]
    path = out_dir / last["file"]
    data = np.fromfile(path, dtype=dtype)
    take = min(n_tokens, data.size // 10)
    if take == 0:
        return val

    data[-take:].tofile(out_dir / "val_000000.bin")
    val.shards = [{"file": "val_000000.bin", "tokens": int(take)}]
    val.total = int(take)

    data[:-take].tofile(path)
    last["tokens"] = int(data.size - take)
    train.total -= take
    return val


def _batched(it: Iterator[dict], n: int) -> Iterator[list[dict]]:
    batch: list[dict] = []
    for item in it:
        batch.append(item)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch
