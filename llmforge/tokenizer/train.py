"""Byte-level BPE, trained on the user's own corpus.

For a from-scratch model the tokenizer is part of the model: a vocabulary fitted to
the actual data compresses it better than a general-purpose one, which directly buys
effective context and training throughput. For fine-tuning we never come here — the
base model's tokenizer is non-negotiable.

Byte-level means no UNK token is possible, so the tokenizer cannot fail on unexpected
input no matter what ends up in the corpus.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

from llmforge.core import paths
from llmforge.core.hashing import hash_obj, short

# One document boundary marker, doubling as BOS, EOS and PAD. Extra chat markers are
# added unconditionally so a tokenizer trained on prose can still be used for SFT
# later without resizing embeddings.
END_OF_TEXT = "<|endoftext|>"
CHAT_START = "<|im_start|>"
CHAT_END = "<|im_end|>"
SPECIAL_TOKENS = [END_OF_TEXT, CHAT_START, CHAT_END]


def choose_vocab_size(n_tokens: int) -> int:
    """Vocabulary big enough to compress well, small enough not to dominate the model.

    Embedding parameters are vocab * d_model. On a 25M-parameter model a 32k vocab
    would be more than half the network, and most of those rows would be undertrained.
    """
    if n_tokens < 5_000_000:
        return 4_096
    if n_tokens < 50_000_000:
        return 8_192
    if n_tokens < 500_000_000:
        return 16_384
    return 32_768


def tokenizer_id(corpus_hash: str, vocab_size: int) -> str:
    """Content-addressed: same corpus and vocab size means the same tokenizer."""
    return short(hash_obj({"corpus": corpus_hash, "vocab": vocab_size, "v": 1}), 16)


def load(tok_dir: Path) -> Tokenizer:
    return Tokenizer.from_file(str(tok_dir / "tokenizer.json"))


def train(
    texts: Iterator[str],
    *,
    vocab_size: int,
    corpus_hash: str,
    force: bool = False,
) -> tuple[Tokenizer, Path]:
    """Train a byte-level BPE tokenizer, or load the cached one."""
    tok_dir = paths.tokenizers_dir() / tokenizer_id(corpus_hash, vocab_size)

    if (tok_dir / "tokenizer.json").exists() and not force:
        return load(tok_dir), tok_dir

    tokenizer = Tokenizer(models.BPE(unk_token=None))
    # add_prefix_space keeps "word" and " word" as distinct tokens, matching how the
    # model will actually see text mid-sentence.
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        # A merge seen once is noise; requiring two keeps the vocabulary meaningful
        # on small corpora where rare strings would otherwise win slots.
        min_frequency=2,
        show_progress=False,
    )

    tokenizer.train_from_iterator(texts, trainer=trainer)

    tok_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(tok_dir / "tokenizer.json"))
    (tok_dir / "meta.json").write_text(
        json.dumps(
            {
                "corpus_hash": corpus_hash,
                "vocab_size": tokenizer.get_vocab_size(),
                "requested_vocab_size": vocab_size,
                "special_tokens": SPECIAL_TOKENS,
                "eos_token": END_OF_TEXT,
                "eos_id": tokenizer.token_to_id(END_OF_TEXT),
            },
            indent=2,
        )
    )
    return tokenizer, tok_dir
