"""Quality filtering and deduplication.

Duplicated text is the most common way a small corpus quietly ruins a model: the
network spends its capacity memorising the repeated span instead of learning the
distribution. Exact duplicates are cheap to catch; near-duplicates (the same article
with a different header, a file saved twice with one line changed) need MinHash.

The MinHash + LSH implementation here is deliberately dependency-free and streaming —
it holds only band signatures, not documents.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

import numpy as np

# Mersenne prime, standard choice for the (a*x + b) mod p universal hash family.
_MERSENNE = (1 << 61) - 1

_WORD_RE = re.compile(r"\w+", re.UNICODE)

# Python's built-in hash() is salted per process, so using it here would make
# deduplication — and therefore the resulting model — differ between runs.
def _stable_hash(data: bytes) -> int:
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "little")


@dataclass
class QualityConfig:
    min_chars: int = 64
    # Below this, a "document" is usually a table fragment, a base64 blob, or
    # navigation chrome that survived HTML stripping.
    min_alpha_ratio: float = 0.5
    # Fraction of lines allowed to be duplicates of another line in the same doc.
    max_line_dup_ratio: float = 0.5
    # Catches minified assets and single-line data dumps.
    max_mean_line_length: int = 5000


def quality_reject(text: str, cfg: QualityConfig) -> str | None:
    """Return a rejection reason, or None if the document is worth keeping."""
    stripped = text.strip()
    if len(stripped) < cfg.min_chars:
        return "too short"

    alpha = sum(c.isalpha() or c.isspace() for c in stripped)
    if alpha / len(stripped) < cfg.min_alpha_ratio:
        return "low alphabetic ratio"

    lines = [ln.strip() for ln in stripped.split("\n") if ln.strip()]
    if lines:
        if len(lines) >= 8:
            unique = len(set(lines))
            if 1.0 - (unique / len(lines)) > cfg.max_line_dup_ratio:
                return "repetitive lines"
        if sum(len(ln) for ln in lines) / len(lines) > cfg.max_mean_line_length:
            return "pathological line length"

    return None


# Bottom-k sketch size. Without a cap, a 10 MB document yields millions of shingles
# and the signature matrix below would need gigabytes. Keeping the k smallest hashes
# is itself a valid MinHash sketch, so similarity estimates survive the truncation.
_MAX_SHINGLES = 4096


def _shingles(text: str, k: int = 5) -> np.ndarray:
    """Hashed word k-grams. Word-level rather than character-level so that
    reformatting (rewrapped lines, changed indentation) doesn't defeat matching."""
    words = _WORD_RE.findall(text.lower())
    if len(words) < k:
        return np.empty(0, dtype=np.uint64)

    grams = {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}
    hashes = np.fromiter(
        (_stable_hash(g.encode()) & 0xFFFFFFFFFFFFFFF for g in grams),
        dtype=np.uint64,
        count=len(grams),
    )

    if hashes.size > _MAX_SHINGLES:
        hashes = np.partition(hashes, _MAX_SHINGLES)[:_MAX_SHINGLES]
    return hashes


@dataclass
class Deduper:
    """Streaming exact + near-duplicate detector.

    bands * rows = the number of MinHash permutations. 16x4 puts the LSH threshold
    near 0.7 Jaccard similarity: aggressive enough to catch reformatted copies,
    conservative enough to keep genuinely distinct documents that share boilerplate.
    """

    bands: int = 16
    rows: int = 4
    near_dupes: bool = True
    seed: int = 0

    _exact: set[int] = field(default_factory=set, init=False)
    _buckets: set[tuple[int, int]] = field(default_factory=set, init=False)
    n_exact: int = field(default=0, init=False)
    n_near: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        n = self.bands * self.rows
        rng = np.random.default_rng(self.seed)
        # Odd multipliers keep the hash family well-behaved mod a prime.
        self._a = rng.integers(1, _MERSENNE, size=n, dtype=np.uint64) | 1
        self._b = rng.integers(0, _MERSENNE, size=n, dtype=np.uint64)

    def _signature(self, shingles: np.ndarray) -> np.ndarray:
        # (n_perm, n_shingles) then min over shingles. Materialising this is fine:
        # documents are chunked well before they get large enough to matter.
        hashed = (self._a[:, None] * shingles[None, :] + self._b[:, None]) % _MERSENNE
        return hashed.min(axis=1)

    def is_duplicate(self, text: str, near: bool = True) -> bool:
        """Check and record in one pass. Returns True if `text` was seen before.

        `near` disables fuzzy matching for this record. Instruction data is full of
        examples that legitimately share an answer across differently-worded
        questions — that repetition is what teaches robustness to phrasing — and
        fuzzy matching would delete most of the dataset.
        """
        key = _stable_hash(text.strip().encode("utf-8", errors="replace"))
        if key in self._exact:
            self.n_exact += 1
            return True
        self._exact.add(key)

        if not (self.near_dupes and near):
            return False

        shingles = _shingles(text)
        if shingles.size == 0:
            return False  # too short to fingerprint meaningfully

        sig = self._signature(shingles)
        band_keys = [
            (b, _stable_hash(sig[b * self.rows : (b + 1) * self.rows].tobytes()))
            for b in range(self.bands)
        ]

        # A hit in any single band is enough — that is the point of LSH banding.
        if any(k in self._buckets for k in band_keys):
            self.n_near += 1
            return True

        self._buckets.update(band_keys)
        return False
