"""Feeding tokens to the training loop.

Shards are memory-mapped, so the OS page cache does the buffering and startup is
instant regardless of corpus size. Batches are sampled at random offsets rather than
read sequentially: on a corpus small enough to be seen several times, sequential
order would make every epoch identical.

Sampling is driven by an explicitly seeded generator keyed to the step number, so the
same run resumed from a checkpoint sees the same batches it would have seen.
"""

from __future__ import annotations

import hashlib

import numpy as np
import torch

from llmforge.data.pack import PackedDataset


def _split_salt(split: str) -> int:
    """A per-split constant so train and val never draw the same offsets.

    Derived with a stable digest rather than hash(), which is salted per process and
    would make an identical run sample different batches each time.
    """
    return int.from_bytes(hashlib.blake2b(split.encode(), digest_size=4).digest(), "little")


class TokenStream:
    """Random-access batch sampler over a split's memory-mapped shards."""

    def __init__(self, packed: PackedDataset, split: str, seq_len: int, seed: int = 0):
        self.seq_len = seq_len
        self.split = split
        self.seed = seed
        self.salt = _split_salt(split)

        self.shards = packed.open_split(split)
        # A sequence needs seq_len inputs plus one more token for the final target.
        self.usable = np.array([max(0, s.size - seq_len - 1) for s in self.shards], dtype=np.int64)

        if self.usable.sum() == 0:
            total = sum(s.size for s in self.shards)
            raise ValueError(
                f"{split} split holds {total:,} tokens, too few for sequence length "
                f"{seq_len}. Use a shorter context or a larger corpus."
            )

        # Sample shards proportionally to how much of them is usable, so every token
        # is equally likely regardless of how the corpus was sharded.
        self.weights = self.usable / self.usable.sum()
        self.total_tokens = int(sum(s.size for s in self.shards))

    def batch(self, batch_size: int, step: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """One (inputs, targets) pair. Deterministic given `step`."""
        # Deriving the generator from (seed, salt, step) rather than advancing a
        # single stream is what makes resume-from-checkpoint reproduce the original
        # batch order. numpy requires non-negative seed components.
        rng = np.random.default_rng((self.seed, self.salt, abs(int(step))))

        shard_ids = rng.choice(len(self.shards), size=batch_size, p=self.weights)

        x = np.empty((batch_size, self.seq_len), dtype=np.int64)
        y = np.empty((batch_size, self.seq_len), dtype=np.int64)

        for i, shard_id in enumerate(shard_ids):
            shard = self.shards[shard_id]
            start = int(rng.integers(0, self.usable[shard_id] + 1))
            window = shard[start : start + self.seq_len + 1].astype(np.int64)
            x[i] = window[:-1]
            y[i] = window[1:]

        # pin_memory + non_blocking is standard, but on unified memory there is no
        # transfer to overlap, so the plain path is equivalent and simpler.
        return (
            torch.from_numpy(x).to(device, non_blocking=True),
            torch.from_numpy(y).to(device, non_blocking=True),
        )

    def deterministic_batches(
        self, batch_size: int, n_batches: int, device: torch.device
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """A fixed set of batches, identical at every evaluation.

        Validation loss is only comparable across steps if it is measured on the same
        data every time. The per-split salt keeps these from colliding with training
        batches at the same step numbers.
        """
        return [self.batch(batch_size, step=i, device=device) for i in range(n_batches)]
