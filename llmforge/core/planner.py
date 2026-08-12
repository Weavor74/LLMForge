"""Decide what to train, and be honest about it.

This is the difference between a pile of flags and something you can point at a
folder. Given how many tokens the corpus actually has and what this machine actually
measures, it picks an architecture, a batch shape, a schedule, and a step count —
then states how long that will take and how good to expect the result to be.

The rules encoded here are conventional, not clever. The value is in applying them
consistently and reporting the consequences before the compute is spent.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel

from llmforge.core import memory
from llmforge.core.config import CorpusAnalysis, RunPlan
from llmforge.core.hardware import Hardware

# Chinchilla-optimal is ~20 tokens per parameter. Below that a model is undertrained
# for its size; well above it, extra capacity would have been the better spend.
TOKENS_PER_PARAM = 20

# Repeating a small corpus helps up to a point, then memorises it. Four passes is the
# conventional ceiling before returns go sharply negative.
MAX_EPOCHS = 4

# Fraction of measured dense TFLOP/s a real training loop achieves. Small models are
# bandwidth-bound rather than compute-bound, so this is well under what the matmul
# benchmark suggests. Refined from live measurements once a run starts.
ASSUMED_MFU = 0.25


class Tier(BaseModel):
    """One rung of the model-size ladder."""

    name: str
    n_layer: int
    n_head: int
    n_kv_head: int
    d_model: int
    seq_len: int
    lr: float
    # Tokens per optimizer step. Larger models tolerate — and need — larger batches.
    batch_tokens: int


# Shapes follow the GPT-2/Llama family conventions. Every tier uses grouped-query
# attention, and d_ff is derived (see `_ffn_dim`) rather than stored.
TIERS: list[Tier] = [
    Tier(name="nano",   n_layer=6,  n_head=6,  n_kv_head=2, d_model=384,  seq_len=512,  lr=3.0e-3, batch_tokens=32_768),
    Tier(name="micro",  n_layer=8,  n_head=8,  n_kv_head=4, d_model=512,  seq_len=1024, lr=1.5e-3, batch_tokens=65_536),
    Tier(name="small",  n_layer=12, n_head=12, n_kv_head=4, d_model=768,  seq_len=1024, lr=6.0e-4, batch_tokens=262_144),
    Tier(name="medium", n_layer=24, n_head=16, n_kv_head=8, d_model=1024, seq_len=2048, lr=3.0e-4, batch_tokens=524_288),
    Tier(name="large",  n_layer=24, n_head=16, n_kv_head=8, d_model=2048, seq_len=2048, lr=2.0e-4, batch_tokens=524_288),
    Tier(name="xl",     n_layer=32, n_head=24, n_kv_head=8, d_model=3072, seq_len=4096, lr=1.5e-4, batch_tokens=1_048_576),
    Tier(name="xxl",    n_layer=32, n_head=32, n_kv_head=8, d_model=4096, seq_len=4096, lr=1.2e-4, batch_tokens=2_097_152),
]


def _ffn_dim(d_model: int, multiple_of: int = 128) -> int:
    """SwiGLU hidden width: 8/3 of d_model, rounded up to a hardware-friendly multiple.

    The 8/3 keeps a gated FFN's parameter count level with the classic 4x ungated one.
    """
    raw = int(8 * d_model / 3)
    return multiple_of * math.ceil(raw / multiple_of)


class TrainPlan(RunPlan):
    """A complete, executable from-scratch training configuration."""

    mode: Literal["pretrain"] = "pretrain"
    tier: str

    # architecture
    vocab_size: int
    n_layer: int
    n_head: int
    n_kv_head: int
    d_model: int
    d_ff: int
    dropout: float = 0.0
    n_params: int

    # how the model was made to fit
    gradient_checkpointing: bool = False
    optimizer: Literal["adamw", "adamw8bit"] = "adamw"

    # data consumption
    total_tokens: int
    epochs: float

    def model_kwargs(self) -> dict:
        """The subset that constructs a `ModelConfig`."""
        return {
            "vocab_size": self.vocab_size,
            "n_layer": self.n_layer,
            "n_head": self.n_head,
            "n_kv_head": self.n_kv_head,
            "d_model": self.d_model,
            "d_ff": self.d_ff,
            "max_seq_len": self.seq_len,
            "dropout": self.dropout,
            "gradient_checkpointing": self.gradient_checkpointing,
        }


def _params_for(tier: Tier, vocab_size: int) -> int:
    from llmforge.pretrain.model import ModelConfig

    return ModelConfig(
        vocab_size=vocab_size,
        n_layer=tier.n_layer,
        n_head=tier.n_head,
        n_kv_head=tier.n_kv_head,
        d_model=tier.d_model,
        d_ff=_ffn_dim(tier.d_model),
        max_seq_len=tier.seq_len,
    ).n_params()


def choose_tier(available_tokens: int, vocab_size: int) -> Tier:
    """Largest tier the corpus can train to a reasonable degree.

    A tier is affordable when the data — allowing repeats — reaches the point where
    the model is no longer badly undertrained. Half of Chinchilla is the cut-off:
    below that the parameters are mostly wasted.
    """
    budget = available_tokens * MAX_EPOCHS
    affordable = [
        t for t in TIERS if budget >= _params_for(t, vocab_size) * TOKENS_PER_PARAM * 0.5
    ]
    # Nothing is affordable for a very small corpus; train the smallest and warn.
    return affordable[-1] if affordable else TIERS[0]


def estimate_hours(
    *, n_params: int, n_layer: int, d_model: int, seq_len: int, total_tokens: int, hw: Hardware
) -> float:
    """Wall-clock projection from a FLOP count and measured throughput."""
    # 6 FLOPs per parameter per token covers forward and backward. The second term is
    # attention, which does not scale with parameters and starts to matter at long
    # context.
    flops_per_token = 6 * n_params + 12 * n_layer * seq_len * d_model
    total_flops = flops_per_token * total_tokens

    achievable = hw.bf16_tflops * 1e12 * ASSUMED_MFU
    return total_flops / achievable / 3600


def plan_pretrain(
    analysis: CorpusAnalysis,
    *,
    vocab_size: int,
    hw: Hardware,
    tier_name: str | None = None,
    seq_len: int | None = None,
    seed: int = 1337,
) -> TrainPlan:
    """Build a complete plan for training a model from scratch on this corpus."""
    available = analysis.tokens
    if available <= 0:
        raise ValueError("corpus has no tokens")

    if tier_name:
        matches = [t for t in TIERS if t.name == tier_name]
        if not matches:
            raise ValueError(
                f"unknown tier '{tier_name}' — choose from {', '.join(t.name for t in TIERS)}"
            )
        tier = matches[0]
    else:
        tier = choose_tier(available, vocab_size)

    seq_len = seq_len or tier.seq_len
    d_ff = _ffn_dim(tier.d_model)

    from llmforge.pretrain.model import ModelConfig

    cfg = ModelConfig(
        vocab_size=vocab_size,
        n_layer=tier.n_layer,
        n_head=tier.n_head,
        n_kv_head=tier.n_kv_head,
        d_model=tier.d_model,
        d_ff=d_ff,
        max_seq_len=seq_len,
    )
    n_params = cfg.n_params()

    # --- how many tokens to train on -------------------------------------------
    # Train to the compute-optimal point, unless the corpus runs out first — in which
    # case repeat it, up to the epoch ceiling. When there is data to spare this stops
    # part-way through a single pass, which is correct: the rest of the corpus would
    # buy less than spending the same compute on a larger model.
    ideal = n_params * TOKENS_PER_PARAM
    ceiling = available * MAX_EPOCHS
    total_tokens = int(min(ideal, ceiling))
    epochs = total_tokens / available

    # --- how many devices ---------------------------------------------------------
    strategy, n_gpus = _parallel_strategy(
        n_params=n_params, tier=tier, d_ff=d_ff, vocab_size=vocab_size,
        seq_len=seq_len, hw=hw,
    )

    # --- making it fit -----------------------------------------------------------
    # Escalate only as far as needed. Each rung buys memory and costs speed, so the
    # first configuration that fits is the right one.
    checkpointing, optimizer_name, micro_batch, mem = _fit_strategy(
        n_params=n_params, tier=tier, d_ff=d_ff, vocab_size=vocab_size,
        seq_len=seq_len, hw=hw,
    )

    # Every rank consumes its own micro-batch, so the batch a step actually covers
    # scales with the number of devices.
    tokens_per_micro = micro_batch * seq_len * n_gpus
    grad_accum = max(1, round(tier.batch_tokens / tokens_per_micro))
    tokens_per_step = tokens_per_micro * grad_accum

    total_steps = max(1, round(total_tokens / tokens_per_step))
    # Recompute so the reported token count matches what will actually be consumed.
    total_tokens = total_steps * tokens_per_step
    epochs = total_tokens / available

    # --- schedule ----------------------------------------------------------------
    warmup_steps = max(1, min(round(total_steps * 0.02), 500))

    hours = estimate_hours(
        n_params=n_params,
        n_layer=tier.n_layer,
        d_model=tier.d_model,
        seq_len=seq_len,
        total_tokens=total_tokens,
        hw=hw,
    )
    if n_gpus > 1:
        # Scaling is sublinear: DDP pays an all-reduce per step, FSDP additionally
        # gathers parameters. 90% and 75% efficiency are conservative rules of thumb.
        efficiency = 0.9 if strategy == "ddp" else 0.75
        hours /= n_gpus * efficiency
    if checkpointing:
        hours *= 1.33  # the recomputed forward pass

    # Around 20 evaluations over the run: frequent enough to see the curve, rare
    # enough not to distort throughput.
    eval_every = max(1, total_steps // 20)

    plan = TrainPlan(
        tier=tier.name,
        vocab_size=vocab_size,
        n_layer=tier.n_layer,
        n_head=tier.n_head,
        n_kv_head=tier.n_kv_head,
        d_model=tier.d_model,
        d_ff=d_ff,
        seq_len=seq_len,
        dropout=_dropout_for(epochs),
        n_params=n_params,
        gradient_checkpointing=checkpointing,
        optimizer=optimizer_name,
        strategy=strategy,
        n_gpus=n_gpus,
        micro_batch=micro_batch,
        grad_accum=grad_accum,
        tokens_per_step=tokens_per_step,
        total_steps=total_steps,
        total_tokens=total_tokens,
        epochs=round(epochs, 2),
        lr=tier.lr,
        min_lr=tier.lr / 10,
        warmup_steps=warmup_steps,
        eval_every=eval_every,
        sample_every=eval_every,
        checkpoint_every=max(1, total_steps // 5),
        seed=seed,
        compile=hw.compile_ok,
        estimated_hours=round(hours, 2),
        estimated_memory_gb=round(mem.total_gb, 1),
        notes=[],
    )
    plan.notes = _notes(plan, analysis, hw, available)
    return plan


def _parallel_strategy(
    *, n_params: int, tier: Tier, d_ff: int, vocab_size: int, seq_len: int, hw: Hardware
) -> tuple[str, int]:
    """Decide how to spread the run across whatever devices are present.

    DDP where a full replica fits on one device — it is faster, since parameters
    never move. FSDP only when the model does not fit, because sharding means
    gathering parameters on every forward and backward.
    """
    if hw.n_gpus <= 1:
        return "single", 1

    _, estimate = memory.largest_micro_batch(
        n_params=n_params, n_layer=tier.n_layer, d_model=tier.d_model, d_ff=d_ff,
        vocab_size=vocab_size, seq_len=seq_len,
        total_memory_gb=hw.memory_gb, utilisation=hw.utilisation,
        gradient_checkpointing=True,
    )
    return ("ddp" if estimate.fits else "fsdp"), hw.n_gpus


def _fit_strategy(
    *, n_params: int, tier: Tier, d_ff: int, vocab_size: int, seq_len: int, hw: Hardware
) -> tuple[bool, str, int, memory.MemoryEstimate]:
    """Find the cheapest configuration that fits, escalating only as far as needed.

    Ordered by what they cost: plain training is fastest; gradient checkpointing adds
    roughly a third more compute; an 8-bit optimizer adds quantisation overhead to
    every step. A model that fits without them should not pay for them.
    """
    ladder = [
        (False, "adamw", memory.BYTES_PER_PARAM_TRAINING),
        (True, "adamw", memory.BYTES_PER_PARAM_TRAINING),
        (True, "adamw8bit", memory.BYTES_PER_PARAM_8BIT_ADAM),
    ]

    fallback = None
    for checkpointing, optimizer_name, bytes_per_param in ladder:
        micro_batch, estimate = memory.largest_micro_batch(
            n_params=n_params,
            n_layer=tier.n_layer,
            d_model=tier.d_model,
            d_ff=d_ff,
            vocab_size=vocab_size,
            seq_len=seq_len,
            total_memory_gb=hw.memory_gb,
            utilisation=hw.utilisation,
            bytes_per_param=bytes_per_param,
            gradient_checkpointing=checkpointing,
        )
        fallback = (checkpointing, optimizer_name, micro_batch, estimate)
        if estimate.fits:
            return fallback

    # Nothing fits even at the bottom of the ladder. Return it anyway and let the
    # notes say so — the estimate is approximate and the fit-check is the real judge.
    return fallback


def _dropout_for(epochs: float) -> float:
    """Regularise only when the data will be seen repeatedly.

    On a single pass every batch is new and dropout is pure loss of throughput.
    """
    if epochs <= 1.2:
        return 0.0
    return 0.1


def _notes(plan: TrainPlan, analysis: CorpusAnalysis, hw: Hardware, available: int) -> list[str]:
    """What the user should understand before pressing Start."""
    notes: list[str] = []

    ratio = plan.total_tokens / plan.n_params
    if ratio < TOKENS_PER_PARAM * 0.5:
        notes.append(
            f"{ratio:.1f} tokens per parameter, against the ~{TOKENS_PER_PARAM} that "
            f"makes a model worth its size. This corpus cannot fill a model even this "
            f"small, so expect imitation of surface style and little more."
        )

    if plan.epochs >= 3:
        notes.append(
            f"{plan.epochs:.1f} passes over the corpus. Watch validation loss — when it "
            f"turns upward while training loss keeps falling, it is memorising."
        )

    if analysis.kind == "instruction":
        notes.append(
            "This corpus looks like instruction data. Training from scratch on it "
            "teaches the format but not the knowledge behind the answers; fine-tuning "
            "an existing model on the same folder would use it far better."
        )

    if plan.estimated_hours > 24:
        notes.append(
            f"Projected ~{plan.estimated_hours / 24:.1f} days. The estimate is refined "
            f"from measured throughput once training starts."
        )

    if plan.n_gpus > 1:
        notes.append(
            f"Training across {plan.n_gpus} GPUs with "
            + (
                "DDP — each holds a full copy and gradients are averaged every step."
                if plan.strategy == "ddp"
                else "FSDP — parameters, gradients and optimizer state are sharded "
                "across devices, which is what makes a model this size trainable."
            )
        )

    if plan.gradient_checkpointing:
        notes.append(
            "Gradient checkpointing is on: activations are recomputed in the backward "
            "pass rather than stored. That is what makes a model this size fit, and it "
            "costs roughly a third more compute per step."
        )

    if plan.optimizer == "adamw8bit":
        notes.append(
            "Using an 8-bit optimizer to fit — AdamW's moments are quantized, taking "
            "optimizer state from 8 bytes per parameter to 2. Widely used and close to "
            "lossless, but it is a compromise made for memory."
        )

    if plan.estimated_memory_gb > hw.memory_gb * hw.utilisation:
        notes.append(
            f"Estimated {plan.estimated_memory_gb:.0f} GB against a "
            f"{hw.memory_gb * hw.utilisation:.0f} GB budget — this may not fit even at "
            f"the smallest batch. Choose a smaller tier, or a machine with more memory."
        )

    if not hw.compile_ok:
        notes.append("torch.compile is unavailable — training eagerly, roughly 30% slower.")

    if not hw.flash_sdpa:
        notes.append(
            "No fused attention backend; long-context steps will use more memory than "
            "the estimate assumes."
        )

    return notes
