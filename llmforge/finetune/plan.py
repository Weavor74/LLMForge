"""Planning a fine-tune.

Same job as the pretraining planner, different constraints. Here the architecture is
given, so the decisions are: which adaptation method the memory budget allows, how
long the sequences need to be, and how many passes over the data are appropriate
before it starts memorising.
"""

from __future__ import annotations

from typing import Literal

from llmforge.core import memory
from llmforge.core.config import CorpusAnalysis, RunPlan
from llmforge.core.hardware import Hardware
from llmforge.finetune.base import BaseModelInfo

# LoRA rank. 16 is the usual default: enough capacity for style and domain
# adaptation, small enough that the adapter stays a rounding error in memory.
DEFAULT_LORA_RANK = 16
# Alpha at twice the rank is the common convention, giving a scaling factor of 2.
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_DROPOUT = 0.05

# Which projections to adapt. Attention plus MLP consistently beats attention alone.
LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
]

# Fine-tuning uses far lower rates than pretraining: the model is already trained and
# the job is to nudge it, not to move it. But the right rate scales with model size —
# the 1e-5..2e-5 figures quoted for fine-tuning are for 7B-and-up models, and applying
# them to a few hundred million parameters produces a run that measurably learns
# nothing. Each entry is (parameter count below which, learning rate).
FULL_LR_BY_SIZE = [
    (1e9, 1e-4),
    (4e9, 5e-5),
    (15e9, 2e-5),
    (float("inf"), 1e-5),
]

# LoRA runs an order of magnitude hotter than full fine-tuning: the adapter starts at
# zero and has further to travel, and the base weights are not at risk.
LORA_LR_BY_SIZE = [
    (4e9, 2e-4),
    (15e9, 1e-4),
    (float("inf"), 5e-5),
]


def learning_rate_for(n_params: int, method: str) -> float:
    table = FULL_LR_BY_SIZE if method == "full" else LORA_LR_BY_SIZE
    return next(lr for threshold, lr in table if n_params < threshold)

# Instruction data is small and repeats badly. Three passes is the usual ceiling.
DEFAULT_EPOCHS = 3
MAX_SEQ_LEN = 4096

# Floor on optimizer steps per epoch. Below this the schedule barely leaves warmup
# and the weights hardly move, however many epochs are requested.
MIN_STEPS_PER_EPOCH = 8

# A cosine schedule spread over a handful of steps spends most of its life at a
# near-zero rate, so a short run learns almost nothing regardless of the peak rate.
# Below this many steps we add epochs rather than accept a run that cannot move.
MIN_TOTAL_STEPS = 60
MAX_AUTO_EPOCHS = 12

# Fine-tuning achieves lower utilisation than pretraining: gradient checkpointing
# recomputes the forward pass, and QLoRA dequantizes weights on the fly.
FINETUNE_MFU = 0.15
QLORA_MFU_PENALTY = 0.6


class FinetunePlan(RunPlan):
    """A complete, executable fine-tuning configuration."""

    mode: Literal["finetune"] = "finetune"

    base_model: str
    base_params: int
    base_label: str
    architecture: str

    method: Literal["full", "lora", "qlora"]
    lora_rank: int = DEFAULT_LORA_RANK
    lora_alpha: int = DEFAULT_LORA_ALPHA
    lora_dropout: float = DEFAULT_LORA_DROPOUT
    lora_targets: list[str] = LORA_TARGETS

    gradient_checkpointing: bool = True
    trainable_params: int = 0

    # What the data looks like: supervised conversations, or raw text for continued
    # pretraining. These want different treatment and the corpus decides which.
    supervised: bool
    n_examples: int
    epochs: float


def adapter_param_count(info: BaseModelInfo, rank: int) -> int:
    """Rough LoRA parameter count: two low-rank matrices per adapted projection."""
    per_projection = 2 * rank * info.d_model
    return info.n_layer * len(LORA_TARGETS) * per_projection


def plan_finetune(
    analysis: CorpusAnalysis,
    info: BaseModelInfo,
    *,
    hw: Hardware,
    n_examples: int,
    total_tokens: int,
    p95_length: int,
    method: str | None = None,
    seq_len: int | None = None,
    epochs: float | None = None,
    seed: int = 1337,
) -> FinetunePlan:
    """Build a fine-tuning plan for this base model and corpus.

    `epochs=None` means "decide for me", which allows the run to be lengthened when
    the dataset is too small for the requested passes to produce a usable schedule.
    """
    if n_examples == 0:
        raise ValueError("corpus produced no training examples")

    epochs_auto = epochs is None
    supervised = analysis.kind == "instruction"

    # --- sequence length --------------------------------------------------------
    # Long enough for almost every example, capped by the model and by memory. Sizing
    # to the longest example would waste most of every batch on padding.
    if seq_len is None:
        seq_len = min(_round_up_pow2(p95_length), info.max_position, MAX_SEQ_LEN)
    seq_len = max(seq_len, 128)

    # --- method -----------------------------------------------------------------
    adapter_params = adapter_param_count(info, DEFAULT_LORA_RANK)

    if method:
        if method not in memory.BASE_BYTES:
            raise ValueError(
                f"unknown method '{method}' — choose from full, lora, qlora"
            )
        estimate = memory.estimate_finetune_memory(
            n_params=info.n_params,
            n_layer=info.n_layer,
            d_model=info.d_model,
            vocab_size=info.vocab_size,
            method=method,
            micro_batch=1,
            seq_len=seq_len,
            total_memory_gb=hw.memory_gb,
            adapter_params=0 if method == "full" else adapter_params,
        )
    else:
        method, estimate = memory.choose_finetune_method(
            n_params=info.n_params,
            n_layer=info.n_layer,
            d_model=info.d_model,
            vocab_size=info.vocab_size,
            seq_len=seq_len,
            total_memory_gb=hw.memory_gb,
            adapter_params=adapter_params,
        )

    trainable = info.n_params if method == "full" else adapter_params

    # --- batch shape -------------------------------------------------------------
    micro_batch = _largest_micro_batch(info, method, seq_len, hw, adapter_params)

    # Effective batches of ~64 sequences are the SFT convention, but that convention
    # assumes thousands of examples. On a small dataset a batch that size yields a
    # handful of optimizer steps and the model never moves, so cap it to leave at
    # least MIN_STEPS_PER_EPOCH updates per pass.
    conventional = 64 if supervised else 32
    target_sequences = max(1, min(conventional, n_examples // MIN_STEPS_PER_EPOCH))

    grad_accum = max(1, round(target_sequences / micro_batch))
    tokens_per_step = micro_batch * grad_accum * seq_len

    sequences_per_step = micro_batch * grad_accum
    requested_epochs = epochs if epochs is not None else DEFAULT_EPOCHS
    epochs = requested_epochs
    total_steps = max(1, round(n_examples * epochs / sequences_per_step))

    # Only stretch the run when the caller left the epoch count to us.
    auto_extended = False
    if epochs_auto and total_steps < MIN_TOTAL_STEPS:
        epochs = min(MAX_AUTO_EPOCHS, epochs * MIN_TOTAL_STEPS / total_steps)
        total_steps = max(1, round(n_examples * epochs / sequences_per_step))
        auto_extended = epochs > requested_epochs

    # --- schedule ----------------------------------------------------------------
    lr = learning_rate_for(info.n_params, method)
    warmup_steps = max(1, min(round(total_steps * 0.03), 100))
    eval_every = max(1, total_steps // 10)

    hours = _estimate_hours(
        info=info,
        method=method,
        total_tokens=int(total_tokens * epochs),
        seq_len=seq_len,
        hw=hw,
    )

    plan = FinetunePlan(
        base_model=info.ref,
        base_params=info.n_params,
        base_label=info.label,
        architecture=info.architecture,
        method=method,
        trainable_params=trainable,
        supervised=supervised,
        n_examples=n_examples,
        epochs=epochs,
        seq_len=seq_len,
        micro_batch=micro_batch,
        grad_accum=grad_accum,
        tokens_per_step=tokens_per_step,
        total_steps=total_steps,
        lr=lr,
        min_lr=lr / 10,
        warmup_steps=warmup_steps,
        weight_decay=0.0 if method != "full" else 0.01,
        eval_every=eval_every,
        sample_every=eval_every,
        checkpoint_every=max(1, total_steps // 5),
        seed=seed,
        # torch.compile and peft's hooks interact badly enough that the speedup is
        # not worth the failure modes; fine-tuning runs eagerly.
        compile=False,
        estimated_hours=round(hours, 2),
        estimated_memory_gb=round(estimate.total_gb, 1),
    )
    plan.notes = _notes(plan, analysis, info, estimate, hw)
    if auto_extended:
        plan.notes.insert(
            0,
            f"Extended to {epochs:.0f} passes: at {requested_epochs:.0f} this dataset "
            f"gives too few optimizer steps for the schedule to accomplish anything. "
            f"Watch validation loss for memorisation, or set --epochs yourself.",
        )
    return plan


def _round_up_pow2(n: int) -> int:
    size = 128
    while size < n and size < MAX_SEQ_LEN:
        size *= 2
    return size


def _largest_micro_batch(
    info: BaseModelInfo, method: str, seq_len: int, hw: Hardware, adapter_params: int
) -> int:
    best, size = 1, 1
    while size <= 16:
        estimate = memory.estimate_finetune_memory(
            n_params=info.n_params,
            n_layer=info.n_layer,
            d_model=info.d_model,
            vocab_size=info.vocab_size,
            method=method,
            micro_batch=size,
            seq_len=seq_len,
            total_memory_gb=hw.memory_gb,
            adapter_params=0 if method == "full" else adapter_params,
        )
        if not estimate.fits:
            break
        best = size
        size *= 2
    return best


def _estimate_hours(
    *, info: BaseModelInfo, method: str, total_tokens: int, seq_len: int, hw: Hardware
) -> float:
    # LoRA still backpropagates through the whole network, so the FLOP count barely
    # changes; what changes is the optimizer step, which is negligible here.
    flops_per_token = 6 * info.n_params + 12 * info.n_layer * seq_len * info.d_model
    # Gradient checkpointing recomputes the forward pass: roughly a third more work.
    flops_per_token *= 1.33

    mfu = FINETUNE_MFU * (QLORA_MFU_PENALTY if method == "qlora" else 1.0)
    achievable = hw.bf16_tflops * 1e12 * mfu
    return flops_per_token * total_tokens / achievable / 3600


def _notes(
    plan: FinetunePlan,
    analysis: CorpusAnalysis,
    info: BaseModelInfo,
    estimate: memory.MemoryEstimate,
    hw: Hardware,
) -> list[str]:
    notes: list[str] = []

    if not estimate.fits:
        notes.append(
            f"Estimated {estimate.total_gb:.0f} GB against a {estimate.budget_gb:.0f} GB "
            f"budget — this may not fit. The fit-check will reduce the batch, and will "
            f"stop the run immediately if even one sequence is too large."
        )

    if plan.method == "qlora":
        notes.append(
            "Using QLoRA: the base model is quantized to 4 bits to fit. That costs "
            "some fidelity, and dequantizing on the fly makes each step slower."
        )
    elif plan.method == "lora":
        share = plan.trainable_params / plan.base_params
        notes.append(
            f"Training a LoRA adapter — {plan.trainable_params / 1e6:.1f}M parameters, "
            f"{share:.2%} of the base. The original weights are frozen, so the result "
            f"is a small file you can apply to or remove from the base model."
        )

    if not plan.supervised:
        notes.append(
            "This corpus is raw text, not instruction data, so this is continued "
            "pretraining: the model will absorb your domain's style and vocabulary "
            "but will not learn to follow new instructions."
        )

    if plan.n_examples < 100:
        notes.append(
            f"Only {plan.n_examples} training examples. Expect very little to change; "
            f"a few hundred to a few thousand is where fine-tuning starts to bite."
        )

    if not info.has_chat_template and plan.supervised:
        notes.append(
            "The base model ships no chat template, so ChatML markers are being used. "
            "If it was instruction-tuned with a different format, the result will be "
            "worse than fine-tuning the raw base model would have been."
        )

    if plan.estimated_hours > 24:
        notes.append(
            f"Projected ~{plan.estimated_hours / 24:.1f} days. This machine has the "
            f"memory for models this size but not the bandwidth to move through them "
            f"quickly."
        )

    return notes
