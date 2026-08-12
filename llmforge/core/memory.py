"""Memory budgeting.

On the GB10 there is one 128 GB pool shared by CPU and GPU, so running out of memory
is not a graceful event — it can take the desktop down with it. The estimates here
are deliberately approximate and deliberately pessimistic; the training loop follows
them with a real fit-check that halves the micro-batch until a step actually runs.

Being roughly right here just means the fit-check usually succeeds first try.
"""

from __future__ import annotations

from dataclasses import dataclass

# Training keeps fp32 master weights and runs compute under bf16 autocast:
#   4 bytes parameter + 4 bytes gradient + 8 bytes AdamW moments.
# Weights in bf16 would halve this but cost stability on long runs, and capacity is
# the one resource this machine is not short of.
BYTES_PER_PARAM_TRAINING = 16

# With an 8-bit optimizer the AdamW moments drop from 8 bytes to 2, taking the total
# from 16 to 10. This is what makes models fit that otherwise would not.
BYTES_PER_PARAM_8BIT_ADAM = 10

# Activation bytes per token per layer per unit of d_model, measured empirically for
# a bf16 pre-norm SwiGLU block with fused attention. Covers the residual stream, the
# norm and attention intermediates, and the (wider) MLP intermediates.
ACTIVATION_BYTES_PER_TOKEN_UNIT = 2.0

# Fallback fraction of memory a training process may claim, used when no hardware
# profile is supplied. Real callers pass `hw.utilisation`, which is higher on a
# dedicated card than on unified memory shared with the desktop.
SAFE_UTILISATION = 0.75


# --- fine-tuning -----------------------------------------------------------
# Bytes per parameter of the *frozen* base, by method.
BASE_BYTES = {
    "full": 16,      # trainable: fp32 master + grad + AdamW moments
    "lora": 2,       # bf16 weights only — no gradients, no optimizer state
    "qlora": 0.55,   # nf4 packs 4 bits plus per-block scales
}

# LoRA adapters are a small fraction of the base, but they carry the full optimizer
# footprint that the frozen weights avoid.
ADAPTER_BYTES_PER_PARAM = 16

# Gradient checkpointing recomputes activations in the backward pass instead of
# storing them. Roughly this much of the activation memory survives.
CHECKPOINTING_RETENTION = 0.15


@dataclass
class MemoryEstimate:
    params_gb: float
    activations_gb: float
    logits_gb: float
    total_gb: float
    budget_gb: float

    @property
    def fits(self) -> bool:
        return self.total_gb <= self.budget_gb


def estimate_training_memory(
    *,
    n_params: int,
    n_layer: int,
    d_model: int,
    d_ff: int,
    vocab_size: int,
    micro_batch: int,
    seq_len: int,
    total_memory_gb: float,
    utilisation: float = SAFE_UTILISATION,
    bytes_per_param: int = BYTES_PER_PARAM_TRAINING,
    gradient_checkpointing: bool = False,
) -> MemoryEstimate:
    """Peak memory for one forward/backward at this micro-batch."""
    params_gb = n_params * bytes_per_param / 1e9

    tokens = micro_batch * seq_len
    # The MLP intermediate is d_ff wide, so a block's activation footprint scales
    # with d_model + the gate/up projections rather than d_model alone.
    per_layer_units = d_model * 4 + d_ff * 3
    activations_gb = (
        tokens * n_layer * per_layer_units * ACTIVATION_BYTES_PER_TOKEN_UNIT / 1e9
    )
    if gradient_checkpointing:
        # Activations are recomputed in the backward pass rather than stored, at a
        # cost of roughly one extra forward pass per step.
        activations_gb *= CHECKPOINTING_RETENTION

    # Logits are the single largest tensor in a small model: batch x seq x vocab,
    # held in bf16 and again in fp32 for the cross-entropy, plus its gradient.
    logits_gb = tokens * vocab_size * (2 + 4 + 4) / 1e9

    total = params_gb + activations_gb + logits_gb
    return MemoryEstimate(
        params_gb=params_gb,
        activations_gb=activations_gb,
        logits_gb=logits_gb,
        total_gb=total,
        budget_gb=total_memory_gb * utilisation,
    )


def largest_micro_batch(
    *,
    n_params: int,
    n_layer: int,
    d_model: int,
    d_ff: int,
    vocab_size: int,
    seq_len: int,
    total_memory_gb: float,
    ceiling: int = 64,
    utilisation: float = SAFE_UTILISATION,
    bytes_per_param: int = BYTES_PER_PARAM_TRAINING,
    gradient_checkpointing: bool = False,
) -> tuple[int, MemoryEstimate]:
    """Biggest power-of-two micro-batch predicted to fit.

    Powers of two only: they keep gradient-accumulation arithmetic exact and avoid
    the tail effects of an odd batch dimension in the attention kernels.
    """
    best = 1
    estimate = None

    size = 1
    while size <= ceiling:
        candidate = estimate_training_memory(
            n_params=n_params,
            n_layer=n_layer,
            d_model=d_model,
            d_ff=d_ff,
            vocab_size=vocab_size,
            micro_batch=size,
            seq_len=seq_len,
            total_memory_gb=total_memory_gb,
            utilisation=utilisation,
            bytes_per_param=bytes_per_param,
            gradient_checkpointing=gradient_checkpointing,
        )
        if not candidate.fits:
            break
        best, estimate = size, candidate
        size *= 2

    if estimate is None:
        # Even a single sequence is predicted not to fit. Return it anyway: the
        # estimate is approximate, and the fit-check will deliver the real verdict.
        estimate = estimate_training_memory(
            n_params=n_params,
            n_layer=n_layer,
            d_model=d_model,
            d_ff=d_ff,
            vocab_size=vocab_size,
            micro_batch=1,
            seq_len=seq_len,
            total_memory_gb=total_memory_gb,
            utilisation=utilisation,
            bytes_per_param=bytes_per_param,
            gradient_checkpointing=gradient_checkpointing,
        )
    return best, estimate


def estimate_finetune_memory(
    *,
    n_params: int,
    n_layer: int,
    d_model: int,
    vocab_size: int,
    method: str,
    micro_batch: int,
    seq_len: int,
    total_memory_gb: float,
    adapter_params: int = 0,
    gradient_checkpointing: bool = True,
    utilisation: float = SAFE_UTILISATION,
) -> MemoryEstimate:
    """Peak memory for one fine-tuning step.

    The three methods differ almost entirely in what the *frozen* base costs: full
    fine-tuning pays optimizer state on every parameter, LoRA pays it on almost none,
    and QLoRA additionally compresses the frozen weights to 4 bits.
    """
    weights_gb = n_params * BASE_BYTES[method] / 1e9
    adapter_gb = adapter_params * ADAPTER_BYTES_PER_PARAM / 1e9

    tokens = micro_batch * seq_len
    # A transformer block's activations, without the architecture detail we do not
    # have for an arbitrary base model. d_ff is folded into the constant.
    activations_gb = tokens * n_layer * d_model * 12 * 2 / 1e9
    if gradient_checkpointing:
        activations_gb *= CHECKPOINTING_RETENTION

    # Logits dominate for large vocabularies: batch x seq x vocab, in bf16 and again
    # in fp32 for the loss, plus the gradient.
    logits_gb = tokens * vocab_size * (2 + 4 + 4) / 1e9

    total = weights_gb + adapter_gb + activations_gb + logits_gb
    return MemoryEstimate(
        params_gb=weights_gb + adapter_gb,
        activations_gb=activations_gb,
        logits_gb=logits_gb,
        total_gb=total,
        budget_gb=total_memory_gb * utilisation,
    )


def choose_finetune_method(
    *,
    n_params: int,
    n_layer: int,
    d_model: int,
    vocab_size: int,
    seq_len: int,
    total_memory_gb: float,
    adapter_params: int,
    allow_full: bool = True,
    utilisation: float = SAFE_UTILISATION,
) -> tuple[str, MemoryEstimate]:
    """Pick the highest-quality method that fits, at a micro-batch of one.

    Ordered by how much of the model actually learns: full fine-tuning updates
    everything, LoRA updates a low-rank correction, QLoRA does the same on a
    quantized base and pays for it in fidelity. We take the best that fits rather
    than the fastest, because the run is going to be slow regardless.
    """
    order = ["full", "lora", "qlora"] if allow_full else ["lora", "qlora"]

    fallback = None
    for method in order:
        estimate = estimate_finetune_memory(
            n_params=n_params,
            n_layer=n_layer,
            d_model=d_model,
            vocab_size=vocab_size,
            method=method,
            micro_batch=1,
            seq_len=seq_len,
            total_memory_gb=total_memory_gb,
            adapter_params=0 if method == "full" else adapter_params,
            utilisation=utilisation,
        )
        fallback = (method, estimate)
        if estimate.fits:
            return method, estimate

    # Nothing fits. Return the cheapest option and let the caller warn — the estimate
    # is approximate and the fit-check gives the real answer.
    return fallback
