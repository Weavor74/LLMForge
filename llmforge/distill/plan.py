"""Planning a distillation.

Distillation trains a small model to imitate a large one. Instead of learning only
from "the next token was X", the student learns the teacher's whole probability
distribution — which carries far more information per token and is why a distilled
model beats one of the same size trained on the same data from scratch.

The constraint that shapes everything here: student and teacher must share a
vocabulary, so the student is built around the *teacher's* tokenizer rather than one
fitted to the corpus.
"""

from __future__ import annotations

from typing import Literal

from llmforge.core import memory
from llmforge.core.config import CorpusAnalysis, RunPlan
from llmforge.core.hardware import Hardware
from llmforge.core.planner import MAX_EPOCHS, TIERS, _ffn_dim
from llmforge.finetune.base import BaseModelInfo

# Softening applied to both distributions before comparing them. Above 1 this exposes
# the teacher's opinion about tokens it did *not* pick, which is the signal that makes
# distillation worth doing at all.
DEFAULT_TEMPERATURE = 2.0

# Share of the loss taken from the teacher rather than from the true next token.
# Weighting the teacher heavily is standard; the hard labels mainly keep the student
# anchored where the teacher is wrong.
DEFAULT_ALPHA = 0.7

# A student this close to its teacher in size is not worth distilling — just use the
# teacher, or fine-tune it.
MAX_STUDENT_FRACTION = 0.5

# Fraction of measured dense throughput a distillation step achieves. The teacher
# runs inference-only, which is bandwidth-bound rather than compute-bound.
DISTILL_MFU = 0.2

# Dequantizing a 4-bit teacher on every forward costs roughly this much extra.
QUANTIZED_TEACHER_PENALTY = 1.5

# A cosine schedule spread over a couple of steps spends its whole life in warmup or
# near-zero, so a short run learns nothing regardless of the rate. When the corpus is
# smaller than one batch, shrink the batch rather than accept a one-step run.
MIN_TOTAL_STEPS = 30


class DistillPlan(RunPlan):
    """A complete, executable distillation configuration."""

    mode: Literal["distill"] = "distill"

    teacher: str
    teacher_params: int
    teacher_label: str
    teacher_load_4bit: bool = False

    # The student is one of the from-scratch tiers, built on the teacher's vocabulary.
    tier: str
    vocab_size: int
    n_layer: int
    n_head: int
    n_kv_head: int
    d_model: int
    d_ff: int
    dropout: float = 0.0
    n_params: int

    temperature: float = DEFAULT_TEMPERATURE
    alpha: float = DEFAULT_ALPHA

    total_tokens: int
    epochs: float

    def model_kwargs(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "n_layer": self.n_layer,
            "n_head": self.n_head,
            "n_kv_head": self.n_kv_head,
            "d_model": self.d_model,
            "d_ff": self.d_ff,
            "max_seq_len": self.seq_len,
            "dropout": self.dropout,
        }


def _student_params(tier, vocab_size: int) -> int:
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


def choose_student(teacher: BaseModelInfo, available_tokens: int) -> object:
    """Largest tier that is both affordable from the data and smaller than the teacher.

    Distillation is more data-efficient than pretraining — the teacher supplies a full
    distribution per token rather than one label — so the token budget can support a
    bigger student here than `choose_tier` would allow.
    """
    ceiling = teacher.n_params * MAX_STUDENT_FRACTION
    budget = available_tokens * MAX_EPOCHS

    affordable = [
        t
        for t in TIERS
        if _student_params(t, teacher.vocab_size) <= ceiling
        # 5 tokens per parameter, against 20 for from-scratch training.
        and budget >= _student_params(t, teacher.vocab_size) * 5
    ]
    if affordable:
        return affordable[-1]

    # Nothing qualifies; fall back to the smallest tier that is at least smaller than
    # the teacher, and let the notes explain.
    smaller = [t for t in TIERS if _student_params(t, teacher.vocab_size) <= ceiling]
    return smaller[0] if smaller else TIERS[0]


def plan_distill(
    analysis: CorpusAnalysis,
    teacher: BaseModelInfo,
    *,
    hw: Hardware,
    available_tokens: int,
    tier_name: str | None = None,
    seq_len: int | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 1337,
) -> DistillPlan:
    """Build a plan for distilling `teacher` into a smaller model on this corpus."""
    if available_tokens <= 0:
        raise ValueError("corpus has no tokens")

    if tier_name:
        matches = [t for t in TIERS if t.name == tier_name]
        if not matches:
            raise ValueError(
                f"unknown tier '{tier_name}' — choose from {', '.join(t.name for t in TIERS)}"
            )
        tier = matches[0]
    else:
        tier = choose_student(teacher, available_tokens)

    # The teacher's context is a hard ceiling: it cannot score what it cannot read.
    seq_len = min(seq_len or tier.seq_len, teacher.max_position)
    d_ff = _ffn_dim(tier.d_model)

    from llmforge.pretrain.model import ModelConfig

    student = ModelConfig(
        vocab_size=teacher.vocab_size,
        n_layer=tier.n_layer,
        n_head=tier.n_head,
        n_kv_head=tier.n_kv_head,
        d_model=tier.d_model,
        d_ff=d_ff,
        max_seq_len=seq_len,
    )
    n_params = student.n_params()

    # --- token budget -----------------------------------------------------------
    # 5 tokens per parameter rather than 20: each one carries a distribution.
    ideal = n_params * 5
    total_tokens = int(min(ideal, available_tokens * MAX_EPOCHS))
    epochs = total_tokens / available_tokens

    # --- memory and batch shape -------------------------------------------------
    # The teacher is frozen and inference-only, so it costs weights plus activations
    # and no optimizer state. Quantize it if bf16 would crowd out the student.
    teacher_bf16_gb = teacher.n_params * 2 / 1e9
    budget = hw.memory_gb * memory.SAFE_UTILISATION
    load_4bit = teacher_bf16_gb > budget * 0.5
    teacher_gb = teacher.n_params * (0.55 if load_4bit else 2) / 1e9

    micro_batch, mem = memory.largest_micro_batch(
        n_params=n_params,
        n_layer=tier.n_layer,
        d_model=tier.d_model,
        d_ff=d_ff,
        vocab_size=teacher.vocab_size,
        seq_len=seq_len,
        # Reserve what the teacher will occupy before sizing the student's batch.
        total_memory_gb=max(1.0, hw.memory_gb - teacher_gb / memory.SAFE_UTILISATION),
    )

    # Shrink the batch so the run gets a workable number of optimizer steps. On a
    # small corpus the conventional batch — and sometimes a single micro-batch —
    # exceeds the entire dataset, which produces a one-step run and a model that
    # never moved. Memory allowed the larger batch; the data does not justify it.
    affordable_micro = max(1, total_tokens // (seq_len * MIN_TOTAL_STEPS))
    micro_batch = max(1, min(micro_batch, affordable_micro))

    tokens_per_micro = micro_batch * seq_len
    batch_tokens = min(tier.batch_tokens, max(tokens_per_micro, total_tokens // MIN_TOTAL_STEPS))
    grad_accum = max(1, round(batch_tokens / tokens_per_micro))
    tokens_per_step = tokens_per_micro * grad_accum

    total_steps = max(1, round(total_tokens / tokens_per_step))
    total_tokens = total_steps * tokens_per_step
    epochs = total_tokens / available_tokens

    # The teacher's forward pass is the dominant cost whenever it is much larger than
    # the student — which is the whole point of distilling. Estimating from the
    # student alone understates the run by orders of magnitude.
    teacher_flops = 2 * teacher.n_params * total_tokens
    student_flops = 6 * n_params * total_tokens
    achievable = hw.bf16_tflops * 1e12 * DISTILL_MFU * max(1, hw.n_gpus) * 0.9
    hours = (teacher_flops + student_flops) / achievable / 3600
    if load_4bit:
        hours *= QUANTIZED_TEACHER_PENALTY

    eval_every = max(1, total_steps // 20)

    plan = DistillPlan(
        teacher=teacher.ref,
        teacher_params=teacher.n_params,
        teacher_label=teacher.label,
        teacher_load_4bit=load_4bit,
        tier=tier.name,
        vocab_size=teacher.vocab_size,
        n_layer=tier.n_layer,
        n_head=tier.n_head,
        n_kv_head=tier.n_kv_head,
        d_model=tier.d_model,
        d_ff=d_ff,
        dropout=0.0 if epochs <= 1.2 else 0.1,
        n_params=n_params,
        temperature=temperature,
        alpha=alpha,
        seq_len=seq_len,
        micro_batch=micro_batch,
        grad_accum=grad_accum,
        tokens_per_step=tokens_per_step,
        total_steps=total_steps,
        total_tokens=total_tokens,
        epochs=round(epochs, 2),
        lr=tier.lr,
        min_lr=tier.lr / 10,
        warmup_steps=max(1, min(round(total_steps * 0.02), 500)),
        eval_every=eval_every,
        sample_every=eval_every,
        checkpoint_every=max(1, total_steps // 5),
        seed=seed,
        # The teacher's forward pass runs inside the step; compiling only the student
        # is fine, but the interaction is not worth the failure modes.
        compile=False,
        estimated_hours=round(hours, 2),
        estimated_memory_gb=round(mem.total_gb + teacher_gb, 1),
    )
    plan.notes = _notes(plan, teacher, analysis)
    return plan


def _notes(plan: DistillPlan, teacher: BaseModelInfo, analysis: CorpusAnalysis) -> list[str]:
    notes: list[str] = []

    ratio = plan.n_params / teacher.n_params
    notes.append(
        f"Distilling {teacher.label} into a {plan.n_params / 1e6:.0f}M student "
        f"({ratio:.1%} of the teacher). The student learns the teacher's full "
        f"probability distribution, not just the next token, so it needs far less "
        f"data than training it from scratch would."
    )

    notes.append(
        f"The student uses the teacher's tokenizer ({plan.vocab_size:,} tokens), "
        f"because comparing two distributions requires the same vocabulary. That "
        f"makes the embedding table {plan.vocab_size * plan.d_model / 1e6:.0f}M "
        f"parameters on its own."
    )

    teacher_share = (2 * teacher.n_params) / (2 * teacher.n_params + 6 * plan.n_params)
    if teacher_share > 0.8:
        notes.append(
            f"{teacher_share:.0%} of the compute in this run is the teacher's forward "
            f"pass, not the student's training — a {teacher.label} teacher is scored "
            f"over every token, every epoch. If that time is unacceptable, have the "
            f"teacher generate answers once and fine-tune the student on those instead."
        )

    embedding_share = plan.vocab_size * plan.d_model / plan.n_params
    if embedding_share > 0.5:
        notes.append(
            f"{embedding_share:.0%} of the student is its embedding table, because it "
            f"must share the teacher's {plan.vocab_size:,}-token vocabulary. Very "
            f"little of this model is doing the actual reasoning."
        )

    if ratio > 0.3:
        notes.append(
            "The student is close to the teacher's size, so the gain over simply "
            "using the teacher is small."
        )

    if plan.teacher_load_4bit:
        notes.append(
            "The teacher is being loaded in 4-bit to leave room for the student. Its "
            "targets will be slightly less precise, and each step slower."
        )

    if plan.total_tokens < plan.n_params * 2:
        notes.append(
            f"Only {plan.total_tokens / plan.n_params:.1f} tokens per student "
            f"parameter. Distillation is efficient but not magic — expect a weak model."
        )

    if analysis.kind == "instruction":
        notes.append(
            "This corpus is instruction data. Distillation here transfers the "
            "teacher's behaviour on these prompts; it is the teacher's responses, not "
            "the ones in your files, that the student is learning to reproduce."
        )

    return notes
