"""Tests for distillation.

The loss is where the subtle mistakes live. A distillation objective that is merely
*plausible* trains without complaint and produces a student no better than one trained
on hard labels alone, so the properties that make it work are asserted directly.
"""

from __future__ import annotations

import pytest
import torch

from llmforge.core.config import CorpusAnalysis
from llmforge.core.hardware import Hardware
from llmforge.distill.plan import (
    DEFAULT_ALPHA,
    DEFAULT_TEMPERATURE,
    MAX_STUDENT_FRACTION,
    choose_student,
    plan_distill,
)
from llmforge.distill.train import distillation_loss
from llmforge.finetune.base import BaseModelInfo

HW = Hardware(
    gpu="NVIDIA GB10",
    capability="12.1",
    bf16_tflops=70.0,
    bandwidth_gbps=210.0,
    memory_gb=128.0,
    compile_ok=True,
    flash_sdpa=True,
)


def teacher(n_params: int, *, vocab: int = 32000, n_layer: int = 32, d_model: int = 4096):
    return BaseModelInfo(
        ref="org/teacher",
        is_local=False,
        n_params=n_params,
        n_layer=n_layer,
        d_model=d_model,
        vocab_size=vocab,
        max_position=4096,
        architecture="LlamaForCausalLM",
        torch_dtype="bfloat16",
        has_chat_template=True,
    )


def analysis(kind: str = "raw") -> CorpusAnalysis:
    return CorpusAnalysis(
        root="/tmp/c", content_hash="abc", kind=kind, n_documents=1000, n_chars=4_000_000
    )


def make_plan(n_params=8_000_000_000, tokens=2_000_000_000, **kw):
    return plan_distill(
        analysis(kw.pop("kind", "raw")),
        teacher(n_params, **{k: kw.pop(k) for k in ("vocab", "n_layer", "d_model") if k in kw}),
        hw=kw.pop("hw", HW),
        available_tokens=tokens,
        **kw,
    )


# --------------------------------------------------------------------------
# the objective
# --------------------------------------------------------------------------


def _logits(values: list[float]) -> torch.Tensor:
    """One batch, one position, `len(values)` vocabulary entries."""
    return torch.tensor([[values]], dtype=torch.float32)


def test_matching_the_teacher_drives_the_kl_term_to_zero():
    logits = _logits([2.0, 1.0, 0.1, -1.0])
    targets = torch.tensor([[0]])
    _, kl, _ = distillation_loss(logits, logits.clone(), targets, temperature=2.0, alpha=1.0)
    assert kl.item() == pytest.approx(0.0, abs=1e-5)


def test_disagreeing_with_the_teacher_costs_more_than_agreeing():
    teacher_logits = _logits([3.0, 0.0, 0.0, 0.0])
    close = _logits([2.5, 0.1, 0.0, 0.0])
    far = _logits([0.0, 0.0, 0.0, 3.0])
    targets = torch.tensor([[0]])

    _, kl_close, _ = distillation_loss(close, teacher_logits, targets, temperature=2.0, alpha=1.0)
    _, kl_far, _ = distillation_loss(far, teacher_logits, targets, temperature=2.0, alpha=1.0)
    assert kl_far > kl_close


def test_temperature_squared_keeps_the_gradient_scale_stable():
    """Softening by T shrinks gradients by ~1/T^2; the T^2 factor undoes that.

    Without it the distillation term quietly stops mattering as temperature rises,
    which looks like "distillation does not help" rather than like a bug.
    """
    teacher_logits = _logits([3.0, 1.0, 0.0, -1.0])
    targets = torch.tensor([[0]])

    def grad_norm(temperature: float) -> float:
        student = _logits([1.0, 0.5, 0.0, 0.0]).requires_grad_(True)
        # alpha=1 makes the total the KL term alone. The returned kl is detached —
        # it is a diagnostic — so the gradient has to come from the loss itself.
        total, _, _ = distillation_loss(
            student, teacher_logits, targets, temperature=temperature, alpha=1.0
        )
        total.backward()
        return student.grad.norm().item()

    low, high = grad_norm(1.0), grad_norm(4.0)
    # Same order of magnitude. Without the T^2 factor this ratio would be ~16x.
    assert 0.2 < high / low < 5.0


def test_alpha_selects_between_the_two_signals():
    student = _logits([0.1, 0.2, 3.0, 0.0])
    teacher_logits = _logits([3.0, 0.0, 0.0, 0.0])
    targets = torch.tensor([[2]])  # the student is right, the teacher disagrees

    all_teacher, kl, ce = distillation_loss(
        student, teacher_logits, targets, temperature=2.0, alpha=1.0
    )
    all_labels, _, _ = distillation_loss(
        student, teacher_logits, targets, temperature=2.0, alpha=0.0
    )

    assert all_teacher.item() == pytest.approx(kl.item(), rel=1e-5)
    assert all_labels.item() == pytest.approx(ce.item(), rel=1e-5)


def test_loss_is_a_weighted_blend():
    student = _logits([0.5, 1.5, 0.0, 0.2])
    teacher_logits = _logits([2.0, 0.5, 0.0, 0.0])
    targets = torch.tensor([[1]])

    total, kl, ce = distillation_loss(
        student, teacher_logits, targets, temperature=2.0, alpha=0.7
    )
    assert total.item() == pytest.approx(0.7 * kl.item() + 0.3 * ce.item(), rel=1e-5)


def test_kl_does_not_shrink_with_vocabulary_size():
    """`batchmean` averages over positions, not over vocabulary entries.

    With reduction='mean' the term would be divided by the vocabulary size too, making
    it vanishingly small for a 150k-token model — silently disabling distillation.
    """
    targets = torch.tensor([[0]])

    def kl_for(vocab: int) -> float:
        torch.manual_seed(0)
        student = torch.randn(1, 1, vocab)
        teacher_logits = torch.randn(1, 1, vocab)
        _, kl, _ = distillation_loss(
            student, teacher_logits, targets, temperature=2.0, alpha=1.0
        )
        return kl.item()

    small, large = kl_for(128), kl_for(32768)
    # Grows with vocabulary, as a sum over more entries should — not collapses.
    assert large > small


# --------------------------------------------------------------------------
# student sizing
# --------------------------------------------------------------------------


def test_student_is_always_smaller_than_its_teacher():
    for size in (1e9, 8e9, 70e9):
        info = teacher(int(size))
        tier = choose_student(info, 10_000_000_000)
        plan = plan_distill(
            analysis(), info, hw=HW, available_tokens=10_000_000_000, tier_name=tier.name
        )
        assert plan.n_params <= info.n_params * MAX_STUDENT_FRACTION


def test_bigger_teacher_permits_a_bigger_student():
    small = make_plan(1_000_000_000, tokens=50_000_000_000)
    large = make_plan(70_000_000_000, tokens=50_000_000_000)
    assert large.n_params >= small.n_params


def test_student_inherits_the_teacher_vocabulary():
    plan = make_plan(vocab=151936)
    assert plan.vocab_size == 151936


def test_context_cannot_exceed_the_teacher():
    info = teacher(8_000_000_000)
    info.max_position = 512
    plan = plan_distill(analysis(), info, hw=HW, available_tokens=1_000_000_000)
    assert plan.seq_len <= 512


def test_large_teacher_is_quantized_to_make_room():
    assert make_plan(70_000_000_000).teacher_load_4bit is True
    assert make_plan(1_000_000_000).teacher_load_4bit is False


def test_quantized_teacher_is_flagged():
    plan = make_plan(70_000_000_000)
    assert any("4-bit" in n for n in plan.notes)


def test_memory_estimate_includes_the_teacher():
    """The teacher sits in memory for the whole run; omitting it would under-report."""
    plan = make_plan(8_000_000_000)
    assert plan.estimated_memory_gb > 8_000_000_000 * 2 / 1e9


def test_defaults_are_applied():
    plan = make_plan()
    assert plan.temperature == DEFAULT_TEMPERATURE
    assert plan.alpha == DEFAULT_ALPHA


def test_explicit_tier_is_honoured():
    assert make_plan(70_000_000_000, tier_name="micro").tier == "micro"


def test_unknown_tier_rejected():
    with pytest.raises(ValueError, match="unknown tier"):
        make_plan(tier_name="gigantic")


def test_empty_corpus_rejected():
    with pytest.raises(ValueError, match="no tokens"):
        make_plan(tokens=0)


def test_epoch_ceiling_respected():
    plan = make_plan(tokens=1_000_000)
    assert plan.epochs <= 4.01


def test_plan_arithmetic_is_self_consistent():
    plan = make_plan()
    assert plan.tokens_per_step == plan.micro_batch * plan.seq_len * plan.grad_accum
    assert plan.total_tokens == plan.total_steps * plan.tokens_per_step


def test_distillation_does_not_compile():
    """The teacher forward runs inside the step; compiling around it is not worth
    the failure modes."""
    assert make_plan().compile is False


def test_notes_explain_the_shared_vocabulary():
    plan = make_plan()
    assert any("tokenizer" in n for n in plan.notes)


def test_instruction_corpus_gets_a_caveat():
    plan = make_plan(kind="instruction")
    assert any("teacher's responses" in n or "teacher's behaviour" in n for n in plan.notes)


def test_small_corpus_still_gets_a_workable_number_of_steps():
    """A batch larger than the whole corpus yields one optimizer step and a model
    that never moved. Same defect class as the fine-tuning planner had."""
    plan = make_plan(8_000_000_000, tokens=200_000)
    assert plan.total_steps >= 5, f"only {plan.total_steps} steps"


def test_large_corpus_keeps_the_conventional_batch():
    plan = make_plan(8_000_000_000, tokens=500_000_000_000)
    assert plan.tokens_per_step >= 32_768
