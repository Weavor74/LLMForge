"""Tests for planning, model shape, and the training scaffolding.

The training loop itself is exercised end-to-end by actually running it; what is
tested here is the arithmetic around it, where a wrong answer is plausible-looking
and expensive.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from llmforge.core import memory
from llmforge.core.config import CorpusAnalysis
from llmforge.core.hardware import Hardware
from llmforge.core.planner import (
    MAX_EPOCHS,
    TIERS,
    _ffn_dim,
    choose_tier,
    plan_pretrain,
)
from llmforge.pretrain.model import ModelConfig, Transformer
from llmforge.pretrain.train import lr_at

HW = Hardware(
    gpu="NVIDIA GB10",
    capability="12.1",
    bf16_tflops=70.0,
    bandwidth_gbps=210.0,
    memory_gb=128.0,
    compile_ok=True,
    flash_sdpa=True,
)


def analysis(tokens: int, kind: str = "raw") -> CorpusAnalysis:
    return CorpusAnalysis(
        root="/tmp/corpus",
        content_hash="deadbeef",
        kind=kind,
        n_documents=1000,
        n_chars=tokens * 4,
        est_tokens=tokens,
        exact_tokens=tokens,
    )


# --------------------------------------------------------------------------
# model shape
# --------------------------------------------------------------------------


@pytest.mark.parametrize("tier", TIERS, ids=lambda t: t.name)
def test_param_formula_matches_construction(tier):
    """The planner sizes models without building them, so the formula must be exact."""
    cfg = ModelConfig(
        vocab_size=8192,
        n_layer=tier.n_layer,
        n_head=tier.n_head,
        n_kv_head=tier.n_kv_head,
        d_model=tier.d_model,
        d_ff=_ffn_dim(tier.d_model),
        max_seq_len=tier.seq_len,
    )
    # On the meta device parameters carry shape but no storage, so the multi-billion
    # tiers can be counted without allocating tens of gigabytes.
    with torch.device("meta"):
        built = sum(p.numel() for p in Transformer(cfg).parameters())
    assert cfg.n_params() == built


def test_untied_embeddings_counted_separately():
    kwargs = dict(
        vocab_size=8192, n_layer=2, n_head=4, n_kv_head=2, d_model=128, d_ff=384, max_seq_len=128
    )
    tied = ModelConfig(**kwargs, tie_embeddings=True).n_params()
    untied = ModelConfig(**kwargs, tie_embeddings=False).n_params()
    assert untied - tied == 8192 * 128


def test_invalid_head_configuration_rejected():
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(
            vocab_size=100, n_layer=2, n_head=5, n_kv_head=2, d_model=128, d_ff=256, max_seq_len=64
        )


def test_ffn_dim_is_hardware_aligned():
    for d in (384, 512, 768, 1024, 2048):
        assert _ffn_dim(d) % 128 == 0
        assert _ffn_dim(d) >= 8 * d / 3


# --------------------------------------------------------------------------
# tier selection
# --------------------------------------------------------------------------


def test_tier_grows_with_corpus():
    """More data should never select a smaller model."""
    sizes = [
        TIERS.index(choose_tier(t, 32768))
        for t in (1_000_000, 100_000_000, 5_000_000_000, 200_000_000_000)
    ]
    assert sizes == sorted(sizes)


def test_tiny_corpus_gets_smallest_tier():
    assert choose_tier(500_000, 8192).name == "nano"


def test_huge_corpus_gets_largest_tier():
    assert choose_tier(500_000_000_000, 32768).name == TIERS[-1].name


def test_explicit_tier_is_honoured():
    plan = plan_pretrain(analysis(4_000_000), vocab_size=4096, hw=HW, tier_name="medium")
    assert plan.tier == "medium"


def test_unknown_tier_rejected():
    with pytest.raises(ValueError, match="unknown tier"):
        plan_pretrain(analysis(4_000_000), vocab_size=4096, hw=HW, tier_name="enormous")


# --------------------------------------------------------------------------
# plan arithmetic
# --------------------------------------------------------------------------


def test_batch_shape_is_self_consistent():
    plan = plan_pretrain(analysis(2_000_000_000), vocab_size=32768, hw=HW)
    assert plan.tokens_per_step == plan.micro_batch * plan.seq_len * plan.grad_accum
    assert plan.total_tokens == plan.total_steps * plan.tokens_per_step


def test_never_exceeds_the_epoch_ceiling():
    """A small corpus must not be silently looped a hundred times."""
    tokens = 3_000_000
    plan = plan_pretrain(analysis(tokens), vocab_size=4096, hw=HW)
    assert plan.epochs <= MAX_EPOCHS + 0.01


def test_large_corpus_stops_at_chinchilla():
    """With data to spare, training should stop at the compute-optimal point rather
    than consuming everything available."""
    plan = plan_pretrain(analysis(10_000_000_000_000), vocab_size=32768, hw=HW)
    assert plan.epochs < 1.0
    assert plan.total_tokens / plan.n_params == pytest.approx(20, rel=0.15)


def test_at_least_one_pass_over_the_corpus():
    plan = plan_pretrain(analysis(50_000), vocab_size=4096, hw=HW)
    assert plan.total_tokens >= 50_000


def test_warmup_is_a_small_slice_of_the_run():
    plan = plan_pretrain(analysis(5_000_000_000), vocab_size=32768, hw=HW)
    assert 0 < plan.warmup_steps <= max(1, plan.total_steps // 2)


def test_dropout_only_when_data_repeats():
    single_pass = plan_pretrain(analysis(10_000_000_000_000), vocab_size=32768, hw=HW)
    many_passes = plan_pretrain(analysis(2_000_000), vocab_size=4096, hw=HW)
    assert single_pass.dropout == 0.0
    assert many_passes.dropout > 0.0


def test_undertrained_corpus_is_flagged():
    plan = plan_pretrain(analysis(2_000_000), vocab_size=4096, hw=HW)
    assert any("tokens per parameter" in n for n in plan.notes)


def test_instruction_corpus_suggests_finetuning():
    plan = plan_pretrain(analysis(5_000_000, kind="instruction"), vocab_size=4096, hw=HW)
    assert any("fine-tuning" in n for n in plan.notes)


def test_missing_compiler_is_flagged():
    hw = Hardware(**{**HW.__dict__, "compile_ok": False})
    plan = plan_pretrain(analysis(5_000_000), vocab_size=4096, hw=hw)
    assert plan.compile is False
    assert any("torch.compile" in n for n in plan.notes)


def test_empty_corpus_rejected():
    with pytest.raises(ValueError, match="no tokens"):
        plan_pretrain(analysis(0), vocab_size=4096, hw=HW)


# --------------------------------------------------------------------------
# memory model
# --------------------------------------------------------------------------


def test_memory_grows_with_batch():
    def total(mb):
        return memory.estimate_training_memory(
            n_params=100_000_000, n_layer=12, d_model=768, d_ff=2048,
            vocab_size=32768, micro_batch=mb, seq_len=1024, total_memory_gb=128,
        ).total_gb

    assert total(1) < total(4) < total(16)


def test_micro_batch_shrinks_as_memory_shrinks():
    def pick(total_gb):
        return memory.largest_micro_batch(
            n_params=1_000_000_000, n_layer=24, d_model=2048, d_ff=5504,
            vocab_size=32768, seq_len=2048, total_memory_gb=total_gb,
        )[0]

    assert pick(8) <= pick(32) <= pick(128)


def test_micro_batch_never_zero():
    """Even an absurd model must yield a runnable batch size for the fit-check."""
    mb, _ = memory.largest_micro_batch(
        n_params=70_000_000_000, n_layer=80, d_model=8192, d_ff=28672,
        vocab_size=128000, seq_len=8192, total_memory_gb=1,
    )
    assert mb == 1


def test_budget_leaves_headroom():
    est = memory.estimate_training_memory(
        n_params=1_000_000, n_layer=2, d_model=128, d_ff=256,
        vocab_size=1024, micro_batch=1, seq_len=128, total_memory_gb=100,
    )
    assert est.budget_gb < 100


# --------------------------------------------------------------------------
# learning rate schedule
# --------------------------------------------------------------------------


def _plan_for_schedule():
    return plan_pretrain(analysis(5_000_000_000), vocab_size=32768, hw=HW)


def test_warmup_rises_to_peak():
    plan = _plan_for_schedule()
    assert lr_at(0, plan) < plan.lr
    assert lr_at(plan.warmup_steps - 1, plan) == pytest.approx(plan.lr)


def test_schedule_decays_to_minimum():
    plan = _plan_for_schedule()
    assert lr_at(plan.total_steps - 1, plan) == pytest.approx(plan.min_lr, rel=1e-3)


def test_schedule_is_monotonic_after_warmup():
    plan = _plan_for_schedule()
    steps = range(plan.warmup_steps, plan.total_steps, max(1, plan.total_steps // 50))
    values = [lr_at(s, plan) for s in steps]
    assert values == sorted(values, reverse=True)


def test_schedule_stays_in_range_after_warmup():
    """Warmup deliberately starts near zero; the floor only applies to the decay."""
    plan = _plan_for_schedule()
    for step in range(plan.warmup_steps, plan.total_steps, max(1, plan.total_steps // 100)):
        assert plan.min_lr * 0.99 <= lr_at(step, plan) <= plan.lr * 1.01


def test_warmup_starts_near_zero_and_never_overshoots():
    plan = _plan_for_schedule()
    assert 0 < lr_at(0, plan) < plan.lr * 0.5
    for step in range(plan.warmup_steps):
        assert lr_at(step, plan) <= plan.lr * 1.01


def test_schedule_clamps_past_the_end():
    """Overrunning the planned step count must not send the rate negative."""
    plan = _plan_for_schedule()
    assert lr_at(plan.total_steps * 3, plan) == pytest.approx(plan.min_lr, rel=1e-3)


# --------------------------------------------------------------------------
# token streaming
# --------------------------------------------------------------------------


class _FakePacked:
    """A PackedDataset stand-in backed by in-memory arrays."""

    def __init__(self, sizes: list[int]):
        self._shards = [
            np.arange(offset, offset + n, dtype=np.uint16)
            for offset, n in zip(np.cumsum([0] + sizes[:-1]), sizes, strict=True)
        ]

    def open_split(self, split: str):
        return self._shards


def test_batches_are_deterministic_for_a_given_step():
    import torch

    from llmforge.pretrain.data import TokenStream

    stream = TokenStream(_FakePacked([10_000]), "train", seq_len=128, seed=7)
    device = torch.device("cpu")
    a, _ = stream.batch(4, step=42, device=device)
    b, _ = stream.batch(4, step=42, device=device)
    assert torch.equal(a, b)


def test_different_steps_give_different_batches():
    import torch

    from llmforge.pretrain.data import TokenStream

    stream = TokenStream(_FakePacked([10_000]), "train", seq_len=128, seed=7)
    device = torch.device("cpu")
    a, _ = stream.batch(4, step=1, device=device)
    b, _ = stream.batch(4, step=2, device=device)
    assert not torch.equal(a, b)


def test_targets_are_inputs_shifted_by_one():
    import torch

    from llmforge.pretrain.data import TokenStream

    stream = TokenStream(_FakePacked([10_000]), "train", seq_len=64, seed=3)
    x, y = stream.batch(2, step=5, device=torch.device("cpu"))
    # Shards hold consecutive integers, so the shift is directly visible.
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_evaluation_batches_are_stable_across_calls():
    import torch

    from llmforge.pretrain.data import TokenStream

    stream = TokenStream(_FakePacked([10_000]), "val", seq_len=64, seed=3)
    device = torch.device("cpu")
    first = stream.deterministic_batches(2, 3, device)
    second = stream.deterministic_batches(2, 3, device)
    assert all(torch.equal(a[0], b[0]) for a, b in zip(first, second, strict=True))


def test_train_and_val_streams_do_not_mirror_each_other():
    """Both splits index from step 0; without a per-split salt they would draw
    identical offsets and validation would stop being independent."""
    import torch

    from llmforge.pretrain.data import TokenStream

    packed = _FakePacked([10_000])
    device = torch.device("cpu")
    train = TokenStream(packed, "train", seq_len=64, seed=1)
    val = TokenStream(packed, "val", seq_len=64, seed=1)
    a, _ = train.batch(4, step=0, device=device)
    b, _ = val.batch(4, step=0, device=device)
    assert not torch.equal(a, b)


def test_split_too_small_for_context_is_rejected():
    from llmforge.pretrain.data import TokenStream

    with pytest.raises(ValueError, match="too few"):
        TokenStream(_FakePacked([100]), "val", seq_len=1024)


def test_multiple_shards_are_all_reachable():
    import torch

    from llmforge.pretrain.data import TokenStream

    stream = TokenStream(_FakePacked([5_000, 5_000, 5_000]), "train", seq_len=64, seed=0)
    seen = set()
    for step in range(60):
        x, _ = stream.batch(8, step=step, device=torch.device("cpu"))
        # Shard identity is recoverable from the value range.
        seen.update((int(v) // 5_000) for v in x[:, 0])
    assert seen == {0, 1, 2}
