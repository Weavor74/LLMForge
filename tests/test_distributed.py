"""Tests for multi-GPU behaviour.

Only world size one can actually be executed here, so these check two things that can
be verified on one GPU: that the single-device path is genuinely unchanged, and that
the arithmetic deciding *how* to spread a run is right. Real multi-GPU scaling is
untested — there is one GPU in this machine.
"""

from __future__ import annotations

import pytest

from llmforge.core import distributed as dist
from llmforge.core.config import CorpusAnalysis
from llmforge.core.hardware import Hardware
from llmforge.core.planner import plan_pretrain


def machine(n_gpus: int, memory_gb: float = 131.0, unified: bool = True) -> Hardware:
    return Hardware(
        gpu="TestGPU",
        capability="9.0",
        bf16_tflops=100.0,
        bandwidth_gbps=1000.0,
        memory_gb=memory_gb,
        compile_ok=True,
        flash_sdpa=True,
        n_gpus=n_gpus,
        unified_memory=unified,
    )


def corpus(tokens: int = 500_000_000_000) -> CorpusAnalysis:
    return CorpusAnalysis(
        root="/c", content_hash="h", kind="raw",
        n_documents=1_000_000, n_chars=tokens * 4,
        est_tokens=tokens, exact_tokens=tokens,
    )


# --------------------------------------------------------------------------
# process-group helpers
# --------------------------------------------------------------------------


def test_single_process_is_not_distributed(monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    assert not dist.is_distributed()
    assert dist.world_size() == 1
    assert dist.rank() == 0
    assert dist.is_main()


def test_environment_defines_the_topology(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("LOCAL_RANK", "2")

    assert dist.is_distributed()
    assert dist.world_size() == 4
    assert dist.rank() == 2
    assert not dist.is_main()


def test_only_rank_zero_owns_side_effects(monkeypatch):
    """Every rank trains; one writes. Otherwise ranks race over the same files."""
    monkeypatch.setenv("WORLD_SIZE", "8")
    for r in range(8):
        monkeypatch.setenv("RANK", str(r))
        assert dist.is_main() == (r == 0)


def test_all_reduce_is_identity_on_one_process(monkeypatch):
    import torch

    monkeypatch.delenv("WORLD_SIZE", raising=False)
    assert dist.all_reduce_mean(3.5, torch.device("cpu")) == 3.5


def test_wrapping_is_a_no_op_on_one_process(monkeypatch):
    import torch

    monkeypatch.delenv("WORLD_SIZE", raising=False)
    model = torch.nn.Linear(4, 4)
    assert dist.wrap_model(model, "ddp", torch.device("cpu")) is model


def test_unwrap_reaches_the_inner_module():
    import torch

    inner = torch.nn.Linear(4, 4)

    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.module = inner

    assert dist.unwrap(Wrapper()) is inner
    assert dist.unwrap(inner) is inner


def test_unknown_strategy_rejected(monkeypatch):
    import torch

    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(ValueError, match="unknown parallel strategy"):
        dist.wrap_model(torch.nn.Linear(4, 4), "pipeline", torch.device("cpu"))


# --------------------------------------------------------------------------
# strategy selection
# --------------------------------------------------------------------------


def test_one_gpu_stays_single():
    plan = plan_pretrain(corpus(), vocab_size=32768, hw=machine(1), tier_name="large")
    assert plan.strategy == "single"
    assert plan.n_gpus == 1


def test_several_gpus_replicate_when_the_model_fits():
    """DDP is preferred wherever a replica fits — parameters never have to move."""
    plan = plan_pretrain(corpus(), vocab_size=32768, hw=machine(8), tier_name="large")
    assert plan.strategy == "ddp"
    assert plan.n_gpus == 8


def test_a_model_too_large_for_one_device_is_sharded():
    """FSDP is the only way to train something bigger than a single GPU holds."""
    plan = plan_pretrain(
        corpus(), vocab_size=32768, hw=machine(8, memory_gb=24, unified=False),
        tier_name="xxl",
    )
    assert plan.strategy == "fsdp"


def test_batch_scales_with_device_count():
    """Each rank consumes its own micro-batch, so a step covers more tokens."""
    one = plan_pretrain(corpus(), vocab_size=32768, hw=machine(1), tier_name="medium")
    many = plan_pretrain(corpus(), vocab_size=32768, hw=machine(4), tier_name="medium")
    assert many.tokens_per_step >= one.tokens_per_step


def test_more_devices_means_less_time():
    one = plan_pretrain(corpus(), vocab_size=32768, hw=machine(1), tier_name="large")
    many = plan_pretrain(corpus(), vocab_size=32768, hw=machine(8), tier_name="large")
    assert many.estimated_hours < one.estimated_hours
    # Sublinear: collectives are not free, so 8 GPUs are not 8 times faster.
    assert many.estimated_hours > one.estimated_hours / 8


def test_multi_gpu_is_explained_in_the_notes():
    plan = plan_pretrain(corpus(), vocab_size=32768, hw=machine(4), tier_name="large")
    assert any("4 GPUs" in n for n in plan.notes)


# --------------------------------------------------------------------------
# hardware portability
# --------------------------------------------------------------------------


def test_unified_memory_gets_a_wider_safety_margin():
    """Unified memory is shared with the desktop, so a run may claim less of it."""
    assert machine(1, unified=True).utilisation < machine(1, unified=False).utilisation


def test_aggregate_capacity_counts_every_device():
    hw = machine(8, memory_gb=80)
    assert hw.total_memory_gb == 640
    assert hw.total_tflops == 800


def test_fingerprint_distinguishes_machines():
    assert machine(1).fingerprint != machine(8).fingerprint


def test_bigger_machine_permits_a_bigger_batch():
    small = plan_pretrain(
        corpus(), vocab_size=32768, hw=machine(1, memory_gb=16, unified=False),
        tier_name="medium",
    )
    large = plan_pretrain(
        corpus(), vocab_size=32768, hw=machine(1, memory_gb=640, unified=False),
        tier_name="medium",
    )
    assert large.micro_batch > small.micro_batch


def test_tight_memory_escalates_to_save_it():
    """On a small card the planner should reach for checkpointing rather than fail."""
    plan = plan_pretrain(
        corpus(), vocab_size=32768, hw=machine(1, memory_gb=24, unified=False),
        tier_name="xl",
    )
    assert plan.gradient_checkpointing or plan.optimizer == "adamw8bit"


def test_roomy_memory_pays_for_nothing_it_does_not_need():
    plan = plan_pretrain(
        corpus(), vocab_size=32768, hw=machine(1, memory_gb=640, unified=False),
        tier_name="small",
    )
    assert not plan.gradient_checkpointing
    assert plan.optimizer == "adamw"
