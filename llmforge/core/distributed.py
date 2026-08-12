"""Running across several GPUs.

Two strategies, chosen by whether the model fits on one device:

- **DDP** replicates the model per GPU and all-reduces gradients. Near-linear speedup,
  but every device holds a full copy of parameters, gradients and optimizer state.
- **FSDP** shards all three across devices, so N GPUs hold roughly 1/N each. Slower
  per step because parameters are gathered on demand, and the only way to train a
  model larger than one device.

Everything here is a no-op at world size one, which is the configuration this was
developed against. A single-GPU run takes exactly the path it took before.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

import torch

# torchrun sets these; their absence means a plain single-process run.
_RANK = "RANK"
_LOCAL_RANK = "LOCAL_RANK"
_WORLD_SIZE = "WORLD_SIZE"


def is_distributed() -> bool:
    return int(os.environ.get(_WORLD_SIZE, "1")) > 1


def world_size() -> int:
    return int(os.environ.get(_WORLD_SIZE, "1"))


def rank() -> int:
    return int(os.environ.get(_RANK, "0"))


def local_rank() -> int:
    return int(os.environ.get(_LOCAL_RANK, "0"))


def is_main() -> bool:
    """Whether this process owns side effects.

    Every rank trains; only one writes checkpoints, metrics and registry updates.
    Without this the ranks race each other and corrupt what they write.
    """
    return rank() == 0


def setup() -> torch.device:
    """Join the process group and claim this rank's device."""
    if not is_distributed():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(f"cuda:{local_rank()}")
    torch.cuda.set_device(device)

    if not torch.distributed.is_initialized():
        # NCCL for GPU collectives; gloo only where there is no CUDA.
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend)

    return device


def cleanup() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def all_reduce_mean(value: float, device: torch.device) -> float:
    """Average a scalar across ranks, so reported loss reflects the whole batch."""
    if not is_distributed():
        return value
    tensor = torch.tensor([value], device=device, dtype=torch.float32)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return float(tensor.item() / world_size())


@contextmanager
def main_first():
    """Let rank 0 run a block before the others.

    For work that is safe to repeat but wasteful to do concurrently — populating a
    cache, downloading a base model — where the ranks would otherwise all fetch the
    same thing at once.
    """
    if is_distributed() and not is_main():
        barrier()
    try:
        yield
    finally:
        if is_distributed() and is_main():
            barrier()


def wrap_model(model: torch.nn.Module, strategy: str, device: torch.device):
    """Apply the chosen parallelism. Returns the module to call in the training step."""
    if not is_distributed() or strategy == "single":
        return model

    if strategy == "ddp":
        from torch.nn.parallel import DistributedDataParallel

        return DistributedDataParallel(
            model,
            device_ids=[local_rank()],
            # Our forward uses every parameter every step, so the extra traversal
            # that finds unused ones would be pure overhead.
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )

    if strategy == "fsdp":
        from torch.distributed.fsdp import FullyShardedDataParallel, MixedPrecision
        from torch.distributed.fsdp.wrap import ModuleWrapPolicy

        from llmforge.pretrain.model import Block

        return FullyShardedDataParallel(
            model,
            # Shard at block granularity: the unit small enough to gather cheaply and
            # large enough that the gathering is worth it.
            auto_wrap_policy=ModuleWrapPolicy({Block}),
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,  # gradient reduction in fp32 for stability
                buffer_dtype=torch.bfloat16,
            ),
            device_id=local_rank(),
            use_orig_params=True,  # required for torch.compile and for param groups
        )

    raise ValueError(f"unknown parallel strategy '{strategy}'")


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """The underlying module, for saving weights without a wrapper's key prefixes."""
    return getattr(model, "module", model)


def gather_state_dict(model: torch.nn.Module, strategy: str) -> dict:
    """A full, unsharded state dict on rank 0.

    Under FSDP each rank holds a slice; saving that directly would produce a
    checkpoint that only reloads at the same world size.
    """
    if strategy != "fsdp" or not is_distributed():
        return unwrap(model).state_dict()

    from torch.distributed.fsdp import FullStateDictConfig, FullyShardedDataParallel, StateDictType

    policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FullyShardedDataParallel.state_dict_type(
        model, StateDictType.FULL_STATE_DICT, policy
    ):
        return model.state_dict()
