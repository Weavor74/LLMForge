"""Measured hardware capability, cached.

The planner needs to know what *this* machine can do, not what a spec sheet claims —
on the GB10 the two differ by a wide margin, and the gap is what decides whether a
plan takes four hours or forty.

Nothing here is specific to one machine. The profile is measured, fingerprinted, and
re-measured automatically when the hardware underneath it changes, so a workspace
copied to a bigger box plans against the bigger box rather than against its origin.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from llmforge.core import paths

# Used only when no GPU can be probed at all — for instance when planning on a laptop
# before moving the run elsewhere. Deliberately modest so estimates are not flattering.
DEFAULT_TFLOPS = 50.0
DEFAULT_BANDWIDTH_GBPS = 200.0
DEFAULT_MEMORY_GB = 24.0

# Share of memory a training process may claim. Unified memory is shared with the CPU
# and the desktop session, so it gets a wider margin than a dedicated card does.
UNIFIED_UTILISATION = 0.75
DISCRETE_UTILISATION = 0.90


@dataclass
class Hardware:
    gpu: str
    capability: str
    # Dense bf16 matmul throughput, measured. Theoretical peak is never reached.
    bf16_tflops: float
    bandwidth_gbps: float
    # Memory of a single device. Aggregate across devices is `total_memory_gb`.
    memory_gb: float
    compile_ok: bool
    flash_sdpa: bool

    n_gpus: int = 1
    # CPU and GPU sharing one pool, as on GB10 and Apple silicon. Changes both the
    # safe utilisation and what an out-of-memory event does to the rest of the system.
    unified_memory: bool = False
    flash_attn: bool = False
    gpus: list[str] = field(default_factory=list)

    @property
    def total_memory_gb(self) -> float:
        """Memory across every visible device.

        Only reachable by a run that shards across them; a single-device run is bound
        by `memory_gb` however many cards are present.
        """
        return self.memory_gb * self.n_gpus

    @property
    def utilisation(self) -> float:
        return UNIFIED_UTILISATION if self.unified_memory else DISCRETE_UTILISATION

    @property
    def total_tflops(self) -> float:
        return self.bf16_tflops * self.n_gpus

    @property
    def fingerprint(self) -> str:
        """Identifies the hardware a profile was measured on.

        A cached profile whose fingerprint no longer matches is measuring a machine
        that is no longer here.
        """
        return f"{self.gpu}x{self.n_gpus}@{self.capability}"

    def describe(self) -> str:
        if self.n_gpus > 1:
            return (
                f"{self.n_gpus}x {self.gpu} — {self.total_memory_gb:.0f} GB total, "
                f"{self.total_tflops:.0f} TFLOP/s"
            )
        return f"{self.gpu} — {self.memory_gb:.0f} GB, {self.bf16_tflops:.0f} TFLOP/s"

    def to_dict(self) -> dict:
        return asdict(self)


def attn_implementation() -> str:
    """Which attention kernel transformers should use.

    flash-attn is faster where it exists but has no wheel for some platforms, sm_121
    aarch64 among them. SDPA is the portable fallback and is itself fused.
    """
    return "flash_attention_2" if _flash_attn_available() else "sdpa"


def _profile_path():
    return paths.workspace() / "hardware.json"


def _detect_unified_memory() -> bool:
    """Whether the GPU shares system memory rather than having its own.

    Detected by comparing what CUDA reports against host RAM: on a unified system
    they are the same pool, so the two figures land within a few percent.
    """
    try:
        import os

        import torch

        if not torch.cuda.is_available():
            return False

        _, gpu_total = torch.cuda.mem_get_info()
        host_total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        return abs(gpu_total - host_total) / max(host_total, 1) < 0.15
    except Exception:
        return False


def _flash_attn_available() -> bool:
    """flash-attn is faster than SDPA where it exists, and has no wheel for some
    platforms (sm_121 aarch64 among them), so it is used only when present."""
    import importlib.util

    return importlib.util.find_spec("flash_attn") is not None


def measure() -> Hardware:
    """Run the probes. A few seconds of benchmarking."""
    from llmforge import doctor as doc

    checks = doc.run_checks(skip_compile=False)
    env = doc.collect_environment(checks)

    n_gpus = 1
    gpus: list[str] = []
    try:
        import torch

        if torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
            gpus = [torch.cuda.get_device_name(i) for i in range(n_gpus)]
    except Exception:
        pass

    return Hardware(
        gpu=env.get("gpu", "unknown"),
        capability=env.get("capability", "unknown"),
        bf16_tflops=env.get("bf16_tflops", DEFAULT_TFLOPS),
        bandwidth_gbps=env.get("bandwidth_gbps", DEFAULT_BANDWIDTH_GBPS),
        memory_gb=env.get("cuda_total_gb", DEFAULT_MEMORY_GB),
        compile_ok=env.get("compile", False),
        flash_sdpa="flash" in env.get("sdpa_backends", []),
        n_gpus=n_gpus,
        unified_memory=_detect_unified_memory(),
        flash_attn=_flash_attn_available(),
        gpus=gpus,
    )


def profile(refresh: bool = False) -> Hardware:
    """Cached hardware profile, re-measured when the machine has changed."""
    path = _profile_path()

    if not refresh and path.exists():
        try:
            cached = Hardware(**json.loads(path.read_text()))
        except (json.JSONDecodeError, TypeError):
            cached = None  # schema moved on, or the file was truncated

        if cached is not None and _matches_current_machine(cached):
            return cached

    hw = measure()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(hw.to_dict(), indent=2))
    return hw


def _matches_current_machine(cached: Hardware) -> bool:
    """Cheap check that a cached profile still describes the hardware present.

    Reading the device name costs microseconds; re-benchmarking costs seconds. This
    keeps the common case fast while making a moved workspace correct by default.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            # No GPU to contradict the cache; the fallbacks are already conservative.
            return True

        capability = torch.cuda.get_device_capability(0)
        current = (
            f"{torch.cuda.get_device_name(0)}x{torch.cuda.device_count()}"
            f"@{capability[0]}.{capability[1]}"
        )
        return current == cached.fingerprint
    except Exception:
        return True
