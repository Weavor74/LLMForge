"""Preflight checks.

aarch64 + CUDA 13 + sm_121 is a young combination: wheels install happily and then
fail at the first fused kernel. Everything LLMForge relies on at training time gets
exercised here, cheaply, before a run burns hours discovering the same thing.

The bandwidth and TFLOPS probes are not vanity numbers — the planner uses measured
throughput to produce honest wall-clock estimates.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

Status = Literal["pass", "warn", "fail", "skip"]

# GB10 Grace-Blackwell. Kernel coverage for this target is still patchy upstream,
# which is why several of the checks below exist at all. It is not a requirement —
# LLMForge runs anywhere CUDA does, and the planner adapts to what it measures.
GB10_CAPABILITY = (12, 1)

# Below this, bf16 tensor cores either do not exist or are not worth using.
MIN_CAPABILITY = (8, 0)


@dataclass
class Check:
    name: str
    status: Status
    detail: str
    # Free-form measurements worth persisting into a run lockfile.
    data: dict = field(default_factory=dict)


def _fail(name: str, detail: str) -> Check:
    return Check(name, "fail", detail)


# --------------------------------------------------------------------------
# host
# --------------------------------------------------------------------------


def check_python() -> Check:
    v = sys.version_info
    detail = f"{v.major}.{v.minor}.{v.micro} ({platform.machine()})"
    if (v.major, v.minor) != (3, 12):
        return Check(
            "python",
            "warn",
            f"{detail} — torch cu130 aarch64 wheels are built for cp312",
        )
    return Check("python", "pass", detail, {"python": f"{v.major}.{v.minor}.{v.micro}"})


def check_disk() -> Check:
    from llmforge.core import paths

    root = paths.workspace()
    # Walk up to the nearest existing ancestor; the workspace may not exist yet.
    probe = root
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent

    usage = shutil.disk_usage(probe)
    free_gb = usage.free / 1e9
    detail = f"{free_gb:.0f} GB free at {root}"
    data = {"disk_free_gb": round(free_gb, 1)}

    # Packed shards plus a few checkpoints of a mid-size model run well into
    # the tens of GB; below 50 GB a real run will die partway through.
    if free_gb < 50:
        return Check("disk", "fail", f"{detail} — need at least 50 GB", data)
    if free_gb < 200:
        return Check("disk", "warn", f"{detail} — tight for multi-checkpoint runs", data)
    return Check("disk", "pass", detail, data)


def check_host_memory() -> Check:
    """On GB10 the CPU and GPU share one pool, so host RAM *is* the training budget."""
    try:
        page = os.sysconf("SC_PAGE_SIZE")
        total_gb = page * os.sysconf("SC_PHYS_PAGES") / 1e9
        avail_gb = page * os.sysconf("SC_AVPHYS_PAGES") / 1e9
    except (ValueError, OSError) as e:
        return Check("unified memory", "warn", f"could not read: {e}")

    detail = f"{total_gb:.0f} GB total, {avail_gb:.0f} GB available (unified CPU+GPU)"
    data = {"unified_total_gb": round(total_gb, 1), "unified_available_gb": round(avail_gb, 1)}
    if avail_gb < 16:
        return Check("unified memory", "warn", f"{detail} — little headroom", data)
    return Check("unified memory", "pass", detail, data)


# --------------------------------------------------------------------------
# torch + gpu
# --------------------------------------------------------------------------


def check_torch() -> Check:
    try:
        import torch
    except ImportError as e:
        return _fail("torch", f"not installed: {e}")

    built = torch.version.cuda or "cpu-only build"
    detail = f"{torch.__version__} (CUDA {built})"
    data = {"torch": torch.__version__, "torch_cuda": built}

    if not torch.version.cuda:
        return _fail("torch", f"{detail} — a CPU-only wheel got installed")
    if not built.startswith("13"):
        return Check("torch", "warn", f"{detail} — expected a cu130 build for sm_121", data)
    return Check("torch", "pass", detail, data)


def check_cuda() -> Check:
    import torch

    if not torch.cuda.is_available():
        return _fail("cuda", "torch.cuda.is_available() is False — driver or wheel mismatch")

    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    count = torch.cuda.device_count()

    detail = f"{name}, sm_{cap[0]}{cap[1]}"
    if count > 1:
        detail = f"{count}x {detail}"

    data = {
        "gpu": name,
        "capability": f"{cap[0]}.{cap[1]}",
        "n_gpus": count,
        "gpus": [torch.cuda.get_device_name(i) for i in range(count)],
    }

    if cap < MIN_CAPABILITY:
        return Check(
            "cuda",
            "warn",
            f"{detail} — pre-Ampere, so bf16 training will be slow or unsupported",
            data,
        )
    return Check("cuda", "pass", detail, data)


def check_gpu_memory() -> Check:
    import torch

    free, total = torch.cuda.mem_get_info()
    count = torch.cuda.device_count()

    detail = f"{free / 1e9:.0f} GB free of {total / 1e9:.0f} GB per device"
    if count > 1:
        detail += f", {total * count / 1e9:.0f} GB across {count} devices"

    return Check(
        "gpu memory",
        "pass",
        detail,
        {
            "cuda_free_gb": round(free / 1e9, 1),
            "cuda_total_gb": round(total / 1e9, 1),
        },
    )


def check_bf16_matmul() -> Check:
    """Every training path here runs in bf16. Prove the tensor cores produce real numbers."""
    import torch

    n = 4096
    try:
        a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)

        for _ in range(3):  # warm up kernel selection and clocks
            c = a @ b
        torch.cuda.synchronize()

        iters = 20
        t0 = time.perf_counter()
        for _ in range(iters):
            c = a @ b
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    except RuntimeError as e:
        return _fail("bf16 matmul", f"kernel failed: {e}")

    if not torch.isfinite(c.float()).all():
        return _fail("bf16 matmul", "produced non-finite values")

    tflops = (2 * n**3 * iters) / elapsed / 1e12
    return Check(
        "bf16 matmul", "pass", f"{tflops:.0f} TFLOP/s dense", {"bf16_tflops": round(tflops, 1)}
    )


def check_memory_bandwidth() -> Check:
    """The GB10's defining constraint. Measured, because every time estimate depends on it."""
    import torch

    try:
        n = 512 * 1024 * 1024 // 2  # 512 MB of bf16
        src = torch.empty(n, device="cuda", dtype=torch.bfloat16)
        dst = torch.empty_like(src)

        for _ in range(2):
            dst.copy_(src)
        torch.cuda.synchronize()

        iters = 20
        t0 = time.perf_counter()
        for _ in range(iters):
            dst.copy_(src)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    except RuntimeError as e:
        return Check("memory bandwidth", "warn", f"probe failed: {e}")

    moved_bytes = src.numel() * src.element_size() * 2 * iters  # read + write
    gbps = moved_bytes / elapsed / 1e9
    data = {"bandwidth_gbps": round(gbps, 1)}

    # Spec is 273 GB/s aggregate; a healthy copy lands well above half of that.
    if gbps < 100:
        return Check("memory bandwidth", "warn", f"{gbps:.0f} GB/s — below expected", data)
    return Check("memory bandwidth", "pass", f"{gbps:.0f} GB/s", data)


def check_sdpa() -> Check:
    """We use SDPA instead of flash-attn, whose wheels don't build on sm_121 aarch64.

    Worth verifying a fused backend actually engages: the math fallback works but
    materialises the full attention matrix, which is unaffordable at long context.
    """
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    b, h, s, d = 2, 8, 1024, 64
    q, k, v = (torch.randn(b, h, s, d, device="cuda", dtype=torch.bfloat16) for _ in range(3))

    working = []
    for backend, label in (
        (SDPBackend.FLASH_ATTENTION, "flash"),
        (SDPBackend.EFFICIENT_ATTENTION, "mem-efficient"),
        (SDPBackend.MATH, "math"),
    ):
        try:
            with sdpa_kernel(backend):
                out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
            if torch.isfinite(out.float()).all():
                working.append(label)
        except RuntimeError:
            continue

    data = {"sdpa_backends": working}
    if not working:
        return _fail("sdpa attention", "no working backend — training is impossible")
    if working == ["math"]:
        return Check(
            "sdpa attention",
            "warn",
            "only the math fallback works — long context will be memory-hungry",
            data,
        )
    return Check("sdpa attention", "pass", ", ".join(working), data)


def check_torch_compile() -> Check:
    """Compilation is worth 20-40% here, but needs a working host compiler toolchain."""
    import torch

    def fn(x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(x) * 2.0 + x

    try:
        compiled = torch.compile(fn, mode="default")
        x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
        t0 = time.perf_counter()
        out = compiled(x)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    except Exception as e:  # inductor raises a wide variety of backend errors
        return Check(
            "torch.compile",
            "warn",
            f"unavailable, will train eagerly: {type(e).__name__}: {str(e)[:120]}",
            {"compile": False},
        )

    if not torch.isfinite(out.float()).all():
        return Check("torch.compile", "warn", "produced non-finite values", {"compile": False})
    return Check("torch.compile", "pass", f"cold compile {elapsed:.1f}s", {"compile": True})


# --------------------------------------------------------------------------
# optional deps
# --------------------------------------------------------------------------


def check_finetune_extras() -> Check:
    """Phase 3 only. Absence is fine; it just means the fine-tune path isn't installed."""
    required = {"peft": "peft", "trl": "trl", "datasets": "datasets", "accelerate": "accelerate"}
    missing = [n for n, mod in required.items() if importlib.util.find_spec(mod) is None]

    if missing:
        return Check(
            "finetune extras",
            "skip",
            f"not installed ({', '.join(missing)}) — `uv sync --extra finetune` to enable",
        )

    # bitsandbytes is the aarch64-fragile one; only QLoRA (34B+ bases) needs it.
    if importlib.util.find_spec("bitsandbytes") is None:
        return Check("finetune extras", "warn", "present, but no bitsandbytes — QLoRA unavailable")
    return Check("finetune extras", "pass", "peft, trl, datasets, accelerate, bitsandbytes")


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

# Ordered. GPU probes are skipped wholesale if torch or CUDA is broken, since
# every one of them would raise the same underlying error.
HOST_CHECKS: list[Callable[[], Check]] = [
    check_python,
    check_disk,
    check_host_memory,
]

GPU_CHECKS: list[Callable[[], Check]] = [
    check_cuda,
    check_gpu_memory,
    check_bf16_matmul,
    check_memory_bandwidth,
    check_sdpa,
    check_torch_compile,
]


def run_checks(skip_compile: bool = False) -> list[Check]:
    """Run every check, degrading gracefully when a prerequisite fails."""
    results = [c() for c in HOST_CHECKS]

    torch_check = check_torch()
    results.append(torch_check)

    if torch_check.status == "fail":
        results.append(Check("gpu probes", "skip", "torch unusable"))
    else:
        import torch

        if not torch.cuda.is_available():
            results.append(_fail("cuda", "torch.cuda.is_available() is False"))
            results.append(Check("gpu probes", "skip", "no CUDA device"))
        else:
            for check in GPU_CHECKS:
                if skip_compile and check is check_torch_compile:
                    results.append(Check("torch.compile", "skip", "--skip-compile"))
                    continue
                results.append(check())

    results.append(check_finetune_extras())
    return results


def collect_environment(checks: list[Check]) -> dict:
    """Flatten measurements into the dict recorded in every run.lock.json."""
    env: dict = {}
    for c in checks:
        env.update(c.data)
    return env
