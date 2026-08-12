"""Run lockfiles.

A plan says what to train. A lockfile records everything else that determined the
result: which exact corpus, which tokenizer, which library versions, which GPU, which
seed. Without it "reproducible" means "probably similar".

Written once when a run starts, and never modified.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from llmforge import __version__

# Anything whose version can change a numerical result.
_TRACKED_PACKAGES = ("torch", "numpy", "tokenizers", "transformers", "peft", "trl", "bitsandbytes")


def _package_versions() -> dict[str, str]:
    out = {}
    for name in _TRACKED_PACKAGES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            continue
    return out


def _git_sha() -> str | None:
    """Source revision, when LLMForge itself is running from a checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() or None if result.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def build(
    *,
    run_id: str,
    mode: str,
    plan: dict,
    corpus_hash: str | None = None,
    corpus_root: str | None = None,
    tokenizer_id: str | None = None,
    base_model: str | None = None,
    hardware: dict | None = None,
) -> dict:
    """Assemble the lockfile contents."""
    return {
        "run_id": run_id,
        "mode": mode,
        "llmforge_version": __version__,
        "git_sha": _git_sha(),
        "inputs": {
            "corpus_hash": corpus_hash,
            "corpus_root": corpus_root,
            "tokenizer_id": tokenizer_id,
            "base_model": base_model,
        },
        "plan": plan,
        "environment": {
            "python": sys.version.split()[0],
            "platform": f"{platform.system()}-{platform.machine()}",
            "packages": _package_versions(),
            "hardware": hardware or {},
        },
    }


def write(run_dir: Path, lock: dict) -> Path:
    path = run_dir / "run.lock.json"
    path.write_text(json.dumps(lock, indent=2, sort_keys=True))
    return path


def read(run_dir: Path) -> dict:
    return json.loads((run_dir / "run.lock.json").read_text())


def diff(a: dict, b: dict) -> list[str]:
    """Human-readable differences between two lockfiles.

    Used by `repro` to explain why a rerun might not match bit-for-bit — a different
    torch version or a different GPU is a legitimate reason for divergence, and the
    user should be told rather than left guessing.
    """
    issues: list[str] = []

    if a["inputs"] != b["inputs"]:
        for key in a["inputs"]:
            if a["inputs"][key] != b["inputs"].get(key):
                issues.append(f"input {key}: {a['inputs'][key]} -> {b['inputs'].get(key)}")

    old_pkgs = a["environment"]["packages"]
    new_pkgs = b["environment"]["packages"]
    for name in sorted(set(old_pkgs) | set(new_pkgs)):
        if old_pkgs.get(name) != new_pkgs.get(name):
            issues.append(f"{name}: {old_pkgs.get(name, 'absent')} -> {new_pkgs.get(name, 'absent')}")

    old_gpu = a["environment"]["hardware"].get("gpu")
    new_gpu = b["environment"]["hardware"].get("gpu")
    if old_gpu != new_gpu:
        issues.append(f"gpu: {old_gpu} -> {new_gpu}")

    if a["plan"] != b["plan"]:
        issues.append("training plan differs")

    return issues
