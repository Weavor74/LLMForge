"""Export formats and quantization levels.

A model that only runs inside LLMForge is not much use. These are the two things
worth exporting to: HuggingFace safetensors, which anything Python can load, and
GGUF, which llama.cpp and Ollama can run anywhere without a GPU.

Only levels that can actually be produced are offered. The `gguf` package quantizes
the legacy formats in pure Python; llama.cpp's "k-quant" schemes (q4_k_m and friends)
live in its C code, so those appear only when `llama-quantize` is on PATH.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from typing import Literal

Format = Literal["safetensors", "gguf"]


@dataclass(frozen=True)
class QuantLevel:
    name: str
    format: Format
    # Bits per weight, for a size estimate before committing to the conversion.
    bits: float
    summary: str
    # k-quants are produced by shelling out to llama.cpp.
    needs_llama_cpp: bool = False

    @property
    def available(self) -> bool:
        return not self.needs_llama_cpp or llama_quantize_path() is not None


def llama_quantize_path() -> str | None:
    """llama.cpp's quantizer, if the user has it installed."""
    return shutil.which("llama-quantize") or shutil.which("quantize")


# Ordered from highest fidelity to smallest within each format.
QUANT_LEVELS: list[QuantLevel] = [
    QuantLevel("bf16", "safetensors", 16, "Native training precision. The faithful choice."),
    QuantLevel("f16", "safetensors", 16, "Half precision, for tools that predate bf16."),
    QuantLevel("f32", "safetensors", 32, "Full precision. Twice the size, no quality gain."),
    # --- gguf, pure Python ---
    QuantLevel("f16", "gguf", 16, "No quantization. Largest, and exactly what you trained."),
    QuantLevel("q8_0", "gguf", 8.5, "8-bit. Effectively lossless at half the size."),
    QuantLevel("q5_1", "gguf", 6.0, "5-bit. Small quality loss."),
    QuantLevel("q5_0", "gguf", 5.5, "5-bit, smaller variant."),
    QuantLevel("q4_1", "gguf", 5.0, "4-bit. Noticeable loss; small files."),
    QuantLevel("q4_0", "gguf", 4.5, "4-bit, smallest of the legacy schemes."),
    # --- gguf, needs llama.cpp ---
    QuantLevel(
        "q6_k", "gguf", 6.6, "6-bit mixed. Better than q5 at similar size.", needs_llama_cpp=True
    ),
    QuantLevel(
        "q5_k_m", "gguf", 5.7, "5-bit mixed. Beats q5_0 for the same bytes.", needs_llama_cpp=True
    ),
    QuantLevel(
        "q4_k_m",
        "gguf",
        4.8,
        "4-bit mixed. The usual default elsewhere — best size-for-quality.",
        needs_llama_cpp=True,
    ),
]


def levels_for(fmt: Format, only_available: bool = False) -> list[QuantLevel]:
    levels = [q for q in QUANT_LEVELS if q.format == fmt]
    return [q for q in levels if q.available] if only_available else levels


def find_level(fmt: Format, name: str) -> QuantLevel:
    for level in levels_for(fmt):
        if level.name == name:
            return level
    available = ", ".join(q.name for q in levels_for(fmt, only_available=True))
    raise ValueError(f"unknown quantization '{name}' for {fmt} — choose from {available}")


def default_level(fmt: Format) -> str:
    # q8_0 rather than a 4-bit scheme: it needs no external tooling, is close to
    # lossless, and the models this builds are small enough that size rarely binds.
    return "bf16" if fmt == "safetensors" else "q8_0"


def estimate_bytes(n_params: int, level: QuantLevel) -> int:
    return int(n_params * level.bits / 8)
