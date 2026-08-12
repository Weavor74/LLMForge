"""Resolving a base model.

The planner has to know how big a model is *before* deciding whether it can be
fine-tuned, and downloading 40 GB of weights only to discover it will not fit is not
an acceptable way to find out. So sizing reads metadata only: the safetensors index
if there is one, and the config as a fallback.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# Bytes per parameter for the dtype a checkpoint is stored in. Used to turn the
# safetensors index's total_size into a parameter count.
_DTYPE_BYTES = {
    "float32": 4, "float16": 2, "bfloat16": 2, "float64": 8, "int8": 1, "uint8": 1,
}


@dataclass
class BaseModelInfo:
    """What we know about a base model without having downloaded it."""

    ref: str
    is_local: bool
    n_params: int
    n_layer: int
    d_model: int
    vocab_size: int
    max_position: int
    architecture: str
    torch_dtype: str
    # Whether the tokenizer ships a chat template. Without one we supply ChatML.
    has_chat_template: bool = False

    @property
    def billions(self) -> float:
        return self.n_params / 1e9

    @property
    def label(self) -> str:
        if self.n_params >= 1e9:
            return f"{self.billions:.1f}B"
        return f"{self.n_params / 1e6:.0f}M"


def _read_json(ref: str, filename: str, local: bool) -> dict | None:
    """Fetch one small metadata file, from disk or the hub."""
    if local:
        path = Path(ref) / filename
        return json.loads(path.read_text()) if path.exists() else None

    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    try:
        return json.loads(Path(hf_hub_download(ref, filename)).read_text())
    except (EntryNotFoundError, OSError):
        return None


def _params_from_config(cfg: dict) -> int:
    """Parameter count derived from a Llama-family config.

    A fallback for repositories without a safetensors index. Approximate for exotic
    architectures, which is why the index is preferred.
    """
    hidden = cfg.get("hidden_size", 0)
    layers = cfg.get("num_hidden_layers", 0)
    heads = cfg.get("num_attention_heads", 1)
    kv_heads = cfg.get("num_key_value_heads", heads)
    ffn = cfg.get("intermediate_size", 4 * hidden)
    vocab = cfg.get("vocab_size", 0)
    head_dim = cfg.get("head_dim", hidden // max(heads, 1))

    attn = hidden * hidden + 2 * hidden * (kv_heads * head_dim) + hidden * hidden
    # Assume a gated FFN: three matrices. Ungated models are over-counted by a third
    # of their FFN, which is acceptable for a planning estimate.
    mlp = 3 * hidden * ffn
    norms = 2 * hidden

    embeddings = vocab * hidden * (1 if cfg.get("tie_word_embeddings", False) else 2)
    return layers * (attn + mlp + norms) + hidden + embeddings


def _params_from_index(ref: str, local: bool, dtype: str) -> int | None:
    """Parameter count from the safetensors shard index — exact, and metadata-only."""
    index = _read_json(ref, "model.safetensors.index.json", local)
    if not index:
        return None
    total_bytes = index.get("metadata", {}).get("total_size")
    if not total_bytes:
        return None
    return int(total_bytes / _DTYPE_BYTES.get(dtype, 2))


def resolve(ref: str) -> BaseModelInfo:
    """Look up a base model by hub id or local path, reading metadata only."""
    local = Path(ref).expanduser().is_dir()
    if local:
        ref = str(Path(ref).expanduser().resolve())

    cfg = _read_json(ref, "config.json", local)
    if cfg is None:
        hint = (
            "the directory has no config.json"
            if local
            else "it does not exist, or it is gated and needs `huggingface-cli login`"
        )
        raise ValueError(
            f"Could not read '{ref}' — {hint}. Give a Hugging Face model id "
            f"(for example Qwen/Qwen3-8B) or a local directory containing one."
        )

    # Some repositories nest the language model config under a multimodal wrapper.
    if "text_config" in cfg and "hidden_size" not in cfg:
        cfg = {**cfg, **cfg["text_config"]}

    dtype = cfg.get("torch_dtype") or cfg.get("dtype") or "bfloat16"
    if not isinstance(dtype, str):
        dtype = "bfloat16"

    n_params = _params_from_index(ref, local, dtype) or _params_from_config(cfg)
    if n_params <= 0:
        raise ValueError(f"Could not determine the size of '{ref}'.")

    tok_cfg = _read_json(ref, "tokenizer_config.json", local) or {}

    architectures = cfg.get("architectures") or ["unknown"]
    return BaseModelInfo(
        ref=ref,
        is_local=local,
        n_params=n_params,
        n_layer=cfg.get("num_hidden_layers", 0),
        d_model=cfg.get("hidden_size", 0),
        vocab_size=cfg.get("vocab_size", 0),
        max_position=cfg.get("max_position_embeddings", 2048),
        architecture=architectures[0],
        torch_dtype=dtype,
        has_chat_template=bool(tok_cfg.get("chat_template")),
    )
