"""Writing a trained run out as a portable model.

Handles all three kinds of run. From-scratch and distilled models are converted from
our layout into Llama's; fine-tunes are already in it, and only need their adapter
merged back into the base weights.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import torch

from llmforge.core import paths, registry
from llmforge.core.registry import RunRecord
from llmforge.export import convert
from llmforge.export.formats import Format, find_level, llama_quantize_path

# gguf's own names for the formats it can write directly.
_GGUF_TYPES = {
    "f32": "F32",
    "f16": "F16",
    "bf16": "BF16",
    "q8_0": "Q8_0",
    "q5_1": "Q5_1",
    "q5_0": "Q5_0",
    "q4_1": "Q4_1",
    "q4_0": "Q4_0",
}

_TORCH_DTYPES = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}


@dataclass
class ExportResult:
    path: Path
    format: Format
    quantization: str
    bytes: int

    @property
    def megabytes(self) -> float:
        return self.bytes / 1e6


def export_run(
    run_id: str,
    *,
    fmt: Format = "gguf",
    quantization: str | None = None,
    checkpoint: str = "best",
    out_dir: Path | None = None,
    progress=None,
) -> ExportResult:
    """Export a finished run. Returns where it landed."""
    record = registry.resolve(run_id)
    level = find_level(fmt, quantization or _default_for(fmt))

    if level.needs_llama_cpp and llama_quantize_path() is None:
        raise ValueError(
            f"{level.name} needs llama.cpp's `llama-quantize`, which is not on PATH. "
            f"Install llama.cpp, or choose a level LLMForge can produce itself."
        )

    destination = out_dir or (paths.models_dir() / record.id)
    destination.mkdir(parents=True, exist_ok=True)

    def report(stage: str) -> None:
        if progress:
            progress(stage, 0, 0)

    if record.mode == "finetune":
        report("merging adapter")
        model_dir = _materialise_finetune(record, checkpoint, destination, level.name, fmt)
    else:
        report("converting weights")
        model_dir = _materialise_scratch(record, checkpoint, destination, level.name, fmt)

    if fmt == "safetensors":
        size = sum(p.stat().st_size for p in model_dir.rglob("*") if p.is_file())
        return ExportResult(model_dir, fmt, level.name, size)

    report("writing gguf")
    gguf_path = _write_gguf(model_dir, destination, record, level.name)

    # The HF directory was only scaffolding for the GGUF conversion.
    if model_dir != destination and model_dir.name == "_hf":
        shutil.rmtree(model_dir, ignore_errors=True)

    return ExportResult(gguf_path, fmt, level.name, gguf_path.stat().st_size)


def _default_for(fmt: Format) -> str:
    from llmforge.export.formats import default_level

    return default_level(fmt)


# ---------------------------------------------------------------------------
# producing a HuggingFace directory
# ---------------------------------------------------------------------------


def _materialise_finetune(
    record: RunRecord, checkpoint: str, destination: Path, quant: str, fmt: Format
) -> Path:
    """Merge a LoRA adapter into its base, or take a full fine-tune as-is."""
    from llmforge.finetune import infer
    from llmforge.finetune.plan import FinetunePlan

    plan = FinetunePlan(**record.plan)
    model, tokenizer = infer.load(record, checkpoint=checkpoint)

    if plan.method != "full":
        # Fold the low-rank update into the frozen weights, so the result is a plain
        # model rather than something that needs peft to load.
        model = model.merge_and_unload()

    target = destination if fmt == "safetensors" else destination / "_hf"
    target.mkdir(parents=True, exist_ok=True)

    dtype = _TORCH_DTYPES.get(quant, torch.bfloat16)
    model.to(dtype)
    model.save_pretrained(str(target), safe_serialization=True)
    tokenizer.save_pretrained(str(target))
    return target


def _materialise_scratch(
    record: RunRecord, checkpoint: str, destination: Path, quant: str, fmt: Format
) -> Path:
    """Convert a from-scratch or distilled model into a Llama-shaped directory."""
    from safetensors.torch import save_file

    from llmforge.data import prepare as prep
    from llmforge.pretrain.model import ModelConfig

    ckpt_path = paths.run_dir(record.id) / "ckpt" / f"{checkpoint}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"no {checkpoint} checkpoint for {record.id}")

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    plan = state["plan"]
    cfg = ModelConfig(
        vocab_size=plan["vocab_size"],
        n_layer=plan["n_layer"],
        n_head=plan["n_head"],
        n_kv_head=plan["n_kv_head"],
        d_model=plan["d_model"],
        d_ff=plan["d_ff"],
        max_seq_len=plan["seq_len"],
    )

    weights = convert.to_llama_state_dict(state["task"]["model"], cfg)
    dtype = _TORCH_DTYPES.get(quant, torch.bfloat16)
    weights = {k: v.to(dtype).contiguous() for k, v in weights.items()}

    target = destination if fmt == "safetensors" else destination / "_hf"
    target.mkdir(parents=True, exist_ok=True)

    save_file(weights, str(target / "model.safetensors"), metadata={"format": "pt"})

    prepared = prep.load_prepared(record.corpus_hash, record.tokenizer_id)
    eos_id = prepared.packed.index["eos_id"]

    (target / "config.json").write_text(
        json.dumps(convert.llama_config(cfg, eos_id=eos_id), indent=2)
    )
    # The tokenizer travels with the model; without it the file is unusable.
    prepared.tokenizer.save(str(target / "tokenizer.json"))
    (target / "generation_config.json").write_text(
        json.dumps({"eos_token_id": eos_id, "pad_token_id": eos_id}, indent=2)
    )
    return target


# ---------------------------------------------------------------------------
# gguf
# ---------------------------------------------------------------------------


def _write_gguf(model_dir: Path, destination: Path, record: RunRecord, quant: str) -> Path:
    """Write a GGUF file from a Llama-shaped HuggingFace directory."""
    import gguf
    import numpy as np
    from safetensors.torch import load_file

    cfg = json.loads((model_dir / "config.json").read_text())
    name = f"{record.id}-{quant}.gguf"

    needs_llama_cpp = quant not in _GGUF_TYPES
    # k-quants are produced by llama.cpp from an f16 file, so write that first.
    write_quant = "f16" if needs_llama_cpp else quant
    staged = destination / (f"_staging-{name}" if needs_llama_cpp else name)

    weights: dict[str, torch.Tensor] = {}
    for shard in sorted(model_dir.glob("*.safetensors")):
        weights.update(load_file(str(shard)))

    writer = gguf.GGUFWriter(str(staged), arch="llama")
    n_head = cfg["num_attention_heads"]
    n_kv_head = cfg.get("num_key_value_heads", n_head)
    head_dim = cfg.get("head_dim") or cfg["hidden_size"] // n_head

    writer.add_name(record.name or record.id)
    writer.add_context_length(cfg["max_position_embeddings"])
    writer.add_embedding_length(cfg["hidden_size"])
    writer.add_block_count(cfg["num_hidden_layers"])
    writer.add_feed_forward_length(cfg["intermediate_size"])
    writer.add_head_count(n_head)
    writer.add_head_count_kv(n_kv_head)
    writer.add_rope_freq_base(cfg.get("rope_theta", 10000.0))
    writer.add_rope_dimension_count(head_dim)
    writer.add_layer_norm_rms_eps(cfg.get("rms_norm_eps", 1e-6))
    writer.add_file_type(getattr(gguf.LlamaFileType, f"MOSTLY_{_GGUF_TYPES[write_quant]}",
                                 gguf.LlamaFileType.MOSTLY_F16))

    _add_tokenizer(writer, model_dir, cfg)

    quant_type = getattr(gguf.GGMLQuantizationType, _GGUF_TYPES[write_quant])
    tied = cfg.get("tie_word_embeddings", True)

    for key, tensor in weights.items():
        gguf_name = _gguf_tensor_name(key, tied)
        if gguf_name is None:
            continue

        array = tensor.to(torch.float32).numpy()

        # Q and K need reordering into llama.cpp's rotary layout; see convert.py.
        if ".attn_q." in gguf_name:
            array = convert.permute_for_gguf(torch.from_numpy(array), n_head, head_dim).numpy()
        elif ".attn_k." in gguf_name:
            array = convert.permute_for_gguf(torch.from_numpy(array), n_kv_head, head_dim).numpy()

        # 1-D tensors (norm gains) stay in fp32 — quantizing them costs accuracy for
        # a negligible saving, which is what every other converter does too. Block
        # quantization also needs the last dimension to be a multiple of the block
        # size, so anything that does not divide evenly falls back to f16.
        target_type = gguf.GGMLQuantizationType.F32 if array.ndim == 1 else quant_type
        block, _ = gguf.GGML_QUANT_SIZES[target_type]
        if block > 1 and array.shape[-1] % block:
            # Block quantization needs the last dimension to divide evenly. Vocabulary
            # sizes usually do; when one does not, f16 is the honest fallback.
            target_type = gguf.GGMLQuantizationType.F16

        payload = gguf.quants.quantize(np.ascontiguousarray(array), target_type)
        # No raw_shape: for a quantized uint8 payload the writer reads the byte shape
        # off the array and derives the logical shape itself. Passing the logical
        # shape here makes it try to interpret elements as bytes.
        writer.add_tensor(gguf_name, payload, raw_dtype=target_type)
        del array, payload

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    if not needs_llama_cpp:
        return staged

    final = destination / name
    subprocess.run(
        [llama_quantize_path(), str(staged), str(final), quant.upper()],
        check=True,
        capture_output=True,
        timeout=3600,
    )
    staged.unlink(missing_ok=True)
    return final


def _gguf_tensor_name(key: str, tied: bool) -> str | None:
    """Map a Llama parameter name onto llama.cpp's naming."""
    if key == "model.embed_tokens.weight":
        return "token_embd.weight"
    if key == "model.norm.weight":
        return "output_norm.weight"
    if key == "lm_head.weight":
        # With tied embeddings llama.cpp reuses token_embd; a duplicate would be
        # rejected as an unknown tensor.
        return None if tied else "output.weight"

    if not key.startswith("model.layers."):
        return None

    parts = key.split(".")
    index = parts[2]
    suffix = ".".join(parts[3:])

    mapping = {
        "self_attn.q_proj.weight": "attn_q.weight",
        "self_attn.k_proj.weight": "attn_k.weight",
        "self_attn.v_proj.weight": "attn_v.weight",
        "self_attn.o_proj.weight": "attn_output.weight",
        "mlp.gate_proj.weight": "ffn_gate.weight",
        "mlp.up_proj.weight": "ffn_up.weight",
        "mlp.down_proj.weight": "ffn_down.weight",
        "input_layernorm.weight": "attn_norm.weight",
        "post_attention_layernorm.weight": "ffn_norm.weight",
    }
    tail = mapping.get(suffix)
    return f"blk.{index}.{tail}" if tail else None


def _add_tokenizer(writer, model_dir: Path, cfg: dict) -> None:
    """Embed the vocabulary, so the GGUF file is self-contained."""
    tokenizer_file = model_dir / "tokenizer.json"
    if not tokenizer_file.exists():
        return

    data = json.loads(tokenizer_file.read_text())
    model = data.get("model", {})
    vocab: dict[str, int] = model.get("vocab", {})
    if not vocab:
        return

    tokens = [""] * (max(vocab.values()) + 1)
    for token, index in vocab.items():
        tokens[index] = token

    added = {t["id"]: t for t in data.get("added_tokens", [])}
    for index, entry in added.items():
        if index < len(tokens):
            tokens[index] = entry["content"]

    # 1 = normal, 3 = control. Special tokens must be marked so they are not emitted
    # as ordinary text during generation.
    types = [3 if i in added else 1 for i in range(len(tokens))]

    writer.add_tokenizer_model("gpt2")
    # Our tokenizer is byte-level BPE in GPT-2's style, where a leading space is
    # merged into the following token. Declaring "default" here makes llama.cpp split
    # on whitespace instead: the file loads, generates fluent-looking text, and scores
    # far worse than the same weights do natively, because almost every token is
    # segmented differently from what the model was trained on.
    writer.add_tokenizer_pre("gpt-2")
    writer.add_token_list(tokens)
    writer.add_token_types(types)

    merges = model.get("merges", [])
    if merges:
        writer.add_token_merges(
            [" ".join(m) if isinstance(m, list) else m for m in merges]
        )

    eos = cfg.get("eos_token_id")
    if eos is not None:
        writer.add_eos_token_id(eos)
        writer.add_bos_token_id(cfg.get("bos_token_id", eos))
