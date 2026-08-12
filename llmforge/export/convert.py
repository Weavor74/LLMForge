"""Converting a from-scratch model into the shape the rest of the world expects.

Our transformer is architecturally a Llama — RoPE, RMSNorm, SwiGLU, GQA, no biases —
so it can be expressed exactly in Llama's parameter naming, and then loaded by
transformers or converted to GGUF. What differs is bookkeeping: we fuse Q, K and V
into one projection, and we name things after what they do rather than after Llama.

Both differences are resolved here, once, so the safetensors and GGUF writers can
share the result.
"""

from __future__ import annotations

import torch

from llmforge.pretrain.model import ModelConfig


def to_llama_state_dict(state: dict[str, torch.Tensor], cfg: ModelConfig) -> dict[str, torch.Tensor]:
    """Rename and un-fuse our weights into Llama's layout."""
    head_dim = cfg.head_dim
    q_width = cfg.n_head * head_dim
    kv_width = cfg.n_kv_head * head_dim

    out: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": state["tok_emb.weight"],
        "model.norm.weight": state["final_norm.weight"],
    }

    for i in range(cfg.n_layer):
        src = f"blocks.{i}"
        dst = f"model.layers.{i}"

        # One fused projection back into three, in the order it was packed.
        qkv = state[f"{src}.attn.qkv_proj.weight"]
        q, k, v = qkv.split([q_width, kv_width, kv_width], dim=0)

        out[f"{dst}.self_attn.q_proj.weight"] = q
        out[f"{dst}.self_attn.k_proj.weight"] = k
        out[f"{dst}.self_attn.v_proj.weight"] = v
        out[f"{dst}.self_attn.o_proj.weight"] = state[f"{src}.attn.o_proj.weight"]

        out[f"{dst}.mlp.gate_proj.weight"] = state[f"{src}.mlp.gate_proj.weight"]
        out[f"{dst}.mlp.up_proj.weight"] = state[f"{src}.mlp.up_proj.weight"]
        out[f"{dst}.mlp.down_proj.weight"] = state[f"{src}.mlp.down_proj.weight"]

        out[f"{dst}.input_layernorm.weight"] = state[f"{src}.attn_norm.weight"]
        out[f"{dst}.post_attention_layernorm.weight"] = state[f"{src}.mlp_norm.weight"]

    # Tied embeddings share storage; transformers expects the key to be absent and
    # infers it from the config, but writing it explicitly is harmless and more
    # portable across loaders.
    if not cfg.tie_embeddings:
        out["lm_head.weight"] = state["lm_head.weight"]

    return out


def llama_config(cfg: ModelConfig, eos_id: int, bos_id: int | None = None) -> dict:
    """A config.json transformers can instantiate LlamaForCausalLM from."""
    return {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": cfg.d_model,
        "intermediate_size": cfg.d_ff,
        "num_hidden_layers": cfg.n_layer,
        "num_attention_heads": cfg.n_head,
        "num_key_value_heads": cfg.n_kv_head,
        "head_dim": cfg.head_dim,
        "vocab_size": cfg.vocab_size,
        "max_position_embeddings": cfg.max_seq_len,
        "rope_theta": cfg.rope_theta,
        "rms_norm_eps": 1e-6,
        "hidden_act": "silu",
        "tie_word_embeddings": cfg.tie_embeddings,
        "attention_bias": False,
        "mlp_bias": False,
        "torch_dtype": "bfloat16",
        "bos_token_id": bos_id if bos_id is not None else eos_id,
        "eos_token_id": eos_id,
        # Our tokenizer has no dedicated pad token; end-of-text doubles as one.
        "pad_token_id": eos_id,
        "transformers_version": "4.44.0",
    }


def permute_for_gguf(weight: torch.Tensor, n_head: int, head_dim: int) -> torch.Tensor:
    """Reorder a Q or K projection from HF's rotary layout into llama.cpp's.

    This is the one genuinely non-obvious step in exporting. Both implementations
    compute the same rotation, but they disagree about which pairs of dimensions are
    rotated together: HF pairs dimension i with i + head_dim/2, while llama.cpp's
    LLAMA architecture pairs adjacent dimensions 2i and 2i+1.

    Skipping this produces a file that loads cleanly and generates fluent nonsense,
    which is exactly the kind of failure worth naming in a comment.
    """
    rows = weight.shape[0]
    rest = weight.shape[1:]
    return (
        weight.reshape(n_head, 2, head_dim // 2, *rest)
        .swapaxes(1, 2)
        .reshape(rows, *rest)
    )
