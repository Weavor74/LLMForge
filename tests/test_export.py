"""Tests for export and evaluation.

The critical property is that an exported model computes what the trained one did.
A conversion that merely *loads* is the dangerous outcome: mismatched rotary layouts
or a mis-split QKV produce a model that generates fluent, confident nonsense.
"""

from __future__ import annotations

import pytest
import torch

from llmforge.export import convert
from llmforge.export.formats import (
    QUANT_LEVELS,
    default_level,
    estimate_bytes,
    find_level,
    levels_for,
)
from llmforge.pretrain.model import ModelConfig, Transformer

CFG = ModelConfig(
    vocab_size=512, n_layer=3, n_head=6, n_kv_head=2, d_model=384, d_ff=1024, max_seq_len=128
)


# --------------------------------------------------------------------------
# format catalogue
# --------------------------------------------------------------------------


def test_every_level_belongs_to_its_format():
    for level in QUANT_LEVELS:
        assert level in levels_for(level.format)


def test_defaults_are_producible_without_extra_tooling():
    """A default that needs software the user has not installed is not a default."""
    for fmt in ("gguf", "safetensors"):
        assert find_level(fmt, default_level(fmt)).needs_llama_cpp is False


def test_unknown_level_is_rejected():
    with pytest.raises(ValueError, match="unknown quantization"):
        find_level("gguf", "q2_k_xxs")


def test_level_from_the_wrong_format_is_rejected():
    with pytest.raises(ValueError, match="unknown quantization"):
        find_level("safetensors", "q4_0")


def test_size_estimates_order_by_precision():
    sizes = [estimate_bytes(1_000_000, find_level("gguf", n)) for n in ("f16", "q8_0", "q4_0")]
    assert sizes == sorted(sizes, reverse=True)


def test_k_quants_are_marked_as_needing_llama_cpp():
    assert find_level("gguf", "q4_k_m").needs_llama_cpp
    assert not find_level("gguf", "q8_0").needs_llama_cpp


# --------------------------------------------------------------------------
# weight conversion
# --------------------------------------------------------------------------


def test_conversion_produces_llama_names():
    model = Transformer(CFG)
    out = convert.to_llama_state_dict(model.state_dict(), CFG)

    assert "model.embed_tokens.weight" in out
    assert "model.norm.weight" in out
    for i in range(CFG.n_layer):
        for suffix in (
            "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
            "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
            "input_layernorm", "post_attention_layernorm",
        ):
            assert f"model.layers.{i}.{suffix}.weight" in out


def test_qkv_split_widths_respect_grouped_query_attention():
    model = Transformer(CFG)
    out = convert.to_llama_state_dict(model.state_dict(), CFG)
    kv_width = CFG.n_kv_head * CFG.head_dim

    assert out["model.layers.0.self_attn.q_proj.weight"].shape[0] == CFG.n_head * CFG.head_dim
    assert out["model.layers.0.self_attn.k_proj.weight"].shape[0] == kv_width
    assert out["model.layers.0.self_attn.v_proj.weight"].shape[0] == kv_width


def test_qkv_split_preserves_the_original_values():
    """The fused projection is packed q, k, v in that order; a wrong split would
    silently shuffle attention weights between roles."""
    model = Transformer(CFG)
    state = model.state_dict()
    out = convert.to_llama_state_dict(state, CFG)

    fused = state["blocks.0.attn.qkv_proj.weight"]
    rebuilt = torch.cat(
        [
            out["model.layers.0.self_attn.q_proj.weight"],
            out["model.layers.0.self_attn.k_proj.weight"],
            out["model.layers.0.self_attn.v_proj.weight"],
        ],
        dim=0,
    )
    assert torch.equal(fused, rebuilt)


def test_tied_embeddings_omit_the_output_head():
    tied = ModelConfig(**{**CFG.__dict__, "tie_embeddings": True})
    out = convert.to_llama_state_dict(Transformer(tied).state_dict(), tied)
    assert "lm_head.weight" not in out


def test_untied_embeddings_include_the_output_head():
    untied = ModelConfig(**{**CFG.__dict__, "tie_embeddings": False})
    out = convert.to_llama_state_dict(Transformer(untied).state_dict(), untied)
    assert "lm_head.weight" in out


def test_config_declares_a_loadable_llama():
    cfg = convert.llama_config(CFG, eos_id=7)
    assert cfg["architectures"] == ["LlamaForCausalLM"]
    assert cfg["hidden_size"] == CFG.d_model
    assert cfg["num_key_value_heads"] == CFG.n_kv_head
    assert cfg["intermediate_size"] == CFG.d_ff
    assert cfg["hidden_act"] == "silu"
    assert cfg["eos_token_id"] == 7


# --------------------------------------------------------------------------
# the rotary permutation
# --------------------------------------------------------------------------


def test_permutation_is_invertible():
    """Applied twice in the wrong direction it must not silently corrupt weights."""
    weight = torch.randn(384, 384)
    permuted = convert.permute_for_gguf(weight, 6, 64)
    restored = permuted.reshape(6, 32, 2, 384).swapaxes(1, 2).reshape(384, 384)
    assert torch.equal(weight, restored)


def test_permutation_preserves_shape_and_contents():
    weight = torch.randn(128, 256)
    permuted = convert.permute_for_gguf(weight, 2, 64)
    assert permuted.shape == weight.shape
    # A permutation moves values without inventing or losing any.
    assert torch.equal(permuted.flatten().sort().values, weight.flatten().sort().values)


def test_permutation_actually_reorders():
    """If this became a no-op, GGUF exports would load and generate nonsense."""
    weight = torch.arange(384 * 8, dtype=torch.float32).reshape(384, 8)
    assert not torch.equal(convert.permute_for_gguf(weight, 6, 64), weight)


def test_permutation_handles_grouped_query_key_widths():
    """K is narrower than Q under GQA and must be permuted against its own head count."""
    kv_width = 2 * 64
    weight = torch.randn(kv_width, 384)
    permuted = convert.permute_for_gguf(weight, 2, 64)
    assert permuted.shape == weight.shape


# --------------------------------------------------------------------------
# eval reporting
# --------------------------------------------------------------------------


def _report(before, after):
    from llmforge.eval.harness import EvalReport

    return EvalReport(
        run_id="r", mode="finetune", before_ppl=before, after_ppl=after, n_examples=10
    )


def test_improvement_is_a_fractional_reduction():
    assert _report(100.0, 50.0).improvement == pytest.approx(0.5)


def test_regression_reports_negative_improvement():
    assert _report(50.0, 100.0).improvement == pytest.approx(-1.0)


def test_improvement_is_undefined_without_a_baseline():
    assert _report(None, 12.0).improvement is None


def test_report_serialises_for_the_api():
    from llmforge.eval.harness import Comparison

    report = _report(10.0, 5.0)
    report.comparisons = [Comparison(prompt="p", before="b", after="a")]
    payload = report.to_dict()

    assert payload["improvement"] == pytest.approx(0.5)
    assert payload["comparisons"][0] == {"prompt": "p", "before": "b", "after": "a"}


@pytest.mark.parametrize(
    "before,after,expected",
    [
        (100.0, 10.0, "clearly"),
        (100.0, 80.0, "modest"),
        (100.0, 99.0, "barely"),
        (100.0, 200.0, "worse"),
    ],
)
def test_notes_describe_what_happened(before, after, expected):
    from llmforge.eval.harness import _finetune_notes

    notes = _finetune_notes(_report(before, after), plan=None)
    assert any(expected in n for n in notes)


def test_large_improvement_warns_about_memorisation():
    from llmforge.eval.harness import _finetune_notes

    notes = _finetune_notes(_report(100.0, 2.0), plan=None)
    assert any("memoris" in n for n in notes)


# --------------------------------------------------------------------------
# checkpoint retention
# --------------------------------------------------------------------------


def test_pruning_keeps_best_and_last(tmp_path):
    from llmforge.core.loop import prune_checkpoints

    for name in ("best.pt", "last.pt", "step_000100.pt", "step_000200.pt", "last.tmp"):
        (tmp_path / name).write_bytes(b"x")

    prune_checkpoints(tmp_path)
    remaining = sorted(p.name for p in tmp_path.iterdir())
    assert remaining == ["best.pt", "last.pt"]


def test_pruning_an_empty_directory_is_harmless(tmp_path):
    from llmforge.core.loop import prune_checkpoints

    assert prune_checkpoints(tmp_path) == []


# --------------------------------------------------------------------------
# end-to-end equivalence
# --------------------------------------------------------------------------


def test_exported_model_computes_what_the_original_did(tmp_path):
    """The property the whole export exists to preserve.

    Converts a real model through the safetensors path and checks transformers
    reproduces its logits. Catches a mis-split QKV, a wrong rotary convention, or a
    config that disagrees with the weights — none of which prevent the file loading.
    """
    transformers = pytest.importorskip("transformers")
    import json

    from safetensors.torch import save_file

    torch.manual_seed(0)
    model = Transformer(CFG).eval()

    weights = convert.to_llama_state_dict(model.state_dict(), CFG)
    weights = {k: v.to(torch.float32).contiguous() for k, v in weights.items()}
    save_file(weights, str(tmp_path / "model.safetensors"), metadata={"format": "pt"})

    config = convert.llama_config(CFG, eos_id=0)
    config["torch_dtype"] = "float32"
    (tmp_path / "config.json").write_text(json.dumps(config))

    exported = transformers.AutoModelForCausalLM.from_pretrained(
        str(tmp_path), dtype=torch.float32
    ).eval()

    idx = torch.randint(0, CFG.vocab_size, (2, 32))
    with torch.no_grad():
        ours, _ = model(idx)
        theirs = exported(input_ids=idx).logits

    assert torch.allclose(ours, theirs, atol=1e-4), (
        f"exported model diverges by {(ours - theirs).abs().max().item():.6f}"
    )


def test_gguf_declares_the_gpt2_pretokenizer(tmp_path, monkeypatch):
    """The pre-tokenizer name must match how the vocabulary was actually built.

    Our tokenizer is byte-level BPE in GPT-2's style, where a leading space merges
    into the following token. Declaring "default" instead makes llama.cpp split on
    whitespace: the file loads and generates fluent-looking text while scoring far
    worse than the identical weights do natively. Measured at 88.5 versus 62.6
    perplexity before this was fixed, so it is worth pinning.
    """
    import gguf

    recorded = {}

    class Recorder:
        def __getattr__(self, name):
            def capture(*args, **kwargs):
                recorded[name] = args[0] if args else None

            return capture

    from llmforge.export import exporter

    writer = Recorder()
    (tmp_path / "tokenizer.json").write_text(
        '{"model": {"vocab": {"a": 0, "b": 1}, "merges": [["a", "b"]]}, "added_tokens": []}'
    )
    exporter._add_tokenizer(writer, tmp_path, {"eos_token_id": 0})

    assert recorded["add_tokenizer_pre"] == "gpt-2"
    assert recorded["add_tokenizer_model"] == "gpt2"
    assert recorded["add_token_merges"] == ["a b"]
    assert gguf  # imported for the side of documenting the dependency
