"""Tests for the fine-tuning path.

Loss masking gets the most attention here because it is the failure that hides: a
model trained on unmasked conversations shows a perfectly healthy loss curve and then
answers in the user's voice.
"""

from __future__ import annotations

import numpy as np
import pytest

from llmforge.core import memory
from llmforge.core.config import CorpusAnalysis
from llmforge.core.hardware import Hardware
from llmforge.data.chat import IGNORE_INDEX, Example, collate
from llmforge.finetune.base import BaseModelInfo, _params_from_config
from llmforge.finetune.plan import (
    MAX_AUTO_EPOCHS,
    MIN_TOTAL_STEPS,
    adapter_param_count,
    learning_rate_for,
    plan_finetune,
)

HW = Hardware(
    gpu="NVIDIA GB10",
    capability="12.1",
    bf16_tflops=70.0,
    bandwidth_gbps=210.0,
    memory_gb=128.0,
    compile_ok=True,
    flash_sdpa=True,
)


def info(n_params: int, *, n_layer: int = 24, d_model: int = 2048, vocab: int = 32000):
    return BaseModelInfo(
        ref="test/model",
        is_local=False,
        n_params=n_params,
        n_layer=n_layer,
        d_model=d_model,
        vocab_size=vocab,
        max_position=4096,
        architecture="LlamaForCausalLM",
        torch_dtype="bfloat16",
        has_chat_template=True,
    )


def analysis(kind: str = "instruction") -> CorpusAnalysis:
    return CorpusAnalysis(
        root="/tmp/c", content_hash="abc", kind=kind, n_documents=500, n_chars=400_000
    )


def make_plan(n_params=1_000_000_000, n_examples=2000, **kw):
    return plan_finetune(
        analysis(kw.pop("kind", "instruction")),
        info(n_params, **{k: kw.pop(k) for k in ("n_layer", "d_model", "vocab") if k in kw}),
        hw=kw.pop("hw", HW),
        n_examples=n_examples,
        total_tokens=n_examples * 200,
        p95_length=kw.pop("p95_length", 512),
        **kw,
    )


# --------------------------------------------------------------------------
# base model sizing
# --------------------------------------------------------------------------


def test_config_param_estimate_is_close_for_a_known_shape():
    """Llama-3-8B's published shape should land within a few percent."""
    cfg = {
        "hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32,
        "num_key_value_heads": 8, "intermediate_size": 14336, "vocab_size": 128256,
        "tie_word_embeddings": False,
    }
    estimate = _params_from_config(cfg)
    assert 7.5e9 < estimate < 8.5e9


def test_tied_embeddings_counted_once():
    cfg = {
        "hidden_size": 2048, "num_hidden_layers": 16, "num_attention_heads": 16,
        "num_key_value_heads": 16, "intermediate_size": 8192, "vocab_size": 50000,
    }
    tied = _params_from_config({**cfg, "tie_word_embeddings": True})
    untied = _params_from_config({**cfg, "tie_word_embeddings": False})
    assert untied - tied == 50000 * 2048


def test_label_formats_by_scale():
    assert info(135_000_000).label == "135M"
    assert info(8_200_000_000).label == "8.2B"


# --------------------------------------------------------------------------
# method selection
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n_params,expected",
    [
        (135_000_000, "full"),
        (600_000_000, "full"),
        (8_000_000_000, "lora"),
        (70_000_000_000, "qlora"),
    ],
)
def test_method_ladder(n_params, expected):
    """Bigger models must fall back to cheaper adaptation, never the reverse."""
    i = info(n_params)
    method, _ = memory.choose_finetune_method(
        n_params=i.n_params, n_layer=i.n_layer, d_model=i.d_model,
        vocab_size=i.vocab_size, seq_len=1024, total_memory_gb=128,
        adapter_params=adapter_param_count(i, 16),
    )
    assert method == expected


def test_smaller_budget_forces_cheaper_method():
    i = info(8_000_000_000)
    kwargs = dict(
        n_params=i.n_params, n_layer=i.n_layer, d_model=i.d_model,
        vocab_size=i.vocab_size, seq_len=1024,
        adapter_params=adapter_param_count(i, 16),
    )
    # 24 GB genuinely still fits an 8B LoRA (~17 GB of bf16 weights); 16 GB does not.
    roomy, _ = memory.choose_finetune_method(**kwargs, total_memory_gb=128)
    tight, _ = memory.choose_finetune_method(**kwargs, total_memory_gb=16)
    assert roomy == "lora" and tight == "qlora"


def test_qlora_is_cheapest():
    kwargs = dict(
        n_params=8_000_000_000, n_layer=32, d_model=4096, vocab_size=32000,
        micro_batch=1, seq_len=1024, total_memory_gb=128, adapter_params=20_000_000,
    )
    sizes = [
        memory.estimate_finetune_memory(**kwargs, method=m).total_gb
        for m in ("full", "lora", "qlora")
    ]
    assert sizes == sorted(sizes, reverse=True)


def test_explicit_method_is_honoured():
    assert make_plan(70_000_000_000, method="lora").method == "lora"


def test_unknown_method_rejected():
    with pytest.raises(ValueError, match="unknown method"):
        make_plan(method="adapters")


def test_impossible_fit_is_flagged_not_hidden():
    plan = make_plan(400_000_000_000, n_layer=120, d_model=16384)
    assert any("may not fit" in n for n in plan.notes)


# --------------------------------------------------------------------------
# learning rate scaling
# --------------------------------------------------------------------------


def test_learning_rate_falls_as_models_grow():
    rates = [learning_rate_for(n, "full") for n in (1e8, 2e9, 8e9, 70e9)]
    assert rates == sorted(rates, reverse=True)


def test_lora_runs_hotter_than_full():
    for n in (5e8, 8e9, 70e9):
        assert learning_rate_for(int(n), "lora") > learning_rate_for(int(n), "full")


def test_small_model_gets_a_usable_rate():
    """A 135M model at 1e-5 measurably learns nothing; this pins the fix in place."""
    assert learning_rate_for(135_000_000, "full") >= 5e-5


# --------------------------------------------------------------------------
# plan arithmetic
# --------------------------------------------------------------------------


def test_short_dataset_is_given_more_passes():
    plan = make_plan(n_examples=120)
    assert plan.total_steps >= MIN_TOTAL_STEPS * 0.8
    assert plan.epochs > 3


def test_epoch_extension_is_capped():
    plan = make_plan(n_examples=10)
    assert plan.epochs <= MAX_AUTO_EPOCHS


def test_explicit_epochs_are_not_overridden():
    plan = make_plan(n_examples=120, epochs=2)
    assert plan.epochs == 2


def test_large_dataset_keeps_the_default_passes():
    plan = make_plan(n_examples=50_000)
    assert plan.epochs == 3


def test_batch_never_exceeds_the_dataset():
    plan = make_plan(n_examples=40)
    assert plan.micro_batch * plan.grad_accum <= 40


def test_sequence_length_follows_the_data():
    short = make_plan(p95_length=200)
    long = make_plan(p95_length=3000)
    assert short.seq_len < long.seq_len


def test_sequence_length_capped_by_the_model():
    plan = plan_finetune(
        analysis(), info(1_000_000_000), hw=HW, n_examples=1000,
        total_tokens=200_000, p95_length=99_999,
    )
    assert plan.seq_len <= 4096


def test_raw_corpus_is_continued_pretraining():
    plan = make_plan(kind="raw")
    assert plan.supervised is False
    assert any("continued pretraining" in n for n in plan.notes)


def test_lora_reports_its_share_of_the_model():
    plan = make_plan(8_000_000_000)
    assert plan.method == "lora"
    assert plan.trainable_params < plan.base_params * 0.05
    assert any("frozen" in n for n in plan.notes)


def test_full_finetune_trains_everything():
    plan = make_plan(200_000_000)
    assert plan.method == "full"
    assert plan.trainable_params == plan.base_params


def test_empty_dataset_rejected():
    with pytest.raises(ValueError, match="no training examples"):
        make_plan(n_examples=0)


def test_finetuning_does_not_compile():
    """peft's hooks and torch.compile interact badly enough not to be worth it."""
    assert make_plan().compile is False


# --------------------------------------------------------------------------
# batching
# --------------------------------------------------------------------------


def _example(n: int, supervised_from: int = 0) -> Example:
    ids = np.arange(1, n + 1, dtype=np.int64)
    labels = ids.copy()
    labels[:supervised_from] = IGNORE_INDEX
    return Example(input_ids=ids, labels=labels)


def test_collate_pads_to_the_longest():
    ids, labels, mask = collate([_example(3), _example(7)], pad_id=0)
    assert ids.shape == labels.shape == mask.shape == (2, 7)


def test_padding_is_masked_from_attention_and_loss():
    ids, labels, mask = collate([_example(3), _example(7)], pad_id=0)
    assert mask[0].tolist() == [1, 1, 1, 0, 0, 0, 0]
    assert (labels[0, 3:] == IGNORE_INDEX).all()
    assert (ids[0, 3:] == 0).all()


def test_collate_preserves_existing_masking():
    ids, labels, _ = collate([_example(6, supervised_from=4)], pad_id=0)
    assert (labels[0, :4] == IGNORE_INDEX).all()
    assert labels[0, 4:6].tolist() == ids[0, 4:6].tolist()


def test_supervised_token_count():
    assert _example(10, supervised_from=4).n_supervised == 6


# --------------------------------------------------------------------------
# masking against a real tokenizer
# --------------------------------------------------------------------------



@pytest.fixture(scope="module")
def tokenizer():
    """The real thing, if it is cached locally. These assertions are worth having but
    must not make the suite depend on the network."""
    try:
        from llmforge.finetune.sft import load_tokenizer

        return load_tokenizer("HuggingFaceTB/SmolLM2-135M-Instruct")
    except Exception:
        pytest.skip("base tokenizer unavailable offline")


def test_only_assistant_turns_are_supervised(tokenizer):
    from llmforge.data.chat import build_example

    example = build_example(
        tokenizer,
        [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "Paris."},
        ],
        512,
    )
    supervised = tokenizer.decode([int(t) for t in example.labels if t != IGNORE_INDEX])
    assert "Paris." in supervised
    assert "capital of France" not in supervised


def test_every_assistant_turn_is_supervised(tokenizer):
    from llmforge.data.chat import build_example

    example = build_example(
        tokenizer,
        [
            {"role": "user", "content": "First question?"},
            {"role": "assistant", "content": "ALPHA answer."},
            {"role": "user", "content": "Second question?"},
            {"role": "assistant", "content": "BETA answer."},
        ],
        512,
    )
    supervised = tokenizer.decode([int(t) for t in example.labels if t != IGNORE_INDEX])
    assert "ALPHA answer." in supervised and "BETA answer." in supervised
    assert "First question" not in supervised and "Second question" not in supervised


def test_labels_match_inputs_wherever_supervised(tokenizer):
    from llmforge.data.chat import build_example

    example = build_example(
        tokenizer,
        [{"role": "user", "content": "Hi there"}, {"role": "assistant", "content": "Hello."}],
        512,
    )
    for token, label in zip(example.input_ids, example.labels, strict=True):
        if label != IGNORE_INDEX:
            assert label == token


def test_conversation_with_no_assistant_turn_is_dropped(tokenizer):
    """Every position masked would contribute NaN to the mean loss."""
    from llmforge.data.chat import build_example

    assert build_example(tokenizer, [{"role": "user", "content": "Only a question?"}], 512) is None


def test_empty_conversation_is_dropped(tokenizer):
    from llmforge.data.chat import build_example

    assert build_example(tokenizer, [], 512) is None


def test_raw_text_example_is_fully_supervised(tokenizer):
    from llmforge.data.chat import build_text_example

    example = build_text_example(tokenizer, "Some ordinary prose for continued pretraining.", 512)
    assert example.n_supervised == len(example)


def test_truncation_keeps_the_supervised_tail(tokenizer):
    from llmforge.data.chat import build_example

    example = build_example(
        tokenizer,
        [
            {"role": "user", "content": "word " * 400},
            {"role": "assistant", "content": "The final answer."},
        ],
        max_len=64,
    )
    assert example is not None and len(example) == 64
    # Truncating from the left must not throw away what we are training on.
    assert example.n_supervised > 0
