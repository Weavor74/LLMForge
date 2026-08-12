"""Tests for the parts of ingest whose failures are silent.

Bad extraction is obvious. Unstable hashing, a validation split that overlaps
training, or a deduper that behaves differently between processes are not — they
just quietly produce a worse model.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import pytest

from llmforge.data.clean import Deduper, QualityConfig, quality_reject
from llmforge.data.extract import Record, extract, record_from_mapping
from llmforge.data.pack import _is_val, render_chat

# --------------------------------------------------------------------------
# structured record interpretation
# --------------------------------------------------------------------------


def test_chat_messages_recognised():
    rec = record_from_mapping(
        {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]},
        "s",
    )
    assert rec is not None and rec.is_chat
    assert [m["role"] for m in rec.messages] == ["user", "assistant"]


def test_sharegpt_role_aliases_normalised():
    rec = record_from_mapping(
        {"conversations": [{"from": "human", "value": "hi"}, {"from": "gpt", "value": "yo"}]}, "s"
    )
    assert [m["role"] for m in rec.messages] == ["user", "assistant"]


def test_alpaca_input_folded_into_prompt():
    rec = record_from_mapping(
        {"instruction": "Translate this", "input": "bonjour", "output": "hello"}, "s"
    )
    assert rec.is_chat
    assert "bonjour" in rec.messages[0]["content"]
    assert rec.messages[1]["content"] == "hello"


def test_system_prompt_preserved():
    rec = record_from_mapping(
        {"system": "Be terse.", "prompt": "hi", "completion": "yo"}, "s"
    )
    assert rec.messages[0] == {"role": "system", "content": "Be terse."}


def test_structured_form_beats_rendered_copy():
    """A record carrying both shapes is a conversation, not prose."""
    rec = record_from_mapping(
        {"text": "user: hi\nassistant: yo", "messages": [{"role": "user", "content": "hi"}]}, "s"
    )
    assert rec.is_chat


def test_metadata_only_record_rejected():
    assert record_from_mapping({"id": 7, "lang": "en", "score": 0.5}, "s") is None


def test_single_long_string_used_as_fallback():
    rec = record_from_mapping({"id": 1, "unusual_key": "x" * 100}, "s")
    assert rec is not None and not rec.is_chat


# --------------------------------------------------------------------------
# quality filtering
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("hi", "too short"),
        ("//8/" * 400, "low alphabetic ratio"),
        ("Same line here.\n" * 60, "repetitive lines"),
        ("word " * 200, None),
    ],
)
def test_quality_reasons(text, expected):
    assert quality_reject(text, QualityConfig()) == expected


# --------------------------------------------------------------------------
# deduplication
# --------------------------------------------------------------------------


def _doc(seed: int, n: int = 200) -> str:
    import random

    rng = random.Random(seed)
    vocab = "alpha bravo charlie delta echo foxtrot golf hotel india juliet".split()
    return " ".join(rng.choice(vocab) for _ in range(n))


def test_exact_duplicate_detected():
    d = Deduper()
    text = _doc(1)
    assert d.is_duplicate(text) is False
    assert d.is_duplicate(text) is True
    assert d.n_exact == 1


def test_near_duplicate_detected():
    d = Deduper()
    text = _doc(2)
    assert d.is_duplicate(text) is False
    assert d.is_duplicate(text + "\n\nA short trailing sentence.") is True
    assert d.n_near == 1


def test_distinct_documents_kept():
    d = Deduper()
    assert d.is_duplicate(_doc(3)) is False
    assert d.is_duplicate(_doc(4)) is False
    assert d.n_near == 0


def test_dedup_is_stable_across_processes():
    """Python's hash() is salted per process; ours must not be.

    If this regresses, two identical ingests produce different corpora and the
    reproducibility guarantee is void.
    """
    script = textwrap.dedent(
        """
        import random
        from llmforge.data.clean import Deduper
        rng = random.Random(11)
        vocab = "alpha bravo charlie delta echo foxtrot".split()
        text = " ".join(rng.choice(vocab) for _ in range(300))
        d = Deduper()
        d.is_duplicate(text)
        print(sorted(d._buckets)[0][1])
        """
    )
    runs = {
        subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=True
        ).stdout.strip()
        for _ in range(2)
    }
    assert len(runs) == 1, "band signatures differ between processes"


# --------------------------------------------------------------------------
# train / val split
# --------------------------------------------------------------------------


def test_val_split_is_deterministic():
    records = [{"source": f"file_{i}.txt"} for i in range(500)]
    first = [_is_val(r, 0.1) for r in records]
    second = [_is_val(r, 0.1) for r in records]
    assert first == second


def test_val_split_hits_approximate_target():
    records = [{"source": f"file_{i}.txt"} for i in range(5000)]
    frac = sum(_is_val(r, 0.1) for r in records) / len(records)
    assert 0.07 < frac < 0.13


def test_val_split_disabled_at_zero():
    assert not any(_is_val({"source": f"f{i}"}, 0.0) for i in range(100))


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------


def test_latin1_file_decoded(tmp_path):
    p = tmp_path / "legacy.txt"
    p.write_bytes("Café résumé naïve. It carries on for a while longer.".encode("latin-1"))
    (rec,) = list(extract(p, "legacy.txt"))
    assert "Café" in rec.text


def test_html_chrome_removed(tmp_path):
    p = tmp_path / "page.html"
    p.write_text(
        "<html><head><script>var x=1;</script><style>b{}</style></head>"
        "<body><nav>Menu</nav><p>Real content here.</p><footer>foot</footer></body></html>"
    )
    (rec,) = list(extract(p, "page.html"))
    assert "Real content here." in rec.text
    for noise in ("var x=1", "Menu", "foot"):
        assert noise not in rec.text


def test_jsonl_mixed_shapes(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text(
        json.dumps({"text": "plain prose"})
        + "\n"
        + json.dumps({"instruction": "do it", "output": "done"})
        + "\n"
        + "{ not valid json\n"  # must be skipped, not fatal
        + json.dumps({"id": 3})
        + "\n"
    )
    recs = list(extract(p, "d.jsonl"))
    assert len(recs) == 2
    assert recs[0].text == "plain prose"
    assert recs[1].is_chat


def test_json_wrapper_unwrapped(tmp_path):
    p = tmp_path / "d.json"
    p.write_text(json.dumps({"version": 1, "data": [{"text": f"doc {i}"} for i in range(5)]}))
    assert len(list(extract(p, "d.json"))) == 5


def test_csv_question_answer_becomes_chat(tmp_path):
    p = tmp_path / "qa.csv"
    p.write_text('id,question,answer\n1,"What is it?","A thing."\n')
    (rec,) = list(extract(p, "qa.csv"))
    assert rec.is_chat


def test_unsupported_extension_skipped(tmp_path):
    from llmforge.data.extract import SkipFile

    p = tmp_path / "x.bin"
    p.write_bytes(b"\x00\x01")
    with pytest.raises(SkipFile):
        list(extract(p, "x.bin"))


# --------------------------------------------------------------------------
# chat rendering
# --------------------------------------------------------------------------


def test_render_chat_marks_turn_boundaries():
    out = render_chat([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}])
    assert out.count("<|im_start|>") == 2
    assert out.count("<|im_end|>") == 2


def test_record_char_len_counts_message_content():
    rec = Record(source="s", messages=[{"role": "user", "content": "12345"}])
    assert rec.char_len() == 5


def test_utf8_preferred_over_legacy_codepages(tmp_path):
    p = tmp_path / "modern.txt"
    p.write_text("Café résumé naïve — with an em dash.", encoding="utf-8")
    (rec,) = list(extract(p, "modern.txt"))
    assert "Café résumé naïve — with an em dash." in rec.text


def test_utf16_bom_decoded(tmp_path):
    p = tmp_path / "wide.txt"
    # str.encode("utf-16") emits a BOM, which is what the decoder keys off.
    p.write_bytes("Wide encoded text that runs on for a bit.".encode("utf-16"))
    (rec,) = list(extract(p, "wide.txt"))
    assert "Wide encoded text" in rec.text
