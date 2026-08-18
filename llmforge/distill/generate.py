"""Having a teacher write the training data.

The other kind of distillation. Instead of scoring the student against the teacher's
distribution on every token of every epoch — which means running the teacher forever —
the teacher answers a set of prompts once, and the student learns from those answers as
ordinary supervised data.

Two things follow from paying the teacher cost once:

- The student is free to use its own tokenizer, so it need not inherit a 150k-token
  vocabulary and spend most of its parameters on an embedding table.
- Re-training the student is minutes, not another full pass of the teacher, so student
  size and hyperparameters can actually be iterated on.

What is lost is the teacher's uncertainty: the student sees which token was chosen, not
how confident the teacher was about the alternatives.

The output is a folder of JSONL that `llmforge create` ingests directly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch

# Generating one prompt at a time wastes almost all of the GPU; this many at once is a
# good balance against the memory a batch of long sequences needs.
DEFAULT_BATCH = 8

# Outputs shorter than this are refusals, empty strings, or artefacts.
MIN_ANSWER_CHARS = 8


@dataclass
class GenerationStats:
    prompts: int = 0
    generated: int = 0
    rejected: int = 0
    stopped: bool = False
    reasons: dict[str, int] = field(default_factory=dict)

    def reject(self, reason: str) -> None:
        self.rejected += 1
        self.reasons[reason] = self.reasons.get(reason, 0) + 1


def load_teacher(ref: str):
    """Load a teacher from a run id, a local directory, or a Hugging Face id.

    Accepting a run id matters: a fine-tuned model is a base plus an adapter, and
    resolving it here means the adapter never has to be merged and written to disk
    just to be used as a teacher.
    """
    from llmforge.core import hardware, registry

    record = registry.find(ref)
    if record is not None and record.mode == "finetune":
        from llmforge.finetune import infer

        model, tokenizer = infer.load(record)
        return model, tokenizer, f"run {record.id}"

    from transformers import AutoModelForCausalLM

    from llmforge.finetune.sft import load_tokenizer

    tokenizer = load_tokenizer(ref)
    model = AutoModelForCausalLM.from_pretrained(
        ref, dtype=torch.bfloat16, attn_implementation=hardware.attn_implementation()
    ).cuda()
    model.eval()
    return model, tokenizer, ref


def prompts_from(source: Path, limit: int | None = None) -> list[str]:
    """Collect prompts from a corpus folder or a plain text file.

    A text file is one prompt per line. A folder is ingested the usual way: its
    conversations contribute their user turns, and prose contributes nothing, since
    a passage of text is not a question worth answering.
    """
    source = Path(source).expanduser()

    if source.is_file():
        lines = [ln.strip() for ln in source.read_text(errors="replace").splitlines()]
        found = [ln for ln in lines if len(ln) > 4]
    else:
        from llmforge.data import ingest as ing

        corpus = ing.ingest(source)
        found = []
        for record in corpus.iter_records():
            messages = record.get("messages")
            if not messages:
                continue
            users = [m.get("content", "") for m in messages if m.get("role") == "user"]
            if users:
                found.append(users[-1].strip())

        if not found:
            raise ValueError(
                f"No prompts found in {source}. Generation needs questions to answer — "
                f"either conversational data, or a text file with one prompt per line."
            )

    # Order-preserving deduplication: asking the same question twice buys nothing.
    seen: set[str] = set()
    unique = []
    for prompt in found:
        if prompt not in seen:
            seen.add(prompt)
            unique.append(prompt)

    return unique[:limit] if limit else unique


@torch.no_grad()
def generate(
    teacher: str,
    prompts: list[str],
    out_dir: Path,
    *,
    samples_per_prompt: int = 1,
    max_new_tokens: int = 512,
    temperature: float = 0.8,
    top_p: float = 0.9,
    batch_size: int = DEFAULT_BATCH,
    system: str | None = None,
    progress=None,
    should_stop=None,
) -> GenerationStats:
    """Answer every prompt with the teacher, writing a corpus as it goes.

    `should_stop` is checked between batches. Generation can run for hours against a
    large teacher, and the partial corpus written so far is perfectly usable — so a
    stop is a legitimate outcome rather than a lost run.
    """
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "generated.jsonl"

    model, tokenizer, label = load_teacher(teacher)

    # Decoder-only generation must pad on the LEFT. With right padding the model
    # continues from padding tokens rather than from the prompt, and produces fluent
    # nonsense — a failure that looks like a bad teacher rather than a bug.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    stats = GenerationStats(prompts=len(prompts))
    # Each sample of a prompt is a separate generation; interleaving keeps the batches
    # varied rather than repeating one prompt N times in a row.
    work = [p for _ in range(samples_per_prompt) for p in prompts]

    with output.open("w", encoding="utf-8") as handle:
        for start in range(0, len(work), batch_size):
            if should_stop and should_stop():
                stats.stopped = True
                break

            batch = work[start : start + batch_size]

            rendered = [
                tokenizer.apply_chat_template(
                    ([{"role": "system", "content": system}] if system else [])
                    + [{"role": "user", "content": p}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for p in batch
            ]
            inputs = tokenizer(
                rendered, return_tensors="pt", padding=True, add_special_tokens=False
            ).to(model.device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=top_p if temperature > 0 else None,
                pad_token_id=tokenizer.pad_token_id,
            )

            width = inputs["input_ids"].shape[1]
            for prompt, sequence in zip(batch, outputs, strict=True):
                answer = tokenizer.decode(
                    sequence[width:], skip_special_tokens=True
                ).strip()

                reason = _reject_reason(answer)
                if reason:
                    stats.reject(reason)
                    continue

                handle.write(
                    json.dumps(
                        {
                            "messages": (
                                ([{"role": "system", "content": system}] if system else [])
                                + [
                                    {"role": "user", "content": prompt},
                                    {"role": "assistant", "content": answer},
                                ]
                            )
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                # Flushed per batch so a long run is resumable-by-inspection and
                # partial output survives an interruption.
                handle.flush()
                stats.generated += 1

            if progress:
                progress("generating", min(start + batch_size, len(work)), len(work))

    (out_dir / "source.json").write_text(
        json.dumps(
            {
                "teacher": label,
                "prompts": len(prompts),
                "samples_per_prompt": samples_per_prompt,
                "temperature": temperature,
                "top_p": top_p,
                "max_new_tokens": max_new_tokens,
                "system": system,
                "generated": stats.generated,
                "rejected": stats.rejected,
                "stopped_early": stats.stopped,
                "reasons": stats.reasons,
            },
            indent=2,
        )
    )

    del model
    torch.cuda.empty_cache()
    return stats


def _reject_reason(answer: str) -> str | None:
    """Drop outputs that would teach the student something useless."""
    if len(answer) < MIN_ANSWER_CHARS:
        return "empty or too short"

    # A model that has fallen into a loop repeats one line for the whole output;
    # training on that teaches the loop.
    lines = [ln.strip() for ln in answer.splitlines() if ln.strip()]
    if len(lines) >= 6 and len(set(lines)) <= len(lines) // 3:
        return "degenerate repetition"

    return None
