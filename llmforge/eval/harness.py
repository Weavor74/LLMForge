"""Did training actually change anything?

Validation loss says a run converged. It does not say whether the result is better
than what you started with — and across runs with different tokenizers it is not even
comparable, since cross-entropy over a 49k vocabulary and over a 4k one measure
different things.

So the question this answers is the one that has a defensible answer: on held-out data
from your own folder, and on prompts drawn from it, how does the model you produced
differ from the model you started with?
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from llmforge.core import hardware, registry
from llmforge.core.registry import RunRecord

# Enough held-out examples for the mean to settle, few enough to run in seconds.
DEFAULT_EVAL_EXAMPLES = 64
DEFAULT_PROMPTS = 5


@dataclass
class Comparison:
    """One prompt, answered by both models."""

    prompt: str
    before: str
    after: str


@dataclass
class EvalReport:
    run_id: str
    mode: str
    # Perplexity on held-out data. `before` is None when there was no starting model.
    before_ppl: float | None
    after_ppl: float | None
    n_examples: int
    comparisons: list[Comparison] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def improvement(self) -> float | None:
        """Fractional reduction in perplexity. Positive means the run helped."""
        if self.before_ppl is None or self.after_ppl is None or self.before_ppl <= 0:
            return None
        return (self.before_ppl - self.after_ppl) / self.before_ppl

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "mode": self.mode,
            "before_ppl": self.before_ppl,
            "after_ppl": self.after_ppl,
            "improvement": self.improvement,
            "n_examples": self.n_examples,
            "comparisons": [
                {"prompt": c.prompt, "before": c.before, "after": c.after}
                for c in self.comparisons
            ],
            "notes": self.notes,
        }


def evaluate(
    run_id: str,
    *,
    n_examples: int = DEFAULT_EVAL_EXAMPLES,
    n_prompts: int = DEFAULT_PROMPTS,
    checkpoint: str = "best",
    progress=None,
) -> EvalReport:
    """Compare a finished run against whatever it started from."""
    record = registry.resolve(run_id)
    if record.status not in ("completed", "cancelled"):
        raise ValueError(f"run {record.id} has not finished ({record.status})")

    if record.mode == "finetune":
        return _evaluate_finetune(record, n_examples, n_prompts, checkpoint, progress)
    return _evaluate_scratch(record, n_examples, n_prompts, checkpoint, progress)


# ---------------------------------------------------------------------------
# fine-tuning: base model versus tuned model
# ---------------------------------------------------------------------------


def _evaluate_finetune(
    record: RunRecord, n_examples: int, n_prompts: int, checkpoint: str, progress
) -> EvalReport:
    from transformers import AutoModelForCausalLM

    from llmforge.data import ingest as ing
    from llmforge.finetune import infer
    from llmforge.finetune.plan import FinetunePlan
    from llmforge.finetune.sft import build_dataset, load_tokenizer

    plan = FinetunePlan(**record.plan)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def report(stage: str) -> None:
        if progress:
            progress(stage, 0, 0)

    report("rebuilding held-out data")
    tokenizer = load_tokenizer(plan.base_model)
    corpus = ing.load_cached(record.corpus_hash)
    if corpus is None:
        raise FileNotFoundError("the corpus this run used is no longer in the workspace")

    dataset = build_dataset(
        corpus, tokenizer, supervised=plan.supervised, max_len=plan.seq_len
    )
    # The validation split is deterministic, so this is the same held-out data the
    # run itself never trained on.
    examples = dataset.val[:n_examples] or dataset.train[:n_examples]

    report(f"scoring {plan.base_model}")
    base = AutoModelForCausalLM.from_pretrained(
        plan.base_model,
        dtype=torch.bfloat16,
        attn_implementation=hardware.attn_implementation(),
    ).to(device)
    base.eval()
    before_ppl = _hf_perplexity(base, examples, dataset.pad_id, device)
    before_answers = _hf_generate(base, tokenizer, _prompts_from(dataset, tokenizer, n_prompts), plan)
    del base
    torch.cuda.empty_cache()

    report("scoring your model")
    tuned, _ = infer.load(record, checkpoint=checkpoint)
    after_ppl = _hf_perplexity(tuned, examples, dataset.pad_id, device)
    prompts = _prompts_from(dataset, tokenizer, n_prompts)
    after_answers = _hf_generate(tuned, tokenizer, prompts, plan)
    del tuned
    torch.cuda.empty_cache()

    comparisons = [
        Comparison(prompt=p, before=b, after=a)
        for p, b, a in zip(prompts, before_answers, after_answers, strict=True)
    ]

    report_obj = EvalReport(
        run_id=record.id,
        mode="finetune",
        before_ppl=before_ppl,
        after_ppl=after_ppl,
        n_examples=len(examples),
        comparisons=comparisons,
    )
    report_obj.notes = _finetune_notes(report_obj, plan)
    return report_obj


@torch.no_grad()
def _hf_perplexity(model, examples, pad_id: int, device, batch_size: int = 4) -> float:
    """Perplexity over the supervised tokens only.

    Averaging over prompt tokens too would dilute the measurement with text the model
    was never asked to produce.
    """
    from llmforge.data.chat import collate

    total_loss = 0.0
    total_batches = 0

    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        if not batch:
            continue
        input_ids, labels, attention = collate(batch, pad_id)
        out = model(
            input_ids=torch.from_numpy(input_ids).to(device),
            attention_mask=torch.from_numpy(attention).to(device),
            labels=torch.from_numpy(labels).to(device),
        )
        total_loss += out.loss.item()
        total_batches += 1

    if not total_batches:
        return float("nan")
    return math.exp(min(total_loss / total_batches, 20))


def _prompts_from(dataset, tokenizer, n: int) -> list[str]:
    """The user turns from held-out conversations, so both models answer the same
    questions the data actually contains."""
    prompts: list[str] = []
    for messages in dataset.val_sources:
        user_turns = [m.get("content", "") for m in messages if m.get("role") == "user"]
        if not user_turns:
            continue
        # The last user turn is the one the final assistant answer responds to.
        text = user_turns[-1].strip()
        if 4 < len(text) < 400:
            prompts.append(text)
        if len(prompts) >= n:
            break
    return prompts


@torch.no_grad()
def _hf_generate(model, tokenizer, prompts: list[str], plan) -> list[str]:
    out: list[str] = []
    for prompt in prompts:
        if plan.supervised:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = prompt
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        generated = model.generate(
            **inputs,
            max_new_tokens=80,
            # Greedy: the two models must differ because they differ, not because
            # they sampled differently.
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        out.append(
            tokenizer.decode(
                generated[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            ).strip()
        )
    return out


def _finetune_notes(report: EvalReport, plan) -> list[str]:
    notes: list[str] = []
    improvement = report.improvement

    if improvement is None:
        return notes

    if improvement > 0.5:
        notes.append(
            f"Perplexity on held-out data fell {improvement:.0%}, from "
            f"{report.before_ppl:.1f} to {report.after_ppl:.1f}. The model clearly "
            f"learned your data."
        )
    elif improvement > 0.05:
        notes.append(
            f"Perplexity fell {improvement:.0%} ({report.before_ppl:.1f} → "
            f"{report.after_ppl:.1f}) — a real but modest change."
        )
    elif improvement > -0.05:
        notes.append(
            f"Perplexity barely moved ({report.before_ppl:.1f} → {report.after_ppl:.1f}). "
            f"Training had little effect: too few examples, too few steps, or a rate "
            f"too low for a model this size."
        )
    else:
        notes.append(
            f"Perplexity got *worse* ({report.before_ppl:.1f} → {report.after_ppl:.1f}). "
            f"Usually too high a learning rate, or too many passes over too little data."
        )

    if improvement > 0.9:
        notes.append(
            "An improvement this large on a small dataset often means memorisation "
            "rather than generalisation. Compare the generations, not just the number."
        )

    return notes


# ---------------------------------------------------------------------------
# from-scratch and distilled: there is no "before"
# ---------------------------------------------------------------------------


def _evaluate_scratch(
    record: RunRecord, n_examples: int, n_prompts: int, checkpoint: str, progress
) -> EvalReport:
    """Score a model that had no starting point.

    A from-scratch model has nothing to compare against, so the report gives held-out
    perplexity and sample continuations. For a distilled model the teacher *is* the
    natural comparison, and it is scored on the same held-out tokens.
    """
    from llmforge.core import paths
    from llmforge.data import prepare as prep
    from llmforge.pretrain.data import TokenStream
    from llmforge.pretrain.train import build_model, sample_text

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def report(stage: str) -> None:
        if progress:
            progress(stage, 0, 0)

    ckpt_path = paths.run_dir(record.id) / "ckpt" / f"{checkpoint}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"no {checkpoint} checkpoint for {record.id}")

    state = torch.load(ckpt_path, map_location=device, weights_only=False)

    if record.mode == "distill":
        from llmforge.distill.plan import DistillPlan

        plan = DistillPlan(**state["plan"])
    else:
        from llmforge.core.planner import TrainPlan

        plan = TrainPlan(**state["plan"])

    report("scoring your model")
    model = build_model(plan, device)
    model.load_state_dict(state["task"]["model"])
    model.eval()

    prepared = prep.load_prepared(record.corpus_hash, record.tokenizer_id)
    stream = TokenStream(prepared.packed, "val", plan.seq_len, seed=plan.seed)
    batches = stream.deterministic_batches(min(plan.micro_batch, 4), n_examples // 4 or 1, device)

    after_ppl = _native_perplexity(model, batches)

    comparisons: list[Comparison] = []
    prompts = _corpus_prompts(prepared, n_prompts)
    for prompt in prompts:
        comparisons.append(
            Comparison(
                prompt=prompt,
                before="",
                after=sample_text(
                    model, prepared, device, prompt=prompt, max_new_tokens=80, temperature=0.7
                ).strip(),
            )
        )

    before_ppl = None
    notes: list[str] = []

    if record.mode == "distill":
        report(f"scoring the teacher, {plan.teacher}")
        try:
            before_ppl = _teacher_perplexity(plan, batches, device)
        except Exception as e:
            notes.append(f"Could not score the teacher for comparison: {type(e).__name__}.")

    del model
    torch.cuda.empty_cache()

    result = EvalReport(
        run_id=record.id,
        mode=record.mode,
        before_ppl=before_ppl,
        after_ppl=after_ppl,
        n_examples=sum(b[0].shape[0] for b in batches),
        comparisons=comparisons,
        notes=notes,
    )
    result.notes.extend(_scratch_notes(result, record.mode))
    return result


@torch.no_grad()
def _native_perplexity(model, batches) -> float:
    total = 0.0
    for x, y in batches:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            _, loss = model(x, y)
        total += loss.item()
    return math.exp(min(total / max(1, len(batches)), 20))


@torch.no_grad()
def _teacher_perplexity(plan, batches, device) -> float:
    """The teacher on exactly the tokens the student was measured on.

    Only meaningful because distillation forces both to share a vocabulary.
    """
    import torch.nn.functional as F

    from llmforge.distill.train import load_teacher

    teacher = load_teacher(plan, device)
    total = 0.0
    for x, y in batches:
        logits = teacher(input_ids=x).logits[..., : plan.vocab_size]
        total += F.cross_entropy(
            logits.float().view(-1, logits.size(-1)), y.reshape(-1)
        ).item()
    del teacher
    torch.cuda.empty_cache()
    return math.exp(min(total / max(1, len(batches)), 20))


def _corpus_prompts(prepared, n: int) -> list[str]:
    """Opening fragments of held-out documents, as continuation prompts."""
    prompts: list[str] = []
    for record in prepared.corpus.iter_records():
        text = record.get("text") or ""
        words = text.split()
        if len(words) > 40:
            prompts.append(" ".join(words[8:16]))
        if len(prompts) >= n:
            break
    return prompts or ["Once upon a time"]


def _scratch_notes(report: EvalReport, mode: str) -> list[str]:
    notes: list[str] = []

    if mode == "distill" and report.before_ppl is not None:
        gap = report.after_ppl / report.before_ppl
        notes.append(
            f"Held-out perplexity: teacher {report.before_ppl:.1f}, your student "
            f"{report.after_ppl:.1f} — {gap:.1f}x the teacher's. Both are measured on "
            f"the same tokens and the same vocabulary, so this comparison is fair."
        )
        if gap > 5:
            notes.append(
                "The student is a long way behind its teacher. More data, or more "
                "passes, is the usual remedy."
            )
    else:
        notes.append(
            f"Held-out perplexity is {report.after_ppl:.1f}. There is no 'before' for a "
            f"model trained from scratch — compare it against another run on the same "
            f"corpus, and read the generations."
        )

    notes.append(
        "Perplexity is only comparable between models sharing a tokenizer. A run with "
        "a larger vocabulary will show a higher number without being worse."
    )
    return notes
