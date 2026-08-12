"""Fine-tuning a base model on the user's corpus.

Handles both shapes the corpus can take: supervised conversations, where only the
assistant's tokens carry loss, and raw text, where the job is continued pretraining
and everything does. The adaptation method — full, LoRA, or QLoRA — comes from the
plan, which chose it against the memory budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from llmforge.core import hardware, loop
from llmforge.core.loop import EventFn
from llmforge.data import chat
from llmforge.data.ingest import Corpus
from llmforge.finetune.plan import FinetunePlan

# Held-out share of examples. Fine-tuning corpora are small, so this is proportionally
# larger than the pretraining split.
VAL_FRACTION = 0.05
MIN_VAL_EXAMPLES = 4


@dataclass
class Dataset:
    """Tokenized examples, split into train and validation."""

    train: list[chat.Example]
    val: list[chat.Example]
    pad_id: int
    # The conversations behind the validation examples, kept verbatim. Recovering a
    # question by decoding the masked tokens back out means parsing whatever chat
    # template the base model uses, which differs per model and yields role markers
    # rather than questions. Keeping the source is exact and template-independent.
    val_sources: list[list[dict]] = field(default_factory=list)

    def truncate(self, max_len: int) -> None:
        """Clamp every example to `max_len`, keeping the tail.

        Examples are tokenized at the base model's ceiling so the planner can see the
        real length distribution; once it has chosen a working context, anything
        longer would blow past the memory the plan was budgeted against. Truncating
        from the left keeps the most recent exchange, which is the supervised part.
        """
        for split in (self.train, self.val):
            for i, example in enumerate(split):
                if len(example) > max_len:
                    split[i] = chat.Example(
                        input_ids=example.input_ids[-max_len:],
                        labels=example.labels[-max_len:],
                    )

    @property
    def n_tokens(self) -> int:
        return sum(len(e) for e in self.train)

    @property
    def p95_length(self) -> int:
        if not self.train:
            return 512
        return int(np.percentile([len(e) for e in self.train], 95))


def load_tokenizer(base_ref: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_ref, trust_remote_code=False)
    chat.ensure_chat_template(tokenizer)

    if tokenizer.pad_token_id is None:
        # Padding is masked out of the loss and the attention, so reusing EOS is safe
        # and avoids resizing the embedding for a token that is never predicted.
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_dataset(
    corpus: Corpus,
    tokenizer,
    *,
    supervised: bool,
    max_len: int,
    progress=None,
) -> Dataset:
    """Tokenize the corpus into supervised examples using the base model's tokenizer."""
    examples: list[chat.Example] = []
    sources: list[list[dict]] = []

    for i, record in enumerate(corpus.iter_records()):
        if "messages" in record:
            example = chat.build_example(tokenizer, record["messages"], max_len)
        elif supervised:
            # A prose document inside an instruction corpus has no assistant turn to
            # supervise; skip it rather than train on unlabelled text.
            example = None
        else:
            example = chat.build_text_example(tokenizer, record.get("text", ""), max_len)

        if example is not None:
            examples.append(example)
            sources.append(record.get("messages") or [])

        if progress and i % 200 == 0:
            progress("tokenizing", i, corpus.analysis.n_documents)

    if not examples:
        raise ValueError(
            "No usable training examples. For fine-tuning, the corpus needs either "
            "conversations (jsonl with messages / instruction+output) or raw text."
        )

    # Deterministic split by position, after a fixed shuffle so the validation set is
    # not simply whichever files sorted last.
    rng = np.random.default_rng(0)
    order = rng.permutation(len(examples))
    n_val = max(MIN_VAL_EXAMPLES, int(len(examples) * VAL_FRACTION))
    n_val = min(n_val, max(1, len(examples) // 5))

    val_idx = set(order[:n_val].tolist())
    train = [e for i, e in enumerate(examples) if i not in val_idx]
    val = [e for i, e in enumerate(examples) if i in val_idx]
    val_sources = [m for i, m in enumerate(sources) if i in val_idx]

    return Dataset(
        train=train, val=val, pad_id=tokenizer.pad_token_id, val_sources=val_sources
    )


def load_model(plan: FinetunePlan, device: torch.device):
    """Load the base model and attach the adaptation the plan chose."""
    from transformers import AutoModelForCausalLM

    kwargs: dict = {
        "dtype": torch.bfloat16,
        "trust_remote_code": False,
        "attn_implementation": hardware.attn_implementation(),
    }

    if plan.method == "qlora":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            # Quantizing the quantization constants too. Saves a little more memory
            # at no measurable quality cost.
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = {"": 0}

    model = AutoModelForCausalLM.from_pretrained(plan.base_model, **kwargs)

    if plan.method != "qlora":
        model.to(device)

    if plan.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        # Without this, checkpointed activations have no grad_fn and the backward
        # pass silently produces no gradients for LoRA parameters.
        model.enable_input_require_grads()

    if plan.method == "full":
        model.config.use_cache = False
        return model

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if plan.method == "qlora":
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=plan.gradient_checkpointing
        )

    # Only adapt projections the architecture actually has; target names vary.
    present = {name.split(".")[-1] for name, _ in model.named_modules()}
    targets = [t for t in plan.lora_targets if t in present]
    if not targets:
        raise ValueError(
            f"None of the expected projection names exist in {plan.architecture}. "
            f"Override --lora-targets with names from this model."
        )

    model = get_peft_model(
        model,
        LoraConfig(
            r=plan.lora_rank,
            lora_alpha=plan.lora_alpha,
            lora_dropout=plan.lora_dropout,
            target_modules=targets,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.config.use_cache = False
    return model


class FinetuneTask(loop.TrainingTask):
    def __init__(
        self,
        plan: FinetunePlan,
        dataset: Dataset,
        tokenizer,
        device: torch.device,
        emit: EventFn,
    ):
        self.plan = plan
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device
        self.emit = emit

        self.model = load_model(plan, device)
        self.model.train()

        self._fit_micro_batch()
        self.val_batches = self._make_val_batches()

    # -- construction helpers -------------------------------------------------

    def _batch_tensors(self, examples: list[chat.Example]):
        input_ids, labels, attention = chat.collate(examples, self.dataset.pad_id)
        return (
            torch.from_numpy(input_ids).to(self.device),
            torch.from_numpy(labels).to(self.device),
            torch.from_numpy(attention).to(self.device),
        )

    def _longest(self, n: int) -> list[chat.Example]:
        """The n longest training examples — the worst case for the fit-check."""
        return sorted(self.dataset.train, key=len, reverse=True)[:n]

    def _fit_micro_batch(self) -> None:
        plan = self.plan
        micro_batch = plan.micro_batch

        while micro_batch >= 1:
            try:
                # Fit against the longest examples, not a random sample: a batch that
                # fits on average and dies on the tail is worse than one that fails now.
                batch = self._longest(micro_batch)
                loss = self._forward(batch)
                loss.backward()
                self.model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                break
            except torch.cuda.OutOfMemoryError:
                self.model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                micro_batch //= 2
                self.emit({
                    "event": "fit_check",
                    "message": f"out of memory, retrying at micro-batch {micro_batch}",
                })
        else:
            raise RuntimeError(
                f"Cannot fit a single sequence of {plan.seq_len} tokens for "
                f"{plan.base_model}. Use a shorter --seq-len or a smaller base model."
            )

        if micro_batch != plan.micro_batch:
            plan.grad_accum = max(1, round(plan.micro_batch * plan.grad_accum / micro_batch))
            plan.micro_batch = micro_batch
            plan.tokens_per_step = micro_batch * plan.grad_accum * plan.seq_len
            self.emit({
                "event": "plan_adjusted",
                "micro_batch": micro_batch,
                "grad_accum": plan.grad_accum,
            })

    def _make_val_batches(self) -> list[list[chat.Example]]:
        size = self.plan.micro_batch
        val = self.dataset.val
        return [val[i : i + size] for i in range(0, len(val), size)] or [self.dataset.train[:1]]

    # -- forward --------------------------------------------------------------

    def _forward(self, examples: list[chat.Example]) -> torch.Tensor:
        input_ids, labels, attention = self._batch_tensors(examples)
        out = self.model(input_ids=input_ids, attention_mask=attention, labels=labels)
        return out.loss

    def _train_batch(self, step: int, micro: int) -> list[chat.Example]:
        """Sample without replacement within an epoch, reshuffling between epochs."""
        plan = self.plan
        index = step * plan.grad_accum + micro
        per_epoch = max(1, len(self.dataset.train) // plan.micro_batch)
        epoch, position = divmod(index, per_epoch)

        rng = np.random.default_rng((plan.seed, epoch))
        order = rng.permutation(len(self.dataset.train))
        start = position * plan.micro_batch
        picks = order[start : start + plan.micro_batch]
        return [self.dataset.train[i] for i in picks]

    # -- the task interface ---------------------------------------------------

    def train_loss(self, step: int, micro: int) -> torch.Tensor:
        return self._forward(self._train_batch(step, micro))

    @torch.no_grad()
    def eval_loss(self) -> float:
        self.model.eval()
        total = 0.0
        for batch in self.val_batches:
            total += self._forward(batch).item()
        self.model.train()
        return total / max(1, len(self.val_batches))

    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def checkpoint_state(self) -> dict:
        if self.plan.method == "full":
            return {"model": self.model.state_dict()}
        # Only the adapter is worth saving: the frozen base is reloadable from its
        # source, and storing it again would multiply checkpoint size by a hundred.
        return {
            "adapter": {
                k: v.detach().cpu()
                for k, v in self.model.state_dict().items()
                if "lora_" in k
            }
        }

    def restore(self, state: dict) -> None:
        if "model" in state:
            self.model.load_state_dict(state["model"])
        else:
            self.model.load_state_dict(state["adapter"], strict=False)

    @torch.no_grad()
    def sample(self) -> str:
        self.model.eval()
        try:
            if self.plan.supervised:
                messages = [{"role": "user", "content": "Summarise what you have learned."}]
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                text = self.tokenizer.bos_token or ""

            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            out = self.model.generate(
                **inputs,
                max_new_tokens=96,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=self.dataset.pad_id,
            )
            generated = out[0][inputs["input_ids"].shape[1] :]
            return self.tokenizer.decode(generated, skip_special_tokens=True)
        except Exception as e:
            # Sampling is a convenience; never let it take down a training run.
            return f"(sampling failed: {type(e).__name__})"
        finally:
            self.model.train()


def train(
    *,
    run_id: str,
    plan: FinetunePlan,
    dataset: Dataset,
    tokenizer,
    run_dir: Path,
    emit: EventFn | None = None,
    resume: bool = False,
    should_stop=None,
) -> dict:
    """Fine-tune a base model to completion."""
    emit = emit or (lambda _: None)
    device = torch.device("cuda")

    torch.manual_seed(plan.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    task = FinetuneTask(plan, dataset, tokenizer, device, emit)
    optimizer = loop.build_optimizer(task, plan)

    summary = loop.run(
        run_id=run_id,
        plan=plan,
        task=task,
        optimizer=optimizer,
        run_dir=run_dir,
        emit=emit,
        resume=resume,
        should_stop=should_stop,
    )

    # Save the adapter in peft's own format so it loads anywhere, not just here.
    if plan.method != "full" and summary["status"] != "failed":
        adapter_dir = run_dir / "adapter"
        task.model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        summary["adapter_dir"] = str(adapter_dir)

    return summary
