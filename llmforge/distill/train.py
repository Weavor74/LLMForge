"""Training a student against a teacher's predictions.

Each step runs the teacher over the batch without gradients, then trains the student
to match its distribution. The loss is the usual pair:

    alpha * T^2 * KL(teacher_T || student_T)  +  (1 - alpha) * CE(student, true tokens)

The T^2 factor is not decoration. Softening the distributions by T shrinks the
gradients by roughly 1/T^2, so without it the distillation term would quietly
lose influence as temperature rises.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from llmforge.core import hardware, loop
from llmforge.core.loop import EventFn
from llmforge.data.prepare import Prepared
from llmforge.distill.plan import DistillPlan
from llmforge.pretrain.data import TokenStream
from llmforge.pretrain.model import ModelConfig, Transformer


def load_teacher(plan: DistillPlan, device: torch.device):
    """Load the teacher for inference only: frozen, eval mode, no gradients."""
    from transformers import AutoModelForCausalLM

    kwargs: dict = {
        "dtype": torch.bfloat16,
        "trust_remote_code": False,
        "attn_implementation": hardware.attn_implementation(),
    }

    if plan.teacher_load_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = {"": 0}

    teacher = AutoModelForCausalLM.from_pretrained(plan.teacher, **kwargs)
    if not plan.teacher_load_4bit:
        teacher.to(device)

    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    teacher.config.use_cache = False
    return teacher


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    temperature: float,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (total, kl_term, ce_term)."""
    # fp32 throughout: a softmax over ~150k vocabulary entries in bf16 loses too much.
    student = student_logits.float()
    teacher = teacher_logits.float()

    student_log_probs = F.log_softmax(student / temperature, dim=-1)
    teacher_probs = F.softmax(teacher / temperature, dim=-1)

    # batchmean, not mean: KL divergence sums over the vocabulary and averages over
    # positions. Using `mean` would divide by the vocabulary size as well and make the
    # term vanishingly small.
    kl = F.kl_div(
        student_log_probs.view(-1, student.size(-1)),
        teacher_probs.view(-1, teacher.size(-1)),
        reduction="batchmean",
    ) * (temperature**2)

    ce = F.cross_entropy(student.view(-1, student.size(-1)), targets.reshape(-1))

    return alpha * kl + (1 - alpha) * ce, kl.detach(), ce.detach()


class DistillTask(loop.TrainingTask):
    def __init__(
        self,
        plan: DistillPlan,
        prepared: Prepared,
        device: torch.device,
        emit: EventFn,
    ):
        self.plan = plan
        self.prepared = prepared
        self.device = device
        self.emit = emit

        emit({"event": "loading_teacher", "teacher": plan.teacher})
        self.teacher = load_teacher(plan, device)

        self.model = Transformer(ModelConfig(**plan.model_kwargs())).to(device)

        self.train_stream = TokenStream(prepared.packed, "train", plan.seq_len, seed=plan.seed)
        self.val_stream = TokenStream(prepared.packed, "val", plan.seq_len, seed=plan.seed)

        self._check_vocabularies()
        self._fit_micro_batch()

        self.val_batches = self.val_stream.deterministic_batches(
            min(plan.micro_batch, 4), plan.eval_batches, device
        )

    def _check_vocabularies(self) -> None:
        """The two models must agree on what token 5000 means.

        A mismatch here does not crash — it silently trains the student against
        distributions over a different vocabulary, producing nonsense.
        """
        teacher_vocab = self.teacher.get_output_embeddings().weight.shape[0]
        student_vocab = self.model.cfg.vocab_size

        if student_vocab > teacher_vocab:
            raise ValueError(
                f"student vocabulary ({student_vocab:,}) exceeds the teacher's "
                f"({teacher_vocab:,}); they must share a tokenizer"
            )
        if student_vocab != teacher_vocab:
            # Common and harmless: many checkpoints pad the embedding matrix past the
            # tokenizer's real vocabulary. Truncating the teacher's logits is correct.
            self.emit({
                "event": "vocab_note",
                "message": (
                    f"teacher embedding is {teacher_vocab:,} wide against a "
                    f"{student_vocab:,}-token vocabulary; extra logits are dropped"
                ),
            })

    def _teacher_logits(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = self.teacher(input_ids=x).logits
        return out[..., : self.model.cfg.vocab_size]

    def _fit_micro_batch(self) -> None:
        plan = self.plan
        micro_batch = plan.micro_batch

        while micro_batch >= 1:
            try:
                x, y = self.train_stream.batch(micro_batch, step=0, device=self.device)
                teacher_logits = self._teacher_logits(x)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    student_logits, _ = self.model(x)
                loss, _, _ = distillation_loss(
                    student_logits, teacher_logits, y,
                    temperature=plan.temperature, alpha=plan.alpha,
                )
                loss.backward()
                self.model.zero_grad(set_to_none=True)
                del teacher_logits, student_logits, loss
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
                "Cannot fit teacher and student together for even one sequence. Use a "
                "shorter context, a smaller teacher, or a smaller student."
            )

        if micro_batch != plan.micro_batch:
            plan.grad_accum = max(1, round(plan.tokens_per_step / (micro_batch * plan.seq_len)))
            plan.micro_batch = micro_batch
            plan.tokens_per_step = micro_batch * plan.seq_len * plan.grad_accum
            self.emit({
                "event": "plan_adjusted",
                "micro_batch": micro_batch,
                "grad_accum": plan.grad_accum,
            })

    # -- the task interface ---------------------------------------------------

    def train_loss(self, step: int, micro: int) -> torch.Tensor:
        plan = self.plan
        x, y = self.train_stream.batch(
            plan.micro_batch, step * plan.grad_accum + micro, self.device
        )
        teacher_logits = self._teacher_logits(x)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            student_logits, _ = self.model(x)

        loss, kl, ce = distillation_loss(
            student_logits, teacher_logits, y,
            temperature=plan.temperature, alpha=plan.alpha,
        )
        # Kept for reporting: the split between the two terms is the most useful
        # diagnostic when a distillation run is not converging.
        self.last_kl = float(kl)
        self.last_ce = float(ce)
        return loss

    @torch.no_grad()
    def eval_loss(self) -> float:
        """Cross-entropy against true tokens, not the distillation objective.

        The student is measured on the same quantity a from-scratch model would be,
        so the two are directly comparable.
        """
        self.model.eval()
        total = 0.0
        for x, y in self.val_batches:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = self.model(x, y)
            total += loss.item()
        self.model.train()
        return total / max(1, len(self.val_batches))

    def trainable_parameters(self):
        return self.model.parameters()

    def checkpoint_state(self) -> dict:
        # Only the student. The teacher is an input, reloadable from its source.
        return {"model": self.model.state_dict()}

    def restore(self, state: dict) -> None:
        self.model.load_state_dict(state["model"])

    def sample(self) -> str:
        from llmforge.pretrain.train import sample_text

        return sample_text(self.model, self.prepared, self.device, max_new_tokens=128)


def train(
    *,
    run_id: str,
    plan: DistillPlan,
    prepared: Prepared,
    run_dir: Path,
    emit: EventFn | None = None,
    resume: bool = False,
    should_stop=None,
) -> dict:
    """Distil a teacher into a student, to completion."""
    emit = emit or (lambda _: None)
    device = torch.device("cuda")

    torch.manual_seed(plan.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    task = DistillTask(plan, prepared, device, emit)
    optimizer = loop.build_optimizer(
        task, plan, param_groups=task.model.param_groups(plan.weight_decay)
    )

    return loop.run(
        run_id=run_id,
        plan=plan,
        task=task,
        optimizer=optimizer,
        run_dir=run_dir,
        emit=emit,
        resume=resume,
        should_stop=should_stop,
    )
