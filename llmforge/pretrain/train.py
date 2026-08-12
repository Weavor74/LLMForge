"""Pretraining as a task over the shared loop.

Everything specific to training a model from scratch lives here: building it, feeding
it packed token shards, and sampling from it. The scheduling, checkpointing and
cancellation machinery is in `llmforge.core.loop`.
"""

from __future__ import annotations

from pathlib import Path

import torch

from llmforge.core import distributed as dist
from llmforge.core import loop
from llmforge.core.loop import EventFn
from llmforge.core.planner import TrainPlan
from llmforge.data.prepare import Prepared
from llmforge.pretrain.data import TokenStream
from llmforge.pretrain.model import ModelConfig, Transformer

# Re-exported: several callers and tests reach for these through this module.
lr_at = loop.lr_at


def build_model(plan: TrainPlan, device: torch.device) -> Transformer:
    model = Transformer(ModelConfig(**plan.model_kwargs()))
    model.to(device)
    return model


class PretrainTask(loop.TrainingTask):
    def __init__(
        self,
        plan: TrainPlan,
        prepared: Prepared,
        device: torch.device,
        emit: EventFn,
    ):
        self.plan = plan
        self.prepared = prepared
        self.device = device
        self.emit = emit

        self.model = build_model(plan, device)
        self.train_stream = TokenStream(prepared.packed, "train", plan.seq_len, seed=plan.seed)
        self.val_stream = TokenStream(prepared.packed, "val", plan.seq_len, seed=plan.seed)

        self.strategy = getattr(plan, "strategy", "single")

        # The planner's memory model is an approximation; this is the real verdict.
        # Run it before wrapping, so an out-of-memory event is a plain one rather than
        # a collective failure that hangs the other ranks.
        self._fit_micro_batch()

        self.model = dist.wrap_model(self.model, self.strategy, device)
        self.step_model = self.model
        self.val_batches = self.val_stream.deterministic_batches(
            min(plan.micro_batch, 8), plan.eval_batches, device
        )

    # -- construction helpers -------------------------------------------------

    def _fit_micro_batch(self) -> None:
        """Halve the micro-batch until a real forward/backward succeeds.

        A run should either start correctly or fail immediately, never die an hour in.
        """
        plan = self.plan
        micro_batch = plan.micro_batch

        while micro_batch >= 1:
            try:
                x, y = self.train_stream.batch(micro_batch, step=0, device=self.device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, loss = self.model(x, y)
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
                "Cannot fit even a single sequence in memory. Reduce the context "
                "length or choose a smaller tier."
            )

        if micro_batch != plan.micro_batch:
            # Preserve tokens-per-step: fewer sequences per pass, more passes per step.
            plan.grad_accum = max(1, round(plan.tokens_per_step / (micro_batch * plan.seq_len)))
            plan.micro_batch = micro_batch
            plan.tokens_per_step = micro_batch * plan.seq_len * plan.grad_accum
            self.emit({
                "event": "plan_adjusted",
                "micro_batch": micro_batch,
                "grad_accum": plan.grad_accum,
            })

    def on_train_begin(self) -> None:
        if not self.plan.compile:
            return
        if getattr(self.plan, "gradient_checkpointing", False):
            # Compiling around a checkpointed graph is slow to build and prone to
            # recompilation; the memory is why checkpointing was turned on.
            self.emit({
                "event": "compile_skipped",
                "message": "gradient checkpointing is on; training eagerly",
            })
            return
        try:
            self.step_model = torch.compile(self.model)
            self.emit({"event": "compiling"})
            # Force compilation now, at the real training shape.
            x, y = self.train_stream.batch(self.plan.micro_batch, step=0, device=self.device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = self.step_model(x, y)
            loss.backward()
            self.model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        except Exception as e:
            self.emit({"event": "compile_failed", "message": str(e)[:200]})
            self.step_model = self.model

    # -- the task interface ---------------------------------------------------

    def _batch_index(self, step: int, micro: int) -> int:
        """A batch index unique to this rank.

        Interleaving by rank means the ranks of one step see disjoint data; without
        it every GPU would compute the same gradient and the run would be N times the
        cost for no benefit.
        """
        within_step = step * self.plan.grad_accum + micro
        return within_step * dist.world_size() + dist.rank()

    def train_loss(self, step: int, micro: int) -> torch.Tensor:
        x, y = self.train_stream.batch(
            self.plan.micro_batch, self._batch_index(step, micro), self.device
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = self.step_model(x, y)
        return loss

    @torch.no_grad()
    def eval_loss(self) -> float:
        self.model.eval()
        total = 0.0
        for x, y in self.val_batches:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = self.model(x, y)
            total += loss.item()
        self.model.train()
        # Every rank evaluates the same fixed batches, so the mean is already shared;
        # averaging keeps it exact if that ever stops being true.
        return dist.all_reduce_mean(total / max(1, len(self.val_batches)), self.device)

    def trainable_parameters(self):
        return self.model.parameters()

    def clip_gradients(self, max_norm: float) -> float:
        if self.strategy == "fsdp" and dist.is_distributed():
            # Gradients are sharded; only FSDP's own method sees the global norm.
            return float(self.model.clip_grad_norm_(max_norm))
        return super().clip_gradients(max_norm)

    def checkpoint_state(self) -> dict:
        return {"model": dist.gather_state_dict(self.model, self.strategy)}

    def restore(self, state: dict) -> None:
        dist.unwrap(self.model).load_state_dict(state["model"])

    def sample(self) -> str:
        # Generation walks the module directly, which a DDP or FSDP wrapper does not
        # support outside a forward pass.
        return sample_text(
            dist.unwrap(self.model), self.prepared, self.device, max_new_tokens=128
        )


def train(
    *,
    run_id: str,
    plan: TrainPlan,
    prepared: Prepared,
    run_dir: Path,
    emit: EventFn | None = None,
    resume: bool = False,
    should_stop=None,
) -> dict:
    """Build a from-scratch model and train it to completion."""
    emit = emit or (lambda _: None)
    device = dist.setup()

    # Same seed on every rank: the model must start identical everywhere, and batch
    # divergence comes from the rank-interleaved index, not from the seed.
    torch.manual_seed(plan.seed)
    # TF32 costs nothing here and helps the few fp32 matmuls that remain.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    task = PretrainTask(plan, prepared, device, emit)
    optimizer = loop.build_optimizer(
        task, plan, param_groups=dist.unwrap(task.model).param_groups(plan.weight_decay)
    )

    try:
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
    finally:
        dist.cleanup()


def sample_text(
    model: Transformer,
    prepared: Prepared,
    device: torch.device,
    *,
    prompt: str = "",
    max_new_tokens: int = 128,
    temperature: float = 0.8,
) -> str:
    """Generate a short sample, for eyeballing progress during training."""
    from llmforge.tokenizer.train import END_OF_TEXT

    tokenizer = prepared.tokenizer
    eos_id = tokenizer.token_to_id(END_OF_TEXT)

    ids = tokenizer.encode(prompt, add_special_tokens=False).ids if prompt else [eos_id]
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    was_training = model.training
    out = model.generate(idx, max_new_tokens, temperature=temperature, eos_id=None)
    if was_training:
        model.train()

    # Return only what was generated. `generate` returns prompt + continuation, and
    # callers print the prompt themselves; without this the prompt appears twice.
    generated = out[0].tolist()[len(ids) :]
    return tokenizer.decode(generated)
