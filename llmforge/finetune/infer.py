"""Loading a fine-tuned model back for inference.

A fine-tune produces either a full set of weights or a small adapter over a base
model that has to be fetched again. Both are reconstructed here from the run record,
so callers do not need to know which method produced the run.
"""

from __future__ import annotations

from pathlib import Path

import torch

from llmforge.core import hardware, paths
from llmforge.core.registry import RunRecord
from llmforge.finetune.plan import FinetunePlan
from llmforge.finetune.sft import load_tokenizer


def load(record: RunRecord, checkpoint: str = "best"):
    """Rebuild the model a fine-tuning run produced. Returns (model, tokenizer)."""
    from transformers import AutoModelForCausalLM

    plan = FinetunePlan(**record.plan)
    run_dir = paths.run_dir(record.id)

    tokenizer = load_tokenizer(plan.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        plan.base_model,
        dtype=torch.bfloat16,
        attn_implementation=hardware.attn_implementation(),
    ).cuda()

    if plan.method == "full":
        state = _load_state(run_dir, checkpoint)
        model.load_state_dict(state["task"]["model"])
    else:
        adapter_dir = run_dir / "adapter"
        if adapter_dir.exists():
            # peft's own format, saved at the end of a completed run.
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(adapter_dir))
        else:
            # A cancelled run has checkpoints but no exported adapter yet.
            from peft import LoraConfig, get_peft_model

            model = get_peft_model(
                model,
                LoraConfig(
                    r=plan.lora_rank,
                    lora_alpha=plan.lora_alpha,
                    lora_dropout=plan.lora_dropout,
                    target_modules=plan.lora_targets,
                    bias="none",
                    task_type="CAUSAL_LM",
                ),
            )
            state = _load_state(run_dir, checkpoint)
            model.load_state_dict(state["task"]["adapter"], strict=False)

    model.eval()
    return model, tokenizer


def _load_state(run_dir: Path, checkpoint: str) -> dict:
    path = run_dir / "ckpt" / f"{checkpoint}.pt"
    if not path.exists():
        raise FileNotFoundError(f"no {checkpoint} checkpoint at {path}")
    return torch.load(path, map_location="cuda", weights_only=False)


def generate(
    model,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    supervised: bool = True,
) -> str:
    """Generate a continuation, applying the chat template for instruction models."""
    if supervised:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
    else:
        text = prompt

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
