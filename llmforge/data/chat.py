"""Turning conversations into supervised examples.

The important part is loss masking. A conversation contains both the user's turn and
the assistant's, and training on all of it teaches the model to write questions as
readily as answers. Only assistant tokens should carry loss; everything else is
context and is masked out.

Getting this wrong is invisible — the loss curve looks fine — and produces a model
that rambles in the user's voice. So the masking is derived by re-rendering each
prefix through the model's own chat template rather than by matching strings.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Ignored by cross-entropy; the conventional sentinel for "no target here".
IGNORE_INDEX = -100

# Used when a base model's tokenizer ships no chat template of its own.
CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


@dataclass
class Example:
    """One tokenized training example."""

    input_ids: np.ndarray
    labels: np.ndarray

    def __len__(self) -> int:
        return len(self.input_ids)

    @property
    def n_supervised(self) -> int:
        return int((self.labels != IGNORE_INDEX).sum())


def ensure_chat_template(tokenizer) -> None:
    """Give the tokenizer a chat template if it lacks one."""
    if getattr(tokenizer, "chat_template", None):
        return
    tokenizer.chat_template = CHATML_TEMPLATE


def _render(tokenizer, messages: list[dict], add_generation_prompt: bool = False) -> str:
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt
    )


def build_example(tokenizer, messages: list[dict[str, str]], max_len: int) -> Example | None:
    """Tokenize a conversation, supervising only the assistant's turns.

    Works by rendering the conversation one turn at a time: the tokens that appear
    when an assistant turn is added, minus those already present, are exactly that
    turn's tokens. This is template-agnostic, so it stays correct for any base model
    rather than assuming ChatML's particular markers.
    """
    if not messages:
        return None

    ids: list[int] = []
    labels: list[int] = []
    previous_len = 0

    for i, message in enumerate(messages):
        prefix = messages[: i + 1]
        is_assistant = message.get("role") == "assistant"

        # For an assistant turn, the generation prompt belongs to the *context*, so
        # render the preceding turns with it appended and treat that as unsupervised.
        if is_assistant:
            context_text = _render(tokenizer, messages[:i], add_generation_prompt=True)
            context_ids = tokenizer(context_text, add_special_tokens=False)["input_ids"]

            full_text = _render(tokenizer, prefix)
            full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

            # The generation prompt's tokens are context; what follows is the answer.
            unsupervised = context_ids[previous_len:]
            answer = full_ids[len(context_ids) :]

            ids.extend(unsupervised)
            labels.extend([IGNORE_INDEX] * len(unsupervised))
            ids.extend(answer)
            labels.extend(answer)
            previous_len = len(full_ids)
        else:
            full_ids = tokenizer(_render(tokenizer, prefix), add_special_tokens=False)[
                "input_ids"
            ]
            added = full_ids[previous_len:]
            ids.extend(added)
            labels.extend([IGNORE_INDEX] * len(added))
            previous_len = len(full_ids)

    if not ids:
        return None

    # Truncate from the left: the most recent exchange is the one worth keeping.
    if len(ids) > max_len:
        ids = ids[-max_len:]
        labels = labels[-max_len:]

    example = Example(
        input_ids=np.array(ids, dtype=np.int64), labels=np.array(labels, dtype=np.int64)
    )

    # An example with nothing to learn from is worse than useless: it contributes a
    # NaN to the mean loss when every position is masked.
    return example if example.n_supervised > 0 else None


def build_text_example(tokenizer, text: str, max_len: int) -> Example | None:
    """A raw-text example for continued pretraining, where everything is supervised."""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not ids:
        return None
    ids = ids[:max_len]
    array = np.array(ids, dtype=np.int64)
    return Example(input_ids=array, labels=array.copy())


def collate(
    examples: list[Example], pad_id: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pad a batch to its longest example. Returns (input_ids, labels, attention_mask)."""
    width = max(len(e) for e in examples)
    n = len(examples)

    input_ids = np.full((n, width), pad_id, dtype=np.int64)
    labels = np.full((n, width), IGNORE_INDEX, dtype=np.int64)
    attention = np.zeros((n, width), dtype=np.int64)

    for i, example in enumerate(examples):
        length = len(example)
        # Right padding: with an explicit attention mask the model never attends to
        # the padding, and the loss ignores it via IGNORE_INDEX.
        input_ids[i, :length] = example.input_ids
        labels[i, :length] = example.labels
        attention[i, :length] = 1

    return input_ids, labels, attention
