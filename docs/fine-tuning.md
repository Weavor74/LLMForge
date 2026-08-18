# Fine-tuning

You give it a folder of text and an existing model. It adapts that model to your
data. The result contains the original model's weights plus your changes.

Use this when you want something that follows instructions about *your* data, and
you want a usable result in hours rather than weeks.

## What each size costs

Memory and time for a 5,000-example instruction dataset at 2048-token context. The
method is chosen automatically against the memory budget.

| base | method | memory | trainable | 1x Spark | 8x H100 |
|---|---|---|---|---|---|
| 0.6B | full | 10 GB | 600M | 0.5h | 0.1h |
| 1.5B | full | 25 GB | 1500M | 1.2h | 0.1h |
| 3B | full | 49 GB | 3000M | 2.5h | 0.3h |
| 8B | LoRA | 18 GB | 29M | 6.6h | 0.7h |
| 13B | LoRA | 29 GB | 46M | 10.7h | 1.2h |
| 32B | LoRA | 68 GB | 73M | 26h | 4.8h |
| 40B | LoRA | 86 GB | 110M | 33h | 6.0h |
| 70B | QLoRA | 46 GB | 147M | 4d | 10.4h |

Note the shape of that table: **a 70B fine-tune needs less memory than a 40B one**,
because QLoRA quantizes the frozen base to 4 bits while LoRA keeps it in bf16. It is
also slower, since those weights are dequantized on every forward pass.

## The three methods

| | what trains | when it is chosen |
|---|---|---|
| **full** | every weight | the model and its optimizer state fit — roughly under 3B here |
| **LoRA** | a small low-rank adapter, base frozen | the base fits in bf16 but full optimizer state does not |
| **QLoRA** | the same adapter over a 4-bit base | the base does not fit in bf16 |

The planner takes the highest-fidelity option that fits, not the fastest, because the
run is going to be slow regardless. Override in **Advanced → Method**.

## Two shapes of data

The corpus decides which, and the analysis reports what it detected:

- **Conversations** (jsonl with `messages`, or instruction/output pairs) become
  supervised fine-tuning. Loss is applied only to assistant turns — the model learns
  to produce answers, not to write the questions.
- **Plain prose** becomes continued pretraining. Every token carries loss. This
  teaches domain style and vocabulary, not instruction-following.

## How much data

A few hundred to a few thousand examples is where fine-tuning starts to bite. Below
about 100, expect very little to change.

On small datasets the planner extends the number of passes automatically, because a
conventional three epochs would give too few optimizer steps for the schedule to
accomplish anything. Watch validation loss: when it turns upward while training loss
keeps falling, the model is memorising.

## Verifying it worked

**Did it work?** on the run page scores your model and the model it started from on
the same held-out data, then puts their answers to real questions side by side.

A very large improvement on a small dataset usually means memorisation rather than
learning. The report says so when it sees one.
