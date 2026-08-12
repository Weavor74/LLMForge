# LLMForge

Point at a folder, get a model.

Select a folder of corpus data and LLMForge builds a language model from scratch.
Select a folder *and* an existing model, and it fine-tunes that model instead.
Same pipeline, same config objects, same reproducibility guarantees either way.

Developed on an NVIDIA DGX Spark (GB10 Grace-Blackwell, aarch64, 128 GB unified
memory), and built to move: the planner measures whatever machine it is on and adapts,
including across several GPUs.

## Why it works this way

Every run is fully described by a resolved config, and the GUI is a config editor —
never a place where state hides. That means anything you build by clicking, you can
rebuild from the command line, and `llmforge repro <run_id>` recreates any past run
from its lockfile.

Defaults are *derived*, not typed. A planner reads your corpus, counts the tokens,
and picks the architecture, batch size, gradient accumulation, and schedule that the
data and the hardware jointly justify — then tells you how long it will take and how
good to expect the result to be, before spending the compute.

## Three things it can build

All three start the same way — you pick a folder — and differ only in where the
training signal comes from.

| | You give it | You get | Use when |
|---|---|---|---|
| **From scratch** | a corpus folder | a model you built, tokenizer and all | you want your own model and have a lot of text |
| **Fine-tune** | a folder + a model | that model, adapted to your data | you want something that follows instructions about your data |
| **Distil** | a folder + a teacher | a *smaller* model that imitates the teacher | you want the teacher's behaviour at a fraction of the size |

## Status

- [x] **Phase 0** — locked CUDA 13 / aarch64 environment, `llmforge doctor` preflight
- [x] **Phase 1** — folder ingest → clean → tokenizer → packed shards
- [x] **Phase 2** — planner + from-scratch transformer + training loop
- [x] **Phase 3** — fine-tune path (full / LoRA / QLoRA, auto-selected)
- [x] **Phase 4** — desktop app: FastAPI + React GUI, subprocess job runner
- [x] **Distillation** — teacher-guided training of a smaller student
- [x] **Phase 5** — eval harness, GGUF/safetensors export, checkpoint retention
- [x] **Scale** — multi-GPU (DDP/FSDP), gradient checkpointing, 8-bit Adam, tiers to 5.8B
- [x] **Portability** — hardware auto-detected and re-measured when the machine changes

## Install it as an app

```bash
uv sync                       # Python side, exact versions from uv.lock
uv run llmforge doctor        # check this machine can actually train
uv run llmforge install-app   # add it to your applications menu
```

`uv sync --extra finetune` additionally installs peft, trl and bitsandbytes, which the
fine-tuning and QLoRA paths need.

The built GUI is committed, so no Node toolchain is required to *run* LLMForge — only
to change the frontend:

```bash
cd web && npm install && npm run build
```

LLMForge then appears in your applications menu. Launching it starts the server and
opens its own window; closing the window puts it away again.

**Training is unaffected by any of that.** Runs are separate processes, so you can quit
the app and reopen it later to watch a run that is still going.

From a terminal instead: `llmforge app`, or `llmforge serve` for a plain server.

## Or from the command line

```bash
# Build a model from scratch out of a folder
llmforge create /path/to/corpus --dry-run              # see the plan, spend nothing
llmforge create /path/to/corpus                        # build it

# Fine-tune an existing model on the same folder
llmforge create /path/to/corpus --base Qwen/Qwen3-8B

# Distil a big model into a small one
llmforge create /path/to/corpus --teacher Qwen/Qwen3-8B

llmforge runs                                          # what has been trained
llmforge sample last -p "How do I ..."                 # talk to it
llmforge resume last                                   # continue an interrupted run

llmforge eval last                                     # did it actually change anything?
llmforge export last -f gguf -q q8_0                   # take it elsewhere
llmforge export --list -f gguf                         # what levels are available
```

## Getting the model out

```bash
llmforge export <run> --format gguf --quantize q8_0
llmforge export <run> --format safetensors
```

**safetensors** produces a directory `transformers` loads directly as a
`LlamaForCausalLM` — from-scratch models included, since they are architecturally
Llama (RoPE, RMSNorm, SwiGLU, GQA). A test asserts the exported model reproduces the
original's logits exactly, because a conversion that merely *loads* is the dangerous
outcome: a mis-split QKV or a mismatched rotary layout yields fluent nonsense.

**GGUF** produces a single file for llama.cpp or Ollama. f16, q8_0, q5_1, q5_0, q4_1
and q4_0 are written directly. The k-quants (q4_k_m and friends) need llama.cpp's
`llama-quantize` on your PATH and are offered only when it is present, rather than
listed and then failing.

## Did it work?

```bash
llmforge eval <run>
```

Scores the model on held-out data from your own folder and — for fine-tunes and
distillations — scores what it started from on exactly the same data, then puts their
answers to real questions from your corpus side by side.

Two honest limits, stated because they are easy to misread:

- **Perplexity only compares models sharing a tokenizer.** A distilled model on a
  49k-token vocabulary will show a higher number than a from-scratch model on a 4k
  one without being worse. Compare generations across those.
- **Large improvements on small datasets are often memorisation.** The report says so
  when it sees one.

## Moving to a bigger machine

Nothing is hardcoded to the machine this was written on. The hardware profile is
measured, fingerprinted, and re-measured automatically when the GPU underneath it
changes, so a workspace copied elsewhere plans against the new box rather than the old
one. The planner then adapts three ways:

- **How to fit.** It escalates only as far as needed — plain training, then gradient
  checkpointing, then an 8-bit optimizer — so a model that fits comfortably pays for
  none of it. This is what lets a 5.8B model train on a single 128 GB device at all.
- **How many devices.** One GPU runs as before. Several run under DDP where a full
  replica fits per device, and FSDP where it does not, which is the only way to train
  a model larger than one GPU holds. Launched automatically via `torchrun`.
- **What that costs.** Memory budgets follow the device — a dedicated card may be
  filled further than unified memory shared with your desktop.

The same corpus and tier, planned on three machines:

| | 1.18B | 3.3B | 5.8B |
|---|---|---|---|
| DGX Spark (1x GB10) | 109 d | 2.5 y | 9.7 y, checkpointed |
| 2x DGX Spark | 61 d, DDP | 1.4 y | 5.4 y |
| 8x H100 80GB | **40 h**, DDP | 14 d | 66 d, FSDP |

Which is the honest summary of training large models from scratch: memory is solvable,
wall-clock is not. Fine-tuning and distilling large models is the practical path on one
machine — LoRA on 8–32B and QLoRA on 70B all work today.

**Multi-GPU is written but untested**: this machine has one GPU, so only the world-size-one
path has actually been run. The arithmetic that chooses a strategy is unit-tested;
the scaling is not.

## On reproducibility

`run.lock.json` records everything that determined a result: corpus content hash,
tokenizer, the full plan, package versions, GPU, driver, and seed. Re-running from it
gives you the same configuration on the same data.

It does **not** give bit-identical weights. GPU reductions are non-deterministic in
their ordering, so two identical runs land close but not equal — around 1% apart in
validation loss in practice. Making that exact costs 10–20% throughput, which is a bad
trade on a bandwidth-bound machine, so it is deliberately not done. What is
reproducible is the recipe, not the floating-point noise.

`create` walks the folder, extracts text from every format it recognises, drops junk
and duplicates, and shows you the plan — model shape, step count, memory, projected
wall-clock, and what quality to expect — before training anything.

Without a model it trains a tokenizer and a transformer from scratch, sized to what
the corpus can actually support. With `--base` it fine-tunes that model instead,
choosing full / LoRA / QLoRA from the memory budget, and choosing between supervised
fine-tuning and continued pretraining based on whether the folder holds conversations
or prose. With `--teacher` it builds a smaller model and trains it against the
teacher's predicted distribution rather than against the text alone.

All three write the same run records and lockfiles, and all three are available
identically from the GUI and the CLI.

## Setup

```bash
uv sync
uv run llmforge doctor
```

`doctor` verifies the parts of the stack that quietly break on sm_121 aarch64: the
CUDA build of torch, bf16 tensor cores, a fused SDPA backend, and `torch.compile`.
It also measures achieved TFLOP/s and memory bandwidth, which the planner uses to
produce honest wall-clock estimates.

For the fine-tuning path:

```bash
uv sync --extra finetune
```

## Layout

```
llmforge/
  core/       config schemas, planner, run registry, memory budgeting
  data/       folder ingest, format detection, cleaning, tokenized packing
  tokenizer/  BPE training
  pretrain/   from-scratch transformer + training loop
  finetune/   SFT: full, LoRA, QLoRA
  distill/    teacher-guided training of a smaller student
  jobs/       subprocess job runner + worker
  eval/       held-out perplexity, before/after comparison
  serve/      load a produced model and chat with it
  jobs/       subprocess run manager
  api/        FastAPI + WebSocket
web/          Vite + React + TypeScript GUI
workspace/    all derived state (override with LLMFORGE_HOME)
```
