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

## Install

```bash
git clone https://github.com/Weavor74/LLMForge.git && cd LLMForge
bash scripts/setup.sh
```

That is the whole thing on any machine. The script inspects what it landed on, picks a
torch build the driver can actually run, installs, and runs the preflight.

**Nothing needs root.** `uv` goes to `~/.local/bin`, the virtual environment lives in
the project directory, and every dependency is a wheel. The only system-level
requirement is an NVIDIA driver, which is already present on any machine with GPUs.

### Moving it to another machine

Clone and run the setup script. Three things adapt on their own:

- **Architecture.** The dependency set resolves on x86_64 and aarch64 alike; the
  lockfile is universal rather than pinned to the machine that produced it.
- **CUDA.** The lockfile pins a CUDA 13 build, which needs driver 580+. On an older
  driver the script installs the cu126 build of the same torch version instead — the
  code is identical, only the wheel differs.
- **Hardware.** The planner measures the GPU it finds and re-measures automatically
  when the fingerprint changes, so a workspace copied from elsewhere plans against the
  machine it is on rather than the one it came from.

A machine with no GPU still installs and runs. It can ingest a corpus, train a
tokenizer and produce a full plan — it simply cannot train, and the preflight says so.

If the preflight reports no CUDA, it tells you which of the three causes it is: no
driver, a CPU-only wheel, or a driver too old for the build — with the fix for each.

### As a desktop application

```bash
uv run llmforge install-app   # adds it to your applications menu
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

### What each size costs

Training from scratch is bounded by two things, and memory is the easier one. These
are Chinchilla-optimal token budgets (~20 per parameter) and this planner's own
estimates, at a 32k vocabulary:

| tier | params | tokens | ≈ text | train memory | 1x Spark | 8x H100 | 64x H100 |
|---|---|---|---|---|---|---|---|
| nano | 0.02B | 0.4B | 2 GB | 0.4 GB | 1h | — | — |
| micro | 0.04B | 0.8B | 3 GB | 1 GB | 3h | — | — |
| small | 0.10B | 2B | 8 GB | 2 GB | 20h | — | — |
| medium | 0.32B | 6B | 25 GB | 5 GB | 9d | 3h | — |
| large | 1.18B | 24B | 94 GB | 19 GB | 111d | 40h | 5h |
| xl | 3.3B | 66B | 266 GB | 53 GB | 3y | 14d | 42h |
| xxl | 5.8B | 116B | 464 GB | 93 GB | 10y | 66d | 8d |
| **8b** | 7.9B | 159B | 635 GB | 127 GB | 21y | 142d | 18d |
| **12b** | 12.0B | 241B | 963 GB | 193 GB | 46y | 309d | 39d |
| **20b** | 20.1B | 402B | 1.6 TB | 322 GB | 125y | 2y | 104d |
| **40b** | 40.2B | 805B | 3.2 TB | 644 GB | 477y | 9y | 396d |
| **60b** | 60.3B | 1205B | 4.8 TB | 964 GB | 1060y | 19y | 2y |
| **80b** | 80.2B | 1605B | 6.4 TB | 1.3 TB | 1871y | 34y | 4y |

"Train memory" is optimizer state, gradients and fp32 master weights — the floor
before a single activation, which is why everything from `8b` up needs FSDP across
many devices simply to hold the model.

Read the last three columns as the answer to "can I train a frontier model from
scratch": on one Spark, nothing above `small` finishes in a week. Even 64 H100s put
an 80B model four years away. Those runs are what a large lab spends months and
millions of dollars on, and the arithmetic here is not pessimism — it is the same
arithmetic they face.

What this does mean: **everything up to `small` is a comfortable afternoon**, `medium`
is a long weekend, and past that you are choosing between fine-tuning and distilling
an existing large model, both of which reach a useful result in hours.

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
