# Training from scratch

You give it a folder of text. It trains a tokenizer on your corpus, builds a
transformer from random initialisation, and trains it. Every weight is yours —
nothing is inherited from another model.

Use this when you want a model you genuinely built, and you have a lot of text.

## What each size costs

Token budgets are Chinchilla-optimal (~20 tokens per parameter). Below roughly half
of that, a model is undertrained and the extra parameters are wasted. Times are this
planner's own estimates at a 32k vocabulary.

| tier | params | tokens | ≈ text | train memory | 1x Spark | 8x H100 | 64x H100 |
|---|---|---|---|---|---|---|---|
| nano | 0.02B | 0.4B | 2 GB | 0.4 GB | 1h | — | — |
| micro | 0.04B | 0.8B | 3 GB | 1 GB | 3h | — | — |
| small | 0.10B | 2B | 8 GB | 2 GB | 20h | — | — |
| medium | 0.32B | 6B | 25 GB | 5 GB | 9d | 3h | — |
| large | 1.18B | 24B | 94 GB | 19 GB | 111d | 40h | 5h |
| xl | 3.3B | 66B | 266 GB | 53 GB | 3y | 14d | 42h |
| xxl | 5.8B | 116B | 464 GB | 93 GB | 10y | 66d | 8d |
| 8b | 7.9B | 159B | 635 GB | 127 GB | 21y | 142d | 18d |
| 12b | 12.0B | 241B | 963 GB | 193 GB | 46y | 309d | 39d |
| 20b | 20.1B | 402B | 1.6 TB | 322 GB | 125y | 2y | 104d |
| 40b | 40.2B | 805B | 3.2 TB | 644 GB | 477y | 9y | 396d |
| 60b | 60.3B | 1205B | 4.8 TB | 964 GB | 1060y | 19y | 2y |
| 80b | 80.2B | 1605B | 6.4 TB | 1.3 TB | 1871y | 34y | 4y |

**Train memory** is optimizer state, gradients and fp32 master weights — the floor
before a single activation is stored. Everything from `8b` upward exceeds one device
and needs FSDP sharding merely to hold the model.

## How to read this

**Up to `small`** — 100M parameters, 8 GB of text, about a day — is a comfortable
single-machine run producing a real model that is entirely yours. This is the
recommended starting point.

**`medium`** is a long weekend and wants 25 GB of text.

**Past that**, wall clock rather than memory is the wall. An 80B model is four years
on 64 H100s. That is not a limitation of this tool; it is the same arithmetic a
frontier lab faces, and why such runs cost months and millions.

## What you actually get

A from-scratch model is bounded by its corpus. A few hundred MB of text produces
something that writes fluent, domain-flavoured prose and cannot answer questions.
That is the correct outcome at that scale, not a defect — general capability needs
billions of tokens.

The planner states the tokens-per-parameter ratio before you commit, and says
plainly when a corpus cannot fill even the smallest model.

## Choosing a size

Leave the tier on `auto` and the planner picks the largest size your corpus can
train to a reasonable degree, allowing up to four passes over the data. Override it
in **Advanced → Model size** when you want to disagree.
