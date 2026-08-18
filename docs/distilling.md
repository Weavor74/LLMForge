# Distilling

You give it a folder of text and a teacher model. It trains a *smaller* model to
reproduce the teacher's predictions. The student is built from random init, so its
weights are entirely yours — the teacher influences training but is not inside the
result.

Use this when you want a teacher's behaviour at a fraction of its size, or when you
need a model you can distribute without redistributing someone else's weights.

## What each teacher costs

Distilling a 2B-token corpus. The student size is chosen automatically.

| teacher | student | teacher 4-bit | memory | teacher share of compute | 1x Spark | 8x H100 |
|---|---|---|---|---|---|---|
| 1.5B | 317M | no | 69 GB | 61% | 6d | 2.0h |
| 8B | 1181M | no | 84 GB | 69% | 107d | 35h |
| 13B | 1181M | no | 94 GB | 79% | 153d | 2d |
| 32B | 1181M | yes | 86 GB | 90% | 494d | 7d |
| 40B | 1181M | yes | 90 GB | 92% | 605d | 8d |
| 70B | 1181M | yes | 82 GB | 95% | 3y | 14d |

**The teacher dominates.** With a 70B teacher, 95% of the compute is running the
teacher over your tokens — the student's own training is noise by comparison. The
teacher is scored on every token of every epoch, which is why these times are long.

## The objective

    alpha * T² * KL(teacher || student)  +  (1 - alpha) * cross-entropy(student, true tokens)

Defaults are temperature 2.0 and alpha 0.7. Temperature above 1 softens both
distributions, exposing what the teacher thought about tokens it *didn't* pick —
which is the extra signal that makes distillation more data-efficient than training
the same model from scratch. Both are adjustable in **Advanced**.

## The vocabulary constraint

Comparing two distributions requires the same vocabulary, so the student inherits the
teacher's tokenizer. This matters more than it sounds: distilling from a model with a
150,000-token vocabulary gives a small student whose embedding table is most of its
parameters — very little of it is doing the reasoning.

The plan reports the embedding share when it exceeds half the model. If you see that,
either choose a larger student or use the generation route below.

## The faster alternative: generate, then fine-tune

**Generate data** has the teacher answer a set of prompts *once*, producing a corpus
you then train a small model on as ordinary fine-tuning.

| | online distillation | generate, then fine-tune |
|---|---|---|
| teacher cost | every token, every epoch | once |
| student tokenizer | must match the teacher | free choice |
| what transfers | the full distribution, including uncertainty | the chosen answers only |
| iterating on the student | another full teacher pass | minutes |

Online distillation transfers more. Generation is dramatically cheaper and frees the
student's vocabulary. With a large teacher and a small student, generation is usually
the better trade.

## Distribution

A distilled student contains no teacher weights — it is trained from random
initialisation. That makes it cleaner to distribute than a fine-tune, which carries
the base model inside it. The teacher's licence may still govern using its outputs to
train other models, so check the licence of whichever teacher you choose.
