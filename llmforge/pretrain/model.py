"""A decoder-only transformer, written from scratch.

Architecture follows what has actually held up since GPT-2: pre-norm blocks, RMSNorm,
rotary position embeddings, SwiGLU feed-forwards, grouped-query attention, and no
biases anywhere. Attention goes through PyTorch's SDPA rather than flash-attn, whose
wheels do not build for sm_121 aarch64 — the doctor verifies a fused backend engages.

Nothing here is hardware-specific; the hardware shows up in how the planner *sizes*
this model, not in what it is.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int
    n_layer: int
    n_head: int
    n_kv_head: int
    d_model: int
    d_ff: int
    max_seq_len: int
    rope_theta: float = 10_000.0
    dropout: float = 0.0
    tie_embeddings: bool = True
    # Recompute activations in the backward pass instead of storing them. Costs about
    # one extra forward pass per step and buys back most of the activation memory,
    # which is what lets a large model fit at a usable batch size.
    gradient_checkpointing: bool = False

    def __post_init__(self) -> None:
        if self.d_model % self.n_head != 0:
            raise ValueError(f"d_model {self.d_model} not divisible by n_head {self.n_head}")
        if self.n_head % self.n_kv_head != 0:
            raise ValueError(f"n_head {self.n_head} not divisible by n_kv_head {self.n_kv_head}")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head

    def n_params(self, embeddings: bool = True) -> int:
        """Parameter count, computed rather than measured so the planner can size
        candidate architectures without building them."""
        d, kv = self.d_model, self.n_kv_head * self.head_dim

        # q, k, v, o. Under GQA, k and v are narrower than q.
        attn = d * d + d * kv * 2 + d * d
        # SwiGLU uses three matrices: gate, up, down.
        mlp = 3 * d * self.d_ff
        # Two RMSNorms per block, one gain vector each.
        norms = 2 * d

        total = self.n_layer * (attn + mlp + norms) + d  # + final norm

        if embeddings:
            # Tied embeddings are one matrix serving as both input and output.
            total += self.vocab_size * d * (1 if self.tie_embeddings else 2)
        return total

    def to_dict(self) -> dict:
        return asdict(self)


class RMSNorm(nn.Module):
    """Cheaper than LayerNorm and empirically just as good: no mean subtraction,
    no bias, one gain vector."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reduce in fp32; bf16 has too little mantissa for a stable mean of squares.
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def build_rope_cache(
    seq_len: int, head_dim: int, theta: float, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute rotary cos/sin tables. Shape (seq_len, head_dim // 2)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(seq_len, device=device).float()
    angles = torch.outer(positions, inv_freq)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate query/key pairs. `x` is (batch, heads, seq, head_dim)."""
    # Split each head into interleaved halves and rotate them against each other.
    x1, x2 = x.float().chunk(2, dim=-1)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    rotated = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    return rotated.to(x.dtype)


class Attention(nn.Module):
    """Grouped-query causal self-attention.

    Fewer key/value heads than query heads shrinks the KV cache at inference and the
    projection parameters at training, at very little quality cost.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.head_dim = cfg.head_dim
        self.dropout = cfg.dropout

        kv_dim = cfg.n_kv_head * cfg.head_dim
        # One fused projection: a single larger GEMM beats three smaller ones,
        # which matters on a bandwidth-bound machine.
        self.qkv_proj = nn.Linear(cfg.d_model, cfg.d_model + 2 * kv_dim, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        kv_dim = self.n_kv_head * self.head_dim

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split([self.n_head * self.head_dim, kv_dim, kv_dim], dim=-1)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # enable_gqa broadcasts the kv heads inside the kernel, avoiding an explicit
        # repeat_interleave that would materialise n_head copies of k and v.
        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
            enable_gqa=self.n_kv_head != self.n_head,
        )

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    """Gated feed-forward. Three matrices instead of two, but better per-parameter
    than ReLU/GELU MLPs, which is why every recent model uses it."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down_proj = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg)
        self.resid_dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.resid_dropout(self.attn(self.attn_norm(x), cos, sin))
        x = x + self.resid_dropout(self.mlp(self.mlp_norm(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.emb_dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            # Input and output embeddings learn the same thing; sharing them saves
            # vocab*d_model parameters, which dominates small models.
            self.lm_head.weight = self.tok_emb.weight

        cos, sin = build_rope_cache(
            cfg.max_seq_len, cfg.head_dim, cfg.rope_theta, torch.device("cpu")
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)

        # Scale down the projections that write into the residual stream, so the
        # stream's variance does not grow with depth at initialisation.
        scale = 1.0 / math.sqrt(2 * cfg.n_layer)
        for block in self.blocks:
            torch.nn.init.normal_(block.attn.o_proj.weight, mean=0.0, std=0.02 * scale)
            torch.nn.init.normal_(block.mlp.down_proj.weight, mean=0.0, std=0.02 * scale)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, T = idx.shape
        if T > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {T} exceeds max_seq_len {self.cfg.max_seq_len}")

        cos = self.rope_cos[:T]
        sin = self.rope_sin[:T]

        x = self.emb_dropout(self.tok_emb(idx))
        for block in self.blocks:
            if self.cfg.gradient_checkpointing and self.training:
                # use_reentrant=False is the supported implementation and the one that
                # works with tied weights and no explicit input gradients.
                x = torch.utils.checkpoint.checkpoint(
                    block, x, cos, sin, use_reentrant=False
                )
            else:
                x = block(x, cos, sin)
        x = self.final_norm(x)

        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # Cross-entropy in fp32: bf16 logits lose too much in the log-softmax.
            loss = F.cross_entropy(
                logits.float().view(-1, logits.size(-1)), targets.reshape(-1)
            )
        return logits, loss

    def param_groups(self, weight_decay: float) -> list[dict]:
        """Decay matrices, not vectors.

        Applying weight decay to norm gains and embeddings measurably hurts; the
        convention is to decay only parameters with 2 or more dimensions.
        """
        decay, no_decay = [], []
        for param in self.parameters():
            if not param.requires_grad:
                continue
            (decay if param.dim() >= 2 else no_decay).append(param)
        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int | None = 50,
        eos_id: int | None = None,
    ) -> torch.Tensor:
        """Straightforward sampling loop, no KV cache.

        Used for the sample-during-training callback and `llmforge sample`, where a
        few hundred tokens is the whole job and cache bookkeeping would not pay off.
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop to the context window; RoPE has no table beyond max_seq_len.
            window = idx[:, -self.cfg.max_seq_len :]
            logits, _ = self(window)
            logits = logits[:, -1, :].float()

            if temperature <= 0:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    k = min(top_k, logits.size(-1))
                    threshold = logits.topk(k, dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < threshold, float("-inf"))
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx, next_id], dim=1)

            if eos_id is not None and (next_id == eos_id).all():
                break
        return idx
