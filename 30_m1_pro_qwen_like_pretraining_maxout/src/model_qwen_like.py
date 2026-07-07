from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass
class QwenLikeConfig:
    vocab_size: int
    block_size: int
    n_embd: int
    num_layers: int
    num_q_heads: int
    num_kv_heads: int
    ffn_multiplier: float = 3.0
    rope_base: float = 1_000_000.0
    qk_norm: bool = True
    weight_tying: bool = True
    use_bias: bool = False
    activation_checkpointing: bool = False
    rope_impl: str = "manual"
    rms_norm_impl: str = "manual"


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x):
        rms = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return (x / rms) * self.weight


class FastRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, self.weight, self.eps)


def make_rms_norm(dim: int, impl: str, eps: float = 1e-6):
    if impl == "manual":
        return RMSNorm(dim, eps=eps)
    if impl == "nn":
        return nn.RMSNorm(dim, eps=eps)
    if impl == "fast":
        return FastRMSNorm(dim, eps=eps)
    raise ValueError(f"Unsupported rms_norm_impl: {impl!r}")


def silu(x):
    return x * mx.sigmoid(x)


def rotate_half_interleaved(x):
    even = x[..., 0::2]
    odd = x[..., 1::2]
    return mx.stack((-odd, even), axis=-1).reshape(x.shape)


def apply_rope(x, base: float):
    # x: [batch, heads, seq_len, head_dim]
    _, _, seq_len, head_dim = x.shape
    if head_dim % 2 != 0:
        raise ValueError("RoPE requires an even head_dim")
    inv_freq = 1.0 / (base ** (mx.arange(0, head_dim, 2).astype(mx.float32) / head_dim))
    positions = mx.arange(seq_len).astype(mx.float32)
    freqs = positions[:, None] * inv_freq[None, :]
    cos = mx.cos(freqs).reshape(1, 1, seq_len, head_dim // 2)
    sin = mx.sin(freqs).reshape(1, 1, seq_len, head_dim // 2)
    cos = mx.stack((cos, cos), axis=-1).reshape(1, 1, seq_len, head_dim)
    sin = mx.stack((sin, sin), axis=-1).reshape(1, 1, seq_len, head_dim)
    return x * cos + rotate_half_interleaved(x) * sin


class QwenLikeAttention(nn.Module):
    def __init__(self, cfg: QwenLikeConfig):
        super().__init__()
        if cfg.n_embd % cfg.num_q_heads != 0:
            raise ValueError("n_embd must be divisible by num_q_heads")
        if cfg.num_q_heads % cfg.num_kv_heads != 0:
            raise ValueError("num_q_heads must be divisible by num_kv_heads for GQA")
        self.n_embd = cfg.n_embd
        self.num_q_heads = cfg.num_q_heads
        self.num_kv_heads = cfg.num_kv_heads
        self.head_dim = cfg.n_embd // cfg.num_q_heads
        self.kv_dim = cfg.num_kv_heads * self.head_dim
        self.rope_base = cfg.rope_base
        self.q_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.use_bias)
        self.k_proj = nn.Linear(cfg.n_embd, self.kv_dim, bias=cfg.use_bias)
        self.v_proj = nn.Linear(cfg.n_embd, self.kv_dim, bias=cfg.use_bias)
        self.o_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.use_bias)
        self.rope_impl = cfg.rope_impl
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=cfg.rope_base) if cfg.rope_impl == "nn" else None
        self.q_norm = make_rms_norm(self.head_dim, cfg.rms_norm_impl) if cfg.qk_norm else None
        self.k_norm = make_rms_norm(self.head_dim, cfg.rms_norm_impl) if cfg.qk_norm else None

    def split_heads(self, x, heads: int):
        batch, seq_len, width = x.shape
        head_dim = width // heads
        x = x.reshape(batch, seq_len, heads, head_dim)
        return mx.transpose(x, (0, 2, 1, 3))

    def merge_heads(self, x):
        batch, heads, seq_len, head_dim = x.shape
        x = mx.transpose(x, (0, 2, 1, 3))
        return x.reshape(batch, seq_len, heads * head_dim)

    def __call__(self, x):
        q = self.split_heads(self.q_proj(x), self.num_q_heads)
        k = self.split_heads(self.k_proj(x), self.num_kv_heads)
        v = self.split_heads(self.v_proj(x), self.num_kv_heads)
        if self.rope_impl == "manual":
            q = apply_rope(q, self.rope_base)
            k = apply_rope(k, self.rope_base)
        elif self.rope_impl == "nn":
            q = self.rope(q)
            k = self.rope(k)
        elif self.rope_impl == "fast":
            q = mx.fast.rope(q, dims=self.head_dim, traditional=False, base=self.rope_base, scale=1.0, offset=0)
            k = mx.fast.rope(k, dims=self.head_dim, traditional=False, base=self.rope_base, scale=1.0, offset=0)
        else:
            raise ValueError(f"Unsupported rope_impl: {self.rope_impl!r}")
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        out = mx.fast.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=1.0 / math.sqrt(self.head_dim),
            mask="causal",
        )
        return self.o_proj(self.merge_heads(out))


class SwiGLU(nn.Module):
    def __init__(self, cfg: QwenLikeConfig):
        super().__init__()
        hidden_dim = int(cfg.ffn_multiplier * cfg.n_embd)
        # Multiples of 256 are friendlier for larger matmuls.
        hidden_dim = max(256, ((hidden_dim + 255) // 256) * 256)
        self.gate = nn.Linear(cfg.n_embd, hidden_dim, bias=cfg.use_bias)
        self.value = nn.Linear(cfg.n_embd, hidden_dim, bias=cfg.use_bias)
        self.down = nn.Linear(hidden_dim, cfg.n_embd, bias=cfg.use_bias)
        self.hidden_dim = hidden_dim

    def __call__(self, x):
        return self.down(silu(self.gate(x)) * self.value(x))


class QwenLikeBlock(nn.Module):
    def __init__(self, cfg: QwenLikeConfig):
        super().__init__()
        self.input_norm = make_rms_norm(cfg.n_embd, cfg.rms_norm_impl)
        self.post_attn_norm = make_rms_norm(cfg.n_embd, cfg.rms_norm_impl)
        self.attn = QwenLikeAttention(cfg)
        self.ffn = SwiGLU(cfg)

    def __call__(self, x):
        x = x + self.attn(self.input_norm(x))
        x = x + self.ffn(self.post_attn_norm(x))
        return x


class QwenLikeDenseLM(nn.Module):
    def __init__(self, cfg: QwenLikeConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = mx.random.normal((cfg.vocab_size, cfg.n_embd)) * 0.02
        self.blocks = [QwenLikeBlock(cfg) for _ in range(cfg.num_layers)]
        self.final_norm = make_rms_norm(cfg.n_embd, cfg.rms_norm_impl)
        if not cfg.weight_tying:
            self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=cfg.use_bias)

    def __call__(self, idx):
        x = self.token_embedding[idx]
        for block in self.blocks:
            x = mx.checkpoint(block)(x) if self.cfg.activation_checkpointing else block(x)
        x = self.final_norm(x)
        if self.cfg.weight_tying:
            return x @ mx.transpose(self.token_embedding, (1, 0))
        return self.lm_head(x)

    def inspect_shapes(self, idx):
        x = self.token_embedding[idx]
        block_shapes = []
        for block in self.blocks:
            x = block(x)
            block_shapes.append(tuple(x.shape))
        logits = self(idx)
        mx.eval(logits)
        components = ["RoPE", "RMSNorm", "SwiGLU", "GQA", "QK-Norm", "Weight Tying"]
        if self.cfg.activation_checkpointing:
            components.append("Activation Checkpointing")
        return {
            "idx": tuple(idx.shape),
            "token_embedding": tuple(self.token_embedding[idx].shape),
            "block_outputs": block_shapes,
            "logits": tuple(logits.shape),
            "attention": {
                "type": "GQA",
                "num_q_heads": self.cfg.num_q_heads,
                "num_kv_heads": self.cfg.num_kv_heads,
                "head_dim": self.cfg.n_embd // self.cfg.num_q_heads,
                "kernel": "mx.fast.scaled_dot_product_attention",
                "mask": "causal",
                "rope_impl": self.cfg.rope_impl,
                "rms_norm_impl": self.cfg.rms_norm_impl,
            },
            "components": components,
        }
