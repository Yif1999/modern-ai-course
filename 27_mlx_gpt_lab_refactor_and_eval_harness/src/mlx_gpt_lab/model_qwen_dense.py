from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x):
        rms = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return (x / rms) * self.weight


def silu(x):
    return x * mx.sigmoid(x)


def rotate_half_interleaved(x):
    even = x[..., 0::2]
    odd = x[..., 1::2]
    return mx.stack((-odd, even), axis=-1).reshape(x.shape)


def apply_rope(x, seq_len: int, base: float = 10000.0):
    head_dim = x.shape[-1]
    if head_dim % 2 != 0:
        raise ValueError("RoPE 需要偶数 head_dim")
    inv_freq = 1.0 / (base ** (mx.arange(0, head_dim, 2).astype(mx.float32) / head_dim))
    positions = mx.arange(seq_len).astype(mx.float32)
    freqs = positions[:, None] * inv_freq[None, :]
    cos = mx.cos(freqs).reshape(1, 1, seq_len, head_dim // 2)
    sin = mx.sin(freqs).reshape(1, 1, seq_len, head_dim // 2)
    cos = mx.stack((cos, cos), axis=-1).reshape(1, 1, seq_len, head_dim)
    sin = mx.stack((sin, sin), axis=-1).reshape(1, 1, seq_len, head_dim)
    return x * cos + rotate_half_interleaved(x) * sin


class QwenStyleCausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, num_heads: int, rope_base: float = 10000.0, use_bias: bool = False):
        super().__init__()
        if n_embd % num_heads != 0:
            raise ValueError("n_embd 必须能被 num_heads 整除")
        self.num_heads = num_heads
        self.head_dim = n_embd // num_heads
        self.rope_base = rope_base
        self.query = nn.Linear(n_embd, n_embd, bias=use_bias)
        self.key = nn.Linear(n_embd, n_embd, bias=use_bias)
        self.value = nn.Linear(n_embd, n_embd, bias=use_bias)
        self.proj = nn.Linear(n_embd, n_embd, bias=use_bias)

    def split_heads(self, x):
        batch, seq_len, n_embd = x.shape
        x = x.reshape(batch, seq_len, self.num_heads, self.head_dim)
        return mx.transpose(x, (0, 2, 1, 3))

    def merge_heads(self, x):
        batch, heads, seq_len, head_dim = x.shape
        x = mx.transpose(x, (0, 2, 1, 3))
        return x.reshape(batch, seq_len, heads * head_dim)

    def __call__(self, x):
        _, seq_len, _ = x.shape
        q = apply_rope(self.split_heads(self.query(x)), seq_len, self.rope_base)
        k = apply_rope(self.split_heads(self.key(x)), seq_len, self.rope_base)
        v = self.split_heads(self.value(x))
        scores = q @ mx.transpose(k, (0, 1, 3, 2))
        scores = scores / math.sqrt(self.head_dim)
        mask = mx.array(np.tril(np.ones((seq_len, seq_len), dtype=np.float32))).reshape(1, 1, seq_len, seq_len)
        scores = mx.where(mask == 1, scores, mx.full(scores.shape, -1e9))
        weights = nn.softmax(scores, axis=-1)
        out = weights @ v
        return self.proj(self.merge_heads(out))


class SwiGLUFeedForward(nn.Module):
    def __init__(self, n_embd: int, hidden_dim: int, use_bias: bool = False):
        super().__init__()
        self.gate = nn.Linear(n_embd, hidden_dim, bias=use_bias)
        self.value = nn.Linear(n_embd, hidden_dim, bias=use_bias)
        self.proj = nn.Linear(hidden_dim, n_embd, bias=use_bias)

    def __call__(self, x):
        return self.proj(silu(self.gate(x)) * self.value(x))


class QwenDenseBlock(nn.Module):
    def __init__(
        self,
        n_embd: int,
        num_heads: int,
        ffn_multiplier: float = 3.0,
        rope_base: float = 10000.0,
        use_bias: bool = False,
    ):
        super().__init__()
        hidden_dim = int(ffn_multiplier * n_embd)
        self.norm1 = RMSNorm(n_embd)
        self.norm2 = RMSNorm(n_embd)
        self.attn = QwenStyleCausalSelfAttention(n_embd, num_heads, rope_base=rope_base, use_bias=use_bias)
        self.ffn = SwiGLUFeedForward(n_embd, hidden_dim, use_bias=use_bias)

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class QwenDenseTinyGPT(nn.Module):
    profile_name = "qwen_dense_tiny"

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_embd: int,
        num_heads: int,
        num_layers: int,
        ffn_multiplier: float = 3.0,
        rope_base: float = 10000.0,
        weight_tying: bool = True,
        use_bias: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        self.weight_tying = weight_tying
        self.token_embedding_table = mx.random.normal((vocab_size, n_embd)) * 0.02
        self.blocks = [
            QwenDenseBlock(
                n_embd=n_embd,
                num_heads=num_heads,
                ffn_multiplier=ffn_multiplier,
                rope_base=rope_base,
                use_bias=use_bias,
            )
            for _ in range(num_layers)
        ]
        self.final_norm = RMSNorm(n_embd)
        if not weight_tying:
            self.lm_head = nn.Linear(n_embd, vocab_size, bias=use_bias)

    def __call__(self, idx):
        x = self.token_embedding_table[idx]
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        if self.weight_tying:
            return x @ mx.transpose(self.token_embedding_table, (1, 0))
        return self.lm_head(x)

    def inspect_shapes(self, idx):
        x = self.token_embedding_table[idx]
        block_shapes = []
        for block in self.blocks:
            x = block(x)
            block_shapes.append(tuple(x.shape))
        x = self.final_norm(x)
        logits = x @ mx.transpose(self.token_embedding_table, (1, 0)) if self.weight_tying else self.lm_head(x)
        mx.eval(logits)
        return {
            "idx": tuple(idx.shape),
            "token_emb": tuple(self.token_embedding_table[idx].shape),
            "position_method": "RoPE on Q/K, no learned position embedding table",
            "block_outputs": block_shapes,
            "logits": tuple(logits.shape),
            "weight_tying": self.weight_tying,
            "modern_components": ["RoPE", "RMSNorm", "SwiGLU", "Pre-Norm", "optional weight tying"],
        }
