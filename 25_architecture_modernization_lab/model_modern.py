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
    head_size = x.shape[-1]
    if head_size % 2 != 0:
        raise ValueError("RoPE requires an even head size")
    inv_freq = 1.0 / (base ** (mx.arange(0, head_size, 2).astype(mx.float32) / head_size))
    positions = mx.arange(seq_len).astype(mx.float32)
    freqs = positions[:, None] * inv_freq[None, :]
    cos = mx.cos(freqs).reshape(1, 1, seq_len, head_size // 2)
    sin = mx.sin(freqs).reshape(1, 1, seq_len, head_size // 2)
    cos = mx.stack((cos, cos), axis=-1).reshape(1, 1, seq_len, head_size)
    sin = mx.stack((sin, sin), axis=-1).reshape(1, 1, seq_len, head_size)
    return x * cos + rotate_half_interleaved(x) * sin


class MultiHeadCausalSelfAttentionRoPE(nn.Module):
    def __init__(self, n_embd: int, num_heads: int):
        super().__init__()
        if n_embd % num_heads != 0:
            raise ValueError("n_embd must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_size = n_embd // num_heads
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)

    def split_heads(self, x):
        batch, seq_len, n_embd = x.shape
        x = x.reshape(batch, seq_len, self.num_heads, self.head_size)
        return mx.transpose(x, (0, 2, 1, 3))

    def merge_heads(self, x):
        batch, heads, seq_len, head_size = x.shape
        x = mx.transpose(x, (0, 2, 1, 3))
        return x.reshape(batch, seq_len, heads * head_size)

    def __call__(self, x):
        _, seq_len, _ = x.shape
        q = self.split_heads(self.query(x))
        k = self.split_heads(self.key(x))
        v = self.split_heads(self.value(x))
        q = apply_rope(q, seq_len)
        k = apply_rope(k, seq_len)
        scores = q @ mx.transpose(k, (0, 1, 3, 2))
        scores = scores / math.sqrt(self.head_size)
        mask = mx.array(np.tril(np.ones((seq_len, seq_len), dtype=np.float32))).reshape(1, 1, seq_len, seq_len)
        scores = mx.where(mask == 1, scores, mx.full(scores.shape, -1e9))
        weights = nn.softmax(scores, axis=-1)
        out = weights @ v
        return self.proj(self.merge_heads(out))


class SwiGLUFeedForward(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        hidden_dim = 4 * n_embd
        self.gate = nn.Linear(n_embd, hidden_dim)
        self.value = nn.Linear(n_embd, hidden_dim)
        self.proj = nn.Linear(hidden_dim, n_embd)

    def __call__(self, x):
        return self.proj(silu(self.gate(x)) * self.value(x))


class ModernTransformerBlock(nn.Module):
    def __init__(self, n_embd: int, num_heads: int):
        super().__init__()
        self.rms1 = RMSNorm(n_embd)
        self.rms2 = RMSNorm(n_embd)
        self.attn = MultiHeadCausalSelfAttentionRoPE(n_embd, num_heads)
        self.ffwd = SwiGLUFeedForward(n_embd)

    def __call__(self, x):
        x = x + self.attn(self.rms1(x))
        x = x + self.ffwd(self.rms2(x))
        return x


class ModernTinyGPT(nn.Module):
    def __init__(self, vocab_size: int, block_size: int, n_embd: int, num_heads: int, num_layers: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        self.token_embedding_table = mx.random.normal((vocab_size, n_embd)) * 0.02
        self.blocks = [ModernTransformerBlock(n_embd, num_heads) for _ in range(num_layers)]
        self.final_norm = RMSNorm(n_embd)
        self.lm_head_bias = mx.zeros((vocab_size,))

    def __call__(self, idx):
        x = self.token_embedding_table[idx]
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = x @ mx.transpose(self.token_embedding_table, (1, 0)) + self.lm_head_bias
        return logits

    def inspect_shapes(self, idx):
        x = self.token_embedding_table[idx]
        block_shapes = []
        for block in self.blocks:
            x = block(x)
            block_shapes.append(tuple(x.shape))
        x = self.final_norm(x)
        logits = x @ mx.transpose(self.token_embedding_table, (1, 0)) + self.lm_head_bias
        mx.eval(logits)
        return {
            "idx": tuple(idx.shape),
            "token_emb": tuple(self.token_embedding_table[idx].shape),
            "position_method": "RoPE on Q/K, no learned position embedding table",
            "block_outputs": block_shapes,
            "logits": tuple(logits.shape),
            "weight_tying": "logits = hidden @ token_embedding_table.T + bias",
        }


def language_model_loss(model: ModernTinyGPT, idx, targets):
    logits = model(idx)
    batch, seq_len, vocab_size = logits.shape
    return nn.losses.cross_entropy(
        logits.reshape(batch * seq_len, vocab_size),
        targets.reshape(batch * seq_len),
        reduction="mean",
    )


def sample_next_id(logits, temperature: float = 0.8, top_k: int = 20) -> int:
    logits_np = np.array(logits, dtype=np.float64)
    logits_np = logits_np / max(temperature, 1e-6)
    logits_np = logits_np - np.max(logits_np)
    probs = np.exp(logits_np)
    probs = probs / probs.sum()
    if 0 < top_k < len(probs):
        keep = np.argpartition(probs, -top_k)[-top_k:]
        filtered = np.zeros_like(probs)
        filtered[keep] = probs[keep]
        probs = filtered / filtered.sum()
    return int(np.random.choice(len(probs), p=probs))


def generate_ids(model: ModernTinyGPT, start_ids, max_new_tokens: int, temperature: float = 0.8, top_k: int = 20):
    ids = [int(i) for i in start_ids]
    for _ in range(max_new_tokens):
        context = ids[-model.block_size :]
        idx = mx.array([context], dtype=mx.int32)
        logits = model(idx)
        last_logits = logits[0, -1, :]
        mx.eval(last_logits)
        ids.append(sample_next_id(last_logits, temperature=temperature, top_k=top_k))
    return ids
