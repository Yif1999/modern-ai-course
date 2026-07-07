from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class MultiHeadCausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, num_heads: int):
        super().__init__()
        if n_embd % num_heads != 0:
            raise ValueError("n_embd 必须能被 num_heads 整除")
        self.num_heads = num_heads
        self.head_dim = n_embd // num_heads
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)

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
        q = self.split_heads(self.query(x))
        k = self.split_heads(self.key(x))
        v = self.split_heads(self.value(x))
        scores = q @ mx.transpose(k, (0, 1, 3, 2))
        scores = scores / math.sqrt(self.head_dim)
        mask = mx.array(np.tril(np.ones((seq_len, seq_len), dtype=np.float32))).reshape(1, 1, seq_len, seq_len)
        scores = mx.where(mask == 1, scores, mx.full(scores.shape, -1e9))
        weights = nn.softmax(scores, axis=-1)
        out = weights @ v
        return self.proj(self.merge_heads(out))


class FeedForward(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.linear1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear2 = nn.Linear(4 * n_embd, n_embd)

    def __call__(self, x):
        return self.linear2(nn.gelu(self.linear1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, n_embd: int, num_heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadCausalSelfAttention(n_embd, num_heads)
        self.ffn = FeedForward(n_embd)

    def __call__(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class BaselineTinyGPT(nn.Module):
    profile_name = "baseline_debug"

    def __init__(self, vocab_size: int, block_size: int, n_embd: int, num_heads: int, num_layers: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        self.token_embedding_table = mx.random.normal((vocab_size, n_embd)) * 0.02
        self.position_embedding_table = mx.random.normal((block_size, n_embd)) * 0.02
        self.blocks = [TransformerBlock(n_embd, num_heads) for _ in range(num_layers)]
        self.final_ln = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def __call__(self, idx):
        _, seq_len = idx.shape
        token_emb = self.token_embedding_table[idx]
        pos_emb = self.position_embedding_table[mx.arange(seq_len)]
        x = token_emb + pos_emb
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.final_ln(x))

    def inspect_shapes(self, idx):
        _, seq_len = idx.shape
        token_emb = self.token_embedding_table[idx]
        pos_emb = self.position_embedding_table[mx.arange(seq_len)]
        x = token_emb + pos_emb
        block_shapes = []
        for block in self.blocks:
            x = block(x)
            block_shapes.append(tuple(x.shape))
        logits = self.lm_head(self.final_ln(x))
        mx.eval(logits)
        return {
            "idx": tuple(idx.shape),
            "token_emb": tuple(token_emb.shape),
            "pos_emb": tuple(pos_emb.shape),
            "position_method": "learned position embedding",
            "block_outputs": block_shapes,
            "logits": tuple(logits.shape),
        }
