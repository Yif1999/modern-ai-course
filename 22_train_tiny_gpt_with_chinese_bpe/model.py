from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class MultiHeadCausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, num_heads: int):
        super().__init__()
        if n_embd % num_heads != 0:
            raise ValueError("n_embd must be divisible by num_heads")
        self.n_embd = n_embd
        self.num_heads = num_heads
        self.head_size = n_embd // num_heads
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)

    def _split_heads(self, x):
        batch, seq_len, _ = x.shape
        x = x.reshape(batch, seq_len, self.num_heads, self.head_size)
        return mx.transpose(x, (0, 2, 1, 3))

    def _merge_heads(self, x):
        batch, num_heads, seq_len, head_size = x.shape
        x = mx.transpose(x, (0, 2, 1, 3))
        return x.reshape(batch, seq_len, num_heads * head_size)

    def __call__(self, x, return_attention: bool = False):
        _, seq_len, _ = x.shape
        q = self._split_heads(self.query(x))
        k = self._split_heads(self.key(x))
        v = self._split_heads(self.value(x))

        scores = q @ mx.transpose(k, (0, 1, 3, 2))
        scores = scores / math.sqrt(self.head_size)

        mask_np = np.tril(np.ones((seq_len, seq_len), dtype=np.float32))
        mask = mx.array(mask_np).reshape(1, 1, seq_len, seq_len)
        scores = mx.where(mask == 1, scores, mx.full(scores.shape, -1e9))

        weights = nn.softmax(scores, axis=-1)
        out = weights @ v
        out = self._merge_heads(out)
        out = self.proj(out)
        if return_attention:
            return out, weights
        return out


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
        self.attn = MultiHeadCausalSelfAttention(n_embd, num_heads)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def __call__(self, x, return_attention: bool = False):
        if return_attention:
            attn_out, weights = self.attn(self.ln1(x), return_attention=True)
            x = x + attn_out
            x = x + self.ffwd(self.ln2(x))
            return x, weights
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int, block_size: int, n_embd: int, num_heads: int, num_layers: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.token_embedding_table = mx.random.normal((vocab_size, n_embd)) * 0.02
        self.position_embedding_table = mx.random.normal((block_size, n_embd)) * 0.02
        self.blocks = [TransformerBlock(n_embd, num_heads) for _ in range(num_layers)]
        self.final_ln = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def __call__(self, idx, return_attention: bool = False):
        _, seq_len = idx.shape
        if seq_len > self.block_size:
            raise ValueError(f"seq_len={seq_len} exceeds block_size={self.block_size}")
        token_emb = self.token_embedding_table[idx]
        pos_emb = self.position_embedding_table[mx.arange(seq_len)]
        x = token_emb + pos_emb

        attentions = []
        for block in self.blocks:
            if return_attention:
                x, weights = block(x, return_attention=True)
                attentions.append(weights)
            else:
                x = block(x)

        logits = self.lm_head(self.final_ln(x))
        if return_attention:
            return logits, attentions
        return logits

    def inspect_shapes(self, idx) -> dict:
        _, seq_len = idx.shape
        token_emb = self.token_embedding_table[idx]
        pos_emb = self.position_embedding_table[mx.arange(seq_len)]
        x = token_emb + pos_emb
        block_shapes = []
        attention_shapes = []
        for block in self.blocks:
            x, weights = block(x, return_attention=True)
            block_shapes.append(tuple(x.shape))
            attention_shapes.append(tuple(weights.shape))
        logits = self.lm_head(self.final_ln(x))
        mx.eval(token_emb, pos_emb, x, logits)
        return {
            "idx": tuple(idx.shape),
            "token_emb": tuple(token_emb.shape),
            "pos_emb": tuple(pos_emb.shape),
            "x_after_embedding_add": tuple((token_emb + pos_emb).shape),
            "block_outputs": block_shapes,
            "attention_weights": attention_shapes,
            "logits": tuple(logits.shape),
        }


def language_model_loss(model: TinyGPT, idx, targets):
    logits = model(idx)
    batch, seq_len, vocab_size = logits.shape
    return nn.losses.cross_entropy(
        logits.reshape(batch * seq_len, vocab_size),
        targets.reshape(batch * seq_len),
        reduction="mean",
    )


def sample_next_id(logits, temperature: float = 0.8, top_k: int | None = 20) -> int:
    logits_np = np.array(logits, dtype=np.float64)
    logits_np = logits_np / max(float(temperature), 1e-6)
    logits_np = logits_np - np.max(logits_np)
    probs = np.exp(logits_np)
    probs = probs / np.sum(probs)

    if top_k is not None and 0 < top_k < len(probs):
        keep = np.argpartition(probs, -top_k)[-top_k:]
        filtered = np.zeros_like(probs)
        filtered[keep] = probs[keep]
        probs = filtered / filtered.sum()

    return int(np.random.choice(len(probs), p=probs))


def generate_ids(
    model: TinyGPT,
    start_ids,
    max_new_tokens: int,
    temperature: float = 0.8,
    top_k: int | None = 20,
) -> list[int]:
    ids = [int(i) for i in start_ids]
    for _ in range(max_new_tokens):
        context = ids[-model.block_size :]
        idx = mx.array([context], dtype=mx.int32)
        logits = model(idx)
        last_logits = logits[0, -1, :]
        mx.eval(last_logits)
        ids.append(sample_next_id(last_logits, temperature=temperature, top_k=top_k))
    return ids
