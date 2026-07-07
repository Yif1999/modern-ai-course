from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np

import mlx.core as mx
import mlx.nn as nn


print("=== MLX Self-Attention Mechanics Demo ===")

mx.random.seed(42)
np.random.seed(42)

current_dir = Path(__file__).resolve().parent
output_dir = current_dir / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

# 一个很小的字符序列
tokens = list("hello ai")
seq_len = len(tokens)

chars = sorted(list(set(tokens)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

ids = [stoi[ch] for ch in tokens]
idx = mx.array([ids], dtype=mx.int32)

print("\nTokens:")
print(tokens)
print("Token ids:")
print(ids)

print("\nInput idx shape:", idx.shape)
print("Meaning: [batch, seq_len]")

batch = 1
n_embd = 8
head_size = 8

# token embedding + position embedding
token_embedding_table = mx.random.normal((vocab_size, n_embd)) * 0.02
position_embedding_table = mx.random.normal((seq_len, n_embd)) * 0.02

token_emb = token_embedding_table[idx]
positions = mx.arange(seq_len)
pos_emb = position_embedding_table[positions]

x = token_emb + pos_emb

mx.eval(token_emb, pos_emb, x)

print("\nEmbedding shapes:")
print("token_emb:", token_emb.shape)
print("pos_emb:", pos_emb.shape)
print("combined x:", x.shape)

# 手写 Q, K, V 投影矩阵
Wq = mx.random.normal((n_embd, head_size)) * 0.02
Wk = mx.random.normal((n_embd, head_size)) * 0.02
Wv = mx.random.normal((n_embd, head_size)) * 0.02

q = x @ Wq
k = x @ Wk
v = x @ Wv

mx.eval(q, k, v)

print("\nQ/K/V shapes:")
print("q:", q.shape)
print("k:", k.shape)
print("v:", v.shape)

# attention scores: QK^T / sqrt(d)
scores = q @ mx.transpose(k, (0, 2, 1))
scores = scores / math.sqrt(head_size)

mx.eval(scores)

print("\nRaw attention scores shape:", scores.shape)
print("Meaning: [batch, seq_len, seq_len]")
print("Each row asks: this position should look at which positions?")

# causal mask：不能看未来
mask_np = np.tril(np.ones((seq_len, seq_len), dtype=np.float32))
mask = mx.array(mask_np)

masked_scores = mx.where(
    mask.reshape(1, seq_len, seq_len) == 1,
    scores,
    mx.full(scores.shape, -1e9),
)

attn_weights = nn.softmax(masked_scores, axis=-1)

out = attn_weights @ v

mx.eval(masked_scores, attn_weights, out)

print("\nAfter causal mask and softmax:")
print("mask shape:", mask.shape)
print("attn_weights shape:", attn_weights.shape)
print("out shape:", out.shape)

print("\nAttention weights matrix:")
attn_np = np.array(attn_weights[0])
np.set_printoptions(precision=3, suppress=True)
print(attn_np)

print("\nInterpretation:")
print("Row i 表示第 i 个 token 在看前面哪些 token。")
print("右上角为 0，因为 causal mask 禁止看未来。")

# 保存 attention heatmap
plt.figure(figsize=(7, 6))
plt.imshow(attn_np, cmap="viridis")
plt.colorbar(label="attention weight")
plt.xticks(range(seq_len), tokens)
plt.yticks(range(seq_len), tokens)
plt.xlabel("tokens being attended to")
plt.ylabel("current token")
plt.title("Causal Self-Attention Weights")
plt.tight_layout()

heatmap_path = output_dir / "attention_mechanics_heatmap.png"
plt.savefig(heatmap_path, dpi=150)
plt.close()

print("\nSaved:", heatmap_path)
print("\nDone.")
