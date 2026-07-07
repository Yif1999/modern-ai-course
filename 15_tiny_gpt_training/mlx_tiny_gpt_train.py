from pathlib import Path
import json
import math
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten


print("=== MLX Tiny GPT Training ===")

mx.random.seed(42)
np.random.seed(42)

current_dir = Path(__file__).resolve().parent
data_dir = current_dir / "data"
output_dir = current_dir / "outputs"
data_dir.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)

text_path = data_dir / "tiny_text.txt"

if not text_path.exists():
    base_text = (
        "hello ai lab\n"
        "hello mlx\n"
        "hello tiny gpt\n"
        "we learn token embedding\n"
        "we learn position embedding\n"
        "we learn self attention\n"
        "we learn multi head attention\n"
        "we learn transformer block\n"
        "tiny gpt stacks transformer blocks\n"
        "each block keeps the same hidden shape\n"
        "residual connection helps information flow\n"
        "layernorm helps stabilize training\n"
        "feed forward network improves representation\n"
        "causal mask prevents looking into the future\n"
        "language model predicts next token\n"
        "the model reads previous tokens and predicts the next token\n"
        "attention lets every position read useful previous positions\n"
        "many small patterns become better text after training\n"
    )
    text_path.write_text(base_text * 6, encoding="utf-8")

text = text_path.read_text(encoding="utf-8")

print("Current dir:", current_dir)
print("Data dir:", data_dir)
print("Output dir:", output_dir)

print("\nRaw text preview:")
print(repr(text[:260]))

# 1. 字符级 tokenizer：建立 vocab / encode / decode。
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}


def encode(s):
    return [stoi[ch] for ch in s]


def decode(ids):
    return "".join([itos[int(i)] for i in ids])


encoded = np.array(encode(text), dtype=np.int32)

print("\nVocab:")
print(chars)
print("vocab_size:", vocab_size)
print("data length:", len(encoded))

# 2. train / val split。
n = int(0.9 * len(encoded))
train_data = encoded[:n]
val_data = encoded[n:]

# 3. Tiny GPT 配置。
block_size = 32
batch_size = 32
n_embd = 64
num_heads = 4
num_layers = 2
head_size = n_embd // num_heads
learning_rate = 3e-3
max_iters = 2000
eval_interval = 100
eval_iters = 20

assert n_embd % num_heads == 0

print("\nHyperparameters:")
print("block_size:", block_size)
print("batch_size:", batch_size)
print("n_embd:", n_embd)
print("num_heads:", num_heads)
print("num_layers:", num_layers)
print("head_size:", head_size)
print("learning_rate:", learning_rate)
print("max_iters:", max_iters)
print("eval_interval:", eval_interval)


def get_batch(split):
    source = train_data if split == "train" else val_data

    if len(source) <= block_size + 1:
        raise ValueError(f"{split} data is too short for block_size={block_size}")

    starts = np.random.randint(
        0,
        len(source) - block_size - 1,
        size=(batch_size,),
    )

    x = np.stack([source[i : i + block_size] for i in starts]).astype(np.int32)
    y = np.stack([source[i + 1 : i + block_size + 1] for i in starts]).astype(
        np.int32
    )

    return mx.array(x), mx.array(y)


class MultiHeadCausalSelfAttention(nn.Module):
    def __init__(self, n_embd, num_heads):
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
        # x: [batch, seq_len, n_embd]
        batch, seq_len, _ = x.shape
        x = x.reshape(batch, seq_len, self.num_heads, self.head_size)
        # [batch, seq_len, num_heads, head_size]
        x = mx.transpose(x, (0, 2, 1, 3))
        # [batch, num_heads, seq_len, head_size]
        return x

    def _merge_heads(self, x):
        # x: [batch, num_heads, seq_len, head_size]
        batch, num_heads, seq_len, head_size = x.shape
        x = mx.transpose(x, (0, 2, 1, 3))
        # [batch, seq_len, num_heads, head_size]
        x = x.reshape(batch, seq_len, num_heads * head_size)
        # [batch, seq_len, n_embd]
        return x

    def __call__(self, x, return_attention=False):
        # x: [batch, seq_len, n_embd]
        _, seq_len, _ = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        # each: [batch, seq_len, n_embd]

        q_heads = self._split_heads(q)
        k_heads = self._split_heads(k)
        v_heads = self._split_heads(v)
        # each: [batch, num_heads, seq_len, head_size]

        scores = q_heads @ mx.transpose(k_heads, (0, 1, 3, 2))
        scores = scores / math.sqrt(self.head_size)
        # [batch, num_heads, seq_len, seq_len]

        mask_np = np.tril(np.ones((seq_len, seq_len), dtype=np.float32))
        mask = mx.array(mask_np).reshape(1, 1, seq_len, seq_len)

        masked_scores = mx.where(
            mask == 1,
            scores,
            mx.full(scores.shape, -1e9),
        )

        attn_weights = nn.softmax(masked_scores, axis=-1)
        # [batch, num_heads, seq_len, seq_len]

        out_heads = attn_weights @ v_heads
        # [batch, num_heads, seq_len, head_size]

        out = self._merge_heads(out_heads)
        # [batch, seq_len, n_embd]

        out = self.proj(out)
        # [batch, seq_len, n_embd]

        if return_attention:
            return out, attn_weights

        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.linear1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear2 = nn.Linear(4 * n_embd, n_embd)

    def __call__(self, x):
        # x: [batch, seq_len, n_embd]
        x = self.linear1(x)
        x = nn.gelu(x)
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, n_embd, num_heads):
        super().__init__()
        self.attn = MultiHeadCausalSelfAttention(n_embd, num_heads)
        self.ffwd = FeedForward(n_embd)
        # Pre-LN：进入 attention / MLP 前先归一化。
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def __call__(self, x, return_attention=False):
        # x: [batch, seq_len, n_embd]
        if return_attention:
            attn_out, attn_weights = self.attn(
                self.ln1(x),
                return_attention=True,
            )
        else:
            attn_out = self.attn(self.ln1(x))
            attn_weights = None

        # residual connection 1
        x = x + attn_out

        ffwd_out = self.ffwd(self.ln2(x))

        # residual connection 2
        x = x + ffwd_out

        if return_attention:
            return x, attn_weights, attn_out, ffwd_out

        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd, num_heads, num_layers):
        super().__init__()

        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.token_embedding_table = mx.random.normal((vocab_size, n_embd)) * 0.02
        self.position_embedding_table = mx.random.normal((block_size, n_embd)) * 0.02

        self.blocks = [
            TransformerBlock(n_embd=n_embd, num_heads=num_heads)
            for _ in range(num_layers)
        ]
        self.final_ln = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def __call__(self, idx, return_debug=False):
        # idx: [batch, seq_len]
        _, seq_len = idx.shape

        if seq_len > self.block_size:
            raise ValueError(f"seq_len={seq_len} exceeds block_size={self.block_size}")

        token_emb = self.token_embedding_table[idx]
        # [batch, seq_len, n_embd]

        positions = mx.arange(seq_len)
        pos_emb = self.position_embedding_table[positions]
        # [seq_len, n_embd]

        x = token_emb + pos_emb
        # [batch, seq_len, n_embd]，pos_emb 自动广播到 batch 维度。

        block_outputs = []
        attention_weights = []

        for block in self.blocks:
            if return_debug:
                x, attn_weights, _, _ = block(x, return_attention=True)
                block_outputs.append(x)
                attention_weights.append(attn_weights)
            else:
                x = block(x)

        normalized = self.final_ln(x)
        logits = self.lm_head(normalized)
        # [batch, seq_len, vocab_size]

        if return_debug:
            return {
                "idx": idx,
                "token_emb": token_emb,
                "pos_emb": pos_emb,
                "x": token_emb + pos_emb,
                "block_outputs": block_outputs,
                "attention_weights": attention_weights,
                "normalized": normalized,
                "logits": logits,
            }

        return logits

    def generate(self, start_ids, max_new_tokens, temperature=0.9, top_k=8):
        if isinstance(start_ids, int):
            ids = [int(start_ids)]
        else:
            ids = [int(i) for i in start_ids]

        for _ in range(max_new_tokens):
            context = ids[-self.block_size :]
            idx = mx.array([context], dtype=mx.int32)

            logits = self(idx)
            logits_last = logits[0, -1, :] / temperature

            probs = nn.softmax(logits_last, axis=-1)
            mx.eval(probs)

            probs_np = np.array(probs, dtype=np.float64)
            probs_np = np.maximum(probs_np, 0.0)

            if top_k is not None and top_k < len(probs_np):
                keep = np.argpartition(probs_np, -top_k)[-top_k:]
                filtered = np.zeros_like(probs_np)
                filtered[keep] = probs_np[keep]
                probs_np = filtered

            probs_np = probs_np / probs_np.sum()
            next_id = np.random.choice(len(probs_np), p=probs_np)
            ids.append(int(next_id))

        return ids


model = TinyGPT(
    vocab_size=vocab_size,
    block_size=block_size,
    n_embd=n_embd,
    num_heads=num_heads,
    num_layers=num_layers,
)
optimizer = optim.AdamW(learning_rate=learning_rate)


def loss_fn(model, idx, targets):
    logits = model(idx)
    batch, seq_len, channels = logits.shape

    logits_flat = logits.reshape(batch * seq_len, channels)
    targets_flat = targets.reshape(batch * seq_len)

    loss = nn.losses.cross_entropy(
        logits_flat,
        targets_flat,
        reduction="mean",
    )

    return loss


value_and_grad_fn = nn.value_and_grad(model, loss_fn)


def estimate_loss():
    out = {}

    for split in ["train", "val"]:
        losses = []

        for _ in range(eval_iters):
            xb, yb = get_batch(split)
            loss = loss_fn(model, xb, yb)
            mx.eval(loss)
            losses.append(float(loss))

        out[split] = sum(losses) / len(losses)

    return out


print("\nInitial batch check:")

xb, yb = get_batch("train")
debug = model(xb, return_debug=True)
loss = loss_fn(model, xb, yb)

mx.eval(
    xb,
    yb,
    debug["token_emb"],
    debug["pos_emb"],
    debug["x"],
    *debug["block_outputs"],
    *debug["attention_weights"],
    debug["normalized"],
    debug["logits"],
    loss,
)

print("idx shape:", debug["idx"].shape)
print("y batch shape:", yb.shape)
print("token_emb shape:", debug["token_emb"].shape)
print("pos_emb shape:", debug["pos_emb"].shape)
print("combined x shape:", debug["x"].shape)

for layer_idx, block_out in enumerate(debug["block_outputs"]):
    print(f"block {layer_idx} output shape:", block_out.shape)

for layer_idx, attn in enumerate(debug["attention_weights"]):
    print(f"block {layer_idx} attention weights shape:", attn.shape)

print("normalized shape:", debug["normalized"].shape)
print("logits shape:", debug["logits"].shape)
print("initial loss:", float(loss))

print("\nFirst input sequence:")
print(repr(decode(np.array(xb[0]).tolist())))

print("\nFirst target sequence:")
print(repr(decode(np.array(yb[0]).tolist())))

print("\nTraining...")

history = []
start_time = time.perf_counter()

for step in range(max_iters):
    xb, yb = get_batch("train")
    loss, grads = value_and_grad_fn(model, xb, yb)
    optimizer.update(model, grads)
    mx.eval(loss, model.parameters(), optimizer.state)

    if step % eval_interval == 0 or step == max_iters - 1:
        losses = estimate_loss()
        elapsed = time.perf_counter() - start_time
        history.append(
            {
                "step": step,
                "train_loss": losses["train"],
                "val_loss": losses["val"],
                "elapsed_sec": elapsed,
            }
        )
        print(
            f"step={step:04d} "
            f"train_loss={losses['train']:.4f} "
            f"val_loss={losses['val']:.4f} "
            f"elapsed={elapsed:.1f}s"
        )

total_time = time.perf_counter() - start_time
print(f"\nTraining time: {total_time:.1f}s")

print("\nSaving loss curve...")

steps = [item["step"] for item in history]
train_losses = [item["train_loss"] for item in history]
val_losses = [item["val_loss"] for item in history]

plt.figure(figsize=(8, 5))
plt.plot(steps, train_losses, label="train loss")
plt.plot(steps, val_losses, label="val loss")
plt.xlabel("step")
plt.ylabel("cross entropy loss")
plt.title("Tiny GPT Training Loss")
plt.legend()
plt.grid(True, alpha=0.3)
loss_curve_path = output_dir / "loss_curve.png"
plt.savefig(loss_curve_path, dpi=150, bbox_inches="tight")
plt.close()
print("Saved:", loss_curve_path)

history_path = output_dir / "loss_history.json"
history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
print("Saved:", history_path)

print("\nGeneration:")

prompt = "hello "
generated_ids = model.generate(
    encode(prompt),
    max_new_tokens=260,
    temperature=0.85,
    top_k=8,
)
generated_text = decode(generated_ids)

print(repr(generated_text))

generated_path = output_dir / "generated_text.txt"
generated_path.write_text(generated_text, encoding="utf-8")
print("Saved:", generated_path)

print("\nSaving attention maps after training...")

sample_text = "tiny gpt learns "
sample_ids = encode(sample_text)
sample_idx = mx.array([sample_ids], dtype=mx.int32)

debug = model(sample_idx, return_debug=True)
mx.eval(*debug["attention_weights"])

sample_tokens = list(sample_text)
seq_len = len(sample_tokens)

layers_to_show = [0, num_layers - 1] if num_layers > 1 else [0]
fig, axes = plt.subplots(
    len(layers_to_show),
    num_heads,
    figsize=(4 * num_heads, 4 * len(layers_to_show)),
)

if len(layers_to_show) == 1:
    axes = np.array([axes])

for row_idx, layer_idx in enumerate(layers_to_show):
    attn_np = np.array(debug["attention_weights"][layer_idx][0])
    # shape: [num_heads, seq_len, seq_len]

    for head_idx in range(num_heads):
        ax = axes[row_idx, head_idx]
        ax.imshow(attn_np[head_idx], cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_title(f"layer {layer_idx} head {head_idx}")
        ax.set_xticks(range(seq_len))
        ax.set_yticks(range(seq_len))
        ax.set_xticklabels(sample_tokens, fontsize=8)
        ax.set_yticklabels(sample_tokens, fontsize=8)
        ax.set_xlabel("attend to")
        ax.set_ylabel("current")

fig.suptitle("Tiny GPT Attention Maps")
fig.tight_layout()

attention_path = output_dir / "tiny_gpt_attention_maps.png"
fig.savefig(attention_path, dpi=150)
plt.close(fig)
print("Saved:", attention_path)

print("\nSaving model parameters...")

weights_path = output_dir / "tiny_gpt_model.safetensors"
flat_params = dict(tree_flatten(model.parameters()))
mx.eval(*flat_params.values())
mx.save_safetensors(str(weights_path), flat_params)
print("Saved:", weights_path)

config_path = output_dir / "tiny_gpt_config.json"
config = {
    "vocab_size": vocab_size,
    "block_size": block_size,
    "batch_size": batch_size,
    "n_embd": n_embd,
    "num_heads": num_heads,
    "num_layers": num_layers,
    "head_size": head_size,
    "learning_rate": learning_rate,
    "max_iters": max_iters,
    "eval_interval": eval_interval,
    "chars": chars,
    "stoi": stoi,
    "itos": {str(k): v for k, v in itos.items()},
}
config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
print("Saved:", config_path)

print("\nDone.")
