from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim


print("=== MLX Multi-Head Causal Self-Attention Language Model ===")

mx.random.seed(42)
np.random.seed(42)

current_dir = Path(__file__).resolve().parent
data_dir = current_dir / "data"
output_dir = current_dir / "outputs"
data_dir.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)

text_path = data_dir / "tiny_text.txt"
if not text_path.exists():
    text_path.write_text(
        "hello ai lab\n"
        "hello mlx\n"
        "hello tiny gpt\n"
        "we learn token embedding\n"
        "we learn position embedding\n"
        "we learn self attention\n"
        "we learn multi head attention\n"
        "one head can look at one relation\n"
        "many heads can look at many relations\n"
        "attention lets tokens look at previous tokens\n"
        "causal mask prevents looking into the future\n"
        "language model predicts next token\n"
        "tiny gpt uses attention and feed forward layers\n"
        "hello hello hello ai ai ai\n",
        encoding="utf-8",
    )

text = text_path.read_text(encoding="utf-8")
print("\nRaw text preview:")
print(repr(text[:180]))

# 1. 建立字符级词表。
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

# 2. 训练集 / 验证集切分。
n = int(0.9 * len(encoded))
train_data = encoded[:n]
val_data = encoded[n:]

block_size = 16
batch_size = 16
n_embd = 32
num_heads = 4
head_size = n_embd // num_heads
assert n_embd % num_heads == 0

print("\nHyperparameters:")
print("block_size:", block_size)
print("batch_size:", batch_size)
print("n_embd:", n_embd)
print("num_heads:", num_heads)
print("head_size:", head_size)


def get_batch(split):
    source = train_data if split == "train" else val_data
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


class MultiHeadCausalAttentionLM(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd, num_heads):
        super().__init__()
        if n_embd % num_heads != 0:
            raise ValueError("n_embd must be divisible by num_heads")

        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        self.num_heads = num_heads
        self.head_size = n_embd // num_heads

        # token embedding: [vocab_size, n_embd]
        self.token_embedding_table = mx.random.normal((vocab_size, n_embd)) * 0.02

        # position embedding: [block_size, n_embd]
        self.position_embedding_table = mx.random.normal((block_size, n_embd)) * 0.02

        # 一次性生成所有 heads 的 Q/K/V。
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)

        # 多个 head concat 后，用输出投影重新混合信息。
        self.proj = nn.Linear(n_embd, n_embd)

        # 语言模型头：每个位置输出 vocab_size 个 logits。
        self.lm_head = nn.Linear(n_embd, vocab_size)

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

    def __call__(self, idx, return_attention=False):
        # idx shape: [batch, seq_len]
        batch, seq_len = idx.shape
        if seq_len > self.block_size:
            raise ValueError(f"seq_len={seq_len} exceeds block_size={self.block_size}")

        token_emb = self.token_embedding_table[idx]
        # [batch, seq_len, n_embd]

        positions = mx.arange(seq_len)
        pos_emb = self.position_embedding_table[positions]
        # [seq_len, n_embd]

        x = token_emb + pos_emb
        # [batch, seq_len, n_embd]

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

        # causal mask: 下三角为 1，右上角为 0，禁止看未来 token。
        mask_np = np.tril(np.ones((seq_len, seq_len), dtype=np.float32))
        mask = mx.array(mask_np)
        masked_scores = mx.where(
            mask.reshape(1, 1, seq_len, seq_len) == 1,
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

        logits = self.lm_head(out)
        # [batch, seq_len, vocab_size]

        if return_attention:
            return {
                "logits": logits,
                "token_emb": token_emb,
                "pos_emb": pos_emb,
                "x": x,
                "q": q,
                "k": k,
                "v": v,
                "q_heads": q_heads,
                "k_heads": k_heads,
                "v_heads": v_heads,
                "scores": scores,
                "attn_weights": attn_weights,
                "out_heads": out_heads,
                "out": out,
            }

        return logits

    def generate(self, start_id, max_new_tokens):
        ids = [int(start_id)]
        for _ in range(max_new_tokens):
            context = ids[-self.block_size :]
            idx = mx.array([context], dtype=mx.int32)
            logits = self(idx)
            logits_last = logits[0, -1, :]
            probs = nn.softmax(logits_last, axis=-1)
            mx.eval(probs)

            probs_np = np.array(probs)
            probs_np = probs_np / probs_np.sum()
            next_id = np.random.choice(len(probs_np), p=probs_np)
            ids.append(int(next_id))
        return ids


model = MultiHeadCausalAttentionLM(
    vocab_size=vocab_size,
    block_size=block_size,
    n_embd=n_embd,
    num_heads=num_heads,
)
optimizer = optim.AdamW(learning_rate=1e-2)

max_iters = 1200
eval_interval = 100


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
        for _ in range(20):
            xb, yb = get_batch(split)
            loss = loss_fn(model, xb, yb)
            mx.eval(loss)
            losses.append(float(loss))
        out[split] = sum(losses) / len(losses)
    return out


print("\nInitial batch check:")
xb, yb = get_batch("train")
debug = model(xb, return_attention=True)
loss = loss_fn(model, xb, yb)
mx.eval(
    debug["logits"],
    debug["token_emb"],
    debug["pos_emb"],
    debug["x"],
    debug["q"],
    debug["k"],
    debug["v"],
    debug["q_heads"],
    debug["k_heads"],
    debug["v_heads"],
    debug["scores"],
    debug["attn_weights"],
    debug["out_heads"],
    debug["out"],
    loss,
)

print("x batch shape:", xb.shape)
print("y batch shape:", yb.shape)
print("token_emb shape:", debug["token_emb"].shape)
print("pos_emb shape:", debug["pos_emb"].shape)
print("combined x shape:", debug["x"].shape)
print("q shape:", debug["q"].shape)
print("k shape:", debug["k"].shape)
print("v shape:", debug["v"].shape)
print("q_heads shape:", debug["q_heads"].shape)
print("k_heads shape:", debug["k_heads"].shape)
print("v_heads shape:", debug["v_heads"].shape)
print("scores shape:", debug["scores"].shape)
print("attention weights shape:", debug["attn_weights"].shape)
print("out_heads shape:", debug["out_heads"].shape)
print("merged out shape:", debug["out"].shape)
print("logits shape:", debug["logits"].shape)
print("initial loss:", float(loss))

print("\nFirst input sequence:")
print(repr(decode(np.array(xb[0]).tolist())))

print("\nFirst target sequence:")
print(repr(decode(np.array(yb[0]).tolist())))

print("\nTraining...")
for step in range(max_iters):
    xb, yb = get_batch("train")
    loss, grads = value_and_grad_fn(model, xb, yb)
    optimizer.update(model, grads)
    mx.eval(loss, model.parameters(), optimizer.state)

    if step % eval_interval == 0 or step == max_iters - 1:
        losses = estimate_loss()
        print(
            f"step={step:04d} "
            f"train_loss={losses['train']:.4f} "
            f"val_loss={losses['val']:.4f}"
        )

print("\nGeneration:")
start_id = stoi["h"]
generated = model.generate(start_id, max_new_tokens=200)
generated_text = decode(generated)
print(repr(generated_text))

output_path = output_dir / "mlx_multi_head_generated_text.txt"
output_path.write_text(generated_text, encoding="utf-8")
print("Saved:", output_path)

print("\nSaving attention maps after training...")
sample_text = "hello ai"
sample_ids = encode(sample_text)
sample_idx = mx.array([sample_ids], dtype=mx.int32)
debug = model(sample_idx, return_attention=True)
mx.eval(debug["attn_weights"])

attn_np = np.array(debug["attn_weights"][0])
# shape: [num_heads, seq_len, seq_len]
sample_tokens = list(sample_text)
seq_len = len(sample_tokens)

fig, axes = plt.subplots(1, num_heads, figsize=(4 * num_heads, 4))
if num_heads == 1:
    axes = [axes]

for h, ax in enumerate(axes):
    ax.imshow(attn_np[h], cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_title(f"head {h}")
    ax.set_xticks(range(seq_len))
    ax.set_yticks(range(seq_len))
    ax.set_xticklabels(sample_tokens)
    ax.set_yticklabels(sample_tokens)
    ax.set_xlabel("attend to")
    ax.set_ylabel("current")

fig.suptitle("Trained Multi-Head Causal Attention Maps")
fig.tight_layout()

attention_path = output_dir / "trained_multi_head_attention_maps.png"
fig.savefig(attention_path, dpi=150)
plt.close(fig)
print("Saved:", attention_path)

print("\nImportant reminder:")
print("This is multi-head causal self-attention.")
print("It is still not a full Transformer block.")
print("No residual connection, no LayerNorm, no FeedForward MLP yet.")

print("\nDone.")
