from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim


print("=== MLX Single-Head Causal Self-Attention Language Model ===")

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
        "attention lets tokens look at previous tokens\n"
        "causal mask prevents looking into the future\n"
        "language model predicts next token\n"
        "tiny gpt uses attention and feed forward layers\n"
        "hello hello hello ai ai ai\n",
        encoding="utf-8",
    )

text = text_path.read_text(encoding="utf-8")

print("\nRaw text preview:")
print(repr(text[:160]))

# 1. vocab
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

# 2. train / val split
n = int(0.9 * len(encoded))
train_data = encoded[:n]
val_data = encoded[n:]

block_size = 16
batch_size = 16
n_embd = 32
head_size = 32


def get_batch(split):
    source = train_data if split == "train" else val_data

    starts = np.random.randint(
        0,
        len(source) - block_size - 1,
        size=(batch_size,),
    )

    x = np.stack([
        source[i:i + block_size]
        for i in starts
    ]).astype(np.int32)

    y = np.stack([
        source[i + 1:i + block_size + 1]
        for i in starts
    ]).astype(np.int32)

    return mx.array(x), mx.array(y)


class SingleHeadCausalAttentionLM(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd, head_size):
        super().__init__()

        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        self.head_size = head_size

        # token embedding: [vocab_size, n_embd]
        self.token_embedding_table = (
            mx.random.normal((vocab_size, n_embd)) * 0.02
        )

        # position embedding: [block_size, n_embd]
        self.position_embedding_table = (
            mx.random.normal((block_size, n_embd)) * 0.02
        )

        # Q, K, V projections
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # language model head
        self.lm_head = nn.Linear(head_size, vocab_size)

    def __call__(self, idx, return_attention=False):
        # idx shape: [batch, seq_len]
        batch, seq_len = idx.shape

        if seq_len > self.block_size:
            raise ValueError(
                f"seq_len={seq_len} exceeds block_size={self.block_size}"
            )

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
        # each: [batch, seq_len, head_size]

        scores = q @ mx.transpose(k, (0, 2, 1))
        scores = scores / math.sqrt(self.head_size)
        # [batch, seq_len, seq_len]

        # causal mask: only allow each position to attend to itself and previous positions
        mask_np = np.tril(np.ones((seq_len, seq_len), dtype=np.float32))
        mask = mx.array(mask_np)

        masked_scores = mx.where(
            mask.reshape(1, seq_len, seq_len) == 1,
            scores,
            mx.full(scores.shape, -1e9),
        )

        attn_weights = nn.softmax(masked_scores, axis=-1)
        # [batch, seq_len, seq_len]

        out = attn_weights @ v
        # [batch, seq_len, head_size]

        logits = self.lm_head(out)
        # [batch, seq_len, vocab_size]

        if return_attention:
            return logits, attn_weights, token_emb, pos_emb, x, q, k, v

        return logits

    def generate(self, start_id, max_new_tokens):
        ids = [int(start_id)]

        for _ in range(max_new_tokens):
            context = ids[-self.block_size:]
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


model = SingleHeadCausalAttentionLM(
    vocab_size=vocab_size,
    block_size=block_size,
    n_embd=n_embd,
    head_size=head_size,
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
logits, attn_weights, token_emb, pos_emb, x, q, k, v = model(
    xb,
    return_attention=True,
)
loss = loss_fn(model, xb, yb)

mx.eval(logits, attn_weights, token_emb, pos_emb, x, q, k, v, loss)

print("x batch shape:", xb.shape)
print("y batch shape:", yb.shape)
print("token_emb shape:", token_emb.shape)
print("pos_emb shape:", pos_emb.shape)
print("combined x shape:", x.shape)
print("q shape:", q.shape)
print("k shape:", k.shape)
print("v shape:", v.shape)
print("attention weights shape:", attn_weights.shape)
print("logits shape:", logits.shape)
print("initial loss:", float(loss))

print("\nFirst input sequence:")
print(repr(decode(np.array(xb[0]))))

print("\nFirst target sequence:")
print(repr(decode(np.array(yb[0]))))

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
generated = model.generate(start_id, max_new_tokens=180)
generated_text = decode(generated)

print(repr(generated_text))

output_path = output_dir / "mlx_attention_generated_text.txt"
output_path.write_text(generated_text, encoding="utf-8")

print("Saved:", output_path)

# 保存训练后 attention map
print("\nSaving attention map after training...")

sample_text = "hello ai"
sample_ids = encode(sample_text)
sample_idx = mx.array([sample_ids], dtype=mx.int32)

logits, attn_weights, *_ = model(sample_idx, return_attention=True)
mx.eval(attn_weights)

attn_np = np.array(attn_weights[0])
sample_tokens = list(sample_text)

plt.figure(figsize=(7, 6))
plt.imshow(attn_np, cmap="viridis")
plt.colorbar(label="attention weight")
plt.xticks(range(len(sample_tokens)), sample_tokens)
plt.yticks(range(len(sample_tokens)), sample_tokens)
plt.xlabel("tokens being attended to")
plt.ylabel("current token")
plt.title("Trained Single-Head Causal Attention Map")
plt.tight_layout()

attention_path = output_dir / "trained_attention_map.png"
plt.savefig(attention_path, dpi=150)
plt.close()

print("Saved:", attention_path)

print("\nImportant reminder:")
print("This is only single-head causal self-attention.")
print("It is not yet a full Transformer block.")
print("No multi-head attention, no residual connection, no LayerNorm, no MLP block yet.")

print("\nDone.")
