from pathlib import Path

import numpy as np

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim


print("=== MLX Bigram Language Model ===")

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
        "we train a tiny language model\n"
        "language model predicts next token\n"
        "hello hello hello ai ai ai\n",
        encoding="utf-8",
    )

text = text_path.read_text(encoding="utf-8")

print("\nRaw text:")
print(repr(text[:100]))

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

block_size = 8
batch_size = 16


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


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        # 手写 embedding table。
        # 每个 token id 对应一行 vocab_size 维 logits。
        self.token_embedding_table = mx.random.normal((vocab_size, vocab_size)) * 0.01

    def __call__(self, idx):
        # idx shape: [batch, seq_len]
        logits = self.token_embedding_table[idx]
        # logits shape: [batch, seq_len, vocab_size]
        return logits

    def generate(self, start_id, max_new_tokens):
        ids = [int(start_id)]

        for _ in range(max_new_tokens):
            idx = mx.array([ids], dtype=mx.int32)
            logits = self(idx)

            # 只取最后一个位置的预测。
            logits_last = logits[0, -1, :]
            probs = nn.softmax(logits_last, axis=-1)

            mx.eval(probs)

            probs_np = np.array(probs)
            probs_np = probs_np / probs_np.sum()

            next_id = np.random.choice(len(probs_np), p=probs_np)
            ids.append(int(next_id))

        return ids


model = BigramLanguageModel(vocab_size)

optimizer = optim.AdamW(learning_rate=1e-2)

max_iters = 1000
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
logits = model(xb)
loss = loss_fn(model, xb, yb)

mx.eval(logits, loss)

print("x shape:", xb.shape)
print("y shape:", yb.shape)
print("logits shape:", logits.shape)
print("initial loss:", float(loss))

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
generated = model.generate(start_id, max_new_tokens=120)
generated_text = decode(generated)

print(repr(generated_text))

output_path = output_dir / "mlx_generated_text.txt"
output_path.write_text(generated_text, encoding="utf-8")

print("Saved:", output_path)
print("\nDone.")
