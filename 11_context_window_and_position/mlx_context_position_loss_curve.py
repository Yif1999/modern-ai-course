from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim


print("=== MLX Context + Position Loss Curve ===")

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
        "we train a tiny language model\n"
        "language model predicts next token\n"
        "context window gives a fixed length input\n"
        "position tells the model where a token is\n"
        "hello hello hello ai ai ai\n",
        encoding="utf-8",
    )

text = text_path.read_text(encoding="utf-8")

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}


def encode(s):
    return [stoi[ch] for ch in s]


def decode(ids):
    return "".join([itos[int(i)] for i in ids])


encoded = np.array(encode(text), dtype=np.int32)

n = int(0.9 * len(encoded))
train_data = encoded[:n]
val_data = encoded[n:]

block_size = 16
batch_size = 16
n_embd = 32
max_iters = 5000
eval_interval = 50

num_train_windows = len(train_data) - block_size - 1
approx_steps_per_epoch = max(1, num_train_windows // batch_size)

print("vocab_size:", vocab_size)
print("data length:", len(encoded))
print("train token length:", len(train_data))
print("val token length:", len(val_data))
print("block_size:", block_size)
print("batch_size:", batch_size)
print("train windows:", num_train_windows)
print("approx steps per epoch:", approx_steps_per_epoch)
print("max_iters:", max_iters)
print("approx epochs:", round(max_iters / approx_steps_per_epoch, 2))


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


class ContextPositionLanguageModel(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd):
        super().__init__()

        self.block_size = block_size
        self.token_embedding_table = mx.random.normal((vocab_size, n_embd)) * 0.02
        self.position_embedding_table = mx.random.normal((block_size, n_embd)) * 0.02
        self.lm_head_w = mx.random.normal((n_embd, vocab_size)) * 0.02
        self.lm_head_b = mx.zeros((vocab_size,))

    def __call__(self, idx):
        batch, seq_len = idx.shape

        if seq_len > self.block_size:
            raise ValueError(f"seq_len={seq_len} exceeds block_size={self.block_size}")

        token_emb = self.token_embedding_table[idx]
        positions = mx.arange(seq_len)
        pos_emb = self.position_embedding_table[positions]
        x = token_emb + pos_emb
        logits = x @ self.lm_head_w + self.lm_head_b
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


model = ContextPositionLanguageModel(vocab_size, block_size, n_embd)
optimizer = optim.AdamW(learning_rate=1e-2)


def loss_fn(model, idx, targets):
    logits = model(idx)
    batch, seq_len, channels = logits.shape
    logits_flat = logits.reshape(batch * seq_len, channels)
    targets_flat = targets.reshape(batch * seq_len)
    return nn.losses.cross_entropy(logits_flat, targets_flat, reduction="mean")


value_and_grad_fn = nn.value_and_grad(model, loss_fn)


def estimate_loss(num_batches=20):
    out = {}

    for split in ["train", "val"]:
        losses = []

        for _ in range(num_batches):
            xb, yb = get_batch(split)
            loss = loss_fn(model, xb, yb)
            mx.eval(loss)
            losses.append(float(loss))

        out[split] = sum(losses) / len(losses)

    return out


def top_next_token_predictions(model, context_text, top_k=5):
    context_ids = encode(context_text)[-block_size:]
    idx = mx.array([context_ids], dtype=mx.int32)
    logits = model(idx)
    logits_last = logits[0, -1, :]
    probs = nn.softmax(logits_last, axis=-1)
    mx.eval(probs)

    probs_np = np.array(probs)
    top_ids = np.argsort(probs_np)[-top_k:][::-1]

    return [
        (decode([token_id]), float(probs_np[token_id]))
        for token_id in top_ids
    ]


curve_steps = []
train_losses = []
val_losses = []

print("\nTraining...")

for step in range(max_iters):
    xb, yb = get_batch("train")
    loss, grads = value_and_grad_fn(model, xb, yb)
    optimizer.update(model, grads)
    mx.eval(loss, model.parameters(), optimizer.state)

    if step % eval_interval == 0 or step == max_iters - 1:
        losses = estimate_loss()
        approx_epoch = step / approx_steps_per_epoch

        curve_steps.append(step)
        train_losses.append(losses["train"])
        val_losses.append(losses["val"])

        print(
            f"step={step:04d} "
            f"approx_epoch={approx_epoch:06.2f} "
            f"train_loss={losses['train']:.4f} "
            f"val_loss={losses['val']:.4f}"
        )


final_losses = estimate_loss(num_batches=50)
print("\nFinal estimate:")
print("train_loss:", round(final_losses["train"], 4))
print("val_loss:", round(final_losses["val"], 4))
print("train_perplexity:", round(math.exp(final_losses["train"]), 4))
print("val_perplexity:", round(math.exp(final_losses["val"]), 4))

curve_path = output_dir / "mlx_loss_curve.png"

plt.figure(figsize=(9, 5))
plt.plot(curve_steps, train_losses, label="train loss")
plt.plot(curve_steps, val_losses, label="val loss")
plt.xlabel("training iteration")
plt.ylabel("cross entropy loss")
plt.title("MLX Context + Position Model Loss Curve")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(curve_path, dpi=150)
plt.close()

generated = model.generate(stoi["h"], max_new_tokens=160)
generated_text = decode(generated)
generated_path = output_dir / "mlx_loss_curve_generated_text.txt"
generated_path.write_text(generated_text, encoding="utf-8")

weights_path = output_dir / "mlx_context_position_model.safetensors"
mx.eval(model.parameters())
model.save_weights(str(weights_path))

prediction_report_path = output_dir / "mlx_prediction_report.txt"
prediction_contexts = [
    "h",
    "he",
    "hel",
    "hello ",
    "we ",
    "token ",
    "model ",
    "hello a",
    "model a",
]

prediction_lines = []

print("\nNext-token prediction probes:")

for context_text in prediction_contexts:
    predictions = top_next_token_predictions(model, context_text)
    line = f"context={context_text!r} -> {predictions}"
    prediction_lines.append(line)
    print(line)

prediction_report_path.write_text(
    "\n".join(prediction_lines) + "\n",
    encoding="utf-8",
)

print("\nGeneration:")
print(repr(generated_text))
print("Saved curve:", curve_path)
print("Saved generated text:", generated_path)
print("Saved weights:", weights_path)
print("Saved prediction report:", prediction_report_path)
print("\nDone.")
