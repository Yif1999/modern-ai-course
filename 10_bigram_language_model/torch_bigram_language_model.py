from pathlib import Path

import torch
import torch.nn.functional as F


print("=== PyTorch Bigram Language Model ===")

torch.manual_seed(42)

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


data = torch.tensor(encode(text), dtype=torch.long)

print("\nVocab:")
print(chars)
print("vocab_size:", vocab_size)
print("data length:", len(data))

# 2. train / val split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

block_size = 8
batch_size = 16


def get_batch(split):
    source = train_data if split == "train" else val_data

    # 每个样本需要 block_size + 1 个 token：
    # 前 block_size 个做输入，后 block_size 个做目标。
    ix = torch.randint(0, len(source) - block_size - 1, (batch_size,))

    x = torch.stack([source[i : i + block_size] for i in ix])
    y = torch.stack([source[i + 1 : i + block_size + 1] for i in ix])

    return x, y


class BigramLanguageModel(torch.nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        # 这里的 embedding table 直接输出 vocab_size 维 logits。
        # 每个当前 token id 会查出一行 logits，用来预测下一个 token。
        self.token_embedding_table = torch.nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx shape: [batch, seq_len]
        logits = self.token_embedding_table(idx)
        # logits shape: [batch, seq_len, vocab_size]

        loss = None

        if targets is not None:
            batch, seq_len, channels = logits.shape

            logits_flat = logits.view(batch * seq_len, channels)
            targets_flat = targets.view(batch * seq_len)

            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx shape: [batch, seq_len]
        for _ in range(max_new_tokens):
            logits, _ = self(idx)

            # 只取最后一个位置的预测。
            logits_last = logits[:, -1, :]
            probs = F.softmax(logits_last, dim=-1)

            # 按概率采样下一个 token。
            idx_next = torch.multinomial(probs, num_samples=1)

            # 拼到当前序列后面。
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


model = BigramLanguageModel(vocab_size)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

max_iters = 1000
eval_interval = 100


@torch.no_grad()
def estimate_loss():
    model.eval()

    out = {}

    for split in ["train", "val"]:
        losses = []

        for _ in range(20):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses.append(loss.item())

        out[split] = sum(losses) / len(losses)

    model.train()
    return out


print("\nInitial batch check:")
xb, yb = get_batch("train")
logits, loss = model(xb, yb)

print("x shape:", tuple(xb.shape))
print("y shape:", tuple(yb.shape))
print("logits shape:", tuple(logits.shape))
print("initial loss:", loss.item())

print("\nTraining...")

for step in range(max_iters):
    xb, yb = get_batch("train")

    logits, loss = model(xb, yb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % eval_interval == 0 or step == max_iters - 1:
        losses = estimate_loss()
        print(
            f"step={step:04d} "
            f"train_loss={losses['train']:.4f} "
            f"val_loss={losses['val']:.4f}"
        )


print("\nGeneration:")

start_id = torch.tensor([[stoi["h"]]], dtype=torch.long)
generated = model.generate(start_id, max_new_tokens=120)[0].tolist()
generated_text = decode(generated)

print(repr(generated_text))

output_path = output_dir / "torch_generated_text.txt"
output_path.write_text(generated_text, encoding="utf-8")

print("Saved:", output_path)
print("\nDone.")
