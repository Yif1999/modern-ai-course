from pathlib import Path

import torch
import torch.nn.functional as F


print("=== PyTorch Context Window + Position Embedding Model ===")

torch.manual_seed(42)

current_dir = Path(__file__).resolve().parent
data_dir = current_dir / "data"
output_dir = current_dir / "outputs"
data_dir.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

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

print("\nRaw text preview:")
print(repr(text[:120]))

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

block_size = 16
batch_size = 16
n_embd = 32


def get_batch(split):
    source = train_data if split == "train" else val_data

    ix = torch.randint(
        0,
        len(source) - block_size - 1,
        (batch_size,),
    )

    x = torch.stack([source[i : i + block_size] for i in ix])
    y = torch.stack([source[i + 1 : i + block_size + 1] for i in ix])

    return x.to(device), y.to(device)


class ContextPositionLanguageModel(torch.nn.Module):
    def __init__(self, vocab_size, block_size, n_embd):
        super().__init__()

        self.block_size = block_size

        # token embedding: 每个 token id 查出一个向量。
        self.token_embedding_table = torch.nn.Embedding(vocab_size, n_embd)

        # position embedding: 每个位置查出一个向量。
        self.position_embedding_table = torch.nn.Embedding(block_size, n_embd)

        # language model head: 把隐藏向量映射回 vocab logits。
        self.lm_head = torch.nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        # idx shape: [batch, seq_len]
        batch, seq_len = idx.shape

        if seq_len > self.block_size:
            raise ValueError(f"seq_len={seq_len} exceeds block_size={self.block_size}")

        token_emb = self.token_embedding_table(idx)
        # token_emb shape: [batch, seq_len, n_embd]

        positions = torch.arange(seq_len, device=idx.device)
        pos_emb = self.position_embedding_table(positions)
        # pos_emb shape: [seq_len, n_embd]

        x = token_emb + pos_emb
        # x shape: [batch, seq_len, n_embd]
        # pos_emb 会自动广播到 batch 维度。

        logits = self.lm_head(x)
        # logits shape: [batch, seq_len, vocab_size]

        loss = None

        if targets is not None:
            batch, seq_len, channels = logits.shape

            logits_flat = logits.reshape(batch * seq_len, channels)
            targets_flat = targets.reshape(batch * seq_len)

            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss, token_emb, pos_emb, x

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # 只保留最后 block_size 个 token 作为上下文窗口。
            idx_cond = idx[:, -self.block_size :]

            logits, _, _, _, _ = self(idx_cond)

            # 只取最后一个位置预测下一个 token。
            logits_last = logits[:, -1, :]
            probs = F.softmax(logits_last, dim=-1)

            # 为了兼容 MPS，把采样放到 CPU 上做。
            idx_next = torch.multinomial(probs.cpu(), num_samples=1).to(idx.device)

            idx = torch.cat((idx, idx_next), dim=1)

        return idx


model = ContextPositionLanguageModel(
    vocab_size=vocab_size,
    block_size=block_size,
    n_embd=n_embd,
).to(device)

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
            _, loss, _, _, _ = model(xb, yb)
            losses.append(loss.item())

        out[split] = sum(losses) / len(losses)

    model.train()
    return out


print("\nInitial batch check:")

xb, yb = get_batch("train")
logits, loss, token_emb, pos_emb, x = model(xb, yb)

print("x batch shape:", tuple(xb.shape))
print("y batch shape:", tuple(yb.shape))
print("token_emb shape:", tuple(token_emb.shape))
print("pos_emb shape:", tuple(pos_emb.shape))
print("combined x shape:", tuple(x.shape))
print("logits shape:", tuple(logits.shape))
print("initial loss:", loss.item())

print("\nFirst input sequence:")
print(repr(decode(xb[0].detach().cpu().tolist())))

print("\nFirst target sequence:")
print(repr(decode(yb[0].detach().cpu().tolist())))

print("\nTraining...")

for step in range(max_iters):
    xb, yb = get_batch("train")

    logits, loss, _, _, _ = model(xb, yb)

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

start_id = torch.tensor([[stoi["h"]]], dtype=torch.long, device=device)
generated = model.generate(start_id, max_new_tokens=160)[0].detach().cpu().tolist()
generated_text = decode(generated)

print(repr(generated_text))

output_path = output_dir / "torch_generated_text.txt"
output_path.write_text(generated_text, encoding="utf-8")

print("Saved:", output_path)

print("\nImportant reminder:")
print("This model has token embedding and position embedding.")
print("But it still has no self-attention.")
print("So each position still cannot really read previous tokens.")

print("\nDone.")
