from pathlib import Path

import torch
import torch.nn.functional as F


print("=== PyTorch Token / Embedding Basics ===")

current_dir = Path(__file__).resolve().parent
data_dir = current_dir / "data"
data_dir.mkdir(parents=True, exist_ok=True)

text_path = data_dir / "tiny_text.txt"
if not text_path.exists():
    text_path.write_text(
        "hello ai lab\n"
        "hello mlx\n"
        "hello tiny gpt\n",
        encoding="utf-8",
    )

text = text_path.read_text(encoding="utf-8")

print("\nRaw text:")
print(repr(text))

# 1. 建立字符级 vocab
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

print("\nVocab:")
print(chars)
print("vocab_size:", vocab_size)


# 2. encode / decode
def encode(s):
    return [stoi[ch] for ch in s]


def decode(ids):
    return "".join([itos[i] for i in ids])


encoded = encode(text)

print("\nEncoded text:")
print(encoded)

print("\nDecoded back:")
print(repr(decode(encoded)))

# 3. 构造 next-token prediction 数据
# 输入是当前 token 序列，标签是下一个 token 序列
block_size = 8
batch_size = 4

data = torch.tensor(encoded, dtype=torch.long)
starts = torch.tensor([0, 1, 2, 3])

x_batch = torch.stack([
    data[start:start + block_size]
    for start in starts
])

y_batch = torch.stack([
    data[start + 1:start + block_size + 1]
    for start in starts
])

print("\nNext-token batch:")
print("x_batch shape:", tuple(x_batch.shape))
print("y_batch shape:", tuple(y_batch.shape))

print("\nFirst input sequence:")
print(x_batch[0].tolist())
print(repr(decode(x_batch[0].tolist())))

print("\nFirst target sequence:")
print(y_batch[0].tolist())
print(repr(decode(y_batch[0].tolist())))

print("\nMeaning:")
print("x_batch 里每个位置是当前 token")
print("y_batch 里对应位置是下一个 token")

# 4. Embedding table
embed_dim = 4
embedding = torch.nn.Embedding(vocab_size, embed_dim)
token_embeddings = embedding(x_batch)

print("\nEmbedding:")
print("embedding.weight shape:", tuple(embedding.weight.shape))
print("token_embeddings shape:", tuple(token_embeddings.shape))
print("Meaning: [batch, seq_len, embed_dim]")

# 5. 一个最小语言模型头：把 embedding 映射成 vocab logits
lm_head = torch.nn.Linear(embed_dim, vocab_size)
logits = lm_head(token_embeddings)

print("\nLogits:")
print("logits shape:", tuple(logits.shape))
print("Meaning: [batch, seq_len, vocab_size]")

# 6. 随机模型的 next-token loss
loss = F.cross_entropy(
    logits.reshape(-1, vocab_size),
    y_batch.reshape(-1),
)

print("\nRandom next-token loss:", loss.item())

# 7. 看一个位置的预测
probs = torch.softmax(logits[0, 0], dim=0)
pred_id = torch.argmax(probs).item()

print("\nPrediction at first position:")
print("input token:", repr(decode([x_batch[0, 0].item()])))
print("target token:", repr(decode([y_batch[0, 0].item()])))
print("pred token:", repr(decode([pred_id])))
print("confidence:", probs[pred_id].item())

print("\nDone.")
