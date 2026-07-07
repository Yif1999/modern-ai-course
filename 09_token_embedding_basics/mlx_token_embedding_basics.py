from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


print("=== MLX Token / Embedding Basics ===")

mx.random.seed(42)

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
    return "".join([itos[int(i)] for i in ids])


encoded = encode(text)

print("\nEncoded text:")
print(encoded)

print("\nDecoded back:")
print(repr(decode(encoded)))

# 3. 构造 next-token prediction 数据
block_size = 8
batch_size = 4
starts = [0, 1, 2, 3]

x_batch_list = [
    encoded[start:start + block_size]
    for start in starts
]

y_batch_list = [
    encoded[start + 1:start + block_size + 1]
    for start in starts
]

x_batch = mx.array(x_batch_list, dtype=mx.int32)
y_batch = mx.array(y_batch_list, dtype=mx.int32)
mx.eval(x_batch, y_batch)

print("\nNext-token batch:")
print("x_batch shape:", x_batch.shape)
print("y_batch shape:", y_batch.shape)

print("\nFirst input sequence:")
print(x_batch_list[0])
print(repr(decode(x_batch_list[0])))

print("\nFirst target sequence:")
print(y_batch_list[0])
print(repr(decode(y_batch_list[0])))

print("\nMeaning:")
print("x_batch 里每个位置是当前 token")
print("y_batch 里对应位置是下一个 token")

# 4. 手写 embedding table
embed_dim = 4
embedding_table = mx.random.normal((vocab_size, embed_dim)) * 0.01
token_embeddings = embedding_table[x_batch]
mx.eval(embedding_table, token_embeddings)

print("\nEmbedding:")
print("embedding_table shape:", embedding_table.shape)
print("token_embeddings shape:", token_embeddings.shape)
print("Meaning: [batch, seq_len, embed_dim]")

# 5. 一个最小语言模型头：把 embedding 映射成 vocab logits
lm_head_w = mx.random.normal((embed_dim, vocab_size)) * 0.01
lm_head_b = mx.zeros((vocab_size,))
logits = token_embeddings @ lm_head_w + lm_head_b
mx.eval(logits)

print("\nLogits:")
print("logits shape:", logits.shape)
print("Meaning: [batch, seq_len, vocab_size]")

# 6. 随机模型的 next-token loss
loss = nn.losses.cross_entropy(
    logits.reshape(-1, vocab_size),
    y_batch.reshape(-1),
    reduction="mean",
)
mx.eval(loss)

print("\nRandom next-token loss:", float(loss))

# 7. 看一个位置的预测
probs = nn.softmax(logits[0, 0], axis=0)
pred_id = mx.argmax(probs)
mx.eval(probs, pred_id)

pred_id_int = int(pred_id)
target_id_int = int(y_batch_list[0][0])
input_id_int = int(x_batch_list[0][0])

print("\nPrediction at first position:")
print("input token:", repr(decode([input_id_int])))
print("target token:", repr(decode([target_id_int])))
print("pred token:", repr(decode([pred_id_int])))
print("confidence:", float(probs[pred_id_int]))

print("\nDone.")
