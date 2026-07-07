from pathlib import Path
import json
import math
import shutil
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim


print("=== MLX Tiny GPT Sampling Strategies ===")

mx.random.seed(42)
np.random.seed(42)

current_dir = Path(__file__).resolve().parent
data_dir = current_dir / "data"
output_dir = current_dir / "outputs"
data_dir.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)

previous_dir = current_dir.parent / "15_tiny_gpt_training"
previous_data_path = previous_dir / "data" / "tiny_text.txt"
previous_weights_path = previous_dir / "outputs" / "tiny_gpt_model.safetensors"
previous_config_path = previous_dir / "outputs" / "tiny_gpt_config.json"

text_path = data_dir / "tiny_text.txt"
if not text_path.exists():
    if previous_data_path.exists():
        shutil.copyfile(previous_data_path, text_path)
    else:
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
print("Training text:", text_path)
print("Text length:", len(text))

loaded_config = None
if previous_config_path.exists():
    loaded_config = json.loads(previous_config_path.read_text(encoding="utf-8"))
    chars = loaded_config["chars"]
else:
    chars = sorted(list(set(text)))

vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}


def encode(s):
    missing = sorted(set(s) - set(stoi))
    if missing:
        raise ValueError(f"Prompt contains tokens outside vocab: {missing}")
    return [stoi[ch] for ch in s]


def decode(ids):
    return "".join([itos[int(i)] for i in ids])


encoded = np.array(encode(text), dtype=np.int32)

print("\nVocab:")
print(chars)
print("vocab_size:", vocab_size)

n = int(0.9 * len(encoded))
train_data = encoded[:n]
val_data = encoded[n:]

block_size = int(loaded_config["block_size"]) if loaded_config else 32
batch_size = int(loaded_config["batch_size"]) if loaded_config else 32
n_embd = int(loaded_config["n_embd"]) if loaded_config else 64
num_heads = int(loaded_config["num_heads"]) if loaded_config else 4
num_layers = int(loaded_config["num_layers"]) if loaded_config else 2
head_size = n_embd // num_heads

# fallback 快速训练用；如果成功加载第 15 课权重，就不会进入训练。
learning_rate = 3e-3
fallback_iters = 800
eval_interval = 200
eval_iters = 10

print("\nModel config:")
print("block_size:", block_size)
print("batch_size:", batch_size)
print("n_embd:", n_embd)
print("num_heads:", num_heads)
print("num_layers:", num_layers)
print("head_size:", head_size)


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
        batch, seq_len, _ = x.shape
        x = x.reshape(batch, seq_len, self.num_heads, self.head_size)
        x = mx.transpose(x, (0, 2, 1, 3))
        return x

    def _merge_heads(self, x):
        batch, num_heads, seq_len, head_size = x.shape
        x = mx.transpose(x, (0, 2, 1, 3))
        x = x.reshape(batch, seq_len, num_heads * head_size)
        return x

    def __call__(self, x):
        _, seq_len, _ = x.shape

        q = self._split_heads(self.query(x))
        k = self._split_heads(self.key(x))
        v = self._split_heads(self.value(x))

        scores = q @ mx.transpose(k, (0, 1, 3, 2))
        scores = scores / math.sqrt(self.head_size)

        mask_np = np.tril(np.ones((seq_len, seq_len), dtype=np.float32))
        mask = mx.array(mask_np).reshape(1, 1, seq_len, seq_len)
        masked_scores = mx.where(mask == 1, scores, mx.full(scores.shape, -1e9))

        weights = nn.softmax(masked_scores, axis=-1)
        out = weights @ v
        out = self._merge_heads(out)
        out = self.proj(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.linear1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear2 = nn.Linear(4 * n_embd, n_embd)

    def __call__(self, x):
        x = self.linear1(x)
        x = nn.gelu(x)
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, n_embd, num_heads):
        super().__init__()
        self.attn = MultiHeadCausalSelfAttention(n_embd, num_heads)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def __call__(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
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

    def __call__(self, idx):
        _, seq_len = idx.shape

        if seq_len > self.block_size:
            raise ValueError(f"seq_len={seq_len} exceeds block_size={self.block_size}")

        token_emb = self.token_embedding_table[idx]
        pos_emb = self.position_embedding_table[mx.arange(seq_len)]
        x = token_emb + pos_emb

        for block in self.blocks:
            x = block(x)

        x = self.final_ln(x)
        logits = self.lm_head(x)
        return logits


model = TinyGPT(
    vocab_size=vocab_size,
    block_size=block_size,
    n_embd=n_embd,
    num_heads=num_heads,
    num_layers=num_layers,
)


def loss_fn(model, idx, targets):
    logits = model(idx)
    batch, seq_len, channels = logits.shape
    loss = nn.losses.cross_entropy(
        logits.reshape(batch * seq_len, channels),
        targets.reshape(batch * seq_len),
        reduction="mean",
    )
    return loss


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


def try_load_previous_weights():
    if not previous_weights_path.exists():
        return False, f"missing weights: {previous_weights_path}"

    try:
        # MLX nn.Module 可以直接从 .safetensors / .npz 加载权重。
        model.load_weights(str(previous_weights_path), strict=True)
        mx.eval(model.parameters())
        return True, str(previous_weights_path)
    except Exception as exc:
        return False, repr(exc)


loaded_weights, load_message = try_load_previous_weights()

if loaded_weights:
    print("\nLoaded previous model weights:")
    print(load_message)
else:
    print("\nCould not load previous model weights.")
    print("Reason:", load_message)
    print("Fallback: training a small Tiny GPT quickly in this lesson...")

    optimizer = optim.AdamW(learning_rate=learning_rate)
    value_and_grad_fn = nn.value_and_grad(model, loss_fn)
    start_time = time.perf_counter()

    for step in range(fallback_iters):
        xb, yb = get_batch("train")
        loss, grads = value_and_grad_fn(model, xb, yb)
        optimizer.update(model, grads)
        mx.eval(loss, model.parameters(), optimizer.state)

        if step % eval_interval == 0 or step == fallback_iters - 1:
            losses = estimate_loss()
            elapsed = time.perf_counter() - start_time
            print(
                f"step={step:04d} "
                f"train_loss={losses['train']:.4f} "
                f"val_loss={losses['val']:.4f} "
                f"elapsed={elapsed:.1f}s"
            )


xb, yb = get_batch("train")
loss = loss_fn(model, xb, yb)
logits = model(xb)
mx.eval(loss, logits)

print("\nBatch / logits check:")
print("x batch shape:", xb.shape)
print("y batch shape:", yb.shape)
print("logits shape:", logits.shape)
print("current train batch loss:", float(loss))


def softmax_np(logits_np, temperature=1.0):
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    scaled = logits_np / temperature
    scaled = scaled - np.max(scaled)
    exp = np.exp(scaled)
    return exp / np.sum(exp)


def apply_top_k(probs, top_k):
    if top_k is None or top_k >= len(probs):
        return probs.copy(), np.arange(len(probs))

    keep = np.argpartition(probs, -top_k)[-top_k:]
    filtered = np.zeros_like(probs)
    filtered[keep] = probs[keep]
    filtered = filtered / filtered.sum()
    return filtered, np.sort(keep)


def apply_top_p(probs, top_p):
    if top_p is None or top_p >= 1.0:
        return probs.copy(), np.arange(len(probs))

    sorted_idx = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_idx]
    cumulative = np.cumsum(sorted_probs)
    cutoff = int(np.searchsorted(cumulative, top_p, side="left"))
    keep_sorted = sorted_idx[: cutoff + 1]

    filtered = np.zeros_like(probs)
    filtered[keep_sorted] = probs[keep_sorted]
    filtered = filtered / filtered.sum()
    return filtered, np.sort(keep_sorted)


def next_logits_np(ids):
    context = ids[-block_size:]
    idx = mx.array([context], dtype=mx.int32)
    logits = model(idx)
    logits_last = logits[0, -1, :]
    mx.eval(logits_last)
    return np.array(logits_last, dtype=np.float64)


def choose_next_id(logits_np, strategy, rng, temperature=1.0, top_k=None, top_p=None):
    if strategy == "greedy":
        probs = softmax_np(logits_np, temperature=1.0)
        return int(np.argmax(probs)), probs, np.arange(len(probs))

    probs = softmax_np(logits_np, temperature=temperature)
    kept = np.arange(len(probs))

    if top_k is not None:
        probs, kept = apply_top_k(probs, top_k)

    if top_p is not None:
        probs, kept = apply_top_p(probs, top_p)

    next_id = int(rng.choice(len(probs), p=probs))
    return next_id, probs, kept


def generate_text(prompt, max_new_tokens, strategy, seed, temperature=1.0, top_k=None, top_p=None):
    ids = encode(prompt)
    rng = np.random.default_rng(seed)

    for _ in range(max_new_tokens):
        logits_np = next_logits_np(ids)
        next_id, _, _ = choose_next_id(
            logits_np=logits_np,
            strategy=strategy,
            rng=rng,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        ids.append(next_id)

    return decode(ids)


prompt = "hello we learn tiny gpt language model"
max_new_tokens = 160

experiments = [
    {
        "name": "greedy",
        "strategy": "greedy",
        "temperature": 1.0,
        "top_k": None,
        "top_p": None,
        "seed": 100,
    },
    {
        "name": "temperature_0.5",
        "strategy": "sample",
        "temperature": 0.5,
        "top_k": None,
        "top_p": None,
        "seed": 101,
    },
    {
        "name": "temperature_1.0",
        "strategy": "sample",
        "temperature": 1.0,
        "top_k": None,
        "top_p": None,
        "seed": 102,
    },
    {
        "name": "temperature_1.5",
        "strategy": "sample",
        "temperature": 1.5,
        "top_k": None,
        "top_p": None,
        "seed": 103,
    },
    {
        "name": "top_k_5",
        "strategy": "sample",
        "temperature": 1.0,
        "top_k": 5,
        "top_p": None,
        "seed": 104,
    },
    {
        "name": "top_k_10",
        "strategy": "sample",
        "temperature": 1.0,
        "top_k": 10,
        "top_p": None,
        "seed": 105,
    },
    {
        "name": "top_p_0.8",
        "strategy": "sample",
        "temperature": 1.0,
        "top_k": None,
        "top_p": 0.8,
        "seed": 106,
    },
    {
        "name": "top_p_0.95",
        "strategy": "sample",
        "temperature": 1.0,
        "top_k": None,
        "top_p": 0.95,
        "seed": 107,
    },
]

print("\nGenerating sampling comparison...")

comparison_lines = [
    "Tiny GPT Sampling Comparison",
    "=" * 40,
    f"current_dir: {current_dir}",
    f"data_dir: {data_dir}",
    f"output_dir: {output_dir}",
    f"loaded_previous_weights: {loaded_weights}",
    f"prompt: {prompt!r}",
    f"max_new_tokens: {max_new_tokens}",
    "",
]

for exp in experiments:
    generated = generate_text(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        strategy=exp["strategy"],
        seed=exp["seed"],
        temperature=exp["temperature"],
        top_k=exp["top_k"],
        top_p=exp["top_p"],
    )

    print(f"\n--- {exp['name']} ---")
    print(repr(generated))

    comparison_lines.extend(
        [
            f"--- {exp['name']} ---",
            f"strategy={exp['strategy']} temperature={exp['temperature']} "
            f"top_k={exp['top_k']} top_p={exp['top_p']} seed={exp['seed']}",
            generated,
            "",
        ]
    )


print("\nAnalyzing one next-token distribution...")

analysis_prompt = "we learn "
analysis_ids = encode(analysis_prompt)
analysis_logits = next_logits_np(analysis_ids)
base_probs = softmax_np(analysis_logits, temperature=1.0)
token_order = np.argsort(base_probs)[::-1]

comparison_lines.extend(
    [
        "One-step distribution after prompt",
        "=" * 40,
        f"analysis_prompt: {analysis_prompt!r}",
        "Top probabilities at temperature=1.0:",
    ]
)

print(f"\nTop tokens after analysis prompt {analysis_prompt!r} at temperature=1.0:")
for rank, token_id in enumerate(token_order[:10], start=1):
    token = repr(decode([token_id]))
    prob = base_probs[token_id]
    line = f"{rank:02d}. token={token:>4} prob={prob:.4f}"
    print(line)
    comparison_lines.append(line)

top_k_probs, top_k_kept = apply_top_k(base_probs, 5)
top_p_probs, top_p_kept = apply_top_p(base_probs, 0.8)

comparison_lines.extend(
    [
        "",
        "Top-k / Top-p filtering table:",
        "token | prob | top_k_5 | top_p_0.8",
        "-" * 36,
    ]
)

print("\nTop-k / Top-p filtering table:")
print("token | prob | top_k_5 | top_p_0.8")
print("-" * 36)

for token_id in token_order[:14]:
    token = repr(decode([token_id]))
    in_top_k = "keep" if token_id in set(top_k_kept.tolist()) else "drop"
    in_top_p = "keep" if token_id in set(top_p_kept.tolist()) else "drop"
    line = f"{token:>5} | {base_probs[token_id]:.4f} | {in_top_k:^7} | {in_top_p:^8}"
    print(line)
    comparison_lines.append(line)

comparison_path = output_dir / "sampling_comparison.txt"
comparison_path.write_text("\n".join(comparison_lines), encoding="utf-8")
print("\nSaved:", comparison_path)


print("\nSaving temperature probability comparison plot...")

temperatures = [0.5, 1.0, 1.5]
top_display = token_order[:12]
x_labels = [decode([token_id]).replace("\n", "\\n") for token_id in top_display]
x = np.arange(len(top_display))

plt.figure(figsize=(10, 5))
bar_width = 0.25

for i, temp in enumerate(temperatures):
    probs = softmax_np(analysis_logits, temperature=temp)
    values = [probs[token_id] for token_id in top_display]
    plt.bar(x + (i - 1) * bar_width, values, width=bar_width, label=f"T={temp}")

plt.xticks(x, x_labels)
plt.ylabel("probability")
plt.title("Temperature changes next-token probability distribution")
plt.legend()
plt.grid(axis="y", alpha=0.3)
temperature_plot_path = output_dir / "temperature_probability_comparison.png"
plt.savefig(temperature_plot_path, dpi=150, bbox_inches="tight")
plt.close()
print("Saved:", temperature_plot_path)


print("\nSaving top-k / top-p filtering visualization...")

display_tokens = token_order[:14]
display_probs = [base_probs[token_id] for token_id in display_tokens]
display_labels = [decode([token_id]).replace("\n", "\\n") for token_id in display_tokens]

colors = []
top_k_set = set(top_k_kept.tolist())
top_p_set = set(top_p_kept.tolist())
for token_id in display_tokens:
    if token_id in top_k_set and token_id in top_p_set:
        colors.append("#2ca02c")
    elif token_id in top_k_set:
        colors.append("#1f77b4")
    elif token_id in top_p_set:
        colors.append("#ff7f0e")
    else:
        colors.append("#d0d0d0")

plt.figure(figsize=(10, 5))
plt.bar(np.arange(len(display_tokens)), display_probs, color=colors)
plt.xticks(np.arange(len(display_tokens)), display_labels)
plt.ylabel("probability at T=1.0")
plt.title("Kept tokens: green=top-k and top-p, blue=top-k only, orange=top-p only")
plt.grid(axis="y", alpha=0.3)
filtering_plot_path = output_dir / "topk_topp_filtering.png"
plt.savefig(filtering_plot_path, dpi=150, bbox_inches="tight")
plt.close()
print("Saved:", filtering_plot_path)

print("\nKey explanations:")
print("temperature 越低，概率会越集中，模型更保守、更重复。")
print("temperature 越高，概率会越分散，模型更随机，也更容易出错。")
print("greedy 每步都取最大概率 token，所以稳定，但容易陷入重复模式。")
print("top-k 只保留概率最高的 k 个 token，直接过滤低概率 token。")
print("top-p 保留累计概率达到 p 的最小 token 集合，候选数量会随分布形状变化。")

print("\nDone.")
