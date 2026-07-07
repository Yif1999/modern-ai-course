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


print("=== MLX KV Cache Intro ===")

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
        raise ValueError(f"Text contains tokens outside vocab: {missing}")
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

learning_rate = 3e-3
fallback_iters = 600
eval_interval = 200
eval_iters = 10

assert n_embd % num_heads == 0

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
        # [batch, seq_len, n_embd] -> [batch, num_heads, seq_len, head_size]
        batch, seq_len, _ = x.shape
        x = x.reshape(batch, seq_len, self.num_heads, self.head_size)
        return mx.transpose(x, (0, 2, 1, 3))

    def _merge_heads(self, x):
        # [batch, num_heads, seq_len, head_size] -> [batch, seq_len, n_embd]
        batch, num_heads, seq_len, head_size = x.shape
        x = mx.transpose(x, (0, 2, 1, 3))
        return x.reshape(batch, seq_len, num_heads * head_size)

    def __call__(self, x, return_attention=False):
        # 训练 / 普通推理路径：整段 context 一次性计算。
        _, seq_len, _ = x.shape

        q = self._split_heads(self.query(x))
        k = self._split_heads(self.key(x))
        v = self._split_heads(self.value(x))

        scores = q @ mx.transpose(k, (0, 1, 3, 2))
        scores = scores / math.sqrt(self.head_size)
        # [batch, num_heads, seq_len, seq_len]

        # causal mask：第 i 个位置只能看 <= i 的位置。
        mask_np = np.tril(np.ones((seq_len, seq_len), dtype=np.float32))
        mask = mx.array(mask_np).reshape(1, 1, seq_len, seq_len)
        masked_scores = mx.where(mask == 1, scores, mx.full(scores.shape, -1e9))

        weights = nn.softmax(masked_scores, axis=-1)
        out = weights @ v
        out = self._merge_heads(out)
        out = self.proj(out)

        if return_attention:
            return out, weights

        return out

    def forward_step(self, x, cache=None, return_debug=False):
        # KV Cache 路径：x 只包含当前这 1 个 token。
        # x: [batch, 1, n_embd]
        q = self._split_heads(self.query(x))
        k_new = self._split_heads(self.key(x))
        v_new = self._split_heads(self.value(x))
        # each: [batch, num_heads, 1, head_size]

        if cache is None:
            k_all = k_new
            v_all = v_new
        else:
            k_all = mx.concatenate([cache["key"], k_new], axis=2)
            v_all = mx.concatenate([cache["value"], v_new], axis=2)

        # 当前 token 的 query 只需要和“过去 + 当前”的所有 key 做 attention。
        scores = q @ mx.transpose(k_all, (0, 1, 3, 2))
        scores = scores / math.sqrt(self.head_size)
        # [batch, num_heads, 1, cache_seq_len]

        weights = nn.softmax(scores, axis=-1)
        out = weights @ v_all
        out = self._merge_heads(out)
        out = self.proj(out)

        new_cache = {
            "key": k_all,
            "value": v_all,
        }

        if return_debug:
            debug = {
                "q": q,
                "k_new": k_new,
                "v_new": v_new,
                "k_all": k_all,
                "v_all": v_all,
                "scores": scores,
                "weights": weights,
                "out": out,
            }
            return out, new_cache, debug

        return out, new_cache


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

    def forward_step(self, x, cache=None, return_debug=False):
        # 每个 block 内部仍然是：LayerNorm -> Attention -> Residual -> MLP -> Residual。
        if return_debug:
            attn_out, new_cache, attn_debug = self.attn.forward_step(
                self.ln1(x),
                cache=cache,
                return_debug=True,
            )
        else:
            attn_out, new_cache = self.attn.forward_step(
                self.ln1(x),
                cache=cache,
            )
            attn_debug = None

        x = x + attn_out
        ffwd_out = self.ffwd(self.ln2(x))
        x = x + ffwd_out

        if return_debug:
            block_debug = {
                "attn": attn_debug,
                "attn_out": attn_out,
                "ffwd_out": ffwd_out,
                "block_out": x,
            }
            return x, new_cache, block_debug

        return x, new_cache


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
        # idx: [batch, seq_len]
        _, seq_len = idx.shape

        if seq_len > self.block_size:
            raise ValueError(f"seq_len={seq_len} exceeds block_size={self.block_size}")

        token_emb = self.token_embedding_table[idx]
        pos_emb = self.position_embedding_table[mx.arange(seq_len)]
        x = token_emb + pos_emb

        for block in self.blocks:
            x = block(x)

        x = self.final_ln(x)
        return self.lm_head(x)

    def forward_step(self, idx_step, caches=None, position=0, return_debug=False):
        # idx_step: [batch, 1]
        if position >= self.block_size:
            raise ValueError(
                f"position={position} exceeds block_size={self.block_size}. "
                "This teaching script keeps generation within one block."
            )

        if caches is None:
            caches = [None for _ in range(self.num_layers)]

        token_emb = self.token_embedding_table[idx_step]
        pos_emb = self.position_embedding_table[mx.array([position])]
        x = token_emb + pos_emb

        new_caches = []
        layer_debug = []

        for layer_idx, block in enumerate(self.blocks):
            if return_debug:
                x, layer_cache, debug = block.forward_step(
                    x,
                    cache=caches[layer_idx],
                    return_debug=True,
                )
                layer_debug.append(debug)
            else:
                x, layer_cache = block.forward_step(x, cache=caches[layer_idx])
            new_caches.append(layer_cache)

        x = self.final_ln(x)
        logits = self.lm_head(x)

        if return_debug:
            debug = {
                "idx_step": idx_step,
                "position": position,
                "token_emb": token_emb,
                "pos_emb": pos_emb,
                "x_after_embeddings": token_emb + pos_emb,
                "layer_debug": layer_debug,
                "logits": logits,
            }
            return logits, new_caches, debug

        return logits, new_caches


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
    return nn.losses.cross_entropy(
        logits.reshape(batch * seq_len, channels),
        targets.reshape(batch * seq_len),
        reduction="mean",
    )


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


def logits_last_no_cache(ids):
    context = ids[-block_size:]
    idx = mx.array([context], dtype=mx.int32)
    logits = model(idx)
    last = logits[0, -1, :]
    mx.eval(last)
    return last


def greedy_id(logits_last):
    mx.eval(logits_last)
    return int(mx.argmax(logits_last))


def init_cache_from_prompt(prompt_ids, shape_lines):
    caches = None
    last_logits = None

    for position, token_id in enumerate(prompt_ids):
        idx_step = mx.array([[token_id]], dtype=mx.int32)
        last_logits, caches, debug = model.forward_step(
            idx_step,
            caches=caches,
            position=position,
            return_debug=True,
        )
        mx.eval(last_logits, *[layer["key"] for layer in caches])

        shape_lines.append(
            f"prefill position={position} token={decode([token_id])!r} "
            f"idx_step={idx_step.shape} logits={last_logits.shape}"
        )

        for layer_idx, layer in enumerate(debug["layer_debug"]):
            attn = layer["attn"]
            shape_lines.append(
                f"  layer={layer_idx} "
                f"query={attn['q'].shape} "
                f"new_key={attn['k_new'].shape} "
                f"cache_key={attn['k_all'].shape} "
                f"cache_value={attn['v_all'].shape} "
                f"scores={attn['scores'].shape} "
                f"weights={attn['weights'].shape}"
            )

    return caches, last_logits


def generate_no_cache(prompt, max_new_tokens):
    ids = encode(prompt)

    for _ in range(max_new_tokens):
        last = logits_last_no_cache(ids)
        ids.append(greedy_id(last))

    return ids


def generate_with_kv_cache(prompt, max_new_tokens, shape_lines=None):
    ids = encode(prompt)
    if len(ids) + max_new_tokens > block_size:
        raise ValueError("This teaching comparison keeps prompt + generation <= block_size")

    if shape_lines is None:
        shape_lines = []

    caches, last_logits = init_cache_from_prompt(ids, shape_lines)

    for step in range(max_new_tokens):
        no_cache_last = logits_last_no_cache(ids)
        mx.eval(no_cache_last, last_logits)

        error = float(mx.max(mx.abs(no_cache_last - last_logits)))
        next_id = greedy_id(last_logits)

        shape_lines.append(
            f"generate step={step} context_len={len(ids)} "
            f"next={decode([next_id])!r} max_logit_error={error:.8f}"
        )

        ids.append(next_id)

        # 为下一轮生成准备“新 token 作为当前 token”产生的 logits 和新 cache。
        if step < max_new_tokens - 1:
            idx_step = mx.array([[next_id]], dtype=mx.int32)
            last_logits, caches, debug = model.forward_step(
                idx_step,
                caches=caches,
                position=len(ids) - 1,
                return_debug=True,
            )
            mx.eval(last_logits, *[layer["key"] for layer in caches])

            for layer_idx, layer in enumerate(debug["layer_debug"]):
                attn = layer["attn"]
                shape_lines.append(
                    f"  after append layer={layer_idx} "
                    f"query={attn['q'].shape} "
                    f"cache_key={attn['k_all'].shape} "
                    f"cache_value={attn['v_all'].shape} "
                    f"scores={attn['scores'].shape} "
                    f"weights={attn['weights'].shape}"
                )

    return ids


def compare_logits_step_by_step(prompt, max_new_tokens):
    ids = encode(prompt)
    shape_lines = [
        "KV Cache Shape Log",
        "=" * 60,
        f"current_dir: {current_dir}",
        f"data_dir: {data_dir}",
        f"output_dir: {output_dir}",
        f"loaded_previous_weights: {loaded_weights}",
        f"prompt: {prompt!r}",
        f"max_new_tokens: {max_new_tokens}",
        "",
        "Shape convention:",
        "idx_step: [batch, 1]",
        "query: [batch, num_heads, 1, head_size]",
        "cache_key/cache_value: [batch, num_heads, cache_seq_len, head_size]",
        "scores/weights: [batch, num_heads, 1, cache_seq_len]",
        "",
    ]

    caches, last_logits = init_cache_from_prompt(ids, shape_lines)

    errors = []
    generated_ids = ids[:]

    for step in range(max_new_tokens):
        no_cache_last = logits_last_no_cache(generated_ids)
        mx.eval(no_cache_last, last_logits)

        error = float(mx.max(mx.abs(no_cache_last - last_logits)))
        errors.append(error)

        next_id = greedy_id(last_logits)
        shape_lines.append(
            f"compare step={step} context_len={len(generated_ids)} "
            f"next={decode([next_id])!r} max_logit_error={error:.8f}"
        )

        generated_ids.append(next_id)

        if step < max_new_tokens - 1:
            idx_step = mx.array([[next_id]], dtype=mx.int32)
            last_logits, caches, debug = model.forward_step(
                idx_step,
                caches=caches,
                position=len(generated_ids) - 1,
                return_debug=True,
            )
            mx.eval(last_logits, *[layer["key"] for layer in caches])

            for layer_idx, layer in enumerate(debug["layer_debug"]):
                attn = layer["attn"]
                shape_lines.append(
                    f"  layer={layer_idx} "
                    f"query={attn['q'].shape} "
                    f"new_key={attn['k_new'].shape} "
                    f"cache_key={attn['k_all'].shape} "
                    f"cache_value={attn['v_all'].shape} "
                    f"scores={attn['scores'].shape} "
                    f"weights={attn['weights'].shape}"
                )

    return generated_ids, errors, shape_lines


prompt = "hello "
max_new_tokens = 20

print("\nRunning no-cache generation...")
no_cache_ids = generate_no_cache(prompt, max_new_tokens)
no_cache_text = decode(no_cache_ids)
print(repr(no_cache_text))

print("\nRunning KV-cache generation and comparing logits...")
cache_ids, logit_errors, shape_lines = compare_logits_step_by_step(
    prompt,
    max_new_tokens,
)
cache_text = decode(cache_ids)
print(repr(cache_text))

max_error = max(logit_errors) if logit_errors else 0.0
texts_match = no_cache_text == cache_text

print("\nGeneration comparison:")
print("texts_match:", texts_match)
print("max logits error:", max_error)

shape_log_path = output_dir / "kv_cache_shape_log.txt"
shape_log_path.write_text("\n".join(shape_lines), encoding="utf-8")
print("Saved:", shape_log_path)

comparison_lines = [
    "KV Cache Generation Comparison",
    "=" * 60,
    f"current_dir: {current_dir}",
    f"data_dir: {data_dir}",
    f"output_dir: {output_dir}",
    f"loaded_previous_weights: {loaded_weights}",
    f"prompt: {prompt!r}",
    f"max_new_tokens: {max_new_tokens}",
    f"texts_match: {texts_match}",
    f"max_logits_error: {max_error:.10f}",
    "",
    "No cache:",
    repr(no_cache_text),
    "",
    "With KV cache:",
    repr(cache_text),
    "",
    "Per-step max logit errors:",
]

for step, error in enumerate(logit_errors):
    comparison_lines.append(f"step={step:02d} max_logit_error={error:.10f}")

comparison_path = output_dir / "kv_cache_generation_comparison.txt"
comparison_path.write_text("\n".join(comparison_lines), encoding="utf-8")
print("Saved:", comparison_path)


print("\nTiming no-cache vs KV-cache generation...")

timing_prompt = "hello "
timing_new_tokens = 20
timing_repeats = 5

no_cache_times = []
cache_times = []

for _ in range(timing_repeats):
    start = time.perf_counter()
    ids = generate_no_cache(timing_prompt, timing_new_tokens)
    mx.eval(logits_last_no_cache(ids))
    no_cache_times.append(time.perf_counter() - start)

for _ in range(timing_repeats):
    start = time.perf_counter()
    ids = generate_with_kv_cache(timing_prompt, timing_new_tokens)
    mx.eval(logits_last_no_cache(ids))
    cache_times.append(time.perf_counter() - start)

timing_lines = [
    "KV Cache Timing",
    "=" * 60,
    f"prompt: {timing_prompt!r}",
    f"new_tokens: {timing_new_tokens}",
    f"repeats: {timing_repeats}",
    "",
    "Important note:",
    "这个脚本是教学版实现，小模型、短序列、Python 循环和 MLX lazy eval 都会影响 timing。",
    "所以这里的时间只能辅助观察，不能当成严肃 benchmark。",
    "",
    f"no_cache_times_sec: {[round(t, 6) for t in no_cache_times]}",
    f"kv_cache_times_sec: {[round(t, 6) for t in cache_times]}",
    f"no_cache_avg_sec: {sum(no_cache_times) / len(no_cache_times):.6f}",
    f"kv_cache_avg_sec: {sum(cache_times) / len(cache_times):.6f}",
]

timing_path = output_dir / "kv_cache_timing.txt"
timing_path.write_text("\n".join(timing_lines), encoding="utf-8")
print("Saved:", timing_path)


print("\nSaving KV Cache concept diagram...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax in axes:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")

axes[0].set_title("No cache: recompute the whole context", fontsize=13)
for step, length in enumerate([4, 5, 6, 7], start=1):
    y = 7.0 - step * 1.25
    axes[0].add_patch(
        plt.Rectangle((0.8, y), 4.6, 0.65, fill=False, lw=2)
    )
    axes[0].text(
        1.05,
        y + 0.23,
        f"step {step}: full forward over {length} tokens",
        fontsize=10,
    )
    axes[0].annotate(
        "",
        xy=(8.0, y + 0.32),
        xytext=(5.5, y + 0.32),
        arrowprops={"arrowstyle": "->", "lw": 1.8},
    )
    axes[0].text(8.15, y + 0.22, "last logits", fontsize=10)

axes[0].text(
    0.8,
    0.8,
    "Problem: old token K/V are recomputed at every step.",
    fontsize=11,
    color="#b22222",
)

axes[1].set_title("With KV cache: compute new token, reuse old K/V", fontsize=13)


def add_box(ax, xy, text, width=4.8, height=0.8, color="#111111"):
    x, y = xy
    ax.add_patch(plt.Rectangle((x, y), width, height, fill=False, lw=2, color=color))
    ax.text(x + 0.18, y + height / 2 - 0.08, text, fontsize=10, color=color)


add_box(axes[1], (1.0, 6.4), "past K/V cache\n[layer, head, seq, dim]", width=4.9, height=0.9)
add_box(axes[1], (1.0, 4.9), "current token -> new Q, K, V", width=4.9, height=0.8)
add_box(axes[1], (1.0, 3.4), "append new K/V to cache", width=4.9, height=0.8)
add_box(axes[1], (1.0, 2.0), "new Q attends to cached K", width=4.9, height=0.8)
add_box(axes[1], (1.0, 0.7), "weights @ cached V -> logits", width=4.9, height=0.8)

for y_start, y_end in [(6.4, 5.7), (4.9, 4.2), (3.4, 2.8), (2.0, 1.5)]:
    axes[1].annotate(
        "",
        xy=(3.45, y_end),
        xytext=(3.45, y_start),
        arrowprops={"arrowstyle": "->", "lw": 1.8},
    )

axes[1].annotate(
    "reuse",
    xy=(5.95, 2.38),
    xytext=(7.05, 6.75),
    arrowprops={"arrowstyle": "->", "lw": 1.8, "connectionstyle": "arc3,rad=0.25"},
    fontsize=10,
    color="#2c6e49",
)
axes[1].annotate(
    "grow cache",
    xy=(5.95, 3.8),
    xytext=(7.05, 3.95),
    arrowprops={"arrowstyle": "->", "lw": 1.8},
    fontsize=10,
    color="#2c6e49",
)
axes[1].text(
    0.7,
    0.15,
    "Core idea: cache old K/V; compute the current Query on demand.",
    fontsize=11,
    color="#2c6e49",
)

fig.suptitle("KV Cache in Autoregressive Generation", fontsize=15)
fig.tight_layout()
diagram_path = output_dir / "kv_cache_concept_diagram.png"
fig.savefig(diagram_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved:", diagram_path)

print("\nDone.")
