from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from tokenizers import Tokenizer


CURRENT_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = CURRENT_DIR / "data" / "processed"
OUTPUT_DIR = CURRENT_DIR / "outputs"
RUNS_DIR = OUTPUT_DIR / "runs"
TOKENIZER_PATH = OUTPUT_DIR / "tokenizer" / "chinese_bpe_tokenizer.json"

PROMPTS = ["人工智能", "大语言模型", "今天我们学习", "本地模型", "中文预训练"]


@dataclass
class RunConfig:
    run_name: str
    dataset_scale: str
    block_size: int
    batch_size: int
    n_embd: int
    num_heads: int
    num_layers: int
    max_iters: int = 1000
    eval_interval: int = 200
    eval_iters: int = 8
    learning_rate: float = 2e-3
    sample_tokens: int = 160
    temperature: float = 0.8
    top_k: int = 20
    seed: int = 42


DEFAULT_RUNS = [
    RunConfig(
        run_name="run_a_baseline",
        dataset_scale="small",
        block_size=64,
        batch_size=16,
        n_embd=64,
        num_heads=4,
        num_layers=2,
    ),
    RunConfig(
        run_name="run_b_more_data",
        dataset_scale="medium",
        block_size=64,
        batch_size=16,
        n_embd=64,
        num_heads=4,
        num_layers=2,
    ),
    RunConfig(
        run_name="run_c_larger_model",
        dataset_scale="small",
        block_size=64,
        batch_size=16,
        n_embd=128,
        num_heads=4,
        num_layers=4,
    ),
    RunConfig(
        run_name="run_d_longer_context",
        dataset_scale="small",
        block_size=128,
        batch_size=16,
        n_embd=64,
        num_heads=4,
        num_layers=2,
    ),
]


class MultiHeadCausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, num_heads: int):
        super().__init__()
        if n_embd % num_heads != 0:
            raise ValueError("n_embd must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_size = n_embd // num_heads
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)

    def split_heads(self, x):
        batch, seq_len, n_embd = x.shape
        x = x.reshape(batch, seq_len, self.num_heads, self.head_size)
        return mx.transpose(x, (0, 2, 1, 3))

    def merge_heads(self, x):
        batch, heads, seq_len, head_size = x.shape
        x = mx.transpose(x, (0, 2, 1, 3))
        return x.reshape(batch, seq_len, heads * head_size)

    def __call__(self, x):
        _, seq_len, _ = x.shape
        q = self.split_heads(self.query(x))
        k = self.split_heads(self.key(x))
        v = self.split_heads(self.value(x))
        scores = q @ mx.transpose(k, (0, 1, 3, 2))
        scores = scores / math.sqrt(self.head_size)
        mask = mx.array(np.tril(np.ones((seq_len, seq_len), dtype=np.float32))).reshape(1, 1, seq_len, seq_len)
        scores = mx.where(mask == 1, scores, mx.full(scores.shape, -1e9))
        weights = nn.softmax(scores, axis=-1)
        out = weights @ v
        return self.proj(self.merge_heads(out))


class FeedForward(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.linear1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear2 = nn.Linear(4 * n_embd, n_embd)

    def __call__(self, x):
        return self.linear2(nn.gelu(self.linear1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, n_embd: int, num_heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadCausalSelfAttention(n_embd, num_heads)
        self.ffwd = FeedForward(n_embd)

    def __call__(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int, block_size: int, n_embd: int, num_heads: int, num_layers: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.token_embedding_table = mx.random.normal((vocab_size, n_embd)) * 0.02
        self.position_embedding_table = mx.random.normal((block_size, n_embd)) * 0.02
        self.blocks = [TransformerBlock(n_embd, num_heads) for _ in range(num_layers)]
        self.final_ln = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def __call__(self, idx):
        _, seq_len = idx.shape
        token_emb = self.token_embedding_table[idx]
        pos_emb = self.position_embedding_table[mx.arange(seq_len)]
        x = token_emb + pos_emb
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.final_ln(x))


def language_model_loss(model: TinyGPT, idx, targets):
    logits = model(idx)
    batch, seq_len, vocab_size = logits.shape
    return nn.losses.cross_entropy(
        logits.reshape(batch * seq_len, vocab_size),
        targets.reshape(batch * seq_len),
        reduction="mean",
    )


def sample_next_id(logits, temperature: float, top_k: int) -> int:
    logits_np = np.array(logits, dtype=np.float64)
    logits_np = logits_np / max(temperature, 1e-6)
    logits_np = logits_np - np.max(logits_np)
    probs = np.exp(logits_np)
    probs = probs / probs.sum()
    if 0 < top_k < len(probs):
        keep = np.argpartition(probs, -top_k)[-top_k:]
        filtered = np.zeros_like(probs)
        filtered[keep] = probs[keep]
        probs = filtered / filtered.sum()
    return int(np.random.choice(len(probs), p=probs))


def generate_ids(model: TinyGPT, start_ids, max_new_tokens: int, temperature: float, top_k: int) -> list[int]:
    ids = [int(i) for i in start_ids]
    for _ in range(max_new_tokens):
        context = ids[-model.block_size :]
        idx = mx.array([context], dtype=mx.int32)
        logits = model(idx)
        last_logits = logits[0, -1, :]
        mx.eval(last_logits)
        ids.append(sample_next_id(last_logits, temperature=temperature, top_k=top_k))
    return ids


class TokenDataset:
    def __init__(self, config: RunConfig):
        scale_dir = PROCESSED_DIR / config.dataset_scale
        metadata_path = scale_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"missing metadata: {metadata_path}")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not self.metadata.get("generated"):
            raise ValueError(f"dataset scale {config.dataset_scale} was not generated: {self.metadata}")
        self.train_tokens = np.load(scale_dir / "train_tokens.npy").astype(np.int32)
        self.val_tokens = np.load(scale_dir / "val_tokens.npy").astype(np.int32)
        self.config = config

    def get_batch(self, split: str):
        source = self.train_tokens if split == "train" else self.val_tokens
        max_start = len(source) - self.config.block_size - 1
        if max_start <= 0:
            raise ValueError(f"{split} split is too small for block_size={self.config.block_size}")
        starts = np.random.randint(0, max_start, size=(self.config.batch_size,))
        x = np.stack([source[i : i + self.config.block_size] for i in starts]).astype(np.int32)
        y = np.stack([source[i + 1 : i + self.config.block_size + 1] for i in starts]).astype(np.int32)
        return mx.array(x), mx.array(y)


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def append_log(path: Path, message: str) -> None:
    print(message)
    with path.open("a", encoding="utf-8") as f:
        f.write(message + "\n")


def decode_ids(tokenizer: Tokenizer, ids) -> str:
    return tokenizer.decode([int(i) for i in ids], skip_special_tokens=True)


def generate_samples(model: TinyGPT, tokenizer: Tokenizer, config: RunConfig) -> dict[str, str]:
    bos_id = tokenizer.token_to_id("<bos>")
    samples = {}
    for prompt in PROMPTS:
        start_ids = [bos_id] + tokenizer.encode(prompt).ids
        ids = generate_ids(
            model,
            start_ids,
            max_new_tokens=config.sample_tokens,
            temperature=config.temperature,
            top_k=config.top_k,
        )
        samples[prompt] = decode_ids(tokenizer, ids)
    return samples


def write_generated_text(path: Path, samples: dict[str, str]) -> None:
    lines = []
    for prompt, text in samples.items():
        lines.append(f"=== prompt: {prompt} ===")
        lines.append(text)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_run_loss(history: list[dict], path: Path, title: str) -> None:
    steps = [row["step"] for row in history]
    train = [row["train_loss"] for row in history]
    val = [row["val_loss"] for row in history]
    plt.figure(figsize=(8, 5))
    plt.plot(steps, train, marker="o", label="train loss")
    plt.plot(steps, val, marker="o", label="val loss")
    plt.xlabel("step")
    plt.ylabel("cross entropy loss")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def run_experiment(config: RunConfig) -> dict:
    run_dir = RUNS_DIR / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "training_log.txt"
    jsonl_path = run_dir / "training_log.jsonl"
    log_path.write_text("", encoding="utf-8")
    jsonl_path.write_text("", encoding="utf-8")

    mx.random.seed(config.seed)
    np.random.seed(config.seed)

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    dataset = TokenDataset(config)
    vocab_size = int(dataset.metadata["vocab_size"])

    run_config = asdict(config)
    run_config.update(
        {
            "vocab_size": vocab_size,
            "train_tokens": int(dataset.metadata["train_tokens"]),
            "val_tokens": int(dataset.metadata["val_tokens"]),
            "dataset_actual_chars": int(dataset.metadata["actual_chars"]),
            "current_dir": str(CURRENT_DIR),
            "run_dir": str(run_dir),
        }
    )
    write_json(run_dir / "config.json", run_config)

    model = TinyGPT(
        vocab_size=vocab_size,
        block_size=config.block_size,
        n_embd=config.n_embd,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
    )
    optimizer = optim.AdamW(learning_rate=config.learning_rate)
    value_and_grad = nn.value_and_grad(model, language_model_loss)

    def estimate_loss() -> dict[str, float]:
        result = {}
        for split in ["train", "val"]:
            losses = []
            for _ in range(config.eval_iters):
                bx, by = dataset.get_batch(split)
                loss = language_model_loss(model, bx, by)
                mx.eval(loss)
                losses.append(float(loss))
            result[split] = sum(losses) / len(losses)
        return result

    append_log(log_path, f"=== {config.run_name} ===")
    append_log(log_path, f"config: {json.dumps(run_config, ensure_ascii=False)}")

    history: list[dict] = []
    best_val = float("inf")
    best_step = 0
    start = time.perf_counter()
    last_train_loss = None

    for step in range(config.max_iters):
        bx, by = dataset.get_batch("train")
        loss, grads = value_and_grad(model, bx, by)
        optimizer.update(model, grads)
        mx.eval(loss, model.parameters(), optimizer.state)
        last_train_loss = float(loss)

        if step % config.eval_interval != 0 and step != config.max_iters - 1:
            continue

        losses = estimate_loss()
        elapsed = time.perf_counter() - start
        tokens_seen = int((step + 1) * config.batch_size * config.block_size)
        tokens_per_second = tokens_seen / max(elapsed, 1e-9)
        row = {
            "step": int(step),
            "train_loss": float(losses["train"]),
            "val_loss": float(losses["val"]),
            "batch_loss": float(last_train_loss),
            "tokens_seen": tokens_seen,
            "tokens_per_second": tokens_per_second,
            "elapsed_sec": elapsed,
        }
        history.append(row)
        with jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if row["val_loss"] < best_val:
            best_val = row["val_loss"]
            best_step = step
            model.save_weights(str(run_dir / "best_model.safetensors"))
        append_log(
            log_path,
            "step={step:04d} train_loss={train:.4f} val_loss={val:.4f} "
            "tokens_seen={tokens} tokens/sec={tps:.1f} elapsed={elapsed:.1f}s".format(
                step=step,
                train=row["train_loss"],
                val=row["val_loss"],
                tokens=tokens_seen,
                tps=tokens_per_second,
                elapsed=elapsed,
            ),
        )
        write_json(run_dir / "loss_history.json", history)
        plot_run_loss(history, run_dir / "loss_curve.png", config.run_name)

    model.save_weights(str(run_dir / "final_model.safetensors"))
    samples = generate_samples(model, tokenizer, config)
    write_generated_text(run_dir / "final_generated_text.txt", samples)

    total_elapsed = time.perf_counter() - start
    final_row = history[-1]
    metrics = {
        "run_name": config.run_name,
        "dataset_scale": config.dataset_scale,
        "dataset_actual_chars": int(dataset.metadata["actual_chars"]),
        "vocab_size": vocab_size,
        "block_size": config.block_size,
        "batch_size": config.batch_size,
        "n_embd": config.n_embd,
        "num_heads": config.num_heads,
        "num_layers": config.num_layers,
        "max_iters": config.max_iters,
        "learning_rate": config.learning_rate,
        "final_train_loss": final_row["train_loss"],
        "final_val_loss": final_row["val_loss"],
        "best_val_loss": best_val,
        "best_step": best_step,
        "tokens_seen": final_row["tokens_seen"],
        "tokens_per_second": final_row["tokens_per_second"],
        "elapsed_sec": total_elapsed,
        "overfit_gap": final_row["val_loss"] - final_row["train_loss"],
        "final_generated_text_path": str(run_dir / "final_generated_text.txt"),
    }
    write_json(run_dir / "metrics.json", metrics)
    append_log(log_path, f"metrics: {json.dumps(metrics, ensure_ascii=False)}")
    return metrics


def selected_runs(names: list[str] | None, max_iters_override: int | None) -> list[RunConfig]:
    runs = []
    wanted = set(names or [])
    for config in DEFAULT_RUNS:
        if wanted and config.run_name not in wanted:
            continue
        if max_iters_override is not None:
            config = RunConfig(**{**asdict(config), "max_iters": max_iters_override})
        runs.append(config)
    return runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="*", default=None, help="run names to execute")
    parser.add_argument("--max-iters", type=int, default=None, help="override max_iters for every selected run")
    args = parser.parse_args()

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = []
    for config in selected_runs(args.runs, args.max_iters):
        metrics.append(run_experiment(config))
    write_json(RUNS_DIR / "latest_scaling_metrics.json", metrics)
    print("=== Scaling experiments finished ===")
    for row in metrics:
        print(row)


if __name__ == "__main__":
    main()
