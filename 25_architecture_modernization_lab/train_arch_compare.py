from __future__ import annotations

import argparse
import json
import shutil
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

import model_baseline
import model_modern


CURRENT_DIR = Path(__file__).resolve().parent
DATA_DIR = CURRENT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = CURRENT_DIR / "outputs"
RUNS_DIR = OUTPUT_DIR / "runs"
TOKENIZER_DIR = OUTPUT_DIR / "tokenizer"
TOKENIZER_PATH = TOKENIZER_DIR / "chinese_bpe_tokenizer.json"

SOURCE_24_DIR = CURRENT_DIR.parent / "24_chinese_gpt_scaling_on_m1_pro"
SOURCE_23_DIR = CURRENT_DIR.parent / "23_chinese_open_dataset_pretraining_run"

PROMPTS = ["人工智能", "大语言模型", "今天我们学习", "本地模型", "中文预训练"]


@dataclass
class TrainConfig:
    block_size: int = 128
    batch_size: int = 16
    n_embd: int = 64
    num_heads: int = 4
    num_layers: int = 2
    learning_rate: float = 2e-3
    max_iters: int = 1000
    eval_interval: int = 200
    eval_iters: int = 8
    sample_tokens: int = 180
    temperature: float = 0.8
    top_k: int = 20
    seed: int = 2025


class TokenDataset:
    def __init__(self, config: TrainConfig):
        self.config = config
        self.train_tokens = np.load(PROCESSED_DIR / "train_tokens.npy").astype(np.int32)
        self.val_tokens = np.load(PROCESSED_DIR / "val_tokens.npy").astype(np.int32)

    def get_batch(self, split: str):
        source = self.train_tokens if split == "train" else self.val_tokens
        max_start = len(source) - self.config.block_size - 1
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


def prepare_data() -> dict:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)

    candidates = [
        {
            "name": "lesson_24_medium",
            "train": SOURCE_24_DIR / "data" / "processed" / "medium" / "train_tokens.npy",
            "val": SOURCE_24_DIR / "data" / "processed" / "medium" / "val_tokens.npy",
            "metadata": SOURCE_24_DIR / "data" / "processed" / "medium" / "metadata.json",
            "tokenizer": SOURCE_24_DIR / "outputs" / "tokenizer" / "chinese_bpe_tokenizer.json",
        },
        {
            "name": "lesson_23_processed",
            "train": SOURCE_23_DIR / "data" / "processed" / "train_tokens.npy",
            "val": SOURCE_23_DIR / "data" / "processed" / "val_tokens.npy",
            "metadata": SOURCE_23_DIR / "data" / "processed" / "metadata.json",
            "tokenizer": SOURCE_23_DIR / "outputs" / "tokenizer" / "chinese_bpe_tokenizer.json",
        },
    ]

    for candidate in candidates:
        if all(candidate[key].exists() for key in ["train", "val", "metadata", "tokenizer"]):
            shutil.copyfile(candidate["train"], PROCESSED_DIR / "train_tokens.npy")
            shutil.copyfile(candidate["val"], PROCESSED_DIR / "val_tokens.npy")
            shutil.copyfile(candidate["metadata"], PROCESSED_DIR / "metadata.json")
            shutil.copyfile(candidate["tokenizer"], TOKENIZER_PATH)
            metadata = json.loads((PROCESSED_DIR / "metadata.json").read_text(encoding="utf-8"))
            metadata["copied_from"] = candidate["name"]
            metadata["local_train_tokens_path"] = str(PROCESSED_DIR / "train_tokens.npy")
            metadata["local_val_tokens_path"] = str(PROCESSED_DIR / "val_tokens.npy")
            metadata["local_tokenizer_path"] = str(TOKENIZER_PATH)
            write_json(PROCESSED_DIR / "metadata.json", metadata)
            return metadata

    raise FileNotFoundError("找不到第 23/24 课可复用的 processed token 数据和 tokenizer。")


def count_parameters(obj) -> int:
    if isinstance(obj, mx.array):
        return int(np.prod(obj.shape))
    if isinstance(obj, dict):
        return sum(count_parameters(value) for value in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(count_parameters(value) for value in obj)
    return 0


def decode_ids(tokenizer: Tokenizer, ids) -> str:
    return tokenizer.decode([int(i) for i in ids], skip_special_tokens=True)


def generate_samples(model, generate_fn, tokenizer: Tokenizer, config: TrainConfig) -> dict[str, str]:
    bos_id = tokenizer.token_to_id("<bos>")
    samples = {}
    for prompt in PROMPTS:
        start_ids = [bos_id] + tokenizer.encode(prompt).ids
        ids = generate_fn(
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


def plot_loss(history: list[dict], path: Path, title: str) -> None:
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


def make_model(run_name: str, vocab_size: int, config: TrainConfig):
    if run_name == "baseline_tiny_gpt":
        return (
            model_baseline.BaselineTinyGPT(
                vocab_size=vocab_size,
                block_size=config.block_size,
                n_embd=config.n_embd,
                num_heads=config.num_heads,
                num_layers=config.num_layers,
            ),
            model_baseline.language_model_loss,
            model_baseline.generate_ids,
            {
                "position": "learned position embedding",
                "norm": "LayerNorm",
                "ffn": "GELU FeedForward",
                "lm_head": "separate Linear(n_embd, vocab_size)",
            },
        )
    if run_name == "modern_tiny_gpt":
        return (
            model_modern.ModernTinyGPT(
                vocab_size=vocab_size,
                block_size=config.block_size,
                n_embd=config.n_embd,
                num_heads=config.num_heads,
                num_layers=config.num_layers,
            ),
            model_modern.language_model_loss,
            model_modern.generate_ids,
            {
                "position": "RoPE on Q/K",
                "norm": "RMSNorm",
                "ffn": "SwiGLU",
                "lm_head": "weight tied with token embedding table",
            },
        )
    raise ValueError(f"unknown run: {run_name}")


def train_one(run_name: str, config: TrainConfig, metadata: dict) -> dict:
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "training_log.txt"
    jsonl_path = run_dir / "training_log.jsonl"
    log_path.write_text("", encoding="utf-8")
    jsonl_path.write_text("", encoding="utf-8")

    mx.random.seed(config.seed)
    np.random.seed(config.seed)

    dataset = TokenDataset(config)
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    model, loss_fn, generate_fn, architecture = make_model(run_name, int(metadata["vocab_size"]), config)
    optimizer = optim.AdamW(learning_rate=config.learning_rate)
    value_and_grad = nn.value_and_grad(model, loss_fn)

    xb, yb = dataset.get_batch("train")
    shape_info = model.inspect_shapes(xb)
    param_count = count_parameters(model.parameters())
    mx.eval(model(xb))

    run_config = {
        **asdict(config),
        "run_name": run_name,
        "architecture": architecture,
        "vocab_size": int(metadata["vocab_size"]),
        "train_tokens": int(len(dataset.train_tokens)),
        "val_tokens": int(len(dataset.val_tokens)),
        "parameter_count": param_count,
        "shape_info": shape_info,
        "data_source": metadata.get("copied_from"),
    }
    write_json(run_dir / "config.json", run_config)
    append_log(log_path, f"=== {run_name} ===")
    append_log(log_path, f"config: {json.dumps(run_config, ensure_ascii=False)}")

    def estimate_loss() -> dict[str, float]:
        result = {}
        for split in ["train", "val"]:
            losses = []
            for _ in range(config.eval_iters):
                bx, by = dataset.get_batch(split)
                loss = loss_fn(model, bx, by)
                mx.eval(loss)
                losses.append(float(loss))
            result[split] = sum(losses) / len(losses)
        return result

    history: list[dict] = []
    best_val = float("inf")
    best_step = 0
    start = time.perf_counter()

    for step in range(config.max_iters):
        bx, by = dataset.get_batch("train")
        loss, grads = value_and_grad(model, bx, by)
        optimizer.update(model, grads)
        mx.eval(loss, model.parameters(), optimizer.state)

        if step % config.eval_interval != 0 and step != config.max_iters - 1:
            continue

        losses = estimate_loss()
        elapsed = time.perf_counter() - start
        tokens_seen = int((step + 1) * config.batch_size * config.block_size)
        tokens_per_second = tokens_seen / max(elapsed, 1e-9)
        row = {
            "step": int(step),
            "train_loss": losses["train"],
            "val_loss": losses["val"],
            "tokens_seen": tokens_seen,
            "tokens_per_second": tokens_per_second,
            "elapsed_sec": elapsed,
        }
        history.append(row)
        if row["val_loss"] < best_val:
            best_val = row["val_loss"]
            best_step = step
            model.save_weights(str(run_dir / "best_model.safetensors"))
        with jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
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
        plot_loss(history, run_dir / "loss_curve.png", run_name)

    model.save_weights(str(run_dir / "final_model.safetensors"))
    samples = generate_samples(model, generate_fn, tokenizer, config)
    write_generated_text(run_dir / "final_generated_text.txt", samples)

    final = history[-1]
    metrics = {
        "run_name": run_name,
        "architecture": architecture,
        "parameter_count": param_count,
        "block_size": config.block_size,
        "batch_size": config.batch_size,
        "n_embd": config.n_embd,
        "num_heads": config.num_heads,
        "num_layers": config.num_layers,
        "learning_rate": config.learning_rate,
        "max_iters": config.max_iters,
        "tokens_seen": final["tokens_seen"],
        "tokens_per_second": final["tokens_per_second"],
        "final_train_loss": final["train_loss"],
        "final_val_loss": final["val_loss"],
        "best_val_loss": best_val,
        "best_step": best_step,
        "overfit_gap": final["val_loss"] - final["train_loss"],
        "elapsed_sec": time.perf_counter() - start,
        "final_generated_text_path": str(run_dir / "final_generated_text.txt"),
    }
    write_json(run_dir / "metrics.json", metrics)
    append_log(log_path, f"metrics: {json.dumps(metrics, ensure_ascii=False)}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-iters", type=int, default=None)
    args = parser.parse_args()

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    metadata = prepare_data()
    config = TrainConfig()
    if args.max_iters is not None:
        config.max_iters = args.max_iters

    metrics = []
    for run_name in ["baseline_tiny_gpt", "modern_tiny_gpt"]:
        metrics.append(train_one(run_name, config, metadata))
    write_json(RUNS_DIR / "architecture_compare_metrics.json", metrics)

    print("=== Architecture comparison finished ===")
    for row in metrics:
        print(row)


if __name__ == "__main__":
    main()
