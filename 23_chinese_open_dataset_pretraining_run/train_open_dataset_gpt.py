from __future__ import annotations

import argparse
import json
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

from model import TinyGPT, generate_ids, language_model_loss


CURRENT_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = CURRENT_DIR / "data" / "processed"
OUTPUT_DIR = CURRENT_DIR / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
SAMPLES_DIR = OUTPUT_DIR / "samples"
TOKENIZER_PATH = OUTPUT_DIR / "tokenizer" / "chinese_bpe_tokenizer.json"
TRAIN_TOKENS_PATH = PROCESSED_DIR / "train_tokens.npy"
VAL_TOKENS_PATH = PROCESSED_DIR / "val_tokens.npy"
METADATA_PATH = PROCESSED_DIR / "metadata.json"
CONFIG_PATH = OUTPUT_DIR / "config.json"
LOG_PATH = OUTPUT_DIR / "training_log.txt"
LOSS_HISTORY_PATH = OUTPUT_DIR / "loss_history.json"
LOSS_CURVE_PATH = OUTPUT_DIR / "loss_curve.png"
FINAL_TEXT_PATH = OUTPUT_DIR / "final_generated_text.txt"


@dataclass
class TrainConfig:
    seed: int = 42
    block_size: int = 128
    batch_size: int = 16
    n_embd: int = 64
    num_heads: int = 4
    num_layers: int = 2
    learning_rate: float = 2e-3
    max_iters: int = 2000
    eval_interval: int = 100
    eval_iters: int = 10
    sample_interval: int = 500
    checkpoint_interval: int = 500
    sample_prompt: str = "人工智能"
    sample_tokens: int = 160
    temperature: float = 0.8
    top_k: int = 20


class TokenDataset:
    def __init__(self, config: TrainConfig):
        self.config = config
        self.train_tokens = np.load(TRAIN_TOKENS_PATH).astype(np.int32)
        self.val_tokens = np.load(VAL_TOKENS_PATH).astype(np.int32)

    def get_batch(self, split: str):
        source = self.train_tokens if split == "train" else self.val_tokens
        starts = np.random.randint(0, len(source) - self.config.block_size - 1, size=(self.config.batch_size,))
        x = np.stack([source[i : i + self.config.block_size] for i in starts]).astype(np.int32)
        y = np.stack([source[i + 1 : i + self.config.block_size + 1] for i in starts]).astype(np.int32)
        return mx.array(x), mx.array(y)


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def log(message: str) -> None:
    print(message)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(message + "\n")


def decode_ids(tokenizer: Tokenizer, ids) -> str:
    return tokenizer.decode([int(i) for i in ids], skip_special_tokens=True)


def save_sample(model: TinyGPT, tokenizer: Tokenizer, config: TrainConfig, step: int) -> Path:
    bos = tokenizer.token_to_id("<bos>")
    start_ids = [bos] + tokenizer.encode(config.sample_prompt).ids
    ids = generate_ids(model, start_ids, config.sample_tokens, temperature=config.temperature, top_k=config.top_k)
    text = decode_ids(tokenizer, ids)
    path = SAMPLES_DIR / f"sample_step_{step:06d}.txt"
    path.write_text(text, encoding="utf-8")
    return path


def plot_loss(history: list[dict]) -> None:
    steps = [item["step"] for item in history]
    train = [item["train_loss"] for item in history]
    val = [item["val_loss"] for item in history]
    plt.figure(figsize=(8, 5))
    plt.plot(steps, train, label="train loss")
    plt.plot(steps, val, label="val loss")
    plt.xlabel("step")
    plt.ylabel("cross entropy loss")
    plt.title("Open Chinese Dataset Tiny GPT Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(LOSS_CURVE_PATH, dpi=150, bbox_inches="tight")
    plt.close()


def save_checkpoint(model: TinyGPT, step: int, train_loss: float, val_loss: float, is_best: bool) -> dict:
    path = CHECKPOINT_DIR / f"step_{step:06d}_model.safetensors"
    model.save_weights(str(path))
    meta = {"step": step, "model_path": str(path), "train_loss": train_loss, "val_loss": val_loss}
    write_json(CHECKPOINT_DIR / f"step_{step:06d}_meta.json", meta)
    write_json(CHECKPOINT_DIR / "latest.json", meta)
    if is_best:
        write_json(CHECKPOINT_DIR / "best.json", meta)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-iters", type=int, default=None)
    args = parser.parse_args()

    config = TrainConfig()
    if args.max_iters is not None:
        config.max_iters = args.max_iters

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("", encoding="utf-8")
    mx.random.seed(config.seed)
    np.random.seed(config.seed)

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    dataset = TokenDataset(config)

    run_config = asdict(config)
    run_config.update(metadata)
    run_config.update({"current_dir": str(CURRENT_DIR), "output_dir": str(OUTPUT_DIR)})
    write_json(CONFIG_PATH, run_config)

    model = TinyGPT(
        vocab_size=metadata["vocab_size"],
        block_size=config.block_size,
        n_embd=config.n_embd,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
    )
    optimizer = optim.AdamW(learning_rate=config.learning_rate)
    value_and_grad = nn.value_and_grad(model, language_model_loss)

    log("=== Open Chinese Dataset Tiny GPT Training ===")
    log(f"Current dir: {CURRENT_DIR}")
    log(f"Vocab size: {metadata['vocab_size']}")
    log(f"Train tokens: {metadata['train_tokens']}")
    log(f"Val tokens: {metadata['val_tokens']}")
    xb, yb = dataset.get_batch("train")
    shape_info = model.inspect_shapes(xb)
    log(f"Shape check: {shape_info}")
    initial = language_model_loss(model, xb, yb)
    mx.eval(initial)
    log(f"Initial batch loss: {float(initial):.4f}")

    def estimate_loss() -> dict:
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

    history = []
    best_val = float("inf")
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
        is_best = losses["val"] < best_val
        if is_best:
            best_val = losses["val"]
        history.append({"step": step, "train_loss": losses["train"], "val_loss": losses["val"], "elapsed_sec": elapsed})
        write_json(LOSS_HISTORY_PATH, history)
        plot_loss(history)
        sample_path = None
        if step % config.sample_interval == 0 or step == config.max_iters - 1:
            sample_path = save_sample(model, tokenizer, config, step)
        checkpoint = None
        if step % config.checkpoint_interval == 0 or step == config.max_iters - 1 or is_best:
            checkpoint = save_checkpoint(model, step, losses["train"], losses["val"], is_best)
        line = f"step={step:04d} train_loss={losses['train']:.4f} val_loss={losses['val']:.4f} elapsed={elapsed:.1f}s"
        if sample_path:
            line += f" sample={sample_path.name}"
        if checkpoint:
            line += f" checkpoint={checkpoint['model_path']}"
        log(line)

    best_meta = json.loads((CHECKPOINT_DIR / "best.json").read_text(encoding="utf-8"))
    model.load_weights(best_meta["model_path"], strict=True)
    final_path = save_sample(model, tokenizer, config, config.max_iters)
    FINAL_TEXT_PATH.write_text(final_path.read_text(encoding="utf-8"), encoding="utf-8")
    total = time.perf_counter() - start
    run_config.update({"best_val_loss": best_val, "best_checkpoint": best_meta["model_path"], "training_time_sec": round(total, 3)})
    write_json(CONFIG_PATH, run_config)
    log(f"Training finished in {total:.1f}s")
    log(f"Best val loss: {best_val:.4f}")
    log(f"Final generated text: {FINAL_TEXT_PATH}")


if __name__ == "__main__":
    main()
