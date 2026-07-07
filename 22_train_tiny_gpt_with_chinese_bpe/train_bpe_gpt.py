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

from model import TinyGPT, generate_ids, language_model_loss
from prepare_bpe_dataset import TRAIN_TOKENS_PATH, VAL_TOKENS_PATH, prepare_bpe_dataset
from tokenizer_bpe import BPETokenizer, TOKENIZER_PATH
from tokenizers import Tokenizer


CURRENT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = CURRENT_DIR / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
SAMPLES_DIR = OUTPUT_DIR / "samples"
CONFIG_PATH = OUTPUT_DIR / "config.json"
LOG_PATH = OUTPUT_DIR / "training_log.txt"
LOSS_HISTORY_PATH = OUTPUT_DIR / "loss_history.json"
LOSS_CURVE_PATH = OUTPUT_DIR / "loss_curve.png"
FINAL_TEXT_PATH = OUTPUT_DIR / "final_generated_text.txt"


@dataclass
class TrainConfig:
    seed: int = 42
    block_size: int = 64
    batch_size: int = 32
    n_embd: int = 64
    num_heads: int = 4
    num_layers: int = 2
    learning_rate: float = 2e-3
    max_iters: int = 1600
    eval_interval: int = 100
    eval_iters: int = 10
    sample_interval: int = 400
    checkpoint_interval: int = 400
    sample_prompt: str = "人工智能"
    sample_tokens: int = 120
    sample_temperature: float = 0.8
    sample_top_k: int = 20


class BPETokenDataset:
    def __init__(self, config: TrainConfig):
        self.config = config
        self.train_tokens = np.load(TRAIN_TOKENS_PATH).astype(np.int32)
        self.val_tokens = np.load(VAL_TOKENS_PATH).astype(np.int32)

    def get_batch(self, split: str):
        source = self.train_tokens if split == "train" else self.val_tokens
        if len(source) <= self.config.block_size + 1:
            raise ValueError(f"{split} tokens are too short for block_size={self.config.block_size}")
        starts = np.random.randint(
            0,
            len(source) - self.config.block_size - 1,
            size=(self.config.batch_size,),
        )
        x = np.stack([source[i : i + self.config.block_size] for i in starts]).astype(np.int32)
        y = np.stack([source[i + 1 : i + self.config.block_size + 1] for i in starts]).astype(np.int32)
        return mx.array(x), mx.array(y)


def ensure_output_dirs() -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    mx.random.seed(seed)
    np.random.seed(seed)


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def log(message: str) -> None:
    print(message)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(message + "\n")


def plot_loss_curve(history: list[dict]) -> None:
    steps = [item["step"] for item in history]
    train_losses = [item["train_loss"] for item in history]
    val_losses = [item["val_loss"] for item in history]
    plt.figure(figsize=(8, 5))
    plt.plot(steps, train_losses, label="train loss")
    plt.plot(steps, val_losses, label="val loss")
    plt.xlabel("step")
    plt.ylabel("cross entropy loss")
    plt.title("Chinese BPE Tiny GPT Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(LOSS_CURVE_PATH, dpi=150, bbox_inches="tight")
    plt.close()


def save_checkpoint(model: TinyGPT, step: int, train_loss: float, val_loss: float, is_best: bool) -> dict:
    model_path = CHECKPOINT_DIR / f"step_{step:06d}_model.safetensors"
    meta_path = CHECKPOINT_DIR / f"step_{step:06d}_meta.json"
    model.save_weights(str(model_path))
    meta = {
        "step": int(step),
        "model_path": str(model_path),
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "saved_at_unix": time.time(),
    }
    write_json(meta_path, meta)
    write_json(CHECKPOINT_DIR / "latest.json", meta)
    if is_best:
        write_json(CHECKPOINT_DIR / "best.json", meta)
    return meta


def load_tokenizer() -> BPETokenizer:
    return BPETokenizer(Tokenizer.from_file(str(TOKENIZER_PATH)))


def save_sample(model: TinyGPT, tokenizer: BPETokenizer, config: TrainConfig, step: int, prefix: str = "sample") -> Path:
    start_ids = tokenizer.encode(config.sample_prompt, add_bos=True)
    ids = generate_ids(
        model,
        start_ids,
        max_new_tokens=config.sample_tokens,
        temperature=config.sample_temperature,
        top_k=config.sample_top_k,
    )
    text = tokenizer.decode(ids)
    path = SAMPLES_DIR / f"{prefix}_step_{step:06d}.txt"
    path.write_text(text, encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument("--force-data", action="store_true")
    args = parser.parse_args()

    config = TrainConfig()
    if args.max_iters is not None:
        config.max_iters = args.max_iters

    ensure_output_dirs()
    LOG_PATH.write_text("", encoding="utf-8")
    set_seed(config.seed)

    metadata = prepare_bpe_dataset(force=args.force_data)
    tokenizer = load_tokenizer()
    dataset = BPETokenDataset(config)

    run_config = asdict(config)
    run_config.update(
        {
            "current_dir": str(CURRENT_DIR),
            "data_dir": str(CURRENT_DIR / "data"),
            "output_dir": str(OUTPUT_DIR),
            "tokenizer_path": str(TOKENIZER_PATH),
            "vocab_size": tokenizer.vocab_size,
            "train_token_count": metadata["train_token_count"],
            "val_token_count": metadata["val_token_count"],
            "bpe_token_count_without_special": metadata["bpe_token_count_without_special"],
            "average_chars_per_bpe_token": metadata["average_chars_per_bpe_token"],
            "tokenizer_source": metadata["tokenizer_source"],
        }
    )
    write_json(CONFIG_PATH, run_config)

    model = TinyGPT(
        vocab_size=tokenizer.vocab_size,
        block_size=config.block_size,
        n_embd=config.n_embd,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
    )
    optimizer = optim.AdamW(learning_rate=config.learning_rate)
    value_and_grad_fn = nn.value_and_grad(model, language_model_loss)

    log("=== Chinese BPE Tiny GPT Training ===")
    log(f"Current dir: {CURRENT_DIR}")
    log(f"Data dir: {CURRENT_DIR / 'data'}")
    log(f"Output dir: {OUTPUT_DIR}")
    log(f"Tokenizer source: {metadata['tokenizer_source']}")
    log(f"Tokenizer path: {TOKENIZER_PATH}")
    log(f"Vocab size: {tokenizer.vocab_size}")
    log(f"BPE tokens without special: {metadata['bpe_token_count_without_special']}")
    log(f"Train tokens: {metadata['train_token_count']}")
    log(f"Val tokens: {metadata['val_token_count']}")

    xb, yb = dataset.get_batch("train")
    shape_info = model.inspect_shapes(xb)
    initial_loss = language_model_loss(model, xb, yb)
    mx.eval(initial_loss)
    log("Shape check:")
    for key, value in shape_info.items():
        log(f"  {key}: {value}")
    log(f"Initial batch loss: {float(initial_loss):.4f}")

    def estimate_loss() -> dict:
        results = {}
        for split in ["train", "val"]:
            losses = []
            for _ in range(config.eval_iters):
                bx, by = dataset.get_batch(split)
                loss = language_model_loss(model, bx, by)
                mx.eval(loss)
                losses.append(float(loss))
            results[split] = sum(losses) / len(losses)
        return results

    history: list[dict] = []
    best_val_loss = float("inf")
    start_time = time.perf_counter()

    for step in range(config.max_iters):
        bx, by = dataset.get_batch("train")
        loss, grads = value_and_grad_fn(model, bx, by)
        optimizer.update(model, grads)
        mx.eval(loss, model.parameters(), optimizer.state)

        if step % config.eval_interval != 0 and step != config.max_iters - 1:
            continue

        losses = estimate_loss()
        elapsed = time.perf_counter() - start_time
        is_best = losses["val"] < best_val_loss
        if is_best:
            best_val_loss = losses["val"]

        history.append(
            {
                "step": step,
                "train_loss": losses["train"],
                "val_loss": losses["val"],
                "elapsed_sec": elapsed,
            }
        )
        write_json(LOSS_HISTORY_PATH, history)
        plot_loss_curve(history)

        sample_path = None
        if step % config.sample_interval == 0 or step == config.max_iters - 1:
            sample_path = save_sample(model, tokenizer, config, step)

        checkpoint_meta = None
        if step % config.checkpoint_interval == 0 or step == config.max_iters - 1 or is_best:
            checkpoint_meta = save_checkpoint(model, step, losses["train"], losses["val"], is_best=is_best)

        line = (
            f"step={step:04d} train_loss={losses['train']:.4f} "
            f"val_loss={losses['val']:.4f} elapsed={elapsed:.1f}s"
        )
        if sample_path is not None:
            line += f" sample={sample_path.name}"
        if checkpoint_meta is not None:
            line += f" checkpoint={checkpoint_meta['model_path']}"
        log(line)

    best_meta = json.loads((CHECKPOINT_DIR / "best.json").read_text(encoding="utf-8"))
    model.load_weights(best_meta["model_path"], strict=True)
    final_ids = generate_ids(
        model,
        tokenizer.encode(config.sample_prompt, add_bos=True),
        max_new_tokens=180,
        temperature=config.sample_temperature,
        top_k=config.sample_top_k,
    )
    final_text = tokenizer.decode(final_ids)
    FINAL_TEXT_PATH.write_text(final_text, encoding="utf-8")

    total_time = time.perf_counter() - start_time
    run_config.update(
        {
            "best_checkpoint": best_meta["model_path"],
            "best_val_loss": best_val_loss,
            "training_time_sec": round(total_time, 3),
        }
    )
    write_json(CONFIG_PATH, run_config)

    log(f"Training finished in {total_time:.1f}s")
    log(f"Best val loss: {best_val_loss:.4f}")
    log(f"Best checkpoint: {best_meta['model_path']}")
    log(f"Loss curve: {LOSS_CURVE_PATH}")
    log(f"Final generated text: {FINAL_TEXT_PATH}")


if __name__ == "__main__":
    main()
