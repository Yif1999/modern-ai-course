from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten


SEED = 42


def build_default_text() -> str:
    base_lines = [
        "hello ai lab",
        "hello mlx",
        "hello tiny gpt",
        "we learn token embedding",
        "we learn position embedding",
        "we learn next token prediction",
        "we learn self attention",
        "we learn multi head attention",
        "we learn transformer block",
        "tiny gpt stacks transformer blocks",
        "each block keeps the hidden shape",
        "residual connection helps information flow",
        "layernorm helps stabilize training",
        "feed forward layers improve representation",
        "causal mask prevents looking into the future",
        "language models predict the next token",
        "train loss shows how well the model fits seen data",
        "val loss shows how well the model generalizes",
        "checkpoints let us resume training later",
        "generated samples help us judge text quality",
        "small models on small text can memorize easily",
        "better engineering makes experiments easier to trust",
    ]
    return ("\n".join(base_lines) + "\n") * 14


class MultiHeadCausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, num_heads: int):
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
        return mx.transpose(x, (0, 2, 1, 3))

    def _merge_heads(self, x):
        batch, num_heads, seq_len, head_size = x.shape
        x = mx.transpose(x, (0, 2, 1, 3))
        return x.reshape(batch, seq_len, num_heads * head_size)

    def __call__(self, x):
        _, seq_len, _ = x.shape

        q = self._split_heads(self.query(x))
        k = self._split_heads(self.key(x))
        v = self._split_heads(self.value(x))

        scores = q @ mx.transpose(k, (0, 1, 3, 2))
        scores = scores / math.sqrt(self.head_size)

        mask = mx.array(np.tril(np.ones((seq_len, seq_len), dtype=np.float32)))
        mask = mask.reshape(1, 1, seq_len, seq_len)
        masked_scores = mx.where(mask == 1, scores, mx.full(scores.shape, -1e9))

        weights = nn.softmax(masked_scores, axis=-1)
        out = weights @ v
        out = self._merge_heads(out)
        out = self.proj(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.linear1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear2 = nn.Linear(4 * n_embd, n_embd)

    def __call__(self, x):
        x = self.linear1(x)
        x = nn.gelu(x)
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, n_embd: int, num_heads: int):
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
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_embd: int,
        num_heads: int,
        num_layers: int,
    ):
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

    def generate(
        self,
        start_ids,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int | None = 8,
    ):
        ids = [int(i) for i in start_ids]

        for _ in range(max_new_tokens):
            context = ids[-self.block_size :]
            idx = mx.array([context], dtype=mx.int32)
            logits = self(idx)
            logits_last = logits[0, -1, :]
            logits_last = logits_last / max(temperature, 1e-6)

            probs = nn.softmax(logits_last, axis=-1)
            mx.eval(probs)

            probs_np = np.array(probs, dtype=np.float64)
            probs_np = np.maximum(probs_np, 0.0)

            if top_k is not None and top_k < len(probs_np):
                keep = np.argpartition(probs_np, -top_k)[-top_k:]
                filtered = np.zeros_like(probs_np)
                filtered[keep] = probs_np[keep]
                probs_np = filtered

            probs_sum = probs_np.sum()
            if probs_sum <= 0:
                next_id = int(np.argmax(np.array(logits_last)))
            else:
                probs_np = probs_np / probs_sum
                next_id = int(np.random.choice(len(probs_np), p=probs_np))

            ids.append(next_id)

        return ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        choices=["none", "latest", "best"],
        default="none",
        help="Resume from a saved checkpoint pointer.",
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Load a checkpoint and only generate text.",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=None,
        help="Override the default max training iterations.",
    )
    args = parser.parse_args()

    print("=== MLX Tiny GPT Training Improvements ===")

    mx.random.seed(SEED)
    np.random.seed(SEED)

    current_dir = Path(__file__).resolve().parent
    data_dir = current_dir / "data"
    output_dir = current_dir / "outputs"
    checkpoints_dir = output_dir / "checkpoints"
    samples_dir = output_dir / "samples"
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    text_path = data_dir / "tiny_text.txt"
    if not text_path.exists():
        text_path.write_text(build_default_text(), encoding="utf-8")
    text = text_path.read_text(encoding="utf-8")

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}

    def encode(s: str):
        missing = sorted(set(s) - set(stoi))
        if missing:
            raise ValueError(f"Text contains tokens outside vocab: {missing}")
        return [stoi[ch] for ch in s]

    def decode(ids):
        return "".join([itos[int(i)] for i in ids])

    encoded = np.array(encode(text), dtype=np.int32)
    split_idx = int(0.9 * len(encoded))
    train_data = encoded[:split_idx]
    val_data = encoded[split_idx:]

    config = {
        "seed": SEED,
        "block_size": 32,
        "batch_size": 32,
        "n_embd": 64,
        "num_heads": 4,
        "num_layers": 2,
        "learning_rate": 3e-3,
        "max_iters": 1500,
        "eval_interval": 100,
        "eval_iters": 10,
        "sample_interval": 300,
        "sample_tokens": 180,
        "sample_prompt": "hello ",
        "vocab_size": vocab_size,
        "data_length": int(len(encoded)),
        "train_size": int(len(train_data)),
        "val_size": int(len(val_data)),
        "text_path": str(text_path),
        "current_dir": str(current_dir),
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
    }
    if args.max_iters is not None:
        config["max_iters"] = int(args.max_iters)

    if len(train_data) <= config["block_size"] + 1:
        raise ValueError("train split is too short for the configured block_size")
    if len(val_data) <= config["block_size"] + 1:
        raise ValueError("val split is too short for the configured block_size")

    config_path = output_dir / "config.json"
    history_path = output_dir / "loss_history.json"
    training_log_path = output_dir / "training_log.txt"
    generate_only_log_path = output_dir / "generate_only_log.txt"
    log_path = generate_only_log_path if args.generate_only else training_log_path
    final_text_path = output_dir / "final_generated_text.txt"
    latest_pointer_path = checkpoints_dir / "latest.json"
    best_pointer_path = checkpoints_dir / "best.json"

    if args.resume == "none" and not args.generate_only:
        log_path.write_text("", encoding="utf-8")
        history = []
    else:
        if history_path.exists():
            history = json.loads(history_path.read_text(encoding="utf-8"))
        else:
            history = []

    def log(message: str):
        print(message)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(message + "\n")

    def write_config(extra: dict | None = None):
        payload = dict(config)
        if extra:
            payload.update(extra)
        config_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def get_batch(split: str):
        source = train_data if split == "train" else val_data
        starts = np.random.randint(
            0,
            len(source) - config["block_size"] - 1,
            size=(config["batch_size"],),
        )
        x = np.stack(
            [source[i : i + config["block_size"]] for i in starts],
        ).astype(np.int32)
        y = np.stack(
            [source[i + 1 : i + config["block_size"] + 1] for i in starts],
        ).astype(np.int32)
        return mx.array(x), mx.array(y)

    model = TinyGPT(
        vocab_size=vocab_size,
        block_size=config["block_size"],
        n_embd=config["n_embd"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
    )
    optimizer = optim.AdamW(learning_rate=config["learning_rate"])

    def loss_fn(model_obj, idx, targets):
        logits = model_obj(idx)
        batch, seq_len, channels = logits.shape
        logits_flat = logits.reshape(batch * seq_len, channels)
        targets_flat = targets.reshape(batch * seq_len)
        return nn.losses.cross_entropy(
            logits_flat,
            targets_flat,
            reduction="mean",
        )

    value_and_grad_fn = nn.value_and_grad(model, loss_fn)

    def estimate_loss():
        out = {}
        for split in ["train", "val"]:
            losses = []
            for _ in range(config["eval_iters"]):
                xb, yb = get_batch(split)
                loss = loss_fn(model, xb, yb)
                mx.eval(loss)
                losses.append(float(loss))
            out[split] = sum(losses) / len(losses)
        return out

    def save_loss_curve():
        if not history:
            return
        steps = [item["step"] for item in history]
        train_losses = [item["train_loss"] for item in history]
        val_losses = [item["val_loss"] for item in history]

        plt.figure(figsize=(8, 5))
        plt.plot(steps, train_losses, label="train loss")
        plt.plot(steps, val_losses, label="val loss")
        plt.xlabel("step")
        plt.ylabel("cross entropy loss")
        plt.title("Tiny GPT Training Loss")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "loss_curve.png", dpi=150, bbox_inches="tight")
        plt.close()

    def save_generated_sample(step: int, prefix: str = "sample"):
        prompt = config["sample_prompt"]
        generated_ids = model.generate(
            encode(prompt),
            max_new_tokens=config["sample_tokens"],
            temperature=0.8,
            top_k=8,
        )
        generated_text = decode(generated_ids)
        sample_path = samples_dir / f"{prefix}_step_{step:06d}.txt"
        sample_path.write_text(generated_text, encoding="utf-8")
        return sample_path, generated_text

    def save_checkpoint(step: int, train_loss: float, val_loss: float, best_val_loss: float):
        model_path = checkpoints_dir / f"step_{step:06d}_model.safetensors"
        optimizer_path = checkpoints_dir / f"step_{step:06d}_optimizer.safetensors"
        meta_path = checkpoints_dir / f"step_{step:06d}_meta.json"

        model.save_weights(str(model_path))
        flat_state = dict(tree_flatten(optimizer.state))
        mx.save_safetensors(str(optimizer_path), flat_state)

        meta = {
            "step": step,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "saved_at_unix": time.time(),
            "model_path": str(model_path),
            "optimizer_path": str(optimizer_path),
            "config_path": str(config_path),
        }
        meta_path.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        latest_pointer_path.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return model_path, optimizer_path, meta_path

    def load_checkpoint(kind: str):
        pointer_path = latest_pointer_path if kind == "latest" else best_pointer_path
        if not pointer_path.exists():
            raise FileNotFoundError(f"Checkpoint pointer not found: {pointer_path}")

        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        model.load_weights(pointer["model_path"], strict=True)
        loaded_opt_state = mx.load(pointer["optimizer_path"])
        optimizer.state = tree_unflatten(loaded_opt_state)
        optimizer.init(model.trainable_parameters())
        return pointer

    if not args.generate_only or not config_path.exists():
        write_config(
            {
                "chars": chars,
                "resume_mode": args.resume,
                "generate_only": args.generate_only,
            }
        )

    log(f"Current dir: {current_dir}")
    log(f"Data dir: {data_dir}")
    log(f"Output dir: {output_dir}")
    log(f"Training text: {text_path}")
    log(f"Vocab size: {vocab_size}")
    log(f"Data length: {len(encoded)}")
    log(f"Train size: {len(train_data)}")
    log(f"Val size: {len(val_data)}")

    start_step = 0
    best_val_loss = float("inf")

    if args.resume != "none":
        pointer = load_checkpoint(args.resume)
        start_step = int(pointer["step"]) + 1
        best_val_loss = float(pointer.get("best_val_loss", pointer["val_loss"]))
        log(
            f"Resumed from {args.resume} checkpoint at step={pointer['step']} "
            f"val_loss={pointer['val_loss']:.4f}"
        )
    elif history:
        history = []

    xb, yb = get_batch("train")
    initial_logits = model(xb)
    initial_loss = loss_fn(model, xb, yb)
    mx.eval(xb, yb, initial_logits, initial_loss)
    log(f"Initial x batch shape: {xb.shape}")
    log(f"Initial y batch shape: {yb.shape}")
    log(f"Initial logits shape: {initial_logits.shape}")
    log(f"Initial batch loss: {float(initial_loss):.4f}")

    if args.generate_only:
        sample_path, generated_text = save_generated_sample(start_step, prefix="generate_only")
        final_text_path.write_text(generated_text, encoding="utf-8")
        log(f"Generate-only output saved: {sample_path}")
        log(f"Final generated text saved: {final_text_path}")
        return

    train_start = time.perf_counter()
    initial_eval_step = 0 if start_step == 0 else start_step - 1
    first_eval = estimate_loss()
    best_val_loss = min(best_val_loss, first_eval["val"])

    should_record_initial_eval = (
        not history or int(history[-1]["step"]) != int(initial_eval_step)
    )
    if should_record_initial_eval:
        history.append(
            {
                "step": initial_eval_step,
                "train_loss": first_eval["train"],
                "val_loss": first_eval["val"],
                "elapsed_sec": 0.0,
            }
        )
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        save_loss_curve()
        save_checkpoint(
            initial_eval_step,
            first_eval["train"],
            first_eval["val"],
            best_val_loss,
        )
        best_pointer_path.write_text(
            latest_pointer_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        sample_path, _ = save_generated_sample(initial_eval_step)
        log(
            f"initial_eval step={initial_eval_step:04d} "
            f"train_loss={first_eval['train']:.4f} "
            f"val_loss={first_eval['val']:.4f} "
            f"sample={sample_path.name}"
        )
    else:
        log(
            f"initial_eval skipped because step={initial_eval_step:04d} "
            "already exists in history"
        )

    for step in range(start_step, config["max_iters"]):
        xb, yb = get_batch("train")
        loss, grads = value_and_grad_fn(model, xb, yb)
        optimizer.update(model, grads)
        mx.eval(loss, model.parameters(), optimizer.state)

        should_eval = step % config["eval_interval"] == 0 or step == config["max_iters"] - 1
        if start_step == 0 and step == 0:
            should_eval = step == config["max_iters"] - 1
        if not should_eval:
            continue

        losses = estimate_loss()
        elapsed = time.perf_counter() - train_start
        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
        history.append(
            {
                "step": step,
                "train_loss": losses["train"],
                "val_loss": losses["val"],
                "elapsed_sec": elapsed,
            }
        )
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        save_loss_curve()
        save_checkpoint(step, losses["train"], losses["val"], best_val_loss)
 
        if abs(losses["val"] - best_val_loss) < 1e-12:
            best_pointer_path.write_text(
                latest_pointer_path.read_text(encoding="utf-8"),
                encoding="utf-8",
            )

        sample_saved = None
        if step % config["sample_interval"] == 0 or step == config["max_iters"] - 1:
            sample_saved, _ = save_generated_sample(step)

        line = (
            f"eval step={step:04d} "
            f"train_loss={losses['train']:.4f} "
            f"val_loss={losses['val']:.4f} "
            f"elapsed={elapsed:.1f}s"
        )
        if sample_saved is not None:
            line += f" sample={sample_saved.name}"
        log(line)

    total_time = time.perf_counter() - train_start

    best_pointer = json.loads(best_pointer_path.read_text(encoding="utf-8"))
    model.load_weights(best_pointer["model_path"], strict=True)

    final_prompt = config["sample_prompt"]
    final_ids = model.generate(
        encode(final_prompt),
        max_new_tokens=260,
        temperature=0.8,
        top_k=8,
    )
    final_text = decode(final_ids)
    final_text_path.write_text(final_text, encoding="utf-8")

    write_config(
        {
            "chars": chars,
            "resume_mode": args.resume,
            "generate_only": args.generate_only,
            "best_val_loss": best_val_loss,
            "best_checkpoint": str(best_pointer["model_path"]),
            "training_time_sec": round(total_time, 3),
        }
    )

    log(f"Training finished in {total_time:.1f}s")
    log(f"Best val loss: {best_val_loss:.4f}")
    log(f"Best checkpoint: {best_pointer['model_path']}")
    log(f"Loss curve saved: {output_dir / 'loss_curve.png'}")
    log(f"Final generated text saved: {final_text_path}")


if __name__ == "__main__":
    main()
