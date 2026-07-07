from __future__ import annotations

import argparse
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from config import CHECKPOINT_DIR, OUTPUT_DIR, SAMPLES_DIR, TrainConfig, ensure_project_dirs
from dataset import TinyTextDataset
from model import TinyGPT, generate_ids, language_model_loss
from prepare_dataset import prepare_dataset
from tokenizer import CharacterTokenizer
from utils import (
    append_log,
    load_checkpoint,
    plot_loss_curve,
    save_checkpoint,
    save_sample_text,
    set_seed,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument("--resume", choices=["none", "latest", "best"], default="none")
    parser.add_argument("--generate-only", action="store_true")
    args = parser.parse_args()

    config = TrainConfig()
    if args.max_iters is not None:
        config.max_iters = args.max_iters

    ensure_project_dirs()
    set_seed(config.seed)

    train_log_path = OUTPUT_DIR / "training_log.txt"
    history_path = OUTPUT_DIR / "loss_history.json"
    config_path = OUTPUT_DIR / "config.json"
    final_text_path = OUTPUT_DIR / "final_generated_text.txt"

    if args.resume == "none" and not args.generate_only:
        train_log_path.write_text("", encoding="utf-8")

    meta = prepare_dataset(config)
    tokenizer = CharacterTokenizer.load(config.vocab_path)
    dataset = TinyTextDataset(config)

    run_config = config.to_dict()
    run_config.update(
        {
            "vocab_size": tokenizer.vocab_size,
            "data_length": meta["data_length"],
            "train_size": meta["train_size"],
            "val_size": meta["val_size"],
            "chars": tokenizer.chars,
            "resume": args.resume,
            "generate_only": args.generate_only,
        }
    )
    write_json(config_path, run_config)

    model = TinyGPT(
        vocab_size=tokenizer.vocab_size,
        block_size=config.block_size,
        n_embd=config.n_embd,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
    )
    optimizer = optim.AdamW(learning_rate=config.learning_rate)
    value_and_grad_fn = nn.value_and_grad(model, language_model_loss)

    def log(message: str) -> None:
        append_log(train_log_path, message)

    log("=== Tiny GPT Training Pipeline ===")
    log(f"Current dir: {run_config['project_dir']}")
    log(f"Data dir: {run_config['data_dir']}")
    log(f"Raw text: {run_config['raw_text_path']}")
    log(f"Processed data: {run_config['processed_path']}")
    log(f"Output dir: {run_config['output_dir']}")
    log(f"Vocab size: {tokenizer.vocab_size}")
    log(f"Train size: {meta['train_size']}")
    log(f"Val size: {meta['val_size']}")

    start_step = 0
    best_val_loss = float("inf")

    if args.resume != "none" or args.generate_only:
        pointer_kind = args.resume if args.resume != "none" else "latest"
        loaded = load_checkpoint(model, optimizer=None if args.generate_only else optimizer, kind=pointer_kind)
        loaded_step = loaded.get("step")
        start_step = 0 if loaded_step is None else int(loaded_step) + 1
        best_val_loss = float(loaded.get("val_loss", float("inf")))
        log(f"Loaded {pointer_kind} checkpoint: {loaded['model_path']}")

    xb, yb = dataset.get_batch("train")
    shape_info = model.inspect_shapes(xb)
    initial_loss = language_model_loss(model, xb, yb)
    mx.eval(xb, yb, initial_loss)

    log("Shape check:")
    for key, value in shape_info.items():
        log(f"  {key}: {value}")
    log(f"Initial batch loss: {float(initial_loss):.4f}")

    def estimate_loss() -> dict:
        out = {}
        for split in ["train", "val"]:
            losses = []
            for _ in range(config.eval_iters):
                batch_x, batch_y = dataset.get_batch(split)
                loss = language_model_loss(model, batch_x, batch_y)
                mx.eval(loss)
                losses.append(float(loss))
            out[split] = sum(losses) / len(losses)
        return out

    def generate_sample(step: int, prefix: str = "sample") -> tuple:
        ids = generate_ids(
            model,
            tokenizer.encode(config.sample_prompt),
            max_new_tokens=config.sample_tokens,
            strategy="sample",
            temperature=config.sample_temperature,
            top_k=config.sample_top_k,
        )
        text = tokenizer.decode(ids)
        path = save_sample_text(step, text, prefix=prefix)
        return path, text

    if args.generate_only:
        sample_path, text = generate_sample(start_step, prefix="generate_only")
        final_text_path.write_text(text, encoding="utf-8")
        log(f"Generate-only sample saved: {sample_path}")
        log(f"Final generated text saved: {final_text_path}")
        return

    history = []
    train_start = time.perf_counter()

    first_eval = estimate_loss()
    best_val_loss = first_eval["val"]
    history.append(
        {
            "step": start_step,
            "train_loss": first_eval["train"],
            "val_loss": first_eval["val"],
            "elapsed_sec": 0.0,
        }
    )
    write_json(history_path, history)
    plot_loss_curve(history)
    initial_sample_path, _ = generate_sample(start_step)
    initial_checkpoint = save_checkpoint(
        model,
        optimizer,
        start_step,
        {"train_loss": first_eval["train"], "val_loss": first_eval["val"], "best_val_loss": best_val_loss},
        config_path,
        is_best=True,
    )
    log(
        f"eval step={start_step:04d} train_loss={first_eval['train']:.4f} "
        f"val_loss={first_eval['val']:.4f} sample={initial_sample_path.name} "
        f"checkpoint={initial_checkpoint['model_path']}"
    )

    for step in range(start_step, config.max_iters):
        batch_x, batch_y = dataset.get_batch("train")
        loss, grads = value_and_grad_fn(model, batch_x, batch_y)
        optimizer.update(model, grads)
        mx.eval(loss, model.parameters(), optimizer.state)

        should_eval = step % config.eval_interval == 0 or step == config.max_iters - 1
        if step == start_step and start_step == 0:
            should_eval = step == config.max_iters - 1
        if not should_eval:
            continue

        losses = estimate_loss()
        elapsed = time.perf_counter() - train_start
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
        write_json(history_path, history)
        plot_loss_curve(history)

        sample_path = None
        if step % config.sample_interval == 0 or step == config.max_iters - 1:
            sample_path, _ = generate_sample(step)

        checkpoint_meta = None
        should_checkpoint = (
            step % config.checkpoint_interval == 0
            or step == config.max_iters - 1
            or is_best
        )
        if should_checkpoint:
            checkpoint_meta = save_checkpoint(
                model,
                optimizer,
                step,
                {
                    "train_loss": losses["train"],
                    "val_loss": losses["val"],
                    "best_val_loss": best_val_loss,
                },
                config_path,
                is_best=is_best,
            )

        line = (
            f"eval step={step:04d} train_loss={losses['train']:.4f} "
            f"val_loss={losses['val']:.4f} elapsed={elapsed:.1f}s"
        )
        if sample_path is not None:
            line += f" sample={sample_path.name}"
        if checkpoint_meta is not None:
            line += f" checkpoint={checkpoint_meta['model_path']}"
        log(line)

    best = load_checkpoint(model, optimizer=None, kind="best")
    final_ids = generate_ids(
        model,
        tokenizer.encode(config.sample_prompt),
        max_new_tokens=280,
        strategy="sample",
        temperature=config.sample_temperature,
        top_k=config.sample_top_k,
    )
    final_text = tokenizer.decode(final_ids)
    final_text_path.write_text(final_text, encoding="utf-8")

    total_time = time.perf_counter() - train_start
    run_config.update(
        {
            "best_checkpoint": best["model_path"],
            "best_val_loss": best_val_loss,
            "training_time_sec": round(total_time, 3),
        }
    )
    write_json(config_path, run_config)

    log(f"Training finished in {total_time:.1f}s")
    log(f"Best val loss: {best_val_loss:.4f}")
    log(f"Best checkpoint: {best['model_path']}")
    log(f"Loss curve saved: {OUTPUT_DIR / 'loss_curve.png'}")
    log(f"Final generated text saved: {final_text_path}")
    log(f"Checkpoints dir: {CHECKPOINT_DIR}")
    log(f"Samples dir: {SAMPLES_DIR}")


if __name__ == "__main__":
    main()
