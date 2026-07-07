from __future__ import annotations

import shutil
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from .checkpoint import save_model_checkpoint
from .dataset import BatchSampler, load_prepared_tokenizer, prepare_data_from_config
from .generate import generate_text
from .model_factory import create_model
from .utils import (
    append_jsonl,
    count_parameters,
    elapsed_seconds,
    load_json,
    make_run_dir,
    plot_loss_curve,
    set_seed,
    write_json,
)


def language_model_loss(model, idx, targets):
    logits = model(idx)
    batch, seq_len, vocab_size = logits.shape
    return nn.losses.cross_entropy(
        logits.reshape(batch * seq_len, vocab_size),
        targets.reshape(batch * seq_len),
        reduction="mean",
    )


def estimate_loss(model, sampler: BatchSampler, eval_iters: int) -> dict[str, float]:
    losses = {}
    for split in ["train", "val"]:
        split_losses = []
        for _ in range(eval_iters):
            x, y = sampler.get_batch(split)
            loss = language_model_loss(model, x, y)
            mx.eval(loss)
            split_losses.append(float(loss))
        losses[split] = sum(split_losses) / len(split_losses)
    return losses


def train_from_config(config_path: Path, project_dir: Path) -> Path:
    config = load_json(config_path)
    set_seed(int(config.get("seed", 0)))
    prepared = prepare_data_from_config(config, project_dir)

    config["vocab_size"] = prepared.vocab_size
    config["train_tokens"] = prepared.train_tokens
    config["val_tokens"] = prepared.val_tokens
    config["config_source"] = str(config_path)

    run_dir = make_run_dir(project_dir, config["run_name"])
    shutil.copy2(config_path, run_dir / "original_config.json")
    write_json(run_dir / "config.json", config)

    sampler = BatchSampler(
        prepared.train_tokens_path,
        prepared.val_tokens_path,
        block_size=int(config["block_size"]),
        batch_size=int(config["batch_size"]),
        seed=int(config.get("seed", 0)),
    )
    tokenizer = load_prepared_tokenizer(prepared.tokenizer_path, config["tokenizer_type"])
    model = create_model(config, prepared.vocab_size)
    optimizer = optim.Adam(learning_rate=float(config["learning_rate"]))
    value_and_grad = nn.value_and_grad(model, language_model_loss)

    x0, _ = sampler.get_batch("train")
    shape_info = model.inspect_shapes(x0)
    param_count = count_parameters(model.parameters())
    config["parameter_count"] = param_count
    config["shape_info"] = shape_info
    write_json(run_dir / "config.json", config)

    max_iters = int(config["max_iters"])
    eval_interval = int(config.get("eval_interval", 100))
    sample_interval = int(config.get("sample_interval", eval_interval))
    checkpoint_interval = int(config.get("checkpoint_interval", eval_interval))
    eval_iters = int(config.get("eval_iters", 4))
    history: list[dict] = []
    best_val = float("inf")
    best_step = 0
    train_start = time.perf_counter()

    for step in range(max_iters):
        x, y = sampler.get_batch("train")
        loss, grads = value_and_grad(model, x, y)
        optimizer.update(model, grads)
        mx.eval(loss, model.parameters(), optimizer.state)

        should_eval = step % eval_interval == 0 or step == max_iters - 1
        if should_eval:
            losses = estimate_loss(model, sampler, eval_iters)
            tokens_seen = (step + 1) * int(config["batch_size"]) * int(config["block_size"])
            row = {
                "step": step,
                "train_loss": losses["train"],
                "val_loss": losses["val"],
                "tokens_seen": tokens_seen,
                "learning_rate": float(config["learning_rate"]),
                "elapsed_sec": elapsed_seconds(train_start),
            }
            history.append(row)
            append_jsonl(run_dir / "training_log.jsonl", row)
            print(
                f"step={step:04d} train_loss={row['train_loss']:.4f} "
                f"val_loss={row['val_loss']:.4f} tokens_seen={tokens_seen}"
            )
            if losses["val"] < best_val:
                best_val = losses["val"]
                best_step = step
                save_model_checkpoint(model, run_dir, step, {"val_loss": best_val}, tag="best_val")

        if step % sample_interval == 0 or step == max_iters - 1:
            sample = generate_text(model, tokenizer, config.get("prompt", "人工智能"), config)
            (run_dir / "samples" / f"sample_step_{step:06d}.txt").write_text(sample, encoding="utf-8")

        if step % checkpoint_interval == 0 or step == max_iters - 1:
            save_model_checkpoint(model, run_dir, step, {"train_loss": float(loss)}, tag="latest")

    total_elapsed = elapsed_seconds(train_start)
    plot_loss_curve(history, run_dir / "loss_curve.png")
    final_text = generate_text(model, tokenizer, config.get("prompt", "人工智能"), config)
    (run_dir / "final_generated_text.txt").write_text(final_text, encoding="utf-8")
    save_model_checkpoint(model, run_dir, max_iters - 1, {"best_val_loss": best_val}, tag="final")

    metrics = {
        "run_name": config["run_name"],
        "model_type": config["model_type"],
        "tokenizer_type": config["tokenizer_type"],
        "parameter_count": param_count,
        "max_iters": max_iters,
        "tokens_seen": max_iters * int(config["batch_size"]) * int(config["block_size"]),
        "elapsed_sec": total_elapsed,
        "tokens_per_second": (max_iters * int(config["batch_size"]) * int(config["block_size"])) / max(total_elapsed, 1e-9),
        "final_train_loss": history[-1]["train_loss"] if history else None,
        "final_val_loss": history[-1]["val_loss"] if history else None,
        "best_val_loss": best_val,
        "best_step": best_step,
        "run_dir": str(run_dir),
    }
    write_json(run_dir / "metrics.json", metrics)
    return run_dir
