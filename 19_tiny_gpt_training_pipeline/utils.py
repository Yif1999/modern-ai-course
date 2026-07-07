from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from config import CHECKPOINT_DIR, OUTPUT_DIR, SAMPLES_DIR, ensure_project_dirs


def set_seed(seed: int) -> None:
    mx.random.seed(seed)
    np.random.seed(seed)


def write_json(path: Path, payload: dict | list) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def append_log(path: Path, message: str) -> None:
    print(message)
    with path.open("a", encoding="utf-8") as f:
        f.write(message + "\n")


def plot_loss_curve(history: list[dict], output_path: Path | None = None) -> None:
    if not history:
        return

    output_path = output_path or (OUTPUT_DIR / "loss_curve.png")
    steps = [item["step"] for item in history]
    train_losses = [item["train_loss"] for item in history]
    val_losses = [item["val_loss"] for item in history]

    plt.figure(figsize=(8, 5))
    plt.plot(steps, train_losses, label="train loss")
    plt.plot(steps, val_losses, label="val loss")
    plt.xlabel("step")
    plt.ylabel("cross entropy loss")
    plt.title("Tiny GPT Training Pipeline Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_checkpoint(
    model,
    optimizer,
    step: int,
    metrics: dict,
    config_path: Path,
    is_best: bool = False,
) -> dict:
    ensure_project_dirs()

    model_path = CHECKPOINT_DIR / f"step_{step:06d}_model.safetensors"
    optimizer_path = CHECKPOINT_DIR / f"step_{step:06d}_optimizer.safetensors"
    meta_path = CHECKPOINT_DIR / f"step_{step:06d}_meta.json"

    model.save_weights(str(model_path))

    optimizer_saved = False
    optimizer_error = None
    try:
        flat_state = dict(tree_flatten(optimizer.state))
        if flat_state:
            mx.save_safetensors(str(optimizer_path), flat_state)
            optimizer_saved = True
    except Exception as exc:  # noqa: BLE001
        optimizer_error = str(exc)

    meta = {
        "step": int(step),
        "saved_at_unix": time.time(),
        "model_path": str(model_path),
        "optimizer_path": str(optimizer_path) if optimizer_saved else None,
        "optimizer_saved": optimizer_saved,
        "optimizer_error": optimizer_error,
        "config_path": str(config_path),
        **metrics,
    }

    write_json(meta_path, meta)
    write_json(CHECKPOINT_DIR / "latest.json", meta)
    if is_best:
        write_json(CHECKPOINT_DIR / "best.json", meta)

    return meta


def load_checkpoint(model, optimizer=None, kind: str = "latest", checkpoint_path: str | None = None) -> dict:
    if checkpoint_path:
        model.load_weights(checkpoint_path, strict=True)
        return {"model_path": checkpoint_path, "step": None, "optimizer_saved": False}

    pointer_path = CHECKPOINT_DIR / f"{kind}.json"
    if not pointer_path.exists():
        raise FileNotFoundError(f"checkpoint pointer not found: {pointer_path}")

    meta = read_json(pointer_path)
    model.load_weights(meta["model_path"], strict=True)

    if optimizer is not None and meta.get("optimizer_path"):
        loaded_state = mx.load(meta["optimizer_path"])
        optimizer.state = tree_unflatten(loaded_state)
        optimizer.init(model.trainable_parameters())

    return meta


def save_sample_text(step: int, text: str, prefix: str = "sample") -> Path:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    path = SAMPLES_DIR / f"{prefix}_step_{step:06d}.txt"
    path.write_text(text, encoding="utf-8")
    return path
