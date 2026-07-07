from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from .utils import write_json


def save_model_checkpoint(model, run_dir: Path, step: int, metrics: dict, tag: str = "latest") -> Path:
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    weights_path = ckpt_dir / f"{tag}_model.safetensors"
    meta_path = ckpt_dir / f"{tag}_meta.json"
    model.save_weights(str(weights_path))
    mx.eval(model.parameters())
    meta = {
        "step": int(step),
        "tag": tag,
        "weights_path": str(weights_path),
        **metrics,
    }
    write_json(meta_path, meta)
    return weights_path


def latest_checkpoint_path(run_dir: Path) -> Path:
    final_path = run_dir / "checkpoints" / "final_model.safetensors"
    latest_path = run_dir / "checkpoints" / "latest_model.safetensors"
    best_path = run_dir / "checkpoints" / "best_val_model.safetensors"
    for path in [final_path, latest_path, best_path]:
        if path.exists():
            return path
    candidates = sorted((run_dir / "checkpoints").glob("*_model.safetensors"))
    if not candidates:
        raise FileNotFoundError(f"没有找到 checkpoint: {run_dir / 'checkpoints'}")
    return candidates[-1]


def load_model_checkpoint(model, run_dir: Path, checkpoint_path: str | Path | None = None):
    path = Path(checkpoint_path) if checkpoint_path else latest_checkpoint_path(run_dir)
    model.load_weights(str(path), strict=True)
    return path
