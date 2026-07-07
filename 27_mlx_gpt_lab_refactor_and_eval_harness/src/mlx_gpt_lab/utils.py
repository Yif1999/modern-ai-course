from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def resolve_path(project_dir: Path, path: str | Path | None) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    return p if p.is_absolute() else project_dir / p


def set_seed(seed: int) -> None:
    mx.random.seed(seed)
    np.random.seed(seed)


def make_run_dir(project_dir: Path, run_name: str) -> Path:
    run_id = time.strftime("%Y%m%d_%H%M%S") + f"_{run_name}"
    run_dir = project_dir / "outputs" / "runs" / run_id
    for subdir in ["samples", "checkpoints", "evals", "benchmarks"]:
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)
    return run_dir


def find_latest_run(project_dir: Path, run_name: str | None = None) -> Path:
    runs_dir = project_dir / "outputs" / "runs"
    if not runs_dir.exists():
        raise FileNotFoundError(f"没有找到 runs 目录: {runs_dir}")
    runs = [p for p in runs_dir.iterdir() if p.is_dir()]
    if run_name:
        runs = [p for p in runs if p.name.endswith(f"_{run_name}") or run_name in p.name]
    if not runs:
        raise FileNotFoundError(f"没有找到匹配的 run: {run_name or '*'}")
    return sorted(runs)[-1]


def count_parameters(params: dict[str, Any]) -> int:
    total = 0
    for _, value in tree_flatten(params):
        if hasattr(value, "shape"):
            total += int(np.prod(value.shape))
    return total


def plot_loss_curve(history: list[dict[str, Any]], output_path: Path) -> None:
    if not history:
        return
    steps = [item["step"] for item in history]
    train_losses = [item["train_loss"] for item in history]
    val_losses = [item["val_loss"] for item in history]

    plt.figure(figsize=(8, 5))
    plt.plot(steps, train_losses, marker="o", linewidth=1.5, label="train loss")
    plt.plot(steps, val_losses, marker="o", linewidth=1.5, label="val loss")
    plt.xlabel("step")
    plt.ylabel("cross entropy")
    plt.title("MLX GPT Lab Loss Curve")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def now_seconds() -> float:
    return time.perf_counter()


def elapsed_seconds(start: float) -> float:
    return time.perf_counter() - start
