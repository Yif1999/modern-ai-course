from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_map

CURRENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CURRENT_DIR / "src"))
sys.path.insert(0, str(CURRENT_DIR / "scripts"))

from model_qwen_like import QwenLikeConfig, QwenLikeDenseLM  # noqa: E402
from train_qwen_like import (  # noqa: E402
    BatchSampler,
    build_optimizer,
    canonical_optimizer_name,
    count_parameters,
    dtype_from_name,
    lm_loss,
    load_json,
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_current_path(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return (CURRENT_DIR.parent / value).resolve()


def current_memory() -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for name, fn_name in [
        ("mlx_active_memory_gb", "get_active_memory"),
        ("mlx_peak_memory_gb", "get_peak_memory"),
        ("mlx_cache_memory_gb", "get_cache_memory"),
    ]:
        fn = getattr(mx, fn_name, None)
        if fn is None:
            out[name] = None
            continue
        try:
            out[name] = float(fn()) / 1024**3
        except Exception:  # noqa: BLE001
            out[name] = None
    return out


def build_model_and_sampler(cfg: dict[str, Any]):
    if cfg.get("memory_limit_gb"):
        mx.set_memory_limit(int(float(cfg["memory_limit_gb"]) * 1024**3))

    metadata_path = Path(cfg.get("metadata_path", CURRENT_DIR / "data" / "metadata" / "lab_bpe_16384_metadata.json"))
    metadata = load_json(metadata_path)
    data_dir = CURRENT_DIR / "data" / "processed"
    train_path = Path(cfg.get("train_tokens_path", metadata.get("train_tokens_path", data_dir / "train_tokens.npy")))
    val_path = Path(cfg.get("val_tokens_path", metadata.get("val_tokens_path", data_dir / "val_tokens.npy")))

    model_cfg = QwenLikeConfig(
        vocab_size=int(metadata["vocab_size"]),
        block_size=int(cfg["block_size"]),
        n_embd=int(cfg["n_embd"]),
        num_layers=int(cfg["num_layers"]),
        num_q_heads=int(cfg["num_q_heads"]),
        num_kv_heads=int(cfg["num_kv_heads"]),
        ffn_multiplier=float(cfg.get("ffn_multiplier", 3.0)),
        rope_base=float(cfg.get("rope_base", 1_000_000.0)),
        qk_norm=bool(cfg.get("qk_norm", True)),
        weight_tying=bool(cfg.get("weight_tying", True)),
        use_bias=bool(cfg.get("use_bias", False)),
        activation_checkpointing=bool(cfg.get("activation_checkpointing", False)),
        rope_impl=str(cfg.get("rope_impl", "manual")),
        rms_norm_impl=str(cfg.get("rms_norm_impl", "manual")),
    )
    model = QwenLikeDenseLM(model_cfg)
    dtype_name = cfg.get("dtype", "bfloat16")
    if dtype_name != "float32":
        model.set_dtype(dtype_from_name(dtype_name))

    optimizer = build_optimizer(cfg)
    sampler = BatchSampler(
        train_path,
        val_path,
        block_size=int(cfg["block_size"]),
        batch_size=int(cfg["batch_size"]),
        seed=int(cfg.get("seed", 2030)),
    )
    mx.eval(model.parameters(), optimizer.state)
    return metadata, model_cfg, model, optimizer, sampler


def train_step(model, optimizer, value_and_grad, sampler, gradient_accumulation_steps: int, *, compiled_vg: bool):
    accumulated_grads = None
    losses = []
    for _ in range(gradient_accumulation_steps):
        x, y = sampler.get_batch("train")
        if compiled_vg:
            loss, grads = value_and_grad(x, y)
        else:
            loss, grads = value_and_grad(model, x, y)
        scaled_grads = tree_map(lambda g: g / gradient_accumulation_steps, grads)
        accumulated_grads = (
            scaled_grads
            if accumulated_grads is None
            else tree_map(lambda total, grad: total + grad, accumulated_grads, scaled_grads)
        )
        mx.eval(loss, accumulated_grads)
        losses.append(float(loss))
    optimizer.update(model, accumulated_grads)
    mx.eval(model.parameters(), optimizer.state)
    return sum(losses) / max(1, len(losses))


def benchmark(cfg: dict[str, Any], *, warmup_steps: int, measure_steps: int) -> dict[str, Any]:
    np.random.seed(int(cfg.get("seed", 2030)))
    mx.random.seed(int(cfg.get("seed", 2030)))
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()

    metadata, model_cfg, model, optimizer, sampler = build_model_and_sampler(cfg)
    raw_value_and_grad = nn.value_and_grad(model, lm_loss)
    compile_value_and_grad = bool(cfg.get("compile_value_and_grad", False))
    if compile_value_and_grad:
        value_and_grad = mx.compile(lambda x, y: raw_value_and_grad(model, x, y))
    else:
        value_and_grad = raw_value_and_grad
    gradient_accumulation_steps = int(cfg.get("gradient_accumulation_steps", cfg.get("grad_accum_steps", 1)))
    tokens_per_step = int(cfg["batch_size"]) * int(cfg["block_size"]) * gradient_accumulation_steps

    warmup_losses = []
    for _ in range(warmup_steps):
        warmup_losses.append(
            train_step(
                model,
                optimizer,
                value_and_grad,
                sampler,
                gradient_accumulation_steps,
                compiled_vg=compile_value_and_grad,
            )
        )

    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()

    step_times = []
    measured_losses = []
    start = time.perf_counter()
    for _ in range(measure_steps):
        step_start = time.perf_counter()
        measured_losses.append(
            train_step(
                model,
                optimizer,
                value_and_grad,
                sampler,
                gradient_accumulation_steps,
                compiled_vg=compile_value_and_grad,
            )
        )
        step_times.append(time.perf_counter() - step_start)
    elapsed = time.perf_counter() - start
    tokens = tokens_per_step * measure_steps
    peak = current_memory()
    final_loss = measured_losses[-1] if measured_losses else None
    if final_loss is not None and not math.isfinite(final_loss):
        raise RuntimeError(f"Non-finite final loss: {final_loss}")

    return {
        "run_name": cfg.get("run_name"),
        "vocab_size": int(metadata["vocab_size"]),
        "parameter_count": count_parameters(model.parameters()),
        "block_size": int(cfg["block_size"]),
        "batch_size": int(cfg["batch_size"]),
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "tokens_per_step": tokens_per_step,
        "warmup_steps": warmup_steps,
        "measure_steps": measure_steps,
        "measured_tokens": tokens,
        "elapsed_sec": elapsed,
        "tokens_per_second": tokens / max(elapsed, 1e-9),
        "avg_step_time_ms": 1000.0 * sum(step_times) / max(len(step_times), 1),
        "min_step_time_ms": 1000.0 * min(step_times) if step_times else None,
        "max_step_time_ms": 1000.0 * max(step_times) if step_times else None,
        "warmup_final_loss": warmup_losses[-1] if warmup_losses else None,
        "measured_final_loss": final_loss,
        "activation_checkpointing": model_cfg.activation_checkpointing,
        "compile_value_and_grad": compile_value_and_grad,
        "rope_impl": model_cfg.rope_impl,
        "rms_norm_impl": model_cfg.rms_norm_impl,
        "optimizer": canonical_optimizer_name(cfg),
        "dtype": cfg.get("dtype", "bfloat16"),
        "mlx_metal_fast_synch": os.environ.get("MLX_METAL_FAST_SYNCH"),
        "memory": peak,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Pure training throughput benchmark for lesson 30.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--measure-steps", type=int, default=8)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = load_json(Path(args.config))
    result = benchmark(cfg, warmup_steps=args.warmup_steps, measure_steps=args.measure_steps)
    if args.output:
        write_json(Path(args.output), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
