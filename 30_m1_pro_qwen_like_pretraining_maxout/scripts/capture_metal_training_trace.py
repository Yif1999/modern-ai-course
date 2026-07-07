from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

CURRENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CURRENT_DIR / "scripts"))

from benchmark_training_throughput import build_model_and_sampler, train_step  # noqa: E402
from train_qwen_like import lm_loss, load_json  # noqa: E402


TRACE_DIR = CURRENT_DIR / "outputs" / "metal_traces"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture a short Metal GPU trace for the current training step.")
    parser.add_argument(
        "--config",
        default=str(CURRENT_DIR / "configs" / "base_0p1b_context1024_adamw_100m_vocab16k_layers18.json"),
    )
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--capture-steps", type=int, default=2)
    parser.add_argument("--output-name", default=None)
    args = parser.parse_args()

    if os.environ.get("MTL_CAPTURE_ENABLED") != "1":
        raise RuntimeError("Metal capture requires launching Python with MTL_CAPTURE_ENABLED=1")

    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_json(Path(args.config))
    metadata, model_cfg, model, optimizer, sampler = build_model_and_sampler(cfg)

    raw_value_and_grad = nn.value_and_grad(model, lm_loss)
    compile_value_and_grad = bool(cfg.get("compile_value_and_grad", False))
    if compile_value_and_grad:
        value_and_grad = mx.compile(lambda x, y: raw_value_and_grad(model, x, y))
    else:
        value_and_grad = raw_value_and_grad

    grad_accum = int(cfg.get("gradient_accumulation_steps", cfg.get("grad_accum_steps", 1)))
    tokens_per_step = int(cfg["batch_size"]) * int(cfg["block_size"]) * grad_accum

    warmup_losses = []
    for _ in range(args.warmup_steps):
        warmup_losses.append(
            train_step(
                model,
                optimizer,
                value_and_grad,
                sampler,
                grad_accum,
                compiled_vg=compile_value_and_grad,
            )
        )

    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    output_name = args.output_name or f"training_step_{stamp}.gputrace"
    if not output_name.endswith(".gputrace"):
        output_name += ".gputrace"
    trace_path = TRACE_DIR / output_name

    started = time.perf_counter()
    mx.metal.start_capture(str(trace_path))
    capture_losses = []
    try:
        for _ in range(args.capture_steps):
            capture_losses.append(
                train_step(
                    model,
                    optimizer,
                    value_and_grad,
                    sampler,
                    grad_accum,
                    compiled_vg=compile_value_and_grad,
                )
            )
    finally:
        mx.metal.stop_capture()
    elapsed = time.perf_counter() - started

    memory = {}
    for key, fn_name in [
        ("active_memory_gb", "get_active_memory"),
        ("peak_memory_gb", "get_peak_memory"),
        ("cache_memory_gb", "get_cache_memory"),
    ]:
        fn = getattr(mx, fn_name, None)
        memory[key] = None if fn is None else float(fn()) / 1024**3

    report = {
        "trace_path": str(trace_path),
        "config": str(Path(args.config).resolve()),
        "model": {
            "vocab_size": int(metadata["vocab_size"]),
            "parameter_count": cfg.get("parameter_count"),
            "n_embd": int(cfg["n_embd"]),
            "num_layers": int(cfg["num_layers"]),
            "num_q_heads": int(cfg["num_q_heads"]),
            "num_kv_heads": int(cfg["num_kv_heads"]),
            "rope_impl": cfg.get("rope_impl", "manual"),
            "rms_norm_impl": cfg.get("rms_norm_impl", "manual"),
            "activation_checkpointing": bool(cfg.get("activation_checkpointing", False)),
            "compile_value_and_grad": compile_value_and_grad,
        },
        "warmup_steps": args.warmup_steps,
        "capture_steps": args.capture_steps,
        "tokens_per_step": tokens_per_step,
        "captured_tokens": tokens_per_step * args.capture_steps,
        "elapsed_sec": elapsed,
        "tokens_per_second_during_capture": (tokens_per_step * args.capture_steps) / max(elapsed, 1e-9),
        "warmup_final_loss": warmup_losses[-1] if warmup_losses else None,
        "capture_final_loss": capture_losses[-1] if capture_losses else None,
        "memory": memory,
        "notes": [
            "Open the .gputrace with Xcode Instruments / Metal debugger.",
            "Warmup steps run before capture so the trace focuses on steady-state training kernels.",
        ],
    }
    report_path = trace_path.with_suffix(".json")
    write_json(report_path, report)
    print("trace:", trace_path)
    print("report:", report_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
