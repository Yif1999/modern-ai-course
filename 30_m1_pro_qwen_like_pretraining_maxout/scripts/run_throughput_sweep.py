from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


CURRENT_DIR = Path(__file__).resolve().parents[1]
BASE_CONFIG = CURRENT_DIR / "configs" / "probe_0p1b_context1024_adamw_bs32_vocab16k_layers18.json"
OUT_DIR = CURRENT_DIR / "outputs" / "throughput_sweep"
CONFIG_DIR = OUT_DIR / "configs"
RESULT_DIR = OUT_DIR / "results"
BENCH_SCRIPT = CURRENT_DIR / "scripts" / "benchmark_training_throughput.py"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def variants() -> list[dict[str, Any]]:
    return [
        {
            "name": "baseline_ckpt_on_bs32_manual",
            "overrides": {"batch_size": 32, "activation_checkpointing": True},
        },
        {
            "name": "fastsync_ckpt_on_bs32_manual",
            "overrides": {"batch_size": 32, "activation_checkpointing": True},
            "env": {"MLX_METAL_FAST_SYNCH": "1"},
        },
        {
            "name": "ops_fast_ckpt_on_bs32",
            "overrides": {
                "batch_size": 32,
                "activation_checkpointing": True,
                "rope_impl": "fast",
                "rms_norm_impl": "fast",
            },
        },
        {
            "name": "ops_nn_ckpt_on_bs32",
            "overrides": {
                "batch_size": 32,
                "activation_checkpointing": True,
                "rope_impl": "nn",
                "rms_norm_impl": "nn",
            },
        },
        {
            "name": "compile_vg_ckpt_on_bs32_manual",
            "overrides": {
                "batch_size": 32,
                "activation_checkpointing": True,
                "compile_value_and_grad": True,
            },
            "warmup_steps": 4,
        },
        {
            "name": "compile_vg_ops_fast_ckpt_on_bs32",
            "overrides": {
                "batch_size": 32,
                "activation_checkpointing": True,
                "compile_value_and_grad": True,
                "rope_impl": "fast",
                "rms_norm_impl": "fast",
            },
            "warmup_steps": 4,
        },
    ]


def write_summary(rows: list[dict[str, Any]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "throughput_sweep_summary.csv"
    fields = [
        "name",
        "ok",
        "tokens_per_second",
        "avg_step_time_ms",
        "peak_memory_gb",
        "batch_size",
        "activation_checkpointing",
        "compile_value_and_grad",
        "rope_impl",
        "rms_norm_impl",
        "fast_synch",
        "parameter_count",
        "error",
        "result_path",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})

    md_path = OUT_DIR / "throughput_sweep_summary.md"
    sorted_rows = sorted(rows, key=lambda r: (not r.get("ok", False), -(r.get("tokens_per_second") or 0)))
    lines = [
        "# Throughput Sweep Summary",
        "",
        "| variant | ok | tok/s | step ms | peak GB | bs | ckpt | compile | rope | norm | fast sync | error |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|",
    ]
    for row in sorted_rows:
        lines.append(
            f"| `{row['name']}` | {row.get('ok')} | "
            f"{row.get('tokens_per_second') or ''} | {row.get('avg_step_time_ms') or ''} | "
            f"{row.get('peak_memory_gb') or ''} | {row.get('batch_size') or ''} | "
            f"{row.get('activation_checkpointing')} | {row.get('compile_value_and_grad')} | "
            f"{row.get('rope_impl') or ''} | {row.get('rms_norm_impl') or ''} | "
            f"{row.get('fast_synch') or ''} | {row.get('error') or ''} |"
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print("summary:", md_path)
    print("csv:", csv_path)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    base = load_json(BASE_CONFIG)
    rows: list[dict[str, Any]] = []

    for variant in variants():
        name = variant["name"]
        cfg = {
            **base,
            "run_name": f"throughput_{name}",
            "generate_samples": False,
            "save_checkpoints": False,
            "eval_interval": 10_000_000,
            "sample_interval": 10_000_000,
            "checkpoint_interval": 10_000_000,
            "memory_limit_gb": 24,
            **variant.get("overrides", {}),
        }
        cfg_path = CONFIG_DIR / f"{name}.json"
        result_path = RESULT_DIR / f"{name}.json"
        log_path = RESULT_DIR / f"{name}.log"
        write_json(cfg_path, cfg)
        env = os.environ.copy()
        env.update(variant.get("env", {}))
        command = [
            sys.executable,
            str(BENCH_SCRIPT),
            "--config",
            str(cfg_path),
            "--warmup-steps",
            str(variant.get("warmup_steps", 3)),
            "--measure-steps",
            str(variant.get("measure_steps", 4)),
            "--output",
            str(result_path),
        ]
        print(f"\n=== {name} ===", flush=True)
        started = time.perf_counter()
        timeout_sec = int(variant.get("timeout_sec", 600))
        try:
            proc = subprocess.run(command, env=env, text=True, capture_output=True, timeout=timeout_sec)
        except subprocess.TimeoutExpired as exc:
            log_path.write_text(
                "COMMAND: " + " ".join(command) + f"\n\nTIMEOUT after {timeout_sec}s\n\n"
                + "STDOUT:\n" + (exc.stdout or "") + "\n\nSTDERR:\n" + (exc.stderr or ""),
                encoding="utf-8",
            )
            row = {
                "name": name,
                "ok": False,
                "error": f"timeout_after_{timeout_sec}s",
                "batch_size": cfg.get("batch_size"),
                "activation_checkpointing": cfg.get("activation_checkpointing"),
                "compile_value_and_grad": cfg.get("compile_value_and_grad", False),
                "rope_impl": cfg.get("rope_impl", "manual"),
                "rms_norm_impl": cfg.get("rms_norm_impl", "manual"),
                "fast_synch": env.get("MLX_METAL_FAST_SYNCH"),
                "result_path": str(result_path),
            }
            rows.append(row)
            print(f"failed: {row['error']}", flush=True)
            write_summary(rows)
            continue
        elapsed = time.perf_counter() - started
        log_path.write_text(
            "COMMAND: " + " ".join(command) + "\n\nSTDOUT:\n" + proc.stdout + "\n\nSTDERR:\n" + proc.stderr,
            encoding="utf-8",
        )

        if proc.returncode == 0 and result_path.exists():
            result = load_json(result_path)
            row = {
                "name": name,
                "ok": True,
                "tokens_per_second": round(float(result["tokens_per_second"]), 2),
                "avg_step_time_ms": round(float(result["avg_step_time_ms"]), 2),
                "peak_memory_gb": round(float((result.get("memory") or {}).get("mlx_peak_memory_gb") or 0), 2),
                "batch_size": result.get("batch_size"),
                "activation_checkpointing": result.get("activation_checkpointing"),
                "compile_value_and_grad": result.get("compile_value_and_grad"),
                "rope_impl": result.get("rope_impl"),
                "rms_norm_impl": result.get("rms_norm_impl"),
                "fast_synch": result.get("mlx_metal_fast_synch"),
                "parameter_count": result.get("parameter_count"),
                "elapsed_sec": round(elapsed, 2),
                "result_path": str(result_path),
            }
            print(
                f"ok tok/s={row['tokens_per_second']} step_ms={row['avg_step_time_ms']} "
                f"peak={row['peak_memory_gb']}GB",
                flush=True,
            )
        else:
            err = (proc.stderr or proc.stdout or "").strip().splitlines()
            row = {
                "name": name,
                "ok": False,
                "error": err[-1] if err else f"returncode={proc.returncode}",
                "batch_size": cfg.get("batch_size"),
                "activation_checkpointing": cfg.get("activation_checkpointing"),
                "compile_value_and_grad": cfg.get("compile_value_and_grad", False),
                "rope_impl": cfg.get("rope_impl", "manual"),
                "rms_norm_impl": cfg.get("rms_norm_impl", "manual"),
                "fast_synch": env.get("MLX_METAL_FAST_SYNCH"),
                "result_path": str(result_path),
            }
            print(f"failed: {row['error']}", flush=True)
        rows.append(row)
        write_summary(rows)


if __name__ == "__main__":
    main()
