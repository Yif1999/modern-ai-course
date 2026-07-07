from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


CURRENT_DIR = Path(__file__).resolve().parents[1]
BASE_CONFIG = CURRENT_DIR / "configs" / "probe_0p1b_context1024_adamw_bs32_vocab16k_layers18.json"
OUT_DIR = CURRENT_DIR / "outputs" / "batch_size_sweep_0p1b"
CONFIG_DIR = OUT_DIR / "configs"
RESULT_DIR = OUT_DIR / "results"
BENCH_SCRIPT = CURRENT_DIR / "scripts" / "benchmark_training_throughput.py"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def batch_sizes() -> list[int]:
    return [16, 24, 32, 40, 48, 56]


def write_summary(rows: list[dict[str, Any]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "batch_size_sweep_0p1b_summary.csv"
    fields = [
        "batch_size",
        "ok",
        "tokens_per_second",
        "tokens_per_step",
        "avg_step_time_ms",
        "peak_memory_gb",
        "active_memory_gb",
        "cache_memory_gb",
        "error",
        "result_path",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})

    md_path = OUT_DIR / "batch_size_sweep_0p1b_summary.md"
    sorted_rows = sorted(rows, key=lambda r: (not r.get("ok", False), -(r.get("tokens_per_second") or 0)))
    lines = [
        "# 0.1B Batch Size Sweep",
        "",
        "固定条件：103M 参数、16k lab tokenizer、block_size=1024、bf16、AdamW、activation checkpointing、fast RoPE/RMSNorm、compiled value_and_grad。",
        "",
        "| batch | ok | tok/s | tokens/step | step ms | peak GB | active GB | cache GB | error |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in sorted_rows:
        lines.append(
            f"| {row.get('batch_size') or ''} | {row.get('ok')} | "
            f"{row.get('tokens_per_second') or ''} | {row.get('tokens_per_step') or ''} | "
            f"{row.get('avg_step_time_ms') or ''} | {row.get('peak_memory_gb') or ''} | "
            f"{row.get('active_memory_gb') or ''} | {row.get('cache_memory_gb') or ''} | "
            f"{row.get('error') or ''} |"
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

    for batch_size in batch_sizes():
        name = f"bs{batch_size}"
        cfg = {
            **base,
            "run_name": f"0p1b_batch_sweep_{name}",
            "metadata_path": "llm-lab/data/metadata/lab_bpe_16384_metadata.json",
            "batch_size": batch_size,
            "activation_checkpointing": True,
            "rope_impl": "fast",
            "rms_norm_impl": "fast",
            "compile_value_and_grad": True,
            "generate_samples": False,
            "save_checkpoints": False,
            "eval_interval": 10_000_000,
            "sample_interval": 10_000_000,
            "checkpoint_interval": 10_000_000,
            "memory_limit_gb": 28,
        }
        cfg_path = CONFIG_DIR / f"{name}.json"
        result_path = RESULT_DIR / f"{name}.json"
        log_path = RESULT_DIR / f"{name}.log"
        write_json(cfg_path, cfg)

        command = [
            sys.executable,
            str(BENCH_SCRIPT),
            "--config",
            str(cfg_path),
            "--warmup-steps",
            "4",
            "--measure-steps",
            "4",
            "--output",
            str(result_path),
        ]
        print(f"\n=== batch_size={batch_size} ===", flush=True)
        try:
            proc = subprocess.run(command, env=os.environ.copy(), text=True, capture_output=True, timeout=900)
        except subprocess.TimeoutExpired as exc:
            log_path.write_text(
                "COMMAND: " + " ".join(command) + "\n\nTIMEOUT after 900s\n\n"
                + "STDOUT:\n" + (exc.stdout or "") + "\n\nSTDERR:\n" + (exc.stderr or ""),
                encoding="utf-8",
            )
            row = {
                "batch_size": batch_size,
                "ok": False,
                "error": "timeout_after_900s",
                "result_path": str(result_path),
            }
            rows.append(row)
            write_summary(rows)
            print("failed:", row["error"], flush=True)
            continue

        log_path.write_text(
            "COMMAND: " + " ".join(command) + "\n\nSTDOUT:\n" + proc.stdout + "\n\nSTDERR:\n" + proc.stderr,
            encoding="utf-8",
        )
        if proc.returncode == 0 and result_path.exists():
            result = load_json(result_path)
            memory = result.get("memory") or {}
            row = {
                "batch_size": batch_size,
                "ok": True,
                "tokens_per_second": round(float(result["tokens_per_second"]), 2),
                "tokens_per_step": int(result["tokens_per_step"]),
                "avg_step_time_ms": round(float(result["avg_step_time_ms"]), 2),
                "peak_memory_gb": round(float(memory.get("mlx_peak_memory_gb") or 0), 2),
                "active_memory_gb": round(float(memory.get("mlx_active_memory_gb") or 0), 2),
                "cache_memory_gb": round(float(memory.get("mlx_cache_memory_gb") or 0), 2),
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
                "batch_size": batch_size,
                "ok": False,
                "error": err[-1] if err else f"returncode={proc.returncode}",
                "result_path": str(result_path),
            }
            print("failed:", row["error"], flush=True)
        rows.append(row)
        write_summary(rows)


if __name__ == "__main__":
    main()
