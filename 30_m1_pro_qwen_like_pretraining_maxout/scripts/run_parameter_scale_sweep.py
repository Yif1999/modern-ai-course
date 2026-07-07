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
OUT_DIR = CURRENT_DIR / "outputs" / "parameter_scale_sweep"
CONFIG_DIR = OUT_DIR / "configs"
RESULT_DIR = OUT_DIR / "results"
BENCH_SCRIPT = CURRENT_DIR / "scripts" / "benchmark_training_throughput.py"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def variants() -> list[dict[str, Any]]:
    common = {
        "metadata_path": "llm-lab/data/metadata/lab_bpe_16384_metadata.json",
        "block_size": 1024,
        "gradient_accumulation_steps": 1,
        "ffn_multiplier": 3.0,
        "rope_base": 1_000_000.0,
        "qk_norm": True,
        "weight_tying": True,
        "use_bias": False,
        "activation_checkpointing": True,
        "rope_impl": "fast",
        "rms_norm_impl": "fast",
        "compile_value_and_grad": True,
        "dtype": "bfloat16",
        "optimizer": "adamw",
        "adamw_betas": [0.9, 0.95],
        "adamw_eps": 1e-6,
        "learning_rate": 3e-4,
        "weight_decay": 0.1,
        "memory_limit_gb": 28,
        "generate_samples": False,
        "save_checkpoints": False,
        "eval_interval": 10_000_000,
        "sample_interval": 10_000_000,
        "checkpoint_interval": 10_000_000,
        "seed": 2070,
    }
    return [
        {
            "name": "0p1b_103m_bs32",
            "overrides": {
                **common,
                "batch_size": 32,
                "n_embd": 640,
                "num_layers": 18,
                "num_q_heads": 10,
                "num_kv_heads": 5,
            },
        },
        {
            "name": "0p2b_182m_bs16",
            "overrides": {
                **common,
                "batch_size": 16,
                "n_embd": 768,
                "num_layers": 24,
                "num_q_heads": 12,
                "num_kv_heads": 6,
            },
        },
        {
            "name": "0p2b_182m_bs24",
            "overrides": {
                **common,
                "batch_size": 24,
                "n_embd": 768,
                "num_layers": 24,
                "num_q_heads": 12,
                "num_kv_heads": 6,
            },
        },
        {
            "name": "0p35b_319m_bs8",
            "overrides": {
                **common,
                "batch_size": 8,
                "n_embd": 1024,
                "num_layers": 24,
                "num_q_heads": 16,
                "num_kv_heads": 8,
            },
        },
        {
            "name": "0p35b_319m_bs12",
            "overrides": {
                **common,
                "batch_size": 12,
                "n_embd": 1024,
                "num_layers": 24,
                "num_q_heads": 16,
                "num_kv_heads": 8,
            },
        },
    ]


def write_summary(rows: list[dict[str, Any]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "parameter_scale_sweep_summary.csv"
    fields = [
        "name",
        "ok",
        "parameter_count",
        "batch_size",
        "tokens_per_second",
        "tokens_per_step",
        "avg_step_time_ms",
        "peak_memory_gb",
        "tokens_per_second_per_100m_params",
        "memory_gb_per_100m_params",
        "error",
        "result_path",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})

    sorted_rows = sorted(rows, key=lambda r: (not r.get("ok", False), -(r.get("tokens_per_second") or 0)))
    md_path = OUT_DIR / "parameter_scale_sweep_summary.md"
    lines = [
        "# Parameter Scale Sweep Summary",
        "",
        "固定条件：16k lab tokenizer、block_size=1024、bf16、AdamW、activation checkpointing、fast RoPE/RMSNorm、compiled value_and_grad。",
        "",
        "| variant | ok | params | bs | tok/s | step ms | peak GB | tok/s / 100M params | GB / 100M params | error |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in sorted_rows:
        lines.append(
            f"| `{row['name']}` | {row.get('ok')} | "
            f"{row.get('parameter_count') or ''} | {row.get('batch_size') or ''} | "
            f"{row.get('tokens_per_second') or ''} | {row.get('avg_step_time_ms') or ''} | "
            f"{row.get('peak_memory_gb') or ''} | "
            f"{row.get('tokens_per_second_per_100m_params') or ''} | "
            f"{row.get('memory_gb_per_100m_params') or ''} | "
            f"{row.get('error') or ''} |"
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print("summary:", md_path)
    print("csv:", csv_path)


def row_from_result(name: str, result: dict[str, Any], result_path: Path) -> dict[str, Any]:
    params = int(result["parameter_count"])
    peak_gb = float((result.get("memory") or {}).get("mlx_peak_memory_gb") or 0)
    tok_s = float(result["tokens_per_second"])
    params_100m = params / 100_000_000
    return {
        "name": name,
        "ok": True,
        "parameter_count": params,
        "batch_size": int(result["batch_size"]),
        "tokens_per_second": round(tok_s, 2),
        "tokens_per_step": int(result["tokens_per_step"]),
        "avg_step_time_ms": round(float(result["avg_step_time_ms"]), 2),
        "peak_memory_gb": round(peak_gb, 2),
        "tokens_per_second_per_100m_params": round(tok_s / params_100m, 2),
        "memory_gb_per_100m_params": round(peak_gb / params_100m, 2),
        "result_path": str(result_path),
    }


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
            **variant["overrides"],
            "run_name": f"scale_sweep_{name}",
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
        print(f"\n=== {name} ===", flush=True)
        started = time.perf_counter()
        try:
            proc = subprocess.run(command, env=os.environ.copy(), text=True, capture_output=True, timeout=900)
        except subprocess.TimeoutExpired as exc:
            log_path.write_text(
                "COMMAND: " + " ".join(command) + "\n\nTIMEOUT after 900s\n\n"
                + "STDOUT:\n" + (exc.stdout or "") + "\n\nSTDERR:\n" + (exc.stderr or ""),
                encoding="utf-8",
            )
            row = {
                "name": name,
                "ok": False,
                "error": "timeout_after_900s",
                "batch_size": cfg.get("batch_size"),
                "result_path": str(result_path),
            }
            rows.append(row)
            write_summary(rows)
            print("failed:", row["error"], flush=True)
            continue

        elapsed = time.perf_counter() - started
        log_path.write_text(
            "COMMAND: " + " ".join(command)
            + f"\n\nELAPSED: {elapsed:.2f}s\n\nSTDOUT:\n"
            + proc.stdout
            + "\n\nSTDERR:\n"
            + proc.stderr,
            encoding="utf-8",
        )
        if proc.returncode == 0 and result_path.exists():
            result = load_json(result_path)
            row = row_from_result(name, result, result_path)
            print(
                f"ok tok/s={row['tokens_per_second']} step_ms={row['avg_step_time_ms']} "
                f"peak={row['peak_memory_gb']}GB params={row['parameter_count']}",
                flush=True,
            )
        else:
            err = (proc.stderr or proc.stdout or "").strip().splitlines()
            row = {
                "name": name,
                "ok": False,
                "error": err[-1] if err else f"returncode={proc.returncode}",
                "batch_size": cfg.get("batch_size"),
                "result_path": str(result_path),
            }
            print("failed:", row["error"], flush=True)
        rows.append(row)
        write_summary(rows)


if __name__ == "__main__":
    main()
