from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


CURRENT_DIR = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_int_list(value: str) -> list[int]:
    out = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not out:
        raise argparse.ArgumentTypeError("expected at least one integer")
    if any(item < 1 for item in out):
        raise argparse.ArgumentTypeError("all values must be >= 1")
    return out


def tail(text: str, max_chars: int = 4000) -> str:
    return text[-max_chars:] if len(text) > max_chars else text


def run_config(config_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    cmd = [sys.executable, str(CURRENT_DIR / "scripts/train_qwen_like.py"), "--config", str(config_path)]
    proc = subprocess.run(cmd, cwd=str(CURRENT_DIR.parent), text=True, capture_output=True)
    elapsed = time.perf_counter() - started

    run_dir = None
    for line in proc.stdout.splitlines():
        if line.startswith("Run dir:"):
            run_dir = line.split("Run dir:", 1)[1].strip()

    metrics = None
    status = None
    if run_dir:
        metrics_path = Path(run_dir) / "metrics.json"
        status_path = Path(run_dir) / "status.json"
        if metrics_path.exists():
            metrics = load_json(metrics_path)
        if status_path.exists():
            status = load_json(status_path)

    return {
        "config": str(config_path),
        "returncode": proc.returncode,
        "ok": proc.returncode == 0,
        "elapsed_sec": elapsed,
        "run_dir": run_dir,
        "metrics": metrics,
        "status": status,
        "stdout_tail": tail(proc.stdout),
        "stderr_tail": tail(proc.stderr),
    }


def make_probe_config(
    base: dict[str, Any],
    *,
    batch_size: int,
    grad_accum_steps: int,
    max_iters: int,
    eval_iters: int,
    optimizer: str,
    activation_checkpointing: str,
    memory_limit_gb: float | None,
) -> dict[str, Any]:
    cfg = dict(base)
    base_name = str(base.get("run_name", "capacity_probe"))
    ckpt_suffix = activation_checkpointing
    if ckpt_suffix == "inherit":
        ckpt_suffix = "ckpt_inherit"
    cfg["run_name"] = f"{base_name}_capacity_bs{batch_size}_ga{grad_accum_steps}_{optimizer}_{ckpt_suffix}"
    cfg["batch_size"] = batch_size
    cfg["gradient_accumulation_steps"] = grad_accum_steps
    cfg["max_iters"] = max_iters
    cfg["eval_interval"] = 1
    cfg["eval_iters"] = eval_iters
    cfg["sample_interval"] = 999999
    cfg["checkpoint_interval"] = 999999
    cfg["generate_samples"] = False
    cfg["save_checkpoints"] = False
    if optimizer != "inherit":
        cfg["optimizer"] = optimizer
    if activation_checkpointing == "on":
        cfg["activation_checkpointing"] = True
    elif activation_checkpointing == "off":
        cfg["activation_checkpointing"] = False
    if memory_limit_gb is not None:
        cfg["memory_limit_gb"] = memory_limit_gb
    return cfg


def write_markdown_summary(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Training Capacity Probe",
        "",
        f"- base_config: `{payload['base_config']}`",
        f"- generated_at: `{payload['generated_at']}`",
        "",
        "| batch_size | grad_accum | effective_batch | ok | tok/s | peak memory GB | optimizer | ckpt | run |",
        "|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in payload["results"]:
        metrics = row.get("metrics") or {}
        status = row.get("status") or {}
        perf = status.get("performance") or {}
        run = Path(row["run_dir"]).name if row.get("run_dir") else ""
        lines.append(
            f"| {row['batch_size']} | {row['gradient_accumulation_steps']} | "
            f"{row['effective_batch_size']} | {row['ok']} | "
            f"{metrics.get('tokens_per_second', '')} | {perf.get('mlx_peak_memory_gb', '')} | "
            f"{row['optimizer']} | {row['activation_checkpointing']} | {run} |"
        )

    best = payload.get("best_passing")
    lines.extend(["", "## Best Passing", ""])
    if best:
        lines.append(
            f"- batch_size={best['batch_size']}, grad_accum={best['gradient_accumulation_steps']}, "
            f"effective_batch={best['effective_batch_size']}"
        )
    else:
        lines.append("- no passing run")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe lesson 30 training batch-size capacity with short training runs.")
    parser.add_argument("--config", default=str(CURRENT_DIR / "configs/lab_probe_0p35b_context1024.json"))
    parser.add_argument("--batch-sizes", type=parse_int_list, default=parse_int_list("1,2,4"))
    parser.add_argument("--gradient-accumulation-steps", type=parse_int_list, default=parse_int_list("1"))
    parser.add_argument("--optimizer", choices=["inherit", "adamw", "adafactor", "lion"], default="inherit")
    parser.add_argument("--activation-checkpointing", choices=["inherit", "on", "off"], default="inherit")
    parser.add_argument("--memory-limit-gb", type=float, default=None)
    parser.add_argument("--max-iters", type=int, default=2)
    parser.add_argument("--eval-iters", type=int, default=1)
    parser.add_argument("--continue-after-fail", action="store_true")
    args = parser.parse_args()

    if args.max_iters < 1:
        raise ValueError("--max-iters must be >= 1")
    if args.eval_iters < 1:
        raise ValueError("--eval-iters must be >= 1")

    base_config_path = Path(args.config)
    base = load_json(base_config_path)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    config_dir = CURRENT_DIR / "outputs" / "probes" / "capacity_configs" / timestamp

    results = []
    best_passing = None
    stop = False
    for grad_accum_steps in args.gradient_accumulation_steps:
        for batch_size in args.batch_sizes:
            probe_cfg = make_probe_config(
                base,
                batch_size=batch_size,
                grad_accum_steps=grad_accum_steps,
                max_iters=args.max_iters,
                eval_iters=args.eval_iters,
                optimizer=args.optimizer,
                activation_checkpointing=args.activation_checkpointing,
                memory_limit_gb=args.memory_limit_gb,
            )
            config_path = config_dir / f"bs{batch_size}_ga{grad_accum_steps}.json"
            write_json(config_path, probe_cfg)

            print(f"Running batch_size={batch_size} grad_accum={grad_accum_steps}: {config_path}", flush=True)
            row = run_config(config_path)
            actual_optimizer = (row.get("metrics") or {}).get("optimizer", probe_cfg.get("optimizer", "adamw"))
            actual_activation_checkpointing = bool(probe_cfg.get("activation_checkpointing", False))
            row.update(
                {
                    "batch_size": batch_size,
                    "gradient_accumulation_steps": grad_accum_steps,
                    "effective_batch_size": batch_size * grad_accum_steps,
                    "optimizer": actual_optimizer,
                    "activation_checkpointing": actual_activation_checkpointing,
                }
            )
            results.append(row)
            if row["ok"]:
                best_passing = row
            elif not args.continue_after_fail:
                stop = True
                break
        if stop:
            break

    payload = {
        "base_config": str(base_config_path),
        "generated_at": timestamp,
        "config_dir": str(config_dir),
        "batch_sizes": args.batch_sizes,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "optimizer": args.optimizer,
        "activation_checkpointing": args.activation_checkpointing,
        "results": results,
        "best_passing": best_passing,
    }
    output_json = CURRENT_DIR / "outputs" / "probes" / f"training_capacity_{timestamp}.json"
    output_md = CURRENT_DIR / "outputs" / "reports" / f"training_capacity_{timestamp}.md"
    write_json(output_json, payload)
    write_markdown_summary(output_md, payload)
    print("Wrote:", output_json)
    print("Wrote:", output_md)
    if best_passing:
        print(
            "Best passing:",
            f"batch_size={best_passing['batch_size']}",
            f"grad_accum={best_passing['gradient_accumulation_steps']}",
            f"effective_batch={best_passing['effective_batch_size']}",
        )
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
