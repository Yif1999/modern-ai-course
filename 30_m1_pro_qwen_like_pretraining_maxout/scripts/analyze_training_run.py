from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any


CURRENT_DIR = Path(__file__).resolve().parents[1]
RUNS_DIR = CURRENT_DIR / "outputs" / "runs"
REPORTS_DIR = CURRENT_DIR / "outputs" / "reports"


def read_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def latest_run() -> Path:
    candidates = [p for p in RUNS_DIR.iterdir() if p.is_dir() and p.name != "demo_run"]
    if not candidates:
        raise FileNotFoundError(f"No runs found in {RUNS_DIR}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "暂无数据"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def seconds_to_hms(seconds: float | None) -> str:
    if seconds is None:
        return "暂无数据"
    seconds = max(0, int(seconds))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def loss_delta(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row.get(key) for row in rows if isinstance(row.get(key), (int, float))]
    if len(values) < 2:
        return None
    return values[-1] - values[0]


def checkpoint_summary(run_dir: Path) -> list[str]:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return []
    return sorted(p.name for p in ckpt_dir.iterdir() if p.is_file())


def sample_summary(run_dir: Path, max_chars: int = 240) -> list[dict[str, str]]:
    samples_dir = run_dir / "samples"
    if not samples_dir.exists():
        return []
    out = []
    for path in sorted(samples_dir.iterdir()):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        out.append({"name": path.name, "preview": text[:max_chars]})
    return out


def diagnose(rows: list[dict[str, Any]]) -> list[str]:
    if len(rows) < 2:
        return ["训练日志还不够多，暂时只能判断流程是否正常，不能判断收敛趋势。"]
    train_delta = loss_delta(rows, "train_loss")
    val_delta = loss_delta(rows, "val_loss")
    notes = []
    if train_delta is not None and val_delta is not None:
        if train_delta < 0 and val_delta < 0:
            notes.append("train loss 和 val loss 都在下降，当前更像训练还在有效推进。")
        elif train_delta < 0 and val_delta > 0:
            notes.append("train loss 下降但 val loss 上升，可能开始出现过拟合，需要观察后续样本和更多评估点。")
        elif abs(train_delta) < 0.05 and abs(val_delta) < 0.05:
            notes.append("train/val loss 变化很小，可能学习率、数据质量或模型容量需要重新检查。")
        else:
            notes.append("loss 走势暂不稳定，需要更多 eval 点判断。")
    return notes


def analyze(run_dir: Path) -> tuple[dict[str, Any], str]:
    config = read_json(run_dir / "config.json") or {}
    status = read_json(run_dir / "status.json") or {}
    metrics = read_json(run_dir / "metrics.json") or {}
    rows = read_jsonl(run_dir / "training_log.jsonl")
    metadata_path = Path(config.get("metadata_path", CURRENT_DIR / "data" / "metadata" / "lab_bpe_32768_metadata.json"))
    metadata = read_json(metadata_path) or {}
    train_tokens = metadata.get("train_tokens")
    tokens_seen = status.get("tokens_seen") or metrics.get("tokens_seen")
    epoch_equivalent = tokens_seen / train_tokens if isinstance(tokens_seen, (int, float)) and train_tokens else None
    throughput_values = [row.get("tokens_per_second") for row in rows if isinstance(row.get("tokens_per_second"), (int, float))]

    summary = {
        "run_id": run_dir.name,
        "state": status.get("state") or "unknown",
        "step": status.get("step"),
        "max_iters": status.get("max_iters") or config.get("max_iters"),
        "progress_percent": status.get("progress_percent"),
        "tokens_seen": tokens_seen,
        "epoch_equivalent": epoch_equivalent,
        "train_loss_initial": rows[0].get("train_loss") if rows else None,
        "train_loss_latest": rows[-1].get("train_loss") if rows else None,
        "val_loss_initial": rows[0].get("val_loss") if rows else None,
        "val_loss_latest": rows[-1].get("val_loss") if rows else None,
        "best_val_loss": status.get("best_val_loss") or metrics.get("best_val_loss"),
        "tokens_per_second_latest": status.get("tokens_per_second") or metrics.get("tokens_per_second"),
        "tokens_per_second_avg_eval_points": mean(throughput_values) if throughput_values else None,
        "mlx_peak_memory_gb": (status.get("performance") or {}).get("mlx_peak_memory_gb"),
        "eta_sec": status.get("eta_sec"),
        "parameter_count": config.get("parameter_count") or metrics.get("parameter_count"),
        "vocab_size": config.get("vocab_size"),
        "block_size": config.get("block_size"),
        "batch_size": config.get("batch_size"),
        "n_embd": config.get("n_embd"),
        "num_layers": config.get("num_layers"),
        "num_q_heads": config.get("num_q_heads"),
        "num_kv_heads": config.get("num_kv_heads"),
        "checkpoint_count": len(checkpoint_summary(run_dir)),
        "sample_count": len(sample_summary(run_dir)),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    samples = sample_summary(run_dir)
    checkpoints = checkpoint_summary(run_dir)
    diagnosis = diagnose(rows)

    md = [
        f"# 训练复盘报告：{run_dir.name}",
        "",
        "## 当前状态",
        "",
        f"- state: `{summary['state']}`",
        f"- step: `{fmt(summary['step'], 0)} / {fmt(summary['max_iters'], 0)}`",
        f"- progress: `{fmt(summary['progress_percent'], 2)}%`",
        f"- tokens_seen: `{fmt(summary['tokens_seen'], 0)}`",
        f"- 等效 epoch: `{fmt(summary['epoch_equivalent'], 3)}`",
        f"- ETA: `{seconds_to_hms(summary['eta_sec'])}`",
        "",
        "## 模型与训练配置",
        "",
        f"- params: `{fmt(summary['parameter_count'], 0)}`",
        f"- vocab_size: `{fmt(summary['vocab_size'], 0)}`",
        f"- block_size: `{fmt(summary['block_size'], 0)}`",
        f"- batch_size: `{fmt(summary['batch_size'], 0)}`",
        f"- n_embd: `{fmt(summary['n_embd'], 0)}`",
        f"- num_layers: `{fmt(summary['num_layers'], 0)}`",
        f"- heads: `q={fmt(summary['num_q_heads'], 0)}, kv={fmt(summary['num_kv_heads'], 0)}`",
        "",
        "## Loss 走势",
        "",
        f"- initial train loss: `{fmt(summary['train_loss_initial'])}`",
        f"- latest train loss: `{fmt(summary['train_loss_latest'])}`",
        f"- initial val loss: `{fmt(summary['val_loss_initial'])}`",
        f"- latest val loss: `{fmt(summary['val_loss_latest'])}`",
        f"- best val loss: `{fmt(summary['best_val_loss'])}`",
        "",
        "## 性能与资源",
        "",
        f"- latest tokens/sec: `{fmt(summary['tokens_per_second_latest'], 2)}`",
        f"- avg tokens/sec at eval points: `{fmt(summary['tokens_per_second_avg_eval_points'], 2)}`",
        f"- MLX peak memory: `{fmt(summary['mlx_peak_memory_gb'], 2)} GB`",
        "",
        "## 产物检查",
        "",
        f"- checkpoints: `{summary['checkpoint_count']}`",
        f"- samples: `{summary['sample_count']}`",
        f"- loss_curve.png: `{(run_dir / 'loss_curve.png').exists()}`",
        f"- final_generated_text.txt: `{(run_dir / 'final_generated_text.txt').exists()}`",
        "",
        "## 初步诊断",
        "",
    ]
    md.extend([f"- {item}" for item in diagnosis])
    md.extend(["", "## 样本预览", ""])
    if samples:
        for item in samples[-5:]:
            md.extend([f"### {item['name']}", "", "```text", item["preview"], "```", ""])
    else:
        md.append("暂无样本。")
    md.extend(["", "## Checkpoints", ""])
    md.extend([f"- `{name}`" for name in checkpoints[-12:]] or ["暂无 checkpoint。"])
    return summary, "\n".join(md)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a lesson 30 training run")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir or latest_run()
    if not run_dir.is_absolute():
        run_dir = (Path.cwd() / run_dir).resolve()
    summary, markdown = analyze(run_dir)
    output = args.output or REPORTS_DIR / f"run_analysis_{run_dir.name}.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")
    json_output = output.with_suffix(".json")
    json_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)
    print(json_output)


if __name__ == "__main__":
    main()
