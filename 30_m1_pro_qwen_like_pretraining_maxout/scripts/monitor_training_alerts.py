from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any


def utcish_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_load_error": str(exc)}


def read_jsonl(path: Path, limit: int = 200) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        # These logs are append-only and small enough for now. Keep this simple
        # and dependency-free; if logs grow huge, this can be replaced by a
        # backwards tail reader without changing the alert schema.
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []
    for line in lines[-limit:]:
        try:
            row = json.loads(line)
        except Exception:
            continue
        rows.append(row)
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except Exception:
        return None
    return result if math.isfinite(result) else None


def mtime_age_seconds(path: Path) -> float | None:
    if not path.exists():
        return None
    return max(0.0, time.time() - path.stat().st_mtime)


def alert(key: str, severity: str, message: str, **details: Any) -> dict[str, Any]:
    return {
        "key": key,
        "severity": severity,
        "message": message,
        "details": details,
    }


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def evaluate_alerts(run_dir: Path, args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    status = load_json(run_dir / "status.json")
    config = load_json(run_dir / "config.json")
    heartbeat_rows = read_jsonl(run_dir / "heartbeat_log.jsonl", limit=240)
    eval_rows = read_jsonl(run_dir / "training_log.jsonl", limit=80)
    latest_state = load_json(run_dir / "checkpoints" / "latest_state.json")
    best_state = load_json(run_dir / "checkpoints" / "best_val_state.json")

    alerts: list[dict[str, Any]] = []
    status_age = mtime_age_seconds(run_dir / "status.json")
    heartbeat_age = mtime_age_seconds(run_dir / "heartbeat_log.jsonl")
    training_log_age = mtime_age_seconds(run_dir / "training_log.jsonl")
    checkpoint_age = mtime_age_seconds(run_dir / "checkpoints" / "latest_state.json")

    state = status.get("state")
    step = int(status.get("step", -1)) if isinstance(status.get("step"), int) else -1
    local_step = int(status.get("local_step", -1)) if isinstance(status.get("local_step"), int) else -1
    eval_interval = int(config.get("eval_interval", 0) or 0)
    checkpoint_interval = int(config.get("checkpoint_interval", 0) or 0)
    heartbeat_interval = int(config.get("heartbeat_interval", 0) or 0)
    tokens_per_second = finite_float(status.get("tokens_per_second"))
    latest_train_loss = finite_float(status.get("last_train_loss", status.get("train_loss")))
    latest_val_loss = finite_float(status.get("last_eval_val_loss", status.get("val_loss")))
    learning_rate = finite_float(status.get("learning_rate"))

    if status.get("_load_error"):
        alerts.append(alert("status_json_invalid", "critical", "status.json 无法解析", error=status["_load_error"]))
    if status_age is None:
        alerts.append(alert("status_missing", "critical", "缺少 status.json"))
    elif status_age > args.status_stale_seconds:
        alerts.append(
            alert(
                "status_stale",
                "critical",
                "status.json 长时间未更新，训练可能卡死或已停止",
                age_seconds=round(status_age, 1),
                threshold_seconds=args.status_stale_seconds,
            )
        )
    elif status_age > args.status_warn_seconds:
        alerts.append(
            alert(
                "status_slow",
                "warning",
                "status.json 更新偏慢",
                age_seconds=round(status_age, 1),
                threshold_seconds=args.status_warn_seconds,
            )
        )

    if heartbeat_interval > 0:
        if heartbeat_age is None:
            alerts.append(alert("heartbeat_missing", "warning", "缺少 heartbeat_log.jsonl"))
        elif heartbeat_age > args.heartbeat_stale_seconds:
            alerts.append(
                alert(
                    "heartbeat_stale",
                    "critical",
                    "heartbeat 长时间未更新",
                    age_seconds=round(heartbeat_age, 1),
                    threshold_seconds=args.heartbeat_stale_seconds,
                )
            )

    if state not in {"running", "completed", "stopped"}:
        alerts.append(alert("unexpected_state", "warning", "训练状态不是 running/completed/stopped", state=state))

    if latest_train_loss is None:
        alerts.append(alert("train_loss_missing_or_nonfinite", "critical", "train loss 缺失或不是有限数", value=status.get("train_loss")))
    if latest_val_loss is not None and not math.isfinite(latest_val_loss):
        alerts.append(alert("val_loss_nonfinite", "critical", "val loss 不是有限数", value=status.get("val_loss")))
    if learning_rate is None:
        alerts.append(alert("learning_rate_missing", "warning", "learning_rate 缺失或不是有限数", value=status.get("learning_rate")))

    losses = [finite_float(row.get("train_loss")) for row in heartbeat_rows]
    losses = [value for value in losses if value is not None]
    if len(losses) >= args.loss_window + 1:
        baseline = losses[-args.loss_window - 1 : -1]
        current = losses[-1]
        mean = statistics.mean(baseline)
        stdev = statistics.pstdev(baseline)
        spike_threshold = mean + max(args.loss_spike_min_delta, args.loss_spike_sigma * stdev)
        if current > spike_threshold and current > mean * args.loss_spike_ratio:
            alerts.append(
                alert(
                    "loss_spike",
                    "warning",
                    "train loss 出现异常尖峰",
                    current=round(current, 5),
                    rolling_mean=round(mean, 5),
                    rolling_stdev=round(stdev, 5),
                    threshold=round(spike_threshold, 5),
                    window=args.loss_window,
                )
            )

    tps_values = [finite_float(row.get("tokens_per_second")) for row in heartbeat_rows[-args.tps_window - 1 : -1]]
    tps_values = [value for value in tps_values if value is not None and value > 0]
    tps_median = median(tps_values)
    if tokens_per_second is not None and tps_median and step > args.ignore_perf_before_step:
        if tokens_per_second < tps_median * args.tps_drop_ratio:
            alerts.append(
                alert(
                    "tokens_per_second_drop",
                    "warning",
                    "tokens/sec 相比近期中位数明显下降",
                    current=round(tokens_per_second, 2),
                    recent_median=round(tps_median, 2),
                    drop_ratio=args.tps_drop_ratio,
                )
            )

    if eval_interval > 0 and step >= 0:
        last_eval_step = status.get("last_eval_step")
        if isinstance(last_eval_step, int):
            if step - last_eval_step > eval_interval * args.interval_grace_multiplier:
                alerts.append(
                    alert(
                        "eval_stale_by_step",
                        "warning",
                        "eval 已超过预期 step 间隔未更新",
                        step=step,
                        last_eval_step=last_eval_step,
                        eval_interval=eval_interval,
                    )
                )
        elif training_log_age is None:
            alerts.append(alert("eval_missing", "warning", "缺少 training_log.jsonl，尚无 eval 记录"))

    if checkpoint_interval > 0 and local_step >= 0:
        latest_global_step = latest_state.get("global_step")
        if isinstance(latest_global_step, int):
            if step + 1 - latest_global_step > checkpoint_interval * args.interval_grace_multiplier:
                alerts.append(
                    alert(
                        "checkpoint_stale_by_step",
                        "warning",
                        "latest checkpoint 已超过预期 step 间隔未更新",
                        step=step,
                        checkpoint_global_step=latest_global_step,
                        checkpoint_interval=checkpoint_interval,
                    )
                )
        elif local_step > checkpoint_interval * args.interval_grace_multiplier:
            alerts.append(alert("checkpoint_missing", "warning", "超过 checkpoint 间隔后仍缺少 latest checkpoint"))

    if checkpoint_age is not None and checkpoint_age > args.checkpoint_stale_seconds and local_step > checkpoint_interval:
        alerts.append(
            alert(
                "checkpoint_file_stale",
                "warning",
                "latest checkpoint 文件更新时间偏旧",
                age_seconds=round(checkpoint_age, 1),
                threshold_seconds=args.checkpoint_stale_seconds,
            )
        )

    summary = {
        "checked_at": utcish_now(),
        "run_dir": str(run_dir),
        "state": state,
        "step": step,
        "local_step": local_step,
        "train_loss": latest_train_loss,
        "val_loss": latest_val_loss,
        "learning_rate": learning_rate,
        "tokens_per_second": tokens_per_second,
        "status_age_seconds": None if status_age is None else round(status_age, 1),
        "heartbeat_age_seconds": None if heartbeat_age is None else round(heartbeat_age, 1),
        "training_log_age_seconds": None if training_log_age is None else round(training_log_age, 1),
        "checkpoint_age_seconds": None if checkpoint_age is None else round(checkpoint_age, 1),
        "latest_checkpoint_global_step": latest_state.get("global_step"),
        "best_val_loss": best_state.get("best_val_loss", status.get("best_val_loss")),
        "alert_count": len(alerts),
        "critical_count": sum(1 for item in alerts if item["severity"] == "critical"),
        "warning_count": sum(1 for item in alerts if item["severity"] == "warning"),
        "alerts": alerts,
    }
    return alerts, summary


def write_alert_outputs(run_dir: Path, alerts: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    status_path = run_dir / "alert_status.json"
    state_path = run_dir / ".alert_state.json"
    previous = load_json(state_path)
    previous_keys = set(previous.get("active_keys", []))
    current_keys = {item["key"] for item in alerts}

    for item in alerts:
        if item["key"] not in previous_keys:
            append_jsonl(
                run_dir / "alerts.jsonl",
                {
                    "event": "opened",
                    "created_at": summary["checked_at"],
                    **item,
                },
            )

    for key in sorted(previous_keys - current_keys):
        append_jsonl(
            run_dir / "alerts.jsonl",
            {
                "event": "resolved",
                "resolved_at": summary["checked_at"],
                "key": key,
            },
        )

    status_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    state_path.write_text(
        json.dumps({"active_keys": sorted(current_keys), "updated_at": summary["checked_at"]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--interval-seconds", type=float, default=60.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--status-warn-seconds", type=float, default=240.0)
    parser.add_argument("--status-stale-seconds", type=float, default=600.0)
    parser.add_argument("--heartbeat-stale-seconds", type=float, default=600.0)
    parser.add_argument("--checkpoint-stale-seconds", type=float, default=1800.0)
    parser.add_argument("--loss-window", type=int, default=30)
    parser.add_argument("--loss-spike-sigma", type=float, default=3.0)
    parser.add_argument("--loss-spike-ratio", type=float, default=1.08)
    parser.add_argument("--loss-spike-min-delta", type=float, default=0.35)
    parser.add_argument("--tps-window", type=int, default=20)
    parser.add_argument("--tps-drop-ratio", type=float, default=0.55)
    parser.add_argument("--ignore-perf-before-step", type=int, default=1400)
    parser.add_argument("--interval-grace-multiplier", type=int, default=3)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    while True:
        alerts, summary = evaluate_alerts(run_dir, args)
        write_alert_outputs(run_dir, alerts, summary)
        print(
            f"[{summary['checked_at']}] step={summary.get('step')} "
            f"alerts={summary['alert_count']} critical={summary['critical_count']} warning={summary['warning_count']}",
            flush=True,
        )
        if args.once:
            break
        time.sleep(max(5.0, args.interval_seconds))


if __name__ == "__main__":
    main()
