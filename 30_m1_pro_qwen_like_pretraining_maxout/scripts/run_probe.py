from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parents[1]


def run_config(config: Path) -> dict:
    cmd = [sys.executable, str(CURRENT_DIR / "scripts/train_qwen_like.py"), "--config", str(config)]
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(CURRENT_DIR.parent), text=True, capture_output=True)
    result = {
        "config": str(config),
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "run_dir": None,
        "metrics": None,
    }
    for line in proc.stdout.splitlines():
        if line.startswith("Run dir:"):
            result["run_dir"] = line.split("Run dir:", 1)[1].strip()
    if result["run_dir"]:
        metrics_path = Path(result["run_dir"]) / "metrics.json"
        if metrics_path.exists():
            result["metrics"] = json.loads(metrics_path.read_text(encoding="utf-8"))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lesson 30 probe configs sequentially")
    parser.add_argument("configs", nargs="*", default=[str(CURRENT_DIR / "configs/probe_micro_256.json")])
    args = parser.parse_args()

    results = []
    for item in args.configs:
        results.append(run_config(Path(item)))

    output_path = CURRENT_DIR / "outputs/probes/probe_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# Probe Results", "", "| config | ok | params | tokens/sec | peak memory | run |", "|---|---:|---:|---:|---:|---|"]
    for row in results:
        metrics = row.get("metrics") or {}
        run_dir = Path(row["run_dir"]).name if row.get("run_dir") else ""
        # Peak memory lives in status.json.
        peak = None
        if row.get("run_dir"):
            status_path = Path(row["run_dir"]) / "status.json"
            if status_path.exists():
                status = json.loads(status_path.read_text(encoding="utf-8"))
                peak = (status.get("performance") or {}).get("mlx_peak_memory_gb")
        lines.append(
            f"| {Path(row['config']).name} | {row['returncode'] == 0} | "
            f"{metrics.get('parameter_count')} | {metrics.get('tokens_per_second')} | {peak} | {run_dir} |"
        )
    (CURRENT_DIR / "outputs/reports/probe_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote:", output_path)


if __name__ == "__main__":
    main()
