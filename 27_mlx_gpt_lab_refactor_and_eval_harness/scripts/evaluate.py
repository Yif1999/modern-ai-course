from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlx_gpt_lab.evals import run_evaluation
from mlx_gpt_lab.utils import find_latest_run, load_json


def resolve_cli_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    return ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="运行轻量中文 eval harness")
    parser.add_argument("--config", default="configs/eval_smoke.json", help="eval 配置文件")
    parser.add_argument("--run-dir", default=None, help="指定 run 目录；不指定则使用最新 run")
    parser.add_argument("--train-config", default=None, help="用于定位 run_name 的训练配置")
    args = parser.parse_args()

    eval_config_path = resolve_cli_path(args.config)

    if args.run_dir:
        run_dir = resolve_cli_path(args.run_dir)
    else:
        run_name = None
        if args.train_config:
            train_config_path = resolve_cli_path(args.train_config)
            run_name = load_json(train_config_path).get("run_name")
        run_dir = find_latest_run(ROOT, run_name)

    output_dir = run_evaluation(ROOT, run_dir, eval_config_path)
    print("Eval output:", output_dir)


if __name__ == "__main__":
    main()
