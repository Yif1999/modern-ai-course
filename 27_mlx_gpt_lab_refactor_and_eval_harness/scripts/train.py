from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlx_gpt_lab.train_loop import train_from_config


def resolve_cli_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    return ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="训练教学型 MLX GPT")
    parser.add_argument("--config", required=True, help="配置文件路径")
    args = parser.parse_args()

    config_path = resolve_cli_path(args.config)
    run_dir = train_from_config(config_path, ROOT)
    print("Run dir:", run_dir)


if __name__ == "__main__":
    main()
