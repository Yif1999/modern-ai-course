from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlx_gpt_lab.dataset import prepare_data_from_config
from mlx_gpt_lab.utils import load_json


def resolve_cli_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    return ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="准备 MLX GPT Lab 数据和 tokenizer")
    parser.add_argument("--config", required=True, help="配置文件路径")
    parser.add_argument("--force", action="store_true", help="强制重新生成 processed 数据")
    args = parser.parse_args()

    config_path = resolve_cli_path(args.config)
    config = load_json(config_path)
    prepared = prepare_data_from_config(config, ROOT, force=args.force)
    print("Prepared data:")
    print("  train_tokens:", prepared.train_tokens_path)
    print("  val_tokens:", prepared.val_tokens_path)
    print("  tokenizer:", prepared.tokenizer_path)
    print("  metadata:", prepared.metadata_path)
    print("  vocab_size:", prepared.vocab_size)


if __name__ == "__main__":
    main()
