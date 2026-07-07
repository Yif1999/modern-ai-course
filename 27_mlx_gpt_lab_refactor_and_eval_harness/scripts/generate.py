from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlx_gpt_lab.generate import generate_text, load_model_tokenizer_from_run
from mlx_gpt_lab.utils import find_latest_run, load_json


def resolve_cli_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    return ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="从 run checkpoint 生成文本")
    parser.add_argument("--config", required=True, help="训练配置文件，用于定位默认 run_name")
    parser.add_argument("--prompt", default="人工智能", help="生成 prompt")
    parser.add_argument("--run-dir", default=None, help="指定 run 目录；不指定则用 config run_name 找最新 run")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    args = parser.parse_args()

    config_path = resolve_cli_path(args.config)
    config = load_json(config_path)
    run_dir = Path(args.run_dir) if args.run_dir else find_latest_run(ROOT, config.get("run_name"))
    if not run_dir.is_absolute():
        run_dir = resolve_cli_path(str(run_dir))

    model, tokenizer, train_config, checkpoint_path = load_model_tokenizer_from_run(ROOT, run_dir)
    text = generate_text(model, tokenizer, args.prompt, train_config, max_new_tokens=args.max_new_tokens)
    output_path = run_dir / "generated_from_script.txt"
    output_path.write_text(text, encoding="utf-8")
    print("Checkpoint:", checkpoint_path)
    print("Saved:", output_path)
    print(text)


if __name__ == "__main__":
    main()
