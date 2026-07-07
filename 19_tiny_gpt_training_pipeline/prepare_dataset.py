from __future__ import annotations

import argparse
import json

import numpy as np

from config import PROCESSED_DATA_DIR, TrainConfig, ensure_project_dirs
from tokenizer import CharacterTokenizer


def build_default_text() -> str:
    lines = [
        "hello ai lab",
        "hello mlx",
        "hello tiny gpt",
        "we learn token embedding",
        "we learn position embedding",
        "we learn next token prediction",
        "we learn self attention",
        "we learn multi head attention",
        "we learn transformer block",
        "tiny gpt stacks transformer blocks",
        "each block keeps the hidden shape",
        "residual connection helps information flow",
        "layernorm helps stabilize training",
        "feed forward layers improve representation",
        "causal mask prevents looking into the future",
        "language models predict the next token",
        "train loss shows how well the model fits seen data",
        "val loss shows how well the model generalizes",
        "checkpoints let us resume training later",
        "generated samples help us judge text quality",
        "a data pipeline turns raw text into token ids",
        "a batch contains many short training sequences",
        "engineering makes experiments easier to reproduce",
        "small models on small text can memorize easily",
    ]
    return ("\n".join(lines) + "\n") * 16


def prepare_dataset(config: TrainConfig | None = None, force: bool = False) -> dict:
    config = config or TrainConfig()
    ensure_project_dirs()

    if not config.raw_text_path.exists():
        config.raw_text_path.write_text(build_default_text(), encoding="utf-8")

    text = config.raw_text_path.read_text(encoding="utf-8")
    tokenizer = CharacterTokenizer.from_text(text)
    encoded = np.array(tokenizer.encode(text), dtype=np.int32)

    if len(encoded) <= config.block_size + 2:
        raise ValueError("raw text is too short for the configured block_size")

    split_idx = int(config.train_split * len(encoded))
    train_ids = encoded[:split_idx]
    val_ids = encoded[split_idx:]

    if len(val_ids) <= config.block_size + 2:
        raise ValueError("validation split is too short; lower block_size or add more text")

    if config.processed_path.exists() and not force:
        meta_path = PROCESSED_DATA_DIR / "dataset_meta.json"
        if meta_path.exists():
            return json.loads(meta_path.read_text(encoding="utf-8"))

    tokenizer.save(config.vocab_path)
    np.savez(
        config.processed_path,
        train_ids=train_ids,
        val_ids=val_ids,
        all_ids=encoded,
    )

    meta = {
        "raw_text_path": str(config.raw_text_path),
        "processed_path": str(config.processed_path),
        "vocab_path": str(config.vocab_path),
        "data_length": int(len(encoded)),
        "train_size": int(len(train_ids)),
        "val_size": int(len(val_ids)),
        "vocab_size": int(tokenizer.vocab_size),
        "chars": tokenizer.chars,
        "train_split": config.train_split,
        "block_size": config.block_size,
    }
    meta_path = PROCESSED_DATA_DIR / "dataset_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Rebuild processed data.")
    args = parser.parse_args()

    meta = prepare_dataset(force=args.force)
    print("Prepared dataset:")
    for key in ["raw_text_path", "processed_path", "vocab_path", "data_length", "train_size", "val_size", "vocab_size"]:
        print(f"{key}: {meta[key]}")


if __name__ == "__main__":
    main()
