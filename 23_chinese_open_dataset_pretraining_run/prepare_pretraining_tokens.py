from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


CURRENT_DIR = Path(__file__).resolve().parent
RAW_DIR = CURRENT_DIR / "data" / "raw"
PROCESSED_DIR = CURRENT_DIR / "data" / "processed"
REPORT_DIR = CURRENT_DIR / "outputs" / "reports"
TOKENIZER_PATH = CURRENT_DIR / "outputs" / "tokenizer" / "chinese_bpe_tokenizer.json"
CORPUS_PATH = RAW_DIR / "open_zh_corpus.txt"
TRAIN_TOKENS_PATH = PROCESSED_DIR / "train_tokens.npy"
VAL_TOKENS_PATH = PROCESSED_DIR / "val_tokens.npy"
METADATA_PATH = PROCESSED_DIR / "metadata.json"
SAMPLING_REPORT_PATH = REPORT_DIR / "open_dataset_sampling_report.json"


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text.replace("\r\n", "\n").replace("\r", "\n"))


def encode_corpus(tokenizer: Tokenizer, text: str) -> np.ndarray:
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    ids: list[int] = []
    for line in normalize_text(text).splitlines():
        line = line.strip()
        if not line:
            continue
        ids.append(bos_id)
        ids.extend(tokenizer.encode(line).ids)
        ids.append(eos_id)
    return np.array(ids, dtype=np.int32)


def prepare_tokens(train_ratio: float = 0.9, block_size_hint: int = 128) -> dict:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    corpus = CORPUS_PATH.read_text(encoding="utf-8")
    token_ids = encode_corpus(tokenizer, corpus)
    split_idx = int(len(token_ids) * train_ratio)
    train_tokens = token_ids[:split_idx]
    val_tokens = token_ids[split_idx:]
    if len(train_tokens) <= block_size_hint + 1 or len(val_tokens) <= block_size_hint + 1:
        raise ValueError("Not enough tokens for the configured block_size hint")

    np.save(TRAIN_TOKENS_PATH, train_tokens)
    np.save(VAL_TOKENS_PATH, val_tokens)

    sampling_report = {}
    if SAMPLING_REPORT_PATH.exists():
        sampling_report = json.loads(SAMPLING_REPORT_PATH.read_text(encoding="utf-8"))

    metadata = {
        "vocab_size": tokenizer.get_vocab_size(),
        "total_tokens": int(len(token_ids)),
        "train_tokens": int(len(train_tokens)),
        "val_tokens": int(len(val_tokens)),
        "block_size_hint": block_size_hint,
        "dataset_name": sampling_report.get("dataset_name"),
        "used_real_dataset": sampling_report.get("used_real_dataset"),
        "fallback": sampling_report.get("fallback"),
        "tokenizer_path": str(TOKENIZER_PATH),
        "corpus_path": str(CORPUS_PATH),
        "corpus_char_count": len(normalize_text(corpus)),
        "bpe_token_count": int(len(token_ids)),
        "average_chars_per_token": len(normalize_text(corpus)) / max(len(token_ids), 1),
        "train_tokens_path": str(TRAIN_TOKENS_PATH),
        "val_tokens_path": str(VAL_TOKENS_PATH),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print("=== Pretraining Tokens Prepared ===")
    for key, value in metadata.items():
        print(f"{key}: {value}")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--block-size-hint", type=int, default=128)
    args = parser.parse_args()
    prepare_tokens(train_ratio=args.train_ratio, block_size_hint=args.block_size_hint)


if __name__ == "__main__":
    main()
