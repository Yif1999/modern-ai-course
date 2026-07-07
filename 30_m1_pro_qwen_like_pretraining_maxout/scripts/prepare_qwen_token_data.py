from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


CURRENT_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = CURRENT_DIR.parent
DATA_DIR = CURRENT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"
CACHE_DIR = DATA_DIR / "cache"


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_prefix(path: Path, max_chars: int | None) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[:max_chars] if max_chars else text


def collect_existing_corpora(max_general_chars: int, max_fun_chars: int) -> tuple[str, list[dict]]:
    sources = []
    pieces = []

    general_candidates = [
        PROJECT_DIR / "23_chinese_open_dataset_pretraining_run/data/raw/open_zh_corpus.txt",
        PROJECT_DIR / "24_chinese_gpt_scaling_on_m1_pro/data/raw/open_zh_corpus.txt",
    ]
    general_remaining = max_general_chars
    for path in general_candidates:
        if general_remaining <= 0:
            break
        text = read_prefix(path, general_remaining)
        if text:
            pieces.append(text)
            sources.append({"path": str(path), "type": "general_zh", "chars": len(text)})
            general_remaining -= len(text)

    fun_path = PROJECT_DIR / "28_chinese_fun_corpus_pipeline/data/processed/fun_corpus.txt"
    fun_text = read_prefix(fun_path, max_fun_chars)
    if fun_text:
        pieces.append(fun_text)
        sources.append({"path": str(fun_path), "type": "fun_corpus", "chars": len(fun_text)})

    return "\n\n".join(pieces), sources


def maybe_stream_more_chinese(
    *,
    enabled: bool,
    dataset_name: str,
    max_chars: int,
    cache_dir: Path,
) -> tuple[str, dict]:
    if not enabled or max_chars <= 0:
        return "", {"enabled": enabled, "chars": 0, "docs": 0, "error": None}
    try:
        from datasets import load_dataset
    except ImportError as exc:
        return "", {"enabled": enabled, "chars": 0, "docs": 0, "error": f"datasets missing: {exc}"}

    try:
        ds = load_dataset(dataset_name, split="train", streaming=True, cache_dir=str(cache_dir))
        pieces = []
        chars = 0
        docs = 0
        for row in ds:
            text = None
            if isinstance(row, dict):
                for key in ["text", "content", "document"]:
                    if isinstance(row.get(key), str):
                        text = row[key]
                        break
            if not text:
                continue
            text = text.strip()
            if len(text) < 80:
                continue
            pieces.append(text)
            chars += len(text)
            docs += 1
            if chars >= max_chars:
                break
        return "\n\n".join(pieces), {"enabled": enabled, "chars": chars, "docs": docs, "error": None}
    except Exception as exc:
        return "", {"enabled": enabled, "chars": 0, "docs": 0, "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Qwen tokenizer data for lesson 30 maxout training")
    parser.add_argument("--tokenizer-name", default="Qwen/Qwen3.6-27B")
    parser.add_argument("--max-general-chars", type=int, default=8_000_000)
    parser.add_argument("--max-fun-chars", type=int, default=2_000_000)
    parser.add_argument("--stream-open-data", action="store_true")
    parser.add_argument("--stream-dataset-name", default="opencsg/chinese-fineweb-edu")
    parser.add_argument("--stream-max-chars", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.02)
    args = parser.parse_args()

    for path in [RAW_DIR, PROCESSED_DIR, METADATA_DIR, CACHE_DIR]:
        path.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", str(CACHE_DIR / "hf"))
    print("Loading tokenizer:", args.tokenizer_name)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name,
        trust_remote_code=True,
        cache_dir=str(CACHE_DIR / "transformers"),
    )

    text, sources = collect_existing_corpora(args.max_general_chars, args.max_fun_chars)
    streamed_text, stream_report = maybe_stream_more_chinese(
        enabled=args.stream_open_data,
        dataset_name=args.stream_dataset_name,
        max_chars=args.stream_max_chars,
        cache_dir=CACHE_DIR / "datasets",
    )
    if streamed_text:
        text = text + "\n\n" + streamed_text if text else streamed_text
        sources.append(
            {
                "path": args.stream_dataset_name,
                "type": "streamed_general_zh",
                "chars": len(streamed_text),
                "docs": stream_report.get("docs"),
            }
        )

    if not text.strip():
        raise RuntimeError("No corpus text found. Run previous lessons or enable streaming.")

    corpus_path = RAW_DIR / "qwen_mixed_corpus.txt"
    corpus_path.write_text(text, encoding="utf-8")

    print("Encoding corpus chars:", len(text))
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) < 4096:
        raise RuntimeError("Encoded corpus is too small for training.")

    split = int(len(token_ids) * (1.0 - args.val_ratio))
    split = min(max(split, 2048), len(token_ids) - 2048)
    train = np.array(token_ids[:split], dtype=np.int32)
    val = np.array(token_ids[split:], dtype=np.int32)
    np.save(PROCESSED_DIR / "train_tokens.npy", train)
    np.save(PROCESSED_DIR / "val_tokens.npy", val)

    special_tokens_map = getattr(tokenizer, "special_tokens_map", {})
    metadata = {
        "tokenizer_name": args.tokenizer_name,
        "vocab_size": int(len(tokenizer.get_vocab())),
        "corpus_path": str(corpus_path),
        "raw_chars": len(text),
        "total_tokens": int(len(token_ids)),
        "train_tokens": int(train.shape[0]),
        "val_tokens": int(val.shape[0]),
        "val_ratio": args.val_ratio,
        "sources": sources,
        "stream_report": stream_report,
        "special_tokens_map": special_tokens_map,
        "chat_template_available": bool(getattr(tokenizer, "chat_template", None)),
    }
    write_json(METADATA_DIR / "qwen_token_data_metadata.json", metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
