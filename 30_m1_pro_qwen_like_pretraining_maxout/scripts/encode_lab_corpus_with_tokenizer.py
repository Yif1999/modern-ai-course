from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
from tokenizers import Tokenizer


CURRENT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = CURRENT_DIR / "data" / "raw"
PROCESSED_DIR = CURRENT_DIR / "data" / "processed"
METADATA_DIR = CURRENT_DIR / "data" / "metadata"
TOKENIZER_DIR = CURRENT_DIR / "data" / "tokenizers"
REPORT_DIR = CURRENT_DIR / "outputs" / "reports"
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def token_preview(tokenizer: Tokenizer, samples: list[str]) -> list[dict[str, Any]]:
    rows = []
    for text in samples:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(encoded.ids, skip_special_tokens=True)
        rows.append(
            {
                "text": text,
                "ids": encoded.ids[:80],
                "tokens": encoded.tokens[:80],
                "token_count": len(encoded.ids),
                "decoded": decoded,
                "roundtrip_ok": decoded == text,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode the current canonical lab corpus with an existing lab BPE tokenizer.")
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--corpus", default=str(RAW_DIR / "lab_bpe_mixed_corpus.txt"))
    parser.add_argument("--docs-jsonl", default=str(RAW_DIR / "lab_bpe_mixed_docs.jsonl"))
    parser.add_argument(
        "--output-prefix",
        default=None,
        help=(
            "Prefix for train/val/metadata outputs. Defaults to lab_bpe_<vocab_size> "
            "for backward compatibility."
        ),
    )
    parser.add_argument("--val-ratio", type=float, default=0.02)
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    tokenizer_path = TOKENIZER_DIR / f"lab_byte_bpe_{args.vocab_size}.json"
    if not corpus_path.exists():
        raise FileNotFoundError(corpus_path)
    if not tokenizer_path.exists():
        raise FileNotFoundError(tokenizer_path)

    for path in [PROCESSED_DIR, METADATA_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    actual_vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    corpus = corpus_path.read_text(encoding="utf-8")
    encoded = tokenizer.encode(corpus, add_special_tokens=False)
    token_ids = encoded.ids

    split = int(len(token_ids) * (1.0 - args.val_ratio))
    split = min(max(split, 2048), len(token_ids) - 2048)
    train = np.array(token_ids[:split], dtype=np.int32)
    val = np.array(token_ids[split:], dtype=np.int32)

    output_prefix = args.output_prefix or f"lab_bpe_{actual_vocab_size}"
    safe_prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", output_prefix).strip("_")
    if not safe_prefix:
        raise ValueError("--output-prefix cannot be empty after sanitizing")

    train_path = PROCESSED_DIR / f"train_tokens_{safe_prefix}.npy"
    val_path = PROCESSED_DIR / f"val_tokens_{safe_prefix}.npy"
    np.save(train_path, train)
    np.save(val_path, val)

    unk_id = tokenizer.token_to_id("<unk>")
    unk_count = int(sum(1 for token_id in token_ids if token_id == unk_id)) if unk_id is not None else 0
    samples = [
        "甲：你觉得人工智能以后会改变什么？\n乙：",
        "老哥稳，这波属于是把本地小模型压榨到极限了。",
        "乙：嗯，这边只有十度，昨天8度！",
        "哈哈哈哈，这是什么离谱操作.jpg",
    ]
    preview = token_preview(tokenizer, samples)

    metadata = {
        "tokenizer_type": "lab_bpe",
        "tokenizer_name": f"lab_byte_bpe_{actual_vocab_size}",
        "tokenizer_path": str(tokenizer_path),
        "vocab_size": actual_vocab_size,
        "requested_vocab_size": args.vocab_size,
        "output_prefix": safe_prefix,
        "special_tokens": SPECIAL_TOKENS,
        "unk_id": unk_id,
        "unk_count": unk_count,
        "unk_ratio": unk_count / len(token_ids),
        "corpus_path": str(corpus_path),
        "docs_jsonl_path": str(Path(args.docs_jsonl)),
        "raw_chars": len(corpus),
        "total_tokens": int(len(token_ids)),
        "chars_per_token": len(corpus) / len(token_ids),
        "train_tokens": int(train.shape[0]),
        "val_tokens": int(val.shape[0]),
        "train_tokens_path": str(train_path),
        "val_tokens_path": str(val_path),
        "val_ratio": args.val_ratio,
        "preview": preview,
        "source": "encoded from current canonical lab corpus using an existing tokenizer",
        "elapsed_sec": time.perf_counter() - started,
    }
    metadata_path = METADATA_DIR / f"{safe_prefix}_metadata.json"
    write_json(metadata_path, metadata)

    report_path = REPORT_DIR / f"{safe_prefix}_encode_report.md"
    lines = [
        f"# Lab BPE {actual_vocab_size:,} Encode Report",
        "",
        f"- corpus: `{corpus_path}`",
        f"- tokenizer: `{tokenizer_path}`",
        f"- raw chars: `{len(corpus):,}`",
        f"- total tokens: `{len(token_ids):,}`",
        f"- chars/token: `{metadata['chars_per_token']:.4f}`",
        f"- train tokens: `{train.shape[0]:,}`",
        f"- val tokens: `{val.shape[0]:,}`",
        f"- unk ratio: `{metadata['unk_ratio']:.6f}`",
        f"- metadata: `{metadata_path}`",
        "",
        "## Preview",
        "",
        "| text | token_count | roundtrip |",
        "|---|---:|---:|",
    ]
    for item in preview:
        text = item["text"].replace("\n", " / ")
        lines.append(f"| {text} | {item['token_count']} | {item['roundtrip_ok']} |")
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print("metadata:", metadata_path)
    print("train:", train_path)
    print("val:", val_path)
    print("report:", report_path)
    print(f"tokens={len(token_ids):,} chars/token={metadata['chars_per_token']:.4f}")


if __name__ == "__main__":
    main()
