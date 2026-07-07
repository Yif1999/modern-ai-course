from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from numpy.lib.format import open_memmap
from tokenizers import Tokenizer


CURRENT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = CURRENT_DIR / "data" / "processed"
METADATA_DIR = CURRENT_DIR / "data" / "metadata"
TOKENIZER_DIR = CURRENT_DIR / "data" / "tokenizers"
REPORT_DIR = CURRENT_DIR / "outputs" / "reports"
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]
DOC_SEPARATOR = "\n\n<|doc_sep|>\n\n"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def iter_docs(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            text = str(row.get("text", "")).strip()
            if text:
                yield text + DOC_SEPARATOR


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    if not value:
        raise ValueError("empty output prefix")
    return value


def count_tokens(tokenizer: Tokenizer, docs_jsonl: Path) -> tuple[int, int, int]:
    total_tokens = 0
    total_docs = 0
    total_chars = 0
    for text in iter_docs(docs_jsonl):
        encoded = tokenizer.encode(text, add_special_tokens=False)
        total_tokens += len(encoded.ids)
        total_docs += 1
        total_chars += len(text)
        if total_docs % 100_000 == 0:
            print(f"count docs={total_docs:,} tokens={total_tokens:,}", flush=True)
    return total_docs, total_chars, total_tokens


def write_tokens(
    *,
    tokenizer: Tokenizer,
    docs_jsonl: Path,
    train_path: Path,
    val_path: Path,
    train_tokens: int,
    val_tokens: int,
) -> tuple[int, int, int]:
    train = open_memmap(train_path, mode="w+", dtype=np.int32, shape=(train_tokens,))
    val = open_memmap(val_path, mode="w+", dtype=np.int32, shape=(val_tokens,))
    train_offset = 0
    val_offset = 0
    total_offset = 0
    docs_written = 0

    for text in iter_docs(docs_jsonl):
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        if not ids:
            continue
        arr = np.asarray(ids, dtype=np.int32)
        end = total_offset + arr.shape[0]

        if total_offset < train_tokens:
            left_count = min(end, train_tokens) - total_offset
            train[train_offset : train_offset + left_count] = arr[:left_count]
            train_offset += left_count
        else:
            left_count = 0

        if end > train_tokens:
            val_start = max(train_tokens - total_offset, 0)
            val_count = end - max(total_offset, train_tokens)
            val[val_offset : val_offset + val_count] = arr[val_start : val_start + val_count]
            val_offset += val_count

        total_offset = end
        docs_written += 1
        if docs_written % 100_000 == 0:
            print(
                f"write docs={docs_written:,} train={train_offset:,}/{train_tokens:,} "
                f"val={val_offset:,}/{val_tokens:,}",
                flush=True,
            )

    train.flush()
    val.flush()
    return docs_written, train_offset, val_offset


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
    parser = argparse.ArgumentParser(description="Stream encode JSONL docs with the lab BPE tokenizer into mmap-friendly .npy arrays.")
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--docs-jsonl", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.02)
    args = parser.parse_args()

    docs_path = Path(args.docs_jsonl)
    tokenizer_path = TOKENIZER_DIR / f"lab_byte_bpe_{args.vocab_size}.json"
    if not docs_path.exists():
        raise FileNotFoundError(docs_path)
    if not tokenizer_path.exists():
        raise FileNotFoundError(tokenizer_path)

    for path in [PROCESSED_DIR, METADATA_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    actual_vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    prefix = safe_name(args.output_prefix)

    print("counting tokens...", flush=True)
    total_docs, total_chars, total_tokens = count_tokens(tokenizer, docs_path)
    if total_tokens < 4096:
        raise ValueError(f"too few tokens: {total_tokens}")

    split = int(total_tokens * (1.0 - args.val_ratio))
    split = min(max(split, 2048), total_tokens - 2048)
    train_tokens = split
    val_tokens = total_tokens - split

    train_path = PROCESSED_DIR / f"train_tokens_{prefix}.npy"
    val_path = PROCESSED_DIR / f"val_tokens_{prefix}.npy"

    print("writing token arrays...", flush=True)
    docs_written, train_written, val_written = write_tokens(
        tokenizer=tokenizer,
        docs_jsonl=docs_path,
        train_path=train_path,
        val_path=val_path,
        train_tokens=train_tokens,
        val_tokens=val_tokens,
    )

    unk_id = tokenizer.token_to_id("<unk>")
    samples = [
        "甲：你觉得人工智能以后会改变什么？\n乙：",
        "老哥稳，这波属于是把本地小模型压榨到极限了。",
        "哈哈哈哈，这是什么离谱操作.jpg",
    ]
    preview = token_preview(tokenizer, samples)
    metadata = {
        "tokenizer_type": "lab_bpe",
        "tokenizer_name": f"lab_byte_bpe_{actual_vocab_size}",
        "tokenizer_path": str(tokenizer_path),
        "vocab_size": actual_vocab_size,
        "requested_vocab_size": args.vocab_size,
        "special_tokens": SPECIAL_TOKENS,
        "unk_id": unk_id,
        "output_prefix": prefix,
        "docs_jsonl_path": str(docs_path),
        "raw_chars": total_chars,
        "total_docs": total_docs,
        "total_tokens": int(total_tokens),
        "chars_per_token": total_chars / total_tokens,
        "train_tokens": int(train_tokens),
        "val_tokens": int(val_tokens),
        "train_tokens_path": str(train_path),
        "val_tokens_path": str(val_path),
        "val_ratio": args.val_ratio,
        "doc_separator": DOC_SEPARATOR,
        "streaming_encoder": True,
        "docs_written": docs_written,
        "train_tokens_written": train_written,
        "val_tokens_written": val_written,
        "preview": preview,
        "elapsed_sec": time.perf_counter() - started,
    }
    metadata_path = METADATA_DIR / f"{prefix}_metadata.json"
    write_json(metadata_path, metadata)

    report_path = REPORT_DIR / f"{prefix}_encode_report.md"
    lines = [
        f"# {prefix} Streaming Encode Report",
        "",
        f"- docs jsonl: `{docs_path}`",
        f"- tokenizer: `{tokenizer_path}`",
        f"- docs: `{total_docs:,}`",
        f"- raw chars: `{total_chars:,}`",
        f"- total tokens: `{total_tokens:,}`",
        f"- chars/token: `{metadata['chars_per_token']:.4f}`",
        f"- train tokens: `{train_tokens:,}`",
        f"- val tokens: `{val_tokens:,}`",
        f"- train path: `{train_path}`",
        f"- val path: `{val_path}`",
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
    print(f"tokens={total_tokens:,} chars/token={metadata['chars_per_token']:.4f}")


if __name__ == "__main__":
    main()
