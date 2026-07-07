from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import unicodedata
from pathlib import Path
from typing import Any


CURRENT_DIR = Path(__file__).resolve().parent
DATA_DIR = CURRENT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
REPORT_DIR = CURRENT_DIR / "outputs" / "reports"

JSONL_PATH = RAW_DIR / "open_zh_sample.jsonl"
CORPUS_PATH = RAW_DIR / "open_zh_corpus.txt"
REPORT_PATH = REPORT_DIR / "open_dataset_sampling_report.json"
PREVIEW_PATH = REPORT_DIR / "open_dataset_preview.txt"

TEXT_FIELD_CANDIDATES = ["text", "content", "document", "raw_content", "article", "body"]


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def is_chinese_char(ch: str) -> bool:
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0x20000 <= code <= 0x2A6DF
        or 0x2A700 <= code <= 0x2B73F
        or 0x2B740 <= code <= 0x2B81F
        or 0x2B820 <= code <= 0x2CEAF
        or 0xF900 <= code <= 0xFAFF
    )


def chinese_ratio(text: str) -> float:
    chars = [ch for ch in text if not ch.isspace()]
    if not chars:
        return 0.0
    return sum(1 for ch in chars if is_chinese_char(ch)) / len(chars)


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"https?://\\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[\\t\\f\\v]+", " ", text)
    text = re.sub(r"\\s+", " ", text)
    return text.strip()


def clean_text(text: str, min_len: int = 80, max_len: int = 4000, min_zh_ratio: float = 0.25) -> str | None:
    text = normalize_text(text)
    if len(text) < min_len:
        return None
    if len(text) > max_len:
        text = text[:max_len].strip()
    if chinese_ratio(text) < min_zh_ratio:
        return None
    return text


def extract_strings(value: Any, prefix: str = "") -> list[tuple[str, str]]:
    if isinstance(value, str):
        return [(prefix, value)]
    if isinstance(value, dict):
        out = []
        for key, item in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            out.extend(extract_strings(item, child_prefix))
        return out
    if isinstance(value, list):
        out = []
        for i, item in enumerate(value[:5]):
            out.extend(extract_strings(item, f"{prefix}[{i}]"))
        return out
    return []


def detect_text_field(example: dict) -> tuple[str | None, str | None]:
    for key in TEXT_FIELD_CANDIDATES:
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            return key, value
    strings = extract_strings(example)
    if not strings:
        return None, None
    strings.sort(key=lambda item: len(item[1]), reverse=True)
    return strings[0]


def build_offline_docs(max_docs: int, max_chars: int) -> list[dict]:
    templates = [
        "人工智能正在进入教育、办公、编程和内容创作等场景。一个小型语言模型虽然参数不多,但仍然可以学习文本中的局部模式。",
        "中文预训练语料需要经过采样、清洗、去重、tokenizer 编码和 train/val 划分。真实数据比 toy text 更混杂,也更接近实际训练。",
        "大语言模型的训练目标通常是 next-token prediction。模型看到前面的 token,预测下一个 token,再通过交叉熵计算 loss。",
        "在本地 MacBook Pro 上训练 Tiny GPT,重点不是追求大模型质量,而是理解数据管线、训练日志、loss 曲线和生成样本。",
        "开源中文数据集可能包含新闻、百科、网页、论坛和技术文章。不同来源的数据质量差异很大,需要进行基础过滤。",
    ]
    docs = []
    chars = 0
    for i in range(max_docs):
        text = templates[i % len(templates)] + f" 这是第 {i} 条本地 fallback 示例文本,用于在网络不可用时完成课程流程。"
        chars += len(text)
        docs.append({"text": text})
        if chars >= max_chars:
            break
    return docs


def stream_hf_dataset(dataset_name: str, split: str):
    from datasets import load_dataset

    return load_dataset(
        dataset_name,
        split=split,
        streaming=True,
        cache_dir=str(CACHE_DIR),
    )


def sample_dataset(
    dataset_name: str,
    max_docs: int,
    max_chars: int,
    offline_fallback: bool,
    split: str,
) -> dict:
    ensure_dirs()
    os.environ["HF_DATASETS_CACHE"] = str(CACHE_DIR)

    print("=== Open Chinese Dataset Sampling ===")
    print("dataset_name:", dataset_name)
    print("max_docs:", max_docs)
    print("max_chars:", max_chars)
    print("streaming:", not offline_fallback)
    print("说明: 使用 streaming / 按需读取, 不会全量下载数据集。")
    print("cache dir:", CACHE_DIR)

    used_real_dataset = False
    fallback = False
    fallback_reason = None
    source_dataset = dataset_name

    if offline_fallback:
        iterable = build_offline_docs(max_docs=max_docs, max_chars=max_chars)
        fallback = True
        fallback_reason = "offline_fallback_requested"
    else:
        try:
            iterable = stream_hf_dataset(dataset_name, split=split)
            used_real_dataset = True
        except Exception as exc:  # noqa: BLE001
            fallback_reason = f"{dataset_name} failed: {exc}"
            try:
                source_dataset = "Skywork/SkyPile-150B"
                iterable = stream_hf_dataset(source_dataset, split=split)
                used_real_dataset = True
            except Exception as exc2:  # noqa: BLE001
                fallback = True
                fallback_reason += f"; Skywork/SkyPile-150B failed: {exc2}"
                iterable = build_offline_docs(max_docs=max_docs, max_chars=max_chars)

    seen_hashes: set[str] = set()
    docs = []
    raw_docs_seen = 0
    raw_chars_seen = 0
    cleaned_chars = 0
    duplicate_count = 0
    field_counts: dict[str, int] = {}

    for example in iterable:
        raw_docs_seen += 1
        if raw_docs_seen > max_docs:
            break

        field_name, raw_text = detect_text_field(example)
        if not raw_text:
            continue
        field_counts[field_name or "<unknown>"] = field_counts.get(field_name or "<unknown>", 0) + 1
        raw_chars_seen += len(raw_text)

        text = clean_text(raw_text)
        if text is None:
            if raw_chars_seen >= max_chars:
                break
            continue

        fingerprint = hashlib.sha1(text.encode("utf-8")).hexdigest()
        if fingerprint in seen_hashes:
            duplicate_count += 1
            if raw_chars_seen >= max_chars:
                break
            continue

        seen_hashes.add(fingerprint)
        docs.append(
            {
                "id": len(docs),
                "text": text,
                "char_count": len(text),
                "chinese_ratio": chinese_ratio(text),
                "source_dataset": source_dataset,
                "text_field": field_name,
            }
        )
        cleaned_chars += len(text)

        if raw_chars_seen >= max_chars:
            break

    if not docs:
        fallback = True
        used_real_dataset = False
        source_dataset = "offline_fallback"
        fallback_reason = fallback_reason or "no documents kept from streaming dataset"
        docs = []
        for i, example in enumerate(build_offline_docs(max_docs=max_docs, max_chars=max_chars)):
            text = clean_text(example["text"], min_len=40)
            if text is None:
                continue
            docs.append(
                {
                    "id": i,
                    "text": text,
                    "char_count": len(text),
                    "chinese_ratio": chinese_ratio(text),
                    "source_dataset": source_dataset,
                    "text_field": "text",
                }
            )
        raw_docs_seen = len(docs)
        raw_chars_seen = sum(item["char_count"] for item in docs)
        cleaned_chars = raw_chars_seen
        field_counts = {"text": len(docs)}

    with JSONL_PATH.open("w", encoding="utf-8") as f:
        for item in docs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    CORPUS_PATH.write_text("\n\n".join(item["text"] for item in docs) + "\n", encoding="utf-8")

    lengths = [item["char_count"] for item in docs]
    ratios = [item["chinese_ratio"] for item in docs]
    report = {
        "dataset_name": source_dataset,
        "requested_dataset_name": dataset_name,
        "used_real_dataset": used_real_dataset,
        "fallback": fallback,
        "fallback_reason": fallback_reason,
        "streaming": used_real_dataset,
        "max_docs": max_docs,
        "max_chars": max_chars,
        "raw_docs_seen": raw_docs_seen,
        "cleaned_docs": len(docs),
        "raw_char_count": raw_chars_seen,
        "clean_char_count": cleaned_chars,
        "avg_length": statistics.mean(lengths) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
        "min_length": min(lengths) if lengths else 0,
        "estimated_chinese_ratio": statistics.mean(ratios) if ratios else 0,
        "duplicate_count": duplicate_count,
        "field_counts": field_counts,
        "used_text_field": max(field_counts, key=field_counts.get) if field_counts else None,
        "jsonl_path": str(JSONL_PATH),
        "corpus_path": str(CORPUS_PATH),
        "jsonl_file_size_bytes": JSONL_PATH.stat().st_size,
        "corpus_file_size_bytes": CORPUS_PATH.stat().st_size,
        "cache_dir": str(CACHE_DIR),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    preview_lines = ["=== Open Dataset Preview ===", json.dumps(report, ensure_ascii=False, indent=2), ""]
    for item in docs[:10]:
        preview_lines.append(f"--- doc {item['id']} len={item['char_count']} zh={item['chinese_ratio']:.3f} ---")
        preview_lines.append(item["text"][:800])
        preview_lines.append("")
    PREVIEW_PATH.write_text("\n".join(preview_lines), encoding="utf-8")

    print("Sampling finished.")
    print("used_real_dataset:", used_real_dataset)
    print("fallback:", fallback)
    print("raw_docs_seen:", raw_docs_seen)
    print("cleaned_docs:", len(docs))
    print("raw_char_count:", raw_chars_seen)
    print("clean_char_count:", cleaned_chars)
    print("used_text_field:", report["used_text_field"])
    print("jsonl:", JSONL_PATH)
    print("corpus:", CORPUS_PATH)
    print("report:", REPORT_PATH)
    print("preview:", PREVIEW_PATH)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="opencsg/chinese-fineweb-edu")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-docs", type=int, default=5000)
    parser.add_argument("--max-chars", type=int, default=5_000_000)
    parser.add_argument("--offline-fallback", action="store_true")
    args = parser.parse_args()
    sample_dataset(
        dataset_name=args.dataset_name,
        split=args.split,
        max_docs=args.max_docs,
        max_chars=args.max_chars,
        offline_fallback=args.offline_fallback,
    )


if __name__ == "__main__":
    main()
