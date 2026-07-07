from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np


CURRENT_DIR = Path(__file__).resolve().parent
DATA_DIR = CURRENT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = CURRENT_DIR / "outputs"
RAW_TEXT_PATH = RAW_DIR / "input_zh.txt"
CLEAN_TEXT_PATH = PROCESSED_DIR / "clean_corpus_zh.txt"
TRAIN_TOKENS_PATH = PROCESSED_DIR / "train_tokens.npy"
VAL_TOKENS_PATH = PROCESSED_DIR / "val_tokens.npy"
VOCAB_PATH = PROCESSED_DIR / "vocab.json"
METADATA_PATH = PROCESSED_DIR / "metadata.json"
SUMMARY_PATH = OUTPUT_DIR / "dataset_summary.txt"


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def build_default_chinese_text() -> str:
    paragraphs = [
        "人工智能实验室从最小的张量开始，逐步理解模型如何学习。",
        "一个语言模型的目标，是根据前面的文本预测下一个 token。",
        "在字符级 tokenizer 中，每一个汉字、标点、空格都可以成为一个 token。",
        "中文文本通常没有天然的空格分词，所以字符级方法很直观，也很容易调试。",
        "但是字符级 tokenizer 的序列会比较长，同样一句话需要更多位置来表示。",
        "数据管线的第一步是收集 raw text，并把它保存到课程目录下。",
        "第二步是清洗文本，去掉空行、重复段落、异常符号和太短的句子。",
        "第三步是构建 vocab，把每个字符映射成一个整数 id。",
        "第四步是 encode，把连续文本变成连续的 token ids。",
        "第五步是划分 train 和 val，用训练集学习，用验证集观察泛化。",
        "Tiny GPT 不直接读取字符串，它读取的是整数 token id 序列。",
        "batch 中的 x 是当前 token 序列，y 是向右移动一位的目标序列。",
        "如果 x 看到的是今天我们学习，那么 y 对应的是天我们学习语。",
        "这种 next-token prediction 是语言模型训练的核心任务。",
        "中文预训练数据需要关注字符数量、中文比例、重复率和文本质量。",
        "过短的文本缺少上下文，过长的文本可以先切成更小的段落。",
        "重复内容太多会让模型更容易记忆，而不是学习更一般的语言规律。",
        "metadata 用来记录数据来源、清洗规则、词表大小和切分比例。",
        "后续如果切换到 BPE tokenizer，raw text 和 cleaned corpus 仍然可以复用。",
        "工程化的数据管线让实验可以重复运行，也方便比较不同 tokenizer 的效果。",
        "我们现在只准备数据，不训练模型，下一步才会把数据接入 Tiny GPT。",
        "一个好的数据管线不追求复杂，而是要清晰、可检查、可复现。",
        "当数据规模变大时，可以继续加入分片、缓存、去重和质量过滤。",
        "这一课的重点是让中文文本变成模型可以消费的 token ids。",
    ]
    return "\n".join(paragraphs) + "\n"


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
    text = text.replace("\u3000", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = "".join(ch for ch in text if ch == "\n" or ch.isprintable())
    return text


def clean_corpus(
    raw_text: str,
    min_len: int = 8,
    max_len: int = 240,
    min_chinese_ratio: float = 0.25,
) -> tuple[str, dict]:
    normalized = normalize_text(raw_text)
    raw_lines = [line.strip() for line in normalized.splitlines()]

    kept: list[str] = []
    seen: set[str] = set()
    counters = Counter()

    for line in raw_lines:
        counters["input_lines"] += 1
        line = re.sub(r"\s+", " ", line).strip()

        if not line:
            counters["empty_lines"] += 1
            continue
        if len(line) < min_len:
            counters["too_short"] += 1
            continue
        if len(line) > max_len:
            counters["too_long"] += 1
            continue
        if chinese_ratio(line) < min_chinese_ratio:
            counters["low_chinese_ratio"] += 1
            continue

        fingerprint = hashlib.sha1(line.encode("utf-8")).hexdigest()
        if fingerprint in seen:
            counters["duplicates"] += 1
            continue

        seen.add(fingerprint)
        kept.append(line)
        counters["kept_lines"] += 1

    clean_text = "\n".join(kept).strip()
    if clean_text:
        clean_text += "\n"

    stats = dict(counters)
    stats["min_len"] = min_len
    stats["max_len"] = max_len
    stats["min_chinese_ratio"] = min_chinese_ratio
    return clean_text, stats


def build_vocab(text: str) -> tuple[list[str], dict[str, int], dict[int, str]]:
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return chars, stoi, itos


def encode(text: str, stoi: dict[str, int]) -> np.ndarray:
    missing = sorted(set(text) - set(stoi))
    if missing:
        raise ValueError(f"text contains characters outside vocab: {missing}")
    return np.array([stoi[ch] for ch in text], dtype=np.int32)


def decode(ids, itos: dict[int, str]) -> str:
    return "".join(itos[int(i)] for i in ids)


def prepare_dataset(
    force: bool = False,
    train_ratio: float = 0.9,
    min_len: int = 8,
    max_len: int = 240,
    min_chinese_ratio: float = 0.25,
) -> dict:
    ensure_dirs()

    if not RAW_TEXT_PATH.exists():
        RAW_TEXT_PATH.write_text(build_default_chinese_text(), encoding="utf-8")

    if (
        TRAIN_TOKENS_PATH.exists()
        and VAL_TOKENS_PATH.exists()
        and VOCAB_PATH.exists()
        and METADATA_PATH.exists()
        and not force
    ):
        return json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    raw_text = RAW_TEXT_PATH.read_text(encoding="utf-8")
    clean_text, clean_stats = clean_corpus(
        raw_text,
        min_len=min_len,
        max_len=max_len,
        min_chinese_ratio=min_chinese_ratio,
    )

    if len(clean_text) < 100:
        raise ValueError("cleaned text is too short; add more Chinese raw text")

    chars, stoi, itos = build_vocab(clean_text)
    token_ids = encode(clean_text, stoi)

    split_idx = int(len(token_ids) * train_ratio)
    train_tokens = token_ids[:split_idx]
    val_tokens = token_ids[split_idx:]

    if len(train_tokens) == 0 or len(val_tokens) == 0:
        raise ValueError("train/val split produced an empty split")

    CLEAN_TEXT_PATH.write_text(clean_text, encoding="utf-8")
    np.save(TRAIN_TOKENS_PATH, train_tokens)
    np.save(VAL_TOKENS_PATH, val_tokens)

    vocab_payload = {
        "type": "character",
        "chars": chars,
        "stoi": stoi,
        "itos": {str(i): ch for i, ch in itos.items()},
        "vocab_size": len(chars),
    }
    VOCAB_PATH.write_text(json.dumps(vocab_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    sample_text = clean_text[:40]
    sample_ids = encode(sample_text, stoi)
    sample_decoded = decode(sample_ids, itos)

    metadata = {
        "raw_text_path": str(RAW_TEXT_PATH),
        "clean_text_path": str(CLEAN_TEXT_PATH),
        "train_tokens_path": str(TRAIN_TOKENS_PATH),
        "val_tokens_path": str(VAL_TOKENS_PATH),
        "vocab_path": str(VOCAB_PATH),
        "summary_path": str(SUMMARY_PATH),
        "raw_char_count": len(raw_text),
        "clean_char_count": len(clean_text),
        "raw_chinese_ratio": chinese_ratio(raw_text),
        "clean_chinese_ratio": chinese_ratio(clean_text),
        "train_ratio": train_ratio,
        "train_token_count": int(len(train_tokens)),
        "val_token_count": int(len(val_tokens)),
        "total_token_count": int(len(token_ids)),
        "vocab_size": int(len(chars)),
        "cleaning": clean_stats,
        "sample_text": sample_text,
        "sample_token_ids": sample_ids.tolist(),
        "sample_decoded": sample_decoded,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_lines = [
        "=== 中文预训练数据管线摘要 ===",
        f"raw text: {RAW_TEXT_PATH}",
        f"clean text: {CLEAN_TEXT_PATH}",
        f"train tokens: {TRAIN_TOKENS_PATH}",
        f"val tokens: {VAL_TOKENS_PATH}",
        f"vocab: {VOCAB_PATH}",
        "",
        f"原始字符数: {metadata['raw_char_count']}",
        f"清洗后字符数: {metadata['clean_char_count']}",
        f"原始中文比例: {metadata['raw_chinese_ratio']:.4f}",
        f"清洗后中文比例: {metadata['clean_chinese_ratio']:.4f}",
        f"train token 数: {metadata['train_token_count']}",
        f"val token 数: {metadata['val_token_count']}",
        f"vocab size: {metadata['vocab_size']}",
        "",
        "Sample encode / decode:",
        f"text: {metadata['sample_text']!r}",
        f"ids: {metadata['sample_token_ids']}",
        f"decoded: {metadata['sample_decoded']!r}",
    ]
    SUMMARY_PATH.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Rebuild processed files.")
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--min-len", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=240)
    parser.add_argument("--min-chinese-ratio", type=float, default=0.25)
    args = parser.parse_args()

    metadata = prepare_dataset(
        force=args.force,
        train_ratio=args.train_ratio,
        min_len=args.min_len,
        max_len=args.max_len,
        min_chinese_ratio=args.min_chinese_ratio,
    )

    print("=== Chinese Open Text Dataset Prepared ===")
    print("Raw text:", metadata["raw_text_path"])
    print("Clean text:", metadata["clean_text_path"])
    print("Train tokens:", metadata["train_tokens_path"])
    print("Val tokens:", metadata["val_tokens_path"])
    print("Vocab:", metadata["vocab_path"])
    print()
    print("原始字符数:", metadata["raw_char_count"])
    print("清洗后字符数:", metadata["clean_char_count"])
    print("原始中文比例:", round(metadata["raw_chinese_ratio"], 4))
    print("清洗后中文比例:", round(metadata["clean_chinese_ratio"], 4))
    print("train token 数:", metadata["train_token_count"])
    print("val token 数:", metadata["val_token_count"])
    print("vocab size:", metadata["vocab_size"])
    print()
    print("清洗统计:")
    for key, value in metadata["cleaning"].items():
        print(f"{key}: {value}")
    print()
    print("Sample encode / decode:")
    print("text:", repr(metadata["sample_text"]))
    print("ids:", metadata["sample_token_ids"])
    print("decoded:", repr(metadata["sample_decoded"]))


if __name__ == "__main__":
    main()
