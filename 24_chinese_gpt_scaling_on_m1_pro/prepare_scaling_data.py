from __future__ import annotations

import argparse
import json
import shutil
import unicodedata
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


CURRENT_DIR = Path(__file__).resolve().parent
DATA_DIR = CURRENT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = CURRENT_DIR / "outputs"
REPORT_DIR = OUTPUT_DIR / "reports"
TOKENIZER_DIR = OUTPUT_DIR / "tokenizer"

SOURCE_COURSE_DIR = CURRENT_DIR.parent / "23_chinese_open_dataset_pretraining_run"
SOURCE_CORPUS_PATH = SOURCE_COURSE_DIR / "data" / "raw" / "open_zh_corpus.txt"
SOURCE_TOKENIZER_PATH = SOURCE_COURSE_DIR / "outputs" / "tokenizer" / "chinese_bpe_tokenizer.json"

CORPUS_PATH = RAW_DIR / "open_zh_corpus.txt"
TOKENIZER_PATH = TOKENIZER_DIR / "chinese_bpe_tokenizer.json"
REPORT_JSON_PATH = REPORT_DIR / "data_preparation_report.json"
REPORT_TXT_PATH = REPORT_DIR / "data_preparation_report.txt"

SCALES = {
    # 当前第 23 课本地语料约 466 万字符，所以这里用 100 万字符作为 small，
    # 用完整本地语料作为 medium，避免为了本课再次消耗网络流量。
    "small": 1_000_000,
    "medium": None,
    "large": 20_000_000,
}


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return unicodedata.normalize("NFKC", text)


def fallback_corpus() -> str:
    paragraphs = [
        "人工智能正在改变农业、教育、医疗和工业生产。小型语言模型可以帮助我们理解预训练流程。",
        "中文预训练数据需要经过抽样、清洗、分词、编码和 train/val 划分。真实数据比 toy text 更复杂。",
        "大语言模型通过 next-token prediction 学习文本规律。模型看到前文 token,预测下一个 token。",
        "在 M1 Pro 上训练 Tiny GPT,重点不是追求最高质量,而是观察 loss、速度和生成样本的变化。",
        "数据量、模型大小、上下文长度和训练步数都会影响语言模型的训练效果。",
    ]
    lines = []
    for i in range(20_000):
        lines.append(paragraphs[i % len(paragraphs)] + f" 这是第 {i} 条本地示例。")
    return "\n".join(lines) + "\n"


def ensure_source_files() -> dict:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    corpus_source = "lesson_23"
    tokenizer_source = "lesson_23"

    if SOURCE_CORPUS_PATH.exists():
        shutil.copyfile(SOURCE_CORPUS_PATH, CORPUS_PATH)
    else:
        CORPUS_PATH.write_text(fallback_corpus(), encoding="utf-8")
        corpus_source = "local_fallback"

    if SOURCE_TOKENIZER_PATH.exists():
        shutil.copyfile(SOURCE_TOKENIZER_PATH, TOKENIZER_PATH)
    else:
        raise FileNotFoundError(
            "找不到第 23 课 tokenizer。请先运行第 23 课 tokenizer 准备脚本，"
            f"或提供 tokenizer: {SOURCE_TOKENIZER_PATH}"
        )

    return {"corpus_source": corpus_source, "tokenizer_source": tokenizer_source}


def slice_corpus(text: str, max_chars: int | None) -> str:
    if max_chars is None or len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last_newline = cut.rfind("\n")
    if last_newline > 1000:
        cut = cut[:last_newline]
    return cut.strip() + "\n"


def encode_corpus(tokenizer: Tokenizer, text: str) -> np.ndarray:
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    if bos_id is None or eos_id is None:
        raise ValueError("tokenizer 必须包含 <bos> 和 <eos>")

    ids: list[int] = []
    for line in normalize_text(text).splitlines():
        line = line.strip()
        if not line:
            continue
        ids.append(bos_id)
        ids.extend(tokenizer.encode(line).ids)
        ids.append(eos_id)
    return np.array(ids, dtype=np.int32)


def prepare_scale(
    scale: str,
    text: str,
    tokenizer: Tokenizer,
    train_ratio: float,
    block_size_hint: int,
) -> dict:
    target_chars = SCALES[scale]
    scale_dir = PROCESSED_DIR / scale
    scale_dir.mkdir(parents=True, exist_ok=True)

    if scale == "large" and len(text) < int(SCALES["large"] or 0):
        meta = {
            "scale": scale,
            "generated": False,
            "reason": "当前本地复用语料不足 2000 万字符；本课未额外下载 large 档。",
            "available_corpus_chars": len(text),
            "target_chars": SCALES["large"],
        }
        (scale_dir / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        return meta

    subset = slice_corpus(text, target_chars)
    token_ids = encode_corpus(tokenizer, subset)
    split_idx = int(len(token_ids) * train_ratio)
    train_tokens = token_ids[:split_idx]
    val_tokens = token_ids[split_idx:]

    if len(train_tokens) <= block_size_hint + 1 or len(val_tokens) <= block_size_hint + 1:
        raise ValueError(f"{scale} 数据太少，无法支持 block_size={block_size_hint}")

    train_path = scale_dir / "train_tokens.npy"
    val_path = scale_dir / "val_tokens.npy"
    np.save(train_path, train_tokens)
    np.save(val_path, val_tokens)

    meta = {
        "scale": scale,
        "generated": True,
        "target_chars": target_chars,
        "actual_chars": len(subset),
        "vocab_size": tokenizer.get_vocab_size(),
        "total_tokens": int(len(token_ids)),
        "train_tokens": int(len(train_tokens)),
        "val_tokens": int(len(val_tokens)),
        "average_chars_per_token": len(subset) / max(len(token_ids), 1),
        "train_tokens_path": str(train_path),
        "val_tokens_path": str(val_path),
        "block_size_hint": block_size_hint,
    }
    (scale_dir / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def write_reports(source_info: dict, corpus_chars: int, scale_metadata: list[dict]) -> None:
    report = {
        **source_info,
        "current_dir": str(CURRENT_DIR),
        "corpus_path": str(CORPUS_PATH),
        "tokenizer_path": str(TOKENIZER_PATH),
        "available_corpus_chars": corpus_chars,
        "scales": scale_metadata,
    }
    REPORT_JSON_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Scaling Data Preparation Report",
        "",
        f"course_dir: {CURRENT_DIR}",
        f"corpus_source: {source_info['corpus_source']}",
        f"tokenizer_source: {source_info['tokenizer_source']}",
        f"available_corpus_chars: {corpus_chars}",
        f"tokenizer_path: {TOKENIZER_PATH}",
        "",
        "## scales",
    ]
    for meta in scale_metadata:
        lines.append("")
        lines.append(f"### {meta['scale']}")
        for key, value in meta.items():
            lines.append(f"- {key}: {value}")
    REPORT_TXT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--block-size-hint", type=int, default=128)
    args = parser.parse_args()

    source_info = ensure_source_files()
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    corpus = normalize_text(CORPUS_PATH.read_text(encoding="utf-8"))

    scale_metadata = []
    for scale in ["small", "medium", "large"]:
        meta = prepare_scale(
            scale=scale,
            text=corpus,
            tokenizer=tokenizer,
            train_ratio=args.train_ratio,
            block_size_hint=args.block_size_hint,
        )
        scale_metadata.append(meta)

    write_reports(source_info, len(corpus), scale_metadata)

    print("=== Scaling Data Prepared ===")
    print("Current dir:", CURRENT_DIR)
    print("Corpus path:", CORPUS_PATH)
    print("Tokenizer path:", TOKENIZER_PATH)
    for meta in scale_metadata:
        print(meta)


if __name__ == "__main__":
    main()
