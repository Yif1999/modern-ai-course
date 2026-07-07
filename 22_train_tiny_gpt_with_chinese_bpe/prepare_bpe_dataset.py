from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tokenizer_bpe import (
    PROCESSED_DIR,
    RAW_TEXT_PATH,
    TOKENIZER_PATH,
    BPETokenizer,
    ensure_dirs,
    ensure_raw_text,
    load_or_train_tokenizer,
    normalize_text,
)


CURRENT_DIR = Path(__file__).resolve().parent
DATA_DIR = CURRENT_DIR / "data"
TRAINING_CORPUS_PATH = PROCESSED_DIR / "bpe_training_corpus.txt"
TRAIN_TOKENS_PATH = PROCESSED_DIR / "train_bpe_tokens.npy"
VAL_TOKENS_PATH = PROCESSED_DIR / "val_bpe_tokens.npy"
METADATA_PATH = PROCESSED_DIR / "bpe_dataset_metadata.json"
TOKENIZATION_REPORT_PATH = CURRENT_DIR / "outputs" / "tokenization_report.txt"

EXTRA_SAMPLE_LINES = [
    "人工智能正在改变我们的学习方式。",
    "大语言模型可以根据上下文预测下一个词。",
    "今天我们用 MLX 在 MacBook Pro 上训练一个中文 Tiny GPT。",
    "2026 年的本地 AI 工具越来越重要。",
    "中文、English、数字123和标点符号！都需要 tokenizer 处理。",
    "BPE token 可以覆盖中文、英文、数字和标点混排文本。",
    "模型训练时 cross entropy 预测的是下一个 BPE token。",
]


def build_training_lines(raw_text: str, repeat: int = 32) -> list[str]:
    base_lines = [normalize_text(line).strip() for line in raw_text.splitlines() if line.strip()]
    lines = base_lines + EXTRA_SAMPLE_LINES
    return lines * repeat


def encode_lines_with_special_tokens(tokenizer: BPETokenizer, lines: list[str]) -> np.ndarray:
    ids: list[int] = []
    for line in lines:
        ids.extend(tokenizer.encode(line, add_bos=True, add_eos=True))
    return np.array(ids, dtype=np.int32)


def char_token_count(text: str) -> int:
    return len(normalize_text(text))


def write_tokenization_report(
    tokenizer: BPETokenizer,
    raw_text: str,
    training_corpus: str,
    token_ids_without_special: list[int],
    train_tokens: np.ndarray,
    val_tokens: np.ndarray,
    tokenizer_source: str,
) -> None:
    samples = EXTRA_SAMPLE_LINES[:5]
    lines: list[str] = []
    lines.append("=== 中文 BPE Tiny GPT Tokenization Report ===")
    lines.append("")
    lines.append(f"tokenizer source: {tokenizer_source}")
    lines.append(f"tokenizer path: {TOKENIZER_PATH}")
    lines.append(f"vocab_size: {tokenizer.vocab_size}")
    lines.append(f"raw text path: {RAW_TEXT_PATH}")
    lines.append(f"raw text char count: {char_token_count(raw_text)}")
    lines.append(f"training corpus char count: {char_token_count(training_corpus)}")
    lines.append(f"BPE token count without special tokens: {len(token_ids_without_special)}")
    lines.append(f"BPE token count with <bos>/<eos>: {int(len(train_tokens) + len(val_tokens))}")
    chars_per_token = char_token_count(training_corpus) / max(len(token_ids_without_special), 1)
    lines.append(f"average chars per BPE token: {chars_per_token:.4f}")
    lines.append(f"train token count: {len(train_tokens)}")
    lines.append(f"val token count: {len(val_tokens)}")
    lines.append("")
    lines.append("Special token ids:")
    lines.append(f"<pad>: {tokenizer.pad_id}")
    lines.append(f"<unk>: {tokenizer.unk_id}")
    lines.append(f"<bos>: {tokenizer.bos_id}")
    lines.append(f"<eos>: {tokenizer.eos_id}")
    lines.append("")
    lines.append("=== Encode / Decode 示例 ===")

    for idx, sample in enumerate(samples, start=1):
        normalized = normalize_text(sample)
        char_tokens = list(normalized)
        bpe_ids = tokenizer.encode(normalized)
        bpe_tokens = tokenizer.encode_tokens(normalized)
        decoded = tokenizer.decode(bpe_ids)
        lines.append("")
        lines.append(f"Sample {idx}: {normalized}")
        lines.append(f"字符级 token 数: {len(char_tokens)}")
        lines.append(f"BPE token 数: {len(bpe_ids)}")
        lines.append(f"压缩比例 BPE/char: {len(bpe_ids) / max(len(char_tokens), 1):.4f}")
        lines.append(f"字符级 tokens: {char_tokens}")
        lines.append(f"BPE tokens: {bpe_tokens}")
        lines.append(f"BPE ids: {bpe_ids}")
        lines.append(f"decoded: {decoded}")
        lines.append(f"decode reversible: {decoded == normalized}")

    preview_ids = token_ids_without_special[:80]
    lines.append("")
    lines.append("=== BPE token 生成结果示例 ===")
    lines.append(f"first 80 ids: {preview_ids}")
    lines.append(f"decoded preview: {tokenizer.decode(preview_ids)}")

    TOKENIZATION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    TOKENIZATION_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_bpe_dataset(force: bool = False, train_ratio: float = 0.9, repeat: int = 32) -> dict:
    ensure_dirs()
    if (
        TRAIN_TOKENS_PATH.exists()
        and VAL_TOKENS_PATH.exists()
        and METADATA_PATH.exists()
        and TOKENIZATION_REPORT_PATH.exists()
        and TOKENIZER_PATH.exists()
        and not force
    ):
        return json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    raw_text = ensure_raw_text()
    training_lines = build_training_lines(raw_text, repeat=repeat)
    training_corpus = "\n".join(training_lines) + "\n"
    TRAINING_CORPUS_PATH.write_text(training_corpus, encoding="utf-8")

    tokenizer, tokenizer_source = load_or_train_tokenizer(training_lines)
    token_ids_without_special = tokenizer.encode(training_corpus)
    token_ids = encode_lines_with_special_tokens(tokenizer, training_lines)

    split_idx = int(len(token_ids) * train_ratio)
    train_tokens = token_ids[:split_idx]
    val_tokens = token_ids[split_idx:]
    if len(train_tokens) < 80 or len(val_tokens) < 80:
        raise ValueError("BPE dataset is too small; increase repeat or lower block_size")

    np.save(TRAIN_TOKENS_PATH, train_tokens)
    np.save(VAL_TOKENS_PATH, val_tokens)

    metadata = {
        "raw_text_path": str(RAW_TEXT_PATH),
        "training_corpus_path": str(TRAINING_CORPUS_PATH),
        "tokenizer_path": str(TOKENIZER_PATH),
        "tokenizer_source": tokenizer_source,
        "train_tokens_path": str(TRAIN_TOKENS_PATH),
        "val_tokens_path": str(VAL_TOKENS_PATH),
        "tokenization_report_path": str(TOKENIZATION_REPORT_PATH),
        "raw_char_count": char_token_count(raw_text),
        "training_corpus_char_count": char_token_count(training_corpus),
        "bpe_token_count_without_special": int(len(token_ids_without_special)),
        "bpe_token_count_with_special": int(len(token_ids)),
        "average_chars_per_bpe_token": char_token_count(training_corpus) / max(len(token_ids_without_special), 1),
        "train_ratio": train_ratio,
        "train_token_count": int(len(train_tokens)),
        "val_token_count": int(len(val_tokens)),
        "vocab_size": int(tokenizer.vocab_size),
        "repeat": repeat,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    write_tokenization_report(
        tokenizer=tokenizer,
        raw_text=raw_text,
        training_corpus=training_corpus,
        token_ids_without_special=token_ids_without_special,
        train_tokens=train_tokens,
        val_tokens=val_tokens,
        tokenizer_source=tokenizer_source,
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--repeat", type=int, default=32)
    args = parser.parse_args()

    metadata = prepare_bpe_dataset(force=args.force, repeat=args.repeat)
    print("=== Chinese BPE Dataset Prepared ===")
    for key in [
        "tokenizer_source",
        "tokenizer_path",
        "raw_text_path",
        "training_corpus_path",
        "raw_char_count",
        "training_corpus_char_count",
        "vocab_size",
        "bpe_token_count_without_special",
        "bpe_token_count_with_special",
        "average_chars_per_bpe_token",
        "train_token_count",
        "val_token_count",
        "tokenization_report_path",
    ]:
        print(f"{key}: {metadata[key]}")


if __name__ == "__main__":
    main()
