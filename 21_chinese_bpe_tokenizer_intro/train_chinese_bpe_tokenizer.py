from __future__ import annotations

import json
import shutil
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tokenizers import Tokenizer, decoders, models, normalizers, trainers


CURRENT_DIR = Path(__file__).resolve().parent
DATA_DIR = CURRENT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = CURRENT_DIR / "outputs"
TOKENIZER_DIR = OUTPUT_DIR / "tokenizers"

RAW_TEXT_PATH = RAW_DIR / "input_zh.txt"
PROCESSED_CORPUS_PATH = PROCESSED_DIR / "bpe_training_corpus.txt"
REPORT_PATH = OUTPUT_DIR / "tokenizer_comparison_report.txt"
PLOT_PATH = OUTPUT_DIR / "token_count_comparison.png"
METADATA_PATH = PROCESSED_DIR / "bpe_metadata.json"

PREVIOUS_RAW_TEXT_PATH = (
    CURRENT_DIR.parent
    / "20_chinese_open_text_pretraining_dataset"
    / "data"
    / "raw"
    / "input_zh.txt"
)

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]
VOCAB_SIZES = [128, 256, 512]

SAMPLE_TEXTS = [
    "人工智能正在改变我们的学习方式。",
    "大语言模型可以根据上下文预测下一个词。",
    "今天我们用 MLX 在 MacBook Pro 上训练一个中文 Tiny GPT。",
    "2026 年的本地 AI 工具越来越重要。",
    "中文、English、数字123和标点符号！都需要 tokenizer 处理。",
]


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)


def build_default_text() -> str:
    lines = [
        "人工智能正在改变我们的学习方式。",
        "大语言模型可以根据上下文预测下一个词。",
        "中文 tokenizer 需要处理汉字、英文、数字和标点。",
        "字符级 tokenizer 很简单，但序列长度通常比较长。",
        "BPE tokenizer 会从字符开始，逐步合并高频片段。",
        "如果一个片段经常出现，例如 人工智能 或 语言模型，它可能被合并成更少的 token。",
        "vocab size 越大，BPE 通常可以保存更多常见片段。",
        "special tokens 用来表示 padding、unknown、begin 和 end。",
        "encode 把文本变成 token ids，decode 把 token ids 还原成文本。",
        "后续 Tiny GPT 会读取 BPE token ids，而不是直接读取字符串。",
    ]
    return ("\n".join(lines) + "\n") * 4


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u3000", " ")
    return text


def load_or_create_raw_text() -> str:
    ensure_dirs()

    if RAW_TEXT_PATH.exists():
        return normalize_text(RAW_TEXT_PATH.read_text(encoding="utf-8"))

    if PREVIOUS_RAW_TEXT_PATH.exists():
        shutil.copyfile(PREVIOUS_RAW_TEXT_PATH, RAW_TEXT_PATH)
        return normalize_text(RAW_TEXT_PATH.read_text(encoding="utf-8"))

    RAW_TEXT_PATH.write_text(build_default_text(), encoding="utf-8")
    return normalize_text(RAW_TEXT_PATH.read_text(encoding="utf-8"))


def build_training_corpus(raw_text: str) -> list[str]:
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    lines.extend(SAMPLE_TEXTS)

    # 小语料会让 BPE 很难看出频率，所以这里重复几次教学样本，突出常见片段。
    repeated = []
    for _ in range(8):
        repeated.extend(lines)

    PROCESSED_CORPUS_PATH.write_text("\n".join(repeated) + "\n", encoding="utf-8")
    return repeated


def train_bpe_tokenizer(corpus_lines: list[str], vocab_size: int) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.decoder = decoders.Fuse()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        show_progress=False,
    )
    tokenizer.train_from_iterator(corpus_lines, trainer=trainer)
    return tokenizer


def char_tokenize(text: str) -> list[str]:
    return list(normalize_text(text))


def compression_ratio(char_count: int, bpe_count: int) -> float:
    if char_count == 0:
        return 0.0
    return bpe_count / char_count


def get_interesting_vocab_tokens(tokenizer: Tokenizer, limit: int = 40) -> list[str]:
    vocab = tokenizer.get_vocab()
    tokens = [token for token in vocab if token not in SPECIAL_TOKENS]
    tokens.sort(key=lambda token: (-len(token), token))
    return tokens[:limit]


def analyze_tokenizers(tokenizers_by_size: dict[int, Tokenizer]) -> tuple[list[str], dict]:
    report_lines: list[str] = []
    token_counts: dict[str, list[int]] = {"char": []}
    decoded_ok: dict[int, list[bool]] = {size: [] for size in tokenizers_by_size}

    report_lines.append("=== 中文 BPE / Subword Tokenizer 入门实验 ===")
    report_lines.append("")
    report_lines.append(f"raw text: {RAW_TEXT_PATH}")
    report_lines.append(f"training corpus: {PROCESSED_CORPUS_PATH}")
    report_lines.append(f"special tokens: {SPECIAL_TOKENS}")
    report_lines.append(f"target vocab sizes: {VOCAB_SIZES}")
    report_lines.append("")

    for size in tokenizers_by_size:
        token_counts[f"bpe_{size}"] = []

    for sample_idx, text in enumerate(SAMPLE_TEXTS, start=1):
        normalized = normalize_text(text)
        char_tokens = char_tokenize(normalized)
        char_count = len(char_tokens)
        token_counts["char"].append(char_count)

        report_lines.append(f"--- Sample {sample_idx} ---")
        report_lines.append(f"text: {normalized}")
        report_lines.append(f"char token count: {char_count}")
        report_lines.append(f"char tokens: {char_tokens}")

        for size, tokenizer in tokenizers_by_size.items():
            encoding = tokenizer.encode(normalized)
            bpe_tokens = encoding.tokens
            bpe_ids = encoding.ids
            decoded = tokenizer.decode(bpe_ids, skip_special_tokens=True)
            ok = decoded == normalized
            decoded_ok[size].append(ok)
            token_counts[f"bpe_{size}"].append(len(bpe_tokens))

            report_lines.append("")
            report_lines.append(f"BPE vocab_size={size}")
            report_lines.append(f"actual vocab size: {tokenizer.get_vocab_size()}")
            report_lines.append(f"bpe token count: {len(bpe_tokens)}")
            report_lines.append(f"compression bpe/char: {compression_ratio(char_count, len(bpe_tokens)):.4f}")
            report_lines.append(f"tokens: {bpe_tokens}")
            report_lines.append(f"ids: {bpe_ids}")
            report_lines.append(f"decoded: {decoded}")
            report_lines.append(f"decode reversible: {ok}")

        report_lines.append("")

    report_lines.append("=== BPE vocab 中较长的 token 片段 ===")
    for size, tokenizer in tokenizers_by_size.items():
        report_lines.append("")
        report_lines.append(f"vocab_size={size}, actual={tokenizer.get_vocab_size()}")
        for token in get_interesting_vocab_tokens(tokenizer):
            report_lines.append(f"- {token}")

    avg_counts = {
        name: float(np.mean(values))
        for name, values in token_counts.items()
    }
    report_lines.append("")
    report_lines.append("=== 平均 token 数 ===")
    for name, value in avg_counts.items():
        report_lines.append(f"{name}: {value:.2f}")

    metadata = {
        "raw_text_path": str(RAW_TEXT_PATH),
        "processed_corpus_path": str(PROCESSED_CORPUS_PATH),
        "report_path": str(REPORT_PATH),
        "plot_path": str(PLOT_PATH),
        "special_tokens": SPECIAL_TOKENS,
        "target_vocab_sizes": VOCAB_SIZES,
        "actual_vocab_sizes": {
            str(size): tokenizers_by_size[size].get_vocab_size()
            for size in tokenizers_by_size
        },
        "samples": SAMPLE_TEXTS,
        "token_counts": token_counts,
        "average_token_counts": avg_counts,
        "decode_reversible": {
            str(size): all(values)
            for size, values in decoded_ok.items()
        },
    }

    return report_lines, metadata


def save_token_count_plot(token_counts: dict[str, list[int]]) -> None:
    labels = [f"S{i}" for i in range(1, len(SAMPLE_TEXTS) + 1)]
    series = list(token_counts.keys())
    x = np.arange(len(labels))
    width = 0.18

    plt.figure(figsize=(11, 5))
    for idx, name in enumerate(series):
        offset = (idx - (len(series) - 1) / 2) * width
        plt.bar(x + offset, token_counts[name], width=width, label=name)

    plt.xticks(x, labels)
    plt.ylabel("Token count")
    plt.title("Character tokenizer vs BPE tokenizers")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close()


def main() -> None:
    raw_text = load_or_create_raw_text()
    corpus_lines = build_training_corpus(raw_text)

    tokenizers_by_size: dict[int, Tokenizer] = {}
    for vocab_size in VOCAB_SIZES:
        tokenizer = train_bpe_tokenizer(corpus_lines, vocab_size)
        tokenizers_by_size[vocab_size] = tokenizer
        tokenizer_path = TOKENIZER_DIR / f"chinese_bpe_vocab{vocab_size}.json"
        tokenizer.save(str(tokenizer_path))

    report_lines, metadata = analyze_tokenizers(tokenizers_by_size)
    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    save_token_count_plot(metadata["token_counts"])

    print("=== Chinese BPE Tokenizer Intro ===")
    print("Raw text:", RAW_TEXT_PATH)
    print("Processed corpus:", PROCESSED_CORPUS_PATH)
    print("Tokenizers dir:", TOKENIZER_DIR)
    print("Report:", REPORT_PATH)
    print("Plot:", PLOT_PATH)
    print()
    for size in VOCAB_SIZES:
        print(
            f"vocab_size={size} "
            f"actual={metadata['actual_vocab_sizes'][str(size)]} "
            f"decode_ok={metadata['decode_reversible'][str(size)]}"
        )
    print()
    print("Average token counts:")
    for name, value in metadata["average_token_counts"].items():
        print(f"{name}: {value:.2f}")


if __name__ == "__main__":
    main()
