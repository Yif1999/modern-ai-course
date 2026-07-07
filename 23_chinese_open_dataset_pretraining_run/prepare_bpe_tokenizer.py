from __future__ import annotations

import argparse
import json
import shutil
import unicodedata
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, normalizers, trainers


CURRENT_DIR = Path(__file__).resolve().parent
RAW_DIR = CURRENT_DIR / "data" / "raw"
OUTPUT_DIR = CURRENT_DIR / "outputs"
REPORT_DIR = OUTPUT_DIR / "reports"
TOKENIZER_DIR = OUTPUT_DIR / "tokenizer"
CORPUS_PATH = RAW_DIR / "open_zh_corpus.txt"
TOKENIZER_PATH = TOKENIZER_DIR / "chinese_bpe_tokenizer.json"
REPORT_PATH = REPORT_DIR / "tokenizer_report.txt"
PREVIOUS_TOKENIZER_PATH = (
    CURRENT_DIR.parent
    / "22_train_tiny_gpt_with_chinese_bpe"
    / "outputs"
    / "tokenizers"
    / "chinese_bpe_tokenizer.json"
)

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text.replace("\r\n", "\n").replace("\r", "\n"))


def ensure_dirs() -> None:
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_corpus_lines(max_lines: int | None = None) -> list[str]:
    text = normalize_text(CORPUS_PATH.read_text(encoding="utf-8"))
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[:max_lines] if max_lines else lines


def train_tokenizer(lines: list[str], vocab_size: int) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.decoder = decoders.Fuse()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        show_progress=False,
    )
    tokenizer.train_from_iterator(lines, trainer=trainer)
    tokenizer.save(str(TOKENIZER_PATH))
    return tokenizer


def unknown_ratio(tokenizer: Tokenizer, lines: list[str], sample_count: int = 200) -> float:
    unk_id = tokenizer.token_to_id("<unk>")
    total = 0
    unk = 0
    for line in lines[:sample_count]:
        ids = tokenizer.encode(line).ids
        total += len(ids)
        unk += sum(1 for item in ids if item == unk_id)
    return unk / max(total, 1)


def interesting_tokens(tokenizer: Tokenizer, limit: int = 80) -> list[str]:
    tokens = [tok for tok in tokenizer.get_vocab() if tok not in SPECIAL_TOKENS]
    tokens.sort(key=lambda tok: (-len(tok), tok))
    return tokens[:limit]


def write_report(tokenizer: Tokenizer, source: str, unk_rate: float, lines: list[str]) -> None:
    samples = lines[:5]
    report = []
    report.append("=== Tokenizer Report ===")
    report.append(f"tokenizer source: {source}")
    report.append(f"tokenizer path: {TOKENIZER_PATH}")
    report.append(f"vocab_size: {tokenizer.get_vocab_size()}")
    report.append(f"special tokens: {SPECIAL_TOKENS}")
    report.append(f"estimated unk ratio: {unk_rate:.6f}")
    report.append("")
    ratios = []
    for i, sample in enumerate(samples, start=1):
        normalized = normalize_text(sample)
        char_count = len(normalized)
        enc = tokenizer.encode(normalized)
        decoded = tokenizer.decode(enc.ids, skip_special_tokens=True)
        ratio = len(enc.tokens) / max(char_count, 1)
        ratios.append(ratio)
        report.append(f"--- sample {i} ---")
        report.append(f"text: {normalized[:500]}")
        report.append(f"char token count: {char_count}")
        report.append(f"BPE token count: {len(enc.tokens)}")
        report.append(f"compression BPE/char: {ratio:.4f}")
        report.append(f"tokens: {enc.tokens[:80]}")
        report.append(f"decoded matches normalized prefix: {decoded == normalized}")
        report.append("")
    report.append(f"average compression ratio: {sum(ratios) / max(len(ratios), 1):.4f}")
    report.append("")
    report.append("=== vocab token preview ===")
    for token in interesting_tokens(tokenizer):
        report.append(f"- {token}")
    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")


def prepare_tokenizer(force_train: bool = False, vocab_size: int = 2048, max_train_lines: int | None = None) -> dict:
    ensure_dirs()
    lines = load_corpus_lines(max_train_lines)
    source = "trained_on_current_corpus"

    if not force_train and PREVIOUS_TOKENIZER_PATH.exists():
        shutil.copyfile(PREVIOUS_TOKENIZER_PATH, TOKENIZER_PATH)
        tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
        source = "copied_from_lesson_22"
        unk_rate = unknown_ratio(tokenizer, lines)
        if unk_rate > 0.01:
            tokenizer = train_tokenizer(lines, vocab_size=vocab_size)
            source = f"trained_on_current_corpus_due_to_unk_ratio_{unk_rate:.4f}"
            unk_rate = unknown_ratio(tokenizer, lines)
    elif TOKENIZER_PATH.exists() and not force_train:
        tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
        source = "loaded_from_current_course"
        unk_rate = unknown_ratio(tokenizer, lines)
    else:
        tokenizer = train_tokenizer(lines, vocab_size=vocab_size)
        unk_rate = unknown_ratio(tokenizer, lines)

    write_report(tokenizer, source, unk_rate, lines)
    meta = {
        "tokenizer_path": str(TOKENIZER_PATH),
        "tokenizer_source": source,
        "vocab_size": tokenizer.get_vocab_size(),
        "unk_ratio": unk_rate,
        "report_path": str(REPORT_PATH),
    }
    (REPORT_DIR / "tokenizer_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print("=== BPE Tokenizer Prepared ===")
    for key, value in meta.items():
        print(f"{key}: {value}")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--vocab-size", type=int, default=2048)
    parser.add_argument("--max-train-lines", type=int, default=None)
    args = parser.parse_args()
    prepare_tokenizer(force_train=args.force_train, vocab_size=args.vocab_size, max_train_lines=args.max_train_lines)


if __name__ == "__main__":
    main()
