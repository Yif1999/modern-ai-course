from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


CURRENT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = CURRENT_DIR / "data" / "raw"
TOKENIZER_DIR = CURRENT_DIR / "data" / "tokenizers"
REPORT_DIR = CURRENT_DIR / "outputs" / "reports"
METADATA_DIR = CURRENT_DIR / "data" / "metadata"

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]
SAMPLES = [
    "甲：你觉得人工智能以后会改变什么？\n乙：",
    "老哥稳，这波属于是把本地小模型压榨到极限了。",
    "乙：嗯，这边只有十度，昨天8度！",
    "中文、English、数字123和标点符号！都需要 tokenizer 处理。",
    "哈哈哈哈，这是什么离谱操作.jpg",
]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def train_byte_bpe(corpus_path: Path, vocab_size: int) -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token="<unk>", byte_fallback=True))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train([str(corpus_path)], trainer=trainer)
    return tokenizer


def load_or_train(corpus_path: Path, vocab_size: int, reuse: bool) -> tuple[Tokenizer, Path, float, bool]:
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer_path = TOKENIZER_DIR / f"lab_byte_bpe_{vocab_size}.json"
    started = time.perf_counter()
    if reuse and tokenizer_path.exists():
        return Tokenizer.from_file(str(tokenizer_path)), tokenizer_path, time.perf_counter() - started, True

    tokenizer = train_byte_bpe(corpus_path, vocab_size)
    actual_vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    tokenizer_path = TOKENIZER_DIR / f"lab_byte_bpe_{actual_vocab_size}.json"
    tokenizer.save(str(tokenizer_path))
    return tokenizer, tokenizer_path, time.perf_counter() - started, False


def sample_preview(tokenizer: Tokenizer) -> list[dict[str, Any]]:
    rows = []
    for text in SAMPLES:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(encoded.ids, skip_special_tokens=True)
        rows.append(
            {
                "text": text,
                "token_count": len(encoded.ids),
                "ids": encoded.ids[:80],
                "tokens": encoded.tokens[:80],
                "decoded": decoded,
                "roundtrip_ok": decoded == text,
            }
        )
    return rows


def read_existing_32k_metadata() -> dict[str, Any] | None:
    path = METADATA_DIR / "lab_bpe_32768_metadata.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def compare_vocab_sizes(corpus_path: Path, vocab_sizes: list[int], reuse: bool) -> dict[str, Any]:
    corpus = corpus_path.read_text(encoding="utf-8")
    raw_chars = len(corpus)
    metadata_32k = read_existing_32k_metadata()
    rows = []

    for vocab_size in vocab_sizes:
        row_started = time.perf_counter()
        tokenizer, tokenizer_path, tokenizer_elapsed, reused = load_or_train(corpus_path, vocab_size, reuse)
        actual_vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)

        if actual_vocab_size == 32768 and metadata_32k and metadata_32k.get("corpus_path") == str(corpus_path):
            total_tokens = int(metadata_32k["total_tokens"])
            encode_elapsed = 0.0
            used_existing_count = True
        else:
            encode_started = time.perf_counter()
            encoded = tokenizer.encode(corpus, add_special_tokens=False)
            total_tokens = len(encoded.ids)
            encode_elapsed = time.perf_counter() - encode_started
            used_existing_count = False

        unk_id = tokenizer.token_to_id("<unk>")
        preview = sample_preview(tokenizer)
        rows.append(
            {
                "requested_vocab_size": vocab_size,
                "actual_vocab_size": actual_vocab_size,
                "tokenizer_path": str(tokenizer_path),
                "tokenizer_reused": reused,
                "used_existing_count": used_existing_count,
                "raw_chars": raw_chars,
                "total_tokens": total_tokens,
                "chars_per_token": raw_chars / total_tokens,
                "unk_id": unk_id,
                "sample_preview": preview,
                "train_or_load_elapsed_sec": tokenizer_elapsed,
                "encode_elapsed_sec": encode_elapsed,
                "total_elapsed_sec": time.perf_counter() - row_started,
            }
        )

    baseline = next((row for row in rows if row["actual_vocab_size"] == 32768), None)
    if baseline:
        base_tokens = baseline["total_tokens"]
        base_params = 32768 * 640
        for row in rows:
            row["token_increase_vs_32768"] = row["total_tokens"] / base_tokens - 1.0
            row["embedding_params_at_n_embd_640"] = row["actual_vocab_size"] * 640
            row["embedding_param_saving_vs_32768"] = 1.0 - (
                row["embedding_params_at_n_embd_640"] / base_params
            )

    return {
        "corpus_path": str(corpus_path),
        "raw_chars": raw_chars,
        "vocab_sizes": rows,
        "decision_rule": "Prefer smaller vocab only if total token increase is less than roughly 15%-20% versus 32k.",
    }


def write_report(payload: dict[str, Any], path: Path) -> None:
    rows = payload["vocab_sizes"]
    lines = [
        "# Lab BPE 词表规模对比",
        "",
        "## 结论口径",
        "",
        "这个报告只比较 tokenizer，不训练模型。",
        "",
        "判断规则：如果 16k / 8k 相比 32k 的 token 数增加小于约 15%-20%，缩小词表大概率值得；如果 token 数膨胀太多，省下来的 embedding / softmax 成本可能会被更长序列吃回去。",
        "",
        "## 总览",
        "",
        "| vocab | tokens | chars/token | token 增幅 vs 32k | embedding 参数节省 vs 32k | tokenizer |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        inc = row.get("token_increase_vs_32768")
        saving = row.get("embedding_param_saving_vs_32768")
        inc_text = "n/a" if inc is None else f"{inc * 100:.2f}%"
        saving_text = "n/a" if saving is None else f"{saving * 100:.2f}%"
        lines.append(
            f"| {row['actual_vocab_size']:,} | {row['total_tokens']:,} | "
            f"{row['chars_per_token']:.4f} | {inc_text} | {saving_text} | "
            f"`{Path(row['tokenizer_path']).name}` |"
        )

    lines.extend(["", "## 样本 token 数", ""])
    for row in rows:
        lines.extend([f"### vocab {row['actual_vocab_size']:,}", ""])
        lines.append("| text | token_count | roundtrip |")
        lines.append("|---|---:|---:|")
        for sample in row["sample_preview"]:
            text = sample["text"].replace("\n", " / ")
            lines.append(f"| {text} | {sample['token_count']} | {sample['roundtrip_ok']} |")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare lab Byte-BPE vocab sizes on the current canonical corpus.")
    parser.add_argument("--corpus", default=str(RAW_DIR / "lab_bpe_mixed_corpus.txt"))
    parser.add_argument("--vocab-sizes", nargs="+", type=int, default=[32768, 16384, 8192])
    parser.add_argument("--reuse", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        raise FileNotFoundError(corpus_path)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    payload = compare_vocab_sizes(corpus_path, args.vocab_sizes, args.reuse)
    json_path = REPORT_DIR / "lab_bpe_vocab_size_comparison.json"
    md_path = REPORT_DIR / "lab_bpe_vocab_size_comparison.md"
    write_json(json_path, payload)
    write_report(payload, md_path)
    print("json:", json_path)
    print("report:", md_path)
    for row in payload["vocab_sizes"]:
        inc = row.get("token_increase_vs_32768")
        inc_text = "n/a" if inc is None else f"{inc * 100:.2f}%"
        print(
            f"vocab={row['actual_vocab_size']:,} tokens={row['total_tokens']:,} "
            f"chars/token={row['chars_per_token']:.4f} increase_vs_32k={inc_text}"
        )


if __name__ == "__main__":
    main()
