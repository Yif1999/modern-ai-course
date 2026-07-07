from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import random
import re
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


CURRENT_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = CURRENT_DIR.parent
DATA_DIR = CURRENT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"
TOKENIZER_DIR = DATA_DIR / "tokenizers"
CACHE_DIR = DATA_DIR / "cache"
REPORT_DIR = CURRENT_DIR / "outputs" / "reports"

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]

TEXT_FIELD_CANDIDATES = [
    "text",
    "content",
    "document",
    "raw_content",
    "article",
    "body",
    "title",
]

GOVERNMENT_KEYWORDS = [
    "人民政府",
    "政府工作报告",
    "工作报告",
    "政策解读",
    "政策法规",
    "法律法规",
    "管理办法",
    "实施方案",
    "实施意见",
    "通知公告",
    "公告",
    "公示",
    "招标",
    "投标",
    "中标",
    "采购项目",
    "项目编号",
    "预算金额",
    "财政局",
    "税务局",
    "教育局",
    "公安局",
    "人社局",
    "住建局",
    "生态环境局",
    "县委",
    "市委",
    "党委",
    "党支部",
    "政协",
    "人大常委会",
    "习近平",
    "贯彻落实",
    "会议精神",
    "责任单位",
    "领导小组",
    "监督检查",
    "依法",
    "条例",
    "规划纲要",
]

BOILERPLATE_PATTERNS = [
    r"版权所有",
    r"ICP备案",
    r"网站地图",
    r"联系我们",
    r"责任编辑",
    r"来源[:：]",
    r"分享到",
    r"打印本页",
]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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


def rough_tokenish_len(text: str) -> int:
    return len([ch for ch in text if not ch.isspace()])


def normalize_preserve_punctuation(text: str) -> str:
    """基础清洗：保留中文标点，不做 NFKC 全角/半角归一化。"""

    text = html.unescape(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    for pattern in BOILERPLATE_PATTERNS:
        text = re.sub(pattern, " ", text)
    text = re.sub(r"[\t\f\v]+", " ", text)
    text = normalize_chinese_spacing(text)
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_chinese_spacing(text: str) -> str:
    """清理中文语料中来自分词/转写的异常空格，但保留正常换行和英文词内空格。"""

    cjk = r"\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
    zh_punct = "，。！？；：、"
    all_punct = "，。！？；：、,.!?;:"

    # 中文标点前后通常不需要半角空格，例如 "你好， 世界" -> "你好，世界"。
    text = re.sub(fr"[ \t]+([{re.escape(all_punct)}])", r"\1", text)
    text = re.sub(fr"([{re.escape(zh_punct)}])[ \t]+(?=[{cjk}A-Za-z0-9“‘\"（(《])", r"\1", text)
    text = re.sub(fr"(?<=[{cjk}])([,.!?;:])[ \t]+(?=[{cjk}])", r"\1", text)

    # LCCC 等社交对话源常带中文分词空格，例如 "我 在 重庆" -> "我在重庆"。
    text = re.sub(fr"(?<=[{cjk}])[ \t]+(?=[{cjk}])", "", text)

    # 聊天语料里中文和数字/英文通常连写，例如 "18 岁" -> "18岁"、"用 MLX" -> "用MLX"。
    # 但保留英文词组内部空格，例如 "MacBook Pro"。
    text = re.sub(fr"(?<=[{cjk}])[ \t]+(?=[A-Za-z0-9])", "", text)
    text = re.sub(fr"(?<=[A-Za-z0-9])[ \t]+(?=[{cjk}])", "", text)

    # 清理被空格拆开的中文省略号/间隔点，例如 "… …" -> "……"。
    text = re.sub(r"([…⋯])[ \t]+(?=[…⋯])", r"\1", text)
    text = re.sub(r"([·・])[ \t]+(?=[·・])", r"\1", text)

    text = re.sub(r"[ \t]{2,}", " ", text)
    return text


def score_government_style(text: str) -> dict[str, Any]:
    matched = [kw for kw in GOVERNMENT_KEYWORDS if kw in text]
    keyword_score = len(matched)
    formal_hits = len(re.findall(r"(第[一二三四五六七八九十\d]+条|本办法|本条例|现将|特此|予以|以下简称)", text))
    punctuation_density = text.count("：") + text.count("；") + text.count("、")
    score = keyword_score * 2 + formal_hits + min(punctuation_density // 12, 3)
    return {
        "score": score,
        "matched_keywords": matched[:8],
        "keyword_count": keyword_score,
        "formal_hits": formal_hits,
    }


def is_too_repetitive(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    if len(compact) < 120:
        return False
    chars = {}
    for ch in compact:
        chars[ch] = chars.get(ch, 0) + 1
    if max(chars.values()) / len(compact) > 0.25:
        return True
    chunks = [compact[i : i + 20] for i in range(0, len(compact) - 20, 20)]
    return len(chunks) > 8 and len(set(chunks)) / len(chunks) < 0.45


def clean_general_text(text: str, min_len: int, max_len: int, min_zh_ratio: float) -> tuple[str | None, str]:
    text = normalize_preserve_punctuation(text)
    if len(text) < min_len:
        return None, "too_short"
    if chinese_ratio(text) < min_zh_ratio:
        return None, "low_chinese_ratio"
    if is_too_repetitive(text):
        return None, "repetitive"
    if len(text) > max_len:
        text = text[:max_len].strip()
    return text, "ok"


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


def detect_text_field(example: dict[str, Any]) -> tuple[str | None, str | None]:
    for key in TEXT_FIELD_CANDIDATES:
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            return key, value
    strings = extract_strings(example)
    if not strings:
        return None, None
    strings.sort(key=lambda item: len(item[1]), reverse=True)
    return strings[0]


def fingerprint(text: str) -> str:
    compact = re.sub(r"\s+", "", text)
    return hashlib.sha1(compact.encode("utf-8")).hexdigest()


def split_local_general(path: Path) -> list[str]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    chunks = [item.strip() for item in re.split(r"\n\s*\n|\n", text) if item.strip()]
    merged = []
    buf: list[str] = []
    buf_len = 0
    for chunk in chunks:
        if len(chunk) < 50:
            buf.append(chunk)
            buf_len += len(chunk)
            continue
        if buf:
            merged.append("\n".join(buf))
            buf = []
            buf_len = 0
        merged.append(chunk)
        if buf_len > 800:
            merged.append("\n".join(buf))
            buf = []
            buf_len = 0
    if buf:
        merged.append("\n".join(buf))
    return merged


def chunk_fun_text(text: str, max_chars: int = 1600) -> list[str]:
    lines = [line.rstrip() for line in text.splitlines()]
    docs = []
    buf: list[str] = []
    buf_len = 0
    for line in lines:
        if not line.strip():
            if buf:
                docs.append("\n".join(buf).strip())
                buf = []
                buf_len = 0
            continue
        if buf and buf_len + len(line) > max_chars:
            docs.append("\n".join(buf).strip())
            buf = []
            buf_len = 0
        buf.append(line)
        buf_len += len(line)
    if buf:
        docs.append("\n".join(buf).strip())
    return [doc for doc in docs if doc]


def add_doc(
    docs: list[dict[str, Any]],
    seen: set[str],
    text: str,
    source_type: str,
    source_name: str,
    stats: dict[str, int],
    *,
    gov_share_limit: float,
    allow_fun: bool = False,
) -> bool:
    if allow_fun:
        cleaned = normalize_preserve_punctuation(text)
        reason = "ok"
    else:
        cleaned, reason = clean_general_text(text, min_len=80, max_len=3600, min_zh_ratio=0.25)
    if cleaned is None:
        stats[f"reject_{reason}"] = stats.get(f"reject_{reason}", 0) + 1
        return False

    fp = fingerprint(cleaned)
    if fp in seen:
        stats["reject_duplicate"] = stats.get("reject_duplicate", 0) + 1
        return False

    gov = score_government_style(cleaned)
    doc_kind = "fun" if allow_fun else "general"
    if not allow_fun and gov["score"] >= 4:
        current_gov_chars = sum(item["char_count"] for item in docs if item["doc_kind"] == "government_limited")
        current_total_chars = max(1, sum(item["char_count"] for item in docs if item["doc_kind"] != "fun"))
        if current_gov_chars / current_total_chars >= gov_share_limit:
            stats["reject_government_over_cap"] = stats.get("reject_government_over_cap", 0) + 1
            return False
        doc_kind = "government_limited"

    seen.add(fp)
    docs.append(
        {
            "id": len(docs),
            "text": cleaned,
            "source_type": source_type,
            "source_name": source_name,
            "doc_kind": doc_kind,
            "char_count": len(cleaned),
            "chinese_ratio": chinese_ratio(cleaned),
            "government_score": gov["score"],
            "government_keywords": gov["matched_keywords"],
        }
    )
    stats[f"keep_{doc_kind}"] = stats.get(f"keep_{doc_kind}", 0) + 1
    stats[f"keep_{source_type}_chars"] = stats.get(f"keep_{source_type}_chars", 0) + len(cleaned)
    return True


def collect_existing_general(docs: list[dict[str, Any]], seen: set[str], stats: dict[str, int], gov_share_limit: float) -> None:
    candidates = [
        PROJECT_DIR / "23_chinese_open_dataset_pretraining_run/data/raw/open_zh_corpus.txt",
        PROJECT_DIR / "24_chinese_gpt_scaling_on_m1_pro/data/raw/open_zh_corpus.txt",
    ]
    seen_source_hashes = set()
    for path in candidates:
        if not path.exists():
            continue
        source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if source_hash in seen_source_hashes:
            stats["skip_duplicate_corpus_file"] = stats.get("skip_duplicate_corpus_file", 0) + 1
            continue
        seen_source_hashes.add(source_hash)
        for piece in split_local_general(path):
            stats["existing_general_candidates"] = stats.get("existing_general_candidates", 0) + 1
            add_doc(
                docs,
                seen,
                piece,
                "existing_general",
                str(path.relative_to(PROJECT_DIR)),
                stats,
                gov_share_limit=gov_share_limit,
            )


def collect_fun(docs: list[dict[str, Any]], seen: set[str], stats: dict[str, int], max_fun_chars: int | None) -> None:
    path = PROJECT_DIR / "28_chinese_fun_corpus_pipeline/data/processed/fun_corpus.txt"
    if not path.exists():
        stats["fun_missing"] = 1
        return
    text = path.read_text(encoding="utf-8", errors="replace")
    if max_fun_chars:
        text = text[:max_fun_chars]
    for piece in chunk_fun_text(text):
        stats["fun_candidates"] = stats.get("fun_candidates", 0) + 1
        add_doc(
            docs,
            seen,
            piece,
            "fun_corpus",
            str(path.relative_to(PROJECT_DIR)),
            stats,
            gov_share_limit=1.0,
            allow_fun=True,
        )


def stream_general(
    docs: list[dict[str, Any]],
    seen: set[str],
    stats: dict[str, int],
    *,
    dataset_name: str,
    target_general_chars: int,
    max_stream_docs: int,
    max_stream_raw_chars: int,
    gov_share_limit: float,
) -> dict[str, Any]:
    from datasets import load_dataset

    os.environ.setdefault("HF_DATASETS_CACHE", str(CACHE_DIR / "datasets"))
    started = time.time()
    report = {
        "dataset_name": dataset_name,
        "streaming": True,
        "error": None,
        "raw_docs_seen": 0,
        "raw_chars_seen": 0,
        "field_counts": {},
    }

    def general_chars() -> int:
        return sum(item["char_count"] for item in docs if item["doc_kind"] in {"general", "government_limited"})

    if general_chars() >= target_general_chars:
        report["skipped_reason"] = "target already satisfied by local corpora"
        return report

    print("Streaming general dataset:", dataset_name)
    print("Target general chars:", target_general_chars)
    ds = load_dataset(dataset_name, split="train", streaming=True, cache_dir=str(CACHE_DIR / "datasets"))
    for row in ds:
        report["raw_docs_seen"] += 1
        if report["raw_docs_seen"] > max_stream_docs:
            break
        field_name, raw_text = detect_text_field(row if isinstance(row, dict) else {})
        if not raw_text:
            stats["stream_reject_no_text"] = stats.get("stream_reject_no_text", 0) + 1
            continue
        report["field_counts"][field_name or "<unknown>"] = report["field_counts"].get(field_name or "<unknown>", 0) + 1
        report["raw_chars_seen"] += len(raw_text)
        stats["stream_candidates"] = stats.get("stream_candidates", 0) + 1
        add_doc(
            docs,
            seen,
            raw_text,
            "streamed_general",
            dataset_name,
            stats,
            gov_share_limit=gov_share_limit,
        )
        if report["raw_docs_seen"] % 1000 == 0:
            print(
                "stream progress:",
                "raw_docs=", report["raw_docs_seen"],
                "general_chars=", general_chars(),
                "docs=", len(docs),
            )
        if general_chars() >= target_general_chars:
            break
        if report["raw_chars_seen"] >= max_stream_raw_chars:
            break
    report["elapsed_sec"] = time.time() - started
    report["final_general_chars"] = general_chars()
    return report


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


def token_preview(tokenizer: Tokenizer, samples: list[str]) -> list[dict[str, Any]]:
    out = []
    for text in samples:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(encoded.ids, skip_special_tokens=True)
        out.append(
            {
                "text": text,
                "ids": encoded.ids[:80],
                "tokens": encoded.tokens[:80],
                "token_count": len(encoded.ids),
                "decoded": decoded,
                "roundtrip_ok": decoded == text,
            }
        )
    return out


def top_source_summary(docs: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for item in docs:
        key = item["doc_kind"]
        if key not in out:
            out[key] = {"docs": 0, "chars": 0}
        out[key]["docs"] += 1
        out[key]["chars"] += item["char_count"]
    return out


def write_docs_jsonl(path: Path, docs: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in docs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_preview(path: Path, docs: list[dict[str, Any]], limit: int = 30) -> None:
    lines = ["# Mixture v2 Preview", ""]
    for item in docs[:limit]:
        lines.extend(
            [
                f"## doc {item['id']} | {item['doc_kind']} | chars={item['char_count']} | gov_score={item['government_score']}",
                "",
                item["text"][:1000],
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare lesson 30 mixture v2 with more general Chinese and all fun corpus.")
    parser.add_argument("--dataset-name", default="opencsg/chinese-fineweb-edu")
    parser.add_argument("--target-general-chars", type=int, default=18_000_000)
    parser.add_argument("--max-stream-docs", type=int, default=80_000)
    parser.add_argument("--max-stream-raw-chars", type=int, default=80_000_000)
    parser.add_argument("--gov-share-limit", type=float, default=0.12)
    parser.add_argument("--max-fun-chars", type=int, default=0, help="0 means use all available fun corpus.")
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=2030)
    parser.add_argument("--no-stream", action="store_true")
    args = parser.parse_args()

    for path in [RAW_DIR, PROCESSED_DIR, METADATA_DIR, TOKENIZER_DIR, CACHE_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    docs: list[dict[str, Any]] = []
    seen: set[str] = set()
    stats: dict[str, int] = {}

    collect_existing_general(docs, seen, stats, gov_share_limit=args.gov_share_limit)
    collect_fun(docs, seen, stats, max_fun_chars=args.max_fun_chars or None)

    stream_report: dict[str, Any] = {"streaming": False, "skipped": args.no_stream}
    if not args.no_stream:
        try:
            stream_report = stream_general(
                docs,
                seen,
                stats,
                dataset_name=args.dataset_name,
                target_general_chars=args.target_general_chars,
                max_stream_docs=args.max_stream_docs,
                max_stream_raw_chars=args.max_stream_raw_chars,
                gov_share_limit=args.gov_share_limit,
            )
        except Exception as exc:  # noqa: BLE001
            stream_report = {
                "streaming": True,
                "dataset_name": args.dataset_name,
                "error": f"{type(exc).__name__}: {exc}",
            }
            print("Streaming failed:", stream_report["error"])

    if not docs:
        raise RuntimeError("No documents collected for mixture v2.")

    random.shuffle(docs)
    for i, item in enumerate(docs):
        item["id"] = i

    docs_jsonl_path = RAW_DIR / "lab_mixture_v2_docs.jsonl"
    corpus_path = RAW_DIR / "lab_mixture_v2_corpus.txt"
    preview_path = REPORT_DIR / "lab_mixture_v2_preview.md"
    write_docs_jsonl(docs_jsonl_path, docs)
    write_preview(preview_path, docs)
    corpus = "\n\n<|doc_sep|>\n\n".join(item["text"] for item in docs) + "\n"
    corpus_path.write_text(corpus, encoding="utf-8")

    tokenizer = train_byte_bpe(corpus_path, args.vocab_size)
    actual_vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    tokenizer_path = TOKENIZER_DIR / f"lab_mixture_v2_byte_bpe_{actual_vocab_size}.json"
    tokenizer.save(str(tokenizer_path))

    encoded = tokenizer.encode(corpus, add_special_tokens=False)
    token_ids = encoded.ids
    split = int(len(token_ids) * (1.0 - args.val_ratio))
    split = min(max(split, 2048), len(token_ids) - 2048)
    train = np.array(token_ids[:split], dtype=np.int32)
    val = np.array(token_ids[split:], dtype=np.int32)

    train_path = PROCESSED_DIR / f"train_tokens_lab_mixture_v2_bpe_{actual_vocab_size}.npy"
    val_path = PROCESSED_DIR / f"val_tokens_lab_mixture_v2_bpe_{actual_vocab_size}.npy"
    np.save(train_path, train)
    np.save(val_path, val)

    unk_id = tokenizer.token_to_id("<unk>")
    unk_count = int(sum(1 for token_id in token_ids if token_id == unk_id)) if unk_id is not None else 0
    kind_summary = top_source_summary(docs)
    char_counts = [item["char_count"] for item in docs]
    gov_docs = [item for item in docs if item["doc_kind"] == "government_limited"]
    samples = [
        "人工智能正在改变我们的学习方式。",
        "今天我们用 MLX 在 MacBook Pro 上训练一个中文 Tiny GPT。",
        "老哥稳，这波属于是把本地小模型压榨到极限了。",
        "中文标点应该保留：逗号，句号。感叹号！",
    ]
    preview = token_preview(tokenizer, samples)

    metadata = {
        "tokenizer_type": "lab_bpe",
        "tokenizer_name": f"lab_mixture_v2_byte_bpe_{actual_vocab_size}",
        "tokenizer_path": str(tokenizer_path),
        "vocab_size": actual_vocab_size,
        "requested_vocab_size": args.vocab_size,
        "special_tokens": SPECIAL_TOKENS,
        "unk_id": unk_id,
        "unk_count": unk_count,
        "unk_ratio": unk_count / len(token_ids),
        "corpus_version": "lab_mixture_v2_general_heavy_fun_all",
        "corpus_path": str(corpus_path),
        "docs_jsonl_path": str(docs_jsonl_path),
        "preview_path": str(preview_path),
        "raw_chars": len(corpus),
        "document_count": len(docs),
        "avg_doc_chars": statistics.mean(char_counts) if char_counts else 0,
        "median_doc_chars": statistics.median(char_counts) if char_counts else 0,
        "total_tokens": int(len(token_ids)),
        "chars_per_token": len(corpus) / len(token_ids),
        "train_tokens": int(train.shape[0]),
        "val_tokens": int(val.shape[0]),
        "train_tokens_path": str(train_path),
        "val_tokens_path": str(val_path),
        "val_ratio": args.val_ratio,
        "target_general_chars": args.target_general_chars,
        "gov_share_limit": args.gov_share_limit,
        "kind_summary": kind_summary,
        "government_limited_docs": len(gov_docs),
        "stats": stats,
        "stream_report": stream_report,
        "preview": preview,
        "normalization": "preserve Chinese punctuation; no NFKC; remove abnormal spaces around Chinese punctuation, between CJK characters, and at CJK/ASCII boundaries",
        "pre_tokenizer": "ByteLevel(add_prefix_space=False)",
        "note": "Mixture v2 keeps all available fun corpus, increases filtered general Chinese, and caps government/report-style documents.",
    }
    metadata_path = METADATA_DIR / f"lab_mixture_v2_bpe_{actual_vocab_size}_metadata.json"
    write_json(metadata_path, metadata)

    report_lines = [
        "# Lab Mixture v2 数据报告",
        "",
        "## 结论",
        "",
        "- 本版本重新构建数据 mixture：通用中文为主，趣味语料全部加入。",
        "- 通用语料经过政府/报告腔识别；这类文本没有完全删除，但被比例上限限制。",
        "- 清洗不使用 `NFKC`，因此中文逗号、句号、感叹号等全角标点会保留。",
        "- 清洗会去掉中文标点前后的异常半角空格，去掉中文字符之间由分词/转写引入的空格，也会去掉中文和数字/英文边界的空格。",
        "",
        "## 规模",
        "",
        f"- 文档数：`{len(docs)}`",
        f"- 字符数：`{len(corpus)}`",
        f"- token 数：`{len(token_ids)}`",
        f"- vocab_size：`{actual_vocab_size}`",
        f"- train tokens：`{train.shape[0]}`",
        f"- val tokens：`{val.shape[0]}`",
        f"- chars/token：`{len(corpus) / len(token_ids):.4f}`",
        f"- `<unk>` ratio：`{unk_count / len(token_ids):.8f}`",
        "",
        "## Mixture 构成",
        "",
    ]
    total_chars = max(1, sum(item["chars"] for item in kind_summary.values()))
    for kind, payload in sorted(kind_summary.items()):
        report_lines.append(f"- `{kind}`：docs=`{payload['docs']}`, chars=`{payload['chars']}`, share=`{payload['chars'] / total_chars:.2%}`")
    report_lines.extend(
        [
            "",
            "## Streaming 报告",
            "",
            "```json",
            json.dumps(stream_report, ensure_ascii=False, indent=2),
            "```",
            "",
            "## 过滤统计",
            "",
            "```json",
            json.dumps(stats, ensure_ascii=False, indent=2),
            "```",
            "",
            "## Tokenizer 样本",
            "",
        ]
    )
    for item in preview:
        report_lines.extend(
            [
                f"### {item['text']}",
                "",
                f"- token_count：`{item['token_count']}`",
                f"- roundtrip_ok：`{item['roundtrip_ok']}`",
                f"- tokens：`{item['tokens'][:40]}`",
                f"- decoded：`{item['decoded']}`",
                "",
            ]
        )
    report_path = REPORT_DIR / "lab_mixture_v2_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print("metadata:", metadata_path)
    print("report:", report_path)


if __name__ == "__main__":
    main()
