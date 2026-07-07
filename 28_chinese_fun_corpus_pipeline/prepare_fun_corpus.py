from __future__ import annotations

import argparse
import json
import math
import re
import time
import unicodedata
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Image, load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

try:
    from transformers import AutoTokenizer
except ImportError:  # pragma: no cover - optional dependency for Qwen compatibility stats
    AutoTokenizer = None


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent
DATA_DIR = CURRENT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"
CACHE_DIR = DATA_DIR / "cache"
OUTPUT_DIR = CURRENT_DIR / "outputs"
REPORT_DIR = OUTPUT_DIR / "reports"
SAMPLE_DIR = OUTPUT_DIR / "samples"

SOURCE_CONFIG_PATH = CURRENT_DIR / "data_sources.json"


def ensure_dirs() -> None:
    for path in [RAW_DIR, PROCESSED_DIR, METADATA_DIR, CACHE_DIR, REPORT_DIR, SAMPLE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def json_dump(obj: Any, path: Path) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def chinese_ratio(text: str) -> float:
    if not text:
        return 0.0
    chinese = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    letters_digits = sum(1 for ch in text if ("\u4e00" <= ch <= "\u9fff") or ch.isascii() and ch.isalnum())
    denom = max(letters_digits, 1)
    return chinese / denom


def normalize_text(text: str) -> str:
    # Use NFC instead of NFKC: NFKC folds full-width Chinese punctuation
    # such as "，" into ASCII ",". For this corpus we want to preserve
    # Chinese typography and internet writing style.
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = normalize_chinese_spacing(text)
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


def clean_text(text: str, min_len: int, max_len: int, min_zh_ratio: float) -> str | None:
    text = normalize_text(text)
    if len(text) < min_len or len(text) > max_len:
        return None
    if chinese_ratio(text) < min_zh_ratio:
        return None
    return text


def flatten_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            out.extend(flatten_strings(item))
        return out
    if isinstance(value, dict):
        out = []
        for item in value.values():
            out.extend(flatten_strings(item))
        return out
    return []


def text_from_fields(record: dict[str, Any], fields: list[str]) -> str:
    parts: list[str] = []
    for field in fields:
        value = record.get(field)
        for item in flatten_strings(value):
            item = normalize_text(item)
            if item:
                parts.append(item)
    return "\n".join(parts)


def chime_record_to_text(record: dict[str, Any]) -> str:
    parts = []
    meme = normalize_text(str(record.get("meme", "")))
    meaning = normalize_text(str(record.get("meaning", "")))
    origin = normalize_text(str(record.get("origin", "")))
    type_cn = normalize_text(str(record.get("type_cn", "")))
    examples = [normalize_text(x) for x in record.get("examples", []) if normalize_text(str(x))]
    if meme:
        parts.append(f"梗：{meme}")
    if type_cn:
        parts.append(f"类型：{type_cn}")
    if meaning:
        parts.append(f"含义：{meaning}")
    if origin:
        parts.append(f"来源：{origin}")
    if examples:
        parts.append("例句：" + " / ".join(examples))
    return "\n".join(parts)


def load_github_json(source: dict[str, Any]) -> list[dict[str, Any]]:
    url = source["url"]
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = response.read().decode("utf-8")
    data = json.loads(payload)
    if isinstance(data, dict):
        data = list(data.values())
    records = []
    for item in data[: source.get("max_samples", len(data))]:
        if not isinstance(item, dict):
            continue
        if source["source_name"] == "CHIME":
            text = chime_record_to_text(item)
        else:
            text = "\n".join(flatten_strings(item))
        records.append({"text": text, "raw": item})
    return records


def load_hf_dataset(source: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset_name = source["dataset_name"]
    config_name = source.get("config_name")
    split = source.get("split", "train")
    max_samples = int(source.get("max_samples", 1000))
    text_fields = list(source.get("text_fields", []))
    decode_images = bool(source.get("decode_images", True))
    load_kwargs: dict[str, Any] = {
        "split": split,
        "streaming": True,
        "cache_dir": str(CACHE_DIR),
    }
    if config_name:
        dataset = load_dataset(dataset_name, config_name, **load_kwargs)
    else:
        dataset = load_dataset(dataset_name, **load_kwargs)

    feature_names = []
    try:
        if dataset.features:
            feature_names = list(dataset.features.keys())
            if not decode_images:
                for key, feature in dataset.features.items():
                    if "Image" in type(feature).__name__ or key.lower() in {"image", "img", "picture"}:
                        dataset = dataset.cast_column(key, Image(decode=False))
    except Exception:
        feature_names = []

    records: list[dict[str, Any]] = []
    seen = 0
    for item in dataset:
        if seen >= max_samples:
            break
        seen += 1
        if not isinstance(item, dict):
            continue
        if text_fields:
            text = text_from_fields(item, text_fields)
        else:
            # If a dataset has no configured text fields, only use obvious string fields.
            inferred_fields = [
                key
                for key, value in item.items()
                if isinstance(value, str)
                and key.lower() not in {"image", "img", "path", "url", "filename", "file_name"}
            ]
            text = text_from_fields(item, inferred_fields)
        records.append({"text": text, "raw": {k: str(v)[:500] for k, v in item.items() if k != "image"}})

    info = {"feature_names": feature_names, "iterated_samples": seen}
    return records, info


def collect_sources(
    sources: list[dict[str, Any]],
    max_docs: int,
    max_chars: int,
    min_len: int,
    max_len: int,
    min_zh_ratio: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    all_records: list[dict[str, Any]] = []
    seen_texts: set[str] = set()
    source_stats: dict[str, Any] = {}
    total_chars = 0

    for source in sources:
        name = source["source_name"]
        stats = {
            "source_name": name,
            "source_type": source["source_type"],
            "enabled": source.get("enabled", False),
            "loaded_records": 0,
            "kept_records": 0,
            "empty_or_no_text": 0,
            "filtered_records": 0,
            "duplicates": 0,
            "error": None,
            "notes": source.get("notes", ""),
        }
        source_stats[name] = stats
        if not source.get("enabled", False):
            continue
        if len(all_records) >= max_docs or total_chars >= max_chars:
            stats["error"] = "global limit reached before this source"
            continue

        try:
            if source["loader"] == "github_json":
                records = load_github_json(source)
                extra_info = {}
            elif source["loader"] == "hf_dataset":
                records, extra_info = load_hf_dataset(source)
                stats.update(extra_info)
            else:
                raise ValueError(f"unknown loader: {source['loader']}")
        except Exception as exc:
            stats["error"] = f"{type(exc).__name__}: {exc}"
            continue

        stats["loaded_records"] = len(records)
        for record in records:
            if len(all_records) >= max_docs or total_chars >= max_chars:
                break
            text = record.get("text") or ""
            if not normalize_text(text):
                stats["empty_or_no_text"] += 1
                continue
            cleaned = clean_text(text, min_len=min_len, max_len=max_len, min_zh_ratio=min_zh_ratio)
            if cleaned is None:
                stats["filtered_records"] += 1
                continue
            dedup_key = re.sub(r"\s+", "", cleaned)
            if dedup_key in seen_texts:
                stats["duplicates"] += 1
                continue
            seen_texts.add(dedup_key)
            out = {
                "source_name": name,
                "source_type": source["source_type"],
                "text": cleaned,
                "char_count": len(cleaned),
                "chinese_ratio": round(chinese_ratio(cleaned), 4),
            }
            all_records.append(out)
            stats["kept_records"] += 1
            total_chars += len(cleaned)

    return all_records, source_stats


def find_previous_tokenizer() -> Path | None:
    candidates = [
        PROJECT_DIR / "23_chinese_open_dataset_pretraining_run/outputs/tokenizer/chinese_bpe_tokenizer.json",
        PROJECT_DIR / "22_train_tiny_gpt_with_chinese_bpe/outputs/tokenizers/chinese_bpe_tokenizer.json",
        PROJECT_DIR / "21_chinese_bpe_tokenizer_intro/outputs/tokenizers/chinese_bpe_512.json",
        PROJECT_DIR / "21_chinese_bpe_tokenizer_intro/outputs/tokenizers/chinese_bpe_256.json",
        PROJECT_DIR / "21_chinese_bpe_tokenizer_intro/outputs/tokenizers/chinese_bpe_128.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def train_fallback_bpe(corpus_path: Path, output_path: Path, vocab_size: int = 1024) -> Path:
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
        min_frequency=2,
    )
    tokenizer.train([str(corpus_path)], trainer=trainer)
    tokenizer.save(str(output_path))
    return output_path


def encode_corpus(tokenizer: Tokenizer, text: str, val_ratio: float) -> tuple[np.ndarray, np.ndarray, list[int]]:
    token_ids = tokenizer.encode(text).ids
    if len(token_ids) < 32:
        raise ValueError("encoded corpus is too small")
    split = max(1, int(len(token_ids) * (1.0 - val_ratio)))
    split = min(split, len(token_ids) - 1)
    train = np.array(token_ids[:split], dtype=np.int32)
    val = np.array(token_ids[split:], dtype=np.int32)
    return train, val, token_ids


def token_lengths_lab(tokenizer: Tokenizer, records: list[dict[str, Any]]) -> list[int]:
    return [len(tokenizer.encode(record["text"]).ids) for record in records]


def try_load_qwen_tokenizer(candidates: list[str]) -> tuple[Any | None, dict[str, Any]]:
    attempts = []
    if AutoTokenizer is None:
        return None, {
            "success": False,
            "actual_tokenizer_name": None,
            "attempts": [{"name": name, "ok": False, "error": "transformers is not installed"} for name in candidates],
        }

    for name in candidates:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                name,
                trust_remote_code=True,
                cache_dir=str(CACHE_DIR / "transformers"),
            )
            attempts.append({"name": name, "ok": True, "error": None})
            return tokenizer, {
                "success": True,
                "actual_tokenizer_name": name,
                "attempts": attempts,
            }
        except Exception as exc:
            attempts.append({"name": name, "ok": False, "error": f"{type(exc).__name__}: {exc}"[:500]})

    return None, {
        "success": False,
        "actual_tokenizer_name": None,
        "attempts": attempts,
    }


def qwen_tokenizer_stats(records: list[dict[str, Any]], candidates: list[str]) -> dict[str, Any]:
    tokenizer, status = try_load_qwen_tokenizer(candidates)
    texts = [record["text"] for record in records]
    total_chars = sum(len(text) for text in texts)
    if tokenizer is None:
        return {
            **status,
            "sample_count": len(records),
            "raw_char_count": total_chars,
            "qwen_token_total": None,
            "avg_tokens_per_sample": None,
            "max_tokens_per_sample": None,
            "over_512": None,
            "over_1024": None,
            "over_2048": None,
            "vocab_size": None,
            "special_tokens_map": None,
            "chat_template_available": False,
            "detected_chat_special_tokens": [],
            "examples": [],
        }

    lengths: list[int] = []
    examples = []
    for idx, text in enumerate(texts):
        ids = tokenizer.encode(text, add_special_tokens=False)
        lengths.append(len(ids))
        if len(examples) < 5:
            decoded = tokenizer.decode(ids[:120], skip_special_tokens=False)
            examples.append(
                {
                    "index": idx,
                    "text_preview": text[:220],
                    "token_count": len(ids),
                    "first_token_ids": ids[:40],
                    "decoded_preview": decoded[:220],
                }
            )

    vocab = tokenizer.get_vocab()
    special_candidates = [
        "<|im_start|>",
        "<|im_end|>",
        "<|endoftext|>",
        "<|fim_prefix|>",
        "<|fim_middle|>",
        "<|fim_suffix|>",
        "<|vision_start|>",
        "<|vision_end|>",
    ]
    detected = [token for token in special_candidates if token in vocab]
    total_tokens = sum(lengths)
    return {
        **status,
        "sample_count": len(records),
        "raw_char_count": total_chars,
        "qwen_token_total": total_tokens,
        "avg_tokens_per_sample": total_tokens / max(len(lengths), 1),
        "max_tokens_per_sample": max(lengths) if lengths else 0,
        "over_512": sum(1 for length in lengths if length > 512),
        "over_1024": sum(1 for length in lengths if length > 1024),
        "over_2048": sum(1 for length in lengths if length > 2048),
        "vocab_size": len(vocab),
        "special_tokens_map": tokenizer.special_tokens_map,
        "chat_template_available": bool(getattr(tokenizer, "chat_template", None)),
        "detected_chat_special_tokens": detected,
        "examples": examples,
        "lengths": lengths,
    }


def qwen_report_markdown(stats: dict[str, Any]) -> str:
    lines = [
        "# Qwen Tokenizer 兼容性报告",
        "",
        "## 加载结果",
        "",
        f"- 是否成功加载：{stats['success']}",
        f"- 实际 tokenizer 名称：{stats.get('actual_tokenizer_name')}",
        f"- vocab_size：{stats.get('vocab_size')}",
        "",
        "## 尝试顺序",
        "",
        "| tokenizer | 是否成功 | 错误 |",
        "|---|---:|---|",
    ]
    for attempt in stats.get("attempts", []):
        lines.append(f"| {attempt['name']} | {attempt['ok']} | {attempt.get('error') or ''} |")

    lines.extend(
        [
            "",
            "## 统计",
            "",
            f"- 样本数：{stats.get('sample_count')}",
            f"- 原始字符数：{stats.get('raw_char_count')}",
            f"- Qwen token 总数：{stats.get('qwen_token_total')}",
            f"- 平均每条样本 Qwen token 数：{stats.get('avg_tokens_per_sample')}",
            f"- 最长样本 Qwen token 数：{stats.get('max_tokens_per_sample')}",
            f"- 超过 512 tokens 的样本数：{stats.get('over_512')}",
            f"- 超过 1024 tokens 的样本数：{stats.get('over_1024')}",
            f"- 超过 2048 tokens 的样本数：{stats.get('over_2048')}",
            f"- 是否有 chat template：{stats.get('chat_template_available')}",
            f"- 检测到的 chat special tokens：{stats.get('detected_chat_special_tokens')}",
            "",
            "## special_tokens_map",
            "",
            "```json",
            json.dumps(stats.get("special_tokens_map"), ensure_ascii=False, indent=2),
            "```",
            "",
            "## Encode / Decode 示例",
            "",
        ]
    )
    for example in stats.get("examples", []):
        lines.extend(
            [
                f"### 样本 {example['index']}",
                "",
                f"- token 数：{example['token_count']}",
                f"- 原文预览：{example['text_preview']}",
                f"- 前 40 个 token ids：{example['first_token_ids']}",
                f"- decode 预览：{example['decoded_preview']}",
                "",
            ]
        )

    lines.extend(
        [
            "## 结论",
            "",
            "- Qwen tokenizer 只用于未来真实 Qwen / MLX-LM / LoRA / SFT 的兼容性检查。",
            "- 当前 Tiny GPT / qwen_dense_tiny 训练仍使用 lab tokenizer。",
            "- 不建议把 Qwen 的大词表直接用于当前小模型训练，否则 LM Head 会显著变大，训练成本和数据需求都会上升。",
        ]
    )
    return "\n".join(lines) + "\n"


def dual_track_report_markdown(
    metadata: dict[str, Any],
    lab_lengths: list[int],
    qwen_stats: dict[str, Any],
) -> str:
    qwen_lengths = qwen_stats.get("lengths") or []
    lab_total = sum(lab_lengths)
    qwen_total = sum(qwen_lengths) if qwen_lengths else None
    sample_char_count = metadata["total_chars"]
    sample_count = max(len(lab_lengths), 1)
    lines = [
        "# Tokenizer 双轨统计报告",
        "",
        "## 两条轨道的用途",
        "",
        "- lab tokenizer：用于我们自己的 MLX GPT Lab / qwen_dense_tiny 小模型训练。",
        "- Qwen tokenizer：只用于未来真实 Qwen / MLX-LM / LoRA / SFT 的兼容性检查。",
        "- 不要把 Qwen 25 万级大词表直接用于当前 Tiny GPT 训练；当前小模型数据量和参数量都不匹配。",
        "",
        "## 对比表",
        "",
        "| 指标 | lab tokenizer | Qwen tokenizer |",
        "|---|---:|---:|",
        f"| 样本数 | {len(lab_lengths)} | {qwen_stats.get('sample_count')} |",
        f"| 样本字符数 | {sample_char_count} | {qwen_stats.get('raw_char_count')} |",
        f"| token 总数 | {lab_total} | {qwen_total} |",
        f"| 平均 tokens / 样本 | {lab_total / sample_count:.2f} | {qwen_stats.get('avg_tokens_per_sample')} |",
        f"| 最长样本 token 数 | {max(lab_lengths) if lab_lengths else 0} | {qwen_stats.get('max_tokens_per_sample')} |",
        f"| 平均每 token 字符数 | {sample_char_count / max(lab_total, 1):.4f} | {qwen_stats.get('raw_char_count') / max(qwen_total, 1) if qwen_total else None} |",
        f"| vocab_size | {metadata['vocab_size']} | {qwen_stats.get('vocab_size')} |",
        "",
        "## 解释",
        "",
        "lab tokenizer 的目标是让本地小模型能低成本训练，所以 vocab_size 较小，LM Head 也较小。",
        "",
        "Qwen tokenizer 的目标是和真实 Qwen 模型保持 token 边界、special tokens、chat template 兼容，方便后续 LoRA / SFT。",
        "",
        "这两者服务的模型规模不同，不应该混用。",
        "",
        "注：这张表按逐条样本统计，不包含拼接 corpus 时插入的段落分隔符；训练用的 `train_tokens.npy` / `val_tokens.npy` 仍来自完整拼接后的 `fun_corpus.txt`。",
    ]
    return "\n".join(lines) + "\n"


def save_samples(records: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if len(grouped[record["source_type"]]) < 10:
            grouped[record["source_type"]].append(record)
    for source_type, items in grouped.items():
        text = []
        for idx, item in enumerate(items, 1):
            text.append(f"## {idx}. {item['source_name']}\n{item['text']}\n")
        (SAMPLE_DIR / f"{source_type}_samples.txt").write_text("\n".join(text), encoding="utf-8")

    preview = []
    for idx, item in enumerate(records[:30], 1):
        preview.append(f"## {idx}. {item['source_type']} / {item['source_name']}\n{item['text']}\n")
    (SAMPLE_DIR / "mixed_samples.txt").write_text("\n".join(preview), encoding="utf-8")


def report_markdown(
    records: list[dict[str, Any]],
    source_stats: dict[str, Any],
    metadata: dict[str, Any],
    top_tokens: list[tuple[str, int]],
) -> str:
    by_type = Counter(record["source_type"] for record in records)
    lines = [
        "# 中文趣味语料数据报告",
        "",
        "## 总览",
        "",
        f"- 清洗后样本数：{metadata['kept_docs']}",
        f"- 总字符数：{metadata['total_chars']}",
        f"- 平均长度：{metadata['avg_chars_per_doc']:.2f}",
        f"- 中文比例估计：{metadata['avg_chinese_ratio']:.4f}",
        f"- 字符级 token 数：{metadata['char_token_count']}",
        f"- BPE token 数：{metadata['bpe_token_count']}",
        f"- `<unk>` token 数：{metadata['unk_token_count']}",
        f"- `<unk>` token 比例：{metadata['unk_token_ratio']:.4%}",
        f"- BPE 平均每 token 字符数：{metadata['avg_chars_per_bpe_token']:.4f}",
        f"- train tokens：{metadata['train_tokens']}",
        f"- val tokens：{metadata['val_tokens']}",
        f"- tokenizer：{metadata['tokenizer_path']}",
        "",
        "## 类别占比",
        "",
        "| 类别 | 样本数 | 占比 |",
        "|---|---:|---:|",
    ]
    total = max(len(records), 1)
    for source_type, count in by_type.most_common():
        lines.append(f"| {source_type} | {count} | {count / total:.2%} |")

    lines.extend(["", "## 数据源结果", "", "| 数据源 | 类型 | 启用 | 加载 | 保留 | 空文本 | 过滤 | 去重 | 备注/错误 |", "|---|---|---:|---:|---:|---:|---:|---:|---|"])
    for stat in source_stats.values():
        err = stat.get("error") or stat.get("notes") or ""
        lines.append(
            "| {source_name} | {source_type} | {enabled} | {loaded_records} | {kept_records} | "
            "{empty_or_no_text} | {filtered_records} | {duplicates} | {err} |".format(
                **stat, err=str(err).replace("\n", " ")[:240]
            )
        )

    lines.extend(["", "## 高频 BPE 片段", "", "| token | count |", "|---|---:|"])
    for token, count in top_tokens[:50]:
        shown = token.replace("\n", "\\n")
        lines.append(f"| `{shown}` | {count} |")

    lines.extend(
        [
            "",
            "## 观察",
            "",
            "- 本课只做数据管线，不训练模型。",
            "- `ToxiCN_MM` 已按用户要求启用，但当前 Hugging Face 可见字段只有图片路径，没有可直接训练的文本字段，所以没有进入文本 corpus。",
            "- 后续 continued pretraining 可以直接读取 `train_tokens.npy` / `val_tokens.npy`。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-docs", type=int, default=12000)
    parser.add_argument("--max-chars", type=int, default=3_000_000)
    parser.add_argument("--min-len", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=2000)
    parser.add_argument("--min-zh-ratio", type=float, default=0.15)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--fallback-lab-vocab-size", type=int, default=8192)
    parser.add_argument(
        "--qwen-tokenizer-candidates",
        nargs="*",
        default=["Qwen/Qwen3.6-27B", "Qwen/Qwen3-0.6B", "Qwen/Qwen2.5-0.5B"],
    )
    args = parser.parse_args()

    start = time.time()
    ensure_dirs()
    sources = json.loads(SOURCE_CONFIG_PATH.read_text(encoding="utf-8"))

    print("=== Chinese Fun Corpus Pipeline ===")
    print("Course dir:", CURRENT_DIR)
    print("Cache dir:", CACHE_DIR)
    print("Max docs:", args.max_docs)
    print("Max chars:", args.max_chars)
    print("Enabled sources:", [s["source_name"] for s in sources if s.get("enabled")])

    records, source_stats = collect_sources(
        sources=sources,
        max_docs=args.max_docs,
        max_chars=args.max_chars,
        min_len=args.min_len,
        max_len=args.max_len,
        min_zh_ratio=args.min_zh_ratio,
    )
    if not records:
        raise RuntimeError("No usable text records collected.")

    raw_jsonl_path = RAW_DIR / "fun_corpus_raw.jsonl"
    with raw_jsonl_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    corpus_text = "\n\n".join(record["text"] for record in records)
    corpus_path = PROCESSED_DIR / "fun_corpus.txt"
    corpus_path.write_text(corpus_text, encoding="utf-8")

    save_samples(records)

    tokenizer_path = find_previous_tokenizer()
    tokenizer_source = "previous_course"
    if tokenizer_path is None:
        tokenizer_path = METADATA_DIR / "fallback_fun_bpe_tokenizer.json"
        train_fallback_bpe(corpus_path, tokenizer_path, vocab_size=args.fallback_lab_vocab_size)
        tokenizer_source = "trained_fallback"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    train_tokens, val_tokens, all_token_ids = encode_corpus(tokenizer, corpus_text, args.val_ratio)
    np.save(PROCESSED_DIR / "train_tokens.npy", train_tokens)
    np.save(PROCESSED_DIR / "val_tokens.npy", val_tokens)

    token_counter = Counter(all_token_ids)
    unk_id = tokenizer.token_to_id("<unk>")
    unk_token_count = token_counter.get(unk_id, 0) if unk_id is not None else 0
    lab_lengths = token_lengths_lab(tokenizer, records)
    vocab = tokenizer.get_vocab()
    id_to_token = {idx: token for token, idx in vocab.items()}
    top_tokens = [(id_to_token.get(token_id, str(token_id)), count) for token_id, count in token_counter.most_common(80)]

    total_chars = sum(record["char_count"] for record in records)
    avg_ratio = sum(record["chinese_ratio"] for record in records) / max(len(records), 1)
    lengths = [record["char_count"] for record in records]
    metadata = {
        "course": "28_chinese_fun_corpus_pipeline",
        "created_at_unix": time.time(),
        "elapsed_seconds": round(time.time() - start, 2),
        "kept_docs": len(records),
        "total_chars": total_chars,
        "avg_chars_per_doc": total_chars / max(len(records), 1),
        "min_chars_per_doc": min(lengths),
        "max_chars_per_doc": max(lengths),
        "avg_chinese_ratio": avg_ratio,
        "char_token_count": len(corpus_text),
        "bpe_token_count": len(all_token_ids),
        "unk_token_count": int(unk_token_count),
        "unk_token_ratio": float(unk_token_count / max(len(all_token_ids), 1)),
        "avg_chars_per_bpe_token": len(corpus_text) / max(len(all_token_ids), 1),
        "train_tokens": int(train_tokens.shape[0]),
        "val_tokens": int(val_tokens.shape[0]),
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_source": tokenizer_source,
        "vocab_size": tokenizer.get_vocab_size(),
        "val_ratio": args.val_ratio,
        "source_stats": source_stats,
        "category_counts": dict(Counter(record["source_type"] for record in records)),
    }
    json_dump(metadata, METADATA_DIR / "train_val_metadata.json")
    lab_tokenizer_metadata = {
        "purpose": "MLX GPT Lab / qwen_dense_tiny training",
        "tokenizer_kind": "lab_bpe",
        "tokenizer_source": tokenizer_source,
        "tokenizer_path": str(tokenizer_path),
        "vocab_size": tokenizer.get_vocab_size(),
        "sample_count": len(records),
        "sample_raw_char_count": total_chars,
        "corpus_char_count": len(corpus_text),
        "sample_token_total": int(sum(lab_lengths)),
        "corpus_token_total": len(all_token_ids),
        "token_total": len(all_token_ids),
        "train_tokens": int(train_tokens.shape[0]),
        "val_tokens": int(val_tokens.shape[0]),
        "avg_tokens_per_sample": sum(lab_lengths) / max(len(lab_lengths), 1),
        "max_tokens_per_sample": max(lab_lengths) if lab_lengths else 0,
        "unk_token_count": int(unk_token_count),
        "unk_token_ratio": float(unk_token_count / max(len(all_token_ids), 1)),
        "note": "This tokenizer is used for local Tiny GPT training. Qwen tokenizer stats are compatibility-only.",
    }
    json_dump(lab_tokenizer_metadata, METADATA_DIR / "lab_tokenizer_metadata.json")

    print("\nLoading Qwen tokenizer for compatibility stats only...")
    qwen_stats = qwen_tokenizer_stats(records, args.qwen_tokenizer_candidates)
    json_dump(qwen_stats, METADATA_DIR / "qwen_token_stats.json")
    json_dump(source_stats, METADATA_DIR / "source_stats.json")

    report = report_markdown(records, source_stats, metadata, top_tokens)
    (REPORT_DIR / "fun_corpus_data_report.md").write_text(report, encoding="utf-8")
    (REPORT_DIR / "qwen_tokenizer_compat_report.md").write_text(qwen_report_markdown(qwen_stats), encoding="utf-8")
    (REPORT_DIR / "tokenizer_dual_track_report.md").write_text(
        dual_track_report_markdown(metadata, lab_lengths, qwen_stats),
        encoding="utf-8",
    )

    print("\nDone.")
    print("Raw jsonl:", raw_jsonl_path)
    print("Corpus:", corpus_path)
    print("Train tokens:", PROCESSED_DIR / "train_tokens.npy", train_tokens.shape)
    print("Val tokens:", PROCESSED_DIR / "val_tokens.npy", val_tokens.shape)
    print("Metadata:", METADATA_DIR / "train_val_metadata.json")
    print("Lab tokenizer metadata:", METADATA_DIR / "lab_tokenizer_metadata.json")
    print("Qwen tokenizer stats:", METADATA_DIR / "qwen_token_stats.json")
    print("Report:", REPORT_DIR / "fun_corpus_data_report.md")
    print("Qwen report:", REPORT_DIR / "qwen_tokenizer_compat_report.md")
    print("Dual track report:", REPORT_DIR / "tokenizer_dual_track_report.md")


if __name__ == "__main__":
    main()
