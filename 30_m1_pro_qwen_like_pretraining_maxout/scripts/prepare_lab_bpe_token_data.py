from __future__ import annotations

import argparse
import gzip
import hashlib
import html
import json
import os
import random
import re
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
from huggingface_hub import hf_hub_download
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
PREPARED_SOURCE_CACHE_DIR = CACHE_DIR / "prepared_sources"
REPORT_DIR = CURRENT_DIR / "outputs" / "reports"

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]
SOURCE_CACHE_VERSION = "chat_mix_clean_20260613_v6_hana_low"
QUALITY_GATE_NAME = "chat_quality_gate_current"
CHAT_SPACING_CATEGORIES = {
    "lccc_dialogue",
    "hana_dialogue",
    "anime_roleplay",
    "game_world_chat",
    "fun_style",
    "bilibili_comment",
}
TEXT_FIELD_CANDIDATES = ["text", "content", "document", "raw_content", "article", "body", "title"]

# 默认源尽量避开单一 FineWeb-Edu 风格。LCCC 是当前首位数据源，
# B 站评论、ACG wiki、玩家社区和角色对话作为次级风格补充。
DEFAULT_SOURCE_PLAN = [
    {"dataset_name": "Skywork/SkyPile-150B", "target_chars": 34_000_000, "max_docs": 260_000, "max_raw_chars": 360_000_000},
    {
        "dataset_name": "Morton-Li/ChineseWebText2.0-HighQuality",
        "target_chars": 26_000_000,
        "max_docs": 220_000,
        "max_raw_chars": 330_000_000,
    },
    {
        "dataset_name": "opencsg/Fineweb-Edu-Chinese-V2.1",
        "target_chars": 2_000_000,
        "max_docs": 35_000,
        "max_raw_chars": 55_000_000,
    },
]

ACG_SOURCE_DEFAULTS = {
    "bilibili_target_chars": 16_000_000,
    "bilibili_max_rows": 2_500_000,
    "moegirl_target_chars": 8_000_000,
    "moegirl_max_rows": 90_000,
    "worldchat_target_chars": 500_000,
    "worldchat_max_rows": 5_000,
    "chatharuhi_target_chars": 6_000_000,
    "chatharuhi_max_rows": 80_000,
}

LCCC_FILES = {
    "base_train": "lccc_base_train.jsonl.gz",
    "base_valid": "lccc_base_valid.jsonl.gz",
    "base_test": "lccc_base_test.jsonl.gz",
    "large": "lccc_large.jsonl.gz",
}

HANA_REPO_URL = "https://www.modelscope.cn/datasets/xuanxixue/HANA.git"
HANA_REPO_DIR = CACHE_DIR / "modelscope" / "HANA_repo"

CATEGORY_CAPS = {
    # 这些不是完全不能用，但必须小比例保留，否则输出会变成公文/法律/医疗/报道腔。
    "formal": 0.03,
    "legal": 0.02,
    "medical": 0.03,
    "recruitment": 0.02,
    "news": 0.08,
    "encyclopedia": 0.10,
}

KEYWORDS = {
    "formal": [
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
        "贯彻落实",
        "会议精神",
        "责任单位",
        "领导小组",
        "监督检查",
        "条例",
        "规划纲要",
        "增值税",
        "国家税务总局",
        "财政部",
        "财税",
        "有限公司",
        "营业部",
        "期货",
        "投资者",
    ],
    "legal": [
        "律师",
        "法院",
        "起诉",
        "判决",
        "赔偿",
        "合同",
        "工伤",
        "劳动关系",
        "债务",
        "清算",
        "股东",
        "法定",
        "纠纷",
        "法律依据",
        "相关知识推荐",
        "法律问答",
        "咨询我",
    ],
    "medical": [
        "医院",
        "医生",
        "患者",
        "治疗",
        "症状",
        "疾病",
        "疫苗",
        "手术",
        "诊断",
        "药物",
        "康复",
        "感染",
        "肿瘤",
        "中医治疗",
        "建议到医院",
    ],
    "recruitment": [
        "事业单位",
        "公开招聘",
        "招聘工作人员",
        "资格审查",
        "面试时间",
        "考生",
        "报名",
        "准考证",
        "岗位表",
        "试讲",
    ],
    "news": [
        "据",
        "报道",
        "记者",
        "新华社",
        "中新网",
        "当地时间",
        "消息称",
        "发布消息",
        "红网时刻",
        "表示",
        "称",
    ],
    "encyclopedia": [
        "什么是",
        "如何",
        "为什么",
        "作用",
        "特点",
        "原理",
        "方法",
        "步骤",
        "症状",
        "预防",
        "养殖",
        "栽培",
        "品种",
        "多年生",
    ],
}

SPAM_KEYWORDS = [
    "美女图片",
    "人体图片",
    "成人视频",
    "成人视频",
    "博彩",
    "赌博",
    "六合彩",
    "重庆时时彩",
    "代开发票",
    "取店名",
    "kaixinseqing",
    "点击进入",
    "app下载",
    "最新网址",
    "原神代肝",
    "代肝",
    "价格私聊",
    "接单私聊",
    "陪玩接单",
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


def slugify(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)
    return value.strip("_")[:80] or "source"


def source_cache_path(label: str, payload: dict[str, Any]) -> Path:
    key = json.dumps(
        {"version": SOURCE_CACHE_VERSION, "label": label, "payload": payload},
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    return PREPARED_SOURCE_CACHE_DIR / f"{slugify(label)}_{digest}.jsonl"


def save_source_cache(path: Path, source_docs: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in source_docs:
            cached = dict(item)
            cached.pop("id", None)
            f.write(json.dumps(cached, ensure_ascii=False) + "\n")


def load_source_cache(path: Path, docs: list[dict[str, Any]], seen: set[str], stats: dict[str, int]) -> dict[str, Any]:
    report = {
        "cache_hit": True,
        "cache_path": str(path),
        "loaded_docs": 0,
        "loaded_chars": 0,
        "skipped_duplicates": 0,
    }
    with path.open(encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            text = item.get("text") or ""
            fp = fingerprint(text)
            if fp in seen:
                report["skipped_duplicates"] += 1
                continue
            seen.add(fp)
            item["id"] = len(docs)
            item["char_count"] = int(item.get("char_count") or len(text))
            item["chinese_ratio"] = float(item.get("chinese_ratio") or chinese_ratio(text))
            docs.append(item)
            category = item.get("category", "unknown")
            report["loaded_docs"] += 1
            report["loaded_chars"] += item["char_count"]
            stats[f"cache_keep_{category}_docs"] = stats.get(f"cache_keep_{category}_docs", 0) + 1
            stats[f"cache_keep_{category}_chars"] = stats.get(f"cache_keep_{category}_chars", 0) + item["char_count"]
    return report


def collect_with_cache(
    label: str,
    collector: Any,
    docs: list[dict[str, Any]],
    seen: set[str],
    stats: dict[str, int],
    *,
    refresh_cache: bool,
    cache_payload: dict[str, Any],
    collector_kwargs: dict[str, Any],
    allow_collect: bool = True,
) -> dict[str, Any]:
    path = source_cache_path(label, cache_payload)
    if path.exists() and not refresh_cache:
        print(f"Using prepared source cache {path}", flush=True)
        report = load_source_cache(path, docs, seen, stats)
        report["source_label"] = label
        report["collector_skipped"] = True
        return report

    if not allow_collect:
        return {
            "cache_hit": False,
            "cache_path": str(path),
            "source_label": label,
            "collector_skipped": True,
            "skipped": True,
            "reason": "prepared source cache missing and live collection disabled",
        }

    before = len(docs)
    started = time.time()
    report = collector(docs, seen, stats, **collector_kwargs)
    new_docs = docs[before:]
    save_source_cache(path, new_docs)
    if isinstance(report, dict):
        report["cache_hit"] = False
        report["cache_path"] = str(path)
        report["cached_docs"] = len(new_docs)
        report["cached_chars"] = sum(int(item.get("char_count", 0)) for item in new_docs)
        report["cache_write_elapsed_sec"] = time.time() - started
    return report


def log_phase(name: str, started: float, phase_times: dict[str, float]) -> float:
    elapsed = time.perf_counter() - started
    phase_times[name] = elapsed
    print(f"phase={name} elapsed_sec={elapsed:.2f}", flush=True)
    return time.perf_counter()


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


def normalize_preserve_punctuation(text: str) -> str:
    """保留中文标点。不要用 NFKC，否则会把全角标点归一成英文标点。"""

    text = html.unescape(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    for pattern in BOILERPLATE_PATTERNS:
        text = re.sub(pattern, " ", text)
    text = re.sub(r"[\t\f\v]+", " ", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def repair_lccc_utterance(text: str) -> str:
    """LCCC 中很多句子是中文分词后带空格的形式，这里还原成更自然的中文。"""

    text = normalize_preserve_punctuation(text)
    text = repair_common_spacing(text)
    text = re.sub(r"(?<=[\u3400-\u9fff])\s+(?=[\u3400-\u9fff])", "", text)
    text = re.sub(r"(?<=[\u3400-\u9fff])\s+(?=[A-Za-z0-9])", "", text)
    text = re.sub(r"(?<=[A-Za-z0-9])\s+(?=[\u3400-\u9fff])", "", text)
    text = re.sub(r"([（【《])\s+", r"\1", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    return text.strip()


def repair_common_spacing(text: str) -> str:
    """全局轻量 spacing 修复，不改变标点类型。

    只处理非常明确的异常：标点前空格、中文标点后接中文/数字/英文时的空格、
    以及 `~ ~ ~` / `… …` / `! !` 这类重复语气符号之间的空格。
    不删除普通英文短语内部空格。
    """

    horizontal_space = r"[\t \u00A0\u3000]+"
    text = re.sub(horizontal_space + r"([，。！？；：、,.!?;:~～…）】》])", r"\1", text)
    text = re.sub(r"([，。！？；：、])" + horizontal_space + r"(?=[\u3400-\u9fffA-Za-z0-9~～…])", r"\1", text)
    text = re.sub(r"([,.!?;:])" + horizontal_space + r"(?=[\u3400-\u9fff])", r"\1", text)
    text = re.sub(r"(?<=[~～…!！?？])" + horizontal_space + r"(?=[~～…!！?？])", "", text)
    return re.sub(r"[ ]{2,}", " ", text).strip()


def repair_chat_spacing(text: str) -> str:
    """对聊天类语料做最终 spacing 修复。

    只对聊天/评论/角色对话类来源使用，避免把普通英文网页文本中的
    正常空格也误删。重点修复 LCCC 这类中文分词预处理遗留的空格：
    `额， 我` -> `额，我`，`~ ~ ~` -> `~~~`。
    """

    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        lines.append(repair_lccc_utterance(stripped) if stripped else "")
    return "\n".join(lines).strip()


def fingerprint(text: str) -> str:
    compact = re.sub(r"\s+", "", text)
    return hashlib.sha1(compact.encode("utf-8")).hexdigest()


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


def has_spam(text: str) -> bool:
    return any(keyword in text for keyword in SPAM_KEYWORDS)


def is_too_repetitive(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    if len(compact) < 120:
        return False
    freq: dict[str, int] = {}
    for ch in compact:
        freq[ch] = freq.get(ch, 0) + 1
    if max(freq.values()) / len(compact) > 0.24:
        return True
    chunks = [compact[i : i + 20] for i in range(0, len(compact) - 20, 20)]
    return len(chunks) > 8 and len(set(chunks)) / len(chunks) < 0.45


def classify_text(text: str) -> tuple[str, dict[str, int]]:
    scores = {name: sum(1 for kw in words if kw in text) for name, words in KEYWORDS.items()}

    # “据”单字太宽，只在有报道/记者等新闻上下文时才把 news 加强。
    if not re.search(r"(据.+报道|记者|新华社|中新网|当地时间|消息称|发布消息|红网时刻)", text):
        scores["news"] = max(0, scores["news"] - 1)

    # 模板式问答/百科强识别。
    if re.search(r"(什么是|如何|为什么|有哪些|怎么申请|怎么治疗|怎么分配|注意事项|具体步骤)", text):
        scores["encyclopedia"] += 2

    priority = ["formal", "legal", "medical", "recruitment", "news", "encyclopedia"]
    for name in priority:
        if scores.get(name, 0) >= 2:
            return name, scores
    if any(
        marker in text
        for marker in [
            "我觉得",
            "你觉得",
            "吗？",
            "怎么办",
            "哈哈",
            "老哥",
            "这波",
            "评论",
            "网友",
            "帖子",
            "回复",
            "楼主",
            "up主",
            "宝贝",
            "亲",
        ]
    ):
        return "natural_dialogish", scores
    return "natural_web", scores


def clean_text(text: str, min_len: int = 80, max_len: int = 2600, min_zh_ratio: float = 0.3) -> tuple[str | None, str]:
    text = normalize_preserve_punctuation(text)
    if len(text) < min_len:
        return None, "too_short"
    if len(text) > max_len:
        text = text[:max_len].strip()
    if chinese_ratio(text) < min_zh_ratio:
        return None, "low_chinese_ratio"
    if has_spam(text):
        return None, "spam"
    if is_too_repetitive(text):
        return None, "repetitive"
    return text, "ok"


def category_char_count(docs: list[dict[str, Any]], category: str) -> int:
    return sum(item["char_count"] for item in docs if item["category"] == category)


def can_keep_category(docs: list[dict[str, Any]], category: str, target_total_chars: int, char_count: int) -> bool:
    cap = CATEGORY_CAPS.get(category)
    if cap is None:
        return True
    return category_char_count(docs, category) + char_count <= int(target_total_chars * cap)


def add_doc(
    docs: list[dict[str, Any]],
    seen: set[str],
    raw_text: str,
    source_name: str,
    source_type: str,
    stats: dict[str, int],
    target_total_chars: int,
    *,
    force_fun: bool = False,
) -> bool:
    if force_fun:
        text = normalize_preserve_punctuation(raw_text)
        if not text:
            stats["reject_fun_empty"] = stats.get("reject_fun_empty", 0) + 1
            return False
        category = "fun_style"
        scores = {}
    else:
        text, reason = clean_text(raw_text)
        if text is None:
            stats[f"reject_{reason}"] = stats.get(f"reject_{reason}", 0) + 1
            return False
        category, scores = classify_text(text)
        if not can_keep_category(docs, category, target_total_chars, len(text)):
            stats[f"reject_cap_{category}"] = stats.get(f"reject_cap_{category}", 0) + 1
            return False

    fp = fingerprint(text)
    if fp in seen:
        stats["reject_duplicate"] = stats.get("reject_duplicate", 0) + 1
        return False
    seen.add(fp)

    docs.append(
        {
            "id": len(docs),
            "text": text,
            "source_name": source_name,
            "source_type": source_type,
            "category": category,
            "category_scores": scores,
            "char_count": len(text),
            "chinese_ratio": chinese_ratio(text),
        }
    )
    stats[f"keep_{category}_docs"] = stats.get(f"keep_{category}_docs", 0) + 1
    stats[f"keep_{category}_chars"] = stats.get(f"keep_{category}_chars", 0) + len(text)
    return True


def add_custom_doc(
    docs: list[dict[str, Any]],
    seen: set[str],
    raw_text: str,
    source_name: str,
    source_type: str,
    category: str,
    stats: dict[str, int],
    *,
    min_len: int = 8,
    max_len: int = 1800,
    min_zh_ratio: float = 0.25,
) -> bool:
    """用于短评论、ACG wiki、玩家聊天等定向来源。

    这些来源的目标不是“通用长文质量”，而是补充口语、弹幕、
    玩家社区和二次元语境，所以不能复用 general web 的 80 字下限。
    """

    text = normalize_preserve_punctuation(raw_text)
    text = re.sub(r"\u200b|\ufeff", "", text)
    if len(text) > max_len:
        text = text[:max_len].strip()
    if len(text) < min_len:
        stats[f"{category}_reject_too_short"] = stats.get(f"{category}_reject_too_short", 0) + 1
        return False
    if chinese_ratio(text) < min_zh_ratio:
        stats[f"{category}_reject_low_chinese_ratio"] = stats.get(f"{category}_reject_low_chinese_ratio", 0) + 1
        return False
    if has_spam(text):
        stats[f"{category}_reject_spam"] = stats.get(f"{category}_reject_spam", 0) + 1
        return False
    if is_too_repetitive(text):
        stats[f"{category}_reject_repetitive"] = stats.get(f"{category}_reject_repetitive", 0) + 1
        return False

    fp = fingerprint(text)
    if fp in seen:
        stats[f"{category}_reject_duplicate"] = stats.get(f"{category}_reject_duplicate", 0) + 1
        return False
    seen.add(fp)

    docs.append(
        {
            "id": len(docs),
            "text": text,
            "source_name": source_name,
            "source_type": source_type,
            "category": category,
            "category_scores": {},
            "char_count": len(text),
            "chinese_ratio": chinese_ratio(text),
        }
    )
    stats[f"keep_{category}_docs"] = stats.get(f"keep_{category}_docs", 0) + 1
    stats[f"keep_{category}_chars"] = stats.get(f"keep_{category}_chars", 0) + len(text)
    return True


def flush_grouped_short_texts(
    docs: list[dict[str, Any]],
    seen: set[str],
    lines: list[str],
    source_name: str,
    source_type: str,
    category: str,
    stats: dict[str, int],
    *,
    min_len: int = 20,
) -> int:
    if not lines:
        return 0
    text = "\n".join(lines)
    before = len(docs)
    add_custom_doc(
        docs,
        seen,
        text,
        source_name=source_name,
        source_type=source_type,
        category=category,
        stats=stats,
        min_len=min_len,
        max_len=1400,
        min_zh_ratio=0.22,
    )
    return docs[-1]["char_count"] if len(docs) > before else 0


def clean_moegirl_markup(text: str, title: str | None = None) -> str:
    text = normalize_preserve_punctuation(text)
    text = re.sub(r"\{\|.*?\|\}", " ", text, flags=re.DOTALL)
    text = re.sub(r"\{\{[^{}]{0,600}\}\}", " ", text)
    text = re.sub(r"\[\[分类:[^\]]+\]\]", " ", text)
    text = re.sub(r"\[\[Category:[^\]]+\]\]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\[\[(?:文件|File|Image|图片):[^\]]+\]\]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\[\[([^|\]]+)\|([^\]]+)\]\]", r"\2", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[https?://[^\]\s]+(?:\s+([^\]]+))?\]", r"\1", text)
    text = re.sub(r"'{2,}", "", text)
    text = re.sub(r"==+\s*(.*?)\s*==+", r"\n\1\n", text)
    text = re.sub(r"<ref[^>]*>.*?</ref>", " ", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    cleaned_lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(("{{", "}}")):
            continue
        line = re.sub(r"\{\{[^{}\n]*(?:\}\})?", " ", line).strip()
        if not line:
            continue
        if re.search(r"(\d{2,4}px\|thumb|thumb\|right|thumb\|left|缩略图|外部链接|参考资料|注释|导航|相关条目)", line, flags=re.IGNORECASE):
            continue
        if re.fullmatch(r"\[?\s*(链接|link)\s*\]?", line, flags=re.IGNORECASE):
            continue
        if line.startswith("[") and len(line) < 40:
            continue
        if line.startswith(("*", "#", "|", "!", ";", ":")) and len(line) < 80:
            continue
        line = re.sub(r"\b\d{2,4}px\|thumb\|(?:right|left|center)\|?", " ", line, flags=re.IGNORECASE)
        line = re.sub(r"\s{2,}", " ", line).strip()
        if line:
            cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if title and title not in text[:80]:
        text = f"{title}\n{text}"
    return text


def collect_bilibili_comments(
    docs: list[dict[str, Any]],
    seen: set[str],
    stats: dict[str, int],
    *,
    target_chars: int,
    max_rows: int,
) -> dict[str, Any]:
    from datasets import load_dataset

    report = {
        "dataset_name": "Midsummra/bilibilicomment",
        "target_chars": target_chars,
        "max_rows": max_rows,
        "streaming": True,
        "raw_rows_seen": 0,
        "kept_docs": 0,
        "kept_chars": 0,
        "error": None,
    }
    print(f"Streaming Bilibili comments target_chars={target_chars}", flush=True)
    started = time.time()
    ds = load_dataset("Midsummra/bilibilicomment", split="train", streaming=True, cache_dir=str(CACHE_DIR / "datasets"))
    group: list[str] = []
    group_chars = 0
    for row in ds:
        report["raw_rows_seen"] += 1
        if report["raw_rows_seen"] > max_rows:
            break
        message = normalize_preserve_punctuation(str(row.get("message", "")))
        message = re.sub(r"\s+", " ", message).strip()
        if len(message) < 2:
            stats["bilibili_comment_reject_too_short_line"] = stats.get("bilibili_comment_reject_too_short_line", 0) + 1
            continue
        if chinese_ratio(message) < 0.18:
            stats["bilibili_comment_reject_low_chinese_line"] = stats.get("bilibili_comment_reject_low_chinese_line", 0) + 1
            continue
        if has_spam(message):
            stats["bilibili_comment_reject_spam_line"] = stats.get("bilibili_comment_reject_spam_line", 0) + 1
            continue
        group.append(message[:180])
        group_chars += len(message)
        if len(group) >= 18 or group_chars >= 700:
            kept = flush_grouped_short_texts(
                docs,
                seen,
                group,
                source_name="Midsummra/bilibilicomment",
                source_type="bilibili_comment",
                category="bilibili_comment",
                stats=stats,
            )
            if kept:
                report["kept_docs"] += 1
                report["kept_chars"] += kept
            group = []
            group_chars = 0
        if report["raw_rows_seen"] % 50_000 == 0:
            print(
                f"Bilibili rows={report['raw_rows_seen']} kept_docs={report['kept_docs']} kept_chars={report['kept_chars']}",
                flush=True,
            )
        if report["kept_chars"] >= target_chars:
            break
    if report["kept_chars"] < target_chars and group:
        kept = flush_grouped_short_texts(
            docs,
            seen,
            group,
            source_name="Midsummra/bilibilicomment",
            source_type="bilibili_comment",
            category="bilibili_comment",
            stats=stats,
        )
        if kept:
            report["kept_docs"] += 1
            report["kept_chars"] += kept
    report["elapsed_sec"] = time.time() - started
    return report


def collect_moegirl(
    docs: list[dict[str, Any]],
    seen: set[str],
    stats: dict[str, int],
    *,
    target_chars: int,
    max_rows: int,
) -> dict[str, Any]:
    from datasets import load_dataset

    report = {
        "dataset_name": "mrzjy/chinese_moegirl_wiki_corpus_raw",
        "target_chars": target_chars,
        "max_rows": max_rows,
        "streaming": True,
        "raw_rows_seen": 0,
        "kept_docs": 0,
        "kept_chars": 0,
        "error": None,
    }
    print(f"Streaming Moegirl ACG wiki target_chars={target_chars}", flush=True)
    started = time.time()
    ds = load_dataset(
        "mrzjy/chinese_moegirl_wiki_corpus_raw",
        split="train",
        streaming=True,
        cache_dir=str(CACHE_DIR / "datasets"),
    )
    for row in ds:
        report["raw_rows_seen"] += 1
        if report["raw_rows_seen"] > max_rows:
            break
        text = clean_moegirl_markup(str(row.get("text", "")), str(row.get("title", "")) if row.get("title") else None)
        before = len(docs)
        add_custom_doc(
            docs,
            seen,
            text,
            source_name="mrzjy/chinese_moegirl_wiki_corpus_raw",
            source_type="acg_wiki",
            category="acg_wiki",
            stats=stats,
            min_len=80,
            max_len=1800,
            min_zh_ratio=0.25,
        )
        if len(docs) > before:
            report["kept_docs"] += 1
            report["kept_chars"] += docs[-1]["char_count"]
        if report["raw_rows_seen"] % 5000 == 0:
            print(
                f"Moegirl rows={report['raw_rows_seen']} kept_docs={report['kept_docs']} kept_chars={report['kept_chars']}",
                flush=True,
            )
        if report["kept_chars"] >= target_chars:
            break
    report["elapsed_sec"] = time.time() - started
    return report


def collect_worldchat(
    docs: list[dict[str, Any]],
    seen: set[str],
    stats: dict[str, int],
    *,
    target_chars: int,
    max_rows: int,
) -> dict[str, Any]:
    from datasets import load_dataset

    report = {
        "dataset_name": "happyme531/YiMeng-JiangHu-WorldChat-1k",
        "target_chars": target_chars,
        "max_rows": max_rows,
        "streaming": True,
        "raw_rows_seen": 0,
        "kept_docs": 0,
        "kept_chars": 0,
        "error": None,
    }
    print(f"Streaming game world chat target_chars={target_chars}", flush=True)
    started = time.time()
    ds = load_dataset(
        "happyme531/YiMeng-JiangHu-WorldChat-1k",
        split="train",
        streaming=True,
        cache_dir=str(CACHE_DIR / "datasets"),
    )
    group: list[str] = []
    group_chars = 0
    for row in ds:
        report["raw_rows_seen"] += 1
        if report["raw_rows_seen"] > max_rows:
            break
        text = normalize_preserve_punctuation(str(row.get("text", "")))
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) < 2 or chinese_ratio(text) < 0.2:
            continue
        group.append(text[:180])
        group_chars += len(text)
        if len(group) >= 12 or group_chars >= 600:
            kept = flush_grouped_short_texts(
                docs,
                seen,
                group,
                source_name="happyme531/YiMeng-JiangHu-WorldChat-1k",
                source_type="game_world_chat",
                category="game_world_chat",
                stats=stats,
            )
            if kept:
                report["kept_docs"] += 1
                report["kept_chars"] += kept
            group = []
            group_chars = 0
        if report["kept_chars"] >= target_chars:
            break
    if group:
        kept = flush_grouped_short_texts(
            docs,
            seen,
            group,
            source_name="happyme531/YiMeng-JiangHu-WorldChat-1k",
            source_type="game_world_chat",
            category="game_world_chat",
            stats=stats,
        )
        if kept:
            report["kept_docs"] += 1
            report["kept_chars"] += kept
    report["elapsed_sec"] = time.time() - started
    return report


def collect_chatharuhi(
    docs: list[dict[str, Any]],
    seen: set[str],
    stats: dict[str, int],
    *,
    target_chars: int,
    max_rows: int,
) -> dict[str, Any]:
    from datasets import load_dataset

    report = {
        "dataset_name": "silk-road/ChatHaruhi-54K-Role-Playing-Dialogue",
        "target_chars": target_chars,
        "max_rows": max_rows,
        "streaming": True,
        "raw_rows_seen": 0,
        "kept_docs": 0,
        "kept_chars": 0,
        "error": None,
    }
    print(f"Streaming ChatHaruhi role dialogue target_chars={target_chars}", flush=True)
    started = time.time()
    ds = load_dataset(
        "silk-road/ChatHaruhi-54K-Role-Playing-Dialogue",
        split="train",
        streaming=True,
        cache_dir=str(CACHE_DIR / "datasets"),
    )
    for row in ds:
        report["raw_rows_seen"] += 1
        if report["raw_rows_seen"] > max_rows:
            break
        user_role = normalize_preserve_punctuation(str(row.get("user_role", "用户"))) or "用户"
        agent_role = normalize_preserve_punctuation(str(row.get("agent_role", "角色"))) or "角色"
        question = normalize_preserve_punctuation(str(row.get("user_question", "")))
        response = normalize_preserve_punctuation(str(row.get("agent_response", "")))
        if not question or not response:
            stats["anime_roleplay_reject_empty_turn"] = stats.get("anime_roleplay_reject_empty_turn", 0) + 1
            continue
        text = f"{user_role}：{question}\n{agent_role}：{response}"
        before = len(docs)
        add_custom_doc(
            docs,
            seen,
            text,
            source_name="silk-road/ChatHaruhi-54K-Role-Playing-Dialogue",
            source_type="anime_roleplay",
            category="anime_roleplay",
            stats=stats,
            min_len=30,
            max_len=1200,
            min_zh_ratio=0.35,
        )
        if len(docs) > before:
            report["kept_docs"] += 1
            report["kept_chars"] += docs[-1]["char_count"]
        if report["raw_rows_seen"] % 10_000 == 0:
            print(
                f"ChatHaruhi rows={report['raw_rows_seen']} kept_docs={report['kept_docs']} kept_chars={report['kept_chars']}",
                flush=True,
            )
        if report["kept_chars"] >= target_chars:
            break
    report["elapsed_sec"] = time.time() - started
    return report


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


def collect_fun(docs: list[dict[str, Any]], seen: set[str], stats: dict[str, int], target_total_chars: int) -> None:
    path = PROJECT_DIR / "28_chinese_fun_corpus_pipeline/data/processed/fun_corpus.txt"
    if not path.exists():
        stats["fun_missing"] = 1
        return
    text = path.read_text(encoding="utf-8", errors="replace")
    for piece in chunk_fun_text(text):
        stats["fun_candidates"] = stats.get("fun_candidates", 0) + 1
        add_doc(
            docs,
            seen,
            piece,
            source_name=str(path.relative_to(PROJECT_DIR)),
            source_type="fun_corpus",
            stats=stats,
            target_total_chars=target_total_chars,
            force_fun=True,
        )


def parse_lccc_dialog(payload: Any) -> list[str]:
    if isinstance(payload, dict):
        value = payload.get("dialog") or payload.get("dialogs") or payload.get("conversation")
    else:
        value = payload
    if not isinstance(value, list):
        return []
    turns = []
    for item in value:
        if isinstance(item, str):
            text = repair_lccc_utterance(item)
            if text:
                turns.append(text)
    return turns


def format_lccc_dialog(turns: list[str]) -> str:
    speakers = ["甲", "乙"]
    lines = []
    for i, turn in enumerate(turns):
        lines.append(f"{speakers[i % 2]}：{turn}")
    return "\n".join(lines)


def collect_lccc(
    docs: list[dict[str, Any]],
    seen: set[str],
    stats: dict[str, int],
    *,
    target_chars: int,
    target_total_chars: int,
    file_key: str,
    max_rows: int,
) -> dict[str, Any]:
    report = {
        "dataset_name": "silver/lccc",
        "file_key": file_key,
        "target_chars": target_chars,
        "max_rows": max_rows,
        "downloaded_file": None,
        "raw_rows_seen": 0,
        "kept_docs": 0,
        "kept_chars": 0,
        "error": None,
    }
    filename = LCCC_FILES[file_key]
    print(f"Downloading/using LCCC {filename}", flush=True)
    path = hf_hub_download(
        repo_id="silver/lccc",
        filename=filename,
        repo_type="dataset",
        cache_dir=str(CACHE_DIR / "lccc"),
    )
    report["downloaded_file"] = path
    print(f"Reading LCCC {path}", flush=True)
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            report["raw_rows_seen"] += 1
            if report["raw_rows_seen"] > max_rows:
                break
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                stats["lccc_reject_json"] = stats.get("lccc_reject_json", 0) + 1
                continue
            turns = parse_lccc_dialog(payload)
            if len(turns) < 2:
                stats["lccc_reject_short_dialog"] = stats.get("lccc_reject_short_dialog", 0) + 1
                continue
            if any(len(turn) < 2 for turn in turns):
                stats["lccc_reject_tiny_turn"] = stats.get("lccc_reject_tiny_turn", 0) + 1
                continue
            text = format_lccc_dialog(turns)
            if chinese_ratio(text) < 0.45:
                stats["lccc_reject_low_chinese_ratio"] = stats.get("lccc_reject_low_chinese_ratio", 0) + 1
                continue
            if has_spam(text) or is_too_repetitive(text):
                stats["lccc_reject_noise"] = stats.get("lccc_reject_noise", 0) + 1
                continue
            fp = fingerprint(text)
            if fp in seen:
                stats["lccc_reject_duplicate"] = stats.get("lccc_reject_duplicate", 0) + 1
                continue
            seen.add(fp)
            docs.append(
                {
                    "id": len(docs),
                    "text": text,
                    "source_name": f"silver/lccc/{filename}",
                    "source_type": "lccc_social_dialogue",
                    "category": "lccc_dialogue",
                    "category_scores": {},
                    "char_count": len(text),
                    "turn_count": len(turns),
                    "chinese_ratio": chinese_ratio(text),
                }
            )
            report["kept_docs"] += 1
            report["kept_chars"] += len(text)
            stats["keep_lccc_dialogue_docs"] = stats.get("keep_lccc_dialogue_docs", 0) + 1
            stats["keep_lccc_dialogue_chars"] = stats.get("keep_lccc_dialogue_chars", 0) + len(text)
            if report["raw_rows_seen"] % 100_000 == 0:
                print(
                    f"LCCC rows={report['raw_rows_seen']} kept_docs={report['kept_docs']} kept_chars={report['kept_chars']}",
                    flush=True,
                )
            if report["kept_chars"] >= target_chars:
                break
    return report


def run_git(args: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    subprocess.run(args, cwd=str(cwd) if cwd else None, env=env, check=True)


def ensure_hana_repo(repo_dir: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "repo_url": HANA_REPO_URL,
        "repo_dir": str(repo_dir),
        "cloned": False,
        "lfs_pulled": False,
    }
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    if not repo_dir.exists():
        env = dict(os.environ)
        env["GIT_LFS_SKIP_SMUDGE"] = "1"
        run_git(["git", "clone", "--depth", "1", HANA_REPO_URL, str(repo_dir)], env=env)
        report["cloned"] = True

    batch_file = repo_dir / "batches" / "batch_001.json"
    needs_lfs = batch_file.exists() and batch_file.read_text(encoding="utf-8", errors="ignore").startswith(
        "version https://git-lfs.github.com/spec"
    )
    if needs_lfs:
        run_git(["git", "lfs", "install", "--local"], cwd=repo_dir)
        run_git(["git", "lfs", "pull"], cwd=repo_dir)
        report["lfs_pulled"] = True
    return report


def parse_hana_dialogue(payload: dict[str, Any]) -> list[str]:
    conversations = payload.get("conversations")
    if not isinstance(conversations, list):
        return []
    turns = []
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        content = repair_lccc_utterance(str(turn.get("content", "")))
        if content:
            turns.append(content)
    return turns


def format_two_speaker_dialog(turns: list[str]) -> str:
    speakers = ["甲", "乙"]
    return "\n".join(f"{speakers[i % 2]}：{turn}" for i, turn in enumerate(turns))


def collect_hana(
    docs: list[dict[str, Any]],
    seen: set[str],
    stats: dict[str, int],
    *,
    target_chars: int,
    max_dialogues: int,
    repo_dir: str | Path,
) -> dict[str, Any]:
    repo_path = Path(repo_dir)
    report: dict[str, Any] = {
        "dataset_name": "xuanxixue/HANA",
        "repo_url": HANA_REPO_URL,
        "repo_dir": str(repo_path),
        "target_chars": target_chars,
        "max_dialogues": max_dialogues,
        "batch_files_seen": 0,
        "raw_dialogues_seen": 0,
        "kept_docs": 0,
        "kept_chars": 0,
        "error": None,
    }
    report["repo_prepare"] = ensure_hana_repo(repo_path)
    batch_files = sorted((repo_path / "batches").glob("batch_*.json"))
    if not batch_files:
        raise FileNotFoundError(f"No HANA batch files found under {repo_path / 'batches'}")

    print(f"Reading HANA target_chars={target_chars}", flush=True)
    for batch_file in batch_files:
        report["batch_files_seen"] += 1
        data = json.loads(batch_file.read_text(encoding="utf-8", errors="replace"))
        dialogues = data.get("dialogues", [])
        if not isinstance(dialogues, list):
            stats["hana_reject_bad_batch"] = stats.get("hana_reject_bad_batch", 0) + 1
            continue
        for payload in dialogues:
            report["raw_dialogues_seen"] += 1
            if report["raw_dialogues_seen"] > max_dialogues:
                break
            if not isinstance(payload, dict):
                stats["hana_reject_bad_payload"] = stats.get("hana_reject_bad_payload", 0) + 1
                continue
            turns = parse_hana_dialogue(payload)
            if len(turns) < 2:
                stats["hana_reject_short_dialog"] = stats.get("hana_reject_short_dialog", 0) + 1
                continue
            if any(len(turn) < 2 for turn in turns):
                stats["hana_reject_tiny_turn"] = stats.get("hana_reject_tiny_turn", 0) + 1
                continue
            text = format_two_speaker_dialog(turns)
            if chinese_ratio(text) < 0.45:
                stats["hana_reject_low_chinese_ratio"] = stats.get("hana_reject_low_chinese_ratio", 0) + 1
                continue
            if has_spam(text) or is_too_repetitive(text):
                stats["hana_reject_noise"] = stats.get("hana_reject_noise", 0) + 1
                continue
            fp = fingerprint(text)
            if fp in seen:
                stats["hana_reject_duplicate"] = stats.get("hana_reject_duplicate", 0) + 1
                continue
            seen.add(fp)
            metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
            docs.append(
                {
                    "id": len(docs),
                    "text": text,
                    "source_name": f"xuanxixue/HANA/{batch_file.name}",
                    "source_type": "hana_synthetic_dialogue",
                    "category": "hana_dialogue",
                    "category_scores": {},
                    "char_count": len(text),
                    "turn_count": len(turns),
                    "topic_category": metadata.get("topic_category"),
                    "conversation_type": metadata.get("conversation_type"),
                    "chinese_ratio": chinese_ratio(text),
                }
            )
            report["kept_docs"] += 1
            report["kept_chars"] += len(text)
            stats["keep_hana_dialogue_docs"] = stats.get("keep_hana_dialogue_docs", 0) + 1
            stats["keep_hana_dialogue_chars"] = stats.get("keep_hana_dialogue_chars", 0) + len(text)
            if report["raw_dialogues_seen"] % 20_000 == 0:
                print(
                    f"HANA rows={report['raw_dialogues_seen']} kept_docs={report['kept_docs']} kept_chars={report['kept_chars']}",
                    flush=True,
                )
            if report["kept_chars"] >= target_chars:
                break
        if report["raw_dialogues_seen"] > max_dialogues or report["kept_chars"] >= target_chars:
            break
    return report


def stream_source(
    docs: list[dict[str, Any]],
    seen: set[str],
    stats: dict[str, int],
    source: dict[str, Any],
    target_total_chars: int,
) -> dict[str, Any]:
    from datasets import load_dataset

    dataset_name = source["dataset_name"]
    target_chars = int(source["target_chars"])
    max_docs = int(source["max_docs"])
    max_raw_chars = int(source["max_raw_chars"])
    report: dict[str, Any] = {
        "dataset_name": dataset_name,
        "target_chars": target_chars,
        "max_docs": max_docs,
        "max_raw_chars": max_raw_chars,
        "streaming": True,
        "error": None,
        "raw_docs_seen": 0,
        "raw_chars_seen": 0,
        "kept_docs": 0,
        "kept_chars": 0,
        "field_counts": {},
    }

    print(f"Streaming {dataset_name} target_chars={target_chars}", flush=True)
    started = time.time()
    ds = load_dataset(dataset_name, split="train", streaming=True, cache_dir=str(CACHE_DIR / "datasets"))
    for row in ds:
        report["raw_docs_seen"] += 1
        if report["raw_docs_seen"] > max_docs:
            break
        field_name, raw_text = detect_text_field(row if isinstance(row, dict) else {})
        if not raw_text:
            stats[f"{dataset_name}_reject_no_text"] = stats.get(f"{dataset_name}_reject_no_text", 0) + 1
            continue
        report["field_counts"][field_name or "<unknown>"] = report["field_counts"].get(field_name or "<unknown>", 0) + 1
        report["raw_chars_seen"] += len(raw_text)
        before = len(docs)
        kept = add_doc(
            docs,
            seen,
            raw_text,
            source_name=dataset_name,
            source_type="streamed_general",
            stats=stats,
            target_total_chars=target_total_chars,
        )
        if kept:
            report["kept_docs"] += 1
            report["kept_chars"] += docs[-1]["char_count"]
        if report["raw_docs_seen"] % 2000 == 0:
            print(
                f"source={dataset_name} raw={report['raw_docs_seen']} kept_docs={report['kept_docs']} "
                f"kept_chars={report['kept_chars']} total_docs={len(docs)}",
                flush=True,
            )
        if report["kept_chars"] >= target_chars:
            break
        if report["raw_chars_seen"] >= max_raw_chars:
            break
    report["elapsed_sec"] = time.time() - started
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


def summarize(docs: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for item in docs:
        category = item["category"]
        if category not in out:
            out[category] = {"docs": 0, "chars": 0}
        out[category]["docs"] += 1
        out[category]["chars"] += item["char_count"]
    return out


QUALITY_ACG_DROP_LINE_RE = re.compile(
    r"(\{\{|\}\}|\{\||\|\}|thumb|\d{2,4}px\||File:|Image:|文件:|图片:|\.jpg\||\.jpeg\||\.png\||\.gif\||https?://|www\.|<references|</references|\[\[|\]\]|外部链接|参考资料|参考来源|相关条目|注释与外部链接|!\[|<table|</td>|rowspan|colspan|\|\s*[A-Za-z_][\w -]{0,24}\s*=|autostart\s*=|loop\s*=|leftbg\s*=|rightbg\s*=|track\s*=|border\s*=)",
    re.IGNORECASE,
)
QUALITY_PRIVATE_OR_REPLACEMENT_RE = re.compile(r"[\uE000-\uF8FF\uFFFD\x7F]")
QUALITY_CJK_SPACED_RUN_RE = re.compile(r"(?:[\u3400-\u9fff][\t \u00A0\u3000]+){3,}[\u3400-\u9fff]")
QUALITY_STRUCTURED_RESIDUE_RE = re.compile(
    r"(!\[|<table|</td>|rowspan|colspan|\{\{|\}\}|\bimage:|\|\s*[A-Za-z_][\w -]{0,24}\s*=|autostart\s*=|loop\s*=|leftbg\s*=|rightbg\s*=|track\s*=|border\s*=)",
    re.IGNORECASE,
)
QUALITY_BILI_STRONG_AD_LINE_RE = re.compile(
    r"(代肝|代练|价格私聊|接单私聊|陪玩接单|有需要私我|无门槛推广|推广兼职|兼职[^。！？\n]{0,20}私我|私聊[^。！？\n]{0,20}接单|接单[^。！？\n]{0,20}私聊|淘宝搜索|点击链接|微信号|QQ群|Q群|加群|群号[:：]?\s*\d{5,}|粉丝群[:：]?\s*\d{5,})",
    re.IGNORECASE,
)
QUALITY_DIALOGISH_FORMAL_RE = re.compile(
    r"(有限公司|公司简介|经营范围|营业部|发展历程|承办单位|活动主题|实施方案|增值税|发票|税务|财政部|国家税务总局|法律问答|法律依据|律师|法院|据.{0,12}报道|记者|新华社|中新网|发布消息|本办法|本条例|公开招聘|资格审查|任职要求|销售指标|客户服务意识|工作经验优先|岗位职责|招生系统|考生|准考证|片方|特辑|领衔主演|独家上线|栏目|节目|扫一扫|孔子学院|博士|企业战略管理|考试成绩)",
)
QUALITY_NATURAL_WEB_FORMAL_RE = re.compile(
    r"(有限公司|公司简介|经营范围|营业执照|公司实力雄厚|经销批发|国家税务总局|本办法|本条例|据.{0,12}报道|记者|新华社|中新网|发布消息|公开招聘|资格审查|岗位职责|任职要求|免责声明|相关阅读|上一篇|下一篇|相关推荐|本文关键词|责任编辑)",
)
QUALITY_SEO_BOILERPLATE_LINE_RE = re.compile(
    r"(免责声明|相关阅读|上一篇|下一篇|相关推荐|本文关键词|责任编辑|点击查看|相关链接)",
)

QUALITY_RULES = {
    "name": QUALITY_GATE_NAME,
    "purpose": "final global quality gate applied after source collection/cache loading",
    "line_filters": {
        "acg_wiki": "drop wiki template/image/table/private-use residue lines",
        "bilibili_comment": "drop strong advertisement/trade signal lines",
        "natural_web/natural_dialogish": "drop SEO boilerplate lines before document-level formal checks",
        "chat_like_sources": "remove segmentation spaces after punctuation and collapse tone markers like ~ ~ ~",
    },
    "doc_filters": {
        "global": "drop documents with replacement/private-use/control-character damage",
        "structured_residue": "drop remaining wiki/markdown/table parameter residue after line filters",
        "cjk_spaced_run": "drop documents with obvious CJK tokenized spacing such as 民 事 判 决 书",
        "natural_web/natural_dialogish": "drop formal/company/tax/legal/news-like documents misclassified into conversational pools",
    },
}


def apply_quality_gate_to_doc(item: dict[str, Any], stats: dict[str, int]) -> str | None:
    """Final global quality gate.

    Source-specific cleaners run when data is first collected. This gate runs after
    collection and after prepared source cache loading, so newer global quality
    rules still apply to cached documents without forcing network re-streaming.
    """

    category = item.get("category", "")
    text = item.get("text") or ""
    text = repair_common_spacing(text)

    if category in CHAT_SPACING_CATEGORIES:
        text = repair_chat_spacing(text)

    if QUALITY_PRIVATE_OR_REPLACEMENT_RE.search(text):
        stats["quality_gate_drop_encoding_damaged_docs"] = stats.get(
            "quality_gate_drop_encoding_damaged_docs", 0
        ) + 1
        return None

    if category == "acg_wiki":
        kept_lines = []
        removed = 0
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if QUALITY_ACG_DROP_LINE_RE.search(stripped):
                removed += 1
                continue
            if QUALITY_PRIVATE_OR_REPLACEMENT_RE.search(stripped):
                removed += 1
                continue
            kept_lines.append(stripped)
        cleaned = "\n".join(kept_lines).strip()
        stats["quality_gate_acg_wiki_removed_lines"] = stats.get("quality_gate_acg_wiki_removed_lines", 0) + removed
        if len(cleaned) < 80 or chinese_ratio(cleaned) < 0.25:
            stats["quality_gate_drop_acg_wiki_docs"] = stats.get("quality_gate_drop_acg_wiki_docs", 0) + 1
            return None
        if QUALITY_CJK_SPACED_RUN_RE.search(cleaned):
            stats["quality_gate_drop_cjk_spaced_docs"] = stats.get("quality_gate_drop_cjk_spaced_docs", 0) + 1
            return None
        if QUALITY_STRUCTURED_RESIDUE_RE.search(cleaned):
            stats["quality_gate_drop_structured_residue_docs"] = stats.get(
                "quality_gate_drop_structured_residue_docs", 0
            ) + 1
            return None
        return cleaned

    if category in {"natural_web", "natural_dialogish"}:
        kept_lines = []
        removed = 0
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if QUALITY_SEO_BOILERPLATE_LINE_RE.search(stripped):
                removed += 1
                continue
            kept_lines.append(stripped)
        text = "\n".join(kept_lines).strip()
        stats["quality_gate_seo_removed_lines"] = stats.get("quality_gate_seo_removed_lines", 0) + removed
        if not text or len(text) < 80:
            stats["quality_gate_drop_seo_empty_docs"] = stats.get("quality_gate_drop_seo_empty_docs", 0) + 1
            return None

    if category == "bilibili_comment":
        kept_lines = []
        removed = 0
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if QUALITY_BILI_STRONG_AD_LINE_RE.search(stripped):
                removed += 1
                continue
            kept_lines.append(stripped)
        cleaned = "\n".join(kept_lines).strip()
        stats["quality_gate_bilibili_removed_lines"] = stats.get("quality_gate_bilibili_removed_lines", 0) + removed
        if len(cleaned) < 20 or chinese_ratio(cleaned) < 0.18:
            stats["quality_gate_drop_bilibili_docs"] = stats.get("quality_gate_drop_bilibili_docs", 0) + 1
            return None
        if QUALITY_CJK_SPACED_RUN_RE.search(cleaned):
            stats["quality_gate_drop_cjk_spaced_docs"] = stats.get("quality_gate_drop_cjk_spaced_docs", 0) + 1
            return None
        if QUALITY_STRUCTURED_RESIDUE_RE.search(cleaned):
            stats["quality_gate_drop_structured_residue_docs"] = stats.get(
                "quality_gate_drop_structured_residue_docs", 0
            ) + 1
            return None
        return cleaned

    if category == "natural_dialogish" and QUALITY_DIALOGISH_FORMAL_RE.search(text):
        stats["quality_gate_drop_natural_dialogish_formal_docs"] = stats.get(
            "quality_gate_drop_natural_dialogish_formal_docs", 0
        ) + 1
        return None

    if category == "natural_web" and QUALITY_NATURAL_WEB_FORMAL_RE.search(text):
        stats["quality_gate_drop_natural_web_formal_docs"] = stats.get(
            "quality_gate_drop_natural_web_formal_docs", 0
        ) + 1
        return None

    if QUALITY_CJK_SPACED_RUN_RE.search(text):
        stats["quality_gate_drop_cjk_spaced_docs"] = stats.get("quality_gate_drop_cjk_spaced_docs", 0) + 1
        return None

    if QUALITY_STRUCTURED_RESIDUE_RE.search(text):
        stats["quality_gate_drop_structured_residue_docs"] = stats.get(
            "quality_gate_drop_structured_residue_docs", 0
        ) + 1
        return None

    return text


def apply_quality_gate(docs: list[dict[str, Any]], stats: dict[str, int]) -> list[dict[str, Any]]:
    cleaned_docs: list[dict[str, Any]] = []
    for item in docs:
        cleaned_text = apply_quality_gate_to_doc(item, stats)
        if cleaned_text is None:
            continue
        if cleaned_text != item.get("text"):
            item = dict(item)
            item["text"] = cleaned_text
        item["id"] = len(cleaned_docs)
        item["char_count"] = len(item["text"])
        item["chinese_ratio"] = chinese_ratio(item["text"])
        cleaned_docs.append(item)
    stats["quality_gate_input_docs"] = len(docs)
    stats["quality_gate_output_docs"] = len(cleaned_docs)
    stats["quality_gate_removed_docs"] = len(docs) - len(cleaned_docs)
    return cleaned_docs


def write_docs_jsonl(path: Path, docs: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for item in docs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_preview(path: Path, docs: list[dict[str, Any]], limit: int = 40) -> None:
    lines = ["# Canonical Lab Corpus Preview", ""]
    for item in docs[:limit]:
        lines.extend(
            [
                f"## doc {item['id']} | {item['category']} | {item['source_name']} | chars={item['char_count']}",
                "",
                item["text"][:1000],
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the canonical lesson 30 lab BPE data.")
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=2030)
    parser.add_argument("--lccc-target-chars", type=int, default=24_000_000)
    parser.add_argument("--lccc-file", choices=sorted(LCCC_FILES), default="large")
    parser.add_argument("--lccc-max-rows", type=int, default=1_500_000)
    parser.add_argument("--hana-target-chars", type=int, default=2_000_000)
    parser.add_argument("--hana-max-dialogues", type=int, default=100_000)
    parser.add_argument("--hana-repo-dir", type=str, default=str(HANA_REPO_DIR))
    parser.add_argument("--match-lccc-to-hana", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bilibili-target-chars", type=int, default=ACG_SOURCE_DEFAULTS["bilibili_target_chars"])
    parser.add_argument("--bilibili-max-rows", type=int, default=ACG_SOURCE_DEFAULTS["bilibili_max_rows"])
    parser.add_argument("--moegirl-target-chars", type=int, default=ACG_SOURCE_DEFAULTS["moegirl_target_chars"])
    parser.add_argument("--moegirl-max-rows", type=int, default=ACG_SOURCE_DEFAULTS["moegirl_max_rows"])
    parser.add_argument("--worldchat-target-chars", type=int, default=ACG_SOURCE_DEFAULTS["worldchat_target_chars"])
    parser.add_argument("--worldchat-max-rows", type=int, default=ACG_SOURCE_DEFAULTS["worldchat_max_rows"])
    parser.add_argument("--chatharuhi-target-chars", type=int, default=ACG_SOURCE_DEFAULTS["chatharuhi_target_chars"])
    parser.add_argument("--chatharuhi-max-rows", type=int, default=ACG_SOURCE_DEFAULTS["chatharuhi_max_rows"])
    parser.add_argument("--skip-lccc", action="store_true")
    parser.add_argument("--skip-hana", action="store_true")
    parser.add_argument("--skip-acg-sources", action="store_true")
    parser.add_argument("--skip-stream", action="store_true")
    parser.add_argument("--stream-cache-only", action="store_true")
    parser.add_argument("--refresh-source-cache", action="store_true")
    parser.add_argument("--reuse-tokenizer", action="store_true")
    args = parser.parse_args()

    for path in [RAW_DIR, PROCESSED_DIR, METADATA_DIR, TOKENIZER_DIR, CACHE_DIR, PREPARED_SOURCE_CACHE_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    os.environ.setdefault("HF_DATASETS_CACHE", str(CACHE_DIR / "datasets"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    pipeline_started = time.perf_counter()
    phase_started = pipeline_started
    phase_times: dict[str, float] = {}

    target_total_chars = (
        sum(item["target_chars"] for item in DEFAULT_SOURCE_PLAN)
        + args.lccc_target_chars
        + args.hana_target_chars
        + args.bilibili_target_chars
        + args.moegirl_target_chars
        + args.worldchat_target_chars
        + args.chatharuhi_target_chars
        + 2_500_000
    )
    docs: list[dict[str, Any]] = []
    seen: set[str] = set()
    stats: dict[str, int] = {}
    source_reports: list[dict[str, Any]] = []
    acg_source_reports: list[dict[str, Any]] = []

    collect_fun(docs, seen, stats, target_total_chars)
    phase_started = log_phase("collect_fun", phase_started, phase_times)

    hana_report: dict[str, Any] = {"skipped": args.skip_hana}
    if not args.skip_hana:
        try:
            hana_kwargs = {
                "target_chars": args.hana_target_chars,
                "max_dialogues": args.hana_max_dialogues,
                "repo_dir": args.hana_repo_dir,
            }
            hana_report = collect_with_cache(
                "hana_dialogue",
                collect_hana,
                docs,
                seen,
                stats,
                refresh_cache=args.refresh_source_cache,
                cache_payload=hana_kwargs,
                collector_kwargs=hana_kwargs,
            )
        except Exception as exc:  # noqa: BLE001
            hana_report = {
                "dataset_name": "xuanxixue/HANA",
                "target_chars": args.hana_target_chars,
                "error": f"{type(exc).__name__}: {exc}",
            }
            print("HANA failed:", hana_report, flush=True)
    phase_started = log_phase("collect_hana", phase_started, phase_times)

    effective_lccc_target_chars = args.lccc_target_chars
    if args.match_lccc_to_hana and not args.skip_hana and hana_report.get("kept_chars"):
        effective_lccc_target_chars = min(args.lccc_target_chars, int(hana_report["kept_chars"]))

    lccc_report: dict[str, Any] = {"skipped": args.skip_lccc}
    if not args.skip_lccc:
        try:
            lccc_collector_kwargs = {
                "target_chars": effective_lccc_target_chars,
                "target_total_chars": target_total_chars,
                "file_key": args.lccc_file,
                "max_rows": args.lccc_max_rows,
            }
            lccc_cache_payload = {
                **lccc_collector_kwargs,
                "configured_target_chars": args.lccc_target_chars,
                "matched_to_hana": bool(args.match_lccc_to_hana),
                "hana_kept_chars": int(hana_report.get("kept_chars") or 0),
            }
            lccc_report = collect_with_cache(
                f"lccc_{args.lccc_file}",
                collect_lccc,
                docs,
                seen,
                stats,
                refresh_cache=args.refresh_source_cache,
                cache_payload=lccc_cache_payload,
                collector_kwargs=lccc_collector_kwargs,
            )
            lccc_report["configured_target_chars"] = args.lccc_target_chars
            lccc_report["matched_to_hana"] = bool(args.match_lccc_to_hana)
            lccc_report["hana_kept_chars"] = int(hana_report.get("kept_chars") or 0)
        except Exception as exc:  # noqa: BLE001
            lccc_report = {
                "dataset_name": "silver/lccc",
                "file_key": args.lccc_file,
                "target_chars": effective_lccc_target_chars,
                "configured_target_chars": args.lccc_target_chars,
                "matched_to_hana": bool(args.match_lccc_to_hana),
                "error": f"{type(exc).__name__}: {exc}",
            }
            print("LCCC failed:", lccc_report, flush=True)
    phase_started = log_phase("collect_lccc", phase_started, phase_times)

    if not args.skip_acg_sources:
        acg_jobs = [
            (
                "bilibili_comment",
                collect_bilibili_comments,
                {"target_chars": args.bilibili_target_chars, "max_rows": args.bilibili_max_rows},
            ),
            (
                "acg_wiki",
                collect_moegirl,
                {"target_chars": args.moegirl_target_chars, "max_rows": args.moegirl_max_rows},
            ),
            (
                "game_world_chat",
                collect_worldchat,
                {"target_chars": args.worldchat_target_chars, "max_rows": args.worldchat_max_rows},
            ),
            (
                "anime_roleplay",
                collect_chatharuhi,
                {"target_chars": args.chatharuhi_target_chars, "max_rows": args.chatharuhi_max_rows},
            ),
        ]
        for source_label, collector, kwargs in acg_jobs:
            if kwargs["target_chars"] <= 0:
                acg_source_reports.append({"source_label": source_label, "skipped": True, "reason": "target_chars <= 0"})
                continue
            try:
                acg_source_reports.append(
                    collect_with_cache(
                        source_label,
                        collector,
                        docs,
                        seen,
                        stats,
                        refresh_cache=args.refresh_source_cache,
                        cache_payload=kwargs,
                        collector_kwargs=kwargs,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                report = {
                    "source_label": source_label,
                    "streaming": True,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                acg_source_reports.append(report)
                print("ACG/community source failed:", report, flush=True)
    phase_started = log_phase("collect_acg_sources", phase_started, phase_times)

    if not args.skip_stream:
        for source in DEFAULT_SOURCE_PLAN:
            try:
                stream_kwargs = {"source": source, "target_total_chars": target_total_chars}
                report = collect_with_cache(
                    f"stream_{source['dataset_name']}",
                    stream_source,
                    docs,
                    seen,
                    stats,
                    refresh_cache=args.refresh_source_cache,
                    cache_payload=stream_kwargs,
                    collector_kwargs=stream_kwargs,
                    allow_collect=not args.stream_cache_only,
                )
            except Exception as exc:  # noqa: BLE001
                report = {
                    "dataset_name": source["dataset_name"],
                    "streaming": True,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                print("Streaming failed:", report, flush=True)
            source_reports.append(report)
    phase_started = log_phase("collect_stream_sources", phase_started, phase_times)

    docs = apply_quality_gate(docs, stats)
    phase_started = log_phase("quality_gate", phase_started, phase_times)

    if len(docs) < 100:
        raise RuntimeError("Collected too few documents. Check dataset access or run without --skip-stream.")

    random.shuffle(docs)
    for i, item in enumerate(docs):
        item["id"] = i
    phase_started = log_phase("shuffle_docs", phase_started, phase_times)

    docs_jsonl_path = RAW_DIR / "lab_bpe_mixed_docs.jsonl"
    corpus_path = RAW_DIR / "lab_bpe_mixed_corpus.txt"
    preview_path = REPORT_DIR / "lab_bpe_32768_preview.md"
    write_docs_jsonl(docs_jsonl_path, docs)
    write_preview(preview_path, docs)

    corpus = "\n\n<|doc_sep|>\n\n".join(item["text"] for item in docs) + "\n"
    corpus_path.write_text(corpus, encoding="utf-8")
    phase_started = log_phase("write_docs_and_corpus", phase_started, phase_times)

    expected_tokenizer_path = TOKENIZER_DIR / f"lab_byte_bpe_{args.vocab_size}.json"
    if args.reuse_tokenizer and expected_tokenizer_path.exists():
        print(f"Reusing tokenizer {expected_tokenizer_path}", flush=True)
        tokenizer = Tokenizer.from_file(str(expected_tokenizer_path))
    else:
        tokenizer = train_byte_bpe(corpus_path, args.vocab_size)
    phase_started = log_phase("load_or_train_tokenizer", phase_started, phase_times)
    actual_vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    tokenizer_path = TOKENIZER_DIR / f"lab_byte_bpe_{actual_vocab_size}.json"
    if not (args.reuse_tokenizer and tokenizer_path.exists()):
        tokenizer.save(str(tokenizer_path))

    encoded = tokenizer.encode(corpus, add_special_tokens=False)
    phase_started = log_phase("encode_corpus", phase_started, phase_times)
    token_ids = encoded.ids
    split = int(len(token_ids) * (1.0 - args.val_ratio))
    split = min(max(split, 2048), len(token_ids) - 2048)
    train = np.array(token_ids[:split], dtype=np.int32)
    val = np.array(token_ids[split:], dtype=np.int32)

    train_path = PROCESSED_DIR / f"train_tokens_lab_bpe_{actual_vocab_size}.npy"
    val_path = PROCESSED_DIR / f"val_tokens_lab_bpe_{actual_vocab_size}.npy"
    np.save(train_path, train)
    np.save(val_path, val)
    phase_started = log_phase("save_token_arrays", phase_started, phase_times)

    unk_id = tokenizer.token_to_id("<unk>")
    unk_count = int(sum(1 for token_id in token_ids if token_id == unk_id)) if unk_id is not None else 0
    category_summary = summarize(docs)
    total_doc_chars = max(1, sum(item["chars"] for item in category_summary.values()))
    char_counts = [item["char_count"] for item in docs]
    samples = [
        "人工智能正在改变我们的学习方式。",
        "今天我们用 MLX 在 MacBook Pro 上训练一个中文 Tiny GPT。",
        "老哥稳，这波属于是把本地小模型压榨到极限了。",
        "中文标点应该保留：逗号，句号。感叹号！",
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
        "unk_count": unk_count,
        "unk_ratio": unk_count / len(token_ids),
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
        "category_summary": category_summary,
        "category_share": {k: v["chars"] / total_doc_chars for k, v in category_summary.items()},
        "category_caps": CATEGORY_CAPS,
        "source_plan": DEFAULT_SOURCE_PLAN,
        "acg_source_defaults": ACG_SOURCE_DEFAULTS,
        "acg_source_reports": acg_source_reports,
        "lccc_report": lccc_report,
        "hana_report": hana_report,
        "source_reports": source_reports,
        "stats": stats,
        "preview": preview,
        "normalization": "preserve Chinese punctuation; no NFKC; remove abnormal spaces around Chinese punctuation, between CJK characters, and at CJK/ASCII boundaries for dialogue sources",
        "pre_tokenizer": "ByteLevel(add_prefix_space=False)",
        "source_cache_version": SOURCE_CACHE_VERSION,
        "quality_gate_name": QUALITY_GATE_NAME,
        "quality_rules": QUALITY_RULES,
        "reuse_tokenizer": args.reuse_tokenizer,
        "refresh_source_cache": args.refresh_source_cache,
        "phase_times": phase_times,
        "total_prepare_elapsed_sec": time.perf_counter() - pipeline_started,
        "note": "Canonical lab corpus. No v1/v2/v3 split; this file is the current training data source.",
    }
    metadata_path = METADATA_DIR / f"lab_bpe_{actual_vocab_size}_metadata.json"
    write_json(metadata_path, metadata)

    report = [
        "# Canonical Lab BPE 数据报告",
        "",
        "## 结论",
        "",
        "- 这是当前唯一主线 lab tokenizer / corpus，不再使用 v1/v2/v3 命名。",
        "- LCCC 是首位对话数据源，用于显著提高日常中文短对话占比。",
        "- HANA 是 AI 生成的结构化中文闲聊数据，优点是格式干净，风险是表达可能更模板化，所以默认只低比例加入。",
        "- B 站评论、萌娘百科、玩家公屏和动漫角色对话是新的次级数据来源。",
        "- 网页数据退居补充层，弱化科普、法律、医疗、新闻、公文腔。",
        "- 第 28 课趣味语料全部加入。",
        "- 清洗不使用 `NFKC`，中文标点保留。",
        "- 对 LCCC / HANA 等对话源，会去掉中文分词空格、中文标点异常空格，以及中文和数字/英文边界空格。",
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
        "## 类别构成",
        "",
    ]
    for category, payload in sorted(category_summary.items(), key=lambda item: -item[1]["chars"]):
        report.append(
            f"- `{category}`：docs=`{payload['docs']}`, chars=`{payload['chars']}`, share=`{payload['chars'] / total_doc_chars:.2%}`"
        )
    report.extend(
        [
            "",
            "## Source reports",
            "",
            "### LCCC",
            "",
            "```json",
            json.dumps(lccc_report, ensure_ascii=False, indent=2),
            "```",
            "",
            "### HANA",
            "",
            "```json",
            json.dumps(hana_report, ensure_ascii=False, indent=2),
            "```",
            "",
            "### ACG / Bilibili / player community sources",
            "",
            "```json",
            json.dumps(acg_source_reports, ensure_ascii=False, indent=2),
            "```",
            "",
            "### Web sources",
            "",
            "```json",
            json.dumps(source_reports, ensure_ascii=False, indent=2),
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
        report.extend(
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
    (REPORT_DIR / f"lab_bpe_{actual_vocab_size}_report.md").write_text("\n".join(report), encoding="utf-8")

    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print("metadata:", metadata_path)


if __name__ == "__main__":
    main()
