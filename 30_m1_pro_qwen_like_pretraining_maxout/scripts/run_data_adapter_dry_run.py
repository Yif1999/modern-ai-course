from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

CURRENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CURRENT_DIR / "scripts"))

from data_adapters import AdapterConfig  # noqa: E402
from data_adapters.factory import build_adapter  # noqa: E402


MANIFEST_PATH = CURRENT_DIR / "data" / "metadata" / "chatlm_scale_data_sources.json"
OUTPUT_DIR = CURRENT_DIR / "outputs" / "data_adapter_preview"
REPORT_DIR = CURRENT_DIR / "outputs" / "reports"
PREPARED_SOURCE_DIR = Path(
    os.environ.get("LAB_PREPARED_SOURCE_DIR", CURRENT_DIR / "data" / "cache" / "prepared_sources")
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def chinese_ratio(text: str) -> float:
    compact = [ch for ch in text if not ch.isspace()]
    if not compact:
        return 0.0
    count = sum(1 for ch in compact if "\u4e00" <= ch <= "\u9fff")
    return count / len(compact)


def simple_noise_flags(text: str) -> list[str]:
    flags = []
    if len(text) < 10:
        flags.append("too_short")
    if chinese_ratio(text) < 0.2:
        flags.append("low_chinese_ratio")
    if re.search(r"https?://|www\\.", text):
        flags.append("url")
    if re.search(r"版权所有|ICP备案|责任编辑|网站地图|联系我们", text):
        flags.append("web_boilerplate")
    if re.search(r"据.+?报道|来源[:：]|新华社|中新网|央视新闻|记者从", text):
        flags.append("news_wire_style")
    if re.search(r"第[一二三四五六七八九十百千万0-9]+条", text) and re.search(r"规定|办法|条例|人民法院|合同|赔偿|清算|债务|责任", text):
        flags.append("legal_template")
    if re.search(r"免责声明|相关阅读|点击查看|更多相关|本文链接|未经授权", text):
        flags.append("seo_boilerplate")
    if re.search(r"<[^>]{1,80}>|\{\||\|\}|class=|style=", text):
        flags.append("markup_noise")
    if len(re.findall(r"\\.(?:jpg|jpeg|png|gif|webp)\\b", text, flags=re.IGNORECASE)) >= 5:
        flags.append("many_media_filenames")
    if re.search(r"(.)\1{8,}", text):
        flags.append("repeated_char")
    if len(text) >= 240:
        spans = [text[i : i + 80] for i in range(0, max(0, len(text) - 80), 40)]
        compact_spans = [re.sub(r"\s+", "", span) for span in spans if len(re.sub(r"\s+", "", span)) >= 50]
        if len(compact_spans) != len(set(compact_spans)):
            flags.append("repeated_span")
    if text.count("\ufffd") > 0 or text.count("�") > 0:
        flags.append("replacement_char")
    return flags


def resolve_existing_source(source: dict[str, Any]) -> dict[str, Any] | None:
    if source["source_name"] != "existing_lab_mixture_v2":
        return None
    path = CURRENT_DIR / "data" / "raw" / "lab_mixture_v2_docs.jsonl"
    if not path.exists():
        return None
    return {
        "source_name": source["source_name"],
        "source_type": source["category"],
        "source_group": source["group"],
        "path": str(path),
        "max_docs": 200,
        "enabled": True,
        "field_map": {"text": "text"},
        "options": {"adapter": "local_jsonl"},
    }


def source_to_adapter_config(source: dict[str, Any], *, include_remote: bool, per_source_docs: int) -> AdapterConfig | None:
    local = resolve_existing_source(source)
    if local:
        local["max_docs"] = per_source_docs
        return AdapterConfig(**local)

    if not include_remote:
        return None

    source_name = source["source_name"]
    field_map: dict[str, str] = {}
    dataset_name = None
    split = "train"
    options: dict[str, Any] = {"adapter": "hf_dataset"}

    if source_name == "Zhihu-KOL":
        dataset_name = "wangrui6/Zhihu-KOL"
        field_map = {"instruction": "INSTRUCTION", "response": "RESPONSE"}
    elif source_name == "BELLE short-answer subset":
        dataset_name = "BelleGroup/train_3.5M_CN"
        field_map = {"conversations": "conversations"}
    elif source_name == "baike_qa2019":
        dataset_name = "shaowenchen/baikeqa_zh"
        field_map = {"question": "title", "context": "desc", "answer": "answer"}
    elif source_name == "Chinese-medical-dialogue-data":
        dataset_name = "ticoAg/Chinese-medical-dialogue"
        field_map = {"instruction": "instruction", "input": "input", "response": "output"}
    elif source_name == "webtext2019zh":
        dataset_name = "json"
        field_map = {"text": "text"}
        options["data_files"] = "https://huggingface.co/datasets/YeungNLP/firefly-pretrain-dataset/resolve/main/webText2019zh.jsonl"
    elif source_name == "Chinese Wikipedia":
        dataset_name = "json"
        field_map = {"text": "text"}
        options["data_files"] = "https://huggingface.co/datasets/YeungNLP/firefly-pretrain-dataset/resolve/main/wiki_zh.jsonl"
    elif source_name == "p208p2002/wudao":
        dataset_name = "p208p2002/wudao"
        field_map = {"text": "content"}
    elif source_name == "BAAI/CCI3-Data":
        # The official BAAI/CCI3-Data repo is gated on Hugging Face. TiWu-Lab/CCI3
        # exposes a streaming-compatible mirror with the same text-oriented role,
        # so use it as the current bounded-sampling fallback.
        dataset_name = "TiWu-Lab/CCI3"
        field_map = {"text": "text"}
    else:
        return None

    return AdapterConfig(
        source_name=source_name,
        source_type=source.get("category", ""),
        source_group=source.get("group", ""),
        dataset_name=dataset_name,
        split=split,
        max_docs=per_source_docs,
        enabled=True,
        streaming=True,
        field_map=field_map,
        options=options,
    )


def prepared_cache_family(path: Path) -> str:
    name = path.name
    for prefix in [
        "acg_wiki",
        "anime_roleplay",
        "bilibili_comment",
        "game_world_chat",
        "hana_dialogue",
        "lccc_large",
        "stream_Morton-Li_ChineseWebText2.0-HighQuality",
        "stream_Skywork_SkyPile-150B",
        "stream_opencsg_Fineweb-Edu-Chinese-V2.1",
    ]:
        if name.startswith(prefix):
            return prefix
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            first_line = f.readline().strip()
        if first_line:
            first = json.loads(first_line)
            if isinstance(first, dict):
                source_name = str(first.get("source_name") or "").strip()
                dataset_name = ""
                metadata = first.get("metadata")
                if isinstance(metadata, dict):
                    dataset_name = str(metadata.get("dataset_name") or "").strip()
                return source_name or dataset_name or path.stem
    except Exception:
        pass
    return path.stem


def prepared_cache_configs(per_source_docs: int, *, dedupe_families: bool = True) -> list[AdapterConfig]:
    if not PREPARED_SOURCE_DIR.exists():
        return []
    paths = sorted(path for path in PREPARED_SOURCE_DIR.glob("*.jsonl") if path.stat().st_size > 0)
    if dedupe_families:
        by_family: dict[str, Path] = {}
        for path in paths:
            family = prepared_cache_family(path)
            current = by_family.get(family)
            # Prefer the largest file for a duplicated source family. This keeps
            # one stable source version while avoiding repeated cache variants.
            if current is None or path.stat().st_size > current.stat().st_size:
                by_family[family] = path
        paths = [by_family[key] for key in sorted(by_family)]
    configs = []
    for path in paths:
        stem = path.name
        source_type = "prepared_cache"
        source_name = stem
        source_group = "prepared_cache"
        try:
            with path.open("r", encoding="utf-8", errors="replace") as f:
                first_line = f.readline().strip()
            if first_line:
                first = json.loads(first_line)
                if isinstance(first, dict):
                    source_name = str(first.get("source_name") or source_name)
                    source_type = str(first.get("source_type") or first.get("category") or source_type)
                    source_group = str(first.get("source_group") or source_group)
        except Exception:
            pass
        for prefix, category in [
            ("lccc_large", "dialogue_short_chat"),
            ("hana_dialogue", "dialogue_short_chat"),
            ("bilibili_comment", "comments_acg_game"),
            ("acg_wiki", "comments_acg_game"),
            ("anime_roleplay", "comments_acg_game"),
            ("game_world_chat", "comments_acg_game"),
            ("stream_", "general_web_backbone"),
        ]:
            if stem.startswith(prefix):
                source_type = category
                break
        configs.append(
            AdapterConfig(
                source_name=source_name,
                source_type=source_type,
                source_group=source_group,
                path=str(path),
                max_docs=per_source_docs,
                enabled=True,
                field_map={"text": "text"},
                options={"adapter": "local_jsonl"},
            )
        )
    return configs


def run_dry(
    manifest_path: Path,
    *,
    include_remote: bool,
    per_source_docs: int,
    include_prepared_cache: bool,
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    preview_path = OUTPUT_DIR / "adapted_preview.jsonl"
    if preview_path.exists():
        preview_path.unlink()

    stats: dict[str, Any] = {
        "manifest": str(manifest_path),
        "include_remote": include_remote,
        "include_prepared_cache": include_prepared_cache,
        "per_source_docs": per_source_docs,
        "sources": {},
        "total_docs": 0,
        "total_chars": 0,
        "category_chars": defaultdict(int),
        "category_docs": defaultdict(int),
        "noise_flags": Counter(),
        "skipped_sources": [],
        "errors": {},
    }
    preview_rows: list[dict[str, Any]] = []

    adapter_configs: list[AdapterConfig] = []
    if include_prepared_cache:
        adapter_configs.extend(prepared_cache_configs(per_source_docs))

    for source in manifest["sources"]:
        cfg = source_to_adapter_config(source, include_remote=include_remote, per_source_docs=per_source_docs)
        if cfg is None:
            stats["skipped_sources"].append(
                {
                    "source_name": source["source_name"],
                    "reason": "no local adapter in dry-run or remote disabled",
                }
            )
            continue
        adapter_configs.append(cfg)

    for cfg in adapter_configs:
        source_docs = 0
        source_chars = 0
        try:
            adapter = build_adapter(cfg)
            rows = []
            for doc in adapter.iter_documents():
                payload = doc.to_json()
                payload["chinese_ratio"] = chinese_ratio(doc.text)
                flags = simple_noise_flags(doc.text)
                payload["noise_flags"] = flags
                rows.append(payload)
                if len(preview_rows) < 80:
                    preview_rows.append(payload)
                source_docs += 1
                source_chars += len(doc.text)
                stats["total_docs"] += 1
                stats["total_chars"] += len(doc.text)
                stats["category_docs"][doc.source_type] += 1
                stats["category_chars"][doc.source_type] += len(doc.text)
                for flag in flags:
                    stats["noise_flags"][flag] += 1
            append_jsonl(preview_path, rows)
        except Exception as exc:  # noqa: BLE001
            stats["errors"][cfg.source_name] = f"{type(exc).__name__}: {exc}"
            continue

        stats["sources"][cfg.source_name] = {
            "docs": source_docs,
            "chars": source_chars,
            "category": cfg.source_type,
            "group": cfg.source_group,
        }

    stats["category_docs"] = dict(stats["category_docs"])
    stats["category_chars"] = dict(stats["category_chars"])
    stats["noise_flags"] = dict(stats["noise_flags"])

    write_json(OUTPUT_DIR / "adapter_dry_run_stats.json", stats)
    write_json(OUTPUT_DIR / "adapter_preview_first_80.json", preview_rows)
    write_markdown_report(stats, REPORT_DIR / "data_adapter_dry_run_report.md")
    return stats


def write_markdown_report(stats: dict[str, Any], path: Path) -> None:
    lines = [
        "# Data Adapter Dry Run Report",
        "",
        f"- include_remote: `{stats['include_remote']}`",
        f"- per_source_docs: `{stats['per_source_docs']}`",
        f"- total_docs: `{stats['total_docs']}`",
        f"- total_chars: `{stats['total_chars']}`",
        "",
        "## Sources",
        "",
        "| source | docs | chars | category | group |",
        "|---|---:|---:|---|---|",
    ]
    for source_name, item in stats["sources"].items():
        lines.append(f"| `{source_name}` | {item['docs']} | {item['chars']} | `{item['category']}` | `{item['group']}` |")

    lines.extend(["", "## Category chars", "", "| category | docs | chars |", "|---|---:|---:|"])
    for category, chars in sorted(stats["category_chars"].items()):
        lines.append(f"| `{category}` | {stats['category_docs'].get(category, 0)} | {chars} |")

    lines.extend(["", "## Noise flags", "", "| flag | count |", "|---|---:|"])
    for flag, count in sorted(stats["noise_flags"].items()):
        lines.append(f"| `{flag}` | {count} |")

    if stats["skipped_sources"]:
        lines.extend(["", "## Skipped sources", ""])
        for item in stats["skipped_sources"]:
            lines.append(f"- `{item['source_name']}`: {item['reason']}")

    if stats["errors"]:
        lines.extend(["", "## Errors", ""])
        for name, error in stats["errors"].items():
            lines.append(f"- `{name}`: {error}")

    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "```text",
            str(OUTPUT_DIR / "adapted_preview.jsonl"),
            str(OUTPUT_DIR / "adapter_dry_run_stats.json"),
            str(OUTPUT_DIR / "adapter_preview_first_80.json"),
            "```",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dry-run data source adapters without large downloads.")
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    parser.add_argument("--include-remote", action="store_true", help="Enable tiny streaming samples for supported HF sources.")
    parser.add_argument("--no-prepared-cache", action="store_true", help="Do not scan local prepared_sources cache.")
    parser.add_argument("--per-source-docs", type=int, default=50)
    args = parser.parse_args()

    stats = run_dry(
        Path(args.manifest),
        include_remote=args.include_remote,
        per_source_docs=args.per_source_docs,
        include_prepared_cache=not args.no_prepared_cache,
    )
    print("total_docs:", stats["total_docs"])
    print("total_chars:", stats["total_chars"])
    print("report:", REPORT_DIR / "data_adapter_dry_run_report.md")
    print("preview:", OUTPUT_DIR / "adapted_preview.jsonl")


if __name__ == "__main__":
    main()
