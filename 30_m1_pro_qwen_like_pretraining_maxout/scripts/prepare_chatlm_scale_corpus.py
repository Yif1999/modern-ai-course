from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

CURRENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CURRENT_DIR / "scripts"))

from data_adapters.factory import build_adapter  # noqa: E402
from run_data_adapter_dry_run import (  # noqa: E402
    MANIFEST_PATH,
    PREPARED_SOURCE_DIR,
    chinese_ratio,
    prepared_cache_configs,
    simple_noise_flags,
    source_to_adapter_config,
)


RAW_DIR = CURRENT_DIR / "data" / "raw"
REPORT_DIR = CURRENT_DIR / "outputs" / "reports"

DEFAULT_CATEGORY_WEIGHTS = {
    "dialogue_short_chat": 0.35,
    "comments_acg_game": 0.20,
    "qa_short_answer": 0.25,
    "general_web_backbone": 0.15,
    "mixed_existing": 0.05,
}

SOURCE_CHAR_CAPS = {
    "acg_wiki": 0.10,
    "Chinese-medical-dialogue-data": 0.02,
    "BAAI/CCI3-Data": 0.04,
    "Chinese Wikipedia": 0.04,
    "p208p2002/wudao": 0.06,
    "BELLE short-answer subset": 0.10,
    "Zhihu-KOL": 0.10,
}

BUILD_PRESETS = {
    "test_100m": ("chatlm_scale_100m_test", 100_000_000),
    "target_1b": ("chatlm_scale_1b", 1_000_000_000),
    "target_2b": ("chatlm_scale_2b", 2_000_000_000),
    "target_4b": ("chatlm_scale_4b", 4_000_000_000),
    "target_10b": ("chatlm_scale_10b", 10_000_000_000),
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_training_text(text: str) -> str:
    """统一训练文本格式，保留中文标点，不把中文标点转成英文标点。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    text = re.sub(r"[ \t]+", " ", text)

    # LCCC 等对话数据常有“额， 我”这类标点后空格；聊天模型训练里这会变成干扰模式。
    text = re.sub(r"([，。！？；：、）】》」』])\s+", r"\1", text)
    text = re.sub(r"\s+([，。！？；：、（【《「『])", r"\1", text)

    # 普通中文聊天里，中英文/数字通常贴合书写；保留英文词内部空格。
    text = re.sub(r"([\u4e00-\u9fff])\s+([A-Za-z0-9])", r"\1\2", text)
    text = re.sub(r"([A-Za-z0-9])\s+([\u4e00-\u9fff])", r"\1\2", text)

    # 只压缩过度空行，不强行改写普通换行。
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def should_keep(text: str, *, min_chars: int) -> tuple[bool, list[str]]:
    flags = simple_noise_flags(text)
    if len(text) < min_chars:
        flags.append("below_min_chars")
    if chinese_ratio(text) < 0.15:
        flags.append("low_chinese_ratio_hard_drop")
    hard_drop = {
        "below_min_chars",
        "low_chinese_ratio_hard_drop",
        "replacement_char",
        "legal_template",
        "seo_boilerplate",
        "markup_noise",
        "many_media_filenames",
        "repeated_span",
    }
    return not any(flag in hard_drop for flag in flags), flags


def text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def category_quotas(target_chars: int) -> dict[str, int]:
    total_weight = sum(DEFAULT_CATEGORY_WEIGHTS.values())
    return {
        category: int(target_chars * weight / total_weight)
        for category, weight in DEFAULT_CATEGORY_WEIGHTS.items()
    }


def source_cap(source_name: str, target_chars: int) -> int | None:
    for key, share in SOURCE_CHAR_CAPS.items():
        if source_name.startswith(key) or source_name == key:
            return int(target_chars * share)
    return None


def adapter_configs(
    manifest: dict[str, Any],
    *,
    include_remote: bool,
    include_prepared_cache: bool,
    max_docs_per_source: int | None,
    max_chars_per_source: int | None,
):
    configs = []
    if include_prepared_cache:
        if not PREPARED_SOURCE_DIR.exists():
            print("prepared cache not found:", PREPARED_SOURCE_DIR)
        else:
            configs.extend(prepared_cache_configs(max_docs_per_source or 10_000_000))

    for source in manifest["sources"]:
        sampling_plan = source.get("sampling_plan", {}) or {}
        source_max_docs = max_docs_per_source
        if source_max_docs is None:
            source_max_docs = sampling_plan.get("max_docs", sampling_plan.get("max_rows", 10_000_000))

        cfg = source_to_adapter_config(
            source,
            include_remote=include_remote,
            per_source_docs=source_max_docs,
        )
        if cfg is None:
            continue
        if max_chars_per_source is not None:
            cfg.max_chars = max_chars_per_source
        elif cfg.max_chars is None and sampling_plan.get("max_chars") is not None:
            cfg.max_chars = int(sampling_plan["max_chars"])
        configs.append(cfg)
    return configs


def build_corpus(
    *,
    manifest_path: Path,
    output_version: str,
    target_tokens: int,
    chars_per_token: float,
    include_remote: bool,
    include_prepared_cache: bool,
    max_docs_per_source: int | None,
    max_chars_per_source: int | None,
    min_chars: int,
    probe_only: bool,
    use_source_caps: bool,
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    target_chars = int(target_tokens * chars_per_token)
    if probe_only:
        target_chars = min(target_chars, 300_000)
        max_docs_per_source = min(max_docs_per_source or 50, 50)

    docs_path = RAW_DIR / f"{output_version}_docs.jsonl"
    corpus_path = RAW_DIR / f"{output_version}_corpus.txt"
    preview_path = REPORT_DIR / f"{output_version}_preview.md"
    report_path = REPORT_DIR / f"{output_version}_build_report.json"
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    quotas = category_quotas(target_chars)
    category_chars = defaultdict(int)
    category_docs = defaultdict(int)
    source_chars = defaultdict(int)
    source_docs = defaultdict(int)
    noise_flags = Counter()
    seen_hashes: set[str] = set()
    preview_rows: list[dict[str, Any]] = []
    errors: dict[str, str] = {}

    configs = adapter_configs(
        manifest,
        include_remote=include_remote,
        include_prepared_cache=include_prepared_cache,
        max_docs_per_source=max_docs_per_source,
        max_chars_per_source=max_chars_per_source,
    )

    total_chars = 0
    total_docs = 0
    duplicates = 0
    dropped = 0

    with docs_path.open("w", encoding="utf-8") as docs_f, corpus_path.open("w", encoding="utf-8") as corpus_f:
        for cfg in configs:
            if total_chars >= target_chars:
                break
            try:
                adapter = build_adapter(cfg)
                for doc in adapter.iter_documents():
                    if total_chars >= target_chars:
                        break
                    text = normalize_training_text(doc.text)
                    keep, flags = should_keep(text, min_chars=min_chars)
                    for flag in flags:
                        noise_flags[flag] += 1
                    if not keep:
                        dropped += 1
                        continue

                    h = text_hash(text)
                    if h in seen_hashes:
                        duplicates += 1
                        continue
                    seen_hashes.add(h)

                    category = doc.source_type or "unknown"
                    quota = quotas.get(category)
                    if quota is not None and category_chars[category] >= quota:
                        continue
                    cap = source_cap(doc.source_name, target_chars) if use_source_caps else None
                    if cap is not None and source_chars[doc.source_name] >= cap:
                        continue

                    row = doc.to_json()
                    row["text"] = text
                    row["char_count"] = len(text)
                    row["chinese_ratio"] = chinese_ratio(text)
                    row["noise_flags"] = flags
                    row["build_version"] = output_version

                    docs_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    corpus_f.write(text + "\n\n<|doc_sep|>\n\n")

                    total_docs += 1
                    total_chars += len(text)
                    category_docs[category] += 1
                    category_chars[category] += len(text)
                    source_docs[doc.source_name] += 1
                    source_chars[doc.source_name] += len(text)
                    if len(preview_rows) < 80:
                        preview_rows.append(row)
            except Exception as exc:  # noqa: BLE001
                errors[cfg.source_name] = f"{type(exc).__name__}: {exc}"

    report = {
        "output_version": output_version,
        "probe_only": probe_only,
        "manifest": str(manifest_path),
        "include_remote": include_remote,
        "include_prepared_cache": include_prepared_cache,
        "target_tokens": target_tokens,
        "estimated_chars_per_token": chars_per_token,
        "effective_target_chars": target_chars,
        "actual_docs": total_docs,
        "actual_chars": total_chars,
        "estimated_tokens": int(total_chars / chars_per_token) if chars_per_token else None,
        "duplicates_removed": duplicates,
        "dropped_docs": dropped,
        "category_docs": dict(category_docs),
        "category_chars": dict(category_chars),
        "source_docs": dict(source_docs),
        "source_chars": dict(source_chars),
        "noise_flags": dict(noise_flags),
        "category_quotas_chars": quotas,
        "source_caps": SOURCE_CHAR_CAPS if use_source_caps else {},
        "errors": errors,
        "outputs": {
            "docs_jsonl": str(docs_path),
            "corpus_txt": str(corpus_path),
            "preview_md": str(preview_path),
            "report_json": str(report_path),
        },
    }

    write_json(report_path, report)
    write_preview(preview_path, report, preview_rows)
    return report


def write_preview(path: Path, report: dict[str, Any], preview_rows: list[dict[str, Any]]) -> None:
    lines = [
        f"# {report['output_version']} Corpus Preview",
        "",
        f"- probe_only: `{report['probe_only']}`",
        f"- include_remote: `{report['include_remote']}`",
        f"- actual_docs: `{report['actual_docs']}`",
        f"- actual_chars: `{report['actual_chars']}`",
        f"- estimated_tokens: `{report['estimated_tokens']}`",
        "",
        "## Category Mix",
        "",
        "| category | docs | chars |",
        "|---|---:|---:|",
    ]
    for category, chars in sorted(report["category_chars"].items()):
        lines.append(f"| `{category}` | {report['category_docs'].get(category, 0)} | {chars} |")

    lines.extend(["", "## Sample Documents", ""])
    for i, row in enumerate(preview_rows[:30], start=1):
        text = row["text"].replace("\n", " / ")
        if len(text) > 260:
            text = text[:260] + "..."
        lines.extend(
            [
                f"### {i}. {row['source_name']} / {row['source_type']}",
                "",
                text,
                "",
            ]
        )

    if report["errors"]:
        lines.extend(["", "## Errors", ""])
        for name, error in report["errors"].items():
            lines.append(f"- `{name}`: {error}")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a bounded ChatLM-scale GPT corpus from unified data adapters.")
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    parser.add_argument("--preset", choices=sorted(BUILD_PRESETS), default=None)
    parser.add_argument("--output-version", default="chatlm_scale_probe")
    parser.add_argument("--target-tokens", type=int, default=100_000_000)
    parser.add_argument("--chars-per-token", type=float, default=1.6)
    parser.add_argument("--include-remote", action="store_true")
    parser.add_argument("--no-prepared-cache", action="store_true")
    parser.add_argument("--max-docs-per-source", type=int, default=None)
    parser.add_argument("--max-chars-per-source", type=int, default=None)
    parser.add_argument("--min-chars", type=int, default=10)
    parser.add_argument("--no-source-caps", action="store_true")
    parser.add_argument("--full", action="store_true", help="Actually aim for target_tokens. Without this, run a bounded probe.")
    args = parser.parse_args()

    output_version = args.output_version
    target_tokens = args.target_tokens
    if args.preset:
        preset_output_version, preset_target_tokens = BUILD_PRESETS[args.preset]
        target_tokens = preset_target_tokens
        if output_version == "chatlm_scale_probe":
            output_version = preset_output_version if args.full else f"{preset_output_version}_probe"

    report = build_corpus(
        manifest_path=Path(args.manifest),
        output_version=output_version,
        target_tokens=target_tokens,
        chars_per_token=args.chars_per_token,
        include_remote=args.include_remote,
        include_prepared_cache=not args.no_prepared_cache,
        max_docs_per_source=args.max_docs_per_source,
        max_chars_per_source=args.max_chars_per_source,
        min_chars=args.min_chars,
        probe_only=not args.full,
        use_source_caps=not args.no_source_caps,
    )
    print("output_version:", report["output_version"])
    print("actual_docs:", report["actual_docs"])
    print("actual_chars:", report["actual_chars"])
    print("estimated_tokens:", report["estimated_tokens"])
    print("preview:", report["outputs"]["preview_md"])
    print("report:", report["outputs"]["report_json"])


if __name__ == "__main__":
    main()
