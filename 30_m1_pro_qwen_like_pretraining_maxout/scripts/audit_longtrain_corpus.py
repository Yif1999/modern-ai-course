from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CURRENT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = CURRENT_DIR / "data" / "raw"
REPORT_DIR = CURRENT_DIR / "outputs" / "reports"


def chinese_ratio(text: str) -> float:
    compact = [ch for ch in text if not ch.isspace()]
    if not compact:
        return 0.0
    return sum(1 for ch in compact if "\u4e00" <= ch <= "\u9fff") / len(compact)


def noise_flags(text: str) -> list[str]:
    flags = []
    if "\ufffd" in text or "�" in text:
        flags.append("replacement_char")
    if re.search(r"https?://|www\.", text):
        flags.append("url")
    if re.search(r"据.+?报道|来源[:：]|新华社|中新网|央视新闻|记者从", text):
        flags.append("news_wire_style")
    if re.search(r"[，。！？；：、,.!?;:]\s+[\u3400-\u9fffA-Za-z0-9~～…]", text):
        flags.append("punctuation_space")
    if len(re.findall(r"\.(?:jpg|jpeg|png|gif|webp)\b", text, flags=re.IGNORECASE)) >= 5:
        flags.append("many_media_filenames")
    if re.search(r"(.)\1{10,}", text):
        flags.append("long_repeated_char")
    if len(text) >= 240:
        spans = [re.sub(r"\s+", "", text[i : i + 80]) for i in range(0, max(0, len(text) - 80), 40)]
        spans = [span for span in spans if len(span) >= 50]
        if len(spans) != len(set(spans)):
            flags.append("repeated_span")
    return flags


def reservoir_add(bucket: list[dict[str, Any]], row: dict[str, Any], *, seen: int, limit: int, rng: random.Random) -> None:
    if len(bucket) < limit:
        bucket.append(row)
        return
    idx = rng.randint(0, seen - 1)
    if idx < limit:
        bucket[idx] = row


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a generated long-train corpus JSONL.")
    parser.add_argument("--docs-jsonl", required=True)
    parser.add_argument("--output-version", required=True)
    parser.add_argument("--sample-per-category", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    docs_path = Path(args.docs_jsonl)
    if not docs_path.exists():
        raise FileNotFoundError(docs_path)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    category_docs = Counter()
    category_chars = Counter()
    source_docs = Counter()
    source_chars = Counter()
    flags = Counter()
    sample_seen = Counter()
    samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    total_docs = 0
    total_chars = 0
    min_chars = None
    max_chars = 0
    chinese_weighted = 0.0

    with docs_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            text = str(row.get("text", ""))
            category = str(row.get("source_type") or row.get("category") or "unknown")
            source = str(row.get("source_name") or "unknown")
            char_count = len(text)
            ratio = chinese_ratio(text)
            total_docs += 1
            total_chars += char_count
            min_chars = char_count if min_chars is None else min(min_chars, char_count)
            max_chars = max(max_chars, char_count)
            chinese_weighted += ratio * char_count
            category_docs[category] += 1
            category_chars[category] += char_count
            source_docs[source] += 1
            source_chars[source] += char_count
            for flag in noise_flags(text):
                flags[flag] += 1

            sample_seen[category] += 1
            sample_row = {
                "source_name": source,
                "source_type": category,
                "char_count": char_count,
                "chinese_ratio": round(ratio, 4),
                "flags": noise_flags(text),
                "text": text,
            }
            reservoir_add(
                samples[category],
                sample_row,
                seen=sample_seen[category],
                limit=args.sample_per_category,
                rng=rng,
            )

    report = {
        "output_version": args.output_version,
        "docs_jsonl": str(docs_path),
        "total_docs": total_docs,
        "total_chars": total_chars,
        "average_chars": total_chars / total_docs if total_docs else 0,
        "min_chars": min_chars,
        "max_chars": max_chars,
        "estimated_chinese_ratio": chinese_weighted / total_chars if total_chars else 0,
        "category_docs": dict(category_docs),
        "category_chars": dict(category_chars),
        "category_char_share": {
            key: value / total_chars if total_chars else 0 for key, value in category_chars.items()
        },
        "source_docs_top": dict(source_docs.most_common(50)),
        "source_chars_top": dict(source_chars.most_common(50)),
        "noise_flags": dict(flags),
    }

    json_path = REPORT_DIR / f"{args.output_version}_audit_report.json"
    sample_path = REPORT_DIR / f"{args.output_version}_sample_audit.json"
    md_path = REPORT_DIR / f"{args.output_version}_audit_report.md"
    write_json(json_path, report)
    write_json(sample_path, samples)

    lines = [
        f"# {args.output_version} Audit Report",
        "",
        f"- docs: `{total_docs:,}`",
        f"- chars: `{total_chars:,}`",
        f"- average chars/doc: `{report['average_chars']:.2f}`",
        f"- estimated Chinese ratio: `{report['estimated_chinese_ratio']:.4f}`",
        f"- docs jsonl: `{docs_path}`",
        f"- JSON report: `{json_path}`",
        f"- sample audit: `{sample_path}`",
        "",
        "## Category Character Mix",
        "",
        "| category | docs | chars | share |",
        "|---|---:|---:|---:|",
    ]
    for category, chars in category_chars.most_common():
        share = chars / total_chars if total_chars else 0
        lines.append(f"| `{category}` | {category_docs[category]:,} | {chars:,} | {share:.2%} |")

    lines.extend(["", "## Noise Flags", "", "| flag | count |", "|---|---:|"])
    for flag, count in flags.most_common():
        lines.append(f"| `{flag}` | {count:,} |")

    lines.extend(["", "## Sample Preview", ""])
    for category, rows in samples.items():
        lines.extend([f"### {category}", ""])
        for row in rows[:5]:
            text = row["text"].replace("\n", " / ")
            if len(text) > 260:
                text = text[:260] + "..."
            lines.append(f"- `{row['source_name']}` chars={row['char_count']} flags={row['flags']}: {text}")
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    print("report:", md_path)
    print("json:", json_path)
    print("samples:", sample_path)
    print(f"docs={total_docs:,} chars={total_chars:,}")


if __name__ == "__main__":
    main()
