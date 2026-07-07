from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


CURRENT_DIR = Path(__file__).resolve().parents[1]
MANIFEST_PATH = CURRENT_DIR / "data" / "metadata" / "chatlm_scale_data_sources.json"
REPORT_DIR = CURRENT_DIR / "outputs" / "reports"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def fmt_int(value: int | float | None) -> str:
    if value is None:
        return ""
    return f"{int(value):,}"


def source_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for source in manifest["sources"]:
        reported = source.get("reported_by_chatlm", {})
        sampling = source.get("sampling_plan", {})
        rows.append(
            {
                "source_name": source["source_name"],
                "group": source.get("group", ""),
                "category": source.get("category", ""),
                "platform": source.get("platform", ""),
                "default_enabled": source.get("default_enabled", False),
                "reported_clean_rows": reported.get("clean_rows", ""),
                "reported_raw_rows": reported.get("raw_rows", ""),
                "max_rows": sampling.get("max_rows", sampling.get("max_docs", "")),
                "max_chars": sampling.get("max_chars", ""),
                "url": source.get("url", source.get("huggingface_url", source.get("modelscope_url", ""))),
                "risks": " | ".join(source.get("risks", [])),
                "intended_use": source.get("intended_use", ""),
            }
        )
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_name",
        "group",
        "category",
        "platform",
        "default_enabled",
        "reported_clean_rows",
        "reported_raw_rows",
        "max_rows",
        "max_chars",
        "url",
        "risks",
        "intended_use",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(manifest: dict[str, Any], rows: list[dict[str, Any]], path: Path) -> None:
    target = manifest["target_for_our_gpt"]
    reference = manifest["reference_recipe"]
    lines = [
        "# ChatLM Scale Source Plan",
        "",
        "## Reference",
        "",
        f"- Reference model: `{reference['name']}`",
        f"- Reference architecture: `{reference['architecture']}`",
        f"- Reported total rows: `{fmt_int(reference['reported_total_dataset_rows'])}`",
        f"- Reported pretraining rows: `{fmt_int(reference['reported_pretraining_rows'])}`",
        "",
        "## Our GPT Target",
        "",
        f"- Model: `{target['model']}`",
        f"- Context: `{target['context']}`",
        f"- Tokenizer: `{target['tokenizer']}`",
        f"- First unique-token target: `{fmt_int(target['first_target_unique_tokens'])}`",
        f"- Stretch unique-token target: `{fmt_int(target['stretch_target_unique_tokens'])}`",
        "",
        "## Target Mixture",
        "",
        "| category | target | role |",
        "|---|---:|---|",
    ]
    for item in manifest["target_mixture"]:
        target_share = item.get("target_share")
        target_text = f"{target_share * 100:.1f}%" if isinstance(target_share, (int, float)) else f"<= {item.get('target_share_max', 0) * 100:.1f}%"
        lines.append(f"| `{item['category']}` | {target_text} | {item['role']} |")

    lines.extend(
        [
            "",
            "## Sources",
            "",
            "| source | group | category | enabled | reported clean rows | planned cap | risk summary |",
            "|---|---|---|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        cap = row["max_rows"] or row["max_chars"]
        lines.append(
            "| "
            f"`{row['source_name']}` | "
            f"{row['group']} | "
            f"{row['category']} | "
            f"{row['default_enabled']} | "
            f"{row['reported_clean_rows']} | "
            f"{cap} | "
            f"{row['risks'][:180]} |"
        )

    lines.extend(
        [
            "",
            "## Guardrail",
            "",
            "This script is planning-only by default. It does not download datasets unless `--execute-download` is explicitly passed.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ChatLM-scale data source plan for the 0.2B GPT run.")
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    parser.add_argument("--execute-download", action="store_true", help="Reserved for future implementation. Not enabled yet.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = load_json(manifest_path)
    if args.execute_download:
        raise SystemExit(
            "Download execution is intentionally not implemented in this planning script yet. "
            "Review the source plan first, then implement bounded source adapters."
        )

    rows = source_rows(manifest)
    csv_path = REPORT_DIR / "chatlm_scale_source_plan.csv"
    md_path = REPORT_DIR / "chatlm_scale_source_plan.md"
    summary_path = REPORT_DIR / "chatlm_scale_source_plan.json"
    write_csv(rows, csv_path)
    write_markdown(manifest, rows, md_path)
    write_json(summary_path, {"manifest": str(manifest_path), "sources": rows})

    print("Wrote:", csv_path)
    print("Wrote:", md_path)
    print("Wrote:", summary_path)
    print("Planning only: no datasets were downloaded.")


if __name__ == "__main__":
    main()
