from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


CURRENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CURRENT_DIR / "scripts"))

from prepare_lab_bpe_token_data import (  # noqa: E402
    PREPARED_SOURCE_CACHE_DIR,
    collect_lccc,
)


REPORT_DIR = CURRENT_DIR / "outputs" / "reports"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def output_path(file_key: str, target_chars: int, max_rows: int) -> Path:
    digest = hashlib.sha1(f"{file_key}:{target_chars}:{max_rows}".encode("utf-8")).hexdigest()[:12]
    return PREPARED_SOURCE_CACHE_DIR / f"lccc_large_expanded_{target_chars}_{digest}.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export a larger cleaned LCCC prepared cache by reusing the existing "
            "LCCC cleaning rules from prepare_lab_bpe_token_data.py."
        )
    )
    parser.add_argument("--target-chars", type=int, default=80_000_000)
    parser.add_argument("--max-rows", type=int, default=2_000_000)
    parser.add_argument("--file-key", choices=["base_train", "large"], default="large")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    PREPARED_SOURCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = output_path(args.file_key, args.target_chars, args.max_rows)
    report_path = REPORT_DIR / f"{out_path.stem}_report.json"

    if out_path.exists() and not args.force:
        report = {
            "status": "exists",
            "output_path": str(out_path),
            "report_path": str(report_path),
            "target_chars": args.target_chars,
            "max_rows": args.max_rows,
            "file_key": args.file_key,
        }
        write_json(report_path, report)
        print("already exists:", out_path)
        print("report:", report_path)
        return

    docs: list[dict[str, Any]] = []
    seen: set[str] = set()
    stats: dict[str, int] = {}
    report = collect_lccc(
        docs,
        seen,
        stats,
        target_chars=args.target_chars,
        target_total_chars=args.target_chars,
        file_key=args.file_key,
        max_rows=args.max_rows,
    )

    with out_path.open("w", encoding="utf-8") as f:
        for doc in docs:
            row = {
                "id": doc.get("id"),
                "text": doc["text"],
                "source_name": "lccc_large_expanded",
                "source_type": "dialogue_short_chat",
                "source_group": "prepared_cache",
                "category": "dialogue_short_chat",
                "char_count": doc.get("char_count", len(doc["text"])),
                "turn_count": doc.get("turn_count"),
                "chinese_ratio": doc.get("chinese_ratio"),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    final_report = {
        **report,
        "status": "written",
        "output_path": str(out_path),
        "output_size_bytes": out_path.stat().st_size,
        "stats": stats,
    }
    write_json(report_path, final_report)

    print("output:", out_path)
    print("report:", report_path)
    print("kept_docs:", report["kept_docs"])
    print("kept_chars:", report["kept_chars"])


if __name__ == "__main__":
    main()
