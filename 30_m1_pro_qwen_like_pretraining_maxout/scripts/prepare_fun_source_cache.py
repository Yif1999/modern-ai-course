from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


CURRENT_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = CURRENT_DIR.parents[0]
sys.path.insert(0, str(CURRENT_DIR / "scripts"))

from prepare_lab_bpe_token_data import (  # noqa: E402
    CACHE_DIR,
    PREPARED_SOURCE_CACHE_DIR,
    collect_bilibili_comments,
    collect_chatharuhi,
    collect_hana,
    collect_lccc,
    collect_moegirl,
    collect_with_cache,
    collect_worldchat,
)


OUTPUT_DIR = CURRENT_DIR / "outputs"
REPORT_DIR = OUTPUT_DIR / "reports"
STATUS_DIR = OUTPUT_DIR / "status"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def report_row(name: str, report: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": name,
        "cache_path": report.get("cache_path"),
        "cache_hit": report.get("cache_hit"),
        "kept_docs": report.get("kept_docs") or report.get("cached_docs") or report.get("loaded_docs") or 0,
        "kept_chars": report.get("kept_chars") or report.get("cached_chars") or report.get("loaded_chars") or 0,
        "raw_rows_seen": report.get("raw_rows_seen") or report.get("raw_dialogues_seen"),
        "elapsed_sec": report.get("elapsed_sec") or report.get("cache_write_elapsed_sec"),
        "error": report.get("error"),
        "skipped": report.get("skipped", False),
        "reason": report.get("reason"),
    }


def run_source(
    *,
    name: str,
    collector: Any,
    collector_kwargs: dict[str, Any],
    cache_payload: dict[str, Any] | None,
    refresh: bool,
) -> tuple[dict[str, Any], dict[str, int]]:
    docs: list[dict[str, Any]] = []
    seen: set[str] = set()
    stats: dict[str, int] = {}
    payload = cache_payload if cache_payload is not None else collector_kwargs
    started = time.time()
    try:
        report = collect_with_cache(
            name,
            collector,
            docs,
            seen,
            stats,
            refresh_cache=refresh,
            cache_payload=payload,
            collector_kwargs=collector_kwargs,
        )
        report["total_elapsed_sec"] = time.time() - started
        return report, stats
    except Exception as exc:  # noqa: BLE001
        return {
            "source_label": name,
            "cache_hit": False,
            "error": f"{type(exc).__name__}: {exc}",
            "total_elapsed_sec": time.time() - started,
            "collector_kwargs": collector_kwargs,
        }, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize reusable prepared JSONL caches for local/community fun sources. "
            "This does not train a tokenizer and does not write token shards."
        )
    )
    parser.add_argument("--output-version", default="fun_source_cache_expansion")
    parser.add_argument("--refresh-source-cache", action="store_true")
    parser.add_argument("--only", nargs="*", default=None, help="Optional source labels to run.")
    parser.add_argument("--skip", nargs="*", default=[], help="Optional source labels to skip.")
    parser.add_argument("--lccc-target-chars", type=int, default=2_000_000_000)
    parser.add_argument("--lccc-max-rows", type=int, default=50_000_000)
    parser.add_argument("--lccc-file", choices=["base_train", "large"], default="large")
    parser.add_argument("--hana-target-chars", type=int, default=200_000_000)
    parser.add_argument("--hana-max-dialogues", type=int, default=5_000_000)
    parser.add_argument("--hana-repo-dir", default=str(CACHE_DIR / "hana" / "HANA"))
    parser.add_argument("--bilibili-target-chars", type=int, default=500_000_000)
    parser.add_argument("--bilibili-max-rows", type=int, default=50_000_000)
    parser.add_argument("--moegirl-target-chars", type=int, default=250_000_000)
    parser.add_argument("--moegirl-max-rows", type=int, default=10_000_000)
    parser.add_argument("--worldchat-target-chars", type=int, default=50_000_000)
    parser.add_argument("--worldchat-max-rows", type=int, default=10_000_000)
    parser.add_argument("--chatharuhi-target-chars", type=int, default=200_000_000)
    parser.add_argument("--chatharuhi-max-rows", type=int, default=5_000_000)
    args = parser.parse_args()

    os.environ.setdefault("HF_DATASETS_CACHE", str(CACHE_DIR / "datasets"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

    for path in [PREPARED_SOURCE_CACHE_DIR, REPORT_DIR, STATUS_DIR]:
        path.mkdir(parents=True, exist_ok=True)

    source_specs: list[tuple[str, Any, dict[str, Any], dict[str, Any] | None]] = [
        (
            "lccc_large_expanded",
            collect_lccc,
            {
                "target_chars": args.lccc_target_chars,
                "target_total_chars": args.lccc_target_chars,
                "file_key": args.lccc_file,
                "max_rows": args.lccc_max_rows,
            },
            {
                "target_chars": args.lccc_target_chars,
                "target_total_chars": args.lccc_target_chars,
                "file_key": args.lccc_file,
                "max_rows": args.lccc_max_rows,
            },
        ),
        (
            "hana_dialogue",
            collect_hana,
            {
                "target_chars": args.hana_target_chars,
                "max_dialogues": args.hana_max_dialogues,
                "repo_dir": args.hana_repo_dir,
            },
            None,
        ),
        (
            "bilibili_comment",
            collect_bilibili_comments,
            {"target_chars": args.bilibili_target_chars, "max_rows": args.bilibili_max_rows},
            None,
        ),
        (
            "acg_wiki",
            collect_moegirl,
            {"target_chars": args.moegirl_target_chars, "max_rows": args.moegirl_max_rows},
            None,
        ),
        (
            "game_world_chat",
            collect_worldchat,
            {"target_chars": args.worldchat_target_chars, "max_rows": args.worldchat_max_rows},
            None,
        ),
        (
            "anime_roleplay",
            collect_chatharuhi,
            {"target_chars": args.chatharuhi_target_chars, "max_rows": args.chatharuhi_max_rows},
            None,
        ),
    ]

    only = set(args.only or [])
    skip = set(args.skip or [])
    rows: list[dict[str, Any]] = []
    all_stats: dict[str, dict[str, int]] = {}
    started = time.time()

    for name, collector, collector_kwargs, cache_payload in source_specs:
        if only and name not in only:
            rows.append({"source": name, "skipped": True, "reason": "not in --only"})
            continue
        if name in skip:
            rows.append({"source": name, "skipped": True, "reason": "listed in --skip"})
            continue

        print(f"source={name} start kwargs={collector_kwargs}", flush=True)
        report, stats = run_source(
            name=name,
            collector=collector,
            collector_kwargs=collector_kwargs,
            cache_payload=cache_payload,
            refresh=args.refresh_source_cache,
        )
        row = report_row(name, report)
        rows.append(row)
        all_stats[name] = stats
        payload = {
            "output_version": args.output_version,
            "elapsed_sec": time.time() - started,
            "latest_source": name,
            "latest_source_report": row,
            "sources": rows,
        }
        write_json(STATUS_DIR / f"{args.output_version}_status.json", payload)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    total_docs = sum(int(row.get("kept_docs") or 0) for row in rows)
    total_chars = sum(int(row.get("kept_chars") or 0) for row in rows)
    final_report = {
        "output_version": args.output_version,
        "elapsed_sec": time.time() - started,
        "prepared_cache_dir": str(PREPARED_SOURCE_CACHE_DIR),
        "total_docs": total_docs,
        "total_chars": total_chars,
        "sources": rows,
        "stats": all_stats,
    }
    write_json(REPORT_DIR / f"{args.output_version}_report.json", final_report)
    write_jsonl(REPORT_DIR / f"{args.output_version}_sources.jsonl", rows)
    write_json(STATUS_DIR / f"{args.output_version}_status.json", final_report)

    print("prepared_cache_dir:", PREPARED_SOURCE_CACHE_DIR)
    print("total_docs:", total_docs)
    print("total_chars:", total_chars)
    print("report:", REPORT_DIR / f"{args.output_version}_report.json")


if __name__ == "__main__":
    main()
