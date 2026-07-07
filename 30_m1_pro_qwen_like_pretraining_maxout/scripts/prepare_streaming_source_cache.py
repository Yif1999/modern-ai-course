from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

CURRENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CURRENT_DIR / "scripts"))

from data_adapters.factory import build_adapter  # noqa: E402
from prepare_chatlm_scale_corpus import (  # noqa: E402
    adapter_configs,
    load_json,
    normalize_training_text,
    should_keep,
)
from run_data_adapter_dry_run import MANIFEST_PATH, PREPARED_SOURCE_DIR, chinese_ratio  # noqa: E402


STATUS_DIR = CURRENT_DIR / "outputs" / "status"
REPORT_DIR = CURRENT_DIR / "outputs" / "reports"
LOG_DIR = CURRENT_DIR / "outputs" / "logs"
LOCK_DIR = CURRENT_DIR / "outputs" / "locks"
CACHE_STATE_DIR = CURRENT_DIR / "data" / "metadata" / "source_cache_states"


def safe_slug(value: str, limit: int = 80) -> str:
    slug = "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in value).strip("_")
    slug = "_".join(part for part in slug.split("_") if part)
    return (slug or "source")[:limit]


def stable_signature(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8", errors="ignore")
    return hashlib.sha1(encoded).hexdigest()[:12]


def text_hash(text: str) -> str:
    compact = "".join(text.split())
    return hashlib.sha1(compact.encode("utf-8", errors="ignore")).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def load_hashes(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def append_hash(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(value + "\n")


def source_cache_paths(output_version: str, cfg) -> dict[str, Path]:
    signature = stable_signature(
        {
            "source_name": cfg.source_name,
            "source_type": cfg.source_type,
            "source_group": cfg.source_group,
            "dataset_name": cfg.dataset_name,
            "split": cfg.split,
            "field_map": cfg.field_map,
            "options": cfg.options,
        }
    )
    name = f"{safe_slug(cfg.source_name)}_{signature}"
    return {
        "jsonl": PREPARED_SOURCE_DIR / f"{name}.jsonl",
        "state": CACHE_STATE_DIR / output_version / f"{name}.json",
        "hashes": CACHE_STATE_DIR / output_version / f"{name}.hashes.txt",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize remote streaming sources into reusable cleaned JSONL caches.")
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    parser.add_argument("--output-version", default="longtrain_10b_source_cache")
    parser.add_argument("--max-docs-per-source", type=int, default=None)
    parser.add_argument("--max-chars-per-source", type=int, default=None)
    parser.add_argument("--min-chars", type=int, default=10)
    parser.add_argument("--source", action="append", default=[], help="Only cache matching source_name. Can repeat.")
    parser.add_argument("--skip-source", action="append", default=[], help="Skip matching source_name. Can repeat.")
    parser.add_argument("--status-interval-docs", type=int, default=10_000)
    parser.add_argument("--status-interval-chars", type=int, default=5_000_000)
    parser.add_argument(
        "--stream-shuffle-buffer",
        type=int,
        default=50_000,
        help="Approximate randomization buffer for HF streaming datasets. 0 disables streaming shuffle.",
    )
    parser.add_argument("--seed", type=int, default=2060)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-rebuild-source", action="append", default=[])
    args = parser.parse_args()

    output_version = safe_slug(args.output_version)
    os.environ.setdefault("HF_DATASETS_CACHE", str(CURRENT_DIR / "data" / "cache" / "datasets"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PREPARED_SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_STATE_DIR.mkdir(parents=True, exist_ok=True)
    LOCK_DIR.mkdir(parents=True, exist_ok=True)

    lock_path = LOCK_DIR / f"{output_version}.source_cache.lock"
    lock_file = lock_path.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(json.dumps({"action": "exit", "reason": f"lock held: {lock_path}"}, ensure_ascii=False), flush=True)
        return
    lock_file.write(f"pid={os.getpid()} started_at_unix={time.time()}\n")
    lock_file.flush()

    status_path = STATUS_DIR / f"{output_version}_status.json"
    ledger_path = REPORT_DIR / f"{output_version}_source_cache_ledger.jsonl"
    report_path = REPORT_DIR / f"{output_version}_source_cache_report.json"
    preview_path = REPORT_DIR / f"{output_version}_source_cache_preview.md"

    manifest = load_json(Path(args.manifest))
    configs = adapter_configs(
        manifest,
        include_remote=True,
        include_prepared_cache=False,
        max_docs_per_source=args.max_docs_per_source,
        max_chars_per_source=args.max_chars_per_source,
    )
    # This stage only materializes remote/HF streams. Local JSONL sources are already cache files.
    configs = [cfg for cfg in configs if cfg.dataset_name]
    if args.source:
        selected = set(args.source)
        configs = [cfg for cfg in configs if cfg.source_name in selected]
    if args.skip_source:
        skipped = set(args.skip_source)
        configs = [cfg for cfg in configs if cfg.source_name not in skipped]
    for cfg in configs:
        cfg.options = dict(cfg.options)
        if args.stream_shuffle_buffer > 0:
            cfg.options["shuffle_buffer"] = args.stream_shuffle_buffer
            cfg.options["seed"] = args.seed

    started = time.time()
    total_docs_seen = 0
    total_docs_kept = 0
    total_chars = 0
    dropped_docs = 0
    duplicate_docs = 0
    current_source = ""
    category_docs = defaultdict(int)
    category_chars = defaultdict(int)
    source_docs = defaultdict(int)
    source_chars = defaultdict(int)
    noise_flags = Counter()
    previews: list[dict[str, Any]] = []
    errors: dict[str, str] = {}

    def status(state: str) -> dict[str, Any]:
        elapsed = max(time.time() - started, 1e-6)
        return {
            "state": state,
            "output_version": output_version,
            "current_source": current_source,
            "sources_total": len(configs),
            "total_docs_seen": total_docs_seen,
            "total_docs_kept": total_docs_kept,
            "total_chars": total_chars,
            "dropped_docs": dropped_docs,
            "duplicate_docs": duplicate_docs,
            "docs_per_sec": total_docs_seen / elapsed,
            "chars_per_sec": total_chars / elapsed,
            "category_docs": dict(category_docs),
            "category_chars": dict(category_chars),
            "source_docs": dict(source_docs),
            "source_chars": dict(source_chars),
            "noise_flags": dict(noise_flags),
            "status_path": str(status_path),
            "ledger_path": str(ledger_path),
            "report_path": str(report_path),
            "prepared_source_dir": str(PREPARED_SOURCE_DIR),
            "updated_at_unix": time.time(),
            "stream_shuffle_buffer": args.stream_shuffle_buffer,
            "seed": args.seed,
        }

    def persist_status(state: str) -> None:
        write_json(status_path, status(state))

    persist_status("running")

    for cfg in configs:
        current_source = cfg.source_name
        paths = source_cache_paths(output_version, cfg)
        source_state = read_json(paths["state"])
        if (
            args.resume
            and source_state.get("state") == "complete"
            and paths["jsonl"].exists()
            and cfg.source_name not in set(args.force_rebuild_source)
        ):
            source_docs[cfg.source_name] += int(source_state.get("docs_kept", 0) or 0)
            source_chars[cfg.source_name] += int(source_state.get("chars_kept", 0) or 0)
            category_docs[cfg.source_type] += int(source_state.get("docs_kept", 0) or 0)
            category_chars[cfg.source_type] += int(source_state.get("chars_kept", 0) or 0)
            total_docs_kept += int(source_state.get("docs_kept", 0) or 0)
            total_chars += int(source_state.get("chars_kept", 0) or 0)
            append_jsonl(ledger_path, {"event": "source_skip_complete", "source_name": cfg.source_name, "cache_path": str(paths["jsonl"]), "at_unix": time.time()})
            persist_status("running")
            continue

        if cfg.source_name in set(args.force_rebuild_source):
            paths["jsonl"].unlink(missing_ok=True)
            paths["hashes"].unlink(missing_ok=True)
            paths["state"].unlink(missing_ok=True)
            source_state = {}
        elif args.resume and paths["jsonl"].exists() and not source_state and not paths["hashes"].exists():
            raise RuntimeError(
                "Found an existing source cache without matching resume state. "
                f"Use --force-rebuild-source {cfg.source_name} or a new output version: {paths['jsonl']}"
            )
        elif not args.resume and (paths["jsonl"].exists() or paths["hashes"].exists() or paths["state"].exists()):
            raise RuntimeError(
                "Refusing to append to an existing source cache without --resume or "
                f"--force-rebuild-source {cfg.source_name}: {paths['jsonl']}"
            )

        seen_hashes = load_hashes(paths["hashes"]) if args.resume else set()
        docs_seen = int(source_state.get("docs_seen", 0) or 0) if args.resume else 0
        docs_kept = int(source_state.get("docs_kept", 0) or 0) if args.resume else 0
        chars_kept = int(source_state.get("chars_kept", 0) or 0) if args.resume else 0
        source_dropped = int(source_state.get("dropped_docs", 0) or 0) if args.resume else 0
        source_duplicate = int(source_state.get("duplicate_docs", 0) or 0) if args.resume else 0
        last_status_docs = docs_seen
        last_status_chars = chars_kept

        append_jsonl(
            ledger_path,
            {
                "event": "source_started",
                "source_name": cfg.source_name,
                "source_type": cfg.source_type,
                "dataset_name": cfg.dataset_name,
                "cache_path": str(paths["jsonl"]),
                "resume": args.resume,
                "docs_seen_existing": docs_seen,
                "docs_kept_existing": docs_kept,
                "chars_kept_existing": chars_kept,
                "at_unix": time.time(),
            },
        )
        persist_status("running")

        try:
            if args.resume and docs_seen > 0:
                cfg.options = dict(cfg.options)
                cfg.options["skip_rows"] = docs_seen
                if cfg.max_docs is not None:
                    cfg.max_docs = max(int(cfg.max_docs) - docs_seen, 0)
                if cfg.max_chars is not None:
                    cfg.max_chars = max(int(cfg.max_chars) - chars_kept, 0)
                append_jsonl(
                    ledger_path,
                    {
                        "event": "source_resume_skip",
                        "source_name": cfg.source_name,
                        "skip_rows": docs_seen,
                        "remaining_max_docs": cfg.max_docs,
                        "remaining_max_chars": cfg.max_chars,
                        "at_unix": time.time(),
                    },
                )
            adapter = build_adapter(cfg)
            with paths["jsonl"].open("a", encoding="utf-8") as out:
                for doc in adapter.iter_documents():
                    total_docs_seen += 1
                    docs_seen += 1
                    text = normalize_training_text(doc.text)
                    keep, flags = should_keep(text, min_chars=args.min_chars)
                    for flag in flags:
                        noise_flags[flag] += 1
                    if not keep:
                        dropped_docs += 1
                        source_dropped += 1
                        continue

                    h = text_hash(text)
                    if h in seen_hashes:
                        duplicate_docs += 1
                        source_duplicate += 1
                        continue

                    row = doc.to_json()
                    row["text"] = text
                    row["source_name"] = doc.source_name
                    row["source_type"] = doc.source_type
                    row["source_group"] = doc.source_group
                    row["source_id"] = doc.source_id
                    row["char_count"] = len(text)
                    row["chinese_ratio"] = chinese_ratio(text)
                    row["noise_flags"] = flags
                    row["text_hash"] = h
                    row["cache_version"] = output_version
                    out.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    append_hash(paths["hashes"], h)
                    seen_hashes.add(h)

                    docs_kept += 1
                    chars_kept += len(text)
                    total_docs_kept += 1
                    total_chars += len(text)
                    category_docs[doc.source_type] += 1
                    category_chars[doc.source_type] += len(text)
                    source_docs[doc.source_name] += 1
                    source_chars[doc.source_name] += len(text)
                    if len(previews) < 80:
                        previews.append(
                            {
                                "source_name": doc.source_name,
                                "source_type": doc.source_type,
                                "char_count": len(text),
                                "chinese_ratio": chinese_ratio(text),
                                "text": text.replace("\n", " / ")[:260],
                            }
                        )

                    if (
                        docs_seen - last_status_docs >= args.status_interval_docs
                        or chars_kept - last_status_chars >= args.status_interval_chars
                    ):
                        last_status_docs = docs_seen
                        last_status_chars = chars_kept
                        write_json(
                            paths["state"],
                            {
                                "state": "running",
                                "source_name": cfg.source_name,
                                "source_type": cfg.source_type,
                                "dataset_name": cfg.dataset_name,
                                "cache_path": str(paths["jsonl"]),
                                "docs_seen": docs_seen,
                                "docs_kept": docs_kept,
                                "chars_kept": chars_kept,
                                "dropped_docs": source_dropped,
                                "duplicate_docs": source_duplicate,
                                "hash_count": len(seen_hashes),
                                "updated_at_unix": time.time(),
                            },
                        )
                        persist_status("running")
                        print(
                            f"source={cfg.source_name} docs={docs_kept:,} chars={chars_kept:,} "
                            f"total_docs={total_docs_kept:,} total_chars={total_chars:,}",
                            flush=True,
                        )

            write_json(
                paths["state"],
                {
                    "state": "complete",
                    "source_name": cfg.source_name,
                    "source_type": cfg.source_type,
                    "dataset_name": cfg.dataset_name,
                    "cache_path": str(paths["jsonl"]),
                    "docs_seen": docs_seen,
                    "docs_kept": docs_kept,
                    "chars_kept": chars_kept,
                    "dropped_docs": source_dropped,
                    "duplicate_docs": source_duplicate,
                    "hash_count": len(seen_hashes),
                    "file_size_bytes": paths["jsonl"].stat().st_size if paths["jsonl"].exists() else 0,
                    "completed_at_unix": time.time(),
                },
            )
            append_jsonl(
                ledger_path,
                {
                    "event": "source_completed",
                    "source_name": cfg.source_name,
                    "source_type": cfg.source_type,
                    "dataset_name": cfg.dataset_name,
                    "cache_path": str(paths["jsonl"]),
                    "docs_seen": docs_seen,
                    "docs_kept": docs_kept,
                    "chars_kept": chars_kept,
                    "dropped_docs": source_dropped,
                    "duplicate_docs": source_duplicate,
                    "at_unix": time.time(),
                },
            )
            persist_status("running")
        except Exception as exc:  # noqa: BLE001
            errors[cfg.source_name] = f"{type(exc).__name__}: {exc}"
            write_json(
                paths["state"],
                {
                    "state": "error",
                    "source_name": cfg.source_name,
                    "source_type": cfg.source_type,
                    "dataset_name": cfg.dataset_name,
                    "cache_path": str(paths["jsonl"]),
                    "docs_seen": docs_seen,
                    "docs_kept": docs_kept,
                    "chars_kept": chars_kept,
                    "dropped_docs": source_dropped,
                    "duplicate_docs": source_duplicate,
                    "error": errors[cfg.source_name],
                    "updated_at_unix": time.time(),
                },
            )
            append_jsonl(ledger_path, {"event": "source_error", "source_name": cfg.source_name, "error": errors[cfg.source_name], "at_unix": time.time()})
            persist_status("running_with_error")

    report = {
        **status("complete" if not errors else "complete_with_errors"),
        "errors": errors,
        "previews": previews,
    }
    write_json(report_path, report)
    preview_lines = [
        f"# {output_version} Source Cache Preview",
        "",
        f"- docs kept: `{total_docs_kept:,}`",
        f"- chars kept: `{total_chars:,}`",
        f"- prepared source dir: `{PREPARED_SOURCE_DIR}`",
        "",
        "## Category Chars",
        "",
        "| category | docs | chars |",
        "|---|---:|---:|",
    ]
    for category, chars in Counter(category_chars).most_common():
        preview_lines.append(f"| `{category}` | {category_docs[category]:,} | {chars:,} |")
    preview_lines.extend(["", "## Samples", ""])
    for row in previews[:30]:
        preview_lines.append(f"- `{row['source_name']}` / `{row['source_type']}` chars={row['char_count']}: {row['text']}")
    if errors:
        preview_lines.extend(["", "## Errors", ""])
        for name, err in errors.items():
            preview_lines.append(f"- `{name}`: {err}")
    preview_path.write_text("\n".join(preview_lines), encoding="utf-8")
    persist_status("complete" if not errors else "complete_with_errors")
    print("status:", status_path)
    print("report:", report_path)
    print("preview:", preview_path)


if __name__ == "__main__":
    main()
