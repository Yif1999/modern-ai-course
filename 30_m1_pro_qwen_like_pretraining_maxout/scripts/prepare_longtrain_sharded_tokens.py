from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.format import open_memmap
from tokenizers import Tokenizer


CURRENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CURRENT_DIR / "scripts"))

from prepare_chatlm_scale_corpus import (  # noqa: E402
    adapter_configs,
    category_quotas,
    load_json,
    normalize_training_text,
    should_keep,
    source_cap,
)
from run_data_adapter_dry_run import MANIFEST_PATH, PREPARED_SOURCE_DIR, chinese_ratio  # noqa: E402


PROCESSED_DIR = CURRENT_DIR / "data" / "processed"
METADATA_DIR = CURRENT_DIR / "data" / "metadata"
TOKENIZER_DIR = CURRENT_DIR / "data" / "tokenizers"
REPORT_DIR = CURRENT_DIR / "outputs" / "reports"
STATUS_DIR = CURRENT_DIR / "outputs" / "status"
LEDGER_DIR = CURRENT_DIR / "outputs" / "ledgers"
LOCK_DIR = CURRENT_DIR / "outputs" / "locks"
DOC_SEPARATOR = "\n\n<|doc_sep|>\n\n"
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]
LEGACY_DEFAULT_OVERFLOW_CATEGORIES = ["qa_short_answer", "general_web_backbone", "mixed_existing"]


class ShardWriter:
    def __init__(self, root: Path, prefix: str, shard_tokens: int, *, resume: bool = False):
        self.root = root
        self.prefix = prefix
        self.shard_tokens = shard_tokens
        self.root.mkdir(parents=True, exist_ok=True)
        self.shards: list[dict[str, Any]] = []
        self.shard_index = 0
        self.offset = 0
        self.total_tokens = 0
        self.current_path: Path | None = None
        self.current: np.memmap | None = None
        if resume:
            self._load_existing_shards()
        self._open_next()

    def _load_existing_shards(self) -> None:
        paths = sorted(self.root.glob(f"{self.prefix}_shard_*.npy"))
        max_index = -1
        for path in paths:
            # Ignore a zero-sized/current temp file from a previously interrupted run.
            if path.stat().st_size <= 128:
                path.unlink(missing_ok=True)
                continue
            match = re.search(rf"{re.escape(self.prefix)}_shard_(\d+)", path.stem)
            if match:
                max_index = max(max_index, int(match.group(1)))
            arr = np.load(path, mmap_mode="r")
            if arr.dtype != np.int32:
                raise TypeError(f"Existing shard {path} has dtype {arr.dtype}, expected int32")
            tokens = int(arr.shape[0])
            self.shards.append({"path": str(path), "tokens": tokens, "bytes": int(path.stat().st_size)})
            self.total_tokens += tokens
        self.shard_index = max_index + 1

    def _open_next(self) -> None:
        self.current_path = self.root / f"{self.prefix}_shard_{self.shard_index:05d}.npy"
        if self.current_path.exists():
            self.current_path.unlink()
        self.current = open_memmap(self.current_path, mode="w+", dtype=np.int32, shape=(self.shard_tokens,))
        self.offset = 0

    def _finalize_current(self, *, final: bool = False) -> None:
        if self.current is None or self.current_path is None:
            return
        self.current.flush()
        tokens = self.offset
        path = self.current_path
        self.current = None
        self.current_path = None
        if tokens == 0:
            path.unlink(missing_ok=True)
            return
        if final and tokens < self.shard_tokens:
            trimmed_path = path.with_name(path.stem + "_trimmed.npy")
            arr = np.load(path, mmap_mode="r")[:tokens]
            np.save(trimmed_path, np.asarray(arr, dtype=np.int32))
            path.unlink(missing_ok=True)
            path = trimmed_path
        self.shards.append({"path": str(path), "tokens": int(tokens), "bytes": int(path.stat().st_size)})
        self.shard_index += 1

    def write(self, ids: list[int] | np.ndarray, *, max_total_tokens: int | None = None) -> int:
        if max_total_tokens is not None:
            remaining_total = max_total_tokens - self.total_tokens
            if remaining_total <= 0:
                return 0
            if len(ids) > remaining_total:
                ids = ids[:remaining_total]
        arr = np.asarray(ids, dtype=np.int32)
        written = 0
        while written < arr.shape[0]:
            if self.current is None:
                self._open_next()
            assert self.current is not None
            space = self.shard_tokens - self.offset
            take = min(space, arr.shape[0] - written)
            self.current[self.offset : self.offset + take] = arr[written : written + take]
            self.offset += take
            self.total_tokens += take
            written += take
            if self.offset >= self.shard_tokens:
                self._finalize_current(final=False)
        return written

    def close(self) -> None:
        self._finalize_current(final=True)


def trim_shards_to_expected_tokens(root: Path, prefix: str, expected_tokens: int) -> int:
    """Trim or remove interrupted shards so resume starts from the last confirmed token.

    A live memmap shard has its full target file size even when only part of it has
    been written. If the process is killed, blindly loading that shard would treat
    the unwritten tail as real token ids. The status file records the last confirmed
    token count, so resume first reconciles the shard files to that count.
    """
    root.mkdir(parents=True, exist_ok=True)
    expected_tokens = max(0, int(expected_tokens))
    paths = sorted(root.glob(f"{prefix}_shard_*.npy"))
    remaining = expected_tokens
    kept = 0

    for path in paths:
        if remaining <= 0:
            path.unlink(missing_ok=True)
            continue

        arr = np.load(path, mmap_mode="r")
        tokens = int(arr.shape[0])
        if tokens <= remaining:
            remaining -= tokens
            kept += tokens
            continue

        tmp_path = path.parent / f".{path.name}.trim.npy"
        np.save(tmp_path, np.asarray(arr[:remaining], dtype=np.int32))
        tmp_path.replace(path)
        kept += remaining
        remaining = 0

    return kept


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def load_json_or_empty(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def load_hashes(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def append_hash(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(value + "\n")


def safe_name(value: str) -> str:
    value = "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in value).strip("_")
    if not value:
        raise ValueError("empty output name")
    return value


def hash_to_unit(text: str) -> float:
    digest = hashlib.sha1(text.encode("utf-8", errors="ignore")).digest()
    value = int.from_bytes(digest[:8], "big")
    return value / float(2**64 - 1)


def parse_category_weights(raw: str | None) -> dict[str, float] | None:
    if raw is None or not raw.strip():
        return None
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("--category-weights-json must be a JSON object")
    weights: dict[str, float] = {}
    for category, value in payload.items():
        if not isinstance(category, str) or not category:
            raise ValueError("--category-weights-json category names must be non-empty strings")
        try:
            weight = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid weight for category {category!r}: {value!r}") from exc
        if weight < 0:
            raise ValueError(f"category weight must be non-negative for {category!r}")
        if weight > 0:
            weights[category] = weight
    if not weights:
        raise ValueError("--category-weights-json must contain at least one positive weight")
    return weights


def category_quotas_from_weights(total_tokens: int, weights: dict[str, float]) -> dict[str, int]:
    total_weight = sum(weights.values())
    raw_quotas = {category: total_tokens * weight / total_weight for category, weight in weights.items()}
    quotas = {category: int(value) for category, value in raw_quotas.items()}
    remainder = total_tokens - sum(quotas.values())
    if remainder > 0:
        categories_by_fraction = sorted(
            raw_quotas,
            key=lambda category: (raw_quotas[category] - quotas[category], weights[category]),
            reverse=True,
        )
        for index in range(remainder):
            quotas[categories_by_fraction[index % len(categories_by_fraction)]] += 1
    return quotas


def document_dedupe_hash(text: str, mode: str, metadata: dict[str, Any]) -> str:
    if mode == "clean_hash":
        clean_hash = str(metadata.get("clean_text_hash") or "").strip()
        if clean_hash:
            return clean_hash
        return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()
    if mode == "exact":
        return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()
    if mode == "compact":
        return hashlib.sha1("".join(text.split()).encode("utf-8", errors="ignore")).hexdigest()
    raise ValueError(f"unsupported dedupe mode: {mode}")


def short_preview(text: str, limit: int = 240) -> str:
    text = text.replace("\n", " / ")
    return text if len(text) <= limit else text[:limit] + "..."


def buffered_shuffle(iterable, *, buffer_size: int, rng: random.Random):
    if buffer_size <= 1:
        yield from iterable
        return
    buffer = []
    for item in iterable:
        if len(buffer) < buffer_size:
            buffer.append(item)
            continue
        idx = rng.randrange(len(buffer))
        yield buffer[idx]
        buffer[idx] = item
    rng.shuffle(buffer)
    yield from buffer


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare sharded lab-BPE token data for long training.")
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    parser.add_argument("--output-version", default="longtrain_10b_lab_bpe_16384")
    parser.add_argument("--vocab-size", type=int, default=16_384)
    parser.add_argument("--target-train-tokens", type=int, default=10_000_000_000)
    parser.add_argument("--val-tokens", type=int, default=10_000_000)
    parser.add_argument("--shard-tokens", type=int, default=25_000_000)
    parser.add_argument("--include-remote", action="store_true")
    parser.add_argument("--no-prepared-cache", action="store_true")
    parser.add_argument("--max-docs-per-source", type=int, default=50_000_000)
    parser.add_argument("--max-chars-per-source", type=int, default=None)
    parser.add_argument("--max-doc-chars", type=int, default=None)
    parser.add_argument("--min-chars", type=int, default=10)
    parser.add_argument(
        "--dedupe-mode",
        choices=["clean_hash", "exact", "compact"],
        default="compact",
        help="Document de-duplication hash mode. Use clean_hash with clean prepared caches.",
    )
    parser.add_argument(
        "--trust-prepared-cache-cleaning",
        action="store_true",
        help="Skip the legacy per-document quality filter when LAB_PREPARED_SOURCE_DIR points at a cleaned cache.",
    )
    parser.add_argument("--no-source-caps", action="store_true")
    parser.add_argument("--category-quota-mode", choices=["hard", "soft", "flexible", "off"], default="flexible")
    parser.add_argument(
        "--category-weights-json",
        default=None,
        help=(
            "Optional JSON object mapping source categories to mixture weights. "
            "Example: '{\"general_web_backbone\":0.75,\"qa_short_answer\":0.18}'."
        ),
    )
    parser.add_argument(
        "--overflow-category",
        action="append",
        default=None,
        help=(
            "Category allowed to exceed its target quota when other categories are exhausted. "
            "Can be passed multiple times. Defaults to the legacy QA/general/mixed overflow only "
            "when --category-weights-json is not set."
        ),
    )
    parser.add_argument("--shuffle-sources", action="store_true", default=True)
    parser.add_argument("--no-shuffle-sources", dest="shuffle_sources", action="store_false")
    parser.add_argument("--shuffle-buffer-docs", type=int, default=10_000)
    parser.add_argument("--source-retries", type=int, default=3)
    parser.add_argument("--retry-sleep-sec", type=float, default=20.0)
    parser.add_argument(
        "--skip-source",
        action="append",
        default=[],
        help="Source name to skip. Can be passed multiple times by watchdog after source-level stalls.",
    )
    parser.add_argument("--status-interval-tokens", type=int, default=5_000_000)
    parser.add_argument("--sample-preview-limit", type=int, default=80)
    parser.add_argument("--doc-separator", default=DOC_SEPARATOR)
    parser.add_argument("--seed", type=int, default=2060)
    parser.add_argument("--resume", action="store_true", help="Resume by appending new shards in an existing output directory.")
    parser.add_argument("--force", action="store_true", help="Allow replacing an existing output directory/metadata. Does not delete old shards.")
    args = parser.parse_args()

    output_version = safe_name(args.output_version)
    tokenizer_path = TOKENIZER_DIR / f"lab_byte_bpe_{args.vocab_size}.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError(tokenizer_path)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    rng = random.Random(args.seed)

    root = PROCESSED_DIR / output_version
    metadata_path = METADATA_DIR / f"{output_version}_metadata.json"
    status_path = STATUS_DIR / f"{output_version}_status.json"
    report_path = REPORT_DIR / f"{output_version}_build_report.json"
    preview_path = REPORT_DIR / f"{output_version}_preview.md"
    ledger_dir = LEDGER_DIR / output_version
    state_path = ledger_dir / "state.json"
    source_ledger_path = ledger_dir / "source_ledger.jsonl"
    doc_hash_path = ledger_dir / "kept_doc_hashes.txt"
    mixture_report_path = REPORT_DIR / f"{output_version}_mixture_report.md"
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = LOCK_DIR / f"{output_version}_prepare.lock"
    lock_file = lock_path.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(
            json.dumps(
                {
                    "output_version": output_version,
                    "action": "exit",
                    "reason": f"another prepare process holds {lock_path}",
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return
    lock_file.write(f"pid={os.getpid()} started_at_unix={time.time()}\n")
    lock_file.flush()
    if root.exists() and not args.resume and not args.force:
        raise FileExistsError(
            f"{root} already exists. Use --resume to append or choose a new --output-version. "
            "This guard prevents accidental overwrites of a long-running dataset build."
        )

    resume_status: dict[str, Any] = {}
    if args.resume and status_path.exists():
        try:
            resume_status = json.loads(status_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            resume_status = {}
        expected_train_tokens = min(int(resume_status.get("train_tokens", 0) or 0), args.target_train_tokens)
        expected_val_tokens = min(int(resume_status.get("val_tokens", 0) or 0), args.val_tokens)
        if expected_train_tokens or expected_val_tokens:
            kept_train = trim_shards_to_expected_tokens(root / "train", "train", expected_train_tokens)
            kept_val = trim_shards_to_expected_tokens(root / "val", "val", expected_val_tokens)
            print(
                "resume shard reconciliation:",
                f"train={kept_train:,}/{expected_train_tokens:,}",
                f"val={kept_val:,}/{expected_val_tokens:,}",
                flush=True,
            )

    train_writer = ShardWriter(root / "train", "train", args.shard_tokens, resume=args.resume)
    val_writer = ShardWriter(
        root / "val",
        "val",
        min(args.shard_tokens, max(args.val_tokens, 2048)),
        resume=args.resume,
    )

    manifest = load_json(Path(args.manifest))
    configs = adapter_configs(
        manifest,
        include_remote=args.include_remote,
        include_prepared_cache=not args.no_prepared_cache,
        max_docs_per_source=args.max_docs_per_source,
        max_chars_per_source=args.max_chars_per_source,
    )
    skip_sources = set(args.skip_source or [])
    if skip_sources:
        before = len(configs)
        configs = [cfg for cfg in configs if cfg.source_name not in skip_sources]
        print(f"skipped sources: {sorted(skip_sources)} ({before - len(configs)} matched)", flush=True)
    if args.shuffle_sources:
        rng.shuffle(configs)

    total_target_tokens = args.target_train_tokens + args.val_tokens
    category_weight_config = parse_category_weights(args.category_weights_json)
    if category_weight_config is None:
        quotas = category_quotas(total_target_tokens)
    else:
        quotas = category_quotas_from_weights(total_target_tokens, category_weight_config)
        for category in {cfg.source_type or "unknown" for cfg in configs}:
            quotas.setdefault(category, 0)
    if args.overflow_category is None:
        overflow_categories = set(LEGACY_DEFAULT_OVERFLOW_CATEGORIES if category_weight_config is None else [])
    else:
        overflow_categories = set(args.overflow_category)
    if category_weight_config is not None:
        overflow_categories = {category for category in overflow_categories if category_weight_config.get(category, 0) > 0}
    doc_separator = args.doc_separator
    persisted_state = load_json_or_empty(state_path) if args.resume else {}
    category_tokens = defaultdict(int, persisted_state.get("category_tokens", {}))
    category_docs = defaultdict(int, persisted_state.get("category_docs", {}))
    category_chars = defaultdict(int, persisted_state.get("category_chars", {}))
    source_tokens = defaultdict(int, persisted_state.get("source_tokens", {}))
    source_docs = defaultdict(int, persisted_state.get("source_docs", {}))
    source_chars = defaultdict(int, persisted_state.get("source_chars", {}))
    noise_flags = Counter(persisted_state.get("noise_flags", {}))
    dropped_docs = int(persisted_state.get("dropped_docs", 0) or 0)
    duplicate_docs = int(persisted_state.get("duplicate_docs", 0) or 0)
    seen_hashes: set[str] = load_hashes(doc_hash_path) if args.resume else set()
    completed_sources = set(persisted_state.get("completed_sources", [])) if args.resume else set()
    source_attempts = defaultdict(int, persisted_state.get("source_attempts", {}))
    previews: list[dict[str, Any]] = list(persisted_state.get("previews", []))[: args.sample_preview_limit]
    errors: dict[str, str] = dict(persisted_state.get("errors", {}))
    docs_seen = int(persisted_state.get("docs_seen", 0) or 0)
    docs_kept = int(persisted_state.get("docs_kept", 0) or 0)
    raw_chars = int(persisted_state.get("raw_chars", 0) or 0)
    started = time.perf_counter()
    started_wall = time.time()
    last_token_write_wall = time.time()
    last_status_tokens = 0
    current_source = ""
    current_attempt = 0
    resumed_train_tokens = train_writer.total_tokens
    resumed_val_tokens = val_writer.total_tokens
    val_probability = min(0.20, max(0.0001, args.val_tokens / max(total_target_tokens, 1) * 2.0))

    def current_status(state: str) -> dict[str, Any]:
        elapsed = time.perf_counter() - started
        total_written = train_writer.total_tokens + val_writer.total_tokens
        total_for_share = max(total_written, 1)
        category_share = {
            category: {
                "tokens": int(tokens),
                "share": float(tokens / total_for_share),
                "target_tokens": int(quotas.get(category, 0)),
                "target_share": float(quotas.get(category, 0) / max(total_target_tokens, 1)),
            }
            for category, tokens in sorted(category_tokens.items())
        }
        return {
            "state": state,
            "output_version": output_version,
            "docs_seen": docs_seen,
            "docs_kept": docs_kept,
            "docs_dropped": dropped_docs,
            "docs_duplicate": duplicate_docs,
            "doc_keep_rate": docs_kept / docs_seen if docs_seen else None,
            "train_tokens": train_writer.total_tokens,
            "val_tokens": val_writer.total_tokens,
            "total_tokens": total_written,
            "tokens_per_seen_doc": total_written / docs_seen if docs_seen else None,
            "tokens_per_kept_doc": total_written / docs_kept if docs_kept else None,
            "current_source": current_source,
            "current_attempt": current_attempt,
            "target_train_tokens": args.target_train_tokens,
            "target_val_tokens": args.val_tokens,
            "target_total_tokens": total_target_tokens,
            "resumed_train_tokens": resumed_train_tokens,
            "resumed_val_tokens": resumed_val_tokens,
            "progress_percent": 100.0 * total_written / max(total_target_tokens, 1),
            "elapsed_sec": elapsed,
            "started_at_unix": started_wall,
            "updated_at_unix": time.time(),
            "last_token_write_at_unix": last_token_write_wall,
            "seconds_since_token_write": time.time() - last_token_write_wall,
            "tokens_per_sec": total_written / elapsed if elapsed > 0 else None,
            "category_tokens": dict(category_tokens),
            "source_tokens_top": dict(Counter(source_tokens).most_common(30)),
            "source_docs_top": dict(Counter(source_docs).most_common(30)),
            "category_share": category_share,
            "category_target_tokens": dict(quotas),
            "category_quota_mode": args.category_quota_mode,
            "overflow_categories": sorted(overflow_categories),
            "category_weight_config": dict(category_weight_config or {}),
            "prepared_source_dir": str(PREPARED_SOURCE_DIR),
            "doc_separator": doc_separator,
            "trust_prepared_cache_cleaning": args.trust_prepared_cache_cleaning,
            "dedupe_mode": args.dedupe_mode,
            "max_doc_chars": args.max_doc_chars,
            "completed_sources": sorted(completed_sources),
            "completed_source_count": len(completed_sources),
            "seen_hash_count": len(seen_hashes),
            "skip_sources": sorted(skip_sources),
            "metadata_path": str(metadata_path),
            "report_path": str(report_path),
            "mixture_report_path": str(mixture_report_path),
            "ledger_dir": str(ledger_dir),
            "source_ledger_path": str(source_ledger_path),
        }

    def persist_status(state: str) -> None:
        write_json(status_path, current_status(state))

    def persist_ledger_state(state: str) -> None:
        write_json(
            state_path,
            {
                **current_status(state),
                "category_docs": dict(category_docs),
                "category_chars": dict(category_chars),
                "source_tokens": dict(source_tokens),
                "source_docs": dict(source_docs),
                "source_chars": dict(source_chars),
                "source_attempts": dict(source_attempts),
                "noise_flags": dict(noise_flags),
                "dropped_docs": dropped_docs,
                "duplicate_docs": duplicate_docs,
                "raw_chars": raw_chars,
                "docs_seen": docs_seen,
                "docs_kept": docs_kept,
                "errors": errors,
                "previews": previews[: args.sample_preview_limit],
                "completed_sources": sorted(completed_sources),
            },
        )

    def persist_progress(state: str) -> None:
        persist_status(state)
        persist_ledger_state(state)

    try:
        persist_progress("running")
        for cfg in configs:
            if train_writer.total_tokens >= args.target_train_tokens and val_writer.total_tokens >= args.val_tokens:
                break
            if cfg.source_name in completed_sources:
                append_jsonl(
                    source_ledger_path,
                    {
                        "event": "source_skipped_completed",
                        "source_name": cfg.source_name,
                        "source_type": cfg.source_type,
                        "at_unix": time.time(),
                    },
                )
                continue
            current_source = cfg.source_name
            source_completed = False
            source_start_tokens = train_writer.total_tokens + val_writer.total_tokens
            source_start_docs_seen = docs_seen
            source_start_docs_kept = docs_kept
            source_start_raw_chars = raw_chars
            append_jsonl(
                source_ledger_path,
                {
                    "event": "source_started",
                    "source_name": cfg.source_name,
                    "source_type": cfg.source_type,
                    "source_group": cfg.source_group,
                    "start_total_tokens": source_start_tokens,
                    "start_train_tokens": train_writer.total_tokens,
                    "start_val_tokens": val_writer.total_tokens,
                    "at_unix": time.time(),
                },
            )
            for attempt in range(1, max(1, args.source_retries) + 1):
                current_attempt = attempt
                source_attempts[cfg.source_name] += 1
                try:
                    adapter = __import__("data_adapters.factory", fromlist=["build_adapter"]).build_adapter(cfg)
                    persist_progress("running")
                    docs_iter = buffered_shuffle(
                        adapter.iter_documents(),
                        buffer_size=args.shuffle_buffer_docs,
                        rng=rng,
                    )
                    for doc in docs_iter:
                        if train_writer.total_tokens >= args.target_train_tokens and val_writer.total_tokens >= args.val_tokens:
                            break
                        docs_seen += 1
                        text = normalize_training_text(doc.text)
                        if args.trust_prepared_cache_cleaning:
                            keep = len(text) >= args.min_chars
                            flags = [] if keep else ["too_short"]
                        else:
                            keep, flags = should_keep(text, min_chars=args.min_chars)
                        for flag in flags:
                            noise_flags[flag] += 1
                        if not keep:
                            dropped_docs += 1
                            continue
                        if args.max_doc_chars is not None and len(text) > args.max_doc_chars:
                            noise_flags["too_long"] += 1
                            dropped_docs += 1
                            continue
                        compact_hash = document_dedupe_hash(text, args.dedupe_mode, doc.metadata)
                        if compact_hash in seen_hashes:
                            duplicate_docs += 1
                            continue

                        category = doc.source_type or "unknown"
                        quota = quotas.get(category)
                        category_at_or_over_quota = quota is not None and category_tokens[category] >= quota
                        if args.category_quota_mode == "hard" and category_at_or_over_quota:
                            break
                        if (
                            args.category_quota_mode == "flexible"
                            and category_at_or_over_quota
                            and category not in overflow_categories
                        ):
                            break

                        cap_chars = source_cap(doc.source_name, int(total_target_tokens * 1.5)) if not args.no_source_caps else None
                        if cap_chars is not None and source_chars[doc.source_name] >= cap_chars:
                            break

                        ids = tokenizer.encode(text + doc_separator, add_special_tokens=False).ids
                        if not ids:
                            continue
                        if args.category_quota_mode in {"hard", "flexible"} and quota is not None:
                            remaining_quota = quota - category_tokens[category]
                            if remaining_quota <= 0 and category not in overflow_categories:
                                continue
                            if (
                                remaining_quota > 0
                                and len(ids) > remaining_quota
                                and category not in overflow_categories
                            ):
                                ids = ids[:remaining_quota]

                        to_val = (
                            val_writer.total_tokens < args.val_tokens
                            and hash_to_unit(compact_hash) < val_probability
                        )
                        if to_val:
                            written = val_writer.write(ids, max_total_tokens=args.val_tokens)
                        else:
                            written = train_writer.write(ids, max_total_tokens=args.target_train_tokens)
                        if written <= 0:
                            continue

                        last_token_write_wall = time.time()
                        seen_hashes.add(compact_hash)
                        append_hash(doc_hash_path, compact_hash)
                        docs_kept += 1
                        raw_chars += len(text)
                        category_tokens[category] += written
                        category_docs[category] += 1
                        category_chars[category] += len(text)
                        source_tokens[doc.source_name] += written
                        source_docs[doc.source_name] += 1
                        source_chars[doc.source_name] += len(text)
                        if len(previews) < args.sample_preview_limit:
                            previews.append(
                                {
                                    "source_name": doc.source_name,
                                    "source_type": category,
                                    "char_count": len(text),
                                    "token_count": len(ids),
                                    "chinese_ratio": chinese_ratio(text),
                                    "text": short_preview(text),
                                }
                            )

                        total_written = train_writer.total_tokens + val_writer.total_tokens
                        if total_written - last_status_tokens >= args.status_interval_tokens:
                            last_status_tokens = total_written
                            persist_progress("running")
                            print(
                                f"tokens={total_written:,}/{total_target_tokens:,} "
                                f"train={train_writer.total_tokens:,} val={val_writer.total_tokens:,} "
                                f"docs={docs_kept:,}",
                                flush=True,
                            )
                    source_completed = True
                    break
                except Exception as exc:  # noqa: BLE001
                    errors[f"{cfg.source_name}:stream:{attempt}"] = f"{type(exc).__name__}: {exc}"
                    append_jsonl(
                        source_ledger_path,
                        {
                            "event": "source_attempt_error",
                            "source_name": cfg.source_name,
                            "source_type": cfg.source_type,
                            "attempt": attempt,
                            "error": f"{type(exc).__name__}: {exc}",
                            "at_unix": time.time(),
                        },
                    )
                    persist_progress("running_with_source_error")
                    if attempt < args.source_retries:
                        time.sleep(args.retry_sleep_sec)
                    else:
                        source_completed = False
            source_end_tokens = train_writer.total_tokens + val_writer.total_tokens
            source_tokens_added = source_end_tokens - source_start_tokens
            if source_completed:
                completed_sources.add(cfg.source_name)
            append_jsonl(
                source_ledger_path,
                {
                    "event": "source_completed" if source_completed else "source_failed",
                    "source_name": cfg.source_name,
                    "source_type": cfg.source_type,
                    "source_group": cfg.source_group,
                    "tokens_added": source_tokens_added,
                    "docs_seen_added": docs_seen - source_start_docs_seen,
                    "docs_kept_added": docs_kept - source_start_docs_kept,
                    "raw_chars_added": raw_chars - source_start_raw_chars,
                    "end_total_tokens": source_end_tokens,
                    "end_train_tokens": train_writer.total_tokens,
                    "end_val_tokens": val_writer.total_tokens,
                    "completed_sources": len(completed_sources),
                    "at_unix": time.time(),
                },
            )
            persist_progress("running")
            print(
                f"source={cfg.source_name} completed={source_completed} "
                f"tokens_added={source_tokens_added:,} "
                f"docs_seen_added={docs_seen - source_start_docs_seen:,} "
                f"docs_kept_added={docs_kept - source_start_docs_kept:,}",
                flush=True,
            )

        train_writer.close()
        val_writer.close()
        state = (
            "complete"
            if train_writer.total_tokens >= args.target_train_tokens and val_writer.total_tokens >= args.val_tokens
            else "exhausted_sources"
        )
        metadata = {
            "dataset_format": "sharded_npy",
            "output_version": output_version,
            "tokenizer_type": "lab_bpe",
            "tokenizer_name": f"lab_byte_bpe_{vocab_size}",
            "tokenizer_path": str(tokenizer_path),
            "vocab_size": vocab_size,
            "special_tokens": SPECIAL_TOKENS,
            "doc_separator": doc_separator,
            "target_train_tokens": args.target_train_tokens,
            "target_val_tokens": args.val_tokens,
            "prepared_source_dir": str(PREPARED_SOURCE_DIR),
            "trust_prepared_cache_cleaning": args.trust_prepared_cache_cleaning,
            "dedupe_mode": args.dedupe_mode,
            "max_doc_chars": args.max_doc_chars,
            "category_weight_config": dict(category_weight_config or {}),
            "category_quota_mode": args.category_quota_mode,
            "overflow_categories": sorted(overflow_categories),
            "shuffle_sources": args.shuffle_sources,
            "shuffle_buffer_docs": args.shuffle_buffer_docs,
            "seed": args.seed,
            "skip_sources": sorted(skip_sources),
            "train_tokens": train_writer.total_tokens,
            "val_tokens": val_writer.total_tokens,
            "total_tokens": train_writer.total_tokens + val_writer.total_tokens,
            "train_shards": train_writer.shards,
            "val_shards": val_writer.shards,
            "shard_tokens": args.shard_tokens,
            "raw_chars": raw_chars,
            "docs_seen": docs_seen,
            "docs_kept": docs_kept,
            "dropped_docs": dropped_docs,
            "duplicate_docs": duplicate_docs,
            "category_tokens": dict(category_tokens),
            "category_docs": dict(category_docs),
            "category_chars": dict(category_chars),
            "category_target_tokens": dict(quotas),
            "source_tokens_top": dict(Counter(source_tokens).most_common(100)),
            "source_tokens": dict(source_tokens),
            "source_docs_top": dict(Counter(source_docs).most_common(100)),
            "source_docs": dict(source_docs),
            "source_chars": dict(source_chars),
            "completed_sources": sorted(completed_sources),
            "source_ledger_path": str(source_ledger_path),
            "state_path": str(state_path),
            "doc_hash_path": str(doc_hash_path),
            "noise_flags": dict(noise_flags),
            "status": state,
            "errors": errors,
            "status_path": str(status_path),
            "report_path": str(report_path),
            "mixture_report_path": str(mixture_report_path),
            "preview_path": str(preview_path),
            "elapsed_sec": time.perf_counter() - started,
        }
        write_json(metadata_path, metadata)
        write_json(report_path, metadata)

        lines = [
            f"# {output_version} Sharded Dataset Report",
            "",
            f"- status: `{state}`",
            f"- train tokens: `{train_writer.total_tokens:,}` / `{args.target_train_tokens:,}`",
            f"- val tokens: `{val_writer.total_tokens:,}` / `{args.val_tokens:,}`",
            f"- train shards: `{len(train_writer.shards)}`",
            f"- val shards: `{len(val_writer.shards)}`",
            f"- docs kept: `{docs_kept:,}`",
            f"- metadata: `{metadata_path}`",
            "",
            "## Category Tokens",
            "",
            "| category | tokens | share | docs |",
            "|---|---:|---:|---:|",
        ]
        total_tokens = max(train_writer.total_tokens + val_writer.total_tokens, 1)
        for category, tokens in Counter(category_tokens).most_common():
            lines.append(f"| `{category}` | {tokens:,} | {tokens / total_tokens:.2%} | {category_docs[category]:,} |")
        lines.extend(
            [
                "",
                "## Source Tokens",
                "",
                "| source | category | tokens | share | docs |",
                "|---|---|---:|---:|---:|",
            ]
        )
        source_to_category = {cfg.source_name: cfg.source_type for cfg in configs}
        for source, tokens in Counter(source_tokens).most_common(100):
            lines.append(
                f"| `{source}` | `{source_to_category.get(source, '')}` | "
                f"{tokens:,} | {tokens / total_tokens:.2%} | {source_docs[source]:,} |"
            )
        lines.extend(["", "## Preview", ""])
        for row in previews:
            lines.append(f"- `{row['source_name']}` / `{row['source_type']}` tokens={row['token_count']}: {row['text']}")
        if errors:
            lines.extend(["", "## Source Errors", ""])
            for name, err in errors.items():
                lines.append(f"- `{name}`: {err}")
        preview_path.write_text("\n".join(lines), encoding="utf-8")
        mixture_lines = [
            f"# {output_version} Mixture Report",
            "",
            f"- status: `{state}`",
            f"- total tokens: `{total_tokens:,}`",
            f"- mode: `{args.category_quota_mode}`",
            f"- overflow categories: `{', '.join(sorted(set(args.overflow_category or [])))}`",
            f"- source ledger: `{source_ledger_path}`",
            f"- state: `{state_path}`",
            "",
            "## Category Actual vs Target",
            "",
            "| category | actual tokens | actual share | target tokens | target share | delta pp |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for category in sorted(set(category_tokens) | set(quotas)):
            actual = category_tokens.get(category, 0)
            target = quotas.get(category, 0)
            actual_share = actual / total_tokens
            target_share = target / max(total_target_tokens, 1)
            mixture_lines.append(
                f"| `{category}` | {actual:,} | {actual_share:.2%} | "
                f"{target:,} | {target_share:.2%} | {(actual_share - target_share) * 100:+.2f} |"
            )
        mixture_lines.extend(["", "## Source Actual", "", "| source | category | tokens | share | docs |", "|---|---|---:|---:|---:|"])
        for source, tokens in Counter(source_tokens).most_common():
            mixture_lines.append(
                f"| `{source}` | `{source_to_category.get(source, '')}` | "
                f"{tokens:,} | {tokens / total_tokens:.2%} | {source_docs[source]:,} |"
            )
        mixture_report_path.write_text("\n".join(mixture_lines), encoding="utf-8")
        persist_progress(state)
        print("metadata:", metadata_path)
        print("report:", report_path)
        print("mixture_report:", mixture_report_path)
        print("status:", status_path)
        print(f"state={state} train_tokens={train_writer.total_tokens:,} val_tokens={val_writer.total_tokens:,}")
    except KeyboardInterrupt:
        train_writer.close()
        val_writer.close()
        persist_progress("interrupted")
        raise


if __name__ == "__main__":
    main()
