from __future__ import annotations

import json
import math
import os
import random
import subprocess
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
DEFAULT_RUNS_DIR = PROJECT_DIR / "outputs" / "runs"


def get_runs_dir() -> Path:
    configured = os.environ.get("MLX_GPT_LAB_RUNS_DIR")
    return Path(configured).expanduser().resolve() if configured else DEFAULT_RUNS_DIR.resolve()


def iso_mtime(path: Path) -> str | None:
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def file_size(path: Path) -> int | None:
    return path.stat().st_size if path.exists() else None


def safe_child(base: Path, *parts: str) -> Path:
    base_resolved = base.resolve()
    candidate = base_resolved.joinpath(*parts).resolve()
    try:
        candidate.relative_to(base_resolved)
    except ValueError as exc:
        raise FileNotFoundError("path escapes runs directory") from exc
    return candidate


def run_dir(run_id: str) -> Path:
    if "/" in run_id or "\\" in run_id or run_id in {"", ".", ".."}:
        raise FileNotFoundError(f"invalid run_id: {run_id}")
    path = safe_child(get_runs_dir(), run_id)
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"run not found: {run_id}")
    return path


def read_json(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.exists():
        return None
    try:
        return json_safe(json.loads(path.read_text(encoding="utf-8")))
    except json.JSONDecodeError:
        return {"_error": f"JSON 解析失败: {path.name}", "_raw": path.read_text(encoding="utf-8", errors="replace")}


def json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    return value


def read_text(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8", errors="replace")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            rows.append({"_line": line_no, "_parse_error": True, "text": line})
    return rows


def resolve_data_path(raw_path: str | None, base_run_dir: Path) -> Path | None:
    if not raw_path:
        return None

    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve() if path.exists() else path

    candidates: list[Path] = [
        base_run_dir / path,
        base_run_dir.parent / path,
        PROJECT_DIR / path,
        PROJECT_DIR.parent / path,
    ]

    runs_dir = get_runs_dir()
    for parent in [runs_dir, *runs_dir.parents[:4]]:
        candidates.append(parent / path)

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate.resolve()
    return candidates[-1]


def dataset_metadata_for_run(path: Path) -> tuple[Path | None, dict[str, Any]]:
    config = read_json(path / "config.json")
    if not isinstance(config, dict):
        config = {}

    metadata_path = resolve_data_path(config.get("metadata_path"), path)
    if metadata_path and metadata_path.exists():
        metadata = read_json(metadata_path)
        if isinstance(metadata, dict):
            return metadata_path, metadata

    local_candidates = [
        path / "metadata.json",
        path / "data_metadata.json",
        path / "dataset_metadata.json",
    ]
    for candidate in local_candidates:
        if candidate.exists():
            metadata = read_json(candidate)
            if isinstance(metadata, dict):
                return candidate.resolve(), metadata

    return metadata_path, {}


def _compact_category_summary(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, dict):
        return []
    rows = []
    for category, stats in value.items():
        if isinstance(stats, dict):
            rows.append({"category": category, **stats})
        else:
            rows.append({"category": category, "value": stats})
    return sorted(rows, key=lambda row: row.get("chars") or 0, reverse=True)


def _normalize_dataset_doc(row: dict[str, Any], fallback_id: int) -> dict[str, Any]:
    text = row.get("text") or row.get("content") or row.get("document") or row.get("prompt") or ""
    if not isinstance(text, str):
        text = json.dumps(text, ensure_ascii=False)
    if len(text) > 1600:
        text = text[:1600] + "\n..."
    return {
        "id": row.get("id", fallback_id),
        "text": text,
        "source_name": row.get("source_name") or row.get("dataset") or row.get("source") or "unknown",
        "source_type": row.get("source_type") or "unknown",
        "category": row.get("category") or "unknown",
        "char_count": row.get("char_count") or len(text),
        "turn_count": row.get("turn_count"),
        "chinese_ratio": row.get("chinese_ratio"),
    }


def _load_tokenizer(metadata: dict[str, Any], base_run_dir: Path):
    tokenizer_path = resolve_data_path(metadata.get("tokenizer_path"), base_run_dir)
    if not tokenizer_path or not tokenizer_path.exists():
        return None, str(tokenizer_path) if tokenizer_path else None, "tokenizer_path 不存在"
    try:
        from tokenizers import Tokenizer
    except Exception as exc:
        return None, str(tokenizer_path), f"tokenizers 依赖不可用: {exc}"
    try:
        return Tokenizer.from_file(str(tokenizer_path)), str(tokenizer_path), None
    except Exception as exc:
        return None, str(tokenizer_path), f"tokenizer 加载失败: {exc}"


def _tokenize_doc(tokenizer: Any, text: str, token_limit: int) -> dict[str, Any] | None:
    if tokenizer is None:
        return None
    encoding = tokenizer.encode(text)
    ids = encoding.ids
    tokens = encoding.tokens
    offsets = encoding.offsets
    preview = []
    for index, (token_id, token) in enumerate(zip(ids[:token_limit], tokens[:token_limit])):
        try:
            decoded = tokenizer.decode([token_id], skip_special_tokens=False)
        except Exception:
            decoded = token
        item: dict[str, Any] = {"index": index, "id": token_id, "token": token, "decoded": decoded}
        if index < len(offsets):
            item["offset"] = offsets[index]
        preview.append(item)
    return {
        "token_count": len(ids),
        "shown": len(preview),
        "truncated": len(ids) > len(preview),
        "preview": preview,
    }


def _sample_docs_jsonl(path: Path, limit: int, mode: str, seed: int | None, scan_limit: int) -> tuple[list[dict[str, Any]], int]:
    samples: list[dict[str, Any]] = []
    rng = random.Random(seed)
    scanned = 0

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, start=1):
            if scanned >= scan_limit:
                break
            line = line.strip()
            if not line:
                continue
            scanned += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                row = {"id": line_no, "text": line, "category": "parse_error", "source_name": path.name}
            if not isinstance(row, dict):
                row = {"id": line_no, "text": row, "category": "non_object", "source_name": path.name}
            doc = _normalize_dataset_doc(row, line_no)
            if mode == "head":
                samples.append(doc)
                if len(samples) >= limit:
                    break
                continue
            if len(samples) < limit:
                samples.append(doc)
            else:
                replace_at = rng.randint(0, scanned - 1)
                if replace_at < limit:
                    samples[replace_at] = doc

    return samples, scanned


def _sample_corpus_text(path: Path, limit: int) -> tuple[list[dict[str, Any]], int]:
    text = path.read_text(encoding="utf-8", errors="replace")
    chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
    if not chunks:
        chunks = [line.strip() for line in text.splitlines() if line.strip()]
    samples = [
        {
            "id": index,
            "text": chunk[:1600] + ("\n..." if len(chunk) > 1600 else ""),
            "source_name": path.name,
            "source_type": "corpus_text",
            "category": "corpus",
            "char_count": len(chunk),
            "turn_count": None,
            "chinese_ratio": None,
        }
        for index, chunk in enumerate(chunks[:limit])
    ]
    return samples, len(chunks)


def _sample_preview_markdown(path: Path, limit: int) -> tuple[list[dict[str, Any]], int]:
    samples: list[dict[str, Any]] = []
    in_preview = False
    scanned = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip() == "## Preview":
            in_preview = True
            continue
        if not in_preview:
            continue
        if line.startswith("## "):
            break
        if not line.startswith("- `"):
            continue
        scanned += 1
        try:
            source_part, text = line[2:].split(": ", 1)
        except ValueError:
            source_part, text = "unknown", line[2:]
        parts = [part.strip().strip("`") for part in source_part.split(" / ")]
        source_name = parts[0] if parts else "unknown"
        category = parts[1].split(" tokens=")[0].strip().strip("`") if len(parts) > 1 else "unknown"
        token_count = None
        if "tokens=" in source_part:
            try:
                token_count = int(source_part.rsplit("tokens=", 1)[1].split("`", 1)[0])
            except Exception:
                token_count = None
        samples.append(
            {
                "id": scanned,
                "text": text,
                "source_name": source_name,
                "source_type": "preview_markdown",
                "category": category,
                "char_count": len(text),
                "token_count": token_count,
                "turn_count": None,
                "chinese_ratio": None,
            }
        )
        if len(samples) >= limit:
            break
    return samples, scanned


def _sample_token_shards(
    metadata: dict[str, Any],
    base_run_dir: Path,
    limit: int,
    mode: str,
    seed: int | None,
    window_tokens: int,
) -> tuple[list[dict[str, Any]], int, str | None]:
    train_shards = metadata.get("train_shards")
    if not isinstance(train_shards, list) or not train_shards:
        return [], 0, "metadata 中没有 train_shards。"

    tokenizer, tokenizer_path, tokenizer_error = _load_tokenizer(metadata, base_run_dir)
    if tokenizer is None:
        return [], 0, tokenizer_error or f"tokenizer 不可用: {tokenizer_path}"

    try:
        import numpy as np
    except Exception as exc:
        return [], 0, f"numpy 依赖不可用: {exc}"

    usable_shards: list[dict[str, Any]] = []
    for index, shard in enumerate(train_shards):
        if not isinstance(shard, dict):
            continue
        shard_path = resolve_data_path(shard.get("path"), base_run_dir)
        if not shard_path or not shard_path.exists():
            continue
        tokens = int(shard.get("tokens") or 0)
        if tokens <= 1:
            continue
        usable_shards.append({"index": index, "path": shard_path, "tokens": tokens})

    if not usable_shards:
        return [], 0, "train_shards 路径不可读。"

    rng = random.Random(seed) if seed is not None else random.Random()
    total_tokens = sum(item["tokens"] for item in usable_shards)
    window_tokens = max(16, min(int(window_tokens), 512))
    samples: list[dict[str, Any]] = []

    def choose_shard(sample_index: int) -> dict[str, Any]:
        if mode == "head":
            return usable_shards[min(sample_index, len(usable_shards) - 1)]
        pick = rng.randrange(total_tokens)
        acc = 0
        for item in usable_shards:
            acc += item["tokens"]
            if pick < acc:
                return item
        return usable_shards[-1]

    for sample_index in range(limit):
        shard = choose_shard(sample_index)
        shard_tokens = int(shard["tokens"])
        max_start = max(0, shard_tokens - window_tokens - 1)
        if mode == "head":
            start = min(sample_index * window_tokens, max_start)
        else:
            start = rng.randint(0, max_start) if max_start > 0 else 0
        length = min(window_tokens, shard_tokens - start)
        try:
            arr = np.load(shard["path"], mmap_mode="r")
            ids = [int(x) for x in arr[start : start + length].tolist()]
            text = tokenizer.decode(ids, skip_special_tokens=False)
        except Exception as exc:
            return samples, len(usable_shards), f"读取 token shard 失败: {exc}"

        preview_text = text[:1600] + ("\n..." if len(text) > 1600 else "")
        samples.append(
            {
                "id": f"{shard['index']}:{start}",
                "text": preview_text,
                "source_name": Path(shard["path"]).name,
                "source_type": "token_shard_window",
                "category": "train_token_window",
                "char_count": len(preview_text),
                "token_count": len(ids),
                "turn_count": None,
                "chinese_ratio": None,
                "token_window": {
                    "shard_index": shard["index"],
                    "shard_path": str(shard["path"]),
                    "start": start,
                    "length": len(ids),
                },
            }
        )

    return samples, len(usable_shards), None


def get_dataset_samples(
    run_id: str,
    limit: int = 12,
    mode: str = "random",
    seed: int | None = None,
    scan_limit: int = 200_000,
    include_tokens: bool = True,
    token_limit: int = 80,
):
    path = run_dir(run_id)
    limit = max(1, min(int(limit), 50))
    scan_limit = max(limit, min(int(scan_limit), 1_000_000))
    mode = mode if mode in {"random", "head"} else "random"

    metadata_path, metadata = dataset_metadata_for_run(path)
    docs_path = resolve_data_path(metadata.get("docs_jsonl_path"), path)
    corpus_path = resolve_data_path(metadata.get("corpus_path"), path)
    preview_path = resolve_data_path(metadata.get("preview_path"), path)

    source = None
    samples: list[dict[str, Any]] = []
    scanned = 0
    note = None
    token_limit = max(1, min(int(token_limit), 240))
    token_window_size = max(64, min(int(token_limit) * 4, 512))

    if docs_path and docs_path.exists():
        samples, scanned = _sample_docs_jsonl(docs_path, limit, mode, seed, scan_limit)
        source = "docs_jsonl"
    elif corpus_path and corpus_path.exists():
        samples, scanned = _sample_corpus_text(corpus_path, limit)
        source = "corpus_text"
        note = "没有找到 docs_jsonl_path，已退回到 corpus 文本切片预览。"
    elif isinstance(metadata.get("train_shards"), list) and metadata.get("train_shards"):
        samples, scanned, shard_error = _sample_token_shards(metadata, path, limit, mode, seed, token_window_size)
        if samples:
            source = "token_shards"
            note = "当前 run 使用 sharded token 数据；样本直接从训练 token 分片随机抽取并 decode。"
        elif preview_path and preview_path.exists():
            samples, scanned = _sample_preview_markdown(preview_path, limit)
            source = "preview_markdown"
            note = f"token 分片随机预览不可用（{shard_error}），已显示构建报告中的固定样本。"
        else:
            note = shard_error or "当前 run 的 token 分片不可预览。"
    elif preview_path and preview_path.exists():
        samples, scanned = _sample_preview_markdown(preview_path, limit)
        source = "preview_markdown"
        note = "当前 run 使用 sharded token 数据，没有原始 docs/corpus；已退回到构建报告中的样本预览。"
    else:
        note = "当前 run 没有关联可读取的数据集 metadata 或样本文件。"

    tokenizer = None
    tokenizer_path = None
    tokenizer_error = None
    if include_tokens and samples:
        tokenizer, tokenizer_path, tokenizer_error = _load_tokenizer(metadata, path)
        if tokenizer is not None:
            for sample in samples:
                sample["tokenization"] = _tokenize_doc(tokenizer, sample.get("text") or "", token_limit)

    return {
        "run_id": run_id,
        "mode": mode,
        "limit": limit,
        "scan_limit": scan_limit,
        "scanned": scanned,
        "source": source,
        "note": note,
        "tokenization": {
            "enabled": include_tokens,
            "token_limit": token_limit,
            "tokenizer_path": tokenizer_path,
            "tokenizer_error": tokenizer_error,
        },
        "metadata": {
            "metadata_path": str(metadata_path) if metadata_path else None,
            "metadata_exists": bool(metadata_path and metadata_path.exists()),
            "docs_jsonl_path": str(docs_path) if docs_path else None,
            "docs_jsonl_exists": bool(docs_path and docs_path.exists()),
            "corpus_path": str(corpus_path) if corpus_path else None,
            "corpus_exists": bool(corpus_path and corpus_path.exists()),
            "preview_path": str(preview_path) if preview_path else None,
            "preview_exists": bool(preview_path and preview_path.exists()),
            "tokenizer_name": metadata.get("tokenizer_name"),
            "tokenizer_type": metadata.get("tokenizer_type"),
            "vocab_size": metadata.get("vocab_size"),
            "document_count": metadata.get("document_count") or metadata.get("total_docs"),
            "raw_chars": metadata.get("raw_chars") or metadata.get("corpus_chars"),
            "total_tokens": metadata.get("total_tokens"),
            "train_tokens": metadata.get("train_tokens"),
            "val_tokens": metadata.get("val_tokens"),
            "chars_per_token": metadata.get("chars_per_token"),
            "category_summary": _compact_category_summary(metadata.get("category_summary"))[:12],
            "category_share": metadata.get("category_share") if isinstance(metadata.get("category_share"), dict) else {},
        },
        "samples": samples,
    }


def _visible_token_text(value: str) -> str:
    if value == "":
        return "∅"
    return (
        value.replace(" ", "·")
        .replace("\n", "↵")
        .replace("\t", "⇥")
        .replace("\r", "␍")
    )


def _display_decoded_token(value: str) -> tuple[str, bool, str]:
    if value == "":
        return "∅", True, "empty"
    if "\ufffd" in value:
        return "byte", False, "byte_fragment"
    if all(unicodedata.category(char).startswith("C") for char in value):
        return "ctrl", False, "control"
    return _visible_token_text(value), True, "text"


def get_tokenizer_vocab(run_id: str, page: int = 1, page_size: int = 256, query: str | None = None):
    path = run_dir(run_id)
    _, metadata = dataset_metadata_for_run(path)
    tokenizer, tokenizer_path, tokenizer_error = _load_tokenizer(metadata, path)
    if tokenizer is None:
        return {
            "run_id": run_id,
            "ok": False,
            "error": tokenizer_error or "tokenizer 不可用",
            "tokenizer_path": tokenizer_path,
            "rows": [],
            "page": 1,
            "page_size": page_size,
            "total": 0,
            "total_pages": 0,
        }

    page = max(1, int(page))
    page_size = max(24, min(int(page_size), 1000))
    query = (query or "").strip()

    vocab = tokenizer.get_vocab()
    by_id: list[tuple[int, str]] = sorted((int(token_id), token) for token, token_id in vocab.items())
    rows = []
    for token_id, raw_token in by_id:
        try:
            decoded = tokenizer.decode([token_id], skip_special_tokens=False)
        except Exception:
            decoded = raw_token
        display, readable, kind = _display_decoded_token(decoded)
        if query and query not in str(token_id) and query not in decoded and query not in display:
            continue
        rows.append(
            {
                "id": token_id,
                "decoded": decoded,
                "display": display,
                "readable": readable,
                "kind": kind,
                "is_special": raw_token.startswith("<") and raw_token.endswith(">"),
            }
        )

    total = len(rows)
    total_pages = max(1, (total + page_size - 1) // page_size) if total else 0
    if total_pages:
        page = min(page, total_pages)
    start = (page - 1) * page_size
    end = start + page_size

    return {
        "run_id": run_id,
        "ok": True,
        "tokenizer_name": metadata.get("tokenizer_name"),
        "tokenizer_type": metadata.get("tokenizer_type"),
        "tokenizer_path": tokenizer_path,
        "vocab_size": metadata.get("vocab_size") or len(vocab),
        "page": page,
        "page_size": page_size,
        "total": total,
        "total_pages": total_pages,
        "query": query,
        "rows": rows[start:end],
    }


def _downsample_evenly(rows: list[dict[str, Any]], max_points: int) -> list[dict[str, Any]]:
    if max_points <= 0:
        return []
    if len(rows) <= max_points:
        return rows
    if max_points == 1:
        return [rows[-1]]

    last_index = len(rows) - 1
    indices = {round(index * last_index / (max_points - 1)) for index in range(max_points)}
    indices.add(0)
    indices.add(last_index)
    return [rows[index] for index in sorted(indices)]


def _run_dir_from_checkpoint_path(raw_path: Any, base_run_dir: Path) -> Path | None:
    if not raw_path:
        return None
    checkpoint_path = resolve_data_path(str(raw_path), base_run_dir)
    if not checkpoint_path:
        return None
    if checkpoint_path.name.endswith(".safetensors") and checkpoint_path.parent.name == "checkpoints":
        candidate = checkpoint_path.parent.parent
        if candidate.exists() and candidate.is_dir():
            return candidate.resolve()

    if checkpoint_path.name.endswith(".safetensors"):
        companions = []
        if checkpoint_path.name.endswith("_model.safetensors"):
            stem = checkpoint_path.name[: -len("_model.safetensors")]
        else:
            stem = checkpoint_path.stem
        companions.extend(
            [
                checkpoint_path.with_name(f"{stem}_state.json"),
                checkpoint_path.with_name(f"{stem}_meta.json"),
            ]
        )
        for companion in companions:
            payload = read_json(companion)
            if not isinstance(payload, dict):
                continue
            for key in ["weights_path", "model_path", "checkpoint_path", "optimizer_path"]:
                source_path = payload.get(key)
                source_run_dir = _run_dir_from_checkpoint_path(source_path, base_run_dir)
                if source_run_dir:
                    return source_run_dir
    return None


def _resume_history_run_dirs(path: Path, max_depth: int = 16) -> list[Path]:
    """Return upstream run dirs in chronological order for a resumed run."""
    current = path.resolve()
    config = read_json(path / "config.json")
    if not isinstance(config, dict):
        return []

    source_dirs: list[Path] = []
    seen = {current}
    checkpoint = config.get("resume_checkpoint_path")
    base_run_dir = path

    for _ in range(max_depth):
        source_dir = _run_dir_from_checkpoint_path(checkpoint, base_run_dir)
        if not source_dir or source_dir in seen:
            break
        source_dirs.append(source_dir)
        seen.add(source_dir)

        source_config = read_json(source_dir / "config.json")
        if not isinstance(source_config, dict):
            break
        checkpoint = source_config.get("resume_checkpoint_path")
        base_run_dir = source_dir

    return list(reversed(source_dirs))


def read_training_log(
    path: Path,
    tail: int | None = None,
    heartbeat_tail: int | None = None,
    heartbeat_points: int | None = None,
) -> list[dict[str, Any]]:
    jsonl_path = path / "training_log.jsonl"
    heartbeat_path = path / "heartbeat_log.jsonl"
    rows_by_step: dict[int, dict[str, Any]] = {}
    merged_rows: list[dict[str, Any]] = []

    def add_rows(
        rows: list[dict[str, Any]],
        source: str,
        *,
        history_run_id: str | None = None,
        overwrite: bool = False,
    ) -> None:
        for row in rows:
            if not isinstance(row, dict):
                continue
            next_row = {**row}
            next_row.setdefault("status_source", source)
            if history_run_id:
                next_row.setdefault("history_source_run_id", history_run_id)
            if source == "heartbeat":
                next_row.setdefault("val_loss", None)
            if isinstance(next_row.get("step"), int):
                step = int(next_row["step"])
                existing = rows_by_step.get(step)
                if overwrite or existing is None or existing.get("status_source") == "heartbeat":
                    rows_by_step[step] = next_row
            else:
                merged_rows.append(next_row)

    for source_run_dir in _resume_history_run_dirs(path):
        source_jsonl = source_run_dir / "training_log.jsonl"
        source_heartbeat = source_run_dir / "heartbeat_log.jsonl"
        source_run_id = source_run_dir.name
        if source_jsonl.exists():
            add_rows(read_jsonl(source_jsonl), "eval", history_run_id=source_run_id)
        if source_heartbeat.exists():
            add_rows(read_jsonl(source_heartbeat), "heartbeat", history_run_id=source_run_id)

    if jsonl_path.exists():
        add_rows(read_jsonl(jsonl_path), "eval", overwrite=True)
    if heartbeat_path.exists():
        add_rows(read_jsonl(heartbeat_path), "heartbeat", overwrite=True)
    if rows_by_step or merged_rows:
        rows = [*sorted(rows_by_step.values(), key=lambda row: int(row.get("step", 0))), *merged_rows]
        if heartbeat_points is not None and heartbeat_points >= 0:
            eval_rows = [row for row in rows if row.get("status_source") != "heartbeat"]
            heartbeat_rows = [row for row in rows if row.get("status_source") == "heartbeat"]
            heartbeat_rows = _downsample_evenly(heartbeat_rows, min(int(heartbeat_points), 5000))
            return sorted([*eval_rows, *heartbeat_rows], key=lambda row: int(row.get("step", 0)))
        if heartbeat_tail is not None and heartbeat_tail >= 0:
            eval_rows = [row for row in rows if row.get("status_source") != "heartbeat"]
            heartbeat_rows = [row for row in rows if row.get("status_source") == "heartbeat"]
            if heartbeat_tail > 0:
                heartbeat_rows = heartbeat_rows[-min(int(heartbeat_tail), 5000) :]
            else:
                heartbeat_rows = []
            return sorted([*eval_rows, *heartbeat_rows], key=lambda row: int(row.get("step", 0)))
        if tail is not None and tail > 0:
            return rows[-min(int(tail), 5000) :]
        return rows

    txt_path = path / "training_log.txt"
    if not txt_path.exists():
        return []
    rows = []
    for line_no, line in enumerate(txt_path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        rows.append({"line": line_no, "text": line})
    if tail is not None and tail > 0:
        return rows[-min(int(tail), 5000) :]
    return rows


def latest_log_row(path: Path) -> dict[str, Any] | None:
    rows = read_training_log(path)
    for row in reversed(rows):
        if "step" in row or "train_loss" in row or "val_loss" in row:
            return row
    return rows[-1] if rows else None


def find_eval_results(path: Path) -> Path | None:
    candidates = [
        path / "eval_results.jsonl",
        path / "evals" / "eval_results.jsonl",
    ]
    return next((p for p in candidates if p.exists()), None)


def find_benchmark(path: Path) -> Path | None:
    candidates = [
        path / "benchmark_results.json",
        path / "benchmarks" / "benchmark.json",
        path / "benchmark.json",
    ]
    return next((p for p in candidates if p.exists()), None)


def list_checkpoints(path: Path) -> list[dict[str, Any]]:
    ckpt_dir = path / "checkpoints"
    if not ckpt_dir.exists():
        return []
    items = []
    for p in sorted(ckpt_dir.iterdir()):
        if p.is_file():
            items.append({"name": p.name, "size": file_size(p), "updated_at": iso_mtime(p)})
    return items


def list_samples(path: Path) -> list[dict[str, Any]]:
    samples_dir = path / "samples"
    if not samples_dir.exists():
        return []
    samples = []
    for p in sorted(samples_dir.iterdir()):
        if p.is_file():
            samples.append({"name": p.name, "size": file_size(p), "updated_at": iso_mtime(p)})
    return samples


def get_sample_text(path: Path, sample_name: str) -> str | None:
    samples_dir = path / "samples"
    sample_path = safe_child(samples_dir, sample_name)
    if not sample_path.exists() or not sample_path.is_file():
        return None
    return read_text(sample_path)


def as_number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.replace(",", ""))
        except ValueError:
            return None
    return None


def as_dict_payload(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def first_number(*values: Any) -> float | None:
    for value in values:
        number = as_number(value)
        if number is not None:
            return number
    return None


def compute_training_estimates(
    config: dict[str, Any] | None,
    metrics: dict[str, Any] | None,
    status: dict[str, Any] | None,
    latest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return teaching estimates for training compute.

    The common dense-transformer rule of thumb is:
    training FLOPs/token ~= 6 * parameter_count.
    These numbers are directional, not a hardware profiler replacement.
    """

    config = as_dict_payload(config)
    metrics = as_dict_payload(metrics)
    status = as_dict_payload(status)
    latest = as_dict_payload(latest)

    parameter_count = first_number(
        status.get("parameter_count"),
        metrics.get("parameter_count"),
        config.get("parameter_count"),
        (status.get("performance") or {}).get("parameter_count"),
        (metrics.get("performance") or {}).get("parameter_count"),
    )
    tokens_seen = first_number(
        status.get("tokens_seen"),
        latest.get("tokens_seen"),
        metrics.get("tokens_seen"),
    )
    tokens_per_second = first_number(
        status.get("tokens_per_second"),
        latest.get("tokens_per_second"),
        metrics.get("tokens_per_second"),
        (status.get("performance") or {}).get("tokens_per_second"),
        (metrics.get("performance") or {}).get("tokens_per_second"),
    )

    estimates: dict[str, Any] = {
        "parameter_count": int(parameter_count) if parameter_count is not None else None,
        "compute_estimate_method": "training_flops ~= 6 * parameter_count * tokens",
    }

    if parameter_count is not None:
        estimates["parameter_count_millions"] = parameter_count / 1e6
        estimates["training_flops_per_token"] = 6.0 * parameter_count

    if parameter_count is not None and tokens_per_second is not None:
        estimates["estimated_training_tflops"] = 6.0 * parameter_count * tokens_per_second / 1e12

    if parameter_count is not None and tokens_seen is not None:
        estimates["estimated_total_pflops"] = 6.0 * parameter_count * tokens_seen / 1e15
        estimates["effective_param_tokens"] = parameter_count * tokens_seen
        estimates["tokens_per_parameter"] = tokens_seen / parameter_count if parameter_count > 0 else None

    return estimates


def enrich_run_numbers(
    payload: dict[str, Any] | None,
    config: dict[str, Any] | None,
    metrics: dict[str, Any] | None,
    status: dict[str, Any] | None,
    latest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    enriched = dict(payload or {})
    estimates = compute_training_estimates(config, metrics, status, latest)
    for key, value in estimates.items():
        if enriched.get(key) is None and value is not None:
            enriched[key] = value
    performance = dict(enriched.get("performance") or enriched.get("telemetry") or {})
    for key in ["estimated_training_tflops", "estimated_total_pflops", "tokens_per_parameter"]:
        if estimates.get(key) is not None:
            performance.setdefault(key, estimates[key])
    if performance:
        enriched["performance"] = performance
    return enriched


def run_summary(path: Path) -> dict[str, Any]:
    config_path = path / "config.json"
    metrics_path = path / "metrics.json"
    status_path = path / "status.json"
    training_jsonl_path = path / "training_log.jsonl"
    training_txt_path = path / "training_log.txt"
    final_text_path = path / "final_generated_text.txt"
    loss_curve_path = path / "loss_curve.png"
    eval_path = find_eval_results(path)
    benchmark_path = find_benchmark(path)
    samples = list_samples(path)
    checkpoints = list_checkpoints(path)

    config = read_json(config_path) or {}
    metrics = read_json(metrics_path) or {}
    status = read_json(status_path) or {}
    latest = latest_log_row(path) or {}
    updated_candidates = [
        p
        for p in [
            config_path,
            metrics_path,
            status_path,
            training_jsonl_path,
            training_txt_path,
            final_text_path,
            loss_curve_path,
            eval_path,
            benchmark_path,
        ]
        if p is not None and p.exists()
    ]
    updated_candidates += [path / "samples" / item["name"] for item in samples]
    updated_at = max((iso_mtime(p) for p in updated_candidates if p.exists()), default=iso_mtime(path))
    status_updated_at = iso_mtime(status_path)
    status_age_seconds = None
    if status_path.exists():
        status_age_seconds = max(0.0, datetime.now(timezone.utc).timestamp() - status_path.stat().st_mtime)

    raw_state = status.get("state")
    if raw_state == "running" and status_age_seconds is not None and status_age_seconds > 900:
        state = "stale"
    elif raw_state in {"running", "completed", "failed", "stopped"}:
        state = raw_state
    elif raw_state in {"restarted", "superseded"}:
        state = "stopped"
    elif final_text_path.exists() or metrics_path.exists():
        state = "completed"
    elif config_path.exists() and status_age_seconds is not None and status_age_seconds <= 900:
        state = "initializing"
    else:
        state = "unknown"

    estimates = compute_training_estimates(config, metrics, status, latest)

    return {
        "run_id": path.name,
        "path": str(path),
        "updated_at": updated_at,
        "status_updated_at": status_updated_at,
        "status_age_seconds": status_age_seconds,
        "state": state,
        "raw_state": raw_state,
        "is_running": state == "running",
        "is_stale": state == "stale",
        "has_config": config_path.exists(),
        "has_training_log": training_jsonl_path.exists() or training_txt_path.exists(),
        "has_metrics": metrics_path.exists(),
        "has_status": status_path.exists(),
        "has_samples": len(samples) > 0,
        "has_final_text": final_text_path.exists(),
        "has_eval_results": eval_path is not None,
        "has_benchmark": benchmark_path is not None,
        "has_loss_curve": loss_curve_path.exists(),
        "checkpoint_count": len(checkpoints),
        "sample_count": len(samples),
        "model_type": config.get("model_type"),
        "tokenizer_type": config.get("tokenizer_type"),
        "latest_step": status.get("step") or latest.get("step") or metrics.get("max_iters"),
        "latest_train_loss": status.get("train_loss") or latest.get("train_loss") or metrics.get("final_train_loss"),
        "latest_val_loss": status.get("val_loss") or latest.get("val_loss") or metrics.get("final_val_loss"),
        "tokens_seen": status.get("tokens_seen") or latest.get("tokens_seen") or metrics.get("tokens_seen"),
        "tokens_per_second": status.get("tokens_per_second") or latest.get("tokens_per_second") or metrics.get("tokens_per_second"),
        "parameter_count": estimates.get("parameter_count"),
        "estimated_training_tflops": estimates.get("estimated_training_tflops"),
        "estimated_total_pflops": estimates.get("estimated_total_pflops"),
        "tokens_per_parameter": estimates.get("tokens_per_parameter"),
    }


def list_runs() -> list[dict[str, Any]]:
    runs_dir = get_runs_dir()
    if not runs_dir.exists():
        return []
    run_dirs = [p for p in runs_dir.iterdir() if p.is_dir()]
    real_run_dirs = [p for p in run_dirs if p.name != "demo_run"]
    if real_run_dirs:
        run_dirs = real_run_dirs
    runs = [run_summary(p) for p in run_dirs]
    state_rank = {"running": 0, "stale": 1, "initializing": 2, "stopped": 3, "failed": 4, "completed": 5}

    def updated_timestamp(row: dict[str, Any]) -> float:
        value = row.get("updated_at")
        if not value:
            return 0.0
        try:
            return datetime.fromisoformat(str(value)).timestamp()
        except ValueError:
            return 0.0

    return sorted(
        runs,
        key=lambda row: (
            state_rank.get(str(row.get("state")), 9),
            -updated_timestamp(row),
        ),
    )


def run_detail(run_id: str) -> dict[str, Any]:
    path = run_dir(run_id)
    summary = run_summary(path)
    return {
        **summary,
        "config": read_json(path / "config.json"),
        "metrics": read_json(path / "metrics.json"),
        "status": get_status(run_id),
        "samples": list_samples(path),
        "checkpoints": list_checkpoints(path),
    }


def get_config(run_id: str):
    return read_json(run_dir(run_id) / "config.json")


def get_metrics(run_id: str):
    path = run_dir(run_id)
    config = read_json(path / "config.json") or {}
    metrics = read_json(path / "metrics.json") or {}
    status = read_json(path / "status.json") or {}
    latest = latest_log_row(path) or {}
    return enrich_run_numbers(metrics, config, metrics, status, latest)


def get_status(run_id: str):
    path = run_dir(run_id)
    config = read_json(path / "config.json") or {}
    metrics = read_json(path / "metrics.json") or {}
    status = read_json(path / "status.json")
    if status is not None:
        return enrich_run_numbers(status, config, metrics, status, latest_log_row(path) or {})
    latest = latest_log_row(path) or {}
    derived = {
        "run_id": run_id,
        "step": latest.get("step") or metrics.get("max_iters"),
        "train_loss": latest.get("train_loss") or metrics.get("final_train_loss"),
        "val_loss": latest.get("val_loss") or metrics.get("final_val_loss"),
        "tokens_seen": latest.get("tokens_seen") or metrics.get("tokens_seen"),
        "tokens_per_second": latest.get("tokens_per_second") or metrics.get("tokens_per_second"),
        "elapsed_sec": latest.get("elapsed_sec") or metrics.get("elapsed_sec"),
        "updated_at": run_summary(path).get("updated_at"),
        "source": "status.json" if (path / "status.json").exists() else "derived",
    }
    return enrich_run_numbers(derived, config, metrics, derived, latest)


def get_final_text(run_id: str):
    text = read_text(run_dir(run_id) / "final_generated_text.txt")
    return {"run_id": run_id, "text": text}


def get_eval_results(run_id: str):
    path = run_dir(run_id)
    eval_path = find_eval_results(path)
    rows = read_jsonl(eval_path) if eval_path else []
    metrics = read_json(path / "evals" / "eval_metrics.json") or read_json(path / "eval_metrics.json")
    return {"run_id": run_id, "path": str(eval_path) if eval_path else None, "rows": rows, "metrics": metrics}


def get_benchmark(run_id: str):
    path = run_dir(run_id)
    benchmark_path = find_benchmark(path)
    benchmark = read_json(benchmark_path) if benchmark_path else None
    metrics = read_json(path / "metrics.json")
    return {"run_id": run_id, "path": str(benchmark_path) if benchmark_path else None, "benchmark": benchmark, "metrics": metrics}


def _run_text(cmd: list[str]) -> str:
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=2)
    except Exception:
        return ""
    return result.stdout if result.returncode == 0 else ""


def _parse_vm_stat() -> dict[str, Any]:
    text = _run_text(["vm_stat"])
    page_size = 16384
    values: dict[str, int] = {}
    for line in text.splitlines():
        if "page size of" in line:
            parts = [part for part in line.split() if part.isdigit()]
            if parts:
                page_size = int(parts[0])
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        digits = "".join(ch for ch in value if ch.isdigit())
        if digits:
            values[key.strip()] = int(digits)

    free = values.get("Pages free", 0)
    speculative = values.get("Pages speculative", 0)
    active = values.get("Pages active", 0)
    inactive = values.get("Pages inactive", 0)
    wired = values.get("Pages wired down", 0)
    compressed = values.get("Pages occupied by compressor", 0)
    used = active + inactive + wired + compressed
    total = used + free + speculative
    if total <= 0:
        return {}
    return {
        "page_size": page_size,
        "used_gb": used * page_size / 1024**3,
        "total_gb": total * page_size / 1024**3,
        "free_gb": (free + speculative) * page_size / 1024**3,
        "used_percent": 100.0 * used / total,
    }


def _system_cpu() -> dict[str, Any]:
    cpu_count = os.cpu_count() or 1
    load1, load5, load15 = os.getloadavg()
    ps_text = _run_text(["ps", "-A", "-o", "%cpu="])
    cpu_sum = 0.0
    for line in ps_text.splitlines():
        try:
            cpu_sum += float(line.strip())
        except ValueError:
            pass
    return {
        "cpu_count": cpu_count,
        "load_1m": load1,
        "load_5m": load5,
        "load_15m": load15,
        "load_1m_percent": min(100.0, 100.0 * load1 / cpu_count),
        "estimated_system_cpu_percent": min(100.0, cpu_sum / cpu_count),
        "raw_ps_cpu_sum": cpu_sum,
    }


def _training_processes() -> list[dict[str, Any]]:
    text = _run_text(["ps", "-axo", "pid,ppid,stat,%cpu,%mem,rss,etime,command"])
    rows = []
    for line in text.splitlines()[1:]:
        if "train_qwen_like.py" not in line:
            continue
        parts = line.split(None, 7)
        if len(parts) < 8:
            continue
        pid, ppid, stat, cpu, mem, rss, etime, command = parts
        try:
            rss_gb = int(rss) * 1024 / 1024**3
        except ValueError:
            rss_gb = None
        rows.append(
            {
                "pid": int(pid),
                "ppid": int(ppid),
                "stat": stat,
                "cpu_percent": float(cpu),
                "mem_percent": float(mem),
                "rss_gb": rss_gb,
                "elapsed": etime,
                "command": command,
            }
        )
    return rows


def get_system_resources() -> dict[str, Any]:
    processes = _training_processes()
    training_cpu = sum(row.get("cpu_percent") or 0.0 for row in processes)
    return {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "cpu": _system_cpu(),
        "memory": _parse_vm_stat(),
        "training_processes": processes,
        "training_cpu_percent": training_cpu,
        "gpu": {
            "util_percent": None,
            "status": "unavailable",
            "note": "Apple Silicon GPU utilization is not exposed by a stable non-sudo MLX API. Use MLX/Metal memory, step time, and tokens/sec as training signals; powermetrics or Metal capture can be added later with extra permissions.",
        },
    }


def ensure_demo_run() -> Path:
    runs_dir = get_runs_dir()
    demo_dir = runs_dir / "demo_run"
    samples_dir = demo_dir / "samples"
    checkpoints_dir = demo_dir / "checkpoints"
    samples_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def write_if_missing(path: Path, text: str) -> None:
        if not path.exists():
            path.write_text(text, encoding="utf-8")

    write_if_missing(
        demo_dir / "config.json",
        json.dumps(
            {
                "run_name": "demo_run",
                "model_type": "qwen_dense_tiny",
                "tokenizer_type": "char",
                "block_size": 16,
                "batch_size": 8,
                "n_embd": 32,
                "num_heads": 4,
                "num_layers": 1,
                "learning_rate": 0.003,
            },
            indent=2,
            ensure_ascii=False,
        ),
    )
    write_if_missing(
        demo_dir / "training_log.jsonl",
        "\n".join(
            [
                json.dumps({"step": 0, "train_loss": 5.1, "val_loss": 5.2, "tokens_seen": 128, "elapsed_sec": 0.1}, ensure_ascii=False),
                json.dumps({"step": 20, "train_loss": 3.0, "val_loss": 3.2, "tokens_seen": 2688, "elapsed_sec": 0.5}, ensure_ascii=False),
                json.dumps({"step": 40, "train_loss": 1.8, "val_loss": 2.0, "tokens_seen": 5120, "elapsed_sec": 0.9}, ensure_ascii=False),
            ]
        )
        + "\n",
    )
    write_if_missing(
        demo_dir / "metrics.json",
        json.dumps(
            {
                "run_name": "demo_run",
                "final_train_loss": 1.8,
                "final_val_loss": 2.0,
                "best_val_loss": 2.0,
                "tokens_seen": 5120,
                "tokens_per_second": 5600.0,
                "parameter_count": 18656,
                "elapsed_sec": 0.9,
            },
            indent=2,
            ensure_ascii=False,
        ),
    )
    write_if_missing(
        demo_dir / "status.json",
        json.dumps(
            {
                "run_id": "demo_run",
                "step": 40,
                "train_loss": 1.8,
                "val_loss": 2.0,
                "tokens_seen": 5120,
                "tokens_per_second": 5600.0,
                "updated_at": datetime.now(tz=timezone.utc).isoformat(),
            },
            indent=2,
            ensure_ascii=False,
        ),
    )
    write_if_missing(samples_dir / "sample_step_0000.txt", "人工智能正在学习中文。")
    write_if_missing(samples_dir / "sample_step_0100.txt", "人工智能可以根据上下文预测下一个 token。")
    write_if_missing(demo_dir / "final_generated_text.txt", "人工智能正在改变学习方式，本地模型让实验更容易观察。")
    write_if_missing(
        demo_dir / "eval_results.jsonl",
        json.dumps(
            {
                "id": "demo_eval_001",
                "prompt": "人工智能",
                "generated": "人工智能正在改变学习方式。",
                "length": 12,
                "repeated_char_ratio": 0.0,
            },
            ensure_ascii=False,
        )
        + "\n",
    )
    write_if_missing(
        demo_dir / "benchmark_results.json",
        json.dumps(
            {
                "tokens_per_second": 128.4,
                "elapsed_sec": 0.47,
                "generated_tokens": 60,
                "parameter_count": 18656,
            },
            indent=2,
            ensure_ascii=False,
        ),
    )
    return demo_dir
