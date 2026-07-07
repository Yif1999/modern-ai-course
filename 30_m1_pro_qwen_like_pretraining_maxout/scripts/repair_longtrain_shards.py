from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np


CURRENT_DIR = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair sharded token data by rolling back to a trusted token count."
    )
    parser.add_argument("--output-version", default="longtrain_10b_lab_bpe_16384")
    parser.add_argument("--safe-total-tokens", type=int, required=True)
    parser.add_argument("--val-tokens", type=int, default=10_000_000)
    parser.add_argument("--target-total-tokens", type=int, default=10_010_000_000)
    parser.add_argument("--shard-tokens", type=int, default=25_000_000)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def shard_index(path: Path, prefix: str) -> int:
    stem = path.stem
    marker = f"{prefix}_shard_"
    if marker not in stem:
        return -1
    return int(stem.split(marker, 1)[1].split("_", 1)[0])


def npy_len(path: Path) -> int:
    return int(np.load(path, mmap_mode="r").shape[0])


def move_to_quarantine(path: Path, quarantine_dir: Path, apply: bool) -> Path:
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    dest = quarantine_dir / path.name
    if dest.exists():
        suffix = datetime.now().strftime("%H%M%S")
        dest = quarantine_dir / f"{path.stem}_{suffix}{path.suffix}"
    if apply:
        shutil.move(str(path), str(dest))
    return dest


def repair_split(
    split_dir: Path,
    prefix: str,
    target_tokens: int,
    quarantine_dir: Path,
    apply: bool,
) -> dict:
    files = sorted(split_dir.glob(f"{prefix}_shard_*.npy"), key=lambda p: shard_index(p, prefix))
    kept_files: list[str] = []
    moved_files: list[dict] = []
    trimmed_file: dict | None = None
    total = 0

    for path in files:
        n = npy_len(path)
        if total >= target_tokens:
            dest = move_to_quarantine(path, quarantine_dir, apply)
            moved_files.append({"from": str(path), "to": str(dest), "tokens": n})
            continue

        if total + n <= target_tokens:
            kept_files.append(str(path))
            total += n
            continue

        keep = target_tokens - total
        dest = move_to_quarantine(path, quarantine_dir, apply)
        trimmed_file = {
            "original": str(path),
            "quarantine": str(dest),
            "original_tokens": n,
            "kept_tokens": keep,
        }
        if apply:
            arr = np.load(dest, mmap_mode="r")
            trimmed = np.asarray(arr[:keep], dtype=np.int32)
            np.save(path, trimmed)
        kept_files.append(str(path))
        total += keep

    return {
        "split": prefix,
        "target_tokens": target_tokens,
        "final_tokens": total,
        "kept_count": len(kept_files),
        "moved_count": len(moved_files),
        "trimmed_file": trimmed_file,
        "moved_files_preview": moved_files[:5],
        "moved_files_tail": moved_files[-5:],
    }


def update_status(args: argparse.Namespace, train_tokens: int, val_tokens: int, apply: bool) -> dict:
    status_dir = CURRENT_DIR / "outputs" / "status"
    status_path = status_dir / f"{args.output_version}_status.json"
    watchdog_path = status_dir / f"{args.output_version}_watchdog_status.json"
    total = train_tokens + val_tokens
    payload = {
        "state": "interrupted_repaired",
        "output_version": args.output_version,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "total_tokens": total,
        "target_total_tokens": args.target_total_tokens,
        "progress_percent": total / args.target_total_tokens * 100,
        "repair_note": "Rolled back to trusted pre-concurrency checkpoint; later shards quarantined.",
        "repaired_at": datetime.now().isoformat(timespec="seconds"),
    }
    if apply:
        status_dir.mkdir(parents=True, exist_ok=True)
        if status_path.exists():
            backup = status_path.with_suffix(status_path.suffix + ".pre_repair_backup")
            shutil.copy2(status_path, backup)
            payload["status_backup"] = str(backup)
        status_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        if watchdog_path.exists():
            backup = watchdog_path.with_suffix(watchdog_path.suffix + ".pre_repair_backup")
            shutil.copy2(watchdog_path, backup)
            watchdog_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    args = parse_args()
    processed_dir = CURRENT_DIR / "data" / "processed" / args.output_version
    quarantine_dir = (
        CURRENT_DIR
        / "data"
        / "processed"
        / f"{args.output_version}_quarantine_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    safe_train_tokens = args.safe_total_tokens - args.val_tokens
    if safe_train_tokens <= 0:
        raise ValueError("safe train token count must be positive")

    report = {
        "mode": "apply" if args.apply else "dry_run",
        "output_version": args.output_version,
        "processed_dir": str(processed_dir),
        "quarantine_dir": str(quarantine_dir),
        "safe_total_tokens": args.safe_total_tokens,
        "safe_train_tokens": safe_train_tokens,
        "safe_val_tokens": args.val_tokens,
    }
    report["train"] = repair_split(
        processed_dir / "train",
        "train",
        safe_train_tokens,
        quarantine_dir / "train",
        args.apply,
    )
    report["val"] = repair_split(
        processed_dir / "val",
        "val",
        args.val_tokens,
        quarantine_dir / "val",
        args.apply,
    )
    report["status"] = update_status(args, safe_train_tokens, args.val_tokens, args.apply)

    report_path = CURRENT_DIR / "outputs" / "reports" / f"{args.output_version}_repair_report.json"
    if args.apply:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
