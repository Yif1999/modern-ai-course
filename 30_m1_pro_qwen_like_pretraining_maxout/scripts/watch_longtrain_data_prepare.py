from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import shlex
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any


CURRENT_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = CURRENT_DIR.parent
STATUS_DIR = CURRENT_DIR / "outputs" / "status"
LOG_DIR = CURRENT_DIR / "outputs" / "logs"
LOCK_DIR = CURRENT_DIR / "outputs" / "locks"


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_log(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"[{now_text()}] {line}\n")


def run_text(args: list[str]) -> str:
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    return result.stdout


def screen_exists(screen_name: str) -> bool:
    out = run_text(["screen", "-ls"])
    for line in out.splitlines():
        stripped = line.strip()
        if not stripped or "." not in stripped:
            continue
        session = stripped.split()[0]
        if session.split(".", 1)[-1] == screen_name:
            return True
    return False


def find_prepare_pids(output_version: str) -> list[int]:
    out = run_text(["ps", "-axo", "pid=,command="])
    pids: list[int] = []
    for line in out.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        pid_text, _, command = stripped.partition(" ")
        if "prepare_longtrain_sharded_tokens.py" not in command:
            continue
        if f"--output-version {output_version}" not in command:
            continue
        if "watch_longtrain_data_prepare.py" in command:
            continue
        try:
            pids.append(int(pid_text))
        except ValueError:
            continue
    return sorted(set(pids))


def is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def stop_prepare_processes(
    *,
    output_version: str,
    screen_name: str,
    log_path: Path,
    graceful_timeout_sec: float,
    dry_run: bool,
) -> None:
    pids = find_prepare_pids(output_version)
    append_log(log_path, f"stopping prepare processes pids={pids} screen={screen_name}")
    if dry_run:
        return

    for pid in pids:
        try:
            os.kill(pid, signal.SIGINT)
            append_log(log_path, f"sent SIGINT pid={pid}")
        except OSError as exc:
            append_log(log_path, f"SIGINT failed pid={pid}: {exc}")

    deadline = time.time() + graceful_timeout_sec
    while time.time() < deadline:
        if not any(is_alive(pid) for pid in pids):
            break
        time.sleep(1.0)

    remaining = [pid for pid in pids if is_alive(pid)]
    if remaining:
        append_log(log_path, f"graceful stop timed out; quitting screen and SIGTERM pids={remaining}")
        subprocess.run(["screen", "-S", screen_name, "-X", "quit"], check=False)
        for pid in remaining:
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass
        time.sleep(5.0)

    remaining = [pid for pid in remaining if is_alive(pid)]
    if remaining:
        append_log(log_path, f"SIGTERM timed out; SIGKILL pids={remaining}")
        for pid in remaining:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass


def start_prepare_screen(args: argparse.Namespace, *, dry_run: bool) -> None:
    prepare_log = LOG_DIR / f"{args.output_version}_data_prepare.log"
    prepared_cache_flag = ""
    include_remote_flag = "--include-remote " if args.include_remote else ""
    no_source_caps_flag = "--no-source-caps " if args.no_source_caps else ""
    no_shuffle_sources_flag = "--no-shuffle-sources " if args.no_shuffle_sources else ""
    watchdog_status = read_json(STATUS_DIR / f"{args.output_version}_watchdog_status.json")
    command = (
        f"cd {PROJECT_DIR} && "
        "source .venv/bin/activate && "
        f"echo '[{now_text()}] watchdog resume start' >> {prepare_log} && "
        "python "
        f"{CURRENT_DIR / 'scripts' / 'prepare_longtrain_sharded_tokens.py'} "
        f"--output-version {args.output_version} "
        f"--vocab-size {args.vocab_size} "
        f"--target-train-tokens {args.target_train_tokens} "
        f"--val-tokens {args.val_tokens} "
        f"--shard-tokens {args.shard_tokens} "
        f"{include_remote_flag}"
        f"{prepared_cache_flag}"
        f"{no_source_caps_flag}"
        f"{no_shuffle_sources_flag}"
        f"--max-docs-per-source {args.max_docs_per_source} "
        f"--category-quota-mode {args.category_quota_mode} "
        f"--source-retries {args.source_retries} "
        f"--retry-sleep-sec {args.retry_sleep_sec} "
        f"--status-interval-tokens {args.status_interval_tokens} "
        f"--shuffle-buffer-docs {args.shuffle_buffer_docs} "
        f"--seed {args.seed} "
        "--resume "
        f">> {prepare_log} 2>&1"
    )
    watchdog_log = LOG_DIR / f"{args.output_version}_watchdog.log"
    append_log(watchdog_log, f"starting screen={args.screen_name} command={command}")
    if dry_run:
        return
    subprocess.run(["screen", "-dmS", args.screen_name, "zsh", "-lc", command], check=False)


def status_mtime(path: Path) -> float | None:
    if not path.exists():
        return None
    return path.stat().st_mtime


def check_once(args: argparse.Namespace) -> dict[str, Any]:
    status_path = STATUS_DIR / f"{args.output_version}_status.json"
    watchdog_status_path = STATUS_DIR / f"{args.output_version}_watchdog_status.json"
    watchdog_log = LOG_DIR / f"{args.output_version}_watchdog.log"

    status = read_json(status_path)
    watchdog_state = read_json(watchdog_status_path)
    now = time.time()

    total_tokens = int(status.get("total_tokens", 0) or 0)
    target_total_tokens = int(
        status.get("target_total_tokens", args.target_train_tokens + args.val_tokens) or 0
    )
    state = str(status.get("state", "missing"))
    current_source = str(status.get("current_source") or "")
    pids = find_prepare_pids(args.output_version)
    running = bool(pids)
    mtime = status_mtime(status_path)

    has_watchdog_history = "last_total_tokens" in watchdog_state
    last_total = int(watchdog_state.get("last_total_tokens", total_tokens))
    last_progress_at = float(watchdog_state.get("last_progress_at_unix", mtime or now))
    if has_watchdog_history and total_tokens > last_total:
        last_progress_at = now if mtime is None else max(mtime, now)
        last_total = total_tokens

    stale_age = now - last_progress_at
    complete = total_tokens >= target_total_tokens or state in {"complete", "exhausted_sources"}
    action = "none"
    reason = ""

    if complete:
        action = "complete"
        reason = f"state={state} total_tokens={total_tokens}"
    elif not running:
        action = "restart"
        reason = "prepare process is not running"
    elif stale_age >= args.stale_sec:
        action = "restart"
        reason = f"no token progress for {stale_age:.0f}s"

    append_log(
        watchdog_log,
        "check "
        f"running={running} pids={pids} state={state} total={total_tokens:,} "
        f"stale_age={stale_age:.0f}s action={action} reason={reason}",
    )

    if action == "restart":
        write_json(
            watchdog_status_path,
            {
                **watchdog_state,
                "output_version": args.output_version,
                "last_total_tokens": last_total,
                "last_progress_at_unix": last_progress_at,
            },
        )
        stop_prepare_processes(
            output_version=args.output_version,
            screen_name=args.screen_name,
            log_path=watchdog_log,
            graceful_timeout_sec=args.graceful_timeout_sec,
            dry_run=args.dry_run,
        )
        time.sleep(args.restart_sleep_sec)
        start_prepare_screen(args, dry_run=args.dry_run)
        last_progress_at = now

    payload = {
        "output_version": args.output_version,
        "checked_at": now_text(),
        "checked_at_unix": now,
        "prepare_running": running,
        "prepare_pids": pids,
        "screen_name": args.screen_name,
        "prepare_screen_exists": screen_exists(args.screen_name),
        "state": state,
        "current_source": current_source,
        "total_tokens": total_tokens,
        "target_total_tokens": target_total_tokens,
        "progress_percent": 100.0 * total_tokens / max(target_total_tokens, 1),
        "last_total_tokens": last_total,
        "last_progress_at_unix": last_progress_at,
        "seconds_since_progress": now - last_progress_at,
        "action": action,
        "reason": reason,
        "dry_run": args.dry_run,
        "status_path": str(status_path),
        "watchdog_log": str(watchdog_log),
    }
    write_json(watchdog_status_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch and resume the 10B longtrain data preparation job.")
    parser.add_argument("--output-version", default="longtrain_10b_lab_bpe_16384")
    parser.add_argument("--screen-name", default="longtrain_10b_data_prepare")
    parser.add_argument("--check-interval-sec", type=float, default=600.0)
    parser.add_argument("--stale-sec", type=float, default=600.0)
    parser.add_argument("--graceful-timeout-sec", type=float, default=60.0)
    parser.add_argument("--restart-sleep-sec", type=float, default=10.0)
    parser.add_argument("--vocab-size", type=int, default=16_384)
    parser.add_argument("--target-train-tokens", type=int, default=10_000_000_000)
    parser.add_argument("--val-tokens", type=int, default=10_000_000)
    parser.add_argument("--shard-tokens", type=int, default=25_000_000)
    parser.add_argument("--max-docs-per-source", type=int, default=50_000_000)
    parser.add_argument("--category-quota-mode", choices=["hard", "soft", "flexible", "off"], default="flexible")
    parser.add_argument("--no-source-caps", action="store_true")
    parser.add_argument("--no-shuffle-sources", action="store_true")
    parser.add_argument(
        "--include-remote",
        action="store_true",
        help="Allow tokenization-stage restarts to stream remote HF datasets directly. Default is cache-first only.",
    )
    parser.add_argument("--source-retries", type=int, default=5)
    parser.add_argument("--retry-sleep-sec", type=float, default=60.0)
    parser.add_argument("--status-interval-tokens", type=int, default=1_000_000)
    parser.add_argument("--shuffle-buffer-docs", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2060)
    parser.add_argument(
        "--include-prepared-cache-on-resume",
        action="store_true",
        help="By default watchdog restarts skip local prepared caches to avoid duplicating already appended cache data.",
    )
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = LOCK_DIR / f"{args.output_version}_watchdog.lock"
    with lock_path.open("w", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            message = {
                "output_version": args.output_version,
                "action": "exit",
                "reason": f"another watchdog instance holds {lock_path}",
            }
            print(json.dumps(message, ensure_ascii=False), flush=True)
            return
        lock_file.write(f"pid={os.getpid()} started_at={now_text()}\n")
        lock_file.flush()

        while True:
            payload = check_once(args)
            print(json.dumps(payload, ensure_ascii=False), flush=True)
            if args.once or payload["action"] == "complete":
                break
            time.sleep(args.check_interval_sec)


if __name__ == "__main__":
    main()
