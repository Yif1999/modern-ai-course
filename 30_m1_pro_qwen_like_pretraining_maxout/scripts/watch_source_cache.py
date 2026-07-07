from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any


CURRENT_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = CURRENT_DIR.parents[0]
OUTPUT_DIR = CURRENT_DIR / "outputs"
STATUS_DIR = OUTPUT_DIR / "status"
LOG_DIR = OUTPUT_DIR / "logs"
LOCK_DIR = OUTPUT_DIR / "locks"


def now_text() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"[{now_text()}] {message}\n")


def run_text(args: list[str]) -> str:
    result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    return result.stdout


def screen_exists(screen_name: str) -> bool:
    output = run_text(["screen", "-ls"])
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped or "." not in stripped:
            continue
        session = stripped.split()[0]
        if session.split(".", 1)[-1] == screen_name:
            return True
    return False


def matching_pids(output_version: str) -> list[int]:
    output = run_text(["ps", "-axo", "pid,command"])
    pids: list[int] = []
    needle = "prepare_streaming_source_cache.py"
    for line in output.splitlines():
        if needle not in line or output_version not in line:
            continue
        parts = line.strip().split(maxsplit=1)
        if not parts:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        if pid != os.getpid():
            pids.append(pid)
    return pids


def is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def stop_job(screen_name: str, output_version: str, log_path: Path, timeout_sec: float) -> None:
    pids = matching_pids(output_version)
    append_log(log_path, f"stopping screen={screen_name} pids={pids}")
    subprocess.run(["screen", "-S", screen_name, "-X", "quit"], check=False)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        remaining = [pid for pid in pids if is_alive(pid)]
        if not remaining:
            return
        time.sleep(2.0)
    remaining = [pid for pid in pids if is_alive(pid)]
    if remaining:
        append_log(log_path, f"SIGTERM timed out; SIGKILL pids={remaining}")
        for pid in remaining:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass


def start_source_cache(args: argparse.Namespace, *, dry_run: bool) -> None:
    run_log = LOG_DIR / f"{args.output_version}_source_cache.log"
    source_args = " ".join(f"--source {name}" for name in args.source)
    skip_args = " ".join(f"--skip-source {name}" for name in args.skip_source)
    force_args = " ".join(f"--force-rebuild-source {name}" for name in args.force_rebuild_source)
    max_docs = f"--max-docs-per-source {args.max_docs_per_source} " if args.max_docs_per_source is not None else ""
    max_chars = f"--max-chars-per-source {args.max_chars_per_source} " if args.max_chars_per_source is not None else ""
    command = (
        f"cd {PROJECT_DIR} && "
        "source .venv/bin/activate && "
        f"echo '[{now_text()}] source cache watchdog start' >> {run_log} && "
        "python "
        f"{CURRENT_DIR / 'scripts' / 'prepare_streaming_source_cache.py'} "
        f"--output-version {args.output_version} "
        f"{max_docs}"
        f"{max_chars}"
        f"--min-chars {args.min_chars} "
        f"--status-interval-docs {args.status_interval_docs} "
        f"--status-interval-chars {args.status_interval_chars} "
        f"--stream-shuffle-buffer {args.stream_shuffle_buffer} "
        f"--seed {args.seed} "
        f"{source_args} "
        f"{skip_args} "
        f"{force_args} "
        "--resume "
        f">> {run_log} 2>&1"
    )
    watchdog_log = LOG_DIR / f"{args.output_version}_source_cache_watchdog.log"
    append_log(watchdog_log, f"starting screen={args.screen_name} command={command}")
    if dry_run:
        return
    subprocess.run(["screen", "-dmS", args.screen_name, "zsh", "-lc", command], check=False)


def check_once(args: argparse.Namespace) -> dict[str, Any]:
    status_path = STATUS_DIR / f"{args.output_version}_status.json"
    watchdog_status_path = STATUS_DIR / f"{args.output_version}_source_cache_watchdog_status.json"
    watchdog_log = LOG_DIR / f"{args.output_version}_source_cache_watchdog.log"
    status = read_json(status_path)
    watchdog_state = read_json(watchdog_status_path)
    restart_count = int(watchdog_state.get("restart_count", 0) or 0)
    state = status.get("state")
    complete = state in {"complete", "complete_with_errors"}
    now = time.time()
    mtime = status_path.stat().st_mtime if status_path.exists() else None
    age = None if mtime is None else now - mtime
    exists = screen_exists(args.screen_name)
    action = "none"

    if complete:
        action = "complete"
    elif not exists:
        if restart_count >= args.max_restarts:
            action = "blocked_max_restarts"
            append_log(watchdog_log, f"blocked start_missing after restart_count={restart_count}")
        else:
            action = "start_missing"
            restart_count += 1
            start_source_cache(args, dry_run=args.dry_run)
    elif age is not None and age > args.stale_sec:
        if restart_count >= args.max_restarts:
            action = "blocked_max_restarts"
            append_log(watchdog_log, f"blocked restart_stale after restart_count={restart_count} status_age_sec={age}")
        else:
            action = "restart_stale"
            restart_count += 1
            stop_job(args.screen_name, args.output_version, watchdog_log, args.graceful_timeout_sec)
            time.sleep(args.restart_sleep_sec)
            start_source_cache(args, dry_run=args.dry_run)

    payload = {
        "output_version": args.output_version,
        "state": state,
        "action": action,
        "restart_count": restart_count,
        "max_restarts": args.max_restarts,
        "status_age_sec": age,
        "screen_name": args.screen_name,
        "screen_exists": screen_exists(args.screen_name),
        "status_path": str(status_path),
        "watchdog_log": str(watchdog_log),
        "updated_at_unix": now,
        "dry_run": args.dry_run,
    }
    write_json(watchdog_status_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch and resume remote source-cache materialization.")
    parser.add_argument("--output-version", default="longtrain_10b_remote_source_cache_v1")
    parser.add_argument("--screen-name", default="longtrain_remote_source_cache")
    parser.add_argument("--check-interval-sec", type=float, default=600.0)
    parser.add_argument("--stale-sec", type=float, default=900.0)
    parser.add_argument("--graceful-timeout-sec", type=float, default=60.0)
    parser.add_argument("--restart-sleep-sec", type=float, default=10.0)
    parser.add_argument("--max-restarts", type=int, default=3)
    parser.add_argument("--max-docs-per-source", type=int, default=None)
    parser.add_argument("--max-chars-per-source", type=int, default=None)
    parser.add_argument("--min-chars", type=int, default=10)
    parser.add_argument("--status-interval-docs", type=int, default=10_000)
    parser.add_argument("--status-interval-chars", type=int, default=5_000_000)
    parser.add_argument("--stream-shuffle-buffer", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=2060)
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--skip-source", action="append", default=[])
    parser.add_argument("--force-rebuild-source", action="append", default=[])
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = LOCK_DIR / f"{args.output_version}_source_cache_watchdog.lock"
    with lock_path.open("w", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps({"action": "exit", "reason": f"lock held: {lock_path}"}, ensure_ascii=False), flush=True)
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
