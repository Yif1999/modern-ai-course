from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import mlx.core as mx


def bytes_to_gb(value: int | float | None) -> float | None:
    if value is None:
        return None
    return float(value) / (1024**3)


def safe_call(fn):
    try:
        return fn()
    except Exception:
        return None


def process_rss_gb() -> float | None:
    def read():
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())], text=True).strip()
        return int(out) * 1024

    return bytes_to_gb(safe_call(read))


def vm_stat_memory() -> dict[str, float | None]:
    def read():
        out = subprocess.check_output(["vm_stat"], text=True)
        page_size = 16384
        values: dict[str, int] = {}
        for line in out.splitlines():
            if ":" not in line:
                continue
            key, raw = line.split(":", 1)
            raw = raw.strip().rstrip(".")
            if raw.replace(".", "", 1).isdigit():
                values[key] = int(float(raw))
        free = values.get("Pages free", 0) + values.get("Pages speculative", 0)
        active = values.get("Pages active", 0)
        inactive = values.get("Pages inactive", 0)
        wired = values.get("Pages wired down", 0)
        compressed = values.get("Pages occupied by compressor", 0)
        used = active + inactive + wired + compressed
        total = used + free
        return {
            "system_memory_used_gb": bytes_to_gb(used * page_size),
            "system_memory_total_gb": bytes_to_gb(total * page_size),
            "system_memory_used_percent": (used / total * 100) if total else None,
        }

    return safe_call(read) or {
        "system_memory_used_gb": None,
        "system_memory_total_gb": None,
        "system_memory_used_percent": None,
    }


def mlx_memory_snapshot() -> dict[str, float | None]:
    snap = {
        "mlx_active_memory_gb": bytes_to_gb(safe_call(mx.get_active_memory)),
        "mlx_peak_memory_gb": bytes_to_gb(safe_call(mx.get_peak_memory)),
        "mlx_cache_memory_gb": bytes_to_gb(safe_call(mx.get_cache_memory)),
    }
    metal = getattr(mx, "metal", None)
    if metal is not None:
        snap.update(
            {
                "metal_active_memory_gb": bytes_to_gb(safe_call(metal.get_active_memory)),
                "metal_peak_memory_gb": bytes_to_gb(safe_call(metal.get_peak_memory)),
                "metal_cache_memory_gb": bytes_to_gb(safe_call(metal.get_cache_memory)),
            }
        )
    return snap


def telemetry_snapshot(
    *,
    step_time_ms: float | None,
    tokens_per_second: float | None,
    eta_sec: float | None,
    progress_percent: float | None,
) -> dict[str, Any]:
    perf: dict[str, Any] = {
        "source": "mlx+process",
        "step_time_ms": step_time_ms,
        "tokens_per_second": tokens_per_second,
        "eta_sec": eta_sec,
        "progress_percent": progress_percent,
        "process_rss_gb": process_rss_gb(),
        "gpu_util_percent": None,
        "gpu_util_note": "Apple Silicon GPU utilization is not available through a stable non-sudo MLX API.",
        "thermal_state": None,
        "updated_at_unix": time.time(),
    }
    perf.update(mlx_memory_snapshot())
    perf.update(vm_stat_memory())
    return perf


def write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
