"""Host resource metrics for the worker control panel."""

from __future__ import annotations

import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .process_manager import runtime_root

_cpu_primed = False
_net_prev: Optional[Tuple[float, int, int]] = None  # mono, bytes_recv, bytes_sent
_CONTROL_STARTED_AT = datetime.now(timezone.utc)
_CONTROL_STARTED_MONO = time.monotonic()


def _prime_cpu() -> None:
    global _cpu_primed
    if _cpu_primed:
        return
    import psutil

    psutil.cpu_percent(interval=None)
    _cpu_primed = True


def _find_mountpoint(path: Path) -> str:
    import psutil

    try:
        resolved = path.resolve()
    except OSError:
        resolved = path
    best = "/"
    for part in psutil.disk_partitions(all=False):
        mp = part.mountpoint
        try:
            mp_path = Path(mp).resolve()
        except OSError:
            mp_path = Path(mp)
        prefix = mp_path.as_posix().rstrip("/") or "/"
        target = resolved.as_posix()
        if prefix == "/" or target == prefix or target.startswith(prefix + "/"):
            if len(prefix) >= len(best.rstrip("/") or "/"):
                best = part.mountpoint
    return best


def _memory_metrics() -> Dict[str, Any]:
    import psutil

    vm = psutil.virtual_memory()
    used_mb = round((vm.total - vm.available) / (1024 * 1024), 1)
    total_mb = round(vm.total / (1024 * 1024), 1)
    percent = round(float(vm.percent), 1)
    return {
        "used_mb": used_mb,
        "total_mb": total_mb,
        "percent": percent,
        "chart_value": percent,
    }


def _cpu_metrics() -> float:
    import psutil

    _prime_cpu()
    return round(float(psutil.cpu_percent(interval=None)), 1)


def _network_metrics() -> Dict[str, Any]:
    global _net_prev
    import psutil

    now = time.monotonic()
    io = psutil.net_io_counters()
    recv = int(io.bytes_recv)
    sent = int(io.bytes_sent)
    rx_rate = 0.0
    tx_rate = 0.0
    if _net_prev is not None:
        prev_t, prev_recv, prev_sent = _net_prev
        dt = max(now - prev_t, 1e-6)
        rx_rate = max(0.0, (recv - prev_recv) / dt)
        tx_rate = max(0.0, (sent - prev_sent) / dt)
    _net_prev = (now, recv, sent)
    total_rate = rx_rate + tx_rate
    return {
        "bytes_recv": recv,
        "bytes_sent": sent,
        "rx_bytes_per_s": round(rx_rate, 1),
        "tx_bytes_per_s": round(tx_rate, 1),
        "chart_value": round(total_rate, 1),
    }


def _parse_nvidia_smi() -> Optional[List[Dict[str, Any]]]:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,utilization.memory,"
                "memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if proc.returncode != 0:
        return None
    gpus: List[Dict[str, Any]] = []
    for line in (proc.stdout or "").strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            util = float(parts[1])
            mem_util = float(parts[2])
            mem_used = float(parts[3])
            mem_total = float(parts[4])
        except ValueError:
            continue
        mem_percent = (
            round((mem_used / mem_total) * 100.0, 1) if mem_total > 0 else 0.0
        )
        gpus.append(
            {
                "name": parts[0],
                "utilization_percent": util,
                "memory_utilization_percent": mem_util,
                "memory_used_mb": mem_used,
                "memory_total_mb": mem_total,
                "memory_percent": mem_percent,
                "chart_value": util,
            }
        )
    return gpus or None


def _gpu_metrics() -> Dict[str, Any]:
    gpus = _parse_nvidia_smi()
    if not gpus:
        return {"available": False, "gpus": [], "chart_value": None}
    primary = gpus[0]
    return {
        "available": True,
        "gpus": gpus,
        "name": primary["name"],
        "utilization_percent": primary["utilization_percent"],
        "memory_used_mb": primary["memory_used_mb"],
        "memory_total_mb": primary["memory_total_mb"],
        "memory_percent": primary["memory_percent"],
        "chart_value": primary["chart_value"],
        "count": len(gpus),
    }


def _disk_entry(path: Path) -> Optional[Dict[str, Any]]:
    import psutil

    try:
        usage = psutil.disk_usage(str(path))
        mount = _find_mountpoint(path)
    except OSError:
        return None
    total = max(int(usage.total), 1)
    used = int(usage.used)
    free = int(usage.free)
    percent = round(float(usage.percent), 1)
    return {
        "path": str(path),
        "mount": mount,
        "total_bytes": total,
        "used_bytes": used,
        "free_bytes": free,
        "percent": percent,
        "total_gb": round(total / (1024**3), 1),
        "used_gb": round(used / (1024**3), 1),
        "free_gb": round(free / (1024**3), 1),
    }


def _disk_metrics() -> Dict[str, Any]:
    candidates: List[Path] = [Path("/")]
    root = runtime_root()
    try:
        if root.exists():
            candidates.append(root)
    except OSError:
        pass

    seen_mounts: set[str] = set()
    partitions: List[Dict[str, Any]] = []
    for path in candidates:
        entry = _disk_entry(path)
        if entry is None:
            continue
        mount = entry["mount"]
        if mount in seen_mounts:
            continue
        seen_mounts.add(mount)
        partitions.append(entry)

    if not partitions:
        return {"partitions": [], "chart_value": None, "primary": None}

    primary = partitions[0]
    for part in partitions:
        if part["path"] != "/":
            primary = part
            break

    return {
        "partitions": partitions,
        "primary": primary,
        "percent": primary["percent"],
        "chart_value": primary["percent"],
    }


def collect_host_metrics() -> Dict[str, Any]:
    """Sample host memory, CPU, network, GPU, and disk fill."""
    _prime_cpu()
    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "started_at": _CONTROL_STARTED_AT.isoformat(),
        "uptime_seconds": round(time.monotonic() - _CONTROL_STARTED_MONO, 1),
        "memory": _memory_metrics(),
        "cpu_percent": _cpu_metrics(),
        "network": _network_metrics(),
        "gpu": _gpu_metrics(),
        "disk": _disk_metrics(),
    }
