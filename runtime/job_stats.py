"""Durable per-job usage stats for the worker control panel.

Stores job counts, bytes in/out, and wall-clock compute time under
``/var/lib/websub-workers/control/stats.sqlite``.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_RUNTIME_ROOT = Path("/var/lib/websub-workers")

_lock = threading.RLock()
_active_by_product: Dict[str, str] = {}  # product_name -> job_id


def runtime_root() -> Path:
    env = os.environ.get("WEBSUB_WORKERS_RUNTIME_ROOT", "").strip()
    return Path(env).expanduser() if env else DEFAULT_RUNTIME_ROOT


def control_root() -> Path:
    return runtime_root() / "control"


def stats_db_path() -> Path:
    return control_root() / "stats.sqlite"


def _connect() -> sqlite3.Connection:
    path = stats_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            product_name TEXT NOT NULL,
            started_at TEXT NOT NULL,
            ended_at TEXT,
            wall_seconds REAL,
            bytes_in INTEGER NOT NULL DEFAULT 0,
            bytes_out INTEGER NOT NULL DEFAULT 0,
            success INTEGER,
            hub_event TEXT
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_jobs_product ON jobs(product_name)"
    )
    conn.commit()


def begin_job(
    product_name: str,
    *,
    hub_event: str = "",
    bytes_in: int = 0,
) -> str:
    """Insert a running job and mark it active for this product_name."""
    job_id = uuid.uuid4().hex
    started = datetime.now(timezone.utc).isoformat()
    name = (product_name or "").strip() or "UNKNOWN"
    with _lock:
        conn = _connect()
        try:
            _ensure_schema(conn)
            conn.execute(
                """
                INSERT INTO jobs (
                    id, product_name, started_at, bytes_in, bytes_out, hub_event
                ) VALUES (?, ?, ?, ?, 0, ?)
                """,
                (job_id, name, started, int(bytes_in or 0), hub_event or ""),
            )
            conn.commit()
        finally:
            conn.close()
        _active_by_product[name] = job_id
    return job_id


def set_bytes_in(job_id: str, bytes_in: int) -> None:
    with _lock:
        conn = _connect()
        try:
            _ensure_schema(conn)
            conn.execute(
                "UPDATE jobs SET bytes_in = ? WHERE id = ?",
                (int(bytes_in or 0), job_id),
            )
            conn.commit()
        finally:
            conn.close()


def add_bytes_out(job_id: str, nbytes: int) -> None:
    if nbytes <= 0:
        return
    with _lock:
        conn = _connect()
        try:
            _ensure_schema(conn)
            conn.execute(
                "UPDATE jobs SET bytes_out = bytes_out + ? WHERE id = ?",
                (int(nbytes), job_id),
            )
            conn.commit()
        finally:
            conn.close()


def add_bytes_out_for_product(product_name: str, nbytes: int) -> None:
    """Accumulate outbound bytes on the active job for ``product_name``."""
    name = (product_name or "").strip()
    if not name or nbytes <= 0:
        return
    with _lock:
        job_id = _active_by_product.get(name)
    if job_id:
        add_bytes_out(job_id, nbytes)


def end_job(
    job_id: str,
    *,
    wall_seconds: float,
    success: bool,
) -> None:
    ended = datetime.now(timezone.utc).isoformat()
    with _lock:
        conn = _connect()
        try:
            _ensure_schema(conn)
            conn.execute(
                """
                UPDATE jobs
                SET ended_at = ?, wall_seconds = ?, success = ?
                WHERE id = ?
                """,
                (
                    ended,
                    float(wall_seconds),
                    1 if success else 0,
                    job_id,
                ),
            )
            conn.commit()
        finally:
            conn.close()
        for key, active_id in list(_active_by_product.items()):
            if active_id == job_id:
                del _active_by_product[key]


def _empty_bucket() -> Dict[str, Any]:
    return {
        "jobs": 0,
        "bytes_in": 0,
        "bytes_out": 0,
        "wall_seconds": 0.0,
        "success_jobs": 0,
        "failed_jobs": 0,
    }


def aggregate_totals() -> Dict[str, Any]:
    """Return host-wide totals across all completed and in-flight jobs."""
    with _lock:
        conn = _connect()
        try:
            _ensure_schema(conn)
            row = conn.execute(
                """
                SELECT
                    COUNT(*) AS jobs,
                    COALESCE(SUM(bytes_in), 0) AS bytes_in,
                    COALESCE(SUM(bytes_out), 0) AS bytes_out,
                    COALESCE(SUM(COALESCE(wall_seconds, 0)), 0) AS wall_seconds,
                    COALESCE(SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END), 0)
                        AS success_jobs,
                    COALESCE(SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END), 0)
                        AS failed_jobs
                FROM jobs
                """
            ).fetchone()
        finally:
            conn.close()
    if row is None:
        return _empty_bucket()
    return {
        "jobs": int(row["jobs"] or 0),
        "bytes_in": int(row["bytes_in"] or 0),
        "bytes_out": int(row["bytes_out"] or 0),
        "wall_seconds": float(row["wall_seconds"] or 0.0),
        "success_jobs": int(row["success_jobs"] or 0),
        "failed_jobs": int(row["failed_jobs"] or 0),
    }


def aggregate_by_product() -> Dict[str, Dict[str, Any]]:
    """Return per-product_name aggregates."""
    with _lock:
        conn = _connect()
        try:
            _ensure_schema(conn)
            rows = conn.execute(
                """
                SELECT
                    product_name,
                    COUNT(*) AS jobs,
                    COALESCE(SUM(bytes_in), 0) AS bytes_in,
                    COALESCE(SUM(bytes_out), 0) AS bytes_out,
                    COALESCE(SUM(COALESCE(wall_seconds, 0)), 0) AS wall_seconds,
                    COALESCE(SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END), 0)
                        AS success_jobs,
                    COALESCE(SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END), 0)
                        AS failed_jobs
                FROM jobs
                GROUP BY product_name
                """
            ).fetchall()
        finally:
            conn.close()
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        name = str(row["product_name"] or "")
        out[name] = {
            "jobs": int(row["jobs"] or 0),
            "bytes_in": int(row["bytes_in"] or 0),
            "bytes_out": int(row["bytes_out"] or 0),
            "wall_seconds": float(row["wall_seconds"] or 0.0),
            "success_jobs": int(row["success_jobs"] or 0),
            "failed_jobs": int(row["failed_jobs"] or 0),
        }
    return out


def collect_stats() -> Dict[str, Any]:
    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "totals": aggregate_totals(),
        "by_product": aggregate_by_product(),
    }


def bytes_out_from_publish_message(hub_message: Dict[str, Any]) -> int:
    """Sum byteLength fields from a dicom/nifti publish message."""
    try:
        files = hub_message.get("event", {}).get("context", {}).get("files") or []
    except AttributeError:
        return 0
    total = 0
    for entry in files:
        if not isinstance(entry, dict):
            continue
        bl = entry.get("byteLength")
        if bl is None and isinstance(entry.get("data"), (bytes, bytearray)):
            bl = len(entry["data"])
        try:
            total += int(bl or 0)
        except (TypeError, ValueError):
            continue
    return total
