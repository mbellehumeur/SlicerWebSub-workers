"""Durable series-level inference result cache under the workers runtime root.

Layout::

    {runtime}/results/{product}/series|{content}/{id}/{model}/{options_hash}/
        meta.json
        output/...

Cache key: product + SeriesInstanceUID (or content sha256) + model + options_hash.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

LOGGER = logging.getLogger("result_cache")

DEFAULT_RUNTIME_ROOT = Path("/var/lib/websub-workers")
OPTIONS_HASH_LEN = 16


@dataclass(frozen=True)
class SeriesKey:
    """Middle path segment under results/{product}/."""

    kind: str  # "series" | "content"
    value: str
    study_instance_uid: Optional[str] = None

    @property
    def path_segment(self) -> str:
        return f"{self.kind}/{self.value}"


@dataclass(frozen=True)
class CacheHit:
    root: Path
    meta: Dict[str, Any]
    output_files: Tuple[Path, ...]
    publish_as: str


def runtime_root() -> Path:
    env = os.environ.get("WEBSUB_WORKERS_RUNTIME_ROOT", "").strip()
    return Path(env).expanduser() if env else DEFAULT_RUNTIME_ROOT


def models_dir() -> Path:
    return runtime_root() / "models"


def jobs_dir() -> Path:
    return runtime_root() / "jobs"


def results_dir() -> Path:
    return runtime_root() / "results"


def _safe_topic_dir_name(topic: str) -> str:
    safe = re.sub(r"[^\w.\-]+", "_", (topic or "").strip())
    return safe or "topic"


def _safe_path_segment(value: str, *, label: str) -> str:
    text = (value or "").strip()
    if not text:
        raise ValueError(f"{label} is empty")
    if ".." in text or "/" in text or "\\" in text:
        raise ValueError(f"{label} contains unsafe path characters: {text!r}")
    # DICOM UIDs are typically [0-9.]; model ids and hashes are alnum/_/-
    if not re.fullmatch(r"[A-Za-z0-9._\-]+", text):
        raise ValueError(f"{label} has invalid characters: {text!r}")
    return text


def options_hash(options: Mapping[str, Any]) -> str:
    payload = json.dumps(options, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:OPTIONS_HASH_LEN]


def job_scratch_dir(product: str, topic: str, *, stamp_ms: Optional[int] = None) -> Path:
    """Ephemeral per-run directory: {runtime}/jobs/{product}/{topic}-{stamp}/."""
    product_seg = _safe_path_segment(product, label="product")
    stamp = stamp_ms if stamp_ms is not None else int(time.time() * 1000)
    job_dir = (
        jobs_dir()
        / product_seg
        / f"{_safe_topic_dir_name(topic)}-{stamp}"
    )
    job_dir.mkdir(parents=True, exist_ok=True)
    return job_dir


def result_dir(
    product: str,
    series_key: SeriesKey,
    model: str,
    options_hash_value: str,
) -> Path:
    product_seg = _safe_path_segment(product, label="product")
    kind = _safe_path_segment(series_key.kind, label="series_key.kind")
    if kind not in ("series", "content"):
        raise ValueError(f"series_key.kind must be 'series' or 'content', got {kind!r}")
    id_seg = _safe_path_segment(series_key.value, label="series_key.value")
    model_seg = _safe_path_segment(model, label="model")
    hash_seg = _safe_path_segment(options_hash_value, label="options_hash")
    return results_dir() / product_seg / kind / id_seg / model_seg / hash_seg


def _context_series_instance_uid(message: Mapping[str, Any]) -> Optional[str]:
    event = message.get("event") or {}
    ctx = event.get("context")
    if not isinstance(ctx, dict):
        return None
    for key in ("seriesInstanceUID", "SeriesInstanceUID", "series_instance_uid"):
        value = ctx.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _iter_candidate_files(input_dir: Path) -> Iterable[Path]:
    if not input_dir.is_dir():
        return
    for path in sorted(input_dir.rglob("*")):
        if path.is_file():
            yield path


def _looks_like_nifti(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _read_dicom_uids(path: Path) -> Optional[Tuple[str, Optional[str]]]:
    try:
        import pydicom
    except ImportError:
        LOGGER.warning("pydicom not installed; cannot read SeriesInstanceUID from %s", path)
        return None
    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    except Exception:
        return None
    series_uid = getattr(ds, "SeriesInstanceUID", None)
    if not series_uid:
        return None
    study_uid = getattr(ds, "StudyInstanceUID", None)
    return str(series_uid), (str(study_uid) if study_uid else None)


def resolve_series_key(
    input_dir: Path,
    message: Mapping[str, Any],
    *,
    primary_nifti: Optional[Path] = None,
) -> SeriesKey:
    """Derive series/ or content/ key from DICOM UIDs, message context, or file hash."""
    context_uid = _context_series_instance_uid(message)
    if context_uid:
        return SeriesKey(kind="series", value=context_uid)

    study_uid: Optional[str] = None
    for path in _iter_candidate_files(input_dir):
        if _looks_like_nifti(path):
            continue
        uids = _read_dicom_uids(path)
        if uids:
            series_uid, study_uid = uids
            return SeriesKey(
                kind="series",
                value=series_uid,
                study_instance_uid=study_uid,
            )

    nifti = primary_nifti
    if nifti is None:
        for path in _iter_candidate_files(input_dir):
            if _looks_like_nifti(path):
                nifti = path
                break
    if nifti is not None and nifti.is_file():
        digest = _file_sha256(nifti)
        return SeriesKey(kind="content", value=digest)

    # Last resort: hash the whole input directory file set (names + contents).
    digest = hashlib.sha256()
    for path in _iter_candidate_files(input_dir):
        digest.update(str(path.relative_to(input_dir)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(_file_sha256(path).encode("ascii"))
        digest.update(b"\0")
    return SeriesKey(kind="content", value=digest.hexdigest())


def lookup_complete(
    product: str,
    series_key: SeriesKey,
    model: str,
    options: Mapping[str, Any],
) -> Optional[CacheHit]:
    ohash = options_hash(options)
    root = result_dir(product, series_key, model, ohash)
    meta_path = root / "meta.json"
    if not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        LOGGER.warning("result cache: unreadable meta at %s", meta_path)
        return None
    if meta.get("status") != "complete":
        return None
    publish_as = str(meta.get("publish_as") or "").strip()
    if publish_as not in ("nifti-send", "dicom-send"):
        LOGGER.warning("result cache: bad publish_as in %s", meta_path)
        return None
    files_meta = meta.get("files")
    if not isinstance(files_meta, list) or not files_meta:
        return None
    output_root = root / "output"
    resolved: List[Path] = []
    for entry in files_meta:
        if isinstance(entry, str):
            rel = entry
        elif isinstance(entry, dict):
            rel = str(entry.get("path") or entry.get("fileName") or "").strip()
        else:
            return None
        if not rel or ".." in rel or rel.startswith(("/", "\\")):
            return None
        path = output_root / rel
        if not path.is_file():
            LOGGER.warning("result cache: missing file %s", path)
            return None
        resolved.append(path)
    return CacheHit(
        root=root,
        meta=meta,
        output_files=tuple(resolved),
        publish_as=publish_as,
    )


def store_complete(
    product: str,
    series_key: SeriesKey,
    model: str,
    options: Mapping[str, Any],
    *,
    output_files: Sequence[Path],
    publish_as: str,
) -> Path:
    """Copy result files into the cache tree and write meta.json (status=complete)."""
    if publish_as not in ("nifti-send", "dicom-send"):
        raise ValueError(f"publish_as must be nifti-send or dicom-send, got {publish_as!r}")
    if not output_files:
        raise ValueError("output_files is empty")

    ohash = options_hash(options)
    root = result_dir(product, series_key, model, ohash)
    output_root = root / "output"
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    rel_files: List[str] = []
    for src in output_files:
        src_path = Path(src)
        if not src_path.is_file():
            raise FileNotFoundError(f"result file missing: {src_path}")
        dest_name = src_path.name
        dest = output_root / dest_name
        shutil.copy2(src_path, dest)
        rel_files.append(dest_name)

    meta: Dict[str, Any] = {
        "status": "complete",
        "product": product,
        "model": model,
        "options": dict(options),
        "options_hash": ohash,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": rel_files,
        "publish_as": publish_as,
        "key_kind": series_key.kind,
    }
    if series_key.kind == "series":
        meta["series_instance_uid"] = series_key.value
    else:
        meta["content_sha256"] = series_key.value
    if series_key.study_instance_uid:
        meta["study_instance_uid"] = series_key.study_instance_uid

    meta_path = root / "meta.json"
    tmp_path = root / "meta.json.tmp"
    tmp_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(meta_path)
    LOGGER.info("result cache: stored complete result at %s", root)
    return root
