#!/usr/bin/env python3
"""ORCHID Hub WebSub client — bridges Slicer Hub ↔ independent Orchid platform.

Orchid installs separately (``./orchid/docker/up.sh``). This process only
connects to the Hub and talks to a running Orchid instance over REST/DICOM.

Run from repo root:

    source .venv/bin/activate
    pip install -r requirements.txt
    # optional for C-STORE ingest: pip install pynetdicom
    python orchid/orchid_websub.py
    python orchid/orchid_websub.py --local

Requires Orchid healthy at ``ORCHID_API_URL`` (default http://127.0.0.1:8000).

On inbound dicom-send: download, results-cache lookup, ingest into Orchid,
poll until complete, publish DICOM SR (+ findings.json), cache under
``/var/lib/websub-workers/results/ORCHID/``.

See ``orchid/README.md``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
_WORKERS_ROOT = _SCRIPT_DIR.parent
for _extra in (
    _SCRIPT_DIR,
    _WORKERS_ROOT,
    _WORKERS_ROOT / "runtime",
    _WORKERS_ROOT / "python_client" / "src",
):
    _extra_str = str(_extra)
    if _extra.is_dir() and _extra_str not in sys.path:
        sys.path.insert(0, _extra_str)

from orchid_client import (  # noqa: E402
    api_base,
    ensure_orchid,
    run_orchid_job,
)
from result_cache import (  # noqa: E402
    lookup_complete,
    resolve_series_key,
    series_key_from_message,
    store_complete,
)
from worker import (  # noqa: E402
    ServiceProviderConfig,
    ServiceProviderContext,
    ServiceProviderHandlers,
    run_sync,
)

LOGGER = logging.getLogger("ORCHID")

PRODUCT_NAME = "ORCHID"
FINAL_STATUS_LINE = "Job complete"
DEFAULT_MODEL = "routing"


def build_status_response(_ctx: ServiceProviderContext) -> Dict[str, Any]:
    items = [{"key": "availability", "value": "online"}]
    try:
        info = ensure_orchid()
        items.append({"key": "orchid_api", "value": api_base()})
        items.append({"key": "orchid_aet", "value": str(info.get("aet") or "")})
        items.append(
            {
                "key": "orchid_dicom_port",
                "value": str(info.get("dicom_port") or ""),
            }
        )
    except Exception as exc:
        items = [
            {"key": "availability", "value": "degraded"},
            {"key": "orchid_error", "value": str(exc)[:200]},
        ]
    return {"source": "status", "product": PRODUCT_NAME, "items": items}


def _event_context(message: Mapping[str, Any]) -> Dict[str, Any]:
    event = message.get("event") or {}
    ctx = event.get("context")
    return ctx if isinstance(ctx, dict) else {}


def _context_files(message: Dict[str, Any]) -> List[Dict[str, Any]]:
    files = _event_context(message).get("files")
    if isinstance(files, list):
        return [entry for entry in files if isinstance(entry, dict)]
    return []


def _manifest_file_stats(message: Dict[str, Any]) -> Tuple[int, int]:
    files = _context_files(message)
    total_bytes = 0
    for entry in files:
        byte_length = entry.get("byteLength")
        if isinstance(byte_length, int) and byte_length >= 0:
            total_bytes += byte_length
    return len(files), total_bytes


def _format_download_status_line(file_count: int, total_bytes: int) -> str:
    if total_bytes > 0:
        total_mb = total_bytes / (1024 * 1024)
        mb_text = f"{total_mb:.1f}" if total_mb < 100 else f"{total_mb:.0f}"
        return f"Downloading study, {file_count} files, total {mb_text} MB."
    return f"Downloading study, {file_count} files."


def _context_str(ctx: Mapping[str, Any], *keys: str) -> Optional[str]:
    for key in keys:
        value = ctx.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _parse_app_slugs(message: Mapping[str, Any]) -> List[str]:
    """Ordered Orchid app slugs from ``context.app`` / ``context.models``."""
    ctx = _event_context(message)
    requested: List[str] = []
    raw_list = ctx.get("models")
    if raw_list is None:
        raw_list = ctx.get("Models")
    if isinstance(raw_list, list):
        for entry in raw_list:
            if isinstance(entry, str) and entry.strip():
                requested.append(entry.strip())
            elif isinstance(entry, dict):
                name = _context_str(
                    entry, "app", "model", "id", "name", "slug", "modelId"
                )
                if name:
                    requested.append(name)
    singular = _context_str(
        ctx, "app", "App", "model", "Model", "modelId", "model_id", "slug"
    )
    if singular and singular not in requested:
        requested.append(singular)
    # Preserve order, drop dupes.
    out: List[str] = []
    for name in requested:
        if name not in out:
            out.append(name)
    return out


def _cache_model_id(app_slugs: Sequence[str]) -> str:
    if not app_slugs:
        return DEFAULT_MODEL
    if len(app_slugs) == 1:
        return app_slugs[0]
    return "+".join(app_slugs)


def _publish_to_requester(
    ctx: ServiceProviderContext, message: Dict[str, Any], status_line: str
) -> None:
    event = message.get("event") or {}
    topic = (event.get("hub.topic") or "").strip()
    target_subscriber = str(message.get("subscriber.name") or "").strip()
    if not topic or not target_subscriber:
        return
    http_status = ctx.publish_status_update_sync(
        topic, target_subscriber, status_line
    )
    LOGGER.info(
        "ORCHID: status-update %r -> %s (HTTP %s)",
        status_line[:120],
        target_subscriber,
        http_status,
    )


def on_send_download_start(
    ctx: ServiceProviderContext, message: Dict[str, Any], _hub_event: str
) -> None:
    file_count, total_bytes = _manifest_file_stats(message)
    if file_count <= 0:
        LOGGER.warning(
            "ORCHID: inbound send has no context.files[]; "
            "skipping download status-update"
        )
        return
    _publish_to_requester(
        ctx, message, _format_download_status_line(file_count, total_bytes)
    )


def on_try_cache_before_download(
    ctx: ServiceProviderContext, message: Dict[str, Any], hub_event: str
) -> bool:
    """Skip study download when SeriesInstanceUID already has a complete cache hit."""
    if hub_event != "dicom-send":
        return False
    series_key = series_key_from_message(message)
    if series_key is None:
        return False
    event = message.get("event") or {}
    topic = (event.get("hub.topic") or "").strip()
    target_subscriber = str(message.get("subscriber.name") or "").strip() or None
    if not topic or not target_subscriber:
        return False

    app_slugs = _parse_app_slugs(message)
    model = _cache_model_id(app_slugs)
    options: Dict[str, Any] = {"model": model}
    if app_slugs:
        options["apps"] = list(app_slugs)

    hit = lookup_complete(PRODUCT_NAME, series_key, model, options)
    if hit is None:
        return False

    _publish_to_requester(ctx, message, f"Using cached result ({model})")
    LOGGER.info(
        "ORCHID: pre-download cache hit %s model=%s series=%s",
        hit.root,
        model,
        series_key.value,
    )
    findings_path = next(
        (p for p in hit.output_files if p.name == "findings.json"), None
    )
    if findings_path and findings_path.is_file():
        try:
            findings = json.loads(findings_path.read_text(encoding="utf-8"))
            if isinstance(findings, dict):
                _publish_findings_status(ctx, message, findings)
        except (OSError, json.JSONDecodeError):
            LOGGER.warning("ORCHID: could not load cached findings.json")
    try:
        _publish_files_sync(
            ctx,
            topic,
            hit.output_files,
            target_subscriber_name=target_subscriber,
        )
    except Exception as exc:
        _publish_to_requester(
            ctx, message, f"ERROR: publish failed ({model}): {exc}"
        )
        LOGGER.exception("ORCHID: pre-download cached publish failed")
        return True
    _publish_to_requester(ctx, message, FINAL_STATUS_LINE)
    return True


def _require_job_routing(
    ctx: ServiceProviderContext,
    message: Dict[str, Any],
    input_dir: Path,
    label: str,
) -> Optional[str]:
    event = message.get("event") or {}
    topic = (event.get("hub.topic") or "").strip()
    LOGGER.info(
        "ORCHID: received %s id=%s topic=%s",
        label,
        message.get("id", ""),
        topic,
    )
    ctx.write_directory_manifest(input_dir)
    if not topic:
        LOGGER.warning("ORCHID: inbound %s missing hub.topic", label)
        return None
    if not str(message.get("subscriber.name") or "").strip():
        LOGGER.warning(
            "ORCHID: inbound %s missing subscriber.name; "
            "cannot send status-update",
            label,
        )
        return None
    return topic


def _publish_files_sync(
    ctx: ServiceProviderContext,
    topic: str,
    files: Sequence[Path],
    *,
    target_subscriber_name: Optional[str] = None,
) -> None:
    for path in files:
        if path.suffix.lower() not in (".dcm", ".dicom") and path.name != "findings.json":
            # Only publish DICOM SR back as dicom-send; JSON goes via status.
            if path.suffix.lower() == ".json":
                continue
        if path.suffix.lower() in (".dcm", ".dicom"):
            future = asyncio.run_coroutine_threadsafe(
                ctx.publish_dicom_send(
                    topic,
                    str(path),
                    target_subscriber_name=target_subscriber_name,
                ),
                ctx.loop,
            )
            http_status = future.result(timeout=300.0)
            LOGGER.info(
                "ORCHID: published dicom-send %s HTTP %s",
                path.name,
                http_status,
            )


def _publish_findings_status(
    ctx: ServiceProviderContext,
    message: Dict[str, Any],
    findings: Mapping[str, Any],
) -> None:
    # Compact RESULT line for viewers that parse status-update JSON.
    compact = {
        "study_id": findings.get("study_id"),
        "study_uid": findings.get("study_uid"),
        "status": findings.get("status"),
        "results": [
            {
                "id": r.get("id"),
                "app_name": r.get("app_name"),
                "kind": r.get("kind"),
                "positive": r.get("positive"),
                "confidence": r.get("confidence"),
                "label": (r.get("payload") or {}).get("label")
                if isinstance(r.get("payload"), dict)
                else None,
                "summary": r.get("summary"),
            }
            for r in (findings.get("results") or [])
            if isinstance(r, dict)
        ],
    }
    line = "RESULT: " + json.dumps(compact, separators=(",", ":"), default=str)
    if len(line) > 3500:
        line = line[:3490] + "…}"
    _publish_to_requester(ctx, message, line)


def on_dicom_send(
    ctx: ServiceProviderContext,
    message: Dict[str, Any],
    input_dir: Path,
    file_count: int,
    total_bytes: int,
) -> None:
    del file_count, total_bytes
    topic = _require_job_routing(ctx, message, input_dir, "dicom-send")
    if topic is None:
        return

    _publish_to_requester(ctx, message, "Download complete.")

    try:
        ensure_orchid(on_status=lambda line: _publish_to_requester(ctx, message, line))
    except Exception as exc:
        _publish_to_requester(ctx, message, f"ERROR: {exc}")
        LOGGER.exception("ORCHID: platform not available")
        return

    app_slugs = _parse_app_slugs(message)
    model = _cache_model_id(app_slugs)
    options: Dict[str, Any] = {"model": model}
    if app_slugs:
        options["apps"] = list(app_slugs)

    series_key = resolve_series_key(input_dir, message)
    job_dir = input_dir.parent
    LOGGER.info(
        "ORCHID: model=%s series=%s/%s apps=%s",
        model,
        series_key.kind,
        series_key.value,
        app_slugs or "(routing)",
    )

    target_subscriber = str(message.get("subscriber.name") or "").strip() or None
    hit = lookup_complete(PRODUCT_NAME, series_key, model, options)
    if hit is not None:
        _publish_to_requester(ctx, message, f"Using cached result ({model})")
        LOGGER.info("ORCHID: cache hit %s", hit.root)
        findings_path = next(
            (p for p in hit.output_files if p.name == "findings.json"), None
        )
        if findings_path and findings_path.is_file():
            try:
                findings = json.loads(findings_path.read_text(encoding="utf-8"))
                if isinstance(findings, dict):
                    _publish_findings_status(ctx, message, findings)
            except (OSError, json.JSONDecodeError):
                LOGGER.warning("ORCHID: could not load cached findings.json")
        try:
            _publish_files_sync(
                ctx,
                topic,
                hit.output_files,
                target_subscriber_name=target_subscriber,
            )
        except Exception as exc:
            _publish_to_requester(
                ctx, message, f"ERROR: publish failed ({model}): {exc}"
            )
            LOGGER.exception("ORCHID: cached publish failed")
            return
        _publish_to_requester(ctx, message, FINAL_STATUS_LINE)
        return

    output_dir = job_dir / "output"
    if output_dir.exists():
        shutil.rmtree(output_dir)

    def _status(line: str) -> None:
        _publish_to_requester(ctx, message, line)

    try:
        result_files, findings = run_orchid_job(
            input_dir,
            output_dir,
            app_slugs=app_slugs or None,
            on_status=_status,
        )
    except Exception as exc:
        _publish_to_requester(ctx, message, f"ERROR: {exc}")
        LOGGER.exception("ORCHID: job failed")
        return

    if not result_files:
        _publish_to_requester(
            ctx, message, f"Job completed with no output ({model})"
        )
        LOGGER.warning("ORCHID: no artifacts model=%s", model)
        _publish_to_requester(ctx, message, FINAL_STATUS_LINE)
        return

    try:
        store_complete(
            PRODUCT_NAME,
            series_key,
            model,
            options,
            output_files=result_files,
            publish_as="dicom-send",
        )
    except Exception:
        LOGGER.exception("ORCHID: failed to store result cache")

    _publish_findings_status(ctx, message, findings)
    _publish_to_requester(ctx, message, f"Publishing result ({model})…")
    try:
        _publish_files_sync(
            ctx,
            topic,
            result_files,
            target_subscriber_name=target_subscriber,
        )
    except Exception as exc:
        _publish_to_requester(
            ctx, message, f"ERROR: publish failed ({model}): {exc}"
        )
        LOGGER.exception("ORCHID: publish failed")
        return

    _publish_to_requester(ctx, message, FINAL_STATUS_LINE)
    LOGGER.info(
        "ORCHID: job complete model=%s files=%d", model, len(result_files)
    )


HANDLERS = ServiceProviderHandlers(
    on_dicom_send=on_dicom_send,
    on_send_download_start=on_send_download_start,
    on_try_cache_before_download=on_try_cache_before_download,
    build_status_response=build_status_response,
)


if __name__ == "__main__":
    try:
        ensure_orchid()
    except Exception as exc:
        LOGGER.error("%s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        # Still start the Hub client so status-request can report degraded;
        # dicom-send will fail fast with the same message.
    run_sync(ServiceProviderConfig(product_name=PRODUCT_NAME), HANDLERS)
