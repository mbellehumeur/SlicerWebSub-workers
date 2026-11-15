#!/usr/bin/env python3
"""EUCAIM Hub service provider — Docker-backed MIC-DKFZ nnU-Net tasks.

Run from repo root (plain Python, no 3D Slicer):

    source .venv/bin/activate
    pip install -r requirements.txt
    python eucaim/eucaim_docker.py
    python eucaim/eucaim_docker.py --local

Default hub is SLICER-HUB-CLOUD; ``--local`` uses ``http://127.0.0.1:2018``.

Requires ``event.context.models`` (array of catalog ids) or ``event.context.model``.
Optional: ``gpus`` (default ``all``; ``none``/``cpu``/``off`` for CPU). Hosts
with Blackwell GPUs (compute capability 12.0+) auto-fall back to CPU unless
``EUCAIM_FORCE_GPU=1``.

On inbound dicom-send: download, results-cache lookup, Docker run on miss,
publish SEG outputs, store under ``/var/lib/websub-workers/results/EUCAIM/``.

See ``eucaim/README.md`` and ``runtime/result_cache.py``.
"""

from __future__ import annotations

import asyncio
import logging
import os
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

from docker_runner import (  # noqa: E402
    DEFAULT_GPUS,
    canonicalize_model_id,
    collect_outputs,
    log_output_tree,
    resolve_image_tag,
    run_dicom_job,
    validate_model,
)
from gpu_compat import (  # noqa: E402
    cpu_fallback_reason,
    should_force_cpu_for_eucaim,
)
from result_cache import (  # noqa: E402
    lookup_complete,
    resolve_series_key,
    store_complete,
)
from worker import (  # noqa: E402
    ServiceProviderConfig,
    ServiceProviderContext,
    ServiceProviderHandlers,
    run_sync,
)

LOGGER = logging.getLogger("EUCAIM")

PRODUCT_NAME = "EUCAIM"
FINAL_STATUS_LINE = "Job complete"
GPUS_ENV = "EUCAIM_GPUS"
FORCE_GPU_ENV = "EUCAIM_FORCE_GPU"
_CPU_GPU_TOKENS = frozenset({"none", "off", "cpu", "false", "0", "no"})


def build_status_response(_ctx: ServiceProviderContext) -> Dict[str, Any]:
    return {
        "source": "status",
        "product": PRODUCT_NAME,
        "items": [{"key": "availability", "value": "online"}],
    }


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


def _parse_models_list(message: Mapping[str, Any]) -> List[str]:
    """Return ordered canonical model ids from ``context.models`` / ``context.model``."""
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
                name = _context_str(entry, "model", "id", "name", "modelId")
                if name:
                    requested.append(name)
    singular = _context_str(ctx, "model", "Model", "modelId", "model_id")
    if singular and singular not in requested:
        requested.append(singular)
    if not requested:
        raise ValueError(
            "missing event.context.models (array) or event.context.model "
            '(e.g. ["liver_tumour"] or "hippocampus_segmentation")'
        )
    models: List[str] = []
    for name in requested:
        canonical = canonicalize_model_id(name)
        if canonical not in models:
            models.append(canonical)
    return models


def _normalize_gpus_value(raw: str) -> str:
    """Return Docker ``--gpus`` value, or ``\"\"`` for CPU (no ``--gpus``)."""
    text = raw.strip()
    if text.lower() in _CPU_GPU_TOKENS:
        return ""
    return text


def _env_flag_true(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _resolve_gpus(message: Mapping[str, Any]) -> str:
    """Docker ``--gpus`` value; empty string means CPU (omit the flag)."""
    ctx = _event_context(message)
    from_ctx = _context_str(ctx, "gpus", "Gpus", "GPU")
    if from_ctx:
        return _normalize_gpus_value(from_ctx)
    env = os.environ.get(GPUS_ENV, "").strip()
    if env:
        return _normalize_gpus_value(env)
    return DEFAULT_GPUS


def _apply_host_gpu_policy(
    gpus: str, *, image_tag: Optional[str] = None
) -> Tuple[str, Optional[str]]:
    """Maybe force CPU when the host GPU cannot run published EUCAIM images."""
    if not gpus:
        return gpus, None
    if _env_flag_true(FORCE_GPU_ENV):
        LOGGER.info(
            "EUCAIM: %s set; skipping Blackwell CPU fallback (gpus=%s image=%s)",
            FORCE_GPU_ENV,
            gpus,
            image_tag or "(unset)",
        )
        return gpus, None
    if not should_force_cpu_for_eucaim(image_tag=image_tag):
        return gpus, None
    reason = cpu_fallback_reason(image_tag=image_tag)
    LOGGER.warning("EUCAIM: %s", reason)
    return "", reason


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
        "EUCAIM: status-update %r -> %s (HTTP %s)",
        status_line,
        target_subscriber,
        http_status,
    )


def on_send_download_start(
    ctx: ServiceProviderContext, message: Dict[str, Any], _hub_event: str
) -> None:
    file_count, total_bytes = _manifest_file_stats(message)
    if file_count <= 0:
        LOGGER.warning(
            "EUCAIM: inbound send has no context.files[]; "
            "skipping download status-update"
        )
        return
    _publish_to_requester(
        ctx, message, _format_download_status_line(file_count, total_bytes)
    )


def _require_job_routing(
    ctx: ServiceProviderContext,
    message: Dict[str, Any],
    input_dir: Path,
    label: str,
) -> Optional[str]:
    event = message.get("event") or {}
    topic = (event.get("hub.topic") or "").strip()
    LOGGER.info(
        "EUCAIM: received %s id=%s topic=%s",
        label,
        message.get("id", ""),
        topic,
    )
    ctx.write_directory_manifest(input_dir)
    if not topic:
        LOGGER.warning("EUCAIM: inbound %s missing hub.topic", label)
        return None
    if not str(message.get("subscriber.name") or "").strip():
        LOGGER.warning(
            "EUCAIM: inbound %s missing subscriber.name; "
            "cannot send status-update",
            label,
        )
        return None
    return topic


def _publish_files_sync(
    ctx: ServiceProviderContext,
    topic: str,
    files: Sequence[Path],
    publish_as: str,
) -> None:
    for path in files:
        if publish_as != "dicom-send":
            raise ValueError(f"unsupported publish_as: {publish_as!r}")
        future = asyncio.run_coroutine_threadsafe(
            ctx.publish_dicom_send(topic, str(path)),
            ctx.loop,
        )
        http_status = future.result(timeout=300.0)
        LOGGER.info(
            "EUCAIM: published %s %s HTTP %s", publish_as, path.name, http_status
        )


def _handle_inbound_send(
    ctx: ServiceProviderContext,
    message: Dict[str, Any],
    input_dir: Path,
    file_count: int,
    total_bytes: int,
    label: str,
    publish_as: str,
) -> None:
    del file_count, total_bytes
    topic = _require_job_routing(ctx, message, input_dir, label)
    if topic is None:
        return

    _publish_to_requester(ctx, message, "Download complete.")

    try:
        models = _parse_models_list(message)
    except ValueError as exc:
        _publish_to_requester(ctx, message, f"ERROR: {exc}")
        LOGGER.warning("EUCAIM: bad job options: %s", exc)
        return

    requested_gpus = _resolve_gpus(message)
    series_key = resolve_series_key(input_dir, message)
    job_dir = input_dir.parent

    LOGGER.info(
        "EUCAIM: models=%s series=%s/%s requested_gpus=%s",
        models,
        series_key.kind,
        series_key.value,
        requested_gpus or "cpu",
    )

    for model in models:
        image = resolve_image_tag(model)
        gpus, gpu_status = _apply_host_gpu_policy(
            requested_gpus, image_tag=image
        )
        if gpu_status:
            _publish_to_requester(ctx, message, f"{gpu_status} ({model})")
        LOGGER.info(
            "EUCAIM: model=%s image=%s gpus=%s",
            model,
            image,
            gpus or "cpu",
        )
        options: Dict[str, Any] = {"model": model}
        LOGGER.info(
            "EUCAIM: cache key kind=%s value=%s model=%s options=%s",
            series_key.kind,
            series_key.value,
            model,
            options,
        )

        hit = lookup_complete(PRODUCT_NAME, series_key, model, options)
        if hit is not None:
            _publish_to_requester(
                ctx, message, f"Using cached result ({model})"
            )
            LOGGER.info("EUCAIM: cache hit %s", hit.root)
            try:
                _publish_files_sync(
                    ctx, topic, hit.output_files, hit.publish_as
                )
            except Exception as exc:
                _publish_to_requester(
                    ctx, message, f"ERROR: publish failed ({model}): {exc}"
                )
                LOGGER.exception(
                    "EUCAIM: cached publish failed model=%s", model
                )
                return
            continue

        data_dir = job_dir / "data" / model
        docker_log = job_dir / f"docker_{model}.log"
        if data_dir.exists():
            shutil.rmtree(data_dir)

        def _status(line: str) -> None:
            _publish_to_requester(ctx, message, line)

        try:
            output_dir = run_dicom_job(
                model,
                input_dir,
                data_dir,
                gpus=gpus,
                on_status=_status,
                log_path=docker_log,
            )
        except Exception as exc:
            _publish_to_requester(ctx, message, f"ERROR: {model}: {exc}")
            LOGGER.exception(
                "EUCAIM: docker/inference failed model=%s log=%s",
                model,
                docker_log,
            )
            return

        log_output_tree(data_dir, logger=LOGGER, label="data tree")
        log_output_tree(output_dir, logger=LOGGER, label="output")
        LOGGER.info("EUCAIM: docker log at %s", docker_log)

        result_files = list(collect_outputs(output_dir, publish_as))
        if not result_files:
            _publish_to_requester(
                ctx, message, f"Job completed with no output ({model})"
            )
            LOGGER.warning(
                "EUCAIM: no publishable outputs model=%s output=%s "
                "data_dir=%s docker_log=%s",
                model,
                output_dir,
                data_dir,
                docker_log,
            )
            continue

        try:
            store_complete(
                PRODUCT_NAME,
                series_key,
                model,
                options,
                output_files=result_files,
                publish_as=publish_as,
            )
        except Exception:
            LOGGER.exception(
                "EUCAIM: failed to store result cache model=%s", model
            )

        _publish_to_requester(
            ctx, message, f"Publishing result ({model})…"
        )
        try:
            _publish_files_sync(ctx, topic, result_files, publish_as)
        except Exception as exc:
            _publish_to_requester(
                ctx, message, f"ERROR: publish failed ({model}): {exc}"
            )
            LOGGER.exception("EUCAIM: publish failed model=%s", model)
            return

        LOGGER.info(
            "EUCAIM: model complete model=%s files=%d", model, len(result_files)
        )

    _publish_to_requester(ctx, message, FINAL_STATUS_LINE)
    LOGGER.info("EUCAIM: job complete models=%s", models)


def on_dicom_send(
    ctx: ServiceProviderContext,
    message: Dict[str, Any],
    input_dir: Path,
    file_count: int,
    total_bytes: int,
) -> None:
    _handle_inbound_send(
        ctx,
        message,
        input_dir,
        file_count,
        total_bytes,
        "dicom-send",
        "dicom-send",
    )


HANDLERS = ServiceProviderHandlers(
    on_dicom_send=on_dicom_send,
    on_send_download_start=on_send_download_start,
    build_status_response=build_status_response,
)


if __name__ == "__main__":
    run_sync(ServiceProviderConfig(product_name=PRODUCT_NAME), HANDLERS)
