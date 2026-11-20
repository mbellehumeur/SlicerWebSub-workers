#!/usr/bin/env python3
"""VOX2SEGLAM Hub service provider — standalone CLI entry point.

Host-native inference is not supported. Prefer the Docker worker::

    docker build -t websub/vox2seglam:cu128 \\
      -f vox2seglam/docker/Dockerfile vox2seglam/docker
    docker build -t websub/tools:cpu \\
      -f tools/docker/Dockerfile tools/docker
    python vox2seglam/vox2seglam_docker.py --local

This module holds Hub handlers; ``vox2seglam_docker.py`` patches
``_run_vox2seglam`` to the cu128 container runner.

Upstream: https://github.com/Neuro-iX/Vox2SegLAM
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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

from result_cache import (  # noqa: E402
    lookup_complete,
    resolve_series_key,
    series_key_from_message,
    store_complete,
)
from tools_runner import (  # noqa: E402
    run_dicom_to_nifti,
    run_landmarks_to_dicom_sr,
    run_nifti_to_dicom_seg,
)
from worker import (  # noqa: E402
    ServiceProviderConfig,
    ServiceProviderContext,
    ServiceProviderHandlers,
    run_sync,
)

LOGGER = logging.getLogger("VOX2SEGLAM")

PRODUCT_NAME = "VOX2SEGLAM"
FINAL_STATUS_LINE = "Segmentation complete"
MODEL_ID = "vox2seglam"
TASK_ENV = "VOX2SEGLAM_TASK"
DEFAULT_TASK = "subcortical"


def _task_name() -> str:
    return (os.environ.get(TASK_ENV) or DEFAULT_TASK).strip() or DEFAULT_TASK


def _inference_options(*, dicom_out: bool) -> Dict[str, Any]:
    options: Dict[str, Any] = {
        "model": MODEL_ID,
        "task": _task_name(),
        "postprocess": True,
        "dicom_out": bool(dicom_out),
    }
    if dicom_out:
        # Invalidate cached SEGs that used generic label_N / lacked landmarks SR.
        options["seg_meta"] = "structures-tsv-v2"
        options["sr_meta"] = "landmarks-sr-v3"
    return options


def build_status_response(_ctx: ServiceProviderContext) -> Dict[str, Any]:
    return {
        "source": "status",
        "product": PRODUCT_NAME,
        "items": [{"key": "availability", "value": "online"}],
    }


def _context_files(message: Dict[str, Any]) -> List[Dict[str, Any]]:
    event = message.get("event") or {}
    ctx = event.get("context")
    if isinstance(ctx, dict):
        files = ctx.get("files")
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


def _format_download_status_line(
    file_count: int, total_bytes: int, hub_event: str
) -> str:
    kind = "DICOM series" if hub_event == "dicom-send" else "NIfTI volume"
    if total_bytes > 0:
        total_mb = total_bytes / (1024 * 1024)
        mb_text = f"{total_mb:.1f}" if total_mb < 100 else f"{total_mb:.0f}"
        return (
            f"Downloading {kind}, {file_count} files, total {mb_text} MB."
        )
    return f"Downloading {kind}, {file_count} files."


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
        "VOX2SEGLAM: status-update %r -> %s (HTTP %s)",
        status_line,
        target_subscriber,
        http_status,
    )


def on_send_download_start(
    ctx: ServiceProviderContext, message: Dict[str, Any], hub_event: str
) -> None:
    file_count, total_bytes = _manifest_file_stats(message)
    if file_count <= 0:
        LOGGER.warning(
            "VOX2SEGLAM: inbound send has no context.files[]; "
            "skipping download status-update"
        )
        return
    _publish_to_requester(
        ctx,
        message,
        _format_download_status_line(file_count, total_bytes, hub_event),
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

    options = _inference_options(dicom_out=True)
    hit = lookup_complete(PRODUCT_NAME, series_key, MODEL_ID, options)
    if hit is None:
        return False

    _publish_to_requester(ctx, message, "Using cached result")
    LOGGER.info(
        "VOX2SEGLAM: pre-download cache hit %s series=%s",
        hit.root,
        series_key.value,
    )
    try:
        _publish_outputs_sync(
            ctx,
            topic,
            hit.output_files,
            hit.publish_as,
            target_subscriber_name=target_subscriber,
        )
    except Exception as exc:
        _publish_to_requester(
            ctx, message, f"ERROR: publish cached result failed: {exc}"
        )
        LOGGER.exception("VOX2SEGLAM: pre-download cached publish failed")
        # Still skip download: result exists; republishing failed.
        return True
    _publish_to_requester(ctx, message, FINAL_STATUS_LINE)
    return True


def _resolve_input_nifti(input_dir: Path) -> Optional[Path]:
    nifti_files = sorted(input_dir.glob("*.nii*"))
    if not nifti_files:
        return None
    return nifti_files[0]


def _publish_result_nifti_sync(
    ctx: ServiceProviderContext,
    topic: str,
    result_path: Path,
    *,
    target_subscriber_name: Optional[str] = None,
) -> None:
    future = asyncio.run_coroutine_threadsafe(
        ctx.publish_nifti_send(
            topic,
            str(result_path),
            target_subscriber_name=target_subscriber_name,
        ),
        ctx.loop,
    )
    http_status = future.result(timeout=300.0)
    LOGGER.info(
        "VOX2SEGLAM: published nifti-send %s HTTP %s",
        result_path.name,
        http_status,
    )


def _publish_result_dicom_sync(
    ctx: ServiceProviderContext,
    topic: str,
    result_path: Path,
    *,
    target_subscriber_name: Optional[str] = None,
) -> None:
    future = asyncio.run_coroutine_threadsafe(
        ctx.publish_dicom_send(
            topic,
            str(result_path),
            target_subscriber_name=target_subscriber_name,
        ),
        ctx.loop,
    )
    http_status = future.result(timeout=300.0)
    LOGGER.info(
        "VOX2SEGLAM: published dicom-send %s HTTP %s",
        result_path.name,
        http_status,
    )


def _publish_outputs_sync(
    ctx: ServiceProviderContext,
    topic: str,
    files: Sequence[Path],
    publish_as: str,
    *,
    target_subscriber_name: Optional[str] = None,
) -> None:
    for path in files:
        if publish_as == "dicom-send":
            _publish_result_dicom_sync(
                ctx,
                topic,
                path,
                target_subscriber_name=target_subscriber_name,
            )
        elif publish_as == "nifti-send":
            _publish_result_nifti_sync(
                ctx,
                topic,
                path,
                target_subscriber_name=target_subscriber_name,
            )
        else:
            raise ValueError(f"unsupported publish_as: {publish_as!r}")


def _require_job_routing(
    ctx: ServiceProviderContext,
    message: Dict[str, Any],
    input_dir: Path,
    label: str,
) -> Optional[str]:
    event = message.get("event") or {}
    topic = (event.get("hub.topic") or "").strip()
    LOGGER.info(
        "VOX2SEGLAM: received %s id=%s topic=%s",
        label,
        message.get("id", ""),
        topic,
    )
    ctx.write_directory_manifest(input_dir)
    if not topic:
        LOGGER.warning("VOX2SEGLAM: inbound %s missing hub.topic", label)
        return None
    if not str(message.get("subscriber.name") or "").strip():
        LOGGER.warning(
            "VOX2SEGLAM: inbound %s missing subscriber.name; "
            "cannot send status-update",
            label,
        )
        return None
    return topic


def _run_vox2seglam(
    nifti_path: Path,
    output_dir: Path,
    *,
    job_dir: Optional[Path] = None,
    on_status: Optional[Any] = None,
) -> Optional[Path]:
    """Inference hook — replaced by ``vox2seglam_docker.py`` for Docker runs."""
    del nifti_path, output_dir, job_dir, on_status
    raise RuntimeError(
        "Host-native Vox2SegLAM is not supported. "
        "Use vox2seglam/vox2seglam_docker.py after building "
        "websub/vox2seglam:cu128."
    )


def _handle_job(
    ctx: ServiceProviderContext,
    message: Dict[str, Any],
    *,
    topic: str,
    nifti_path: Path,
    input_dir: Path,
    inbound_label: str,
    ref_dicom_dir: Optional[Path] = None,
) -> None:
    """Shared inference / cache / publish path for nifti-send and dicom-send."""
    dicom_out = inbound_label == "dicom-send"
    target_subscriber = str(message.get("subscriber.name") or "").strip() or None
    job_dir = input_dir.parent
    output_dir = job_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    options = _inference_options(dicom_out=dicom_out)
    series_key = resolve_series_key(
        input_dir, message, primary_nifti=nifti_path
    )
    LOGGER.info(
        "VOX2SEGLAM: cache key kind=%s value=%s model=%s options=%s",
        series_key.kind,
        series_key.value,
        MODEL_ID,
        options,
    )

    hit = lookup_complete(PRODUCT_NAME, series_key, MODEL_ID, options)
    if hit is not None:
        _publish_to_requester(ctx, message, "Using cached result")
        LOGGER.info("VOX2SEGLAM: cache hit %s", hit.root)
        try:
            _publish_outputs_sync(
                ctx,
                topic,
                hit.output_files,
                hit.publish_as,
                target_subscriber_name=target_subscriber,
            )
        except Exception as exc:
            _publish_to_requester(
                ctx, message, f"ERROR: publish cached result failed: {exc}"
            )
            LOGGER.exception("VOX2SEGLAM: cached publish failed")
            return
        _publish_to_requester(ctx, message, FINAL_STATUS_LINE)
        LOGGER.info(
            "VOX2SEGLAM: job complete (cached) files=%s",
            [p.name for p in hit.output_files],
        )
        return

    def _status(text: str) -> None:
        _publish_to_requester(ctx, message, text[:180])

    try:
        result_path = _run_vox2seglam(
            nifti_path,
            output_dir,
            job_dir=job_dir,
            on_status=_status,
        )
    except Exception as exc:
        _publish_to_requester(
            ctx, message, f"ERROR: segmentation failed: {exc}"
        )
        LOGGER.exception("VOX2SEGLAM: inference failed")
        return

    if result_path is None:
        _publish_to_requester(
            ctx, message, "ERROR: segmentation produced no *_seg.nii.gz"
        )
        return

    publish_as = "dicom-send" if dicom_out else "nifti-send"
    # Match Hub publish format to inbound: DICOM in → SEG/SR only;
    # NIfTI in → seg NIfTI only. Intermediate *_seg.nii.gz stays on disk
    # for conversion when dicom_out.
    output_files: List[Path] = [] if dicom_out else [result_path]

    if dicom_out:
        ref = ref_dicom_dir or input_dir
        docker_out = job_dir / "docker_out"
        structures_tsv = docker_out / "structures.tsv"
        landmarks_csv = docker_out / "landmarks.csv"
        if not structures_tsv.is_file():
            LOGGER.warning(
                "VOX2SEGLAM: structures.tsv missing under %s; "
                "SEG SegmentLabels will fall back to label_N",
                docker_out,
            )
            structures_tsv_arg: Optional[Path] = None
        else:
            structures_tsv_arg = structures_tsv
        try:
            seg_dcm = run_nifti_to_dicom_seg(
                result_path,
                ref,
                job_dir,
                structures_tsv=structures_tsv_arg,
                algorithm_name="Vox2SegLAM",
                on_status=_status,
                log_path=job_dir / "docker_tools.log",
            )
            output_files.append(seg_dcm)
        except Exception as exc:
            _publish_to_requester(
                ctx, message, f"ERROR: DICOM SEG conversion failed: {exc}"
            )
            LOGGER.exception("VOX2SEGLAM: nifti-to-dicom-seg failed")
            return

        if landmarks_csv.is_file():
            try:
                sr_dcm = run_landmarks_to_dicom_sr(
                    landmarks_csv,
                    ref,
                    job_dir,
                    algorithm_name="Vox2SegLAM",
                    on_status=_status,
                    log_path=job_dir / "docker_tools.log",
                )
                output_files.append(sr_dcm)
            except Exception as exc:
                # Landmarks SR is additive; keep SEG on failure.
                _publish_to_requester(
                    ctx,
                    message,
                    f"WARNING: landmarks SR conversion failed: {exc}",
                )
                LOGGER.exception("VOX2SEGLAM: landmarks-to-dicom-sr failed")
        else:
            LOGGER.warning(
                "VOX2SEGLAM: landmarks.csv missing under %s; skipping SR",
                docker_out,
            )

    try:
        store_complete(
            PRODUCT_NAME,
            series_key,
            MODEL_ID,
            options,
            output_files=output_files,
            publish_as=publish_as,
        )
    except Exception:
        LOGGER.exception("VOX2SEGLAM: failed to store result cache")

    _publish_to_requester(ctx, message, "Publishing result…")
    try:
        _publish_outputs_sync(
            ctx,
            topic,
            output_files,
            publish_as,
            target_subscriber_name=target_subscriber,
        )
    except Exception as exc:
        _publish_to_requester(
            ctx, message, f"ERROR: publish result failed: {exc}"
        )
        LOGGER.exception("VOX2SEGLAM: publish failed")
        return

    _publish_to_requester(ctx, message, FINAL_STATUS_LINE)
    LOGGER.info(
        "VOX2SEGLAM: job complete files=%s",
        [p.name for p in output_files],
    )


def on_nifti_send(
    ctx: ServiceProviderContext,
    message: Dict[str, Any],
    input_dir: Path,
    file_count: int,
    total_bytes: int,
) -> None:
    del file_count, total_bytes
    topic = _require_job_routing(ctx, message, input_dir, "nifti-send")
    if topic is None:
        return

    _publish_to_requester(ctx, message, "Download complete.")
    nifti_path = _resolve_input_nifti(input_dir)
    if nifti_path is None:
        _publish_to_requester(ctx, message, "ERROR: no NIfTI file in input")
        return

    LOGGER.info("VOX2SEGLAM: input NIfTI %s", nifti_path)
    _handle_job(
        ctx,
        message,
        topic=topic,
        nifti_path=nifti_path,
        input_dir=input_dir,
        inbound_label="nifti-send",
    )


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
    job_dir = input_dir.parent

    def _status(text: str) -> None:
        _publish_to_requester(ctx, message, text[:180])

    try:
        nifti_path = run_dicom_to_nifti(
            input_dir,
            job_dir,
            on_status=_status,
            log_path=job_dir / "docker_tools.log",
        )
    except Exception as exc:
        _publish_to_requester(
            ctx, message, f"ERROR: DICOM to NIfTI failed: {exc}"
        )
        LOGGER.exception("VOX2SEGLAM: dicom-to-nifti failed")
        return

    LOGGER.info("VOX2SEGLAM: converted DICOM → NIfTI %s", nifti_path)
    _handle_job(
        ctx,
        message,
        topic=topic,
        nifti_path=nifti_path,
        input_dir=input_dir,
        inbound_label="dicom-send",
        ref_dicom_dir=input_dir,
    )


HANDLERS = ServiceProviderHandlers(
    on_dicom_send=on_dicom_send,
    on_nifti_send=on_nifti_send,
    on_send_download_start=on_send_download_start,
    on_try_cache_before_download=on_try_cache_before_download,
    build_status_response=build_status_response,
)


if __name__ == "__main__":
    run_sync(
        ServiceProviderConfig(
            product_name=PRODUCT_NAME,
            events=["dicom-send", "nifti-send", "status-request"],
        ),
        HANDLERS,
    )
