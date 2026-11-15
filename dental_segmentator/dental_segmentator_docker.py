#!/usr/bin/env python3
"""DENTAL_SEG Hub worker — Docker cu128 backend (Blackwell-friendly).

Same product as ``dental_segmentator.py``, but runs inference in
``websub/dental_segmentator:cu128`` instead of host nnU-Net/torch.

Build the image once (from repo root)::

    docker build -t websub/dental_segmentator:cu128 \\
      -f dental_segmentator/docker/Dockerfile dental_segmentator/docker

Then::

    python dental_segmentator/dental_segmentator_docker.py
    python dental_segmentator/dental_segmentator_docker.py --local

Weights download on first container run into
``/var/lib/websub-workers/models/dental_segmentator`` (mounted at ``/models``).

Env:
  DENTAL_SEG_DOCKER_IMAGE  — override image tag (default websub/dental_segmentator:cu128)
  DENTAL_SEG_DEVICE        — cuda | cpu
  DENTAL_SEG_GPUS          — Docker --gpus value (default all; none/cpu to omit)
  DENTAL_SEG_DOCKER_SHM_SIZE — Docker --shm-size (default 8g; required for nnU-Net)
  DENTAL_SEG_MODEL_PATH    — unused for download path; models always under models_dir

Point the Service Provider onMessage at this script for GPU Docker inference.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict

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

import dental_segmentator as ds  # noqa: E402
from docker_runner import run_dental_docker_job  # noqa: E402
from result_cache import (  # noqa: E402
    lookup_complete,
    models_dir,
    resolve_series_key,
    store_complete,
)
from worker import (  # noqa: E402
    ServiceProviderConfig,
    ServiceProviderContext,
    ServiceProviderHandlers,
    run_sync,
)

LOGGER = logging.getLogger("DENTAL_SEG.docker_worker")


def on_nifti_send(
    ctx: ServiceProviderContext,
    message: Dict[str, Any],
    input_dir: Path,
    file_count: int,
    total_bytes: int,
) -> None:
    del file_count, total_bytes
    topic = ds._require_job_routing(ctx, message, input_dir, "nifti-send")
    if topic is None:
        return

    ds._publish_to_requester(ctx, message, "Download complete.")
    nifti_path = ds._resolve_input_nifti(input_dir)
    if nifti_path is None:
        ds._publish_to_requester(ctx, message, "ERROR: no NIfTI file in input")
        return

    LOGGER.info("DENTAL_SEG.docker: input NIfTI %s", nifti_path)
    job_dir = input_dir.parent
    options = ds._inference_options()
    series_key = resolve_series_key(
        input_dir, message, primary_nifti=nifti_path
    )
    LOGGER.info(
        "DENTAL_SEG.docker: cache key kind=%s value=%s model=%s",
        series_key.kind,
        series_key.value,
        ds.MODEL_ID,
    )

    hit = lookup_complete(
        ds.PRODUCT_NAME, series_key, ds.MODEL_ID, options
    )
    if hit is not None:
        result_path = hit.output_files[0]
        ds._publish_to_requester(ctx, message, "Using cached result")
        LOGGER.info("DENTAL_SEG.docker: cache hit %s", hit.root)
        try:
            ds._publish_result_nifti_sync(ctx, topic, result_path)
        except Exception as exc:
            ds._publish_to_requester(
                ctx, message, f"ERROR: publish nifti-send failed: {exc}"
            )
            LOGGER.exception("DENTAL_SEG.docker: cached publish failed")
            return
        ds._publish_to_requester(ctx, message, ds.FINAL_STATUS_LINE)
        LOGGER.info(
            "DENTAL_SEG.docker: job complete (cached) result=%s", result_path
        )
        return

    def _status(text: str) -> None:
        ds._publish_to_requester(ctx, message, text[:180])

    models_cache = models_dir() / ds.MODEL_ID
    try:
        result_path = run_dental_docker_job(
            nifti_path,
            job_dir,
            models_cache,
            on_status=_status,
            log_path=job_dir / "docker_dental.log",
        )
    except FileNotFoundError as exc:
        ds._publish_to_requester(ctx, message, f"ERROR: {exc}")
        LOGGER.exception("DENTAL_SEG.docker: model/output error")
        return
    except Exception as exc:
        ds._publish_to_requester(
            ctx, message, f"ERROR: Docker segmentation failed: {exc}"
        )
        LOGGER.exception("DENTAL_SEG.docker: inference failed")
        return

    try:
        store_complete(
            ds.PRODUCT_NAME,
            series_key,
            ds.MODEL_ID,
            options,
            output_files=[result_path],
            publish_as="nifti-send",
        )
    except Exception:
        LOGGER.exception("DENTAL_SEG.docker: failed to store result cache")

    ds._publish_to_requester(ctx, message, "Publishing result…")
    try:
        ds._publish_result_nifti_sync(ctx, topic, result_path)
    except Exception as exc:
        ds._publish_to_requester(
            ctx, message, f"ERROR: publish nifti-send failed: {exc}"
        )
        LOGGER.exception("DENTAL_SEG.docker: publish failed")
        return

    ds._publish_to_requester(ctx, message, ds.FINAL_STATUS_LINE)
    LOGGER.info("DENTAL_SEG.docker: job complete result=%s", result_path)


HANDLERS = ServiceProviderHandlers(
    on_dicom_send=ds.on_dicom_send,
    on_nifti_send=on_nifti_send,
    on_send_download_start=ds.on_send_download_start,
    build_status_response=ds.build_status_response,
)


if __name__ == "__main__":
    run_sync(
        ServiceProviderConfig(product_name=ds.PRODUCT_NAME),
        HANDLERS,
    )
