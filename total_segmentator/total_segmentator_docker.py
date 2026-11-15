#!/usr/bin/env python3
"""TOTALSEG Hub worker — Docker cu128 backend (Blackwell-friendly).

Same product as ``total_segmentator.py``, but runs inference in
``websub/totalsegmentator:cu128`` instead of host TotalSegmentator/torch.

Build the image once (from repo root)::

    docker build -t websub/totalsegmentator:cu128 \\
      -f total_segmentator/docker/Dockerfile total_segmentator/docker

Then::

    python total_segmentator/total_segmentator_docker.py
    python total_segmentator/total_segmentator_docker.py --local

Weights download on first container run into
``/var/lib/websub-workers/models/totalsegmentator`` (``TOTALSEG_HOME_DIR``).

Env:
  TOTALSEG_DOCKER_IMAGE      — override image tag
  TOTALSEG_DEVICE            — gpu | cpu (mapped to Docker --gpus)
  TOTALSEG_GPUS              — Docker --gpus value (default all)
  TOTALSEG_DOCKER_SHM_SIZE   — Docker --shm-size (default 8g)

Point the Service Provider onMessage at this script for GPU Docker inference.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

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

import total_segmentator as ts  # noqa: E402
from docker_runner import run_totalseg_job  # noqa: E402


def _run_totalsegmentator_docker(
    input_path: Path,
    output_dir: Path,
    hub_event: str,
    task: str = ts.TS_TASK,
    fast: bool = ts.TS_FAST,
) -> Optional[Path]:
    def _status(line: str) -> None:
        ts._status_log_line(line)

    return run_totalseg_job(
        input_path,
        output_dir,
        hub_event,
        task=task,
        fast=fast,
        job_dir=Path(output_dir).resolve().parent,
        on_status=_status,
        log_path=Path(output_dir).resolve().parent / "docker_totalseg.log",
    )


# Swap inference backend; keep download / publish / status from the host module.
ts._run_totalsegmentator = _run_totalsegmentator_docker


if __name__ == "__main__":
    from worker import ServiceProviderConfig, ServiceProviderHandlers, run_sync

    HANDLERS = ServiceProviderHandlers(
        on_dicom_send=ts.on_dicom_send,
        on_nifti_send=ts.on_nifti_send,
        on_send_download_start=ts.on_send_download_start,
        build_status_response=ts.build_status_response,
    )
    run_sync(
        ServiceProviderConfig(product_name=ts.DEFAULT_PRODUCT_NAME),
        HANDLERS,
    )
