#!/usr/bin/env python3
"""TORCHXRAYVISION Hub worker — Docker cu128 backend (Blackwell-friendly).

Same product as ``torchxrayvision.py``, but runs inference in
``websub/torchxrayvision:cu128`` instead of host torch/torchxrayvision.

Build the image once (from repo root)::

    docker build -t websub/torchxrayvision:cu128 \\
      -f torchxrayvision/docker/Dockerfile torchxrayvision/docker

Then::

    python torchxrayvision/torchxrayvision_docker.py
    python torchxrayvision/torchxrayvision_docker.py --local

Weights download on first container run into
``/var/lib/websub-workers/models/torchxrayvision`` (mounted at
``/root/.torchxrayvision``).

Env:
  TXRV_DOCKER_IMAGE  — override image tag (default websub/torchxrayvision:cu128)
  TXRV_DEVICE        — cuda | cpu
  TXRV_GPUS          — Docker --gpus value (default all; none/cpu to omit)
  TXRV_WEIGHTS       — pretrained weights id (default densenet121-res224-all)

Research-only: not for diagnostic or clinical decision making.

Point the Service Provider onMessage at this script for GPU Docker inference.
"""

from __future__ import annotations

import importlib.util
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

# Load the Hub worker helpers by path so we do not collide with the PyPI
# package also named ``torchxrayvision``.
_spec = importlib.util.spec_from_file_location(
    "txrv_worker", _SCRIPT_DIR / "torchxrayvision.py"
)
assert _spec is not None and _spec.loader is not None
txrv = importlib.util.module_from_spec(_spec)
sys.modules["txrv_worker"] = txrv
_spec.loader.exec_module(txrv)

from docker_runner import run_txrv_docker_job  # noqa: E402
from result_cache import models_dir  # noqa: E402
from worker import (  # noqa: E402
    ServiceProviderConfig,
    ServiceProviderHandlers,
    ServiceProviderContext,
    run_sync,
)

LOGGER = logging.getLogger("TORCHXRAYVISION.docker_worker")


def _run_docker_inference(
    image_path: Path,
    job_dir: Path,
    *,
    on_status=None,
) -> Path:
    models_cache = models_dir() / txrv.MODEL_ID
    return run_txrv_docker_job(
        image_path,
        job_dir,
        models_cache,
        on_status=on_status,
        log_path=job_dir / "docker_txrv.log",
    )


def on_dicom_send(
    ctx: ServiceProviderContext,
    message: Dict[str, Any],
    input_dir: Path,
    file_count: int,
    total_bytes: int,
) -> None:
    del file_count, total_bytes
    txrv._handle_dicom_job(
        ctx,
        message,
        input_dir,
        run_inference=_run_docker_inference,
    )


HANDLERS = ServiceProviderHandlers(
    on_dicom_send=on_dicom_send,
    on_nifti_send=txrv.on_nifti_send,
    on_send_download_start=txrv.on_send_download_start,
    on_try_cache_before_download=txrv.on_try_cache_before_download,
    build_status_response=txrv.build_status_response,
)


if __name__ == "__main__":
    run_sync(
        ServiceProviderConfig(product_name=txrv.PRODUCT_NAME),
        HANDLERS,
    )
