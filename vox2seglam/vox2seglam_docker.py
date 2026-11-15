#!/usr/bin/env python3
"""VOX2SEGLAM Hub worker — Docker cu128 backend (Blackwell-friendly).

Same product as ``vox2seglam.py``, but runs inference in
``websub/vox2seglam:cu128``.

Build images once (from repo root)::

    docker build -t websub/vox2seglam:cu128 \\
      -f vox2seglam/docker/Dockerfile vox2seglam/docker
    docker build -t websub/tools:cpu \\
      -f tools/docker/Dockerfile tools/docker

Then::

    python vox2seglam/vox2seglam_docker.py
    python vox2seglam/vox2seglam_docker.py --local

Weights download on first container run into
``/var/lib/websub-workers/models/vox2seglam`` (``VOX2SEGLAM_ASSETS``).
DICOM in / SEG out uses the shared ``websub/tools:cpu`` sidecar.

Env:
  VOX2SEGLAM_DOCKER_IMAGE    — override image tag
  VOX2SEGLAM_DEVICE          — cuda | cpu
  VOX2SEGLAM_GPUS            — Docker --gpus value (default all)
  VOX2SEGLAM_DOCKER_SHM_SIZE — Docker --shm-size (default 8g)
  VOX2SEGLAM_TASK            — subcortical (default) | full-brain
  WEBSUB_TOOLS_DOCKER_IMAGE  — override tools sidecar tag

Point the Service Provider onMessage at this script for GPU Docker inference.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Optional

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

import vox2seglam as v2s  # noqa: E402
from docker_runner import run_vox2seglam_job  # noqa: E402
from worker import ServiceProviderConfig, run_sync  # noqa: E402


def _run_vox2seglam_docker(
    nifti_path: Path,
    output_dir: Path,
    *,
    job_dir: Optional[Path] = None,
    on_status: Optional[Callable[[str], None]] = None,
) -> Optional[Path]:
    return run_vox2seglam_job(
        nifti_path,
        output_dir,
        job_dir=job_dir or Path(output_dir).resolve().parent,
        on_status=on_status,
        log_path=(job_dir or Path(output_dir).resolve().parent)
        / "docker_vox2seglam.log",
    )


# Swap inference backend; keep download / publish / status from the host module.
v2s._run_vox2seglam = _run_vox2seglam_docker


if __name__ == "__main__":
    run_sync(
        ServiceProviderConfig(
            product_name=v2s.PRODUCT_NAME,
            events=["dicom-send", "nifti-send", "status-request"],
        ),
        v2s.HANDLERS,
    )
