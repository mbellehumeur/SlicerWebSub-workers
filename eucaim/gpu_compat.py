"""Host GPU compatibility for EUCAIM Docker images.

Published Harbor EUCAIM task images use CUDA 11.8 / older PyTorch builds that
lack kernels for NVIDIA Blackwell (compute capability 12.0+, e.g. RTX
50-series). On those hosts the worker should skip ``--gpus`` and run on CPU
unless ``EUCAIM_FORCE_GPU`` is set.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Tuple

LOGGER = logging.getLogger("EUCAIM.gpu")

# Blackwell consumer GPUs (sm_120) and newer — harbor CUDA 11.8 images lack kernels.
MIN_INCOMPATIBLE_COMPUTE = (12, 0)


@dataclass(frozen=True)
class GpuInfo:
    name: str
    compute_cap: Tuple[int, int]

    @property
    def compute_cap_str(self) -> str:
        return f"{self.compute_cap[0]}.{self.compute_cap[1]}"


def _parse_compute_cap(raw: str) -> Optional[Tuple[int, int]]:
    text = (raw or "").strip()
    if not text:
        return None
    parts = text.split(".")
    if len(parts) != 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def _query_nvidia_smi() -> List[GpuInfo]:
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        LOGGER.warning("EUCAIM: nvidia-smi probe failed: %s", exc)
        return []
    if proc.returncode != 0:
        LOGGER.warning(
            "EUCAIM: nvidia-smi exited %s: %s",
            proc.returncode,
            (proc.stderr or proc.stdout or "").strip()[:200],
        )
        return []

    gpus: List[GpuInfo] = []
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        if "," not in line:
            continue
        name_part, cap_part = line.rsplit(",", 1)
        cap = _parse_compute_cap(cap_part)
        if cap is None:
            continue
        gpus.append(GpuInfo(name=name_part.strip(), compute_cap=cap))
    return gpus


@lru_cache(maxsize=1)
def list_host_gpus() -> Tuple[GpuInfo, ...]:
    """Cached ``nvidia-smi`` GPU list (empty if unavailable)."""
    return tuple(_query_nvidia_smi())


def eucaim_images_incompatible_gpus(
    gpus: Optional[Tuple[GpuInfo, ...]] = None,
) -> Tuple[GpuInfo, ...]:
    """GPUs whose compute capability exceeds what published EUCAIM images support."""
    if gpus is None:
        gpus = list_host_gpus()
    return tuple(
        g for g in gpus if g.compute_cap >= MIN_INCOMPATIBLE_COMPUTE
    )


def should_force_cpu_for_eucaim(*, image_tag: Optional[str] = None) -> bool:
    """True when host GPU is incompatible with published Harbor EUCAIM images.

    ``image_tag`` is accepted for API symmetry with MHUB; v1 has no cu128
    rebuild exemption (local ``websub/eucaim-*:1.0.0`` tags are still CUDA 11.8
    retags unless rebuilt separately).
    """
    del image_tag
    return bool(eucaim_images_incompatible_gpus())


def cpu_fallback_reason(*, image_tag: Optional[str] = None) -> str:
    """Short human-facing reason for status-update / logs."""
    del image_tag
    bad = eucaim_images_incompatible_gpus()
    if not bad:
        return ""
    parts = [f"{g.name} (sm_{g.compute_cap_str})" for g in bad]
    return (
        "GPU incompatible with published EUCAIM images ("
        + "; ".join(parts)
        + "); running on CPU"
    )
