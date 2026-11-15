"""Host GPU compatibility for EUCAIM Docker images.

Upstream CUDA 11.8 / cu118 EUCAIM images lack kernels for NVIDIA Blackwell
(compute capability 12.0+, e.g. RTX 50-series). Local ``*:cu128`` rebuilds are
Blackwell-capable and skip CPU fallback (same pattern as MHUB).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Tuple

LOGGER = logging.getLogger("EUCAIM.gpu")

# Blackwell consumer GPUs (sm_120) and newer — need torch cu128.
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
    """GPUs whose compute capability exceeds what non-cu128 EUCAIM images support."""
    if gpus is None:
        gpus = list_host_gpus()
    return tuple(
        g for g in gpus if g.compute_cap >= MIN_INCOMPATIBLE_COMPUTE
    )


def is_cu128_image_tag(tag: Optional[str]) -> bool:
    """True for local Blackwell rebuilds (``*:cu128``), which skip CPU fallback."""
    text = (tag or "").strip().lower()
    return bool(text) and text.endswith(":cu128")


def should_force_cpu_for_eucaim(*, image_tag: Optional[str] = None) -> bool:
    """True when host GPU is incompatible with the image that will run.

    Local ``websub/eucaim-*:cu128`` rebuilds are Blackwell-capable — never force
    CPU for those tags.
    """
    if is_cu128_image_tag(image_tag):
        return False
    return bool(eucaim_images_incompatible_gpus())


def cpu_fallback_reason(*, image_tag: Optional[str] = None) -> str:
    """Short human-facing reason for status-update / logs."""
    if is_cu128_image_tag(image_tag):
        return ""
    bad = eucaim_images_incompatible_gpus()
    if not bad:
        return ""
    parts = [f"{g.name} (sm_{g.compute_cap_str})" for g in bad]
    return (
        "GPU incompatible with non-cu128 EUCAIM images ("
        + "; ".join(parts)
        + "); running on CPU"
    )
