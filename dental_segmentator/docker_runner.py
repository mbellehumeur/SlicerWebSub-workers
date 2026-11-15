"""Run Dental Segmentator via the local cu128 Docker image."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable, List, Optional

LOGGER = logging.getLogger("DENTAL_SEG.docker")
StatusFn = Callable[[str], None]

DEFAULT_IMAGE = "websub/dental_segmentator:cu128"
IMAGE_ENV = "DENTAL_SEG_DOCKER_IMAGE"
DEVICE_ENV = "DENTAL_SEG_DEVICE"
GPUS_ENV = "DENTAL_SEG_GPUS"
SHM_SIZE_ENV = "DENTAL_SEG_DOCKER_SHM_SIZE"
DEFAULT_GPUS = "all"
# nnU-Net/PyTorch multiprocessing needs more than Docker's default 64MB /dev/shm.
DEFAULT_SHM_SIZE = "8g"


def image_tag() -> str:
    return (os.environ.get(IMAGE_ENV) or DEFAULT_IMAGE).strip() or DEFAULT_IMAGE


def image_is_local(tag: str) -> bool:
    proc = subprocess.run(
        ["docker", "image", "inspect", tag],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return proc.returncode == 0


def ensure_image(*, on_status: Optional[StatusFn] = None) -> str:
    """Require ``websub/dental_segmentator:cu128`` (or ``DENTAL_SEG_DOCKER_IMAGE``)."""
    tag = image_tag()
    if image_is_local(tag):
        LOGGER.info("DENTAL_SEG docker image local: %s", tag)
        return tag
    hint = (
        f"Docker image {tag} not found. Build it once:\n"
        "  docker build -t websub/dental_segmentator:cu128 "
        "-f dental_segmentator/docker/Dockerfile dental_segmentator/docker"
    )
    LOGGER.error(hint)
    if on_status:
        on_status(f"ERROR: missing image {tag} — build dental_segmentator/docker")
    raise RuntimeError(hint)


def _resolve_gpus() -> str:
    """Docker ``--gpus`` value, or empty for CPU (omit flag)."""
    device = (os.environ.get(DEVICE_ENV) or "").strip().lower()
    if device == "cpu":
        return ""
    raw = (os.environ.get(GPUS_ENV) or "").strip()
    if raw.lower() in ("none", "off", "cpu", "false", "0", "no"):
        return ""
    if raw:
        return raw
    return DEFAULT_GPUS


def _prepare_io(nifti_path: Path, job_dir: Path) -> tuple[Path, Path]:
    docker_in = job_dir / "docker_in"
    docker_out = job_dir / "docker_out"
    if docker_in.exists():
        shutil.rmtree(docker_in)
    if docker_out.exists():
        shutil.rmtree(docker_out)
    docker_in.mkdir(parents=True)
    docker_out.mkdir(parents=True)
    dest = docker_in / nifti_path.name
    try:
        os.link(nifti_path, dest)
    except OSError:
        shutil.copy2(nifti_path, dest)
    return docker_in, docker_out


def _find_prediction_output(pred_dir: Path) -> Optional[Path]:
    preferred = pred_dir / "dental.nii.gz"
    if preferred.is_file():
        return preferred
    matches = sorted(pred_dir.glob("*.nii.gz")) + sorted(pred_dir.glob("*.nii"))
    return matches[0] if matches else None


def run_dental_docker_job(
    nifti_path: Path,
    job_dir: Path,
    models_dir: Path,
    *,
    on_status: Optional[StatusFn] = None,
    log_path: Optional[Path] = None,
) -> Path:
    """Run the dental container; return host path to the labelmap NIfTI."""
    tag = ensure_image(on_status=on_status)
    nifti_path = Path(nifti_path).resolve()
    job_dir = Path(job_dir).resolve()
    models_dir = Path(models_dir).resolve()
    models_dir.mkdir(parents=True, exist_ok=True)

    docker_in, docker_out = _prepare_io(nifti_path, job_dir)
    gpus = _resolve_gpus()
    device = "cpu" if not gpus else (
        (os.environ.get(DEVICE_ENV) or "cuda").strip().lower() or "cuda"
    )

    shm_size = (
        os.environ.get(SHM_SIZE_ENV) or DEFAULT_SHM_SIZE
    ).strip() or DEFAULT_SHM_SIZE

    command: List[str] = ["docker", "run", "--rm", "-t", "--shm-size", shm_size]
    if gpus:
        command += ["--gpus", gpus]
    command += [
        "-e",
        f"DENTAL_SEG_DEVICE={device}",
        "-e",
        "DENTAL_SEG_MODELS=/models",
        "-v",
        f"{docker_in}:/data/input:ro",
        "-v",
        f"{docker_out}:/data/output",
        "-v",
        f"{models_dir}:/models",
        tag,
    ]

    log_file = Path(log_path) if log_path else job_dir / "docker_dental.log"
    LOGGER.info(
        "DENTAL_SEG docker run image=%s gpus=%s models=%s log=%s",
        tag,
        gpus or "cpu",
        models_dir,
        log_file,
    )
    if on_status:
        on_status(
            f"Running DentalSegmentator (Docker {tag}, device={device})…"
        )

    with open(log_file, "w", encoding="utf-8") as log_fh:
        log_fh.write(" ".join(command) + "\n\n")
        log_fh.flush()
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdout is not None
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            text = line.rstrip()
            log_fh.write(line)
            log_fh.flush()
            if text:
                LOGGER.info("DENTAL_SEG.docker| %s", text)
                if on_status and len(text) <= 200 and (
                    text.startswith("Predicting")
                    or text.startswith("Processing")
                    or "Downloading weights" in text
                    or text.startswith("DENTAL_SEG.docker:")
                ):
                    on_status(text[:180])
        proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(
            f"docker run exited {proc.returncode} (see {log_file})"
        )

    result = _find_prediction_output(docker_out)
    if result is None:
        raise FileNotFoundError(
            f"No segmentation NIfTI under {docker_out} (see {log_file})"
        )
    return result.resolve()
