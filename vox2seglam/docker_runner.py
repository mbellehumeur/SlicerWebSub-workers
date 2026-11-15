"""Run Vox2SegLAM via the local cu128 Docker image."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable, List, Optional

LOGGER = logging.getLogger("VOX2SEGLAM.docker")
StatusFn = Callable[[str], None]

DEFAULT_IMAGE = "websub/vox2seglam:cu128"
IMAGE_ENV = "VOX2SEGLAM_DOCKER_IMAGE"
DEVICE_ENV = "VOX2SEGLAM_DEVICE"
GPUS_ENV = "VOX2SEGLAM_GPUS"
SHM_SIZE_ENV = "VOX2SEGLAM_DOCKER_SHM_SIZE"
TASK_ENV = "VOX2SEGLAM_TASK"
DEFAULT_GPUS = "all"
DEFAULT_SHM_SIZE = "8g"
DEFAULT_TASK = "subcortical"


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
    tag = image_tag()
    if image_is_local(tag):
        LOGGER.info("VOX2SEGLAM docker image local: %s", tag)
        return tag
    hint = (
        f"Docker image {tag} not found. Build it once:\n"
        "  docker build -t websub/vox2seglam:cu128 "
        "-f vox2seglam/docker/Dockerfile vox2seglam/docker"
    )
    LOGGER.error(hint)
    if on_status:
        on_status(f"ERROR: missing image {tag} — build vox2seglam/docker")
    raise RuntimeError(hint)


def _resolve_gpus() -> str:
    device = (os.environ.get(DEVICE_ENV) or "").strip().lower()
    if device in ("cpu",):
        return ""
    raw = (os.environ.get(GPUS_ENV) or "").strip()
    if raw.lower() in ("none", "off", "cpu", "false", "0", "no"):
        return ""
    if raw:
        return raw
    return DEFAULT_GPUS


def _cli_device(gpus: str) -> str:
    """Vox2SegLAM CLI uses ``cuda`` / ``cpu``."""
    if not gpus:
        return "cpu"
    raw = (os.environ.get(DEVICE_ENV) or "").strip().lower()
    if raw in ("cpu",):
        return "cpu"
    return "cuda"


def _task_name() -> str:
    return (os.environ.get(TASK_ENV) or DEFAULT_TASK).strip() or DEFAULT_TASK


def _prepare_io(nifti_path: Path, job_dir: Path) -> tuple[Path, Path, str]:
    """Return (host_in, host_out, container input path)."""
    docker_in = job_dir / "docker_in"
    docker_out = job_dir / "docker_out"
    if docker_in.exists():
        shutil.rmtree(docker_in)
    if docker_out.exists():
        shutil.rmtree(docker_out)
    docker_in.mkdir(parents=True)
    docker_out.mkdir(parents=True)

    nifti_path = Path(nifti_path).resolve()
    if not nifti_path.is_file():
        raise FileNotFoundError(f"Vox2SegLAM input not found: {nifti_path}")
    dest = docker_in / nifti_path.name
    try:
        os.link(nifti_path, dest)
    except OSError:
        shutil.copy2(nifti_path, dest)
    container_i = f"/data/input/{nifti_path.name}"
    return docker_in, docker_out, container_i


def _find_seg_nifti(pred_dir: Path) -> Optional[Path]:
    matches = sorted(pred_dir.glob("*_seg.nii.gz")) + sorted(
        pred_dir.glob("*_seg.nii")
    )
    if matches:
        return matches[0]
    # Fallback: any NIfTI that is not under prepared/
    candidates = [
        p
        for p in sorted(pred_dir.rglob("*.nii*"))
        if p.is_file() and "prepared" not in p.parts
    ]
    return candidates[0] if candidates else None


def _clear_dir(path: Path) -> None:
    if not path.is_dir():
        return
    for entry in path.iterdir():
        if entry.is_file():
            entry.unlink()
        elif entry.is_dir():
            shutil.rmtree(entry, ignore_errors=True)


def _docker_once(
    *,
    tag: str,
    docker_in: Path,
    docker_out: Path,
    models_dir: Path,
    container_i: str,
    device: str,
    task: str,
    gpus: str,
    shm_size: str,
    log_fh,
    on_status: Optional[StatusFn],
) -> str:
    """Run one container; return combined stdout. Raises RuntimeError on failure."""
    v2s_args: List[str] = [
        "predict",
        "--input",
        container_i,
        "--output-dir",
        "/data/output",
        "--task",
        task,
        "--postprocess",
        "--download",
        "--device",
        device,
    ]

    command: List[str] = ["docker", "run", "--rm", "-t", "--shm-size", shm_size]
    if gpus:
        command += ["--gpus", gpus]
    command += [
        "-e",
        "VOX2SEGLAM_ASSETS=/models",
        "-v",
        f"{docker_in}:/data/input:ro",
        "-v",
        f"{docker_out}:/data/output",
        "-v",
        f"{models_dir}:/models",
        tag,
        *v2s_args,
    ]

    log_fh.write(" ".join(command) + "\n\n")
    log_fh.flush()
    LOGGER.info("VOX2SEGLAM docker: %s", " ".join(command))

    chunks: List[str] = []
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
            chunks.append(text)
            LOGGER.info("VOX2SEGLAM.docker| %s", text)
            if on_status and len(text) <= 200:
                on_status(text[:180])
    proc.wait()
    output = "\n".join(chunks)
    if proc.returncode != 0:
        raise RuntimeError(
            f"docker exit {proc.returncode}\n{output[-4000:]}"
        )
    return output


def run_vox2seglam_job(
    nifti_path: Path,
    output_dir: Path,
    *,
    job_dir: Optional[Path] = None,
    models_dir: Optional[Path] = None,
    on_status: Optional[StatusFn] = None,
    log_path: Optional[Path] = None,
) -> Optional[Path]:
    """Run Vox2SegLAM in Docker; return path to ``*_seg.nii.gz`` or None."""
    tag = ensure_image(on_status=on_status)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    job_dir = Path(job_dir or output_dir.parent).resolve()
    if models_dir is None:
        from result_cache import models_dir as _models_root

        models_dir = _models_root() / "vox2seglam"
    models_dir = Path(models_dir).resolve()
    models_dir.mkdir(parents=True, exist_ok=True)

    docker_in, docker_out, container_i = _prepare_io(Path(nifti_path), job_dir)
    gpus = _resolve_gpus()
    shm_size = (
        os.environ.get(SHM_SIZE_ENV) or DEFAULT_SHM_SIZE
    ).strip() or DEFAULT_SHM_SIZE
    task = _task_name()
    log_file = Path(log_path) if log_path else job_dir / "docker_vox2seglam.log"

    preferred_device = _cli_device(gpus)
    devices = (
        [preferred_device]
        if preferred_device == "cpu"
        else [preferred_device, "cpu"]
    )

    with open(log_file, "w", encoding="utf-8") as log_fh:
        for device in devices:
            run_gpus = gpus if device == "cuda" else ""
            try:
                started = time.monotonic()
                if on_status:
                    on_status(
                        f"Running Vox2SegLAM (Docker {tag}, "
                        f"device={device}, task={task})…"
                    )
                _clear_dir(docker_out)
                _docker_once(
                    tag=tag,
                    docker_in=docker_in,
                    docker_out=docker_out,
                    models_dir=models_dir,
                    container_i=container_i,
                    device=device,
                    task=task,
                    gpus=run_gpus,
                    shm_size=shm_size,
                    log_fh=log_fh,
                    on_status=on_status,
                )
                elapsed = time.monotonic() - started
                if on_status:
                    on_status(
                        f"Segmentation finished ({device}, {elapsed:.0f}s)"
                    )
                result = _find_seg_nifti(docker_out)
                if result is None:
                    if on_status:
                        on_status(
                            "Segmentation failed: process exited without "
                            "*_seg.nii.gz"
                        )
                    continue
                dest = output_dir / result.name
                if result.resolve() != dest.resolve():
                    shutil.copy2(result, dest)
                return dest.resolve()
            except Exception as exc:
                LOGGER.exception("VOX2SEGLAM docker failed device=%s", device)
                if on_status:
                    on_status(f"Segmentation failed ({device}): {exc}")
                _clear_dir(docker_out)
    return None
