"""Run TotalSegmentator via the local cu128 Docker image."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable, List, Optional

LOGGER = logging.getLogger("TOTALSEG.docker")
StatusFn = Callable[[str], None]

DEFAULT_IMAGE = "websub/totalsegmentator:cu128"
IMAGE_ENV = "TOTALSEG_DOCKER_IMAGE"
DEVICE_ENV = "TOTALSEG_DEVICE"
GPUS_ENV = "TOTALSEG_GPUS"
SHM_SIZE_ENV = "TOTALSEG_DOCKER_SHM_SIZE"
DEFAULT_GPUS = "all"
DEFAULT_SHM_SIZE = "8g"
OUTPUT_DICOM_NAME = "segmentations.dcm"


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
        LOGGER.info("TOTALSEG docker image local: %s", tag)
        return tag
    hint = (
        f"Docker image {tag} not found. Build it once:\n"
        "  docker build -t websub/totalsegmentator:cu128 "
        "-f total_segmentator/docker/Dockerfile total_segmentator/docker"
    )
    LOGGER.error(hint)
    if on_status:
        on_status(f"ERROR: missing image {tag} — build total_segmentator/docker")
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


def _ts_device(gpus: str) -> str:
    """TotalSegmentator CLI uses ``gpu`` / ``cpu`` (not ``cuda``)."""
    if not gpus:
        return "cpu"
    raw = (os.environ.get(DEVICE_ENV) or "").strip().lower()
    if raw in ("cpu",):
        return "cpu"
    return "gpu"


def _prepare_mounts(
    input_path: Path, output_dir: Path, job_dir: Path
) -> tuple[Path, Path, str, str]:
    """Return (host_in, host_out, container_-i, container_-o)."""
    docker_in = job_dir / "docker_in"
    docker_out = job_dir / "docker_out"
    if docker_in.exists():
        shutil.rmtree(docker_in)
    if docker_out.exists():
        shutil.rmtree(docker_out)
    docker_in.mkdir(parents=True)
    docker_out.mkdir(parents=True)

    input_path = Path(input_path).resolve()
    if input_path.is_file():
        dest = docker_in / input_path.name
        try:
            os.link(input_path, dest)
        except OSError:
            shutil.copy2(input_path, dest)
        container_i = f"/data/input/{input_path.name}"
    elif input_path.is_dir():
        for src in sorted(input_path.iterdir()):
            if not src.is_file():
                continue
            dest = docker_in / src.name
            try:
                os.link(src, dest)
            except OSError:
                shutil.copy2(src, dest)
        container_i = "/data/input"
    else:
        raise FileNotFoundError(f"TotalSegmentator input not found: {input_path}")

    container_o = f"/data/output/{OUTPUT_DICOM_NAME}"
    return docker_in, docker_out, container_i, container_o


def _find_output_dicom(pred_dir: Path) -> Optional[Path]:
    primary = pred_dir / OUTPUT_DICOM_NAME
    if primary.is_file():
        return primary
    candidates = sorted(
        p
        for p in pred_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in (".dcm", ".dicom")
    )
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
    container_o: str,
    device: str,
    task: str,
    fast: bool,
    gpus: str,
    shm_size: str,
    log_fh,
    on_status: Optional[StatusFn],
) -> str:
    """Run one container; return combined stdout. Raises RuntimeError on failure."""
    ts_args: List[str] = [
        "-i",
        container_i,
        "-o",
        container_o,
        "--task",
        task,
        "-ot",
        "dicom_seg",
        "-d",
        device,
        "-nr",
        "1",
        "-ns",
        "1",
    ]
    if fast:
        ts_args.append("--fast")

    command: List[str] = ["docker", "run", "--rm", "-t", "--shm-size", shm_size]
    if gpus:
        command += ["--gpus", gpus]
    command += [
        "-e",
        "TOTALSEG_HOME_DIR=/models",
        "-v",
        f"{docker_in}:/data/input:ro",
        "-v",
        f"{docker_out}:/data/output",
        "-v",
        f"{models_dir}:/models",
        tag,
        *ts_args,
    ]

    log_fh.write(" ".join(command) + "\n\n")
    log_fh.flush()
    LOGGER.info("TOTALSEG docker: %s", " ".join(command))

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
            LOGGER.info("TOTALSEG.docker| %s", text)
            if on_status and len(text) <= 200:
                on_status(text[:180])
    proc.wait()
    output = "\n".join(chunks)
    if proc.returncode != 0:
        raise RuntimeError(
            f"docker exit {proc.returncode}\n{output[-4000:]}"
        )
    return output


def run_totalseg_job(
    input_path: Path,
    output_dir: Path,
    hub_event: str,
    task: str = "total",
    fast: bool = False,
    *,
    job_dir: Optional[Path] = None,
    models_dir: Optional[Path] = None,
    on_status: Optional[StatusFn] = None,
    log_path: Optional[Path] = None,
) -> Optional[Path]:
    """Docker-backed replacement for ``_run_totalsegmentator``.

    Signature matches the host helper plus optional mounts; returns SEG path or None.
    """
    del hub_event
    # Late import avoids circular import when total_segmentator_docker patches.
    import total_segmentator as ts  # noqa: WPS433

    tag = ensure_image(on_status=on_status)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    job_dir = Path(job_dir or output_dir.parent).resolve()
    if models_dir is None:
        from result_cache import models_dir as _models_root

        models_dir = _models_root() / "totalsegmentator"
    models_dir = Path(models_dir).resolve()
    models_dir.mkdir(parents=True, exist_ok=True)

    docker_in, docker_out, container_i, container_o = _prepare_mounts(
        Path(input_path), output_dir, job_dir
    )
    gpus = _resolve_gpus()
    shm_size = (
        os.environ.get(SHM_SIZE_ENV) or DEFAULT_SHM_SIZE
    ).strip() or DEFAULT_SHM_SIZE
    use_fast = bool(fast)
    log_file = Path(log_path) if log_path else job_dir / "docker_totalseg.log"

    preferred_device = _ts_device(gpus)
    devices = (
        [preferred_device]
        if preferred_device == "cpu"
        else [preferred_device, "cpu"]
    )

    with open(log_file, "w", encoding="utf-8") as log_fh:
        for device in devices:
            run_gpus = gpus if device == "gpu" else ""
            try:
                while True:
                    started = time.monotonic()
                    if on_status:
                        on_status(
                            f"Running TotalSegmentator (Docker {tag}, "
                            f"device={device})…"
                        )
                    _clear_dir(docker_out)
                    try:
                        _docker_once(
                            tag=tag,
                            docker_in=docker_in,
                            docker_out=docker_out,
                            models_dir=models_dir,
                            container_i=container_i,
                            container_o=container_o,
                            device=device,
                            task=task or "total",
                            fast=use_fast,
                            gpus=run_gpus,
                            shm_size=shm_size,
                            log_fh=log_fh,
                            on_status=on_status,
                        )
                    except RuntimeError as exc:
                        text = str(exc)
                        if "does not work with option --fast" in text:
                            if not use_fast:
                                raise
                            if on_status:
                                on_status(
                                    f"Task {task} does not support --fast; "
                                    "retrying without fast…"
                                )
                            use_fast = False
                            continue
                        if "No non-empty segments found to save" in text:
                            raise ts.TotalSegEmptySegmentsError(
                                "No non-empty segments found to save"
                            ) from exc
                        raise
                    break

                elapsed = time.monotonic() - started
                if on_status:
                    on_status(
                        f"Segmentation finished ({device}, {elapsed:.0f}s)"
                    )
                result = _find_output_dicom(docker_out)
                if result is None:
                    if on_status:
                        on_status(
                            "Segmentation failed: process exited without output"
                        )
                    continue
                # Copy into the job output_dir expected by the host worker.
                dest = output_dir / OUTPUT_DICOM_NAME
                if result.resolve() != dest.resolve():
                    shutil.copy2(result, dest)
                return dest.resolve()
            except ts.TotalSegEmptySegmentsError:
                if on_status:
                    on_status(
                        f"No segmentation to save for task={task} — the model "
                        "found no non-empty structures on this series "
                        "(try task=total, or a different study/FOV)."
                    )
                return None
            except Exception as exc:
                LOGGER.exception("TOTALSEG docker failed device=%s", device)
                if on_status:
                    on_status(f"Segmentation failed ({device}): {exc}")
                _clear_dir(docker_out)
    return None
