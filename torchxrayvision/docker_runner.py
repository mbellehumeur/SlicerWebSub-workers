"""Run TorchXRayVision via the local cu128 Docker image."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable, List, Optional

LOGGER = logging.getLogger("TORCHXRAYVISION.docker")
StatusFn = Callable[[str], None]

DEFAULT_IMAGE = "websub/torchxrayvision:cu128"
IMAGE_ENV = "TXRV_DOCKER_IMAGE"
DEVICE_ENV = "TXRV_DEVICE"
GPUS_ENV = "TXRV_GPUS"
WEIGHTS_ENV = "TXRV_WEIGHTS"
DEFAULT_GPUS = "all"
DEFAULT_WEIGHTS = "densenet121-res224-all"


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
    """Require ``websub/torchxrayvision:cu128`` (or ``TXRV_DOCKER_IMAGE``)."""
    tag = image_tag()
    if image_is_local(tag):
        LOGGER.info("TXRV docker image local: %s", tag)
        return tag
    hint = (
        f"Docker image {tag} not found. Build it once:\n"
        "  docker build -t websub/torchxrayvision:cu128 "
        "-f torchxrayvision/docker/Dockerfile torchxrayvision/docker"
    )
    LOGGER.error(hint)
    if on_status:
        on_status(f"ERROR: missing image {tag} — build torchxrayvision/docker")
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


def _weights() -> str:
    return (
        os.environ.get(WEIGHTS_ENV) or DEFAULT_WEIGHTS
    ).strip() or DEFAULT_WEIGHTS


def _hardlink_or_copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dest)
    except OSError:
        shutil.copy2(src, dest)


def _prepare_io(image_path: Path, job_dir: Path) -> tuple[Path, Path]:
    docker_in = job_dir / "docker_in"
    docker_out = job_dir / "docker_out"
    if docker_in.exists():
        shutil.rmtree(docker_in)
    if docker_out.exists():
        shutil.rmtree(docker_out)
    docker_in.mkdir(parents=True)
    docker_out.mkdir(parents=True)
    _hardlink_or_copy(image_path, docker_in / image_path.name)
    return docker_in, docker_out


def _find_preds_json(out_dir: Path) -> Optional[Path]:
    preferred = out_dir / "preds.json"
    if preferred.is_file():
        return preferred
    matches = sorted(out_dir.glob("*.json"))
    return matches[0] if matches else None


def run_txrv_docker_job(
    image_path: Path,
    job_dir: Path,
    models_dir: Path,
    *,
    on_status: Optional[StatusFn] = None,
    log_path: Optional[Path] = None,
) -> Path:
    """Run the TXRV container; return host path to ``preds.json``."""
    tag = ensure_image(on_status=on_status)
    image_path = Path(image_path).resolve()
    job_dir = Path(job_dir).resolve()
    models_dir = Path(models_dir).resolve()
    models_dir.mkdir(parents=True, exist_ok=True)

    docker_in, docker_out = _prepare_io(image_path, job_dir)
    gpus = _resolve_gpus()
    device = "cpu" if not gpus else (
        (os.environ.get(DEVICE_ENV) or "cuda").strip().lower() or "cuda"
    )
    weights = _weights()

    command: List[str] = ["docker", "run", "--rm", "-t"]
    if gpus:
        command += ["--gpus", gpus]
    command += [
        "-e",
        f"TXRV_DEVICE={device}",
        "-e",
        f"TXRV_WEIGHTS={weights}",
        "-v",
        f"{docker_in}:/data/input:ro",
        "-v",
        f"{docker_out}:/data/output",
        "-v",
        f"{models_dir}:/root/.torchxrayvision",
        tag,
    ]

    log_file = Path(log_path) if log_path else job_dir / "docker_txrv.log"
    LOGGER.info(
        "TXRV docker run image=%s gpus=%s weights=%s models=%s log=%s",
        tag,
        gpus or "cpu",
        weights,
        models_dir,
        log_file,
    )
    if on_status:
        on_status(f"Running TorchXRayVision (Docker {tag}, device={device})…")

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
                LOGGER.info("TXRV.docker| %s", text)
                if on_status and len(text) <= 200 and (
                    text.startswith("TXRV.docker:")
                    or "Downloading" in text
                ):
                    on_status(text[:180])
        proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(
            f"docker run exited {proc.returncode} (see {log_file})"
        )

    result = _find_preds_json(docker_out)
    if result is None:
        raise FileNotFoundError(
            f"No preds.json under {docker_out} (see {log_file})"
        )
    return result.resolve()
