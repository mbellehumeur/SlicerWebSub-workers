"""Run shared ``websub/tools:cpu`` conversion sidecars.

Workers call these helpers for DICOM ↔ NIfTI / DICOM-SEG without baking
conversion deps into product GPU images.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable, List, Optional, Sequence

LOGGER = logging.getLogger("websub.tools")
StatusFn = Callable[[str], None]

DEFAULT_IMAGE = "websub/tools:cpu"
IMAGE_ENV = "WEBSUB_TOOLS_DOCKER_IMAGE"
BUILD_HINT = (
    "docker build -t websub/tools:cpu "
    "-f tools/docker/Dockerfile tools/docker"
)


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


def ensure_tools_image(*, on_status: Optional[StatusFn] = None) -> str:
    tag = image_tag()
    if image_is_local(tag):
        LOGGER.info("websub tools image local: %s", tag)
        return tag
    hint = f"Docker image {tag} not found. Build it once:\n  {BUILD_HINT}"
    LOGGER.error(hint)
    if on_status:
        on_status(f"ERROR: missing image {tag} — build tools/docker")
    raise RuntimeError(hint)


def _hardlink_or_copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if dest.is_dir():
            shutil.rmtree(dest)
        else:
            dest.unlink()
    try:
        os_link = getattr(os, "link", None)
        if os_link is not None:
            os_link(src, dest)
            return
    except OSError:
        pass
    shutil.copy2(src, dest)


def _stage_tree(src_dir: Path, dest_dir: Path) -> None:
    """Hardlink/copy all files under ``src_dir`` into ``dest_dir`` (flat-ish)."""
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True)
    src_dir = Path(src_dir).resolve()
    for path in sorted(src_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(src_dir)
        target = dest_dir / rel
        _hardlink_or_copy(path, target)


def _docker_run(
    *,
    tag: str,
    docker_in: Path,
    docker_out: Path,
    args: Sequence[str],
    log_path: Path,
    on_status: Optional[StatusFn],
) -> None:
    command: List[str] = [
        "docker",
        "run",
        "--rm",
        "-t",
        "-v",
        f"{docker_in.resolve()}:/data/input:ro",
        "-v",
        f"{docker_out.resolve()}:/data/output",
        tag,
        *args,
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("websub tools docker: %s", " ".join(command))
    with open(log_path, "a", encoding="utf-8") as log_fh:
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
        chunks: List[str] = []
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            log_fh.write(line)
            log_fh.flush()
            text = line.rstrip()
            if text:
                chunks.append(text)
                LOGGER.info("websub.tools| %s", text)
                if on_status and len(text) <= 200:
                    on_status(text[:180])
        proc.wait()
        if proc.returncode != 0:
            tail = "\n".join(chunks[-40:])
            raise RuntimeError(
                f"tools docker exit {proc.returncode}\n{tail[-4000:]}"
            )


def run_dicom_to_nifti(
    input_dir: Path,
    job_dir: Path,
    *,
    on_status: Optional[StatusFn] = None,
    log_path: Optional[Path] = None,
) -> Path:
    """Convert a DICOM series dir to ``job_dir/volume.nii.gz``; return that path."""
    tag = ensure_tools_image(on_status=on_status)
    job_dir = Path(job_dir).resolve()
    input_dir = Path(input_dir).resolve()
    docker_in = job_dir / "tools_in_dicom"
    docker_out = job_dir / "tools_out_nifti"
    _stage_tree(input_dir, docker_in)
    if docker_out.exists():
        shutil.rmtree(docker_out)
    docker_out.mkdir(parents=True)

    if on_status:
        on_status("Converting DICOM to NIfTI…")
    _docker_run(
        tag=tag,
        docker_in=docker_in,
        docker_out=docker_out,
        args=["dicom-to-nifti"],
        log_path=Path(log_path) if log_path else job_dir / "docker_tools.log",
        on_status=on_status,
    )

    produced = docker_out / "volume.nii.gz"
    if not produced.is_file():
        # Accept uncompressed fallback
        alt = docker_out / "volume.nii"
        if alt.is_file():
            produced = alt
        else:
            raise RuntimeError(
                "tools dicom-to-nifti produced no volume.nii.gz"
            )

    dest = job_dir / produced.name
    if dest.exists():
        dest.unlink()
    shutil.copy2(produced, dest)
    if on_status:
        on_status(f"NIfTI ready: {dest.name}")
    return dest.resolve()


def run_nifti_to_dicom_seg(
    seg_nifti: Path,
    ref_dicom_dir: Path,
    job_dir: Path,
    *,
    on_status: Optional[StatusFn] = None,
    log_path: Optional[Path] = None,
) -> Path:
    """Convert a labelmap NIfTI + reference series to ``segmentation.seg.dcm``."""
    tag = ensure_tools_image(on_status=on_status)
    job_dir = Path(job_dir).resolve()
    seg_nifti = Path(seg_nifti).resolve()
    ref_dicom_dir = Path(ref_dicom_dir).resolve()
    if not seg_nifti.is_file():
        raise FileNotFoundError(f"seg NIfTI not found: {seg_nifti}")
    if not ref_dicom_dir.is_dir():
        raise FileNotFoundError(f"reference DICOM dir missing: {ref_dicom_dir}")

    docker_in = job_dir / "tools_in_seg"
    docker_out = job_dir / "tools_out_seg"
    if docker_in.exists():
        shutil.rmtree(docker_in)
    docker_in.mkdir(parents=True)
    _hardlink_or_copy(seg_nifti, docker_in / seg_nifti.name)
    _stage_tree(ref_dicom_dir, docker_in / "reference")
    if docker_out.exists():
        shutil.rmtree(docker_out)
    docker_out.mkdir(parents=True)

    if on_status:
        on_status("Converting NIfTI seg to DICOM SEG…")
    _docker_run(
        tag=tag,
        docker_in=docker_in,
        docker_out=docker_out,
        args=[
            "nifti-to-dicom-seg",
            "--seg",
            f"/data/input/{seg_nifti.name}",
            "--reference",
            "/data/input/reference",
        ],
        log_path=Path(log_path) if log_path else job_dir / "docker_tools.log",
        on_status=on_status,
    )

    produced = docker_out / "segmentation.seg.dcm"
    if not produced.is_file():
        # Any .dcm fallback
        dcms = sorted(docker_out.glob("*.dcm")) + sorted(
            docker_out.glob("*.dicom")
        )
        if not dcms:
            raise RuntimeError(
                "tools nifti-to-dicom-seg produced no segmentation.seg.dcm"
            )
        produced = dcms[0]

    dest = job_dir / "output" / produced.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    shutil.copy2(produced, dest)
    if on_status:
        on_status(f"DICOM SEG ready: {dest.name}")
    return dest.resolve()
