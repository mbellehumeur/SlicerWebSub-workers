"""Run MHub models via Docker using the vendored MHubSkill helper."""

from __future__ import annotations

import logging
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, List, Optional, Sequence

LOGGER = logging.getLogger("MHUB.docker")
_OUTPUT_LIST_CAP = 100
StatusFn = Callable[[str], None]

_MHUB_DIR = Path(__file__).resolve().parent
_HELPER_SCRIPTS = _MHUB_DIR / "MHubSkill" / "scripts"
if _HELPER_SCRIPTS.is_dir() and str(_HELPER_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_HELPER_SCRIPTS))

from mhub_helper import (  # noqa: E402
    generate_nifti_config,
    get_model,
)

DEFAULT_GPUS = "all"
# Local Blackwell rebuilds (never published). Prefer when present on the host.
CU128_IMAGE_TEMPLATE_ENV = "MHUB_CU128_IMAGE_TEMPLATE"
DEFAULT_CU128_IMAGE_TEMPLATE = "websub/mhub-{model}:cu128"
OFFICIAL_IMAGE_TEMPLATE = "mhubai/{model}:latest"


def validate_model(model_name: str) -> dict:
    """Return model metadata or raise ValueError if unknown."""
    name = (model_name or "").strip()
    if not name:
        raise ValueError("model name is empty")
    model = get_model(name)
    if not model:
        raise ValueError(f"unknown MHub model: {name}")
    return model


def cu128_image_tag(model: str) -> str:
    """Local-only cu128 tag (``websub/mhub-<model>:cu128`` by default)."""
    tmpl = (
        os.environ.get(CU128_IMAGE_TEMPLATE_ENV) or DEFAULT_CU128_IMAGE_TEMPLATE
    ).strip() or DEFAULT_CU128_IMAGE_TEMPLATE
    return tmpl.format(model=(model or "").strip())


def official_image_tag(model: str) -> str:
    return OFFICIAL_IMAGE_TEMPLATE.format(model=(model or "").strip())


def image_tag(model: str) -> str:
    """Preferred runnable tag: local cu128 if present, else official latest."""
    return resolve_image_tag(model)


def is_cu128_image_tag(tag: str) -> bool:
    text = (tag or "").strip().lower()
    return text.endswith(":cu128") or "/mhub-" in text and ":cu128" in text


def image_is_local(tag: str) -> bool:
    """True if ``docker image inspect`` finds the tag locally."""
    proc = subprocess.run(
        ["docker", "image", "inspect", tag],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return proc.returncode == 0


def resolve_image_tag(model: str) -> str:
    """Return cu128 tag when that image exists locally; else official tag.

    Never pulls or pushes cu128 images — they are operator-built and local-only.
    """
    validate_model(model)
    local = cu128_image_tag(model)
    if image_is_local(local):
        return local
    return official_image_tag(model)


def ensure_image(
    model: str,
    *,
    on_status: Optional[StatusFn] = None,
) -> str:
    """Ensure a runnable image is local; prefer local cu128, else pull official.

    Calls ``on_status`` with short user-facing lines when pulling.
    """
    validate_model(model)
    local = cu128_image_tag(model)
    if image_is_local(local):
        LOGGER.info("MHUB using local cu128 image: %s", local)
        if on_status:
            on_status(f"Using local {local}")
        return local

    tag = official_image_tag(model)
    if image_is_local(tag):
        LOGGER.info("MHUB docker image already local: %s", tag)
        return tag

    msg = f"Pulling {tag}…"
    LOGGER.info("MHUB docker image missing; %s", msg)
    if on_status:
        on_status(msg)
    pull = subprocess.run(
        ["docker", "pull", tag],
        check=False,
    )
    if pull.returncode != 0:
        hint = (
            f"docker pull failed for {tag}. For Blackwell GPU, build a local "
            f"image instead (do not push):\n"
            f"  docker build -t websub/mhub-base:cu128 "
            f"-f mhub/docker/base/Dockerfile mhub/docker/base\n"
            f"  ./mhub/docker/build_model_cu128.sh {model}"
        )
        raise RuntimeError(hint)
    if not image_is_local(tag):
        raise RuntimeError(f"docker pull finished but image still missing: {tag}")
    if on_status:
        on_status(f"Pulled {tag}")
    LOGGER.info("MHUB docker image ready: %s", tag)
    return tag


def _populate_input_data(input_dir: Path, dest_dir: Path) -> int:
    """Fill ``dest_dir`` with real files (hardlink when possible, else copy).

    Host-absolute symlinks break inside the container (DicomImporter sees a
    missing ``/app/data/input_data``), so we never symlink the directory.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for src in sorted(input_dir.iterdir()):
        if not src.is_file():
            continue
        dest = dest_dir / src.name
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        try:
            os.link(src, dest)
        except OSError:
            shutil.copy2(src, dest)
        count += 1
    return count


def prepare_data_dir(input_dir: Path, data_dir: Path) -> Path:
    """Create an MHub ``/app/data`` tree on the host.

    Layout::

        data_dir/
          input_data/    (real files: hardlink/copy from input_dir)
          output_data/   (created empty)

    Returns the ``output_data`` path (publishable artifacts land here).
    """
    input_dir = Path(input_dir).resolve()
    if not input_dir.is_dir():
        raise ValueError(f"input_dir does not exist: {input_dir}")

    data_dir = Path(data_dir)
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    input_data = data_dir / "input_data"
    output_data = data_dir / "output_data"
    output_data.mkdir(parents=True, exist_ok=True)

    n_files = _populate_input_data(input_dir, input_data)
    if n_files <= 0:
        raise ValueError(f"no input files found under {input_dir}")
    LOGGER.info(
        "MHUB data dir ready: %s (input_data=%d files from %s, output_data=%s)",
        data_dir,
        n_files,
        input_dir,
        output_data,
    )
    return output_data


def _build_docker_command(
    model: str,
    data_dir: Path,
    *,
    image: Optional[str] = None,
    workflow: Optional[str] = None,
    config_file: Optional[Path] = None,
    gpus: Optional[str] = None,
    network: str = "none",
) -> List[str]:
    validate_model(model)
    data_dir = Path(data_dir).resolve()
    command = ["docker", "run", "--rm", "-t"]
    if gpus:
        command += ["--gpus", gpus]
    if network:
        command += ["--network", network]

    # Single mount so intermediates (sorted_data, nifti, …) persist on the host.
    command += ["-v", f"{data_dir}:/app/data"]

    config_mount_name = None
    if config_file:
        config_path = Path(config_file).expanduser().resolve()
        if not config_path.is_file():
            raise ValueError(f"Config file not found: {config_path}")
        config_mount_name = config_path.name
        command += ["-v", f"{config_path}:/app/config/{config_mount_name}:ro"]

    command.append(image or resolve_image_tag(model))

    if workflow:
        command += ["--workflow", workflow]
    elif config_mount_name:
        command += ["--config", f"/app/config/{config_mount_name}"]

    return command


def _run_docker_logged(
    command: Sequence[str],
    log_path: Path,
) -> None:
    """Run docker, tee stdout/stderr to ``log_path`` and the logger."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd_text = shlex.join(list(command))
    LOGGER.info("MHUB docker command: %s", cmd_text)
    LOGGER.info("MHUB docker log file: %s", log_path)

    with log_path.open("w", encoding="utf-8") as log_fh:
        log_fh.write(f"$ {cmd_text}\n\n")
        log_fh.flush()
        proc = subprocess.Popen(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log_fh.write(line)
            log_fh.flush()
            text = line.rstrip("\n")
            if text:
                LOGGER.info("MHUB| %s", text)
                print(text, flush=True)
        code = proc.wait()

    if code != 0:
        raise RuntimeError(
            f"docker run failed (exit {code}); see log {log_path}"
        )


def run_dicom_job(
    model: str,
    input_dir: Path,
    data_dir: Path,
    *,
    workflow: Optional[str] = None,
    gpus: str = DEFAULT_GPUS,
    on_status: Optional[StatusFn] = None,
    log_path: Optional[Path] = None,
) -> Path:
    """Run default DICOM workflow; return ``output_data`` path under ``data_dir``."""
    tag = ensure_image(model, on_status=on_status)
    output_data = prepare_data_dir(input_dir, data_dir)
    log_file = Path(log_path) if log_path else data_dir.parent / f"docker_{model}.log"
    LOGGER.info(
        "MHUB docker dicom model=%s image=%s data_dir=%s workflow=%s gpus=%s",
        model,
        tag,
        data_dir,
        workflow or "(default)",
        gpus,
    )
    if on_status:
        on_status(f"Running {tag}…")
    command = _build_docker_command(
        model,
        data_dir,
        image=tag,
        workflow=workflow or None,
        config_file=None,
        gpus=gpus or None,
    )
    _run_docker_logged(command, log_file)
    return output_data


def run_nifti_job(
    model: str,
    input_dir: Path,
    data_dir: Path,
    *,
    modality: str = "ct",
    gpus: str = DEFAULT_GPUS,
    config_path: Optional[Path] = None,
    on_status: Optional[StatusFn] = None,
    log_path: Optional[Path] = None,
) -> Path:
    """Run NIfTI workflow with generated config; return ``output_data`` path."""
    tag = ensure_image(model, on_status=on_status)
    output_data = prepare_data_dir(input_dir, data_dir)
    cfg_path = Path(config_path) if config_path else data_dir.parent / f"nifti_workflow_{model}.yml"
    yaml_text = generate_nifti_config(
        model,
        file_pattern="flat",
        modality=(modality or "ct").lower(),
        output_format="organized",
    )
    cfg_path.write_text(yaml_text, encoding="utf-8")
    log_file = Path(log_path) if log_path else data_dir.parent / f"docker_{model}.log"
    LOGGER.info(
        "MHUB docker nifti model=%s image=%s data_dir=%s config=%s gpus=%s",
        model,
        tag,
        data_dir,
        cfg_path,
        gpus,
    )
    if on_status:
        on_status(f"Running {tag}…")
    command = _build_docker_command(
        model,
        data_dir,
        image=tag,
        workflow=None,
        config_file=cfg_path,
        gpus=gpus or None,
    )
    _run_docker_logged(command, log_file)
    return output_data


def _is_nifti(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def _is_dicom(path: Path) -> bool:
    name = path.name.lower()
    if name.endswith(".dcm") or name.endswith(".dicom"):
        return True
    # DICOM-SEG often has no extension; skip obvious non-DICOM
    if path.suffix.lower() in {".nii", ".gz", ".yml", ".yaml", ".json", ".txt", ".log", ".csv"}:
        return False
    if name.endswith(".nii.gz"):
        return False
    return path.suffix == "" or name.startswith("seg") or "seg" in name


def log_output_tree(
    output_dir: Path,
    *,
    logger: Optional[logging.Logger] = None,
    limit: int = _OUTPUT_LIST_CAP,
    label: str = "output tree",
) -> None:
    """Log relative paths and sizes under ``output_dir`` (capped listing)."""
    log = logger or LOGGER
    root = Path(output_dir)
    if not root.is_dir():
        log.info("MHUB %s: %s (missing or not a directory)", label, root)
        return
    files = [p for p in sorted(root.rglob("*")) if p.is_file()]
    log.info(
        "MHUB %s: %s (%d file(s))",
        label,
        root,
        len(files),
    )
    for path in files[: max(0, limit)]:
        try:
            size = path.stat().st_size
        except OSError:
            size = -1
        rel = path.relative_to(root).as_posix()
        log.info("MHUB %s file: %s (%d bytes)", label, rel, size)
    if len(files) > limit:
        log.info(
            "MHUB %s: … %d more file(s) not listed",
            label,
            len(files) - limit,
        )


def collect_outputs(output_dir: Path, publish_as: str) -> List[Path]:
    """Collect publishable artifacts under ``output_dir``.

    Prefer segmentation products over logs / echoed input copies.
    Returns an empty list when nothing matches (caller decides status).
    """
    if not output_dir.is_dir():
        return []

    files = [p for p in sorted(output_dir.rglob("*")) if p.is_file()]
    if publish_as == "nifti-send":
        segs: List[Path] = []
        combined: List[Path] = []
        other_nifti: List[Path] = []
        for path in files:
            if not _is_nifti(path):
                continue
            rel = str(path.relative_to(output_dir)).replace("\\", "/").lower()
            if "/input/" in f"/{rel}/" or rel.startswith("input/"):
                continue
            if "/logs/" in f"/{rel}/" or rel.startswith("logs/"):
                continue
            if "segmentations" in rel or "/seg/" in f"/{rel}/":
                segs.append(path)
            elif "combined" in rel:
                combined.append(path)
            else:
                other_nifti.append(path)
        chosen = segs or combined or other_nifti
        return chosen

    if publish_as == "dicom-send":
        dicoms: List[Path] = []
        for path in files:
            name = path.name.lower()
            rel = str(path.relative_to(output_dir)).replace("\\", "/").lower()
            if "/logs/" in f"/{rel}/":
                continue
            if name.endswith(".dcm") or name.endswith(".dicom"):
                dicoms.append(path)
            elif "seg" in rel and path.suffix == "":
                dicoms.append(path)
        return dicoms

    raise ValueError(f"unsupported publish_as: {publish_as!r}")
