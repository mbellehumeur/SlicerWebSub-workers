"""Run EUCAIM nnU-Net task images via Docker."""

from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

LOGGER = logging.getLogger("EUCAIM.docker")
_OUTPUT_LIST_CAP = 100
StatusFn = Callable[[str], None]

_EUCAIM_DIR = Path(__file__).resolve().parent
_MODELS_JSON = _EUCAIM_DIR / "models.json"

DEFAULT_GPUS = "all"
LOCAL_IMAGE_TEMPLATE_ENV = "EUCAIM_LOCAL_IMAGE_TEMPLATE"
DEFAULT_LOCAL_IMAGE_TEMPLATE = "websub/eucaim-{model}:1.0.0"

INPUT_MOUNT = "/home/eucaim/nnUNet_input"
OUTPUT_MOUNT = "/home/eucaim/nnUNet_output"
START_SCRIPT = "/bin/bash"
START_ARGS = (
    "/app/start_nnunet.sh",
    "--input_directory",
    INPUT_MOUNT,
    "--output_directory",
    OUTPUT_MOUNT,
)


@lru_cache(maxsize=1)
def load_models_catalog() -> Dict[str, dict]:
    """Return model id → metadata from ``models.json``."""
    if not _MODELS_JSON.is_file():
        raise FileNotFoundError(f"missing catalog: {_MODELS_JSON}")
    data = json.loads(_MODELS_JSON.read_text(encoding="utf-8"))
    models = data.get("models") if isinstance(data, dict) else None
    if not isinstance(models, list):
        raise ValueError(f"invalid models.json: expected models array in {_MODELS_JSON}")
    out: Dict[str, dict] = {}
    for entry in models:
        if not isinstance(entry, dict):
            continue
        model_id = str(entry.get("id") or "").strip()
        image = str(entry.get("image") or "").strip()
        if not model_id or not image:
            continue
        out[model_id] = dict(entry)
    if not out:
        raise ValueError(f"no models found in {_MODELS_JSON}")
    return out


def canonicalize_model_id(model_name: str) -> str:
    """Map a requested model name to the catalog id (accepts ``-`` aliases)."""
    return str(validate_model(model_name)["id"]).strip()


def validate_model(model_name: str) -> dict:
    """Return model metadata or raise ValueError if unknown.

    Accepts catalog ids (``liver_tumour``) and hyphen aliases
    (``liver-tumour``) used by older client pickers.
    """
    name = (model_name or "").strip()
    if not name:
        raise ValueError("model name is empty")
    catalog = load_models_catalog()
    model = catalog.get(name)
    if model is None and "-" in name:
        model = catalog.get(name.replace("-", "_"))
    if not model:
        known = ", ".join(sorted(catalog))
        raise ValueError(f"unknown EUCAIM model: {name} (known: {known})")
    return model


def local_image_tag(model: str) -> str:
    """Local-only tag (``websub/eucaim-<model>:1.0.0`` by default)."""
    tmpl = (
        os.environ.get(LOCAL_IMAGE_TEMPLATE_ENV) or DEFAULT_LOCAL_IMAGE_TEMPLATE
    ).strip() or DEFAULT_LOCAL_IMAGE_TEMPLATE
    return tmpl.format(model=(model or "").strip())


def official_image_tag(model: str) -> str:
    meta = validate_model(model)
    return str(meta["image"]).strip()


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
    """Return local tag when present; else official harbor tag.

    Never pulls local tags — they are operator-built/retagged and local-only.
    """
    validate_model(model)
    local = local_image_tag(model)
    if image_is_local(local):
        return local
    return official_image_tag(model)


def ensure_image(
    model: str,
    *,
    on_status: Optional[StatusFn] = None,
) -> str:
    """Ensure a runnable image is local; prefer local tag, else pull official."""
    validate_model(model)
    local = local_image_tag(model)
    if image_is_local(local):
        LOGGER.info("EUCAIM using local image: %s", local)
        if on_status:
            on_status(f"Using local {local}")
        return local

    tag = official_image_tag(model)
    if image_is_local(tag):
        LOGGER.info("EUCAIM docker image already local: %s", tag)
        return tag

    msg = f"Pulling {tag}…"
    LOGGER.info("EUCAIM docker image missing; %s", msg)
    if on_status:
        on_status(msg)
    pull = subprocess.run(["docker", "pull", tag], check=False)
    if pull.returncode != 0:
        hint = (
            f"docker pull failed for {tag}. Build or retag locally "
            f"(see eucaim/docker/README.md):\n"
            f"  ./eucaim/docker/pull_or_build.sh {model}\n"
            f"  # or RETAG_LOCAL=1 after a successful pull"
        )
        raise RuntimeError(hint)
    if not image_is_local(tag):
        raise RuntimeError(f"docker pull finished but image still missing: {tag}")
    if on_status:
        on_status(f"Pulled {tag}")
    LOGGER.info("EUCAIM docker image ready: %s", tag)
    return tag


def _populate_series_folder(input_dir: Path, dest_dir: Path) -> int:
    """Copy/hardlink DICOM files into a single series folder under ``dest_dir``.

    EUCAIM expects folders of ``.dcm`` under the input mount. Flat worker
    downloads become ``dest_dir/series/<files>``.
    """
    series_dir = dest_dir / "series"
    series_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for src in sorted(input_dir.rglob("*")):
        if not src.is_file():
            continue
        dest = series_dir / src.name
        # Avoid collisions from nested trees with the same basename.
        if dest.exists() or dest.is_symlink():
            dest = series_dir / f"{src.parent.name}_{src.name}"
            if dest.exists() or dest.is_symlink():
                dest.unlink()
        try:
            os.link(src, dest)
        except OSError:
            shutil.copy2(src, dest)
        count += 1
    return count


def prepare_data_dir(input_dir: Path, data_dir: Path) -> Path:
    """Create EUCAIM host I/O tree; return ``output`` path.

    Layout::

        data_dir/
          input/series/   (DICOM files)
          output/         (container writes segmentations/ here)
    """
    input_dir = Path(input_dir).resolve()
    if not input_dir.is_dir():
        raise ValueError(f"input_dir does not exist: {input_dir}")

    data_dir = Path(data_dir)
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    input_root = data_dir / "input"
    output_root = data_dir / "output"
    output_root.mkdir(parents=True, exist_ok=True)

    n_files = _populate_series_folder(input_dir, input_root)
    if n_files <= 0:
        raise ValueError(f"no input files found under {input_dir}")
    LOGGER.info(
        "EUCAIM data dir ready: %s (input=%d files from %s, output=%s)",
        data_dir,
        n_files,
        input_dir,
        output_root,
    )
    return output_root


def _host_uid_gid() -> tuple[int, int]:
    return os.getuid(), os.getgid()


def _build_docker_command(
    model: str,
    data_dir: Path,
    *,
    image: Optional[str] = None,
    gpus: Optional[str] = None,
    network: str = "none",
) -> List[str]:
    validate_model(model)
    data_dir = Path(data_dir).resolve()
    input_root = data_dir / "input"
    output_root = data_dir / "output"
    if not input_root.is_dir():
        raise ValueError(f"missing input dir: {input_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    uid, gid = _host_uid_gid()
    command = ["docker", "run", "--rm", "-t"]
    if gpus:
        command += ["--gpus", gpus]
    if network:
        command += ["--network", network]

    command += [
        "-e",
        f"HOST_UID={uid}",
        "-e",
        f"HOST_GID={gid}",
        "-v",
        f"{input_root}:{INPUT_MOUNT}:ro",
        "-v",
        f"{output_root}:{OUTPUT_MOUNT}",
        image or resolve_image_tag(model),
        START_SCRIPT,
        *START_ARGS,
    ]
    return command


def _run_docker_logged(
    command: Sequence[str],
    log_path: Path,
) -> None:
    """Run docker, tee stdout/stderr to ``log_path`` and the logger."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd_text = shlex.join(list(command))
    LOGGER.info("EUCAIM docker command: %s", cmd_text)
    LOGGER.info("EUCAIM docker log file: %s", log_path)

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
                LOGGER.info("EUCAIM| %s", text)
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
    gpus: str = DEFAULT_GPUS,
    on_status: Optional[StatusFn] = None,
    log_path: Optional[Path] = None,
) -> Path:
    """Run EUCAIM DICOM inference; return ``output`` path under ``data_dir``."""
    tag = ensure_image(model, on_status=on_status)
    output_root = prepare_data_dir(input_dir, data_dir)
    log_file = (
        Path(log_path) if log_path else data_dir.parent / f"docker_{model}.log"
    )
    LOGGER.info(
        "EUCAIM docker dicom model=%s image=%s data_dir=%s gpus=%s",
        model,
        tag,
        data_dir,
        gpus or "cpu",
    )
    if on_status:
        on_status(f"Running {tag}…")
    command = _build_docker_command(
        model,
        data_dir,
        image=tag,
        gpus=gpus or None,
    )
    _run_docker_logged(command, log_file)
    return output_root


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
        log.info("EUCAIM %s: %s (missing or not a directory)", label, root)
        return
    files = [p for p in sorted(root.rglob("*")) if p.is_file()]
    log.info("EUCAIM %s: %s (%d file(s))", label, root, len(files))
    for path in files[: max(0, limit)]:
        try:
            size = path.stat().st_size
        except OSError:
            size = -1
        rel = path.relative_to(root).as_posix()
        log.info("EUCAIM %s file: %s (%d bytes)", label, rel, size)
    if len(files) > limit:
        log.info(
            "EUCAIM %s: … %d more file(s) not listed",
            label,
            len(files) - limit,
        )


def collect_outputs(output_dir: Path, publish_as: str = "dicom-send") -> List[Path]:
    """Collect publishable DICOM SEG artifacts under ``output_dir``.

    Prefers ``segmentations/`` then any top-level ``.dcm`` / ``.dicom``.
    """
    if publish_as != "dicom-send":
        raise ValueError(f"unsupported publish_as: {publish_as!r}")
    if not output_dir.is_dir():
        return []

    files = [p for p in sorted(output_dir.rglob("*")) if p.is_file()]
    segs: List[Path] = []
    other_dicom: List[Path] = []
    for path in files:
        name = path.name.lower()
        rel = str(path.relative_to(output_dir)).replace("\\", "/").lower()
        if "/logs/" in f"/{rel}/":
            continue
        is_dcm = name.endswith(".dcm") or name.endswith(".dicom")
        if not is_dcm and not (
            "seg" in rel and path.suffix == ""
        ):
            continue
        if "segmentations" in rel or rel.startswith("segmentations/"):
            segs.append(path)
        else:
            other_dicom.append(path)
    return segs or other_dicom
