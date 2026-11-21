#!/usr/bin/env python3
"""TORCHXRAYVISION Hub service provider — chest X-ray classification.

Run from repo root (plain Python, no 3D Slicer):

    pip install aiohttp torchxrayvision pydicom
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
    python torchxrayvision/torchxrayvision.py
    python torchxrayvision/torchxrayvision.py --local

Prefer the Docker worker for GPU hosts (Blackwell / cu128)::

    python torchxrayvision/torchxrayvision_docker.py --local

On inbound dicom-send: download chest radiograph(s), run TorchXRayVision
DenseNet classification, publish pathology scores via status-update
(``RESULT: {…}``) and cache ``preds.json``.

On inbound nifti-send: status error (chest X-ray DICOM / PNG / JPG only).

Research-only upstream (not for clinical use):
https://github.com/mlmed/torchxrayvision

See ``torchxrayvision/README.md``.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

from result_cache import (  # noqa: E402
    lookup_complete,
    models_dir,
    resolve_series_key,
    series_key_from_message,
    store_complete,
)
from worker import (  # noqa: E402
    ServiceProviderConfig,
    ServiceProviderContext,
    ServiceProviderHandlers,
    run_sync,
)

LOGGER = logging.getLogger("TORCHXRAYVISION")

PRODUCT_NAME = "TORCHXRAYVISION"
FINAL_STATUS_LINE = "Classification complete"
MODEL_ID = "torchxrayvision"
DEVICE_ENV = "TXRV_DEVICE"
WEIGHTS_ENV = "TXRV_WEIGHTS"
DEFAULT_WEIGHTS = "densenet121-res224-all"
IMAGE_SUFFIXES = (
    ".dcm",
    ".dicom",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
)


def _weights() -> str:
    return (
        os.environ.get(WEIGHTS_ENV) or DEFAULT_WEIGHTS
    ).strip() or DEFAULT_WEIGHTS


def _inference_options() -> Dict[str, Any]:
    return {"model": MODEL_ID, "weights": _weights()}


def build_status_response(_ctx: ServiceProviderContext) -> Dict[str, Any]:
    return {
        "source": "status",
        "product": PRODUCT_NAME,
        "items": [{"key": "availability", "value": "online"}],
    }


def _context_files(message: Dict[str, Any]) -> List[Dict[str, Any]]:
    event = message.get("event") or {}
    ctx = event.get("context")
    if isinstance(ctx, dict):
        files = ctx.get("files")
        if isinstance(files, list):
            return [entry for entry in files if isinstance(entry, dict)]
    return []


def _manifest_file_stats(message: Dict[str, Any]) -> Tuple[int, int]:
    files = _context_files(message)
    total_bytes = 0
    for entry in files:
        byte_length = entry.get("byteLength")
        if isinstance(byte_length, int) and byte_length >= 0:
            total_bytes += byte_length
    return len(files), total_bytes


def _format_download_status_line(file_count: int, total_bytes: int) -> str:
    if total_bytes > 0:
        total_mb = total_bytes / (1024 * 1024)
        mb_text = f"{total_mb:.1f}" if total_mb < 100 else f"{total_mb:.0f}"
        return (
            f"Downloading chest X-ray, {file_count} files, "
            f"total {mb_text} MB."
        )
    return f"Downloading chest X-ray, {file_count} files."


def _publish_to_requester(
    ctx: ServiceProviderContext, message: Dict[str, Any], status_line: str
) -> None:
    event = message.get("event") or {}
    topic = (event.get("hub.topic") or "").strip()
    target_subscriber = str(message.get("subscriber.name") or "").strip()
    if not topic or not target_subscriber:
        return
    http_status = ctx.publish_status_update_sync(
        topic, target_subscriber, status_line
    )
    LOGGER.info(
        "TORCHXRAYVISION: status-update %r -> %s (HTTP %s)",
        status_line[:120],
        target_subscriber,
        http_status,
    )


def on_send_download_start(
    ctx: ServiceProviderContext, message: Dict[str, Any], _hub_event: str
) -> None:
    file_count, total_bytes = _manifest_file_stats(message)
    if file_count <= 0:
        LOGGER.warning(
            "TORCHXRAYVISION: inbound send has no context.files[]; "
            "skipping download status-update"
        )
        return
    _publish_to_requester(
        ctx, message, _format_download_status_line(file_count, total_bytes)
    )


def on_try_cache_before_download(
    ctx: ServiceProviderContext, message: Dict[str, Any], hub_event: str
) -> bool:
    """Skip study download when SeriesInstanceUID already has a complete cache hit."""
    if hub_event != "dicom-send":
        return False
    series_key = series_key_from_message(message)
    if series_key is None:
        return False
    event = message.get("event") or {}
    topic = (event.get("hub.topic") or "").strip()
    target_subscriber = str(message.get("subscriber.name") or "").strip()
    if not topic or not target_subscriber:
        return False

    options = _inference_options()
    hit = lookup_complete(PRODUCT_NAME, series_key, MODEL_ID, options)
    if hit is None:
        return False

    result_path = hit.output_files[0]
    _publish_to_requester(ctx, message, "Using cached result")
    LOGGER.info(
        "TORCHXRAYVISION: pre-download cache hit %s series=%s",
        hit.root,
        series_key.value,
    )
    _publish_result_status(ctx, message, result_path)
    _publish_to_requester(ctx, message, FINAL_STATUS_LINE)
    return True


def _resolve_input_image(input_dir: Path) -> Optional[Path]:
    files = sorted(
        p
        for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not files:
        files = sorted(p for p in input_dir.iterdir() if p.is_file())
    if not files:
        return None
    for p in files:
        if p.suffix.lower() in (".dcm", ".dicom") or p.suffix == "":
            return p
    return files[0]


def _require_job_routing(
    ctx: ServiceProviderContext,
    message: Dict[str, Any],
    input_dir: Path,
    label: str,
) -> Optional[str]:
    event = message.get("event") or {}
    topic = (event.get("hub.topic") or "").strip()
    LOGGER.info(
        "TORCHXRAYVISION: received %s id=%s topic=%s",
        label,
        message.get("id", ""),
        topic,
    )
    ctx.write_directory_manifest(input_dir)
    if not topic:
        LOGGER.warning("TORCHXRAYVISION: inbound %s missing hub.topic", label)
        return None
    if not str(message.get("subscriber.name") or "").strip():
        LOGGER.warning(
            "TORCHXRAYVISION: inbound %s missing subscriber.name; "
            "cannot send status-update",
            label,
        )
        return None
    return topic


def _prediction_device() -> str:
    raw = (os.environ.get(DEVICE_ENV) or "").strip().lower()
    if raw == "cpu":
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def _import_xrv_library() -> Any:
    """Import the PyPI ``torchxrayvision`` package, not this worker module.

    ``torchxrayvision/`` is on ``sys.path`` so a bare ``import torchxrayvision``
    would load this file. Prefer site-packages by temporarily removing our
    product directory from the path.
    """
    import importlib

    script_dir = str(_SCRIPT_DIR)
    removed: List[str] = []
    while script_dir in sys.path:
        sys.path.remove(script_dir)
        removed.append(script_dir)

    cached = sys.modules.get("torchxrayvision")
    worker_mod = None
    if cached is not None and getattr(cached, "PRODUCT_NAME", None) == PRODUCT_NAME:
        worker_mod = cached
        del sys.modules["torchxrayvision"]

    try:
        return importlib.import_module("torchxrayvision")
    finally:
        for path in reversed(removed):
            if path not in sys.path:
                sys.path.insert(0, path)
        if worker_mod is not None:
            sys.modules["torchxrayvision"] = worker_mod


def _run_host_classification(
    image_path: Path,
    job_dir: Path,
    *,
    on_status: Optional[Any] = None,
) -> Path:
    """Classify on the host Python (requires torchxrayvision + torch)."""
    import torch
    import torchvision

    xrv = _import_xrv_library()

    weights = _weights()
    device = _prediction_device()
    if on_status:
        on_status(f"Running TorchXRayVision (host, device={device})…")

    cache_root = models_dir() / MODEL_ID
    preferred_cache = cache_root / "models_data"
    preferred_cache.mkdir(parents=True, exist_ok=True)

    _orig_get_cache_dir = xrv.utils.get_cache_dir

    def _cache_dir() -> str:
        return str(preferred_cache)

    xrv.utils.get_cache_dir = _cache_dir  # type: ignore[assignment]
    try:
        LOGGER.info(
            "TORCHXRAYVISION: host classify input=%s weights=%s device=%s cache=%s",
            image_path,
            weights,
            device,
            preferred_cache,
        )
        img = xrv.utils.load_image(str(image_path))
        transform = torchvision.transforms.Compose(
            [xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)]
        )
        img = transform(img)
        model = xrv.models.get_model(weights)
        model.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(img).unsqueeze(0)
            if device == "cuda":
                tensor = tensor.cuda()
                model = model.cuda()
            preds = model(tensor).cpu()[0].detach().numpy()
        pathologies = list(
            getattr(model, "pathologies", None) or xrv.datasets.default_pathologies
        )
        scores = {
            str(name): float(score) for name, score in zip(pathologies, preds)
        }
    finally:
        xrv.utils.get_cache_dir = _orig_get_cache_dir  # type: ignore[assignment]

    out_dir = job_dir / "txrv_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "preds.json"
    payload = {
        "model": weights,
        "device": device,
        "input": image_path.name,
        "preds": scores,
    }
    out_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return out_path.resolve()


def _result_status_line(preds_path: Path) -> str:
    data = json.loads(preds_path.read_text(encoding="utf-8"))
    preds = data.get("preds") if isinstance(data, dict) else None
    if not isinstance(preds, dict):
        return f"RESULT: {preds_path.name}"
    # Compact single-line JSON for the Hub status channel.
    compact = json.dumps(
        {"model": data.get("model"), "preds": preds},
        separators=(",", ":"),
        sort_keys=True,
    )
    return f"RESULT: {compact}"


def _publish_result_status(
    ctx: ServiceProviderContext, message: Dict[str, Any], preds_path: Path
) -> None:
    _publish_to_requester(ctx, message, _result_status_line(preds_path))


def _handle_dicom_job(
    ctx: ServiceProviderContext,
    message: Dict[str, Any],
    input_dir: Path,
    *,
    run_inference,
) -> None:
    topic = _require_job_routing(ctx, message, input_dir, "dicom-send")
    if topic is None:
        return

    _publish_to_requester(ctx, message, "Download complete.")
    image_path = _resolve_input_image(input_dir)
    if image_path is None:
        _publish_to_requester(
            ctx, message, "ERROR: no chest X-ray image in input"
        )
        return

    LOGGER.info("TORCHXRAYVISION: input image %s", image_path)
    job_dir = input_dir.parent
    options = _inference_options()
    series_key = resolve_series_key(input_dir, message)
    LOGGER.info(
        "TORCHXRAYVISION: cache key kind=%s value=%s model=%s weights=%s",
        series_key.kind,
        series_key.value,
        MODEL_ID,
        options.get("weights"),
    )

    hit = lookup_complete(PRODUCT_NAME, series_key, MODEL_ID, options)
    if hit is not None:
        result_path = hit.output_files[0]
        _publish_to_requester(ctx, message, "Using cached result")
        LOGGER.info("TORCHXRAYVISION: cache hit %s", hit.root)
        _publish_result_status(ctx, message, result_path)
        _publish_to_requester(ctx, message, FINAL_STATUS_LINE)
        LOGGER.info(
            "TORCHXRAYVISION: job complete (cached) result=%s", result_path
        )
        return

    def _status(text: str) -> None:
        _publish_to_requester(ctx, message, text[:180])

    try:
        result_path = run_inference(image_path, job_dir, on_status=_status)
    except Exception as exc:
        _publish_to_requester(
            ctx, message, f"ERROR: TorchXRayVision failed: {exc}"
        )
        LOGGER.exception("TORCHXRAYVISION: inference failed")
        return

    try:
        store_complete(
            PRODUCT_NAME,
            series_key,
            MODEL_ID,
            options,
            output_files=[result_path],
            publish_as="status-update",
        )
    except Exception:
        LOGGER.exception("TORCHXRAYVISION: failed to store result cache")

    _publish_to_requester(ctx, message, "Publishing result…")
    _publish_result_status(ctx, message, result_path)
    _publish_to_requester(ctx, message, FINAL_STATUS_LINE)
    LOGGER.info("TORCHXRAYVISION: job complete result=%s", result_path)


def on_dicom_send(
    ctx: ServiceProviderContext,
    message: Dict[str, Any],
    input_dir: Path,
    file_count: int,
    total_bytes: int,
) -> None:
    del file_count, total_bytes
    _handle_dicom_job(
        ctx,
        message,
        input_dir,
        run_inference=lambda image_path, job_dir, on_status: _run_host_classification(
            image_path, job_dir, on_status=on_status
        ),
    )


def on_nifti_send(
    ctx: ServiceProviderContext,
    message: Dict[str, Any],
    input_dir: Path,
    file_count: int,
    total_bytes: int,
) -> None:
    del file_count, total_bytes
    topic = _require_job_routing(ctx, message, input_dir, "nifti-send")
    if topic is None:
        return
    _publish_to_requester(
        ctx,
        message,
        "ERROR: TorchXRayVision expects chest X-ray DICOM/PNG/JPG "
        "(use dicom-send)",
    )


HANDLERS = ServiceProviderHandlers(
    on_dicom_send=on_dicom_send,
    on_nifti_send=on_nifti_send,
    on_send_download_start=on_send_download_start,
    on_try_cache_before_download=on_try_cache_before_download,
    build_status_response=build_status_response,
)


if __name__ == "__main__":
    run_sync(
        ServiceProviderConfig(product_name=PRODUCT_NAME),
        HANDLERS,
    )
