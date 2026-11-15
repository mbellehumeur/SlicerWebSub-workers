#!/usr/bin/env python3
"""Container entry: classify a chest X-ray with TorchXRayVision.

Expects mounts:
  /data/input              — DICOM / PNG / JPG chest radiograph(s)
  /data/output             — preds.json written here
  /root/.torchxrayvision   — writable host cache for model weights

Env:
  TXRV_DEVICE   — cuda | cpu (default cuda)
  TXRV_WEIGHTS  — pretrained weights id (default densenet121-res224-all)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List

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


def _log(msg: str) -> None:
    print(f"TXRV.docker: {msg}", flush=True)


def _pick_input_image(input_dir: Path) -> Path:
    if not input_dir.is_dir():
        raise SystemExit(f"Missing input dir {input_dir}")
    files: List[Path] = sorted(
        p
        for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not files:
        # DICOM often has no extension — try any file and let load_image decide.
        files = sorted(p for p in input_dir.iterdir() if p.is_file())
    if not files:
        raise SystemExit(f"No image files under {input_dir}")
    # Prefer DICOM when mixed.
    for p in files:
        if p.suffix.lower() in (".dcm", ".dicom") or p.suffix == "":
            return p
    return files[0]


def _resolve_device(requested: str) -> str:
    device = (requested or "cuda").strip().lower()
    if device not in ("cuda", "cpu"):
        device = "cuda"
    if device != "cuda":
        return "cpu"
    try:
        import torch

        _log(
            f"torch={torch.__version__} cuda={torch.version.cuda} "
            f"arch={torch.cuda.get_arch_list() if torch.cuda.is_available() else []}"
        )
        if not torch.cuda.is_available():
            _log("CUDA not available inside container; falling back to cpu")
            return "cpu"
    except ImportError:
        _log("torch missing; falling back to cpu")
        return "cpu"
    return "cuda"


def main() -> int:
    input_dir = Path("/data/input")
    output_dir = Path("/data/output")
    weights = (
        os.environ.get("TXRV_WEIGHTS") or "densenet121-res224-all"
    ).strip() or "densenet121-res224-all"
    device = _resolve_device(os.environ.get("TXRV_DEVICE") or "cuda")

    image_path = _pick_input_image(input_dir)
    _log(f"input={image_path.name} weights={weights} device={device}")

    import torch
    import torchvision
    import torchxrayvision as xrv

    _log(f"cache_dir={xrv.utils.get_cache_dir()}")
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

    pathologies = list(getattr(model, "pathologies", None) or xrv.datasets.default_pathologies)
    scores = {
        str(name): float(score)
        for name, score in zip(pathologies, preds)
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    for old in output_dir.iterdir():
        if old.is_file():
            old.unlink()

    payload = {
        "model": weights,
        "device": device,
        "input": image_path.name,
        "preds": scores,
    }
    out_path = output_dir / "preds.json"
    out_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _log(f"Done: {out_path} ({len(scores)} findings)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
