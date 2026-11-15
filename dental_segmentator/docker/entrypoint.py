#!/usr/bin/env python3
"""Container entry: ensure Dataset111 under /models, run nnU-Net dental predict.

Expects mounts:
  /data/input   — NIfTI (single file or already ``*_0000.nii.gz``)
  /data/output  — prediction output
  /models       — writable host cache for weights (downloaded on miss)

Env:
  DENTAL_SEG_MODELS  — models root (default /models)
  DENTAL_SEG_DEVICE  — cuda | cpu (default cuda)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

WEIGHTS_ZIP_URL = (
    "https://github.com/gaudot/SlicerDentalSegmentator/releases/download/"
    "v1.0.0-alpha/Dataset111_453CT_v100.zip"
)
WEIGHTS_ZIP_NAME = "Dataset111_453CT_v100.zip"
CASE_ID = "dental"


def _log(msg: str) -> None:
    print(f"DENTAL_SEG.docker: {msg}", flush=True)


def _cache_has_model(root: Path) -> bool:
    try:
        _resolve_model_folder(root)
        return True
    except FileNotFoundError:
        return False


def _resolve_model_folder(model_root: Path) -> Path:
    if (model_root / "fold_0").is_dir() or (
        (model_root / "dataset.json").is_file()
        and any(model_root.glob("fold_*"))
    ):
        return model_root

    for fold in sorted(model_root.rglob("fold_0")):
        parent = fold.parent
        if (parent / "plans.json").is_file() or (
            parent / "dataset.json"
        ).is_file():
            return parent
        return parent

    dataset_json = next(model_root.rglob("dataset.json"), None)
    if dataset_json is not None:
        dataset_dir = dataset_json.parent
        trainers = sorted(dataset_dir.glob("nnUNetTrainer*"))
        if trainers:
            return trainers[0]
        return dataset_dir

    raise FileNotFoundError(
        f"No nnUNet model (fold_0 / dataset.json) under {model_root}"
    )


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".partial")
    if partial.is_file():
        partial.unlink()
    _log(f"Downloading weights (~220 MB) from {url}")
    try:
        with urllib.request.urlopen(url, timeout=120) as resp:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            last_pct = -1
            with open(partial, "wb") as out:
                while True:
                    chunk = resp.read(1024 * 256)
                    if not chunk:
                        break
                    out.write(chunk)
                    done += len(chunk)
                    if total > 0:
                        pct = int(100 * done / total)
                        if pct >= last_pct + 10 or pct == 100:
                            last_pct = pct
                            _log(
                                f"Downloading weights… {pct}% "
                                f"({done / (1024 * 1024):.0f}/"
                                f"{total / (1024 * 1024):.0f} MB)"
                            )
    except urllib.error.URLError as exc:
        if partial.is_file():
            partial.unlink()
        raise SystemExit(f"Failed to download weights: {exc}") from exc
    partial.replace(dest)
    _log(f"Downloaded {dest.name}")


def _ensure_models(models_root: Path) -> Path:
    if _cache_has_model(models_root):
        folder = _resolve_model_folder(models_root)
        _log(f"Using cached model at {folder}")
        return folder

    models_root.mkdir(parents=True, exist_ok=True)
    zip_path = models_root / WEIGHTS_ZIP_NAME
    if not zip_path.is_file():
        _download(WEIGHTS_ZIP_URL, zip_path)
    else:
        _log(f"Using cached zip {zip_path}")

    _log("Unzipping weights…")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(models_root)
    except zipfile.BadZipFile as exc:
        zip_path.unlink(missing_ok=True)
        raise SystemExit(
            f"Corrupt weights zip {zip_path}; deleted — re-run to download"
        ) from exc

    if not _cache_has_model(models_root):
        raise SystemExit(
            f"Unzipped under {models_root} but no nnUNet model found"
        )
    folder = _resolve_model_folder(models_root)
    _log(f"Weights ready at {folder}")
    return folder


def _pick_input_nifti(input_dir: Path) -> Path:
    if not input_dir.is_dir():
        raise SystemExit(f"Missing input dir {input_dir}")
    files = sorted(
        p
        for p in input_dir.iterdir()
        if p.is_file()
        and p.name.lower().endswith((".nii.gz", ".nii"))
    )
    if not files:
        raise SystemExit(f"No NIfTI under {input_dir}")
    for p in files:
        if p.name.endswith("_0000.nii.gz") or p.name.endswith("_0000.nii"):
            return p
    return files[0]


def _stage_input(nifti: Path, stage_dir: Path) -> Path:
    stage_dir.mkdir(parents=True, exist_ok=True)
    for old in stage_dir.iterdir():
        if old.is_file():
            old.unlink()
    dest = stage_dir / f"{CASE_ID}_0000.nii.gz"
    if nifti.resolve() != dest.resolve():
        shutil.copy2(nifti, dest)
    return dest


def _find_output(pred_dir: Path) -> Path:
    preferred = pred_dir / f"{CASE_ID}.nii.gz"
    if preferred.is_file():
        return preferred
    matches = sorted(pred_dir.glob("*.nii.gz")) + sorted(pred_dir.glob("*.nii"))
    if not matches:
        raise SystemExit(f"No segmentation NIfTI under {pred_dir}")
    return matches[0]


def main() -> int:
    models_root = Path(
        os.environ.get("DENTAL_SEG_MODELS", "/models")
    ).expanduser()
    input_dir = Path("/data/input")
    output_dir = Path("/data/output")
    work_in = Path("/tmp/nnunet_in")
    device = (os.environ.get("DENTAL_SEG_DEVICE") or "cuda").strip().lower()
    if device not in ("cuda", "cpu"):
        device = "cuda"

    if device == "cuda":
        try:
            import torch

            _log(
                f"torch={torch.__version__} cuda={torch.version.cuda} "
                f"arch={torch.cuda.get_arch_list() if torch.cuda.is_available() else []}"
            )
            if not torch.cuda.is_available():
                _log("CUDA not available inside container; falling back to cpu")
                device = "cpu"
        except ImportError:
            _log("torch missing; falling back to cpu")
            device = "cpu"

    model_folder = _ensure_models(models_root)
    nifti = _pick_input_nifti(input_dir)
    staged = _stage_input(nifti, work_in)
    output_dir.mkdir(parents=True, exist_ok=True)
    for old in output_dir.iterdir():
        if old.is_file():
            old.unlink()

    args = [
        "nnUNetv2_predict_from_modelfolder",
        "-i",
        str(work_in),
        "-o",
        str(output_dir),
        "-m",
        str(model_folder),
        "-f",
        "0",
        "-device",
        device,
        "-npp",
        "1",
        "-nps",
        "1",
        "--disable_tta",
        "--disable_progress_bar",
    ]
    if device == "cpu":
        args.append("--not_on_device")

    _log(f"Running: {' '.join(args)}")
    _log(f"staged input={staged}")
    proc = subprocess.run(args, check=False)
    if proc.returncode != 0:
        return proc.returncode

    result = _find_output(output_dir)
    _log(f"Done: {result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
