#!/usr/bin/env python3
"""Shared conversion CLI for websub/tools:cpu.

Subcommands:
  dicom-to-nifti       /data/input (DICOM series) → /data/output/volume.nii.gz
  nifti-to-dicom-seg   labelmap NIfTI + reference DICOM → SEG

Mounts:
  /data/input   — source files (read-only from host)
  /data/output  — conversion products
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np


def _log(msg: str) -> None:
    print(f"websub.tools: {msg}", flush=True)


def _list_nifti(dir_path: Path) -> List[Path]:
    return sorted(
        p
        for p in dir_path.iterdir()
        if p.is_file()
        and (
            p.name.endswith(".nii")
            or p.name.endswith(".nii.gz")
            or p.suffix.lower() == ".nii"
        )
    )


def _cmd_dicom_to_nifti(input_dir: Path, output_dir: Path) -> Path:
    """Convert a single DICOM series to ``volume.nii.gz`` via dcm2niix."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input dir missing: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="dcm2niix_") as tmp:
        tmp_dir = Path(tmp)
        cmd = [
            "dcm2niix",
            "-z",
            "y",
            "-b",
            "n",
            "-f",
            "volume",
            "-o",
            str(tmp_dir),
            str(input_dir),
        ]
        _log("running: " + " ".join(cmd))
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if proc.stdout:
            for line in proc.stdout.splitlines():
                _log(f"dcm2niix| {line}")
        if proc.stderr:
            for line in proc.stderr.splitlines():
                _log(f"dcm2niix| {line}")
        if proc.returncode != 0:
            raise RuntimeError(
                f"dcm2niix exited {proc.returncode}\n{proc.stderr[-2000:]}"
            )

        niftis = _list_nifti(tmp_dir)
        if not niftis:
            raise RuntimeError(
                "dcm2niix produced no NIfTI under temporary output"
            )
        if len(niftis) > 1:
            names = ", ".join(p.name for p in niftis)
            raise RuntimeError(
                "dcm2niix produced multiple NIfTI files "
                f"({names}); expected a single series under /data/input"
            )
        dest = output_dir / "volume.nii.gz"
        if dest.exists():
            dest.unlink()
        # Normalize name even if dcm2niix wrote volume.nii
        src = niftis[0]
        if src.name.endswith(".nii.gz"):
            shutil.move(str(src), str(dest))
        else:
            # Re-compress uncompressed .nii
            import nibabel as nib

            img = nib.load(str(src))
            nib.save(img, str(dest))
        _log(f"wrote {dest}")
        return dest


def _load_dicom_series(ref_dir: Path) -> List:
    """Load single-frame DICOM images sorted by ImagePositionPatient."""
    import highdicom as hd
    import pydicom

    paths: List[Path] = []
    for path in sorted(ref_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name.startswith("."):
            continue
        # Skip obvious non-DICOM
        if path.suffix.lower() in (".nii", ".gz", ".json", ".csv", ".txt", ".log"):
            continue
        paths.append(path)

    if not paths:
        raise FileNotFoundError(f"no DICOM files under {ref_dir}")

    datasets = []
    for path in paths:
        try:
            ds = hd.imread(str(path))
        except Exception:
            try:
                ds = pydicom.dcmread(str(path), force=True)
            except Exception:
                continue
            if not hasattr(ds, "SOPInstanceUID"):
                continue
        if getattr(ds, "SOPClassUID", None) and str(ds.SOPClassUID).startswith(
            "1.2.840.10008.5.1.4.1.1.66"
        ):
            # Skip existing SEG / RT
            continue
        if not hasattr(ds, "Rows") or not hasattr(ds, "Columns"):
            continue
        datasets.append(ds)

    if not datasets:
        raise RuntimeError(f"no readable image DICOM under {ref_dir}")

    series_uids = {
        str(getattr(ds, "SeriesInstanceUID", "") or "") for ds in datasets
    }
    series_uids.discard("")
    if len(series_uids) > 1:
        raise RuntimeError(
            f"reference dir has {len(series_uids)} SeriesInstanceUID values; "
            "expected a single series"
        )

    def _sort_key(ds) -> Tuple:
        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp is not None and len(ipp) >= 3:
            return (float(ipp[2]), float(ipp[1]), float(ipp[0]))
        return (int(getattr(ds, "InstanceNumber", 0) or 0),)

    datasets.sort(key=_sort_key)
    _log(f"loaded {len(datasets)} reference DICOM frames")
    return datasets


def _nifti_labelmap_to_frames(
    seg_path: Path, source_images: Sequence
) -> np.ndarray:
    """Return uint8 labelmap shaped (frames, rows, cols) aligned to DICOM."""
    import nibabel as nib
    import SimpleITK as sitk

    n_frames = len(source_images)
    rows = int(source_images[0].Rows)
    cols = int(source_images[0].Columns)

    nii = nib.load(str(seg_path))
    data = np.asanyarray(nii.dataobj)
    if data.ndim == 4:
        data = data[..., 0]
    if data.ndim != 3:
        raise RuntimeError(f"expected 3D labelmap, got shape {data.shape}")

    # nibabel array is typically (i, j, k); DICOM frames are (k, row, col)
    # with row=y, col=x. Prefer SimpleITK for spatial resampling to DICOM grid.
    sitk_img = sitk.ReadImage(str(seg_path))
    target = sitk.Image(cols, rows, n_frames, sitk.sitkUInt16)

    # Build target geometry from first/last DICOM when spacing/orientation exist
    first = source_images[0]
    spacing_x = float(getattr(first, "PixelSpacing", [1.0, 1.0])[1])
    spacing_y = float(getattr(first, "PixelSpacing", [1.0, 1.0])[0])
    # Slice spacing from adjacent ImagePositionPatient when possible
    if n_frames >= 2:
        ipp0 = np.array(source_images[0].ImagePositionPatient, dtype=float)
        ipp1 = np.array(source_images[1].ImagePositionPatient, dtype=float)
        spacing_z = float(np.linalg.norm(ipp1 - ipp0)) or 1.0
    else:
        spacing_z = float(
            getattr(first, "SpacingBetweenSlices", None)
            or getattr(first, "SliceThickness", None)
            or 1.0
        )

    iop = getattr(first, "ImageOrientationPatient", None)
    ipp = getattr(first, "ImagePositionPatient", None)
    if iop is not None and ipp is not None:
        iop = [float(x) for x in iop]
        row_dir = np.array(iop[0:3], dtype=float)
        col_dir = np.array(iop[3:6], dtype=float)
        slice_dir = np.cross(row_dir, col_dir)
        direction = (
            row_dir[0],
            col_dir[0],
            slice_dir[0],
            row_dir[1],
            col_dir[1],
            slice_dir[1],
            row_dir[2],
            col_dir[2],
            slice_dir[2],
        )
        target.SetDirection(direction)
        target.SetOrigin(tuple(float(x) for x in ipp))
        target.SetSpacing((spacing_x, spacing_y, spacing_z))
    else:
        # Fall back to shape-only nearest copy when geometry tags are missing
        _log("warning: missing IPP/IOP; using nearest shape match")
        arr = np.asarray(data)
        # Guess axis order: put the axis closest to n_frames first
        if arr.shape[2] == n_frames:
            arr = np.transpose(arr, (2, 1, 0))
        elif arr.shape[0] == n_frames:
            pass
        elif arr.shape[1] == n_frames:
            arr = np.transpose(arr, (1, 0, 2))
        else:
            raise RuntimeError(
                f"cannot align NIfTI shape {arr.shape} to "
                f"{n_frames}x{rows}x{cols} without DICOM geometry"
            )
        if arr.shape[1:] != (rows, cols):
            raise RuntimeError(
                f"NIfTI in-plane {arr.shape[1:]} != DICOM {(rows, cols)}"
            )
        return np.asarray(np.rint(arr), dtype=np.uint16).astype(np.uint8)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(target)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    resampler.SetOutputPixelType(sitk.sitkUInt16)
    resampled = resampler.Execute(sitk_img)
    arr = sitk.GetArrayFromImage(resampled)  # (z, y, x) == (frames, rows, cols)
    if arr.shape != (n_frames, rows, cols):
        raise RuntimeError(
            f"resampled shape {arr.shape} != expected "
            f"{(n_frames, rows, cols)}"
        )
    return np.asarray(np.rint(arr), dtype=np.uint16).astype(np.uint8)


def _cmd_nifti_to_dicom_seg(
    seg_path: Path,
    reference_dir: Path,
    output_dir: Path,
) -> Path:
    """Write ``segmentation.seg.dcm`` (BINARY DICOM SEG) from a labelmap."""
    import highdicom as hd
    from pydicom.sr.codedict import codes

    if not seg_path.is_file():
        raise FileNotFoundError(f"seg NIfTI not found: {seg_path}")
    if not reference_dir.is_dir():
        raise FileNotFoundError(f"reference DICOM dir missing: {reference_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    source_images = _load_dicom_series(reference_dir)
    labelmap = _nifti_labelmap_to_frames(seg_path, source_images)
    labels = sorted(int(v) for v in np.unique(labelmap) if int(v) > 0)
    if not labels:
        raise RuntimeError("labelmap has no positive labels")
    if max(labels) > 255:
        raise RuntimeError(
            f"label values up to {max(labels)}; expected uint8-range labels"
        )

    _log(f"building SEG for {len(labels)} labels")

    # highdicom BINARY expects (frames, rows, cols, segments) bool/0-1
    mask = np.zeros(
        (labelmap.shape[0], labelmap.shape[1], labelmap.shape[2], len(labels)),
        dtype=bool,
    )
    for i, lab in enumerate(labels):
        mask[..., i] = labelmap == lab

    algorithm_identification = hd.AlgorithmIdentificationSequence(
        name="websub-tools",
        version="1.0",
        family=codes.cid7162.ArtificialIntelligence,
    )

    descriptions = []
    for i, lab in enumerate(labels):
        descriptions.append(
            hd.seg.SegmentDescription(
                segment_number=i + 1,
                segment_label=f"label_{lab}",
                segmented_property_category=codes.cid7150.AnatomicalStructure,
                segmented_property_type=codes.cid7166.Tissue,
                algorithm_type=hd.seg.SegmentAlgorithmTypeValues.AUTOMATIC,
                algorithm_identification=algorithm_identification,
            )
        )

    seg_dataset = hd.seg.Segmentation(
        source_images=list(source_images),
        pixel_array=mask,
        segmentation_type=hd.seg.SegmentationTypeValues.BINARY,
        segment_descriptions=descriptions,
        series_instance_uid=hd.UID(),
        series_number=100,
        sop_instance_uid=hd.UID(),
        instance_number=1,
        manufacturer="SlicerWebSub",
        manufacturer_model_name="websub-tools",
        software_versions="1.0",
        device_serial_number="websub-tools",
        series_description="Segmentation",
    )

    dest = output_dir / "segmentation.seg.dcm"
    if dest.exists():
        dest.unlink()
    seg_dataset.save_as(str(dest))
    _log(f"wrote {dest}")
    return dest


def _resolve_seg_nifti(input_dir: Path, seg_arg: Optional[str]) -> Path:
    if seg_arg:
        path = Path(seg_arg)
        if not path.is_file():
            # Allow path relative to /data/input
            alt = input_dir / path.name
            if alt.is_file():
                return alt
            raise FileNotFoundError(f"seg NIfTI not found: {seg_arg}")
        return path
    niftis = list(input_dir.glob("*.nii")) + list(input_dir.glob("*.nii.gz"))
    niftis = [p for p in niftis if p.is_file()]
    if not niftis:
        raise FileNotFoundError(f"no NIfTI under {input_dir}")
    if len(niftis) > 1:
        # Prefer *_seg*
        preferred = [p for p in niftis if "_seg" in p.name]
        if len(preferred) == 1:
            return preferred[0]
        raise RuntimeError(
            f"multiple NIfTI under {input_dir}; pass --seg explicitly"
        )
    return niftis[0]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="websub-tools")
    sub = parser.add_subparsers(dest="command", required=True)

    p_in = sub.add_parser("dicom-to-nifti", help="DICOM series → volume.nii.gz")
    p_in.add_argument("--input", default="/data/input")
    p_in.add_argument("--output-dir", default="/data/output")

    p_out = sub.add_parser(
        "nifti-to-dicom-seg", help="labelmap NIfTI + ref DICOM → SEG"
    )
    p_out.add_argument("--input", default="/data/input")
    p_out.add_argument("--output-dir", default="/data/output")
    p_out.add_argument(
        "--seg",
        default=None,
        help="Path to labelmap NIfTI (default: single *.nii* under --input)",
    )
    p_out.add_argument(
        "--reference",
        default=None,
        help="Reference DICOM dir (default: <input>/reference)",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        if args.command == "dicom-to-nifti":
            _cmd_dicom_to_nifti(Path(args.input), Path(args.output_dir))
            return 0
        if args.command == "nifti-to-dicom-seg":
            input_dir = Path(args.input)
            ref = (
                Path(args.reference)
                if args.reference
                else input_dir / "reference"
            )
            seg = _resolve_seg_nifti(input_dir, args.seg)
            _cmd_nifti_to_dicom_seg(seg, ref, Path(args.output_dir))
            return 0
        parser.error(f"unknown command {args.command!r}")
        return 2
    except Exception as exc:
        _log(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
