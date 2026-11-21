#!/usr/bin/env python3
"""Shared conversion CLI for websub/tools:cpu.

Subcommands:
  dicom-to-nifti         /data/input (DICOM series) → /data/output/volume.nii.gz
  nifti-to-dicom-seg     labelmap NIfTI + reference DICOM → SEG
  landmarks-to-dicom-sr  landmarks.csv + reference DICOM → Comprehensive 3D SR

Mounts:
  /data/input   — source files (read-only from host)
  /data/output  — conversion products
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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


def _load_structures_tsv(path: Path) -> Dict[int, str]:
    """Map label index → display name from a Vox2SegLAM-style structures.tsv."""
    mapping: Dict[int, str] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if reader.fieldnames is None:
            raise RuntimeError(f"empty structures TSV: {path}")
        fields = {name.strip().lower(): name for name in reader.fieldnames}
        index_key = fields.get("index")
        name_key = fields.get("name")
        if not index_key or not name_key:
            raise RuntimeError(
                f"structures TSV missing index/name columns: {path}"
            )
        for row in reader:
            raw_index = (row.get(index_key) or "").strip()
            raw_name = (row.get(name_key) or "").strip()
            if not raw_index or not raw_name:
                continue
            try:
                index = int(raw_index)
            except ValueError as exc:
                raise RuntimeError(
                    f"invalid index {raw_index!r} in {path}"
                ) from exc
            if index > 0:
                mapping[index] = raw_name
    return mapping


def _load_landmarks_csv(
    path: Path,
) -> List[Tuple[int, str, float, float, float]]:
    """Return (id, name, x_mm, y_mm, z_mm) rows from landmarks.csv (LPS mm)."""
    rows: List[Tuple[int, str, float, float, float]] = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise RuntimeError(f"empty landmarks CSV: {path}")
        fields = {name.strip().lower(): name for name in reader.fieldnames}
        id_key = fields.get("landmark_id")
        name_key = fields.get("landmark_name")
        x_key = fields.get("x_mm")
        y_key = fields.get("y_mm")
        z_key = fields.get("z_mm")
        missing = [
            key
            for key, value in (
                ("landmark_id", id_key),
                ("landmark_name", name_key),
                ("x_mm", x_key),
                ("y_mm", y_key),
                ("z_mm", z_key),
            )
            if value is None
        ]
        if missing:
            raise RuntimeError(
                f"landmarks CSV missing columns {missing}: {path}"
            )
        assert id_key and name_key and x_key and y_key and z_key
        for row in reader:
            try:
                landmark_id = int((row.get(id_key) or "").strip())
                name = (row.get(name_key) or "").strip()
                x_mm = float((row.get(x_key) or "").strip())
                y_mm = float((row.get(y_key) or "").strip())
                z_mm = float((row.get(z_key) or "").strip())
            except ValueError as exc:
                raise RuntimeError(
                    f"invalid landmarks row in {path}: {row}"
                ) from exc
            if not name:
                continue
            rows.append((landmark_id, name, x_mm, y_mm, z_mm))
    return rows


def _cmd_nifti_to_dicom_seg(
    seg_path: Path,
    reference_dir: Path,
    output_dir: Path,
    *,
    structures_tsv: Optional[Path] = None,
    algorithm_name: str = "websub-tools",
) -> Path:
    """Write ``segmentation.seg.dcm`` (BINARY DICOM SEG) from a labelmap."""
    import highdicom as hd
    from pydicom.sr.codedict import codes

    if not seg_path.is_file():
        raise FileNotFoundError(f"seg NIfTI not found: {seg_path}")
    if not reference_dir.is_dir():
        raise FileNotFoundError(f"reference DICOM dir missing: {reference_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    label_names: Dict[int, str] = {}
    if structures_tsv is not None:
        if not structures_tsv.is_file():
            raise FileNotFoundError(
                f"structures TSV not found: {structures_tsv}"
            )
        label_names = _load_structures_tsv(structures_tsv)
        _log(f"loaded {len(label_names)} structure names from {structures_tsv.name}")

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

    algo_name = (algorithm_name or "websub-tools").strip() or "websub-tools"
    algorithm_identification = hd.AlgorithmIdentificationSequence(
        name=algo_name,
        version="1.0",
        family=codes.cid7162.ArtificialIntelligence,
    )

    descriptions = []
    for i, lab in enumerate(labels):
        segment_label = label_names.get(lab) or f"label_{lab}"
        # DICOM SegmentLabel is LO (max 64 chars)
        if len(segment_label) > 64:
            segment_label = segment_label[:64]
        desc = hd.seg.SegmentDescription(
            segment_number=i + 1,
            segment_label=segment_label,
            segmented_property_category=codes.cid7150.AnatomicalStructure,
            segmented_property_type=codes.cid7166.Tissue,
            algorithm_type=hd.seg.SegmentAlgorithmTypeValues.AUTOMATIC,
            algorithm_identification=algorithm_identification,
        )
        # Slicer Live prefers SegmentDescription; OHIF uses SegmentLabel.
        # highdicom has no ctor arg for this tag — set it explicitly.
        desc.SegmentDescription = segment_label
        descriptions.append(desc)

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
        manufacturer_model_name=algo_name,
        software_versions="1.0",
        device_serial_number="websub-tools",
        series_description=f"{algo_name} Segmentation",
    )

    dest = output_dir / "segmentation.seg.dcm"
    if dest.exists():
        dest.unlink()
    seg_dataset.save_as(str(dest))
    _log(f"wrote {dest}")
    return dest


def _cmd_landmarks_to_dicom_sr(
    landmarks_csv: Path,
    reference_dir: Path,
    output_dir: Path,
    *,
    algorithm_name: str = "websub-tools",
) -> Path:
    """Write ``landmarks.sr.dcm`` (Comprehensive 3D SR) from landmarks.csv."""
    import highdicom as hd
    from pydicom.sr.codedict import codes

    if not landmarks_csv.is_file():
        raise FileNotFoundError(f"landmarks CSV not found: {landmarks_csv}")
    if not reference_dir.is_dir():
        raise FileNotFoundError(f"reference DICOM dir missing: {reference_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    landmarks = _load_landmarks_csv(landmarks_csv)
    if not landmarks:
        raise RuntimeError(f"no landmarks in {landmarks_csv}")

    source_images = _load_dicom_series(reference_dir)
    first = source_images[0]
    frame_of_reference_uid = str(
        getattr(first, "FrameOfReferenceUID", "") or ""
    )
    if not frame_of_reference_uid:
        raise RuntimeError(
            "reference DICOM missing FrameOfReferenceUID; "
            "cannot encode SCOORD3D landmarks"
        )

    algo_name = (algorithm_name or "websub-tools").strip() or "websub-tools"
    observer_device_context = hd.sr.ObserverContext(
        observer_type=codes.DCM.Device,
        observer_identifying_attributes=hd.sr.DeviceObserverIdentifyingAttributes(
            uid=hd.UID(),
            name=algo_name,
            manufacturer_name="SlicerWebSub",
            model_name=algo_name,
        ),
    )
    observation_context = hd.sr.ObservationContext(
        observer_device_context=observer_device_context,
    )

    imaging_measurements = []
    for landmark_id, name, x_mm, y_mm, z_mm in landmarks:
        referenced_region = hd.sr.ImageRegion3D(
            graphic_type=hd.sr.GraphicTypeValues3D.POINT,
            graphic_data=np.array([[x_mm, y_mm, z_mm]], dtype=float),
            frame_of_reference_uid=frame_of_reference_uid,
        )
        # OHIF label path: finding.text ← Finding CodeMeaning. CodeValue must
        # fit in VR SH (≤16) so it stays in CodeValue (not LongCodeValue);
        # CORNERSTONEFREETEXT is 19 chars and OHIF then shows "empty".
        display_name = name if len(name) <= 64 else name[:64]
        tracking_id = display_name
        finding_type = hd.sr.CodedConcept(
            value="LMARK",
            scheme_designator="99WEBSUB",
            meaning=display_name,
        )
        imaging_measurements.append(
            hd.sr.PlanarROIMeasurementsAndQualitativeEvaluations(
                tracking_identifier=hd.sr.TrackingIdentifier(
                    uid=hd.UID(),
                    identifier=tracking_id,
                ),
                referenced_region=referenced_region,
                finding_type=finding_type,
                text_evaluations=[
                    hd.sr.TextEvaluation(
                        name=codes.DCM.SelectedRegionDescription,
                        value=name,
                    )
                ],
            )
        )

    _log(f"building Comprehensive3DSR for {len(imaging_measurements)} landmarks")
    measurement_report = hd.sr.MeasurementReport(
        observation_context=observation_context,
        procedure_reported=codes.LN.BrainMRI,
        imaging_measurements=imaging_measurements,
    )

    sr_dataset = hd.sr.Comprehensive3DSR(
        evidence=list(source_images),
        content=measurement_report,
        series_number=101,
        series_instance_uid=hd.UID(),
        sop_instance_uid=hd.UID(),
        instance_number=1,
        manufacturer="SlicerWebSub",
        series_description=f"{algo_name} Landmarks",
    )

    dest = output_dir / "landmarks.sr.dcm"
    if dest.exists():
        dest.unlink()
    sr_dataset.save_as(str(dest))
    _log(f"wrote {dest}")
    return dest


def _resolve_input_file(
    input_dir: Path, arg: Optional[str], *, preferred_names: Sequence[str]
) -> Path:
    if arg:
        path = Path(arg)
        if path.is_file():
            return path
        alt = input_dir / path.name
        if alt.is_file():
            return alt
        raise FileNotFoundError(f"file not found: {arg}")
    for name in preferred_names:
        candidate = input_dir / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"none of {list(preferred_names)} found under {input_dir}"
    )


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
    p_out.add_argument(
        "--structures-tsv",
        default=None,
        help="Optional structures.tsv (index/name) for SegmentLabel values",
    )
    p_out.add_argument(
        "--algorithm-name",
        default="websub-tools",
        help="Algorithm / manufacturer model name written into the SEG",
    )

    p_sr = sub.add_parser(
        "landmarks-to-dicom-sr",
        help="landmarks.csv + ref DICOM → Comprehensive 3D SR",
    )
    p_sr.add_argument("--input", default="/data/input")
    p_sr.add_argument("--output-dir", default="/data/output")
    p_sr.add_argument(
        "--landmarks",
        default=None,
        help="Path to landmarks.csv (default: landmarks.csv under --input)",
    )
    p_sr.add_argument(
        "--reference",
        default=None,
        help="Reference DICOM dir (default: <input>/reference)",
    )
    p_sr.add_argument(
        "--algorithm-name",
        default="websub-tools",
        help="Algorithm / series description prefix written into the SR",
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
            structures = None
            if args.structures_tsv:
                structures = _resolve_input_file(
                    input_dir,
                    args.structures_tsv,
                    preferred_names=("structures.tsv",),
                )
            else:
                candidate = input_dir / "structures.tsv"
                if candidate.is_file():
                    structures = candidate
            _cmd_nifti_to_dicom_seg(
                seg,
                ref,
                Path(args.output_dir),
                structures_tsv=structures,
                algorithm_name=args.algorithm_name,
            )
            return 0
        if args.command == "landmarks-to-dicom-sr":
            input_dir = Path(args.input)
            ref = (
                Path(args.reference)
                if args.reference
                else input_dir / "reference"
            )
            landmarks = _resolve_input_file(
                input_dir,
                args.landmarks,
                preferred_names=("landmarks.csv",),
            )
            _cmd_landmarks_to_dicom_sr(
                landmarks,
                ref,
                Path(args.output_dir),
                algorithm_name=args.algorithm_name,
            )
            return 0
        parser.error(f"unknown command {args.command!r}")
        return 2
    except Exception as exc:
        _log(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
