# websub/tools (CPU)

Shared conversion sidecar for Hub workers: DICOM → NIfTI, NIfTI labelmap →
DICOM SEG, and landmarks CSV → DICOM SR. No GPU / no model weights.

## Build

From the workers repo root:

```bash
docker build -t websub/tools:cpu \
  -f tools/docker/Dockerfile \
  tools/docker
```

Override tag with `WEBSUB_TOOLS_DOCKER_IMAGE` when calling
[`runtime/tools_runner.py`](../../runtime/tools_runner.py).

## Manual runs

DICOM series → `volume.nii.gz`:

```bash
docker run --rm \
  -v /path/to/dicom:/data/input:ro \
  -v /path/to/out:/data/output \
  websub/tools:cpu \
  dicom-to-nifti
```

Labelmap NIfTI + reference series → `segmentation.seg.dcm`:

```bash
# Layout under input:
#   seg.nii.gz
#   structures.tsv   (optional; index/name → SegmentLabel)
#   reference/       (original DICOM instances)
docker run --rm \
  -v /path/to/tools_in:/data/input:ro \
  -v /path/to/out:/data/output \
  websub/tools:cpu \
  nifti-to-dicom-seg \
  --structures-tsv /data/input/structures.tsv \
  --algorithm-name Vox2SegLAM
```

Landmarks CSV + reference series → `landmarks.sr.dcm` (Comprehensive 3D SR):

```bash
# Layout under input:
#   landmarks.csv    (landmark_id, landmark_name, x_mm, y_mm, z_mm; LPS)
#   reference/
docker run --rm \
  -v /path/to/tools_in:/data/input:ro \
  -v /path/to/out:/data/output \
  websub/tools:cpu \
  landmarks-to-dicom-sr \
  --algorithm-name Vox2SegLAM
```

## Used by

- [`vox2seglam/`](../../vox2seglam/) — DICOM in / SEG + landmarks SR out around GPU inference
- Intended for dental / other NIfTI-native workers later
