# websub/tools (CPU)

Shared conversion sidecar for Hub workers: DICOM → NIfTI and NIfTI labelmap →
DICOM SEG. No GPU / no model weights.

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
#   reference/   (original DICOM instances)
docker run --rm \
  -v /path/to/tools_in:/data/input:ro \
  -v /path/to/out:/data/output \
  websub/tools:cpu \
  nifti-to-dicom-seg
```

## Used by

- [`vox2seglam/`](../../vox2seglam/) — DICOM in / SEG out around GPU inference
- Intended for dental / lung_screening / other NIfTI-native workers later
