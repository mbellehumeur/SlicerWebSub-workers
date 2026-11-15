# Vox2SegLAM Hub service provider

Product **`VOX2SEGLAM`**: joint subcortical segmentation + landmarks on T1w
MRI via [Neuro-iX/Vox2SegLAM](https://github.com/Neuro-iX/Vox2SegLAM).

Inference runs in Docker (`websub/vox2seglam:cu128`, PyTorch **cu128** for
Blackwell / RTX 50). Weights are **not** in the image; they download into
`/var/lib/websub-workers/models/vox2seglam` on first run.

DICOM conversion uses the shared CPU sidecar `websub/tools:cpu`
([`tools/`](../tools/), [`runtime/tools_runner.py`](../runtime/tools_runner.py)).

## Build images

From the workers repo root:

```bash
docker build -t websub/vox2seglam:cu128 \
  -f vox2seglam/docker/Dockerfile \
  vox2seglam/docker

docker build -t websub/tools:cpu \
  -f tools/docker/Dockerfile \
  tools/docker
```

See [`docker/README.md`](docker/README.md) and
[`tools/docker/README.md`](../tools/docker/README.md).

## Run (Hub worker)

```bash
python vox2seglam/vox2seglam_docker.py --local
```

Omit `--local` for the default cloud hub.

## Slicer WebSub setup

| Field | Value |
|-------|--------|
| Product | `VOX2SEGLAM` |
| Version | `1.0` |
| onMessage script | `vox2seglam/vox2seglam_docker.py` |

Subscribed events: `dicom-send`, `nifti-send`, `status-request`.

## Input / output

| Inbound | Convert | Infer | Publish |
|---------|---------|-------|---------|
| `dicom-send` (T1 series) | `websub/tools:cpu` → NIfTI | `vox2seglam:cu128` | `nifti-send` (`*_seg.nii.gz`) **and** `dicom-send` (SEG) |
| `nifti-send` (T1 NIfTI) | — | same | `nifti-send` only |

- Prep (orientation/resolution) is inside the inference container.
- Landmarks (`.mrk.json`, `landmarks.csv`) stay under the job `docker_out/` for debugging.
- Default task: `subcortical` + `--postprocess`. Override with
  `VOX2SEGLAM_TASK=full-brain`.

## Env

| Env | Meaning |
|-----|---------|
| `VOX2SEGLAM_DOCKER_IMAGE` | Image tag (default `websub/vox2seglam:cu128`) |
| `VOX2SEGLAM_DEVICE` | `cuda` or `cpu` |
| `VOX2SEGLAM_GPUS` | Docker `--gpus` (default `all`; `none`/`cpu` to omit) |
| `VOX2SEGLAM_DOCKER_SHM_SIZE` | Docker `--shm-size` (default `8g`) |
| `VOX2SEGLAM_TASK` | `subcortical` (default) or `full-brain` |
| `WEBSUB_TOOLS_DOCKER_IMAGE` | Tools sidecar tag (default `websub/tools:cpu`) |
