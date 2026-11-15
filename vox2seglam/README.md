# Vox2SegLAM Hub service provider

Product **`VOX2SEGLAM`**: joint subcortical segmentation + landmarks on T1w
MRI via [Neuro-iX/Vox2SegLAM](https://github.com/Neuro-iX/Vox2SegLAM).

Inference runs in Docker (`websub/vox2seglam:cu128`, PyTorch **cu128** for
Blackwell / RTX 50). Weights are **not** in the image; they download into
`/var/lib/websub-workers/models/vox2seglam` on first run.

## Build image

From the workers repo root:

```bash
docker build -t websub/vox2seglam:cu128 \
  -f vox2seglam/docker/Dockerfile \
  vox2seglam/docker
```

See [`docker/README.md`](docker/README.md).

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

Subscribed events: `nifti-send`, `status-request`.

## Input / output

- **In:** one T1 NIfTI via `nifti-send` (any orientation/resolution; prep is
  inside the container).
- **Out:** `*_seg.nii.gz` published back as `nifti-send`. Landmarks
  (`.mrk.json`, `landmarks.csv`) stay under the job `docker_out/` for debugging.
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
