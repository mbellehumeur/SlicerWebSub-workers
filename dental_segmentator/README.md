# Dental Segmentator Hub service provider

Standalone Python service provider for Dental Segmentator — fully automatic CT/CBCT
dental segmentation (maxilla, mandible, upper/lower teeth, mandibular canal).

Upstream: [gaudot/SlicerDentalSegmentator](https://github.com/gaudot/SlicerDentalSegmentator).
Kitware write-up: [kitware.com/dental-segmentator](https://www.kitware.com/dental-segmentator/).
Pretrained nnU-Net weights: [Zenodo](https://zenodo.org/doi/10.5281/zenodo.10829674) /
[GitHub releases](https://github.com/gaudot/SlicerDentalSegmentator/releases).

On **`nifti-send`**, the server stages the volume as `{case}_0000.nii.gz` and runs:

`nnUNetv2_predict_from_modelfolder -i … -o … -m <model> -f 0`

then publishes the labelmap back with **`nifti-send`**.

**`dicom-send`** is not implemented yet (convert to NIfTI / use nifti-send).

Framework: `worker.py` (shared with other service providers).

## Dependencies

**Host worker** ([`dental_segmentator.py`](dental_segmentator.py)):

```bash
pip install aiohttp nnunetv2
# CUDA torch (required for GPU). Plain `pip install torch` installs a +cpu build
# on Windows and dental_segmentator will fall back to CPU even if a GPU is present:
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

CUDA recommended; CPU works but is slow (set `DENTAL_SEG_DEVICE=cpu`). A `+cpu`
torch build ignores the GPU — check `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`.

**Docker worker** ([`dental_segmentator_docker.py`](dental_segmentator_docker.py)) — preferred on
RTX 50 / Blackwell: inference runs in `websub/dental_segmentator:cu128` (PyTorch
cu128). Host only needs Docker + the Hub venv (`aiohttp`, etc.), not host torch.

```bash
docker build -t websub/dental_segmentator:cu128 \
  -f dental_segmentator/docker/Dockerfile dental_segmentator/docker
```

See [`docker/README.md`](docker/README.md). Weights still land under
`/var/lib/websub-workers/models/dental_segmentator` on first run (not baked into
the image).

## Model weights

By default the service provider **auto-downloads** [Dataset111_453CT_v100.zip](https://github.com/gaudot/SlicerDentalSegmentator/releases/download/v1.0.0-alpha/Dataset111_453CT_v100.zip) (~220 MB) on the first job and caches it under:

`/var/lib/websub-workers/models/dental_segmentator`

Completed labelmaps are stored under
`/var/lib/websub-workers/results/DENTAL_SEG/` (keyed by series UID or NIfTI
content hash + model + options). A repeat request with the same key republishes
the cached NIfTI and skips nnU-Net. See [docs/install.md](../docs/install.md).

Optional override — point at a pre-unzipped folder that contains `dataset.json` and/or `fold_0` (Git Bash rewrites `/path/...`; use a real Windows path):

```bash
# Git Bash / MSYS — quote a drive path:
export DENTAL_SEG_MODEL_PATH='C:/Users/marti/models/dental'
# optional: cuda | cpu | mps
export DENTAL_SEG_DEVICE=cuda
```

PowerShell:

```powershell
$env:DENTAL_SEG_MODEL_PATH = "C:\Users\marti\models\dental"
$env:DENTAL_SEG_DEVICE = "cuda"
```

## Slicer WebSub setup

In **Service Providers**, add or edit a row:

| Field | Value |
|-------|--------|
| Product | `DENTAL_SEG` |
| Version | `1.0` |
| Description | e.g. dental CT/CBCT segmentation |
| Hub | `SLICER-HUB` or `SLICER-HUB-CLOUD` |
| onMessage script | `dental_segmentator/dental_segmentator.py` (host) **or** `dental_segmentator/dental_segmentator_docker.py` (cu128 container) |

Click **Connect**. Subscribed events: `dicom-send`, `nifti-send`, `status-request`.

## Run standalone (no Slicer UI)

From the workers repo root (venv activated):

```bash
# Host nnU-Net (needs cu128 torch on the host for RTX 50)
python dental_segmentator/dental_segmentator.py --local

# Docker cu128 image (build image first; see docker/README.md)
python dental_segmentator/dental_segmentator_docker.py --local
```

Default hub is **SLICER-HUB-CLOUD**; `--local` uses `http://127.0.0.1:2018`.

Docker worker env: `DENTAL_SEG_DOCKER_IMAGE`, `DENTAL_SEG_DEVICE`, `DENTAL_SEG_GPUS`,
`DENTAL_SEG_DOCKER_SHM_SIZE` (default `8g`; needed for nnU-Net shared memory).

## End-to-end job flow (nifti-send)

```
VolView / IRA                       Hub                         dental_segmentator.py
     |                               |                                  |
     |  nifti-send (.nii.gz)         |                                  |
     |------------------------------>|--------------------------------->|
     |                               |     download → input_dir         |
     |  status-update                |<---------------------------------|
     |                               |     nnUNetv2_predict_from_modelfolder
     |  status-update (progress)     |<---------------------------------|
     |  nifti-send (labelmap)        |<---------------------------------|
     |  status-update (complete)     |<---------------------------------|
```

Cite when using results:

> Dot G, et al. DentalSegmentator: robust open source deep learning-based CT and CBCT image segmentation. Journal of Dentistry (2024) doi:10.1016/j.jdent.2024.105130
