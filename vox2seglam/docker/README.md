# Vox2SegLAM Docker (cu128)

Blackwell-friendly image: PyTorch from the **cu128** wheel index, installed in
`/opt/venv`. Model weights are **not** in the image; `vox2seglam` downloads
them into `VOX2SEGLAM_ASSETS` (`/models`) on first run (use `--download` for
non-interactive fetch).

Upstream: https://github.com/Neuro-iX/Vox2SegLAM

## Build

From the workers repo root:

```bash
docker build -t websub/vox2seglam:cu128 \
  -f vox2seglam/docker/Dockerfile \
  vox2seglam/docker
```

Smoke GPU:

```bash
docker run --rm --gpus all --entrypoint python \
  websub/vox2seglam:cu128 \
  -c "import torch; print(torch.__version__, torch.cuda.get_arch_list()); print(torch.randn(1).cuda())"
```

## Run (manual)

```bash
mkdir -p /tmp/v2s_in /tmp/v2s_out
# place a T1 .nii.gz in /tmp/v2s_in
docker run --rm --gpus all --shm-size=8g \
  -e VOX2SEGLAM_ASSETS=/models \
  -v /tmp/v2s_in:/data/input:ro \
  -v /tmp/v2s_out:/data/output \
  -v /var/lib/websub-workers/models/vox2seglam:/models \
  websub/vox2seglam:cu128 \
  predict --input /data/input/scan.nii.gz \
  --output-dir /data/output \
  --postprocess --download --device cuda
```

Outputs include `*_seg.nii.gz` (label map on the scan grid), Slicer markups
`*.mrk.json`, and `landmarks.csv`. The Hub worker publishes the seg NIfTI
(and DICOM SEG via `websub/tools:cpu` when the job arrived as `dicom-send`).

## Hub worker

Use [`../vox2seglam_docker.py`](../vox2seglam_docker.py) as the Service
Provider onMessage script (product `VOX2SEGLAM`). Also build
[`../../tools/docker`](../../tools/docker) (`websub/tools:cpu`) for DICOM I/O.
