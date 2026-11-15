# TotalSegmentator Docker (cu128)

Blackwell-friendly image: PyTorch from the **cu128** wheel index, installed in
`/opt/venv`. Model weights are **not** in the image; TotalSegmentator downloads
them into `TOTALSEG_HOME_DIR` (`/models`) on first run.

## Build

From the workers repo root:

```bash
docker build -t websub/totalsegmentator:cu128 \
  -f total_segmentator/docker/Dockerfile \
  total_segmentator/docker
```

Smoke GPU:

```bash
docker run --rm --gpus all --entrypoint python \
  websub/totalsegmentator:cu128 \
  -c "import torch; print(torch.__version__, torch.cuda.get_arch_list()); print(torch.randn(1).cuda())"
```

## Run (manual)

```bash
mkdir -p /tmp/ts_in /tmp/ts_out
# place a .nii.gz in /tmp/ts_in
docker run --rm --gpus all --shm-size=8g \
  -e TOTALSEG_HOME_DIR=/models \
  -v /tmp/ts_in:/data/input:ro \
  -v /tmp/ts_out:/data/output \
  -v /var/lib/websub-workers/models/totalsegmentator:/models \
  websub/totalsegmentator:cu128 \
  -i /data/input/study.nii.gz \
  -o /data/output/segmentations.dcm \
  --task total -ot dicom_seg -d gpu -nr 1 -ns 1
```

(`--shm-size=8g` is required for nnU-Net/PyTorch multiprocessing.)

## Hub worker

Use [`../total_segmentator_docker.py`](../total_segmentator_docker.py) as the Service
Provider onMessage script (product still `TOTALSEG`).
