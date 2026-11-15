# Dental Segmentator Docker (cu128)

Blackwell-friendly image: PyTorch from the **cu128** wheel index, installed in
`/opt/venv` (avoids pip-as-root / system-site installs). Model weights are
**not** in the image; they download on first run into the mounted `/models`
volume (same cache as the host worker).

## Build

From the workers repo root:

```bash
docker build -t websub/dental_segmentator:cu128 \
  -f dental_segmentator/docker/Dockerfile \
  dental_segmentator/docker
```

Smoke GPU:

```bash
docker run --rm --gpus all --entrypoint python \
  websub/dental_segmentator:cu128 \
  -c "import torch; print(torch.__version__, torch.cuda.get_arch_list()); print(torch.randn(1).cuda())"
```

Expect `sm_120` (or similar) in the arch list on RTX 50-series.

## Run (manual)

```bash
mkdir -p /tmp/dental_in /tmp/dental_out
# place a .nii.gz in /tmp/dental_in
docker run --rm --gpus all --shm-size=8g \
  -v /tmp/dental_in:/data/input:ro \
  -v /tmp/dental_out:/data/output \
  -v /var/lib/websub-workers/models/dental_segmentator:/models \
  websub/dental_segmentator:cu128
```

(`--shm-size=8g` is required — default 64 MB `/dev/shm` breaks PyTorch/nnU-Net workers.)

CPU:

```bash
docker run --rm --shm-size=8g -e DENTAL_SEG_DEVICE=cpu \
  -v /tmp/dental_in:/data/input:ro \
  -v /tmp/dental_out:/data/output \
  -v /var/lib/websub-workers/models/dental_segmentator:/models \
  websub/dental_segmentator:cu128
```

## Hub worker

Use [`../dental_segmentator_docker.py`](../dental_segmentator_docker.py) as the
Service Provider onMessage script (product still `DENTAL_SEG`). It mounts the
models cache and job I/O the same way.
