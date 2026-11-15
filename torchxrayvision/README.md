# TorchXRayVision Hub service provider

Upstream: [mlmed/torchxrayvision](https://github.com/mlmed/torchxrayvision) — chest X-ray
pathology classification (18 findings). **Research / software development only —
not for diagnostic or clinical decision making.**

Upstream ships a **pip package only** (no official Docker image). This product wraps
it in a local Blackwell-capable image and Hub worker.

## Docker image (recommended)

```bash
docker build -t websub/torchxrayvision:cu128 \
  -f torchxrayvision/docker/Dockerfile torchxrayvision/docker
```

Weights are **not** baked in. On first run they download into
`/var/lib/websub-workers/models/torchxrayvision` (mounted at
`/root/.torchxrayvision` inside the container).

```bash
python torchxrayvision/torchxrayvision_docker.py --local
```

| Env | Meaning |
|-----|---------|
| `TXRV_DOCKER_IMAGE` | Image tag (default `websub/torchxrayvision:cu128`) |
| `TXRV_DEVICE` | `cuda` or `cpu` |
| `TXRV_GPUS` | Docker `--gpus` value (`all`, or `none`/`cpu` to omit) |
| `TXRV_WEIGHTS` | Pretrained id (default `densenet121-res224-all`) |

## Host inference (optional)

Requires `torch`, `torchvision`, `torchxrayvision`, and `pydicom` in this Python:

```bash
pip install -e python_client aiohttp torchxrayvision pydicom
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python torchxrayvision/torchxrayvision.py --local
```

## Slicer WebSub / control panel

Control panel prefers the Docker entry (`txrv` → `torchxrayvision_docker.py`).

| Field | Value |
|-------|--------|
| Product | `TORCHXRAYVISION` |
| Version | `1.0` |
| Description | e.g. TorchXRayVision chest X-ray AI |
| Hub | `SLICER-HUB` or `SLICER-HUB-CLOUD` |
| onMessage script | `torchxrayvision/torchxrayvision_docker.py` |

Hub events: `dicom-send`, `nifti-send` (rejected — use DICOM/PNG/JPG), `status-request`.

## Job flow

1. Inbound `dicom-send` → download chest radiograph into job scratch under
   `/var/lib/websub-workers/jobs/`.
2. Run DenseNet classification (`preds.json`).
3. Publish `RESULT: {…pathology scores…}` via `status-update`, then
   `Classification complete`.
4. Cache under `/var/lib/websub-workers/results/TORCHXRAYVISION/…`.

## Product card

`torchxrayvision.info.json` (`id: txrv`) is the catalog / info-card entry.
Copy into hub-js-client with `node tools/sync_inference_info.js`.

## Worklist cases

Integrated Worklist org **TorchXRayVision** (`torchxrayvision`) lists public IDC
CR/DX chest radiographs (`worklist/torchxrayvision-manifest.json`).
