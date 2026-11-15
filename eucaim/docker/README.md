# EUCAIM Docker images

Upstream: [MIC-DKFZ/nnunet-eucaim-docker](https://github.com/MIC-DKFZ/nnunet-eucaim-docker).

Task images ship with weights baked in (CUDA 11.8 base). Prefer pulling from
Harbor; optionally retag locally as `websub/eucaim-<model>:1.0.0` so the worker
uses them without pulling again.

## Pull

```bash
./eucaim/docker/pull_or_build.sh              # all catalog models
./eucaim/docker/pull_or_build.sh liver_tumour # one model
```

Layers land under `/var/lib/docker`, not the git tree.

## Local tag (optional)

After a successful pull:

```bash
docker tag \
  harbor.eucaim.cancerimage.eu/processing-tools/nnunet-liver-tumour:1.0.0 \
  websub/eucaim-liver_tumour:1.0.0
```

The worker prefers `websub/eucaim-<model>:1.0.0` when that tag exists locally.

## Build from upstream (offline / no Harbor)

Clone the upstream repo and follow its README (`nnunet-base` then
`nnunet-task-*`). Tag the resulting image as `websub/eucaim-<model>:1.0.0`.

Do **not** vendor the full upstream tree into this git repo.

## Blackwell / RTX 50-series

Harbor images are CUDA 11.8 and are expected to fall back to CPU on Blackwell
hosts (see [`../gpu_compat.py`](../gpu_compat.py)). A cu128 rebuild is future
work; for now use CPU or a compatible GPU.
