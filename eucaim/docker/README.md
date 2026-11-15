# EUCAIM Docker images (local cu128)

Upstream source: [MIC-DKFZ/nnunet-eucaim-docker](https://github.com/MIC-DKFZ/nnunet-eucaim-docker)
(clone expected at `/home/mbellehumeur/src/nnunet-eucaim-docker`, override with
`EUCAIM_UPSTREAM`).

**Harbor is not used.** Build Blackwell-capable images on this host and tag them
`websub/eucaim-<model>:cu128`. The worker never `docker pull`s.

## Build (base + all tasks)

From the SlicerWebSub-workers repo root (Docker group required):

```bash
sg docker -c './eucaim/docker/build_cu128.sh'
# or one model:
sg docker -c './eucaim/docker/build_cu128.sh hippocampus_segmentation'
```

This builds:

1. `websub/eucaim-base:cu128` (CUDA 12.8 + PyTorch cu128) from
   [`base/Dockerfile`](base/Dockerfile) with upstream `nnunet-base/` as context
2. Tags that base as `nnunet:base-1.0.0` so upstream task Dockerfiles work
3. Each selected task → `websub/eucaim-<id>:cu128`

Layers land under `/var/lib/docker`, not the git tree. Builds are large.

## Worker

[`../models.json`](../models.json) points at `websub/eucaim-*:cu128`.
[`../gpu_compat.py`](../gpu_compat.py) keeps `--gpus` on Blackwell for `:cu128`.

Control panel must run with Docker access:

```bash
sg docker -c 'python -m control'
```

## Legacy CUDA 11.8 tags

Older `websub/eucaim-*:1.0.0` / Harbor-named tags (if still on disk) are unused
by the worker after the cu128 catalog change. Safe to leave or prune with
`docker image rm`.
