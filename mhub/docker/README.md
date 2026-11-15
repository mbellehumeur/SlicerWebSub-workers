# MHub cu128 images (local only)

Blackwell / RTX 50-friendly rebuilds of selected [MHub](https://mhub.ai)
models. Tags are **`websub/mhub-<model>:cu128`** and stay on this machine.

**Do not** `docker push` these images (or `websub/mhub-base:cu128`) to Docker
Hub or any other registry.

## Why

Published `mhubai/*:latest` images ship older PyTorch without sm_120 kernels.
The MHUB worker falls back to CPU on those hosts unless a local cu128 image
exists for the model.

## Build base (once)

From the workers repo root:

```bash
docker build -t websub/mhub-base:cu128 \
  -f mhub/docker/base/Dockerfile \
  mhub/docker/base
```

Smoke:

```bash
docker run --rm --gpus all --entrypoint python websub/mhub-base:cu128 \
  -c "import torch; print(torch.__version__, torch.cuda.get_arch_list()); print(torch.randn(1).cuda())"
```

## Build wave-1 models

```bash
./mhub/docker/build_model_cu128.sh nnunet_liver
./mhub/docker/build_model_cu128.sh lungmask
./mhub/docker/build_model_cu128.sh monai_prostate158
./mhub/docker/build_model_cu128.sh platipy
./mhub/docker/build_model_cu128.sh bamf_nnunet_ct_liver
```

(`totalsegmentator` is covered by `websub/totalsegmentator:cu128` — skip here.)

Each script:

1. Sparse-checkouts the model from [MHubAI/models](https://github.com/MHubAI/models)
2. Rewrites `FROM mhubai/base:latest` → `FROM websub/mhub-base:cu128`
3. Drops default-PyPI `torch==…` pins (base already has cu128)
4. Tags **`websub/mhub-<id>:cu128`** locally only

Scratch under `mhub/docker/build/` is gitignored.

## Worker behaviour

[`../docker_runner.py`](../docker_runner.py) prefers a local
`websub/mhub-<model>:cu128` when present; otherwise pulls `mhubai/<model>:latest`.
Blackwell CPU fallback applies only to official images, not to `:cu128` tags.

Env:

| Env | Meaning |
|-----|---------|
| `MHUB_CU128_IMAGE_TEMPLATE` | default `websub/mhub-{model}:cu128` |
| `MHUB_FORCE_GPU=1` | keep `--gpus` even for official images on Blackwell |

## Add another model later

```bash
./mhub/docker/build_model_cu128.sh <model_id>
```

Then smoke GPU and run a real study through the MHUB worker. Still: **no push**.
