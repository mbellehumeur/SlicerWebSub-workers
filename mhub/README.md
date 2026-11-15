# MHub.ai Hub service provider

Standalone Python service provider for [MHub.ai](https://mhub.ai)
([GitHub org](https://github.com/MHubAI/)). It receives a study (`dicom-send` /
`nifti-send`), runs `mhubai/<model>:latest` via Docker, reports progress via
`status-update`, publishes results, and ends with `Job complete`.

Vendored beside the worker:

- [`MHubSkill/`](https://github.com/MHubAI/MHubSkill) — model catalog, SegDB,
  workflow config generation, Docker run helpers ([`docker_runner.py`](docker_runner.py)).

Framework: `worker.py` (shared with other service providers).

## Prerequisites (host)

Follow **[docs/install.md](../docs/install.md)**: runtime dirs
`/var/lib/websub-workers/{models,jobs,results}`, Docker Engine, NVIDIA Container
Toolkit, and a Python venv with `pip install -r requirements.txt`.

You can pre-pull images, or let the worker pull on demand (it checks
`docker image inspect` first and status-updates `Pulling mhubai/<model>:latest…`
when missing):

```bash
docker pull mhubai/totalsegmentator:latest
```

## JS / hub context contract

Jobs are started by the SlicerWebSub-js MHUB client. The worker expects these
fields on inbound `dicom-send` / `nifti-send` `event.context`:

| Field | Required | Meaning |
|-------|----------|---------|
| `files` | yes | Study / series payload (framework downloads these) |
| `models` | yes* | Array of MHub model ids (e.g. `["nnunet_liver"]`); each is run in order |
| `model` | yes* | Single model id (legacy/alternate); merged with `models` if both set |
| `workflow` | no | Container `--workflow` name (DICOM path only) |
| `modality` | no | NIfTI workflow modality (`ct` / `mr` / `pt`; default `ct`) |
| `gpus` | no | Docker `--gpus` value (default `all`; override with env `MHUB_GPUS`). Use `none` / `cpu` / `off` / `false` / `0` to omit `--gpus` (CPU). |
| `seriesInstanceUID` | no | Preferred series key for NIfTI when no DICOM UIDs are present |

\* Require **`models` and/or `model`**. Ids must exist in `MHubSkill/data/models_summary.json`.  
Aliases for singular model: `Model`, `modelId`, `model_id`.

### GPU / CPU policy

Published `mhubai/*` images use older PyTorch CUDA builds that do **not** support
NVIDIA Blackwell (compute capability 12.0+, e.g. RTX 50-series). On those hosts
the worker probes `nvidia-smi`, status-updates a short reason, and runs **without**
`--gpus` (CPU).

**Local cu128 rebuilds** (optional, stay on this host — do **not** push):

```bash
docker build -t websub/mhub-base:cu128 -f mhub/docker/base/Dockerfile mhub/docker/base
./mhub/docker/build_model_cu128.sh nnunet_liver   # → websub/mhub-nnunet_liver:cu128
```

When `websub/mhub-<model>:cu128` exists locally, the worker uses it and keeps
`--gpus` on Blackwell. See [`docker/README.md`](docker/README.md).

| Env | Effect |
|-----|--------|
| `MHUB_GPUS=cpu` (or `none` / `off`) | Always CPU |
| `MHUB_FORCE_GPU=1` | Keep `--gpus` even for official images on incompatible cards |
| `MHUB_CU128_IMAGE_TEMPLATE` | default `websub/mhub-{model}:cu128` |

Probe helper: [`gpu_compat.py`](gpu_compat.py).

## Event mapping

| Inbound | Docker | Publish |
|---------|--------|---------|
| `dicom-send` | default MHub DICOM workflow → SEG | `dicom-send` |
| `nifti-send` | generated flat NIfTI config | `nifti-send` |

After each successful Docker run the worker:

- Mounts a full host tree as `/app/data` (`jobs/…/data/<model>/`) so MHub
  intermediates (`sorted_data`, nifti, etc.) stay on disk for debugging
- Writes Docker stdout to `jobs/…/docker_<model>.log`
- Logs the full data-tree and `output_data` listings (paths + sizes)

If the container finishes but `output_data` has nothing publishable (no SEG /
NIfTI), the status-update is **Job completed with no output (`{model}`)** — not
an ERROR — then the job continues / ends with **Job complete**. Inspect
`data/<model>/` and `docker_<model>.log` for why. Docker crashes and unknown
models still report `ERROR: …`.

## Results cache

Outputs are stored under the shared results tree
([`runtime/result_cache.py`](../runtime/result_cache.py)):

```text
/var/lib/websub-workers/results/MHUB/
  series/{SeriesInstanceUID}/   # or content/{sha256} for NIfTI without UID
    {model}/                    # e.g. totalsegmentator
      {options_hash}/           # hash of {model, workflow?, modality?}
        meta.json
        output/
```

Cache key: **product (`MHUB`) + series/content id + model + options_hash**.
`gpus` is **not** part of the hash. On a complete hit, the worker republishes
cached files and skips Docker.

## Slicer WebSub setup

In **Service Providers**, add or edit a row:

| Field | Value |
|-------|--------|
| Product | `MHUB` |
| Version | `1.0` |
| Description | e.g. MHub.ai model hub |
| Hub | `SLICER-HUB` or `SLICER-HUB-CLOUD` |
| onMessage script | `mhub/mhub.py` |

## Run standalone (no Slicer UI)

From the workers repo root (venv activated):

```bash
python mhub/mhub.py
python mhub/mhub.py --local
```

Default hub is **SLICER-HUB-CLOUD**; `--local` uses `http://127.0.0.1:2018`.
