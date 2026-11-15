# EUCAIM Hub service provider

Standalone Python service provider for
[MIC-DKFZ/nnunet-eucaim-docker](https://github.com/MIC-DKFZ/nnunet-eucaim-docker)
(EUCAIM grant). It receives a study (`dicom-send`), runs a **local** task image
from the catalog in [`models.json`](models.json), reports progress via
`status-update`, publishes DICOM SEG results, and ends with `Job complete`.

Framework: `worker.py` (shared with other service providers). Results cache:
[`runtime/result_cache.py`](../runtime/result_cache.py).

## Prerequisites (host)

Follow **[docs/install.md](../docs/install.md)**: runtime dirs
`/var/lib/websub-workers/{models,jobs,results}`, Docker Engine, NVIDIA Container
Toolkit, and a Python venv with `pip install -r requirements.txt`.

**Harbor is not used.** Build Blackwell-capable images once:

```bash
sg docker -c './eucaim/docker/build_cu128.sh'
# or one model:
sg docker -c './eucaim/docker/build_cu128.sh liver_tumour'
```

Tags: `websub/eucaim-<model>:cu128`. See [`docker/README.md`](docker/README.md).

Start the control panel (or worker) with Docker group access:

```bash
sg docker -c 'python -m control'
```

## Hub / JS context contract

Jobs are started by a SlicerWebSub client (to be built in another repo). The
worker expects these fields on inbound `dicom-send` `event.context`:

| Field | Required | Meaning |
|-------|----------|---------|
| `files` | yes | Study / series payload (framework downloads these) |
| `models` | yes* | Array of catalog ids (e.g. `["liver_tumour"]`); each is run in order |
| `model` | yes* | Single model id (legacy/alternate); merged with `models` if both set |
| `gpus` | no | Docker `--gpus` value (default `all`; override with env `EUCAIM_GPUS`). Use `none` / `cpu` / `off` / `false` / `0` to omit `--gpus` (CPU). |

\* Require **`models` and/or `model`**. Ids must exist in [`models.json`](models.json).
Aliases for singular model: `Model`, `modelId`, `model_id`. Hyphen aliases
(e.g. `liver-tumour`) are accepted.

### GPU / CPU policy

Local `websub/eucaim-*:cu128` images use PyTorch cu128 and **keep `--gpus`** on
NVIDIA Blackwell (e.g. RTX 50-series). Non-cu128 tags (if forced) still fall
back to CPU on sm_120+.

| Env | Effect |
|-----|--------|
| `EUCAIM_GPUS=cpu` (or `none` / `off`) | Always CPU |
| `EUCAIM_FORCE_GPU=1` | Keep `--gpus` even for non-cu128 images on incompatible cards |

Probe helper: [`gpu_compat.py`](gpu_compat.py).

## Event mapping

| Inbound | Docker | Publish |
|---------|--------|---------|
| `dicom-send` | EUCAIM `start_nnunet.sh` → SEG under `segmentations/` | `dicom-send` |

NIfTI is **not** supported in v1 (upstream containers expect DICOM series folders).

After each successful Docker run the worker:

- Mounts host `input/` → `/home/eucaim/nnUNet_input` and `output/` →
  `/home/eucaim/nnUNet_output` with `HOST_UID` / `HOST_GID`
- Writes Docker stdout to `jobs/…/docker_<model>.log`
- Publishes DICOM SEG from `output/segmentations/` (and any top-level `.dcm`)

If the container finishes but nothing publishable is found, the status-update is
**Job completed with no output (`{model}`)** — not an ERROR — then the job
continues / ends with **Job complete**.

## Results cache

```text
/var/lib/websub-workers/results/EUCAIM/
  series/{SeriesInstanceUID}/
    {model}/
      {options_hash}/
        meta.json
        output/
```

Cache key: **product (`EUCAIM`) + series/content id + model + options_hash**.
`gpus` is **not** part of the hash. On a complete hit, the worker republishes
cached files and skips Docker.

## Catalog (source of truth)

[`models.json`](models.json) is the worker validation catalog and the intended
source for a future client UI.

| id | Local image |
|----|-------------|
| `abdominal_organ_segmentation` | `websub/eucaim-abdominal_organ_segmentation:cu128` |
| `brain_tumour_pet` | `websub/eucaim-brain_tumour_pet:cu128` |
| `colon_cancer` | `websub/eucaim-colon_cancer:cu128` |
| `hippocampus_segmentation` | `websub/eucaim-hippocampus_segmentation:cu128` |
| `kidney_tumour` | `websub/eucaim-kidney_tumour:cu128` |
| `liver_tumour` | `websub/eucaim-liver_tumour:cu128` |
| `pancreas_tumour` | `websub/eucaim-pancreas_tumour:cu128` |

## Client (other repo)

Not implemented here. When building the SlicerWebSub / IRA client later:

1. Register Service Provider product `EUCAIM`, `hubEvent` `dicom-send`.
2. Load model pickers from `eucaim/models.json` (or an export derived from it).
3. Kick off jobs with `dicom-send` + `context.models` (and optional `gpus`).
4. Listen for `status-update` / published SEG / `Job complete` like the MHUB client.

## Slicer WebSub setup

In **Service Providers**, add or edit a row:

| Field | Value |
|-------|--------|
| Product | `EUCAIM` |
| Version | `1.0` |
| Description | e.g. EUCAIM nnU-Net tasks |
| Hub | `SLICER-HUB` or `SLICER-HUB-CLOUD` |
| onMessage script | `eucaim/eucaim_docker.py` |

Prefer starting via the control panel: `sg docker -c 'python -m control'`.

## Run standalone (no Slicer UI)

From the workers repo root (venv activated, docker group):

```bash
sg docker -c 'python eucaim/eucaim_docker.py'
sg docker -c 'python eucaim/eucaim_docker.py --local'
```

Default hub is **SLICER-HUB-CLOUD**; `--local` uses `http://127.0.0.1:2018`.
