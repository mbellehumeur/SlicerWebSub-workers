# EUCAIM Hub service provider

Standalone Python service provider for
[MIC-DKFZ/nnunet-eucaim-docker](https://github.com/MIC-DKFZ/nnunet-eucaim-docker)
(EUCAIM grant). It receives a study (`dicom-send`), runs a task image from the
catalog in [`models.json`](models.json), reports progress via `status-update`,
publishes DICOM SEG results, and ends with `Job complete`.

Framework: `worker.py` (shared with other service providers). Results cache:
[`runtime/result_cache.py`](../runtime/result_cache.py).

## Prerequisites (host)

Follow **[docs/install.md](../docs/install.md)**: runtime dirs
`/var/lib/websub-workers/{models,jobs,results}`, Docker Engine, NVIDIA Container
Toolkit, and a Python venv with `pip install -r requirements.txt`.

Pre-pull images, or let the worker pull on demand:

```bash
./eucaim/docker/pull_or_build.sh liver_tumour
# or:
docker pull harbor.eucaim.cancerimage.eu/processing-tools/nnunet-liver-tumour:1.0.0
```

Optional local tags (preferred when present): `websub/eucaim-<model>:1.0.0`.
See [`docker/README.md`](docker/README.md).

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
Aliases for singular model: `Model`, `modelId`, `model_id`.

### GPU / CPU policy

Published EUCAIM harbor images use CUDA **11.8** and do **not** support NVIDIA
Blackwell (compute capability 12.0+, e.g. RTX 50-series). On those hosts the
worker probes `nvidia-smi`, status-updates a short reason, and runs **without**
`--gpus` (CPU).

| Env | Effect |
|-----|--------|
| `EUCAIM_GPUS=cpu` (or `none` / `off`) | Always CPU |
| `EUCAIM_FORCE_GPU=1` | Keep `--gpus` even for harbor images on incompatible cards |

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
source for a future client UI (same role as MHub’s `models_summary.json`).

| id | Harbor image |
|----|----------------|
| `abdominal_organ_segmentation` | `…/nnunet-abdominal-organ-segmentation:1.0.0` |
| `brain_tumour_pet` | `…/nnunet-brain-tumour-pet:1.0.0` |
| `colon_cancer` | `…/nnunet-colon-cancer:1.0.0` |
| `hippocampus_segmentation` | `…/nnunet-hippocampus-segmentation:1.0.0` |
| `kidney_tumour` | `…/nnunet-kidney-tumour:1.0.0` |
| `liver_tumour` | `…/nnunet-liver-tumour:1.0.0` |
| `pancreas_tumour` | `…/nnunet-pancreas-tumour:1.0.0` |

Prefix: `harbor.eucaim.cancerimage.eu/processing-tools/`.

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
| onMessage script | `eucaim/eucaim.py` |

## Run standalone (no Slicer UI)

From the workers repo root (venv activated):

```bash
python eucaim/eucaim.py
python eucaim/eucaim.py --local
```

Default hub is **SLICER-HUB-CLOUD**; `--local` uses `http://127.0.0.1:2018`.
