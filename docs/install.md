# Host install (workers + mHub Docker)

Commands for you to run on the worker machine. Keep **code** in the git tree; keep **heavy data** and **Docker layers** outside it.

## Directory map

| Role | Path |
|------|------|
| Source (git + Cursor Open Folder) | `/home/mbellehumeur/src/SlicerWebSub-workers` |
| Runtime root | `/var/lib/websub-workers/` |
| Model weights / caches | `/var/lib/websub-workers/models/` |
| Per-run input/output scratch | `/var/lib/websub-workers/jobs/` |
| Durable inference results | `/var/lib/websub-workers/results/` |
| Docker images/layers | `/var/lib/docker` (default; move later only if `/` is too small) |

**Rule:** repo = code; `/var/lib/websub-workers` = heavy data; Docker graph ≠ git tree.

Override the runtime root if needed:

```bash
export WEBSUB_WORKERS_RUNTIME_ROOT=/var/lib/websub-workers
```

### Results cache layout

Key = **product + SeriesInstanceUID (or content hash) + model + options_hash**:

```text
/var/lib/websub-workers/results/
  {product}/                         # e.g. DENTAL_SEG, MHUB, EUCAIM
    series/{SeriesInstanceUID}/      # DICOM series, or:
    content/{sha256}/                # NIfTI without series UID
      {model}/
        {options_hash}/
          meta.json
          output/                    # published artifact(s)
```

On a complete hit (`meta.json` `status=complete` and listed files present), workers republish cached outputs and skip inference.

## 1. Runtime directories

From the repo root:

```bash
./scripts/setup_runtime_dirs.sh
```

Creates `/var/lib/websub-workers/{models,jobs,results}` (idempotent) and chowns them to your user. Requires `sudo`.

Check:

```bash
ls -la /var/lib/websub-workers
```

## 2. Docker Engine

On Ubuntu, install the distro packages (example for this host):

```bash
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-v2
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"
```

Log out and back in (or `newgrp docker` / install `util-linux-extra` if needed) so the `docker` group applies.

Check:

```bash
docker --version
docker run --rm hello-world
```

Leave the Docker data root at `/var/lib/docker` unless disk space on `/` becomes a problem.

## 3. NVIDIA Container Toolkit (GPU in containers)

Confirm the **host** driver first:

```bash
nvidia-smi
```

Install the toolkit so containers can use the GPU. Requires `curl`. `nvidia-container-toolkit` is usually **not** in Ubuntu repos; use NVIDIA’s apt repository ([install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)):

```bash
sudo apt-get install -y curl

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Check GPU access inside a container (adjust the CUDA image tag if the pull fails):

```bash
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi
```

**RTX 50-series / Blackwell:** published `mhubai/*` and EUCAIM Harbor images
(CUDA 11.8) do not include CUDA kernels for these GPUs. The MHUB and EUCAIM
workers detect compute capability ≥ 12.0 via `nvidia-smi` and run models on
**CPU** unless you set `MHUB_FORCE_GPU=1` / `EUCAIM_FORCE_GPU=1` (only after
rebuilding images with a Blackwell-capable stack). See
[`mhub/README.md`](../mhub/README.md) and [`eucaim/README.md`](../eucaim/README.md).

For Dental Segmentator / TotalSegmentator / Vox2SegLAM on this GPU, build the
cu128 images and use the Docker workers
(`dental_segmentator/dental_segmentator_docker.py`,
`total_segmentator/total_segmentator_docker.py`,
`vox2seglam/vox2seglam_docker.py`). Weights stay under
`/var/lib/websub-workers/models/{dental_segmentator,totalsegmentator,vox2seglam}`
(downloaded on first run). See
[`dental_segmentator/docker/README.md`](../dental_segmentator/docker/README.md),
[`total_segmentator/docker/README.md`](../total_segmentator/docker/README.md),
and [`vox2seglam/docker/README.md`](../vox2seglam/docker/README.md).

For **MHub** models on Blackwell, optionally build **local-only**
`websub/mhub-base:cu128` and `websub/mhub-<model>:cu128` (do **not** push to a
registry). The MHUB worker prefers those tags when present; otherwise it pulls
`mhubai/<model>:latest` and falls back to CPU. See
[`mhub/docker/README.md`](../mhub/docker/README.md).

For **EUCAIM** models, pull Harbor tags (or retag as
`websub/eucaim-<model>:1.0.0`) via [`eucaim/docker/pull_or_build.sh`](../eucaim/docker/pull_or_build.sh).
Layers stay under `/var/lib/docker`. See
[`eucaim/docker/README.md`](../eucaim/docker/README.md).

## 4. Smoke checks (no large mHub / EUCAIM pulls)

```bash
docker --version
python3 mhub/MHubSkill/scripts/mhub_helper.py info
python3 -c "from pathlib import Path; import sys; sys.path.insert(0,'eucaim'); from docker_runner import load_models_catalog, validate_model; print(sorted(load_models_catalog())); validate_model('liver_tumour')"
```

Do **not** pull large `mhubai/<model>:latest` or EUCAIM Harbor images until you
intentionally cache a model. Those layers land under `/var/lib/docker`; model
weight caches for workers should use `/var/lib/websub-workers/models/`.

## 5. Python deps (workers)

Ubuntu system Python is PEP 668 managed — use a venv:

```bash
sudo apt-get install -y python3-venv python3-full
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

This installs the in-repo Hub client (`./python_client`) and `pydicom` (series UID reads for the results cache).

See the root [README](../README.md) for running the hub and workers.
