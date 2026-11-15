# TorchXRayVision Docker image

```bash
# from repo root
docker build -t websub/torchxrayvision:cu128 \
  -f torchxrayvision/docker/Dockerfile torchxrayvision/docker
```

Mounts:

| Host | Container |
|------|-----------|
| job `docker_in/` | `/data/input:ro` |
| job `docker_out/` | `/data/output` |
| `/var/lib/websub-workers/models/torchxrayvision` | `/root/.torchxrayvision` |

Entrypoint writes `/data/output/preds.json`.
