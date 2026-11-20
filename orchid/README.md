# Orchid — independent platform + WebSub client

Upstream: [kalsiva/orchid](https://github.com/kalsiva/orchid) — open-source radiology
AI orchestration (DICOM ingest, marketplace, routing, DICOM SR, validation).

**Orchid installs and runs on its own.** This folder only adds:

1. Thin Docker Compose wrappers / data-dir override
2. A Hub **WebSub client** (`orchid_websub.py`) that pushes studies into Orchid and
   publishes SR / findings back
3. Optional `/analyze` adapters so WebSub-backed models (e.g. TorchXRayVision) can
   appear in Orchid’s marketplace

> Research / evaluation only. Orchid is **not a medical device** and ships demo
> models for illustration. Do not use in clinical care.

## 1. Install Orchid (independent)

```bash
git clone https://github.com/kalsiva/orchid.git ~/src/orchid
export ORCHID_SRC=~/src/orchid

# optional: bind data under the workers runtime root
cp orchid/docker/compose.override.example.yml "${ORCHID_SRC}/compose.override.yml"

./scripts/setup_runtime_dirs.sh   # creates /var/lib/websub-workers/orchid/{data,jobs}
./orchid/docker/up.sh
curl -s http://127.0.0.1:8000/api/health
```

| Port | Role |
|------|------|
| `8000` | Web UI + REST API |
| `11112` | DICOM C-STORE (`ORCHID` AE title) |

Stop with `./orchid/docker/down.sh`. The control panel does **not** start/stop the
platform — only the WebSub client.

### Start on reboot (systemd)

Unit: [`orchid/systemd/websub-orchid.service`](systemd/websub-orchid.service)
(oneshot → `up.sh` / `down.sh`, after `docker.service`).

```bash
sudo cp orchid/systemd/websub-orchid.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now websub-orchid.service
systemctl status websub-orchid.service
curl -s http://127.0.0.1:8000/api/health
```

Disable with `sudo systemctl disable --now websub-orchid.service`.

## 2. WebSub client (Hub ↔ Orchid)

Requires Orchid healthy first. Then:

```bash
python orchid/orchid_websub.py --local
# or Start → Orchid in the control panel
```

| Env | Default | Purpose |
|-----|---------|---------|
| `ORCHID_API_URL` | `http://127.0.0.1:8000` | Orchid REST base |
| `ORCHID_API_KEY` | *(empty)* | Optional `X-Api-Key` |
| `ORCHID_DICOM_HOST` | `127.0.0.1` | C-STORE host |
| `ORCHID_DICOM_PORT` | `11112` | C-STORE port |
| `ORCHID_DICOM_AET` | `ORCHID` | Called AE title |
| `ORCHID_CALLING_AET` | `WEBSUB` | Calling AE title |
| `ORCHID_INGEST` | `auto` | `auto` \| `http` \| `cstore` |
| `ORCHID_SETTLE_SECONDS` | `5` | Extra wait after last instance |
| `ORCHID_POLL_TIMEOUT` | `300` | Max seconds waiting for study complete |

### Context contract

Inbound Hub `dicom-send`:

- Optional `event.context.app` or `event.context.models` — Orchid marketplace
  **slug(s)** used for results-cache keying / filtering. If omitted, cache key is
  `routing` and Orchid’s own rules decide which apps run.
- Orchid’s installed apps + routing rules still control inference.

Flow: download → (cache lookup) → HTTP upload or C-STORE → poll
`GET /api/studies/{id}` → download SR(s) + `findings.json` → publish → cache under
`/var/lib/websub-workers/results/ORCHID/`.

### Slicer Service Provider row

| Field | Value |
|-------|--------|
| Product | `ORCHID` |
| Hub | `SLICER-HUB` or `SLICER-HUB-CLOUD` |
| onMessage script | `orchid/orchid_websub.py` |

The worker control panel (`http://127.0.0.1:8765`) shows an **Orchid radiology AI
platform** section at the bottom with Compose container status and API health
(`GET /api/orchid`).

## 3. Optional TXRV `/analyze` adapter

Orchid AI apps speak `POST /analyze` with one PNG frame. 3D SEG products
(mHub / EUCAIM) are a poor fit; classification (TorchXRayVision) is.

```bash
docker build -t websub/orchid-adapter-txrv:cpu \
  -f orchid/adapters/Dockerfile orchid/adapters

docker run -d --name orchid-adapter-txrv -p 9201:9201 \
  -v /var/lib/websub-workers/models/torchxrayvision:/root/.torchxrayvision \
  websub/orchid-adapter-txrv:cpu

./orchid/adapters/register.sh
```

Registers slug `websub-txrv` against `http://host.docker.internal:9201` (override
with `ORCHID_ADAPTER_URL` / `ORCHID_API_URL`).

## Product card

`orchid.info.json` (`id: orchid`) is discovered by the control panel.
