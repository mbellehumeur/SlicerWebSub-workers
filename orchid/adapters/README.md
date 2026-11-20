# Orchid `/analyze` adapters

Thin HTTP services that speak Orchid’s AI app contract so WebSub-backed models
can appear in the Orchid marketplace.

**v1:** TorchXRayVision (classification). 3D SEG products (mHub / EUCAIM) are not
a good fit for the single-PNG `/analyze` frame — keep those Hub-native.

## Build & run

```bash
docker build -t websub/orchid-adapter-txrv:cpu \
  -f orchid/adapters/Dockerfile orchid/adapters

docker run -d --name orchid-adapter-txrv -p 9201:9201 \
  -v /var/lib/websub-workers/models/torchxrayvision:/root/.torchxrayvision \
  websub/orchid-adapter-txrv:cpu

curl -s http://127.0.0.1:9201/health
```

If Orchid runs in Docker and the adapter on the host, register with
`ORCHID_ADAPTER_URL=http://host.docker.internal:9201` (Linux may need
`--add-host=host.docker.internal:host-gateway` on the Orchid container).

## Register in Orchid

```bash
# Orchid platform must already be up
./orchid/adapters/register.sh
```

| Env | Default |
|-----|---------|
| `ORCHID_API_URL` | `http://127.0.0.1:8000` |
| `ORCHID_ADAPTER_URL` | `http://host.docker.internal:9201` |
| `ORCHID_ADAPTER_SLUG` | `websub-txrv` |
