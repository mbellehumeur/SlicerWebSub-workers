# SlicerWebSub-workers

Standalone Hub **workers** (evidence creators / inference products).

## Layout

| Path | Role |
|------|------|
| `worker.py` | Shared Hub worker framework (`run_sync`, handlers) |
| `hub_presets.py` | Hub endpoint presets |
| `runtime/` | Shared runtime helpers |
| `<product>/` | One folder per product at repo root |

## Setup

```bash
pip install -r requirements.txt
pip install -e ../SlicerWebSub/WebSubInterface/python_client
```

## Run

```bash
# terminal A — hub
python ../SlicerWebSub/WebSubInterface/hub/hub.py --port 2018

# terminal B — worker
python dental_segmentator/dental_segmentator.py --local
python total_segmentator/total_segmentator.py --local
```

## WebSub workers

Workers in this repository connect inference containers to the hub over a websocket. A worker receives instructions to process a dataset with specific options, downloads the data (from PACS or from the viewer over the hub), runs inference, and returns results to the viewer. That lets users see results without archiving them to PACS first. Workers can run on a local machine (only the hub needs a public domain), avoiding cloud GPU leasing expenses.

Workers subscribe to user topics for status-request and dicom/nifti events. In the example below, Dental Segmentator sends binary results back to the user through the hub.

<p align="center">
  <a href="https://youtu.be/vNtK8ldMcw0">
    <img src="https://img.youtube.com/vi/vNtK8ldMcw0/hqdefault.jpg" alt="Dental Segmentator demo" width="80%">
  </a>
</p>

Since workers do not log in as a user, they need a *Machine-to-Machine* application entry in the customer's authorization server. That provides a client id and client secret used against hub `POST /oauth/token` (`client_credentials`). Configure the hub with `HUB_OIDC_M2M_AUDIENCE` (and optional `HUB_SP_CLIENT_IDS`); put the M2M credentials on the worker via `HUB_SP_CLIENT_ID` / `HUB_SP_CLIENT_SECRET`. See [SlicerWebSub hub OIDC docs](https://github.com/mbellehumeur/SlicerWebSub/blob/main/WebSubInterface/hub/docs/oidc.md).

## Simplified, secure deployment of medical imaging services

This architecture protects workers by eliminating direct inbound internet exposure entirely. No hostname is required and no changes to the networking environment are needed. No VPN or proxy to configure.

Each worker establishes only outbound encrypted connections to the hub, which functions exclusively as a routing appliance. Because no inbound ports need to be opened on hospital or enterprise networks, the workers remain protected behind existing firewalls and are never directly reachable from the public internet.

It also simplifies providing resources in-house since the IT department only needs to add a hostname and rules for the hub. They do not have to touch their networking every time a new worker is available for use. They only have to configure a shared worker key for it in their authorization server.

For the hub, the architecture provides a significantly reduced attack surface and minimizes operational security risk since it maintains no storage or database.

<p align="center">
  <img src="https://raw.githubusercontent.com/mbellehumeur/SlicerWebSub/main/WebSubInterface/docs/images/deployment.png" alt="Deployment architecture" width="100%">
</p>

After installation, the workers' outbound ports can also be locked down, allowing access to the hub and sites needed by the resource only.

In theory, the hub can be cloud deployed as a serverless application. In practice, many of those low-cost offerings do not support websocket services and a docker based offering is necessary like Azure WebApps or AWS Elastic Beanstalk.

For high availability deployment a hot standby configuration can be used. The “reset server” button in the hub admin portal allows testing workflow behavior during failover.

The hub provides a test mock auth endpoint that assigns a user when none is provided. For public web applications that do not need user authentication but want to use the workers, the mock endpoints provide the required functionality.

Since the workers are not on the internet, you will get shared keys for the auth server. The hub can use domain name certificates.

## Python client usage

The shared Python package lives in the sibling SlicerWebSub repo:

```bash
pip install -e ../SlicerWebSub/WebSubInterface/python_client
```

Minimal example against a local hub on port 2018:

```python
import asyncio
from hub_client import (
    HubClientOptions,
    HubConfig,
    SessionConfig,
    SlicerHubClient,
    generate_subscriber_name,
    hub_event_name,
)

HUB = HubConfig(
    hub_endpoint="http://127.0.0.1:2018/api/hub",
    authorization_endpoint="http://127.0.0.1:2018/oauth/login",
    token_endpoint="http://127.0.0.1:2018/oauth/token",
    client_id="130c3d9c-4157-4dd1-aa1d-slicer",
    client_secret="client_secret_3d_Slicer",  # public mock-OAuth placeholder
)


async def main() -> None:
    session = SessionConfig(
        topic="USER-1",
        subscriber_name=generate_subscriber_name("DEMO"),
        product_name="DEMO",
        product_version="1.0",
        actors=["ID"],
        events=["imagingstudy-open", "imagingstudy-close"],
        user_name="demo-user",
    )
    client = SlicerHubClient(
        HUB,
        session,
        HubClientOptions(auto_reconnect=True),
    )
    await client.connect()
    print(f"listening as {client.subscriber_name} on {client.topic}")
    try:
        while True:
            msg = await client.message_queue.get()
            print("event:", hub_event_name(msg), "id:", msg.get("id"))
    finally:
        await client.close(hub_unsubscribe=True)


if __name__ == "__main__":
    asyncio.run(main())
```

More detail (live smokes, dataType tokens): [python_client README](https://github.com/mbellehumeur/SlicerWebSub/blob/main/WebSubInterface/python_client/README.md).
