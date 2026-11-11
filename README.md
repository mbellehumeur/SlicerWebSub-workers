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
