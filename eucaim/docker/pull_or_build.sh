#!/usr/bin/env bash
# Pull EUCAIM nnU-Net task images from Harbor (or list catalog ids).
# Usage:
#   ./eucaim/docker/pull_or_build.sh              # all models
#   ./eucaim/docker/pull_or_build.sh liver_tumour # one model
# Optional: RETAG_LOCAL=1 retags each pulled image as websub/eucaim-<id>:1.0.0
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS_JSON="${ROOT}/models.json"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 required to read models.json" >&2
  exit 1
fi

mapfile -t ENTRIES < <(
  python3 - "$MODELS_JSON" "$@" <<'PY'
import json, sys
from pathlib import Path

path = Path(sys.argv[1])
want = [a.strip() for a in sys.argv[2:] if a.strip()]
data = json.loads(path.read_text(encoding="utf-8"))
models = data.get("models") or []
by_id = {m["id"]: m for m in models if isinstance(m, dict) and m.get("id")}
if want:
    missing = [i for i in want if i not in by_id]
    if missing:
        raise SystemExit(f"unknown model id(s): {', '.join(missing)}")
    selected = [by_id[i] for i in want]
else:
    selected = list(by_id.values())
for m in selected:
    print(f"{m['id']}\t{m['image']}")
PY
)

if [[ ${#ENTRIES[@]} -eq 0 ]]; then
  echo "no models selected" >&2
  exit 1
fi

for entry in "${ENTRIES[@]}"; do
  id="${entry%%$'\t'*}"
  image="${entry#*$'\t'}"
  echo "Pulling ${image} …"
  docker pull "${image}"
  if [[ "${RETAG_LOCAL:-0}" == "1" ]]; then
    local_tag="websub/eucaim-${id}:1.0.0"
    echo "Tagging ${local_tag}"
    docker tag "${image}" "${local_tag}"
  fi
done

echo "Done."
