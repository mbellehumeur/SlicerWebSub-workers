#!/usr/bin/env bash
# Start the independent Orchid platform (upstream docker compose).
# Does not start the WebSub client — use control panel or orchid_websub.py.
set -euo pipefail

ORCHID_SRC="${ORCHID_SRC:-${HOME}/src/orchid}"
RUNTIME_ROOT="${WEBSUB_WORKERS_RUNTIME_ROOT:-/var/lib/websub-workers}"
DATA_DIR="${RUNTIME_ROOT}/orchid/data"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OVERRIDE_EXAMPLE="${SCRIPT_DIR}/compose.override.example.yml"

if [[ ! -f "${ORCHID_SRC}/docker-compose.yml" ]]; then
  echo "error: Orchid checkout not found at ORCHID_SRC=${ORCHID_SRC}" >&2
  echo "  git clone https://github.com/kalsiva/orchid.git \"${ORCHID_SRC}\"" >&2
  exit 1
fi

mkdir -p "${DATA_DIR}"

COMPOSE=(docker compose -f docker-compose.yml)
if [[ -f "${ORCHID_SRC}/compose.override.yml" ]]; then
  COMPOSE+=(-f compose.override.yml)
elif [[ -f "${OVERRIDE_EXAMPLE}" ]]; then
  echo "note: no ${ORCHID_SRC}/compose.override.yml — using example bind for data dir"
  COMPOSE+=(-f "${OVERRIDE_EXAMPLE}")
fi

cd "${ORCHID_SRC}"
echo "ORCHID_SRC=${ORCHID_SRC}"
echo "data → ${DATA_DIR}"
"${COMPOSE[@]}" up --build -d "$@"
echo
echo "Orchid UI/API:  http://127.0.0.1:8000"
echo "DICOM C-STORE:  ORCHID@127.0.0.1:11112"
echo "Health:         curl -s http://127.0.0.1:8000/api/health"
