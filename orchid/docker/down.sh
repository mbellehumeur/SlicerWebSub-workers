#!/usr/bin/env bash
# Stop the independent Orchid platform (upstream docker compose).
set -euo pipefail

ORCHID_SRC="${ORCHID_SRC:-${HOME}/src/orchid}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OVERRIDE_EXAMPLE="${SCRIPT_DIR}/compose.override.example.yml"

if [[ ! -f "${ORCHID_SRC}/docker-compose.yml" ]]; then
  echo "error: Orchid checkout not found at ORCHID_SRC=${ORCHID_SRC}" >&2
  exit 1
fi

COMPOSE=(docker compose -f docker-compose.yml)
if [[ -f "${ORCHID_SRC}/compose.override.yml" ]]; then
  COMPOSE+=(-f compose.override.yml)
elif [[ -f "${OVERRIDE_EXAMPLE}" ]]; then
  COMPOSE+=(-f "${OVERRIDE_EXAMPLE}")
fi

cd "${ORCHID_SRC}"
"${COMPOSE[@]}" down "$@"
