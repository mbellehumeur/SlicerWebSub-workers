#!/usr/bin/env bash
# Create the host runtime layout for SlicerWebSub workers
# (models + job scratch + durable results cache).
# Idempotent: safe to re-run. Requires sudo for /var/lib paths.
set -euo pipefail

RUNTIME_ROOT="${WEBSUB_WORKERS_RUNTIME_ROOT:-/var/lib/websub-workers}"
MODELS_DIR="${RUNTIME_ROOT}/models"
JOBS_DIR="${RUNTIME_ROOT}/jobs"
RESULTS_DIR="${RUNTIME_ROOT}/results"
CONTROL_DIR="${RUNTIME_ROOT}/control"
ORCHID_DIR="${RUNTIME_ROOT}/orchid"
ORCHID_DATA_DIR="${ORCHID_DIR}/data"
ORCHID_JOBS_DIR="${ORCHID_DIR}/jobs"

# Prefer the user who invoked sudo; fall back to the current user.
OWNER="${SUDO_USER:-${USER:-$(id -un)}}"
if [[ -z "${OWNER}" || "${OWNER}" == "root" ]]; then
  echo "error: refuse to chown runtime dirs to root; set SUDO_USER or run as a normal user with sudo" >&2
  exit 1
fi

echo "Runtime root: ${RUNTIME_ROOT}"
echo "Owner:        ${OWNER}"

sudo mkdir -p \
  "${MODELS_DIR}" \
  "${JOBS_DIR}" \
  "${RESULTS_DIR}" \
  "${CONTROL_DIR}" \
  "${ORCHID_DATA_DIR}" \
  "${ORCHID_JOBS_DIR}"
sudo chown -R "${OWNER}:${OWNER}" "${RUNTIME_ROOT}"

echo
echo "Layout:"
ls -la "${RUNTIME_ROOT}"
echo
echo "Done. Models:  ${MODELS_DIR}"
echo "      Jobs:    ${JOBS_DIR}"
echo "      Results: ${RESULTS_DIR}"
echo "      Control: ${CONTROL_DIR}"
echo "      Orchid:  ${ORCHID_DIR} (data + jobs)"
