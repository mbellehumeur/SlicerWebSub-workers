#!/usr/bin/env bash
# Build local Blackwell EUCAIM images (websub/eucaim-*:cu128). No Harbor.
#
# Usage (from SlicerWebSub-workers repo root, with docker group):
#   ./eucaim/docker/build_cu128.sh              # base + all tasks
#   ./eucaim/docker/build_cu128.sh liver_tumour # base (if needed) + one task
#
# Optional:
#   EUCAIM_UPSTREAM=/path/to/nnunet-eucaim-docker
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DOCKER_DIR="$(cd "$(dirname "$0")" && pwd)"
UPSTREAM="${EUCAIM_UPSTREAM:-/home/mbellehumeur/src/nnunet-eucaim-docker}"
BASE_CTX="${UPSTREAM}/nnunet-base"
BASE_DOCKERFILE="${DOCKER_DIR}/base/Dockerfile"
BASE_TAG="websub/eucaim-base:cu128"
UPSTREAM_BASE_ALIAS="nnunet:base-1.0.0"

if [[ ! -d "${BASE_CTX}" ]]; then
  echo "missing upstream nnunet-base at ${BASE_CTX}" >&2
  echo "Set EUCAIM_UPSTREAM to your nnunet-eucaim-docker clone." >&2
  exit 1
fi
if [[ ! -f "${BASE_DOCKERFILE}" ]]; then
  echo "missing ${BASE_DOCKERFILE}" >&2
  exit 1
fi

# model_id -> relative task dir under upstream
declare -A TASK_DIRS=(
  [abdominal_organ_segmentation]=nnunet-task-abdominal
  [brain_tumour_pet]=nnunet-task-brain-pet
  [colon_cancer]=nnunet-task-colon
  [hippocampus_segmentation]=nnunet-task-hippocampus
  [kidney_tumour]=nnunet-task-kidney
  [liver_tumour]=nnunet-task-liver
  [pancreas_tumour]=nnunet-task-pancreas
)

ALL_IDS=(
  abdominal_organ_segmentation
  brain_tumour_pet
  colon_cancer
  hippocampus_segmentation
  kidney_tumour
  liver_tumour
  pancreas_tumour
)

WANT=("$@")
if [[ ${#WANT[@]} -eq 0 ]]; then
  SELECTED=("${ALL_IDS[@]}")
else
  SELECTED=()
  for id in "${WANT[@]}"; do
    id="${id//-/_}"
    if [[ -z "${TASK_DIRS[$id]+x}" ]]; then
      echo "unknown model id: $id" >&2
      echo "known: ${ALL_IDS[*]}" >&2
      exit 1
    fi
    SELECTED+=("$id")
  done
fi

echo "======== BUILD ${BASE_TAG} ========"
docker build \
  --build-arg "USER_UID=$(id -u)" \
  --build-arg "USER_GID=$(id -g)" \
  -t "${BASE_TAG}" \
  -f "${BASE_DOCKERFILE}" \
  "${BASE_CTX}"

echo "Tagging ${BASE_TAG} as ${UPSTREAM_BASE_ALIAS} for task Dockerfiles"
docker tag "${BASE_TAG}" "${UPSTREAM_BASE_ALIAS}"

for id in "${SELECTED[@]}"; do
  ctx="${UPSTREAM}/${TASK_DIRS[$id]}"
  tag="websub/eucaim-${id}:cu128"
  if [[ ! -d "${ctx}" ]]; then
    echo "missing task context: ${ctx}" >&2
    exit 1
  fi
  echo "======== BUILD ${tag} ========"
  docker build -t "${tag}" "${ctx}/"
  docker image inspect "${tag}" --format "OK {{.RepoTags}}"
done

echo "======== ALL DONE ========"
echo "Images:"
docker images --format '{{.Repository}}:{{.Tag}}\t{{.Size}}' | grep -E 'websub/eucaim-.*:cu128|nnunet:base' || true
