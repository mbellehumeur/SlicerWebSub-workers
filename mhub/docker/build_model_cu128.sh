#!/usr/bin/env bash
# Build a local websub/mhub-<model>:cu128 image from MHubAI/models.
# Images stay on this machine — do not docker push.
#
# Usage (from workers repo root):
#   ./mhub/docker/build_model_cu128.sh nnunet_liver
#   ./mhub/docker/build_model_cu128.sh lungmask
#
# Requires: websub/mhub-base:cu128 already built.

set -euo pipefail

MODEL_ID="${1:-}"
if [[ -z "${MODEL_ID}" ]]; then
  echo "Usage: $0 <mhub_model_id>" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BASE_TAG="${MHUB_BASE_IMAGE:-websub/mhub-base:cu128}"
OUT_TAG="${MHUB_CU128_IMAGE_PREFIX:-websub/mhub-}${MODEL_ID}:cu128"
BUILD_ROOT="${MHUB_CU128_BUILD_DIR:-$ROOT/mhub/docker/build}"
CTX="${BUILD_ROOT}/${MODEL_ID}"
REPO_URL="${MHUB_MODELS_REPO_URL:-https://github.com/MHubAI/models.git}"
REPO_BRANCH="${MHUB_MODELS_REPO_BRANCH:-main}"

if ! docker image inspect "${BASE_TAG}" >/dev/null 2>&1; then
  echo "Missing base image ${BASE_TAG}." >&2
  echo "Build it first:" >&2
  echo "  docker build -t ${BASE_TAG} -f mhub/docker/base/Dockerfile mhub/docker/base" >&2
  exit 1
fi

echo "=== Local-only build (do not push) ==="
echo "model:  ${MODEL_ID}"
echo "base:   ${BASE_TAG}"
echo "tag:    ${OUT_TAG}"
echo "ctx:    ${CTX}"

rm -rf "${CTX}"
mkdir -p "${CTX}"

# Sparse-checkout only this model's tree + keep Dockerfile at build root
TMP_GIT="${CTX}/.src"
mkdir -p "${TMP_GIT}"
git -C "${TMP_GIT}" init -q
git -C "${TMP_GIT}" remote add origin "${REPO_URL}"
git -C "${TMP_GIT}" sparse-checkout set "models/${MODEL_ID}"
git -C "${TMP_GIT}" fetch -q --depth 1 origin "${REPO_BRANCH}"
git -C "${TMP_GIT}" checkout -q FETCH_HEAD

MODEL_DIR="${TMP_GIT}/models/${MODEL_ID}"
if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "Model path not found in repo: models/${MODEL_ID}" >&2
  exit 1
fi

UPSTREAM_DF="${MODEL_DIR}/dockerfiles/Dockerfile"
if [[ ! -f "${UPSTREAM_DF}" ]]; then
  echo "No Dockerfile at models/${MODEL_ID}/dockerfiles/Dockerfile" >&2
  exit 1
fi

# Build context is the model folder (import_mhub_model.sh expects /app workspace
# but upstream builds from dockerfiles/ with COPY of model via import script).
# Upstream: docker build from models/<id>/dockerfiles with context often the
# dockerfiles dir; import script clones into the image WORKDIR /app.
#
# We place a rewritten Dockerfile in CTX and use CTX as context; the RUN
# import_mhub_model.sh still fetches from GitHub inside the build.

cp "${UPSTREAM_DF}" "${CTX}/Dockerfile.orig"
# Rewrite FROM + drop default-PyPI torch pins (base already has cu128).
python3 - "${CTX}/Dockerfile.orig" "${CTX}/Dockerfile" "${BASE_TAG}" <<'PY'
import re
import sys

src, dst, base = sys.argv[1], sys.argv[2], sys.argv[3]
text = open(src, encoding="utf-8").read()
text = re.sub(
    r"^FROM\s+mhubai/base(?::\S+)?\s*$",
    f"FROM {base}",
    text,
    flags=re.M,
)

lines = text.splitlines()
out: list[str] = []
i = 0
while i < len(lines):
    line = lines[i]
    if re.search(r"\bpip3?\s+install\b", line):
        block = [line]
        while block[-1].rstrip().endswith("\\") and i + 1 < len(lines):
            i += 1
            block.append(lines[i])
        joined = "\n".join(block)
        if re.search(r"\btorch==", joined) and "cu128" not in joined:
            header = None
            pkgs: list[str] = []
            for bl in block:
                raw = bl.rstrip()
                body = raw[:-1].rstrip() if raw.endswith("\\") else raw
                if header is None:
                    m = re.match(
                        r"^(.*?pip3?\s+install(?:\s+--[^\s\\]+)*)\s*(.*)$",
                        body,
                    )
                    if not m:
                        header = body
                        continue
                    header = m.group(1).rstrip()
                    rest = m.group(2).strip()
                    if rest and not rest.startswith("torch=="):
                        pkgs.append(rest)
                    continue
                tok = body.strip()
                if tok.startswith("torch==") or not tok:
                    continue
                pkgs.append(tok)
            assert header is not None
            if not pkgs:
                out.append(
                    "# [websub cu128] skipped torch-only pip install — base has cu128"
                )
            elif len(pkgs) == 1:
                out.append(f"{header} \\")
                out.append(f"    {pkgs[0]}")
            else:
                out.append(f"{header} \\")
                for j, pkg in enumerate(pkgs):
                    suf = " \\" if j < len(pkgs) - 1 else ""
                    out.append(f"    {pkg}{suf}")
        else:
            out.extend(block)
        i += 1
        continue
    out.append(line)
    i += 1

open(dst, "w", encoding="utf-8").write("\n".join(out) + "\n")
print(f"Wrote {dst}")
PY

echo "Building ${OUT_TAG} (local only)…"
docker build \
  --build-arg "MHUB_MODELS_REPO=${REPO_URL}::${REPO_BRANCH}" \
  -t "${OUT_TAG}" \
  -f "${CTX}/Dockerfile" \
  "${CTX}"

echo
echo "Built ${OUT_TAG}"
echo "Do NOT run: docker push ${OUT_TAG}"
echo "Smoke GPU:"
echo "  docker run --rm --gpus all --entrypoint python ${OUT_TAG} \\"
echo "    -c \"import torch; print(torch.__version__, torch.cuda.is_available())\""
