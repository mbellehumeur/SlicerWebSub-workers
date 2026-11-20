#!/usr/bin/env bash
# Register the WebSub TXRV /analyze adapter in a running Orchid marketplace.
set -euo pipefail

ORCHID_API_URL="${ORCHID_API_URL:-http://127.0.0.1:8000}"
ORCHID_API_KEY="${ORCHID_API_KEY:-}"
# URL Orchid containers use to reach the adapter on the host.
ORCHID_ADAPTER_URL="${ORCHID_ADAPTER_URL:-http://host.docker.internal:9201}"
SLUG="${ORCHID_ADAPTER_SLUG:-websub-txrv}"
NAME="${ORCHID_ADAPTER_NAME:-WebSub TorchXRayVision}"
MATCH_MODALITY="${ORCHID_ADAPTER_MODALITY:-CR}"
MATCH_BODY="${ORCHID_ADAPTER_BODY_PART:-CHEST}"

AUTH_HDR=()
if [[ -n "${ORCHID_API_KEY}" ]]; then
  AUTH_HDR=(-H "X-Api-Key: ${ORCHID_API_KEY}")
fi

echo "Orchid API:  ${ORCHID_API_URL}"
echo "Adapter URL: ${ORCHID_ADAPTER_URL}"
echo "Slug:        ${SLUG}"

# Health of Orchid + adapter (adapter health via Orchid test after register).
curl -sf "${ORCHID_API_URL}/api/health" >/dev/null \
  || { echo "error: Orchid not healthy at ${ORCHID_API_URL}" >&2; exit 1; }

EXISTING="$(curl -sf "${AUTH_HDR[@]}" "${ORCHID_API_URL}/api/apps" \
  | python3 -c "
import json,sys
slug=${SLUG!r}
apps=json.load(sys.stdin)
for a in apps:
    if a.get('slug')==slug:
        print(a['id']); break
" || true)"

if [[ -n "${EXISTING}" ]]; then
  APP_ID="${EXISTING}"
  echo "App already registered id=${APP_ID}; ensuring installed+enabled"
  curl -sf "${AUTH_HDR[@]}" -X POST "${ORCHID_API_URL}/api/apps/${APP_ID}/install" >/dev/null || true
  curl -sf "${AUTH_HDR[@]}" -X PATCH "${ORCHID_API_URL}/api/apps/${APP_ID}" \
    -H "content-type: application/json" \
    -d "{\"endpoint_url\":\"${ORCHID_ADAPTER_URL}\",\"enabled\":true}" >/dev/null
else
  RESP="$(curl -sf "${AUTH_HDR[@]}" -X POST "${ORCHID_API_URL}/api/apps" \
    -H "content-type: application/json" \
    -d "$(python3 - <<PY
import json
print(json.dumps({
  "slug": "${SLUG}",
  "name": "${NAME}",
  "vendor": "SlicerWebSub-workers",
  "version": "1.0.0",
  "description": "TorchXRayVision chest X-ray classification via WebSub Orchid adapter (research only).",
  "task": "classification",
  "modality": "${MATCH_MODALITY}",
  "body_part": "${MATCH_BODY}",
  "endpoint_url": "${ORCHID_ADAPTER_URL}",
  "regulatory": "Research Use Only",
  "tags": ["websub", "torchxrayvision", "chest"],
  "icon": "🫁",
  "installed": True,
  "enabled": True,
}))
PY
)")"
  APP_ID="$(python3 -c "import json,sys; print(json.load(sys.stdin)['id'])" <<<"${RESP}")"
  echo "Registered app id=${APP_ID}"
fi

# Connectivity test from Orchid → adapter
TEST="$(curl -sf "${AUTH_HDR[@]}" -X POST "${ORCHID_API_URL}/api/apps/${APP_ID}/test" || true)"
echo "Connectivity test: ${TEST:-"(failed)"}"

# Routing rule (skip if one already points at this app for CR/CHEST)
RULE_EXISTS="$(curl -sf "${AUTH_HDR[@]}" "${ORCHID_API_URL}/api/rules" \
  | python3 -c "
import json,sys
app_id=int('${APP_ID}')
for r in json.load(sys.stdin):
    if r.get('app_id')==app_id and (r.get('match_modality') or '').upper() in ('', '${MATCH_MODALITY}'):
        print(r['id']); break
" || true)"

if [[ -n "${RULE_EXISTS}" ]]; then
  echo "Routing rule already exists id=${RULE_EXISTS}"
else
  RULE="$(curl -sf "${AUTH_HDR[@]}" -X POST "${ORCHID_API_URL}/api/rules" \
    -H "content-type: application/json" \
    -d "$(python3 - <<PY
import json
print(json.dumps({
  "name": "WebSub TXRV ${MATCH_MODALITY}/${MATCH_BODY}",
  "priority": 40,
  "enabled": True,
  "match_modality": "${MATCH_MODALITY}",
  "match_body_part": "${MATCH_BODY}",
  "app_id": int("${APP_ID}"),
}))
PY
)")"
  echo "Created rule: ${RULE}"
fi

echo "Done. Marketplace slug=${SLUG} app_id=${APP_ID}"
