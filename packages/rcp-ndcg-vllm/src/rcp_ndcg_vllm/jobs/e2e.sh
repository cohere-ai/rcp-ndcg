#!/usr/bin/env bash
# T4 end to end: the run scenarios of GPU-VALIDATION.md as a harness stage driven inside the pod
# (rcp_ndcg_vllm.e2e; not pytest).  The operator submits this like a recipe wave (jobs/submit.sh
# --script e2e); no lane submits a job.
#
#   e2e.sh <RC_STAGE_URI> <OUT_URI> --wave <NAME> [--wave-list <file>]
#
#     RC_STAGE_URI  gs://.../<rc-name> (or a local path), the RC that rc_build.sh staged
#     OUT_URI       gs://.../waves/<wave>: the T4 report and every run directory upload here
#     --wave NAME   the wave's list: <stage>/wave-lists/<NAME>.txt -- scenario ids (or paths), one per
#                   line.  The four shipped scenarios: text-four-phases, outage, identity, vidore.
#
# In order: the node's three environments via bootstrap.sh envs (the engine environment untouched, the
# client environment the product's own install mechanism), then the T4 stage through the client
# mechanism... the coordinator of every phase installs the release from the staged wheelhouse through
# the runners' install-source (node-runtime items 1, 2 and 10), and the stage asserts the process
# boundaries.  The auth script mounted at $RCP_GCS_AUTH_FILE is executed (by bootstrap.sh), never
# printed; HF_TOKEN reaches the engines from the job's secret.  E2E_DRY=1 prints the plan instead.
#
# Assumes about the node (checked, not assumed): the engine image's python3 imports vllm (bootstrap.sh
# fails otherwise), nvidia-smi with the GPUs the scenarios' slots declare, and the Hub reachable with
# the token secret.

set -euo pipefail

usage() {
  echo "usage: e2e.sh <RC_STAGE_URI> <OUT_URI> --wave <NAME> [--wave-list <file>]" >&2
}

RC_STAGE_URI="${1:-}"
OUT_URI="${2:-}"
if [[ -n "$RC_STAGE_URI" && -n "$OUT_URI" ]]; then
  shift 2
else
  usage
  exit 2
fi
WAVE_NAME=""
WAVE_LIST_FILE=""
while (($#)); do
  case "$1" in
    --wave) WAVE_NAME="${2:?--wave needs a name}"; shift 2 ;;
    --wave-list) WAVE_LIST_FILE="${2:?--wave-list needs a file}"; shift 2 ;;
    *) echo "e2e: unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done
[[ -n "$WAVE_NAME" ]] || { echo "e2e: needs --wave <NAME>" >&2; usage; exit 2; }

BOOTSTRAP_SH="${RCP_BOOTSTRAP_SH:-/etc/rcp/files/bootstrap/bootstrap.sh}"
export AUTH_SCRIPT="${RCP_GCS_AUTH_FILE:-/etc/rcp/gcs_auth.sh}"

STATE="$(mktemp -d "${TMPDIR:-/tmp}/rcp-e2e.XXXXXX")"

if [[ "${E2E_DRY:-0}" == "1" ]]; then
  echo "e2e (dry): would run, in order, on one node:"
  echo "  1. bootstrap  $BOOTSTRAP_SH envs $RC_STAGE_URI --state $STATE"
  echo "                (three environments: the engine untouched, the client the product's own uvx"
  echo "                mechanism from the staged wheelhouse)"
  echo "  2. scenarios  the wave list for '$WAVE_NAME' (${WAVE_LIST_FILE:-<stage>/wave-lists/$WAVE_NAME.txt}):"
  echo "                text-four-phases | outage | identity | vidore (ids of <stage>/scenarios/)"
  echo "  3. the T4     \$STATE/client python -m rcp_ndcg_vllm.e2e --scenarios @<wave list> --out \$STATE/e2e"
  echo "                (each scenario: the T0 smoke of its judge, then its run(s) as the product's phased"
  echo "                job script -- four phases, engines and coordinators under one supervision block)"
  echo "  4. upload     the report (E2E.md, status.json) and the run directories to $OUT_URI"
  exit 0
fi

echo "e2e: building the node's environments (bootstrap.sh envs)"
[[ -f "$BOOTSTRAP_SH" ]] || { echo "e2e: no bootstrap.sh at $BOOTSTRAP_SH (RCP_BOOTSTRAP_SH)" >&2; exit 2; }
bash "$BOOTSTRAP_SH" envs "$RC_STAGE_URI" --state "$STATE"

# The staged RC: a local stage is its own directory; a gs:// stage arrives at <state>/stage.
if [[ -d "$RC_STAGE_URI" ]]; then
  STAGE="$RC_STAGE_URI"
else
  STAGE="$STATE/stage"
fi
[[ -f "$STAGE/manifest.json" ]] || { echo "e2e: $STAGE holds no manifest.json (not an rc_build.sh stage)" >&2; exit 1; }
VERSION="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["version"])' "$STAGE/manifest.json")"

if [[ -z "$WAVE_LIST_FILE" ]]; then
  WAVE_LIST_FILE="$STAGE/wave-lists/$WAVE_NAME.txt"
fi
[[ -f "$WAVE_LIST_FILE" ]] || {
  echo "e2e: no wave list for '$WAVE_NAME' at $WAVE_LIST_FILE (a scenario id per line)" >&2
  exit 1
}

scenario_args=(--scenarios "@$WAVE_LIST_FILE")
for root in "$STAGE/scenarios" "$STAGE"/extra/*/scenarios; do
  [[ -d "$root" ]] && scenario_args+=(--scenarios-root "$root")
done
recipe_args=()
for root in "$STAGE/recipes" "$STAGE"/extra/*/recipes; do
  [[ -d "$root" ]] && recipe_args+=(--recipes-root "$root")
done

echo "e2e: driving the scenarios of '$WAVE_NAME' ($(grep -cv '^#' "$WAVE_LIST_FILE" || true) entries) with the client mechanism"
exec "$STATE/client" python -m rcp_ndcg_vllm.e2e \
  "${scenario_args[@]}" "${recipe_args[@]}" \
  --out "$STATE/e2e" --runs-dir "$STATE/runs" \
  --wheelhouse "$STAGE/wheelhouse" --constraints "$STAGE/requirements-constraints.txt" --version "$VERSION" \
  --upload "$OUT_URI"