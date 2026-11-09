#!/usr/bin/env bash
# W13(c)'s gate: the late-interaction MaxSim scoring path, as its own wave (the reviews' D4: the
# equivalence waves run stages 1-2 only, and stage 3 / --quality never runs, so the product's
# client-side MaxSim scorer -- the path the retrieval view and `rcp-ndcg eval score` consume -- had
# no wave).  The node wrapper for the probe module rcp_ndcg_test/jobs/maxsim_probe.py (mounted beside
# this script by submit.sh).
#
#   maxsim_probe.sh <RC_STAGE_URI> <OUT_URI>
#
#     RC_STAGE_URI  gs://.../<rc-name> (or a local path), the RC that rc_build.sh staged; its
#                   wheelhouse builds the client mechanism, its recipes install the plugin wheel into
#                   the engine environment and build the family's reference venv (bootstrap.sh envs)
#     OUT_URI       gs://.../waves/<wave>: maxsim-report.json, maxsim-report.md, maxsim-rows.json and
#                   the serve logs upload here
#
# In order:
#   1. environments  bootstrap.sh envs with a wave list naming the family's variants: the client
#                    mechanism (the product + the harness, from the staged wheelhouse), the recipe's
#                    plugin wheel in the engine environment (--no-deps, freeze-diff guarded) and the
#                    family's reference venv at <state>/reference/pplx-embed-v2-late (its reference
#                    declares device: cuda: a GPU of its own)
#   2. probe         the client mechanism runs maxsim_probe.py per variant: one engine start, the
#                    pairs file's text rows through the product's role client, the reference's
#                    --mode embed, then the gate -- the product's own MaxSim scorer on the served
#                    vectors against the card's direct float64 MaxSim on the reference's, under the
#                    recipe's declared per-token floor's derived per-pair bound and the published
#                    median-tau rank gate; the expected MaxSim rows are recorded as data
#   3. upload        the reports, the rows and the serve logs; a failed transfer fails the job (no
#                    silent green with no artefacts)
#
# Environment: RCP_GCS_AUTH_FILE (the mounted auth script; executed, never printed), HF_TOKEN (from the
# mounted token file, exported by submit.sh's wrapper; never printed), RCP_BOOTSTRAP_SH /
# RCP_MAXSIM_PROBE_PY / RCP_GCS_HELPER_SH / RCP_GCS_HELPER_PY (the mounted helpers),
# RCP_MAXSIM_RECIPES (pplx-embed-v2-late-0.6b,pplx-embed-v2-late-9b; the family's variants, one probe
# run each), RCP_MAXSIM_ROWS (empty: every text row), RCP_MAXSIM_PORT_BASE (8100),
# RCP_MAXSIM_GPUS (0; the engine's CUDA_VISIBLE_DEVICES), RCP_MAXSIM_REFERENCE_GPUS (1; the reference's),
# RCP_MAXSIM_STARTUP_TIMEOUT_S (1800; one engine's load budget).  MAXSIM_DRY=1 prints the plan and runs
# nothing.  The transfer runs jobs/gcs.sh's gcs_cp (the one dispatch home: gcloud/gsutil when one is
# present, else the python gcsfs helper).

set -euo pipefail

usage() { echo "usage: maxsim_probe.sh <RC_STAGE_URI> <OUT_URI>" >&2; }

RC_STAGE_URI="${1:-}"
OUT_URI="${2:-}"
if [[ -n "$RC_STAGE_URI" && -n "$OUT_URI" ]]; then
  shift 2
else
  usage
  exit 2
fi
(($# == 0)) || { echo "maxsim_probe: unknown argument: $1" >&2; usage; exit 2; }

BOOTSTRAP_SH="${RCP_BOOTSTRAP_SH:-/etc/rcp/files/bootstrap/bootstrap.sh}"
PROBE_PY="${RCP_MAXSIM_PROBE_PY:-/etc/rcp/files/maxsimprobe/maxsim_probe.py}"
GCS_SH="${RCP_GCS_HELPER_SH:-/etc/rcp/files/gcshelper/gcs.sh}"
GCS_HELPER_PY="${RCP_GCS_HELPER_PY:-/etc/rcp/files/gcshelper/gcs.py}"
FAMILY="pplx-embed-v2-late"
RECIPES="${RCP_MAXSIM_RECIPES:-$FAMILY-0.6b,$FAMILY-9b}"
ROWS="${RCP_MAXSIM_ROWS:-}"
PORT_BASE="${RCP_MAXSIM_PORT_BASE:-8100}"
GPUS="${RCP_MAXSIM_GPUS:-0}"
REFERENCE_GPUS="${RCP_MAXSIM_REFERENCE_GPUS:-1}"
STARTUP_TIMEOUT="${RCP_MAXSIM_STARTUP_TIMEOUT_S:-1800}"

if [[ "${MAXSIM_DRY:-0}" == "1" ]]; then
  echo "maxsim_probe (dry): would run, in order, on one node:"
  echo "  1. bootstrap  $BOOTSTRAP_SH envs $RC_STAGE_URI --state <state> --recipes <stage>/recipes"
  echo "                --wave-list <state>/maxsim-wave.txt ($RECIPES; one reference venv for the family:"
  echo "                <state>/reference/$FAMILY, its own GPU)"
  echo "  2. probe      for each variant of $RECIPES: <state>/client python $PROBE_PY --out <state>/probe/<recipe>"
  echo "                --recipe <recipe> --pairs <stage>/pairs/<recipe>.jsonl \\"
  echo "                --reference-python <state>/reference/$FAMILY/bin/python \\"
  echo "                --port-base $PORT_BASE --gpus $GPUS --reference-gpus $REFERENCE_GPUS \\"
  echo "                --startup-timeout-s $STARTUP_TIMEOUT"
  if [[ -n "$ROWS" ]]; then echo "                --rows $ROWS"; fi
  echo "                (one engine start per variant; the gate: the product's MaxSim scorer on the"
  echo "                 served vectors vs the card's direct float64 MaxSim on the reference's)"
  echo "  3. upload     <state>/probe/* to ${OUT_URI%/}/maxsim-probe/ (a failed transfer fails the job)"
  exit 0
fi

AUTH_SCRIPT="${RCP_GCS_AUTH_FILE:-/etc/rcp/gcs_auth.sh}"
[[ -f "$AUTH_SCRIPT" ]] || { echo "maxsim_probe: no auth script at $AUTH_SCRIPT (set RCP_GCS_AUTH_FILE)" >&2; exit 1; }
for helper in "$BOOTSTRAP_SH" "$PROBE_PY" "$GCS_SH" "$GCS_HELPER_PY"; do
  [[ -f "$helper" ]] || { echo "maxsim_probe: missing mounted helper $helper" >&2; exit 1; }
done

# The mounted auth script runs before anything else (executed, never printed): it sets up the credentials
# every transfer path uses.  Then the transfer path is chosen once (jobs/gcs.sh).
set +e
auth_out="$(mktemp)"
bash "$AUTH_SCRIPT" >"$auth_out" 2>&1
auth_status=$?
rm -f "$auth_out"
set -e
if ((auth_status != 0)); then
  echo "maxsim_probe: the GCS auth script failed with exit code $auth_status (its output is not echoed)" >&2
  exit 1
fi
# shellcheck disable=SC1090  # the helper is mounted at a job-specific path
source "$GCS_SH"
gcs_sdk_on_path  # an SDK the auth script installed, searched in RCP_GCLOUD_SDK_DIRS (gcs.sh)
export GCS_PY="${GCS_PY:-$(command -v python3)}"
export GCS_WHEELHOUSE=""  # set once the stage is local (below): the helper prefers the staged gcsfs wheel
export GCS_TOOLS_DIR="${GCS_TOOLS_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/rcp-maxsim-gcs.XXXXXX")}"

STATE="$(mktemp -d "${TMPDIR:-/tmp}/rcp-maxsim.XXXXXX")"
echo "maxsim_probe: state $STATE"

# The wave list the client mechanism, the plugin wheel and the family's reference venv are built from:
# both variants of the one family, so envs runs once and every probe run reuses its venv.
WAVE_LIST="$STATE/maxsim-wave.txt"
tr ',' '\n' <<<"$RECIPES" >"$WAVE_LIST"

# The staged RC: a local stage is its own directory; a gs:// stage arrives at <state>/stage, fetched by
# bootstrap.sh envs itself (the sdpa-probe pattern).
STAGE="$RC_STAGE_URI"
if [[ "$RC_STAGE_URI" == gs://* ]]; then
  STAGE="$STATE/stage"
fi

echo "maxsim_probe: building the node's environments (bootstrap.sh envs)"
bash "$BOOTSTRAP_SH" envs "$RC_STAGE_URI" --state "$STATE" --recipes "$STAGE/recipes" --wave-list "$WAVE_LIST"
if [[ -d "$STAGE/wheelhouse" ]]; then
  export GCS_WHEELHOUSE="$STAGE/wheelhouse"  # the staged gcsfs wheel, preferred over PyPI by the helper
fi

REFERENCE_PYTHON="$STATE/reference/$FAMILY/bin/python"
[[ -x "$REFERENCE_PYTHON" ]] || {
  echo "maxsim_probe: no reference venv at $REFERENCE_PYTHON" >&2
  exit 1
}

PROBE_OUT="$STATE/probe"
mkdir -p "$PROBE_OUT"
status=0
while IFS= read -r recipe; do
  [[ -n "$recipe" ]] || continue
  PAIRS="$STAGE/pairs/$recipe.jsonl"
  [[ -f "$PAIRS" ]] || { echo "maxsim_probe: no staged pairs at $PAIRS" >&2; exit 1; }
  echo "maxsim_probe: running the gate ($recipe)"
  probe_args=(
    --out "$PROBE_OUT/$recipe"
    --recipe "$recipe"
    --pairs "$PAIRS"
    --reference-python "$REFERENCE_PYTHON"
    --port-base "$PORT_BASE"
    --gpus "$GPUS"
    --reference-gpus "$REFERENCE_GPUS"
    --startup-timeout-s "$STARTUP_TIMEOUT"
  )
  if [[ -n "$ROWS" ]]; then probe_args+=(--rows "$ROWS"); fi
  set +e
  "$STATE/client" python "$PROBE_PY" "${probe_args[@]}" >"$STATE/probe-$recipe.log" 2>&1
  run_status=$?
  set -e
  tail -n 40 "$STATE/probe-$recipe.log" || true
  [[ -f "$PROBE_OUT/$recipe/maxsim-report.md" ]] || {
    echo "maxsim_probe: no report written for $recipe (exit $run_status)" >&2
    exit 1
  }
  if ((run_status != 0)); then
    echo "maxsim_probe: $recipe did not pass (exit $run_status); the report and its logs are in $PROBE_OUT/$recipe" >&2
    status=$run_status
  fi
done <"$WAVE_LIST"

echo "maxsim_probe: uploading to ${OUT_URI%/}/maxsim-probe/"
if ! gcs_cp "$PROBE_OUT" "${OUT_URI%/}/maxsim-probe" dir; then
  echo "maxsim_probe: UPLOAD-FAILED: the reports are at $PROBE_OUT on the node and nowhere else" >&2
  exit 1
fi
if ((status != 0)); then
  echo "maxsim_probe: FAILED (a variant's gate did not pass; the reports were uploaded from $PROBE_OUT)" >&2
  exit "$status"
fi

echo "maxsim_probe: DONE (reports uploaded from $PROBE_OUT)"
