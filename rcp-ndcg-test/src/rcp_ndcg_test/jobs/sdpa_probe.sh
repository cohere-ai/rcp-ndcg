#!/usr/bin/env bash
# D4's probe: the plugin's TORCH_SDPA backend against the engine's own attention path, one node, one GPU.
# The node wrapper for the probe module rcp_ndcg_test/jobs/sdpa_probe.py (mounted beside this script by
# submit.sh).  The decision rule this job serves is written before it runs: DECISION-D4.md in the
# migration campaign's exec/ directory; the report carries the same rule and the same gate as data.
#
#   sdpa_probe.sh <RC_STAGE_URI> <OUT_URI>
#
#     RC_STAGE_URI  gs://.../<rc-name> (or a local path), the RC that rc_build.sh staged; its wheelhouse
#                   builds the node's client mechanism and its recipes install the plugin wheel into the
#                   engine environment (bootstrap.sh envs)
#     OUT_URI       gs://.../waves/<wave>: sdpa-report.json, sdpa-report.md, sdpa-vectors.json and the two
#                   arms' serve logs upload here
#
# In order:
#   1. environments  bootstrap.sh envs with a wave list naming the probed recipe: the client mechanism (the
#                    product + the harness, from the staged wheelhouse) and the recipe's plugin wheel in the
#                    engine environment (--no-deps, freeze-diff guarded)
#   2. probe         the client mechanism runs sdpa_probe.py: it starts the recipe's engine twice on the same
#                    port, the second start with the `torch-sdpa-attention` patch and its
#                    `--attention-backend TORCH_SDPA` pin removed, sends the recipe's pairs through the
#                    product's own role client to each arm (twice per arm: the null control), compares the
#                    pooled vectors against the recorded gate, and writes the report + the verdict
#   3. upload        the report, the vectors and the serve logs; a failed transfer fails the job (no silent
#                    green with no artefacts)
#
# Environment: RCP_GCS_AUTH_FILE (the mounted auth script; executed, never printed), HF_TOKEN (from the
# mounted token file, exported by submit.sh's wrapper; never printed), RCP_BOOTSTRAP_SH / RCP_SDPA_PROBE_PY /
# RCP_GCS_HELPER_SH / RCP_GCS_HELPER_PY (the mounted helpers), RCP_SDPA_RECIPE (embeddinggemma-2; the variant
# probed), RCP_SDPA_ROWS (empty: every pairs row), RCP_SDPA_PORT_BASE (8100; both arms serve on it in turn),
# RCP_SDPA_GPUS (0; CUDA_VISIBLE_DEVICES for the engine), RCP_SDPA_STARTUP_TIMEOUT_S (1800; one engine's load
# budget).  SDPA_DRY=1 prints the plan and runs nothing.  The transfer runs jobs/gcs.sh's gcs_cp (the one
# dispatch home: gcloud/gsutil when one is present, else the python gcsfs helper).

set -euo pipefail

usage() { echo "usage: sdpa_probe.sh <RC_STAGE_URI> <OUT_URI>" >&2; }

RC_STAGE_URI="${1:-}"
OUT_URI="${2:-}"
if [[ -n "$RC_STAGE_URI" && -n "$OUT_URI" ]]; then
  shift 2
else
  usage
  exit 2
fi
(($# == 0)) || { echo "sdpa_probe: unknown argument: $1" >&2; usage; exit 2; }

BOOTSTRAP_SH="${RCP_BOOTSTRAP_SH:-/etc/rcp/files/bootstrap/bootstrap.sh}"
PROBE_PY="${RCP_SDPA_PROBE_PY:-/etc/rcp/files/sdpaprobe/sdpa_probe.py}"
GCS_SH="${RCP_GCS_HELPER_SH:-/etc/rcp/files/gcshelper/gcs.sh}"
GCS_HELPER_PY="${RCP_GCS_HELPER_PY:-/etc/rcp/files/gcshelper/gcs.py}"
RECIPE="${RCP_SDPA_RECIPE:-embeddinggemma-2}"
ROWS="${RCP_SDPA_ROWS:-}"
PORT_BASE="${RCP_SDPA_PORT_BASE:-8100}"
GPUS="${RCP_SDPA_GPUS:-0}"
STARTUP_TIMEOUT="${RCP_SDPA_STARTUP_TIMEOUT_S:-1800}"

if [[ "${SDPA_DRY:-0}" == "1" ]]; then
  echo "sdpa_probe (dry): would run, in order, on one node:"
  echo "  1. bootstrap  $BOOTSTRAP_SH envs $RC_STAGE_URI --state <state> --recipes <stage>/recipes"
  echo "                --wave-list <state>/sdpa-wave.txt ($RECIPE)"
  echo "  2. probe      <state>/client python $PROBE_PY --out <state>/probe --recipe $RECIPE \\"
  echo "                --pairs <stage>/pairs/$RECIPE.jsonl --port-base $PORT_BASE --gpus $GPUS \\"
  echo "                --startup-timeout-s $STARTUP_TIMEOUT"
  if [[ -n "$ROWS" ]]; then echo "                --rows $ROWS"; fi
  echo "                (two engine starts: the recipe as declared, then with $RECIPE's torch-sdpa-attention"
  echo "                 patch and its --attention-backend TORCH_SDPA pin removed)"
  echo "  3. upload     <state>/probe/* to ${OUT_URI%/}/sdpa-probe/ (a failed transfer fails the job)"
  exit 0
fi

AUTH_SCRIPT="${RCP_GCS_AUTH_FILE:-/etc/rcp/gcs_auth.sh}"
[[ -f "$AUTH_SCRIPT" ]] || { echo "sdpa_probe: no auth script at $AUTH_SCRIPT (set RCP_GCS_AUTH_FILE)" >&2; exit 1; }
for helper in "$BOOTSTRAP_SH" "$PROBE_PY" "$GCS_SH" "$GCS_HELPER_PY"; do
  [[ -f "$helper" ]] || { echo "sdpa_probe: missing mounted helper $helper" >&2; exit 1; }
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
  echo "sdpa_probe: the GCS auth script failed with exit code $auth_status (its output is not echoed)" >&2
  exit 1
fi
# shellcheck disable=SC1090  # the helper is mounted at a job-specific path
source "$GCS_SH"
gcs_sdk_on_path  # an SDK the auth script installed, searched in RCP_GCLOUD_SDK_DIRS (gcs.sh)
export GCS_PY="${GCS_PY:-$(command -v python3)}"
export GCS_WHEELHOUSE=""  # set once the stage is local (below): the helper prefers the staged gcsfs wheel
export GCS_TOOLS_DIR="${GCS_TOOLS_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/rcp-sdpa-gcs.XXXXXX")}"

STATE="$(mktemp -d "${TMPDIR:-/tmp}/rcp-sdpa.XXXXXX")"
echo "sdpa_probe: state $STATE"

# The wave list the client mechanism and the engine's plugin install are built from.
WAVE_LIST="$STATE/sdpa-wave.txt"
printf '%s\n' "$RECIPE" >"$WAVE_LIST"

# The staged RC: a local stage is its own directory; a gs:// stage arrives at <state>/stage, fetched by
# bootstrap.sh envs itself.
STAGE="$RC_STAGE_URI"
if [[ "$RC_STAGE_URI" == gs://* ]]; then
  STAGE="$STATE/stage"
fi

echo "sdpa_probe: building the node's environments (bootstrap.sh envs)"
bash "$BOOTSTRAP_SH" envs "$RC_STAGE_URI" --state "$STATE" --recipes "$STAGE/recipes" --wave-list "$WAVE_LIST"
if [[ -d "$STAGE/wheelhouse" ]]; then
  export GCS_WHEELHOUSE="$STAGE/wheelhouse"  # the staged gcsfs wheel, preferred over PyPI by the helper
fi

PAIRS="$STAGE/pairs/$RECIPE.jsonl"
[[ -f "$PAIRS" ]] || { echo "sdpa_probe: no staged pairs at $PAIRS" >&2; exit 1; }

PROBE_OUT="$STATE/probe"
mkdir -p "$PROBE_OUT"
echo "sdpa_probe: running the probe ($RECIPE)"
probe_args=(
  --out "$PROBE_OUT"
  --recipe "$RECIPE"
  --pairs "$PAIRS"
  --port-base "$PORT_BASE"
  --gpus "$GPUS"
  --startup-timeout-s "$STARTUP_TIMEOUT"
)
if [[ -n "$ROWS" ]]; then probe_args+=(--rows "$ROWS"); fi
set +e
"$STATE/client" python "$PROBE_PY" "${probe_args[@]}" >"$STATE/probe.log" 2>&1
status=$?
set -e
tail -n 40 "$STATE/probe.log" || true
[[ -f "$PROBE_OUT/sdpa-report.md" ]] || { echo "sdpa_probe: no report written (exit $status)" >&2; exit 1; }

echo "sdpa_probe: uploading to ${OUT_URI%/}/sdpa-probe/"
if ! gcs_cp "$PROBE_OUT" "${OUT_URI%/}/sdpa-probe" dir; then
  echo "sdpa_probe: UPLOAD-FAILED: the report is at $PROBE_OUT on the node and nowhere else" >&2
  exit 1
fi
if ((status != 0)); then
  echo "sdpa_probe: FAILED (probe exit $status; the partial report was uploaded from $PROBE_OUT)" >&2
  exit "$status"
fi

echo "sdpa_probe: DONE (report uploaded from $PROBE_OUT)"
