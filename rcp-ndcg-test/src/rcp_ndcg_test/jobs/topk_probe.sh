#!/usr/bin/env bash
# topk-embed-v1's controlled GPU probe: the node wrapper for the probe module
# rcp_ndcg_test/jobs/topk_probe.py (mounted beside this script by submit.sh).
#
#   topk_probe.sh <RC_STAGE_URI> <OUT_URI>
#
#     RC_STAGE_URI  gs://.../<rc-name> (or a local path), the RC that rc_build.sh staged; its
#                   wheelhouse/recipes/constraints build the node's client environment and the
#                   family's reference venv (bootstrap.sh envs)
#     OUT_URI       gs://.../waves/<wave>: probe-report.json, probe-report.md and the per-row
#                   reference dumps upload here
#
# In order:
#   1. environments  bootstrap.sh envs with a wave list naming the chosen variant: the engine
#                    environment gains the recipe's plugin wheel (--no-deps, freeze-diff guarded) and
#                    the family's reference venv (sentence-transformers + the checkpoint's own code)
#                    lands at <state>/reference/topk-embed-v1
#   2. check         topk_probe.py --check-engine: the image's vLLM API contract in seconds and no
#                    weights (the single-process switch, the engine's model_executor branch, the
#                    offline pooling entry points); a failure stops the job here
#   3. probe         the engine image's own python runs topk_probe.py: E0 the one-row control (the
#                    wave's recorded vectors when RCP_TOPK_CONTROL_VECTORS names them), E1
#                    preprocessing isolation, E2 the same tensors into the engine tower, E3 layer
#                    dumps, E4 the image-token embeddings, E5 cross-feed, E6 the grid sweep, E7 the
#                    text path layer by layer; the reference side re-executes the same file under
#                    <state>/reference/topk-embed-v1 (on its own GPU)
#   4. upload        probe-report.json, probe-report.md and the per-row dumps; a failed transfer
#                    fails the job (no silent green with no artefacts)
#
# Environment: RCP_GCS_AUTH_FILE (the mounted auth script; executed, never printed), HF_TOKEN (from the
# mounted token file, exported by submit.sh's wrapper; never printed), RCP_BOOTSTRAP_SH /
# RCP_TOPK_PROBE_PY / RCP_TOPK_WAVE_COSINES / RCP_GCS_HELPER_SH / RCP_GCS_HELPER_PY (the mounted
# helpers), RCP_TOPK_VARIANT (small; the variant whose recipe and checkpoint the probe uses),
# RCP_TOPK_ROWS (empty: every image document row), RCP_TOPK_TEXT_ROWS / RCP_TOPK_QUERY_ROWS (empty:
# no text-only row; the named pairs rows are probed as text documents / queries), RCP_TOPK_DEVICE
# (cuda:0; the engine),
# RCP_TOPK_REFERENCE_DEVICE (cuda:1; the reference subprocess), RCP_TOPK_GPU_FRACTION (0.45; vLLM's
# gpu_memory_utilization), RCP_TOPK_DTYPE / RCP_TOPK_REFERENCE_DTYPE (bfloat16; the engine's LLM dtype
# and the checkpoint's own load dtype on the reference side), RCP_TOPK_GDN_BACKEND (unset: the engine's
# own resolver default, `auto` -> flashinfer on this image; `flashinfer` / `triton` / `cutedsl` request
# the tag's --gdn-prefill-backend, and the report records the ENGAGED backend from the loaded layers --
# a refused request falls back inside vLLM and must not be read as that backend's result),
# RCP_TOPK_CONTROL_VECTORS (unset: the
# control is `not_run` and the report says so), RCP_TOPK_MODEL / RCP_TOPK_REVISION (unset: read from the
# staged family.yaml in the client environment, the recipe's own pin).
# TOPK_DRY=1 prints the plan and runs nothing.  The transfer runs jobs/gcs.sh's gcs_cp (the one dispatch
# home: gcloud/gsutil when one is present, else the python gcsfs helper).

set -euo pipefail

usage() { echo "usage: topk_probe.sh <RC_STAGE_URI> <OUT_URI>" >&2; }

RC_STAGE_URI="${1:-}"
OUT_URI="${2:-}"
if [[ -n "$RC_STAGE_URI" && -n "$OUT_URI" ]]; then
  shift 2
else
  usage
  exit 2
fi
(($# == 0)) || { echo "topk_probe: unknown argument: $1" >&2; usage; exit 2; }

BOOTSTRAP_SH="${RCP_BOOTSTRAP_SH:-/etc/rcp/files/bootstrap/bootstrap.sh}"
PROBE_PY="${RCP_TOPK_PROBE_PY:-/etc/rcp/files/topkprobe/topk_probe.py}"
WAVE_COSINES="${RCP_TOPK_WAVE_COSINES:-/etc/rcp/files/topkprobe/wave-cosines-topk-small.json}"
GCS_SH="${RCP_GCS_HELPER_SH:-/etc/rcp/files/gcshelper/gcs.sh}"
GCS_HELPER_PY="${RCP_GCS_HELPER_PY:-/etc/rcp/files/gcshelper/gcs.py}"
VARIANT="${RCP_TOPK_VARIANT:-small}"
ROWS="${RCP_TOPK_ROWS:-}"
TEXT_ROWS="${RCP_TOPK_TEXT_ROWS:-}"
QUERY_ROWS="${RCP_TOPK_QUERY_ROWS:-}"
DEVICE="${RCP_TOPK_DEVICE:-cuda:0}"
REFERENCE_DEVICE="${RCP_TOPK_REFERENCE_DEVICE:-cuda:1}"
GPU_FRACTION="${RCP_TOPK_GPU_FRACTION:-0.45}"
DTYPE="${RCP_TOPK_DTYPE:-bfloat16}"
REFERENCE_DTYPE="${RCP_TOPK_REFERENCE_DTYPE:-bfloat16}"
GDN_BACKEND="${RCP_TOPK_GDN_BACKEND:-}"
CONTROL_VECTORS="${RCP_TOPK_CONTROL_VECTORS:-}"
RECIPE="topk-embed-v1-$VARIANT"

if [[ "${TOPK_DRY:-0}" == "1" ]]; then
  echo "topk_probe (dry): would run, in order, on one node:"
  echo "  1. bootstrap  $BOOTSTRAP_SH envs $RC_STAGE_URI --state <state> --recipes <stage>/recipes"
  echo "                --wave-list <state>/topk-wave.txt ($RECIPE)"
  echo "  2. check      python3 $PROBE_PY --check-engine (the image's vLLM API, seconds and no weights;"
  echo "                a failure stops the job here, before any weights load)"
  echo "  3. probe      python3 $PROBE_PY --out <state>/probe --pairs <stage>/pairs/$RECIPE.jsonl \\"
  echo "                --model <checkpoint> --revision <pinned> --recipe-dir <stage>/recipes/topk-embed-v1 \\"
  echo "                --reference-python <state>/reference/topk-embed-v1/bin/python \\"
  echo "                --device $DEVICE --reference-device $REFERENCE_DEVICE \\"
  echo "                --dtype $DTYPE --reference-dtype $REFERENCE_DTYPE \\"
  echo "                --gpu-memory-fraction $GPU_FRACTION --wave-cosines $WAVE_COSINES"
  [[ -n "$ROWS" ]] && echo "                --rows $ROWS"
  [[ -n "$TEXT_ROWS" ]] && echo "                --text-rows $TEXT_ROWS"
  [[ -n "$QUERY_ROWS" ]] && echo "                --query-rows $QUERY_ROWS"
  [[ -n "$GDN_BACKEND" ]] && echo "                --gdn-backend $GDN_BACKEND"
  [[ -n "$CONTROL_VECTORS" ]] && echo "                --control-vectors $CONTROL_VECTORS"
  echo "  4. upload     <state>/probe/* to ${OUT_URI%/}/topk-probe/ (a failed transfer fails the job)"
  exit 0
fi

AUTH_SCRIPT="${RCP_GCS_AUTH_FILE:-/etc/rcp/gcs_auth.sh}"
[[ -f "$AUTH_SCRIPT" ]] || { echo "topk_probe: no auth script at $AUTH_SCRIPT (set RCP_GCS_AUTH_FILE)" >&2; exit 1; }
for helper in "$BOOTSTRAP_SH" "$PROBE_PY" "$GCS_SH" "$GCS_HELPER_PY"; do
  [[ -f "$helper" ]] || { echo "topk_probe: missing mounted helper $helper" >&2; exit 1; }
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
  echo "topk_probe: the GCS auth script failed with exit code $auth_status (its output is not echoed)" >&2
  exit 1
fi
# shellcheck disable=SC1090  # the helper is mounted at a job-specific path
source "$GCS_SH"
gcs_sdk_on_path  # an SDK the auth script installed, searched in RCP_GCLOUD_SDK_DIRS (gcs.sh)
export GCS_PY="${GCS_PY:-$(command -v python3)}"
export GCS_WHEELHOUSE=""  # set once the stage is local (below): the helper prefers the staged gcsfs wheel
export GCS_TOOLS_DIR="${GCS_TOOLS_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/rcp-topkprobe-gcs.XXXXXX")}"

STATE="$(mktemp -d "${TMPDIR:-/tmp}/rcp-topkprobe.XXXXXX")"
echo "topk_probe: state $STATE"

# The wave list the client/reference environments are built from (one reference venv per family; the
# chosen variant's own recipe is the one that installs its plugin wheel and builds the reference env).
WAVE_LIST="$STATE/topk-wave.txt"
printf '%s\n' "$RECIPE" >"$WAVE_LIST"

# The staged RC: a local stage is its own directory; a gs:// stage arrives at <state>/stage, fetched by
# bootstrap.sh envs itself (the judgecheck pattern).
STAGE="$RC_STAGE_URI"
if [[ "$RC_STAGE_URI" == gs://* ]]; then
  STAGE="$STATE/stage"
fi

echo "topk_probe: building the node's environments (bootstrap.sh envs)"
bash "$BOOTSTRAP_SH" envs "$RC_STAGE_URI" --state "$STATE" --recipes "$STAGE/recipes" --wave-list "$WAVE_LIST"
if [[ -d "$STAGE/wheelhouse" ]]; then
  export GCS_WHEELHOUSE="$STAGE/wheelhouse"  # the staged gcsfs wheel, preferred over PyPI by the helper
fi

REFERENCE_PYTHON="$STATE/reference/topk-embed-v1/bin/python"
[[ -x "$REFERENCE_PYTHON" ]] || {
  echo "topk_probe: no reference venv at $REFERENCE_PYTHON" >&2
  exit 1
}

MODEL="${RCP_TOPK_MODEL:-}"
REVISION="${RCP_TOPK_REVISION:-}"
if [[ -z "$MODEL" || -z "$REVISION" ]]; then
  # The recipe is the one home of the checkpoint pin; read it in the client environment.  The family
  # declares two variants, so the loader is told which one; ``recipe.model`` is the HF repo id (the
  # variant id rides ``recipe.client["model"]``, which is NOT the checkpoint).
  read -r MODEL REVISION < <("$STATE/client" python - "$STAGE/recipes/topk-embed-v1/family.yaml" "$RECIPE" <<'PY'
import sys

from rcp_ndcg_vllm.recipe import load_recipe

recipe = load_recipe(sys.argv[1], variant=sys.argv[2])
print(recipe.model, recipe.revision)
PY
)
fi
[[ -n "$MODEL" && -n "$REVISION" ]] || { echo "topk_probe: no checkpoint pin (set RCP_TOPK_MODEL/RCP_TOPK_REVISION)" >&2; exit 1; }

# The engine image's vLLM API, checked before the weights load: a moved accessor or pooling entry
# point fails here in seconds instead of after a multi-GPU model load (the probe's own report would
# then only carry one traceback).  The check is the probe module's ``--check-engine``.
echo "topk_probe: checking the engine image's vLLM API"
set +e
python3 "$PROBE_PY" --check-engine
check_status=$?
set -e
[[ $check_status -eq 0 ]] || {
  echo "topk_probe: the engine image's vLLM API moved (the check above names it); not loading the weights" >&2
  exit 1
}

PAIRS="$STAGE/pairs/$RECIPE.jsonl"
[[ -f "$PAIRS" ]] || { echo "topk_probe: no staged pairs at $PAIRS" >&2; exit 1; }

PROBE_OUT="$STATE/probe"
mkdir -p "$PROBE_OUT"
echo "topk_probe: running the probe ($RECIPE)"
probe_args=(
  --out "$PROBE_OUT"
  --model "$MODEL"
  --revision "$REVISION"
  --pairs "$PAIRS"
  --recipe-dir "$STAGE/recipes/topk-embed-v1"
  --reference-python "$REFERENCE_PYTHON"
  --device "$DEVICE"
  --reference-device "$REFERENCE_DEVICE"
  --dtype "$DTYPE"
  --reference-dtype "$REFERENCE_DTYPE"
  --gpu-memory-fraction "$GPU_FRACTION"
)
[[ -n "$ROWS" ]] && probe_args+=(--rows "$ROWS")
[[ -n "$TEXT_ROWS" ]] && probe_args+=(--text-rows "$TEXT_ROWS")
[[ -n "$QUERY_ROWS" ]] && probe_args+=(--query-rows "$QUERY_ROWS")
[[ -n "$GDN_BACKEND" ]] && probe_args+=(--gdn-backend "$GDN_BACKEND")
[[ -n "$CONTROL_VECTORS" ]] && probe_args+=(--control-vectors "$CONTROL_VECTORS")
[[ -f "$WAVE_COSINES" ]] && probe_args+=(--wave-cosines "$WAVE_COSINES")
set +e
python3 "$PROBE_PY" "${probe_args[@]}" >"$STATE/probe.log" 2>&1
status=$?
set -e
tail -n 40 "$STATE/probe.log" || true
[[ -f "$PROBE_OUT/probe-report.md" ]] || { echo "topk_probe: no report written (exit $status)" >&2; exit 1; }

echo "topk_probe: uploading to ${OUT_URI%/}/topk-probe/"
if ! gcs_cp "$PROBE_OUT" "${OUT_URI%/}/topk-probe" dir; then
  echo "topk_probe: UPLOAD-FAILED: the report is at $PROBE_OUT on the node and nowhere else" >&2
  exit 1
fi
if ((status != 0)); then
  echo "topk_probe: FAILED (probe exit $status; the partial report was uploaded from $PROBE_OUT)" >&2
  exit "$status"
fi

echo "topk_probe: DONE (report uploaded from $PROBE_OUT)"
