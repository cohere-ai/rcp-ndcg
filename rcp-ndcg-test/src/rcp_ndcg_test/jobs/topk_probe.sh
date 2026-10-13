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
# and the checkpoint's own load dtype on the reference side), RCP_TOPK_CONTROL_VECTORS (unset: the
# control is `not_run` and the report says so), RCP_TOPK_MODEL / RCP_TOPK_REVISION (unset: read from the
# staged family.yaml in the client environment, the recipe's own pin).
# TOPK_DRY=1 prints the plan and runs nothing.  The transfer runs gcloud or gsutil when either is
# present, else the python gcsfs helper (jobs/gcs.sh owns the dispatch).

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
  [[ -n "$CONTROL_VECTORS" ]] && echo "                --control-vectors $CONTROL_VECTORS"
  echo "  4. upload     <state>/probe/* to ${OUT_URI%/}/topk-probe/ (a failed transfer fails the job)"
  exit 0
fi

for helper in "$BOOTSTRAP_SH" "$PROBE_PY"; do
  [[ -f "$helper" ]] || { echo "topk_probe: missing mounted helper $helper" >&2; exit 1; }
done

AUTH_SCRIPT="${RCP_GCS_AUTH_FILE:-/etc/rcp/gcs_auth.sh}"
if [[ -f "$AUTH_SCRIPT" ]]; then
  # shellcheck disable=SC1090
  source "$AUTH_SCRIPT"
fi

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
[[ -n "$CONTROL_VECTORS" ]] && probe_args+=(--control-vectors "$CONTROL_VECTORS")
[[ -f "$WAVE_COSINES" ]] && probe_args+=(--wave-cosines "$WAVE_COSINES")
set +e
python3 "$PROBE_PY" "${probe_args[@]}" >"$STATE/probe.log" 2>&1
status=$?
set -e
tail -n 40 "$STATE/probe.log" || true
[[ -f "$PROBE_OUT/probe-report.md" ]] || { echo "topk_probe: no report written (exit $status)" >&2; exit 1; }

echo "topk_probe: uploading to ${OUT_URI%/}/topk-probe/"
upload() {
  if command -v gcloud >/dev/null; then
    gcloud storage cp --quiet --recursive "$PROBE_OUT"/* "${OUT_URI%/}/topk-probe/"
  elif command -v gsutil >/dev/null; then
    gsutil -q -m cp -r "$PROBE_OUT"/* "${OUT_URI%/}/topk-probe/"
  elif [[ -f "$GCS_SH" ]]; then
    # shellcheck disable=SC1090
    source "$GCS_SH"
    upload_tree "$PROBE_OUT" "${OUT_URI%/}/topk-probe"
  elif [[ -f "$GCS_HELPER_PY" ]]; then
    python3 "$GCS_HELPER_PY" upload-tree "$PROBE_OUT" "${OUT_URI%/}/topk-probe"
  else
    echo "topk_probe: no transfer path (gcloud, gsutil, $GCS_SH, $GCS_HELPER_PY)" >&2
    return 1
  fi
}
if ! upload; then
  echo "topk_probe: UPLOAD-FAILED: the report is at $PROBE_OUT on the node and nowhere else" >&2
  exit 1
fi

echo "topk_probe: DONE (probe exit $status; report uploaded from $PROBE_OUT)"
