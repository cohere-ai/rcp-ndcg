#!/usr/bin/env bash
# topk-embed-v1's controlled GPU probe: the node wrapper for jobs/investigate/topk_probe.py.
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
#   1. environments  bootstrap.sh envs with a wave list naming the two topk variants: the engine
#                    environment gains the recipe's plugin wheel (--no-deps, freeze-diff guarded) and
#                    the family's reference venv (sentence-transformers + the checkpoint's own code)
#                    lands at <state>/reference/topk-embed-v1
#   2. probe         the engine image's own python runs topk_probe.py: E1 preprocessing isolation,
#                    E2 the same tensors into both towers, E3 layer dumps, E4 the image-token
#                    embeddings, E5 cross-feed, E6 the grid sweep, E7 the attention window; the
#                    reference side re-executes the same file under <state>/reference/topk-embed-v1
#   3. upload        probe-report.json, probe-report.md and the per-row dumps
#
# Environment: RCP_GCS_AUTH_FILE (the mounted auth script; executed, never printed), HF_TOKEN (from the
# mounted token file, exported by submit.sh's wrapper; never printed), RCP_BOOTSTRAP_SH /
# RCP_TOPK_PROBE_PY / RCP_GCS_HELPER_SH / RCP_GCS_HELPER_PY (the mounted helpers), RCP_TOPK_VARIANT
# (small; the variant whose recipe and checkpoint the probe uses), RCP_TOPK_ROWS (empty: every media
# row), RCP_TOPK_DEVICE (cuda:0), RCP_TOPK_MODEL / RCP_TOPK_REVISION (unset: read from the staged
# family.yaml in the client environment, the recipe's own pin).
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
GCS_SH="${RCP_GCS_HELPER_SH:-/etc/rcp/files/gcshelper/gcs.sh}"
GCS_HELPER_PY="${RCP_GCS_HELPER_PY:-/etc/rcp/files/gcshelper/gcs.py}"
VARIANT="${RCP_TOPK_VARIANT:-small}"
ROWS="${RCP_TOPK_ROWS:-}"
DEVICE="${RCP_TOPK_DEVICE:-cuda:0}"
RECIPE="topk-embed-v1-$VARIANT"

if [[ "${TOPK_DRY:-0}" == "1" ]]; then
  echo "topk_probe (dry): would run, in order, on one node:"
  echo "  1. bootstrap  $BOOTSTRAP_SH envs $RC_STAGE_URI --state <state> --recipes <stage>/recipes"
  echo "                --wave-list <state>/topk-wave.txt (topk-embed-v1-small, topk-embed-v1-xsmall)"
  echo "  2. probe      python3 $PROBE_PY --out <state>/probe --pairs <stage>/pairs/$RECIPE.jsonl \\"
  echo "                --model <checkpoint> --revision <pinned> --recipe-dir <stage>/recipes/topk-embed-v1 \\"
  echo "                --reference-python <state>/reference/topk-embed-v1/bin/python --device $DEVICE"
  [[ -n "$ROWS" ]] && echo "                --rows $ROWS"
  echo "  3. upload     <state>/probe/* to ${OUT_URI%/}/topk-probe/"
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

# The wave list the client/reference environments are built from (one reference venv per family).
WAVE_LIST="$STATE/topk-wave.txt"
printf 'topk-embed-v1-small\ntopk-embed-v1-xsmall\n' >"$WAVE_LIST"

# The staged RC: a local stage is its own directory; a gs:// stage arrives at <state>/stage (bootstrap).
STAGE="$RC_STAGE_URI"
if [[ "$RC_STAGE_URI" == gs://* ]]; then
  STAGE="$STATE/stage"
  if command -v gcloud >/dev/null; then
    gcloud storage cp --quiet --recursive "${RC_STAGE_URI%/}/recipes" "${RC_STAGE_URI%/}/pairs" \
      "${RC_STAGE_URI%/}/wheelhouse" "${RC_STAGE_URI%/}/constraints" "$STAGE" 2>/dev/null || true
  fi
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
  # The recipe is the one home of the checkpoint pin; read it in the client environment.
  read -r MODEL REVISION < <("$STATE/client" python - "$STAGE/recipes/topk-embed-v1/family.yaml" <<'PY'
import sys

from rcp_ndcg_vllm.recipe import load_recipe

recipe = load_recipe(sys.argv[1])
print(recipe.client["model"], recipe.revision)
PY
)
fi
[[ -n "$MODEL" && -n "$REVISION" ]] || { echo "topk_probe: no checkpoint pin (set RCP_TOPK_MODEL/RCP_TOPK_REVISION)" >&2; exit 1; }

PAIRS="$STAGE/pairs/$RECIPE.jsonl"
[[ -f "$PAIRS" ]] || { echo "topk_probe: no staged pairs at $PAIRS" >&2; exit 1; }

PROBE_OUT="$STATE/probe"
mkdir -p "$PROBE_OUT"
echo "topk_probe: running the probe ($RECIPE)"
set +e
python3 "$PROBE_PY" \
  --out "$PROBE_OUT" \
  --model "$MODEL" \
  --revision "$REVISION" \
  --pairs "$PAIRS" \
  --recipe-dir "$STAGE/recipes/topk-embed-v1" \
  --reference-python "$REFERENCE_PYTHON" \
  --device "$DEVICE" \
  ${ROWS:+--rows "$ROWS"} \
  >"$STATE/probe.log" 2>&1
status=$?
set -e
tail -n 40 "$STATE/probe.log" || true
[[ -f "$PROBE_OUT/probe-report.md" ]] || { echo "topk_probe: no report written (exit $status)" >&2; exit 1; }

echo "topk_probe: uploading to ${OUT_URI%/}/topk-probe/"
if command -v gcloud >/dev/null; then
  gcloud storage cp --quiet --recursive "$PROBE_OUT"/* "${OUT_URI%/}/topk-probe/" || true
elif command -v gsutil >/dev/null; then
  gsutil -q -m cp -r "$PROBE_OUT"/* "${OUT_URI%/}/topk-probe/" || true
elif [[ -f "$GCS_SH" ]]; then
  # shellcheck disable=SC1090
  source "$GCS_SH"
  upload_tree "$PROBE_OUT" "${OUT_URI%/}/topk-probe" || true
elif [[ -f "$GCS_HELPER_PY" ]]; then
  python3 "$GCS_HELPER_PY" upload-tree "$PROBE_OUT" "${OUT_URI%/}/topk-probe" || true
else
  echo "topk_probe: no transfer path (gcloud, gsutil, $GCS_SH, $GCS_HELPER_PY)" >&2
fi

echo "topk_probe: DONE (probe exit $status; report in $PROBE_OUT)"
