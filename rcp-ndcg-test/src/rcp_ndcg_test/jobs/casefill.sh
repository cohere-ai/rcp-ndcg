#!/usr/bin/env bash
# The cuda reference-case fills: the four recipes whose pending reference cases need a GPU reference
# (the release checklist's "fill every pending_gpu expected value"; submitted as submit.sh --script
# casefill).
#
#   casefill.sh <RC_STAGE_URI> <OUT_URI>
#
#     RC_STAGE_URI  gs://.../<rc-name> (or a local path), the RC that rc_build.sh staged; its staged
#                   cases/ tree (rc_build.sh stages rcp-ndcg-test/cases) is what the fills edit -- a copy
#                   in the state directory, so the stage itself stays as built
#     OUT_URI       gs://.../waves/<wave>: the edited case YAMLs (<OUT_URI>/cases/<recipe>/<case>.yaml)
#                   and casefill-report.json upload here; the operator reviews and copies the YAMLs into
#                   the repository (a reviewed step, never done on the node)
#
# In order, failing fast at the first failure:
#   1. environments  bootstrap.sh envs with a wave list of the four recipes: the client mechanism and the
#                    per-family reference venvs the fills run (one venv per family, owner decision 35)
#   2. fills         per recipe, rcp_ndcg_test.case_fill through the client wrapper with --device cuda
#                    over the staged cases copy; a recipe whose fill fails is recorded and the next
#                    recipe still runs, and every fill's report (what it filled, what it skipped and why)
#                    merges into the job's report
#   3. upload        the edited YAMLs only (the media inputs are the repository's) and the report
#
# A pending case the fill cannot produce is skipped, its file untouched and its reason recorded: a
# media-bearing case's values cannot come from the text-only reference (the reference's media mode
# reports input facts, not model outputs), so a media-value fill is a follow-up; a kind the role's mode
# does not produce, and a kind: none case, carry no values by design.  The job exits 1 when any recipe
# failed (after uploading everything it filled); it exits 0 when the four recipes' fills ran, even when
# cases were skipped.
#
# The four recipes and their reference family directories (the venv keys under <state>/reference/); the
# pairs are pinned to the recipe catalog and the committed cases by tests/test_jobs_scripts.py:
#   topk-embed-v1-small   -> topk-embed-v1     (8 text fills; 6 media + 1 kind: none skipped)
#   qwen3-embedding-0.6b  -> qwen3-embedding   (8 text fills)
#   qwen3-vl-embedding-2b -> qwen3-vl-embedding (5 text fills; 4 media skipped)
#   qwen3-vl-reranker-2b  -> qwen3-vl-reranker  (5 text fills; 4 media skipped)
#
# Environment: RCP_GCS_AUTH_FILE (the mounted auth script; executed, never printed), HF_TOKEN (from the
# mounted token file, exported by submit.sh's wrapper; never printed), RCP_BOOTSTRAP_SH / RCP_REPORT_PY /
# RCP_GCS_HELPER_SH / RCP_GCS_HELPER_PY (the mounted helpers), RCP_CASEFILL_BUDGET_S (the reference
# subprocess's wall-clock budget, 7200 by default -- a VL reference's model load is minutes) and
# RCP_CASEFILL_DATE (the date the notes record, today by default).  CASEFILL_DRY=1 prints the plan and
# runs nothing.  The transfer runs gcloud or gsutil when either is present, else the python gcsfs helper
# (jobs/gcs.sh owns the dispatch); a Cloud SDK the auth script installs is put on PATH from the first of
# RCP_GCLOUD_SDK_DIRS holding a CLI.

set -euo pipefail

usage() { echo "usage: casefill.sh <RC_STAGE_URI> <OUT_URI>" >&2; }

RC_STAGE_URI="${1:-}"
OUT_URI="${2:-}"
if [[ -n "$RC_STAGE_URI" && -n "$OUT_URI" ]]; then
  shift 2
else
  usage
  exit 2
fi
(($# == 0)) || { echo "casefill: unknown argument: $1" >&2; usage; exit 2; }

# The four recipes and their family directories (the reference venv keys); pinned to the catalog by the test.
RECIPES=(
  "topk-embed-v1-small topk-embed-v1"
  "qwen3-embedding-0.6b qwen3-embedding"
  "qwen3-vl-embedding-2b qwen3-vl-embedding"
  "qwen3-vl-reranker-2b qwen3-vl-reranker"
)

BOOTSTRAP_SH="${RCP_BOOTSTRAP_SH:-/etc/rcp/files/bootstrap/bootstrap.sh}"
REPORT_PY="${RCP_REPORT_PY:-/etc/rcp/files/report/report.py}"
GCS_SH="${RCP_GCS_HELPER_SH:-/etc/rcp/files/gcshelper/gcs.sh}"
GCS_HELPER_PY="${RCP_GCS_HELPER_PY:-/etc/rcp/files/gcshelper/gcs.py}"
AUTH_SCRIPT="${RCP_GCS_AUTH_FILE:-/etc/rcp/gcs_auth.sh}"
BUDGET_S="${RCP_CASEFILL_BUDGET_S:-7200}"
FILLED_AT="${RCP_CASEFILL_DATE:-$(date -u +%F)}"

if [[ "${CASEFILL_DRY:-0}" == "1" ]]; then
  echo "casefill (dry): would run, in order, on one node, failing fast at the first failure:"
  echo "  1. bootstrap  $BOOTSTRAP_SH envs $RC_STAGE_URI --state <state> --recipes <stage>/recipes"
  echo "                --wave-list <state>/case-fill-wave.txt (one recipe id per line:"
  printf '                %s\n' "${RECIPES[@]%% *}"
  echo "  2. fills      \$STATE/client python -m rcp_ndcg_test.case_fill per recipe:"
  for row in "${RECIPES[@]}"; do
    read -r recipe family <<<"$row"
    echo "                --recipe $recipe --reference-python <state>/reference/$family/bin/python \\"
    echo "                --device cuda --filled-at $FILLED_AT --budget-s $BUDGET_S --cases-root <stage>/cases"
  done
  echo "  3. upload     <state>/filled-cases/<recipe>/*.yaml -> $OUT_URI/cases/<recipe>/ and the report"
  echo "                to $OUT_URI/casefill-report.json"
  echo "casefill (dry): the plan only; nothing ran"
  exit 0
fi

command -v python3 >/dev/null || { echo "casefill: python3 is not on PATH (an assumption of the node)" >&2; exit 1; }
[[ -f "$AUTH_SCRIPT" ]] || { echo "casefill: no auth script at $AUTH_SCRIPT (set RCP_GCS_AUTH_FILE)" >&2; exit 1; }
[[ -f "$BOOTSTRAP_SH" && -f "$REPORT_PY" && -f "$GCS_SH" && -f "$GCS_HELPER_PY" ]] || {
  echo "casefill: a mounted helper is missing (bootstrap.sh: $BOOTSTRAP_SH, report.py: $REPORT_PY," \
    "gcs.sh: $GCS_SH, gcs.py: $GCS_HELPER_PY)" >&2
  exit 1
}

# The mounted auth script runs before anything else (executed, never printed): it sets up the
# credentials every transfer path uses.  Then the transfer path is chosen once (jobs/gcs.sh).
set +e
auth_out="$(mktemp)"
bash "$AUTH_SCRIPT" >"$auth_out" 2>&1
auth_status=$?
rm -f "$auth_out"
set -e
if ((auth_status != 0)); then
  echo "casefill: the GCS auth script failed with exit code $auth_status (its output is not echoed)" >&2
  exit 1
fi
# shellcheck disable=SC1090  # the helper is mounted at a job-specific path
source "$GCS_SH"
gcs_sdk_on_path  # an SDK the auth script installed, searched in RCP_GCLOUD_SDK_DIRS (gcs.sh)
export GCS_PY="${GCS_PY:-$(command -v python3)}"
export GCS_WHEELHOUSE=""  # the stage is not local yet; the helper fetches gcsfs from PyPI if it must
export GCS_TOOLS_DIR="${GCS_TOOLS_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/rcp-casefill-gcs.XXXXXX")}"

STATE="$(mktemp -d "${TMPDIR:-/tmp}/rcp-casefill.XXXXXX")"
STARTED="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "casefill: state $STATE"

# The wave list the reference environments are built from (bootstrap envs resolves one family per recipe).
WAVE_LIST="$STATE/case-fill-wave.txt"
for row in "${RECIPES[@]}"; do
  printf '%s\n' "${row%% *}"
done >"$WAVE_LIST"

# The staged RC: a local stage is its own directory; a gs:// stage arrives at <state>/stage (bootstrap).
if [[ -d "$RC_STAGE_URI" ]]; then
  STAGE="$RC_STAGE_URI"
else
  STAGE="$STATE/stage"
fi

echo "casefill: building the node's environments and the four families' reference venvs (bootstrap.sh envs)"
bash "$BOOTSTRAP_SH" envs "$RC_STAGE_URI" --state "$STATE" --recipes "$STAGE/recipes" --wave-list "$WAVE_LIST"

[[ -f "$STAGE/manifest.json" ]] || { echo "casefill: $STAGE holds no manifest.json (not an rc_build.sh stage)" >&2; exit 1; }
[[ -d "$STAGE/cases" ]] || {
  echo "casefill: $STAGE holds no cases/ (rc_build.sh stages rcp-ndcg-test/cases as <stage>/cases; rebuild the RC)" >&2
  exit 1
}

# The fills edit a copy of the staged cases tree: a local stage is the operator's checkout.
CASES="$STATE/cases"
mkdir -p "$CASES"
cp -r "$STAGE/cases"/. "$CASES"/

REPORT="$STATE/casefill-report.json"
python3 "$REPORT_PY" init --file "$REPORT" --schema rcp-ndcg.casefill-report.v1 --started "$STARTED"
python3 "$REPORT_PY" merge --file "$REPORT" --key bootstrap --fragment "$STATE/bootstrap.json"

# fragment RECIPE ERROR: one failed recipe's entry, JSON-quoted (the combiner below merges the entries
# under the dot-free ``fills`` key -- a recipe id contains dots, which report.py's --key reads as a path).
fragment() {
  python3 - "$1" "$2" "$STATE/casefill-$1.fragment.json" <<'PYEOF'
import json
import sys

print(json.dumps({"state": "failed", "error": sys.argv[2]}), file=open(sys.argv[3], "w", encoding="utf-8"))
PYEOF
}

FAILED_RECIPES=()
for row in "${RECIPES[@]}"; do
  read -r recipe family <<<"$row"
  reference_python="$STATE/reference/$family/bin/python"
  if [[ ! -x "$reference_python" ]]; then
    echo "casefill: no reference environment for family $family at $reference_python" >&2
    fragment "$recipe" "no reference environment for family $family at $reference_python"
    FAILED_RECIPES+=("$recipe")
    continue
  fi
  echo "casefill: filling $recipe ($family) with the reference on cuda"
  if "$STATE/client" python -m rcp_ndcg_test.case_fill \
      --recipe "$recipe" --reference-python "$reference_python" --device cuda \
      --filled-at "$FILLED_AT" --budget-s "$BUDGET_S" --cases-root "$CASES" \
      >"$STATE/casefill-$recipe.json" 2>"$STATE/casefill-$recipe.err"; then
    : # the fill's own JSON report is the entry the combiner merges
  else
    status=$?
    reason="$(tail -1 "$STATE/casefill-$recipe.err" | tr -d '\r')"
    echo "casefill: $recipe failed (exit $status): $reason" >&2
    fragment "$recipe" "exit $status: $reason"
    FAILED_RECIPES+=("$recipe")
  fi
done

# The report's fills section: every recipe's own report, or its failure entry (a failed fill's stdout
# file exists but is empty, so the fragment is checked first).
python3 - "$STATE" "${RECIPES[@]}" <<'PYEOF'
import json
import sys
from pathlib import Path

state = Path(sys.argv[1])
fills: dict[str, object] = {}
for row in sys.argv[2:]:
    recipe = row.split(" ", 1)[0]
    fragment = state / f"casefill-{recipe}.fragment.json"
    report = state / f"casefill-{recipe}.json"
    if fragment.is_file():
        fills[recipe] = json.loads(fragment.read_text(encoding="utf-8"))
    elif report.is_file():
        fills[recipe] = json.loads(report.read_text(encoding="utf-8"))
    else:  # pragma: no cover - a fill that wrote neither is recorded, never silently dropped
        fills[recipe] = {"state": "failed", "error": "no fill report was written"}
(state / "fills.json").write_text(json.dumps(fills), encoding="utf-8")
PYEOF
python3 "$REPORT_PY" merge --file "$REPORT" --key fills --fragment "$STATE/fills.json"

# The edited YAMLs only (the report names them): <state>/filled-cases/<recipe>/<case>.yaml.
python3 - "$STATE" "${RECIPES[@]}" <<'PYEOF'
import json
import shutil
import sys
from pathlib import Path

state = Path(sys.argv[1])
for row in sys.argv[2:]:
    recipe = row.split(" ", 1)[0]
    report_path = state / f"casefill-{recipe}.json"
    if not report_path.is_file():
        continue
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        continue  # a failed fill wrote no report; its record is in the job report
    out = state / "filled-cases" / recipe
    for entry in report.get("filled", []):
        out.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(entry["path"], out / Path(entry["path"]).name)
PYEOF
if [[ -d "$STATE/filled-cases" ]]; then
  echo "casefill: uploading the edited case YAMLs to $OUT_URI/cases/"
  gcs_cp "$STATE/filled-cases" "$OUT_URI/cases" dir
fi

if ((${#FAILED_RECIPES[@]})); then
  python3 "$REPORT_PY" fail --file "$REPORT" --step casefill --reason "the fills failed for: ${FAILED_RECIPES[*]}"
fi
python3 "$REPORT_PY" emit --file "$REPORT"
gcs_cp "$REPORT" "$OUT_URI/casefill-report.json" file
if ((${#FAILED_RECIPES[@]})); then
  echo "casefill: ${#FAILED_RECIPES[@]} recipe(s) failed: ${FAILED_RECIPES[*]} (their fills are recorded in the report)" >&2
  exit 1
fi
echo "casefill: the four recipes' fills finished; the edited YAMLs and the report are under $OUT_URI/"
