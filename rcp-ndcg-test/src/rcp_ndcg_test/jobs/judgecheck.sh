#!/usr/bin/env bash
# The release checklist's judge item: `rcp-ndcg judge check` against each judge recipe's served engine,
# in the recipe's documented thinking modes, with the engine's media token count (submitted as submit.sh
# --script judgecheck).
#
#   judgecheck.sh <RC_STAGE_URI> <OUT_URI>
#
#     RC_STAGE_URI  gs://.../<rc-name> (or a local path), the RC that rc_build.sh staged; its staged
#                   wheelhouse/recipes/constraints build the node's client environment (bootstrap.sh envs)
#     OUT_URI       gs://.../waves/<wave>: the per-judge JSONs (<OUT_URI>/<recipe>/<mode>.json and
#                   <OUT_URI>/<recipe>/media.json) and judgecheck-report.json upload here
#
# In order, continuing past a failed judge (a failed judge is recorded and the next still runs):
#   1. environments  bootstrap.sh envs with a wave list of the ten judges: the client environment
#                    (rcp-ndcg + rcp-ndcg-test + rcp-ndcg-vllm from the staged wheels) every command runs in
#   2. per judge      serve the recipe's own argv (`rcp-ndcg-vllm serve <id> --port <N>`), wait for
#                     GET /v1/models within RCP_JUDGECHECK_STARTUP_S, then, with the engine up:
#                       thinking modes:  `rcp-ndcg judge check --judge <id> --judge-model <id>
#                                        --set judge.base_url=http://127.0.0.1:<N>/v1
#                                        [--set judge.extra_body=<the mode's switch>] --json`
#                       the media probe:      `python -m rcp_ndcg_test.jobs.judgecheck_probe media ...`
#                                             (one prepared probe image beside its no-media baseline; the
#                                             engine's media delta against the client's counted tokens)
#                     stop the engine's process group, record the verdict, the engine's version and the
#                     NVFP4/Marlin backend lines from the engine log, move on
#   3. upload        the per-judge JSONs and the report
#
# The thinking modes are the recipe's documented ones (jobs/judgecheck_probe.py owns the table): the
# Gemma 4 judges' two (off, the shipped default; on, the recipe's declared `chat_template_kwargs`
# `enable_thinking` choice) and every other judge's single documented mode (the shipped client block, no
# toggle passed -- a guessed flag would silently run another instrument).  `--set judge.extra_body`
# REPLACES the whole block, so a toggle body is the recipe's own `extra_body` with the switch merged in.
#
# Environment: RCP_GCS_AUTH_FILE (the mounted auth script; executed, never printed), HF_TOKEN (from the
# mounted token file, exported by submit.sh's wrapper; never printed), RCP_BOOTSTRAP_SH / RCP_REPORT_PY /
# RCP_JUDGECHECK_PROBE_PY / RCP_GCS_HELPER_SH / RCP_GCS_HELPER_PY (the mounted helpers),
# RCP_JUDGECHECK_PORT_BASE (8100; judge i serves on base + i), RCP_JUDGECHECK_STARTUP_S (3600; the
# /v1/models wait) and RCP_JUDGECHECK_CHECK_S / RCP_JUDGECHECK_MEDIA_S (900; the per-command budgets).
# JUDGECHECK_DRY=1 prints the plan and runs nothing.  The transfer runs gcloud or gsutil when either is
# present, else the python gcsfs helper (jobs/gcs.sh owns the dispatch).

set -euo pipefail

usage() { echo "usage: judgecheck.sh <RC_STAGE_URI> <OUT_URI>" >&2; }

RC_STAGE_URI="${1:-}"
OUT_URI="${2:-}"
if [[ -n "$RC_STAGE_URI" && -n "$OUT_URI" ]]; then
  shift 2
else
  usage
  exit 2
fi
(($# == 0)) || { echo "judgecheck: unknown argument: $1" >&2; usage; exit 2; }

# The ten shipped judges, the release checklist's six first (a time-boxed run covers them before the four
# Gemma 4 judges); the ids are pinned to the recipe catalog by tests/test_jobs_scripts.py.  The thinking
# modes per judge are the helper's documented table (jobs/judgecheck_probe.py), read at run time.
RECIPES=(
  "qwen3.5-397b-a17b-nvfp4"
  "gpt-oss-120b"
  "qwen3.6-27b-fp8"
  "qwen3.8-27b-fp8"
  "qwen3.8-flash-next-fp8"
  "qwen3.8-flash-next-nvfp4"
  "gemma-4-12b-it"
  "gemma-4-26b-a4b-it"
  "gemma-4-26b-a4b-nvfp4"
  "gemma-4-31b-it-nvfp4"
)

BOOTSTRAP_SH="${RCP_BOOTSTRAP_SH:-/etc/rcp/files/bootstrap/bootstrap.sh}"
REPORT_PY="${RCP_REPORT_PY:-/etc/rcp/files/report/report.py}"
PROBE_PY="${RCP_JUDGECHECK_PROBE_PY:-/etc/rcp/files/judgecheckprobe/judgecheck_probe.py}"
GCS_SH="${RCP_GCS_HELPER_SH:-/etc/rcp/files/gcshelper/gcs.sh}"
GCS_HELPER_PY="${RCP_GCS_HELPER_PY:-/etc/rcp/files/gcshelper/gcs.py}"
AUTH_SCRIPT="${RCP_GCS_AUTH_FILE:-/etc/rcp/gcs_auth.sh}"
PORT_BASE="${RCP_JUDGECHECK_PORT_BASE:-8100}"
STARTUP_S="${RCP_JUDGECHECK_STARTUP_S:-3600}"
CHECK_S="${RCP_JUDGECHECK_CHECK_S:-900}"
MEDIA_S="${RCP_JUDGECHECK_MEDIA_S:-900}"

if [[ "${JUDGECHECK_DRY:-0}" == "1" ]]; then
  echo "judgecheck (dry): would run, in order, on one node, continuing past a failure:"
  echo "  1. bootstrap  $BOOTSTRAP_SH envs $RC_STAGE_URI --state <state> --recipes <stage>/recipes"
  echo "                --wave-list <state>/judgecheck-wave.txt (one recipe id per line:"
  printf '                %s\n' "${RECIPES[@]}"
  echo "  2. per judge: serve the recipe, wait for /v1/models, the recipe's documented thinking modes, the media probe, stop"
  echo "                the modes are the helper's documented table: the Gemma 4 judges' two (off, on), the others'"
  echo "                single documented mode (default, the shipped client block -- their recipes document no toggle)"
  for index in "${!RECIPES[@]}"; do
    recipe="${RECIPES[$index]}"
    port=$((PORT_BASE + index))
    echo "                serve    <state>/client rcp-ndcg-vllm serve $recipe --port $port"
    echo "                wait     GET http://127.0.0.1:$port/v1/models within ${STARTUP_S}s"
    echo "                modes    <state>/client python -m rcp_ndcg_test.jobs.judgecheck_probe modes --judge $recipe"
    echo "                check    per mode: <state>/client rcp-ndcg judge check --judge $recipe --judge-model $recipe \\"
    echo "                         --set judge.base_url=http://127.0.0.1:$port/v1 [--set judge.extra_body=<the mode's body>] --json"
    echo "                media    <state>/client python -m rcp_ndcg_test.jobs.judgecheck_probe media --judge $recipe \\"
    echo "                         --judge-model $recipe --set judge.base_url=http://127.0.0.1:$port/v1 --out <state>/media-$recipe.json"
    echo "                stop     kill the engine's process group"
  done
  echo "  3. upload     <state>/check-<recipe>-<mode>.json -> $OUT_URI/<recipe>/<mode>.json, the media JSONs and"
  echo "                the report to $OUT_URI/judgecheck-report.json"
  echo "judgecheck (dry): the plan only; nothing ran"
  exit 0
fi

command -v python3 >/dev/null || { echo "judgecheck: python3 is not on PATH (an assumption of the node)" >&2; exit 1; }
command -v setsid >/dev/null || {
  echo "judgecheck: setsid is not on PATH; the engine's process group is stopped through it" >&2
  exit 1
}
[[ -f "$AUTH_SCRIPT" ]] || { echo "judgecheck: no auth script at $AUTH_SCRIPT (set RCP_GCS_AUTH_FILE)" >&2; exit 1; }
[[ -f "$BOOTSTRAP_SH" && -f "$REPORT_PY" && -f "$PROBE_PY" && -f "$GCS_SH" && -f "$GCS_HELPER_PY" ]] || {
  echo "judgecheck: a mounted helper is missing (bootstrap.sh: $BOOTSTRAP_SH, report.py: $REPORT_PY," \
    "judgecheck_probe.py: $PROBE_PY, gcs.sh: $GCS_SH, gcs.py: $GCS_HELPER_PY)" >&2
  exit 1
}

# The mounted auth script runs before anything else (executed, never printed): it sets up the credentials
# every transfer path uses.  Then the transfer path is chosen once (jobs/gcs.sh).
set +e
auth_out="$(mktemp)"
bash "$AUTH_SCRIPT" >"$auth_out" 2>&1
auth_status=$?
rm -f "$auth_out"
set -e
if ((auth_status != 0)); then
  echo "judgecheck: the GCS auth script failed with exit code $auth_status (its output is not echoed)" >&2
  exit 1
fi
# shellcheck disable=SC1090  # the helper is mounted at a job-specific path
source "$GCS_SH"
gcs_sdk_on_path  # an SDK the auth script installed, searched in RCP_GCLOUD_SDK_DIRS (gcs.sh)
export GCS_PY="${GCS_PY:-$(command -v python3)}"
export GCS_WHEELHOUSE=""  # the stage is not local yet; the helper fetches gcsfs from PyPI if it must
export GCS_TOOLS_DIR="${GCS_TOOLS_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/rcp-judgecheck-gcs.XXXXXX")}"

STATE="$(mktemp -d "${TMPDIR:-/tmp}/rcp-judgecheck.XXXXXX")"
STARTED="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "judgecheck: state $STATE"

# wait_models PORT TIMEOUT_S PID: 0 when GET /v1/models answers 2xx with a JSON body within the timeout;
# 2 when the engine's whole process group is gone (a dead engine fails fast, never the full timeout);
# 1 at the timeout.
wait_models() {
  python3 - "$1" "$2" "$3" <<'PYEOF'
import json
import os
import sys
import time
import urllib.error
import urllib.request

port, timeout, pid = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
start = time.monotonic()
deadline = start + timeout
ready = False
while not ready and time.monotonic() < deadline:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as reply:  # noqa: S310
            if 200 <= reply.status < 300:
                json.loads(reply.read().decode("utf-8"))
                ready = True
    except (urllib.error.URLError, OSError, ValueError):
        pass
    if not ready:
        try:  # the session's process group: the engine's workers live in it, the launcher may have exited
            os.killpg(pid, 0)
        except ProcessLookupError:
            sys.exit(2)
        time.sleep(1 if time.monotonic() - start < 60 else 5)  # a fast first minute, then the slow poll
sys.exit(0 if ready else 1)
PYEOF
}

# stop_engine PID: SIGTERM the engine's process group (the session setsid started), then SIGKILL it once
# the grace period is up -- whether or not the leader already exited: a tensor-parallel worker that
# outlives an OOM-killed API server keeps the GPUs the next judge needs.
stop_engine() {
  local pid="$1"
  kill -TERM -- "-$pid" 2>/dev/null || true
  for _ in 1 2 3 4 5 6; do
    kill -0 -- "-$pid" 2>/dev/null || break  # the whole group is gone
    sleep 5
  done
  kill -KILL -- "-$pid" 2>/dev/null || true
  sleep 1
  return 0
}

# The wave list the client environment is built from (bootstrap envs resolves one reference per recipe; the
# judges declare none, so this only names the wave).
WAVE_LIST="$STATE/judgecheck-wave.txt"
printf '%s\n' "${RECIPES[@]}" >"$WAVE_LIST"

# The staged RC: a local stage is its own directory; a gs:// stage arrives at <state>/stage (bootstrap).
if [[ -d "$RC_STAGE_URI" ]]; then
  STAGE="$RC_STAGE_URI"
else
  STAGE="$STATE/stage"
fi

echo "judgecheck: building the node's client environment (bootstrap.sh envs)"
bash "$BOOTSTRAP_SH" envs "$RC_STAGE_URI" --state "$STATE" --recipes "$STAGE/recipes" --wave-list "$WAVE_LIST"

[[ -f "$STAGE/manifest.json" ]] || { echo "judgecheck: $STAGE holds no manifest.json (not an rc_build.sh stage)" >&2; exit 1; }
[[ -x "$STATE/client" ]] || { echo "judgecheck: bootstrap built no client wrapper at $STATE/client" >&2; exit 1; }

ENGINE_VLLM="$(python3 -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo unknown)"

REPORT="$STATE/judgecheck-report.json"
python3 "$REPORT_PY" init --file "$REPORT" --schema rcp-ndcg.judgecheck-report.v1 --started "$STARTED"
python3 "$REPORT_PY" merge --file "$REPORT" --key bootstrap --fragment "$STATE/bootstrap.json"
python3 - "$STATE/engine.json" "$ENGINE_VLLM" <<'PYEOF'
import json
import sys

print(json.dumps({"vllm": sys.argv[2]}), file=open(sys.argv[1], "w", encoding="utf-8"))
PYEOF
python3 "$REPORT_PY" merge --file "$REPORT" --key engine --fragment "$STATE/engine.json"

FAILED_RECIPES=()
for index in "${!RECIPES[@]}"; do
  recipe="${RECIPES[$index]}"
  port=$((PORT_BASE + index))
  engine_log="$STATE/engine-$recipe.log"
  : >"$STATE/modes-$recipe.txt"  # exists even when the engine dies first (the record assembly reads it)
  started_s="$(date +%s)"
  echo "judgecheck: serving $recipe on port $port"
  setsid "$STATE/client" rcp-ndcg-vllm serve "$recipe" --port "$port" >"$engine_log" 2>&1 &
  engine_pid=$!

  set +e
  wait_models "$port" "$STARTUP_S" "$engine_pid"
  ready_status=$?
  set -e
  if ((ready_status != 0)); then
    if ((ready_status == 2)); then
      reason="the engine's process group exited before /v1/models answered"
    else
      reason="the engine answered no /v1/models within ${STARTUP_S}s"
    fi
    tail_line="$(tail -1 "$engine_log" 2>/dev/null | tr -d '\r')"
    [[ -n "$tail_line" ]] && reason="$reason: $tail_line"
    echo "judgecheck: $recipe failed: $reason" >&2
    stop_engine "$engine_pid"
    python3 - "$STATE/recipe-$recipe.json" "$recipe" "$port" "$reason" "$(( $(date +%s) - started_s ))" <<'PYEOF'
import json
import sys

path, recipe, port, reason, wall_s = sys.argv[1:6]
document = {"state": "failed", "error": reason, "port": int(port), "wall_s": int(wall_s)}
print(json.dumps(document), file=open(path, "w", encoding="utf-8"))
PYEOF
    FAILED_RECIPES+=("$recipe")
    continue
  fi

  # What the engine serves (the served entry of /v1/models), for the report.
  python3 - "$port" >"$STATE/models-$recipe.json" <<'PYEOF'
import json
import sys
import urllib.request

with urllib.request.urlopen(f"http://127.0.0.1:{int(sys.argv[1])}/v1/models", timeout=30) as reply:  # noqa: S310
    models = json.loads(reply.read().decode("utf-8"))
rows = models.get("data", [])
print(json.dumps({"models": [row.get("id") for row in rows], "entry": rows[0] if rows else {}}))
PYEOF

  # The recipe's documented thinking modes (the helper owns the table): <name>\t<extra_body> per line.
  mode_lines="$STATE/modes-$recipe.txt"
  if ! "$STATE/client" python -m rcp_ndcg_test.jobs.judgecheck_probe modes --judge "$recipe" \
      >"$mode_lines" 2>"$STATE/modes-$recipe.err"; then
    tail_line="$(tail -1 "$STATE/modes-$recipe.err" 2>/dev/null | tr -d '\r')"
    echo "judgecheck: $recipe: the thinking-mode table failed: $tail_line" >&2
    : >"$mode_lines"
  fi

  while IFS=$'\t' read -r mode body; do
    [[ -n "$mode" ]] || continue
    check_out="$STATE/check-$recipe-$mode.json"
    set +e
    if [[ -n "${body:-}" ]]; then
      timeout "$CHECK_S" "$STATE/client" rcp-ndcg judge check \
        --judge "$recipe" --judge-model "$recipe" \
        --set "judge.base_url=http://127.0.0.1:$port/v1" \
        --set "judge.extra_body=$body" \
        --json >"$check_out" 2>"$STATE/check-$recipe-$mode.err"
    else
      timeout "$CHECK_S" "$STATE/client" rcp-ndcg judge check \
        --judge "$recipe" --judge-model "$recipe" \
        --set "judge.base_url=http://127.0.0.1:$port/v1" \
        --json >"$check_out" 2>"$STATE/check-$recipe-$mode.err"
    fi
    check_status=$?
    set -e
    if ((check_status != 0)) && [[ ! -s "$check_out" ]]; then
      tail_line="$(tail -1 "$STATE/check-$recipe-$mode.err" 2>/dev/null | tr -d '\r')"
      echo "judgecheck: $recipe/$mode: judge check exited $check_status: $tail_line" >&2
      rm -f "$check_out"  # no JSON document: the failure is recorded in the report, not uploaded as a check
    fi
  done <"$mode_lines"

  media_out="$STATE/media-$recipe.json"
  set +e
  timeout "$MEDIA_S" "$STATE/client" python -m rcp_ndcg_test.jobs.judgecheck_probe media \
    --judge "$recipe" --judge-model "$recipe" \
    --set "judge.base_url=http://127.0.0.1:$port/v1" \
    --out "$media_out" >"$STATE/media-$recipe.stdout" 2>"$STATE/media-$recipe.err"
  media_status=$?
  set -e
  if ((media_status != 0)) && [[ ! -s "$media_out" ]]; then
    tail_line="$(tail -1 "$STATE/media-$recipe.err" 2>/dev/null | tr -d '\r')"
    python3 - "$media_out" "$media_status" "$tail_line" <<'PYEOF'
import json
import sys

document = {"checked": False, "ok": False, "outcome": "failed", "detail": f"exit {sys.argv[2]}: {sys.argv[3]}"}
print(json.dumps(document), file=open(sys.argv[1], "w", encoding="utf-8"))
PYEOF
  fi

  # The engine log's NVFP4/Marlin lines (the release checklist's backend note), bounded and verbatim.
  python3 - "$engine_log" >"$STATE/backends-$recipe.json" <<'PYEOF'
import json
import re
import sys

pattern = re.compile(r"nvfp4|marlin|w4a16|fp4", flags=re.IGNORECASE)
lines = []
with open(sys.argv[1], encoding="utf-8", errors="replace") as handle:
    for line in handle:
        if pattern.search(line):
            lines.append(line.strip())
print(json.dumps(lines[:5]))
PYEOF

  stop_engine "$engine_pid"

  python3 - "$STATE" "$recipe" "$port" "$(( $(date +%s) - started_s ))" <<'PYEOF'
import json
import sys
from pathlib import Path

state, recipe, port, wall_s = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
modes = {}
mode_names = (state / f"modes-{recipe}.txt")
for line in (mode_names.read_text(encoding="utf-8") if mode_names.is_file() else "").splitlines():
    if not line.strip():
        continue
    mode = line.split("\t", 1)[0]
    path = state / f"check-{recipe}-{mode}.json"
    ok, detail = False, "judge check wrote no JSON document"
    if path.is_file():
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            ok = bool(document.get("ok"))
            detail = (document.get("error") or {}).get("message", "")
        except (OSError, json.JSONDecodeError):
            detail = "the check wrote no readable JSON document"
    modes[mode] = {"state": "ok" if ok else "failed", "ok": ok, "detail": detail}
media = {"checked": False, "outcome": "missing", "detail": "the media probe wrote no report"}
media_path = state / f"media-{recipe}.json"
if media_path.is_file():
    try:
        media = json.loads(media_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        media = {"checked": False, "outcome": "unreadable", "detail": "the media probe wrote an unreadable report"}
models = json.loads((state / f"models-{recipe}.json").read_text(encoding="utf-8"))
backends = json.loads((state / f"backends-{recipe}.json").read_text(encoding="utf-8"))
# The judge's own record: verified only when every documented mode passed AND the engine media check
# passed.  A text-only judge records checked:false with no outcome (not applicable); a mismatch,
# refusal, not_checked, a crashed probe and a missing report all fail the judge.
media_applies = bool(media.get("checked")) or media.get("outcome") is not None
media_ok = bool(media.get("ok")) or not media_applies
document = {
    "state": "verified" if modes and all(row["ok"] for row in modes.values()) and media_ok else "failed",
    "port": port,
    "wall_s": wall_s,
    "models": models,
    "modes": modes,
    "media": media,
    "backends": backends,
}
if set(modes) == {"default"}:
    document["modes_note"] = "single documented mode: the recipe documents no thinking-mode toggle"
print(json.dumps(document), file=open(state / f"recipe-{recipe}.json", "w", encoding="utf-8"))
PYEOF
  if ! python3 - "$STATE/recipe-$recipe.json" <<'PYEOF'
import json
import sys

document = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(0 if document["state"] == "verified" else 1)
PYEOF
  then
    echo "judgecheck: $recipe failed a thinking mode or the media check (recorded in the report)" >&2
    FAILED_RECIPES+=("$recipe")
  fi
done

# The report's recipes section: every judge's own record, or its failure entry.
python3 - "$STATE" "${RECIPES[@]}" <<'PYEOF'
import json
import sys
from pathlib import Path

state = Path(sys.argv[1])
recipes: dict[str, object] = {}
for recipe in sys.argv[2:]:
    path = state / f"recipe-{recipe}.json"
    recipes[recipe] = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {
        "state": "failed",
        "error": "no record was written",
    }
(state / "recipes.json").write_text(json.dumps(recipes), encoding="utf-8")
PYEOF
python3 "$REPORT_PY" merge --file "$REPORT" --key recipes --fragment "$STATE/recipes.json"

# The per-judge JSONs and the media reports (the checks' own documents verbatim).
for index in "${!RECIPES[@]}"; do
  recipe="${RECIPES[$index]}"
  while IFS=$'\t' read -r mode _body; do
    [[ -n "$mode" ]] || continue
    if [[ -s "$STATE/check-$recipe-$mode.json" ]]; then
      gcs_cp "$STATE/check-$recipe-$mode.json" "$OUT_URI/$recipe/$mode.json" file
    fi
  done <"$STATE/modes-$recipe.txt"
  if [[ -s "$STATE/media-$recipe.json" ]]; then
    gcs_cp "$STATE/media-$recipe.json" "$OUT_URI/$recipe/media.json" file
  fi
done

if ((${#FAILED_RECIPES[@]})); then
  python3 "$REPORT_PY" fail --file "$REPORT" --step judgecheck --reason "the judge checks failed for: ${FAILED_RECIPES[*]}"
fi
python3 "$REPORT_PY" emit --file "$REPORT"
gcs_cp "$REPORT" "$OUT_URI/judgecheck-report.json" file
if ((${#FAILED_RECIPES[@]})); then
  echo "judgecheck: ${#FAILED_RECIPES[@]} recipe(s) failed: ${FAILED_RECIPES[*]} (their checks are recorded in the report)" >&2
  exit 1
fi
echo "judgecheck: the ten judges' checks finished; the JSONs and the report are under $OUT_URI/"
