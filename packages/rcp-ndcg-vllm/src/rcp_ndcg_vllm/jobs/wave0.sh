#!/usr/bin/env bash
# Wave 0: the one-job node test, run before any recipe wave (GPU-VALIDATION.md, node-runtime items).
#
#   wave0.sh <RC_STAGE_URI> <OUT_URI>
#
#     RC_STAGE_URI  gs://.../<rc-name> (or a local path): the development build rc_build.sh staged
#     OUT_URI       gs://.../waves/wave0: the report and the engines' logs upload here (the report also
#                   to <RC_STAGE_URI>/reports/, so the stage holds what every wave ran against)
#
# In order, failing fast with a one-line reason at the first failure, every step's fragment in the report:
#   (a) host      the image (digest as passed in), driver, GPUs, free disk, /dev/shm, python versions
#   (b) bootstrap the three environments via bootstrap.sh; the engine's pip freeze must not change
#                 (no plugin wheel in wave 0)
#   (c) reach     the Hub with the token secret (a metadata call), and a gs:// write/list/read/delete
#                 round-trip through rcp_ndcg.storage from the client environment; PyPI is not required
#   (d) engines   two engines at once on two slots (the pinned embedding model, and any small second
#                 model), each with its own CUDA_VISIBLE_DEVICES, port, VLLM_PORT, TMPDIR and log dir
#   (e) embed     the texts under an explicit budget through the product's fit() and EmbeddingClient,
#                 the engine's /tokenize per input (the engine is the tokenization truth, R29)
#   (f) evict     the models out of the HF cache, with the free disk before and after (item 8)
#   (g) stop      everything stopped, and no engine process left on the node
#
# Environment: RCP_GCS_AUTH_FILE (mounted auth script; executed, never printed), HF_TOKEN (the job's
# secret; never printed), RCP_IMAGE / RCP_IMAGE_DIGEST (recorded), WAVE0_DRY=1 (print the plan, run
# nothing), RCP_REPORT_PY / RCP_HOST_PY / RCP_BOOTSTRAP_SH (the mounted helpers) and the WAVE0_* knobs.
#
# Assumes about the node (checked, not assumed): nvidia-smi and python3 on PATH; at least
# WAVE0_MIN_GPUS GPUs; /dev/shm of at least WAVE0_MIN_SHM_GIB (submit sizes it); free disk above
# WAVE0_MIN_FREE_GIB (the weights land on the container filesystem); the Hub and the bucket reachable;
# the image's python3 imports vllm (the engine environment).

set -euo pipefail

usage() { echo "usage: wave0.sh <RC_STAGE_URI> <OUT_URI>" >&2; }

RC_STAGE_URI="${1:?usage: wave0.sh <RC_STAGE_URI> <OUT_URI>}"
OUT_URI="${2:?usage: wave0.sh <RC_STAGE_URI> <OUT_URI>}"
[[ -n "$RC_STAGE_URI" && -n "$OUT_URI" ]] || { usage; exit 2; }

# --- the knobs (the model and revision are the researched pins; the rest is the node test's own) -------
WAVE0_MODEL="${WAVE0_MODEL:-Qwen/Qwen3-Embedding-0.6B}"
WAVE0_REVISION="${WAVE0_REVISION:-97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3}"
WAVE0_SERVED_NAME="${WAVE0_SERVED_NAME:-qwen3-embedding-0-6b}"
WAVE0_SECOND_MODEL="${WAVE0_SECOND_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"
WAVE0_SECOND_SERVED_NAME="${WAVE0_SECOND_SERVED_NAME:-all-minilm-l6-v2}"
WAVE0_SECOND_ARGS="${WAVE0_SECOND_ARGS:---runner pooling}"
WAVE0_BUDGET="${WAVE0_BUDGET:-512}"
WAVE0_COUNT="${WAVE0_COUNT:-20}"
WAVE0_OVER="${WAVE0_OVER:-5}"
WAVE0_PORT_BASE="${WAVE0_PORT_BASE:-8100}"
WAVE0_VLLM_PORT_BASE="${WAVE0_VLLM_PORT_BASE:-9200}"
WAVE0_STARTUP_TIMEOUT_S="${WAVE0_STARTUP_TIMEOUT_S:-1800}"
WAVE0_MIN_GPUS="${WAVE0_MIN_GPUS:-2}"
WAVE0_MIN_SHM_GIB="${WAVE0_MIN_SHM_GIB:-8}"
WAVE0_MIN_FREE_GIB="${WAVE0_MIN_FREE_GIB:-50}"

AUTH_SCRIPT="${RCP_GCS_AUTH_FILE:-/etc/rcp/gcs_auth.sh}"
REPORT_PY="${RCP_REPORT_PY:-/etc/rcp/files/report/report.py}"
HOST_PY="${RCP_HOST_PY:-/etc/rcp/files/wave0host/wave0_host.py}"
BOOTSTRAP_SH="${RCP_BOOTSTRAP_SH:-/etc/rcp/files/bootstrap/bootstrap.sh}"

if [[ "${WAVE0_DRY:-0}" == "1" ]]; then
  echo "wave0 (dry): would run, in order, on one node, failing fast at the first failure:"
  echo "  0. preflight  tools nvidia-smi, python3, gcloud|gsutil; env RCP_GCS_AUTH_FILE, HF_TOKEN;"
  echo "                >= $WAVE0_MIN_GPUS GPUs, /dev/shm >= ${WAVE0_MIN_SHM_GIB} GiB, free disk >= ${WAVE0_MIN_FREE_GIB} GiB"
  echo "  a. host       $HOST_PY: image (digest as passed), driver, GPUs, free disk, /dev/shm, pythons"
  echo "  b. bootstrap  $BOOTSTRAP_SH envs $RC_STAGE_URI --state <state>"
  echo "                (three environments; the engine's pip freeze must not change - no plugin in wave 0)"
  echo "  c. reach      the Hub metadata for $WAVE0_MODEL@$WAVE0_REVISION with HF_TOKEN, and a gs://"
  echo "                write/list/read/delete round-trip through rcp_ndcg.storage under $OUT_URI (client env)"
  echo "  d. engines    vllm serve $WAVE0_MODEL@$WAVE0_REVISION on slot 0 (port $WAVE0_PORT_BASE, VLLM_PORT"
  echo "                $WAVE0_VLLM_PORT_BASE) and $WAVE0_SECOND_MODEL on slot 1, both up at the same time"
  echo "  e. embed      product fit() + EmbeddingClient over $WAVE0_COUNT texts ($WAVE0_OVER over a"
  echo "                $WAVE0_BUDGET-token budget); the engine's /tokenize per input through the client env"
  echo "  f. evict      $WAVE0_MODEL and $WAVE0_SECOND_MODEL out of the HF cache (free disk before/after)"
  echo "  g. stop       both engines' process groups stopped, then no vllm process left on the node"
  echo "  finish        the report -> $OUT_URI/wave0-report.json and $RC_STAGE_URI/reports/"
  echo "wave0 (dry): the plan only; nothing ran"
  exit 0
fi

for tool in python3 nvidia-smi; do
  command -v "$tool" >/dev/null || { echo "wave0: $tool is not on PATH (an assumption of the node)"; exit 1; }
done
command -v gcloud >/dev/null || command -v gsutil >/dev/null || {
  echo "wave0: neither gcloud nor gsutil is on PATH (the report uploads need one)"
  exit 1
}
[[ -f "$AUTH_SCRIPT" ]] || { echo "wave0: no auth script at $AUTH_SCRIPT (set RCP_GCS_AUTH_FILE)"; exit 1; }
[[ -n "${HF_TOKEN:-}" ]] || { echo "wave0: HF_TOKEN is not set (the job's secret); the Hub check needs it"; exit 1; }
[[ -f "$REPORT_PY" && -f "$HOST_PY" && -f "$BOOTSTRAP_SH" ]] || {
  echo "wave0: report.py ($REPORT_PY), wave0_host.py ($HOST_PY) or bootstrap.sh ($BOOTSTRAP_SH) is not mounted"
  exit 1
}

WORK="$(mktemp -d "${TMPDIR:-/tmp}/rcp-wave0.XXXXXX")"
REPORT="$WORK/wave0-report.json"
STATE="$WORK/state"
SPEC="$WORK/engines-spec.json"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$WORK/engines" "$WORK/logs"
export WAVE0_MODEL WAVE0_REVISION WAVE0_SERVED_NAME WAVE0_SECOND_MODEL WAVE0_SECOND_SERVED_NAME
export WAVE0_SECOND_ARGS WAVE0_PORT_BASE WAVE0_VLLM_PORT_BASE WAVE0_STARTUP_TIMEOUT_S

report() { python3 "$REPORT_PY" "$@"; }
fragment_error() { # fragment_error FILE: the fragment's one-line error, or a fallback
  python3 -c 'import json, sys
document = json.load(open(sys.argv[1]))
print(document.get("error") or ("" if document.get("passed", True) else "the probe recorded passed: false"))' "$1" 2>/dev/null || echo "the step failed (no fragment)"
}

fail_step() { # fail_step STEP REASON: one line, the report, the engines stopped, the uploads, out
  echo "wave0: step $1 failed: $2" >&2
  report fail --file "$REPORT" --step "$1" --reason "$2" >/dev/null
  stop_engines
  upload_artifacts
  report emit --file "$REPORT"
  exit 1
}

upload() { # upload LOCAL REMOTE: one copy, gcloud first; a failure is reported, not fatal
  local src="$1" dst="$2"
  if command -v gcloud >/dev/null; then
    gcloud storage cp "$src" "$dst" 2>/dev/null || echo "wave0: the upload of $src to $dst failed; continuing" >&2
  else
    gsutil cp "$src" "$dst" 2>/dev/null || echo "wave0: the upload of $src to $dst failed; continuing" >&2
  fi
}

upload_artifacts() {
  upload "$REPORT" "${OUT_URI%/}/wave0-report.json"
  upload "$REPORT" "${RC_STAGE_URI%/}/reports/wave0-report-$STAMP.json"
  if [[ -d "$WORK/logs" ]]; then
    if command -v gcloud >/dev/null; then
      gcloud storage cp -r "$WORK/logs/*" "${OUT_URI%/}/logs/" 2>/dev/null || echo "wave0: the engine logs' upload failed; continuing" >&2
    else
      gsutil -m cp -r "$WORK/logs/*" "${OUT_URI%/}/logs/" 2>/dev/null || echo "wave0: the engine logs' upload failed; continuing" >&2
    fi
  fi
}

stop_engines() { # the bash fallback the EXIT trap runs; the probe's engines-stop is the real one
  local state_file="$STATE/engines.json" pid
  [[ -f "$state_file" ]] || return 0
  while IFS= read -r pid; do
    [[ -n "$pid" ]] || continue
    kill -TERM -- "-$pid" 2>/dev/null || true
  done < <(python3 -c 'import json, sys
print("\n".join(str(engine["pid"]) for engine in json.load(open(sys.argv[1])).get("engines", [])))' "$state_file" 2>/dev/null)
  sleep 5
  while IFS= read -r pid; do
    [[ -n "$pid" ]] || continue
    kill -KILL -- "-$pid" 2>/dev/null || true
  done < <(python3 -c 'import json, sys
print("\n".join(str(engine["pid"]) for engine in json.load(open(sys.argv[1])).get("engines", [])))' "$state_file" 2>/dev/null)
}
trap stop_engines EXIT

run_probe() { # run_probe REPORT_KEY PROBE_ARGS...: run the probe in the client env, merge, gate on passed
  local key="$1"
  shift
  if ! "$STATE/client" python -m rcp_ndcg_vllm.jobs.wave0_probe "$@" \
    >"$WORK/$key.json" 2>"$WORK/$key.err"; then
    tail -3 "$WORK/$key.err" >&2 || true
    fail_step "$key" "$(fragment_error "$WORK/$key.json" "the probe failed")"
  fi
  report merge --file "$REPORT" --key "$key" --fragment "$WORK/$key.json" >/dev/null
  if ! python3 -c 'import json, sys
raise SystemExit(0 if json.load(open(sys.argv[1])).get("passed", True) else 1)' "$WORK/$key.json"; then
    fail_step "$key" "$(fragment_error "$WORK/$key.json" "the step recorded passed: false")"
  fi
}

report init --file "$REPORT" --schema rcp-ndcg.wave0-report.v1

# --- preflight: the assumptions the node must meet ----------------------------------------------------

GPUS="$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo 0)"
if ((GPUS < WAVE0_MIN_GPUS)); then
  fail_step preflight "the node has $GPUS GPUs, wave 0 needs $WAVE0_MIN_GPUS (two slots at once)"
fi
SHM_BYTES="$(python3 -c 'import os
stats = os.statvfs("/dev/shm")
print(stats.f_bsize * stats.f_bavail)')"
if ((SHM_BYTES < (WAVE0_MIN_SHM_GIB << 30))); then
  fail_step preflight "/dev/shm is $((SHM_BYTES >> 30)) GiB, under the assumed ${WAVE0_MIN_SHM_GIB} GiB (submit with worker.shared_memory)"
fi
FREE_BYTES="$(python3 -c 'import shutil; print(shutil.disk_usage("/").free)')"
if ((FREE_BYTES < (WAVE0_MIN_FREE_GIB << 30))); then
  fail_step preflight "only $((FREE_BYTES >> 30)) GiB free on /, under the assumed ${WAVE0_MIN_FREE_GIB} GiB (the weights land on the container filesystem)"
fi

# --- (a) the node's facts -----------------------------------------------------------------------------

python3 "$HOST_PY" --report "$WORK/host.json" --workdir "$WORK/engines" >/dev/null \
  || fail_step host "the host probe failed"
report merge --file "$REPORT" --key host --fragment "$WORK/host.json" >/dev/null
if [[ -z "$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["image_digest"] or "")' "$WORK/host.json")" ]]; then
  echo "wave0: note: RCP_IMAGE_DIGEST was not passed; the report records it as unknown (resolve it operator-side)" >&2
fi

# --- (b) the three environments (and the untouched engine freeze) -------------------------------------

BOOTSTRAP_START="$(date +%s)"
if ! "$BOOTSTRAP_SH" envs "$RC_STAGE_URI" --state "$STATE" >"$WORK/bootstrap.log" 2>&1; then
  tail -5 "$WORK/bootstrap.log" >&2 || true
  fail_step bootstrap "bootstrap.sh envs failed (the last lines of its log are above)"
fi
BOOTSTRAP_S="$(( $(date +%s) - BOOTSTRAP_START ))"
[[ -f "$STATE/bootstrap.json" ]] || fail_step bootstrap "bootstrap.sh wrote no report fragment at $STATE/bootstrap.json"
[[ -x "$STATE/client" ]] || fail_step bootstrap "bootstrap.sh built no client wrapper at $STATE/client"
python3 - "$STATE/bootstrap.json" "$BOOTSTRAP_S" <<'PYEOF'
import json
import sys

document = json.load(open(sys.argv[1]))
document["install_s"] = int(sys.argv[2])
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    handle.write(json.dumps(document, indent=2) + "\n")
PYEOF
report merge --file "$REPORT" --key bootstrap --fragment "$STATE/bootstrap.json" >/dev/null

# --- (c) reachability: the Hub with the token secret, and the GCS round-trip through the product ------

if ! python3 "$HOST_PY" --help >/dev/null 2>&1; then
  fail_step reach "wave0_host.py is unusable"
fi
if ! python3 "$HOST_PY" hub --model "$WAVE0_MODEL" --revision "$WAVE0_REVISION" --report "$WORK/hub.json" >/dev/null 2>"$WORK/hub.err"; then
  tail -3 "$WORK/hub.err" >&2 || true
  fail_step reach "the Hub metadata call failed for $WAVE0_MODEL (the token or the Hub is the suspect)"
fi
report merge --file "$REPORT" --key reach.hub --fragment "$WORK/hub.json" >/dev/null
run_probe reach.gcs gcs --out-uri "$OUT_URI" --report "$WORK/gcs.json"

# --- (d) two engines on two slots, at the same time ----------------------------------------------------

python3 - "$SPEC" <<'PYEOF'
"""The two slots' spec: the pinned embedding model and any small second model, isolation per slot."""
import json
import os
import sys

port = int(os.environ["WAVE0_PORT_BASE"])
vllm_port = int(os.environ["WAVE0_VLLM_PORT_BASE"])
second_args = os.environ["WAVE0_SECOND_ARGS"].split()
work = sys.argv[2]
spec = {
    "startup_timeout_s": int(os.environ["WAVE0_STARTUP_TIMEOUT_S"]),
    "slots": [
        {
            "slot": 0,
            "model": os.environ["WAVE0_MODEL"],
            "served_model_name": os.environ["WAVE0_SERVED_NAME"],
            "argv": [
                "vllm",
                "serve",
                os.environ["WAVE0_MODEL"],
                "--revision",
                os.environ["WAVE0_REVISION"],
                "--served-model-name",
                os.environ["WAVE0_SERVED_NAME"],
                "--host",
                "0.0.0.0",
                "--port",
                str(port),
                "--runner",
                "pooling",
                "--dtype",
                "bfloat16",
            ],
            "cuda_visible_devices": "0",
            "port": port,
            "vllm_port": vllm_port,
            "tmpdir": f"{work}/engines/slot-0/tmp",
            "log_dir": f"{work}/logs/slot-0",
        },
        {
            "slot": 1,
            "model": os.environ["WAVE0_SECOND_MODEL"],
            "served_model_name": os.environ["WAVE0_SECOND_SERVED_NAME"],
            "argv": [
                "vllm",
                "serve",
                os.environ["WAVE0_SECOND_MODEL"],
                "--served-model-name",
                os.environ["WAVE0_SECOND_SERVED_NAME"],
                "--host",
                "0.0.0.0",
                "--port",
                str(port + 1),
                *os.environ["WAVE0_SECOND_ARGS"].split(),
            ],
            "cuda_visible_devices": "1",
            "port": port + 1,
            "vllm_port": vllm_port + 1,
            "tmpdir": f"{work}/engines/slot-1/tmp",
            "log_dir": f"{work}/logs/slot-1",
        },
    ],
}
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    handle.write(json.dumps(spec, indent=2) + "\n")
PYEOF
run_probe engines "$STATE" engines-start --spec "$SPEC" --state "$STATE/engines.json" --report "$WORK/engines.json"

# --- (e) the product's budget, client and the engine's /tokenize ---------------------------------------

run_probe embed embed --base-url "http://127.0.0.1:$WAVE0_PORT_BASE" \
  --served-model-name "$WAVE0_SERVED_NAME" \
  --tokenizer "$WAVE0_MODEL@$WAVE0_REVISION" \
  --budget "$WAVE0_BUDGET" --count "$WAVE0_COUNT" --over "$WAVE0_OVER" \
  --report "$WORK/embed.json"

# --- (f) the eviction ----------------------------------------------------------------------------------

run_probe evict evict --model "$WAVE0_MODEL" --model "$WAVE0_SECOND_MODEL" --report "$WORK/evict.json"

# --- (g) stop everything, assert no engine process remains ---------------------------------------------

run_probe stop engines-stop --state "$STATE/engines.json" --report "$WORK/stop.json"

# --- the report ----------------------------------------------------------------------------------------

report merge --file "$REPORT" --key finished --literal "{\"at\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" >/dev/null
stop_engines  # the trap also runs this; explicit here so the exit code below is the step's
trap - EXIT
upload_artifacts
report emit --file "$REPORT"
PASSED="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["passed"])' "$REPORT")"
[[ "$PASSED" == "True" ]] || exit 1
echo "wave0: all steps passed; the report is at ${OUT_URI%/}/wave0-report.json"
