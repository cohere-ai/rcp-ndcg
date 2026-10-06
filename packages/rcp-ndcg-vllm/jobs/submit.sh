#!/usr/bin/env bash
# The operator's submission of the GPU waves: submit the job(s) of one or more waves against a staged
# release candidate. No lane submits a job: the operator runs this (or the kjobs-go command it prints).
#
#   submit.sh [options] <RC_STAGE_URI> <OUT_PREFIX> <WAVE_NAME> [<WAVE_NAME>...]
#
#     RC_STAGE_URI  gs://.../<rc-name> (or a local path), the RC that rc_build.sh staged
#     OUT_PREFIX    gs://.../waves, the outputs' parent; the wave uploads to <OUT_PREFIX>/<wave>
#     WAVE_NAME     one wave per job, named rcp-<wave>; its recipe list is the staged
#                   <RC>/wave-lists/<wave>.txt (bootstrap.sh looks it up on the node)
#
# Options:
#   --max-jobs N   at most N jobs in flight: wave i (0-based) for i >= N is submitted with
#                  depends_on=<release of wave i-N>, so the platform runs at most N at once (default 1)
#   --priority C   the Kueue priority class: dev-high or dev-medium; passed as the override
#                  priority_class=<C> to kjobs-go submit (which renders <C>-training-priority;
#                  default dev-medium; verify with a dry run)
#   --script NAME  what the job runs on the node: bootstrap (a recipe wave; the default) or wave0
#                  (the node test; its script is mounted next to bootstrap.sh)
#
# Environment (required, no defaults: this script must not name any machine's paths or buckets):
#   RCP_KJOBS_CONFIG   the job-CLI config file (the job CLI's -f argument)
#   RCP_GCS_AUTH_FILE  the auth script mounted at /etc/rcp/gcs_auth.sh on the node; named, never read
#   RCP_HF_TOKEN_FILE  the Hugging Face token file, passed as a kjobs secret by command substitution
#                      INSIDE this script - the value is never echoed, never logged, never printed
#   RCP_SHARED_MEMORY  the job's worker.shared_memory (defaults to 128Gi, sized for eight engines)
#   KJOBS              the job CLI, "kjobs-go" by default; KJOBS=echo prints every command instead of
#                      running it (the plan, with the secret value elided)
#   RCP_SUBMIT_DIR     where the job CLI's output files land (default: a mktemp dir)

set -euo pipefail

usage() {
  echo "usage: submit.sh [--max-jobs N] [--priority CLASS] [--script bootstrap|wave0] \\
         <RC_STAGE_URI> <OUT_PREFIX> <WAVE_NAME> [<WAVE_NAME>...]" >&2
}

MAX_JOBS="1"
PRIORITY="dev-medium"
SCRIPT_NAME="bootstrap"
RC_STAGE_URI=""
OUT_PREFIX=""
WAVES=()

while (($#)); do
  case "$1" in
    --max-jobs) MAX_JOBS="${2:?--max-jobs needs a number}"; shift 2 ;;
    --priority) PRIORITY="${2:?--priority needs a class}"; shift 2 ;;
    --script) SCRIPT_NAME="${2:?--script needs bootstrap or wave0}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --*) echo "submit.sh: unknown option: $1" >&2; usage; exit 2 ;;
    *) if [[ -z "$RC_STAGE_URI" ]]; then RC_STAGE_URI="$1"; elif [[ -z "$OUT_PREFIX" ]]; then OUT_PREFIX="$1"; else WAVES+=("$1"); fi; shift ;;
  esac
done
[[ -n "$RC_STAGE_URI" && -n "$OUT_PREFIX" && ${#WAVES[@]} -ge 1 ]] || { usage; exit 2; }
[[ "$SCRIPT_NAME" == "bootstrap" || "$SCRIPT_NAME" == "wave0" ]] || {
  echo "submit.sh: --script must be bootstrap or wave0, got $SCRIPT_NAME" >&2
  exit 2
}
[[ "$MAX_JOBS" =~ ^[0-9]+$ && "$MAX_JOBS" -ge 1 ]] || { echo "submit.sh: --max-jobs must be a positive number" >&2; exit 2; }

: "${RCP_KJOBS_CONFIG:?set RCP_KJOBS_CONFIG to the job-CLI config file (the -f argument)}"
: "${RCP_GCS_AUTH_FILE:?set RCP_GCS_AUTH_FILE to the auth script to mount at /etc/rcp/gcs_auth.sh}"
: "${RCP_HF_TOKEN_FILE:?set RCP_HF_TOKEN_FILE to the Hugging Face token file (passed as a secret, never read into a variable twice, never echoed)}"
[[ -f "$RCP_KJOBS_CONFIG" ]] || { echo "submit.sh: no config file at $RCP_KJOBS_CONFIG (RCP_KJOBS_CONFIG)" >&2; exit 2; }
[[ -f "$RCP_GCS_AUTH_FILE" ]] || {
  echo "submit.sh: no auth file at $RCP_GCS_AUTH_FILE (RCP_GCS_AUTH_FILE; it is only named, never read)" >&2
  exit 2
}
[[ -f "$RCP_HF_TOKEN_FILE" ]] || {
  echo "submit.sh: no token file at $RCP_HF_TOKEN_FILE (RCP_HF_TOKEN_FILE; it is read once by command substitution)" >&2
  exit 2
}

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KJOBS="${KJOBS:-kjobs-go}"
SHARED_MEMORY="${RCP_SHARED_MEMORY:-128Gi}"
OUT_DIR="${RCP_SUBMIT_DIR:-$(mktemp -d /tmp/rcp-submit.XXXXXX)}"
ECHO_ONLY=false
if [[ "$KJOBS" == "echo" ]]; then
  ECHO_ONLY=true
fi

case "$(basename "$HERE")" in
  jobs) WAVE0_SH="$HERE/../src/rcp_ndcg_vllm/jobs/wave0.sh" ;;
  *) WAVE0_SH="$HERE/jobs/wave0.sh" ;;
esac
[[ "$SCRIPT_NAME" != "wave0" || -f "$WAVE0_SH" ]] || {
  echo "submit.sh: wave0.sh not found at $WAVE0_SH (run submit.sh from the checkout)" >&2
  exit 2
}

run() { # run, or print when KJOBS=echo (the secret's VALUE never reaches this function's argv in echo mode)
  if $ECHO_ONLY; then
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

# The release name a submitted job got, from the CLI's own instruction line
# ("kjobs ... logs <release-name>"); the depends_on chain needs it. One name per submitted job.
release_of() { # release_of LOG_FILE: the job's release name, or fail loudly
  local name
  name="$(sed -n 's/.* logs \([^ ]*\)$/\1/p' "$1" | tail -1)"
  [[ -n "$name" ]] || {
    echo "submit.sh: could not read the submitted job's release name from $1 (the CLI's output format changed?)" >&2
    exit 1
  }
  printf '%s\n' "$name"
}

declare -a RELEASES=()
SUBMITTED=0
for wave in "${WAVES[@]}"; do
  JOB_NAME="rcp-$wave"
  OUT_URI="${OUT_PREFIX%/}/$wave"
  LOG="$OUT_DIR/kjobs-$wave.log"

  # The job's argv. The HF token is expanded by THIS command substitution, into the argv only; in echo
  # mode the placeholder text is printed instead of the value, and the value itself is never printed.
  if $ECHO_ONLY; then
    TOKEN_OVERRIDE=("secret.HF_TOKEN=\$(cat \"\$RCP_HF_TOKEN_FILE\")")
  else
    TOKEN_OVERRIDE=("secret.HF_TOKEN=$(cat "$RCP_HF_TOKEN_FILE")")
  fi
  args=(
    submit -f "$RCP_KJOBS_CONFIG"
    "app=$JOB_NAME"
    "priority_class=$PRIORITY"
    "worker.shared_memory=$SHARED_MEMORY"
  )
  if [[ "$SCRIPT_NAME" == "wave0" ]]; then
    args+=(
      "worker.command=/bin/bash /etc/rcp/files/wave0/wave0.sh $RC_STAGE_URI $OUT_URI"
      "files.wave0.from_file=$WAVE0_SH" "files.wave0.mount_path=/etc/rcp/files/wave0/wave0.sh"
      "files.wave0host.from_file=$HERE/wave0_host.py" "files.wave0host.mount_path=/etc/rcp/files/wave0host/wave0_host.py"
      # wave 0's step (b) runs the bootstrap: mounted beside its own script.
      "files.bootstrap.from_file=$HERE/bootstrap.sh" "files.bootstrap.mount_path=/etc/rcp/files/bootstrap/bootstrap.sh"
    )
  else
    # The recipe wave: bootstrap.sh builds the three environments and runs the wave's list.
    args+=(
      "worker.command=/bin/bash /etc/rcp/files/bootstrap/bootstrap.sh wave $RC_STAGE_URI $OUT_URI --wave $wave"
      "files.bootstrap.from_file=$HERE/bootstrap.sh" "files.bootstrap.mount_path=/etc/rcp/files/bootstrap/bootstrap.sh"
    )
  fi
  args+=(
    "files.report.from_file=$HERE/report.py" "files.report.mount_path=/etc/rcp/files/report/report.py"
    "files.gcsauth.from_file=$RCP_GCS_AUTH_FILE" "files.gcsauth.mount_path=/etc/rcp/gcs_auth.sh"
    "${TOKEN_OVERRIDE[@]}"
  )
  # At most MAX_JOBS in flight: wave i for i >= MAX_JOBS waits for the release of wave i - MAX_JOBS.
  if ((SUBMITTED >= MAX_JOBS)); then
    dep="${RELEASES[$((SUBMITTED - MAX_JOBS))]}"
    args+=("depends_on=$dep")
  fi

  echo "submit.sh: $JOB_NAME ($SCRIPT_NAME, priority $PRIORITY, shm $SHARED_MEMORY) -> $OUT_URI"
  if $ECHO_ONLY; then
    run "$KJOBS" "${args[@]}"
    RELEASES+=("rcp-$wave") # echo mode runs nothing; the plan prints the app name
  else
    set +e
    run "$KJOBS" "${args[@]}" >"$LOG" 2>&1
    status=$?
    set -e
    if ((status != 0)); then
      echo "submit.sh: the job CLI failed for $JOB_NAME (exit $status); its output is in $LOG (never echoed here)" >&2
      exit 1
    fi
    RELEASES+=("$(release_of "$LOG")")
    echo "submit.sh: $JOB_NAME submitted as ${RELEASES[-1]} (logs: kjobs logs $LOG)"
  fi
  SUBMITTED=$((SUBMITTED + 1))
done

if $ECHO_ONLY; then
  echo "submit.sh: KJOBS=echo printed the plan; nothing was submitted"
else
  echo "submit.sh: $SUBMITTED job(s) submitted; the job CLI's outputs are under $OUT_DIR (grep them, never echo them)"
fi
