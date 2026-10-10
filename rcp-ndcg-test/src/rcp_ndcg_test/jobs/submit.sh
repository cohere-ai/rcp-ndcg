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
#   --script NAME  what the job runs on the node: bootstrap (a recipe wave; the default), wave0
#                  (the node test; its script is mounted next to bootstrap.sh) or e2e (the T4 run
#                  scenarios: e2e.sh drives them in the pod; the wave list names scenario ids)
#
# Environment (required, no defaults: this script must not name any machine's paths or buckets):
#   RCP_KJOBS_CONFIG   the job-CLI config file (the job CLI's -f argument)
#   RCP_GCS_AUTH_FILE  the auth script mounted at /etc/rcp/gcs_auth.sh on the node; named, never read
#   RCP_HF_TOKEN_FILE  the Hugging Face token file, mounted into the job and read there by the wrapper
#                      (its value never reaches a process argv, an echo or a log)
#   RCP_SHARED_MEMORY  the job's worker.shared_memory (defaults to 128Gi, sized for eight engines)
#   KJOBS              the job CLI, "kjobs-go" by default; KJOBS=echo prints every command instead of
#                      running it (the plan, with the secret value elided)
#   RCP_SUBMIT_DIR     where the job CLI's output files land (default: a mktemp dir; created when it
#                      does not exist)
#   RCP_HF_TOKEN_MOUNT where the mounted token appears inside the job (default /etc/rcp/hf_token; the
#                      wrapper reads it there)

set -euo pipefail

usage() {
  echo "usage: submit.sh [--max-jobs N] [--priority CLASS] [--script bootstrap|wave0|e2e] \\
         <RC_STAGE_URI> <OUT_PREFIX> <WAVE_NAME> [<WAVE_NAME>...]" >&2
}

MAX_JOBS="1"
PRIORITY="dev-medium"
SCRIPT_NAME="bootstrap"
IMAGE=""
RC_STAGE_URI=""
OUT_PREFIX=""
WAVES=()

while (($#)); do
  case "$1" in
    --max-jobs) MAX_JOBS="${2:?--max-jobs needs a number}"; shift 2 ;;
    --priority) PRIORITY="${2:?--priority needs a class}"; shift 2 ;;
    --script) SCRIPT_NAME="${2:?--script needs bootstrap, wave0 or e2e}"; shift 2 ;;
    --image) IMAGE="${2:?--image needs a repository:tag}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --*) echo "submit.sh: unknown option: $1" >&2; usage; exit 2 ;;
    *) if [[ -z "$RC_STAGE_URI" ]]; then RC_STAGE_URI="$1"; elif [[ -z "$OUT_PREFIX" ]]; then OUT_PREFIX="$1"; else WAVES+=("$1"); fi; shift ;;
  esac
done
[[ -n "$RC_STAGE_URI" && -n "$OUT_PREFIX" && ${#WAVES[@]} -ge 1 ]] || { usage; exit 2; }
[[ "$SCRIPT_NAME" == "bootstrap" || "$SCRIPT_NAME" == "wave0" || "$SCRIPT_NAME" == "e2e" ]] || {
  echo "submit.sh: --script must be bootstrap, wave0 or e2e, got $SCRIPT_NAME" >&2
  exit 2
}
[[ "$MAX_JOBS" =~ ^[0-9]+$ && "$MAX_JOBS" -ge 1 ]] || { echo "submit.sh: --max-jobs must be a positive number" >&2; exit 2; }

: "${RCP_KJOBS_CONFIG:?set RCP_KJOBS_CONFIG to the job-CLI config file (the -f argument)}"
: "${RCP_GCS_AUTH_FILE:?set RCP_GCS_AUTH_FILE to the auth script to mount at /etc/rcp/gcs_auth.sh}"
: "${RCP_HF_TOKEN_FILE:?set RCP_HF_TOKEN_FILE to the Hugging Face token file (mounted into the job and read there, never passed in an argv)}"
[[ -f "$RCP_KJOBS_CONFIG" ]] || { echo "submit.sh: no config file at $RCP_KJOBS_CONFIG (RCP_KJOBS_CONFIG)" >&2; exit 2; }
[[ -f "$RCP_GCS_AUTH_FILE" ]] || {
  echo "submit.sh: no auth file at $RCP_GCS_AUTH_FILE (RCP_GCS_AUTH_FILE; it is only named, never read)" >&2
  exit 2
}
[[ -f "$RCP_HF_TOKEN_FILE" ]] || {
  echo "submit.sh: no token file at $RCP_HF_TOKEN_FILE (RCP_HF_TOKEN_FILE; it is mounted into the job, never passed in an argv)" >&2
  exit 2
}

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KJOBS="${KJOBS:-kjobs-go}"
SHARED_MEMORY="${RCP_SHARED_MEMORY:-128Gi}"
OUT_DIR="${RCP_SUBMIT_DIR:-}"
SCRATCH_OUT_DIR=false
if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="$(mktemp -d /tmp/rcp-submit.XXXXXX)"
  SCRATCH_OUT_DIR=true
else
  # The operator's directory: created when it does not exist, so the job CLI's log always lands.
  mkdir -p "$OUT_DIR"
fi
ECHO_ONLY=false
if [[ "$KJOBS" == "echo" ]]; then
  ECHO_ONLY=true
fi

# The token is mounted, never passed: the wrapper reads it from the mounted file inside the job and execs the
# worker with HF_TOKEN in its environment (engines inherit it). Its own content holds no secret.
HF_TOKEN_MOUNT="${RCP_HF_TOKEN_MOUNT:-/etc/rcp/hf_token}"
HF_TOKEN_WRAPPER=/etc/rcp/files/hftoken/hf_token_env.sh
HF_TOKEN_WRAPPER_LOCAL="$OUT_DIR/hf_token_env.sh"
cat >"$HF_TOKEN_WRAPPER_LOCAL" <<WRAPPER
#!/usr/bin/env bash
set -euo pipefail
export HF_TOKEN="\$(cat "$HF_TOKEN_MOUNT")"
exec "\$@"
WRAPPER
chmod 700 "$HF_TOKEN_WRAPPER_LOCAL"

case "$(basename "$HERE")" in
  jobs) WAVE0_SH="$HERE/wave0.sh" ;;
  *) WAVE0_SH="$HERE/jobs/wave0.sh" ;;
esac
case "$(basename "$HERE")" in
  jobs) E2E_SH="$HERE/e2e.sh" ;;
  *) E2E_SH="$HERE/jobs/e2e.sh" ;;
esac
[[ "$SCRIPT_NAME" != "wave0" || -f "$WAVE0_SH" ]] || {
  echo "submit.sh: wave0.sh not found at $WAVE0_SH (run submit.sh from the checkout)" >&2
  exit 2
}
[[ "$SCRIPT_NAME" != "e2e" || -f "$E2E_SH" ]] || {
  echo "submit.sh: e2e.sh not found at $E2E_SH (run submit.sh from the checkout)" >&2
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

# resolve_digest IMAGE: the manifest digest of a public image, no credentials asked.
# Docker Hub needs a token dance (an anonymous pull token, then the manifest head); the gcloud CLI
# covers Google-hosted images when it is installed. Unresolvable prints to stderr and returns 1.
resolve_digest() {
  local image="$1" token digest
  if [[ "$image" != *"/"* ]]; then
    image="library/$image"
  fi
  token="$(python3 -c 'import json, sys, urllib.request
request = urllib.request.Request(
    "https://auth.docker.io/token?service=registry.docker.io&scope=repository:" + sys.argv[1] + ":pull"
)
with urllib.request.urlopen(request, timeout=30) as reply:
    print(json.loads(reply.read().decode("utf-8"))["token"])' "${image%:*}" 2>/dev/null || true)"
  if [[ -n "$token" ]]; then
    digest="$(python3 -c 'import sys, urllib.request
image, token = sys.argv[1], sys.argv[2]
accept = "application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.list.v2+json, application/vnd.docker.distribution.manifest.v2+json"
request = urllib.request.Request(
    "https://registry-1.docker.io/v2/" + sys.argv[3] + "/manifests/" + sys.argv[4],
    headers={"Authorization": "Bearer " + token, "Accept": accept},
)
with urllib.request.urlopen(request, timeout=30) as reply:
    print(reply.headers.get("Docker-Content-Digest", ""))' "$image" "$token" "${image%:*}" "${image##*:}" 2>/dev/null || true)"
    if [[ -n "${digest:-}" ]]; then
      printf '%s\n' "$digest"
      return 0
    fi
  fi
  if command -v gcloud >/dev/null; then
    digest="$(gcloud container images describe "$image" --format 'value(image_summary.digest)' 2>/dev/null || true)"
    if [[ -n "${digest:-}" ]]; then
      printf '%s\n' "$digest"
      return 0
    fi
  fi
  echo "submit.sh: no digest resolver answered for $image (docker hub or gcloud)" >&2
  return 1
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

# The image the job records: --image (default the pinned wave-0 image), and its digest. The digest
# resolution needs no credentials for a public image; a private one is resolved with the gcloud CLI.
# Unresolvable is not fatal: the report records it as unknown with the way out.
IMAGE="${IMAGE:-vllm/vllm-openai:v0.31.0}"
if [[ -n "${RCP_IMAGE_DIGEST:-}" ]]; then
  IMAGE_DIGEST="$RCP_IMAGE_DIGEST"
else
  IMAGE_DIGEST="$(resolve_digest "$IMAGE" || true)"
fi
if [[ -n "${IMAGE_DIGEST:-}" ]]; then
  echo "submit.sh: image digest for $IMAGE: $IMAGE_DIGEST"
else
  echo "submit.sh: warning: could not resolve the digest of $IMAGE; the wave-0 report will record it as" \
    "unknown. Resolve it operator-side and export RCP_IMAGE_DIGEST=... , e.g.:" \
    "gcloud container images describe <image> --format 'value(image_summary.digest)'" >&2
fi

# --- the plan: one row per (wave, engine image) -------------------------------------------------
# Owner decisions 38 and 35: a recipe may pin another engine image by digest, and a GPU job runs one
# container image.  A bootstrap wave's list is therefore grouped by the recipes' resolved engine.image
# and one job is submitted per image; wave0/e2e stay one job per wave.
declare -a PLAN=()
declare -A DIGESTS=()
SUBMITTED=0

fetch_stage_file() { # fetch_stage_file REL_PATH LOCAL [--recursive]: copy one staged file/dir out of the stage
  # A directory copy needs an existing local destination and the source's CONTENTS (the gcs.sh rule:
  # gcloud/gsutil refuse a missing destination and would otherwise nest the source directory under it).
  local rel="$1" local_path="$2" recursive="${3:-}"
  if [[ "$recursive" == "--recursive" ]]; then
    mkdir -p "$local_path"
    if [[ "$RC_STAGE_URI" == gs://* ]]; then
      if command -v gcloud >/dev/null; then
        gcloud storage cp --quiet --recursive "${RC_STAGE_URI%/}/$rel"/* "$local_path"
      else
        gsutil -q -m cp -r "${RC_STAGE_URI%/}/$rel"/* "$local_path"
      fi
    else
      cp -r "${RC_STAGE_URI%/}/$rel"/* "$local_path"
    fi
    return 0
  fi
  mkdir -p "$(dirname "$local_path")"
  if [[ "$RC_STAGE_URI" == gs://* ]]; then
    if command -v gcloud >/dev/null; then
      gcloud storage cp --quiet "${RC_STAGE_URI%/}/$rel" "$local_path"
    else
      gsutil -q cp "${RC_STAGE_URI%/}/$rel" "$local_path"
    fi
  else
    cp -r "${RC_STAGE_URI%/}/$rel" "$local_path"
  fi
}

plan_rows() { # plan_rows PLAN_JSON: one "image<TAB>list-file" row per engine image
  python3 - "$1" <<'PYEOF'
import json
import sys

plan = json.load(open(sys.argv[1], encoding="utf-8"))
for group in plan["groups"]:
    print(group["image"], group["file"], sep="\t")
PYEOF
}

group_wave() { # group_wave WAVE: append one plan row per engine image (the bootstrap script only)
  local wave="$1"
  local work="$OUT_DIR/group-$wave"
  local repo_root image file suffix job out
  mkdir -p "$work"
  if ! fetch_stage_file "wave-lists/$wave.txt" "$work/$wave.txt" \
    || ! fetch_stage_file "recipes" "$work/recipes" --recursive; then
    echo "submit.sh: warning: cannot read the staged wave list/recipes for $wave;" \
      "submitting one job on $IMAGE" >&2
    PLAN+=("$wave"$'\x1f'"$IMAGE"$'\x1f'""$'\x1f'"rcp-$wave"$'\x1f'"${OUT_PREFIX%/}/$wave")
    return 0
  fi
  repo_root="$(cd "$HERE/../../../.." && pwd)"
  if ! (cd "$repo_root" && uv run --no-sync python -m rcp_ndcg_test.jobs.wavegroups \
      --wave "$wave" --recipes "@$work/$wave.txt" --recipes-root "$work/recipes" \
      --out "$work/lists") >"$work/plan.json" 2>"$work/plan.err"; then
    echo "submit.sh: cannot group the wave '$wave' by engine image (rcp_ndcg_test.jobs.wavegroups):" >&2
    sed -n '1,5p' "$work/plan.err" >&2
    exit 1
  fi
  while IFS=$'\t' read -r image file; do
    [[ -n "$image" ]] || continue
    suffix="${file#"$wave."}"
    suffix="${suffix%.txt}"
    job="rcp-$wave-$suffix"
    out="${OUT_PREFIX%/}/$wave/$suffix"
    PLAN+=("$wave"$'\x1f'"$image"$'\x1f'"$work/lists/$file"$'\x1f'"$job"$'\x1f'"$out")
  done < <(plan_rows "$work/plan.json")
}

for wave in "${WAVES[@]}"; do
  if [[ "$SCRIPT_NAME" == "bootstrap" && "${RCP_GROUP_IMAGES:-1}" != "0" ]]; then
    group_wave "$wave"
  else
    PLAN+=("$wave"$'\x1f'"$IMAGE"$'\x1f'""$'\x1f'"rcp-$wave"$'\x1f'"${OUT_PREFIX%/}/$wave")
  fi
done

for plan_row in "${PLAN[@]}"; do
  IFS=$'\x1f' read -r wave row_image listfile JOB_NAME OUT_URI <<<"$plan_row"
  if [[ -z "${DIGESTS[$row_image]:-}" ]]; then
    DIGESTS["$row_image"]="$(resolve_digest "$row_image" || true)"
  fi
  ROW_IMAGE_DIGEST="${DIGESTS[$row_image]}"
  if [[ -n "$ROW_IMAGE_DIGEST" ]]; then
    echo "submit.sh: image digest for $row_image: $ROW_IMAGE_DIGEST"
  else
    echo "submit.sh: warning: could not resolve the digest of $row_image; the wave-0 report will record" \
      "it as unknown. Resolve it operator-side and export RCP_IMAGE_DIGEST=... , e.g.:" \
      "gcloud container images describe <image> --format 'value(image_summary.digest)'" >&2
  fi
  if [[ -n "$listfile" ]]; then
    LOG="$OUT_DIR/kjobs-$JOB_NAME.log"
  else
    LOG="$OUT_DIR/kjobs-$wave.log"  # the ungrouped wave's log keeps its historical name
  fi

  # The job's argv. The HF token never enters it: the token file is mounted, and a wrapper mounted beside it
  # exports HF_TOKEN from the file inside the job (a value in the argv is readable through /proc/<pid>/cmdline
  # by every user of the submitting host for the duration of the job CLI's call).
  HF_TOKEN_PREFIX="/bin/bash $HF_TOKEN_WRAPPER "
  args=(
    submit -f "$RCP_KJOBS_CONFIG"
    "app=$JOB_NAME"
    "priority_class=$PRIORITY"
    "worker.shared_memory=$SHARED_MEMORY"
    "env.RCP_IMAGE=$row_image"
    "env.RCP_IMAGE_DIGEST=$ROW_IMAGE_DIGEST"
  )
  if [[ "$SCRIPT_NAME" == "wave0" ]]; then
    args+=(
      "worker.command=${HF_TOKEN_PREFIX}/bin/bash /etc/rcp/files/wave0/wave0.sh $RC_STAGE_URI $OUT_URI"
      "files.wave0.from_file=$WAVE0_SH" "files.wave0.mount_path=/etc/rcp/files/wave0/wave0.sh"
      "files.wave0host.from_file=$HERE/wave0_host.py" "files.wave0host.mount_path=/etc/rcp/files/wave0host/wave0_host.py"
      # wave 0's step (b) runs the bootstrap: mounted beside its own script.
      "files.bootstrap.from_file=$HERE/bootstrap.sh" "files.bootstrap.mount_path=/etc/rcp/files/bootstrap/bootstrap.sh"
    )
  elif [[ "$SCRIPT_NAME" == "e2e" ]]; then
    # The T4 scenarios: e2e.sh builds the environments and drives the wave's scenario ids.
    args+=(
      "worker.command=${HF_TOKEN_PREFIX}/bin/bash /etc/rcp/files/e2e/e2e.sh $RC_STAGE_URI $OUT_URI --wave $wave"
      "files.e2e.from_file=$E2E_SH" "files.e2e.mount_path=/etc/rcp/files/e2e/e2e.sh"
      "files.bootstrap.from_file=$HERE/bootstrap.sh" "files.bootstrap.mount_path=/etc/rcp/files/bootstrap/bootstrap.sh"
    )
  else
    # The recipe wave: bootstrap.sh builds the environments and runs the wave's list.  A grouped row
    # mounts its per-image list and passes --wave-list; the ungrouped fallback resolves the staged one.
    wave_list_arg=""
    if [[ -n "$listfile" ]]; then
      wave_list_arg=" --wave-list /etc/rcp/files/wavelist/$(basename "$listfile")"
    fi
    args+=(
      "worker.command=${HF_TOKEN_PREFIX}/bin/bash /etc/rcp/files/bootstrap/bootstrap.sh wave $RC_STAGE_URI $OUT_URI --wave $wave$wave_list_arg"
      "files.bootstrap.from_file=$HERE/bootstrap.sh" "files.bootstrap.mount_path=/etc/rcp/files/bootstrap/bootstrap.sh"
    )
    if [[ -n "$listfile" ]]; then
      args+=(
        "files.wavelist.from_file=$listfile" "files.wavelist.mount_path=/etc/rcp/files/wavelist/$(basename "$listfile")"
      )
    fi
  fi
  args+=(
    "files.report.from_file=$HERE/report.py" "files.report.mount_path=/etc/rcp/files/report/report.py"
    "files.refdeps.from_file=$HERE/reference_deps.py" "files.refdeps.mount_path=/etc/rcp/files/refdeps/reference_deps.py"
    "files.gcshelper.from_file=$HERE/gcs.sh" "files.gcshelper.mount_path=/etc/rcp/files/gcshelper/gcs.sh"
    "files.gcspy.from_file=$HERE/gcs.py" "files.gcspy.mount_path=/etc/rcp/files/gcshelper/gcs.py"
    "files.gcsauth.from_file=$RCP_GCS_AUTH_FILE" "files.gcsauth.mount_path=/etc/rcp/gcs_auth.sh"
    "files.hftoken.from_file=$RCP_HF_TOKEN_FILE" "files.hftoken.mount_path=$HF_TOKEN_MOUNT"
    "files.hftokenenv.from_file=$HF_TOKEN_WRAPPER_LOCAL" "files.hftokenenv.mount_path=$HF_TOKEN_WRAPPER"
  )
  # At most MAX_JOBS in flight: job i for i >= MAX_JOBS waits for the release of job i - MAX_JOBS.
  if ((SUBMITTED >= MAX_JOBS)); then
    dep="${RELEASES[$((SUBMITTED - MAX_JOBS))]}"
    args+=("depends_on=$dep")
  fi

  echo "submit.sh: $JOB_NAME ($SCRIPT_NAME, image $row_image, priority $PRIORITY, shm $SHARED_MEMORY) -> $OUT_URI"
  if $ECHO_ONLY; then
    run "$KJOBS" "${args[@]}"
    RELEASES+=("$JOB_NAME") # echo mode runs nothing; the plan prints the app name
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
  if $SCRATCH_OUT_DIR; then
    rmdir "$OUT_DIR" 2>/dev/null || true  # the plan wrote nothing into the scratch dir
  fi
  echo "submit.sh: KJOBS=echo printed the plan; nothing was submitted"
else
  echo "submit.sh: $SUBMITTED job(s) submitted; the job CLI's outputs are under $OUT_DIR (grep them, never echo them)"
fi
