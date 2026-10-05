#!/usr/bin/env bash
# The operator's submission: stage the code tarball of the current commit, then submit the wave job.
#
#   submit.sh <WAVE_NAME> <RECIPES_FILE> <STAGE_PREFIX> <OUT_PREFIX>
#
#     WAVE_NAME      the wave's name; the job is named rcp-<WAVE_NAME>
#     RECIPES_FILE   the recipes file, staged into the tarball (the wave runner reads it from the code)
#     STAGE_PREFIX   gs://YOUR-BUCKET/stage - the tarball goes to <STAGE_PREFIX>/<WAVE_NAME>/code.tar.gz
#     OUT_PREFIX     gs://YOUR-BUCKET/waves, the outputs' parent; the wave uploads to <OUT_PREFIX>/<WAVE_NAME>
#
# Environment (required, no defaults: this script must not name any machine's paths):
#   EXTRA_DIRS        space-separated extra directories to add to the tarball (pairs files, plugins), copied as-is
#   RCP_KJOBS_CONFIG  the job-CLI config file (the job CLI's -f argument)
#   RCP_GCS_AUTH_FILE the auth script mounted at /etc/rcp/gcs_auth.sh on the node; named, never read or printed
#   KJOBS             the job CLI, "kjobs-go" by default; KJOBS=echo prints every command instead of running it
#
# This script never reads, prints or renders the auth file: it only names its path for the mount.

set -euo pipefail

WAVE_NAME="${1:?usage: submit.sh <WAVE_NAME> <RECIPES_FILE> <STAGE_PREFIX> <OUT_PREFIX>}"
RECIPES_FILE="${2:?usage: submit.sh <WAVE_NAME> <RECIPES_FILE> <STAGE_PREFIX> <OUT_PREFIX>}"
STAGE_PREFIX="${3:?usage: submit.sh <WAVE_NAME> <RECIPES_FILE> <STAGE_PREFIX> <OUT_PREFIX>}"
OUT_PREFIX="${4:?usage: submit.sh <WAVE_NAME> <RECIPES_FILE> <STAGE_PREFIX> <OUT_PREFIX>}"

: "${RCP_KJOBS_CONFIG:?set RCP_KJOBS_CONFIG to the job-CLI config file (the -f argument)}"
: "${RCP_GCS_AUTH_FILE:?set RCP_GCS_AUTH_FILE to the auth script to mount at /etc/rcp/gcs_auth.sh}"
[[ -f "$RCP_KJOBS_CONFIG" ]] || {
  echo "submit.sh: no config file at $RCP_KJOBS_CONFIG (RCP_KJOBS_CONFIG)" >&2
  exit 2
}
[[ -f "$RCP_GCS_AUTH_FILE" ]] || {
  echo "submit.sh: no auth file at $RCP_GCS_AUTH_FILE (RCP_GCS_AUTH_FILE; it is only named, never read)" >&2
  exit 2
}

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KJOBS="${KJOBS:-kjobs-go}"

STAGE_URI="${STAGE_PREFIX%/}/$WAVE_NAME/code.tar.gz"
OUT_URI="${OUT_PREFIX%/}/$WAVE_NAME"
JOB_NAME="rcp-$WAVE_NAME"

ECHO_ONLY=false
if [[ "$KJOBS" == "echo" ]]; then
  ECHO_ONLY=true
fi

run() { # run, or print when KJOBS=echo
  if $ECHO_ONLY; then
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

STAGE="$(mktemp -d /tmp/rcp-stage.XXXXXX)"
trap 'rm -rf "$STAGE"' EXIT

# The current commit of rcp-ndcg, plus any extra directories.
mkdir -p "$STAGE/code"
git archive HEAD | tar -x -C "$STAGE/code"
if [[ -n "${EXTRA_DIRS:-}" ]]; then
  for extra in $EXTRA_DIRS; do
    cp -r "$extra" "$STAGE/code/"
  done
fi
cp "$RECIPES_FILE" "$STAGE/code/recipes.txt"
tar -czf "$STAGE/code.tar.gz" -C "$STAGE/code" .

run gcloud storage cp "$STAGE/code.tar.gz" "$STAGE_URI"

run "$KJOBS" submit -f "$RCP_KJOBS_CONFIG" \
  "app=$JOB_NAME" \
  "worker.command=/bin/bash /etc/rcp/files/bootstrap/bootstrap.sh $STAGE_URI $OUT_URI" \
  "files.bootstrap.from_file=$HERE/bootstrap.sh" \
  "files.gcsauth.from_file=$RCP_GCS_AUTH_FILE" \
  "files.gcsauth.mount_path=/etc/rcp/gcs_auth.sh"
