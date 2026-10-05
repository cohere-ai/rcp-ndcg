#!/usr/bin/env bash
# The node entry point of a wave: authenticate, fetch the code tarball, install it into the image's Python,
# run the wave, and leave the uploads to the runner (which also copies <out> after each recipe).
#
#   bootstrap.sh <STAGE_URI> <OUT_URI>
#
#     STAGE_URI  gs://.../code.tar.gz, the staged tarball submit.sh built from the current commit
#     OUT_URI    gs://.../wave, where run_wave.py --upload copies the wave's outputs
#
# The GPU host is the stock engine image: vLLM, torch, transformers, numpy, pydantic, httpx, PyYAML and jinja2
# are already installed, so the packages from the tarball go in with --no-deps and pip never touches the image's
# vLLM or torch.  The auth script mounted at /etc/rcp/gcs_auth.sh is executed, never printed, and its output and
# environment are never echoed.

set -euo pipefail

STAGE_URI="${1:?usage: bootstrap.sh <STAGE_URI> <OUT_URI>}"
OUT_URI="${2:?usage: bootstrap.sh <STAGE_URI> <OUT_URI>}"

AUTH_SCRIPT="/etc/rcp/gcs_auth.sh"

# The auth script is executed, never printed and never sourced into this shell's output.
set +x
AUTH_OUTPUT="$(mktemp)"
if ! "$AUTH_SCRIPT" >"$AUTH_OUTPUT" 2>&1; then
  echo "bootstrap: the GCS auth script failed; see the node's own logs (its output is not echoed here)" >&2
  exit 1
fi
rm -f "$AUTH_OUTPUT"
set -x

WORK="$(mktemp -d /tmp/rcp-wave.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT

fetch() { # one download, gcloud first and gsutil as the fallback
  gcloud storage cp "$1" "$2" || gsutil cp "$1" "$2"
}

mkdir -p "$WORK/code"
fetch "$STAGE_URI" "$WORK/code.tar.gz"
tar -xzf "$WORK/code.tar.gz" -C "$WORK/code"

cd "$WORK/code"

# Packages whose dependencies the image already satisfies (vLLM brings torch, transformers, numpy, pydantic,
# httpx, PyYAML and jinja2): install from the tarball without resolving dependencies, so pip never touches the
# image's vLLM or torch. The extras are named for the record; with --no-deps pip skips their resolution.
pip install --no-deps \
  "$WORK/code/packages/rcp-ndcg-core" \
  "$WORK/code/packages/rcp-ndcg-vllm[reference,metrics]"
# The pipeline package (rcp-ndcg): same mode, then its remaining runtime dependencies, which the image does not
# carry, from the public index - constrained to the image's own torch and transformers so pip cannot replace them.
pip freeze | grep -iE '^(torch|torchvision|transformers)==' >"$WORK/constraints.txt" || true
pip install --no-deps "$WORK/code"
# rcp-ndcg's remaining runtime dependencies, which the image does not carry. The list mirrors the root
# pyproject.toml's dependencies; bm25s is pinned there and repeated here. Edit both together.
pip install --constraint "$WORK/constraints.txt" pandas scipy pyarrow "bm25s==0.2.13" PyStemmer click tqdm \
  python-dotenv openai fsspec gcsfs Pillow sentence-transformers jinja2

# A wave may ship vllm.general_plugins plugin packages; plugins.txt lists their directories, one per line.
if [[ -f "$WORK/code/plugins.txt" ]]; then
  while IFS= read -r plugin_dir; do
    [[ -z "$plugin_dir" || "$plugin_dir" == \#* ]] && continue
    pip install --no-deps "$WORK/code/$plugin_dir"
  done <"$WORK/code/plugins.txt"
fi

GPUS="$(nvidia-smi --list-gpus 2>/dev/null | wc -l)"
GPUS="${GPUS//[[:space:]]/}"
if ! [[ "$GPUS" =~ ^[1-9][0-9]*$ ]]; then
  GPUS=8
fi
WAVE_ARGS=(--gpus "$GPUS" --out "$WORK/out" --upload "$OUT_URI" --record \
  --recipes-root "$WORK/code/packages/rcp-ndcg-vllm/recipes")
if [[ -f "$WORK/code/recipes.txt" ]]; then
  WAVE_ARGS+=(--recipes "@$WORK/code/recipes.txt")
fi
if [[ -d "$WORK/code/pairs" ]]; then
  WAVE_ARGS+=(--pairs-dir "$WORK/code/pairs")
fi

python -m rcp_ndcg_vllm.jobs.run_wave "${WAVE_ARGS[@]}"
