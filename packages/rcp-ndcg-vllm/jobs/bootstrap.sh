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

derive_light_deps() { # the non-image dependencies of the three packages, read from their pyproject.toml files
  python3 - "$1" <<'PY'
"""Derive the light dependencies from the manifests (tomllib), not from a hand-mirrored list."""

import re
import sys
import tomllib
from pathlib import Path

root = Path(sys.argv[1])
IMAGE_PROVIDED = {
    # the stock engine image carries the whole heavy stack; pip never touches these
    "vllm", "torch", "torchvision", "transformers", "safetensors", "tokenizers",
    "numpy", "pydantic", "pydantic-core", "pyyaml", "httpx", "jinja2",
    "aiohttp", "pillow", "tqdm", "fsspec",
}
requirements: list[str] = []
for manifest in (
    root / "pyproject.toml",
    root / "packages" / "rcp-ndcg-core" / "pyproject.toml",
    root / "packages" / "rcp-ndcg-vllm" / "pyproject.toml",
):
    if not manifest.is_file():
        continue
    data = tomllib.loads(manifest.read_text(encoding="utf-8"))
    requirements.extend(data["project"]["dependencies"])
# The reference extra of this package, whose torch/transformers the image provides but
# sentence-transformers it does not.
vllm_manifest = tomllib.loads((root / "packages" / "rcp-ndcg-vllm" / "pyproject.toml").read_text())
requirements += vllm_manifest["project"].get("optional-dependencies", {}).get("reference", [])
requirements += vllm_manifest["project"].get("optional-dependencies", {}).get("metrics", [])

out: list[str] = []
for requirement in requirements:
    name = re.split(r"[\s;\[<>=!~]", requirement, maxsplit=1)[0].strip().lower()
    if name in IMAGE_PROVIDED or name in {"rcp-ndcg", "rcp-ndcg-core"}:
        continue  # the image provides it, or the tarball install above already put it in
    requirement = re.sub(r"\s*;.*$", "", requirement).strip()  # drop environment markers
    if requirement and requirement not in out:
        out.append(requirement)
print(" ".join(out))
PY
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
# The light dependencies, derived from the packages' own pyproject.toml files at run time (never hand-mirrored):
# root (rcp-ndcg) + core + this package's base deps and reference extra, minus everything the image provides.
read -r -a LIGHT_DEPS <<<"$(derive_light_deps "$WORK/code")"
if (( ${#LIGHT_DEPS[@]} )); then
  pip install --constraint "$WORK/constraints.txt" "${LIGHT_DEPS[@]}"
fi

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
