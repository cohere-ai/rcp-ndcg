#!/usr/bin/env bash
# The node bootstrap: the stock image's three environments (GPU-VALIDATION.md, "the out-of-the-box image,
# separate environments"), built from one staged release candidate, then optionally the wave.
#
#   bootstrap.sh envs <STAGE_URI> --state <DIR> [--recipes <dir> --wave-list <file>]
#   bootstrap.sh wave <STAGE_URI> <OUT_URI> --wave <NAME>
#
#   STAGE_URI   gs://.../<rc-name>, the RC that rc_build.sh staged (or a local path to such a directory)
#   OUT_URI     gs://..., where run_wave.py --upload copies the wave's outputs
#   --state DIR where the environments' state lands: client (a wrapper that runs the client mechanism),
#               reference/ (the reference venv), the freeze files and bootstrap.json (the report)
#   --wave NAME the wave's list: <stage>/wave-lists/<NAME>.txt (or extra/*/wave-lists/<NAME>.txt)
#   --recipes/--wave-list  (envs mode) the recipes whose plugin wheels install into the engine
#               environment; without them nothing installs and the freeze must not change at all (wave 0)
#
# The three environments, never mixed:
#   engine    the image's own Python, which runs vllm serve - untouched, except recipe plugin wheels
#             installed with --no-deps; a pip freeze before/after must differ by exactly those wheels
#             (in wave 0: by nothing at all), or this script fails before any engine starts.
#   client    NO separate venv: every client command runs through the product's own install mechanism
#             (rcp_ndcg.runners.script.install_argv: uvx --find-links <wheelhouse> --no-index, held to the
#             staged constraints file) - the code users run, exercised by the waves.
#   reference one venv PER FAMILY, keyed by the family's reference.lock (owner decision 35): each is a
#             venv over the image's torch/CUDA (--system-site-packages) with the family's own lock pins
#             installed into it (--no-deps from the staged wheelhouse, then jobs/reference_deps.py
#             completes the venv's own missing deps to a fixed point); a family declaring own-torch
#             gets a venv without --system-site-packages and its lock's stack.  Every family is built
#             once per pod and reused by its variants; the import check fails loudly with the family
#             named.  The recipes' reference.py runs as <state>/reference/<family>/bin/python
#             subprocesses.
#
# uv is installed with pip --target (the product's own bootstrap_uv location), never into the engine
# environment; UV_CACHE_DIR lives in the state directory. The auth script mounted at
# $RCP_GCS_AUTH_FILE is executed, never printed; HF_TOKEN reaches the engines from the job's secret.
# A Cloud SDK it installs is put on PATH from the first of RCP_GCLOUD_SDK_DIRS holding a CLI (jobs/gcs.sh:
# colon-separated, the SDK's usual locations when unset, no search when empty).
# The staged files must hash to the manifest rc_build.sh wrote; anything else fails fast, one line.

set -euo pipefail

usage() {
  echo "usage: bootstrap.sh envs <STAGE_URI> --state <DIR> [--recipes <dir> --wave-list <file>]
       bootstrap.sh wave <STAGE_URI> <OUT_URI> --wave <NAME>" >&2
}

# The mounted auth script is executed FIRST - never printed, its output never echoed - because it sets
# up the credentials every transfer path uses (gcloud, gsutil, or the python gcsfs helper when the
# image ships neither CLI; see jobs/gcs.sh, which owns that dispatch).
auth() {
  if [[ ! -f "$AUTH_SCRIPT" ]]; then
    echo "bootstrap: no auth script at $AUTH_SCRIPT (set RCP_GCS_AUTH_FILE); GCS steps cannot run" >&2
    return 1
  fi
  local out
  out="$(mktemp)"
  local status=0
  # bash: mounted files carry no execute bit; only the exit code is reported, never the output.
  bash "$AUTH_SCRIPT" >"$out" 2>&1 || status=$?
  rm -f "$out"
  if ((status != 0)); then
    echo "bootstrap: the GCS auth script failed with exit code $status (its output is not echoed)" >&2
    return 1
  fi
}

now_s() { date +%s; }

# --- the freeze-diff guard: the engine environment may gain exactly the declared plugins -------------

# freeze_of PYTHON: the engine environment's pip freeze, one package per line.
freeze_of() { "$1" -m pip freeze --disable-pip-version-check 2>/dev/null; }

# freeze_name_of PATH_OR_SPEC: the canonical distribution name of an installed plugin, from a wheel
# filename (dashes escaped to underscores; the first dash separates name and version) or from a pip
# spec (a name, with or without a version specifier; its dashes are part of the name). Both normalize
# PEP 503 (lowercase, runs of '-_.' to one '-'), which is what pip's freeze line collapses to.
freeze_name_of() {
  local stem="${1##*/}"
  if [[ "$stem" == *.whl ]]; then
    stem="${stem%.whl}"
    stem="${stem%%-*}"  # a wheel filename's name part carries no dashes (they are escaped to '_')
  else
    stem="${stem%%[<>=!~; \[ ]*}"  # a version specifier or extras bracket keeps only the name
  fi
  printf '%s\n' "$(printf '%s' "$stem" | sed -E 's/[-_.]+/-/g' | tr '[:upper:]' '[:lower:]')"
}

# freeze_diff_guard BEFORE AFTER ALLOWED: fails when the freeze changed beyond the allowed names
# (anything added, removed or upgraded whose canonical name is not one of the declared plugin wheels).
freeze_diff_guard() {
  local before="$1" after="$2" allowed="$3" changed
  changed="$(
    { comm -13 <(sort -u "$before") <(sort -u "$after"); comm -23 <(sort -u "$before") <(sort -u "$after"); } \
      | awk '{ line = tolower($0); sub(/ @ .*/, "", line); gsub(/[-_.]+/, "-", line); sub(/==.*/, "", line); print line }' \
      | sort -u | grep -Fvx -f "$allowed" || true
  )"
  if [[ -n "$changed" ]]; then
    echo "bootstrap: the engine environment changed beyond the declared plugins" >&2
    echo "  changed: $changed" >&2
    return 1
  fi
  return 0
}

# install_plugin_wheels SPECS_FILE ALLOWED_FILE FAILED_FILE: install one plugin spec per line into the
# engine environment.  A spec that names a staged file installs from the staged tree; a name installs
# from the staged wheelhouses ONLY (--no-index --find-links "$STAGE_DIR/wheelhouse", plus each existing
# "$STAGE_DIR"/extra/*/wheelhouse - a wheel staged through rc_build's EXTRA_DIRS lands there - never an
# index: the stage is the whole truth).  A spec that cannot install (a plugin found nowhere) is appended to
# FAILED_FILE with its exact name - run_wave's --failed-plugins then fails exactly the recipes that
# name it - and this returns 0 even when every plugin failed: one failing recipe never stops the job.
install_plugin_wheels() {
  local specs_file="$1" allowed_file="$2" failed_file="$3" plugin plugin_path extra_wheelhouse
  local -a links=(--find-links "$STAGE_DIR/wheelhouse")
  for extra_wheelhouse in "$STAGE_DIR"/extra/*/wheelhouse; do
    if [[ -d "$extra_wheelhouse" ]]; then
      links+=(--find-links "$extra_wheelhouse")
    fi
  done
  while IFS= read -r plugin; do
    [[ -z "$plugin" ]] && continue
    plugin_path=""
    for candidate in "$STAGE_DIR/recipes/$plugin" "$STAGE_DIR/$plugin" "$RECIPES_ROOT/$plugin" "$plugin"; do
      if [[ -n "$candidate" && -e "$candidate" ]]; then
        plugin_path="$candidate"
        break
      fi
    done
    if [[ -n "$plugin_path" ]]; then
      if ! "$ENGINE_PYTHON" -m pip install --quiet --no-deps "$plugin_path"; then
        echo "bootstrap: the recipe's plugin $plugin could not be installed from $plugin_path;" \
          "the recipes that name it will fail (with the exact name)" >&2
        printf '%s\n' "$plugin" >>"$failed_file"
        continue
      fi
    else
      # Not a staged file: installed as named from the staged wheelhouse only.
      echo "bootstrap: the recipe's plugin $plugin is not staged; installing it from the staged wheelhouses" \
        "(${links[*]})" >&2
      if ! "$ENGINE_PYTHON" -m pip install --quiet --no-deps --no-index "${links[@]}" "$plugin"; then
        echo "bootstrap: the plugin $plugin is neither staged nor in a staged wheelhouse;" \
          "the recipes that name it will fail (with the exact name)" >&2
        printf '%s\n' "$plugin" >>"$failed_file"
        continue
      fi
    fi
    freeze_name_of "${plugin_path:-$plugin}" >>"$allowed_file"
  done <"$specs_file"
  return 0
}

# --- the reference venv's torch stack: the image's own, pinned and verified ---------------------------

# --- the reference environments: one venv per family, over the image's torch/CUDA -------------------

# reference_install REFERENCE_PYTHON REQUIREMENTS WHEELHOUSE: install the family lock's pins from the
# staged wheelhouse only, --no-deps: the family's own pins (transformers, sentence-transformers,
# flash-attn, ...) install INTO the venv and take precedence over the image's copies (owner decision
# 35), and --no-deps never resolves the image stack's own dependency tree (the image does not register
# it).  The lock carries no torch/CUDA-stack pins unless the family declares own-torch; in the default
# --system-site-packages venv the image's stack is read through the venv, never replaced.
reference_install() {
  local -a links=(--find-links "$3")
  local extra_wheelhouse
  for extra_wheelhouse in "${STAGE_DIR:-}"/extra/*/wheelhouse; do
    [[ -d "$extra_wheelhouse" ]] && links+=(--find-links "$extra_wheelhouse")
  done
  if ! "$1" -m pip install --quiet --no-deps --no-index "${links[@]}" -r "$2"; then
    echo "bootstrap: the family reference install failed (pip's output is above); the staged wheelhouse" >&2
    echo "  cannot satisfy the family's reference.lock - regenerate the lock or stage its wheels" >&2
    return 1
  fi
}

# reference_complete REFDEPS_PY REFERENCE_PYTHON WHEELHOUSE: complete the reference venv's OWN
# distributions' missing dependencies to a fixed point, each --no-deps from the staged wheelhouse(s)
# (the image's distributions are never completed - that would shadow its CUDA stack).
reference_complete() {
  local -a links=("$3")
  local extra_wheelhouse
  for extra_wheelhouse in "${STAGE_DIR:-}"/extra/*/wheelhouse; do
    [[ -d "$extra_wheelhouse" ]] && links+=("$extra_wheelhouse")
  done
  if ! "$2" "$1" "${links[@]}"; then
    echo "bootstrap: completing the reference venv's own missing dependencies failed (the output is" >&2
    echo "  above). Add them to the family's reference.lock, or stage their wheels in the wheelhouse." >&2
    return 1
  fi
}

# torch_probe PYTHON: the torch build that PYTHON sees, as one JSON line {"version":..., "cuda":...}
# where ``cuda`` is torch.version.cuda (null on a CPU build).  A python torch cannot import reports
# nulls: the probe records, never raises.
torch_probe() {
  "$1" - <<'PYEOF'
import json

try:
    import torch

    print(json.dumps({"version": torch.__version__, "cuda": torch.version.cuda}))
except Exception:  # noqa: BLE001 - the probe records, never raises
    print(json.dumps({"version": None, "cuda": None}))
PYEOF
}

# check_reference_torch REF_JSON IMAGE_JSON: does the reference see the image's own CUDA torch build?
# Prints true/false and: fails when the reference replaced the image's torch stack, cannot see torch at
# all, or sees a CPU torch where the image ships CUDA (a CPU torch on a GPU node is a failed bootstrap).
check_reference_torch() {
  python3 - "$1" "$2" <<'PYEOF'
import json
import sys

reference = json.load(open(sys.argv[1]))
image = json.load(open(sys.argv[2]))
if not image.get("version") and not reference.get("version"):
    print("false")  # the image carries no torch: nothing to keep
    raise SystemExit(0)
kept_image_build = bool(image.get("version")) and reference == image
is_image_build = bool(kept_image_build and image.get("cuda"))
print("true" if is_image_build else "false")
if kept_image_build:
    # The reference sees exactly the image's torch build (recorded CUDA or not: the image's own CPU
    # build is still the image's build, so nothing was replaced).
    raise SystemExit(0)
if image.get("version") and not reference.get("version"):
    print(
        f"bootstrap: the reference venv cannot import torch at all; the image's build is "
        f"{image.get('version')}",
        file=sys.stderr,
    )
    raise SystemExit(1)
if image.get("cuda") and not reference.get("cuda"):
    print(
        f"bootstrap: the reference venv's torch {reference.get('version')} is a CPU build where the "
        f"image ships CUDA torch {image.get('version')}: a CPU torch on a GPU node is a failed bootstrap",
        file=sys.stderr,
    )
    raise SystemExit(1)
print(
    f"bootstrap: the reference venv replaced the image's torch build {image.get('version')} "
    f"(cuda {image.get('cuda')}) with {reference.get('version')} (cuda {reference.get('cuda')}); a "
    "requirement that would replace the image's torch stack conflicts with the image's pins",
    file=sys.stderr,
)
raise SystemExit(1)
PYEOF
}

main() {
# The report and the reference completion are mounted helpers (subprocess scripts, never printed);
# both are defined before every use (set -u is active, and defining them late once broke the run).
REPORT_PY="${RCP_REPORT_PY:-/etc/rcp/files/report/report.py}"
REFERENCE_DEPS_PY="${RCP_REFERENCE_DEPS_PY:-/etc/rcp/files/refdeps/reference_deps.py}"
# The auth script runs before anything else (credentials first; then the transfer path is chosen -
# gcloud, gsutil, or the python helper when the image ships neither CLI - and recorded).
export AUTH_SCRIPT="${RCP_GCS_AUTH_FILE:-/etc/rcp/gcs_auth.sh}"
GCS_SH="${RCP_GCS_HELPER_SH:-/etc/rcp/files/gcshelper/gcs.sh}"
GCS_HELPER_PY="${RCP_GCS_HELPER_PY:-/etc/rcp/files/gcshelper/gcs.py}"
[[ -f "$GCS_SH" && -f "$GCS_HELPER_PY" ]] || {
  echo "bootstrap: the mounted GCS helpers are missing (gcs.sh: $GCS_SH, gcs.py: $GCS_HELPER_PY)" >&2
  exit 1
}
auth || exit 1
# shellcheck disable=SC1090  # the helper is mounted at a job-specific path
source "$GCS_SH"
gcs_sdk_on_path  # an SDK the auth script installed, searched in RCP_GCLOUD_SDK_DIRS (gcs.sh)
export GCS_PY="${GCS_PY:-$(command -v python3)}"
export GCS_WHEELHOUSE=""  # set once the stage is local; the helper prefers the staged gcsfs wheel

MODE="${1:-}"
[[ "$MODE" == "envs" || "$MODE" == "wave" ]] || { usage; exit 2; }
shift
STARTED="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

if [[ "$MODE" == "wave" ]]; then
  STAGE_URI="${1:?usage: bootstrap.sh wave <STAGE_URI> <OUT_URI> --wave <NAME>}"
  OUT_URI="${2:?usage: bootstrap.sh wave <STAGE_URI> <OUT_URI> --wave <NAME>}"
  shift 2
  WAVE_NAME=""
  WAVE_LIST_FILE=""
  while (($#)); do
    case "$1" in
      --wave) WAVE_NAME="${2:?--wave needs a name}"; shift 2 ;;
      --wave-list) WAVE_LIST_FILE="${2:?--wave-list needs a file}"; shift 2 ;;
      *) echo "bootstrap: unknown wave argument: $1" >&2; exit 2 ;;
    esac
  done
  [[ -n "$WAVE_NAME" ]] || { echo "bootstrap: wave mode needs --wave <NAME>" >&2; exit 2; }
  STATE="$(mktemp -d "${TMPDIR:-/tmp}/rcp-bootstrap.XXXXXX")"
  export GCS_TOOLS_DIR="${GCS_TOOLS_DIR:-$STATE/gcs-tools}"
  TRANSFER="$(gcs_transfer_detect)"
  export TRANSFER
  echo "bootstrap: GCS transfer path: $TRANSFER" >&2
else
  STAGE_URI=""
  STATE=""
  WAVE_LIST_FILE=""
  RECIPES_DIR=""
  while (($#)); do
    case "$1" in
      --state) STATE="${2:?--state needs a directory}"; shift 2 ;;
      --recipes) RECIPES_DIR="${2:?--recipes needs a directory}"; shift 2 ;;
      --wave-list) WAVE_LIST_FILE="${2:?--wave-list needs a file}"; shift 2 ;;
      *) STAGE_URI="${STAGE_URI:-$1}"; shift ;;
    esac
  done
  [[ -n "$STAGE_URI" && -n "$STATE" ]] || { usage; exit 2; }
  export GCS_TOOLS_DIR="${GCS_TOOLS_DIR:-$STATE/gcs-tools}"
  TRANSFER="$(gcs_transfer_detect)"
  export TRANSFER
  echo "bootstrap: GCS transfer path: $TRANSFER" >&2
  if [[ -n "$WAVE_LIST_FILE" || -n "$RECIPES_DIR" ]]; then
    [[ -n "$WAVE_LIST_FILE" && -n "$RECIPES_DIR" ]] || {
      echo "bootstrap: --wave-list and --recipes go together (the plugins of the listed recipes)" >&2
      exit 2
    }
  fi
  mkdir -p "$STATE"
fi

# --- the staged RC -----------------------------------------------------------------------------------

fetch_stage() { # fetch_stage SRC_URI DST: copy the staged RC's contents into DST (same layout on every path)
  # gcs_cp copies a directory source's CONTENTS under the destination, so the stage lands directly at
  # DST (manifest.json at DST/manifest.json) whichever transfer runs; the destination is pre-created -
  # gcloud refuses a missing local one.
  local src="${1%/}" dst="$2"
  [[ ! -e "$dst" ]] || { echo "bootstrap: $dst already exists" >&2; return 1; }
  mkdir -p "$dst"
  gcs_cp "$src" "$dst/" dir
}

STAGE="${STAGE_URI#gs://}"
if [[ "$STAGE" == "$STAGE_URI" ]]; then
  [[ -d "$STAGE_URI" ]] || { echo "bootstrap: not a stage directory: $STAGE_URI" >&2; exit 1; }
  STAGE_DIR="$STAGE_URI"
else
  fetch_start="$(now_s)"
  STAGE_DIR="$STATE/stage"
  fetch_stage "gs://$STAGE" "$STAGE_DIR"
  echo "bootstrap: stage downloaded in $(( $(now_s) - fetch_start ))s via $TRANSFER" >&2
fi
[[ -f "$STAGE_DIR/manifest.json" ]] || {
  echo "bootstrap: $STAGE_DIR holds no manifest.json (not an rc_build.sh stage)" >&2
  exit 1
}

manifest_field() { # manifest_field NAME: the manifest's top-level field
  python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$STAGE_DIR/manifest.json" "$1"
}

ENVS_START="$(now_s)"
VERSION="$(manifest_field version)"
[[ -n "$VERSION" ]] || { echo "bootstrap: the manifest carries no version" >&2; exit 1; }
echo "bootstrap: staged RC version $VERSION (commit $(manifest_field commit))" >&2

# The staged files must hash to the manifest (a tampered or partial stage never installs).
echo "bootstrap: verifying the staged files against the manifest" >&2
python3 - "$STAGE_DIR" <<'PYEOF'
import hashlib
import json
import sys
from pathlib import Path

stage = Path(sys.argv[1])
manifest = json.loads((stage / "manifest.json").read_text(encoding="utf-8"))
bad = [
    entry["path"]
    for entry in manifest["files"]
    if not (stage / entry["path"]).is_file()
    or hashlib.sha256((stage / entry["path"]).read_bytes()).hexdigest() != entry["sha256"]
]
if bad:
    print(f"bootstrap: these staged files do not hash to the manifest: {bad}", file=sys.stderr)
    raise SystemExit(1)
print(f"bootstrap: {len(manifest['files'])} staged files verified", file=sys.stderr)
PYEOF

# --- uv, outside the engine environment --------------------------------------------------------------

UV_DIR="${TMPDIR:-/tmp}/rcp-ndcg-uv"  # the product's own bootstrap_uv location (one home for uv)
if ! command -v uvx >/dev/null; then
  echo "bootstrap: installing uv with pip --target (never into the engine environment)" >&2
  python3 -m pip install --quiet --target "$UV_DIR" uv
fi
export PATH="$UV_DIR/bin:$PATH"
export UV_CACHE_DIR="$STATE/uv-cache"
mkdir -p "$UV_CACHE_DIR"

# --- the engine environment: measured, untouched until the (optional) plugin wheels -------------------

ENGINE_PYTHON="$(command -v python3)"
"$ENGINE_PYTHON" -c "import vllm" 2>/dev/null || {
  echo "bootstrap: the image's $ENGINE_PYTHON cannot import vllm; this is not the engine image" >&2
  exit 1
}
ENGINE_PYTHON_VERSION="$("$ENGINE_PYTHON" -c 'import platform; print(platform.python_version())')"
ENGINE_VLLM_VERSION="$("$ENGINE_PYTHON" -c 'import vllm; print(vllm.__version__)')"
freeze_of "$ENGINE_PYTHON" >"$STATE/engine-freeze-before.txt"
echo "bootstrap: engine environment: python $ENGINE_PYTHON_VERSION, vllm $ENGINE_VLLM_VERSION" >&2

# --- the client environment: the product's own install mechanism (uvx), from the staged wheelhouse ---

CLIENT_SPEC="rcp-ndcg-vllm[test]==${VERSION}"
CLIENT_WITH="rcp-ndcg[hf]==${VERSION}"
CLIENT_TEST="rcp-ndcg-test==${VERSION}"  # unpublished, staged in the wheelhouse: the wave runner and the checks live here
CLIENT_ARGS=(uvx --from "$CLIENT_SPEC" --with "$CLIENT_WITH" --with "$CLIENT_TEST"
  --constraints "$STAGE_DIR/requirements-constraints.txt"
  --find-links "$STAGE_DIR/wheelhouse" --no-index)
install_start="$(now_s)"
# The CUDA-lock wheels (nvidia-*, triton) ride in the wheelhouse for the engine; the CLIENT environment
# must never install them (its specs pull no torch, and the manifest marks them inert). The probe runs
# IN the client environment - under the client mechanism, not this shell's python (the engine
# environment legitimately carries them on the stock image).
INERT_NAMES="$(python3 - "$STAGE_DIR/manifest.json" <<'INERTPY'
import json
import re
import sys

manifest = json.load(open(sys.argv[1]))
names = sorted({
    re.sub(r"[-_.]+", "-", wheel.rsplit("/", 1)[-1].split("-")[0]).lower()
    for wheel in manifest.get("cpu_inert_wheels", [])
})
print(" ".join(names))
INERTPY
)"
# shellcheck disable=SC2086  # INERT_NAMES is a deliberate word list
"${CLIENT_ARGS[@]}" python - "$STAGE_DIR/requirements-constraints.txt" $INERT_NAMES <<'PYEOF' >"$STATE/client-versions.json"
import json
import sys
from importlib.metadata import PackageNotFoundError, version

import rcp_ndcg
import rcp_ndcg_core
import rcp_ndcg_vllm


def installed(dist):
    try:
        return version(dist)
    except PackageNotFoundError:
        return None


print(json.dumps({
    "rcp-ndcg": rcp_ndcg.__version__,
    "rcp-ndcg-core": version("rcp-ndcg-core"),
    "rcp-ndcg-vllm": version("rcp-ndcg-vllm"),
    "inert_present": {dist: installed(dist) for dist in sys.argv[2:]},
}))
PYEOF
client_install_s="$(( $(now_s) - install_start ))"
python3 - "$STATE/client-versions.json" "$VERSION" <<'VCHK' || exit 1
import json
import sys

versions = json.load(open(sys.argv[1]))
expected = sys.argv[2]
inert = versions.pop("inert_present", {})
bad = {name: seen for name, seen in versions.items() if seen != expected}
if bad:
    print(f"bootstrap: the client installed {bad}, not the manifest's {expected}", file=sys.stderr)
    raise SystemExit(1)
# The client environment is checked for the CUDA-lock wheels the manifest marks inert: none may be there.
present = {dist: seen for dist, seen in inert.items() if seen}
if present:
    print(f"bootstrap: the client environment gained CUDA-lock wheels it must not have: {present}", file=sys.stderr)
    raise SystemExit(1)
VCHK


# The state's client wrapper: every later client command runs the same mechanism (one home per concept).
cat >"$STATE/client" <<WRAPPER
#!/usr/bin/env bash
# The client environment of this node (built by bootstrap.sh): the product's install mechanism
# (uvx --find-links wheelhouse --no-index, held to the staged constraints), from the staged wheelhouse.
set -euo pipefail
export UV_CACHE_DIR="${UV_CACHE_DIR}"
export PATH="${UV_DIR}/bin:\$PATH"
exec ${CLIENT_ARGS[*]} "\$@"
WRAPPER
chmod +x "$STATE/client"
echo "bootstrap: client environment ready in ${client_install_s}s ($CLIENT_SPEC)" >&2
python3 - "$STATE/client-versions.json" "$client_install_s" <<'PYEOF' >"$STATE/client.json"
import json
import sys

versions = json.load(open(sys.argv[1]))
versions["mechanism"] = "uvx --find-links <wheelhouse> --no-index --constraints <staged>"
versions["install_s"] = int(sys.argv[2])
print(json.dumps(versions, indent=2))
PYEOF

# --- the wave's recipes and list (both modes; wave 0 has neither) -------------------------------------

RECIPES_ROOT=""
if [[ "$MODE" == "wave" ]]; then
  if [[ -z "$WAVE_LIST_FILE" ]]; then
    found=""
    for candidate in "$STAGE_DIR/wave-lists/$WAVE_NAME.txt" "$STAGE_DIR"/extra/*/wave-lists/"$WAVE_NAME.txt"; do
      [[ -f "$candidate" ]] && { found="$candidate"; break; }
    done
    [[ -n "$found" ]] || { echo "bootstrap: no wave list for '$WAVE_NAME' under $STAGE_DIR (wave-lists/)" >&2; exit 1; }
    WAVE_LIST_FILE="$found"
  fi
  RECIPES_ROOT="$STATE/recipes"
  mkdir -p "$RECIPES_ROOT"
  for root in "$STAGE_DIR/recipes" "$STAGE_DIR"/extra/*/recipes; do
    [[ -d "$root" ]] || continue
    for recipe_dir in "$root"/*/; do
      ln -sfn "$(realpath "$recipe_dir")" "$RECIPES_ROOT/$(basename "$recipe_dir")"
    done
  done
elif [[ -n "${RECIPES_DIR:-}" ]]; then
  # Envs mode with the plugins' recipes named explicitly: the given directory is the root.
  RECIPES_ROOT="$RECIPES_DIR"
else
  # Envs mode without a wave list (wave 0): the staged family directories (the reference environments
  # resolve from them; the first family exercises the mechanism).
  RECIPES_ROOT="$STAGE_DIR/recipes"
fi

# --- the engine's plugin wheels (none in wave 0), under the freeze-diff guard ------------------------

: >"$STATE/plugin-allowed.txt"
: >"$STATE/plugin-failures.txt"
if [[ -n "${WAVE_LIST_FILE:-}" && -n "$RECIPES_ROOT" ]]; then
  # The collect step has already skipped (and reported) any invalid recipe of the wave list; the wave
  # marks those failed with the validation message.  What remains are the plugin specs to install.
  "$STATE/client" python -m rcp_ndcg_test.jobs.plugins collect \
    --recipes-root "$RECIPES_ROOT" --recipes "@$WAVE_LIST_FILE" >"$STATE/plugin-specs.txt"
  if [[ -s "$STATE/plugin-specs.txt" ]]; then
    echo "bootstrap: installing the recipes' plugin wheels into the engine environment (--no-deps)" >&2
    install_plugin_wheels "$STATE/plugin-specs.txt" "$STATE/plugin-allowed.txt" "$STATE/plugin-failures.txt"
  fi
fi
freeze_of "$ENGINE_PYTHON" >"$STATE/engine-freeze-after.txt"
ENGINE_MEASURE_S="$(( $(now_s) - ENVS_START ))"
freeze_diff_guard "$STATE/engine-freeze-before.txt" "$STATE/engine-freeze-after.txt" "$STATE/plugin-allowed.txt"
if [[ ! -s "$STATE/plugin-allowed.txt" ]]; then
  # The plugin canary (wave 0): with no plugin wheel installed, the plugin ecosystem's own dependency
  # (fla) must not be importable in the engine environment - a leak the freeze guard cannot name.
  if "$ENGINE_PYTHON" -c 'import fla' >/dev/null 2>&1; then
    echo "bootstrap: the plugin canary failed: fla is importable in the untouched engine environment" >&2
    exit 1
  fi
  PLUGIN_CANARY="fla not importable"
else
  PLUGIN_CANARY="skipped: this wave installs plugin wheels"
fi
export PLUGIN_CANARY

# --- the reference environments: one venv per family, over the image's torch/CUDA -------------------

ref_start="$(now_s)"
# The wave's families (owner decision 35): one environment per family, keyed by the family's
# reference.lock, reused by every variant of the family.  The list resolves through the product's own
# loader in the client environment; a recipe that does not load is reported and left to the wave's
# failed row.  In envs mode without a wave list (wave 0) the first staged family exercises the
# mechanism.  A family declaring own-torch gets a venv WITHOUT --system-site-packages (its lock pins
# the stack); every other family reads the image's torch/CUDA through --system-site-packages and
# installs only its own pins into the venv.
: >"$STATE/reference-rows.jsonl"
if [[ -n "${WAVE_LIST_FILE:-}" && -n "${RECIPES_ROOT:-}" ]]; then
  "$STATE/client" python -m rcp_ndcg_test.jobs.reference_env families \
    --recipes-root "$RECIPES_ROOT" --recipes "@$WAVE_LIST_FILE" >"$STATE/reference-families.tsv"
elif [[ -n "${RECIPES_ROOT:-}" && -d "$RECIPES_ROOT" ]]; then
  first_family=""
  for candidate in "$RECIPES_ROOT"/*/; do
    [[ -f "$candidate/reference.lock" ]] || continue
    first_family="$(basename "$candidate")"
    break
  done
  if [[ -n "$first_family" ]]; then
    "$STATE/client" python -m rcp_ndcg_test.jobs.reference_env families \
      --recipes-root "$RECIPES_ROOT" --family "$first_family" >"$STATE/reference-families.tsv"
  fi
fi
while IFS=$'\t' read -r family lock own_torch; do
  [[ -n "$family" ]] || continue
  env_dir="$STATE/reference/$family"
  family_start="$(now_s)"
  echo "bootstrap: building the reference environment for family $family (own-torch $own_torch)" >&2
  if [[ "$own_torch" == "true" ]]; then
    uv venv --seed "$env_dir" >/dev/null
  else
    uv venv --system-site-packages --seed "$env_dir" >/dev/null
  fi
  reference_install "$env_dir/bin/python" "$lock" "$STAGE_DIR/wheelhouse"
  reference_complete "$REFERENCE_DEPS_PY" "$env_dir/bin/python" "$STAGE_DIR/wheelhouse"
  check_json="$STATE/reference-$family-check.json"
  if ! "$STATE/client" python -m rcp_ndcg_test.jobs.reference_env check \
      --python "$env_dir/bin/python" --lock "$lock" >"$check_json"; then
    echo "bootstrap: the reference environment for family $family failed its import check" >&2
    exit 1
  fi
  TORCH_IS_IMAGE_BUILD="false"
  torch_probe "$env_dir/bin/python" >"$STATE/reference-$family-torch.json"
  if [[ "$own_torch" != "true" ]]; then
    # The default family reads the image's torch build: a CPU torch on a GPU node, a replaced stack or
    # no torch at all is a failed bootstrap, named with the family.
    torch_probe "$ENGINE_PYTHON" >"$STATE/engine-torch.json"
    if ! TORCH_IS_IMAGE_BUILD="$(check_reference_torch "$STATE/reference-$family-torch.json" "$STATE/engine-torch.json")"; then
      echo "bootstrap: the reference venv of family $family does not carry the image's torch build" >&2
      echo "  (the message above is the reason)" >&2
      exit 1
    fi
  fi
  python3 - "$check_json" "$lock" "$family" "$own_torch" "$TORCH_IS_IMAGE_BUILD" "$(( $(now_s) - family_start ))" <<'PYEOF' >>"$STATE/reference-rows.jsonl"
import json
import sys
from pathlib import Path

check_file, lock, family, own_torch, torch_is_image_build, install_s = sys.argv[1:7]
checked = json.loads(Path(check_file).read_text(encoding="utf-8"))
header = {}
for line in Path(lock).read_text(encoding="utf-8").splitlines():
    if not line.startswith("# "):
        continue
    name, _, value = line[2:].partition(":")
    if name in ("image", "own-torch") and value:
        header[name] = value.strip()
print(json.dumps({
    "family": family,
    "lock_sha256": checked.get("lock_sha256"),
    "image": header.get("image"),
    "own_torch": own_torch == "true",
    "torch_is_image_build": torch_is_image_build == "true",
    "install_s": int(install_s),
    "facts": checked.get("facts", {}),
}))
PYEOF
done <"$STATE/reference-families.tsv"
ref_s="$(( $(now_s) - ref_start ))"
python3 - "$STATE/reference-rows.jsonl" "$ref_s" <<'PYEOF' >"$STATE/reference.json"
import json
import sys
from pathlib import Path

rows = [json.loads(line) for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines() if line.strip()]
print(json.dumps({
    "families": {row["family"]: row for row in rows},
    "n_families": len(rows),
    "install_s": int(sys.argv[2]),
}, indent=2))
PYEOF
echo "bootstrap: $(( $(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["n_families"])' "$STATE/reference.json") )) reference environment(s) ready in ${ref_s}s" >&2

# --- the report --------------------------------------------------------------------------------------

REPORT="$STATE/bootstrap.json"
python3 "$REPORT_PY" init --file "$REPORT" --schema rcp-ndcg.bootstrap-report.v1 --started "$STARTED"
python3 "$REPORT_PY" merge --file "$REPORT" --key engine --fragment <(
  python3 - "$STATE/plugin-allowed.txt" "$STATE/plugin-failures.txt" "$ENGINE_PYTHON_VERSION" "$ENGINE_VLLM_VERSION" "$ENGINE_MEASURE_S" \
    "$STATE/engine-freeze-after.txt" "$STAGE_DIR/wheelhouse" "$PLUGIN_CANARY" <<'PYEOF'
import json
import sys
from pathlib import Path

allowed, failed, python_version, vllm_version, measure_s, freeze_file, wheelhouse, canary = sys.argv[1:9]
plugins = [line.strip() for line in Path(allowed).read_text(encoding="utf-8").splitlines() if line.strip()]
plugins_failed = [line.strip() for line in Path(failed).read_text(encoding="utf-8").splitlines() if line.strip()]
freeze = [
    line.strip()
    for line in Path(freeze_file).read_text(encoding="utf-8").splitlines()
    if line.strip()
]
print(json.dumps({
    "python": python_version, "vllm": vllm_version, "freeze_unchanged": True, "plugins": plugins,
    "plugins_failed": plugins_failed,
    "measure_s": int(measure_s), "freeze": freeze, "wheelhouse": wheelhouse, "plugin_canary": canary,
}))
PYEOF
)
python3 "$REPORT_PY" merge --file "$REPORT" --key client --fragment "$STATE/client.json"
python3 "$REPORT_PY" merge --file "$REPORT" --key reference --fragment "$STATE/reference.json"
python3 "$REPORT_PY" emit --file "$REPORT"

# --- wave mode: the wave runs through the client mechanism -------------------------------------------

if [[ "$MODE" == "wave" ]]; then
  pairs_args=()
  for candidate in "$STAGE_DIR/pairs" "$STAGE_DIR"/extra/*/pairs; do
    if [[ -d "$candidate" ]]; then
      pairs_args+=(--pairs-dir "$candidate")
      break
    fi
  done
  gpus="$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo 0)"
  echo "bootstrap: running the wave '$WAVE_NAME' on $gpus GPUs (list: $WAVE_LIST_FILE)" >&2
  store_args=()
  if [[ -n "${RCP_REFERENCE_STORE:-}" ]]; then
    # A previous wave's downloaded reference store: the wave computes only the missing/stale outputs.
    store_args=(--reference-store "$RCP_REFERENCE_STORE")
  fi
  exec "$STATE/client" python -m rcp_ndcg_test.jobs.run_wave \
    --recipes "@$WAVE_LIST_FILE" --recipes-root "$RECIPES_ROOT" --gpus "$gpus" \
    --out "$STATE/wave" --upload "$OUT_URI" --record \
    --failed-plugins "$STATE/plugin-failures.txt" \
    --reference-root "$STATE/reference" "${store_args[@]+${store_args[@]}}" \
    "${pairs_args[@]+${pairs_args[@]}}"
fi
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
