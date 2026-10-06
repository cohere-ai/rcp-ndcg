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
#   reference a venv with --system-site-packages over the image's torch and CUDA, installing only what
#             requirements-reference.txt names from the wheelhouse; the recipes' reference.py runs as
#             <state>/reference/bin/python subprocesses.
#
# uv is installed with pip --target (the product's own bootstrap_uv location), never into the engine
# environment; UV_CACHE_DIR lives in the state directory. The auth script mounted at
# $RCP_GCS_AUTH_FILE is executed, never printed; HF_TOKEN reaches the engines from the job's secret.
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
  local sdk_bin
  for sdk_bin in "$HOME/google-cloud-sdk/bin" /root/google-cloud-sdk/bin /opt/google-cloud-sdk/bin \
    /usr/lib/google-cloud-sdk/bin /usr/local/google-cloud-sdk/bin; do
    if [[ -x "$sdk_bin/gcloud" || -x "$sdk_bin/gsutil" ]]; then
      export PATH="$sdk_bin:$PATH"
      break
    fi
  done
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

main() {
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

fetch_stage() { # fetch_stage SRC_URI PARENT: copy the stage directory into PARENT and print where it landed
  # A recursive copy into an existing directory nests the source under it (gcloud, gsutil and fsspec alike), so
  # copy into the parent and use <parent>/<basename>; never pre-create the target.
  local src="${1%/}" parent="$2" name
  name="$(basename "$src")"
  mkdir -p "$parent"
  [[ ! -e "$parent/$name" ]] || { echo "bootstrap: $parent/$name already exists" >&2; return 1; }
  gcs_cp "$src" "$parent/" >&2
  [[ -d "$parent/$name" ]] || { echo "bootstrap: the stage copy did not produce $parent/$name" >&2; return 1; }
  printf '%s\n' "$parent/$name"
}

STAGE="${STAGE_URI#gs://}"
if [[ "$STAGE" == "$STAGE_URI" ]]; then
  [[ -d "$STAGE_URI" ]] || { echo "bootstrap: not a stage directory: $STAGE_URI" >&2; exit 1; }
  STAGE_DIR="$STAGE_URI"
else
  fetch_start="$(now_s)"
  STAGE_DIR="$(fetch_stage "gs://$STAGE" "$STATE")"
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
CLIENT_ARGS=(uvx --from "$CLIENT_SPEC" --with "$CLIENT_WITH"
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
fi

# --- the engine's plugin wheels (none in wave 0), under the freeze-diff guard ------------------------

: >"$STATE/plugin-allowed.txt"
if [[ -n "${WAVE_LIST_FILE:-}" && -n "$RECIPES_ROOT" ]]; then
  PLUGINS="$("$STATE/client" python -m rcp_ndcg_vllm.jobs.plugins collect \
    --recipes-root "$RECIPES_ROOT" --recipes "@$WAVE_LIST_FILE")"
else
  PLUGINS=""
fi
if [[ -n "$PLUGINS" ]]; then
  echo "bootstrap: installing the recipes' plugin wheels into the engine environment (--no-deps)" >&2
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
      "$ENGINE_PYTHON" -m pip install --quiet --no-deps "$plugin_path"
    else
      # Not a staged file: a name on an index or in the wheelhouse (the item-9 fallback, declared).
      echo "bootstrap: the recipe's plugin $plugin is not staged; installing it as named" >&2
      "$ENGINE_PYTHON" -m pip install --quiet --no-deps "$plugin"
    fi
    freeze_name_of "${plugin_path:-$plugin}" >>"$STATE/plugin-allowed.txt"
  done <<<"$PLUGINS"
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

# --- the reference environment: the image's torch, read through --system-site-packages ---------------

ref_start="$(now_s)"
# The reference reads the image's torch and CUDA through --system-site-packages and installs only what
# is missing - pip's job, not uv's: uv ignores system site-packages during resolution and would install
# the wheelhouse's CPU torch over the image's CUDA build; pip sees the system distributions and skips
# them (a paper reference that needs other versions gets REFERENCE_REQUIREMENTS and its own venv).
uv venv --system-site-packages --seed "$STATE/reference" >/dev/null
"$STATE/reference/bin/python" -m pip install --quiet --no-index \
  --find-links "$STAGE_DIR/wheelhouse" \
  -r "${REFERENCE_REQUIREMENTS:-$STAGE_DIR/requirements-reference.txt}"
ref_s="$(( $(now_s) - ref_start ))"
"$STATE/reference/bin/python" - <<'PYEOF' >"$STATE/reference-versions.json"
import json
import sys

versions = {"python": ".".join(str(part) for part in sys.version_info[:3])}
try:
    import torch

    versions["torch"] = torch.__version__
except ImportError:
    versions["torch"] = None  # the image's torch is not visible: the reference will fail on use
try:
    import transformers

    versions["transformers"] = transformers.__version__
except ImportError:
    versions["transformers"] = None
print(json.dumps(versions))
PYEOF
python3 - "$STATE/reference-versions.json" "$ref_s" <<'PYEOF' >"$STATE/reference.json"
import json
import sys

versions = json.load(open(sys.argv[1]))
versions["install_s"] = int(sys.argv[2])
print(json.dumps(versions, indent=2))
PYEOF
echo "bootstrap: reference environment ready in ${ref_s}s ($(cat "$STATE/reference-versions.json"))" >&2

# --- the report --------------------------------------------------------------------------------------

REPORT_PY="${RCP_REPORT_PY:-/etc/rcp/files/report/report.py}"
REPORT="$STATE/bootstrap.json"
python3 "$REPORT_PY" init --file "$REPORT" --schema rcp-ndcg.bootstrap-report.v1 --started "$STARTED"
python3 "$REPORT_PY" merge --file "$REPORT" --key engine --fragment <(
  python3 - "$STATE/plugin-allowed.txt" "$ENGINE_PYTHON_VERSION" "$ENGINE_VLLM_VERSION" "$ENGINE_MEASURE_S" \
    "$STATE/engine-freeze-after.txt" "$STAGE_DIR/wheelhouse" "$PLUGIN_CANARY" <<'PYEOF'
import json
import sys
from pathlib import Path

allowed, python_version, vllm_version, measure_s, freeze_file, wheelhouse, canary = sys.argv[1:8]
plugins = [line.strip() for line in Path(allowed).read_text(encoding="utf-8").splitlines() if line.strip()]
freeze = [
    line.strip()
    for line in Path(freeze_file).read_text(encoding="utf-8").splitlines()
    if line.strip()
]
print(json.dumps({
    "python": python_version, "vllm": vllm_version, "freeze_unchanged": True, "plugins": plugins,
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
  exec "$STATE/client" python -m rcp_ndcg_vllm.jobs.run_wave \
    --recipes "@$WAVE_LIST_FILE" --recipes-root "$RECIPES_ROOT" --gpus "$gpus" \
    --out "$STATE/wave" --upload "$OUT_URI" --record \
    --reference-python "$STATE/reference/bin/python" "${pairs_args[@]+${pairs_args[@]}}"
fi
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
