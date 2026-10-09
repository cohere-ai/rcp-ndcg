#!/usr/bin/env bash
# Build one release candidate exactly as .github/workflows/release.yml builds a release, stage it with the
# wheelhouse the node installs from, and write the manifest. The operator runs this on a machine with uv,
# Python 3.12, and access to PyPI (and the bucket); the node never builds anything.
#
#   rc_build.sh <RC_NAME> [<COMMIT>]
#
#     RC_NAME   the candidate's name (e.g. rc0); everything stages to <RCP_STAGE_PREFIX>/<RC_NAME>/
#     COMMIT    the commit to build (default HEAD); built from a detached worktree, so a dirty checkout is fine
#
# Environment:
#   RCP_STAGE_PREFIX  required: the private stage location, from the operator's command line only
#                     (a tracked file names no machine or bucket); <prefix>/<RC_NAME>/ is created
#   EXTRA_DIRS        optional, space-separated extra directories staged under extra/<basename>/
#                     (private pairs, wave lists or recipes)
#
# Staged layout (what a wave installs from; see docs/how-to/release-candidates.md):
#   <RC_NAME>/dist/                        the six release files, as release.yml builds them
#   <RC_NAME>/wheelhouse/                  the release wheels + every locked dependency (the CPU torch build)
#   <RC_NAME>/requirements-constraints.txt the lock's export, the install's constraints file
#   <RC_NAME>/recipes/                     the recipes, from the rcp-ndcg-vllm wheel's package data
#                                          (layout-move item 3: the recipes ship inside the wheel)
#   <RC_NAME>/wave-lists/<wave>.txt        the wave lists (one recipe id per line)
#   <RC_NAME>/pairs/                       the stage-2 pairs files, from rcp-ndcg-test/pairs/ (one home)
#   <RC_NAME>/extra/<name>/                the EXTRA_DIRS entries, as they are
#   <RC_NAME>/manifest.json                the commit, the version and the SHA-256 of every staged file

set -euo pipefail

# The published distributions, in release order (release.yml builds exactly these, one --package per
# invocation; the fourth workspace member rcp-ndcg-test is unpublished and is never built -- GPU-E1:
# an --all-packages build staged its wheel and the dist/ check then refused the extra file).
PUBLISHED_PACKAGES=("rcp-ndcg-core" "rcp-ndcg" "rcp-ndcg-vllm")

build_published() { # build_published OUT_DIR: the three published distributions, by name, as release.yml
  local package
  for package in "${PUBLISHED_PACKAGES[@]}"; do
    uv build --package "$package" --out-dir "$1"
  done
}

# The pairs files' one home in the checkout (the wave runner consumes <pairs-dir>/<recipe>.jsonl).
PACKAGES_PAIRS="rcp-ndcg-test/pairs"

stage_pairs() {
  # stage_pairs <SRC-checkout> <STAGE-dir>: stage <SRC>/rcp-ndcg-test/pairs/ as <STAGE>/pairs/.
  # One home: a stray <SRC>/pairs/ or a pre-layout <SRC>/rcp-ndcg-vllm/pairs/ is refused (never silently
  # staged), and no pairs at all stages none (the wave runner then reports its missing pairs).
  local src="$1" stage="$2" stray
  if [[ -d "$src/$PACKAGES_PAIRS" ]]; then
    cp -r "$src/$PACKAGES_PAIRS" "$stage/pairs"
    echo "rc_build: staged the pairs files from $PACKAGES_PAIRS"
  elif [[ -d "$src/pairs" || -d "$src/rcp-ndcg-vllm/pairs" ]]; then
    for stray in "$src/pairs" "$src/rcp-ndcg-vllm/pairs"; do
      [[ -d "$stray" ]] || continue
      echo "rc_build: pairs have one home: $PACKAGES_PAIRS (found a stray $stray instead; move the files)" >&2
    done
    return 1
  else
    echo "rc_build: no $PACKAGES_PAIRS in the checkout; staging no pairs (the wave runner reports its missing pairs)"
  fi
}

# stage_recipes SRC STAGE [WHEEL]: the recipes the node runs are the BUILT rcp-ndcg-vllm wheel's package
# data (layout-move item 3: the recipes ship inside the wheel).  The whole rcp_ndcg_vllm/recipes/ directory
# is extracted from the wheel -- never a per-recipe file list and never the pre-move source path -- so a
# recipe file missing from the wheel fails this build, not a wave that silently runs an older recipe set.
# WHEEL names the wheel explicitly (the test/rehearsal path); without it the newest wheel in <SRC>/dist is
# used, which is what every real build has just produced and checked.
stage_recipes() {
  local src="$1" stage="$2" wheel="${3:-}" candidate scratch
  if [[ -z "$wheel" ]]; then
    for candidate in "$src"/dist/rcp_ndcg_vllm-*.whl; do
      [[ -f "$candidate" ]] && { wheel="$candidate"; break; }
    done
  fi
  [[ -n "$wheel" && -f "$wheel" ]] || {
    echo "rc_build: no rcp_ndcg_vllm wheel in $src/dist (stage_recipes)" >&2
    return 1
  }
  scratch="$stage/.wheel-recipes"
  rm -rf "$scratch"
  python3 -m zipfile -e "$wheel" "$scratch"
  if [[ ! -d "$scratch/rcp_ndcg_vllm/recipes" ]]; then
    echo "rc_build: the wheel $(basename "$wheel") carries no rcp_ndcg_vllm/recipes/ package data" >&2
    rm -rf "$scratch"
    return 1
  fi
  rm -rf "$stage/recipes"
  cp -r "$scratch/rcp_ndcg_vllm/recipes" "$stage/recipes"
  rm -rf "$scratch"
  echo "rc_build: staged the recipes from $(basename "$wheel")'s package data"
}

# The wave lists' one home in the checkout (owner decision, 2026-10-09: the lists are committed under the
# tooling home, decision 20; generated from the recipe catalog by jobs/wavelist.py, never hand-maintained).
PACKAGES_WAVE_LISTS="rcp-ndcg-test/wave-lists"

# stage_wave_lists SRC STAGE: stage <SRC>/rcp-ndcg-test/wave-lists/ as <STAGE>/wave-lists/ (what the
# node's bootstrap resolves a wave name against).  One home: a stray <SRC>/wave-lists/ is refused; no
# lists at all stages none (EXTRA_DIRS can carry private ones), said on stderr.
stage_wave_lists() {
  local src="$1" stage="$2"
  if [[ -d "$src/$PACKAGES_WAVE_LISTS" ]]; then
    cp -r "$src/$PACKAGES_WAVE_LISTS" "$stage/wave-lists"
    echo "rc_build: staged the wave lists from $PACKAGES_WAVE_LISTS"
  elif [[ -d "$src/wave-lists" ]]; then
    echo "rc_build: wave lists have one home: $PACKAGES_WAVE_LISTS (found a stray $src/wave-lists instead; move the lists)" >&2
    return 1
  else
    echo "rc_build: no $PACKAGES_WAVE_LISTS in the checkout; staging no wave lists" >&2
  fi
}

# stage_tree SRC STAGE [WHEEL]: everything the node reads beside the wheels -- the recipes (from the built
# wheel's package data), the wave lists, the pairs files and the EXTRA_DIRS entries.  Split from the build
# so the staging half is testable (and rehearsable) without building or uploading.
stage_tree() {
  local src="$1" stage="$2" wheel="${3:-}" extra
  mkdir -p "$stage"
  stage_recipes "$src" "$stage" "$wheel" || return 1
  stage_wave_lists "$src" "$stage" || return 1
  stage_pairs "$src" "$stage" || return 1
  if [[ -n "${EXTRA_DIRS:-}" ]]; then
    for extra in $EXTRA_DIRS; do
      [[ -d "$extra" ]] || { echo "rc_build: EXTRA_DIRS entry is not a directory: $extra" >&2; return 1; }
      mkdir -p "$stage/extra"
      cp -r "$extra" "$stage/extra/$(basename "$extra")"
    done
  fi
}

# check_pins SRC VERSION: the release.yml pin checks.  rcp-ndcg pins rcp-ndcg-core exactly; the lean
# rcp-ndcg-vllm names NO sibling package (decision 18: there is no lockstep version pin between the two),
# read with tomllib over every dependency scope -- never by matching TOML text.
check_pins() {
  local src="$1" version="$2"
  grep -qx "  \"rcp-ndcg-core==${version}\"," "$src/rcp-ndcg/pyproject.toml" || {
    echo "rc_build: rcp-ndcg does not pin rcp-ndcg-core==${version}" >&2
    return 1
  }
  python3 - "$src/rcp-ndcg-vllm/pyproject.toml" <<'PYEOF' || return 1
import re
import sys
import tomllib
from pathlib import Path

# name, optional [extras], then the specifier: "rcp-ndcg[calibrate]==0.0.1" -> ("rcp-ndcg", "==0.0.1")
requirement_re = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?\s*(.*)$")

def canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()

manifest = Path(sys.argv[1])
project = tomllib.loads(manifest.read_text(encoding="utf-8"))["project"]
declared = list(project.get("dependencies", []))
for extra in project.get("optional-dependencies", {}).values():
    declared += extra
siblings = []
for requirement in declared:
    match = requirement_re.match(requirement.split(";", 1)[0])
    if match is not None and canonical(match.group(1)) == "rcp-ndcg":
        siblings.append(requirement.strip())
if siblings:
    print(
        f"rc_build: {manifest} must name no sibling package (decision 18: no lockstep version pin between "
        f"rcp-ndcg and rcp-ndcg-vllm), found {siblings}",
        file=sys.stderr,
    )
    raise SystemExit(1)
PYEOF
}

rc_build_main() {
RC_NAME="${1:?usage: rc_build.sh <RC_NAME> [<COMMIT>]}"
COMMIT="${2:-HEAD}"
: "${RCP_STAGE_PREFIX:?set RCP_STAGE_PREFIX to the stage location (a gs:// or /shared/ URI; the RC stages to <prefix>/<RC_NAME>)}"

: "${CPU_INDEX:=https://download.pytorch.org/whl/cpu}"
: "${PYPI_INDEX:=https://pypi.org/simple}"

for tool in uv git python3; do
  command -v "$tool" >/dev/null || { echo "rc_build: $tool is not on PATH (build it where release.yml's job runs)" >&2; exit 2; }
done
if ! command -v gcloud >/dev/null && ! command -v gsutil >/dev/null; then
  echo "rc_build: neither gcloud nor gsutil is on PATH (staging is a storage upload)" >&2
  exit 2
fi

WORK="$(mktemp -d "${TMPDIR:-/tmp}/rcp-rc-build.XXXXXX")"
trap 'git worktree remove --force "$WORK/checkout" >/dev/null 2>&1 || true; rm -rf "$WORK"' EXIT

echo "rc_build: checking out $COMMIT into a detached worktree"
git worktree add --detach "$WORK/checkout" "$COMMIT" >/dev/null
SRC="$WORK/checkout"
cd "$SRC"

VERSION="$(sed -n 's/^version = "\(.*\)"/\1/p' rcp-ndcg/pyproject.toml | head -1)"
[[ -n "$VERSION" ]] || { echo "rc_build: cannot read the version from rcp-ndcg/pyproject.toml" >&2; exit 1; }
echo "rc_build: version $VERSION from $COMMIT"

# --- the build and check steps of release.yml ---------------------------------------------

echo "rc_build: building the three published distributions, by name (never --all-packages: the"
echo "  unpublished workspace member rcp-ndcg-test must not stage a wheel)"
build_published dist

echo "rc_build: checking the versions and the pins"
expected=(
  "dist/rcp_ndcg-${VERSION}-py3-none-any.whl" "dist/rcp_ndcg-${VERSION}.tar.gz"
  "dist/rcp_ndcg_core-${VERSION}-py3-none-any.whl" "dist/rcp_ndcg_core-${VERSION}.tar.gz"
  "dist/rcp_ndcg_vllm-${VERSION}-py3-none-any.whl" "dist/rcp_ndcg_vllm-${VERSION}.tar.gz"
)
for file in "${expected[@]}"; do
  [[ -f "$file" ]] || { echo "rc_build: missing $file: the checkout's version is not $VERSION" >&2; exit 1; }
done
test "$(find dist -type f -not -name '.*' | wc -l)" -eq "${#expected[@]}" || {
  echo "rc_build: dist/ holds other versions" >&2
  exit 1
}
check_pins "$SRC" "$VERSION" || exit 1

echo "rc_build: checking requirements-constraints.txt against the lock"
command_line="$(sed -n '2s/^#    //p' requirements-constraints.txt)"
[[ -n "$command_line" ]] || { echo "rc_build: requirements-constraints.txt carries no export command" >&2; exit 1; }
export_command="${command_line/ -o requirements-constraints.txt/ --no-header -o $WORK/exported.txt}"
(cd "$SRC" && eval "$export_command")
diff <(grep -v '^#' requirements-constraints.txt) <(grep -v '^#' "$WORK/exported.txt") || {
  echo "rc_build: requirements-constraints.txt is not the lock's export (regenerate it)" >&2
  exit 1
}

echo "rc_build: twine check"
uvx twine check dist/*

# The wheelhouse: the release wheels plus every locked dependency for the node's platform (the CPU
# torch build included), so a node install never asks an index (node-runtime item 3).
echo "rc_build: building the wheelhouse (this downloads the locked dependencies; a few minutes)"
mkdir -p stage/"$RC_NAME"/wheelhouse
cp -r dist stage/"$RC_NAME"/dist
cp dist/* stage/"$RC_NAME"/wheelhouse/
cp requirements-constraints.txt stage/"$RC_NAME"/requirements-constraints.txt
# REF-ENVS SEAM (owner decision 35): the reference environment is per recipe family next; the ref-envs
# lane stages one reference lock (and its inputs) per family here and the bootstrap installs each into
# its own venv.  Until that lands, the single shared requirements-reference.txt is what every reference
# installs from -- the two lines below (this copy and the pip download) move together; do not redesign
# them without the ref-envs lane.
cp rcp-ndcg-vllm/requirements-reference.txt stage/"$RC_NAME"/requirements-reference.txt
uv venv "$WORK/dl" --python 3.12 >/dev/null
uv pip install --python "$WORK/dl/bin/python" pip >/dev/null
"$WORK/dl/bin/python" -m pip download --quiet \
  -r stage/"$RC_NAME"/requirements-constraints.txt \
  -r rcp-ndcg-vllm/requirements-reference.txt \
  -c stage/"$RC_NAME"/requirements-constraints.txt \
  "rcp-ndcg-vllm[test]==${VERSION}" "rcp-ndcg[hf]==${VERSION}" \
  --dest stage/"$RC_NAME"/wheelhouse \
  --find-links stage/"$RC_NAME"/wheelhouse \
  --only-binary :all: \
  --index-url "$CPU_INDEX" --extra-index-url "$PYPI_INDEX"

# A fresh-venv install from the wheelhouse alone: the candidate installs and answers (release smoke).
echo "rc_build: fresh-venv install smoke from the wheelhouse"
uv venv "$WORK/smoke" --python 3.12
uv pip install --python "$WORK/smoke/bin/python" --no-index \
  --find-links stage/"$RC_NAME"/wheelhouse \
  --constraints stage/"$RC_NAME"/requirements-constraints.txt \
  "rcp-ndcg==${VERSION}"
"$WORK/smoke/bin/rcp-ndcg" --version
"$WORK/smoke/bin/rcp-ndcg" --help >/dev/null

# The staged tree beside the wheels: the recipes from the built wheel's package data, the wave lists
# from rcp-ndcg-test/wave-lists, the pairs from rcp-ndcg-test/pairs, and the EXTRA_DIRS entries
# (layout-move item 3: no separate plugin wheels -- the folded models ship inside rcp-ndcg-vllm).
stage_tree "$SRC" "stage/$RC_NAME" || exit 1

python3 - "$SRC" "$RC_NAME" "$VERSION" "$WORK/dl/bin/python" <<'PYEOF'
"""Write manifest.json: the commit, the version and the SHA-256 of every staged file."""
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

src, rc_name, version, dl_python = sys.argv[1:5]
stage = Path("stage") / rc_name
files = []
for path in sorted(stage.rglob("*")):
    if path.is_file() and path.name != "manifest.json":
        files.append(
            {
                "path": str(path.relative_to(stage)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
        )
commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=src).stdout.strip()
py = subprocess.run([dl_python, "-c", "import sys; print('%d.%d.%d' % sys.version_info[:3])"], capture_output=True, text=True).stdout.strip()
uv_v = subprocess.run(["uv", "--version"], capture_output=True, text=True).stdout.strip()
# The CUDA-lock wheels (nvidia-*, triton) ride along because the constraints file pins them on Linux;
# a CPU client never installs them (the client specs pull no torch at all). The manifest names them so
# a reviewer can tell the inert bulk from what the node actually reads.
_INERT_PREFIXES = ("nvidia-", "nvidia_", "triton")
cpu_inert = sorted(
    entry["path"].rsplit("/", 1)[-1]
    for entry in files
    if entry["path"].endswith(".whl")
    and entry["path"].rsplit("/", 1)[-1].lower().startswith(_INERT_PREFIXES)
)
manifest = {
    "schema": "rcp-ndcg.rc-manifest.v1",
    "rc_name": rc_name,
    "version": version,
    "commit": commit,
    "created": datetime.now(UTC).isoformat(timespec="seconds"),
    "built_with": {"python": py, "uv": uv_v},
    "cpu_inert_wheels": cpu_inert,
    "files": files,
}
(stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(f"rc_build: manifest lists {len(files)} files ({len(cpu_inert)} inert on a CPU client)")
PYEOF

echo "rc_build: staging to ${RCP_STAGE_PREFIX%/}/$RC_NAME/"
if command -v gcloud >/dev/null; then
  gcloud storage cp -r stage/"$RC_NAME" "${RCP_STAGE_PREFIX%/}/"
else
  gsutil -m cp -r stage/"$RC_NAME" "${RCP_STAGE_PREFIX%/}/"
fi
echo "rc_build: staged $(du -sh stage/"$RC_NAME" | cut -f1) to ${RCP_STAGE_PREFIX%/}/$RC_NAME/ (manifest: manifest.json)"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  rc_build_main "$@"
fi
