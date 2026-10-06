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
#                     (private plugin wheels, pairs, private recipes or wave lists)
#
# Staged layout (what a wave installs from; see docs/how-to/release-candidates.md):
#   <RC_NAME>/dist/                        the six release files, as release.yml builds them
#   <RC_NAME>/wheelhouse/                  the release wheels + every locked dependency (the CPU torch build)
#   <RC_NAME>/requirements-constraints.txt the lock's export, the install's constraints file
#   <RC_NAME>/recipes/                     the recipe directories (each with recipe.yaml and reference.py)
#   <RC_NAME>/plugins/                     public plugin packages, when the package ships any
#   <RC_NAME>/wave-lists/<wave>.txt        the wave lists (one recipe id per line)
#   <RC_NAME>/pairs/                       the stage-2 pairs files, when the checkout has any
#   <RC_NAME>/extra/<name>/                the EXTRA_DIRS entries, as they are
#   <RC_NAME>/manifest.json                the commit, the version and the SHA-256 of every staged file

set -euo pipefail

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

VERSION="$(sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml | head -1)"
[[ -n "$VERSION" ]] || { echo "rc_build: cannot read the version from pyproject.toml" >&2; exit 1; }
echo "rc_build: version $VERSION from $COMMIT"

# --- the build and check steps of release.yml ---------------------------------------------

echo "rc_build: building the three distributions (uv build --all-packages, then the workspace outsider)"
uv build --all-packages --out-dir dist
(cd packages/rcp-ndcg-vllm && uv build --out-dir ../../dist)

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
grep -qx "  \"rcp-ndcg-core==${VERSION}\"," pyproject.toml || {
  echo "rc_build: rcp-ndcg does not pin rcp-ndcg-core==${VERSION}" >&2
  exit 1
}
grep -qx "  \"rcp-ndcg==${VERSION}\"," packages/rcp-ndcg-vllm/pyproject.toml || {
  echo "rc_build: rcp-ndcg-vllm does not pin rcp-ndcg==${VERSION}" >&2
  exit 1
}

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
cp packages/rcp-ndcg-vllm/requirements-reference.txt stage/"$RC_NAME"/requirements-reference.txt
uv venv "$WORK/dl" --python 3.12 >/dev/null
uv pip install --python "$WORK/dl/bin/python" pip >/dev/null
"$WORK/dl/bin/python" -m pip download --quiet \
  -r stage/"$RC_NAME"/requirements-constraints.txt \
  -r packages/rcp-ndcg-vllm/requirements-reference.txt \
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

# The staged tree beside the wheels: recipes, plugins, wave lists, pairs, and the EXTRA_DIRS entries.
mkdir -p stage/"$RC_NAME"
cp -r packages/rcp-ndcg-vllm/recipes stage/"$RC_NAME"/recipes
if [[ -d packages/rcp-ndcg-vllm/plugins ]]; then
  cp -r packages/rcp-ndcg-vllm/plugins stage/"$RC_NAME"/plugins
fi
if [[ -d wave-lists ]]; then
  cp -r wave-lists stage/"$RC_NAME"/wave-lists
else
  echo "rc_build: the checkout has no wave-lists/ directory; stage one via EXTRA_DIRS or commit it" >&2
fi
if [[ -d pairs ]]; then
  cp -r pairs stage/"$RC_NAME"/pairs
fi
if [[ -n "${EXTRA_DIRS:-}" ]]; then
  for extra in $EXTRA_DIRS; do
    [[ -d "$extra" ]] || { echo "rc_build: EXTRA_DIRS entry is not a directory: $extra" >&2; exit 1; }
    mkdir -p stage/"$RC_NAME"/extra
    cp -r "$extra" stage/"$RC_NAME"/extra/"$(basename "$extra")"
  done
fi

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
manifest = {
    "schema": "rcp-ndcg.rc-manifest.v1",
    "rc_name": rc_name,
    "version": version,
    "commit": commit,
    "created": datetime.now(UTC).isoformat(timespec="seconds"),
    "built_with": {"python": py, "uv": uv_v},
    "files": files,
}
(stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(f"rc_build: manifest lists {len(files)} files")
PYEOF

echo "rc_build: staging to ${RCP_STAGE_PREFIX%/}/$RC_NAME/"
if command -v gcloud >/dev/null; then
  gcloud storage cp -r stage/"$RC_NAME" "${RCP_STAGE_PREFIX%/}/"
else
  gsutil -m cp -r stage/"$RC_NAME" "${RCP_STAGE_PREFIX%/}/"
fi
echo "rc_build: staged $(du -sh stage/"$RC_NAME" | cut -f1) to ${RCP_STAGE_PREFIX%/}/$RC_NAME/ (manifest: manifest.json)"
