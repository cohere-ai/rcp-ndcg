#!/usr/bin/env bash
# The --no-deps freeze check the GPU wave runs before any engine starts: installing the
# plugin wheel into a venv over the engine environment must change `pip freeze` by
# exactly this one distribution — the engine environment stays untouched except for the
# plugin itself.
#
# Usage: check_no_deps_freeze.sh <wheel> [venv-python]
#   <wheel>       the rcp_ndcg_vllm_pplx-*.whl built by `uv build rcp-ndcg-vllm/plugins/pplx`
#   [venv-python] the python to build the venv from; default `python3` (on the node: the
#                 engine image's own python, via a --system-site-packages venv so the
#                 engine environment itself is never touched)
set -euo pipefail

WHEEL=$1
PYTHON=${2:-python3}

VENV=$(mktemp -d /tmp/pplx-freeze-XXXXXX)
trap 'rm -rf "$VENV"' EXIT

"$PYTHON" -m venv --system-site-packages "$VENV/venv"
PIP="$VENV/venv/bin/pip"

"$PIP" freeze | sort > "$VENV/before.txt"
"$PIP" install --no-deps --quiet "$WHEEL"
"$PIP" freeze | sort > "$VENV/after.txt"

CHANGED=$(diff "$VENV/before.txt" "$VENV/after.txt" || true)
ADDED=$(printf '%s\n' "$CHANGED" | grep -c '^>' || true)

echo "$CHANGED"
if [ "$ADDED" -ne 1 ] || ! printf '%s\n' "$CHANGED" | grep '^>' | grep -q 'rcp-ndcg-vllm-pplx'; then
  echo "FAIL: installing $WHEEL did not add exactly the rcp-ndcg-vllm-pplx distribution:" >&2
  echo "$CHANGED" >&2
  exit 1
fi
if printf '%s\n' "$CHANGED" | grep -q '^<'; then
  echo "FAIL: the install removed or changed other distributions:" >&2
  printf '%s\n' "$CHANGED" | grep '^<' >&2
  exit 1
fi
echo "OK: pip freeze changed by exactly one distribution: $(printf '%s\n' "$CHANGED" | grep '^>')"
