#!/usr/bin/env bash
# Create the locked environment with the given extras, with CPU torch instead of the CUDA wheels.
#
#   .github/scripts/cpu-env.sh dev [docs mteb ...]
#
# On Linux the lock resolves torch with its CUDA dependencies (several GB). This installs everything else from the
# lock, then torch and torchvision from the PyTorch CPU index. Run commands afterwards with `uv run --no-sync`, so
# that uv does not swap the CPU build back. On macOS the locked torch is already a CPU build.
set -euo pipefail

extras=()
for extra in "$@"; do extras+=(--extra "$extra"); done

if [[ "$(uname -s)" != "Linux" ]]; then
  uv sync --locked "${extras[@]}"
  exit 0
fi

skip=(--no-install-package torch --no-install-package torchvision --no-install-package triton)
while read -r name; do skip+=(--no-install-package "$name"); done < <(sed -n 's/^name = "\(nvidia[^"]*\)"/\1/p' uv.lock)
uv sync --locked "${extras[@]}" "${skip[@]}"

torchvision=$(grep -A1 '^name = "torchvision"$' uv.lock | sed -n 's/^version = "\(.*\)"/\1/p' | head -1)
uv pip install "torch==2.9.1" "torchvision==${torchvision}" --index-url https://download.pytorch.org/whl/cpu
uv run --no-sync python -c "import torch; assert not torch.cuda.is_available(); print('torch', torch.__version__)"
