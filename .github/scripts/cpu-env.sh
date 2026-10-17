#!/usr/bin/env bash
# Create the locked environment with the given extras, with CPU torch instead of the CUDA wheels.
#
#   .github/scripts/cpu-env.sh dev [docs mteb ...]
#
# On Linux the lock resolves torch with its CUDA dependencies (several GB). This installs everything else from the
# lock, then torch (and torchvision, when the extras lock it) at the locked versions from the PyTorch CPU index. Run
# commands afterwards with `uv run --no-sync`, so that uv does not swap the CPU build back. On macOS the locked torch
# is already a CPU build.
set -euo pipefail

extras=()
for extra in "$@"; do extras+=(--extra "$extra"); done

if [[ "$(uname -s)" != "Linux" ]]; then
  uv sync --locked "${extras[@]}"
  exit 0
fi

# The versions these extras lock (the lock holds one torch, shared by every extra that needs it).
torch=()
while read -r requirement; do torch+=("$requirement"); done < <(
  uv export --frozen --no-hashes --no-emit-workspace "${extras[@]}" | sed -n 's/^\(torch\|torchvision\)==\([^ ;]*\).*/\1==\2/p'
)

# torch's CUDA dependencies: the nvidia-* and cuda-* wheels, and triton.
skip=(--no-install-package torch --no-install-package torchvision --no-install-package triton)
while read -r name; do skip+=(--no-install-package "$name"); done < <(sed -n 's/^name = "\(\(nvidia\|cuda\)-[^"]*\)"/\1/p' uv.lock)
uv sync --locked "${extras[@]}" "${skip[@]}"

if ((${#torch[@]})); then
  uv pip install "${torch[@]}" --index-url https://download.pytorch.org/whl/cpu
  uv run --no-sync python -c "import torch; assert not torch.cuda.is_available(); print('torch', torch.__version__)"
fi
