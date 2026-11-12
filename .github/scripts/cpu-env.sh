#!/usr/bin/env bash
# Create the locked environment with the given extras, with CPU torch instead of the CUDA wheels.
#
#   .github/scripts/cpu-env.sh dev [docs mteb ...]
#
# On Linux the lock resolves torch with its CUDA dependencies (several GB). This installs everything else from the
# lock, then torch (and torchvision, when the extras lock it) at the locked versions from the PyTorch CPU index. Run
# commands afterwards with `uv run --no-sync`, so that uv does not swap the CPU build back. On macOS the locked torch
# is already a CPU build.
#
# Both branches pin `--python 3.12`, and the pin is deliberate rather than incidental: a runner image's default
# Python moves (3.15 on the macOS image; 3.14.4 on the Linux one, which is how a `functools.partial` descriptor
# change reached main -- the seven corpus metrics passed `self` as a third positional argument), and `uv sync`
# follows it. The pin is the interpreter the lock, basedpyright's `pythonVersion` and every other CI venv use, so a
# job tests the tree the developer's own run tests. It says nothing about support: `requires-python` is `>=3.12`,
# 3.14 stays supported, and the product must keep working on it.
set -euo pipefail

extras=()
for extra in "$@"; do extras+=(--extra "$extra"); done

if [[ "$(uname -s)" != "Linux" ]]; then
  # The runner's own Python moves (3.15 on the macOS image), and torch has no wheel for it: pin the version the
  # lock and every other CI venv use.
  uv sync --locked --python 3.12 "${extras[@]}"
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
# `--python 3.12`: the same pin as the macOS branch above, for the reason stated there.
uv sync --locked --python 3.12 "${extras[@]}" "${skip[@]}"

if ((${#torch[@]})); then
  uv pip install "${torch[@]}" --index-url https://download.pytorch.org/whl/cpu
  uv run --no-sync python -c "import torch; assert not torch.cuda.is_available(); print('torch', torch.__version__)"
fi
