#!/usr/bin/env bash
# The paper's engine for its second judge, gpt-oss-120b in its released precision (MXFP4 mixture-of-experts
# weights): SGLang on one node of eight H100 GPUs (tensor parallel 4 x data parallel 2). The judge preset
# `gpt_oss_120b` reaches it at http://127.0.0.1:8000/v1. Run it on the GPU host; the weights are read from $HF_HOME.
#
# For a run config's `serve:` section, `image` is $IMAGE and `command` is the ENGINE array below.
set -euo pipefail

IMAGE=${IMAGE:-lmsysorg/sglang:v0.5.17-cu129}
ENGINE=(
  python3 -m sglang.launch_server
  --model-path openai/gpt-oss-120b
  --served-model-name gpt-oss-120b
  --host 0.0.0.0 --port 8000
  # Parallelism. SGLang's current docs recommend `python3 -m sglang_router.launch_server --dp-size 2` for data
  # parallelism; the paper ran the flags below.
  --tensor-parallel-size 4 --data-parallel-size 2
  # The judge preset's context_tokens.
  --context-length 131072
  # Scheduling and concurrency.
  --max-running-requests 2048 --num-continuous-decode-steps 256 --cuda-graph-max-bs 256 --schedule-policy lpm
  --tokenizer-worker-num 16
  # The reasoning goes to its own channel, and the answer schema constrains only the answer.
  --reasoning-parser gpt-oss
  --enable-metrics --enable-metrics-for-all-schedulers
)

exec docker run --rm --gpus all --ipc=host --shm-size 32g -p 8000:8000 \
  -e HF_HOME=/models -e HF_TOKEN -v "${HF_HOME:-$HOME/.cache/huggingface}:/models" \
  "$IMAGE" "${ENGINE[@]}"
