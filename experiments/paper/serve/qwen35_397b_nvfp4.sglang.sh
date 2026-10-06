#!/usr/bin/env bash
# The paper's engine for its primary judge, Qwen3.5-397B-A17B in its NVFP4 release: SGLang on one node of eight
# Blackwell GPUs (tensor parallel 4 x data parallel 2). The judge preset `qwen35_397b_nvfp4` reaches it at
# http://127.0.0.1:8000/v1. Run it on the GPU host; the weights (about 250 GB) are read from $HF_HOME.
#
# For a job's engine phase, `image` is $IMAGE and `command` is the ENGINE array below.
set -euo pipefail

IMAGE=${IMAGE:-lmsysorg/sglang:v0.5.17-cu129}
ENGINE=(
  python3 -m sglang.launch_server
  --model-path nvidia/Qwen3.5-397B-A17B-NVFP4
  --revision 0368c1b3233414cd4a617b8ff9515e25752dc16c
  --served-model-name qwen3.5-397b
  --host 0.0.0.0 --port 8000
  # Parallelism. SGLang's current docs recommend `python3 -m sglang_router.launch_server --dp-size 2` for data
  # parallelism; the paper ran the flags below.
  --tensor-parallel-size 4 --data-parallel-size 2
  # NVFP4 weights, FP8 KV cache.
  --quantization modelopt_fp4 --kv-cache-dtype fp8_e4m3
  --mem-fraction-static 0.85
  # The judge preset's context_tokens.
  --context-length 262144
  # Scheduling and concurrency.
  --max-running-requests 1024 --chunked-prefill-size 32768 --max-prefill-tokens 32768
  --stream-interval 30 --tokenizer-worker-num 6
  --disable-radix-cache
  # Blackwell NVFP4 kernels.
  --attention-backend trtllm_mha --moe-runner-backend flashinfer_trtllm --fp4-gemm-backend flashinfer_cutlass
  --enable-flashinfer-allreduce-fusion
  # The hybrid (Gated DeltaNet) cache without an extra state buffer, since the radix cache is off.
  --mamba-scheduler-strategy no_buffer --page-size 64
  # The reasoning goes to its own channel, and the answer schema constrains only the answer.
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder
  --model-loader-extra-config '{"enable_multithread_load": "true", "num_threads": 64}'
  --enable-metrics --enable-metrics-for-all-schedulers
)

exec docker run --rm --gpus all --ipc=host --shm-size 32g -p 8000:8000 \
  -e HF_HOME=/models -e HF_TOKEN -v "${HF_HOME:-$HOME/.cache/huggingface}:/models" \
  "$IMAGE" "${ENGINE[@]}"
