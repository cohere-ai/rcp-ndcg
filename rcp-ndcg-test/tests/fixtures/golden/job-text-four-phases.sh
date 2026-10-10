#!/usr/bin/env bash
#SBATCH --job-name=rcp-RUN-ID
#SBATCH --output=/e2e/out/logs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=0-08:00:00
set -euo pipefail
if ! command -v uvx >/dev/null; then
  python3 -m pip install --quiet --target "${TMPDIR:-/tmp}/rcp-ndcg-uv" uv
  export PATH="${TMPDIR:-/tmp}/rcp-ndcg-uv/bin:$PATH"
fi
mkdir -p /tmp/rcp-e2e-text-four-phases/tmp-8100 /tmp/rcp-e2e-text-four-phases/tmp-8110 /tmp/rcp-e2e-text-four-phases/tmp-8120
read -r -d '' WORKER_1 <<'RCP_NDCG_WORKER_1' || true
#!/usr/bin/env bash
set -euo pipefail
cd /e2e/runs/rcp-text-four-phases
export PYTHONPATH=/e2e/out/probe-site
export RCP_E2E_PROBE_JSONL=/e2e/out/client-probe.jsonl
exec uvx --from 'rcp-ndcg[calibrate,hf,s3,azure]==0.0.1' --constraints /stage/requirements-constraints.txt --find-links /stage/wheelhouse --no-index rcp-ndcg run resume --run /e2e/runs/rcp-text-four-phases --only retrieve
RCP_NDCG_WORKER_1
read -r -d '' ENGINE_ENCODER <<'RCP_NDCG_ENGINE_ENCODER' || true
export CUDA_VISIBLE_DEVICES=0
export VLLM_PORT=9100
export TMPDIR=/tmp/rcp-e2e-text-four-phases/tmp-8100
export RCP_NDCG_VLLM_PATCHES=''
exec vllm serve Qwen/Qwen3-Embedding-0.6B --revision 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3 --served-model-name qwen3-embedding-0.6b --host 0.0.0.0 --port 8100 --tensor-parallel-size 1 --runner pooling --dtype bfloat16 --max-model-len 32768 --hf-overrides '{}' --pooler-config '{}'
RCP_NDCG_ENGINE_ENCODER
if ((BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 3))); then
  echo "rcp-ndcg: this job needs bash 4.3 or later (for wait -n), and its bash is $BASH_VERSION; use an image or node with a newer bash" >&2
  exit 1
fi
if ! command -v python3 >/dev/null; then
  echo "rcp-ndcg: this job needs python3 on PATH (the engine readiness probe uses its standard library), and there is none; use an image or node that has python3" >&2
  exit 1
fi
rcp_ndcg_stop() {  # SIGTERM the processes, SIGKILL what is left after the grace period, and reap them
  local pid waited=0
  local -a left
  for pid in "$@"; do kill -TERM "$pid" 2>/dev/null || true; done
  while true; do
    left=()
    for pid in "$@"; do if kill -0 "$pid" 2>/dev/null; then left+=("$pid"); fi; done
    if ((${#left[@]} == 0)); then break; fi
    if ((waited >= 200)); then  # tenths of a second
      kill -KILL "${left[@]}" 2>/dev/null || true
      break
    fi
    sleep 0.1
    waited=$((waited + 1))
  done
  for pid in "$@"; do  # reap what this script started, so the next phase's wait -n cannot see it
    wait "$pid" 2>/dev/null || true
  done
}
rcp_ndcg_cleanup() {
  trap '' TERM INT
  local -a pids=()
  if [[ -n "${RCP_NDCG_COORDINATOR_PID:-}" ]]; then pids+=("$RCP_NDCG_COORDINATOR_PID"); fi
  if [[ -n "${RCP_NDCG_ENGINE_PID:-}" ]]; then pids+=("${RCP_NDCG_ENGINE_PID}"); fi
  if ((${#pids[@]})); then rcp_ndcg_stop "${pids[@]}"; fi
}
trap rcp_ndcg_cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
srun --overlap --nodes=1 --ntasks-per-node=1 --kill-on-bad-exit=1 --wait=10 --gres=gpu:1 bash -c "$ENGINE_ENCODER" &
RCP_NDCG_ENGINE_PID=$!
rcp_ndcg_any_ready() {  # PORT PATH HOST...
  local port="$1" path="$2"; shift 2
  for host in "$@"; do
    python3 -c 'import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=5)' "http://$host:$port$path" >/dev/null 2>&1 && return 0
  done
  return 1
}
rcp_ndcg_wait_ready() {  # PID_VAR TIMEOUT PORT PATH HOST...
  local pid_var="$1" timeout="$2" port="$3" path="$4"; shift 4
  local deadline=$((SECONDS + timeout)) status
  echo "rcp-ndcg: waiting up to $timeout s for an engine replica to answer $path" >&2
  until rcp_ndcg_any_ready "$port" "$path" "$@"; do
    if [[ -n "${!pid_var:-}" ]] && ! kill -0 "${!pid_var}" 2>/dev/null; then
      status=0
      wait "${!pid_var}" || status=$?
      printf -v "$pid_var" ''
      echo "rcp-ndcg: the engine exited with status $status before it answered $path; its output is above" >&2
      exit 1
    fi
    if ((SECONDS >= deadline)); then
      echo "rcp-ndcg: no engine replica answered $path within $timeout s (serve.startup_timeout_s); stopping the job" >&2
      exit 1
    fi
    sleep 10
  done
}
rcp_ndcg_wait_ready RCP_NDCG_ENGINE_PID 1800 8100 /v1/models 127.0.0.1
export RCP_NDCG_ENGINES='{"encoder": {"urls": ["http://127.0.0.1:8100/v1"], "wait_on_outage_s": 300}}'
bash -c "$WORKER_1" &
RCP_NDCG_COORDINATOR_PID=$!
status=0
wait -n || status=$?
if kill -0 "$RCP_NDCG_COORDINATOR_PID" 2>/dev/null; then
  echo "rcp-ndcg: the encoder engine exited with status $status while the run was going; stopping the run (submit it again with rcp-ndcg run resume --run <run dir> --runner <this runner>)" >&2
  exit 1
fi
failed=0
for pid in ${RCP_NDCG_ENGINE_PID}; do
  if ! kill -0 "$pid" 2>/dev/null; then
    es=0
    wait "$pid" 2>/dev/null || es=$?
    if [ "$es" -ne 0 ] && [ "$es" -ne 127 ]; then failed=$es; fi
  fi
done
if [ "$failed" -ne 0 ]; then
  echo "rcp-ndcg: the encoder engine exited with status $failed while the run was going; stopping the run (submit it again with rcp-ndcg run resume --run <run dir> --runner <this runner>)" >&2
  exit 1
fi
if [ "$status" -ne 0 ]; then
  exit "$status"
fi
rcp_ndcg_stop "$RCP_NDCG_ENGINE_PID"
read -r -d '' WORKER_2 <<'RCP_NDCG_WORKER_2' || true
#!/usr/bin/env bash
set -euo pipefail
cd /e2e/runs/rcp-text-four-phases
export PYTHONPATH=/e2e/out/probe-site
export RCP_E2E_PROBE_JSONL=/e2e/out/client-probe.jsonl
exec uvx --from 'rcp-ndcg[calibrate,hf,s3,azure]==0.0.1' --constraints /stage/requirements-constraints.txt --find-links /stage/wheelhouse --no-index rcp-ndcg run resume --run /e2e/runs/rcp-text-four-phases --only rerank
RCP_NDCG_WORKER_2
read -r -d '' ENGINE_RERANKER <<'RCP_NDCG_ENGINE_RERANKER' || true
export CUDA_VISIBLE_DEVICES=0
export VLLM_PORT=9110
export TMPDIR=/tmp/rcp-e2e-text-four-phases/tmp-8110
export RCP_NDCG_VLLM_PATCHES=''
exec vllm serve Qwen/Qwen3-Reranker-0.6B --revision e61197ed45024b0ed8a2d74b80b4d909f1255473 --served-model-name qwen3-reranker-0.6b --host 0.0.0.0 --port 8110 --tensor-parallel-size 1 --runner pooling --dtype bfloat16 --max-model-len 10000 --hf-overrides '{"architectures": ["Qwen3ForSequenceClassification"], "classifier_from_token": ["no", "yes"], "is_original_qwen3_reranker": true}' --chat-template /e2e/recipes/qwen3-reranker/template.jinja --pooler-config '{"use_activation": true}'
RCP_NDCG_ENGINE_RERANKER
if ((BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 3))); then
  echo "rcp-ndcg: this job needs bash 4.3 or later (for wait -n), and its bash is $BASH_VERSION; use an image or node with a newer bash" >&2
  exit 1
fi
if ! command -v python3 >/dev/null; then
  echo "rcp-ndcg: this job needs python3 on PATH (the engine readiness probe uses its standard library), and there is none; use an image or node that has python3" >&2
  exit 1
fi
rcp_ndcg_stop() {  # SIGTERM the processes, SIGKILL what is left after the grace period, and reap them
  local pid waited=0
  local -a left
  for pid in "$@"; do kill -TERM "$pid" 2>/dev/null || true; done
  while true; do
    left=()
    for pid in "$@"; do if kill -0 "$pid" 2>/dev/null; then left+=("$pid"); fi; done
    if ((${#left[@]} == 0)); then break; fi
    if ((waited >= 200)); then  # tenths of a second
      kill -KILL "${left[@]}" 2>/dev/null || true
      break
    fi
    sleep 0.1
    waited=$((waited + 1))
  done
  for pid in "$@"; do  # reap what this script started, so the next phase's wait -n cannot see it
    wait "$pid" 2>/dev/null || true
  done
}
rcp_ndcg_cleanup() {
  trap '' TERM INT
  local -a pids=()
  if [[ -n "${RCP_NDCG_COORDINATOR_PID:-}" ]]; then pids+=("$RCP_NDCG_COORDINATOR_PID"); fi
  if [[ -n "${RCP_NDCG_ENGINE_PID:-}" ]]; then pids+=("${RCP_NDCG_ENGINE_PID}"); fi
  if ((${#pids[@]})); then rcp_ndcg_stop "${pids[@]}"; fi
}
trap rcp_ndcg_cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
srun --overlap --nodes=1 --ntasks-per-node=1 --kill-on-bad-exit=1 --wait=10 --gres=gpu:1 bash -c "$ENGINE_RERANKER" &
RCP_NDCG_ENGINE_PID=$!
rcp_ndcg_any_ready() {  # PORT PATH HOST...
  local port="$1" path="$2"; shift 2
  for host in "$@"; do
    python3 -c 'import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=5)' "http://$host:$port$path" >/dev/null 2>&1 && return 0
  done
  return 1
}
rcp_ndcg_wait_ready() {  # PID_VAR TIMEOUT PORT PATH HOST...
  local pid_var="$1" timeout="$2" port="$3" path="$4"; shift 4
  local deadline=$((SECONDS + timeout)) status
  echo "rcp-ndcg: waiting up to $timeout s for an engine replica to answer $path" >&2
  until rcp_ndcg_any_ready "$port" "$path" "$@"; do
    if [[ -n "${!pid_var:-}" ]] && ! kill -0 "${!pid_var}" 2>/dev/null; then
      status=0
      wait "${!pid_var}" || status=$?
      printf -v "$pid_var" ''
      echo "rcp-ndcg: the engine exited with status $status before it answered $path; its output is above" >&2
      exit 1
    fi
    if ((SECONDS >= deadline)); then
      echo "rcp-ndcg: no engine replica answered $path within $timeout s (serve.startup_timeout_s); stopping the job" >&2
      exit 1
    fi
    sleep 10
  done
}
rcp_ndcg_wait_ready RCP_NDCG_ENGINE_PID 1800 8110 /v1/models 127.0.0.1
export RCP_NDCG_ENGINES='{"reranker": {"urls": ["http://127.0.0.1:8110/v1"], "wait_on_outage_s": 300}}'
bash -c "$WORKER_2" &
RCP_NDCG_COORDINATOR_PID=$!
status=0
wait -n || status=$?
if kill -0 "$RCP_NDCG_COORDINATOR_PID" 2>/dev/null; then
  echo "rcp-ndcg: the reranker engine exited with status $status while the run was going; stopping the run (submit it again with rcp-ndcg run resume --run <run dir> --runner <this runner>)" >&2
  exit 1
fi
failed=0
for pid in ${RCP_NDCG_ENGINE_PID}; do
  if ! kill -0 "$pid" 2>/dev/null; then
    es=0
    wait "$pid" 2>/dev/null || es=$?
    if [ "$es" -ne 0 ] && [ "$es" -ne 127 ]; then failed=$es; fi
  fi
done
if [ "$failed" -ne 0 ]; then
  echo "rcp-ndcg: the reranker engine exited with status $failed while the run was going; stopping the run (submit it again with rcp-ndcg run resume --run <run dir> --runner <this runner>)" >&2
  exit 1
fi
if [ "$status" -ne 0 ]; then
  exit "$status"
fi
rcp_ndcg_stop "$RCP_NDCG_ENGINE_PID"
read -r -d '' WORKER_3 <<'RCP_NDCG_WORKER_3' || true
#!/usr/bin/env bash
set -euo pipefail
cd /e2e/runs/rcp-text-four-phases
export PYTHONPATH=/e2e/out/probe-site
export RCP_E2E_PROBE_JSONL=/e2e/out/client-probe.jsonl
exec uvx --from 'rcp-ndcg[calibrate,hf,s3,azure]==0.0.1' --constraints /stage/requirements-constraints.txt --find-links /stage/wheelhouse --no-index rcp-ndcg run resume --run /e2e/runs/rcp-text-four-phases --only tournament --only rubric
RCP_NDCG_WORKER_3
read -r -d '' ENGINE_JUDGE <<'RCP_NDCG_ENGINE_JUDGE' || true
export CUDA_VISIBLE_DEVICES=0
export VLLM_PORT=9120
export TMPDIR=/tmp/rcp-e2e-text-four-phases/tmp-8120
export RCP_NDCG_VLLM_PATCHES=''
exec vllm serve nvidia/Qwen3.8-Flash-Next-NVFP4 --revision fc694b54fb0174e0913e6adf86691ef85a4ead47 --served-model-name judge --host 0.0.0.0 --port 8120 --tensor-parallel-size 1 --runner generate --dtype auto --max-model-len 131072 --hf-overrides '{}' --pooler-config '{}' --limit-mm-per-prompt '{"image": 10}' --quantization modelopt_fp4 --reasoning-parser qwen3
RCP_NDCG_ENGINE_JUDGE
if ((BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 3))); then
  echo "rcp-ndcg: this job needs bash 4.3 or later (for wait -n), and its bash is $BASH_VERSION; use an image or node with a newer bash" >&2
  exit 1
fi
if ! command -v python3 >/dev/null; then
  echo "rcp-ndcg: this job needs python3 on PATH (the engine readiness probe uses its standard library), and there is none; use an image or node that has python3" >&2
  exit 1
fi
rcp_ndcg_stop() {  # SIGTERM the processes, SIGKILL what is left after the grace period, and reap them
  local pid waited=0
  local -a left
  for pid in "$@"; do kill -TERM "$pid" 2>/dev/null || true; done
  while true; do
    left=()
    for pid in "$@"; do if kill -0 "$pid" 2>/dev/null; then left+=("$pid"); fi; done
    if ((${#left[@]} == 0)); then break; fi
    if ((waited >= 200)); then  # tenths of a second
      kill -KILL "${left[@]}" 2>/dev/null || true
      break
    fi
    sleep 0.1
    waited=$((waited + 1))
  done
  for pid in "$@"; do  # reap what this script started, so the next phase's wait -n cannot see it
    wait "$pid" 2>/dev/null || true
  done
}
rcp_ndcg_cleanup() {
  trap '' TERM INT
  local -a pids=()
  if [[ -n "${RCP_NDCG_COORDINATOR_PID:-}" ]]; then pids+=("$RCP_NDCG_COORDINATOR_PID"); fi
  if [[ -n "${RCP_NDCG_ENGINE_PID:-}" ]]; then pids+=("${RCP_NDCG_ENGINE_PID}"); fi
  if ((${#pids[@]})); then rcp_ndcg_stop "${pids[@]}"; fi
}
trap rcp_ndcg_cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
srun --overlap --nodes=1 --ntasks-per-node=1 --kill-on-bad-exit=1 --wait=10 --gres=gpu:1 bash -c "$ENGINE_JUDGE" &
RCP_NDCG_ENGINE_PID=$!
rcp_ndcg_any_ready() {  # PORT PATH HOST...
  local port="$1" path="$2"; shift 2
  for host in "$@"; do
    python3 -c 'import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=5)' "http://$host:$port$path" >/dev/null 2>&1 && return 0
  done
  return 1
}
rcp_ndcg_wait_ready() {  # PID_VAR TIMEOUT PORT PATH HOST...
  local pid_var="$1" timeout="$2" port="$3" path="$4"; shift 4
  local deadline=$((SECONDS + timeout)) status
  echo "rcp-ndcg: waiting up to $timeout s for an engine replica to answer $path" >&2
  until rcp_ndcg_any_ready "$port" "$path" "$@"; do
    if [[ -n "${!pid_var:-}" ]] && ! kill -0 "${!pid_var}" 2>/dev/null; then
      status=0
      wait "${!pid_var}" || status=$?
      printf -v "$pid_var" ''
      echo "rcp-ndcg: the engine exited with status $status before it answered $path; its output is above" >&2
      exit 1
    fi
    if ((SECONDS >= deadline)); then
      echo "rcp-ndcg: no engine replica answered $path within $timeout s (serve.startup_timeout_s); stopping the job" >&2
      exit 1
    fi
    sleep 10
  done
}
rcp_ndcg_wait_ready RCP_NDCG_ENGINE_PID 3600 8120 /v1/models 127.0.0.1
export RCP_NDCG_ENGINES='{"judge": {"urls": ["http://127.0.0.1:8120/v1"], "wait_on_outage_s": 300}}'
bash -c "$WORKER_3" &
RCP_NDCG_COORDINATOR_PID=$!
status=0
wait -n || status=$?
if kill -0 "$RCP_NDCG_COORDINATOR_PID" 2>/dev/null; then
  echo "rcp-ndcg: the judge engine exited with status $status while the run was going; stopping the run (submit it again with rcp-ndcg run resume --run <run dir> --runner <this runner>)" >&2
  exit 1
fi
failed=0
for pid in ${RCP_NDCG_ENGINE_PID}; do
  if ! kill -0 "$pid" 2>/dev/null; then
    es=0
    wait "$pid" 2>/dev/null || es=$?
    if [ "$es" -ne 0 ] && [ "$es" -ne 127 ]; then failed=$es; fi
  fi
done
if [ "$failed" -ne 0 ]; then
  echo "rcp-ndcg: the judge engine exited with status $failed while the run was going; stopping the run (submit it again with rcp-ndcg run resume --run <run dir> --runner <this runner>)" >&2
  exit 1
fi
if [ "$status" -ne 0 ]; then
  exit "$status"
fi
rcp_ndcg_stop "$RCP_NDCG_ENGINE_PID"
read -r -d '' WORKER_4 <<'RCP_NDCG_WORKER_4' || true
#!/usr/bin/env bash
set -euo pipefail
cd /e2e/runs/rcp-text-four-phases
export RCP_NDCG_ENGINES='{}'
export PYTHONPATH=/e2e/out/probe-site
export RCP_E2E_PROBE_JSONL=/e2e/out/client-probe.jsonl
exec uvx --from 'rcp-ndcg[calibrate,hf,s3,azure]==0.0.1' --constraints /stage/requirements-constraints.txt --find-links /stage/wheelhouse --no-index rcp-ndcg run resume --run /e2e/runs/rcp-text-four-phases --only calibrate --only evaluate
RCP_NDCG_WORKER_4
bash -c "$WORKER_4"
