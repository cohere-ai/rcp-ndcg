# k8s-run: the Kubernetes runner on the real cluster

The release checklist's real-infrastructure item, Kubernetes only
(`handover/RELEASE-CHECKLIST.md:74`: "one SLURM and one Kubernetes run of the one-container and StatefulSet
job shapes"). **SLURM is explicitly out of scope and was not run** (see the end of this report).

Worktree `/root/repos/rcp-ndcg-lanes/wt-k8s-run`, branch `lane/k8s-run`, base commit `310e1002` (the rfc-0001
tip, the commit the running RC wave `rc0-202610140620` was staged from); the lane's HEAD is the commit that
contains this report (`git log -1` in the worktree). Submitting host: the lane box, env built with
`uv sync --locked --extra dev` (see `.lane/env-build.log`).

## Verdicts

| shape | what the runner documents | verdict |
|---|---|---|
| (a) **one-container** | a single-replica engine, whose phase runs in the coordinator's own pod | **RAN** — Job `Complete 1/1` in 96 s; engine answered `/v1/models`; two real `POST /v1/embeddings` requests; `retrieve` completed; outputs in the run's mirror |
| (b) **StatefulSet** | several replicas of one engine role (run-scoped StatefulSet + headless Service, owned by the Job) | **BLOCKED** — `statefulsets.apps` `create` is forbidden for the submitting identity in **every** namespace (0 of 179 allow it). The runner applied the Job, the StatefulSet apply was refused, the runner deleted the Job and recorded the run failed. Nothing ran |

## The cluster and the staging

- Context `gke-us-central1-prod` (the current `kubectl` context; the job CLI config `jobs/config.yaml` names
  the queue `emb-queue` and priority `dev-high`), namespace `default`, user `fabian.schmidt@cohere.com`.
- GPU placement (the waves' own values, read off the running wave job): `nodeSelector
  {cohere.com/mode: gpu, nvidia.com/gpu.product: NVIDIA-B200}` and the toleration
  `nvidia.com/gpu=present:NoSchedule`. Both shapes asked **1 GPU per engine replica**.
- Service account `svc-modeling` (the GKE Workload Identity binding that reaches GCS; the run's dataset and
  mirror are `gs://`).
- **The release install.** `0.0.1` is not on PyPI and the GitHub release does not exist yet, so the coordinator
  cannot use its default install (`uvx --from rcp-ndcg[calibrate,hf,s3,azure]==0.0.1` against PyPI + the
  release constraints URL). The product's documented Kubernetes answer is a **wheelhouse** (`runner.options.
  wheelhouse`, `constraints`); the staged RC bucket is private, so the wheelhouse was served from inside the
  cluster by a Job + Service we own (`handover/reports/k8s-run/wheels.yaml`, name prefix
  `rcp-ndcg-k8srun-`): the pod downloads the tip's staged wheelhouse
  (`gs://cohere-dev-central-2/fabian/rcp-ndcg-rfc/rc0-202610140620/wheelhouse/`, 5.8 GB, plus that stage's
  `requirements-constraints.txt`) and one extra `uv` wheel from our own prefix, and serves the directory over
  HTTP at `http://rcp-ndcg-k8srun-wheels:8080`. Both runs used
  `wheelhouse: http://rcp-ndcg-k8srun-wheels:8080` and
  `constraints: http://rcp-ndcg-k8srun-wheels:8080/requirements-constraints.txt`.
- **The dataset** is a tiny corpus-layout JSONL dataset written for this lane (3 queries, 8 documents, 15
  judged pairs; `handover/reports/k8s-run/dataset/`), staged to
  `gs://cohere-dev-central-2/fabian/rcp-ndcg-k8s-run/20261014-k8srun/dataset/` and read by the pod over `gs://`
  (a Kubernetes job sees no local files, and the config refuses a local input). The mirror is
  `gs://cohere-dev-central-2/fabian/rcp-ndcg-k8s-run/20261014-k8srun/runs/<shape>`.

### The two runs' configs

Both configs are committed verbatim: `handover/reports/k8s-run/k8s-one-container.yaml` and
`k8s-statefulset.yaml`. Their `runner:` block is the same (context, namespace, service account, GPU node
selector, toleration, wheelhouse/constraints, `resources {cpus: 8, memory_gb: 32, time_limit_s: 3600}`,
`backoff_limit: 0`, `ttl_seconds_after_finished: 900`).

Shape (a) serves the **shipped recipe `qwen3-embedding-0.6b`** (1 GPU) and runs `steps: [retrieve]`;
shape (b) serves the shipped recipe **`gemma-4-26b-a4b-nvfp4`** (the smallest judge recipe by bytes, 1 GPU per
replica) with `replicas: 2` and runs `steps: [tournament]`.

Two deviations from the plain recipe spelling, both forced and both recorded below:

1. The **client blocks are inlined** (the exact block `rcp_ndcg.inference.recipes.recipe_client_data` returns for
   each recipe) instead of `recipe: <id>`, because the coordinator's uvx environment cannot resolve a shipped
   recipe id (finding F2, with its own in-cluster reproduction).
2. The **engine command installs the serving wheel and runs the shipped recipe id**:
   `python3 -m pip install ... uv && python3 -m pip install ... rcp-ndcg-vllm==0.0.1 && exec rcp-ndcg-vllm
   serve <recipe-id> --port 8000`. This is the documented inline form (`docs/how-to/serve-a-model.md` §1), and
   it is the only place the recipe is resolved — the engine image is the recipe's own
   `vllm/vllm-openai:v0.31.0`.

## Shape (a): the one-container shape — RAN

### Command and result

```console
$ cd /root/repos/rcp-ndcg-lanes/wt-k8s-run
$ uv run --no-sync rcp-ndcg run start .lane/k8s-run/k8s-one-container.yaml --runner kubernetes
Run 20261014-072801-k8s-one-container-49e823 submitted (submitted)
  directory  runs/20261014-072801-k8s-one-container-49e823
  requests   0 judge requests
    retrieve     pending
  job rcp-20261014-072801-k8s-one-container-49e823 (kubernetes): default/rcp-20261014-072801-k8s-one-container-49e823 pending
  mirror     gs://cohere-dev-central-2/fabian/rcp-ndcg-k8s-run/20261014-k8srun/runs/one-container (last upload ...)

$ uv run --no-sync rcp-ndcg run status --run runs/20261014-072801-k8s-one-container-49e823
Run 20261014-072801-k8s-one-container-49e823 completed (ran)
  directory  runs/20261014-072801-k8s-one-container-49e823
  requests   0 judge requests
    retrieve     completed      1.4s
  job rcp-20261014-072801-k8s-one-container-49e823 (kubernetes): default/rcp-20261014-072801-k8s-one-container-49e823 completed
  mirror     gs://.../runs/one-container (last upload ...)
  note       read from the mirror ..., which is ahead of this directory (restore it with run resume --mirror).
```

`kubectl get job -n default`:

```console
NAME                                           STATUS     COMPLETIONS   DURATION   AGE
rcp-20261014-072801-k8s-one-container-49e823   Complete   1/1           96s        6m
```

The pod ran on a B200 node (`gke-gke-us-central1-prod-b200-9-59689f0d-0qs9`), 1 GPU.

### The object the cluster created (`one-container-job-final.yaml`)

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: rcp-20261014-072801-k8s-one-container-49e823
  namespace: default
  labels: {app.kubernetes.io/name: rcp-ndcg, rcp-ndcg/job: rcp-20261014-072801-k8s-one-container-49e823}
spec:
  backoffLimit: 0
  activeDeadlineSeconds: 3600
  ttlSecondsAfterFinished: 900
  template:
    spec:
      restartPolicy: Never
      containers:                      # one phase: the last phase is the main container, so no init container
      - name: phase-1
        image: vllm/vllm-openai:v0.31.0   # the engine's image (engine + coordinator in one container)
        resources: {limits: {nvidia.com/gpu: 1}}
        volumeMounts: [scratch:/scratch, dshm:/dev/shm]
      volumes: [scratch (emptyDir), dshm (emptyDir, Memory, 16Gi)]
      securityContext: {runAsNonRoot: false, seccompProfile: {type: RuntimeDefault}}
      serviceAccountName: svc-modeling
      automountServiceAccountToken: false
      nodeSelector: {cohere.com/mode: gpu, nvidia.com/gpu.product: NVIDIA-B200}
      tolerations: [{key: nvidia.com/gpu, operator: Equal, value: present, effect: NoSchedule}]
status:
  conditions: [{type: SuccessCriteriaMet, ...}, {type: Complete, status: "True", reason: CompletionsReached}]
  succeeded: 1
  completionTime: "2026-10-14T07:29:42Z"
```

No StatefulSet and no Service belong to this job (`kubectl get statefulset,svc -n default -l
rcp-ndcg/job=rcp-20261014-072801-k8s-one-container-49e823` → `No resources found`); the one-container shape
really is one object.

The container's rendered script (in the Job's `command`) is the supervision script: it starts the engine in the
background (`bash -c "$ENGINE_ENCODER"` with `CUDA_VISIBLE_DEVICES=0`), waits for `/v1/models`, exports
`RCP_NDCG_ENGINES='{"encoder": {"urls": ["http://127.0.0.1:8000/v1"], "wait_on_outage_s": 300}}'`, runs the
coordinator through `uvx` against the wheelhouse, and stops the engine after it exits.

### The engine became ready and served real requests (`one-container-key-lines.txt`)

```console
rcp-ndcg: waiting up to 1800 s for an engine replica to answer /v1/models
(APIServer pid=8) INFO ... [launcher.py:69] Route: /v1/models, Methods: GET
(APIServer pid=8) INFO ... [launcher.py:69] Route: /v1/embeddings, Methods: POST
INFO:     Application startup complete.
INFO:     127.0.0.1:48000 - "GET /v1/models HTTP/1.1" 200 OK
...
 Downloaded torch
Installed 88 packages in 1.05s
2026-10-14 07:29:36,752 | INFO | rcp_ndcg.runs.pipeline | [run] retrieve: starting
INFO:     127.0.0.1:49436 - "POST /v1/embeddings HTTP/1.1" 200 OK
INFO:     127.0.0.1:49450 - "POST /v1/embeddings HTTP/1.1" 200 OK
```

The engine was the recipe's own serve command: the container installed `uv` and `rcp-ndcg-vllm==0.0.1` from the
wheel server (with `--trusted-host`, finding F4) and ran `rcp-ndcg-vllm serve qwen3-embedding-0.6b --port 8000`
(`vllm serve Qwen/Qwen3-Embedding-0.6B@97b0c614... --runner pooling ...`).

### The run's outputs (from the mirror)

`run status` on the submitting host reads the mirror, which is ahead of the local directory (the pod wrote it):

```console
$ gcloud storage ls -r gs://.../runs/one-container/
.../candidates.parquet   .../manifest.json   .../run.yaml   .../logs/run.log
```

The mirror's `manifest.json` (`one-container-mirror-manifest.json`): `status: completed`, step `retrieve`
`completed`, output `candidates.parquet` sha256 `2cacddfd…`. The file is the 3 × 8 retrieval view (24 rows,
columns `system, dataset, query_id, doc_id, score`) — produced through the served engine, so the run really did
reach the engine and record what came back.

## Shape (b): the StatefulSet shape — BLOCKED (RBAC)

### Command and exact blocker

```console
$ uv run --no-sync rcp-ndcg run start .lane/k8s-run/k8s-statefulset.yaml --runner kubernetes
error [PROVIDER]: `kubectl --context gke-us-central1-prod apply -f -` exited 1: Error from server (Forbidden):
error when creating "STDIN": statefulsets.apps is forbidden: User "fabian.schmidt@cohere.com" cannot create
resource "statefulsets" in API group "apps" in the namespace "default": requires one of
["container.statefulSets.create"] permission(s) in Cloud IAM or a Kubernetes RBAC role with verb "create" for
resource "statefulsets".
```

The runner's own behaviour around it (from `rcp_ndcg.runners.kubernetes.KubernetesRunner.submit`): it applies
the Job first, then the objects the Job owns; when that fails it **deletes the Job it just applied** so nothing
is left running. The run's record (`runs/20261014-073121-k8s-statefulset-9cc6f8/logs/jobs.json`) carries the
error above with a null handle, and `run status` shows:

```console
Run 20261014-073121-k8s-statefulset-9cc6f8 failed (ran)
    tournament   pending              0/? windows
  job rcp-20261014-073121-k8s-statefulset-9cc6f8 (kubernetes):  unknown
  note       the submission failed: RunnerError: `kubectl ... apply -f -` exited 1: ... statefulsets.apps is
             forbidden: User "fabian.schmidt@cohere.com" cannot create resource "statefulsets" ...
```

No `rcp-20261014-073121-k8s-statefulset-9cc6f8` Job and no StatefulSet remain in the namespace.

The config committed here carries the same engine-command fix as shape (a) (finding F4); the attempt above used
the pre-fix spelling, which never mattered: the StatefulSet apply is refused before any engine pod could be
created.

### The RBAC fact, exactly (`rbac.txt`)

```console
$ kubectl auth can-i create statefulsets -n default
no - requires one of ["container.statefulSets.create"] permission(s) in Cloud IAM or a Kubernetes RBAC role
with verb "create" for resource "statefulsets".

$ (loop over all 179 namespaces)
namespaces allowing statefulset create: 0

$ kubectl auth can-i create services -n default      -> yes
$ kubectl auth can-i create jobs.batch -n default    -> yes
```

`kubectl auth can-i --list -n default` also shows the platform's blessed multi-pod path is not StatefulSets:
`jobsets.jobset.x-k8s.io` is create/update/delete, `deployments.apps` and `statefulsets.apps` are read-only or
absent. The product's `KubernetesRunner` renders StatefulSets, so on this cluster the multi-replica shape cannot
be submitted by this identity at all. This is a platform-policy blocker, not a product bug: the shape was
exercised as far as the cluster allows (Job applied → engine objects refused → Job cleaned up).

### What the runner rendered for the shape (`statefulset-render.yaml`, from `--dry-run`)

```yaml
# Job (coordinator only: the judge is a StatefulSet, so this pod asks for no GPU)
kind: Job
spec: {backoffLimit: 0, ttlSecondsAfterFinished: 900, template: {spec: {restartPolicy: Never,
  containers: [{name: phase-1, image: ghcr.io/astral-sh/uv:python3.12-trixie-slim,
                resources: {requests: {cpu: 8, memory: 32Gi}, limits: {cpu: 8, memory: 32Gi}}}]}}}
---
kind: StatefulSet
metadata: {name: rcp-2026...-<hash>, namespace: default,
           labels: {app.kubernetes.io/name: rcp-ndcg-engine-judge, rcp-ndcg/job: rcp-2026...}}
spec:
  serviceName: rcp-2026...-<hash>          # the headless Service below
  replicas: 2                              # serve.judge.replicas
  podManagementPolicy: Parallel
  template:
    spec:
      containers:
      - name: engine
        image: vllm/vllm-openai:v0.31.0
        command: [bash, -lc, "pip install ... rcp-ndcg-vllm==0.0.1 && exec rcp-ndcg-vllm serve
                  gemma-4-26b-a4b-nvfp4 --port 8000"]
        resources: {limits: {nvidia.com/gpu: 1}}     # per replica
        startupProbe:  {httpGet: {path: /v1/models, port: 8000}, periodSeconds: 10, failureThreshold: 360}
        readinessProbe: {httpGet: {path: /v1/models, port: 8000}, periodSeconds: 10}
      volumes: [dshm (Memory, 16Gi), scratch (emptyDir), engine-tmp (emptyDir)]
      nodeSelector: {cohere.com/mode: gpu, nvidia.com/gpu.product: NVIDIA-B200}
      tolerations: [{key: nvidia.com/gpu, operator: Equal, value: present, effect: NoSchedule}]
      serviceAccountName: svc-modeling
---
kind: Service
spec: {clusterIP: None, publishNotReadyAddresses: true,
       selector: {app.kubernetes.io/name: rcp-ndcg-engine-judge, rcp-ndcg/job: rcp-2026...},
       ports: [{name: http, port: 8000, targetPort: 8000}]}
```

The coordinator phase's rendered script waits for the replicas' stable names
(`<sts>-0.<svc>.<ns>.svc:8000`, `<sts>-1.<svc>.<ns>.svc:8000`) with `RCP_NDCG_ENGINE_PID_REMOTE` (no local
engine to watch), exports them as `RCP_NDCG_ENGINES`, and runs the `tournament` step. The coordinator Job's
pod is `restartPolicy: Never` (as in shape (a)); the StatefulSet's engine pod sets no `restartPolicy`, so it
is the StatefulSet default `Always` — a crashed replica is restarted by the controller while the run-scoped
engine lives.

To show the blocker is exactly the StatefulSet and not the rest of the shape, the rendered Service was applied
with a server dry run:

```console
$ kubectl apply --dry-run=server -f statefulset-service-only.yaml
service/rcp-20261014-073159-k8s-statefulset-ca56ec-64e053df created (server dry run)
```

## Findings

**F1 — the multi-replica (StatefulSet) shape cannot run on this cluster.** `statefulsets.apps` `create` is
denied for the submitting identity in all 179 namespaces; the product's runner is the only thing that would
create one. Verdict for the checklist item: the one-container shape is verified on real infrastructure; the
StatefulSet shape needs either the RBAC grant (`container.statefulSets.create` for the user/namespace) or a
platform-supported runner (JobSet) before it can be verified. Evidence: `statefulset-start.txt`,
`statefulset-run-status.txt`, `rbac.txt`, `statefulset-render.yaml`.

**F2 — `recipe: <shipped-id>` cannot resolve in a Kubernetes coordinator.** The coordinator installs
`rcp-ndcg[calibrate,hf,s3,azure]==0.0.1` (`COORDINATOR_EXTRAS`,
`rcp-ndcg/src/rcp_ndcg/runners/script.py:51`); the `rcp-ndcg` wheel declares no dependency on `rcp-ndcg-vllm`
and no `vllm` extra, and the uvx environment is isolated from anything in the image, so the serving package is
never present. Reproduced in the exact coordinator environment (the same uvx spec against the same wheelhouse),
`recipecheck.log`:

```console
$ uvx --from 'rcp-ndcg[calibrate,hf,s3,azure]==0.0.1' --constraints http://rcp-ndcg-k8srun-wheels:8080/...
       --find-links http://rcp-ndcg-k8srun-wheels:8080 --no-index python -c '... recipe_client_data(...)'
Installed 88 packages in 324ms
PRODUCT ERROR: ConfigError - recipe: ... needs the serving recipes of rcp-ndcg-vllm (No module named 'rcp_ndcg_vllm')
```

The two runs therefore carry the recipes' **client blocks inlined** and serve the recipes by id from the engine
side. A release that wants `recipe:` in a Kubernetes run config needs one of: the serving package added to
`COORDINATOR_EXTRAS`/the wheelhouse install spec, or a coordinator image whose uvx environment carries it.

**F3 — the plain Job is not Kueue-managed, so it does not use the waves' queue/priority.** The waves are
JobSets carrying `kueue.x-k8s.io/queue-name: emb-queue` and `kueue.x-k8s.io/priority-class:
dev-high-training-priority` (a **WorkloadPriorityClass**, value 925; `kubectl get priorityclass
dev-high-training-priority` → NotFound). The product's Job carries only its own two labels, Kueue's
`manageJobsWithoutQueueName` is off, and no Workload was created for our runs (`kubectl get workloads -n
default` has no `rcp-*` row): the runs scheduled as ordinary Jobs and consumed no `emb-queue` quota.
`runner.options.priority_class` renders a pod-level `priorityClassName` and cannot carry the Kueue
WorkloadPriorityClass; there is no option for the queue label. For this lane that was acceptable (1 GPU, one
pod, the GPU nodes had room), but a release run that must respect the queue would need the label (or the
platform's JobSet path). Evidence: the Job's labels in `one-container-job-final.yaml`, `kubectl get
workloads`.

**F4 — a plain-HTTP wheelhouse needs `--trusted-host` for pip (uv is fine).** The first attempt failed in the
engine with pip's "The repository located at rcp-ndcg-k8srun-wheels is not a trusted or secure host and is
being ignored"; uv accepts an `http://` `--find-links`/`--constraints` (verified locally). The committed configs
install `uv` and the plugin with `--trusted-host rcp-ndcg-k8srun-wheels` and set `PIP_TRUSTED_HOST` for the
coordinator's own uv bootstrap. Operational note for anyone serving a wheelhouse over plain HTTP.

## Residual risks and notes

- The runs used the tip's staged RC wheelhouse (`rc0-202610140620`, githead `310e1002` — the same commit as the
  submitting host), not a published release; a released `0.0.1` would install from PyPI + the GitHub constraints
  URL, which were both unavailable here (PyPI 404, GitHub release 404).
- The wheel server (a Job + Service we own, name prefix `rcp-ndcg-k8srun-`) is lane infrastructure, not part of
  the product: the product only ever sees the `wheelhouse` URL. It was deleted after the runs.
- Shape (a) is one data point: a single-replica encoder, a tiny dataset, `retrieve` only. It does not exercise
  the judge engine, the multi-phase shape, `resume`, or an outage.
- The mirror is ahead of the submitting host's run directory (the pod uploaded the outputs). The run directory
  can be restored with `rcp-ndcg run resume --run <dir> --mirror <uri>`; it was not, to leave the as-submitted
  record intact.
- The product's `note` on a failed submission (F1) is good; the runner's cleanup (deleting the Job it applied)
  was verified by absence.

## Cleanup

- The shape (a) Job had `ttl_seconds_after_finished: 900`; it was deleted explicitly after recording (the
  failed shape (b) Job was deleted by the runner itself).
- The wheel-server Job and Service, and the three probe Jobs (`rcp-ndcg-k8srun-probe`, `-gpuprobe`,
  `-gcsprobe`, `-recipecheck`) were deleted after recording; their TTLs (300–900 s) would have removed them
  anyway. Nothing with the `rcp-ndcg-k8srun-` prefix is left.
- The two waves named in the task were not touched by this lane: every object this lane created, queried for
  deletion or deleted was named `rcp-ndcg-k8srun-*` or `rcp-2026*` (the runs' own names), and no command ever
  named a `fabianschmidt-*` object. At the end of the lane the judges wave
  (`fabianschmidt-rcp-judges-j1-202610140706-difi-worker-0`) was still running, and the e2e wave
  (`fabianschmidt-cpu-75-rcp-e2e-202610140703-rxwg`) had finished and been cleaned up by its own
  `ttlSecondsAfterFinished: 60` (its JobSet is gone; see `cleanup.txt`).
- The staged GCS prefix `gs://cohere-dev-central-2/fabian/rcp-ndcg-k8s-run/20261014-k8srun/` (dataset,
  wheels-extra, mirror) is left in place; it is small and lane-scoped.

## SLURM — not run

Out of scope for this lane by instruction. No SLURM runner command was issued and no `sbatch` was submitted.

## Evidence index (`handover/reports/k8s-run/`)

| file | what it is |
|---|---|
| `k8s-one-container.yaml`, `k8s-statefulset.yaml` | the two run configs, verbatim |
| `dataset/` | the tiny JSONL dataset (queries/corpus/qrels) staged to `gs://` |
| `wheels.yaml` | the lane's wheel-server Job + Service (infrastructure, not the product) |
| `one-container-start.txt` | `run start` output (submitted, job handle, mirror) |
| `one-container-run-status.txt` | `run status` after completion |
| `one-container-job-final.yaml` | the created Job object, verbatim (shapes, status) |
| `one-container-pod-final.yaml` | the pod object (node, container statuses) |
| `one-container-pod.log` | the pod's full log (engine + coordinator) |
| `one-container-key-lines.txt` | readiness / install / request lines extracted from it |
| `one-container-get-job.txt`, `one-container-get-sts-svc.txt` | `kubectl get` after completion (no sts/svc) |
| `one-container-mirror-manifest.json` | the run's manifest from the mirror (step + output sha256) |
| `statefulset-start.txt` | the exact blocked submission command + output |
| `statefulset-run-status.txt` | the run's failed status with the error note |
| `statefulset-render.yaml` | the Job + StatefulSet + Service the runner rendered (`--dry-run`) |
| `statefulset-service-only.yaml` | the rendered Service, dry-run applied (accepted) |
| `rbac.txt` | `kubectl auth can-i` evidence for the blocker |
| `recipecheck.log` | the recipe-pointer gap reproduced in the coordinator's own environment |
