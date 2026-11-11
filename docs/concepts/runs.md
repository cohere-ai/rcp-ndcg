# Runs, engines and runners

This page is the orchestration half. It covers run configs and steps, the job runners, the engines that a job
starts, and durability. [Judges, the judgement store and estimates](judges.md) covers the judge itself. A run
config's `serve:` section names one engine per role: the judge, the retrieval encoder and the reranker. It
starts your image with your command, verbatim, in the phases of the run that use it. On SLURM without a
container runtime, it starts your command on the node.

## Runs and job runners

A run config names the dataset, the candidates, the judge and the steps. The steps are `retrieve`, `fuse`,
`rerank`, `tournament`, `rubric`, `calibrate` and `evaluate`. `rcp run start my_run.yaml` runs them
into `runs/<run_id>/`. The `manifest.json` there records every step's identity, inputs, outputs and usage,
which is judge calls and tokens. `rcp run resume --run runs/<run_id>` reuses every step whose identity is
unchanged. A judging step's identity includes the prompt's text hash, the tokenizer's SHA-256 and the pool
selection: `candidates.source`, `.system` and `.rankings`. A resume that flips the system therefore
re-judges that system's pools. The steps that read the dataset carry its resolved commit, the title join mode
and the instruction policy. Those are content fields of the role configs. They also carry the text-formatting
rule's version. A resume after the formatting changed therefore re-runs them, instead of reusing the old
strings. A local dataset has no commit, so it carries the digest of the files that its reader reads. That is a
`jsonl:` file's bytes, or a `beir:` or `images:` directory's sorted listing. That is the same digest that the
judgement store's dataset identity carries, so editing the file in place re-runs them. A judging step whose
store holds the old identity refuses, and it names the change and its way out. `calibrate` keys on the
judgement stores that it reads, whose content already covers the formatting. `rcp run status` shows the
state of each step. Estimate a run with `--estimate` before you start it.

A run's `fuse` step performs the fusion itself. The config is:
`candidates: {from: fuse, fuse: {systems: {s1: <rankings file>, s2: <file>#<system>}, rrf_k: 60}}`

The step fuses the named systems' rankings with reciprocal rank fusion. There must be at least two systems.
Each one is a rankings file, or one system of a file that holds several (`<file>#<system>`). The arithmetic is
`rcp.retrieval.fuse`, the same that `rcp retrieval fuse` calls. The run's candidate pools are the
fused order. The step's record in `manifest.json` names the systems, the constant and the depth. A reader
therefore sees that the fusion happened. A declared fusion that cannot be performed is refused with a typed
error, never a silent ranking by one system. It fails in four cases. A system's rankings file is not there. A
system was not retrieved into its file. A file of several systems has no `#<system>` selector. The fusion has
fewer than two systems.

`step_budget_s:` (seconds, default `null`) gives every step a wall-clock budget. The transport checks it
before each request and after every park. The judging pass checks it before each phase's windows. A step over
budget stops with `StepBudgetExceededError` (exit code 9). The judgement store keeps every record that it
wrote, and a resume continues from there. It is a runtime knob, so changing it never re-keys a step. `null`
leaves the steps unbudgeted.

`run resume --set KEY=VALUE` (a YAML literal) changes the run's config. The run keeps the change only if the
resume succeeds. A resume that fails or is refused leaves `run.yaml`, the recorded config and the run's
status as they were. A step that it re-ran before failing is redone by the next resume. A judging step can
start a stage under the changed config. Its store then holds judgements of that config, so the change stays.
The run is `failed` until a resume finishes it. `run resume --only STEP` runs just those steps now, and it
never changes the run's recorded `steps`. `--only rerank` regenerates a missing first stage with the
configured `retrieve` step. In a fused run it uses the `fuse` step. It also applies to a restore that left no
`work/`, which the mirror skips. The documented path is not wedged by it. A later resume records the retrieve
step that it ran on the way.

The steps run on this machine by default. `rcp run start my_run.yaml --runner slurm` (or `kubernetes`)
submits the run as one job instead. That job runs `rcp run resume` on the scheduler. `run resume` always
runs the run in the process that calls it. `run status`, `run logs` and `run cancel` follow it, from any
process. `--detach` makes `run start` return at once, and the run is then followed with `run status`. Without
`--runner`, the run becomes a background job of the local runner. The MCP tool `run_start` always detaches:
it returns the run directory, and the agent polls `run_status`.
`rcp run start my_run.yaml --runner slurm --dry-run` prints the step plan and the sbatch script that
would be submitted. For `kubernetes` it prints the manifests. It writes nothing.

The plan (``--dry-run``, ``Pipeline.plan()``) is computed from the run directory as it stands. A step that
would run and rewrite an input that another step reads is not simulated. The plan can therefore differ from
what ``run()`` does. One example is a `from: dataset` run whose pools changed. It prints "would skip" for a
judging step that the run's own preamble then makes stale. Another is a restore without `work/`. It prints
"would run" for a rerank step that the regenerated first stage makes current. The plan is a forecast, never a
promise. ``run()`` decides from the state that it actually produces.

The runner's options go in the run config. Each runner validates them with its own model: `LocalOptions`,
`SlurmOptions` and `KubernetesOptions` in `rcp.runners`. It validates them when the config is read and
before anything is written. A typo therefore leaves no run directory behind. Three of them describe the job
rather than the runner: `resources` (`gpus`, `cpus`, `memory_gb`, `time_limit_s`), `image` and `env`:

```yaml
runner:
  name: slurm
  options:
    partition: gpu
    account: my-project
    log_dir: logs/slurm
    setup: ["source .venv/bin/activate"]
    resources: {cpus: 8, memory_gb: 64, time_limit_s: 86400}
    env: {HF_HOME: /shared/hf}
```

The three job fields apply to the runner that the config names. The run can be handed to another runner:
`run start --runner kubernetes`, or `run resume --runner slurm`. With `resources`, `image` or `env` set,
that handover is refused with a message naming them, never silently dropped. `env` may not name
`RCP_NDCG_ENGINES`, because the phase overlay owns that variable; a job's own value would be silently
overridden. A run without `serve:` hands its engines over with `RCP_NDCG_ENGINES` in the process environment,
or with `run resume --engine <role>=<url>`.

- **SLURM.** Resources become `--gres=gpu:N`, `--cpus-per-task`, `--mem` and `--time`, and `sbatch_args` passes anything
  else through. With `serve:`, a phase holds its engine replicas and, on the phase's first node, the coordinator. A
  node's GPUs are the sum of what runs on it: the coordinator's own request, plus each engine's. The largest engine
  shares the coordinator's node. CPUs and memory add up. An engine whose `memory_gb` is unstated gets the node's whole
  memory (`--mem=0`), whatever the coordinator asks ([partitioning](#gpus-are-partitioned-per-node-not-shared)). The job
  runs in the node's environment (`setup`) or in a container (`container_runtime: apptainer` or `pyxis`, with
  `container_mounts`). A container runs the stock coordinator image below unless `image` names another. With
  `container_runtime: none`, an `image` or `container_mounts` is refused, because the node's environment and filesystem
  are already there. A phased job's coordinator then runs as a step of its own (`srun --overlap`). Its own `--gres`
  reservation and node pin hold there too. A coordinator that asks for no GPU sees no device. A job with no phases has
  no engines, so its coordinator runs in the batch shell and needs no reservation. Runs, stores and caches live on the
  cluster's shared filesystem. A node without internet access needs the weights and data staged beforehand: set
  `HF_HOME` to a shared cache and `HF_HUB_OFFLINE=1`. The online runs themselves fill the shared cache. A dataset run
  resolves its revision online once. It records the commit behind the branch (`refs/<ref>` in the cache). An offline run
  without `--revision` therefore reads from the same cache. A cache with no recorded ref needs `--revision <full sha>`.
  A download pinned to a commit stages such a cache. Until the revision is given, the run warns (`UNPINNED_REVISION`, on
  stderr and in the `--json` envelope's `warnings`). A file that the online run never fetched is an error naming the
  fix, not an empty table.
- **Kubernetes.** Each run is a `batch/v1` Job, applied with the `kubectl` on your `PATH`, and `context` if set.
  Resources become the containers' requests and limits (`nvidia.com/gpu`, `cpu`, `memory`) and the Job's
  `activeDeadlineSeconds`. A phase container asks for the sum of its engines' GPU requests plus the coordinator's. It
  also asks for the engines' CPUs and memory plus the coordinator's. A CPU or memory amount that the engine leaves
  unstated is left unlimited, whatever the coordinator asks. The coordinator's share therefore never caps the engine.
  `secrets` are exposed to every container as environment: an HF token, or the mirror's credentials. Every pod carries a
  `RuntimeDefault` seccomp profile, no privilege escalation, and no service-account token unless
  `automount_service_account_token: true`. A job that reaches the cluster API, or a store through the cluster's workload
  identity, sets it. An image with a non-root `USER` declares `run_as_non_root: true`, which the kubelet enforces. The
  stock coordinator and `vllm/vllm-openai` images run as root, so it is off by default. `tolerations`, `affinity` and
  `priority_class` are rendered on the Job's pod and every engine pod. `engine_node_selector` places the engine pods. A
  single-replica engine runs in the Job's pod, so its selector is merged into that pod's `nodeSelector`. A key that the
  two disagree on is refused. A credential never goes in `env`. A name that looks like one (`*_TOKEN`, `*_KEY`,
  `*SECRET*`, `*PASSWORD*`) is refused when the config is read. Export it in the submitting environment (local, SLURM)
  or name a `secrets` entry (Kubernetes). An environment value is never recorded in clear in `run.yaml`, the manifest or
  a mirror copy. The pod's disk is scratch, an `emptyDir` at `/scratch`. The model weights and the Hugging Face cache
  live on a declared `cache_volume`. That volume is a `PersistentVolumeClaim` mounted at `/cache`, shared by every
  replica and kept across restarts. Otherwise they live on that scratch volume, never the container's writable layer.
  Every engine process gets its own `TMPDIR` under `/scratch/tmp`. `/dev/shm` is a Memory `emptyDir` bounded by
  `dshm_size` (`runner.options.dshm_size`, `16Gi` by default). The engines' shared-memory working set fits, and a
  runaway engine fails its own pod instead of exhausting the node. Raise it for a phase that runs several engines in one
  container, whose requirements add up. A run-scoped engine (several replicas) lives in its own StatefulSet for the
  whole run. A finished Job that owns one is deleted, with its engines, an hour after it finishes, unless
  `ttl_seconds_after_finished` says otherwise. `run status` reports a pod that the scheduler cannot place as `pending`,
  with the scheduler's own reason in the note. The Job's pods and the engine pods alike are covered. A pod list that
  cannot be read reports `unknown` with a note, never `running`. A multi-phase job whose init container is still
  initializing reads `pending`, with the container's own `PodInitializing` reason in the note. The phase's engine is
  still starting, not stalled. The pod needs a `mirror:` ([durability](#durability-local-runs-and-a-mirror)), since it
  does not see the submitting host's files either. Every input must be a URI that it can read: `hf://`, `s3://`, `gs://`
  or `https://`. A run that names a local dataset, rankings file, evaluation system, judge config file or prompt file is
  refused. The error names them before anything is written. A judge or prompt goes by its shipped name, or inline in the
  run config. The packaged `tiny` example reads local files, so it is not submittable to Kubernetes as it is.

### The coordinator installs itself

The coordinator is the process that runs `rcp run resume`. It needs no GPU and no image of its own by
default. The job's `resources.gpus` is 0 unless a config declares some, and the runners then reserve exactly
what the config declares. On SLURM that is a `--gres` on the phase's first node. On Kubernetes it is a
container request ([partitioned](#gpus-are-partitioned-per-node-not-shared) ahead of the engines' slices). In
a container it runs a stock image with uv and Python 3.12 (`ghcr.io/astral-sh/uv:python3.12-trixie-slim`
unless the runner's `image` names another). That image installs the same release as the submitting host when
the job starts:

```bash
uvx --from 'rcp[calibrate,hf,s3,azure]==<version>' \
  --constraints https://github.com/cohere-ai/rcp-ndcg/releases/download/v<version>/requirements-constraints.txt \
  --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \
  rcp run resume --run <run_dir> --mirror <uri>
```

The constraints file, attached to each GitHub release, is the release's `uv.lock` exported. Every dependency
is therefore the version that the release was tested with. `unsafe-best-match` lets uv consider the PyTorch
CPU index next to PyPI for every package. torch then resolves to its CPU build at the locked version. Without
it, uv stops at the first index that has a package, and it fails. On Kubernetes uv's cache lives on the
scratch volume. Without PyPI, install from the repository instead. Use
`--from 'rcp[calibrate,hf,s3,azure] @ git+https://github.com/cohere-ai/rcp-ndcg@v<version>'` in the
larger `ghcr.io/astral-sh/uv:python3.12-trixie` image, which has git.

**Before the release is on PyPI**, the coordinator installs from a **wheelhouse** instead of PyPI. One example
is a release candidate or an RC wave on a cluster. The same holds on a node without any network access. Set
the runner's `wheelhouse` and `constraints` options: `runner.options.wheelhouse` and
`runner.options.constraints`. The generic `--set runner.options.wheelhouse=...` overrides them from the
command line. On a node without any network access, name `constraints` too. `uvx` fetches the default release
URL at job start, even under `--no-index`. The wheelhouse is a directory of staged wheels, readable on the
node, or an `http(s)://` URL of one. uv's `--find-links` and `--constraints` read local paths (or `file://`
URLs) and http(s) URLs with a host, and no bucket scheme. A GCS-staged wheelhouse is exposed through its
`https://` URL, or mounted where the job runs. A `gs://` /`s3://` value is refused when the config is read,
not at job start. On Kubernetes a directory must reach the pod another way, because the Kubernetes runner
takes no volume mounts. Bake it into the coordinator's `image:`, or name the `https://` URL. On SLURM,
`container_mounts` mounts a shared one. The rendered `uvx` then takes everything from the wheelhouse and asks
no index. The constraints file that you name replaces the release's:

```yaml
runner:
  name: kubernetes
  options:
    wheelhouse: https://storage.googleapis.com/my-bucket/wheelhouse/<version>
    constraints: https://storage.googleapis.com/my-bucket/wheelhouse/<version>/requirements-constraints.txt
```

The local runner runs the coordinator in this host's environment and installs nothing, so a wheelhouse there
is refused. On SLURM the option applies with a container runtime. With `container_runtime: none` the node's
own environment provides the release, and a wheelhouse is refused.

`jobs/rc_build.sh` in `rcp-test` does all of this in one command: build, checks, wheelhouse, stage and
manifest. See [Release candidates and the GPU waves](../how-to/release-candidates.md). Built by hand, the same
commands are:

```bash
# the three published wheels, by name, as release.yml builds them (never --all-packages: the
# unpublished fourth workspace member rcp-test must not be swept into the release files; the
# pyproject.toml versions must match)
uv build --package=rcp-core --out-dir /shared/wheelhouse/<version>
uv build --package=rcp --out-dir /shared/wheelhouse/<version>
uv build --package=rcp-vllm --out-dir /shared/wheelhouse/<version>
# the unpublished harness wheel (the node's client environment imports rcp_test), in its own directory
uv build --package=rcp-test --out-dir /shared/wheelhouse/<version>/harness
# the locked dependencies, pinned exactly (the command the committed requirements-constraints.txt records)
uv export --frozen --no-hashes --no-emit-workspace --no-dev --extra calibrate --extra hf --extra s3 --extra azure \
  -o /shared/wheelhouse/<version>/requirements-constraints.txt
pip download -r /shared/wheelhouse/<version>/requirements-constraints.txt \
  -c /shared/wheelhouse/<version>/requirements-constraints.txt \
  'rcp-vllm[test]==<version>' 'rcp[hf]==<version>' 'rcp-test==<version>' \
  $(for lock in rcp-vllm/src/rcp_vllm/recipes/*/reference.lock; do
      grep -q '^# own-torch: true' "$lock" || printf -- '-r %s ' "$lock";
    done) \
  --dest /shared/wheelhouse/<version> --find-links /shared/wheelhouse/<version> --only-binary :all: \
  --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
# The own-torch families' locks pin a CUDA torch: download them from PyPI alone (PEP 440 orders a
# ``+cpu`` local version above the plain release, so the CPU index must not see them).
pip download \
  $(for lock in rcp-vllm/src/rcp_vllm/recipes/*/reference.lock; do
      grep -q '^# own-torch: true' "$lock" && printf -- '-r %s ' "$lock";
    done) \
  --dest /shared/wheelhouse/<version> --index-url https://pypi.org/simple
```

`uv export --frozen ...` writes the same file that the release attaches, with the command in its header.
`pip download` resolves it for the node's platform, and it pulls torch from the cpu index first. The reference
environments' pins come from the families' `reference.lock` files: one venv per family, built from its lock. A
family pin with no index wheel must be staged in an `EXTRA_DIRS` wheelhouse. One example is `flash-attn`, a
GitHub-release wheel. Stage the directory to the URL that the nodes read, and mount it for containers. The job
then installs exactly the release, whatever PyPI serves that night.

### Starting the engines with the run

Without `serve:`, the job runs the coordinator only. Each role config's `base_url` is used as it is: your own
engines, gateways or hosted APIs. With `serve:`, the run names one engine per role. The `slurm` and
`kubernetes` runners start each engine inside the run's own job. The engines and the coordinator are therefore
scheduled together and end together. `judge` serves the judge, `encoder` serves the retrieval config's
encoder, and `reranker` serves its reranker. The served model name is that config's `model`, and a role
config whose engine is served names no `base_url`. The job's URLs for it reach the step at runtime, never the
config. Setting both is refused rather than silently overridden. A judge that names a `base_url` must name the
job's engine's own loopback URL (`http://127.0.0.1:<serve.judge.port>/v1`). Any other value is refused, since
the job's engine would silently replace it at runtime.

```yaml
judge: recipe:gpt-oss-120b
candidates:
  from: retrieval
  retrieval: {kind: dense, encoder: {api: openai_embeddings, model: octen-embedding-8b,
                                      tokenizer: Octen/Octen-Embedding-8B, max_tokens: 8192}}
  rerank: {api: rerank, model: qwen3-reranker-8b, use_activation: true,
           tokenizer: Qwen/Qwen3-Reranker-8B, max_tokens: 8192, query_max_tokens: 4096}
steps: [retrieve, rerank, tournament, rubric, calibrate, evaluate]
serve:
  encoder:
    image: vllm/vllm-openai:v0.33.0                # your engine and your tag
    command: [vllm, serve, Octen/Octen-Embedding-8B, --runner, pooling, --served-model-name, octen-embedding-8b,
              --host, 0.0.0.0, --port, "8000"]
    env: {HF_HOME: /models}
    resources: {gpus: 1}                           # per replica
  judge:
    image: vllm/vllm-openai:v0.33.0
    command: [vllm, serve, openai/gpt-oss-120b, --served-model-name, gpt-oss-120b, --reasoning-parser, openai_gptoss,
              --max-model-len, "131072", --tensor-parallel-size, "4", --data-parallel-size, "2",
              --host, 0.0.0.0, --port, "8000"]
    resources: {gpus: 8}
    outage_timeout_s: 900                          # fail the run if every replica stops answering this long
runner:
  name: kubernetes
  options: {namespace: eval, secrets: [hf-token]}
mirror: s3://my-bucket/runs/nano-gpt-oss
step_budget_s: 3600                               # stop a step after an hour; null (the default) is unbudgeted
```

Every engine has the same fields, whatever role it serves: `image`, `command`, `env`, `resources`,
`replicas`, `port`, `readiness_path`, `startup_timeout_s` and `outage_timeout_s`. The `command` is yours,
verbatim, as an argv list or one string. It must serve the role config's `model` name on `port`, on all
interfaces when there are several replicas. `nodes_per_replica` must be 1. A replica that spans several nodes
is not implemented, and another value fails when the config is read. A served encoder or reranker runs one
replica, because this release's retrieval clients address one URL. The judge may run several. A hosted model
(Cohere, Voyage, Gemini) is not served by a job's engine at all, so drop the role. The `image` is required on
Kubernetes and with the SLURM runner's `container_runtime: apptainer` or `pyxis`. It must name an exact tag
or a digest. `:latest` and an untagged reference are refused, because the recipe declares the image that it
was validated against. With `container_runtime: none`, the SLURM default, the command runs on the node
itself, so the role's `image` is refused there. Set a container runtime to run the engine in its image.
Otherwise drop `image` to run the command on the node. The same refusal covers the coordinator's own `image`
and `container_mounts` with `container_runtime: none`, because the node provides both. A command can name a
parallelism flag: `--tensor-parallel-size` /`-tp`, `--data-parallel-size` /`-dp`, `--pipeline-parallel-size`
/`-pp` or `--prefill-context-parallel-size` /`-pcp`. Their product is the engine's world size. Such a command,
or one that names `--port`, must agree with `resources.gpus` and `port`. A disagreement is refused when the
config is read. The runner reserves the devices and probes the port that the config declares. The local
runner, and a run in this process, start no engine, and they refuse phases that would start one. Start the
engines yourself, and pass their URLs with `run resume --engine <role>=<url>[,<url>]`.

#### Phases

The run's steps run in phases, the *job phases*. They are distinct from the schedule phases of the window
schedules. The job's GPUs are therefore the maximum over phases, not the sum over engines. Consecutive steps
that call the same served engines share a phase. Steps that call no served engine form phases without one:
BM25 retrieval, hosted APIs, `calibrate` and `evaluate`. The run above becomes four phases. Each one starts
only its engines and waits for readiness. It runs `rcp run resume --run <dir> --only <steps>`, and stops
the engines before the next phase starts:

1. `retrieve` — the encoder (1 GPU);
2. `rerank` — the reranker;
3. `tournament`, `rubric` — the judge (8 GPUs);
4. `calibrate`, `evaluate` — no engine.

The engines' URLs reach each phase's coordinator through `RCP_NDCG_ENGINES`, as JSON
`{"judge": {"urls": ["http://node1:8000/v1"], "wait_on_outage_s": 900}, ...}`. The coordinator applies them
to the role configs at runtime only. They are never written into `run.yaml`, and they never enter a step
identity. A run resumed by hand is therefore byte-identical with or without the variable. A job's own `env`
cannot name the variable. The phase overlay owns it, and it refuses it at config time.
`run resume --engine role=url[,url]` is the same overlay on the command line, for engines that you started
yourself. It is repeatable. A job that starts an encoder or reranker engine starts one replica for it. This
release's retrieval clients address one replica URL, and more is refused.

A phase refuses to start its engine on a port that already answers. A previous phase's engine can outlive its
`srun` client, or one can be started by hand. Such an engine would answer the readiness probe. The phase's own
engine would then die on `EADDRINUSE`, after the coordinator had already talked to the wrong one. After a
phase's coordinator exits 0, the engines are stopped. Each engine's port is waited on until it stops
answering, up to the same 20-second stop grace. The next phase can therefore never reach the previous phase's
engine.

#### GPUs are partitioned per node, not shared

A vLLM-class engine pre-allocates most of each GPU that it sees (`--gpu-memory-utilization`). Two co-located
engines that both see every GPU therefore fail with out-of-memory. The runner therefore treats a phase's GPUs
as a fixed pool that the engines **partition**, per node:

- **The request**. A node's GPU request is the **sum** of what runs on it. That sum is the coordinator's own
  `resources.gpus` plus the engine placed there. The coordinator runs on the phase's first node, or in the phase's
  container. One replica per node is placed on SLURM, and each of the container's engines on Kubernetes. The job asks
  for the maximum of that over the phases. SLURM's `--gres` is per node, so one value is asked on every node;
  Kubernetes' limit is the container's.
- **The devices.** Every co-located engine process gets a disjoint `CUDA_VISIBLE_DEVICES` slice. Take a 4-GPU judge and
  a 1-GPU encoder in one phase container. That needs a hand-built `JobPhase`, because the `serve:` planner names one
  engine per phase. The container asks for 5, and the judge runs with `0,1,2,3`, the encoder with `4`. The slices
  continue across a role's replicas when they share a GPU set. Two replicas of a 2-GPU engine get `0,1` and `2,3`, with
  distinct ports. An engine that declares no GPUs gets the empty slice. It sees no device, never all of it. The
  coordinator is confined the same way. On Kubernetes it exports its own reserved slice (`0..resources.gpus-1`, empty
  for none). On SLURM its step carries the empty `CUDA_VISIBLE_DEVICES` when it asks for no GPU. A job env entry named
  `CUDA_VISIBLE_DEVICES` is refused. The runners that partition GPUs assign it from `resources.gpus`. An operator's
  value would widen the coordinator's view past the devices that it reserved. An engine's own `serve.env` may still
  declare a slice; the e2e driver's node-runtime slots do. The runner's slice wins where it assigns one: co-located
  engines, and a GPU-less engine. A GPU engine's `serve.env` value stands over SLURM's per-step grant, and over a
  several-replica Kubernetes engine pod's allocation.
- **SLURM**. Each role's replicas are pinned to a disjoint slice of the allocation's nodes, one replica per node. No two
  engine processes share a node. A step's `--gres` is its own engine's count. SLURM's per-step `CUDA_VISIBLE_DEVICES` is
  set per step with unique devices (gres.html, "GPU Management"). It could still overlap across steps, because the
  engine steps run under `srun --overlap`. srun(1) documents that as allowing steps to "share all resources (CPUs,
  memory, and GRES) with all other steps" (SLURM 26.05). The coordinator's task therefore claims its own `--gres` under
  every container runtime, the default included. That is its reservation, on the phase's first node. The node pinning
  keeps the engine steps apart, and a node's `--gres` carries the sum of what runs on it. A cluster that constrains
  devices per step (`ConstrainDevices=yes`) should be checked against these slices.

What the runners submit:

| | One replica per engine | Several replicas of an engine |
|---|---|---|
| **Kubernetes** | each engine phase is an init container whose engines run in one container of the (single) engine's `image` (a phase's engines share one image and, if several, need distinct ports), the last phase the main container; the container's command is the supervision script below, which starts the phase's engines and its coordinator side by side and talks to them on `localhost`; `restartPolicy: Never` fails the pod on the first phase that exits non-zero, and the run directory on the pod's `emptyDir` is mounted by every container | as before, plus one StatefulSet of engine pods (`podManagementPolicy: Parallel`) behind a headless Service per role with several replicas, both owned by the Job, so `run cancel` or the Job's TTL (one hour by default for a Job that owns engines, or `ttl_seconds_after_finished`) deletes them; such engines are **run-scoped** — they live for the whole run, not one phase — and the phases that use them wait for the pods' stable names, at most `startup_timeout_s` until one replica answers; a phase whose engines are all StatefulSet replicas waits in the coordinator's image |
| **SLURM** | one sbatch asking for the maximum nodes and GPUs over the phases, one supervision block per engine phase: each role's engines are one background step (`srun --overlap`, one replica per node, pinned to their slice of the allocation's nodes), the URLs built from the node list, the coordinator on the first node | the same, with the several-replica roles' steps on their slices; the whole allocation holds the largest phase |

### When the engine fails, the job fails

A job that starts its engine never outlives it. Holding an allocation for a dead or hung engine costs more
than starting again. The judgement stores are append-only, a mirror keeps them, and `run resume` asks only for
the windows that they lack. The supervision script of each phase is the same on SLURM and in a Kubernetes
container. It does four things:

- starts each of the phase's engines once, in the background, and never restarts them;
- waits until every role has a replica answering `readiness_path`, for at most `startup_timeout_s` (1800 s by
  default). An engine that exits before it answers fails the job at once. Its status and its own output are in the job's
  log;
- then runs the phase's coordinator in the background, and ends the phase with whichever ends first. When an engine
  exits, the coordinator is stopped (`SIGTERM`, then `SIGKILL` 20 s later). The job exits 1 with a message naming the
  engine. When the coordinator exits, the phase's engines are stopped and reaped. The next phase starts only if the
  coordinator exited 0. A failing phase ends the job with the coordinator's status;
- stops the phase's engines and the coordinator when the scheduler cancels or preempts the job (`SIGTERM` or `SIGINT`).
  No engine keeps running.

On SLURM, an engine step runs with `srun --kill-on-bad-exit=1 --wait=10`. One replica that fails ends the
whole step at once, and with it the job. A replica that exits with status 0 ends it 10 s later. On Kubernetes
with several replicas, the StatefulSet restarts an engine pod that dies. The run relies on the judge instead.
A job that starts engines hands each role an outage wait of `outage_timeout_s` through `RCP_NDCG_ENGINES`. A
judge that finds no replica answering for that long stops with `BackendUnavailableError` and a non-zero exit.
The wait is 900 s by default. The same bound applies to every job with `serve:`, so an engine that hangs
without exiting fails the run too. A judge whose engine the job does not start keeps its own
`wait_on_outage_s`.

A failed job is not retried by default. To run it again, engine included, submit the run again with
`rcp run resume --run <dir> --runner slurm`. The new job takes the runner options of the run's last job.
A `--set runner.options.<key>=...` change wins over the record. It takes that job's `serve:` section too. When
the run has a mirror, the new job restores the directory from the mirror first. It then resumes the run where
it stopped, asking only for the windows that its stores lack. `run resume` without `--runner` resumes in this
process, which starts no engine. On Kubernetes, a resubmission of the same run id is not available. The runner
refuses to apply over an existing Job, and Kubernetes never restarts one. Once the Job is gone, its handle
cannot be resolved. A finished Job is deleted with its engines by `ttl_seconds_after_finished`, an hour by
default for a Job that owns engines. `run resume --runner kubernetes` therefore refuses both ways. Start a new
run with its own run id and mirror prefix instead. On SLURM `sbatch` always creates a new job. `backoff_limit`
(0 by default) lets the Job retry a failed pod by itself. A retried pod resumes the run from its mirror.

The pod of one replica needs nothing from the cluster but an image and a command. Any launcher that takes
those two runs it, and no Kubernetes feature beyond a plain Job is used. The engine image needs `bash` 4.3 or
later, `python3` and `pip`; the stock vLLM image has them. The readiness probe uses Python's standard
library, and the coordinator runs in the engine image through `uvx`. An image without uv gets it first with
`python3 -m pip install --target`. That comes from the job's staged wheelhouse (`--no-index --find-links`)
when one is given, and from PyPI otherwise. An image that has uv needs no `pip`. The job checks these before
it starts the engine. An image without `bash` fails to start the container. An image can lack a recent enough
`bash`, `python3`, or both uv and `pip`. It then stops the job at once, with a message naming what is
missing. On SLURM, the node that runs the batch script is checked for `bash` 4.3 and `python3` the same way.

### Deployment modes

A suite (`dataset: suite:<name>`) is one run config and one coordinator. The judge's concurrency interleaves
the queries of all its datasets. The deployment shape therefore depends on the platform and the number of
nodes, not on how many datasets are judged:

| | One node | Several nodes |
|---|---|---|
| **One dataset or a suite, Kubernetes** | `serve: {judge: {replicas: 1}}`: one engine phase — the engine and the coordinator in one container of the Job's pod, engine-native data parallelism inside the node | `serve: {judge: {replicas: N}}`: N run-scoped engine pods, one coordinator balancing over them |
| **One dataset or a suite, SLURM** | `serve: {judge: {replicas: 1}}`: one sbatch, the engine as a background step | `serve: {judge: {replicas: N}}`: one N-node sbatch, one engine per node |
| **A shared or long-lived engine** | no `serve:`: point `judge.base_url` at it (below) | no `serve:`: list the replicas in `judge.base_url`, or point it at their gateway |

Replicas are independent engines. Do not span one engine's data parallelism over nodes for a
mixture-of-experts judge, and both paper judges are one. Its expert layers then synchronise every forward pass
across the nodes.

### The job interface

`rcp.runners.get_runner(name, **options)` returns the `local`, `slurm` or `kubernetes` runner. It can
also return a runner that another installed package registers under the `rcp.runners` entry-point group.
A job is one command line with its image, resources and environment (`JobSpec.argv`). It can also be the
phases to run in order (`JobSpec.phases`). It is exactly one of the two. A job without phases runs `argv`,
and a phased job's commands are its phases' `argv`. A plugin runner that renders a job's phases declares
`renders_phases = True` and takes the phased job. A runner without it is handed the whole-run command as the
job's `argv`. A job whose phases start engines is refused. `render` shows what would be submitted without
submitting anything:

```python
from rcp.runners import JobPhase, JobSpec, Resources, ServeConfig, get_runner

engine = ServeConfig(
    image="vllm/vllm-openai:v0.33.0",
    command="vllm serve openai/gpt-oss-120b --served-model-name gpt-oss-120b --reasoning-parser openai_gptoss",
    resources=Resources(gpus=8),
)
resume = ("rcp", "run", "resume", "--run", "/shared/runs/nano-nfcorpus")
job = JobSpec(
    name="nano-nfcorpus",
    resources=Resources(cpus=8, memory_gb=32, time_limit_s=86400),
    env={"HF_HOME": "/shared/hf"},
    phases=(  # a phased job takes no argv: its commands are its phases' argv
        JobPhase(engines={"judge": engine}, argv=(*resume, "--only", "tournament", "--only", "rubric")),
        JobPhase(argv=(*resume, "--only", "calibrate", "--only", "evaluate")),
    ),
)

slurm = get_runner("slurm", partition="gpu", account="my-project", log_dir="logs/slurm",
                   container_runtime="pyxis")  # the engine runs in the engine's image
print(slurm.render([job])["nano-nfcorpus"])  # the sbatch script

kubernetes = get_runner("kubernetes", namespace="eval", secrets=["hf-token"])
print(kubernetes.render([job])["nano-nfcorpus"])  # the Job: engine phases as init containers, the last phase
# the main container; several-replica engines as run-scoped StatefulSets owned by the Job
```

`runner.submit([job])` submits and returns handles. `runner.status(handle)`, `runner.logs(handle, tail=100)`
and `runner.cancel(handle)` follow them. A name that more than one installed distribution provides is refused,
so a plugin cannot shadow `local`, `slurm` or `kubernetes`.
`rcp.testing.runner_conformance(runner, job=...)` is the seam's contract that a plugin's own tests call.
It covers the four methods, the answers' shapes, and `render`, `renders_phases` and `run_root` when declared.

## Run events

A run publishes what it does as data. A dashboard, a `wandb` run or a progress line of your own can therefore
follow it without patching the pipeline. A subscriber is a callable that receives a `RunEvent`. It is installed
as an entry point in the `rcp.events` group, which is optional and which the product declares none of:

```toml
[project.entry-points."rcp.events"]
my-dashboard = "my_package.events:subscribe"
```

`subscribe(event)` is called once per event, synchronously, in the order the run produces them. The vocabulary is
four phases, one event each:

| `phase` | when the run emits it | `detail` |
|---|---|---|
| `step_started` | the step is recorded as running, before its work | `{}` |
| `step_finished` | the step completed and its record is written | `{"status": "completed", "duration_s": <seconds>}` |
| `step_failed` | the step raised, was interrupted or cancelled, or a judging step refused its store's identity | `{"error": "<the manifest's redacted description>", "status": "failed"}` — `"cancelled"` when `run cancel` stopped it |
| `usage` | a step reported judge usage, right after the event that closed it | `{}`; the counts are in `usage` |

The event is `RunEvent(run_id, step, phase, at, usage, detail)`. It carries the run's id, the step's name and the
phase. It carries when the phase happened, in UTC. It carries the step's judge counts as the manifest records
them: `requests`, `failed_requests`, `input_tokens`, `output_tokens` and `cached_input_tokens`. Those counts are
`None` on every phase but `usage`. It also carries the phase's detail. It is a frozen pydantic model, so
`event.model_dump(mode="json")` is the JSON that a dashboard stores.

The ordering guarantee, in full:

- a step that runs emits one `step_started`, then exactly one `step_finished` or `step_failed`;
- a step's `usage` event, when it has one, follows the event that closed the step, so a failed judging step still
  reports what it asked the judge before it failed;
- the steps come in run order (the configured order, narrowed by `run resume --only`);
- a step a resume skips emits nothing: the event stream is what this invocation did, and the manifest is what the
  run has done;
- delivery is synchronous: the run does not produce the next event until the subscriber returns, so a slow
  subscriber slows the run, and no event is ever out of order;
- a subscriber that raises, or whose entry point does not load, is logged once for the run and then ignored. The
  run completes. The other subscribers still receive every event. The manifest records exactly what it records
  without a subscriber.

The model is `rcp.support.events.RunEvent`; a subscriber needs no import to be called (any callable taking the
event does). To follow a run from the same process, build an emitter over your own callables:

```python
from rcp.support.events import Emitter, RunEvent


def subscribe(event: RunEvent) -> None:
    print(event.phase, event.step, event.detail)


Emitter([subscribe]).emit(RunEvent(run_id="20260724-151233-nano-a1b2c3", step="retrieve", phase="step_started"))
```

`Emitter.installed()` is the emitter `Pipeline.run` builds: the group's subscribers, in entry-point name order. A
run with no subscriber installs nothing and emits nothing.

## Durability: local runs and a mirror

A run directory and a judgement store (`judge --out`) are written on a local or shared filesystem. They never
go straight to a bucket. A remote runs directory or `--out` is refused. Inputs are still read from anywhere:
`hf://`, `s3://` or `https://`.

A mirror copies what a run writes to any fsspec URI while it runs. A job that is preempted therefore loses at
most one interval of work. Set it with `--mirror s3://bucket/runs/<name>` on `run start`, `run resume` and
`judge tournament|rubric`, or with `mirror:` in the run config. `mirror_interval_s:` sets the interval, 60 by
default, and `judge --mirror` takes `--mirror-interval` for the same knob. A job that a runner starts carries
it on its `run resume` command line. On Kubernetes the mirror is how the run reaches its pod. The pod's disk
is scratch, so the prepared run directory is uploaded to the mirror before the Job is submitted. The pod
restores it into `/scratch/runs/<run_id>`. A Kubernetes run without a mirror is refused before anything is
written.

- The judgement stores and the text census go up every `mirror_interval_s`. They go up once more when the run ends, also
  on `SIGTERM` or `SIGINT`. They go up as immutable parts. `<file>.parts/<start>-<end>` holds the bytes appended since
  the last upload, up to the last complete line. Every other file goes up whole when it changed: the manifest,
  `run.yaml`, `identity.json`, the calibration, the reports and the log. `work/` is scratch space and is not mirrored.
- `run resume --run <dir> --mirror <uri>` first rebuilds the directory from the mirror on a node that lacks it, or that
  holds a shorter store. The parts go in offset order, and a gap or an overlap is refused with exit 12. The command then
  resumes as usual, so only the windows that the store lacks are asked. A directory whose manifest is older than the
  mirror's also takes the mirror's newer whole files. The run went on in a job elsewhere. The restore writes into the
  run directory before anything else, also with `--dry-run` or `--estimate`.
- A local or shared path (`/shared/mirrors/nano` or `file:///shared/mirrors/nano`) is a mirror too. Its directories
  are created as the mirror writes. A whole file there is published atomically: a temp file beside it, then one
  rename. A concurrent `restore()` on another host therefore never reads a partial `manifest.json` or
  `identity.json`. An object store writes each object whole anyway.
- The mirror is **run-scoped**. `restore` refuses a mirror whose `manifest.json` names another run. The run id is the
  local manifest's, salvaged from the damaged bytes when it does not parse. Otherwise it is the directory's name, when a
  job restores into a fresh directory. `run status` ignores such a manifest with a note, instead of adopting the other
  run's id and metrics. A damaged local `manifest.json` is replaced by the mirror's instead of making the restore crash;
  the recovery `RunManifest.load` names works.
- `logs/jobs.json` is host-local state: the runner and the job handles of the submitting host. The mirror never uploads
  or restores it. A restore therefore can never replace the handle of a job that this host can cancel.
- `run status` shows the mirror's last upload and its lag (`data.mirror.last_upload_at`, `data.mirror.lag_s`). The text
  output shows the note and the mirror's state, its last error included. The state file itself is published atomically
  too. An unparseable one reads as "never ran" with a warning instead of failing `run status`. A reader racing a flush,
  or a writer the kernel killed, produces one. A mirror that the client cannot read falls back to the local state with a
  note rather than aborting. Examples are a 403, an expired credential or a transport failure.
- A mirror URI that carries credentials (userinfo, a query, a fragment) is accepted. The job must reach the store with
  it. It is redacted wherever it is written down. `run.yaml`, the manifest, the state file, `run status` and every log
  line show `safe_url`'s form. The live config and the job's command line keep the full URI. A resume that reads the
  redacted `run.yaml` takes the credentials from the environment (the store SDK's own variables). It can also take them
  from a `--mirror` override. A secret-looking `env` value is refused when the config is read. It is redacted if it
  reaches a recorded file by another route. The run directory and the records that the mirror uploads (`run.yaml`,
  `manifest.json`, `logs/mirror.json`) are owner-only (`0700`/`0600`). The host-local `logs/jobs.json` is owner-only
  too. A shared cluster filesystem therefore does not expose them.

**The guarantee.** Durable is the last uploaded part. The mirror never rewrites an uploaded part, so a
graceful stop (`SIGTERM`/`SIGINT`, the block's end) uploads everything written. A hard kill (`SIGKILL`, a
power loss) loses at most the current interval's appended lines. The resume re-asks exactly the lost windows.
A judgement is keyed by its stable `record_id`, so a re-asked window is never duplicated. A window that is
already stored is never asked again. There is **one live writer per store**. The store's `flock` is same-host
only, so two hosts writing one store diverge. The diverged writer's next flush then refuses with `DataError`
("no longer extends what the mirror holds") rather than clobbering. The run continues unmirrored, and
`run status` shows the recorded error. Mirror the rewritten store to a new URI to start a fresh writer. Parts
and superseded files (`<file>.parts/`, a store's `.superseded/`) are never garbage-collected. A long-lived
mirror therefore keeps every part that it ever uploaded.

**Any fsspec filesystem is a mirror target**, because the mirror uses exactly three of its operations. It
writes an object with `pipe_file` on a remote target. A local or shared target publishes whole files
atomically through `storage.publish_bytes`. It reads an object with `cat_file`, and it lists a prefix with
`ls`. A remote target never asks whether an object exists, renames or appends. A local target's whole-file
write is one temp-file rename, and its temp files are skipped. GCS works as installed, because `gcsfs` is a
dependency. S3 needs `s3fs` (`pip install "rcp[s3]"`), and Azure needs `adlfs` (`[azure]`). A protocol
with no filesystem installed or registered stops the run at start with exit 10, naming what is missing.
`hf://` works, but every write to the Hub is a commit and its rate limits apply. Publish a finished run there,
and mirror a running one to an object store.

Your own storage is a small `fsspec.AbstractFileSystem` with those three methods. Register it under a protocol
with `fsspec.register_implementation`, or from your package's `fsspec.specs` entry point:

```python
import fsspec
from fsspec import AbstractFileSystem


class MyStore(AbstractFileSystem):
    """Objects in a dict; a real one would call its service's put, get and list."""

    protocol = "mystore"
    objects: dict[str, bytes] = {}

    def pipe_file(self, path, value, **kwargs):
        self.objects[self._strip_protocol(path)] = bytes(value)

    def cat_file(self, path, start=None, end=None, **kwargs):
        return self.objects[self._strip_protocol(path)]

    def ls(self, path, detail=True, **kwargs):
        prefix = self._strip_protocol(path).rstrip("/") + "/"
        names = {prefix + key[len(prefix) :].split("/")[0] for key in self.objects if key.startswith(prefix)}
        if not names:
            raise FileNotFoundError(path)
        entries = [{"name": n, "type": "file" if n in self.objects else "directory", "size": 0} for n in names]
        return entries if detail else sorted(names)


fsspec.register_implementation("mystore", MyStore, clobber=True)
```

Then `--mirror mystore://team/runs/nano` mirrors to it, and so does `mirror: mystore://team/runs/nano`. The
same class serves `rcp.runs.Mirror` directly:

```python
from pathlib import Path

from rcp.runs import Mirror, restore

Path("run/judgements").mkdir(parents=True)
Path("run/judgements/rubric.jsonl").write_text('{"window": 1}\n', encoding="utf-8")
Mirror("run", "mystore://team/runs/nano").flush()
assert restore("elsewhere", "mystore://team/runs/nano") == ["judgements/rubric.jsonl"]
```

## One engine, many judge jobs

For many runs against one long-lived engine, serve it once as a Deployment behind a Service. Run GPU-less
judge jobs against it. The manifest below serves `gpt-oss-120b` with the stock vLLM image. The judge config of
the run points `base_url` at the Service (`http://judge-engine:8000/v1`), and nothing else changes. Engine
health comes from `/health`, and the engine's Prometheus `/metrics` shows running and queued requests and
KV-cache use.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: judge-engine
spec:
  replicas: 1
  selector:
    matchLabels: {app: judge-engine}
  template:
    metadata:
      labels: {app: judge-engine}
    spec:
      containers:
        - name: vllm
          image: vllm/vllm-openai:v0.33.0         # pin the tag you tested
          command: [vllm, serve]
          args: ["openai/gpt-oss-120b", "--served-model-name", "gpt-oss-120b", "--reasoning-parser", "openai_gptoss",
                 "--tensor-parallel-size", "4", "--max-model-len", "131072", "--port", "8000"]
          ports: [{containerPort: 8000}]
          envFrom: [{secretRef: {name: hf-token}}]
          resources:
            limits: {nvidia.com/gpu: 4}
          startupProbe:
            httpGet: {path: /health, port: 8000}
            periodSeconds: 10
            failureThreshold: 360                 # up to an hour to load the weights
          readinessProbe:
            httpGet: {path: /health, port: 8000}
            periodSeconds: 10
          volumeMounts: [{name: dshm, mountPath: /dev/shm}]
      volumes:
        - name: dshm
          emptyDir: {medium: Memory, sizeLimit: 16Gi}
---
apiVersion: v1
kind: Service
metadata:
  name: judge-engine
spec:
  selector: {app: judge-engine}
  ports: [{port: 8000, targetPort: 8000}]
```

A restarted judge job resumes its run from its mirror. The judgement store asks the judge only for the missing
windows.
