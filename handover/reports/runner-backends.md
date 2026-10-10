# Lane runner-backends: SLURM and Kubernetes resources, lifecycle and confinement (runner review B6, B12, C1, C2, C5, C6, D)

## Status

DONE. Base `rfc-0001` at `b18d34c4` (runner-security already merged). Every claim of the brief was re-verified on
the base first: C1, V5, B6, B12, C5, C6, D and the redaction follow-up all reproduced; C2 did not reproduce as a
defect (the code already asks the per-node sum, not `nodes x max`; the pin test was added and the docs reworded to
the arithmetic). Three adversarial verifier rounds on `cohere-oss-v2/deepseek-v4-1-flash:xhigh`: round 1 (two
independent lenses) FAIL, round 2 (one confirmation) FAIL, round 3 (one confirmation) PASS. Every finding from
rounds 1 and 2 is fixed with a red-then-green test; round 3's two minors and three nits are fixed too. The current
`rfc-0001` was merged five times (`b8832a2e`, `67e6ef25`, `6c388950`, `b7af0c5c`, `c00a5f3e`) with no semantic drift, and the
final gate is on the merged head.

## Commits

| Commit | Subject |
|---|---|
| `0daa2228` | runners: the coordinator is confined to its reserved devices on SLURM and Kubernetes (runner review C1) |
| `8769bad5` | slurm: an image or mount the node runtime cannot honour is refused, not ignored (runner review V5) |
| `36be3315` | kubernetes: a pod the scheduler cannot place is pending, and its reason reaches run status (runner review B6) |
| `ca8d5bbe` | kubernetes: engine cleanup, no orphan on a failed submission, and declarable scheduling knobs (runner review B12) |
| `57c4051d` | kubernetes: a declared weights/cache volume, the HF cache on a volume and a per-engine TMPDIR (runner review C5) |
| `deadbd1c` | serve: the engine command is cross-checked against resources.gpus and the declared port (runner review C6) |
| `030f24e5` | runners: a local configs options are honoured, plan_phases co-location is pinned, and the schema check covers the rendered pods (runner review D) |
| `417e86b6` | runs: every recorded URI is redacted, not just the mirror and env (runner-security follow-up) |
| `3f796658` | docs, CHANGELOG and snapshots for the runner-backends lane |
| `9405913e` | runs: the scheduler note is a str before it is redacted (basedpyright) |
| `c8f6b3ec` | e2e: the four-phase golden follows the coordinator step (runner review C1) |
| `5d71444f` | Merge rfc-0001 (b8832a2e: wave-integrity) |
| `cb91bf4c` | Merge rfc-0001 (67e6ef25: run-integrity, mrl-harness) |
| `093d569c` | round-1 review fixes: the engine-free host list, credential-free identities, the engine TMPDIR volume, the runner-owned CUDA variable, engine pods in status, the systems selector and the V5 hints |
| `e2ee9779` | round-2 review fixes: the golden mkdir, the pod label is the capped Job name, an unschedulable pod beside a running coordinator, the cancel error, the CUDA scope and the resubmission docs |
| `9abeaf20` | Merge rfc-0001 (6c388950: late-keep) |
| `baed724f` | round-3 review nits: the CUDA scope in docs, the duplicated mrl_dim description, the local runner message and the cancel wording |
| `89042e44` | Merge rfc-0001 (b7af0c5c: harness-media) |
| `54123201` | Merge rfc-0001 (c00a5f3e: judge-fixes) |
| `96b0268e` | handover: the runner-backends lane report |

## What changed

### C2 -- SLURM `--gres` is per node

Re-verified, not a defect on the base: `render_job` computes `max over phases(res.gpus + max engine gpus)` and
emits it once; SLURM applies it to every node, so the job asks the per-node sum, never `nodes x max`. Added
`tests/runners/test_gpu_partition.py::test_slurm_asks_the_per_node_sum_not_the_node_count_times_it` (a two-node
shape asks `--gres=gpu:6`, never `gpu:12`); the docs' request bullet now names the placement (one replica per
node) that makes the maximum the node's sum, and the `..._the_sum` test's assertions stay.

### C1 -- the coordinator is confined to its reserved devices

* SLURM: the coordinator is a step of its own (`_coordinator_step`: `srun --overlap --nodes=1 --ntasks=1`, the
  `--nodelist` pin when multi-node, `--gres=gpu:N` when `resources.gpus > 0`) under every container runtime,
  the default `none` included, so its `--gres` claim and pin are no longer dropped. With `resources.gpus == 0`
  its worker exports the empty `CUDA_VISIBLE_DEVICES` (a step without `--gres` is granted the job's whole GRES).
  The engine-free phase uses the same step (and gains `--overlap` and the pin under pyxis).
* Kubernetes: every coordinator container exports its reserved slice
  (`device_slices([job.resources.gpus])[0]`, empty for none), disjoint from the engines' slices in the same
  container.
* `CUDA_VISIBLE_DEVICES` is the job runners': a job/`runner.options.env` entry of that name is refused
  (`JobSpec`/`JobOptions`), and `merge_phase_env` never exports a job's value for it (the runner's slice wins
  where it has one, the name is dropped where it does not, so SLURM's per-step grant stands). An engine's own
  `serve.env` may still declare a slice (the e2e driver's node-runtime slots do); the runner's slice wins where
  it assigns one, and a GPU engine's `serve.env` value stands over SLURM's per-step grant and over a
  several-replica Kubernetes engine pod's allocation.
* The `engine-free phased job` blocker found in round 1: `one_node = nodes <= 1` (an all-engine-free phased job
  has `nodes == 0`, and the host list is only read when `nodes > 1`), with
  `tests/runners/test_supervise.py::test_an_engine_free_phase_never_references_an_undefined_host_list` executing
  the rendered script.

### V5 -- SLURM `image` under `container_runtime: none`

`SlurmOptions._nothing_is_dropped_on_the_node` refuses `image` and `container_mounts` with a typed
`ConfigError` whose hint names `container_runtime: apptainer | pyxis`; `render_job` refuses a job's own `image`
there with the same hint. The engine's image refusal was already there.

### B6 -- an unsatisfiable GPU request is reported

`KubernetesRunner.status` reads the Job's pods (the Job's own and the run-scoped engine pods, selected by the
`rcp-ndcg/job` label) and reports `pending` when any pod's `PodScheduled` condition is False or none is Running,
`unknown` when the pod list cannot be read, and `running` only when a pod is Running and nothing is
unschedulable. `KubernetesRunner.note(handle)` returns the scheduler's message or a container's waiting reason;
`execution.status` appends it (redacted) to the run's note, guarded against a plugin runner that raises.

### B12 -- lifecycle, orphan submissions and scheduling knobs

* A finished Job that owns run-scoped engine StatefulSets is deleted, with them, an hour after it finishes
  (`DEFAULT_ENGINE_TTL_S`) unless `ttl_seconds_after_finished` says otherwise, so the engines no longer hold
  their GPUs forever.
* A submission that fails between the Job apply and the engine-objects apply deletes the Job it just applied; a
  compensating delete that itself fails raises a `RunnerError` naming the orphan, retryable false, and
  `run cancel` repeats the recorded submission error.
* `tolerations`, `affinity` and `priority_class` are declarable and render on the Job's pod and every engine
  pod. `engine_node_selector` is merged into the Job pod's selector when a single-replica engine runs there (a
  conflicting key is refused; a job with no engine refuses the selector).

### C5 -- the engine cache and per-engine TMPDIR

`KubernetesOptions.cache_volume` mounts a PersistentVolumeClaim at `/cache` in every container and points
`HF_HOME` there (else at the pod's scratch emptyDir, never the container's writable layer). Each engine process
gets its own `TMPDIR` (`/scratch/tmp/<role>` for a co-located engine, `/scratch/tmp/coordinator` for the
coordinator, an `engine-tmp` emptyDir mounted at `/scratch/tmp/engine` for a StatefulSet engine pod, so the
directory exists); `engine_script` creates the effective (post-`serve.env`) directory before the engine starts.

### C6 -- the engine command is cross-checked

`ServeConfig` refuses a command whose `--tensor-parallel-size` x `--data-parallel-size` product differs from
`resources.gpus` (the equals form included, DP-only included), or whose `--port` differs from `port`; the
Kubernetes runner's duplicate-port refusal for co-located engines was already there.

### D -- plan_phases, ignored options, the schema check

* `plan_phases` co-locates two engine roles when the caller's `uses` names both
  (`tests/support/test_serve.py::test_a_step_that_uses_two_roles_co_locates_their_engines`); the docs no longer
  imply `serve:` produces a two-engine phase (it names one engine per phase; the co-location example is a
  hand-built `JobPhase`).
* A config that names `runner: {name: local}` and sets any of `log_dir`, `detach`, `cwd`, `env` or `resources`
  is handed to the `LocalRunner` through `rcp-ndcg run start` and through the library's `rcp_ndcg.run()`
  (`execution.local_options_set`, one home); a bare local config still runs in-process.
* The `tests/runners/k8s_schema.py` structural check accepts what the runner renders now: `tolerations`,
  `affinity`, `priorityClassName`, `persistentVolumeClaim` volumes and the `emptyDir` fields.

### Redaction follow-up

`RunConfig.recorded()` redacts the dataset URI and its reader `*_uri` options, the rankings file, the evaluation
systems (keeping the semantic `#<system>` selector), the runner's `wheelhouse`/`constraints` and the mirror;
`redact_runner_options` is the one home for the install-source list (also used for `logs/jobs.json`). The step
identities (the manifest's and the judging store's `identity.json`) are built from the redacted URIs, so they
hash the same live and recorded and a credential never reaches a mirrored record; the manifest's revision key
is redacted too. Tests per field in `tests/runs/test_secrets.py`.

## Verification

**Round 1** (two independent verifiers, `deepseek-v4-1-flash:xhigh`, fresh context, one workflow; both on
`cb91bf4c`). Lens A (correctness): **VERDICT: FAIL** -- 2 blockers + 8 findings; Lens B (regressions/hygiene):
**VERDICT: FAIL** -- 2 blockers/majors + 6 minors. The load-bearing findings and their fixes:

* Blocker: SLURM rendered `${RCP_NDCG_HOSTS[0]}` for an all-engine-free phased job (`nodes == 0`), aborting
  under `set -u`. Fixed with `one_node = nodes <= 1` and an executed test (`093d569c`).
* Blocker: the step identities still recorded credentialed dataset/rankings/evaluation URIs in the mirrored
  `manifest.json` and `judgements/identity.json`, and live vs recorded hashed differently. Fixed by redacting
  inside the identity builders (`redact_dataset_payload`/`redact_candidates_payload`/
  `redact_evaluation_payload` in `runs/config.py`, used by `DatasetSource.identity()` and `Pipeline._identity`;
  `judging._dataset_identity`), with a live==recorded hash test (`093d569c`).
* Major: the StatefulSet engine pod's `TMPDIR` did not exist (tempfile fell back to `/tmp`). Fixed with the
  `engine-tmp` emptyDir mounted at the path (`093d569c`).
* Major: a job env `CUDA_VISIBLE_DEVICES` defeated the coordinator's reservation. Fixed by the refusal, the
  `PHASE_ENV` addition and the order-preserving `merge_phase_env` (`093d569c`, `e2ee9779`).
* Major (Lens B): `recorded()` stripped the semantic `#<system>` selector. Fixed with `safe_systems_location`
  (`093d569c`).
* Major (Lens B): an unschedulable engine StatefulSet pod was invisible to `status`/`note`. Fixed by selecting
  the engine pods through the Job's `rcp-ndcg/job` label (`093d569c`), then by making the label the capped Job
  name and reporting an unschedulable pod as `pending` beside a running coordinator (`e2ee9779`).
* Minors fixed: unreadable pod list -> `unknown`; the library's local options; the V5 hint; `engine_script`'s
  mkdir of the effective TMPDIR; the TTL/resubmission docs; the failed compensating delete named; one
  `redact_runner_options`; a raising `note` guarded (`093d569c`).

**Round 2** (one fresh confirmation verifier, both lenses, `e2ee9779`). **VERDICT: FAIL** -- 1 blocker + 2
majors + 4 minors, all fixed (`e2ee9779`):

* Blocker: the committed e2e golden was stale after the mkdir/`PHASE_ENV` change. Fixed by making
  `merge_phase_env` keep the merged export position and regenerating the golden (the diff is exactly the three
  `mkdir -p` lines); `tests/test_e2e_scenarios.py` passes.
* Major: the CUDA scope claim did not cover a GPU engine's `serve.env` over SLURM's per-step grant; the docs
  and CHANGELOG now say exactly where the runner's slice wins and where the engine's own stands.
* Major: the TTL/resubmission docs advice was not actionable (a Kubernetes resubmission of the same run id is
  refused both while the Job exists and after it is gone); the paragraph now says so and points at a new run.
* Minors: the local-runner CUDA message; the unschedulable-beside-running status; `run cancel`'s message; the
  long-name label collision (the label is `k8s_name(job.name)`).

**Round 3** (one fresh confirmation verifier, both lenses, `9abeaf20` + `e2ee9779`). **VERDICT: PASS** with 2
minors and 3 nits, all fixed (`baed724f`): the CUDA scope wording now covers the SLURM GPU-engine and
several-replica Kubernetes cases; the duplicated `mrl_dim` docstring (a late-keep merge artifact) is gone and
the schemas regenerated; the `merge_phase_env` comment, the `base.py` docstring and the cancel wording are
accurate. The verifier confirmed the golden renders byte-identically, the label/Service/StatefulSet selectors
agree, the status shapes (unschedulable, scheduled, backoff) are right, and the merge with late-keep is
intact; the unpublished suite was 1008 passed / 227 skipped.

## Checks

Last commands on the code head (`54123201`, rfc-0001 `c00a5f3e` merged; the report commit `96b0268e` adds only
this file), and their result lines:

```
uv run --no-sync ruff format --check .        -> 602 files already formatted
uv run --no-sync ruff check .                 -> All checks passed!
uv run --no-sync basedpyright                 -> 0 errors, 0 warnings, 0 notes
flock /tmp/rcp_heavy.lock uv run --no-sync pytest tests/ -q -n 4 -p no:cacheprovider
                                              -> 3931 passed, 102 skipped
uv run --no-sync pytest tests/contract tests/docs -q
                                              -> 301 passed, 55 skipped
uv run --no-sync pytest rcp-ndcg-test/tests -q -p no:cacheprovider
                                              -> 1038 passed, 227 skipped (the gate)
uv run --no-sync mkdocs build --strict        -> Documentation built
bin/gate lane/runner-backends                 -> GATE: PASS (slot 2; see below)
```

The gate on `54123201` (slot 2):

```
rev lane/runner-backends = 54123201 (slot 2)
ruff-check exit=0 All checks passed!
ruff-format exit=0 602 files already formatted
basedpyright exit=0 0 errors, 0 warnings, 0 notes
pytest exit=0 3971 passed, 102 skipped in 67.89s (0:01:07)
contract-docs exit=0 301 passed, 55 skipped in 42.51s
mkdocs exit=0 Documentation built in 1.44 seconds
test-pkg exit=0 1038 passed, 227 skipped in 756.23s (0:12:36)
recipes exit=0 recipes: no failure outside the baseline (0 baseline failures remain, 0 fixed; pytest exit 0)
vllm-pkg exit=0 49 passed in 5.27s
vllm-models exit=0 92 passed, 7 skipped in 113.51s (0:01:53)
run_all exit=0 leaderboards: 1022 checks, 987 match, 35 known deviations, 0 failed;
              human study: 67 checks, 67 match; external LLM judges: 82 checks, 82 match
public-names exit=0 public-names: clean (0 baselined hits remain)
clean exit=0 clean
GATE: PASS
```

The failing-test-first evidence: the four new C1 tests (`test_the_coordinator_claims_its_gpus_on_the_node_runtime_too`,
`test_a_gpu_less_coordinator_never_sees_the_jobs_gres`, `test_a_gpu_less_coordinator_sees_no_device_on_kubernetes`,
`test_slurm_asks_the_per_node_sum_not_the_node_count_times_it`), the V5, B6, B12, C5, C6, D and redaction tests
were red on the unfixed base; the round-1/2 fix tests were red on the pre-fix heads (the verifiers reproduced
each failure independently, and three mutations -- the SLURM `cuda`, the K8s pod check and `_redact_uris` --
each turned the corresponding tests red).

## Open questions

* **The Kubernetes TTL default is 3600 s.** The brief asked for cleanup after a run; a non-`None` default was
  needed (the review's K2 closure test accepts one), and one hour keeps the Job's logs readable while bounding
  the idle GPUs. An operator who wants a different grace sets `ttl_seconds_after_finished`.
* **A Kubernetes resubmission of the same run id is not available.** `_refuse_existing_job` refuses to apply
  over an existing Job (run-integrity's K1 fix) and `_refuse_live_jobs` refuses a handle the runner can no
  longer resolve, so neither deleting the Job nor letting the TTL fire makes `run resume --runner kubernetes`
  work; the docs now say a new run. A `run prune`/`--force` that deletes a terminal Job and clears the handle
  would restore the documented path -- a product gap for a later lane.
* **A GPU engine's `serve.env` slice can override SLURM's per-step grant.** The e2e driver's node-runtime slots
  rely on that (they have no SLURM client and the srun shim drops `--gres`), so the behavior is documented
  rather than refused. If the runners should own the variable strictly, the harness needs another pinning route.
* **Credentialed dataset URIs now require the credentials from the environment on a resume/job.** The identity
  and every recorded copy are credential-free (the point of item 9); a Kubernetes job that restores the
  redacted `run.yaml` cannot read a credentialed `s3://user:pass@...` dataset unless the store reads the
  credentials from the environment. The live config and the submitting host's flush keep them.
* **The C6 check covers explicit command flags only.** A command of the form `rcp-ndcg-vllm serve <id>` renders
  its `--tensor-parallel-size` inside the engine, so the run config's `serve.<role>.resources.gpus` is not
  cross-checked against the recipe's own `resources.gpus` (the review's serve F5); only a command that names
  the flags itself is checked.
* **`_recorded_options`/`redact_runner_options` redacts `wheelhouse` and `constraints` only.** A plugin
  runner's other URI-valued option is not covered; the brief named the two install sources.
* The SLURM per-step `CUDA_VISIBLE_DEVICES` disjointness under `srun --overlap` is documented SLURM behaviour
  that no CPU test can verify (no cluster); the review's residual uncertainty stands.

## CHANGELOG entry

The exact text added under `## Unreleased` (`### Public surface` and `### Fixed`):

> - **Runner options for engine lifecycle, placement and the engine cache** (runner review B6, B12, C5, C6, D):
>   `KubernetesOptions` gains `cache_volume` (a PersistentVolumeClaim mounted at `/cache` in every container: the
>   model weights and the HF cache, shared by every replica and kept across restarts), `tolerations`, `affinity`
>   and `priority_class` (rendered on the Job's pod and every engine pod). `KubernetesRunner.note(handle)` reports
>   why a job's pods are not running (the scheduler's message, an image-pull failure), and `run status` puts it in
>   its note; `KubernetesRunner.status` reports a Job whose pods are all Pending as `pending`, never `running`.
>   `rcp_ndcg.runners.kubernetes` exports `CACHE` and `DEFAULT_ENGINE_TTL_S`; `rcp_ndcg.runners.script.engine_script`
>   takes an optional `env` (the runner's per-replica cache and TMPDIR, under the engine's own). `ServeConfig`
>   refuses a command whose `--tensor-parallel-size` x `--data-parallel-size` product differs from
>   `resources.gpus`, or whose `--port` differs from `port`.
> - **The coordinator is confined to the devices it reserved on both backends** (runner review C1): on SLURM the
>   coordinator runs as a step of its own (`srun --overlap`) under every container runtime, so its own `--gres`
>   reservation and node pin hold with the default `container_runtime: none` too; with `resources.gpus: 0` it
>   exports the empty `CUDA_VISIBLE_DEVICES`, so a step srun(1) would grant the job's whole GRES sees no device.
>   On Kubernetes every coordinator container exports its reserved slice (`0..resources.gpus-1`, empty for none),
>   disjoint from the engines' slices in the same container. `CUDA_VISIBLE_DEVICES` is the job runners': a job env
>   entry of that name is refused (the runner assigns it from `resources.gpus`; the local runner inherits the
>   submitting environment), and the renderer never exports the job's value over the runner's slice or the
>   scheduler's per-step devices. An engine's own `serve.env` may still declare a slice (the e2e driver's
>   node-runtime slots do): the runner's slice wins where it assigns one (co-located engines, a GPU-less engine),
>   while a GPU engine's `serve.env` value stands over SLURM's per-step grant and over a several-replica
>   Kubernetes engine pod's allocation.
> - **An image or a mount the node runtime cannot honour is refused, not ignored** (runner review V5): a SLURM
>   job's or the runner's `image` and `container_mounts` are refused with a hint naming
>   `container_runtime: apptainer | pyxis` when `container_runtime: none` (the engine's image already was),
>   instead of rendering a script that never uses them.
> - **A Kubernetes pod the scheduler cannot place is pending, and its reason reaches `run status`** (runner
>   review B6): `JobStatus.active` counts a Pending pod, so an unsatisfiable GPU request used to read `running`
>   forever; the Job's own pods and the run-scoped engine pods (the `rcp-ndcg/job` label, which is the Job's
>   capped name) decide -- an unschedulable pod reports `pending` even beside a running coordinator -- and the
>   pod's `PodScheduled` condition (or a container's waiting reason) becomes the run's note. A pod list that
>   cannot be read reports `unknown` with a note, never a `running` that may never resolve.
> - **A run-scoped engine's StatefulSet is cleaned up after the run** (runner review B12): a finished Job that
>   owns several-replica engines is deleted, with them, an hour after it finishes unless
>   `runner.options.ttl_seconds_after_finished` says otherwise, so the engines no longer hold their GPUs forever;
>   a submission that fails between the Job apply and the engine-objects apply deletes the Job it just applied,
>   and a compensating delete that itself fails raises an error naming the Job that may still run (no GPU job is
>   left that the run's record does not name and no CLI command can cancel).
> - **The Kubernetes engine cache lives on a volume, and every engine has its own TMPDIR** (runner review C5):
>   `cache_volume` mounts a PersistentVolumeClaim at `/cache` and `HF_HOME` points there (else at the pod's
>   scratch emptyDir, never the container's writable layer); each engine process gets a distinct
>   `TMPDIR=/scratch/tmp/<role>` (the coordinator its own, the run-scoped engine pod its own emptyDir mounted at
>   the path), created before the engine starts, and `engine_script` creates the effective directory after the
>   engine's own `serve.env` override.
> - **An accepted local-runner option is honoured, not dropped** (runner review D): a config that names
>   `runner: {name: local}` and sets any of `log_dir`, `detach`, `cwd`, `env` or `resources` is handed to the
>   local runner instead of running in-process, which ignored them -- through `rcp-ndcg run start` and through
>   the library's `rcp_ndcg.run()` alike.
> - **The `mrl_dim` schema description is no longer duplicated** (a merge artifact of the late-keep change): the
>   stale "below dim" block is gone, so `schemas/run-config.v1.json` describes `k == dim` as the identity
>   selection once.
> - **Every URI a run records is redacted, not just the mirror and `env`** (runner-security follow-up): the
>   dataset and its reader `*_uri` options, the rankings file, the evaluation systems, the runner's `wheelhouse`
>   and `constraints`, the manifest's revision keys, and the step identities (the manifest's and the judging
>   store's `identity.json`, hashed in their redacted form so a live and a resumed config key alike) pass through
>   `safe_url`, so userinfo and query never reach the mirrored `run.yaml`, `manifest.json`, `logs/jobs.json` or
>   `judgements/identity.json`. An evaluation system's `#<system>` selector is semantic and is kept; the live
>   config and the job's command line keep the credentials the stores need.

## Public surface changes

* `KubernetesOptions`: new `cache_volume`, `tolerations`, `affinity`, `priority_class`; `engine_node_selector`
  now merges into the Job pod's `nodeSelector` for a single-replica engine (a conflicting key is refused) and
  is refused for a job with no engine.
* `KubernetesRunner`: new `note(handle)` method; `status` returns `pending` for unschedulable/non-running pods
  and `unknown` when the pod list cannot be read; `submit` deletes what it applied when a later apply fails and
  raises a typed `RunnerError` naming an orphan it could not delete; a Job that owns engine StatefulSets gets
  `ttlSecondsAfterFinished=3600` unless the option says otherwise.
* `rcp_ndcg.runners.kubernetes` exports `CACHE` and `DEFAULT_ENGINE_TTL_S`; the `rcp-ndcg/job` label is the
  capped Job name (`k8s_name`), and `_label_value`'s old truncation rule is gone.
* `rcp_ndcg.runners.script.engine_script` takes an optional `env`; `PHASE_ENV` includes
  `CUDA_VISIBLE_DEVICES`; `merge_phase_env` never exports the overlay's names from the job's env.
* `JobSpec`/`JobOptions` refuse an env entry named `CUDA_VISIBLE_DEVICES`.
* `SlurmOptions` refuses `image`/`container_mounts` with `container_runtime: none`; `render_job` refuses a job
  image there.
* `ServeConfig` refuses a command whose `--tensor-parallel-size` x `--data-parallel-size` product differs from
  `resources.gpus`, or whose `--port` differs from `port`.
* `RunConfig.recorded()` redacts the dataset/reader-option/rankings/evaluation/install-source URIs and keeps
  the `#<system>` selector; the manifest's revision key and the step identities are credential-free.
* `rcp_ndcg.runs.execution.local_options_set` (internal) is shared by the CLI and the library.
* Schemas regenerated: `schemas/run-config.v1.json`, `run-start.v1.json`, `run-status.v1.json` (and
  `index.v1.json` for the `mrl_dim` description); snapshot `tests/contract/snapshots/python_api.json`.
* CLI: no new flag or exit code. `run start` with a config that names local and sets options now submits to the
  `LocalRunner` instead of running in-process.

## Files outside scope

* `rcp-ndcg-test/tests/fixtures/golden/job-text-four-phases.sh` (the e2e golden: the C1 coordinator step, then
  the round-1 `mkdir` lines; regenerated twice, reviewed).
* `rcp-ndcg/src/rcp_ndcg/inference/config.py` (the duplicated `mrl_dim` docstring, a merge artifact of the
  late-keep lane; the stale block removed and the schemas regenerated).
* `docs/concepts/runs.md`, `CHANGELOG.md`, `schemas/*`, `tests/contract/snapshots/python_api.json` (docs,
  changelog and the public-surface regeneration every public change owes).

## For the next lanes

* **Kubernetes resubmission** (above): `run resume --runner kubernetes` cannot resubmit a run whose Job exists
  or is gone; a `run prune` or a `submit --force` that deletes a terminal Job and clears the record's handle
  would restore the documented path. The docs now describe the new-run route.
* **`serve.env` CUDA vs the scheduler** (above): the harness's node-runtime slots pin engines through
  `serve.env`; if the product should own the variable for GPU engines too, the harness needs another route.
* **The `mrl_dim` docstring** was fixed here because the merge carried the duplicate; the late-keep lane's own
  report should note the schema re-export.
* The review's other runner items (B7-B9, C3) are run-integrity's; this lane merged its fixes and the docs say
  what the merged code does.
