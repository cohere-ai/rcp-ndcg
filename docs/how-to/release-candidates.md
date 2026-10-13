# Release candidates and the GPU waves

How a release candidate is built, what the GPU node installs from it, and how wave 0 proves the node
before any recipe wave. The waves themselves and their tiers are described in [validate a recipe on
GPUs](validate-a-recipe.md)
and the [recipe guide](add-a-model.md); this page is the operator's procedure. Everything below uses
`gs://YOUR-BUCKET/rcp-ndcg` as the private stage location — a tracked file never names a real one, so
the operator passes the real prefix on the command line or as `RCP_STAGE_PREFIX`.

## Build a release candidate

`rc_build.sh` builds the three published distributions with the same commands the release workflow runs, checks
them (versions against the checkout, the pins, the constraints file against the lock, twine), builds the
**harness wheel** (`rcp-ndcg-test`, the unpublished test distribution the node's client environment imports)
into its own `<stage>/harness/` directory, builds the **wheelhouse** (every locked dependency beside the
release wheels, the CPU torch build included), smoke-installs from it in a fresh venv, stages everything, and
writes a hash manifest:

```bash
export RCP_STAGE_PREFIX=gs://YOUR-BUCKET/stage            # private location; operator's command line only
export EXTRA_DIRS="/path/to/private-pairs /path/to/private-wave-lists"
rcp-ndcg-test/src/rcp_ndcg_test/jobs/rc_build.sh rc0                # or rc_build.sh rc0 <commit>
```

The build runs in a detached worktree of the given commit, so a dirty checkout is fine. The staged tree
under `<prefix>/rc0/` is what the node installs from:

| Path | Contents |
|---|---|
| `dist/` | the six release files, as `release.yml` builds them |
| `harness/` | the unpublished `rcp-ndcg-test` wheel (the harness), built by name into its own directory; the node's client environment installs it from here |
| `wheelhouse/` | the release wheels plus every locked dependency for the node's platform |
| `requirements-constraints.txt` | the lock's export — the install's constraints file |
| `recipes/` | the recipe families (family.yaml + its variants table, the family's reference.py, its template, its `reference.in`/`reference.lock`), from the rcp-ndcg-vllm wheel's package data |
| `wave-lists/<wave>.txt` | one recipe id per line, per wave (the T4 scenario wave's: one scenario id per line), from `rcp-ndcg-test/wave-lists/`; `submit.sh` derives one `<wave>.<image-slug>.txt` per engine image at submit time (owner decisions 38/35) |
| `scenarios/<id>.yaml` | the T4 run scenarios, for a `--script e2e` wave (stage them beside `recipes/`) |
| `pairs/` | the stage-2 pairs files, from `rcp-ndcg-test/pairs/` |
| `cases/` | the reference cases, from `rcp-ndcg-test/cases/` (the `--script casefill` job reads and edits them on the node; the wheel drops `cases/`) |
| `extra/<name>/` | the `EXTRA_DIRS` entries (private pairs, wave lists or recipes, and the wheelhouses for family pins with no index wheel such as `flash-attn`), as they are |
| `manifest.json` | the commit, the version, the CUDA-lock wheels inert on a CPU client (`nvidia-*`, `triton`), the SHA-256 of every staged file |

The wheelhouse is what makes a node install exact and independent of PyPI's state that night: the node
installs with `--find-links <wheelhouse> --no-index`, never asking an index (PyPI is only ever a
fallback). Building it downloads the locked dependencies once, on the operator's machine, from the CPU
torch index and PyPI only.

## The three environments on the node

A GPU job runs one container image: `submit.sh` groups a bootstrap wave's recipe list by each variant's
`engine.image` and submits one job per image (owner decisions 38/35) — the stock `vllm/vllm-openai:v0.31.0`
for most recipes, the digest-pinned nightly for `embeddinggemma-2` — with three environments that are never
mixed. `bootstrap.sh` builds them from a staged RC:

- **engine** — the image's own Python, which runs `vllm serve`. Untouched, except recipe plugin wheels
  installed with `--no-deps`: a spec that names a staged file installs from the staged tree; a name installs
  from the staged **wheelhouses only** (`--no-index --find-links <stage>/wheelhouse`, plus one `--find-links`
  per existing `<stage>/extra/<name>/wheelhouse`, where a wheel staged through `EXTRA_DIRS` lands; never an
  index). A plugin found nowhere is recorded with its exact name (the report's engine block lists it under
  `plugins_failed`)
  and the wave marks exactly the recipes that name it failed — one failing recipe never stops the wave, end
  to end (a recipe that fails validation is likewise reported on stderr, skipped by the collect step, and
  marked failed in the wave report with the validation message). A `pip freeze` before and after must differ
  by exactly the installed plugin wheels (in wave 0, by nothing at all), or the bootstrap fails before any
  engine starts.
- **client** — no separate venv: every client command runs through the product's own install mechanism
  (`rcp_ndcg.runners.script.install_argv`: `uvx --find-links <wheelhouse> --no-index`, held to the
  staged constraints file) plus the staged harness wheel (`--with rcp-ndcg-test==<version>` from
  `<stage>/harness/`), so the waves exercise exactly the code users run and the wave tooling itself
  (`rcp_ndcg_test`) is present; the client probe imports it and checks its version against the manifest.
  `uv` itself is installed with `pip --target` (the product's own `bootstrap_uv` location), never into
  the engine environment.
- **reference** — one venv **per family**, keyed by the family's
  `recipes/<family>/reference.lock`: `--system-site-packages` over the image's torch/CUDA for the default
  families, a venv of its own for a family declaring `# own-torch: true`. The install runs
  `pip install --no-deps` from the staged wheelhouse(s) only, so the family's own pins (transformers,
  sentence-transformers, flash-attn, ...) take precedence over the image's copies and one recipe's pin can
  no longer break the wave. What `--no-deps` cannot pull — the reference venv's **own** distributions'
  missing dependencies (sentence-transformers' scikit-learn, scipy, joblib, threadpoolctl) — is completed
  from the wheelhouse to a fixed point by the mounted `reference_deps.py`; the image's distributions are
  never completed (that would shadow its CUDA stack), and a wheelhouse gap fails with the requirement
  names and the way out. Afterwards an import check runs in the client environment against the family's
  python: torch imports (the image's build for the default families, the lock's pin under own-torch) and
  every pinned distribution is installed at its pin and imports — a failure names the family and fails the
  bootstrap. The check also reports a lock pin another pinned distribution's requirement forbids (the r4
  wave's `huggingface-hub==2.2.0` against the same lock's `transformers==5.17.0`, which declares
  `huggingface-hub<2.0,>=1.5.0`): `uv pip compile --no-deps` never consults another pin's requirements, so
  an unpinned name in `reference.in` can resolve outside a pinned distribution's range and the completion
  step then moves the venv's copy; the check names both pins and the requirement. The report's `reference`
  block is a map of the built families (`lock_sha256`, `own_torch`,
  `torch_is_image_build`, the import facts), and each family's venv is reused by every variant of that family
  (its `freeze.txt` beside the venv is what `equivalence.json` records).
  The recipes' references run as subprocesses of their family's python, never inside the client, and their
  outputs are stored under `<out>/references` (reused when unchanged; `--reference-store` points at a
  previous wave's store).

The bootstrap verifies the staged files against the manifest before installing anything, records the
install times and the versions of all three environments, and fails if the installed versions differ
from the manifest. On the node it is mounted at `/etc/rcp/files/bootstrap/bootstrap.sh` (submitted by
`submit.sh`) and takes the same arguments a job passes it:

```bash
bootstrap.sh envs <STAGE_URI> --state <DIR>     # the three environments only (what wave 0 calls)
bootstrap.sh wave <STAGE_URI> <OUT_URI> --wave wave-a
```

The wave mode then runs the wave's list through `run_wave.py` from the client wrapper: one engine per
recipe on its GPUs, and one failing recipe never stops the wave. Each recipe's steps run in a worker of
their own under a declared per-step wall-clock budget (a stuck request fails that step and stops only its
engine), the pod log gets one `run_wave: <recipe> <step> start|passed|failed <secs>s` line per step,
`status.json` is rewritten atomically after every step, and with `--upload` each finished recipe's
directory lands the moment the recipe ends -- a cancelled pod keeps the evidence of everything that
finished ([validate a recipe on GPUs](validate-a-recipe.md) has the budgets, the log and the partial
results).  A recipe that fails validation is a
failed row in `wave.json`/`WAVE.md` with the validation message, a recipe whose plugin could not be
installed is a failed row with the plugin's exact name, and a recipe that measurably cannot fit the
free disk fails early (measured at the nearest existing parent on a fresh pod, where the cache does
not exist yet). The serve step records the engine's own outcome — started, answered, stopped cleanly
is a success whatever a later step's verdict is — and every slot's `TMPDIR` is a short per-slot
directory, because vLLM's ZMQ IPC socket path must fit AF_UNIX's 107 characters whatever the recipe
id is.

### Which gates a submitted wave runs

The submitted wave passes `--record` only (plus the staged pairs directory, the reference python and the
staged plugin wheel): it runs **T0 smoke**, **T2 equivalence** (stages 1 and 2) and the **recorder** for
every recipe. With `RCP_RECORD_CORPUS=1` in the operator's environment, `submit.sh` forwards
`env.RCP_RECORD_CORPUS=1` into the job spec and the bootstrap adds `--record-corpus`: that is the
**recording pass**, run as its own wave (a corpus failure is a corpus re-run, never a recipe failure --
the corpus step is part of the recipe's end state). The other gates are operator-run against the same
staged RC, with the wave runner in the client environment the bootstrap built:

- **T1 observations** — `--record-corpus` (with `--changed-since <previous recording wave.json>` to
  re-record only what moved; the flag takes a previous recording wave's `fingerprints`/`engine_versions`
  maps -- the repository corpus index is a different schema). It needs the client environment and the
  reference python a `bootstrap.sh envs` run built on the node; run the wave runner from that state's
  client wrapper with the staged recipes and pairs, and pass the staged plugin wheel
  (`--plugin-wheel <stage>/wheelhouse/rcp_ndcg_vllm-<version>-*.whl`, the wheel the engine environment
  installed) so the corpus step cross-checks its modules against the behaviour fingerprint's
  `plugin_sha256` inputs. `--changed-since` needs the pod's engine version before engines start: the
  bootstrap exports `RCP_ENGINE_PYTHON` for it. The corpora land at
  `<out>/observations/vllm-<version>/<recipe>/<fingerprint>/<stamp>/` and upload with the wave's final
  whole-tree copy.
- **T3 quality** — `--quality --paper-numbers <file>`. Its reference subprocess needs `mteb`,
  `rcp_ndcg[mteb]` and the harness (`rcp_ndcg_test`, the staged wheel) in the reference environment,
  which the bootstrap does not stage today, so run it in an environment built for it (the reference venv
  plus those packages) and keep its `quality.json` beside the recipe's status.
- **The negative controls (a)-(f)** — `--controls`, with the same client/reference environments as T2.
- **T4 end to end** is its own job: `submit.sh --script e2e` (below).
- **The cuda reference-case fills** are their own job too: `submit.sh --script casefill`. The job builds
the four cuda families' reference venvs through `bootstrap.sh envs` and runs
`python -m rcp_ndcg_test.case_fill --device cuda` per recipe over the staged `cases/` copy, then uploads
the edited YAMLs to `<OUT_PREFIX>/<wave>/cases/<recipe>/` and a `casefill-report.json` beside them; the
operator reviews the diff and copies the YAMLs into the repository. A pending case the fill cannot
produce is recorded skipped, its file untouched (a `kind: none` case carries no values by design); a
media-bearing case's `media/` files enter the pairs row's `media` field, so it fills like a text one; a
case's run-level `instruction` enters the pairs row only when the recipe's client places it (a recipe
declaring `instruction: none` drops it by declaration); a
case the reference refuses is a recorded exception (its reason in the report and in its note, the case
left `pending_gpu`, the recipe's other cases still filling).
- **The judge item of the release checklist** is its own job: `submit.sh --script judgecheck` (below).

A recipe's `status` flips to `verified` only from a wave whose document records the gates it needs; the
wave document's `recipes[].steps` names exactly which ran in that job.

## Submitting the waves

`submit.sh` submits one job per wave against a staged RC — at most `--max-jobs` in flight (a wave
beyond the cap is submitted with `depends_on` on the wave that many places ahead), with the wave's
Kueue priority class and the shared-memory size for eight engines:

```bash
export RCP_KJOBS_CONFIG=/path/to/jobs-config.yaml    # the job CLI's -f config (required, no default)
export RCP_GCS_AUTH_FILE=/path/to/gcs_auth.sh        # mounted at /etc/rcp/gcs_auth.sh; named, never read
export RCP_HF_TOKEN_FILE=/path/to/token              # mounted into the job and read there, never passed in an argv
export RCP_SUBMIT_DIR=/path/to/job-outputs           # optional: where the job CLI's output files land
rcp-ndcg-test/src/rcp_ndcg_test/jobs/submit.sh gs://YOUR-BUCKET/rc0 gs://YOUR-BUCKET/waves wave-a wave-b
```

Options: `--max-jobs N` (default 1), `--priority dev-high|dev-medium` (the `priority_class=` override,
rendered as `<class>-training-priority`; verify with a dry run),
`--script bootstrap|wave0|e2e|casefill|judgecheck|topk-probe`,
and `KJOBS=echo` to print the plan instead of submitting. The job CLI's output goes to a file under
`RCP_SUBMIT_DIR` (default: a fresh temp directory; the directory is created when it does not exist);
only job names and states are printed. `RCP_RECORD_CORPUS=1` is forwarded into every submitted job's
spec (the recording pass, above).

For a bootstrap wave, `submit.sh` downloads the staged wave list and recipes, groups the ids by each
variant's `engine.image` (`python -m rcp_ndcg_test.jobs.wavegroups`), and submits **one job per image**:
`app=rcp-<wave>-<image-slug>`, the recipe's image in `env.RCP_IMAGE`, the filtered list mounted at
`/etc/rcp/files/wavelist/<wave>.<image-slug>.txt` and passed as `--wave-list`. A wave whose recipes all
share one image is one job; `RCP_GROUP_IMAGES=0` disables the grouping (the old one-job-per-wave plan).
When the staged list or recipes cannot be read (a stage without them), the submitter warns and submits
the single ungrouped job, so the failure is visible rather than silent.

## Wave 0, the node test

Before any recipe wave, wave 0 runs **as one job on one node** from a development build staged like an
RC. It fails fast with a one-line reason at the first failure and writes one JSON report (to
`<OUT_URI>/wave0-report.json`, to `<RC_STAGE_URI>/reports/`, and to the job's stdout):

1. **preflight** — the node's assumptions, checked not assumed: `nvidia-smi` and `python3` on `PATH`,
   at least two GPUs, `/dev/shm` at least 8 GiB (the submission sizes it), at least 50 GiB free (the
   weights land on the container filesystem), the auth script and the HF token present.
2. **host** — the image (its digest as passed in `RCP_IMAGE_DIGEST`; `submit.sh` resolves it when it
   can — the Docker Hub registry for a public image, `gcloud container images describe` otherwise —
   and passes it in `env.RCP_IMAGE_DIGEST`, so the report never says null), the driver, the GPUs, free
   disk, `/dev/shm`, the python versions, and which GCS transfer path ran (`gcloud`, `gsutil`, or the
   python `gcsfs` helper — the image ships neither CLI). A Cloud SDK the auth script installs is put
   on `PATH` from the first directory of `RCP_GCLOUD_SDK_DIRS` that holds `gcloud` or `gsutil`
   (colon-separated; unset, the SDK's usual locations such as `$HOME/google-cloud-sdk/bin` and
   `/usr/lib/google-cloud-sdk/bin`; set empty, no directory is searched).
3. **bootstrap** — the three environments, and the engine's `pip freeze` unchanged (no plugin in
   wave 0).
4. **reach** — the Hub with the token secret (a metadata call; the reply's commit must be the pinned
   revision) and a `gs://` write/list/read/delete round-trip through `rcp_ndcg.storage` from the client
   environment. PyPI is not required.
5. **engines** — `vllm serve Qwen/Qwen3-Embedding-0.6B@97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` on one
   slot and a small second model on another, **at the same time**, each slot with its own
   `CUDA_VISIBLE_DEVICES`, HTTP port, `VLLM_PORT`, `TMPDIR` and log directory: both must answer their
   own model names, or the slots collide.
6. **embed** — from the client environment, the product's wired `EmbeddingClient` (the config's
   budget, fitted inside the client) over 20 texts (5 of them over the budget, cut by the product); the
   engine's `/tokenize` of every rendered input must equal the product's token ids (the engine is the
   tokenization truth), recorded per input.
7. **evict** — the models out of the HF cache, the free disk before and after (the engine may still
   hold the weights mapped, so the stop step re-measures).
8. **stop** — both engines' process groups stopped, then no engine process left on the node.
9. **uploads** — every report/log upload attempt, with its source, destination and error: a failed
    upload is never silent. The logs go up first so their records reach the uploaded report; the
    report's own final copy is the one attempt that cannot appear in the artifact (the stdout emit
    carries the complete record).

Run it through `submit.sh` (which mounts `wave0.sh` beside `bootstrap.sh`), or as the plain
`kjobs-go submit` line it prints — the exact command, with `KJOBS=echo` showing the overrides:

```bash
export RCP_STAGE_PREFIX=gs://YOUR-BUCKET/stage
rcp-ndcg-test/src/rcp_ndcg_test/jobs/submit.sh --script wave0 --priority dev-high \
  "$RCP_STAGE_PREFIX/rc0" gs://YOUR-BUCKET/waves wave0
```

which prints, in dry run, the submission the operator runs (the token file is mounted into the job and read
there by the wrapper; its value never reaches a process argv, an echo or a log):

```bash
# the from_file paths are absolute where submit.sh runs (its own checkout); shown shortened here
kjobs-go submit -f "$RCP_KJOBS_CONFIG" \
  app=rcp-wave0 priority_class=dev-high worker.shared_memory=128Gi \
  env.RCP_IMAGE=vllm/vllm-openai:v0.31.0 \
  env.RCP_IMAGE_DIGEST=sha256:0123...abcd \
  worker.command='/bin/bash /etc/rcp/files/hftoken/hf_token_env.sh /bin/bash /etc/rcp/files/wave0/wave0.sh '"$RCP_STAGE_PREFIX"'/rc0 gs://YOUR-BUCKET/waves/wave0' \
  files.wave0.from_file=rcp-ndcg-test/src/rcp_ndcg_test/jobs/wave0.sh \
  files.wave0.mount_path=/etc/rcp/files/wave0/wave0.sh \
  files.wave0host.from_file=rcp-ndcg-test/src/rcp_ndcg_test/jobs/wave0_host.py \
  files.wave0host.mount_path=/etc/rcp/files/wave0host/wave0_host.py \
  files.gcshelper.from_file=rcp-ndcg-test/src/rcp_ndcg_test/jobs/gcs.sh \
  files.gcshelper.mount_path=/etc/rcp/files/gcshelper/gcs.sh \
  files.gcspy.from_file=rcp-ndcg-test/src/rcp_ndcg_test/jobs/gcs.py \
  files.gcspy.mount_path=/etc/rcp/files/gcshelper/gcs.py \
  files.bootstrap.from_file=rcp-ndcg-test/src/rcp_ndcg_test/jobs/bootstrap.sh \
  files.bootstrap.mount_path=/etc/rcp/files/bootstrap/bootstrap.sh \
  files.report.from_file=rcp-ndcg-test/src/rcp_ndcg_test/jobs/report.py \
  files.report.mount_path=/etc/rcp/files/report/report.py \
  files.refdeps.from_file=rcp-ndcg-test/src/rcp_ndcg_test/jobs/reference_deps.py \
  files.refdeps.mount_path=/etc/rcp/files/refdeps/reference_deps.py \
  files.gcsauth.from_file="$RCP_GCS_AUTH_FILE" files.gcsauth.mount_path=/etc/rcp/gcs_auth.sh \
  files.hftoken.from_file="$RCP_HF_TOKEN_FILE" files.hftoken.mount_path=/etc/rcp/hf_token \
  files.hftokenenv.from_file="$RCP_SUBMIT_DIR/hf_token_env.sh" \
  files.hftokenenv.mount_path=/etc/rcp/files/hftoken/hf_token_env.sh
```

`WAVE0_DRY=1` prints the plan without running anything. The knobs (`WAVE0_MODEL`, `WAVE0_REVISION`,
`WAVE0_SECOND_MODEL`, `WAVE0_BUDGET`, `WAVE0_PORT_BASE`, `WAVE0_STARTUP_TIMEOUT_S`, the minimums) are
environment variables with the researched defaults; the report is `rcp-ndcg.wave0-report.v1`, its
schema exported at `rcp-ndcg-test/schema/wave0-report.schema.json`.

## The T4 scenario wave (`--script e2e`)

The end-to-end run scenarios of the GPU validation (a full `rcp-ndcg run` with `serve:` by role, its
phases executed in the pod, engines and coordinators under one supervision block per phase) run as one
job per wave through `e2e.sh`, which builds the three environments and drives
`python -m rcp_ndcg_test.e2e`. The wave list names **scenario ids** (or paths), one per line:

```bash
rcp-ndcg-test/src/rcp_ndcg_test/jobs/submit.sh --script e2e --priority dev-high \
  "$RCP_STAGE_PREFIX/rc0" gs://YOUR-BUCKET/waves e2e
```

The four shipped scenarios (`rcp-ndcg-test/scenarios/`, schema
`schema/scenario.schema.json`) are `text-four-phases` (a NanoBEIR subset: served encoder -> served
reranker -> served judge -> calibrate + evaluate, run twice and compared on identities and outputs),
`outage` (the judge engine killed mid-tournament: the run parks and recovers once, the
`wait_on_outage_s` expiry fails with `BackendUnavailableError`, and a resume finishes the run),
`identity` (the same run directory again on new engine ports: nothing recomputes) and `vidore` (a
ViDoRe v3 subset with page images). The judge candidates and their pinned commits live in the
scenarios, each with the `fallback` its T0 smoke concedes to (the Flash-Next NVFP4 candidate falls
back to its FP8 release). The wave's outputs (`E2E.md`, `status.json`, the run directories) upload to
the wave's `OUT_URI`.

The stage must carry the scenario YAMLs under `scenarios/` beside `recipes/` (until `rc_build.sh`
stages them with the recipe data, copy `rcp-ndcg-test/scenarios/*.yaml` into the stage).
`E2E_DRY=1 bash e2e.sh ...` prints the plan the node would run.

## The judge-check job (`--script judgecheck`)

The release checklist's judge item — `rcp-ndcg judge check` against each judge recipe's served engine,
plus the engine's media token count — runs as one job through `judgecheck.sh`, which builds the node's
client environment with `bootstrap.sh envs` and then, **one judge at a time** (the big judges need
tensor parallelism, so they never share the node):

```bash
rcp-ndcg-test/src/rcp_ndcg_test/jobs/submit.sh --script judgecheck --priority dev-high \
  "$RCP_STAGE_PREFIX/rc0" gs://YOUR-BUCKET/waves judgecheck
```

Per judge it serves the recipe's own argv (`rcp-ndcg-vllm serve <id> --port <N>`), waits for
`GET /v1/models` within `RCP_JUDGECHECK_STARTUP_S` (3600 s; a dead engine fails fast), runs
`rcp-ndcg judge check --judge <id> --judge-model <id> --set judge.base_url=http://127.0.0.1:<N>/v1 --json`
for each **documented** thinking mode of the recipe (`jobs/judgecheck_probe.py` owns the table: the Gemma 4
judges' `chat_template_kwargs.enable_thinking` choice merged over the recipe's own `extra_body`; the other
recipes document none, so their single documented mode is the shipped client block), runs the engine media
probe (one prepared probe image beside its no-media baseline under the processor family's stock budget; the
engine's media delta against the client's counted tokens), stops the engine's process group, records the
verdicts, the engine version and the NVFP4/Marlin backend lines from the engine log, and moves on — a
failed judge is recorded, never fatal. The outputs are `<OUT_PREFIX>/<wave>/<recipe>/<mode>.json`, the
`<recipe>/media.json` probes, and `judgecheck-report.json` beside them; the job exits 1 when any judge
failed. `JUDGECHECK_DRY=1 bash judgecheck.sh ...` prints the plan.

The knobs: `RCP_JUDGECHECK_PORT_BASE` (8100; judge *i* serves on base + *i*),
`RCP_JUDGECHECK_STARTUP_S` (3600), `RCP_JUDGECHECK_CHECK_S` / `RCP_JUDGECHECK_MEDIA_S` (900; the per-command
budgets).

## The topk probe job (`--script topk-probe`)

One recipe family's engine-vs-reference divergence, isolated experiment by experiment on a node: the
`topk-embed-v1` image-document gap. The job mounts `topk_probe.sh`, its `topk_probe.py` module and the
wave's per-row cosine fixture beside `bootstrap.sh`, builds the node's client environment and the family's
reference venv with `bootstrap.sh envs`, runs the probe on the engine image's own python, and uploads
`probe-report.json`, `probe-report.md` and the per-row dumps to `<OUT_PREFIX>/<wave>/topk-probe/`.

```bash
rcp-ndcg-test/src/rcp_ndcg_test/jobs/submit.sh --script topk-probe --priority dev-high \
  "$RCP_STAGE_PREFIX/rc0" gs://YOUR-BUCKET/waves topk-probe
```

The probe's experiments: **E0** the control, the probe's own engine vectors against a wave's recorded
vectors for one row (a file passed with `RCP_TOPK_CONTROL_VECTORS`; with none the report says `not_run`
and the numbers are uncalibrated, never a silent pass), **E1** the preprocessing isolation (the engine's
pixels against the checkpoint's, every filter/dtype variant against both), **E2/E3** the same tensors into
the engine's vision tower with forward hooks per layer, so the first tensor whose max-abs-diff crosses
`max(1e-2, 10 x the in-run bf16 floor)` names the layer, **E4** the image-token embeddings (the merger
output), **E5** the cross-feed (the reference tower on the engine's pixels and the engine tower on the
reference's), **E6** the grid sweep (each image document row's end-to-end per-token cosine beside the
wave's recorded minimum) and **E7** the text path layer by layer.

The knobs: `RCP_TOPK_VARIANT` (small; the variant whose recipe and checkpoint the probe uses),
`RCP_TOPK_ROWS` (empty: every image document row), `RCP_TOPK_DEVICE` (cuda:0; the engine),
`RCP_TOPK_REFERENCE_DEVICE` (cuda:1; the reference subprocess), `RCP_TOPK_GPU_FRACTION` (0.45; the
engine's `gpu_memory_utilization`), `RCP_TOPK_CONTROL_VECTORS` (unset: the control is `not_run`),
`RCP_TOPK_MODEL` / `RCP_TOPK_REVISION` (unset: the staged recipe's own checkpoint pin). Each is forwarded
into the job only when the operator sets it. `TOPK_DRY=1 bash topk_probe.sh <stage> <out>` prints the plan
the node would run.
