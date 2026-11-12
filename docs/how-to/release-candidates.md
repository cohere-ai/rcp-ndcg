# Release candidates and the GPU waves

This page describes how to build a release candidate. It describes what the GPU node installs from it. It
describes how wave 0 proves the node before any recipe wave. The waves themselves and their tiers are described
in [validate a recipe on GPUs](validate-a-recipe.md) and the [recipe guide](add-a-model.md). This page is the
operator's procedure. Everything below uses `gs://YOUR-BUCKET/rcp` as the private stage location. A
tracked file never names a real one. Thus the operator passes the real prefix on the command line or as
`RCP_STAGE_PREFIX`.

## Build a release candidate

`rc_build.sh` builds the three published distributions with the same commands that the release workflow runs. It
checks them: versions against the checkout, the pins, the constraints file against the lock, twine. It builds the **harness wheel** (`rcp-test`, the unpublished test distribution that the node's client environment imports). The wheel goes into its own `<stage>/harness/` directory. It builds the **wheelhouse**: every locked dependency beside
the release wheels, the CPU torch build included. It smoke-installs from the wheelhouse in a fresh venv. It
stages everything and writes a hash manifest:

```bash
export RCP_STAGE_PREFIX=gs://YOUR-BUCKET/stage            # private location; operator's command line only
export EXTRA_DIRS="/path/to/private-pairs /path/to/private-wave-lists"
packages/rcp-test/src/rcp_test/jobs/rc_build.sh rc0                # or rc_build.sh rc0 <commit>
```

The build runs in a detached worktree of the given commit, so a dirty checkout is fine. The staged tree under
`<prefix>/rc0/` is what the node installs from:

| Path | Contents |
|---|---|
| `dist/` | the six release files, as `release.yml` builds them |
| `harness/` | the unpublished `rcp-test` wheel (the harness), built by name into its own directory; the node's client environment installs it from here |
| `wheelhouse/` | the release wheels plus every locked dependency for the node's platform |
| `requirements-constraints.txt` | the lock's export — the install's constraints file |
| `recipes/` | the recipe families (family.yaml + its variants table, the family's reference.py, its template, its `reference.in`/`reference.lock`), from the rcp-vllm wheel's package data |
| `wave-lists/<wave>.txt` | one recipe id per line, per wave (the T4 scenario wave's: one scenario id per line), from `packages/rcp-test/wave-lists/`; `submit.sh` derives one `<wave>.<image-slug>.txt` per engine image at submit time |
| `scenarios/<id>.yaml` | the T4 run scenarios, for a `--script e2e` wave (stage them beside `recipes/`) |
| `pairs/` | the stage-2 pairs files, from `packages/rcp-test/pairs/` |
| `cases/` | the reference cases, from `packages/rcp-test/cases/` (the `--script casefill` job reads and edits them on the node; the wheel drops `cases/`) |
| `extra/<name>/` | the `EXTRA_DIRS` entries (private pairs, wave lists or recipes, and the wheelhouses for family pins with no index wheel such as `flash-attn`), as they are |
| `manifest.json` | the commit, the version, the CUDA-lock wheels inert on a CPU client (`nvidia-*`, `triton`), the SHA-256 of every staged file |

The wheelhouse is what makes a node install exact and independent of PyPI's state that night. The node installs
with `--find-links <wheelhouse> --no-index` and never asks an index. PyPI is only ever a fallback. Building the
wheelhouse downloads the locked dependencies once, on the operator's machine, from the CPU torch index and PyPI
only.

## The three environments on the node

A GPU job runs one container image. `submit.sh` groups a bootstrap wave's recipe list by each variant's
`engine.image` and submits one job per image. The image is the stock `vllm/vllm-openai:v0.33.0` for every recipe, the checkpoint `embeddinggemma-2` included. It carries its model plugin, not another image. The job has
three environments that are never mixed. `bootstrap.sh` builds them from a staged RC:

- **engine** — the image's own Python, which runs `vllm serve`. It is untouched, except recipe plugin wheels
  installed with `--no-deps`. A spec that names a staged file installs from the staged tree. A name installs from the staged **wheelhouses only**. The command uses `--no-index --find-links <stage>/wheelhouse`, plus one `--find-links` per existing `<stage>/extra/<name>/wheelhouse`, where a wheel staged through `EXTRA_DIRS` is placed. It never uses an index. A plugin found nowhere is recorded with its exact name; the report's engine block lists it under
  `plugins_failed`. The wave marks exactly the recipes that name it failed. One failing recipe never stops the
  wave, end to end. A recipe that fails validation is likewise reported on stderr. The collect step skips it, and the wave report marks it failed with the validation message. A `pip freeze` before and after must differ by exactly the installed plugin wheels. In wave 0, it differs by nothing at all. If it differs otherwise, the bootstrap fails before any engine starts.
- **client** — no separate venv. Every client command runs through the product's own install mechanism (`rcp.runners.script.install_argv`: `uvx --find-links <wheelhouse> --no-index`, held to the staged constraints file). It also gets the staged harness wheel (`--with rcp-test==<version>` from `<stage>/harness/`).
  Thus the waves exercise exactly the code that users run, and the wave tooling itself (`rcp_test`) is
  present. The client probe imports it and checks its version against the manifest. `uv` itself is installed
  with `pip --target` (the product's own `bootstrap_uv` location), never into the engine environment.
- **reference** — one venv **per family**, keyed by the family's `recipes/<family>/reference.lock`. It uses
  `--system-site-packages` over the image's torch/CUDA for the default families. A family declaring `# 
  own-torch: true` gets a venv of its own. The install runs `pip install --no-deps` from the staged
  wheelhouse(s) only. Thus the family's own pins (transformers, sentence-transformers, flash-attn, ...) take precedence over the image's copies. One recipe's pin can no longer break the wave. `--no-deps` cannot
  pull the reference venv's **own** distributions' missing dependencies (sentence-transformers' scikit-learn,
  scipy, joblib, threadpoolctl). The mounted `reference_deps.py` completes them from the wheelhouse to a fixed
  point. The image's distributions are never completed; that would shadow its CUDA stack. A wheelhouse gap
  fails with the requirement names and the way out. Afterwards an import check runs in the client environment
  against the family's python. torch imports: the image's build for the default families, or the lock's pin under own-torch. Every pinned distribution is installed at its pin and imports. A failure names the family
  and fails the bootstrap. The check also reports a lock pin that another pinned distribution's requirement
  forbids. Example: the r4 wave's `huggingface-hub==2.2.0` against the same lock's `transformers==5.17.0`,
  which declares `huggingface-hub<2.0,>=1.5.0`. `uv pip compile --no-deps` never consults another pin's
  requirements, so an unpinned name in `reference.in` can resolve outside a pinned distribution's range. The
  completion step then moves the venv's copy. The check names both pins and the requirement. The report's
  `reference` block is a map of the built families (`lock_sha256`, `own_torch`, `torch_is_image_build`, the
  import facts). Each family's venv is reused by every variant of that family; its `freeze.txt` beside the venv
  is what `equivalence.json` records. The recipes' references run as subprocesses of their family's python,
  never inside the client. Their outputs are stored under `<out>/references` (reused when unchanged;
  `--reference-store` points at a previous wave's store).

The bootstrap verifies the staged files against the manifest before it installs anything. It records the install
times and the versions of all three environments. It fails if the installed versions differ from the manifest.
On the node it is mounted at `/etc/rcp/files/bootstrap/bootstrap.sh` (submitted by `submit.sh`). It takes the same arguments that a job passes it:

```bash
bootstrap.sh envs <STAGE_URI> --state <DIR>     # the three environments only (what wave 0 calls)
bootstrap.sh wave <STAGE_URI> <OUT_URI> --wave wave-a
```

The wave mode then runs the wave's list through `run_wave.py` from the client wrapper. It runs one engine per
recipe on its GPUs, and one failing recipe never stops the wave. Each recipe's steps run in a worker of their
own under a declared per-step wall-clock budget. A stuck request fails that step and stops only its engine. The
pod log gets one `run_wave: <recipe> <step> start|passed|failed <secs>s` line per step. `status.json` is
rewritten atomically after every step. With `--upload`, each finished recipe's directory is copied when the recipe ends. Thus a cancelled pod keeps the evidence of everything that finished. [validate a recipe on GPUs](validate-a-recipe.md) has the budgets, the log and the partial results. A recipe that fails validation is
a failed row in `wave.json`/`WAVE.md` with the validation message. A recipe whose plugin could not be installed
is a failed row with the plugin's exact name. A recipe that measurably cannot fit the free disk fails early. The check is measured at the nearest existing parent on a fresh pod, where the cache does not exist yet. The serve step records the engine's own outcome. Started, answered and stopped cleanly is a success whatever a later step's verdict is. Every slot's `TMPDIR` is a short per-slot directory. vLLM's ZMQ IPC socket path must fit AF_UNIX's 107 characters whatever the recipe id is.

### Which gates a submitted wave runs

The submitted wave passes `--record` only, plus the staged pairs directory, the reference python and the staged
plugin wheel. It runs **T0 smoke**, **T2 equivalence** (stages 1 and 2) and the **recorder** for every recipe.
With `RCP_RECORD_CORPUS=1` in the operator's environment, `submit.sh` forwards `env.RCP_RECORD_CORPUS=1` into
the job spec, and the bootstrap adds `--record-corpus`. That is the **recording pass**, run as its own wave. A corpus failure is a corpus re-run, never a recipe failure. The corpus step is part of the recipe's end state.
An operator runs the other gates against the same staged RC. The wave runner uses the client environment that the bootstrap built:

- **T1 observations** — `--record-corpus`, with `--changed-since <previous recording wave.json>` to re-record only what moved. The flag takes a previous recording wave's `fingerprints`/`engine_versions` maps; the repository corpus index is a different schema. It needs the client environment and the reference python
  that a `bootstrap.sh envs` run built on the node. Run the wave runner from that state's client wrapper with
  the staged recipes and pairs. Pass the staged plugin wheel (`--plugin-wheel <stage>/wheelhouse/rcp_vllm-<version>-*.whl`, the wheel that the engine environment installed). Then the corpus step cross-checks its modules against the behaviour fingerprint's `plugin_sha256` inputs.
  `--changed-since` needs the pod's engine version before engines start; the bootstrap exports
  `RCP_ENGINE_PYTHON` for it. The corpora are written to
  `<out>/observations/vllm-<version>/<recipe>/<fingerprint>/<stamp>/` and upload with the wave's final
  whole-tree copy. The repository's copy of a recording is committed as the full corpus's redacted view when the recording carries private infrastructure. Examples are the pod's Kubernetes Service links and a staging path in the serve argv. The manifest's `engine.redacted` entry says what was removed. Each subset `index.json` points at
  the full corpus with the `gs://YOUR-BUCKET/...` placeholder. [use verified fake
  engines](use-verified-fake-engines.md) has the procedure.
- **T3 quality** — `--quality --paper-numbers <file>`. Its reference subprocess needs `mteb`, `cohere-rcp[mteb]` and the harness (`rcp_test`, the staged wheel) in the reference environment. The bootstrap does not stage that environment today. Run it in an environment built for it (the reference venv plus those packages). Keep its `quality.json` beside the recipe's status.
- **The negative controls (a)-(f)** — `--controls`, with the same client/reference environments as T2.
- **T4 end to end** is its own job: `submit.sh --script e2e` (below).
- **The cuda reference-case fills** are their own job too: `submit.sh --script casefill`. The job builds the four cuda families' reference venvs through `bootstrap.sh envs`. It runs `python -m rcp_test.case_fill --device cuda` per recipe over the staged `cases/` copy. It then uploads the edited YAMLs to
  `<OUT_PREFIX>/<wave>/cases/<recipe>/` and a `casefill-report.json` beside them. The operator reviews the diff
  and copies the YAMLs into the repository. A pending case that the fill cannot produce is recorded skipped, its file untouched. A `kind: none` case carries no values by design. A media-bearing case's `media/` files
  enter the pairs row's `media` field, so it fills like a text one. A case's run-level `instruction` enters the pairs row only when the recipe's client places it. A recipe declaring `instruction: none` drops it by declaration. A case that the reference refuses is a recorded exception. Its reason is in the report and in its note. The case is left `pending_gpu`, and the recipe's other cases still fill.
- **The judge item of the release checklist** is its own job: `submit.sh --script judgecheck` (below).

A recipe's `status` becomes `verified` only from a wave whose document records the gates it needs. The wave
document's `recipes[].steps` names exactly which ran in that job. The document also records the tested RC
(`rc.name`, `rc.commit`, `rc.manifest_sha256`). Every RC carries version `0.0.1`, so without it two RCs built
from different commits would produce indistinguishable evidence.

## Tagging the release

The tag's commit must equal the tested RC's commit. The tag pipeline rebuilds from the tag. A tag on another commit would publish bytes that no GPU node ever saw. The shared release check takes the wave documents whose evidence changed the statuses. It compares their recorded `rc.commit` with the tag commit:

```bash
git fetch --tags origin
python .github/scripts/check_release.py 0.0.1 --commit "$(git rev-parse v0.0.1)" \
  --wave /path/to/waves/wave-a/wave.json --wave /path/to/waves/wave-b/wave.json
```

A mismatch is refused naming both commits. A wave document without the `rc` record is refused too. It predates the recording and cannot tie the tag to the tested RC. `release.yml` runs the same check on a `workflow_dispatch` of the tag with the `rc_wave` input. Thus the checklist can enforce it in the tag pipeline itself.

## Submitting the waves

`submit.sh` submits one job per wave against a staged RC. It runs at most `--max-jobs` in flight. A wave beyond the cap is submitted with `depends_on` on the wave that many places ahead. It uses the wave's scheduler priority
class and the shared-memory size for eight engines:

```bash
export RCP_JOBS_CONFIG=/path/to/jobs-config.yaml     # the job CLI's -f config (required, no default)
export RCP_GCS_AUTH_FILE=/path/to/gcs_auth.sh        # mounted at /etc/rcp/gcs_auth.sh; named, never read
export RCP_HF_TOKEN_FILE=/path/to/token              # mounted into the job and read there, never passed in an argv
export RCP_SUBMIT_DIR=/path/to/job-outputs           # optional: where the job CLI's output files land
packages/rcp-test/src/rcp_test/jobs/submit.sh gs://YOUR-BUCKET/rc0 gs://YOUR-BUCKET/waves wave-a wave-b
```

Options: `--max-jobs N` (default 1), `--priority <class>` (the `priority_class=` override; default `RCP_PRIORITY`, else `default`; verify with a dry run) and `--script bootstrap|wave0|e2e|casefill|judgecheck|topk-probe|sdpa-probe|maxsim-probe`. `RCP_JOBS_CLI=echo` prints the plan instead of submitting. `RCP_JOBS_CLI` names the job CLI
(default `jobs`): any CLI that takes `submit -f <config> key=value...`. The job CLI's output goes to a file under `RCP_SUBMIT_DIR`. The default is a fresh temp directory, and the directory is created when it does not exist. Only job names and states are printed. `RCP_RECORD_CORPUS=1` is forwarded into every submitted job's
spec (the recording pass, above).

For a bootstrap wave, `submit.sh` downloads the staged wave list and recipes. It groups the ids by each variant's `engine.image` (`python -m rcp_test.jobs.wavegroups`) and submits **one job per image**. The job is `app=rcp-<wave>-<image-slug>`, with the recipe's image in `env.RCP_IMAGE`. The filtered list is mounted at `/etc/rcp/files/wavelist/<wave>.<image-slug>.txt` and passed as `--wave-list`. A wave whose recipes all
share one image is one job. `RCP_GROUP_IMAGES=0` disables the grouping (the old one-job-per-wave plan).
The staged list or recipes may be unreadable (a stage without them). Then the submitter warns and submits the single ungrouped job. Thus the failure is visible rather than silent.

## Wave 0, the node test

Before any recipe wave, wave 0 runs **as one job on one node**. The job uses a development build staged like an RC. It fails fast with a one-line reason at the first failure. It writes one JSON report (to
`<OUT_URI>/wave0-report.json`, to `<RC_STAGE_URI>/reports/`, and to the job's stdout):

1. **preflight** — the node's assumptions, checked not assumed. `nvidia-smi` and `python3` are on `PATH`. There
   are at least two GPUs. `/dev/shm` has at least 8 GiB (the submission sizes it). At least 50 GiB are free
   (the weights go on the container filesystem). The auth script and the HF token are present.
2. **host** — the image, its driver, the GPUs, free disk, `/dev/shm` and the python versions. It also records the GCS transfer path: `gcloud`, `gsutil`, or the python `gcsfs` helper; the image ships neither CLI. The image digest is passed in `RCP_IMAGE_DIGEST`. `submit.sh` resolves it when it can: the Docker Hub registry for a public image, `gcloud container images describe` otherwise. It passes the digest in `env.RCP_IMAGE_DIGEST`, so the report never says null. A Cloud SDK that the auth script installs is put on `PATH`. The first directory of `RCP_GCLOUD_SDK_DIRS` that holds `gcloud` or `gsutil` supplies it. The list is colon-separated. When unset, the SDK's usual locations are used, such as `$HOME/google-cloud-sdk/bin` and `/usr/lib/google-cloud-sdk/bin`. When set empty, no directory is searched.
3. **bootstrap** — the three environments, and the engine's `pip freeze` unchanged (no plugin in
   wave 0).
4. **reach** — the Hub with the token secret (a metadata call; the reply's commit must be the pinned revision). It also does a `gs://` write/list/read/delete round-trip through `rcp.storage` from the client environment. PyPI is not required.
5. **engines** — `vllm serve Qwen/Qwen3-Embedding-0.6B@97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` on one
   slot and a small second model on another, **at the same time**. Each slot has its own
   `CUDA_VISIBLE_DEVICES`, HTTP port, `VLLM_PORT`, `TMPDIR` and log directory. Both must answer their
   own model names, or the slots collide.
6. **embed** — the product's wired `EmbeddingClient` runs from the client environment over 20 texts. It uses the config's budget, fitted inside the client. 5 of them are over the budget and are cut by the product. The engine's `/tokenize` of every rendered input must equal the product's token ids. The engine is the tokenization truth. The result is recorded per input.
7. **evict** — the models out of the HF cache, and the free disk before and after. The engine may still hold the weights mapped, so the stop step re-measures.
8. **stop** — both engines' process groups stopped, then no engine process left on the node.
9. **uploads** — every report/log upload attempt, with its source, destination and error. A failed
    upload is never silent. The logs go up first so their records reach the uploaded report. The report's own final copy is the one attempt that cannot appear in the artifact. The stdout emit carries the complete record.

Run it through `submit.sh` (which mounts `wave0.sh` beside `bootstrap.sh`), or as the plain `submit` line it prints. That is the exact command, with `RCP_JOBS_CLI=echo` showing the overrides:

```bash
export RCP_STAGE_PREFIX=gs://YOUR-BUCKET/stage
packages/rcp-test/src/rcp_test/jobs/submit.sh --script wave0 --priority high \
  "$RCP_STAGE_PREFIX/rc0" gs://YOUR-BUCKET/waves wave0
```

which prints, in dry run, the submission the operator runs. The token file is mounted into the job and read there by the wrapper. Its value never reaches a process argv, an echo or a log:

```bash
# the from_file paths are absolute where submit.sh runs (its own checkout); shown shortened here
jobs submit -f "$RCP_JOBS_CONFIG" \
  app=rcp-wave0 priority_class=high worker.shared_memory=128Gi \
  env.RCP_IMAGE=vllm/vllm-openai:v0.33.0 \
  env.RCP_IMAGE_DIGEST=sha256:0123...abcd \
  worker.command='/bin/bash /etc/rcp/files/hftoken/hf_token_env.sh /bin/bash /etc/rcp/files/wave0/wave0.sh '"$RCP_STAGE_PREFIX"'/rc0 gs://YOUR-BUCKET/waves/wave0' \
  files.wave0.from_file=packages/rcp-test/src/rcp_test/jobs/wave0.sh \
  files.wave0.mount_path=/etc/rcp/files/wave0/wave0.sh \
  files.wave0host.from_file=packages/rcp-test/src/rcp_test/jobs/wave0_host.py \
  files.wave0host.mount_path=/etc/rcp/files/wave0host/wave0_host.py \
  files.gcshelper.from_file=packages/rcp-test/src/rcp_test/jobs/gcs.sh \
  files.gcshelper.mount_path=/etc/rcp/files/gcshelper/gcs.sh \
  files.gcspy.from_file=packages/rcp-test/src/rcp_test/jobs/gcs.py \
  files.gcspy.mount_path=/etc/rcp/files/gcshelper/gcs.py \
  files.bootstrap.from_file=packages/rcp-test/src/rcp_test/jobs/bootstrap.sh \
  files.bootstrap.mount_path=/etc/rcp/files/bootstrap/bootstrap.sh \
  files.report.from_file=packages/rcp-test/src/rcp_test/jobs/report.py \
  files.report.mount_path=/etc/rcp/files/report/report.py \
  files.refdeps.from_file=packages/rcp-test/src/rcp_test/jobs/reference_deps.py \
  files.refdeps.mount_path=/etc/rcp/files/refdeps/reference_deps.py \
  files.gcsauth.from_file="$RCP_GCS_AUTH_FILE" files.gcsauth.mount_path=/etc/rcp/gcs_auth.sh \
  files.hftoken.from_file="$RCP_HF_TOKEN_FILE" files.hftoken.mount_path=/etc/rcp/hf_token \
  files.hftokenenv.from_file="$RCP_SUBMIT_DIR/hf_token_env.sh" \
  files.hftokenenv.mount_path=/etc/rcp/files/hftoken/hf_token_env.sh
```

`WAVE0_DRY=1` prints the plan without running anything. The knobs (`WAVE0_MODEL`, `WAVE0_REVISION`,
`WAVE0_SECOND_MODEL`, `WAVE0_BUDGET`, `WAVE0_PORT_BASE`, `WAVE0_STARTUP_TIMEOUT_S`, the minimums) are
environment variables with the researched defaults. The report is `rcp-ndcg.wave0-report.v1`, its
schema exported at `packages/rcp-test/schema/wave0-report.schema.json`.

## The T4 scenario wave (`--script e2e`)

The end-to-end run scenarios of the GPU validation run as one job per wave through `e2e.sh`. A scenario is a full `rcp run` with `serve:` by role. Its phases run in the pod, with engines and coordinators under one supervision block per phase. The script builds the three environments and drives `python -m rcp_test.e2e`.
The wave list names **scenario ids** (or paths), one per line:

```bash
packages/rcp-test/src/rcp_test/jobs/submit.sh --script e2e --priority high \
  "$RCP_STAGE_PREFIX/rc0" gs://YOUR-BUCKET/waves e2e
```

The four shipped scenarios are in `packages/rcp-test/scenarios/`, with schema `schema/scenario.schema.json`. `text-four-phases` is a NanoBEIR subset: served encoder -> served reranker -> served judge -> calibrate + evaluate. It runs twice and is compared on identities and outputs. `outage` kills the judge engine mid-tournament: the run parks and recovers once. The `wait_on_outage_s` expiry fails with `BackendUnavailableError`, and a resume finishes the run. `identity` runs the same run directory again on new engine ports: nothing recomputes. `vidore` is a ViDoRe v3 subset with page images. The judge candidates and their pinned commits live in the scenarios. Each has the `fallback` that its T0 smoke concedes to: the Flash-Next NVFP4 candidate falls back to its FP8 release. The wave's outputs (`E2E.md`, `status.json`, the run directories) upload to
the wave's `OUT_URI`.

The stage must carry the scenario YAMLs under `scenarios/` beside `recipes/`. Until `rc_build.sh`
stages them with the recipe data, copy `packages/rcp-test/scenarios/*.yaml` into the stage.
`E2E_DRY=1 bash e2e.sh ...` prints the plan that the node would run.

## The judge-check job (`--script judgecheck`)

The release checklist's judge item runs as one job through `judgecheck.sh`. The item is `rcp judge check`
against each judge recipe's served engine, plus the engine's media token count. The script builds the
node's client environment with `bootstrap.sh envs` and then serves the judges **packed onto the node's GPUs**.
A judge's slice is the `resources.gpus` that its recipe declares, the value that renders the engine's `--tensor-parallel-size`. The job takes it first-fit in the list's order. Thus as many judges serve at the same time as the devices hold. Two never share a device; the big judges keep the GPUs they need. A judge that declares
more GPUs than the node has is refused before anything is served.

```bash
packages/rcp-test/src/rcp_test/jobs/submit.sh --script judgecheck --priority high \
  "$RCP_STAGE_PREFIX/rc0" gs://YOUR-BUCKET/waves judgecheck
```

Per judge, the job serves the recipe's own argv (`rcp-vllm serve <id> --port <N>`) with that judge's own
`CUDA_VISIBLE_DEVICES`, `VLLM_PORT` and `TMPDIR`. It waits for `GET /v1/models` within
`RCP_JUDGECHECK_STARTUP_S` (3600 s; a dead engine fails fast). It runs `rcp judge check --judge <id>
--judge-model <id> --set judge.base_url=http://127.0.0.1:<N>/v1 --json` for each **documented** thinking mode
of the recipe. `jobs/judgecheck_probe.py` owns the table. For the Gemma 4 judges, it merges the `chat_template_kwargs.enable_thinking` choice over the recipe's own `extra_body`. The other recipes document none, so their single documented mode is the shipped client block. It runs the engine media probe: one prepared probe image beside its no-media baseline under the processor family's stock budget. The engine's media delta is checked against the client's counted tokens. It stops the engine's process group. It records the verdicts, the engine version and the NVFP4/Marlin backend lines from the engine log. Then it moves on. A failed judge is
recorded, never fatal. A judge's record is `verified` only when every documented mode passed **and** its engine media check passed. A text-only judge's check is not applicable. A media mismatch, refusal or `not_checked`
fails the judge exactly like a failed mode. The outputs are `<OUT_PREFIX>/<wave>/<recipe>/<mode>.json`, the
`<recipe>/media.json` probes, and `judgecheck-report.json` beside them. Its `plan` section records the packing: the devices, and how many judges serve at once. It also records each judge's slice, port, `VLLM_PORT` and `TMPDIR`. The job
exits 1 when any judge failed. `JUDGECHECK_DRY=1 bash judgecheck.sh ...` prints the plan.

The knobs: `RCP_JUDGECHECK_GPUS` (the node's device indices, a comma list such as `0,1,2,3`). Also set `RCP_JUDGECHECK_PORT_BASE` (8100; judge *i* serves on base + *i*). Also set `RCP_JUDGECHECK_VLLM_PORT_BASE` (9200; judge *i*'s internal `VLLM_PORT` is base + *i*). The other knobs are `RCP_JUDGECHECK_STARTUP_S` (3600) and `RCP_JUDGECHECK_CHECK_S` / `RCP_JUDGECHECK_MEDIA_S` (900; the per-command budgets). When RCP_JUDGECHECK_GPUS is unset, the node's own devices come from `nvidia-smi --list-gpus`. A node with neither is refused, rather than served with every device visible to every engine.

## The topk probe job (`--script topk-probe`)

The job isolates one recipe family's engine-vs-reference divergence, experiment by experiment on a node: the
`topk-embed-v1` image-document gap. The job mounts `topk_probe.sh`, its `topk_probe.py` module and the wave's
per-row cosine fixture beside `bootstrap.sh`. It builds the node's client environment and the family's reference
venv with `bootstrap.sh envs`. It checks the image's vLLM API (`topk_probe.py --check-engine`: seconds, no weights). A moved engine API stops the job before the model loads. It runs the probe on the engine image's own
python. It uploads `probe-report.json`, `probe-report.md` and the per-row dumps to
`<OUT_PREFIX>/<wave>/topk-probe/`.

```bash
packages/rcp-test/src/rcp_test/jobs/submit.sh --script topk-probe --priority high \
  "$RCP_STAGE_PREFIX/rc0" gs://YOUR-BUCKET/waves topk-probe
```

The probe's experiments start with **E0**, the control. It compares the probe's own engine vectors against a wave's recorded vectors for one row. The file is passed with `RCP_TOPK_CONTROL_VECTORS`. With none the report says `not_run` and the numbers are uncalibrated, never a silent pass. **E1** isolates the preprocessing: the engine's pixels against the checkpoint's, every filter/dtype variant against both. **E2/E3** feed the same tensors into the engine's vision tower with forward hooks per layer. The first tensor whose max-abs-diff crosses `max(1e-2, 10 x the in-run bf16 floor)` names the layer. **E4** covers the image-token embeddings (the merger output). **E5** is the cross-feed: the reference tower on the engine's pixels and the engine tower on the reference's. **E6** is the grid sweep: each image document row's end-to-end per-token cosine beside the wave's recorded minimum. **E7** covers the text path layer by layer.

The knobs control the probe. `RCP_TOPK_VARIANT` is the variant whose recipe and checkpoint the probe uses (default small). `RCP_TOPK_ROWS` names the rows (empty: every image document row). `RCP_TOPK_DEVICE` is the engine (cuda:0). `RCP_TOPK_REFERENCE_DEVICE` is the reference subprocess (cuda:1). `RCP_TOPK_GPU_FRACTION` is the engine's `gpu_memory_utilization` (0.45). `RCP_TOPK_CONTROL_VECTORS` is the control file (unset: the control is `not_run`). `RCP_TOPK_MODEL` / `RCP_TOPK_REVISION` are the checkpoint (unset: the staged recipe's own checkpoint pin). Each is forwarded into the job only when
the operator sets it. `TOPK_DRY=1 bash topk_probe.sh <stage> <out>` prints the plan the node would run.
