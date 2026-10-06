# Release candidates and the GPU waves

How a release candidate is built, what the GPU node installs from it, and how wave 0 proves the node
before any recipe wave. The waves themselves are described in [the serving concepts](../concepts/serving.md)
and the [recipe guide](add-a-model.md); this page is the operator's procedure. Everything below uses
`gs://YOUR-BUCKET/rcp-ndcg` as the private stage location — a tracked file never names a real one, so
the operator passes the real prefix on the command line or as `RCP_STAGE_PREFIX`.

## Build a release candidate

`rc_build.sh` builds the three distributions with the same commands the release workflow runs, checks
them (versions against the checkout, the pins, the constraints file against the lock, twine), builds
the **wheelhouse** (every locked dependency beside the release wheels, the CPU torch build included),
smoke-installs from it in a fresh venv, stages everything, and writes a hash manifest:

```bash
export RCP_STAGE_PREFIX=gs://YOUR-BUCKET/stage            # private location; operator's command line only
export EXTRA_DIRS="/path/to/private-plugins /path/to/private-pairs"
packages/rcp-ndcg-vllm/jobs/rc_build.sh rc0                # or rc_build.sh rc0 <commit>
```

The build runs in a detached worktree of the given commit, so a dirty checkout is fine. The staged tree
under `<prefix>/rc0/` is what the node installs from:

| Path | Contents |
|---|---|
| `dist/` | the six release files, as `release.yml` builds them |
| `wheelhouse/` | the release wheels plus every locked dependency for the node's platform |
| `requirements-constraints.txt` | the lock's export — the install's constraints file |
| `recipes/` | the recipe directories (recipe.yaml, reference.py, templates) |
| `plugins/` | public plugin packages, when the package ships any |
| `wave-lists/<wave>.txt` | one recipe id per line, per wave |
| `pairs/` | the stage-2 pairs files, when the checkout has any |
| `requirements-reference.txt` | what the reference venv installs from the wheelhouse (the image's torch stays) |
| `extra/<name>/` | the `EXTRA_DIRS` entries (private plugins, pairs, wave lists), as they are |
| `manifest.json` | the commit, the version, the CUDA-lock wheels inert on a CPU client (`nvidia-*`, `triton`), the SHA-256 of every staged file |

The wheelhouse is what makes a node install exact and independent of PyPI's state that night: the node
installs with `--find-links <wheelhouse> --no-index`, never asking an index (PyPI is only ever a
fallback). Building it downloads the locked dependencies once, on the operator's machine, from the CPU
torch index and PyPI only.

## The three environments on the node

Every wave runs on the stock `vllm/vllm-openai:v0.31.0` image — no custom image, no build — with three
environments that are never mixed. `bootstrap.sh` builds them from a staged RC:

- **engine** — the image's own Python, which runs `vllm serve`. Untouched, except recipe plugin wheels
  installed with `--no-deps`: a `pip freeze` before and after must differ by exactly those wheels (in
  wave 0, by nothing at all), or the bootstrap fails before any engine starts.
- **client** — no separate venv: every client command runs through the product's own install mechanism
  (`rcp_ndcg.runners.script.install_argv`: `uvx --find-links <wheelhouse> --no-index`, held to the
  staged constraints file), so the waves exercise exactly the code users run. `uv` itself is installed
  with `pip --target` (the product's own `bootstrap_uv` location), never into the engine environment.
- **reference** — a venv with `--system-site-packages` over the image's torch and CUDA, installing only
  what `requirements-reference.txt` names from the wheelhouse; the recipes' references run as
  subprocesses of that python, never inside the client.

The bootstrap verifies the staged files against the manifest before installing anything, records the
install times and the versions of all three environments, and fails if the installed versions differ
from the manifest. On the node it is mounted at `/etc/rcp/files/bootstrap/bootstrap.sh` (submitted by
`submit.sh`) and takes the same arguments a job passes it:

```bash
bootstrap.sh envs <STAGE_URI> --state <DIR>     # the three environments only (what wave 0 calls)
bootstrap.sh wave <STAGE_URI> <OUT_URI> --wave wave-a
```

## Submitting the waves

`submit.sh` submits one job per wave against a staged RC — at most `--max-jobs` in flight (a wave
beyond the cap is submitted with `depends_on` on the wave that many places ahead), with the wave's
Kueue priority class and the shared-memory size for eight engines:

```bash
export RCP_KJOBS_CONFIG=/path/to/jobs-config.yaml    # the job CLI's -f config (required, no default)
export RCP_GCS_AUTH_FILE=/path/to/gcs_auth.sh        # mounted at /etc/rcp/gcs_auth.sh; named, never read
export RCP_HF_TOKEN_FILE=/path/to/token              # passed as a kjobs secret, never read or echoed
packages/rcp-ndcg-vllm/jobs/submit.sh gs://YOUR-BUCKET/rc0 gs://YOUR-BUCKET/waves wave-a wave-b
```

Options: `--max-jobs N` (default 1), `--priority dev-high|dev-medium` (the `priority_class=` override,
rendered as `<class>-training-priority`; verify with a dry run), `--script bootstrap|wave0`, and
`KJOBS=echo` to print the plan instead of submitting. The job CLI's output goes to a file; only job
names and states are printed.

## Wave 0, the node test

Before any recipe wave, wave 0 runs **as one job on one node** from a development build staged like an
RC. It fails fast with a one-line reason at the first failure and writes one JSON report (to
`<OUT_URI>/wave0-report.json`, to `<RC_STAGE_URI>/reports/`, and to the job's stdout):

1. **preflight** — the node's assumptions, checked not assumed: `nvidia-smi` and `python3` on `PATH`,
   at least two GPUs, `/dev/shm` at least 8 GiB (the submission sizes it), at least 50 GiB free (the
   weights land on the container filesystem), the auth script and the HF token present.
2. **host** — the image (its digest as passed in `RCP_IMAGE_DIGEST`, since a container cannot see its
   own), the driver, the GPUs, free disk, `/dev/shm`, the python versions.
3. **bootstrap** — the three environments, and the engine's `pip freeze` unchanged (no plugin in
   wave 0).
4. **reach** — the Hub with the token secret (a metadata call; the reply's commit must be the pinned
   revision) and a `gs://` write/list/read/delete round-trip through `rcp_ndcg.storage` from the client
   environment. PyPI is not required.
5. **engines** — `vllm serve Qwen/Qwen3-Embedding-0.6B@97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` on one
   slot and a small second model on another, **at the same time**, each slot with its own
   `CUDA_VISIBLE_DEVICES`, HTTP port, `VLLM_PORT`, `TMPDIR` and log directory: both must answer their
   own model names, or the slots collide.
6. **embed** — from the client environment, the product's `fit()` under an explicit budget and the
   product's embedding client over 20 texts (5 of them over the budget, cut by the product); the
   engine's `/tokenize` of every rendered input must equal the product's token ids (the engine is the
   tokenization truth), recorded per input.
7. **evict** — the models out of the HF cache, the free disk before and after (the engine may still
   hold the weights mapped, so the stop step re-measures).
8. **stop** — both engines' process groups stopped, then no engine process left on the node.

Run it through `submit.sh` (which mounts `wave0.sh` beside `bootstrap.sh`), or as the plain
`kjobs-go submit` line it prints — the exact command, with `KJOBS=echo` showing the overrides:

```bash
export RCP_STAGE_PREFIX=gs://YOUR-BUCKET/stage
packages/rcp-ndcg-vllm/jobs/submit.sh --script wave0 --priority dev-high \
  "$RCP_STAGE_PREFIX/rc0" gs://YOUR-BUCKET/waves wave0
```

which prints, in dry run, the submission the operator runs (`secret.HF_TOKEN` is expanded from
`RCP_HF_TOKEN_FILE` by command substitution inside the script, never printed):

```bash
# the from_file paths are absolute where submit.sh runs (its own checkout); shown shortened here
kjobs-go submit -f "$RCP_KJOBS_CONFIG" \
  app=rcp-wave0 priority_class=dev-high worker.shared_memory=128Gi \
  worker.command='/bin/bash /etc/rcp/files/wave0/wave0.sh '"$RCP_STAGE_PREFIX"'/rc0 gs://YOUR-BUCKET/waves/wave0' \
  files.wave0.from_file=packages/rcp-ndcg-vllm/src/rcp_ndcg_vllm/jobs/wave0.sh \
  files.wave0.mount_path=/etc/rcp/files/wave0/wave0.sh \
  files.wave0host.from_file=packages/rcp-ndcg-vllm/jobs/wave0_host.py \
  files.wave0host.mount_path=/etc/rcp/files/wave0host/wave0_host.py \
  files.bootstrap.from_file=packages/rcp-ndcg-vllm/jobs/bootstrap.sh \
  files.bootstrap.mount_path=/etc/rcp/files/bootstrap/bootstrap.sh \
  files.report.from_file=packages/rcp-ndcg-vllm/jobs/report.py \
  files.report.mount_path=/etc/rcp/files/report/report.py \
  files.gcsauth.from_file="$RCP_GCS_AUTH_FILE" files.gcsauth.mount_path=/etc/rcp/gcs_auth.sh \
  secret.HF_TOKEN="$(cat "$RCP_HF_TOKEN_FILE")"
```

`WAVE0_DRY=1` prints the plan without running anything. The knobs (`WAVE0_MODEL`, `WAVE0_REVISION`,
`WAVE0_SECOND_MODEL`, `WAVE0_BUDGET`, `WAVE0_PORT_BASE`, `WAVE0_STARTUP_TIMEOUT_S`, the minimums) are
environment variables with the researched defaults; the report is `rcp-ndcg.wave0-report.v1`, its
schema exported at `packages/rcp-ndcg-vllm/schema/wave0-report.schema.json`.
