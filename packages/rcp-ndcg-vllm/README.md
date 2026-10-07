# rcp-ndcg-vllm

Serving recipes, the equivalence harness, the engine recorder and the GPU wave runner for
[rcp-ndcg](https://github.com/cohere-ai/rcp-ndcg) models served with [vLLM](https://docs.vllm.ai).

This package is **outside the root uv workspace and lock** on purpose: it is installed into an engine image that
already carries vLLM, torch and transformers, and a torch pin here would fight every vLLM release. The harness
imports `rcp-ndcg` (the recipe's `client` block constructs the product's endpoint configs, stage 1 runs the
product's `fit`); only the engine is reached over HTTP, and the reference runs as a subprocess so the harness
process never imports torch.

Install it standalone with `pip install rcp-ndcg-vllm` (or, from a checkout of the repository,
`pip install packages/rcp-ndcg-vllm`), which pulls in the pinned `rcp-ndcg`; inside the engine image use
`pip install --no-deps rcp-ndcg-vllm` together with the already-installed product, which keeps the image's own
vLLM and torch.

## Layout

- `src/rcp_ndcg_vllm/recipe.py` — the recipe schema (frozen pydantic models, `extra="forbid"`), its validators,
  `load_recipe`, `iter_recipes`, `serve_argv` and `client_config`.
- `recipes/<id>/` — one directory per served model: `recipe.yaml`, an optional `template.jinja` chat template,
  `reference.py` (the subprocess reference) and an optional `requirements-reference.txt` (the reference's own environment, installed by the node's
  bootstrap instead of the package's shared one). The recipe lanes write these.
- `scenarios/<id>.yaml` — one T4 end-to-end run scenario each (the run config the product runs, the
  engines by role, the judge and its fallback, and the scenario's routine: `run`, `outage` or
  `identity`). Staged to the node beside `recipes/`; the wave list names the ids.
- `src/rcp_ndcg_vllm/equivalence/` — the three-stage equivalence check of the design's section on in-process
  scoring, as functions and a CLI (`python -m rcp_ndcg_vllm.equivalence`), driven through the product's role
  clients and audited on the captured wire.
- `src/rcp_ndcg_vllm/record.py` — records one fixed request/response exchange per engine route under
  `<out>/<engine>-<version>/<recipe-id>/`, the fixtures the engine adapters' contract tests replay.
- `src/rcp_ndcg_vllm/e2e.py` — the T4 end-to-end stage and its driver (`python -m rcp_ndcg_vllm.e2e`):
  materializes the product's run config from a scenario, renders its phased job script with the
  product's SLURM renderer (`container_runtime: none`) and runs it in the pod (see "T4 end to end"
  below).
- `src/rcp_ndcg_vllm/jobs/` — `run_wave.py` (packs recipes onto one node's GPUs, with per-slot isolation and
  the free-disk check and eviction), `wave0_probe.py` and `wave0_report.py` (wave 0's probes and the report
  schema), and `plugins.py` (the plugin wheels a wave's recipes install into the engine environment).
- `jobs/` — the node and operator scripts: `rc_build.sh` (build a release candidate exactly as `release.yml`
  does, stage it with the wheelhouse and a hash manifest), `bootstrap.sh` (the node's three environments),
  `submit.sh` (one job per wave, priority class, shared memory, the token as a secret; `--script
  bootstrap|wave0|e2e`), `gcs.sh` and
  `gcs.py` (the gs:// transfer: the CLIs when the image has one, else gcsfs into a tools directory
  outside the engine environment), and `wave0_host.py` and `report.py` (wave 0's stdlib helpers,
  mounted onto the node). `src/rcp_ndcg_vllm/jobs/e2e.sh` is the T4 scenario wave's entry, mounted like
  `wave0.sh`.
- `schema/recipe.schema.json`, `schema/scenario.schema.json` and `schema/wave0-report.schema.json` — the
  exported JSON Schemas of `Recipe`, `Scenario` and of the wave-0 report.

## T4 end to end: the run scenarios

The end-to-end scenarios of the GPU validation run as a harness stage inside the pod (never as pytest):
one scenario materializes the product's `RunConfig` (its validators decide), stages the run directory,
renders the run's **phased job script** with the product's SLURM renderer (`container_runtime: none`)
and executes that script in the pod. Around the product's rendering the driver adds only what the pod
lacks or the scenario needs — it never copies the product:

- **the client mechanism** (node-runtime items 1-2): every phase's coordinator command is wrapped with
  `rcp_ndcg.runners.script.install_argv` from the staged wheelhouse (`--find-links <wheelhouse>
  --no-index`, the staged constraints file), so the coordinator installs the release the users' way and
  never runs from the engine environment;
- **the srun stand-in**: the SLURM renderer starts each engine step under `srun`; a pod has no SLURM
  client, so the driver puts a tiny `srun` on `PATH` that runs the step's command in its own session
  and stops that session when it is stopped. The engine's `CUDA_VISIBLE_DEVICES`, `VLLM_PORT` and
  `TMPDIR` come from each scenario's `serve` block (node-runtime item 7);
- **the process-boundary probe** (node-runtime item 10): the phase workers put a one-file
  `sitecustomize` on `PYTHONPATH` that records each coordinator's `sys.prefix`, executable and
  versions, and the stage records the engine environment's `python3` beside it and asserts the two
  never name the same interpreter;
- **the T0 judge smoke**: each judge is booted, `GET /v1/models` and one chat completion must answer,
  and the winner serves the scenario (a candidate whose smoke fails concedes to the scenario's
  `fallback` — the Flash-Next NVFP4 candidate to its FP8 release);
- **the outage and identity controls**: the `outage` scenario's judge is a run-scoped engine the
  driver kills mid-tournament and restarts (the product's model: engines the
  supervision script started end the job when they exit; engines that live elsewhere are "restarted
  by whatever runs them, and the judge bounds the outage instead"), and `identity` re-runs an existing
  run directory on moved ports.

What two identical runs are compared on (`text-four-phases`, `pair: 2`): every step's identity and
identity hash, the input-deterministic steps' outputs byte for byte, and the judgement windows' counts
and families — never the judged values themselves, since judgements may differ at temperature > 0. The
stage writes `<out>/E2E.md` and `<out>/status.json` (schema `rcp-ndcg-vllm.e2e-report.v1`) beside the
run directories.

```bash
python -m rcp_ndcg_vllm.e2e --scenarios text-four-phases --scenarios-root scenarios \
  --recipes-root recipes --out /tmp/e2e --wheelhouse /stage/wheelhouse
```

On the node the entry is `src/rcp_ndcg_vllm/jobs/e2e.sh` (submitted by `jobs/submit.sh --script e2e`);
`E2E_DRY=1` prints its plan. Offline counterparts of every GPU check live in the root suite: the
rendered phased script as a golden file (this package's tests), the four-phase supervision re-run with
the fake engines as its engines (`tests/runners/test_supervision_replay.py`; its `fake_engines()` is the
one seam where the verified emulators of `rcp_ndcg.testing.engines` plug in when they land), and the
observed outage behaviour as a transport test (`tests/inference/test_outage_observed.py`).

## Validate a recipe on CPU (stage 1 only)

```bash
pip install rcp-ndcg-vllm            # pulls the pinned rcp-ndcg (stage 1 runs the product's fit)
python -m rcp_ndcg_vllm.equivalence --recipe recipes/<id> --pairs pairs.jsonl --out /tmp/equiv --stages 1
```

See `docs/how-to/add-a-model.md` in the repository for the guide, and the recipe schema's docstrings for every
field.
