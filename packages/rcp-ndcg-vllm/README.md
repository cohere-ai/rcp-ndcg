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
- `src/rcp_ndcg_vllm/equivalence/` — the three-stage equivalence check of the design's section on in-process
  scoring, as functions and a CLI (`python -m rcp_ndcg_vllm.equivalence`), driven through the product's role
  clients and audited on the captured wire.
- `src/rcp_ndcg_vllm/record.py` — records one fixed request/response exchange per engine route under
  `<out>/<engine>-<version>/<recipe-id>/`, the fixtures the engine adapters' contract tests replay; and
  `record_corpus`, the observation corpus's collector (raw-first `RECORD_SCHEMA` records, the request
  plan's rows twice in one process and once after an engine restart, batches 1/2/8/32, the reverse-order
  rerank variant, the protocol probes and `/tokenize` per input).
- `src/rcp_ndcg_vllm/observe/` — the observation lane: `requests.py` (the deterministic request
  generator: `GENERATOR_VERSION`, `SEED`, `PINNED_DATASET_COMMITS`, the synthetic adversarial set as
  text; it writes `pairs/<recipe>.jsonl` and `pairs/manifest.json`), `sources.py` (the suite catalogs),
  `corpus.py` (the corpus format, the behaviour fingerprint, the acceptance checks, the repository
  subset and `--changed-since`) and `controls.py` (the negative controls (a)-(f) as recipe variants;
  a passing control is a blocker).  The format is documented in `schema/observation-corpus.md`.
- `src/rcp_ndcg_vllm/quality.py` — the T3 quality stage: the task matrix as data, the served-path and
  `mteb` command lines, the served-vs-reference-vs-paper-vs-published comparison table with its
  deviation notes and `QUALITY.md`, and the golden-replay selection.
- `pairs/` — one stage-2 pairs file per recipe from the generator, plus the manifest (GENERATOR_VERSION,
  strata presence, per-row provenance, file hashes); `jobs/rc_build.sh` stages it (its one home).
- `src/rcp_ndcg_vllm/jobs/` — `run_wave.py` (packs recipes onto one node's GPUs, with per-slot isolation and
  the free-disk check and eviction; `--record-corpus` writes one observation corpus per recipe and
  `--changed-since <index>` re-records only the recipes whose behaviour fingerprint changed),
  `wave0_probe.py` and `wave0_report.py` (wave 0's probes and the report
  schema), and `plugins.py` (the plugin wheels a wave's recipes install into the engine environment).
- `jobs/` — the node and operator scripts: `rc_build.sh` (build a release candidate exactly as `release.yml`
  does, stage it with the wheelhouse and a hash manifest), `bootstrap.sh` (the node's three environments),
  `submit.sh` (one job per wave, priority class, shared memory, the token as a secret), `gcs.sh` and
  `gcs.py` (the gs:// transfer: the CLIs when the image has one, else gcsfs into a tools directory
  outside the engine environment), and `wave0_host.py` and `report.py` (wave 0's stdlib helpers,
  mounted onto the node).
- `schema/recipe.schema.json` and `schema/wave0-report.schema.json` — the exported JSON Schemas of `Recipe`
  and of the wave-0 report; `schema/observation-corpus.md` — the pairs and observation-corpus schema
  (shared with the fake-engines lane).

## Generate the pairs files and validate a recipe on CPU (stage 1 only)

```bash
pip install rcp-ndcg-vllm            # pulls the pinned rcp-ndcg (stage 1 runs the product's fit)
python -m rcp_ndcg_vllm.observe.requests --out pairs --reference-python $(which python)   # pairs + manifest
python -m rcp_ndcg_vllm.equivalence --recipe recipes/<id> --pairs pairs/<id>.jsonl --out /tmp/equiv --stages 1
```

See `docs/how-to/add-a-model.md` in the repository for the guide, and the recipe schema's docstrings for every
field.
