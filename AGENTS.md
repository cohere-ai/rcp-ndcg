# Contributing to RCP-nDCG: notes for coding agents

This file is for agents (and people) who change this repository. To use the package from another project, read
`skills/rcp-ndcg/` and `docs/` instead.

## Set up, check, test

```bash
uv sync --locked --extra dev                     # Python 3.12 or later
uv run ruff format --check . && uv run ruff check .
uv run basedpyright                              # the four src trees: zero errors, blocking in CI
uv run pytest tests/ -n 4                        # the whole suite, offline
uv run pytest tests/contract                     # the public surface: CLI tree, exit codes, schemas, __all__
uv run pytest tests/docs                         # Markdown links, navigation, and every documentation snippet
```

On Linux the lock resolves torch with its CUDA wheels. CI, like a CPU-only machine, skips them and installs the CPU
build instead, then runs every command with `uv run --no-sync` (see `.github/workflows/ci.yml`). CI needs no
secrets. Run the narrowest test first, then the whole suite before you finish. The tests that gate on `[data]`,
`[mteb]` or the `mcp` SDK (the pdf and hf readers, the transformers parity check, the MCP SDK round trip) skip in a
bare `dev` environment; CI's `gated` job installs their extras and runs the whole tree with every gate open.

`uv run pytest tests/conformance` is not a root command: the conformance suite lives in `rcp-ndcg-test` and
runs per that package's README.

## Layout and layering

- `rcp-ndcg-core` (`rcp_ndcg_core`): the metric, the gains, the scoring protocols, the public records and
  the IRT estimators. numpy and pydantic only; torch is imported lazily inside `irt/` and nowhere else.
- `rcp-ndcg/src/rcp_ndcg`: the pipeline and the CLI. Imports point inward only, in this order:
   `rcp_ndcg_core → support → storage → data → inference → retrieval → judging → calibration → eval → runners → runs → results → schemas | mcp → cli`.
  Eager imports have no cycles; `schemas` and `mcp` import the CLI's command table lazily, to describe and serve it.
  The product reads six entry-point groups, one per extension seam: `rcp_ndcg.runners` (job execution),
  `rcp_ndcg.adapters` (wire adapters, per role), `rcp_ndcg.readers` and `rcp_ndcg.writers` (dataset formats),
  `rcp_ndcg.results` (result sinks) and `rcp_ndcg.fake_transports` (the transport providers behind a `fake://`
  URL). `tests/contract/snapshots/packaging.json` records the list as data and is the authority; the test
  package's own `rcp_ndcg.emulators` group is read by `rcp_ndcg_test`, never by the product.
- `experiments/`: paper reproduction from public data. It imports the package; the package never imports it.
- `examples/`: runnable examples (`tests/docs` runs them). `skills/rcp-ndcg/`: the skill for agents that use the
  package. `schemas/`: exported JSON Schemas, generated and committed.

## One home per concept

Before adding a helper, `git grep` for an existing one. A second implementation of any of these blocks a merge:

| Concept | Home |
|---|---|
| nDCG, tie rules, Count-nDCG, the paper's scoring protocols | `rcp_ndcg_core.metric`, `rcp_ndcg_core.protocol` |
| The gain `g(theta)`, pass probabilities, item and query parameters | `rcp_ndcg_core.gain`, `rcp_ndcg_core.schemas` |
| Calibration (with or without the tournament, pooled judges), scoring and insertion of documents | `rcp_ndcg.calibration` over `rcp_ndcg_core.irt` |
| Judging: the client, the schedules, the judgement store, cost estimates | `rcp_ndcg.judging` |
| Prompts (tournament, rubric, vision and video variants) | `rcp-ndcg/src/rcp_ndcg/judging/prompts/`, loaded by name |
| Text, image and video preprocessing, caps and chunking | `rcp_ndcg.data.text_policy` (the text policy, chunking, `Preprocessing`), `rcp_ndcg.data.resolution` (image and video policies), `rcp_ndcg.data.prepare` (media sent to a judge), `rcp_ndcg.data.templates` |
| The geometry a checkpoint's processor recomputes (the smart resize, the fixed point) | `rcp_ndcg.data.resolution`. Declared exception: `rcp_ndcg_vllm/recipes/*/reference.py` are dependency-free reference subprocesses that must not import the product, so a family's reference mirrors the product's rule where the engine's own processor is the thing being checked; the mirror is a recorded copy, validated against the product on GPU (the equivalence waves) |
| Judge/role text budgets, templates and their cut policy | `rcp_ndcg.data.text_budget` (the fit), `rcp_ndcg.data.census` (the cut record and census), `rcp_ndcg.storage.census` (the census files' record I/O), `rcp_ndcg.data.templates` |
| What a document reads as (MTEB's title join, `title: separate`) and the two query instructions (the task instruction's placement, the per-query append) | `rcp_ndcg_core.records` (`Document.model_content`, `Query.format_query`/`format_content`) -- the rule once; the role clients' `normalise` stage applies it, and the sparse (BM25) path has its own join (`retrieval/_api._sparse_corpus`, mteb's BM25) |
| Postprocess of model output (L2 normalisation, chunk-score aggregation, the late-interaction skip ids) | `rcp_ndcg.data.postprocess` |
| The MRL head (the truncation cut, the learned projection, the declared set/range) | `rcp_ndcg.data.mrl` |
| The role clients' one preparation pipeline (stage order, per-row records) | `rcp_ndcg.inference.clients._base` (`STAGES`, the runners) |
| Serving recipes, the recipe schema, `serve` | `rcp_ndcg_vllm` |
| Model plugins for served checkpoints | `rcp_ndcg_vllm.models/` |
| Reference cases, conformance, model-level fakes, equivalence/recording/GPU job tooling | `rcp_ndcg_test` |
| Wire adapters, the transport, replicas, parking, provenance probe | `rcp_ndcg.inference` |
| Evaluation: scoring rankings, comparisons, explanations, MTEB tasks | `rcp_ndcg.eval` |
| Job execution (local, SLURM, Kubernetes, plugins) | `rcp_ndcg.runners` |
| Paths, cache and storage URIs | `rcp_ndcg.support`, `rcp_ndcg.storage` |
| Entry-point plugin lookup (the names in a group, the one duplicate-name policy, the load check) | `rcp_ndcg.support.entrypoints`, used by the readers/writers, results and runners registries |
| Identity and hashing | `rcp_ndcg.support.identity` |
| Errors and exit codes | `rcp_ndcg.errors` |

All click code lives in `rcp_ndcg.cli`, as thin adapters over library functions. Library code never prints; it
returns typed results and raises typed errors from `rcp_ndcg.errors`.

## Rules

- **Stage B has exactly five criteria, C1 to C5.** The rubric prompt defines them. Do not add, remove or reword a
  criterion; a changed prompt is a new judgement family and never pools with the shipped one.
- **Every fix gets a failing test first.** Write the test, watch it fail on the unfixed code, then fix.
- **Every public change updates the snapshot and the CHANGELOG.** A change to a public name, a CLI command or flag,
  an exit code or a schema shows up in `tests/contract/snapshots/` or `schemas/`. Regenerate them with
  `uv run pytest tests/contract --update-snapshots`, review the diff, and add an entry to `CHANGELOG.md`. CI fails a
  pull request that changes either without a CHANGELOG entry.
- **Numerics do not move by accident.** The metric, the gain, the protocols, the schedules and the calibration fit
  reproduce the paper, and anchor tests pin them. A change to their numbers is deliberate, tested, and in the
  CHANGELOG.
- **No network in tests**, except tests marked `@pytest.mark.network` that also skip themselves unless
  `RCP_NDCG_NETWORK_TESTS=1` is set (the marker alone skips nothing). Use `rcp_ndcg.testing.FakeJudge` and small
  in-test data instead of a live judge.
- **Tests write only to `tmp_path`**, never into the checkout.
- **Nothing is cut or defaulted silently.** A truncation is declared policy and recorded; a missing value is an
  error with a hint, not a default that changes numbers.
- **Every public module declares `__all__`**, and every public function has a docstring that states its inputs,
  outputs and units (logits, gain in [0, 1], `_s`, `_chars`, `_tokens`).
- **Docs describe current behaviour.** When behaviour changes, update the page in `docs/` that describes it, and
  keep every snippet runnable: `tests/docs` runs them. No numbers without a reproducible source.
- **Public names only.** No private infrastructure, hosts, buckets, people or unreleased models in code, configs,
  docs, tests or commit messages. Use placeholders such as `gs://YOUR-BUCKET/...` and `registry.example.com`.
- **Explicit budgets.** A self-hosted role config declares `tokenizer` + `max_tokens`; a hosted vendor profile
  without a tokenizer declares `budget_source: vendor` and sends content uncut. Over budget, the default is
  `cut`, recorded in the census; chunk aggregation is `max`; media are counted in tokens, never money.
- **Anchors are reserved, never engine-side cut.** Client-side cuts are anchor-preserving; the paper code's
  anchor drops are declared `known_deviations` in a recipe and compared under the cap only.
- **Recipe ids are the lowercased canonical Hub repo name** (never a redirecting short name), pinned by a test;
  a recipe's CHANGELOG bullets fold into the one release entry.
- **Every recipe ships validated.** A recipe merges only with its GPU waves green (one 0.0.1 with everything);
  `status: unverified` does not ship in a tag.
- **No GPU pytests.** GPU work produces observation corpora per `OBSERVATIONS-SPEC`; verified fake engines on
  CPU with conformance + golden replays stand in for tests. Model-level fakes live in `rcp-ndcg-test`; the
  generic `fake://` stays in the product.
- **Versions move together.** All distributions carry the tag version; `rcp-ndcg` pins
  `rcp-ndcg-core==<version>`; `rcp-ndcg-test` is never published.

## Releasing

Push a tag `v<version>` whose version is that of all four `pyproject.toml` files (`rcp-ndcg/`,
`rcp-ndcg-core/`, `rcp-ndcg-vllm/` and the unpublished `rcp-ndcg-test/`; the root manifest is the uv workspace
only). `.github/workflows/release.yml` builds the three published distributions (one `--package` per member of
the four-member workspace; `rcp-ndcg-test` is never built), checks each version against the
tag, that `rcp-ndcg` pins `rcp-ndcg-core==<version>`, and `requirements-constraints.txt` against the lock (the
pins, semantically -- `.github/scripts/check_constraints.py`), and runs
`twine check` on every file. Each package publishes to PyPI with trusted publishing through its own GitHub environment
(one publish job per package, below), because PyPI identifies a pending trusted publisher by owner, repository,
workflow file and environment only, not the project name; `publish-rcp-ndcg` waits for `publish-core`, which it pins
exactly, and `publish-vllm` waits for both (it names no sibling: the lean package pins no lockstep
version): the publish order is `core` -> `rcp-ndcg` ->
`vllm`. The GitHub release attaches the constraints
file. When `uv.lock` changes, regenerate the constraints file with `python .github/scripts/check_constraints.py
--write` (the export command is in its header). One-time setup (done): on pypi.org, add a trusted publisher to each
project (a
pending one before the first upload) with owner `cohere-ai`, repository `rcp-ndcg`, workflow `release.yml` and the
environment from the table, and create each environment in the repository's settings. No secret is needed.

The layout move added three release gates on top of `twine check`, each also run by CI's own jobs: a
**fresh-venv wheel install** of every published distribution (the wheel, not the checkout, must import and
work); the **`--no-deps` freeze check** for `rcp-ndcg-vllm` in a venv that has only pydantic and PyYAML
(`pip freeze` before and after differs by exactly that wheel); and `rcp-ndcg-vllm serve <id> --dry-run` for
every recipe (the serve argv renders for every variant). The node-side gates (the wheelhouse, the reference
environments, the GPU waves) are in `docs/how-to/release-candidates.md`.

| PyPI project | GitHub environment |
|---|---|
| `rcp-ndcg` | `pypi` |
| `rcp-ndcg-core` | `pypi-core` |
| `rcp-ndcg-vllm` | `pypi-vllm` |

## Where to look

- The concepts and their formulas: `docs/concepts/`. The paper: https://arxiv.org/abs/2609.35739.
- The shipped serving recipes and their catalog: the `rcp_ndcg_vllm` package's recipe data, described in
  [recipes and serving models](docs/reference/recipes.md).
- The command line: `rcp-ndcg --help`, `rcp-ndcg schema show commands`, `docs/reference/cli.md`.
- The public surface as data: `tests/contract/snapshots/` and `schemas/`, and the rules around it in
  [compatibility and versioning](docs/reference/versioning.md).
