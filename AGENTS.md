# Contributing to RCP-nDCG: notes for coding agents

This file is for agents (and people) who change this repository. To use the package from another project, read
`skills/rcp/` and `docs/` instead.

## Set up, check, test

```bash
uv sync --locked --extra dev                     # Python 3.12 or later
uv run ruff format --check . && uv run ruff check .
uv run basedpyright                              # the four src trees: zero errors, blocking in CI
uv run pytest tests/ -n 4                        # the root tree: the invariants, the contract and docs suites
uv run pytest packages/rcp/tests packages/rcp-core/tests -n 4   # the product's and the core's suites
uv run pytest packages/rcp-test/tests -n 4       # the harness's suite, in its own process
uv run pytest packages/rcp-vllm/tests -q         # the serving package's suite, in its own process
uv run pytest tests/contract                     # the public surface: CLI tree, exit codes, schemas, __all__
uv run pytest tests/docs                         # Markdown links, navigation, and every documentation snippet
```

The four suite invocations are one per tree: `packages/rcp-test/tests` and `packages/rcp-vllm/tests` are
never collected with the root tree (both trees are named `tests`), so no single `pytest tests/` runs the
repository. CI runs the first two as one invocation (`pytest tests/ packages/rcp/tests packages/rcp-core/tests`).

On Linux the lock resolves torch with its CUDA wheels. CI, like a CPU-only machine, skips them and installs the CPU
build instead, then runs every command with `uv run --no-sync` (see `.github/workflows/ci.yml`). CI needs no
secrets. Run the narrowest test first, then the four suite invocations above before you finish. The tests that gate on `[data]`,
`[mteb]` or the `mcp` SDK (the pdf and hf readers, the transformers parity check, the MCP SDK round trip) skip in a
bare `dev` environment; CI's `gated` job installs their extras and runs the whole tree with every gate open.

`uv run pytest tests/conformance` is not a root command: the conformance suite lives in `rcp-test` and
runs per that package's README.

## Layout and layering

- `rcp-core` (`rcp_core`): the metric, the gains, the scoring protocols, the public records and
  the IRT estimators. numpy and pydantic only; torch is imported lazily inside `irt/` and nowhere else.
- `packages/rcp/src/rcp`: the pipeline and the CLI. Imports point inward only, in this order:
   `rcp_core → support → storage → data → inference → retrieval → judging → calibration → eval → runners → runs → results → schemas | mcp → cli`.
  Eager imports have no cycles; `schemas` and `mcp` import the CLI's command table lazily, to describe and serve it.
  The product reads eight entry-point groups, one per extension seam: `rcp.runners` (job execution),
  `rcp.adapters` (wire adapters, per role), `rcp.readers` and `rcp.writers` (dataset formats),
  `rcp.results` (result sinks), `rcp.fake_transports` (the transport providers behind a `fake://`
  URL), `rcp.metrics` (the scoring metrics) and `rcp.events` (the run-event subscribers).
  `tests/contract/snapshots/packaging.json` records the list as data and is the authority; the test
  package's own `rcp.emulators` group is read by `rcp_test`, never by the product.
- `experiments/`: paper reproduction from public data. It imports the package; the package never imports it.
- `examples/`: runnable examples (`tests/docs` runs them). `skills/rcp/`: the skill for agents that use the
  package. `schemas/`: exported JSON Schemas, generated and committed.

## One home per concept

Before adding a helper, `git grep` for an existing one. A second implementation of any of these blocks a merge:

| Concept | Home |
|---|---|
| nDCG, tie rules, Count-nDCG, the paper's scoring protocols | `rcp_core.metric`, `rcp_core.protocol` |
| The gain `g(theta)`, pass probabilities, item and query parameters | `rcp_core.gain`, `rcp_core.schemas` |
| Calibration (with or without the tournament, pooled judges), scoring and insertion of documents | `rcp.calibration` over `rcp_core.irt` |
| Judging: the client, the schedules, the judgement store, cost estimates | `rcp.judging` |
| Prompts (tournament, rubric, vision and video variants) | `packages/rcp/src/rcp/judging/prompts/`, loaded by name |
| Text, image and video preprocessing, caps and chunking | `rcp.data.text_policy` (the text policy, chunking, `Preprocessing`), `rcp.data.resolution` (image and video policies), `rcp.data.prepare` (media sent to a judge), `rcp.data.templates` |
| The geometry a checkpoint's processor recomputes (the smart resize, the fixed point) | `rcp.data.resolution`. Declared exception: `packages/rcp-vllm/src/rcp_vllm/recipes/*/reference.py` are dependency-free reference subprocesses that must not import the product (no reference's `reference.in` pins a workspace distribution, so none can), so a family's reference mirrors the product's rule where the engine's own processor is the thing being checked; the mirror is a recorded copy, validated against the product on GPU (the equivalence waves) |
| Judge/role text budgets, templates and their cut policy | `rcp.data.text_budget` (the fit), `rcp.data.census` (the cut record and census), `rcp.storage.census` (the census files' record I/O), `rcp.data.templates` |
| What a document reads as (MTEB's title join, `title: separate`) and the two query instructions (the task instruction's placement, the per-query append) | `rcp_core.records` (`Document.model_content`, `Query.format_query`/`format_content`) -- the rule once; the role clients' `normalise` stage applies it, and the sparse (BM25) path has its own join (`retrieval/_api._sparse_corpus`, mteb's BM25) |
| Postprocess of model output (L2 normalisation, chunk-score aggregation, the late-interaction skip ids) | `rcp.data.postprocess` |
| The MRL head (the truncation cut, the learned projection, the declared set/range) | `rcp.data.mrl` |
| The role clients' one preparation pipeline (stage order, per-row records) | `rcp.inference.clients._base` (`STAGES`, the runners) |
| Serving recipes, the recipe schema, `serve` | `rcp_vllm` |
| Model plugins for served checkpoints | `rcp_vllm.models/` |
| Reference cases, conformance, model-level fakes, equivalence/recording/GPU job tooling | `rcp_test` |
| Wire adapters, the transport, replicas, parking, provenance probe | `rcp.inference` |
| Evaluation: scoring rankings, comparisons, explanations, MTEB tasks | `rcp.eval` |
| Job execution (local, SLURM, Kubernetes, plugins) | `rcp.runners` |
| Paths, cache and storage URIs | `rcp.support`, `rcp.storage` |
| Entry-point plugin lookup (the names in a group, the one duplicate-name policy, the load check) | `rcp.support.entrypoints`, used by the readers/writers, results, runners and metrics registries and by the run-event subscribers |
| Identity and hashing | `rcp.support.identity` |
| Errors and exit codes | `rcp.errors` |

All click code lives in `rcp.cli`, as thin adapters over library functions. Library code never prints; it
returns typed results and raises typed errors from `rcp.errors`.

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
  `RCP_NDCG_NETWORK_TESTS=1` is set (the marker alone skips nothing). Use `rcp.testing.FakeJudge` and small
  in-test data instead of a live judge.
- **Tests write only to `tmp_path`**, never into the checkout.
- **Nothing is cut or defaulted silently.** A truncation is declared policy and recorded; a missing value is an
  error with a hint, not a default that changes numbers.
- **Every public module declares `__all__`**, and every public function has a docstring that states its inputs,
  outputs and units (logits, gain in [0, 1], `_s`, `_chars`, `_tokens`).
- **The public surface is frozen.** `PUBLIC_MODULES` in `rcp.support.api_docs` is the list, the
  snapshot is the definition of public, and every public name is documented -- `rcp docs api --out
  docs/reference/api` generates the per-module pages the contract suite keeps current. The rules, the
  deprecation path and the update flow are on
  [compatibility and versioning](docs/reference/versioning.md#the-001-freeze).
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
- **Every recipe ships validated.** A recipe merges only with its GPU waves green (one release with everything);
  `status: unverified` does not ship in a tag, and a `failed` variant ships only when the owner's explicit
  disposition list names it (for 0.0.1, exactly the two `topk-embed-v1` variants).
- **No GPU pytests.** GPU work produces observation corpora per `OBSERVATIONS-SPEC`; verified fake engines on
  CPU with conformance + golden replays stand in for tests. Model-level fakes live in `rcp-test`; the
  generic `fake://` stays in the product.
- **Versions move together.** All distributions carry the tag version; `rcp` pins
  `rcp-core==<version>`; `rcp-test` is never published.

## Releasing

Push a tag `v<version>` whose version is that of all four `pyproject.toml` files (`packages/rcp/`,
`packages/rcp-core/`, `packages/rcp-vllm/` and the unpublished `packages/rcp-test/`; the root manifest is the uv workspace
only). `.github/workflows/release.yml` builds the three published distributions (one `--package` per member of
the four-member workspace; `rcp-test` is never built), checks each version against the
tag, that `rcp` pins `rcp-core==<version>`, and `requirements-constraints.txt` against the lock (the
pins, semantically -- `.github/scripts/check_constraints.py`), and runs
`twine check` on every file. Each package publishes to PyPI with trusted publishing through its own GitHub environment
(one publish job per package, below), because PyPI identifies a pending trusted publisher by owner, repository,
workflow file and environment only, not the project name; `publish-rcp` waits for `publish-core`, which it pins
exactly, and `publish-vllm` waits for both (it names no sibling: the lean package pins no lockstep
version): the publish order is `core` -> `rcp` ->
`vllm`. The GitHub release attaches the constraints
file. When `uv.lock` changes, regenerate the constraints file with `python .github/scripts/check_constraints.py
--write` (the export command is in its header). One-time setup: on pypi.org, add a trusted publisher to each project (a
pending one before the first upload; the `rcp-core` project is new after the rename, so it needs its pending
publisher before the first `publish-core` run) with owner `cohere-ai`, repository `rcp`, workflow `release.yml` and the
environment from the table, and create each environment in the repository's settings. No secret is needed.

The layout move added three release gates on top of `twine check`, each also run by CI's own jobs: a
**fresh-venv wheel install** of every published distribution (the wheel, not the checkout, must import and
work); the **`--no-deps` freeze check** for `rcp-vllm` in a venv that has only pydantic and PyYAML
(`pip freeze` before and after differs by exactly that wheel); and `rcp-vllm serve <id> --dry-run` for
every recipe (the serve argv renders for every variant). The node-side gates (the wheelhouse, the reference
environments, the GPU waves) are in `docs/how-to/release-candidates.md`.

| PyPI project | GitHub environment |
|---|---|
| `rcp` | `pypi` |
| `rcp-core` | `pypi-core` |
| `rcp-vllm` | `pypi-vllm` |

## Where to look

- The concepts and their formulas: `docs/concepts/`. The paper: https://arxiv.org/abs/2609.35739.
- The shipped serving recipes and their catalog: the `rcp_vllm` package's recipe data, described in
  [recipes and serving models](docs/reference/recipes.md).
- The command line: `rcp --help`, `rcp schema show commands`, `docs/reference/cli.md`.
- The public surface as data: `tests/contract/snapshots/` and `schemas/`, and the rules around it in
  [compatibility and versioning](docs/reference/versioning.md).
