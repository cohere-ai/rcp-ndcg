# Contributing to RCP-nDCG: notes for coding agents

This file is for agents (and people) who change this repository. To use the package from another project, read
`skills/rcp-ndcg/` and `docs/` instead.

## Set up, check, test

```bash
uv sync --locked --extra dev                     # Python 3.12 or later
uv run ruff format --check . && uv run ruff check .
uv run basedpyright                              # src and the core package: zero errors, blocking in CI
uv run pytest tests/ -n 4                        # the whole suite, offline
uv run pytest tests/contract                     # the public surface: CLI tree, exit codes, schemas, __all__
uv run pytest tests/docs                         # Markdown links, navigation, and every documentation snippet
```

On Linux the lock resolves torch with its CUDA wheels. CI, like a CPU-only machine, skips them and installs the CPU
build instead, then runs every command with `uv run --no-sync` (see `.github/workflows/ci.yml`). CI needs no
secrets. Run the narrowest test first, then the whole suite before you finish.

## Layout and layering

- `packages/rcp-ndcg-core` (`rcp_ndcg_core`): the metric, the gains, the scoring protocols, the public records and
  the IRT estimators. numpy and pydantic only; torch is imported lazily inside `irt/` and nowhere else.
- `src/rcp_ndcg`: the pipeline and the CLI. Imports point inward only, in this order:
  `rcp_ndcg_core → support → storage → data → inference → retrieval → llm → calibration → eval → runners → runs → schemas | mcp → cli`.
  Eager imports have no cycles; `schemas` and `mcp` import the CLI's command table lazily, to describe and serve it.
  Job runners are loaded through the `rcp_ndcg.runners` entry-point group, the one plugin seam for job execution
  (adapters have their own: `rcp_ndcg.adapters`).
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
| Judging: the client, the schedules, the judgement store, cost estimates | `rcp_ndcg.llm` |
| Prompts (tournament, rubric, vision and video variants) | `src/rcp_ndcg/llm/prompts/`, loaded by name |
| Text, image and video preprocessing, caps and chunking | `rcp_ndcg.data.preprocess` (text), `rcp_ndcg.data.resolution` (image and video policies), `rcp_ndcg.data.prepare` (media sent to a judge) |
| Wire adapters, the transport, replicas, parking, provenance probe | `rcp_ndcg.inference` |
| Evaluation: scoring rankings, comparisons, explanations, MTEB tasks | `rcp_ndcg.eval` |
| Job execution (local, SLURM, Kubernetes, plugins) | `rcp_ndcg.runners` |
| Paths, cache and storage URIs | `rcp_ndcg.support`, `rcp_ndcg.storage` |
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

## Releasing

Push a tag `v<version>` whose version is that of both `pyproject.toml` files. `.github/workflows/release.yml` builds
`rcp-ndcg` and `rcp-ndcg-core`, checks their versions against the tag and `requirements-constraints.txt` against the
lock, publishes both to PyPI with trusted publishing, and attaches the constraints file to the GitHub release. When
`uv.lock` changes, regenerate the constraints file with the command in its header. One-time setup: on pypi.org, add
a trusted publisher to each project (a pending one before the first upload) with owner `cohere-ai`, repository
`rcp-ndcg`, workflow `release.yml` and environment `pypi`, and create the environment `pypi` in the repository's
settings. No secret is needed.

## Where to look

- The concepts and their formulas: `docs/concepts/`. The paper: https://arxiv.org/abs/2609.35739.
- The command line: `rcp-ndcg --help`, `rcp-ndcg schema show commands`, `docs/reference/cli.md`.
- The public surface as data: `tests/contract/snapshots/` and `schemas/`.
