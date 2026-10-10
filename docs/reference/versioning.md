# Compatibility and versioning

What this release promises to keep stable, what may move in a `0.0.x` patch, and how the versioned artifacts
outside the package version (recipes, observation corpora, the run layout, the result record) carry their own
versions. The normative spelling of every name below lives in the repository, not on this page:
`tests/contract/snapshots/` and `schemas/` are the contract, and the tests in `tests/contract/` enforce it.

## What is public

- **The Python names** in the `__all__` of the public modules. `PUBLIC_MODULES` in `tests/contract/surface.py`
  lists them and `tests/contract/snapshots/python_api.json` pins every name: the facade `rcp_ndcg`; the core
  `rcp_ndcg_core` with `.gain`, `.irt`, `.metric`, `.protocol` and `.records`; `rcp_ndcg.calibration`,
  `.data`, `.data.preprocess`, `.data.revisions`, `.errors`, `.eval`, `.eval.mteb`, `.examples`, `.inference`,
  `.judging`, `.results`, `.retrieval`, `.runners`, `.runs` and `.testing`; and `rcp_ndcg_vllm.recipe` (the
  serving package's recipe module).
- **The command line**: the command tree, its flags, its output schemas and the exit codes
  (`tests/contract/snapshots/cli.json`, `exit_codes.json`; [the command line](cli.md)).
- **The MCP tools** `rcp-ndcg mcp serve` exposes (`tests/contract/snapshots/mcp_tools.json`).
- **The packaging**: distributions, extras and entry-point groups (`tests/contract/snapshots/packaging.json`).
- **The JSON Schemas** exported to `schemas/` (46 `*.v1.json` files in this release) and the
  `rcp-ndcg-vllm` recipe schemas (`tests/contract/snapshots/vllm_cli.json`).

Everything else is internal and may change without notice: `rcp_ndcg.cli`, `rcp_ndcg.storage`,
`rcp_ndcg.support`, `rcp_ndcg.data.io`, the submodules of `retrieval`, `judging`, `calibration`, `runs` and
`runners`, the rest of `rcp_ndcg_vllm`, and all of the unpublished `rcp-ndcg-test`. A change to a public name,
flag, exit code or schema is a reviewed diff against the snapshot and gets a CHANGELOG entry; a pull request
that changes either without one fails CI.

## The `0.0.x` rules

While the version is `0.x`, a change that breaks the public surface bumps the minor version and an additive
change bumps the patch version; from `1.0` on, semantic versioning applies.

A `0.0.x` patch release:

- may add a public name, a flag, a schema field (optional or defaulted), a sink or a runner, and may fix
  behaviour whose documented contract did not promise the old output;
- may change internal modules, messages, logs, performance and the wording of a hint;
- does not rename or remove a public name, change an exit code, change the meaning of a schema field, or change
  a shipped number's meaning. The metric, the gains, the scoring protocols, the schedules and the calibration
  fit reproduce the paper and are pinned by anchor tests: a change to their numbers is deliberate, tested and in
  the CHANGELOG ([REPRODUCIBILITY.md](https://github.com/cohere-ai/rcp-ndcg/blob/main/REPRODUCIBILITY.md)).

The four distributions carry the same version; `rcp-ndcg` pins `rcp-ndcg-core==<version>`; `rcp-ndcg-vllm`
carries the same version but pins no sibling; `rcp-ndcg-test` is never published.

### Deprecating a public name

A public name that must go is first **deprecated for one minor release**: it keeps working, warns where the
surface has a warning channel (a Python `DeprecationWarning`, a click `deprecated=True` command, a schema
description), and the CHANGELOG entry says what replaces it. It is removed in the next minor release, with its
own CHANGELOG entry. Nothing is removed silently, and an internal name needs no deprecation at all.

## Serving recipes: `schema_version`

The recipe file format is the versioned contract between `rcp-ndcg-vllm` (which ships the data) and `rcp-ndcg`
(which reads it): every family file carries `schema_version` (currently `"1"`), and `rcp_ndcg` refuses a recipe
whose version it cannot read, naming the versions it knows
(`rcp_ndcg.inference.recipes.RECIPE_SCHEMA_VERSIONS`) before any field is merged. The check runs first, so a
newer `rcp-ndcg-vllm` and an older `rcp-ndcg` fail with a versioned message instead of a schema surprise.
`schema_version` bumps when the file format changes incompatibly; adding an optional field does not bump it.
There is no lockstep version pin between the two packages: the reader names the versions it understands, and a
recipe of the operator's own goes through the same check ([recipes and serving models](recipes.md)).

## The behaviour fingerprint

A recipe's **behaviour fingerprint** keys everything that can change what the model returns: the model id and
revision, the engine image and its version floor, the `serve` block, the source hash of every plugin module the
engine runs, the template file's bytes, the tokenizer's SHA-256, and the client fields that change the request
bytes. Its rule is versioned (`rcp-fp/4` in this release, `rcp_ndcg_test.fingerprint.FINGERPRINT_SCHEMA`):
changing the input set or their canonicalisation is a new rule (`rcp-fp/5`), so corpora keyed by different rules
never collide. A bump is deliberate and recorded in the fingerprint module, in the
[verified fake engines](../how-to/use-verified-fake-engines.md) page and in the CHANGELOG; every committed corpus
is then re-keyed (only when the moved inputs shaped none of its recorded exchanges) or re-recorded on the next
wave, with `tests/conformance/stale.json` naming the ones awaiting recording.

## Observation corpora and their records

The GPU waves record an observation corpus per recipe; the corpus format is `rcp-ndcg-test`'s, and each of its
pieces is versioned separately (`rcp_ndcg_test.corpus`):

- the corpus manifest's `schema`: `rcp-ndcg.observation-corpus/1`;
- a repository subset's `index.json`: `rcp-ndcg.observation-subset/1`;
- one exchange record: `RECORD_SCHEMA` (`1`); a bump never invalidates an old corpus, because `load_corpus`
  migrates older records forward;
- the normalisation rule applied before a comparison: `NORMALISATION_VERSION` (`1`); a changed rule is a new
  version, never a silent edit;
- the conformance verifier's record: `rcp-ndcg.verification/1`.

The format has one home and one reader (`rcp_ndcg_test.corpus`); the pairs files' generator carries its own
`GENERATOR_VERSION` and media-set version. A change to any of these is recorded in the package's own history,
not in `rcp-ndcg`'s.

## Artifact tags

Every artifact `rcp-ndcg` writes carries its own version tag, `rcp-ndcg.<name>.v1`, independent of the package
version and bumped only when that artifact changes incompatibly. The tag is the record's `schema` field, and a
reader refuses a tag it does not know. Within one `v1`, fields are only ever added (optional or defaulted), so a
reader of the first `v1` keeps reading later `v1` records; a change to an existing field's meaning or type is a
new tag, never a silent reinterpretation. The rule, spelled out for the exported results, is on
[the results record](results-record.md); it holds for the run manifest and layout
(`rcp-ndcg.run-manifest.v1`, `rcp-ndcg.run-layout.v1`), the judgement store
(`rcp-ndcg.judgement-store.v1`), the calibration (`rcp-ndcg.calibration.v1`) and every other exported schema.
`rcp-ndcg schema export --out DIR` writes them all; the repository's `schemas/` is that output, committed, and
`tests/contract` pins it byte for byte.

## The versions of a release

| Version | Home | Bumps when |
|---|---|---|
| the package version | all four `pyproject.toml` files, `rcp-ndcg --version` | a public-surface change (minor) or an addition (patch), per the `0.0.x` rules above |
| a recipe's `schema_version` | `rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/*/family.yaml` | the recipe file format changes incompatibly |
| the behaviour fingerprint rule | `rcp_ndcg_test.fingerprint.FINGERPRINT_SCHEMA` (`rcp-fp/4`) | the fingerprint's input set or canonicalisation changes |
| the corpus schemas | `rcp_ndcg_test.corpus` (`rcp-ndcg.observation-corpus/1`, `RECORD_SCHEMA`, `NORMALISATION_VERSION`) | the corpus format, record shape or normalisation rule changes |
| an artifact tag | `schemas/*.v1.json` | that artifact changes incompatibly |
| the run layout | `rcp_ndcg.runs.layout.LAYOUT_VERSION` (`rcp-ndcg.run-layout.v1`) | the on-disk run layout changes incompatibly |
