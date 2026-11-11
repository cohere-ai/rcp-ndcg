# Compatibility and versioning

What this release promises to keep stable, what may move in a `0.0.x` patch, and how the versioned artifacts
outside the package version (recipes, observation corpora, the run layout, the result record) carry their own
versions. The normative spelling of every name below lives in the repository, not on this page:
`tests/contract/snapshots/` and `schemas/` are the contract, and the tests in `tests/contract/` enforce it.

## What is public

- **The Python names** in the `__all__` of the public modules. `PUBLIC_MODULES` in
  `rcp.support.api_docs` lists them (the same list generates the API pages and the contract snapshot),
  and `tests/contract/snapshots/python_api.json` pins every name: the facade `rcp`; the core
  `rcp_core` with `.gain`, `.irt`, `.metric`, `.protocol` and `.records`; `rcp.calibration`,
  `.data`, `.data.preprocess`, `.data.revisions`, `.errors`, `.eval`, `.eval.mteb`, `.examples`, `.inference`,
  `.judging`, `.results`, `.retrieval`, `.runners`, `.runs` and `.testing`; and `rcp_vllm.recipe` (the
  serving package's recipe module).
- **The command line**: the command tree, its flags, its output schemas and the exit codes
  (`tests/contract/snapshots/cli.json`, `exit_codes.json`; [the command line](cli.md)).
- **The MCP tools** `rcp mcp serve` exposes (`tests/contract/snapshots/mcp_tools.json`).
- **The packaging**: distributions, extras and entry-point groups (`tests/contract/snapshots/packaging.json`).
- **The JSON Schemas** exported to `schemas/` (47 `*.v1.json` files in this release) and the
  `rcp-vllm` recipe schemas (`tests/contract/snapshots/vllm_cli.json`).

Everything else is internal and may change without notice: `rcp.cli`, `rcp.storage`,
`rcp.support`, `rcp.data.io`, the submodules of `retrieval`, `judging`, `calibration`, `runs` and
`runners`, the rest of `rcp_vllm`, and all of the unpublished `rcp-test`. A change to a public name,
flag, exit code or schema is a reviewed diff against the snapshot and gets a CHANGELOG entry; a pull request
that changes either without one fails CI.

## The 0.0.1 freeze

0.0.1 freezes the Python surface deliberately and once:

- **The snapshot is the definition of public.** A name is public exactly when it is in the `__all__` of one of
  the modules :data:`rcp.support.api_docs.PUBLIC_MODULES` lists, and the snapshot pins it. A page that
  misses a name does not make it private, and an internal module's `__all__` makes nothing public.
- **Public means documented and pinned.** Every pinned name is documented: the curated concept and how-to
  pages carry the narrative, and `rcp docs api --out docs/reference/api` generates one page per public
  module (each name with its kind, signature and one-line role) from the snapshot and the modules' docstrings.
  `tests/contract` fails when a pinned name is on no page, when a pinned module has no page, or when the
  committed pages differ from a fresh generation, so the docs cannot drift from the surface.
- **Everything else is internal** and may change without notice, including a name removed from a public
  module's `__all__`: it is no longer part of the surface even while the module still resolves it.
- **A public name that must go is deprecated for one minor release** (see below); at 0.0.1, with nothing
  released yet, a rename or removal is made directly and recorded in the CHANGELOG under `### Public surface`.

A surface change is a deliberate, reviewed diff: `pytest tests/contract` classifies it (BREAKING / ADDITIVE),
`pytest tests/contract --update-snapshots` rewrites the snapshots and `schemas/`, and
`rcp docs api` regenerates the API pages.

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

The four distributions carry the same version; `rcp` pins `rcp-core==<version>`; `rcp-vllm`
carries the same version but pins no sibling; `rcp-test` is never published.

### Deprecating a public name

A public name that must go is first **deprecated for one minor release**: it keeps working, warns where the
surface has a warning channel (a Python `DeprecationWarning`, a click `deprecated=True` command, a schema
description), and the CHANGELOG entry says what replaces it. It is removed in the next minor release, with its
own CHANGELOG entry. Nothing is removed silently, and an internal name needs no deprecation at all.

## Serving recipes: `schema_version`

The recipe file format is the versioned contract between `rcp-vllm` (which ships the data) and `rcp`
(which reads it): every family file carries `schema_version` (currently `"1"`), and `rcp` refuses a recipe
whose version it cannot read, naming the versions it knows
(`rcp.inference.recipes.RECIPE_SCHEMA_VERSIONS`) before any field is merged. The check runs first, so a
newer `rcp-vllm` and an older `rcp` fail with a versioned message instead of a schema surprise.
`schema_version` bumps when the file format changes incompatibly; adding an optional field does not bump it.
There is no lockstep version pin between the two packages: the reader names the versions it understands, and a
recipe of the operator's own goes through the same check ([recipes and serving models](recipes.md)).

## The behaviour fingerprint

A recipe's **behaviour fingerprint** keys everything that can change what the model returns: the model id and
revision, the engine image and its version floor, the `serve` block, the source hash of every plugin module the
engine runs, the template file's bytes, the tokenizer's SHA-256, and the client fields that change the request
bytes. Its rule is versioned (`rcp-fp/4` in this release, `rcp_test.fingerprint.FINGERPRINT_SCHEMA`):
changing the input set or their canonicalisation is a new rule (`rcp-fp/5`), so corpora keyed by different rules
never collide. A bump is deliberate and recorded in the fingerprint module, in the
[verified fake engines](../how-to/use-verified-fake-engines.md) page and in the CHANGELOG; every committed corpus
is then re-keyed (only when the moved inputs shaped none of its recorded exchanges) or re-recorded on the next
wave, with `tests/conformance/stale.json` naming the ones awaiting recording.

## Observation corpora and their records

The GPU waves record an observation corpus per recipe; the corpus format is `rcp-test`'s, and each of its
pieces is versioned separately (`rcp_test.corpus`):

- the corpus manifest's `schema`: `rcp-ndcg.observation-corpus/1`;
- a repository subset's `index.json`: `rcp-ndcg.observation-subset/1`;
- one exchange record: `RECORD_SCHEMA` (`2`); a bump never invalidates an old corpus, because `load_corpus`
  migrates older records forward (the `1` -> `2` migration stamps the engine version a version 1 record could
  not carry as explicitly unavailable, and never invents one);
- the normalisation rule applied before a comparison: `NORMALISATION_VERSION` (`1`); a changed rule is a new
  version, never a silent edit;
- the conformance verifier's record: `rcp-ndcg.verification/1`.

The format has one home and one reader (`rcp_test.corpus`); the pairs files' generator carries its own
`GENERATOR_VERSION` and media-set version. A change to any of these is recorded in the package's own history,
not in `rcp`'s.

## Artifact tags

Every artifact `rcp` writes carries its own version tag, `rcp.<name>.v1`, independent of the package version
and bumped only when that artifact changes incompatibly. The tag is the record's `schema` field, and a reader
refuses a tag it does not know. Within one `v1`, fields are only ever added (optional or defaulted), so a reader
of the first `v1` keeps reading later `v1` records; a change to an existing field's meaning or type is a new tag,
never a silent reinterpretation. The rule, spelled out for the exported results, is on
[the results record](results-record.md); it holds for the run manifest and layout
(`rcp.run-manifest.v1`, `rcp-ndcg.run-layout.v1`), the judgement store (`rcp.judgement-store.v1`), the
calibration (`rcp.calibration.v1`) and every other exported schema.

0.0.1 moved the tag from `rcp-ndcg.<name>.v1` to `rcp.<name>.v1`; the run layout's own tag
(`rcp-ndcg.run-layout.v1`) is not a schema id and did not move. An artifact written before the move carries the
old spelling, and every reader accepts both. A new artifact carries the new spelling. A judgement store keeps
the tag it was created with, so a resume writes its identity back under that tag and never migrates the file in
place. The judgement records a resume appends do carry the new tag, so the code of that day opens the store and
then refuses a record it does not know: finish an old store with a new install, or start a new one.
`rcp schema export --out DIR` writes them all; the repository's `schemas/` is that output, committed, and
`tests/contract` pins it byte for byte.

## The versions of a release

| Version | Home | Bumps when |
|---|---|---|
| the package version | all four `pyproject.toml` files, `rcp --version` | a public-surface change (minor) or an addition (patch), per the `0.0.x` rules above |
| a recipe's `schema_version` | `rcp-vllm/src/rcp_vllm/recipes/*/family.yaml` | the recipe file format changes incompatibly |
| the behaviour fingerprint rule | `rcp_test.fingerprint.FINGERPRINT_SCHEMA` (`rcp-fp/4`) | the fingerprint's input set or canonicalisation changes |
| the corpus schemas | `rcp_test.corpus` (`rcp-ndcg.observation-corpus/1`, `RECORD_SCHEMA`, `NORMALISATION_VERSION`) | the corpus format, record shape or normalisation rule changes |
| an artifact tag | `schemas/*.v1.json` | that artifact changes incompatibly |
| the run layout | `rcp.runs.layout.LAYOUT_VERSION` (`rcp-ndcg.run-layout.v1`) | the on-disk run layout changes incompatibly |
