# Compatibility and versioning

This page states what the release keeps stable, and what can move in a `0.0.x` patch. It also states how the versioned
artifacts carry their own versions. These artifacts are the recipes, the observation corpora, the run layout and the
result record. Their versions are separate from the package version. The normative spelling of every name below lives
in the repository, not on this page. `tests/contract/snapshots/` and `schemas/` are the contract, and the tests in
`tests/contract/` enforce it.

## What is public

- **The Python names** in the `__all__` of the public modules. `PUBLIC_MODULES` in `rcp_ndcg.support.api_docs` lists
  them, and the same list generates the API pages and the contract snapshot.
  `tests/contract/snapshots/python_api.json` pins every name. The list holds the facade `rcp_ndcg` and the core
  `rcp_ndcg_core` with `.gain`, `.irt`, `.metric`, `.protocol` and `.records`. It also holds `rcp_ndcg.calibration`,
  `.data`, `.data.preprocess`, `.data.revisions`, `.errors`, `.eval`, `.eval.mteb`, `.examples`, `.inference`,
  `.judging`, `.results`, `.retrieval`, `.runners`, `.runs` and `.testing`. The last name is `rcp_ndcg_vllm.recipe`,
  the recipe module of the serving package.
- **The command line**: the command tree, its flags, its output schemas and the exit codes
  (`tests/contract/snapshots/cli.json`, `exit_codes.json`; [the command line](cli.md)).
- **The MCP tools** `rcp-ndcg mcp serve` exposes (`tests/contract/snapshots/mcp_tools.json`).
- **The packaging**: distributions, extras and entry-point groups (`tests/contract/snapshots/packaging.json`).
- **The JSON Schemas** exported to `schemas/` (47 `*.v1.json` files in this release) and the `rcp-ndcg-vllm` recipe
  schemas (`tests/contract/snapshots/vllm_cli.json`).

Everything else is internal and can change without notice. The internal modules are `rcp_ndcg.cli`,
`rcp_ndcg.storage`, `rcp_ndcg.support`, `rcp_ndcg.data.io` and the submodules of `retrieval`, `judging`,
`calibration`, `runs` and `runners`. The rest of `rcp_ndcg_vllm` is internal, and so is all of the unpublished
`rcp-ndcg-test`. A change to a public name, a flag, an exit code or a schema is a reviewed diff. It is a diff against
the snapshot, and it gets a CHANGELOG entry. A pull request that changes either one without an entry fails CI.

## The 0.0.1 freeze

0.0.1 freezes the Python surface on purpose and one time:

- **The snapshot is the definition of public.** A name is public exactly when it is in the `__all__` of a listed
  module. The list is in :data:`rcp_ndcg.support.api_docs.PUBLIC_MODULES`, and the snapshot pins the name. A page that
  misses a name does not make it private. The `__all__` of an internal module makes nothing public.
- **Public means documented and pinned.** Every pinned name is documented. The curated concept and how-to pages give
  the explanation. `rcp-ndcg docs api --out docs/reference/api` generates one page per public module from the snapshot
  and the docstrings of the modules. Each name on that page has its kind, its signature and a one-line role.
  `tests/contract` fails when a pinned name is on no page, or when a pinned module has no page. It also fails when the
  committed pages differ from a fresh generation. The docs therefore cannot differ from the surface.
- **Everything else is internal** and can change without notice. This includes a name removed from the `__all__` of a
  public module. That name is no longer part of the surface, even while the module still resolves it.
- **A public name that must go is deprecated for one minor release** (see below). At 0.0.1 nothing is released yet, so
  a rename or a removal is made directly. It is recorded in the CHANGELOG under `### Public surface`.

A surface change is a deliberate, reviewed diff. `pytest tests/contract` classifies it (BREAKING / ADDITIVE).
`pytest tests/contract --update-snapshots` rewrites the snapshots and `schemas/`, and `rcp-ndcg docs api` regenerates
the API pages.

## The `0.0.x` rules

While the version is `0.x`, a change that breaks the public surface bumps the minor version. An additive change bumps
the patch version. From `1.0` on, semantic versioning applies.

A `0.0.x` patch release:

- may add a public name, a flag, a schema field (optional or defaulted), a sink or a runner. It may also fix behaviour
  when the documented contract did not promise the old output;
- may change internal modules, messages, logs, performance and the wording of a hint;
- does not rename or remove a public name, and does not change an exit code. It does not change the meaning of a
  schema field or of a shipped number. The metric, the gains, the scoring protocols, the schedules and the calibration
  fit reproduce the paper. Anchor tests pin them. A change to their numbers is deliberate and tested, and it goes into
  the CHANGELOG ([REPRODUCIBILITY.md](https://github.com/cohere-ai/rcp-ndcg/blob/main/REPRODUCIBILITY.md)).

The four distributions carry the same version. `rcp-ndcg` pins `rcp-ndcg-core==<version>`. `rcp-ndcg-vllm` carries the
same version but pins no sibling, and `rcp-ndcg-test` is never published.

### Deprecating a public name

A public name that must go is first **deprecated for one minor release**. It keeps working, and it warns where the
surface has a warning channel. The channel is a Python `DeprecationWarning`, a click `deprecated=True` command or a
schema description. The CHANGELOG entry says what replaces it. The name is removed in the next minor release, with its
own CHANGELOG entry. Nothing is removed silently. An internal name needs no deprecation at all.

## Serving recipes: `schema_version`

The recipe file format is the versioned contract between `rcp-ndcg-vllm`, which ships the data, and `rcp-ndcg`, which
reads it. Every family file carries `schema_version` (currently `"1"`). `rcp_ndcg` refuses a recipe whose version it
cannot read. It names the versions that it knows (`rcp_ndcg.inference.recipes.RECIPE_SCHEMA_VERSIONS`) before it
merges any field. The check runs first. A newer `rcp-ndcg-vllm` and an older `rcp-ndcg` therefore fail with a
versioned message, not with an unexpected schema error. `schema_version` bumps when the file format changes in an
incompatible way. An added optional field does not bump it. There is no lockstep version pin between the two packages.
The reader names the versions that it understands. A recipe of the operator's own goes through the same check
([recipes and serving models](recipes.md)).

## The behaviour fingerprint

A recipe's **behaviour fingerprint** is a key over everything that can change what the model returns. The key covers
the model id and revision, the engine image and its version floor, and the `serve` block. It also covers the source
hash of every plugin module that the engine runs. The bytes of the template file and the SHA-256 of the tokenizer are
also part of the key. So are the client fields that change the bytes of the request. Its rule is versioned (`rcp-fp/4`
in this release, `rcp_ndcg_test.fingerprint.FINGERPRINT_SCHEMA`). A change to the input set or to their
canonicalisation is a new rule (`rcp-fp/5`). Corpora keyed by different rules therefore never collide. A bump is
deliberate. It is recorded in the fingerprint module, in the
[verified fake engines](../how-to/use-verified-fake-engines.md) page and in the CHANGELOG. Every committed corpus is
then re-keyed or re-recorded on the next wave. It is re-keyed only when the moved inputs shaped none of its recorded
exchanges. `tests/conformance/stale.json` names the corpora that await recording.

## Observation corpora and their records

The GPU waves record one observation corpus for each recipe. The corpus format belongs to `rcp-ndcg-test`, and each
piece of it has its own version (`rcp_ndcg_test.corpus`):

- the corpus manifest's `schema`: `rcp-ndcg.observation-corpus/1`;
- a repository subset's `index.json`: `rcp-ndcg.observation-subset/1`;
- one exchange record: `RECORD_SCHEMA` (`2`). A bump never invalidates an old corpus, because `load_corpus` migrates
  older records forward. The `1` -> `2` migration stamps the engine version that a version 1 record could not carry as
  explicitly unavailable. It never invents one;
- the normalisation rule that the code applies before a comparison: `NORMALISATION_VERSION` (`1`). A changed rule is a
  new version, never a silent edit;
- the conformance verifier's record: `rcp-ndcg.verification/1`.

The format has one home and one reader (`rcp_ndcg_test.corpus`). The generator of the pairs files carries its own
`GENERATOR_VERSION` and its own media-set version. A change to any of these is recorded in the history of the package,
not in the history of `rcp-ndcg`.

## Artifact tags

Every artifact that `rcp-ndcg` writes carries its own version tag, `rcp-ndcg.<name>.v1`. The tag does not depend on
the package version. It bumps only when that artifact changes in an incompatible way. The tag is the `schema` field of
the record, and a reader refuses a tag that it does not know. Within one `v1`, fields are only ever added, and an
added field is optional or defaulted. A reader of the first `v1` therefore keeps reading later `v1` records. A change
to the meaning or the type of an existing field is a new tag, never a silent reinterpretation.
[The results record](results-record.md) states the rule for the exported results. The same rule holds for the run
manifest and layout (`rcp-ndcg.run-manifest.v1`, `rcp-ndcg.run-layout.v1`) and the judgement store
(`rcp-ndcg.judgement-store.v1`). It also holds for the calibration (`rcp-ndcg.calibration.v1`) and every other
exported schema. `rcp-ndcg schema export --out DIR` writes them all. The `schemas/` directory of the repository is
that output, committed, and `tests/contract` pins it byte for byte.

## The versions of a release

| Version | Home | Bumps when |
|---|---|---|
| the package version | all four `pyproject.toml` files, `rcp-ndcg --version` | a public-surface change (minor) or an addition (patch), as the rules above say |
| a recipe's `schema_version` | `rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/*/family.yaml` | the recipe file format changes in an incompatible way |
| the behaviour fingerprint rule | `rcp_ndcg_test.fingerprint.FINGERPRINT_SCHEMA` (`rcp-fp/4`) | the input set or the canonicalisation of the fingerprint changes |
| the corpus schemas | `rcp_ndcg_test.corpus` (`rcp-ndcg.observation-corpus/1`, `RECORD_SCHEMA`, `NORMALISATION_VERSION`) | the corpus format, the record shape or the normalisation rule changes |
| an artifact tag | `schemas/*.v1.json` | that artifact changes in an incompatible way |
| the run layout | `rcp_ndcg.runs.layout.LAYOUT_VERSION` (`rcp-ndcg.run-layout.v1`) | the on-disk run layout changes in an incompatible way |
