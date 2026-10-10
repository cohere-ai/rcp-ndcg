# Lane freeze — the 0.0.1 public-surface freeze: the curated surface, both `Family` renames, the generated API pages

**Status: DONE.** Branch `lane/freeze`, base `rfc-0001` at `aae1aafe` (the current tip at the time of writing; the
branch already contained it, so there was nothing to merge — re-checked against `origin/rfc-0001` before the
gate). The gated commit is `2690bfa9`; `bin/gate lane/freeze` ran in the foreground on it and printed
`GATE: PASS` (slot 2; every step `exit=0`: ruff-check, ruff-format, basedpyright, pytest, contract-docs, mkdocs,
test-pkg, recipes, vllm-pkg, vllm-models, run_all, public-names, clean). This report is the commit after it and
changes prose only.

## 1. Commits

| commit | what |
|---|---|
| `2690bfa9` | The 0.0.1 public-surface freeze: the curated surface, both `Family` renames, and the generated API pages (83 files) |
| (this report) | `handover: the freeze lane report` |

No co-author lines, no attribution, nothing pushed, `rfc-0001` not merged into.

## 2. What changed (per brief item)

### The two renames

- `rcp_ndcg_core.schemas.Family` → **`JudgementFamily`** (the judgement instrument family). Every use moved:
  `JudgementSet.families`, the calibration fit and `family_of`, `JudgementStore.claim`/`StageEntry`,
  `RunManifest.families`, `runs.pipeline`, `judging.reparse`, the CLI's calibration models, the tests
  (`tests/calibration/*`, `tests/core/test_public_records.py`, `tests/judging/*`, `tests/cli/test_eval.py`),
  the docstrings and the docs (`docs/concepts/rubric.md`). The core facade and the core schema `__all__` carry
  the new name.
- `rcp_ndcg_vllm.recipe.Family` → **`RecipeFamily`** (the shipped recipe family). `load_family`,
  `iter_families`, `load_recipes_of`, `_expand_variant`, `family_json_schema` and the package facade moved
  with it; `docs/how-to/add-a-model.md`, `docs/reference/recipes.md` and the recipes README name the new class.
- Regenerated the documented way: the exported schemas (`pytest tests/contract --update-snapshots`; six
  schemas carried `Family` in a description), the contract snapshots (`python_api.json`, `cli.json`) and the
  recipe family schema (`rcp_ndcg_vllm.recipe.family_json_schema()` → `rcp-ndcg-vllm/schema/family.schema.json`;
  one line: `"title": "Family"` → `"title": "RecipeFamily"`). The recipe **goldens did not move**: the golden
  test is green with the committed bytes (`rcp-ndcg-test/tests/recipes/test_family_goldens.py`, 49 passed), so
  nothing was regenerated.
- `KNOWN_SECOND_HOMES` in the contract suite is now empty (no public name has two homes) and the one-home test
  pins that.

### The curated surface

Every one of the 23 public modules' `__all__` was reviewed once, against the rule "keep what a user or a plugin
seam genuinely needs; implementation detail is not public". The 12 names the QA freeze found on no page are
classified:

| name | classification |
|---|---|
| `NullResultSink`, `ParquetResultSink`, `RESULTS_GROUP`, `RESULT_SCHEMA`, `registered_result_sinks`, `result_sink_class` | **documented**: the results-export seam stays public (decision 40); the generated `rcp_ndcg.results` page documents them |
| `ChangeMechanism`, `ResolvedRevision`, `StoredRankings` | **documented**: public types that appear in signatures or are returned (`apply_text_policy`'s changes, `resolve_revision`, `stored_rankings_model`); the generated pages document them |
| `hub_cache_dir`, `hub_offline` | **private**: renamed `_hub_cache_dir`/`_hub_offline`; the hub reader (`data/io/hub.py`) is the only caller and was updated |
| `REFERENCE_SYSTEMS` | **removed from the public surface**: it is the run pipeline's constant, not a results-export concept; `rcp_ndcg.results` no longer re-exports it (it stays in `runs.pipeline`, where the CLI and the exporter import it) |

The review found no other accidental export: the other names on no narrative page are the advanced surface the
release notes describe (estimator classes, registry constants, plugin-seam interfaces, model records), and the
generated pages now document every one of them. The surface counts:

| | modules | distinct names | exported name entries |
|---|---|---|---|
| before | 23 | 376 | 453 |
| after | 23 | 374 | 450 |

### The contract collector's broad exception (ARCH-7)

`tests/contract/surface.py`'s `collect_python` caught every `Exception` around `getattr(module, name)` and
recorded a fake `requires_extra`, so a name listed in `__all__` but absent at runtime could not fail the
snapshot test. It now catches only an `ImportError` (a lazy re-export an optional extra alone resolves) and the
engine plugin's own `RuntimeError` whose message starts with `rcp-ndcg-vllm:` (the vLLM-absent refusal of
`rcp_ndcg_vllm.models.topk`); anything else — an `AttributeError` for a broken `__all__` entry included —
propagates. A test pins it by injecting a bogus name into `rcp_ndcg.errors.__all__` and requiring
`AttributeError`.

### "Public ⇒ documented", mechanical

- New `rcp_ndcg.support.api_docs` owns the one list of public modules (`PUBLIC_MODULES`, moved out of
  `tests/contract/surface.py`, which now imports it) and renders the pages: one Markdown page per public module
  from the pinned snapshot plus the modules' own docstrings — each name with its kind, signature and one-line
  role (a function's parameters and return, a model's fields, an enum's values, a constant's value; the role
  comes from the object's own docstring, a class's own `__doc__` so an undocumented subclass cannot inherit
  `str`'s, or the `#:` comment / string literal beside a constant).
- New CLI command `rcp-ndcg docs api --out docs/reference/api [--snapshot PATH]` (`rcp_ndcg.cli.docs`, a `docs`
  group in the lazy command table) wraps it; `docs/reference/api/` holds the 23 committed pages, listed in
  `mkdocs.yml` under "API reference (generated)".
- `tests/contract/test_public_names.py` is rewritten: every name in the snapshot must appear on a page under
  `docs/`, every pinned module must have a page, a fresh generation must equal the committed pages byte for
  byte, and the snapshot must name exactly `PUBLIC_MODULES`. The `--update-snapshots` flag rewrites the pages
  with the snapshots, so one flow keeps everything current; the freeze list
  (`tests/contract/undocumented_public_names.json`) is deleted.
- The collector's `_kind` now classifies an alias of a builtin type (`ID = str`, `JobHandle = str`) as a
  `type_alias` rather than a class, so the generated pages render `ID = str` instead of `class ID()` (four
  entries; a small surface-data correction the generator surfaced).

### The freeze record

- `docs/reference/versioning.md` gains **The 0.0.1 freeze**: the snapshot is the definition of public; public
  means documented and pinned; the generated pages are the long tail; everything else is internal and may
  change; at 0.0.1 a rename or removal is made directly (the deprecation path applies from the next minor
  release on); the update flow. `docs/reference/public-surface.md` is the short form and the index.
- `AGENTS.md`'s rules now carry the freeze bullet pointing at the versioning page; `CHANGELOG.md`'s versioning
  preamble and the `### Changed` freeze bullet name the new mechanism; `handover/RELEASE-CHECKLIST.md`'s freeze
  item is updated.
- `docs/reference/cli.md` documents the new `docs api` command.

## 3. Verification

**Every fix had a failing test first.** Before the implementation:

- `uv run --no-sync pytest tests/contract/test_public_surface.py -q -p no:cacheprovider` → **4 failed, 9
  passed**: `test_the_family_classes_carry_specific_names` (`assert "JudgementFamily" in
  rcp_ndcg_core.schemas.__all__`), `test_a_broken_all_entry_fails_collection` (no `AttributeError` — the broad
  handler swallowed it), `test_the_curated_surface_classifies_the_nowhere_names` (`assert 'REFERENCE_SYSTEMS'
  not in [...]`), `test_one_home_per_concept` (`AssertionError: {'Family': ['rcp_ndcg_core.schemas',
  'rcp_ndcg_vllm.recipe']}`).
- `uv run --no-sync pytest tests/contract/test_public_names.py -q -p no:cacheprovider` → collection error,
  `ModuleNotFoundError: No module named 'rcp_ndcg.support.api_docs'`.

**Mutations after the fix** (each test goes red):

- appending a line to `docs/reference/api/rcp_ndcg.results.md` →
  `test_the_generated_pages_are_current` FAILED (the committed pages must equal a fresh generation);
- deleting `docs/reference/api/rcp_ndcg.data.revisions.md` → 3 FAILED (`test_every_public_name_is_documented`,
  `test_the_generated_pages_cover_the_pinned_modules`, `test_the_generated_pages_are_current`).

**Commands and results:**

| command | result |
|---|---|
| `uv run --no-sync ruff format --check .` | 619 files already formatted |
| `uv run --no-sync ruff check .` | All checks passed |
| `uv run --no-sync basedpyright` | 0 errors, 0 warnings, 0 notes |
| `heavy uv run --no-sync pytest tests/ -q -n 4 -p no:cacheprovider` | 4127 passed, 153 skipped |
| `RCP_NDCG_TEST_TIMEOUT=300 uv run --no-sync python -X faulthandler -m pytest rcp-ndcg-test/tests -q -p no:cacheprovider` | 1103 passed, 228 skipped |
| `uv run --no-sync pytest tests/contract tests/docs tests/cli -q -p no:cacheprovider` | 608 passed, 113 skipped |
| `uv run --no-sync mkdocs build --strict -d <scratch>/site` | Documentation built |
| `uv run --no-sync pytest rcp-ndcg-test/tests/recipes/test_family_goldens.py -q` | 49 passed (goldens unchanged) |
| `uv run --no-sync rcp-ndcg docs api --out docs/reference/api` | 23 pages written, regeneration check green |
| `timeout 7200 bin/gate lane/freeze` (foreground) | **GATE: PASS** (all steps `exit=0`; run_all 1022/987/35/0, public-names clean, tree clean) |

## 4. Open questions

- **70 of the 374 public names have no one-line role** on their generated page because the source has no
  docstring (37 constants, 14 type aliases, 11 pydantic models such as `Document`, `Query`, `ImagePart`, 7
  literal aliases and the `JobStatus` enum). The page still lists each name with its kind and signature, so
  "documented" holds; adding docstrings to those objects is a later, additive improvement.
- **Two API sections.** The generated pages live under `docs/reference/api/` (the brief's path) while the three
  curated API pages live under `docs/api/`; a later docs pass may merge the sections, which would be a nav and
  link move, not a surface change.
- **`rcp-ndcg docs api` reads a repository-relative snapshot by default** (`tests/contract/snapshots/
  python_api.json`). It is the repository's own generator, like `schema export`; running it outside the
  repository needs `--snapshot PATH`, and a missing file is a typed `UsageError` with that hint.

## 5. CHANGELOG entry

Under the 0.0.1 section's `### Public surface` (the exact text added):

> - **The public surface is curated and mechanically documented (the 0.0.1 freeze).** Every one of the 23
>   public modules' `__all__` was reviewed once: the results-export seam stays public (decision 40:
>   `RESULTS_GROUP`, `RESULT_SCHEMA`, `NullResultSink`, `ParquetResultSink`, `registered_result_sinks`,
>   `result_sink_class`), the public types that appeared on no page are documented (`ChangeMechanism`,
>   `ResolvedRevision`, `StoredRankings`), and the accidental exports are gone: `hub_cache_dir`/`hub_offline`
>   are private (`_hub_cache_dir`/`_hub_offline`, the hub reader's own helpers) and `REFERENCE_SYSTEMS` is no
>   longer re-exported by `rcp_ndcg.results` (it stays the run pipeline's internal constant).
> - **Both `Family` classes carry their own names**: `rcp_ndcg_core.schemas.Family` is `JudgementFamily` (the
>   judgement instrument family) and `rcp_ndcg_vllm.recipe.Family` is `RecipeFamily` (the shipped recipe
>   family). Every use, the exported schemas and the recipe family schema are regenerated.
> - **Every public name is documented**: `rcp-ndcg docs api --out docs/reference/api` generates one page per
>   public module (each name with its kind, signature and one-line role) from the pinned snapshot and the
>   modules' docstrings; `docs/reference/api/` holds the 23 committed pages, and `tests/contract` fails when a
>   pinned name is on no page, when a module has no page, or when the pages drift from the snapshot.
> - The contract snapshots, the exported schemas and the generated API pages record every name above;
>   regenerate them with `uv run pytest tests/contract --update-snapshots` and `rcp-ndcg docs api --out
>   docs/reference/api` (review the diff, then add the CHANGELOG entry).

The `### Changed` bullet "The public surface is frozen for the 0.0.1 line" now names `PUBLIC_MODULES` in
`rcp_ndcg.support.api_docs`, the snapshot and the generated pages instead of the deleted freeze list, and the
versioning preamble's `PUBLIC_MODULES` reference points at the new home.

## 6. Public surface changes

- **Renamed names**: `rcp_ndcg_core.schemas.Family` → `JudgementFamily` (also through the core facade);
  `rcp_ndcg_vllm.recipe.Family` → `RecipeFamily` (also through the serving facade).
- **Removed names**: `rcp_ndcg.data.revisions.hub_cache_dir`, `hub_offline` (private now);
  `rcp_ndcg.results.REFERENCE_SYSTEMS` (no longer public anywhere).
- **CLI**: new group `rcp-ndcg docs` with `docs api --out DIR [--snapshot PATH]` (output schema
  `rcp-ndcg.docs-api.v1`); `tests/contract/snapshots/cli.json` carries the tree.
- **Schemas**: new `schemas/docs-api.v1.json`; regenerated `calibration-summary`, `calibration`,
  `judgement-store`, `judgement`, `run-manifest` and `run-summary` (the rename in their descriptions); the
  recipe family schema's title. 47 `*.v1.json` files now (the versioning page's count updated).
- **Snapshot data**: `python_api.json` (376 → 374 distinct names; `ID`/`JobHandle` classified as type
  aliases), `cli.json`.
- **Docs**: `docs/reference/api/` (23 generated pages), the versioning/public-surface/recipes/cli/rubric/
  add-a-model pages, `mkdocs.yml`, `AGENTS.md`, `CHANGELOG.md`.

## 7. Files outside scope

- `rcp-ndcg-test/tests/recipes/test_pplx_embed_v1.py` — `ruff format` added the two blank lines the tip
  introduced before `_reference_module()` (the commit after the QA gate left the file unformatted, and the
  gate's `ruff-format` step would fail on it).
- `handover/RELEASE-CHECKLIST.md` — the freeze item now names the new mechanism (the checklist is the owner's
  release record).
- `rcp-ndcg/src/rcp_ndcg/testing/__init__.py` — the `adapter_contract` docstring named the internal design
  document; reworded to "the adapter seam's contract", because the generated page made it user-facing and the
  docs wording test refuses design ids.

## 8. For the next lanes

- The public surface is frozen at 23 modules / 374 names / 450 entries. A new public name is a deliberate diff:
  add it to a module's `__all__`, run `pytest tests/contract --update-snapshots`, then `rcp-ndcg docs api --out
  docs/reference/api`, and add the CHANGELOG entry under `### Public surface`.
- `PUBLIC_MODULES` has one home (`rcp_ndcg.support.api_docs`); `tests/contract/surface.py` imports it, and the
  generated pages and the snapshot are checked against it. Do not reintroduce a second list.
- The release checklist's freeze item is done; the only open freeze-adjacent item is the owner's GPU/CI work.
