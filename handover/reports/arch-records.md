# Lane `core-records`: one home for the pipeline records (owner decision 37)

**Status: DONE.** Branch `lane/core-records`, gated on the merged tree `a4f7c084` (GATE: PASS).

## Commits

| Commit | Subject |
|---|---|
| `1a6030fc` | One home for the pipeline records (owner decision 37): the private `_records` becomes the public `rcp_ndcg_core.records` with `__all__`, exported through the core and `rcp_ndcg.data` facades; `QueryRow`/`DocumentRow` are deleted, their strict rules (unknown keys refused, numeric ids read as strings) move into the records, `Dataset.from_records` takes the records or plain dicts with their field names and aliases and `Dataset.queries`/`corpus` hold the records; the formatting rules stay on the records (workstream 10's one home); every importer uses the public path; the IRT estimators the pipeline refits are readable from `rcp_ndcg_core.irt` (lazily, keeping the package torch-free); snapshots, CHANGELOG, docs and AGENTS.md regenerated |
| `2d1f26e7` | The verifiers' minor findings: the coercion rule is pinned on every record (`RankingExample`'s numeric ids and qrels keys read as strings, with a test), the records' mutability is stated in the CHANGELOG and `docs/data.md`, the CHANGELOG preamble names `rcp_ndcg_core.records`, the MASTER's remaining-private-import bullet drops the gone `_records` path, and the records module drops the five dead type aliases its `__all__` never exported |
| `a4f7c084` | Merge `rfc-0001` (`d630e4a6`, lane `sz-qwen3`) into `lane/core-records` |

The gate ran on `a4f7c084`. `rfc-0001` was at `d630e4a6` when merged; the merge was conflict-free and the sz-qwen3
CHANGELOG entry and recipe data are intact.

## What changed (per brief item)

1. **One public module.** `git mv _records.py records.py` (no shim), `__all__` declared (`ID`,
   `TEXT_FORMATTING_VERSION`, `Document`, `DocumentTitle`, `Input`, `Query`, `RankingExample`, `Text`,
   `mteb_document_text`); the core facade and `rcp_ndcg.data` re-export `Document`, `ID`, `Input`, `Query`,
   `RankingExample`, `Text`; `rcp_ndcg_core.records` is in `tests/contract/surface.py`'s `PUBLIC_MODULES`. All 42
   importers (the four packages plus `tests/`) moved to `rcp_ndcg_core.records`; docstring cross-references too.
   `import rcp_ndcg_core._records` fails, pinned by a test.
2. **`QueryRow`/`DocumentRow` deleted.** `Dataset.from_records` validates `Query`/`Document` (instances pass
   through, dicts validate with the records' names and aliases); `Dataset.queries`/`corpus` hold the records; the
   row models' rules moved into the records: `extra="forbid"` and `coerce_numbers_to_str=True` on `Input` (so every
   record, `RankingExample` included, refuses unknown keys and reads numeric ids as strings), each with a test.
   `frozen=True` was **not** moved: the records' own validators mutate in place (`Text._derive_text_from_content`
   derives `text` from `content`; `RankingExample` sorts/reorders), and the tables now hold the pipeline's working
   objects rather than frozen copies — stated in the CHANGELOG and `docs/data.md`. `QrelRow` and `RankingRow` stay.
3. **Formatting stays on the records.** The row-model wrappers went with the rows; the one home is
   `Document.model_content` / `Query.format_query` / `Query.format_content` / `mteb_document_text` in
   `rcp_ndcg_core.records` (workstream 10's placement, which the data layer and the role clients both read — the
   data layer cannot import `inference`, so the records are the only home both can use). The verifiers compared
   formatting against the base over 44 title/body/instruction/task/media cases: byte-identical.
4. **`_logging` and the private-surface items.** The `_logging` docstring now maps `rcp_ndcg_core.records` to
   `rcp_ndcg.core.records`. Decisions, one line each:
   - `BradleyTerryEstimator`, `RaschEstimator`: **exported** from `rcp_ndcg_core.irt`, lazily (PEP 562
     `__getattr__` plus a `TYPE_CHECKING` import for the checkers); the product reads them from the facade instead
     of `_bradley_terry`/`_rasch`. The package stays torch-free at import and the estimators import torch on first
     read; `tests/core/irt/test_torch_free_surface.py` was updated to allow exactly those two names and still fails
     if torch becomes eager.
   - `select_opponents`: **kept private** in `rcp_ndcg_core.irt._insertion`; the public concept is
     `rcp_ndcg.calibration.select_opponents` (a different signature and policy), and exporting the kernel under the
     same name would add a second public home for one name (the one-home test would need an exemption).
   - `PARSE_VERSION`: **kept private** in `rcp_ndcg.judging._parsing.common`; it is imported only within
     `rcp_ndcg` (not across packages), and the value travels publicly in `Family.parse_version`.
5. **Snapshots, CHANGELOG, docs.** `tests/contract/snapshots/python_api.json` regenerated (the classified diff:
   `rcp_ndcg_core.records` added; both facades gain the records and lose the rows; `rcp_ndcg_core.irt` gains the two
   estimators; the `from_records` annotations change). `schemas/` is unchanged: no exported schema ever contained
   the records (the fresh-export test passes). CHANGELOG `### Public surface` carries two entries (the records; the
   IRT estimators). `docs/data.md` gained the in-memory-record paragraph (public paths, dict aliases, mutability);
   `docs/api/evaluate.md` names the records instead of the row models; `AGENTS.md`'s one-home row points at
   `rcp_ndcg_core.records`. Decision 37 was already recorded in `handover/00-MASTER.md` (commit `937146ed`, before
   this lane), so no new decision text was needed; the stale "remaining private-name imports" bullet there now names
   only `irt._*`.

## Verification

Test-first, shown red before the fix (tests written first):

```
$ uv run --no-sync pytest tests/core/test_schemas.py tests/data/test_records.py -q -p no:cacheprovider
tests/data/test_records.py:9: in <module>
    from rcp_ndcg_core.records import Document, Query
E   ModuleNotFoundError: No module named 'rcp_ndcg_core.records'
```

**Round 1 — two fresh adversarial verifiers (DeepSeek-V4.1-flash, `:xhigh`), run in parallel, neither seeing the
other's output.** Both returned **VERDICT: PASS**, no blocker or major finding; per COMMON, no round 2.

- **Verifier 1 (lens A, correctness).** Findings: (F1) `coerce_numbers_to_str` on `Input` also widens
  `RankingExample` (numeric `doc_ids`/qrels keys now coerce), untested and unmentioned; (F2) `frozen` was dropped,
  so the tables hold mutable records, undocumented; (F3) the MASTER's remaining-private-import bullet still named
  `_records`. It reproduced every claim itself: facades, `_records` unimportable, `from_records` records/dicts/
  aliases/coercion/refusal, one home (only the pre-existing unrelated `Family` pair is a second home), torch-free
  import + lazy torch, the mutation tests (4 tests red without the strict config; the blocked-torch probe red with
  eager torch), 61/10/304/3635/634 test results, and `run_all` 1022/987/35/0, 67/67, 82/82.
- **Verifier 2 (lens B, regressions and hygiene).** Findings: (F1) the CHANGELOG preamble's public-module list
  omitted `rcp_ndcg_core.records`; (F2) same `RankingExample` coercion widening; (F3) same mutability point;
  (F4) the five unused type aliases (`Qrels`, `QrelsDict`, `SearchResults`, `Metric`, `Results`) were dead code and
  not in `__all__`. It also mutation-tested the new tests (strict config removed → 2 red; coercion removed → 2 red;
  shim restored → the private-path test red; eager torch → the torch-free probe red) and restored every file
  byte-identically, ran the full suites and checks, confirmed scope (58 files, no unrelated ones, `schemas/`
  untouched), one home, no leftovers, docs truth, snapshot green without `--update-snapshots`, and a clean checkout.

**Actions on the findings.** F1/F2 (coercion): kept the rule on `Input` — it is the records' rule, and the records
module's own docstring already says a numeric id reads as its string form — and made it deliberate with a
`RankingExample` test (`test_every_record_coerces_numeric_ids_to_strings`) plus a CHANGELOG clause. F2/F3
(mutability): stated in the CHANGELOG bullet and `docs/data.md`; no code change (restoring immutability would
fight the records' in-place validators). F3 (MASTER bullet): updated. F1 (CHANGELOG preamble): added
`rcp_ndcg_core.records`. F4 (dead aliases): deleted. All fixes are in `2d1f26e7`; the checks were re-run green.

## Checks

The gate ran on the merged tree `a4f7c084` (slot 4):

```
ruff-check      exit=0 All checks passed!
ruff-format     exit=0 578 files already formatted
basedpyright    exit=0 0 errors, 0 warnings, 0 notes
pytest          exit=0 3636 passed, 102 skipped in 58.38s
contract-docs   exit=0 304 passed, 55 skipped in 71.61s
mkdocs          exit=0 Documentation built
test-pkg        exit=0 639 passed, 424 skipped in 422.30s
recipes         exit=0 no failure outside the baseline (0 baseline failures remain, 0 fixed; pytest exit 0)
vllm-pkg        exit=0 40 passed in 4.77s
vllm-models     exit=0 72 passed, 7 skipped in 102.97s
run_all         exit=0 leaderboards 1022/987/35/0; human study 67/67; external LLM judges 82/82
public-names    exit=0 clean (0 baselined hits remain)
clean           exit=0 clean
GATE: PASS
```

Before the gate, the merged tree was also checked directly: full root suite 3636 passed/102 skipped,
`tests/contract tests/docs` 304 passed/55 skipped, `rcp-ndcg-test/tests` 639 passed/424 skipped,
`mkdocs build --strict` green, and `run_all` reproduced 1022/987/35/0, 67/67, 82/82.

## Docs updated

- `docs/data.md` — the in-memory-record paragraph (public paths, dict aliases, the records are the tables' working
  objects).
- `docs/api/evaluate.md` — the "Inputs in memory" sentence names the records, not the deleted row models.
- `AGENTS.md` — the one-home table's join/instruction row points at `rcp_ndcg_core.records`.
- `CHANGELOG.md` — the two `### Public surface` entries below.
- `handover/00-MASTER.md` — the remaining-private-import bullet no longer names `rcp_ndcg_core._records`.

Greps run (all now empty for the private path outside historical reports): `git grep -n
"rcp_ndcg_core\._records"` over the four `src` trees, `tests/`, `docs/`, `examples/`, `skills/`,
`README.md`, `REPRODUCIBILITY.md`, `mkdocs.yml`; `git grep -n "QueryRow\|DocumentRow"` over the same set.

## CHANGELOG entry

```markdown
- **The pipeline records are public, and the compatibility rows are gone** (owner decision 37): the records
  `Document`, `Query`, `RankingExample` (with `Text`, `Input` and `ID`) live in `rcp_ndcg_core.records` -- renamed
  from the private `_records`, `__all__` declared, no shim -- and are re-exported by the `rcp_ndcg_core` and
  `rcp_ndcg.data` facades, so a reader/writer plugin imports a public path. `Dataset.from_records` takes the
  records themselves or plain dicts with their field names and aliases, and `Dataset.queries`/`Dataset.corpus` hold
  them; the compatibility row models `QueryRow`/`DocumentRow` are deleted, with their strict rules moved into the
  records (unknown keys refused, numeric ids read as strings on every record, `RankingExample` included). The
  records are the pipeline's working objects, not frozen copies: `Dataset.queries`/`Dataset.corpus` hold them and
  a mutation of a passed-in record is visible in the dataset. The formatting rules stay where workstream 10 put
  them -- one home, `Document.model_content`/`Query.format_query`/`Query.format_content` on the records, read by
  the data layer and the role clients alike. The snapshots and schemas are regenerated; every importer in the
  repository uses the public path.
- **The IRT estimator classes the pipeline refits are readable from the package**: `rcp_ndcg_core.irt` exports
  `BradleyTerryEstimator` and `RaschEstimator` (read lazily: touching them imports torch, as the stand-alone
  fits already do, while importing the package stays torch-free). `rcp_ndcg.calibration` and `rcp_ndcg.judging`
  read them from `rcp_ndcg_core.irt` instead of its private submodules.
```

## Public surface changes

- `PUBLIC_MODULES` gains `rcp_ndcg_core.records` (snapshot section with the nine exported names).
- `rcp_ndcg_core.__all__` gains `Document`, `ID`, `Input`, `Query`, `RankingExample`, `Text`.
- `rcp_ndcg.data.__all__` gains `Document`, `ID`, `Input`, `Query`, `RankingExample`, `Text`; loses `DocumentRow`,
  `QueryRow` (deleted).
- `Dataset.from_records` parameter annotations change from `Iterable[QueryRow | Mapping[...]]` /
  `Iterable[DocumentRow | Mapping[...]]` to the records (signature snapshot).
- `rcp_ndcg_core.irt.__all__` gains `BradleyTerryEstimator`, `RaschEstimator` (lazy, torch-backed).
- No CLI, exit-code or `schemas/` change.

## Files outside scope

None. (The brief's scope covers the records, the data layer, the importers, tests, snapshots, CHANGELOG, docs,
`AGENTS.md` and the handover; the IRT estimator decision is brief item 4.)

## For the next lanes

- `rcp_ndcg_core.records` is the import path; `QueryRow`/`DocumentRow` no longer exist. A reader/writer plugin
  should type its seam with `Query`/`Document`/`RankingExample`.
- `select_opponents` and `PARSE_VERSION` remain private by decision (see item 4); `irt._*` imports still exist
  where the kernel is genuinely internal.
- `KNOWN_SECOND_HOMES` still carries `{"Family"}` for the two unrelated `Family` concepts (IRT judgement vs
  recipe family); the brief's "empty" is not achievable without renaming one of them, which is out of this lane's
  scope. No record name is a second home.
- The historical l10a CHANGELOG entry still mentions `DocumentRow.title` as a field of that change; the new
  final-state bullet supersedes it. A release-notes pass may want to rewrite it, but it is not false of the change
  it describes.
