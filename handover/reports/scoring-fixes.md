# Lane `scoring-fixes` — the scoring-chain review (F1–F11)

Base `rfc-0001` @ `28afb3b7`; merged `rfc-0001` @ `0fbe4685` (merge commit `57c8707f`). Branch `lane/scoring-fixes`.
Gate: PASS on the final head (and on the round-1 head). The review is
`reviews/scoring-chain-review-2026-10-09.md`; the binding rules are `COMMON.md` and `AGENTS.md`.

## Status

DONE.

## Commits

| Commit | Subject |
|---|---|
| `fbd63dc9` | calibration: the refit is order-canonical, no-evidence documents are visible, Count-nDCG gains, validated thetas |
| `9941d8e1` | core irt: refuse a Rasch observation the estimator cannot attach, and anchor the Bradley-Terry loss |
| `d2f01cda` | eval: a Count-nDCG route from the rubric store, group_mean deltas under the protocol's rule, the right hint |
| `bfa2e356` | contract: snapshots and schemas for the scoring fixes |
| `ec691dc8` | docs: the scoring fixes (CHANGELOG, calibration, metric, primitives, cli, evaluate) |
| `57c8707f` | Merge branch 'rfc-0001' into lane/scoring-fixes (brings `0fbe4685`, lane judge-gemma) |
| `5faf7041` | scoring fixes: verifier round 1 |
| `8a51517d` | scoring fixes: verifier round 2 |

## What changed

- **F1 (major)** — `_ordered` (`rcp_ndcg/calibration/_projection.py`) now gives planned windows
  (`window_seq=None`) a total order by `record_id` after the scheduled ones. Both the comparison list and the
  estimator's `doc_ids` inherit it, so two stores holding the same windows in either order give bit-identical
  theta, SE, items, queries and fingerprint; scheduled windows keep their positions, so no fitted number moved.
- **F2 (operator decision)** — the paper's numbers stay (a document with no comparison gets the query's mean
  ability), but it is visible: `CalibrationCoverage.no_tournament_evidence_documents` (new field, in
  `coverage.json`), the new warning code `NO_VALID_TOURNAMENT_EVIDENCE`, and a `DataError` under
  `calibrate(..., strict=True)` / `calibration fit --strict`. The list covers the documents of a query the fit
  reads; a query whose windows are all invalid is `uncalibrated` instead (no ability is written for it).
- **F3** — `rcp_ndcg.calibration.count_gains(judgements)` is the one derivation (rubric windows, sibling chunks
  pooled inside a window, per-criterion pass counts through `count_gain`), keyed as `Calibration.gains()` is.
  `evaluate(..., count_gains=...)` takes it; `rcp-ndcg eval score --metrics count_ndcg --judgements STORE`
  (repeatable) is the CLI route; `ReportInputs.judgements` lets `eval explain --report` re-score a saved
  Count-nDCG report; the `eval_score` MCP tool takes `judgements` too. The missing-count-gains hint now names
  the rubric windows and this derivation, not the tournament store.
- **F9** — `tests/core/irt/test_bradley_terry_anchor.py` minimises the documented Bradley-Terry objective with
  scipy in float64 (no `BradleyTerryEstimator` in the independent path), requires agreement within 1e-4, and
  pins the fitted values; a doubled ridge moves the fit 0.025, far outside the tolerance. No released
  tournament store exists offline in the repository, so the anchor is the synthetic minimiser the brief allows.
- **F4** — `score_delta(..., scores_a=, scores_b=, ties=)` and `explain` compute the deltas under the report's
  own tie rule (a `group_mean` class is credited its mean gain), the query's own id is dropped from the ideal
  under `drop_identical_ids`, and the display order (the protocol's where it has one, `doc_id_desc` for
  `group_mean`) is documented. The equality with the report's per-query values is stated for the RCP-gain path;
  the qrel fallback compares grades the protocol may round or map.
- **F5** — `select_opponents` raises a typed `DataError` when a query has no other calibrated document (instead
  of returning `[[doc_id]]`, which `judge` refused later) and takes `provisional_theta=`, in logits on the
  calibration's scale (the EAP of `score_documents`), mapped internally onto the query's Bradley-Terry scale;
  the default is still the query's median fitted ability.
- **F6** — `Calibration.load` validates every `thetas.parquet` row (`_theta_row`): a known `source`, a finite
  theta, a finite or missing SE; a corrupt row is a `DataError` naming the file and the document. `_write_json`
  turns `allow_nan=False`'s bare `ValueError` into a `DataError` naming the artifact.
- **F7** — a missing Bradley-Terry SE is written `None`, not 0.0 ("certain"); `bradley_terry` returns
  `float | None` and `calibrate` keeps `None` when no SE was formed.
- **F8** — the diagonal approximation is stated where `theta_se` is documented (`docs/concepts/calibration.md`,
  `ThetaRow.theta_se`, the coverage description) and the generated schemas carry it.
- **F10** — the `_projection.py` module docstring now says the refit is the live tournament's observations
  fitted cold (agreement to convergence tolerance, not bit for bit).
- **F11** — `RaschEstimator.add_criteria` refuses an unknown document (the rubric schedule never relied on the
  drop: its only caller pools document ids from the query's own units).
- The whole tree: `ruff check`/`ruff format --check` clean, `basedpyright` 0 errors, the full suite, the
  contract/docs suites, the test package and the gate all green; the CHANGELOG, snapshots and schemas are
  updated.

## Verification

**Round 1** — two fresh independent verifiers, DeepSeek-V4.1-flash (thinking xhigh), lenses A (correctness
against the brief) and B (regressions and hygiene), neither seeing the other's output.

- Verifier A: **VERDICT FAIL**. Findings: (1) major — the docs commit duplicated a code fence in
  `docs/concepts/calibration.md`, so the page's `Invalid windows`, `Documents without a tournament ability` and
  `In code` sections rendered inside a code block; (2) minor — the new "equals the report's per-query values"
  claim is false on the qrel-only path (rounding, exponential qrel gain, `drop_identical_ids`); (3) minor — the
  no-evidence warning claimed a ridge SE even when the written SE is `None`, and said "1 documents";
  (4) minor — `provisional_theta` was documented on the calibrated scale but compared on the Bradley-Terry
  scale; (5) minor — the all-invalid-query case was overclaimed in a docstring and the docs.
- Verifier B: **VERDICT FAIL**. Findings: (1) major — the same docs fence; (2) major — the `eval_score` MCP
  tool advertised `count_ndcg` but its input allowlist omitted `judgements`, so the tool could never satisfy
  it; (3) major — the same `provisional_theta` scale mismatch; (4) minor — the new test helpers hand-wrote
  `identity.json` instead of using `JudgementStore.claim`; (5) minor — `THETA_SOURCES` duplicated the `Source`
  literal.
- Fixes (commit `5faf7041`): removed both stray fences; added `judgements` to the MCP tool (and a test);
  `provisional_theta` is now the calibration's scale and is converted inside; the no-evidence list covers only
  the queries the fit reads and the message is accurate and pluralised; the equality claim is scoped to the RCP
  path in the docstrings, docs and schema; the test stores use `claim`; `THETA_SOURCES = get_args(Source)`.

**Round 2** — one fresh confirmation verifier (lens A+B), same model.

- **VERDICT FAIL**. Findings: (A) major — `explain` did not drop the query's own id from the ideal under a
  `drop_identical_ids` protocol, so its deltas were not the report's gap; (B) minor — the singular no-evidence
  message read "gives them"; (C) minor — the coverage description and the generated schemas still claimed the
  ridge SE unconditionally; (D) minor — the docs overstated `uncalibrated_documents` for a pure-tournament
  document of an all-invalid query; (E) minor — `score_delta` silently ignored a single score mapping.
- Fixes (commit `8a51517d`): the query id joins `excluded` in `explain` under `drop_identical_ids`;
  `score_delta` refuses one mapping alone; the singular pronoun is fixed; the coverage description and schemas
  are scoped and regenerated; the docs sentence is scoped to rubric-judged documents. Each fix has a failing
  test first (`test_explain_drops_the_query_id_from_the_ideal_when_the_protocol_says_so`,
  `test_score_delta_refuses_one_score_mapping_alone`).

**Round 3** — one fresh confirmation verifier (lens A+B), same model, on `8a51517d`.

- **VERDICT PASS**: all five round-2 fixes confirmed with its own reproductions; F1 bit-identity re-checked
  over all 24 permutations of four planned windows plus two real stores read in both orders; the full suite,
  lint, types, contract/docs and the anchors green; no new finding. One cosmetic note (the strict-refusal hint
  says "them" for a single document) is left in Open questions.

## Checks

Last commands (all on the final head `8a51517d` unless noted):

- `bin/gate lane/scoring-fixes` → **GATE: PASS** (slot 8): ruff-check 0, ruff-format 541 formatted,
  basedpyright 0, pytest 3301 passed/93 skipped, contract-docs 288 passed/52 skipped, mkdocs ok,
  test-pkg 570 passed/225 skipped, recipes "no failure outside the baseline (34 baseline failures remain,
  0 fixed)", vllm-pkg 1 passed, vllm-models 70 passed/7 skipped, leaderboards **1022 checks, 987 match,
  35 known deviations, 0 failed**, human study 67/67, external judges 82/82, public-names clean, clean.
- `heavy uv run --no-sync pytest tests/ -q -n 4 -p no:cacheprovider` → 3301 passed, 93 skipped.
- `uv run --no-sync pytest tests/contract tests/docs -q -p no:cacheprovider` → 288 passed, 52 skipped.
- `uv run --no-sync pytest tests/experiments tests/core/test_paper_labels.py
  tests/calibration/test_calibration_anchor.py tests/core/irt/test_bradley_terry_anchor.py -q` → 79 passed
  (anchors unchanged).
- `uv run --no-sync ruff format --check .` → 541 files formatted; `uv run --no-sync ruff check .` → clean;
  `uv run --no-sync basedpyright` → 0 errors, 0 warnings.
- `uv run --no-sync mkdocs build --strict -d <scratch>/site` → built.

Anchor bar, all unchanged: the paper per-query values (`tests/core/test_paper_labels.py`), the leaderboards
(1022/987/35/0), the rubric-only refit 1e-3 and the released-gain 1e-12 pins and the tournament 2PL recovery
(`tests/calibration/test_calibration_anchor.py`), and the refit-rule test — all pass with no assertion touched.

## Docs updated

- `docs/concepts/calibration.md` — `theta_se` is documented (diagonal SE, missing SE, no-evidence documents,
  the all-invalid query), the coverage layout line, the no-evidence section.
- `docs/concepts/metric.md` — Count-nDCG's derivation and its two routes.
- `docs/concepts/primitives.md` — `select_opponents`' `provisional_theta` scale and the no-opponent refusal.
- `docs/api/evaluate.md` — `count_gains`' derivation, the `explain` tie rule and display order, the new
  `score_delta` signature.
- `docs/reference/cli.md` — `eval score` computes Count-nDCG; `--judgements` on `eval score`; `--strict` also
  refuses a no-evidence document; the new warning code in the list.
- `skills/rcp-ndcg/SKILL.md` — the `--metrics count_ndcg --judgements` route.
- `CHANGELOG.md` — the entries below.

Grep commands run over `docs/`, `README.md`, `REPRODUCIBILITY.md`, `skills/`, `examples/`,
`experiments/**/*.md`, docstrings and CLI help texts: `git grep -n -i
"count_gains\|count_ndcg\|NO_VALID_TOURNAMENT_EVIDENCE\|no_tournament_evidence\|provisional_theta\|--judgements"
-- docs skills examples experiments README.md REPRODUCIBILITY.md`; `git grep -n "theta_se" -- docs
rcp-ndcg-core/src rcp-ndcg/src skills`; `git grep -n "uncalibrated_documents\|coverage.json" -- docs`;
`git grep -n "select_opponents" -- docs skills examples`. Every hit is either updated above or describes
behaviour that did not change.

## Open questions

- **The strict-refusal hint's pronoun**: `calibrate(..., strict=True)` says "judge a valid tournament window for
  them" even for one document (cosmetic; the warning message itself is correctly singular/plural). Left as
  reported by the round-3 verifier.
- **F2's remaining policy choice**: the no-evidence documents still enter the 2PL fit at `theta_BT = mean`, as
  the review described; the lane made that visible rather than excluding their rubric rows. A future lane could
  take the review's other option (exclude their rows and list them under `uncalibrated_documents`).
- **The qrel fallback is documented, not exact**: `explain`'s deltas compare the query's grades, so under
  `round_digits` or `qrel_gain="exponential"` they differ from the report's qrel-nDCG. Making them exact means
  mapping the grades through the protocol's gain and rounding in the fallback — out of this brief's scope.
- **`count_gains` is a helper, not a `Calibration` view**: a `Calibration.count_gains()` would need the counts
  in the artifact layout (a new file and field). The helper is the brief's allowed alternative.
- **F9's anchor is synthetic**: no released tournament store exists offline in the repository, so a change to
  the BT loss is caught by the independent minimiser and the pinned values, not by released data.

## CHANGELOG entry

Under `## Unreleased` (exact text):

### Public surface

- **Count-nDCG has its product path** (scoring-chain review F3): `rcp_ndcg.calibration.count_gains(judgements)`
  is the one derivation of the rubric-only gains (per window, per criterion, through `count_gain`), keyed as
  `Calibration.gains()` is; `evaluate(..., count_gains=...)` takes it, and `rcp-ndcg eval score --metrics
  count_ndcg --judgements STORE` (repeatable) is the command-line route. `ReportInputs` gains `judgements`, so
  `eval explain --report` re-scores a saved Count-nDCG report, and the `eval_score` MCP tool takes `judgements`
  too. The hint for missing count gains names the rubric windows and this derivation instead of the tournament
  store.
- **`rcp_ndcg.errors.WarningCode` gains `NO_VALID_TOURNAMENT_EVIDENCE`** (review F2): a document whose
  tournament windows are all invalid carries no comparison, and the fit says so instead of presenting the mean
  ability as judged. `CalibrationCoverage` gains `no_tournament_evidence_documents` (the
  `"<dataset>||<query_id>/<doc_id>"` list, in `coverage.json`).
- **`select_opponents(..., provisional_theta=)`** (review F5): the new document's own best guess, in logits on
  the calibration's scale (the scale of `score_documents`' EAP; the call maps it onto the query's Bradley-Terry
  scale); `None` (the default) is the query's median fitted ability, the behaviour so far.
- **`score_delta(..., scores_a=, scores_b=, ties=)`** (review F4): with the systems' score mappings and the
  protocol's tie rule the deltas are the report's metric (a `group_mean` class is credited its mean gain).

### Fixed

- **The calibration's refit is order-canonical** (scoring-chain review F1): a planned window
  (`window_seq=None`) has no schedule position, so the projections now order those by `record_id`; the same
  windows read in any store order give bit-identical Bradley-Terry abilities, standard errors, item parameters
  and fingerprint, as `docs/concepts/calibration.md` promises. Scheduled windows keep their positions, so no
  fitted number moved.
- **A document the tournament showed without a valid window is visible** (review F2): its ability stays the
  paper's (the query's mean, the ridge's standard error only when the query has other comparisons), and the fit
  lists it under `coverage.no_tournament_evidence_documents` and warns with `NO_VALID_TOURNAMENT_EVIDENCE`;
  `calibrate(..., strict=True)` (`calibration fit --strict`) refuses it. A missing Bradley-Terry standard error
  is written as `None` (review F7), not as 0.0 ("certain"), and `Calibration.load` validates the
  `thetas.parquet` rows (review F6): an unknown `source`, a non-finite theta or an infinite SE is a
  `DataError`; a JSON artifact that would hold a NaN names the file instead of raising a bare `ValueError`.
- **`explain` computes its gaps under the report's tie rule** (review F4): for a `group_mean` protocol the
  displayed order is document id descending, but the selection/ordering deltas now credit an equal-score class
  its mean gain, so they equal the report's per-query values; the display order is documented.
- **`RaschEstimator.add_criteria` refuses an unknown document** (review F11) as
  `BradleyTerryEstimator.add_comparison` does, instead of dropping the observation silently (the rubric
  schedule never relied on the drop).

### Changed

- **`select_opponents` refuses a query with no opponents** (review F5) with a typed `DataError` naming the
  query and the documents the calibration holds, instead of returning `[[doc_id]]` (not a window: `judge`
  refused it later).
- **The Bradley-Terry refit is documented as a cold refit** (review F10): it fits the live tournament's own
  observations from zero, so it agrees with the live fit to convergence tolerance, not bit for bit. The
  diagonal standard-error approximation is stated where `theta_se` is documented (review F8).

## Public surface changes

- New public function `rcp_ndcg.calibration.count_gains(judgements, *, dataset=None)` (exported in
  `rcp_ndcg.calibration.__all__`).
- New field `CalibrationCoverage.no_tournament_evidence_documents` (`schemas/calibration-coverage.v1.json`,
  `schemas/calibration-summary.v1.json`).
- New warning code `NO_VALID_TOURNAMENT_EVIDENCE` (`rcp_ndcg.errors.WarningCode`; `schemas/cli.v1.json`,
  `schemas/eval-report.v1.json`).
- `eval score` gains `--judgements` (repeatable) and accepts `--metrics count_ndcg`; `ReportInputs` gains
  `judgements` (`schemas/eval-report.v1.json`); the `eval_score` MCP tool takes `judgements`
  (`tests/contract/snapshots/mcp_tools.json`).
- `select_opponents(..., provisional_theta=None)` (keyword-only; `tests/contract/snapshots/python_api.json`).
- `score_delta(..., scores_a=None, scores_b=None, ties="doc_id_desc")` (not in `rcp_ndcg.eval.__all__`, so not
  pinned by the Python-API snapshot, like `bootstrap_interval`).
- `QueryExplanation.ScoreDelta`'s description (`schemas/query-explanation.v1.json`).
- No exit code changed. `tests/contract/snapshots/` and `schemas/` are regenerated and reviewed.

## Files outside scope

None. The lane touched its assigned modules (`rcp_ndcg.calibration/**`, `rcp_ndcg_core.irt/**`,
`rcp_ndcg.eval`'s `explain`/`evaluate`, `cli/eval.py`, `errors.py`) plus `mcp.py` (the `eval_score` input list
the new flag needs; reported here), tests, snapshots, schemas, docs, the skill page and the CHANGELOG. The
`handover/*` files in the commit range come from the pre-lane merge of `rfc-0001` (`0fbe4685`, lane
judge-gemma), not from this lane.

## For the next lanes

- `explain`'s qrel fallback can be made exact by mapping grades through the protocol's `qrel_gain` and
  `round_digits` and by applying the full excluded set (including `drop_identical_ids`, now done) to the
  ideal — a small, self-contained change if a lane wants the fallback to equal qrel-nDCG everywhere.
- If a `Calibration.count_gains()` view is wanted, the counts have to be written into the artifact layout
  (a new file and dataclass field, with a schema bump); the helper is deliberately layout-free.
- F2's no-evidence documents still inform the 2PL at `theta_BT = mean`; the review's alternative (exclude
  their rubric rows) remains open as a policy decision.
- The strict-refusal hint's "them" for one document is a one-line cosmetic fix.
