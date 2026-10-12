# Lane names — the internal design-doc citations leave the shipped tree

**Lane:** `lane/names` (worktree outside the operator's checkout). **Base:** `rfc-0001` at `3217b5ed`, merged
with its tip `9d119c5e` as `68c1882d` before the gate. **Gate:** `timeout 7200 bin/gate lane/names` on
`68c1882d` (slot 3) — **GATE: PASS**, every step exit 0 (the SUMMARY is in section 4). No GPU work; nothing
pushed; `rfc-0001` was not merged into.

## 1. Status

Done, in the brief's three items plus the supervisor-approved case-insensitive widening:

1. **The design-document citations are gone from the shipped tree.** 38 files carried one: the brief's 35
   (30 `qwen3-reranker` case YAMLs, four test modules and the retrieval config) plus three the case-insensitive
   guard surfaced (the ctxl family's two source lines and its two regenerated goldens; the two lowercase
   comments in `tests/inference/test_types.py` are extra lines in a file already among the 35). Every citation
   is replaced with the public behavioural statement it stood for, never a renamed id:

   - `rcp-ndcg/src/rcp_ndcg/retrieval/config.py`: the docstring keeps "role configs selected by `api`", the
     `(RFC-0001 section 7.2)` citation dropped.
   - `tests/test_no_inprocess_models.py`: "the package carries no model that loads weights" stands on its own.
   - `tests/support/test_serve.py`: the phase rule is stated directly — consecutive steps that call the same
     served engines share a phase, steps that call no served engine run in a phase without one, and the paper
     run becomes four phases.
   - `tests/inference/test_types.py`: the identity rule (one `tokenizer_sha256` key, never the tokenizer's
     name) and the two section comments (the wire types; the serve-by-role types).
   - `tests/runs/test_pipeline.py`: the retrieve and rerank step identities hold the candidates config's
     content payload, never its runtime fields.
   - the 30 case YAMLs: "gated at the declared abs 0.02 (the stage-2 gate for probability-scale scores: at most
     0.02 for 99% of documents, 0.05 for all)".
   - the ctxl family's 1b/6b sources: "that path was removed when the in-process path left the package" /
     "removed when the in-process path left the package" instead of the merge reference.
   - `CHANGELOG.md`: the `qwen3-reranker-0.6b` line loses the `(on branch lane/l5-packaging at this HEAD; ...)`
     parenthetical; the public `experiments/paper/rerankers/reference/qwen3.py` path stays.

   **The generator home: none.** No in-tree script emits the 30 case notes (the cases lanes' builders lived
   outside the repository; the files were committed by hand in `c64c80de`, "Add the qwen3-reranker generated
   strata cases (30 files, 10 per recipe)"). They were fixed in place with one uniform replacement,
   `RFC-0001 stage-2 gate` -> `stage-2 gate`, applied to all 30 at once (the same wording).

2. **The CHANGELOG parenthetical is dropped** (the one `lane/` reference in the shipped tree), keeping the
   public path and the surrounding probability/bfloat16/batch statement.

3. **The rule is self-enforcing.** The repo-wide hygiene module
   (`rcp-ndcg-test/tests/recipes/test_recipe_hygiene.py`) gains
   `test_no_shipped_file_cites_an_internal_design_or_lane_id`: it scans every tracked file (`git ls-files`)
   outside `handover/` and `experiments/` (the scaffolding, deleted at release) for
   `\bRFC[- ]?\d{3,4}\b|\blane/`, case-insensitively — a lowercased citation of the same id is the same id.
   The docs wording test forbids those ids in Markdown and the gate's public-names step forbids operator
   paths; this scan is the one home for the rest of the shipped tree. Red-first: the guard failed on the
   pre-existing 36 hits and on a planted citation before the sweep, and passes after it (section 4).

## 2. Commits

| commit | what |
|---|---|
| `c2608426` | the shipped-tree guard, red-first on the pre-existing citations |
| `0f596819` | the brief's 35 files swept; the CHANGELOG parenthetical dropped |
| `6831d882` | the case-insensitive sweep: the two `test_types` comments, the ctxl sources, the two goldens regenerated |
| `68c1882d` | merge of `rfc-0001` (`9d119c5e`) before the gate |

This report is the commit after `68c1882d` (handover Markdown only). No co-author lines, no attribution,
nothing pushed.

## 3. What changed (files)

- `rcp-ndcg-test/tests/recipes/test_recipe_hygiene.py`: the module docstring, `DESIGN_AND_LANE_IDS`,
  `SCAFFOLDING`, `_repository_root`, `_shipped_files` and the new test.
- `rcp-ndcg-test/cases/qwen3-reranker-{0.6b,4b,8b}/gen-*.yaml`: 30 files, one note phrase each.
- `rcp-ndcg/src/rcp_ndcg/retrieval/config.py`, `tests/test_no_inprocess_models.py`,
  `tests/support/test_serve.py`, `tests/inference/test_types.py`, `tests/runs/test_pipeline.py`.
- `rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/ctxl-rerank-v2-instruct-multilingual/family.yaml` (the 1b/6b
  sources) and `rcp-ndcg-test/tests/recipes/golden/ctxl-rerank-v2-instruct-multilingual-{1b,6b}.json`
  (regenerated with `--update-goldens`; only the sources text moved).
- `CHANGELOG.md` (the historical `0.0.1` line), this report.

## 4. Verification

The guard before the sweep (red on the real citations):

```
uv run --no-sync pytest rcp-ndcg-test/tests/recipes/test_recipe_hygiene.py::test_no_shipped_file_cites_an_internal_design_or_lane_id -q
E  CHANGELOG.md:2588: derived from `experiments/paper/rerankers/reference/qwen3.py` (on branch lane/l5-packaging at ...
E  rcp-ndcg-test/cases/qwen3-reranker-0.6b/gen-mixed-lengths.yaml:86: ... (the RFC-0001 stage-2 gate ...
E  ... (36 hits: the 30 case YAMLs, the four test modules, the retrieval config and the CHANGELOG line) ...
1 failed
```

The planted citation (appended to a clean tracked file, then reverted):

```
# The planted citation: RFC-0001 section 9.9.
E  rcp-ndcg-test/src/rcp_ndcg_test/generators.py:436: # The planted citation: RFC-0001 section 9.9.
1 failed
```

After the brief's 35-file sweep the case-sensitive form was green, and the case-insensitive form (the
supervisor's call) was red on the four remaining files; after that sweep: `3 passed`. The whole-tree scan is
empty: `git grep -in -E '\brfc[- ]?[0-9]{3,4}\b|\blane/' -- ':!handover/' ':!experiments/'` -> 0.

The affected root modules: `tests/inference/test_types.py tests/runs/test_pipeline.py tests/support/test_serve.py
tests/test_no_inprocess_models.py` — **239 passed**. The guard module — **3 passed**. The family goldens were
regenerated (`--update-goldens`, **49 passed**) and their diff is the sources text only. The case loader —
`rcp-ndcg-test/tests/test_cases.py` **76 passed**. Full root suite through the heavy slot: **4134 passed, 155
skipped** before the merge (the gate re-ran it on the merged tree: **4136 passed, 155 skipped**);
`tests/contract tests/docs` — **413 passed, 107 skipped**; the test package — **1140 passed, 229 skipped**;
`ruff format --check`, `ruff check`, `basedpyright` (0 errors) and `mkdocs build --strict` clean.

The gate's recipes flake check: the first gate run's `test_zembed_1.py::test_stage1_on_cpu_passes_anchors_and_render`
hit its 300 s per-test timeout under a load average of ~134; the same test passes alone on the lane tree
(`1 passed in 353.60s`, the session including collection). It is not touched by this lane.

### Gate

First run on `68c1882d` (slot 4) — **GATE: FAIL**, two environmental flakes under load (three gates plus the
GPU waves): `test-pkg exit=139` (SIGSEGV at `test_wavegroups.py::test_the_real_catalog_has_exactly_two_engine_images`,
the known flake) and `recipes exit=1` (`test_zembed_1.py::test_stage1_on_cpu_passes_anchors_and_render`, the
300 s timeout). Re-run after the load dropped (one other gate), same revision:

```
rev lane/names = 68c1882d (slot 3)
ruff-check exit=0 All checks passed!
ruff-format exit=0 624 files already formatted
basedpyright exit=0 0 errors, 0 warnings, 0 notes
pytest exit=0 4136 passed, 155 skipped in 56.34s
contract-docs exit=0 413 passed, 107 skipped in 41.71s
mkdocs exit=0 INFO    -  Documentation built in 2.96 seconds
test-pkg exit=0 ================ 1140 passed, 229 skipped in 753.42s (0:12:33) =================
recipes exit=0 recipes: no failure outside the baseline (0 baseline failures remain, 0 fixed; pytest exit 0)
vllm-pkg exit=0 50 passed in 4.75s
vllm-models exit=0 102 passed, 8 skipped in 59.07s
run_all exit=0   external_judges  ok
leaderboards: 1022 checks, 987 match, 35 known deviations, 0 failed
human study: 67 checks, 67 match, 0 known deviations, 0 failed
external LLM judges: 82 checks, 82 match, 0 known deviations, 0 failed
public-names exit=0 public-names: clean (0 baselined hits remain)
clean exit=0 clean
GATE: PASS
```

The commit after the gate changes only this report (handover Markdown); the guard, `public-names` and
`mkdocs --strict` were re-run on the final tip and stay clean.

## 5. Open questions

1. **The guard's RFC pattern would also flag a future citation of a public IETF RFC by number.** No such
   citation exists in the tree, and the docs wording test has carried the same pattern for Markdown; if a
   legitimate external RFC citation ever lands, the pattern (one regex) is the place to narrow.
2. **The guard reads tracked files only.** An untracked file is not shipped (and the gate's `clean` step
   forbids untracked files anyway); a future workflow that ships an untracked artifact would be outside this
   net.

## Docs updated

- No docs page changed: the sweep touched docstrings, test comments, case notes, recipe sources and the
  CHANGELOG; no page describes them.
- Greps run before the last commit: `git grep -in -E 'rfc[- ]?0001|lane/l5-packaging' -- ':!handover/'
  ':!experiments/'` (0 hits); `git grep -n "stage-2 gate" -- docs README.md skills` (one hit,
  `docs/how-to/adopt-a-model.md:128`, still true).

## CHANGELOG entry

None: no public name, flag, exit code or schema changed. The one CHANGELOG edit is the historical `0.0.1`
line for `qwen3-reranker-0.6b`, which loses the branch parenthetical (the public reference path stays); it is
not a new entry.

## Public surface changes

None. `tests/contract/snapshots/` and `schemas/` untouched.

## Files outside scope

- `rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/ctxl-rerank-v2-instruct-multilingual/family.yaml` and its two
  goldens: approved by the supervisor when the case-insensitive guard surfaced the lowercased citations (the
  brief's pattern was uppercase).

## For the next lanes

- The hygiene guard scans the whole tracked tree minus `handover/` and `experiments/`: any new shipped file
  must avoid `RFC[- ]?\d{3,4}` (case-insensitively) and `lane/`.
- The 30 `qwen3-reranker` case notes have no in-tree generator; a re-derivation must keep their public wording
  or commit a generator beside the other cases' `make_*.py`.
