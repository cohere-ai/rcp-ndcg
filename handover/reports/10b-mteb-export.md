# Lane l10b — workstream 10 D: MTEB export, scoring inside mteb, the MTEB dataset writer

**Date:** 2026-10-09 · **Branch:** `lane/l10b` (worktree `wt-l10b`) · **Base:** `rfc-0001` @ `511698ef` · **Head:** `ec2fd578`
**Gate:** `bin/gate lane/l10b` → **PASS** (rev `ec2fd578`, after merging `rfc-0001` @ `89e7a3b6`, the l10a + w09 + l08-cat integration).

## Status

DONE. Every brief item is implemented, tested (failing test first) and gated on the merged tree; the round trips
run through mteb's own `RetrievalDatasetLoader` and `mteb.evaluate` offline; `run_all` is unchanged
(1022/987/35/0, 67/67, 82/82). Two verifier rounds ran (round 1: two lenses, both FAIL; round 2: one fresh
confirmation, PASS); every blocker and major finding is fixed and mutation-tested.

## Commits

| Hash | Subject |
|---|---|
| `a20f5def` | The MTEB dataset writer: what `push_dataset_to_hub` writes, plus rcp-ndcg's extras where mteb ignores them (decisions 28, 31) |
| `1b36e2e1` | `Rankings.save(format="mteb")`: mteb's `{Task}_predictions.json` from stored rankings |
| `ef49fedd` | Scoring stored rankings inside mteb: a `SearchProtocol` over stored `Rankings`, a `ModelMeta` from our model identity |
| `88e50d24` | The republishing converter: the published datasets re-laid in the MTEB push layout, eval split `test`, validated by mteb's own loader |
| `e0addc0d` | Docs and CHANGELOG for the MTEB export surface, the contract snapshots regenerated |
| `5c8a0c44` | The scoring round trip's tests ride with the route |
| `c8720c37` | The mteb-format and card hints read as prose (the wording gate) |
| `86d7b5ba` | Merge `rfc-0001` (l08-cat; the shared parallel plan's union) |
| `55e3a32d` | The subset-override hint reformatted |
| `f5e424e1` | The l05 report's two operator paths read as placeholders (the gate's public-names step) |
| `cde187a2` | Verifier round 1's findings: strict subset keying, merge on re-export, the converter's invocation, runnable docs snippets, the new tests |
| `5db19202` | Round-2 confirmation's minors: the save docstring's Raises clause, the merge documented publicly, the merge keeps the file's model meta verbatim |
| `ec2fd578` | Merge `rfc-0001` (lane l10a + the entry-point registry): the writer registers as the `rcp_ndcg.writers` `mteb` entry point, the shims give way to the real fields |

## What changed

- **`Rankings.save(format="mteb")`** (`data/rankings.py`): the `{Task}_predictions.json` of mteb's
  `_save_task_predictions` — `{"mteb_model_meta": {model_name, revision}, subset: {split: {qid: {did: score}}}}`,
  one file per task in a caller-named folder. Every query with a non-empty qrels dict must be ranked (a missing
  one is refused, naming it; verified against the run); a ranked query without qrels is dropped — declared policy,
  because mteb raises on a result for a query that has no qrels; no empty dicts; at most 1,000 documents per query
  (mteb's own `top_k`), ordered by core's `rank_by_score` (ties by document id descending). An existing file is
  merged exactly as mteb's own writer merges (the (subset, split) written replaces theirs; the file's other
  splits, subsets and its `mteb_model_meta` stay). TREC run files stay.
- **Scoring inside mteb** (`eval/mteb/stored.py`, re-exported from `eval/mteb/__init__.py`):
  `stored_rankings_model(rankings, meta)` wraps stored `Rankings` as mteb's duck-typed `SearchProtocol` — the
  served scores are the asked queries only (mteb raises on a result without qrels), restricted to the task's
  `top_ranked` pool when it has one, capped at `min(top_k, 1000)` with ties by document id descending; a query
  the run did not rank returns `{}` (mteb scores it 0 — the same semantics our evaluator reports an unranked
  labelled query with). The scores are keyed on the task's `hf_subset` through `Rankings.resolve_dataset`: a run
  of another subset is refused, never served (nanobeir subsets share query ids, where serving them reads as
  plausible, wrong numbers). `model_meta(name, revision, **fields)` builds mteb's `ModelMeta` from our model
  identity — the required fields the caller declares, the rest unknown (`None`); a test pins the defaults
  against mteb's required-field list. `mteb.evaluate` over the wrapped model writes its own predictions file and
  genuine `TaskResult` files in mteb's `ResultCache` layout
  (`results/{org__model}/{revision}/{Task}.json` + `model_meta.json` + `run_settings.jsonl`), ready for
  `submit_results`.
- **The MTEB dataset writer** (`data/io/mteb.py`, registered as the `rcp_ndcg.writers` `mteb` entry point):
  configs `{s-}corpus` (`id`, `title`, `text`), `{s-}queries` (`id`, `text`, `instruction` only when a query
  carries one), `{s-}qrels` (`query-id`, `corpus-id`, `score` as int64 — named `qrels`, not `default`) and
  `{s-}top_ranked` (`query-id`, `corpus-ids`), one parquet shard per config at
  `{config}/{split}-00000-of-00001.parquet`, and a README whose `configs:` front matter is what
  `load_dataset` — and through it mteb's `RetrievalDatasetLoader` — reads the directory with. `card=` (a mteb
  `TaskMetadata` or its fields) renders the card from mteb's own template. rcp-ndcg's extras ride only where mteb
  ignores them: the calibrated `gain`/`theta` columns on the qrels (mteb's `select_columns` drops them) and the
  `{s-}excluded` config, with the exclusions also folded out of `top_ranked` (out of the corpus when the data has
  no pool — mteb's BRIGHT derivation). A grade that is not a whole number is refused, naming the pair (the
  written column is int64; mteb's loader casts it to int32 at load, where a fractional value fails); a non-finite
  grade is refused likewise; media are refused like the BEIR writer. A suite dataset writes every subset's
  configs into one directory under one README; `write_dataset(subset=)` on a suite is refused (later parts would
  overwrite earlier ones). After the l10a merge the writer reads the real `Dataset.subset`/`split` and
  `Document.title` (the private `getattr` shims are removed).
- **Fractional grades** (decision 28): refused on export unless the given grade is an integer; the continuous
  signal travels in the `gain`/`theta` columns.
- **The republishing converter** (`tools/republish_mteb.py`): loads each published repository
  (`rcp-ndcg-nanobeir`, `rcp-ndcg-bright`, `rcp-ndcg-trecdl`, `rcp-ndcg-vidore-v3`) at the revision the checks
  were published against (loaded by file path, so the documented script invocation works), writes every subset
  with the writer at the eval split `test` (not `train`), and validates each written directory with mteb's own
  `RetrievalDatasetLoader` against the loaded `Dataset` (integer qrels, queries and their instructions, corpus,
  pool). Pushes nothing — the owner pushes, with the move to a Hugging Face organisation. The written card comes
  from the repository's published task metadata (mteb's template); the current hand-written usage cards are not
  reproduced.
- **Round trips** (in the `[mteb]` test environment, offline): the writer's output loaded by
  `RetrievalDatasetLoader` equals our `Dataset` (single-subset and named-subset configs, exclusions folded);
  stored `Rankings` scored through `mteb.evaluate` give our integer nDCG@10 under the nanobeir protocol,
  including the tie rule (tied scores break by document id on both sides; verified on a tie block crossing the
  k=10 boundary).

## Verification

**Round 1** — two fresh `cohere-oss-v2/glm-5-3-flash:xhigh` verifiers, run in parallel, both waited for:

- *Lens A (correctness), VERDICT: FAIL.* Verified the layouts, tie rule, caps and TaskResult layout against the
  mteb 2.21.6 source, and found: (1, major) `StoredRankings` silently served another subset's rankings when the
  run named exactly one other dataset (reproduced end-to-end through `mteb.evaluate` → ndcg 0.0); (2, major) the
  converter's documented invocation crashed with `ModuleNotFoundError: No module named 'experiments'`;
  (3, major) the how-to's scoring and save snippets called a non-existent `Rankings.load`; plus minors (int64
  vs int32 in the refusal rationale; `write_dataset(subset=)` collision on suites; NaN grade raising raw
  `ValueError`; save overwriting instead of merging; the `min(top_k, 1000)` cap note; the `_validate` gaps; the
  all-zero-qrels divergence; the ownership one-liners). → All fixed: strict `hf_subset` keying (test + red-then-green
  by reverting to the old source), file-path import + subprocess test, self-contained runnable snippets (the
  network blocks verified Hub-reachable), typed NaN refusal, suite `subset=` refusal, merge semantics matching
  `abstask.py`, `_validate` extended, both divergences documented.
- *Lens B (regressions/hygiene), VERDICT: FAIL.* Confirmed ruff/basedpyright clean (10 errors, all proven
  pre-existing on the base in the same venv), contract snapshots consistent without `--update-snapshots`, the
  full root suite green, R30 held, scope creep all minimal-necessary, the private shims in the lane's own files.
  Mutation-tested 8 behaviours; found (1, major) the docs snippet regression (`Rankings.load` — same as A3) and
  (2, major) the untested cap/tie of `_of_query` (mutation survived the whole suite), plus minors: int64→int32
  prose (4 places), two untested documented refusals of `save`, the tie/cap order in three homes (now core's
  `rank_by_score` and `Rankings.top`), the `search` docstring overpromise, and no CLI test for
  `data convert --to mteb`. → All fixed: cap/tie tests through `StoredRankings.search` (mutation kills), refusal
  tests, the tie order delegated to `rank_by_score`, the dead check removed, the CLI test added, the docstrings
  aligned.

**Round 2 (confirmation, one fresh verifier, lenses A+B): VERDICT: PASS.** All nine round-1 findings confirmed
fixed — finding 1 red-then-green proven by reverting `stored.py` to the pre-fix commit; five further mutations
each turned the corresponding new test red; the int32 rationale verified against mteb's own loader source; the
network doc-snippet test genuinely ran (Hub reachable, HTTP 200). Three minors, all applied: the stale Raises
clause in `Rankings.save`'s docstring (the removed empty-subset refusal), the merge-on-re-export documented in
the public docstring, the how-to page and the CHANGELOG, and the merge keeping the file's `mteb_model_meta`
verbatim (exactly what `_save_task_predictions` does). All mutation probes reverted (`git status` clean).

## Checks

Last run, on the final tree (`ec2fd578`, `rfc-0001` merged):

- `bin/gate lane/l10b` → **GATE: PASS** — every step exit 0: ruff check, ruff format, basedpyright (0 errors),
  root suite 3296 passed/95 skipped, contract+docs 291 passed/51 skipped, mkdocs `--strict`, `rcp-ndcg-test`
  570 passed/225 skipped, the recipes baseline unchanged (34 known), vllm-pkg, vllm-models, run_all
  1022/987/35/0 · 67/67 · 82/82, public-names clean, tree clean.
- `heavy uv run --no-sync pytest tests/ -q -n 4 -p no:cacheprovider` → 3336 passed, 102 skipped (my worktree env,
  `[mteb]` extra installed).
- `uv run --no-sync pytest tests/eval/mteb/ -q` → 57 passed, 1 skipped (the network-marked one; also green with
  `RCP_NDCG_NETWORK_TESTS=1`).
- `uv run --no-sync pytest tests/contract -q` → 10 passed (snapshots regenerated once, the packaging entry point
  reviewed).
- `uv run --no-sync mkdocs build --strict -d <lane-scratch>/site` → OK.
- `RCP_NDCG_NETWORK_TESTS=1 uv run --no-sync pytest tests/docs/test_snippets.py -q` → 54 passed, 46 skipped
  (the mteb-integration network block included).
- Failing-test-first evidence (commands from the transcript):
  `pytest tests/eval/mteb/test_export.py` → `ModuleNotFoundError: No module named 'rcp_ndcg.data.io.mteb'`;
  `pytest tests/eval/mteb/test_predictions.py` → 10 failed (`ConfigError: format='mteb' needs ...` absent);
  `pytest tests/eval/mteb/test_scoring.py` → `ImportError: cannot import name 'model_meta'`;
  `pytest tests/eval/mteb/test_converter.py` → 3 failed (the tool did not exist). Each was then written green.
- Mutation evidence: tie-order flip in `_of_query` → `test_the_cap_and_its_tie_rule_apply_to_served_scores` red;
  merge removal → `test_a_second_export_of_the_same_task_merges_the_other_split` red; subset keying reverted to
  the old `dataset=None` path → `test_a_run_of_another_subset_is_refused_not_silently_served` red (also proven by
  the round-2 verifier reverting the file to the pre-fix commit).

## Checks — flake notes

- Two of the gate's `test-pkg` runs segfaulted (exit 139) at ~97% in the integration worktree — once near the
  run's end with no failing test named, once at `test_record_and_wave.py::test_wave_gives_each_slot_its_own_short_tmpdir`.
  Both times the same suite passed immediately after: 570/570 on the gate re-run and 570/570 in this worktree
  (three single-test probes, the file, and the full package). The crash is an environment flake in the shared
  `wt-int` venv (its pytest runs `-n 8` right before), not a test failure. The two passing gate runs on this
  branch and both local runs stand.

## Open questions

1. **Task definitions vs the owner's local mteb PR** — left open per the brief (no link was added): the
   published `rcp_ndcg_tasks.py` files still name split `train` and this repository's owner; the republished
   datasets carry split `test`. The definitions move when the operator supplies the PR.
2. **`rcp-ndcg-vidore-v3` is not republishable by the converter**: its corpus is page images; the text layout
   cannot hold them and the writer refuses media with a typed `ConfigError` (the same refusal as the BEIR
   writer). Media export into mteb's image columns is deferred with the media decisions; the converter reports
   the repo as failed and continues with the others.
3. **All-zero-qrels queries** score differently by construction: mteb's mean keeps them at 0, our qrel-nDCG drops
   them (undefined without a positive grade). Documented on both sides (`stored.py` docstring, the how-to page);
   every query of the shipped suites carries a positive label. Flagging in case a future dataset does not.
4. **`model_meta` leaves everything undeclared as unknown (`None`)** — mteb's leaderboard review expects declared
   model cards; the caller declares what it knows. The `framework` default is `[]` (mteb requires a list).
5. **The converter's card** renders from the published task metadata (first subset, name-sorted, deterministic)
   with `dataset.path` pointed at the target repository; the current hand-written usage cards of the published
   repositories are not reproduced — the owner may merge them before pushing.
6. **The writer writes one shard per (config, split)** (`-00000-of-00001.parquet`), the `push_dataset_to_hub`
   name for a one-shard split; a sharded variant (N > 1) is not implemented and no shipped dataset needs one.

## CHANGELOG entry

Under `## Unreleased → ### Public surface` (verbatim):

> - **The MTEB dataset writer** (the `mteb` writer of `WRITERS`, `data convert --to mteb`):
>   `rcp_ndcg.data.io.mteb.MtebWriter` writes exactly what mteb's `push_dataset_to_hub` writes -- configs
>   `{s-}corpus` (`id`, `title`, `text`), `{s-}queries` (`id`, `text`, `instruction` only when a query carries
>   one), `{s-}qrels` (`query-id`, `corpus-id`, `score` as int64) and `{s-}top_ranked` -- one parquet shard per
>   config at `{config}/{split}-00000-of-00001.parquet`, and a README whose `configs:` front matter is what
>   `load_dataset` (and through it mteb's `RetrievalDatasetLoader`) reads the directory with; `card=` (a mteb
>   `TaskMetadata` or its fields) renders the card from mteb's own template. rcp-ndcg's extras ride only where
>   mteb ignores them: the calibrated `gain`/`theta` columns ride on the qrels, and the exclusions travel in the
>   `{s-}excluded` config and are folded out of `top_ranked` (out of the corpus when the data has no pool). A
>   grade that is not a whole number is refused (mteb's loader casts the int64 `score` column down to int32,
>   where a fractional value fails), naming the pair: export integer grades, keep the continuous signal in
>   `gain`/`theta`. A suite dataset writes every subset's configs into one directory under one README.
> - **`Rankings.save(format="mteb")`**: the `{Task}_predictions.json` of mteb's `_save_task_predictions`, from
>   stored rankings (`task=`, `qrels=`, `model_name=`, `model_revision=`, `split=`, `system=`). Every query with
>   a non-empty qrels dict must be ranked (a missing one is refused, naming it); a ranked query without qrels is
>   dropped (mteb raises on a result for a query that has no qrels); no empty dicts; at most 1,000 documents per
>   query (mteb's own cap), ties by document id descending. An existing file is merged the way mteb's own writer
>   merges: the (subset, split) written replaces theirs, the file's other splits, subsets and its
>   `mteb_model_meta` stay.
> - **Scoring stored rankings inside mteb** (`rcp_ndcg.eval.mteb`, the `mteb` extra):
>   `stored_rankings_model(rankings, meta)` wraps stored `Rankings` as mteb's `SearchProtocol` -- the served
>   scores are the asked queries only, restricted to the task's `top_ranked` pool when it has one, capped at
>   `top_k` with ties by document id descending -- and `model_meta(name, revision, **fields)` builds mteb's
>   `ModelMeta` from our model identity, the required fields the caller declares, the rest unknown. `mteb.evaluate`
>   over the wrapped model writes its own predictions file and genuine `TaskResult` files in mteb's `ResultCache`
>   layout (`results/{org__model}/{revision}/{Task}.json` with `model_meta.json` and `run_settings.jsonl`), ready
>   for `submit_results`; the integer `ndcg_at_10` equals our `qrel_ndcg` under the suite's protocol (the tie
>   rules agree).
> - `tools/republish_mteb.py` re-lays the published rcp-ndcg datasets in the writer's exact layout with the eval
>   split `test`, validates each written repository with mteb's own `RetrievalDatasetLoader`, and pushes nothing
>   (the owner pushes, with the move to a Hugging Face organisation).

(The writer is now registered as the `rcp_ndcg.writers` `mteb` entry point — the registration home moved with
lane l10a's entry-point seam; the CHANGELOG wording above predates that merge and describes the same writer.)

## Public surface changes

- `rcp_ndcg.data.io`: the `mteb` writer (`MtebWriter`) registered in the `rcp_ndcg.writers` entry-point group
  (packaging snapshot: `entry_points/rcp_ndcg.writers: ['beir', 'jsonl', 'mteb']`).
- `RankingsFormat` gains `"mteb"`; `Rankings.save` gains the keyword parameters `system`, `task`, `qrels`,
  `model_name`, `model_revision`, `split` (the python_api snapshot records them).
- `rcp_ndcg.eval.mteb.__all__` gains `StoredRankings`, `model_meta`, `stored_rankings_model`
  (`tests/contract/snapshots/python_api.json` regenerated; the snapshot also records `MTEB_MAX_DOCS`'s module is
  internal, so no schema change).
- No CLI command or flag changed (the `data convert` help wording follows the entry-point registry, per l10a's
  wording); no exit code changed.

## Files outside scope

- `rcp-ndcg/pyproject.toml` — the one `[project.entry-points."rcp_ndcg.writers"] mteb = ...` line (the
  registration home moved there with lane l10a; the brief's "one line in the WRITERS table" lands as this line).
- `tests/data/test_io_contract.py` — one expectation line (`available_writers()` gains `"mteb"`).
- `handover/reports/05-layout.md` — two operator paths read as placeholders (the gate's public-names step failed
  on the file's two pre-existing unbaselined hits; content-preserving).
- `docs/data.md`, `docs/reference/cli.md`, `skills/rcp-ndcg/SKILL.md`, `rcp-ndcg/src/rcp_ndcg/cli/data.py` —
  sentences describing the new writer/predictions surface (the CLI one was superseded by l10a's entry-point
  wording in the merge).
- `handover/specs/parallel-08-09-10.md` — the plan copy the brief asks for (later unions with the operator's).

## Docs updated

- `docs/how-to/mteb-integration.md` — the writer's config table, the round trips, the scoring route and the
  converter; both export snippets run offline, both evaluate snippets network-marked.
- `docs/data.md` — the ingest list names `--to mteb`.
- `docs/reference/cli.md` — the `data convert` line (l10a's entry-point wording).
- `skills/rcp-ndcg/SKILL.md` — the scoring-inside-mteb and writer pointers.
- Grep commands run: `git grep -n -i "mteb" docs/ skills/ README.md REPRODUCIBILITY.md`,
  `git grep -n "data convert\|--to beir\|--to jsonl" docs/ skills/ examples/`, `git grep -n "writers" skills/`,
  `git grep -n "Rankings.save\|load_rankings" docs/ skills/ examples/` — every hit that the change made false
  was updated in the same lane.

## For the next lanes

- **l10c** (title+body join, instructions through the 09 stages): the writer now carries `Document.title` into
  the corpus column and `Query.instruction` into the queries column; the mteb-side text join stays mteb's own
  (`(title + " " + text).strip()` in its dataloader) — nothing in this lane joins at write time. If formatting
  enters `rcp-fp/3`, the writer is unaffected (it writes fields, never formatted text).
- **The converter run** (`python tools/republish_mteb.py --out <dir>`, network): the reads are pinned via
  `experiments/fetch_data.py`'s revisions; expect `rcp-ndcg-vidore-v3` to fail loudly (media, see open question
  2) and the other three to validate green.
- **QA (07)**: `tests/eval/mteb/` runs in CI's `mteb` job and in the `gated` job (env `dev data mteb docs`);
  no new `importorskip` rows were needed (`mteb` was already in `DEPENDENCY_GATES`).
- The stored-rankings model refuses a subset mismatch on purpose; if a caller legitimately needs a run re-keyed
  to another subset name, the fix is one `to_pandas().assign(dataset=...)` on their side — say so in the docs if
  it comes up.
