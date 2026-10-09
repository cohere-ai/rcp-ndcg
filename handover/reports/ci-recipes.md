# Report `ci-recipes`: CI runs the recipe pins offline and the Hub-backed ones on every push (lane ci-recipes)

**Status:** DONE. Gate: **PASS** on `04302258` (the final code tip; this report is the commit after it). A
dispatched CI run on the lane branch is the operator's (the standing rules forbid a lane push), asked under
Open questions.

## Commits

| Commit | Subject |
|---|---|
| `63105d25` | The recipe gate classifies by fixture: only the Hub-backed items are network-marked |
| `97806664` | A recipe-tests job runs the Hub-backed recipe tests on every push and PR |
| `078a7562` | fetch_tokenizer writes atomically and keeps a pinned file a concurrent worker wrote |
| `04302258` | Verifier round 1: the set holds only downloading fixtures, the embeddinggemma pins run offline, and the atomic write is pinned |
| `a6c67da8` | Merge branch 'rfc-0001' into lane/ci-recipes (rfc-0001 tip `446765e5`, merged cleanly) |

## What changed (per brief item)

1. **Classification by fixture (one place).** `rcp-ndcg-test/tests/recipes/conftest.py` declares
   `NETWORK_FIXTURES` -- the fixtures whose construction downloads a Hub tokenizer file or the checkpoint's
   chat template (`tokenizer`, `tokenizer_dir`, `zerank_tokenizer`, `chat_template`,
   `_seed_checkpoint_template`, `recipe_cpu`, `checkpoint`, `snapshot`, `recipe`, `pairs_path`) -- and the
   collection hook marks an item `network` (and, with `RCP_NDCG_NETWORK_TESTS` unset, skips it with the
   reason naming the variable) only when it requests one of them or carries an explicit
   `@pytest.mark.network`; an explicit `@pytest.mark.offline` opts an item out. The blanket marking and the
   `OFFLINE_MODULES` special case are gone. The `offline` marker is registered in
   `rcp-ndcg-test/pyproject.toml`.
2. **The explicit overrides.** Twenty-one test functions in six modules fetch their tokenizer inside the body
   (a module helper such as `_tokenizer_file`/`tokenizer_file`/`_skip_unless_hub_reachable`, no fixture) and
   now carry `@pytest.mark.network` themselves -- sixteen from the first pass, the five qwen3-embedding ones
   from the verifier round-1 follow-up (they used `hub_cache`, which only sets a cache directory and does
   not download, so `hub_cache` left the declared set). `test_network_gate.py` drives the hook with a
   stand-in item and pins the rule, the declared set and both overrides offline.
3. **The test-pkg job now runs the offline pins.** Recipe directory, offline: **32 passed / 348 skipped
   before, 207 passed / 180 skipped after**; whole `rcp-ndcg-test/tests` offline: **616 passed / 349
   skipped before** (derived: the rest of the tree is untouched, so before = after - the recipe-directory
   delta of 175 passed and 168 skipped) **/ 791 passed / 181 skipped after** (measured in the gate's
   `test-pkg` step). The embeddinggemma module's ten pins run offline because its checkpoint-template seed
   fixture is no longer autouse (only the two harness tests request it).
4. **The `recipe-tests` job** (`.github/workflows/ci.yml`): runs on every push and PR (no `if:`),
   `cpu-env.sh dev` (its `[hf]` extra and torch's jinja2 are what the tests need), `RCP_NDCG_NETWORK_TESTS=1`,
   `RCP_NDCG_VLLM_TOKENIZER_CACHE` and `HF_HUB_CACHE` on one `actions/cache` path
   (`~/.cache/rcp-ndcg-vllm-tokenizers`, keyed by the recipes' bytes), `-q -n 4 -p no:cacheprovider
   -o faulthandler_timeout=120`, a 45-minute job bound, public Hub only and no secret, and the clean-checkout
   step the other jobs use.
5. **The proof the tests can fail (local; run links are the operator's).** Mutating one contract pin
   (`EXPECTED_SERVE["max_model_len"]` in `test_qwen3_embedding.py`) reds the offline pin
   (`serve.max_model_len == 32768, expected 40960`); mutating one mutant (the `[:-1]` drop in
   `test_ctxl_rerank_v2_instruct_multilingual.py::test_dropping_the_trailing_anchor_segment_reddens_the_template_check`)
   reds the network mutant for all three variants. Both mutations were reverted byte-exactly (the files hash
   to their pre-mutation state).
6. **A real concurrency bug the new job exposed, fixed.** With `-n 4` against one tokenizer cache, two
   workers fetch the same file at once: the plain `write_bytes` let one worker's download clobber the
   other's pinned file (and be read half-written), which failed the sha256 pin on the first full network run.
   `_served.fetch_tokenizer` now writes through `tempfile.mkstemp` + `os.replace`, retries once on a pin
   mismatch (a pinned file a concurrent worker wrote is kept), and unlinks its temporary in a `finally`.
   `test_served_fetch.py` pins the race, the atomicity (a reader thread never sees a partial file) and the
   no-leftover property offline.
7. **The baseline workaround.** No page under `docs/` mentions the recipe-test baseline or the 34 known
   failures; the only mentions are historical lane reports under `handover/reports/` (kept as history). The
   gate's own `recipes` step now reports `0 baseline failures remain, 34 fixed`, so the operator-side
   baseline file has served its purpose.

## Verification

- **Failing first.** The `fetch_tokenizer` race test fails on the pre-fix code with
  `the downloaded model/tokenizer.json does not match the pinned sha256` (the corrupt late download clobbers
  the pinned file); the new atomicity test fails when `_served.py`'s temporary-file block is replaced by a
  plain `write_bytes` (`a reader saw a partial download`). Both were reverted byte-exactly after the check.
- **Verifier round 1** (DeepSeek-V4.1-flash:xhigh, fresh context, two lenses run in parallel and awaited
  in-turn; the two verifiers never saw each other's output):
  - Lens A (correctness): **VERDICT: PASS**. Independently reproduced the offline/network counts, the
    fixture-closure classification, the eight `fetch_tokenizer` cases, the CI job's shape and a full
    cold-cache network run (385 passed, 1 skipped in 409 s), and the two mutations red. Findings: F1
    `hub_cache` was not itself a downloading fixture (fixed: the five body-download tests now carry the
    marker and `hub_cache` left the set); F2 the embeddinggemma pins were gated on an autouse seed fixture
    (fixed: the seed is no longer autouse); F3 a failed write could leak the temporary (fixed: `finally`
    unlink); F4 a note on the trigger wording (the repo's push-to-main + PR convention, no change).
  - Lens B (regressions/hygiene): **VERDICT: PASS**. Ruff, basedpyright 0, the full root suite (3540
    passed/101 skipped), `rcp-ndcg-test/tests` 780/191 (before the round-1 fixes), contract+docs 304/55,
    mkdocs strict, the cold-cache network job green, four mutations red in a scratch worktree, the three
    action pins verified against the real tags, no public-surface change. Findings: (1) a residual
    corrupt-download-first race could still raise in a worker (fixed: one retry, then the pinned file is
    kept); (2) the atomic write was not pinned by a test (fixed: the reader-thread test, shown red on the
    non-atomic variant); (3) the temporary leak on a failed write (fixed as F3). The known NFS flake
    (`TestPublishConcurrency::test_concurrent_writers_never_share_a_temp_file`) failed in the lane's root
    run and passed in the verifier's and in isolation -- an environment artifact, not a regression.
  - Round 1 found no blocker and no major, so no round 2 (COMMON: round 2 only for a blocker or a major).
    All agreed minors were fixed in `04302258`.
- **Gate.** `bin/gate lane/ci-recipes` on `04302258` (the final code tip): **GATE: PASS** -- ruff check and
  format, basedpyright 0 errors, root pytest 3540 passed/101 skipped, contract+docs 304 passed/55 skipped,
  mkdocs strict, test-pkg 791 passed/181 skipped, recipes `no failure outside the baseline (0 baseline
  failures remain, 34 fixed)`, vllm-pkg 40 passed, vllm-models 71 passed/7 skipped, run_all leaderboards
  1022 checks/987 match/35 known deviations/0 failed, human study 67/67, external LLM judges 82/82,
  public-names clean, checkout clean. The gate's `test-pkg` step measured the offline suite at **791 passed,
  181 skipped** (from 616/349 at the base).

## Checks (last run of each; the final tip `04302258` unless noted)

- `uv run --no-sync ruff format --check .` -- 569 files already formatted; `ruff check .` -- all passed.
- `uv run --no-sync basedpyright` -- 0 errors, 0 warnings, 0 notes.
- `uv run --no-sync pytest rcp-ndcg-test/tests/recipes -q -n 4 -p no:cacheprovider -o faulthandler_timeout=120`
  (offline) -- 207 passed, 180 skipped.
- `RCP_NDCG_NETWORK_TESTS=1 RCP_NDCG_VLLM_TOKENIZER_CACHE=<scratch> HF_HUB_CACHE=<scratch>/hub uv run --no-sync
  pytest rcp-ndcg-test/tests/recipes -q -n 4 -p no:cacheprovider -o faulthandler_timeout=120` -- fresh-cache
  cold run 385 passed/1 skipped in 373 s on the pre-round-1 tip; the changed modules re-run fresh 24 passed;
  the gate's `recipes` step green on the final tip.
- `uv run --no-sync pytest rcp-ndcg-test/tests -q -p no:cacheprovider` -- 791 passed, 181 skipped (the
  gate's `test-pkg` step on the final tip).
- `uv run --no-sync pytest tests/contract tests/docs -q -p no:cacheprovider` -- 304 passed, 55 skipped.
- `timeout 3600 heavy uv run --no-sync pytest tests/ -q -n 4 -p no:cacheprovider -o faulthandler_timeout=120`
  -- 3539 passed, 101 skipped, 1 NFS flake (passes in isolation and in the gate).
- `uv run --no-sync mkdocs build --strict -d <scratch>/site` -- built.
- `bin/gate lane/ci-recipes` -- GATE: PASS (SUMMARY at revision `04302258`).

## Open questions

- **The dispatched CI run and the run links.** The standing rules forbid a lane push, so the mutation proof
  is local: mutate `EXPECTED_SERVE["max_model_len"]` in `rcp-ndcg-test/tests/recipes/test_qwen3_embedding.py`
  (the offline pin) and drop the `[:-1]` slice in
  `test_ctxl_rerank_v2_instruct_multilingual.py::test_dropping_the_trailing_anchor_segment_reddens_the_template_check`
  (the network mutant); the first reds `test-pkg`, the second `recipe-tests`. Please dispatch `ci.yml` on the
  lane branch (green) and, if the run links are wanted, on a scratch branch carrying those two one-line
  mutations (red in the two jobs), then paste the links into this report.
- **The `zembed` reference skip.** The network job reports 1 skip:
  `test_reference_render_ids_match_the_model_own_remote_code` needs
  `RCP_ZEMBED_1_EMBEDDING_REFERENCE_PYTHON` (a python with torch/transformers/sentence-transformers), which
  the `dev` environment does not carry; the job is green, and the test runs in the GPU waves' reference
  environment. Say the word if the job should provision that interpreter.
- **A transient media-control skip.** One cold-cache full run skipped
  `test_qwen3_vl_reranker.py::test_the_media_stage_holds_the_client_to_the_card` ("the checkpoint's own
  pixel budget is not readable" after three 1-second retries); later runs passed it. That retry loop is the
  test's own resilience, not the job's; if it recurs, widening the retry is a one-line change in that test.
- **A pre-existing stale-cache path.** `fetch_tokenizer` returns a cached file whose hash does not match the
  pin when the download fails with `OSError` (the deliberate "better than an error" branch). It predates
  this lane; the verifiers noted it as a residual risk for an offline run with a corrupt warm cache.
- **The operator-side baseline.** `bin/recipes-step` (outside the repository) still compares against the
  baseline ids file; with all 34 fixed it reports `34 fixed`. The file can be retired by the operator.

## CHANGELOG entry

None. No public surface changed (`git diff 446765e5..HEAD -- tests/contract/snapshots schemas` is empty), and
no documented behaviour changed: the recipe tests' gating is not described in `docs/`, and the pages that
mention network-marked recipe checks remain true.

## Public surface changes

None. No CLI, exit code, schema, `__all__` or snapshot changed.

## Files outside scope

- `rcp-ndcg-test/pyproject.toml` -- the `offline` marker registration (one line), so the documented override
  does not warn.
- `rcp-ndcg-test/tests/recipes/test_served_fetch.py` -- new offline test module for the `fetch_tokenizer`
  concurrency fix.
- `rcp-ndcg-test/tests/recipes/test_embeddinggemma_2.py`, `test_qwen3_embedding.py` -- the round-1
  classification follow-ups (the seed fixture is no longer autouse; five tests carry the explicit marker).

## Docs updated

No page under `docs/` changed: nothing there describes the recipe-test gating, the baseline or the new job,
and no claim became false. The rule's home is the `conftest.py` docstring (rewritten with the change), the
gate test's docstring, and the two `ci.yml` comments. Greps run over `docs/`, `README.md`,
`REPRODUCIBILITY.md`, `skills/`, `examples/`, `experiments/`, `mkdocs.yml` and the package READMEs for
`RCP_NDCG_NETWORK_TESTS`, `RCP_NDCG_VLLM_TOKENIZER_CACHE`, `network-gated`, `network-marked`, `recipe-tests`,
`NETWORK_FIXTURES`, `fetch_tokenizer` and `baseline`; the only hits are the still-true
`rcp-ndcg-test/README.md` case-validation sentence and the historical reports.

## For the next lanes

- The recipe gate's rule is one declared set plus the two marker overrides: a new Hub-backed fixture belongs
  in `NETWORK_FIXTURES`; a test that downloads in its body carries `@pytest.mark.network`; a test that
  requests a Hub fixture but needs no download carries `@pytest.mark.offline`. `test_network_gate.py` pins
  the set, so a rename fails there instead of silently running offline.
- The `recipe-tests` job is the first CI job to share one tokenizer cache across `-n 4` workers. Any new
  cache-writing helper should write through a temporary and rename (and re-check the pin) like
  `fetch_tokenizer`; a plain write is the bug this lane hit.
- The gate's recipe step still runs the whole network suite with the operator's shared cache, so the
  baseline comparison can now be dropped.
