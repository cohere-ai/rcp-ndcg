# Report CI: the first GitHub CI run on the M3 work is green (lane ci-fix)

**Status:** DONE. Gate: PASS (`GATE: PASS` in `gates/8d2ed18c/SUMMARY`, re-run twice; see Verification).

## What landed
- **gated tests job — recipe root (CI run 37822235213, failure 1).** The job installed
  `./packages/rcp-ndcg-vllm` non-editable, but the recipes live beside the package in the checkout (outside
  the wheel until the layout move makes them package data), so `iter_recipes()` found no recipe root at
  `.venv/lib/python3.12/recipes`. The job now installs `uv pip install mcp -e ./packages/rcp-ndcg-vllm`
  (`.github/workflows/ci.yml`), pinned by a new packaging test
  (`tests/docs/test_packaging.py::test_every_ci_install_of_the_serving_recipes_package_is_editable`) that
  fails on any non-editable install of the package, including line-continued install commands. The
  `DEPENDENCY_GATES` opening check still passes (it drops `-e` when tokenizing the install line).
- **gated tests job — MCP SDK tool set (failure 2).** The test's expected set was the wrong side: it
  predated the `eval_score` and `run_start` tools, while the server's registration (13 tools) is pinned by
  `test_the_read_only_tools_mirror_their_commands`, `test_the_builtin_loop_speaks_json_rpc` and
  `test_run_start_is_served_with_its_config_as_the_one_required_input`, and both server implementations
  answer `tools/list` through the same `rcp_ndcg.mcp.tool_manifest()`. The SDK test now expects
  `READ_ONLY | {"eval_score", "run_cancel", "estimate", "run_start"}` and additionally pins the SDK listing
  equal to `tool_manifest()` (the "same results" contract).
- **mteb tasks job — bright subset names (failure 3).** Not the installed mteb (2.21.6 ships none of the
  tasks) and not a regression in our code: all four published dataset repos were renamed (bright commit
  `ed768cfc` "Rename the tasks to ...RCPReranking, add the open-corpus view") — `_TASK_METADATA` is now
  keyed by published task name and the files ship a new `_SUBSETS` block mapping subset aliases to those
  names, so `get_tasks("bright", ["aops"])` was refused as unknown. Fixed in
  `src/rcp_ndcg/eval/mteb/__init__.py`:
  - new public `task_subsets(source)` reads the published `_SUBSETS` as data (``{}`` for the older
    subset-keyed files); a malformed block is a typed `DataError` with a hint;
  - `get_tasks` accepts each subset under its own name and under its published task name (same task either
    way, metadata-identical; naming both is refused as a repeat), an unknown name lists both spellings, and
    `names=None` builds the published default view (the alias map's `task` column, so the ViDoRe files'
    8 OCR variants stay out of the default 8);
  - `_make_task` gives the retrieval view its published name (`RCPReranking` -> `RCPRetrieval`, type
    `Retrieval`) for the task-keyed files; older files keep the legacy `.retrieval` suffix;
  - the network test asserts the new published names (`BrightAopsRCPReranking`) for both spellings; two new
    offline tests (a synthetic task-keyed file with `_hub_text` monkeypatched) pin the resolution and the
    alias map parsing without the Hub.

## Commits
- `352d2290` The first GitHub CI run is green: the gated job installs rcp-ndcg-vllm editable, the MCP SDK
  test expects the server's 13 tools, and get_tasks resolves the published files' renamed tasks through
  _SUBSETS
- `8d2ed18c` Verifier round 1: task_subsets refuses a malformed _SUBSETS with a typed error and a hint, the
  empty-names hint names task_subsets, and the CI editable-install pin catches line-continued installs

Base: `rfc-0001` tip `881505fd` (the branch was cut from it; the merge is a no-op — "Already up to date").

## Verification
- **Failing first.** Reproduced all three failures locally in the worktree venv prepared exactly as the CI
  gated job (`.github/scripts/cpu-env.sh dev data mteb docs`, `uv pip install mcp ./packages/rcp-ndcg-vllm`
  non-editable): recipe `RecipeError` at `.venv/lib/python3.12/recipes`, the MCP tool-set diff (missing
  `eval_score`, `run_start`), and `ConfigError: unknown subsets ['aops']` with
  `RCP_NDCG_NETWORK_TESTS=1` — each matching `ci-fix/scratch/ci-failed.log`. The new tests fail on the
  unfixed code: `task_subsets` ImportError; `get_tasks("bright", ["aops"])` raises the same `ConfigError`
  (also shown by the round-1 verifiers via a scratch worktree at `881505fd` with PYTHONPATH shadowing); the
  packaging pin fails on the pre-fix `uv pip install mcp ./packages/rcp-ndcg-vllm` line and on a
  line-continued non-editable install.
- **Verifier round 1** (GLM-5.3-flash:xhigh, fresh context, run in parallel, awaited in-turn):
  - Lens A (correctness): **VERDICT: PASS**. All three originally failing tests green; each new/changed test
    red on the unfixed code (scratch worktree at 881505fd); a 30-check independent repro against the real
    published files of all four suites confirmed every `get_tasks` semantic (both spellings, refusals before
    network, old-file behavior, vidore OCR exclusion, retrieval naming). Minors: pin test only matched
    single-line installs; raw `KeyError` on a malformed `_SUBSETS`; ViDoRe OCR retrieval name cosmetic.
  - Lens B (regressions/hygiene): **VERDICT: PASS**. Full suite 3310 passed/59 skipped under `heavy`;
    contract+docs 274 passed; rcp-ndcg-test 124 passed; ruff clean; basedpyright 0 errors on the dev-only
    set (the 6 extras-set errors are pre-existing in untouched code and absent on the types job's env);
    4 product-code mutations each turned the new tests red and were reverted byte-exactly; CHANGELOG
    claims checked against the CI log and the live Hub. Minors: `task_subsets` untyped on malformed
    `_SUBSETS`; empty-names hint omits `task_subsets`.
  - Round 1 found no blocker or major, so no round 2. The three agreed minors were fixed (failing tests
    first; the ViDoRe OCR one is answered, not changed: no published OCR retrieval name exists and the
    product has no `text_only`, so the legacy suffix is the only defined behavior there).
- **Gate** (`bin/gate lane/ci-fix` on the merged tree): PASS — ruff check/format, basedpyright 0 errors,
  pytest 3263 passed/83 skipped, contract+docs 274 passed, mkdocs strict, test-pkg 124 passed, vllm-pkg 334
  passed/209 skipped, run_all leaderboards 1022 checks/987 match/35 known deviations/0 failed, human study
  67/67, external LLM judges 82/82, checkout clean. One intermediate gate run failed on
  `vllm-pkg::test_stage1_validation_runs_a_skip_list_recipe_on_the_offline_fake`
  (`OSError: [Errno 39] Directory not empty`, a TemporaryDirectory teardown race while the wt-l05 lane was
  running heavily); the identical tree passes the job twice (334 passed) — flake, not a regression.

## Checks (last run of each)
- `uv run --no-sync ruff format --check .` — 513 files already formatted; `ruff check .` — all passed.
- `uv run --no-sync basedpyright` (dev-only env, as CI's types job) — 0 errors.
- `heavy uv run --no-sync pytest tests/ -q -n 4 -p no:cacheprovider` — 3310 passed, 59 skipped.
- `uv run --no-sync pytest tests/contract tests/docs -q` — 274 passed, 52 skipped.
- `uv run --no-sync pytest packages/rcp-ndcg-test/tests -q` — 124 passed.
- `RCP_NDCG_NETWORK_TESTS=1 uv run --no-sync pytest tests/eval/mteb -q` — 7 passed.
- `uv run --no-sync mkdocs build --strict -d .../scratch/site` — built.
- `bin/gate lane/ci-fix` — GATE: PASS.

## Docs updated
- `docs/api/evaluate.md`: `names` lists subsets or their published task names.
- `docs/how-to/mteb-integration.md`: the retrieval view is always built locally (mteb ships only the
  reranking names); each subset is accepted under both spellings.
- Grep run: `git grep -n -i "task_subsets\|_SUBSETS"` and `git grep -n "get_tasks("` and
  `git grep -n "iter_recipes\|default_recipes_root"` over docs/, README.md, skills/, examples/,
  experiments/ — no stale text left (the `get_tasks("nanobeir", ["NanoFiQA2018Retrieval"])` snippets still
  work: the new files alias exactly those names; `mkdocs build --strict` and `tests/docs` green).

## CHANGELOG entry
Under `## Unreleased`:

- `### Public surface`: "`rcp_ndcg.eval.mteb.task_subsets(source)` reads a published suite file's
  `_SUBSETS` alias map (each subset's published task name, read as data; `{}` for the files that predate
  the task-name keys) -- the lookup `rcp_ndcg.eval.mteb.get_tasks` resolves its `names` through."
- `### Fixed`: "**The first GitHub CI run is green** (run 37822235213): the gated job installs
  `rcp-ndcg-vllm` editable (the recipes live beside the package in the checkout, so the non-editable install
  left the recipe-backed case validation without a recipe root; pinned by a packaging test); the MCP SDK
  round-trip test's expected tool set follows the server's registration -- the SDK server and the built-in
  loop list the same 13 tools through `rcp_ndcg.mcp.tool_manifest()`, and the test's expected set predated
  `eval_score` and `run_start`; and `rcp_ndcg.eval.mteb.get_tasks` accepts the published files' renamed
  tasks: the suites' current releases ("Rename the tasks to ...RCPReranking, add the open-corpus view") key
  `_TASK_METADATA` by published task name and ship the alias map `_SUBSETS`, so a subset name (`aops`) was
  refused as unknown. A subset and its published task name both resolve now (the same task either way;
  naming both is refused as a repeat), the retrieval view carries its published `...RCPRetrieval` name (the
  older files keep the `.retrieval` suffix), and without `names` the published default view is built (the
  ViDoRe files' OCR variants stay out of it)."

## Public surface changes
- New: `rcp_ndcg.eval.mteb.task_subsets(source) -> dict[str, str]` (in `__all__`;
  `tests/contract/snapshots/python_api.json` regenerated; no CLI, exit-code or schema change).

## Files outside scope
- None. (`.github/workflows/ci.yml` is named by the brief's failure 1.)

## For the next lanes
- The layout move (l05) will make the recipes package data; when it does, drop
  `test_every_ci_install_of_the_serving_recipes_package_is_editable`'s premise (the editable pin) with it,
  and the gated job's `-e` can go if wanted.
- `task_subsets` + the alias map also make the nanobeir docs snippet work against the renamed files without
  a docs change; when a future mteb ships the PR-5516 tasks in-tree, `get_tasks` returns mteb's own at the
  same revision (unchanged behavior, now on the new names).
- Open question for the owner: the ViDoRe OCR table entries (a `text_only` view the product does not expose)
  get the legacy `.retrieval` suffix in retrieval mode; there is no published name to match, so none was
  invented.

## Open questions
- None blocking. The two convention notes above (l05 interaction; OCR retrieval naming) are recorded for
  the next lanes.
