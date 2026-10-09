# Report RF-engine: two engine-side fixes ahead of the recipe-fix lane

**Status:** DONE.

## Commits

- `e142df3c` pplx: the pooling contract accepts vLLM's kernel-warmup dummy ids
- `a71ca3fc` vllm: an opt-in engine patch module backports the pooling-hang fix
- `6ea6e883` vllm patches: the docstrings name the opt-in the code implements
- `a8f76b65` Merge `rfc-0001` (`89e7a3b6`) into `lane/rf-engine`
- `6968bc4b` Merge `rfc-0001` (`28afb3b7`) into `lane/rf-engine`
- `a13fbeb6` vllm patches: no internal lane shorthand in the shipped docstrings
- `4868f609` Merge `rfc-0001` (`7013f28d`) into `lane/rf-engine` (the gated head)

## What changed

Per brief item:

1. **The pooling-hang backport (recipe-fix item 3).** New
   `rcp-ndcg-vllm/src/rcp_ndcg_vllm/patches/pooling_full_context.py` wraps
   `vllm.v1.core.sched.scheduler.Scheduler.__init__`; after the original `__init__` it stores
   `num_sampled_tokens_per_step = 0` exactly when `vllm_config.model_config.runner_type == "pooling"`
   (upstream vllm-project/vllm#48039, commit `e6fc81bc78`; at v0.31.0 `scheduler.py:146-148` reserves the
   slot and `:687-692` subtracts it). It logs one line when it applies and one inert line when the attribute
   is already 0 (a release that carries the fix), never touches a generate runner, and is idempotent (a
   wrapped-`__init__` marker). The module docstring names the upstream PR, the commit and the removal
   condition. `patches/__init__.py` defines the opt-in contract -- `RCP_NDCG_VLLM_PATCHES` (comma-separated),
   `PATCH_NAMES`, `opted_in_patch_names`, `apply_opted_in_patches`; an unknown name warns and installs
   nothing. The registration seam is `rcp_ndcg_vllm.models.register()` (the one `vllm.general_plugins`
   entry point vLLM loads in every engine process), which now calls `apply_opted_in_patches()`. Nothing in
   the package imports vLLM or torch at import time. No `--max-num-batched-tokens` flag exists anywhere.
2. **The pplx plugin's warmup (recipe-fix item 11a).** `models/pplx/pooling_core.py` now treats a token-id
   row whose first id is 0 as one of vLLM's engine dummies -- the kernel warmup's `list(range(prompt_len))`
   = `[0, 1]` at `vllm/v1/worker/gpu/warmup.py:256-257` (measured on the wave) and the all-zero pooler
   sizing grid at `vllm/v1/worker/gpu/pool/pooling_runner.py:178-183` -- and pools it as one span; a
   non-zero input without a role prefix is still the contract refusal. The test was red first with the
   measured ids.

## Verification

Round 1: two fresh verifiers on DeepSeek-V4.1-flash (lens A correctness against the brief; lens B
regressions and hygiene), launched in parallel, each told the other existed. Both ran the suites and their
own reproductions and returned **VERDICT: PASS**.

- Lens A findings: (1) minor -- two module docstrings claimed the serve path already renders a recipe's
  declared patches into `RCP_NDCG_VLLM_PATCHES`, but `recipe.py` has no `patches` field and `serve.py`
  only execs the engine (the env passes through by inheritance). Fixed in `6ea6e883`. (2) minor/residual --
  the dummy window widened from all-zero to leading-zero, so an out-of-contract `[0, 1, 2, ...]` pools
  silently where the base raised; reproduced, no product path reaches it (every client prepends a role
  prefix). Kept: the brief asks to validate real requests only, and the engine's warmup is the only
  legitimate leading-zero producer. Recorded in Open questions.
- Lens B findings: the same docstring finding (fixed); no blocker or major. Mutation tests: restoring
  `not any(token_ids)` reddened the new pplx test (`ValueError ... got [0, 1]`); dropping the pooling guard
  reddened the generate test (`assert 0 == 1`); dropping the wrapper marker reddened idempotence; dropping
  the opt-in gate reddened the no-opt-in test.
- Round 2: none. COMMON.md prescribes a second round only for a blocker or a major; both round-1 findings
  were minor.

## Checks

Red-first evidence (before the fixes):
- `pytest rcp-ndcg-vllm/tests/models/pplx/test_contract_core.py::test_pooler_kernel_warmup_ids_are_single_span`
  -> `1 failed ... ValueError: an input's first token id must be the query prefix ... got [0, 1]`.
- the patches suite -> `ModuleNotFoundError: No module named 'rcp_ndcg_vllm.patches'`.

Last runs on the merged tree (`4868f609`):
- `ruff format --check .` -> `542 files already formatted`; `ruff check .` -> `All checks passed!`;
  `basedpyright` -> `0 errors, 0 warnings, 0 notes`.
- narrow (`rcp-ndcg-vllm/tests/patches rcp-ndcg-vllm/tests/models/pplx`) -> `31 passed, 6 skipped`.
- vllm package -> `9 passed` and `71 passed, 7 skipped`.
- full root suite through the heavy wrapper -> `3269 passed, 93 skipped`.
- `tests/contract tests/docs` -> `287 passed, 52 skipped`; `mkdocs build --strict` -> built.
- `rcp-ndcg-test/tests` -> `570 passed, 225 skipped`.
- lane gate on `lane/rf-engine` -> `GATE: PASS`: every step `exit=0`; run_all leaderboards `1022 checks,
  987 match, 35 known deviations, 0 failed`; human study `67 checks, 67 match`; external judges `82 checks,
  82 match`; `public-names: clean`; tree `clean`.

## Open questions

- The first gate attempt failed in one shared integration slot's environment
  (`failed to remove directory .../torch/__pycache__: Directory not empty`); a retry passed on another slot,
  so the gate requirement is met, but that slot's venv may deserve the operator's attention.
- The dummy rule is "first id 0" rather than the tighter "all-zero or exactly `list(range(n))`" (verifier A's
  optional hardening). Kept as shipped because the engine's warmup ids are the only legitimate leading-zero
  rows and the brief's wording is "validate real requests only".
- `rfc-0001` advanced three times during the lane (merged `89e7a3b6`, `28afb3b7`, `7013f28d`); the gated head
  `4868f609` has `rfc-0001` as an ancestor. The last incoming merge was `handover/`-only.
- COMMON.md's round-2 rule (blocker or major only) was followed over the generic addendum's "run a new pair
  after fixing findings"; the fixed finding was a docstring correction with no code path, so no failing test
  could express it -- the verifier's reproduction is the evidence.

## CHANGELOG entry

Added under `## Unreleased`, `### Fixed`:

```markdown
- **An opt-in engine patch ships the pooling-hang backport** (`rcp_ndcg_vllm.patches`): the
  `pooling-full-context` patch backports vllm-project/vllm#48039 (commit `e6fc81bc78`) by wrapping
  `Scheduler.__init__`, so a pooling runner stores `num_sampled_tokens_per_step = 0` and a chunked prompt of
  exactly `max_model_len` tokens schedules its last token. The engine process applies it only when its
  `RCP_NDCG_VLLM_PATCHES` names it (a comma-separated list; `rcp-ndcg-vllm serve` passes the environment
  through), logs one line when it applies, one inert line when the running vLLM already carries the fix, and
  never touches a generate runner. Delete the patch when `engine.image` moves to the first vLLM release that
  carries `e6fc81bc78`.
- **The pplx contextual plugin serves on vLLM v0.31.0**: the pooling contract's role-prefix
  validation fired on the engine's own warmup input (measured `[0, 1]`, the kernel warmup's
  `list(range(decode_query_len + 1))` at `vllm/v1/worker/gpu/warmup.py:256-257`), so the engine died at
  startup. An input whose first id is 0 is now recognised as one of the engine's dummies -- the kernel
  warmup and the all-zero pooler sizing grid -- and pools as a single span, which vLLM discards; only a
  non-zero input without a role prefix is a contract refusal.
```

## Public surface changes

- New modules (internal to the package; `tests/contract/surface.py` pins only `rcp_ndcg_vllm.recipe` as
  public, so no snapshot or schema changed): `rcp_ndcg_vllm.patches` exports `PATCHES_ENV`, `PATCH_NAMES`,
  `opted_in_patch_names`, `apply_opted_in_patches`; `rcp_ndcg_vllm.patches.pooling_full_context` exports
  `PATCH_NAME`, `apply`.
- New engine-side environment variable: `RCP_NDCG_VLLM_PATCHES` (comma-separated patch names). No CLI
  command, flag, exit code or schema changed. No new dependency; the wheel stays pure `py3-none-any` and
  ships the two new files (verifier-checked).

## Files outside scope

none (the recipe schema `recipe.py` and the recipes were not touched).

## For the next lanes

- **recipe-fix**: the engine-side opt-in contract is `RCP_NDCG_VLLM_PATCHES`; the only name today is
  `pooling-full-context`. Add `serve.patches` to the recipe schema, validate its values against
  `rcp_ndcg_vllm.patches.PATCH_NAMES`, and render them into the engine environment in `serve.py` before
  `os.execvp`. Item 18's rule still decides the recipes (jina-embeddings-v5-text-small and
  zembed-1-embedding today; the rerank budgets sit far below their `max_model_len`). The module retires
  itself with an inert log line once `engine.image` carries `e6fc81bc78`.
- **pplx recipe**: the plugin now survives the engine's warmup at a lowered `max_model_len`; the recipe still
  needs item 11b's served length and the matching client budget.
- The e1-p5 measurement is reproducible from `vllm/v1/worker/gpu/warmup.py:256-257` (a pooling model has
  `decode_query_len == 1`, so the warmup row is `[0, 1]`).
