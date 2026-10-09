# Report: lane `l08-judges` — the judge role, ten judge recipes, `judge check` and the R5 pre-flight

Lane `l08-judges` (workstream 08 B and D plus the Gemma 4 judges), 2026-10-09. Base: `rfc-0001` after the
families, overrides and embeddinggemma-2 merges (`247c3d53`); the lane merged `rfc-0001` at `446765e5` (lane
l10d) before its final gate.

## Status

DONE. All eight brief items are delivered; the gate passed on the final head (see Checks). Nothing is
GPU-validated: every judge recipe is `status: unverified` until the E2 judge wave (decision 1).

## Commits

| Commit | Subject |
|---|---|
| `44f3d621` | The judge role in the recipe schema, and the ten judge recipes as eight families |
| `8c69dd48` | Judges resolve through the recipe path; the judge conformance probe |
| `f82534ed` | The T4 scenarios, examples, docs and paper configs point at the judge recipes |
| `aba2bbbe` | The gpt-oss recipe declares the paper's sampling divergence; the JG report answers the owner |
| `e5455d3e` | Merge `rfc-0001` into `lane/l08-judges` (lane l10d: the MTEB export follows PR #5516) |
| `be2d032d` | Round-1 verifier fixes: the TREC-DL cap divergence, the URL-less fake judge, the untested guards |

## What changed (per brief item)

1. **The judge role** (decisions 15, 18, 19). The recipe schema gains `role: judge` with `client.api: chat`
   (the recipe-facing spelling of the judge role's registered `openai_chat` adapter; both names resolve to one
   adapter and neither enters an identity) and **no `reference`** (`Recipe.reference` is `None` for a judge and
   required for every other role; the equivalence harness refuses a judge by name through the new
   `rcp_ndcg_test.equivalence.reference.reference_of`). A judge recipe's `client` block **is** rcp-ndcg's judge
   config, validated by `rcp_ndcg.judging.JudgeConfig` when rcp-ndcg reads `recipe: <id>` (R30: never a second
   model). The serve block carries the vLLM flags: the tensor parallel size from `resources.gpus`, the context
   length, and -- through `serve.extra_args`, the declared verbatim-flag mechanism -- the reasoning parser, the
   quantisation and the KV-cache dtype, with `serve.limit_mm_per_prompt` for the media caps. `extra_args` joined
   the declared per-variant `serve` fields (the flash-next and 26B NVFP4 variants differ there) and `licence`
   became a per-variant fact (the NVFP4 and FP8 releases of one model are licensed differently). `schema_version`
   is unchanged (`"1"`), the recipe and family JSON Schemas are regenerated, and rcp-ndcg's compatibility check
   (`RECIPE_SCHEMA_VERSIONS`) already refuses an unknown major.
2. **Ten judge recipes as families** (decision 34): the six catalog judges (`qwen3.5-397b-a17b-nvfp4`,
   `gpt-oss-120b`, `qwen3.6-27b-fp8`, `qwen3.8-27b-fp8`, `qwen3.8-flash-next-nvfp4`,
   `qwen3.8-flash-next-fp8`) plus the four Gemma 4 judges (`gemma-4-12b-it`, `gemma-4-26b-a4b-it`,
   `gemma-4-26b-a4b-nvfp4`, `gemma-4-31b-it-nvfp4`), grouped into eight family directories per the catalog's
   decision-34 grouping. Every variant is pinned, `status: unverified`, renders through
   `rcp-ndcg-vllm serve <id> --dry-run`, and has its per-variant golden snapshot
   (`rcp-ndcg-test/tests/recipes/golden/<id>.json`). Decision 41 corrected the catalog's TP2 serve blocks: all
   four Gemma judges are `resources.gpus: 1`, `qwen3.8-flash-next-nvfp4` is TP1, `qwen3.8-flash-next-fp8` TP2,
   `qwen3.5-397b-a17b-nvfp4` TP2, the rest TP1, and every family's notes state the B200 and H100 arithmetic
   plus the H100 `serve --set resources.gpus=<n>` override.
3. **Presets become recipes** (D): `--judge <recipe-id>`, `--judge recipe:<id-or-path>` and
   `judge: recipe:<id>` resolve through the same mapping as `reranker: recipe:<id>`
   (`JudgeConfig.load` / `judge_config_data`; `JudgeConfig.base_url` is optional for a recipe, the URL arriving
   from a run's `serve:` block or `RCP_NDCG_ENGINES`, and `JudgeConfig.recipe` records the recipe's identity in
   the judgement store). The four self-hosted presets are removed; `gpt5_hosted` stays a vendor profile;
   `--judge ./file.yaml` keeps working for any OpenAI-compatible endpoint. The T4 scenarios name
   `recipe:`/`fallback_recipe:` (their judge command and client block come from the recipe; `judge.config` may
   carry runtime fields only, refused against the product's own role declaration), the examples and paper
   configs point at recipes, and the docs describe the three routes.
4. **`rcp-ndcg judge check --judge <config|recipe|fake>`**: a short conformance probe that sends the shipped
   tournament and rubric prompts over two fixed windows and reports, per stage, whether the answer schema was
   accepted, whether the answer parsed with the stage's own parser, and whether the endpoint reported a
   reasoning channel beside the answer (`separated`/`absent`, with the advisory). Public:
   `rcp_ndcg.judging.check_judge`, `JudgeCheck`, `JudgeCheckReport`, the command's
   `rcp-ndcg.judge-check-report.v1` schema; tested on the fake engine and a mock endpoint (schema refused,
   unparseable answer, reasoning channel present/absent).
5. **R5 pre-flight (freeze-critical)**: the comparison table is below. The four text/vision shipped prompts are
   byte-identical to the paper's submission prompts; the two video prompts have no submission counterpart and
   are declared new families; the rendered chat-template input is identical between vLLM v0.31.0 and the
   paper's SGLang image for all ten judges. The pre-flight also found two run-level sampling divergences
   (gpt-oss and qwen3.6), both now declared in the recipe notes for the owner to decide.
6. **`docs/concepts/judges.md`** documents the three routes, the shipped catalog with the decision-41 GPU
   counts, `judge check`, and the multi-variant family-directory rule; the contract snapshots, exported
   schemas and CHANGELOG are current.
7. **The Gemma 4 judges**: the owner's eight answers are recorded in `handover/reports/JG-judge-gemma.md`
   (appended section) and implemented in the recipes (one 26B family, thinking off by default with thinking on
   as a declared client choice, the catalog's sampling convention with the cards' values an opt-in, 131072
   context, `wire: frames` video only, no bf16 31B, backends recorded in E2, audio out of scope).
8. **Decision 41**: every judge recipe's notes state both GPU classes' arithmetic; the recipes declare the
   one-B200 shape (TP1 where the weights fit, else the smallest TP that fits a B200 with useful cache) and the
   H100 shape as a documented `serve --set` override; the paper's TP4 x DP2 is recorded as history in the
   qwen3.5-397b and gpt-oss notes.

## Verification

**Round 1** (two fresh verifiers, `deepseek-v4-1-flash:xhigh`, on `e5455d3e`; they did not see each other's
output):

- **Lens A (correctness against the brief): VERDICT FAIL** — one major, two minor.
  - MAJOR: the paper's TREC-DL runs drove `qwen36_27b_fp8` with `strategy.max_response_tokens: 12288`, a
    per-call override over the client's 16384, undeclared in the qwen3.6 recipe while the identical gpt-oss
    case was declared. **Fixed** in `be2d032d`: the divergence paragraph is in
    `recipes/qwen3.6-27b/family.yaml`, the golden regenerated, and the table below records it.
  - MINOR: `docs/concepts/judges.md` promised that the judge route "takes the id in the path's family" for a
    multi-variant directory (the loader refuses it). **Fixed**: the docs now say the route reads a
    single-variant directory and a multi-variant family is refused by name.
  - MINOR: "a script reads `ok`" was ambiguous under `--json` (the envelope's `ok` is true while `data.ok` is
    false). **Fixed**: the docs name `data.ok`.
  - Reproduced clean, claim by claim: the ten recipes (ids, role, api, no reference, gpus, dry-run, goldens,
    tokenizer SHAs against the real checkpoints, B200/H100 arithmetic re-derived); the five judge routes plus
    `judge: recipe:<id>`; `base_url` refusal; the `chat` alias identity; `judge check` (fake, recipe+fake URL,
    `--set` refusal, 400-endpoint exit 0) and that it uses the product's own render/parse path; the presets'
    removal and the scenarios' recipe-derived judge config with the CONTENT override refused; the R5 sha256s
    and the vLLM-vs-SGLang render (strings and token ids); docs truth elsewhere. The one `rcp-ndcg-test`
    failure it saw under `-n 4` was traced to the parallel lens-B verifier's concurrent mutation of a tracked
    recipe file and did not reproduce in four clean runs.
- **Lens B (regressions and hygiene): VERDICT PASS** — seven minor findings, no blocker, no major.
  - The judge/reference validator survived both full suites under mutation; the T4 slot-gpus and CONTENT
    guards and `reference_of` had no test caller. **Fixed**: five new red-first tests
    (`rcp-ndcg-test/tests/test_recipe.py`, `test_e2e_scenarios.py`); each was mutation-checked red (validator
    disabled → both reference tests fail; both guards disabled → both scenario tests fail).
  - `FakeJudge`/`from_config` raised `IndexError` on a URL-less config (reachable now that `base_url` is
    optional). **Fixed** in `judging/_fake.py` with the typed `ValueError`; red-first test in
    `tests/judging/test_fake_judge.py`.
  - `experiments/paper/README.md` still said the package ships judge presets. **Fixed**.
  - `run start/resume --judge` help omitted the recipe form. **Fixed**; snapshots regenerated.
  - `judge_config_data`'s docstring overpromised the install line for a bare id without rcp-ndcg-vllm.
    **Fixed**: the docstring states the bare-id behaviour and the `recipe:`-form install line.
  - The gpt-oss recipe cited `handover/reports/08bd-judge-recipes.md` before it existed; this report commit
    resolves it.
  - Refuted concerns (all passed): no pre-existing recipe golden changed (10 additions, 0 modifications) and
    `rcp-fp/3`'s inputs are additive, so no recorded corpus key moved; the corpus-key recomputation matches
    `tests/conformance/stale.json`'s declared set; R30 (the harness imports the product's `JudgeConfig` and
    `declared_roles`); one home per new concept; generated files current; docs/CHANGELOG truthful; hygiene and
    standing rules clean.

**Round 2** (one fresh confirmation verifier, lens A+B, `deepseek-v4-1-flash:xhigh`, on `be2d032d`):
**VERDICT PASS**.

- F1 fixed and re-verified against the paper's configs: all three TREC-DL deploy configs
  (`msmarco_trecdl_dl_tournament.yaml`, `_rasch.yaml`, `_dl20_tournament_restart.yaml`) declare
  `strategy.max_response_tokens: 12288`, the deploy strategies build `GenerationParams(max_tokens=...)` and
  `model_copy` it over the client's, so the override wins; the qwen35 runs use the non-deploy strategies and
  carry no override (the primary judge matches the paper); no other paper judge has an undeclared run-level
  override. The golden diff is the notes string only.
- Both docs fixes reproduced (the multi-variant directory refused by name; the failed probe's
  `{"ok": true, "data": {"ok": false}}`).
- The `_fake.py` fix is red-first: with the hunks reverted in a scratch copy, `tests/judging/test_fake_judge.py`
  fails with the `IndexError` at `_fake.py:210`.
- All five new tests kill their mutations (each validator disabled in a scratch copy → the five tests fail
  with `DID NOT RAISE`); the worktree was left untouched and clean.
- README/help/snapshots verified; the `judge_config_data` docstring verified with `rcp_ndcg_vllm` blocked
  (bare id → unknown config name with the `recipe:` hint; `recipe:` → the install line); the regression sweep
  green (3554 / 301 / 631, ruff and basedpyright clean); in `be2d032d` only `qwen3.6-27b-fp8.json` changed
  (notes) and the ten judge goldens' fingerprints and tokenizer SHAs are byte-identical to `e5455d3e`.
- It found three pre-existing stale citations outside this lane's files, listed under Open questions.

## R5 pre-flight

### The shipped prompts against the paper's submission prompts

Byte comparison of `rcp-ndcg/src/rcp_ndcg/judging/prompts/*.txt` against the paper's submission prompts (the
paper's prompt tree, read-only, checked 2026-10-09):

| Shipped prompt | Submission prompt | Bytes | sha256 (first 8) | Verdict |
|---|---|---|---|---|
| `tournament.txt` | `tournament_listwise.txt` | 5994 | `9badbb1d` | byte-identical |
| `tournament_vision.txt` | `tournament_listwise_vision.txt` | 6497 | `64742877` | byte-identical |
| `rubric.txt` | `rasch_pointwise_prompt.txt` | 5683 | `d3908db0` | byte-identical |
| `rubric_vision.txt` | `rasch_pointwise_vision.txt` | 6852 | `1d692751` | byte-identical |
| `tournament_video.txt` | none | 6550 | `7bd9a638` | **new family**: the submission prompt tree has no video prompt (the paper's prompt modality is text or image), so the video prompts never pool with a paper family; their own `prompt_hash` keys their family and their calibration is the video corpus's |
| `rubric_video.txt` | none | 6904 | `403128c0` | **new family** (as above) |

Corroboration from the paper's rubric-ablation copies: its `T0_shipped.txt` is byte-identical to
`tournament_listwise.txt`, and its `R00_shipped.txt` is `rasch_pointwise_prompt.txt` plus one trailing newline;
no ablation copy differs from the shipped text otherwise. The loader's `.strip()` is a no-op on these files
(neither side carries surrounding whitespace), so the rendered prompt text is the file's bytes.

The prompts render through the same placeholders on both sides: the paper's `wrap_xml` and the product's
`rcp_ndcg.judging._templates.wrap_xml` produce the same `<documents><doc id="doc_N">` blocks (positional ids,
the same media marker), and both clients send the rendered text as one user message.

### The rendered chat-template input: vLLM v0.31.0 against the paper's SGLang image

The paper's judges ran on `lmsysorg/sglang:v0.5.17-cu129` (the submission's engine scripts). The rendered
input was re-derived from both engines' source and each checkpoint's own chat template at its pin:

- vLLM v0.31.0: `ChatCompletionRequest.build_chat_params` merges `add_generation_prompt=True`,
  `continue_final_message=False` and `reasoning_effort` (dropped when `None` by `merge_kwargs`), and adds
  `enable_thinking` only when the request names a `reasoning_effort`; `safe_apply_chat_template` filters the
  kwargs to the template's accepted variables and calls
  `tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)`.
- SGLang v0.5.17: the generic chat path sets `extra_template_kwargs` only from a request `reasoning_effort` or
  `chat_template_kwargs` (the paper's judges name neither), calls
  `apply_chat_template(messages, tokenize=False, add_generation_prompt=True)`, then encodes the rendered text,
  passing `add_special_tokens=False` only when `tokenizer.encode("")` is non-empty.
- Neither engine adds a system prompt: both clients send one user message, and the paper's `build_messages`
  adds a system turn only when the caller sets `system_prompt` (the judge path does not).

| Judge recipe | Chat template (bytes) | Render vLLM == SGLang | `tokenizer.encode("")` | Template reads `reasoning_effort` |
|---|---|---|---|---|
| `qwen3.5-397b-a17b-nvfp4` | 7756 | yes | empty | no |
| `gpt-oss-120b` | 16738 | yes | empty | yes (unset in both) |
| `qwen3.6-27b-fp8` | 7764 | yes | empty | no |
| `qwen3.8-27b-fp8` | 8952 | yes | empty | yes (unset in both) |
| `qwen3.8-flash-next-nvfp4` | 8952 | yes | empty | yes (unset in both) |
| `qwen3.8-flash-next-fp8` | 8952 | yes | empty | yes (unset in both) |
| `gemma-4-12b-it` | 18683 | yes | empty | no |
| `gemma-4-26b-a4b-it` | 18683 | yes | empty | no |
| `gemma-4-26b-a4b-nvfp4` | 16934 | yes | empty | no |
| `gemma-4-31b-it-nvfp4` | 16934 | yes | empty | no |

So the rendered strings and token ids are identical for every judge: no new family arises from the engine
change. The engine difference is output-side only -- vLLM's reasoning parser splits the thinking channel and
starts the JSON-schema grammar after reasoning ends, which is what the recipes' `--reasoning-parser` flag and
the `judge check` probe cover. The Gemma 4 templates' judge-shaped render (thinking off and on) was
independently re-derived through the tokenizer's own `apply_chat_template`; the Google canonical and NVIDIA
re-export files render identically for a single-turn judge request, which is the evidence behind the one 26B
family.

### Declared divergences the pre-flight found (not prompt or rendering)

Two run-level sampling settings differ between the shipped recipes (which follow the paper's presets) and the
paper's runs (whose strategies overrode the client per call). Both are CONTENT fields of the judgement family,
so a run under these recipes never pools with a paper run; both are declared in the recipes' notes and are
**open questions for the owner**:

| Recipe | Paper's runs | Recipe | Why |
|---|---|---|---|
| `gpt-oss-120b` | `temperature: 0.0`; pooled second-judge runs drove `max_response_tokens: 12288` | `temperature: null`, `max_output_tokens: 8192` | the preset's record is null/8192; the paper's client files declare 0.0 and its strategies override the cap |
| `qwen3.6-27b-fp8` | deploy runs drove `max_response_tokens: 12288` | `max_output_tokens: 16384` | the preset declares 16384; the paper's strategies override the cap |

The qwen3.5-397b runs carry no `max_response_tokens` override, so the primary judge matches the paper.

## Checks

The commands run on the final merged head (the gate runs all of them again; this is the lane's own record):

- `uv run --no-sync ruff format --check . && uv run --no-sync ruff check .` — 570 files already formatted,
  all checks passed.
- `uv run --no-sync basedpyright` — 0 errors, 0 warnings, 0 notes.
- `heavy uv run --no-sync pytest tests/ -q -n 4 -p no:cacheprovider` — 3554 passed, 101 skipped.
- `uv run --no-sync pytest tests/contract tests/docs -q -p no:cacheprovider` — 301 passed, 55 skipped.
- `heavy uv run --no-sync pytest rcp-ndcg-test/tests -q -n 4 -p no:cacheprovider` — 631 passed, 349 skipped.
- `uv run --no-sync pytest rcp-ndcg-vllm/tests -q -p no:cacheprovider` — 102 passed, 11 skipped.
- `uv run --no-sync mkdocs build --strict -d <scratch>/site` — built.
- `RCP_EXPERIMENTS_DATA=<data> uv run --no-sync python experiments/run_all.py` — leaderboards ok, human study
  ok, external judges 82 checks / 82 match / 0 failed.
- `rcp-ndcg-vllm serve <id> --dry-run` for all ten judge variants — exit 0, the argv in each golden.
- `bin/gate lane/l08-judges` — GATE: PASS (see the final report for the per-step lines).

## Open questions

1. **The two declared sampling divergences** (R5 pre-flight, table above): should the release's `gpt-oss-120b`
   and `qwen3.6-27b-fp8` recipes move to the paper's run-level values (`temperature: 0.0` for gpt-oss,
   `max_output_tokens: 12288` for both), or stay with the presets' records? Either choice is a new judgement
   family with its own calibration note; the recipes declare the difference so nothing pools silently.
2. **The judge route and multi-variant family directories**: `--judge recipe:./dir` refuses a directory with
   more than one variant (no `--variant` on the judging commands). A `--variant` flag on the judge route is a
   small addition if the owner wants user families with several sizes.
3. **The video prompts' calibration**: `tournament_video.txt`/`rubric_video.txt` are new families with no
   paper counterpart; the video corpus's calibration plan is the E2 wave's, and a per-checkpoint video budget
   (70 soft tokens per frame for the Gemma 4 judges) is later product work.
4. **`judge check` exits 0 on a failed probe** (like `rcp-ndcg doctor`); the verdict is the report's `ok`
   (`data.ok` under `--json`). A `--strict` flag would give scripts a non-zero exit if wanted.
5. **The qwen3.6 context**: the recipe follows the paper's preset and declares no `context_tokens`, so
   documents are sent whole; the other judges carry per-window budgets. An owner decision to declare 262144
   would change the instrument.
6. **All ten recipes are `status: unverified`**; the E2 judge wave must record the NVFP4 linear/MoE backends
   (owner answer 7), the media token counts, both thinking modes and the T4 reruns before any `verified`.
7. **Three pre-existing stale citations outside this lane's files** (found by the round-2 verifier; not fixed
   here because the lane touches only its assigned files): `rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/pplx-embed-v2-context/family.yaml`
   cites `rcp-ndcg-vllm/plugins/pplx/`, which the plugin fold moved to
   `rcp-ndcg-vllm/src/rcp_ndcg_vllm/models/pplx/` (a note edit plus a golden regeneration); the recipes
   `README.md` cites `tests/fixtures/recipes/` (the fixtures live under `rcp-ndcg-test/tests/fixtures/recipes/`);
   and `docs/concepts/runs.md` still builds `rcp-ndcg-vllm/plugins/*`.
8. **The qwen3.6 context**: the paper's client files declared `context_size: 262144`, equal to their engine's
   `--context-length`, so its per-window truncation never bound a TREC-DL window; the recipe's no-`context_tokens`
   policy sends documents whole under the same engine bound -- the same effective policy, now stated in the
   recipe's notes.

## CHANGELOG entry

The Unreleased entries added by this lane (`### Public surface`, `### Changed`, `### Removed`, `### Fixed`):

> **Judges are recipes, and the judge role is in the recipe schema** (workstream 08 B/D, decisions 15, 18, 19,
> 41): the recipe schema gains `role: judge`, whose `client.api: chat` is the judge role's chat-completions
> wire -- the registered `openai_chat` adapter's recipe-facing spelling, so both names resolve to one adapter
> and neither enters an identity. A judge recipe carries **no `reference`** (`Recipe.reference` is `None`; every
> other role still requires one, and the equivalence harness refuses a judge by name). A judge recipe's `client`
> block **is** rcp-ndcg's judge config, validated by `rcp_ndcg.judging.JudgeConfig` when rcp-ndcg reads
> `recipe: <id>` (R30: never a second model). Ten judge recipes ship as eight families (decision 34): the six of
> decision 15 and the four Gemma 4 judges, each pinned, `status: unverified`, with its per-variant golden
> snapshot and its memory arithmetic for one B200 and one H100 (decision 41). The family schema gains a
> per-variant `licence` and `extra_args` joins the declared per-variant `serve` fields.
>
> **`--judge <recipe-id>` and `judge: recipe:<id>`** resolve through the same path as `reranker: recipe:<id>`:
> `JudgeConfig.load` accepts a recipe id, `recipe:<id>` or `recipe:<path>`, a shipped vendor profile or a YAML
> path; `JudgeConfig.base_url` may be unset for a recipe and `JudgeConfig.recipe` records the recipe's identity
> in the judgement store. `rcp_ndcg.inference.recipes` gains `recipe_source`.
>
> **`rcp-ndcg judge check --judge <config|recipe|fake>`**: a short conformance probe (schema accepted, answer
> parses, reasoning channel separated). Public: `rcp_ndcg.judging.check_judge`, `JudgeCheck`,
> `JudgeCheckReport`, `rcp-ndcg.judge-check-report.v1`.
>
> **`JudgeConfig` gains `recipe`** and its `base_url` becomes optional; `known_adapters("judge")` lists `chat`
> beside `openai_chat`.
>
> **Changed**: the self-hosted judge presets become recipes; the T4 scenarios, examples, docs and paper configs
> point at them; `judge check` joins the group.
>
> **Removed**: the self-hosted judge configs (`qwen35_397b_nvfp4`, `qwen35_397b_fp8`, `gpt_oss_120b`,
> `qwen36_27b_fp8`); `gpt5_hosted` stays.
>
> **Fixed**: `JudgeConfig.is_fake` on a URL-less config no longer raises `IndexError`.

## Public surface changes

- Python: `rcp_ndcg.judging.check_judge`, `JudgeCheck`, `JudgeCheckReport` (new); `rcp_ndcg.inference.recipes.recipe_source`
  (new); `JudgeConfig.recipe` (new field), `JudgeConfig.base_url` (now optional); `rcp_ndcg_vllm.recipe`
  (`Family.licence` per variant, `Variant.licence`, `extra_args` in the per-variant serve fields, `role: judge`,
  `ReferenceSpec | None`); `rcp_ndcg_test.e2e.JudgeCandidate`, `judge_candidates`, `judge_client_data`,
  `JudgeEngine.recipe`/`fallback_recipe`/`fallback_slot` (the scenario schema changed); `known_adapters("judge")`
  lists `chat`.
- CLI: `rcp-ndcg judge check` (new; `--judge`, `--judge-url`, `--judge-model`, `--set judge.*`); the `judge`
  group's help; `--judge` accepts recipe ids on `judge` and `run start/resume`.
- Exit codes: unchanged (`judge check` exits 0 with the report's `ok` as the verdict).
- Schemas: `rcp-ndcg.judge-check-report.v1` (new); `rcp-ndcg.judge-config.v1` and `rcp-ndcg.run-config.v1`
  regenerated (the optional `base_url` and the new `recipe` field); the recipe and family JSON Schemas
  regenerated; `rcp-ndcg-test/schema/scenario.schema.json` regenerated; the contract snapshots
  (`cli.json`, `mcp_tools.json`, `python_api.json`) regenerated.

## Files outside scope

- `rcp-ndcg-test/src/rcp_ndcg_test/equivalence/{reference,stages,media}.py` — the `reference_of` accessor and
  the nine call sites: the recipe schema's `ReferenceSpec | None` made the harness's type checks block; the
  accessor is the minimal typed fix and refuses a judge by name.
- `rcp-ndcg-test/src/rcp_ndcg_test/fingerprint.py` — the judge client fields' classification and the judge role
  in `_CLIENT_MODELS`: without it the fingerprint refuses every judge recipe (and the classification is
  additive, so no existing fingerprint moves).
- `experiments/paper/README.md` and `rcp-ndcg/pyproject.toml` — one sentence each about the presets becoming
  recipes.
- `tests/retrieval/test_paper_configs.py`, `rcp-ndcg-vllm/tests/models/test_wheel_contract.py`,
  `rcp-ndcg-test/tests/test_quality.py` — the recipe-count and judge-coverage pins follow the ten new variants.

## For the next lanes

- **E2 judge wave**: per recipe, the T0 serve smoke (record the engine version, the served model entry, the
  dtype, the NVFP4 linear/MoE backends and any Marlin fallback), the structured-output and parse conformance in
  both thinking modes, the media probe, the 26B MoE probe, and the T4 scenario reruns; then `status: verified`.
  The recipes' notes carry the concurrency arithmetic to check against.
- **The owner's two sampling decisions** (Open questions 1): a recipe edit plus a golden regeneration and a
  calibration note if the paper's run-level values win.
- **A per-checkpoint Gemma 4 video budget** (70 soft tokens per frame plus the timestamp charge) if a
  `wire: video_url` corpus is ever wanted; today `wire: frames` is the only correct path.
- **`handover/` is deleted before the release**: this report, the JG report and the specs go with it.
