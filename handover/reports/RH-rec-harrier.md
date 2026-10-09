# Lane report: rec-harrier (the recipe family `harrier-oss-v1`)

**Status:** DONE. The family, its tests, its pairs files, its three goldens, the catalog/docs/checklist
rows and the CHANGELOG are in; the operator addendum's five items and every verifier finding are fixed;
`bin/gate lane/rec-harrier` reports **GATE: PASS** on the final revision (every step green).

**Base and merges.** The lane started from `lane/rfam` @ dc6c5986 and merged `lane/rfam` @ 2c45386a
(its then-latest tip, which carries `rfc-0001`/w09 and the permanent golden guard) as e4f4be84.
`lane/rfam` has still not merged into `rfc-0001` (`git log rfc-0001 --merges | grep rfam` empty), so per
the brief the latest `lane/rfam` was merged, not `rfc-0001`. `lane/rfam` has since advanced to ecc769be
(32 commits: `rfc-0001`'s l08-sglang, mrl-cards and l10a merges plus rfam's round-2 fixes); a trial merge
was **conflict-free but reverted**: it brings 7 base-inherited failures (the harness's e2e/golden-replay
and stage-3 tests build `jsonl:` dataset URIs that l10a's data-model change no longer registers,
`rcp_ndcg.data.dataset`'s scheme list) — see For the next lanes. The verified tree therefore stays at
2c45386a's merge plus this lane's work.

## Commits

- bd6f210f The request generator resolves a multi-variant family by its variant id and reads the declared empty policies through the plain-dict client
- 3a93c36f The recipe family harrier-oss-v1 (270m, 0.6b, 27b): one family, three served recipe ids
- 4d5ee3e4 The catalog, the release checklist and the CHANGELOG record the harrier-oss-v1 family (22 recipes)
- d02cb299 The family layout's follow-ups on the base: NOTICE paths, the docs snippet, the wheel contract, the snapshot and the lint debt
- 68649c57 The verifiers' findings: the sentence-transformers truncation cap, the restored generator test, the remaining client reads and the docs' family layout
- 10f608c0 The gate's format and public-names follow-ups
- e4f4be84 Merge branch 'lane/rfam' into lane/rec-harrier
- edcff076 The second review and the merge: the harrier goldens, the head-pipeline pin, the pooling opt-in note, the over-cap wording, the generator's over-cap reason and the reference-requirements docs truth
- 308401d7 Merge branch 'lane/rfam' into lane/rec-harrier (rfam's verifier round: the golden guard's shrink-only fix, DELTAS cleanup, the over-share wording)
- 99b07666 The confirmation verifier's minors: the shipped-families table gains the harrier row and the MRL pin widens to the abbreviation
- 130cd839 The zerank family test's docstring says an over-share query is reported non-gating

## What changed (per brief item)

1. **One family, decided with evidence.** One family, three sizes. Byte-pinned shared pipeline:
   `modules.json` byte-identical at all three pinned revisions (sha256 `84e40c8e…`: sentence-transformers
   Transformer -> Pooling -> Normalize); `config_sentence_transformers.json` byte-identical
   (`ad209692…`); `mteb_v2_eval_prompts.json` byte-identical (`08aaf10d…`); `1_Pooling/config.json`
   differs only in `word_embedding_dimension` (640/1024/5376). The cards are the same text with the same
   usage for all three sizes. The backbones differ (Gemma3TextModel for the 270m — 18 layers all
   `full_attention` — and the 27b — 62 layers in the 5:1 `sliding_attention` pattern, `sliding_window`
   1024, linear rope factor 8 — and Qwen3Model for the 0.6b), but each variant's tokenizer follows from
   its model id (the loader injects `client.tokenizer = model@revision`), and the only template-visible
   consequence is which post-processor token the route appends (the anchor declaration covers both).
   Nothing else differs per size: the variant rows carry only id/model/revision/notes/sources, no
   overrides at all, and a split would duplicate the whole contract for a tokenizer that is already
   per-variant data.

2. **Serving on vLLM v0.31.0** (cites in `family.yaml`'s sources/comments, checked in the v0.31.0
   clone): `--runner pooling` only; `convert` auto-resolves to `embed` for both backbones through the
   `*Model` suffix default (`vllm/config/model.py:2288`, `try_match_architecture_defaults :2296-2311`;
   `:1310-1311` is the `*ForCausalLM` fallback); the registry serves `Gemma3TextModel` directly
   (`registry.py:228`) and normalizes the unregistered `Qwen3Model` to `Qwen3ForCausalLM`
   (`registry.py:1287-1313`, called at `:1348`/`:1402`) with the `lm_head` tied
   (`tie_word_embeddings: true`); `as_embedding_model` wraps the class
   (`model_loader/utils.py:272-277`, `adapters.py:250-268`); the pooler resolves from the checkpoints'
   own ST metadata (`transformers_utils/config.py:982-1066` -> LAST + normalize, applied at
   `config/model.py:733-753`; an empty `--pooler-config {}` overrides nothing); causal attention
   (`gemma3.py:153-156`, `175-182`, `198`; the card's decoder-only); `bfloat16` (`gemma3_text` refuses
   float16 at `config/model.py:2330-2335`); no plugin, no trust-remote-code, no template file
   (`io_processor.py:136-160`). Chunked prefill/warmup: causal + LAST supports chunked prefill
   (`config/model.py:2132-2166`), the default `max_num_batched_tokens` is 16384 on the >=160 GB GPUs
   (`arg_utils.py:2860-2866`), and the profile run is one `max_num_batched_tokens` forward
   (`gpu_model_runner.py:528`, `:6464`) — no startup forward exceeds the 32-bit limits.

3. **The prompts.** The query frame is the checkpoints' own `web_search_query` prompt, byte-exact
   (`Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ` —
   WITH the trailing space this checkpoint's prompt carries); documents are bare. The checkpoints'
   `mteb_v2_eval_prompts.json` (131 per-task instructions) are decision 33's `Dataset.task_instruction`
   (model-owned, per task); this family's target placement is the card's own fold
   `Instruct: <instruction>\nQuery: <text>` — this recipe's frame is that fold with the retrieval default
   baked in — and the field's product-side plumbing lands with workstream 10 (the embed role's template
   refuses an instruction span today), so until then a task-specialised serving recipe is a new content
   identity. The other preconfigured prompts (`sts_query`, `bitext_query`) are other tasks' frames.

4. **Budgets and over-cap.** `client.max_tokens` 32768 for every variant = the card's "Max Tokens" column
   and its transformers snippet's `max_length`; `serve.max_model_len` 32768. The 27b's 131072
   `max_position_embeddings` is NOT served: the card declares 32,768 for every variant, and one forward of
   131072 tokens would push the MLP activation to 131072 x 21504 = 2,818,572,288 elements — over 2^31
   (2,147,483,648), the GPU-E1 fault class — while 32768 x 21504 = 704,643,072 stays well under. The
   card's ST path truncates at the Transformer module's `max_seq_length`, which sentence-transformers
   infers when the checkpoint sets none as `min(config.max_position_embeddings,
   tokenizer.model_max_length)` (>=3.0 `models/Transformer.py`): 32768 / 32768 / 131072; the
   post-processor appends the pooled anchor AFTER truncation, so the anchor survives that cut too
   (measured; pinned by a test). The client cuts verbatim text with the frame and the anchor reserved,
   so both sides keep every anchor and the content boundary can differ over the cap ->
   `over_cap_cut_differs`; for the 27b the ST cut is beyond the client's boundary, so the comparison is
   whole-prompt vs content-cut — `over_cap_cut_differs` is the only matching declaration
   (`anchor_drop_over_cap` would be false). No separate query cap exists in the referent. The pooling
   hang at exactly `max_model_len` (v0.31.0; fixed upstream in vllm#48039) is declared: ALL THREE
   variants need the pooling-full-context opt-in, no flag workaround ships, and the family opts in when
   the recipe opt-in field exists (lane rf-engine).

5. **Reference.** One `reference.py` for the family, parameterised by the resolved recipe the harness
   passes (`--recipe`): the card's own sentence-transformers usage — `SentenceTransformer(model,
   revision=…, model_kwargs={"dtype": "auto"})`, `encode(queries, prompt_name="web_search_query")`,
   documents bare; the render mode reads the prompt from the checkpoint's own
   `config_sentence_transformers.json` and needs only `huggingface-hub`; a `--tokenizer` spec naming
   another checkpoint or revision is refused. `requirements-reference.txt`: one file for the family,
   floors justified (`torch>=2.0`; `transformers>=4.56` for the card's `dtype=` kwarg and both
   backbones' floors; `sentence-transformers>=3.0`; `huggingface-hub>=0.30`), installed `--no-deps` over
   the image's freeze.

6. **Tests and artifacts.** `rcp-ndcg-test/tests/recipes/test_harrier_oss_v1.py` (47 network tests): the
   contract pinned per variant with two contract mutants red and two behavioural mutations red; the card
   example's measured ids; the family prompt-file byte-identity; the ST-cap inference and anchor
   survival; the head pipeline (`modules.json` exactly Transformer -> Pooling -> Normalize, one SHA-256,
   and `1_Pooling/config.json` lasttoken with the variant's dim); the Matryoshka/MRL absence; stage 1 on
   CPU with the real tokenizers; the over-cap table; the foreign-tokenizer refusal. The three goldens
   are captured with the guard's writer and committed. The pairs files come from the request generator
   (24 rows each, `validation.render_check: passed`, 0 pruned) and the manifest's entries match their
   files. `TASK_MATRIX` gains the three ids; the catalog/docs counts (22) and the RELEASE-CHECKLIST
   (retrieval recipes 22) are updated; `status: unverified`.

## Verification

**Round 1** — two fresh verifiers in parallel (both FAIL):
- Verifier 1 (correctness): MAJOR — the reference's `max_seq_length` was misstated ("runs uncut"; the ST
  path infers and truncates). MAJOR — a test function silently absorbed by my insertion. MINOR — a
  registry cite, the convert/pooler wording, citation drift, the forward-looking task-instruction
  phrasing. Residual risk (brief-sanctioned): the pooling hang is note-only.
- Verifier 2 (regressions/hygiene): the same MAJOR test absorption; MINOR — a stale "19" count, the
  `recipe.yaml` docs claims, the remaining getattr-on-dict reads in the generator, the absent-empty
  reason wording, an unpinned mteb read, the resolution-order edge.
Fixes: 68649c57/10f608c0.

**The operator's second review** (`reviews/recipe-review-2-2026-10-09.md`) and its addendum: BLOCKER
(the hang trigger — now the declared opt-in note), MAJOR (no golden — captured; the head pipeline
prose-only — pinned), MINOR (`length:over_cap` reason false — fixed and the three manifest entries
regenerated; the 27b over-cap wording — made precise; no Matryoshka pin — added), P1 (the
reference-requirements docs claim — corrected). Also: the golden writer's first-capture fallback
(`tokenizer_sha256`) so a new variant can be captured; the merge of rfam's tip.

**Round 2** — one fresh confirmation verifier (c9d39ec7, FAIL): MAJOR — the shrink-only guard's
predicate could never fail (rfam's fix was not yet merged); MAJOR — the branch was behind rfam's live
tip; MINOR — the docs family table omitted harrier, the planner's empty-doc default disagreed with the
product's, the MRL pin was too narrow. Fixes: 308401d7 (the merge, carrying rfam's guard fix) and
99b07666.

**Round 3** — one fresh confirmation verifier (1db161a7, **PASS**): all five fixes verified with
reproductions; every required check reproduced exactly (605/278, 270, 62, 3203/82, the network family
test 47, the goldens 27, lint and public-names clean, gate PASS). Findings: F1 (minor, coordination) —
rfam's live tip advanced to ecc769be; nothing was lost by the merge (I merged the then-tip), but the
integration should take rfam's tip first (see For the next lanes); F2 (minor) — `test_zerank.py`'s
docstring still said "gates red", fixed in 130cd839.

The attempted ecc769be merge (clean, then reverted): it brings 7 base-inherited failures in the
harness's e2e/golden-replay and stage-3 tests — they build `jsonl:` dataset URIs that l10a's data-model
change no longer registers (`rcp_ndcg.data.dataset`'s scheme list, `ConfigError: unknown dataset URI
scheme 'jsonl'`). Those files are not this lane's and the drift is between l10a and the harness; the
lane's verified tree therefore keeps rfam's 2c45386a.

## Checks

`bin/gate lane/rec-harrier` on 130cd839: **GATE: PASS**

```
ruff-check exit=0 All checks passed!
ruff-format exit=0 523 files already formatted
basedpyright exit=0 0 errors, 0 warnings, 0 notes
pytest exit=0 3203 passed, 82 skipped
contract-docs exit=0 270 passed, 52 skipped
mkdocs exit=0 Documentation built
test-pkg exit=0 605 passed, 278 skipped
recipes exit=0 no failure outside the baseline (0 baseline failures remain, 34 fixed; pytest exit 0)
vllm-pkg exit=0 1 passed
vllm-models exit=0 70 passed, 7 skipped
run_all exit=0 1022 checks, 987 match, 35 known deviations, 0 failed; 67/67; 82/82
public-names exit=0 clean
clean exit=0 clean
GATE: PASS
```

One earlier gate run on 99b07666 failed `pytest` on
`tests/runs/test_execution.py::TestStatus::test_a_pod_run_is_read_from_its_mirror_and_restored_from_it`
with an NFS silly-rename artifact (`.nfsbab…`); the test passes 3/3 sequentially and the re-run gate is
fully green.

Lane commands (last runs): the network-gated family test 47 passed (timeout 900); the zerank family 39
passed; the goldens 27 passed; `rcp-ndcg-test/tests` 605 passed/278 skipped; `tests/` 3203 passed;
`tests/contract tests/docs` 270 passed; `rcp-ndcg-vllm/tests` 62 passed; the torch-CPU model venv 70
passed; ruff/format/basedpyright clean; `mkdocs build --strict` OK; `run_all` 1022/987/35/0, 67/67,
82/82.

## Open questions

- **rfam's tip and the l10a drift.** `lane/rfam` is at ecc769be (which merges `rfc-0001`'s l08-sglang,
  mrl-cards and l10a). Merging it here was conflict-free but carries 7 failures in the harness's e2e /
  golden-replay / stage-3 tests (`jsonl:` dataset URIs vs l10a's reader registry). The integration
  should merge rfam's tip into `rfc-0001` first (or reconcile l10a's schemes with the harness) and then
  this lane; the lane's own surface is unaffected by that drift.
- The MRL-cards spec (decision 39, on rfam's newer tip) independently records harrier's three variants
  as MRL kind **none** ("not card-supported") — consistent with this family's omitted `dimensions` and
  its Matryoshka/MRL-absence test.
- The recipe-fix lane's pooling-full-context opt-in field does not exist yet; the family notes declare
  that all three variants need it.
- The 27b on `resources.gpus: 1` (~54 GB bf16 weights) is arithmetic, not a measurement; the wave
  confirms.
- Two committed manifest entries of other recipes (`jina-reranker-v3`, `topk-embed-v1-small`) keep the
  old "refuses" verb in their absent-empty reason (their declared `empty_doc: omit_zero` values are
  correct); regenerating them would refresh the wording — left out to keep this lane's manifest diff to
  its own recipes.
- The brief's "decision 35: `reference.in`" is not in the repository (the master notes end at decision
  34 and no `reference.in` convention exists); the tree's established `requirements-reference.txt`
  convention was followed.

## CHANGELOG entry

Under `## Unreleased` > `### Public surface` (the harrier bullet as committed; the second bullet is the
family-loader surface entry the regenerated contract snapshot needed):

- **The recipe family `harrier-oss-v1`** (microsoft/harrier-oss-v1-270m @ `31de22b6`, -0.6b @ `f9b9dc8d`,
  -27b @ `0c0fc62f`, MIT; 22 public recipes): one family, three sizes (owner decision 34), every size its
  own served, tested recipe id. … (see `CHANGELOG.md`; the entry states the shared byte-identical ST
  pipeline, the per-variant backbones/tokenizers, the vLLM v0.31.0 serving path, the byte-pinned
  `web_search_query` frame, the mteb_v2 instructions as decision 33's task-instruction data, the 32768
  budgets and the 27b's 131072 arithmetic, the ST truncation-cap inference, the card's own
  sentence-transformers reference with `over_cap_cut_differs`, the generator fixes and the three pairs
  files, and `status: unverified`).
- **Recipe families (owner decision 34)**: `rcp_ndcg_vllm.recipe` exports the family loader — `Family`,
  `Variant`, `load_family`, `iter_families`, `resolve_recipe` — and `load_recipe` now takes a variant id
  (or a single-variant family path); the contract snapshot is regenerated for the added names.

## Public surface changes

- Three new served recipe ids (`harrier-oss-v1-270m`, `-0.6b`, `-27b`), their three pairs files and
  manifest entries, and their three goldens.
- `rcp_ndcg_test.quality.TASK_MATRIX` gains the three ids under text embedders.
- No Python names, CLI flags, exit codes or schemas added by this lane; `tests/contract/snapshots/python_api.json`
  was regenerated for rfam's loader names as a base follow-up.

## Files outside scope

- `rcp-ndcg-test/src/rcp_ndcg_test/observe/requests.py` (the multi-variant validation reload, the
  plain-dict client reads, the absent-empty and `length:over_cap` reasons) and
  `rcp-ndcg-test/tests/test_observe_requests.py` (the tests; rfam's family-directory reload test is the
  canonical one and replaced my duplicate).
- `rcp-ndcg-test/src/rcp_ndcg_test/quality.py` (the TASK_MATRIX rows — an offline root test enforces them).
- `tests/retrieval/test_paper_configs.py` (the shipped-recipe count 19 -> 22).
- Base follow-ups the gate needed: `NOTICE` x5 (family paths + the harrier attribution),
  `tests/docs/test_packaging.py`, `tests/docs/test_docs_recipes.py`, `rcp-ndcg-vllm/tests/models/test_wheel_contract.py`,
  `rcp-ndcg-vllm/tests/models/pplx/test_contract_core.py`, `tests/contract/snapshots/python_api.json`,
  and ruff-format-only edits to six base files.
- `rcp-ndcg-test/tests/recipes/test_family_goldens.py` (the first-capture fallback) and the three
  `golden/harrier-oss-v1-*.json`; `golden/{zerank-1-small,zerank-2}-reranker.json` + `golden/DELTAS.json`
  (operator-path placeholders, consistent across both so the guard stays green — superseded by rfam's own
  redaction in its newer tip).
- `docs/reference/recipes.md`, `docs/how-to/add-a-model.md`, `docs/how-to/release-candidates.md`,
  `rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/README.md`, `docs/how-to/serve-a-model.md` (family layout,
  counts, the reference-environment truth).
- `rcp-ndcg-test/tests/recipes/test_zerank.py` (its docstring's "gates red" corrected).

## For the next lanes

- **Integration:** merge `lane/rfam`'s live tip (ecc769be) into `rfc-0001` first; it currently carries 7
  harness-vs-l10a failures (`jsonl:` dataset URIs, `rcp_ndcg.data.dataset`). Then merge this lane; the
  only expected conflicts are the files listed under Files outside scope.
- **GPU/RC:** the three recipes need their waves (`status: unverified`); the 27b's one-GPU weight fit;
  the at-budget pooling hang until vllm#48039 (the family opts into the pooling-full-context field once
  it exists).
- **rfam:** the `Family` name collision was fixed in its newer tip; the golden guard's writer now
  captures a new variant's tokenizer hash through `tokenizer_sha256`.

## Docs updated

- `rcp-ndcg-vllm/README.md` — three catalog rows and the family wording.
- `docs/reference/recipes.md` — the 14-families/22-variants count, the shipped-families table's harrier
  row and the family layout.
- `docs/index.md`, `docs/quickstart.md`, `docs/how-to/serve-a-model.md` — the 22 count.
- `docs/how-to/add-a-model.md` — the family directory section, the example (a one-variant family), the
  validation command and the reference-environment truth.
- `docs/how-to/release-candidates.md` — the staged family directories.
- `handover/RELEASE-CHECKLIST.md` — retrieval recipes 22.
- `CHANGELOG.md` — the harrier family bullet and the family-loader surface bullet.
Greps run: `git grep -n -i "harrier"`; `git grep -n "19 recipes|18 recipes|19 shipped|catalog of the 19|19th recipe"`;
`git grep -n "22 recipes|22 public|catalog of the 22|Retrieval recipes (22)"`; `git grep -n "recipe.yaml" -- docs
rcp-ndcg-vllm/README.md …`; the mkdocs strict build.
