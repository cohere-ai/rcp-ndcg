# Lane SZ-sz-qwen3: size variants (Qwen3 families) (owner decision 34)

## Status

DONE. Four size variants join three shipped families, each with its own tested recipe id, pairs file, golden
and catalog row; the gate passes on the merged tree. The operator runs the GPU validation per variant
(all four rows stay `status: unverified`).

## Commits

- `1f123ee1` The Qwen3 families' size variants: `qwen3-embedding-4b`/`-8b`, `qwen3-vl-embedding-8b` and
  `qwen3-vl-reranker-8b` join their families as variant rows with their per-size facts, and the
  `qwen3-embedding` reference is parameterised by the variant it serves.
- `b5f93e9f` The three Qwen3 family test modules parametrize over their variants: every variant's contract is
  pinned field by field, its stage-1 network test runs with its own real tokenizer, and two mutants stay red
  per variant.
- `c9fb8ebf` The four variants' pairs files (generated, stage-1 validated) and their permanent goldens, and
  the tokenizer store covers their specs (the bytes were already vendored by hash).
- `c45a14d1` The request generator's stage-1 validation reads the variant it was handed instead of re-loading
  the family directory, so a variant of a multi-variant family generates its pairs file.
- `a5abad6d` The catalog, the docs and the release checklist carry the four new variants (23 retrieval
  recipes), and the CHANGELOG records the size ladders.
- `50acc261` The T3 quality task matrix and the recipe-id rule cover the four new variants (23 shipped
  recipes).
- `a30e3415` Merge branch `rfc-0001` into `lane/sz-qwen3` (rfc-0001 `89e7a3b6`).
- `92499037` The gate's parallel recipe run keeps the Hub reachable: no recipe test module writes
  `HF_HUB_OFFLINE` at import (the pplx module's process-wide flag leaked into every sibling worker), the wheel
  contract counts the 23 variants, and the pplx model test reads the family directory.
- `6167f94f` Verifier round 1: the generator reads the client block as the mapping it is (the empty stratum
  returns and the four new pairs files carry it), the variant notes state the chat-template difference and the
  corrected KV/unit facts, the test hash pins are asserted, the mutants are two per variant, the hub-flag
  guard covers every write form, and the reference-environment paths name the family directories.
- `734df841` Verifier round 2: the VL-reranker 8b sources line states the checkpoint's 4095 px floor, and the
  hub-flag guard covers the dict-literal and plain `os.putenv` write forms.
- `6f3d993e` Verifier round 3: the VL-reranker 8b row no longer calls the card script vendored
  (`reference.py` re-implements it), and the hub-flag guard covers the remaining write spellings without
  matching a subscript read.
- `a0d5642c` Merge branch `lane/rfam` into `lane/sz-qwen3` (lane/rfam `5eeb312c`).
- `3ffa29ea` Merge branch `lane/rfam` into `lane/sz-qwen3` (lane/rfam `15c31e61`, the report-only follow-up;
  at that point lane/rfam == rfc-0001).

## What changed

1. **Pins (brief item 1).** All four variants are pinned from the Hub API at 40-hex revisions, `gated: false`,
   `cardData.license: apache-2.0`, checked 2026-10-09:
   `Qwen/Qwen3-Embedding-4B@5cf2132abc99cad020ac570b19d031efec650f2b`,
   `Qwen/Qwen3-Embedding-8B@1d8ad4ca9b3dd8059ad90a75d4983776a23d44af`,
   `Qwen/Qwen3-VL-Embedding-8B@2c4565515e0f265c6511776e7193b22c0968ddc7`,
   `Qwen/Qwen3-VL-Reranker-8B@b212dc8c91a8164aef1ea2de9c1a867611e75c04`. The family licence is unchanged.
2. **Per-size facts (brief item 2).** Each new variant row's `notes` and `sources` carry the facts read at the
   pinned revision with `file:key` citations: architecture class and `model_type`; `hidden_size` (= the served
   vector width for the embedders), layers, heads/KV heads/head_dim, `intermediate_size`; context
   (`config.json:max_position_embeddings`); `torch_dtype`; `tie_word_embeddings`; weight bytes and shard count
   (`model.safetensors.index.json:metadata.total_size`); the card's advertised context/dims/MRL; and the GPU
   count computed from bf16 weights + KV at the recipe's `max_model_len` (1 GPU everywhere; the largest total
   is ~21 GiB for the VL reranker 8b). The only per-size override is `serve.max_model_len: 40960` for
   `qwen3-embedding-4b`/`-8b` (their `config.json` says 40960 where the 0.6b's says 32768; the card's table
   advertises 32K; `client.max_tokens` stays 8192, so no served request reaches either cap). The notes also
   state the one-forward 32-bit arithmetic and which variants do not meet the pooling-hang condition (none of
   these four: their full-budget probe is 8192, below the engine's `max_model_len` except in the VL-embedding
   case where 8192 == `max_model_len` and a full-budget prompt is a single chunk).
3. **Family membership (brief item 3).** The variant notes report the diff against the family's existing
   variant at its pinned revision, with hashes: tokenizer bytes (`vocab.json`/`merges.txt` byte-identical for
   the text embeddings; `tokenizer.json` byte-identical for both VL families), the 14 special added tokens,
   the post-processor, the prompts (`config_sentence_transformers.json`), pooling
   (`1_Pooling/config.json`/`1_LogitScore/config.json`), normalisation (`modules.json`) and the card scripts.
   Differences beyond the per-size fields are reported, not hidden: the text embeddings' 0.6b carries four
   extra non-special added tokens, a different `eos_token_id` and a different (thinking-aware) chat template;
   the reranker 8b's `modules.json` names its score module directory `1_CausalScoreHead` where the 2b names
   `1_LogitScore` (same `LogitScore` class, same ids). No new per-variant field and no new family was needed.
   The `qwen3-embedding` family's ONE reference now reads the variant's model and revision from `--recipe`
   (the other two references already did, and rfam's merge added the matching resolved-recipe guard).
4. **Tests, pairs, catalog (brief item 4).** Variant rows; the three family test modules parametrized over
   their variants (every resolved `serve`/`client`/`reference` field pinned per variant, two mutants red per
   variant); a stage-1 network test per variant with its own real, hash-pinned tokenizer; one pairs file per
   variant from `rcp_ndcg_test.observe.requests` (rows 24/24/37/35, `render_check: passed`, media variants
   also `media_check: passed`); the docs/catalog rows (`rcp-ndcg-vllm/README.md`,
   `docs/reference/recipes.md`, `docs/index.md`, `docs/quickstart.md`, `docs/how-to/serve-a-model.md`) and
   `handover/RELEASE-CHECKLIST.md` all say 23 retrieval recipes; the CHANGELOG entry below; the permanent
   goldens for the four new variants; the tokenizer store's index covers their specs (the bytes were already
   vendored by hash). The T3 `TASK_MATRIX` and the wheel/paper-config recipe counts cover the new variants.
5. **Status (brief item 5).** All four rows are `status: unverified`; the operator's GPU wave sets it from
   evidence.

## Verification

Three verifier rounds on DeepSeek-V4.1-flash (`:xhigh`), fresh context, each
told the other exists and to refute rather than confirm. Round 1 ran two lenses in parallel (A correctness,
B regressions/hygiene); rounds 2 and 3 ran one fresh confirmation verifier (A+B).

**Round 1, lens A (correctness): FAIL.**
- F1 major: the 4b/8b notes did not report that the checkpoint's chat template differs from the 0.6b's (and
  the family's shared comment cites only the 0.6b's measurement). Fixed: both notes and the tokenizer sources
  now state the difference and the measured id counts (24/23 without a generation prompt, 27/26 with one,
  against the shared comment's 0.6b figures), all inert on the served route.
- F2 minor: KV arithmetic was unit-inconsistent (5.9 GiB, 1.2 GiB, 4.8 GiB). Fixed to 6.04 GB = 5.6 GiB,
  1.21 GB = 1.1 GiB and 4.83 GB = 4.5 GiB.
- F3 minor: the VL-reranker note said the checkpoint's pixel floor is 4096; the file says 4095. Fixed in the
  note and the sources line.
- F4 minor: `tie_word_embeddings false` was cited at `config.json:text_config`; the 8b key is top-level.
  Fixed in both VL rows' notes and sources.
- F5 minor: the per-variant `tokenizer_sha256` in the embedding test was dead data. Fixed: the card-example
  test asserts it, and the VL-reranker snapshot hash-pins all seven tokenizer files.
- F6 minor: "two mutants per variant" was false (one each for the 4b/8b). Fixed: the list has two per
  variant.

**Round 1, lens B (regressions and hygiene): FAIL.**
- F1 major: the generator read `recipe.client` with `getattr` on a plain dict, so every empty policy read as a
  refusal; the four new pairs files silently dropped the `content:empty` stratum their siblings carry and the
  manifest recorded a false reason. Fixed: the six sites read the mapping (`recipe.client.get(...)`), a
  regression test pins it, and the four pairs files were regenerated (24/24/37/35) with their manifest
  entries; no existing pairs file or manifest entry moved.
- F2 minor: the mutants comment/coverage (same as lens A F6). Fixed.
- F3 minor: the hub-flag guard regex missed write forms. Fixed (and extended again in rounds 2-3).
- F4 minor: stale standalone-recipe paths in the two VL `requirements-reference.txt` files. Fixed; the same
  defect in the `qwen3-embedding` shared source line was deferred (fixing it moves the 0.6b's permanent
  golden) and was fixed by rfam's merge.
- F5 minor: the `tie_word_embeddings` citation (same as lens A F4). Fixed.
- Confirmed clean: R30, one home per concept, docs truth, snapshots, generated artifacts, and the gate steps
  then green except the pre-existing `public-names` hits (identical at the base).

**Round 2 (confirmation, A+B): FAIL (two minors).**
- R2-F1 minor: the VL-reranker 8b *sources* line still said 4096. Fixed, golden regenerated.
- R2-F2 minor: the guard still missed `os.environ.update({...})` / `os.putenv(...)`. Fixed.
- R2-F3: the deferred `qwen3-embedding` shared source line was confirmed a defensible deferral (it reds all
  three goldens; the defect predates the lane and is rfam-owned).
- Both round-1 majors were confirmed fixed; nothing regressed.

**Round 3 (final confirmation, A+B): FAIL (one minor, plus the deferral).**
- R3-F1 minor: the VL-reranker 8b row called the card script "vendored" though the family ships no vendored
  file (`reference.py` re-implements it). Fixed after the round (the protocol caps at three rounds): the note
  and sources now say the script is byte-identical at every pinned revision and re-implemented by
  `reference.py`, and the golden was regenerated; the golden guard and the module were re-run green by me.
- R3-F2: the deferral, fixed by rfam's merge.
- R3-F3 minor: the guard still missed exotic write spellings and matched a subscript read comparison. Fixed
  after the round (the guard now matches the named write spellings and no read form).
- Everything else re-verified end to end: the facts, the membership diffs, the reference parameterisation,
  the pins/mutants, the pairs and manifest, the goldens, the counts, the suites and the gate.

**After the rfam merge (mine).** `lane/rfam` had merged into rfc-0001; I merged `5eeb312c` then `15c31e61`
(the latter report-only) into the branch, resolved the conflicts in rfam's favour for anything not this
lane's (the generator's family-directory reload, its equivalent regression test, the pplx comment), kept this
lane's reference parameterisation and variant parametrization where rfam's single-variant version did not
support the family's new sizes, adapted rfam's two new reference tests to the variant ids, and regenerated
the four goldens for the merged tree (rfam's shared source-line fix moves all three `qwen3-embedding`
variants). Re-ran every suite and the gate on the merged tree.

## Checks

Last run on `3ffa29ea` (the merged tree):

- `uv run --no-sync ruff format --check .` -> 546 files already formatted.
- `uv run --no-sync ruff check .` -> All checks passed.
- `uv run --no-sync basedpyright` -> 0 errors, 0 warnings, 0 notes.
- `heavy uv run --no-sync pytest tests/ -q -n 4 -p no:cacheprovider` -> 3388 passed, 96 skipped.
- `uv run --no-sync pytest tests/contract tests/docs -q -p no:cacheprovider` -> 294 passed, 52 skipped.
- `heavy uv run --no-sync pytest rcp-ndcg-test/tests -q -n 4 -p no:cacheprovider` -> 609 passed, 264 skipped.
- Network recipe tests, three family modules, `RCP_NDCG_NETWORK_TESTS=1` with a real tokenizer cache ->
  22 + 24 + 14 passed.
- `uv run --no-sync pytest rcp-ndcg-test/tests/recipes/test_family_goldens.py -q` -> 28 passed.
- Pairs regeneration for the four variants with `--reference-python` -> byte-identical files and identical
  manifest entries.
- `uv run --no-sync mkdocs build --strict` -> built.
- `bin/gate lane/sz-qwen3` on `3ffa29ea` -> **GATE: PASS** (every step exit 0; `recipes` 0 failures outside
  the baseline, 34 baseline failures fixed; `public-names` clean). Two earlier gate runs on the same rev
  segfaulted (exit 139) in the shared `test-pkg` step at two different wave/corpus tests with no Python
  traceback; the same suite passes locally (609/609) and the third gate run was green, so those are gate
  environment flakes, not lane failures.

## Open questions

- **The last round's minor was fixed after the round.** R3-F1's wording fix and R3-F3's guard extension were
  made after the round-3 verifier returned; I re-ran the golden guard, the VL-reranker module and the guard
  test green, but no fourth verifier saw them (the protocol caps at three rounds). If the owner wants an
  independent confirmation of those two edits, a single fresh verifier pass over `3ffa29ea`..HEAD suffices.
- **The `qwen3-embedding` shared `serve.chat_template` comment** still cites the 0.6b's template id counts
  without an in-place "0.6b" qualifier; the 4b/8b notes say so and give the correct counts. A future edit to
  the family could qualify the shared comment (it moves all three goldens).
- **`vocab_size`** (151665 at the 4b/8b vs 151669 at the 0.6b), `max_window_layers` and
  `transformers_version` also differ between the text-embedding sizes; the first is a genuine per-size fact
  the notes do not name (no serving effect). Named for a future pass, not fixed here.
- **Gate flakes.** The shared `test-pkg` step segfaulted twice on this rev and passed on the third run; the
  operator may want to know the gate environment can crash a wave/corpus test at random with no traceback.

## CHANGELOG entry

Under `## Unreleased`, `### Public surface`:

> - **The Qwen3 families carry their public size ladders**: `qwen3-embedding` gains `qwen3-embedding-4b` and
>   `qwen3-embedding-8b`, `qwen3-vl-embedding` gains `qwen3-vl-embedding-8b` and `qwen3-vl-reranker` gains
>   `qwen3-vl-reranker-8b` -- 23 retrieval recipes. Every row pins its Hub revision, its per-size facts
>   (dims, context limit, weight bytes, GPU count) and, where the checkpoint's own `config.json` differs from
>   the family's value, a `serve.max_model_len` override (`qwen3-embedding-4b/-8b`: 40960); the
>   `qwen3-embedding` family's ONE reference reads the variant's model and revision from `--recipe` (it no
>   longer pins the 0.6B checkpoint), and the two media families' references are documented as the family's,
>   serving every size. Each new variant ships its contract pins, its stage-1 test, its pairs file and its
>   golden. (`observe.requests`' stage-1 validation reads the variant it was handed instead of re-loading the
>   family directory, so a variant of a multi-variant family generates its pairs file.)

## Public surface changes

- Four new recipe ids: `qwen3-embedding-4b`, `qwen3-embedding-8b`, `qwen3-vl-embedding-8b`,
  `qwen3-vl-reranker-8b` (each served, `recipe:`-resolvable, contract-tested, stage-1-tested and
  GPU-validated on its own); the catalog and the docs now list 23 retrieval recipes. No CLI command, flag,
  exit code or JSON Schema changed: `tests/contract/snapshots/` and `schemas/` are unchanged by this lane.
- The `qwen3-embedding` family's reference now takes the variant's model and revision from the resolved
  recipe (`--recipe`), which is the family-layout contract the other families already followed.

## Files outside scope

Changed minimally, each listed here as the brief requires:

- `rcp-ndcg-test/src/rcp_ndcg_test/observe/requests.py` (the generator: stage-1 validation reloads through
  the family directory, and the client block is read as the mapping it is at six sites) and
  `rcp-ndcg-test/tests/test_observe_requests.py` (its regression tests).
- `rcp-ndcg-test/src/rcp_ndcg_test/quality.py` (T3 `TASK_MATRIX` covers the four variants).
- `rcp-ndcg-vllm/tests/models/test_wheel_contract.py` (the recipe count 23) and
  `rcp-ndcg-vllm/tests/models/pplx/test_contract_core.py` (reads the family directory).
- `rcp-ndcg-test/tests/recipes/test_network_gate.py` (the import-time hub-flag guard) and
  `rcp-ndcg-test/tests/recipes/test_pplx_embed_v2_context.py` (no process-wide `HF_HUB_OFFLINE` write).
- `tests/retrieval/test_paper_configs.py` (the recipe-id count 23).
- `rcp-ndcg-test/corpora/vllm-0.31.0/_tokenizers/index.json` (the four new specs, pointing at bytes already
  vendored by hash).
- `rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/qwen3-embedding/reference.py` (the family reference's variant
  parameterisation; the family's file, which rfam also edited for the embed mode -- resolved to this lane's
  superset after the merge, with rfam's tests adapted).

## For the next lanes

- The GPU wave validates each of the four variants on the stock image; the two text embeddings also exercise
  the 40960-token `--max-model-len` warm-up for the first time, and the two VL variants their media and
  `/tokenize` gates.
- A future pass over the `qwen3-embedding` family can qualify the shared chat-template comment for the 0.6b
  and name `vocab_size` among the per-size facts.
- The `HF_HUB_OFFLINE`-at-import landmine is closed for the recipe tests (a guard test now scans the
  directory); any new recipe module that needs offline behaviour should pass `local_files_only` to its own
  hub call rather than setting the process flag.
