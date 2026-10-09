# Lane `sz-pplx`: size variants (pplx) — the `pplx-embed-v1` family and `pplx-embed-v2-late-9b`

## 1. Status

**DONE.** Gate `PASS` on the ported tree `5ddde69b` (**16 families / 30 variants**; the port merged `rfc-0001`
at `247c3d53` and re-applied the lane's additions on top of the recipe line), and again on the report commit.
All three variants ship `status: unverified`; the operator runs the GPU validation per variant.

## 2. Commits

| hash | subject |
|---|---|
| `5f53e0dc` | pplx plugin: the pplx-embed-v1 config class and the late sizes' generation-head replacement |
| `dc8a1730` | The pplx-embed-v1 family: the 0.6B and 4B dense embedders |
| `38f66cb4` | The pplx-embed-v2-late-9b variant and the family's media-token count fix |
| `8fb930eb` | The three new variants' goldens and their vendored tokenizers |
| `35c1d148` | The three new variants' pairs files and the regenerated manifest |
| `b74e2ded` | The harness follows the new variants: generator, task matrix and count pins |
| `3ea47532` | Docs, CHANGELOG and the release checklist for the pplx sizes |
| `13da2341` | Merge lane/rfam (`3ce7357a`: the golden guard's shrink-only revert check, DELTAS key-uniqueness and evidence, the operator-path goldens, the over-share note wording) |
| `91fd031a` | Merge rfc-0001 (`e3a356f1`: workstream 10's data I/O and MTEB readers, and the MRL-cards handover spec) |
| `ea657672` | The late reference's embed mode loads the resolved variant's checkpoint (verifier round 1 blocker) |
| `cfc05e7f` | Verifier round 1 minors: the census shapes, the byte-count conventions and the client-dict reads |
| `25eda4ec` | Verifier round 2 minors: the v1 manifest's empty-doc reason and the last flat ~4 GB claim |
| `5629b5ec` | Merge lane/rfam (`911931da`: the recipe-families report and the family-layout handover spec updates, with rfc-0001's sync-hardening) |
| `fec0a198` | The rfam merge's golden reconciliation: the merged pplx-late notes in the 0.6b delta and the 9b golden |
| `3af4b77e` | Merge rfc-0001 (`c5ccc851`: the scoring-fixes lane and the MTEB PR alignment) |
| `ad7366d3` | Merge rfc-0001 (`247c3d53`: the recipe line — 15 families with the sz-misc/rec-overrides/rec-egemma2/rec-harrier variants — plus l10b, mrl-core, rf-engine, sync-hardening and scoring-fixes) |
| `5ddde69b` | Port sz-pplx onto rfc-0001's recipe line: the v1 family, the late 9b variant and the harness rows onto the 16-family catalog |

Merged: `lane/rfam` at `911931da`, `rfc-0001` at `c5ccc851`, then the recipe line's `rfc-0001` at `247c3d53`.
The port kept rfam's family layout, one test module per family, per-variant goldens and pairs manifest; the
lane's own additions were re-applied on top (the per-recipe test modules rfam deleted stayed deleted).

## 3. What changed

- **Pins (brief item 1).** `perplexity-ai/pplx-embed-v1-0.6b` @ `2c4d510dd4a732063c31a0f70193e35067b51fd8`,
  `-4b` @ `06456497a00540a582918fe8dcd3a5eabb207772`, `pplx-embed-v2-late-9b` @
  `0f49a9977fe06b83377d598094c5c0204ce18ad9`; all MIT, all not gated, checked 2026-10-09 (the cards'
  created/last-modified dates are in the variant notes).
- **Per-size facts (brief item 2).** v1: dims 1024/2560, hidden 1024/2560, 28/36 layers, context 32768,
  weights float32, 596,049,920 / 4,022,468,096 F32 params (2,384,199,680 / 16,089,872,384 tensor bytes), bf16
  weights ~1.2 / ~8.0 GB + KV 3.8 / 4.8 GB at 32768 tokens → one 80 GB-class GPU each. Late 9b: head
  in_features 4096, 32 layers (8 full-attention), hidden 4096, context 262144, float32, 8,392,695,024 params
  (33,570,780,096 tensor bytes), bf16 ~16.8 GB + KV → one 80 GB-class GPU. Every number is cited file:key in
  the family/variant notes; the 32-bit checks (q bytes and MLP elements at the variant's `max_model_len`) are
  below 2^31 for all three.
- **Family proof (brief item 3).** v1 0.6b/4b: `tokenizer.json`, `tokenizer_config.json`, `configuration.py`,
  `modeling.py`, `modules.json` and `st_quantize.py` are SHA-256-identical at the two revisions (only
  `special_tokens_map.json` differs, in `sep_token`'s serialization); same architecture class
  (`PPLXQwen3Model` = `Qwen3Model` with an all-to-all mask), same no-prompt/no-instruction wire, same MEAN
  pooling, no head, no normalisation module. Late 9b/0.6b: the tokenizer, every small ST/config file, the
  prompts, the per-shape caps, the 32 punctuation skip words, the chat template and the pixel pin are
  byte-identical; only the per-size geometry and the Dense head's in_features differ.
- **Serving path (brief item 4).** v1: `hf_overrides {architectures: [Qwen3ForCausalLM], is_causal: false}` +
  the pooling runner's `embed` conversion; the plugin registers the checkpoint's own config class
  (`PplxV1Config`) so `config.json` parses locally and `trust_remote_code` stays false; the checkpoint's ST
  metadata resolves MEAN + no activation; raw text on `/v1/embeddings`; 32768 budget; `empty_doc: omit_zero`
  (an empty render is zero tokens); the reference is the card's sentence-transformers path stopped before the
  trailing `FlexibleQuantizer`, with `over_cap_cut_differs`. Late 9b: the family's plugin class now also
  replaces the generation head with vLLM's `StageMissingLayer` (neither checkpoint ships `lm_head` tensors),
  and the Dense-head loader shape-checks `[128, 4096]`.
- **Rows, tests, pairs, docs (brief item 4).** Three variant rows; the v1 family test module (contract per
  variant, two mutants, tokenizer parity, cap derivation, quantizer, stage 1, card cut, empty policy); the
  late module parametrized over both sizes (contract, head shapes, shared blocks, media count, stage 1,
  mutations); pairs files from the request generator (v1 23 rows each; late 9b 33 rows); goldens + vendored
  tokenizers; the catalog rows in `rcp-ndcg-vllm/README.md` and `docs/reference/recipes.md`; the CHANGELOG; the
  RELEASE-CHECKLIST retrieval count 22.
- **`status: unverified` (brief item 5)** on all three variants.
- **Verifier-driven fixes.** The late reference's `embed` mode now loads the resolved variant's checkpoint
  (it had hardcoded the 0.6b); the late reference's media token count is the media item's (patches + 2, not
  + 3); the request generator reads the client policy from the client dict (eight `getattr` sites that always
  returned the default); the census shapes, byte-count conventions and the per-size generation-head figures.

## 4. Verification

Two independent verifiers ran together in round 1 (fresh context, DeepSeek-V4.1-flash at `xhigh`), each told the
other existed; one fresh confirmation verifier ran in round 2.

**Round 1, lens A (correctness): VERDICT FAIL.**
- Blocker: the late reference's `embed` mode ignored `--recipe` and loaded the 0.6b checkpoint for the 9b
  variant — a wrong oracle for stage 2. Fixed in `ea657672`: `_recipe_facts` reads the recipe's model/revision
  and threads them through `embed`/`_load_reference`; a stub-`sentence_transformers` test records what the
  subprocess loads and failed on the 9b before the fix, passes for both variants now.
- Minor: two wrong shapes in the plugin test's safetensors census → corrected (0.6b vision patch
  `[768, 3, 2, 16, 16]`, 9b gated q_proj `[8192, 4096]`), and the census test now asserts key shapes against
  literals.
- Minor: byte counts mixed file size and tensor-data size → the notes/sources label both.
- Minor: `requirements-reference.txt` named the pre-family path → names the family's shared reference and both
  pinned revisions.
- Minor: the v1 `pooler_config` comment said "none is sent" while the argv carries `{}` → reworded.
- Minor: the manifest's v1 `content:empty` reason said `empty_doc: unknown` → fixed in round 2 (see below).

**Round 1, lens B (regressions/hygiene): VERDICT PASS.** Mutations (late `+2`→`+3`, the generator revert, the
v1 `EXPECTED_MODULES`) all red; the full suites matched the gate; the R30, one-home, public-names, count and
merge-hygiene checks clean. Its minors (census shapes, the flat ~4 GB claim, byte-count conventions, the
client-dict `getattr` bug, the tautological head-shape test) were fixed in `cfc05e7f`/`25eda4ec`; the census
test now pins literals and the head-shape test compares its constant to a literal.

**Round 2, one fresh confirmation verifier (both lenses): VERDICT FAIL (minors only).** It reproduced the
pre-fix blocker (the 9b loaded the 0.6b) and confirmed the fix, verified the census shapes and byte counts
against the pinned safetensors headers, confirmed the generator fix at all eight sites and the 9b pairs file
byte-identical to a fresh run, and re-ran every suite. Its two minors were fixed in `25eda4ec`:
- the v1 manifest entries were regenerated (only the `content:empty` reason moves; the pairs files are
  byte-identical);
- `late_data.py`'s last flat "~4 GB" claim now states the per-size served-bf16 figures.

**Port onto the recipe line (`ad7366d3`, `5ddde69b`).** `rfc-0001` moved under the lane (the recipe line landed:
rfam's family layout with 15 families, one test module per family, per-variant goldens and the pairs manifest,
plus l10b, mrl-core, rf-engine, sync-hardening, scoring-fixes and the sz-misc/rec-overrides/rec-egemma2/
rec-harrier variants). The merge kept the recipe line's generated machinery and the lane's additions were
re-applied on top: the v1 family directory, the late 9b row and the reference/plugin fixes (which the auto-merge
carried), the T3 rows and the 16-family / 30-variant counts, the regenerated pairs manifest and the 0.6b's
declared notes/sources deltas. rfam's per-recipe test modules that the recipe line deleted stayed deleted; the
other lanes' reports in the tree are `rfc-0001`'s versions. Every suite was re-run on the ported tree (the
results in section 5) and the gate passed again.

## 5. Checks

Last run on the ported tree `5ddde69b` (gate log `gates/5ddde69b/SUMMARY`):

- `bin/gate lane/sz-pplx` → **GATE: PASS** (ruff check/format clean, basedpyright 0 errors, `pytest` 3439
  passed / 96 skipped, contract+docs 295 passed / 52 skipped, mkdocs strict, `rcp-ndcg-test` 621 passed / 396
  skipped, the network-gated recipes 0 failures outside the baseline, vllm-pkg 40 passed, vllm-models 72 passed
  / 7 skipped, run_all clean, public-names clean, clean tree).
- `RCP_NDCG_NETWORK_TESTS=1 ... pytest rcp-ndcg-test/tests/recipes/test_pplx_embed_v1.py
  rcp-ndcg-test/tests/recipes/test_pplx_embed_v2_late.py` → 64 passed.
- `pytest rcp-ndcg-test/tests/recipes/test_family_goldens.py rcp-ndcg-test/tests/test_observe_requests.py
  rcp-ndcg-vllm/tests/models/pplx` → 55 passed, 6 skipped.
- `ruff format --check .` → 557 files already formatted; `ruff check .` → all checks passed; `basedpyright` →
  0 errors, 0 warnings, 0 notes.
- The full suites also passed standalone: root `tests/` 3439/96; `rcp-ndcg-test/tests` 621/396;
  `rcp-ndcg-vllm/tests` 103/11; contract+docs 295/52.

## 6. Open questions

- **The exactly-`max_model_len` pooling hang applies to both v1 variants**: their client budget equals the
  model's 32768 context, so a full-budget request is exactly the input GPU-E1 found hangs when chunked. The
  notes name the jina workaround (`--max-num-batched-tokens 32768`) and the recipe-fix lane's opt-in patch;
  neither is shipped here. The late 9b is unaffected (4096 < 4352).
- **The v1 int8 storage view is deliberately not applied**: the served engine returns the float mean-pooled
  vector and the reference stops before the checkpoint's `FlexibleQuantizer`. The card's own `encode` output is
  int8; the wave's T3 sanity column should quantify the float-vs-int8 cosine.
- **The late plugin's `no_init_weights` head replacement is source-pinned only on CPU** (the class-level tests
  skip without vLLM); the GPU wave must confirm the 9b loads and the head allocation is gone.
- **The MRL-cards spec (decision 39) is not applied to these recipes**: the product now carries `mrl_kind` and
  friends, but no shipped recipe declares them (a later lane's scope). The v1 family's notes say MRL is
  available per the card but not declared.
- **The `content:empty` corner for the late sizes** is covered by the recipe's CPU test and now present in the
  pairs files; the wave may sample it.
- The manifest's `generator.module` moved to `rcp_ndcg_test.observe.requests` and the file entries re-sorted as
  a regeneration side effect (the generator moved in the layout move); no recipe's content changed.

## CHANGELOG entry

Added under `## Unreleased` (Public surface):

- **The `pplx-embed-v1` family** (perplexity-ai/pplx-embed-v1-0.6b @ `2c4d510d`, -4b @ `06456497`, MIT; the
  catalog grows to 30 recipes): dense text embedders on a diffusion-continued-pretrained Qwen3 backbone with
  bidirectional attention — one mean-pooled float vector per text (1024 dims at 0.6B, 2560 at 4B), no
  instruction, Matryoshka-capable, and an int8/binary storage view the checkpoint's sentence-transformers
  pipeline applies after pooling. Served on the stock image with `hf_overrides {architectures:
  [Qwen3ForCausalLM], is_causal: false}`, the plugin's local config class, the checkpoint's own ST pooling
  metadata, raw text on `/v1/embeddings`, the 32768-token context and `empty_doc: omit_zero`; the reference is
  the card's sentence-transformers path stopped before its `FlexibleQuantizer`, with `over_cap_cut_differs`.
- **The `pplx-embed-v2-late-9b` variant** (perplexity-ai/pplx-embed-v2-late-9b @ `0f49a997`, MIT): the same
  multimodal late-interaction family at 9B — one 128-dim vector per kept token from a 32-layer hybrid Qwen3.5
  backbone with the Dense head [128, 4096]; the shared prompts, caps, skip words, chat template and pixel pin
  are byte-identical to the 0.6B's, so the variant row carries the per-size facts only.
- **The pplx plugin serves both late sizes**: `PplxLateMultiVectorModel` replaces the generation-only head with
  vLLM's `StageMissingLayer` (neither checkpoint ships `lm_head` tensors), so no uninitialised head parameter
  and no unused generation-head allocation; the Dense-head loader shape-checks the shipped `linear.weight`
  against the served projector.

Added under `### Fixed`:

- **The pplx-embed-v2-late reference's media token count** is the media item's own count (merged patches + the
  vision wrapper, not the `[D] ` prompt token); the 9B's pairs validation failed every image row by one token
  until the fix.
- **The request generator validates a variant of a multi-variant family**: `_validate_and_prune` re-resolved the
  recipe with `load_recipe`, which refuses a family directory under decision 34; it now re-reads through the
  family directory. The generator also reads the client policy from the client dict (`.get`, not `getattr`).

Added under `### Changed`:

- **The T3 task matrix gains the pplx sizes** (the v1 sizes under text embedders; the late 9b under visual
  documents and late interaction, text).

## Public surface changes

None to `rcp-ndcg`/`rcp-ndcg-core` names, the CLI, the exit codes or the schemas; `tests/contract` and
`schemas/` are unchanged. The change is in `rcp-ndcg-vllm` package data and the unpublished harness: the
catalog grows to **16 families / 30 variants**, the folded pplx plugin gains the `PplxV1Config` registration
and the late generation-head replacement, and the request generator/T3 matrix/count pins move with the catalog.
No snapshot regeneration was needed (the gate's contract step passes unchanged).

## Files outside scope

- `rcp-ndcg-test/src/rcp_ndcg_test/observe/requests.py` and `rcp-ndcg-test/tests/test_observe_requests.py` — the
  generator's multi-variant/client-dict fixes (needed for the brief's pairs files).
- `rcp-ndcg-test/src/rcp_ndcg_test/quality.py` and `rcp-ndcg-test/tests/test_quality.py` — the T3 matrix rows
  and its count pin.
- `tests/retrieval/test_paper_configs.py`, `rcp-ndcg-vllm/tests/models/test_wheel_contract.py` — the
  recipe-count pins (30 recipes / 16 families).
- `docs/data.md` — a committed conflict marker in the merged `rfc-0001` (not this lane's) resolved as the union
  of the two sides.
- `rcp-ndcg-test/corpora/vllm-0.31.0/_tokenizers/` — the two vendored tokenizers and their index entries (the
  golden guard's offline tokenizer input).

## Docs updated

- `rcp-ndcg-vllm/README.md` — the catalog table gains the v1 sizes and the late 9b row; the plugin paragraph.
- `docs/reference/recipes.md` — the counts (14/22), the family table and the plugin paragraph.
- `docs/index.md`, `docs/quickstart.md` — the recipe counts (22).
- `handover/RELEASE-CHECKLIST.md` — the retrieval list and count (22).
- `docs/how-to/add-a-model.md` — no change needed (the family format it documents already covers these sizes).
- Greps run: `pplx-embed-v1`, `pplx-embed-v2-late-9b`, `19 recipes|19 variants|13 families|13 shipped|two pplx`,
  `1,761|1761|three tokens|+ 3` over `docs/`, `README.md`, `REPRODUCIBILITY.md`, `skills/`, `examples/`,
  `experiments/`, `mkdocs.yml` and `rcp-ndcg-vllm/README.md`; the remaining hits are the current catalogue
  rows and the late family's document-total budget line (correct).

## For the next lanes

- The operator runs T0–T3 per variant; the v1 sizes need the recipe-fix lane's opt-in hang patch (or the jina
  flag) for a full-budget request; the late 9b needs the normal late-family wave (the compile risk and the
  media path are the family's existing items).
- The MRL-cards spec (decision 39) should be applied to these three variants by the MRL lane.
- `lane/rfam` is merged into `rfc-0001` now (the recipe line); this lane's history carries the merges of
  `rfc-0001` at `c5ccc851` and `247c3d53` and re-applies its additions on the merged catalog (16 families /
  30 variants).
- The late plugin's class-level tests need the engine image's vLLM (the gate's CPU venv skips them); the wave
  is the first place `no_init_weights` and the 9b head shape meet real vLLM.
