# Lane `e2b` — the E2 round-6 fix run (the r4 reference failures and pplx-context's nesting)

## 1. Status

**DONE.** All five brief items are fixed (no item was declared away); each fix has a failing-first CPU test,
and the reproductions were run offline against the pinned stacks before the fix. `bin/gate lane/e2b` is
**GATE: PASS** on `e14417f1` (slot 3; every step exit=0). The report commit follows that revision.

The branch's base **is** the current `rfc-0001` tip: the branch was cut at `8cc7fe51` and a fresh
`git fetch origin rfc-0001` at gate time still resolved to `8cc7fe51`, so there was nothing to merge. The
seven lane commits are `ac605bef`, `2c68c549`, `17ef4aa4`, `fb437c1e`, `60b5ded7`, `bc8ca05e`, `e14417f1`.

## 2. Commits

| Commit | Subject |
|---|---|
| `ac605bef` | The pplx-embed-v2-context reference pairs one matrix per query/document text |
| `2c68c549` | The qwen3-vl-embedding reference processes each input on its own |
| `17ef4aa4` | The qwen3-vl-reranker reference hands the model tensor token type ids |
| `fb437c1e` | The qwen3-vl-reranker-8b declares its text config's tie_word_embeddings |
| `60b5ded7` | The qwen3-vl-embedding notes carry the per-input deviation |
| `bc8ca05e` | The topk-embed-v1 layer_type shim reads the new name where 5.17 puts it |
| `e14417f1` | The E2 round-4 CHANGELOG entries: the five reference and recipe fixes |
| (this report) | The lane report |

## 3. What changed (per brief item)

### 3.1 pplx-embed-v2-context-9b-preview: one matrix per query/document text — **fixed**

The reference's `embed` wrote `[document_vectors[0]]`: only the FIRST document's vectors, while the served
client sends one request per text and the harness pairs one matrix per text positionally. The wave's pairs
rows carry 1/2/4 documents (15/2/6 of the 23 rows), so stage 2 read "the engine returned 4 matrix/matrices,
the reference 1" on every multi-document row (the brief's 14 failing comparisons; the min_cosine row went
with the shape). The remote `encode`'s own contract is one `(chunk_count, embedding_dim)` array per
document (`modeling_pplx_contextual.py`'s `encode` docstring), and `embed` was already encoding all of them
and then dropping all but the first.

`embed` now emits `query_vectors` as the query's one-element list and `document_vectors` as one matrix per
document in the row's own order (a `_matrix` helper, the one nested-float conversion). The module and
function docstrings state the per-text contract.

Failing first: `test_the_reference_answers_one_matrix_per_query_and_document_text` (a fake model answering
one array per document, like the remote code) — `assert 1 == 2` on the document count before the fix.

### 3.2 qwen3-vl-embedding-2b/-8b: the reference's processor call — **fixed** (the reference's call, not a pin)

`StopIteration` in transformers' `processing_utils.get_text_with_replacements` is a **batch-composition**
defect, not a version regression: the processor replaces every multimodal placeholder spelling found in the
batch's texts from ONE shared iterator per modality (`replacement_text = next(replacements_iters[mm_type])`),
so a text row carrying the tokenizer's literal image/video placeholder spellings consumes the batch's media
replacements and the iterator runs dry. The wave's pairs carry exactly such a row — the generator's
`content:special_token_spellings` stratum (row 18; query and first document, built from the tokenizer's
added tokens at generation time), which the served engine tokenizes as plain special ids and embeds like any
other text.

Reproduced offline, without weights, against the checkpoint's own processor (transformers 5.19.0,
qwen-vl-utils 0.0.14, the locally cached Qwen3-VL-Embedding-2B processor) and the wave's pairs: the query
batch of 38 inputs (one image row, the spellings row) raised `StopIteration` at
`processing_utils.py:906` (`next(replacements_iters[mm_type])`); each input on its own answered. The same
shared iterator exists in transformers 5.10.4 (`tr5104/transformers/processing_utils.py`, same
`replacement_text = next(...)` line), so pinning the version the card "needs" cannot fix this — the
reference's call had to change.

`mode_embed` now calls the card's `process` once per input (`_one_vector_per_input`); the card pools per
sequence and its batch is padding-only, so each input's vector is unchanged. The deviation is declared in
the reference's module docstring and in the family notes with the measured evidence.

Failing first: `test_the_reference_processes_each_input_on_its_own` (a fake card recording every call's
input count) — `assert [2, 3] == [1, 1, 1, 1, 1]` before the fix; after it, the test also pins the
positional pairing of the assembled rows.

### 3.3 qwen3-vl-reranker-2b: the reference's model call — **fixed**

The card's `tokenize` calls the processor without `return_tensors` (it truncates `input_ids` in place,
which needs lists), so transformers answers `mm_token_type_ids` as a **list of lists**, and the pinned
model's `get_rope_index` indexes it as a tensor:
`input_token_type = input_token_type[attention_mask[batch_idx].bool()]`
(`modeling_qwen3_vl.py:996` at the pinned transformers 5.17.0). Indexing a Python list with a bool tensor is
exactly the brief's `TypeError: only integer tensors of a single element can be converted to an index`.
Text-only rows never reach `get_rope_index` (`can_compute_mrope` needs `image_grid_thw`/`video_grid_thw`),
which is why only the image-bearing rows failed.

Reproduced offline with the checkpoint's own processor and the reference's `tokenize` body at transformers
5.17.0: without `return_tensors`, `mm_token_type_ids` is a `list` and `mm[0][attention[0].bool()]` raises the
exact TypeError; with `return_tensors="pt"` it is a tensor and the indexing works (`pixel_values` and
`image_grid_thw` came back as tensors either way, so the types list was the only non-tensor key the model
needs).

`tokenize` now rebuilds the types for the truncated/padded ids with the processor's own
`create_mm_token_type_ids` — truncation only removes tokens, so the kept tokens' types are unchanged — and
hands the model a tensor of the padded shape. Declared in the function docstring and the family notes.

Failing first: `test_the_reference_hands_the_model_tensor_token_type_ids` (a stub processor answering the
list form) — `assert isinstance([[0, ...]], torch.Tensor)` was False before the fix.

### 3.4 qwen3-vl-reranker-8b: the engine init — **fixed** (recipe `hf_overrides`, no deviation)

The 8b's `config.json` declares `tie_word_embeddings: false` at the top level only; its `text_config`
carries no such key, while the 2b's `text_config` does (both checked at their pinned revisions: the 8b's
config was fetched at `b212dc8c…`). vLLM v0.31.0's seq-cls weight loader reads
`text_config.tie_word_embeddings` (`model_executor/models/adapters.py:542` and `:611`), so the 8b died at
model load with `AttributeError: 'Qwen3VLTextConfig' object has no attribute 'tie_word_embeddings'` while
the 2b served.

The 8b variant now declares, through `overrides.serve.hf_overrides`, the family's scorer keys restated (a
per-variant serve key replaces the family's value at that path — no deep merge) plus
`text_config: {tie_word_embeddings: false}`. vLLM applies a dict-valued `hf_overrides` entry to the nested
config object (`config/model.py` `_apply_dict_overrides` -> `_update_nested`), which is the seam vLLM itself
uses for the same field elsewhere (`config/vllm.py:1010`, `config/model.py:1233`). Untied is the 8b's own
fact: it ships an `lm_head`. The 2b keeps the family's dict untouched (its checkpoint already declares the
tie inside `text_config`; the two sizes disagree). The variant notes carry the evidence and mark the fix as
GPU-confirmed by the next wave's T0 model load.

Failing first: `test_the_8b_declares_the_text_config_tie_word_embeddings_vllm_reads` (plus the per-variant
contract/argv tests, which now expect the 8b's dict) — all four red before the recipe change.

### 3.5 topk-embed-v1-small/-xsmall: the remote code's `layer_type` — **fixed** (the alias reaches the class)

The round-3 shim's guard was `hasattr(layer_cls, "block_type")`, but transformers 5.17's
`Qwen3_5DecoderLayer.__init__` sets `block_type` on the INSTANCE (`self.block_type =
config.layer_types[layer_idx]`), so `hasattr` on the class is False and the alias was never installed; the
checkpoint's remote `hf_backbone.py:178 patch_packing` re-crashed with
`AttributeError: 'PackedQwen3_5DecoderLayer' object has no attribute 'layer_type'` (the remote subclass
inherits only what the shim installs on the base class). Verified against the real transformers 5.17.0
class in a scratch environment: `hasattr(Qwen3_5DecoderLayer, "block_type")` is False, the fixed shim
installs the alias, and a subclass with `__class__` swapped the way `patch_packing` does reads `layer_type`
correctly.

The shim now reads the new name from the class OR from the attribute the class's `__init__` assigns
(`_carries_new_name`: the class attribute, else `"block_type" in layer_cls.__init__.__code__.co_names`). A
class that still carries `layer_type` stays untouched and a class with neither name stays a no-op, so the
round-3 policy (never mask a class that has neither name) is preserved. The family notes carry the r4
evidence; no transformers pin changed — the reference still runs on the image's stack.

Failing first: `test_the_layer_type_shim_aliases_only_the_new_name` (a stub with the real class's shape: the
instance attribute) — `AttributeError: 'InstanceAttribute' object has no attribute 'layer_type'` before the
fix.

## 4. Verification

- Every fix's test was written first and shown red on the unfixed code (the failure lines are in §3).
- The offline reproductions in §3.2, §3.3, §3.4 and §3.5 used the pinned stacks from local caches
  (transformers 5.19.0 / 5.17.0 / 5.10.4, qwen-vl-utils 0.0.14, torchvision 0.29.1+cpu, PyAV 17.0.1) and
  the checkpoints' own files; no model weights were loaded and nothing was written outside a scratch
  directory outside the repository.
- The goldens were regenerated the documented way (`pytest rcp-ndcg-test/tests/recipes/test_family_goldens.py
  --update-goldens`); the diff is the 8b's `serve.hf_overrides`/argv/fingerprint plus the notes text of the
  three touched families. No schema changed.
- The touched families' recipe modules were re-run with the network gate open
  (`RCP_NDCG_NETWORK_TESTS=1`, the tokenizer cache): 108 passed.

## 5. Checks

| Check | Result |
|---|---|
| `bin/gate lane/e2b` | **GATE: PASS** (`e14417f1`, slot 3; every step exit=0) |
| gate pytest | 4127 passed, 153 skipped |
| gate contract-docs | 410 passed, 105 skipped |
| gate test-pkg | 1132 passed, 228 skipped |
| gate recipes (network on) | no failure outside the baseline (pytest exit 0) |
| gate vllm-pkg / vllm-models | 50 passed / 100 passed, 7 skipped |
| gate run_all / public-names / clean | leaderboards 987 match + 35 known deviations + 0 failed; human study 67/67; judges 82/82; public-names clean; tree clean |
| `ruff format --check` / `ruff check` / `basedpyright` | clean / clean / 0 errors |
| `mkdocs build --strict` | built |

Note for reproduction: an ambient `UV_EXTRA_INDEX_URL` (an unauthenticated private index) makes the
`rcp-ndcg-vllm/tests/models` wheel-build tests fail in this environment; `bin/gate` unsets it, and with it
unset the module is green (7 passed).

## 6. Open questions

- The qwen3-vl-embedding per-input fix is numerically argued (the card pools per sequence and its batch is
  padding-only) but not GPU-measured: the next wave's stage 2 re-measures the served-vs-reference cosines,
  which is where a batch-vs-single kernel difference would show.
- The 8b's `text_config.tie_word_embeddings` override is verified against the vLLM v0.31.0 source
  (`_apply_dict_overrides`/`_update_nested`) and the checkpoint's config, but not against a live engine
  (no vLLM on CPU): the next wave's T0 model load is the confirmation, and the notes say so.
- The pplx-context reference now encodes every document (up to four per row) where it encoded one; the
  stage-2 subprocess gets slower on those rows, not differently-shaped. The reference's GPU cost is the
  wave's to observe.

## Docs updated

None. No page describes these reference-internal calls as they were; the pages that state the contracts
(`docs/how-to/add-a-model.md`'s per-text embed contract, `docs/how-to/validate-a-recipe.md`'s stage-2 and
reference-store rules) were already correct and the fixes bring the references to them. Greps run:

- `git grep -n -i "FIRST document|first document's vectors" -- docs skills examples experiments README.md handover/specs`
  (one hit, the render mode's "its first document for the document shape" — still true)
- `git grep -n "document_vectors|query_vectors|one matrix per text" -- docs skills README.md`
- `git grep -n "Qwen3VLEmbedder|mm_token_type_ids|tie_word_embeddings" -- docs skills`
- `git grep -n "layer_type" -- docs skills examples README.md`
- `git grep -n -i "qwen3-vl-reranker" -- docs README.md` and `git grep -n "hf_overrides" -- docs`

## CHANGELOG entry

Under `## 0.0.1 — <date at tag time>` / `### Fixed`, five entries were added (the exact text is in
`CHANGELOG.md`):

- **The pplx-embed-v2-context-9b-preview reference answers one matrix per query/document text** (the E2
  round-4 stage-2 shape mismatch) — the reference wrote only the FIRST document's vectors; `embed` now
  emits the query's one-element list and one entry per document in the row's own order.
- **The qwen3-vl-embedding reference processes each input on its own** (the E2 round-4 stage-2
  `StopIteration`) — the batch's shared per-modality placeholder iterator runs dry on a text row carrying
  the tokenizer's literal placeholder spellings; the reference calls the card's `process` once per input.
- **The qwen3-vl-reranker reference hands the model a tensor `mm_token_type_ids`** (the E2 round-4 2b
  stage-2 `TypeError`) — the processor's list form is what the pinned model's `get_rope_index` cannot
  index; the reference rebuilds the types for the truncated/padded ids.
- **The qwen3-vl-reranker-8b declares `text_config.tie_word_embeddings` for the engine** (the E2 round-4
  engine init) — vLLM's seq-cls weight loader reads the nested key the 8b's config.json does not carry.
- **The topk-embed-v1 `layer_type` shim reads the new name where transformers 5.17 puts it** (the E2
  round-4 reference crash) — the renamed attribute is an instance attribute, so the class-level check never
  fired.

## Public surface changes

None: no public name, CLI command or flag, exit code, schema or snapshot moved. The recipe *data* changed
(one variant's `serve.hf_overrides`, notes text) and the goldens were regenerated; the recipe schema is
unchanged.

## Files outside scope

None.

## For the next lanes

- The next wave should re-run the five families and read the raw `status.json`/equivalence documents:
  pplx-context (stage 2 counts per document), qwen3-vl-embedding (stage 2 cosines and the media rows),
  qwen3-vl-reranker 2b (stage 2 scores on image rows) and 8b (T0 model load, then stage 2), topk-small/
  -xsmall (the reference subprocess load).
- If the qwen3-vl-embedding reference's per-input loop shows a measurable served-vs-reference shift, the
  cause is not the batching (the pooling is per sequence) but the wave should say so with numbers.
- The stored stage-2 reference outputs are keyed by the family reference hash: the three changed references
  invalidate their entries, so the next wave recomputes them (no action needed).

---

# Round 7 — the last five on `rc0-202610120530`

## 1. Status

**DONE.** All five brief items are fixed or declared, each code change has a failing-first CPU test (the
declarations have red-first recipe tests), and `bin/gate lane/e2b` is **GATE: PASS** on `921bc766`
(slot 2; every step exit=0). The branch's base is the current `rfc-0001` tip: `233bdda2` was merged first
(clean, `fee88753`). Round-7 commits: `e86d80cd`, `56a521c2`, `fc3f28f7`, `5b3fec8d`, `96aa4fad`,
`921bc766`, plus this report commit.

The evidence is the wave's own artifacts, downloaded from the run's results prefix into a scratch
directory outside the repository (the raw `status.json`/`equivalence.json`/`serve.log` of the five
recipes, and the pplx-context reference store), plus the pinned tokenizers and the plugin/recipe code.

## 2. Per item

### 2.1 qwen3-vl-embedding-2b/-8b — **declared** (the measured bf16 floors)

Both sides bf16 at both sizes (no cast to fix). Stage 2 measured over the 282 gated comparisons (94
vectors at full width, k=64 and the size's MRL width):

- 2b: min cosine 0.998541 at full width and k=2048, 0.998906 at k=64; 14 of 282 comparisons below the
  published 0.999 (6 full width, 6 at k=2048, 2 at k=64) → `overrides.gates.vec_min_cosine: 0.9985`.
- 8b: 0.998335 at full width and k=4096, 0.998674 at k=64; 15 of 282 below (7 full width, 7 at k=4096,
  1 at k=64) → `0.9983`.

Each variant's notes carry the per-MRL numbers and the family notes carry the summary; the per-variant
tests (`test_the_variant_declares_its_measured_bf16_cosine_floor`) pin the values and the evidence
strings, red first (`assert None == 0.9985`), and the goldens are regenerated. The bounds sit just below
the measured floors, so a real regression still fails.

### 2.2 qwen3-vl-reranker-2b/-8b — **declared** (the measured probability bound), with one value corrected

Stage 2 measured, over 55 documents and 34 queries: `p99_documents_within` 0.9818 at the published 0.02
tolerance (the failing gate), max |delta| 0.0293 (2b) / 0.0273 (8b) inside the published 0.05
all-documents bound, Kendall tau median 1.0.

The brief named `overrides.gates.prob_p99_abs: 0.98`; that field is the **|delta| tolerance** 99% of
documents must meet (the qwen3-reranker family's own field: 0.025/0.04 from measured maxima
0.0234/0.0391), so 0.98 would make the p99 gate vacuous. Escalated; the supervisor confirmed **0.03**.
Both variants declare it (the tight tolerance covering the measured maximum, 100% of the wave's
documents), the variant + family notes carry the numbers, the test is red first (`assert None == 0.03`),
and the goldens are regenerated.

### 2.3 pplx-embed-v2-context — **fixed**: two real defects and one comparison convention

The wave's "6 of 198 vectors" were 3 comparisons at cosine 0.0 (row 15's empty document, ×3 MRL widths)
and 3 at 0.576 (row 18's query, the spellings stratum). All three classes are resolved:

**(a) The empty document (0.0) — a plugin defect, fixed.** The client sends the document render `[D] `
under the declared split parse, and the pinned tokenizer renders the prefix's trailing space as its OWN
token (id 220) exactly when the content's first token cannot carry it or the content is empty (measured: a
content-initial space merges into the next token — 256/262/… — so a leading 220 in the body is always the
prefix's). The reference's char spans start at char 4 and pool no token there (the downloaded reference
store shows row 15's document vector at norm 0.0), while the plugin stripped only the two literal prefix
ids and pooled the space token — a nonzero vector where the reference pools zero. `pool_sequence` now
drops a leading `PREFIX_SPACE_TOKEN_ID` (220) from the document body, ids and hidden states alike; the
plugin's CPU test pins the empty document's zero vector and the first chunk's span on a spelling-bearing
document (red first: the import of the constant failed).

**(b) The zero-vector convention — the harness, fixed.** Even with the plugin returning the zero vector,
the harness's `_cosine` mapped the pair to 0.0 and failed a perfect agreement. Two zero vectors now score
1.0 (the same degenerate output); a zero against a non-zero vector still scores 0.0, so a one-sided
degenerate output still fails. A unit test pins both directions and the gate summary; red first
(`0.0 == 1.0`).

**(c) Row 18's query (0.576) — the client's query parse, fixed per the supervisor's Fix A.** The
checkpoint's own `prepare_inputs(task="query")` split-tokenizes the query BODY (`split_special_tokens=True`)
and prepends the `[Q] ` prefix as the one added id 248077 — 182 body ids on that row — while the client
sent one added-token parse of the whole render (71 ids): the engine and the reference were compared on
different inputs. The two constructions coincide for spelling-free queries, which is why the round-2
measurement passed. The product now declares it per recipe: `PoolingEndpoint.query_split_special_tokens:
bool | None` (CONTENT; refused beside `text`/`messages`, and without a template whose query shape is a
fixed head followed by one content span), and `TemplateSpec.split_ids` is the one home of the rule — the
fixed head tokenizes with the file's own parse (its `[Q] ` crosses as the reference's prefix id) and the
content span with the split parse. The pooling client's query leg sends those ids; the harness's
`fitting.query_split_special_tokens` reads the declaration and its render check derives the reference's ids
through the same `split_ids` call, so stage 1 and stage 2 stay on one rule. Unset, every other run keeps
one parse of the whole render and is not re-keyed (the fingerprint omits unset fields). Tests, red first:
the client's `[prefix id] + split(body)` and the config refusals; the template method's rule and refusals;
the recipe's wire test on a spellings query; and a stage-1 render-check test that a mutation (disabling the
derivation) reds. The recipe declares `query_split_special_tokens: true` and its notes record the evidence.

### 2.4 topk-embed-v1-small/-xsmall — **fixed** (the rope shim)

The checkpoint's wrapper builds 2-D rope positions for text (`within.unsqueeze(0)`, shape `(1, total)`; its
image branch stacks `(3, 1, total)`), while transformers 5.17's `Qwen3_5RotaryEmbedding.forward` reads only
the 3-D mrope form (`position_ids[:, :, None, :]`, `modeling_qwen3_5.py:192`): every text-only embed died
with `IndexError: too many indices for tensor of dimension 2`. `reference.py` declares
`_promote_qwen3_5_rope_position_ids`, which wraps the rope class's forward: a 2-D input is promoted to
`(3, bs, positions)` (the three axes equal — the text-only case the remote's own image branch shows the
layout for) and a 3-D input passes through unchanged, so the image path is untouched and a second install
is a no-op. The checkpoint's 5.9.0 pin stays out (the family's `reference.in` records why a 5.9.0 cannot
replace the image's stack); the CPU test pins both shapes and the idempotent install (red first:
`AttributeError: ... has no attribute '_promote_qwen3_5_rope_position_ids'`).

### 2.5 embeddinggemma-2 — **fixed** (the serve flag, recorded)

The engine now reaches model load and dies in the TRITON_ATTN prefill kernel:
`triton.runtime.errors.OutOfResources: out of resource: shared memory, Required: 266240, Hardware limit:
232448` at `triton_prefill_attention.py`'s `context_attention_fwd`, inside `compile_or_warm_up_model`. The
demand is compile-time — the full-attention layers' head_dim 512 with the kernel's BLOCK 128 — so neither
of the brief's two candidates applies: `--enforce-eager` does not change the attention backend (and the
crash is not a graph/compile one; the engine already overrides the cudagraph mode to NONE for pooling),
and `--max-num-batched-tokens` does not change the kernel's BLOCK sizes (they come from the dtype and
head_dim, not the sequence length). The lever is the attention backend: the image's own Gemma4 config hook
had already selected FA4 for this model's heterogeneous head dimensions ("Gemma4 model has heterogeneous
head dimensions … Using FA4 for all layers", the serve log), and the plugin's folded EmbeddingGemma2 hook
then overrode it to TRITON_ATTN — on this v0.31.0-based image the Gemma4 hook sets only
`flash_attn_version=4` and leaves `backend` None, so the folded hook's "no backend requested" branch fires.
The recipe pins `serve.extra_args: [--attention-backend, FLASH_ATTN]`, restoring the hook's own selection;
FA4 supports any head size, every attention type (encoder-only included) and SM10.x, and an invalid backend
fails loudly at startup. The test pins the argv and the recorded reasoning in the notes (red first:
`ValueError: '--attention-backend' is not in list`).

## 3. Checks

| Check | Result |
|---|---|
| `bin/gate lane/e2b` | **GATE: PASS** (`921bc766`, slot 2; every step exit=0) |
| gate pytest | 4131 passed, 153 skipped |
| gate contract-docs | 410 passed, 105 skipped |
| gate test-pkg | 1139 passed, 229 skipped |
| gate recipes (network on) | no failure outside the baseline (pytest exit 0) |
| gate vllm-pkg / vllm-models | 50 passed / 102 passed, 8 skipped |
| gate run_all / public-names / clean | leaderboards 987 match + 35 known deviations + 0 failed; human study 67/67; judges 82/82; public-names clean; tree clean |
| `ruff format --check` / `ruff check` / `basedpyright` | clean / clean / 0 errors |
| `mkdocs build --strict` | built |
| touched families with the network gate open | 133 passed (qwen3-vl embedding + reranker, topk, embeddinggemma-2, pplx-context) |

Note for reproduction: an ambient `UV_EXTRA_INDEX_URL` still makes the `rcp-ndcg-vllm/tests/models`
wheel-build tests fail in this environment; `bin/gate` unsets it, and with it unset the module is green.

## 4. Decisions taken (escalated, then confirmed)

- The qwen3-vl-reranker bound: 0.03, not the brief's literal 0.98 (the field's meaning; see 2.2).
- pplx-context row 18: Fix A (the product's query construction), per the supervisor — not a declared
  deviation.
- The pplx empty document and the zero-vector convention: both fixed (the plugin defect first).
- embeddinggemma-2: the FA4 pin, not `--enforce-eager`/`--max-num-batched-tokens` (recorded in the notes).

## 5. Docs updated

- `docs/concepts/late-interaction.md` — the token-ids wire's two per-side split declarations (the
  document parse's bullet corrected: the query side is no longer "the file's own parse" by default, and the
  new `query_split_special_tokens` bullet carries the reference-construction rule).
- `docs/reference/api/rcp_ndcg.inference.md`, `docs/reference/api/rcp_ndcg.retrieval.md` — regenerated
  (`rcp-ndcg docs api --out docs/reference/api`) for the new field.
- `schemas/index.v1.json`, `schemas/run-config.v1.json`, `tests/contract/snapshots/python_api.json` —
  regenerated (`pytest tests/contract --update-snapshots`).
- Greps run: `git grep -n "query_split_special_tokens" -- docs README.md REPRODUCIBILITY.md skills
  examples experiments mkdocs.yml` (one narrative hit, updated); `git grep -n "split_ids" -- docs README.md
  skills examples` (none — the method is documented by the generated snapshot/API pages);
  `git grep -n "attention-backend" -- docs README.md skills examples` (none — recipe data, recorded in the
  recipe notes).

## CHANGELOG entry

Under `## 0.0.1 — <date at tag time>`: one `### Public surface` bullet (the query-side declaration and
`TemplateSpec.split_ids`) and seven `### Fixed` bullets (the pplx empty-document prefix-space rule, the
pplx query construction, the harness's zero-vector convention, the topk rope shim, the two qwen3-vl bound
declarations and the embeddinggemma-2 attention backend). The exact text is in `CHANGELOG.md`.

## Public surface changes

- New: `PoolingEndpoint.query_split_special_tokens` (CONTENT) and `TemplateSpec.split_ids`; the contract
  snapshot, the exported schemas (`index.v1.json`, `run-config.v1.json`) and the generated API pages
  record them.
- Recipe data: the qwen3-vl embedding/reranker `overrides.gates` values, the embeddinggemma-2
  `serve.extra_args` pair and the pplx-context `client.query_split_special_tokens`; goldens regenerated.

## Files outside scope

None.

## For the next lanes

- Re-run the five families on a new RC (the plugin wheel's hash moves again: the pplx plugin module, the
  recipe data and the reference file all changed, so the carriers re-key): qwen3-vl-embedding 2b/8b
  (the declared floors now gate), qwen3-vl-reranker 2b/8b (the declared tolerance), pplx-context (the empty
  row and the spellings row), topk-small/-xsmall (the reference subprocess load) and embeddinggemma-2 (T0
  model load with the FA4 pin).
- embeddinggemma-2's FA4 pin is CPU-verified only (argv + the serve-log/config-hook evidence): the next
  wave's T0 model load is the confirmation, and the flag is removed with the fold when the engine image
  moves to a release whose registry carries the model.
- The qwen3-vl-reranker bounds are declared at 0.03; if a later wave measures a tighter maximum, the
  declarations are the one-line tightening (each bound sits just below its measured floor by design).
