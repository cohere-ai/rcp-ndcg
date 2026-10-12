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
