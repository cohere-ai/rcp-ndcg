# Report 06: lane `rec-pplx-late` — the recipe `pplx-embed-v2-late-0.6b`

**Status:** DONE. The recipe, its plugin extension, its test, pairs file, docs, CHANGELOG and checklist row are
merged with `rfc-0001` (final merge `800866be`, carrying rfc-0001's `07bf75bd`) and `bin/gate lane/rec-pplx-late`
passes on the merged head. Two verifier rounds' findings (round 1, both lenses) were fixed; all findings were
minor. Every recipe `status` stays `unverified` — the operator runs the GPU validation.

## What the model is (all facts verified at the pinned revision `8fc2de24534aa3610d85fa59c463313a5f096455`)

A multimodal late-interaction (ColBERT-style) retriever on a Qwen3.5 backbone: one L2-normalized 128-dim vector
per kept token, client-side fp32 MaxSim. The checkpoint is a NATIVE sentence-transformers export — no custom
Python code anywhere (no `auto_map`, no modeling file; the card says so): Transformer -> `1_Dense`
(Linear 1024→128, no bias, Identity) -> `2_MultiVectorMask` (the 32 ASCII punctuation characters dropped
document-side; `skiplist_tasks: ["document"]`; `keep_only_token_ids: null`) -> `3_Normalize`. The role prompts
`"[Q] "`/`"[D] "` are ADDED special tokens (ids 248077/248078) of the checkpoint's own tokenizer. The backbone
is Qwen3.5 (12 layers, hybrid linear/full attention, `text_config.is_causal: false`, hidden 1024), fp32 weights,
`tie_word_embeddings: true`, no lm_head, a Qwen3.5-VL vision tower (153 tensors), and the trained head ships as a
SEPARATE `1_Dense/model.safetensors` (one tensor, `linear.weight` [128, 1024], F32). `sentence_bert_config.json`:
query_length 1024, document_length 4096, query_expansion null. The card: text and image documents, text queries,
"Use separate encoding calls for text-only and image-only batches. Mixed text+image inputs are not supported.",
"PyLate inserts Q/D markers at the second position; this model expects them first."

## Serving decision (vLLM v0.31.0, tag `db9527a468`, cloned into the lane scratch; every fact cited file:line in the recipe's `sources`)

- **A plugin model class is required, extending the EXISTING pplx plugin (no third plugin).** Two stock blockers:
  (a) `architectures[0] "Qwen3_5Model"` is absent from vLLM v0.31.0's registry (its qwen3_5 family is registered
  under `Qwen3_5ForCausalLM` `:203`, `ColQwen3_5` `:284`, `Qwen3_5ForConditionalGeneration` `:596`); a flags-only
  serve falls through to the transformers-backend fallback (a generic AutoModel host), which cannot serve this
  checkpoint. (b) The Dense head's file is invisible to the stock weight discovery
  (`default_loader.py:221` globs `hf_folder/*.safetensors` non-recursively), while `download_weights_from_hf`
  downloads it (huggingface_hub pattern semantics; measured on 1.32.0, the image pins `>=1.31.0`), so the plugin
  fetches it from the same snapshot and renames it onto `custom_text_proj.weight`.
- **The base is the native `ColQwen3_5Model`** (the topk plugin's pattern): its stock mapper maps this
  checkpoint's top-level `language_model.*` → `language_model.model.*`; `visual.*` matches; `embed_dim` comes
  from `serve.hf_overrides {embed_dim: 128}` (declared, not the in-tree `or 128` default); `custom_text_proj` is
  Linear(hidden, 128, bias=True) with a zero-initialised bias (score-equivalent to the checkpoint's bias-less
  head, marked loaded under both qualnames); the tied lm_head loads as the skipped alias of `embed_tokens`.
- **No config class, no remote code**: `model_type qwen3_5` is native; vLLM's own config registry
  (`_CONFIG_REGISTRY["qwen3_5"]`, transformers_utils/config.py:404-428) parses it and registers it with
  transformers' AutoConfig, `trust_remote_code` forced false; the checkpoint carries no `auto_map`. The fla
  canary (`import fla` must FAIL in the engine) is a wave item.
- **The shared risk, flagged (brief item)**: this model's forward runs the same decorated Qwen3.5 text stack
  (`@support_torch_compile`, qwen3_5.py:209) whose torch.compile the context-9b engine currently crashes in at
  startup on GPU (under investigation). The recipe serves without `--enforce-eager` first; the notes declare it
  as the one-entry fallback if the crash reproduces. Left as an Open question for the family.
- **dtype bfloat16** (the GDN prefill kernels refuse fp32, `chunk.py:213-215`); the reference runs the
  checkpoint's own fp32 — the declared served-vs-reference deviation, quantified by the stage-2 gates and the
  wave. The head runs in the pooling runner's default fp32 head dtype.
- **The wire is `request_shape: text`**: the role prefixes are added special tokens (248077/248078) and the
  reference (sentence-transformers, no `split_special_tokens` anywhere) tokenizes the rendered text with the
  same added-token parse the engine runs — measured parity, unlike the context-9b recipe, which needed
  `token_ids` because its remote code splits special tokens.
- **Budgets** 4096 (document) / 1024 (query), the sentence-transformers per-task caps; `max_model_len 4352`;
  the tokenizer.json embeds no truncation to reset. `over_cap_cut_differs` (the card cuts ids at the caps).
- **`empty_doc: send` is exact** (the reference's empty document renders the bare `[D] ` prompt — one kept
  vector; the prefix id is no skiplist word). No normalization corners: the prompts prepend verbatim.
- **Media**: `input [text, image]`, `media_sides ["document"]` (the card's usage encodes queries as text), one
  image per prompt, the R20 nested pixel pin at the shipped processor's effective budget (3136..1800964 px,
  min/max_pixels overriding the size dict in transformers' Qwen2VLImageProcessor init) with
  `engine_pixel_pinning: true` (the product's R20 rule forced the pin: the client policy lies outside the
  family's stock 65536..16777216 range). Image documents are the named no-verify path (the client refuses media
  under skip ids — the same named gap as topk; the manifest records the refusals). The checkpoint ships a video
  processor; the card's surface is text+image, so `max_videos: 0` (the engine could serve a clip; nothing sends
  one; noted).

## What changed (per brief item)

1. **The recipe directory** `packages/rcp-ndcg-vllm/recipes/pplx-embed-v2-late-0.6b/`: `recipe.yaml` (role
   multi_vector, client `vllm_pooling`, `request_shape: text`, dim 128, `embed_dtype: float16`,
   `document_skip_token_ids` = the 32 punctuation ids derived from the skiplist with the checkpoint's own
   tokenizer, per-shape budgets from sentence_bert_config.json, prompts from config_sentence_transformers.json
   as `{special:[Q] }`/`{special:[D] }` fixed segments, `status: unverified`), `reference.py` (the card's own
   sentence-transformers path, decision 9: verbatim prompt prepend, the card's offset-based cut, ST
   MultiVectorEncoder embed with no dtype override, the card's media rules), `requirements-reference.txt` (the
   checkpoint's own pins over the image's torch), no template file (the checkpoint's own chat template rides the
   revision).
2. **Serving**: the pplx plugin wheel gains `late.py` (`PplxLateMultiVectorModel(ColQwen3_5Model)`) +
   `late_data.py` (the checkpoint facts as data) and registers the second architecture `"Qwen3_5Model"`. The
   contextual sibling's behavior is unchanged. The plugin loads the head from `1_Dense/model.safetensors`
   (shape-checked; a shipped bias loads over the zeros when a revision carries one), marks the zero bias under
   both qualnames, and asserts every backbone tensor is inside the pinned name spaces.
3. **The recipe test** `tests/recipes/test_pplx_embed_v2_late_0_6b.py` (16 tests): the full contract through the
   shared `_contract.py` helper (both directions), two named mutants red (plus ten more in the verifier's
   independent mutation run), measured wire facts, the skiplist derivation and asymmetry, stage 1 on CPU (16/16
   with the real tokenizer), the over-length prefix property, the over-cap card-cut parity, the emoji corner,
   the empty-document render, the image wrapper pin, the anchor/frame mutations red, the internal-labels scan.
   The pairs file `pairs/pplx-embed-v2-late-0.6b.jsonl` from `rcp_ndcg_vllm.observe.requests` (33 rows at
   MEDIA_SET_VERSION 3, stage-1-validated, media refusals recorded) + the manifest row. `quality.py`'s
   T3 TASK_MATRIX gains the recipe (visual documents + late interaction, text; the coverage test forces it).
4. **NOTICE** rows (the vLLM subclassing of `late.py`) in all six byte-identical copies; the recipe catalog rows
   (README table, 19), `docs/reference/recipes.md`, `docs/how-to/serve-a-model.md`, `docs/how-to/add-a-model.md`,
   `docs/index.md`, `docs/quickstart.md`, `handover/RELEASE-CHECKLIST.md` (19 retrieval recipes), CHANGELOG.

## Verification

- **Round 1, lens A (correctness, PASS)**: every serving claim verified against the Hub at the pinned revision,
  the vLLM v0.31.0 tag, transformers 5.17.0 and sentence-transformers 6.0.1 wheels. Findings (all minor, all
  fixed): the flags-only-serve mechanism restated precisely (the transformers-backend fallback mis-resolves and
  crashes at construction, rather than resolution failing); the CHANGELOG's pairs row count (32 → 33, the
  MEDIA_SET_VERSION 3 regeneration); the docstring's future head-bias claim now implemented (a shipped
  `linear.bias` loads over the zeros); a citation nit (`default_loader.py:221`, not :226-233). The 3
  qwen3-vl-`*` recipe-test failures, the packaging contract failure and 7 docs failures in its scratch venvs
  were reproduced on clean rfc-0001 → environmental/pre-existing, not this lane.
- **Round 1, lens B (regressions and hygiene, PASS)**: R30 clean (no removed harness helper; the test rides the
  shared helper and the product's fit; the reference is the card's path); one home per concept (no topk imports;
  only the sanctioned data-module mirror); ten recipe mutants + two plugin-constant mutants red naming the field
  (one survival found and fixed: the zero-bias marking now pins its literals); the promised vLLM mapper
  cross-check test added (`importorskip("vllm")`, runs on the GPU wave); docs truth (two stale "18 recipes"
  counts fixed); CHANGELOG truth; no snapshot/schema regeneration needed; NOTICE byte-identical ×6; no private
  names; no AI attribution; the gate PASS read at the lane's SUMMARY; pairs manifest self-consistent; scope
  exactly the brief plus the two in-brief touches.

## Checks (last runs, final head `800866be`)

- `bin/gate lane/rec-pplx-late`: **GATE: PASS** — ruff format/check clean; basedpyright 0 errors; root suite
  3263 passed/83 skipped; tests/contract+tests/docs 274 passed/52 skipped; mkdocs --strict builds; rcp-ndcg-test
  124 passed; vllm-pkg 342 passed/225 skipped; run_all 1022/987/35/0, 67/67, 82/82; worktree clean.
- The recipe's network tests: `RCP_NDCG_NETWORK_TESTS=1 ... pytest tests/recipes/test_pplx_embed_v2_late_0_6b.py`
  → 16 passed (one file, `timeout 900`, tokenizer cache in the lane scratch).
- The pplx plugin suite → 26 passed, 4 skipped (vllm-import and wheel checks run on the GPU wave).
- Mutants: `serve.max_model_len` and `reference.kind` red through the shipped tests; an independent 10-mutant
  sweep (both verifiers) red naming the field; the plugin's ARCHITECTURE/DENSE_HEAD_TENSOR mutants red.

## Open questions

- **The torch.compile risk is shared with pplx-embed-v2-context-9b-preview** (the same decorated Qwen3.5 text
  stack; the context-9b engine currently crashes at startup on GPU inside torch.compile, under investigation).
  This recipe serves without `--enforce-eager` first; if the crash reproduces, `--enforce-eager` is the declared
  one-entry fallback. The owner may want a family-level decision.
- **Image documents are the named no-verify path** (the client refuses media under `document_skip_token_ids` —
  topk's gap, shared). This checkpoint's `keep_only_token_ids` is null, so its own image-side mask would keep
  every non-punctuation token — the gap is the client's position accounting only. Unblocking it is the product
  change MASTER section 9 already tracks (do it in 09 or declare the recipe text-only for 0.0.1).
- **Image queries**: the card's usage encodes queries as text; the recipe declares `media_sides: ["document"]`
  (the client refuses image queries). The ST pipeline itself would encode them — the conservative reading.
- **The pre-existing red on rfc-0001** (both verifiers reproduced it on a clean worktree): the
  `qwen3-vl-embedding-2b` (×2) and `qwen3-vl-reranker-2b` media-stage tests fail with
  `RCP_NDCG_NETWORK_TESTS=1` — the media-inputs lane's new video rows vs the stub engine's video emulation.
  Not this lane's files; for the media-inputs owner.
- `handover/00-MASTER.md` decisions 1 and the M3 table still say "18 public recipes" (historical records, kept;
  workstream 07 folds the counts when it completes the checklist).

## For the next lanes

- The GPU wave for this recipe: the usual T0-T4 plus the compile-crash watch (above), the probe-image count
  (`<= 1 + 2 + 1758`), the fla canary, `/tokenize` equality past 1024, and the skip-ids count cross-check.
- The plugin's two `importorskip("vllm")` tests (the mapper cross-check and the stock-subclass pin) skip on CPU
  and run on the wave — add them to the wave's plugin pytest step if it filters by skip reasons.
- The recipe count moves to 20 when lane 08's judge recipes land; the "19" counts updated here are
  `docs/index.md`, `docs/quickstart.md`, `docs/reference/recipes.md`, `docs/how-to/serve-a-model.md`, the vllm
  README, and `handover/RELEASE-CHECKLIST.md`.
