# Report: lane `rec-egemma2` — the recipe family `embeddinggemma-2` (text, image, video)

**Status: DONE.** The lane began BLOCKED-OWNER (step 1: vLLM v0.31.0 cannot serve `google/embeddinggemma-2`); the
owner's decision 38 (2026-10-09) pinned this recipe to a vLLM nightly by digest and the lane then built the
family. `status: unverified` — the operator runs the GPU validation.

**Base and merge:** the lane was built on `lane/rfam` @ `dc6c5986` (its head at session start). The rfam lane
advanced during the lane to `da344613`, which carries the rfc-0001 merge (`681a8cea`, via `26b3c24a`), the
family goldens (`c6a6f45e`), the family schema and the workstream-09 merge; per the brief (`rfc-0001` merged
into rfam) the lane merged the latest `lane/rfam` in `5199688f`, so `rfc-0001` @ `681a8cea` is an ancestor of
the lane head. The base head `dc6c5986` carried six pre-existing lint/format failures, repaired mechanically in
`539e015a` (listed under "Files outside scope").

## Commits

1. `539e015a` — the base `lane/rfam` lint repair (six files, mechanical only).
2. `df80617b` — the step-1 report (vLLM v0.31.0 cannot serve it; the three options), later superseded.
3. `1eea0f2d` — the report's round-2 fixups.
4. `ec35e9fd` — the report records the gate result on the blocked tree.
5. `4b5ee313` — the Gemma 4 media geometry and the content-only prompt home (product + schemas + snapshots).
6. `f9358cc7` — the recipe family `embeddinggemma-2`.
7. `6e25005c` — the report (the pre-merge tree).
8. `5199688f` — the merge of `lane/rfam` (`da344613`: the rfc-0001 merge, the family goldens and schema).
9. `62dd1343` — the round-1 verifier findings (the gemma4 media fit, the request generator's family-client
   reads, the variant's golden and vendored tokenizer, the geometry cross-check, the docs corrections).
10. The report update after the round-2 confirmation (the lane's final message carries its hash).

## Step 1 — the engine decision (kept for the record; decided by decision 38)

vLLM v0.31.0 (`db9527a46873454610df6dbedf79a36d6bf1a7f6`) cannot serve the checkpoint: no
`EmbeddingGemma2Model` in its registry and no model file; its `requirements/common.txt:10` pins
`transformers >= 5.10.4, < 5.18.0` and the image carries 5.17.0, whose wheel has no `embedding_gemma2` package
(the class landed in **transformers 5.19.0**, released 2026-10-06); and v0.31.0 has no Gemma video backend. The
architecture landed in vLLM commit `02b83919aa2e` (PR #60254, 2026-10-06, +1,142 model +106 config +178 video
+1 registry +687 test) with the mypy follow-up `bb87d227d4b9`. A plugin on v0.31.0 would have to vendor the
HF config/processor (~1,000 lines) plus the vLLM model/config/video backend (~1,430 lines) — a backport, not a
lean plugin. Decision 38: this recipe pins the nightly
`vllm/vllm-openai:nightly-8cbd5d03006c33185f402249ff2b448efd594986@sha256:b25e8a046fdbe948987b3dba06ef2cf9ea0f02b36d9482a25113a445ee52ad21`
and carries the switch-to-release note; the default stays `vllm/vllm-openai:v0.31.0`. Recorded in
`handover/00-MASTER.md` as decision 38.

## Step 2 — the serving facts (cited at the nightly commit `8cbd5d03006c...`)

- **Pooling**: `EmbeddingGemma2Model` is a `VllmModelForPooling` (`registry.py:249`;
  `embedding_gemma2.py:982-1103`), `@default_pooling_type(seq_pooling_type="MEAN", tok_pooling_type="ALL")`
  with `DispatchPooler.for_embedding` — mean pooling over every token, the checkpoint's
  `1_Pooling/config.json` (`mean`, `include_prompt: true`, 768 dims), L2-normalised by the default
  `PoolerNormalize` head.
- **Attention**: bidirectional, `AttentionType.ENCODER_ONLY` (`embedding_gemma2.py:160`; the module docstring
  says "bidirectional Gemma4-derived"), `_WINDOW_OFFSET = 1` for the HF `|q-k| <= W` sliding mask.
- **dtype**: `bfloat16` (the card: bfloat16 or float32, never float16, `README.md:189-193`).
- **max_model_len**: 8192. The card's 8K context (`README.md:68`, `:229-235`); the nightly config caps an
  unset `max_model_len` at 8192 (`config.py:293-340`), and the recipe declares 8192 explicitly.
- **Image path**: the checkpoint's `Gemma4ImageProcessor` (`processor_config.json`: patch 16, pooling 3,
  `max_soft_tokens` 280, `image_seq_length` 280). It resizes by `get_aspect_ratio_preserving_size`
  (transformers 5.19.0 `models/gemma4/image_processing_gemma4.py`): scale by
  `sqrt(max_patches x patch^2 / area)`, floor both edges to `patch x pooling` = 48; the prompt gets
  `boi + n soft tokens + eoi` (`embedding_gemma2.py:491-521`). **The resize is not idempotent**, and the engine
  runs the processor on the bytes the client sends: the client prepares the resize's fixed point (a 4096x576
  page: 2112x288 then 2160x288; a 3000x20 strip: eight passes to 13344x48), so the engine keeps the prepared
  image. The product's `ImagePolicy.target_size` iterates `gemma4_resize` to that fixed point
  (`resolution.py:gemma4_fixed_point`) and counts the pooled patches.
- **Video path**: the nightly's `EmbeddingGemma2VideoBackend` (`vllm/multimodal/video.py:422-523`) samples by
  `fps`, capped at `max_frames`; it **ignores** `num_frames` (`:457`). The prompt renders one
  `boi + n soft tokens + eoi` block **per frame** (no temporal patch; `embedding_gemma2.py:541-577`), with the
  video processor's 140 soft tokens per frame (`processor_config.json`). The recipe pins
  `--media-io-kwargs '{"video": {"fps": 60, "max_frames": 32}}'`: 60 fps capped at 32 realizes the declared 32
  uniformly spaced frames for any clip of at least 32/60 s, and the client's own frame-count check admits a
  container only with at least 32 frames, so a normal clip passes above that. The card's default is 1 fps and
  calls the rate configurable (`README.md:252`); the declared instrument is the 32-frame cap.
- **Tokenizer**: the post-processor adds `<bos>` (2) and `<eos>` (1) around every sequence
  (`tokenizer.json` `post_processor`), declared with `add_special_tokens: true`; the chat template renders a
  user turn's text and media placeholders only (no role markers, no default system turn).
- **The 32-bit and exactly-max_model_len checks do not arise at 8192**: GPU-E1's 32-bit offset fault needs a
  262144-token warmup and the pooling hang was observed at 32768; this recipe caps `max_model_len` at 8192
  (the card's context), so the engine's warmup is one 8192-token sequence. The pairs' `length:at_budget` row
  (8188 tokens) and the wave's smoke exercise the boundary.

## Step 3 — the family (`rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/embeddinggemma-2/`)

`family.yaml` (one variant `embeddinggemma-2`, `google/embeddinggemma-2` @ `914f7f89142e33e77833254d9c9b90c3cef7303b`):
`role: embed`, `input: [text, image, video]`, the digest-pinned nightly with `min_version: "0.31.1.dev0"`, the
pooling runner, `max_model_len: 8192`, `dtype: bfloat16`, `limit_mm_per_prompt {image: 1, video: 1}`, the video
pin in `extra_args`, the card's `query_prompt`/`doc_prompt` (`task: search result | query: ` /
`title: none | text: `) beside a **content-only** template (both shapes, `anchor: mean`,
`add_special_tokens: true`), `image_processor: gemma4` with `image_policy {max_soft_tokens: 280}`,
`video_policy {num_frames: 32, wire: video_url, engine_video_pinning: true}`, `max_tokens: 8192`,
`on_overflow: cut`, `empty_doc: send`, `normalize: true`, and
`reference.known_deviations: [over_cap_cut_differs]`. Audio is refused (out of scope for 0.0.1; the checkpoint
also ships a `gemma4_audio` encoder).

## Step 4 — the reference (`reference.py`, `requirements-reference.txt`)

The card's own usage (`README.md:101-119`): sentence-transformers `SentenceTransformer` with
`prompt_name="SearchQuery"` / `"Document"`, one L2-normalised 768-d vector per text; the declared prompts are
cross-checked against the checkpoint's `config_sentence_transformers.json` at run time. `--mode render`
returns the prefix + raw text uncut (the declared `over_cap_cut_differs`); `--mode media` restates the
checkpoint's Gemma 4 resize (fixed point) and the engine's pinned frame sampling. The environment pins
`transformers==5.19.0` (the first release with the classes) and `sentence-transformers>=6.1.0` (the
checkpoint's own requirement); torch comes from the image.

## Step 5 — tests, pairs, docs

- `rcp-ndcg-test/tests/recipes/test_embeddinggemma_2.py` (14 tests): the contract pins every resolved
  `serve`/`client`/`reference` field plus the top-level facts and the digest/min_version; two drift mutants
  (`serve.max_model_len`, `reference.kind`) red naming the field; the argv pins; the endpoint build; the
  reference's geometry/frame sampling and its cross-check against the product's over a size grid; the
  vendored tokenizer's hash; the checkpoint prompts and chat template; **stage 1 on CPU** with the
  real tokenizer (both shapes, the anchor audit reads every input, the render check exact); the **media stage
  offline** (13 media items: eight image buckets, the captioned page, the mixed batch, the query image, two
  clips); the internal-label scan; the family-directory shape.
- `rcp-ndcg-test/pairs/embeddinggemma-2.jsonl`: 37 rows (13 media rows) from the request generator with
  `--reference-python` validation, pruned rows 0; the manifest carries the file's hash, rows and strata.
- **Golden**: `rcp-ndcg-test/tests/recipes/golden/embeddinggemma-2.json` (the per-variant golden contract),
  written by the rfam golden writer; the checkpoint's tokenizer is vendored into the fingerprint's store
  (`corpora/vllm-0.31.0/_tokenizers/`, gzipped) so the golden's tokenizer SHA is reproducible offline.
- Docs/catalog: `rcp-ndcg-vllm/README.md` (the catalog row), `docs/reference/recipes.md`,
  `docs/index.md`, `docs/quickstart.md`, `docs/how-to/serve-a-model.md` (20 recipes, the nightly exception),
  `handover/RELEASE-CHECKLIST.md` (20 retrieval recipes + the nightly wave item), `CHANGELOG.md`.

## Product changes (declared; the brief authorises an additive `data/resolution` entry)

- `rcp_ndcg.data.resolution`: `ImageProcessor` gains `"gemma4"`; `ProcessorGeometry` gains `resize`,
  `soft_tokens`, `video_soft_tokens`, `per_frame_wrapper`; `PROCESSORS["gemma4"]`; `gemma4_resize` (the
  faithful port) and `gemma4_fixed_point` (the preparation rule); `ImagePolicy.max_soft_tokens`;
  `_soft_budget_problem`; `_video_frame_tokens`/`_container_tokens` per-frame-wrapper accounting.
- `rcp_ndcg.inference.config`: the prompt-prefix refusal fires only when the template's own shape renders a
  fixed segment, so a content-only template may carry `query_prompt`/`doc_prompt` (the only way the task
  prefix reaches a content-only chat render).
- `rcp_ndcg_vllm.recipe`: `EngineSpec.min_version` accepts a setuptools-scm dev series (`0.31.1.dev0`) for a
  digest-pinned nightly; the recipe schema regenerated.
- `rcp_ndcg_test.quality`: the T3 task matrix gains `embeddinggemma-2` (visual documents, vidore), as the
  coverage test forces.

## Verification

- **Round 1, lens A (correctness, VERDICT FAIL: 1 major + 5 minors)**: it reproduced the engine decision
  (the digest, v0.31.0's missing registry/transformers, the nightly's commit and pin), every serving fact at
  the nightly clone, the recipe declaration and the reference, and differentially verified the product's
  `gemma4_resize`/`gemma4_fixed_point` against transformers 5.19.0 (0 mismatches over 910 size/budget
  combinations; the fixed point over 16,445 sizes, worst 12 passes). Findings, all fixed: (F1, major) the
  brief's pre-report merge had been skipped — `lane/rfam` had advanced to `da344613` with the rfc-0001 merge
  and the family goldens, and the new variant failed `test_family_goldens.py`; fixed by the merge
  `5199688f` and the golden `golden/embeddinggemma-2.json` (with the checkpoint's tokenizer vendored into the
  fingerprint store); (F2, major) `fit_media_to_budget` hit a bare `AssertionError` for a gemma4 soft-token
  policy; fixed to skip the shrink step and drop whole items, with a failing test first; (F3) 13440 -> 13344
  in three texts; (F4) the request generator read `empty_doc` off the family client dict (a default), so the
  empty-document row was dropped; fixed to `.get(...)` (all eight such reads) and the pairs regenerated
  (37 rows, `content:empty` present); (F5) the dead `TOKENIZER_SHA256` now pins the vendored store's bytes;
  (F6) the 32-bit/exactly-max_model_len checks are stated as not arising at the 8192 cap; (F7) the
  `ImagePolicy`/`prepare.py` docstrings describe the soft-token budget; (F8) the report's docstring wording
  and count.
- **Round 1, lens B (regressions and hygiene, VERDICT FAIL: 1 blocker + 6 minors)**: it verified the product
  blast radius (schemas/snapshots regenerated, the snapshot drift only the lane's changes plus rfam's family
  entries), the pairs/manifest integrity, the mutants (gemma4_fixed_point -> gemma4_resize red;
  per_frame_wrapper=False red), the base lint repair's AST identity and the full suite. Findings, all fixed:
  (F1, blocker) the skipped merge (same as lens A's F1); (F2) the CHANGELOG's "19 public recipes" -> 20;
  (F3) the stock-image-only pages (`release-candidates.md`, `validate-a-recipe.md`, the vllm README) now name
  the digest-pinned exception; (F4) the `rcp_ndcg.inference.config` product change is listed under Files
  outside scope; (F5) the spliced manifest's stale `generator.module` corrected to
  `rcp_ndcg_test.observe.requests`; (F6) the requirements-reference comments no longer claim unused
  `pyyaml`/`tokenizers`; (F7) the reference/product geometry duplication now has an explicit cross-check test
  and a note.
- **Round 2 (one fresh confirmation verifier, lens A+B)**: _to be filled after the round-2 result._

## Checks (last runs)

- `uv run --no-sync ruff format --check .` → 523 files already formatted; `uv run --no-sync ruff check .` →
  all checks passed; `uv run --no-sync basedpyright` → 0 errors.
- `heavy uv run --no-sync pytest tests/ -q -n 4` → **3215 passed, 82 skipped, 0 failed** on the merged tree
  (the four pre-existing family-layout failures are gone: the `lane/rfam` merge carries their fixes).
- `uv run --no-sync pytest rcp-ndcg-test/tests -q` → **598 passed, 241 skipped, 0 failed** (offline),
  including the family golden test (24) and the request-generator tests.
- `RCP_NDCG_NETWORK_TESTS=1 ... pytest rcp-ndcg-test/tests/recipes/test_embeddinggemma_2.py` → **14 passed**
  (one file, `timeout 900`).
- `tests/contract tests/docs` → 266+ passed, 52 skipped, 0 failed after the schema regeneration.
- `bin/gate lane/rec-egemma2` (final head): _to be filled after the gate._

## Open questions

- The GPU wave must verify the nightly image's actual transformers is 5.19.x, that the architecture loads, the
  pinned video sampling (60 fps, `max_frames` 32) and the Gemma 4 geometry at the engine, and the media
  stage's engine count (the offline media stage compares client vs reference; the stub engine's
  `smart_resize` path cannot emulate the Gemma 4 resize, so the engine count is GPU-only).
- The card's 1 fps default is superseded by the declared 60 fps/32-frame pin; a clip whose frame rate exceeds
  60 fps and whose duration is under 0.53 s can make the engine sample fewer than 32 (the one declared open
  corner; the client's frame-count check already refuses clips under 32 frames).
- The image budget is the checkpoint's own (280 image, 140 video frame); no `mm_processor_kwargs` pin is
  declared, and the recipe schema's pixel-pin check does not yet compare a `max_soft_tokens` pin (a future
  gemma4 recipe that pins a non-stock budget would need that check).

## CHANGELOG entry

The three bullets under `## Unreleased` → `### Public surface` in commit `4b5ee313`: the recipe family
`embeddinggemma-2` (decision 38, the digest pin and the switch-to-release note), `ImagePolicy.max_soft_tokens`
and the `gemma4` processor family (the fixed-point resize, one wrapper per video frame, the dev-series engine
floor), and the prompt-prefix refusal relaxation.

## Public surface changes

- `ImagePolicy.max_soft_tokens` (new field), `ImageProcessor` gains `"gemma4"`,
  `rcp_ndcg.data.resolution.PROCESSORS["gemma4"]`, `gemma4_resize` and `gemma4_fixed_point`;
  `schemas/{index,judge-config,run-config}.v1.json` and `tests/contract/snapshots/python_api.json`
  regenerated.
- `EngineSpec.min_version` accepts `MAJOR.MINOR.PATCH(rcN)?(.devN)?`; the recipe schema regenerated.
- The `EmbeddingEndpoint` prompt-prefix refusal now depends on the template's fixed segments.
- The recipe `embeddinggemma-2` (catalog row; `status: unverified`).

## Files outside scope

- `rcp-ndcg/src/rcp_ndcg/data/prepare.py` — the gemma4 soft-token media fit (the shrink step is skipped
  and whole items drop; a bare `AssertionError` before; failing test first in `tests/data/test_prepare.py`).
- `rcp-ndcg/src/rcp_ndcg/inference/config.py` — the prompt-prefix refusal relaxation (the recipe's task
  prefix needs it on the messages route; a failing test first, `tests/inference/test_client_budget.py`).
- `rcp-ndcg-test/src/rcp_ndcg_test/observe/requests.py` — the generator's family-client reads
  (`getattr(dict, ...)` returned defaults, dropping the empty-document row and every declared instruction,
  query cap and dimensions; fixed to `.get(...)`).
- `rcp-ndcg-test/corpora/vllm-0.31.0/_tokenizers/` — the checkpoint's tokenizer vendored (gzipped) for the
  golden's offline tokenizer SHA (the store's convention).
- `tests/retrieval/test_paper_configs.py` — the shipped-recipe count 19 → 20 (the new recipe).
- `rcp-ndcg-test/src/rcp_ndcg_test/quality.py` — the T3 task matrix row for `embeddinggemma-2` (the coverage
  test forces every recipe into the matrix).
- `handover/00-MASTER.md` — decision 38, as the owner instructed.
- `handover/RELEASE-CHECKLIST.md` — the count and the nightly wave item (the brief names the checklist).
- The base lint repair of `539e015a` (six files, mechanical; the base `lane/rfam` was ruff-red).

## Docs updated

- `rcp-ndcg-vllm/README.md` (the catalog row and the nightly exception), `docs/reference/recipes.md`
  (14 families / 20 variants), `docs/index.md`, `docs/quickstart.md` (the counts),
  `docs/how-to/serve-a-model.md` (20 recipes and the nightly exception),
  `docs/how-to/release-candidates.md` and `docs/how-to/validate-a-recipe.md` (the digest-pinned engine
  environment), `handover/RELEASE-CHECKLIST.md` (20 retrieval recipes and the nightly wave item),
  `CHANGELOG.md` (the entry and the pplx recipe count). Sweep commands:
  `git grep -n -i -e embeddinggemma -e embedding_gemma -- docs README.md REPRODUCIBILITY.md skills examples
  experiments mkdocs.yml`, `git grep -n "19 recipes\|19 retrieval\|19 public\|19 shipped\|stock .vllm/vllm-openai"
  -- docs rcp-ndcg-vllm/README.md handover/RELEASE-CHECKLIST.md CHANGELOG.md`.

## For the next lanes

- The GPU wave for `embeddinggemma-2`: the nightly image's transformers version, the architecture load, the
  media stage against the engine, the video pin, and the switch-to-release check when a vLLM release carries
  `02b83919aa2e` with transformers >= 5.19.0.
- A gemma4-aware stub-engine option would let the offline media stage compare the engine count too (its
  `smart_resize` path cannot emulate the soft-token resize).
- The `ImagePolicy.max_soft_tokens` budget is not yet part of the recipe schema's pixel-pin agreement check;
  the next gemma4 recipe with a pinned non-stock budget needs it.
