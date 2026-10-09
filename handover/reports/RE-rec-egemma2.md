# Report: lane `rec-egemma2` — the recipe family `embeddinggemma-2` (text, image, video)

**Status: BLOCKED-OWNER.** Step 1 decided the engine: the stock image `vllm/vllm-openai:v0.31.0` cannot serve
`google/embeddinggemma-2`, and the brief says to stop there. No recipe family, tests, pairs file, catalog row,
CHANGELOG entry or checklist change was written: every one of them would declare an engine that cannot load the
checkpoint. The exact gaps, the vLLM-main commits and the three options for the owner are below, followed by the
model facts already verified at the pinned revision so the next lane can start from them.

**Base and merge:** the lane was built on the latest `lane/rfam` @ `dc6c5986` (its head at session start and at
report time). `rfam` was **not** merged into `rfc-0001` (`git merge-base --is-ancestor lane/rfam rfc-0001` exits
1), so per the brief the `rfc-0001` merge was not due and the latest `lane/rfam` was the base. The base head
carried six pre-existing lint/format failures (below, "Files outside scope"), which this lane repaired
mechanically so `bin/gate` can run.

## Commits

1. `539e015a` — the base `lane/rfam` lint repair (six files; see "Files outside scope"; mechanical only).
2. This report's commit (the lane's final message carries its hash).

## Step 1 — the engine decision (every claim reproduced in the lane scratch)

**Can vLLM v0.31.0 serve it? No.** Three blockers; the first two are fatal on their own, the third (the missing
video backend and config verification) is fatal to the video modality and the serving defaults:

1. **The architecture is not in the v0.31.0 registry.** The tag is
   `db9527a46873454610df6dbedf79a36d6bf1a7f6` (`vllm/model_executor/models/registry.py`). Its Gemma entries are
   `GemmaForCausalLM` (:106), `Gemma2ForCausalLM` (:107), `Gemma3ForCausalLM` (:108), `Gemma3nForCausalLM` (:110),
   `Gemma4ForCausalLM` (:111), `Gemma2Model`/`Gemma3TextModel` (:227-228), `Gemma3ForConditionalGeneration` (:409),
   `Gemma3nForConditionalGeneration` (:410-413), `Gemma4ForConditionalGeneration` (:418),
   `Gemma4UnifiedForConditionalGeneration` (:419-421), `Gemma4DSparkModel` (:658), `Gemma4MTPModel` (:684). A
   case-insensitive `git grep -i "embeddinggemma\|embedding_gemma"` over `vllm/` at the tag returns no hit at all:
   there is no `EmbeddingGemma2Model` and no `embedding_gemma2` model file.
2. **The image's transformers lacks the checkpoint's config and processor.** v0.31.0's
   `requirements/common.txt:10` pins `transformers >= 5.10.4, < 5.18.0`; the image (`vllm/vllm-openai:v0.31.0`)
   carries **transformers 5.17.0** (GPU-E1: "the reference environment over the image's torch 2.13.0 and
   transformers 5.17.0"). The released 5.17.0 and 5.18.0 wheels contain **no** `transformers/models/embedding_gemma2/`
   package and no `embedding_gemma2` entry in `modeling_auto.py` (the discriminator; the top-level
   `__init__.py` is a generated lazy structure in every release and carries no model-class names). The class
   landed in the released
   **transformers 5.19.0** wheel (published 2026-10-06T16:38:56Z): `configuration_embedding_gemma2.py` (200 lines),
   `modeling_embedding_gemma2.py` (933), `processing_embedding_gemma2.py` (410),
   `video_processing_embedding_gemma2.py` (374). The checkpoint was saved with
   `transformers_version: 5.18.0.dev0` (`config.json`), but the first release that carries the class is 5.19.0.
   The checkpoint has **no `auto_map`** (`config.json`), so `--trust-remote-code` cannot supply any of it.
3. **The engine-side multimodal machinery for it is not in v0.31.0 either.** v0.31.1rc0 adds
   `EmbeddingGemma2VideoBackend` to `vllm/multimodal/video.py` (+178 lines; v0.31.0 has no Gemma video backend)
   and the config verification `EmbeddingGemma2ModelConfig(Gemma4Config)` (+105 lines in
   `vllm/model_executor/models/config.py`). The model class itself (1,142 lines) imports the missing HF config
   and processor (`transformers.video_utils.VideoMetadata` does exist in 5.17.0/5.18.0); the Gemma4 helpers it
   reuses (`Gemma4MLP`,
   `gemma4_layer_config`, `Gemma4ForConditionalGeneration`, `Gemma4MultiModalProcessor`, `Gemma4ProcessingInfo`,
   `Gemma4MultimodalEmbedder`, `_get_max_soft_tokens`, `_SUPPORTED_SOFT_TOKENS`) do exist at v0.31.0, so the
   vLLM-side gap is the model file, the config verification and the video backend.

**Does vLLM `main` have it? Yes.**
`02b83919aa2e` (2026-10-06T17:43:38Z, PR #60254, "[Model] Support EmbeddingGemma2 multimodal pooling
architecture") adds `vllm/model_executor/models/embedding_gemma2.py` (+1,142), `vllm/model_executor/models/config.py`
(+106), `vllm/multimodal/video.py` (+178), one registry entry, `vllm/v1/attention/ops/triton_prefill_attention.py`
(+3/-2) and a 687-line test file; `bb87d227d4b9` (2026-10-06T19:10:00Z) is a mypy follow-up. At report time `main`
is `2c99ee9333821030daf84718b2a76ea28d94ba3c` (2026-10-09T09:29:03Z), and its `requirements/common.txt` pins
`transformers >= 5.16.1, < 5.20.0` (so it resolves 5.19.0, the version with the class).

**The tag between the two, `v0.31.1rc0`, does not unblock it.** The tag
(`e37e51dd246cf421c554e7d6f53e178e3fd29085`, 2026-10-06T18:52:05Z) contains the model file, but its
`requirements/common.txt` still pins `transformers >= 5.16.1, < 5.19.0`: an image built from that tag would carry
transformers 5.18.0, whose wheel has no `EmbeddingGemma2Config`/`EmbeddingGemma2Processor`, so the engine would
fail when it imports the architecture. There is also **no published image** for the tag: Docker Hub returns 404
for `vllm/vllm-openai:v0.31.1rc0`. The first nightly images whose commit allows 5.19.0 are the 2026-10-08
(`81198e97ba...`) and 2026-10-09 (`8cbd5d03006c33185f402249ff2b448efd594986`) builds; the post-commit 2026-10-06
evening (`bb87d227...`) and 2026-10-07 (`43b4aaea...`) nightlies contain the model but still pin `< 5.19.0` and
are unusable for it (the 2026-10-06 morning build predates the commit and has no model file at all).

**Could a plugin in `rcp_ndcg_vllm.models` serve it on v0.31.0?** It could in principle, but it is a backport,
not a lean plugin, and it breaks decision 2's freeze rule unless it vendors everything. Honest size:

- vLLM side: the model file (1,142 lines), `EmbeddingGemma2ModelConfig` (105), `EmbeddingGemma2VideoBackend`
  (178), one registry entry and the three-line triton change: 1,142 + 105 + 178 + 1 + 3 = **1,429 lines**.
- HF side, which the image's transformers 5.17.0 lacks and decision 2 does not allow upgrading (only a
  pure-Python plugin wheel installed `--no-deps`, under the freeze-diff guard): `configuration_embedding_gemma2.py`
  (200 lines), `processing_embedding_gemma2.py` (410), `video_processing_embedding_gemma2.py` (374) plus the
  `AutoConfig`/`AutoProcessor` registration shims ≈ **1,000 lines**; the vLLM class also instantiates the vision
  tower through `AutoModel.from_config` and needs the `gemma4_vision` class, which 5.17.0 does have.
- **Total ≈ 2,400+ lines across ~7 files, plus tests (upstream's own test is 687 lines) and the
  `vllm/multimodal/video.py` backend work.** It duplicates upstream code that a release image will ship, needs
  tracking against upstream, and carries the whole model's numerics on this repo's shoulders. Not recommended.

## The options for the owner

1. **Plugin on v0.31.0** — as estimated above; the only way to keep the stock image. Large, duplicates upstream,
   and the plugin would be deleted once the image moves.
2. **A nightly image pinned by digest containing the commit, with the switch-to-release note.** The newest
   nightly I checked (2026-10-09T07:24:03Z, commit `8cbd5d03006c33185f402249ff2b448efd594986`) contains the model
   and allows transformers 5.19.0:
   `vllm/vllm-openai:nightly-8cbd5d03006c33185f402249ff2b448efd594986@sha256:b25e8a046fdbe948987b3dba06ef2cf9ea0f02b36d9482a25113a445ee52ad21`
   (the CUDA 12.9 variant is
   `cu129-nightly-8cbd5d03006c33185f402249ff2b448efd594986@sha256:e3b7fa9a57277484cc332b33c38ef556c4c489dae454dd2ce596a9b039a109fb`).
   The wave must verify the image's actual transformers version at startup (the pin allows, but does not
   guarantee, 5.19.0) and the recipe's `engine.image`/`min_version` fields would carry the nightly tag; when
   vLLM ships a release containing the commit (the next release after v0.31.0), switch the pin back to it.
3. **Defer** — keep `embeddinggemma-2` out of 0.0.1 and pick it up when a release image contains
   `02b83919aa2e` with `transformers >= 5.19.0` in its pin.

## Model facts verified at the pinned revision (for the next lane)

All at `google/embeddinggemma-2` @ `914f7f89142e33e77833254d9c9b90c3cef7303b` (Hub API: `gated: false`,
`private: false`, `license: apache-2.0`):

- `config.json`: `architectures: [EmbeddingGemma2Model]`, `model_type: embedding_gemma2`;
  `text_config.model_type: embedding_gemma2_text` (24 layers, `sliding_window: 512`, full attention every 6th
  layer, `max_position_embeddings: 262144`); `vision_config.model_type: gemma4_vision`;
  `audio_config.model_type: gemma4_audio`; `image_token_id`/`video_token_id`/`audio_token_id` present;
  `transformers_version: 5.18.0.dev0`; **no `auto_map`**.
- `1_Pooling/config.json`: `pooling_mode: mean`, `include_prompt: true`, `embedding_dimension: 768`.
- `sentence_bert_config.json`: `modality_config` for text, image, audio, video and message, all
  `method: forward`, `method_output_name: last_hidden_state`.
- `config_sentence_transformers.json`: the task prompts (`task: search result | query: ` for the query side;
  `title: none | text: ` for the document side), `similarity_fn_name: cosine`.
- `processor_config.json`: `processor_class: EmbeddingGemma2Processor`; image processor `Gemma4ImageProcessor`,
  `image_seq_length: 280`; video processor `EmbeddingGemma2VideoProcessor`, `fps: 1`, `max_frames: 32`,
  `max_soft_tokens: 140`, `overflow_strategy: uniform`; audio feature extractor `Gemma4AudioFeatureExtractor`
  (out of scope for 0.0.1 per the brief's scope sentence; the deferral itself is
  `handover/10-data-io-and-mteb.md:135` — "Deferred, both additive: conversation-style queries and audio". The
  brief labels it decision 27, but decision 27 in `handover/00-MASTER.md` is the title/MTEB decision; when the
  family is written it must declare audio refused).
- The card: 8K-token context window; 280 tokens per image (default), 140 tokens per video frame (default);
  MRL truncation at 128/256/512 dims; **bfloat16 or float32, never float16** (fp16 returns NaN or silently
  degraded embeddings); the vision soft-token budget is configurable over 70..1120.
- vLLM `main`'s implementation (the shape any unblocking would serve): bidirectional encoder-only attention
  (`AttentionType.ENCODER_ONLY`, `_WINDOW_OFFSET = 1` for the HF `|q-k| <= W` mask), mean pooling
  (`@default_pooling_type(seq_pooling_type="MEAN", tok_pooling_type="ALL")` +
  `DispatchPooler.for_embedding`), and a config that caps `max_model_len` at 8192 (from
  `sentence_bert_config.json` or the default) unless explicitly set — the card's 8K, not the config's
  `max_position_embeddings: 262144`.
- Not done (blocked): the served `max_model_len`/32-bit and exactly-max hang checks, the engine's media
  preparation, the video-sampling pin, the family, its reference, tests, pairs, docs and checklist rows.

## What changed

Nothing in the product. The report, plus a mechanical repair of six files' pre-existing lint/format failures
inherited from the base `lane/rfam` (listed under "Files outside scope"; no behaviour changed). The recipe
family directory, its tests, pairs file, catalog rows, CHANGELOG and RELEASE-CHECKLIST counts were deliberately
not touched (brief: stop at step 1).

## Verification

- **Round 1, lens A (correctness, PASS with 6 minor findings)**: the verifier reproduced every step-1 claim with
  its own commands (the v0.31.0 registry and requirements, the transformers 5.17.0/5.18.0/5.19.0 wheels, the main
  commits and their file counts, the v0.31.1rc0 tag and its pin, the nightly digests, the model facts at the
  pinned revision, the plugin import check and the stop-at-step-1 scope). Findings, all fixed in this report:
  (a) `vllm/multimodal/video.py` is +178 lines, not +179 (the backend block is 178 lines), so the vLLM-side
  total is 1,429; (b) "the 2026-10-06/07 nightlies contain the model" was over-broad — the 2026-10-06 morning
  build predates the commit; (c) the `transformers/__init__.py` criterion is vacuous (generated lazy structure);
  the package directory plus the auto-mapping entry is the discriminator; (d) the brief's "decision 27" for the
  audio deferral is a mislabel — decision 27 in `handover/00-MASTER.md` is the title/MTEB decision, and the
  audio deferral is `handover/10-data-io-and-mteb.md:135`; (e) "any one of which is fatal" softened (the video
  backend/config gap is fatal to video and the serving defaults, not to loading the text tower); (f)
  `VideoMetadata` exists in 5.17.0/5.18.0, so "missing" now covers only the config and processor.
- **Round 1, lens B (regressions and hygiene, FAIL, one major)**: scope correct (only the report changed),
  stop-at-step-1 correct, no product or public-surface change, no false test claims, no private names, no AI
  attribution. Major finding: the base `lane/rfam` @ `dc6c5986` was lint-red (`ruff format --check`: 4 files;
  `ruff check`: 5 errors in 4 files), so the planned gate could not pass; this lane repaired the six files
  mechanically (below, "Files outside scope") and the report discloses it. Minor findings, all fixed: the gate
  line was a forward reference under "commands run" (reworded); the base/merge statement was missing (added);
  the decision-27 citation (fixed as above); the COMMON-mandated "Docs updated" section was missing (added).
- **Round 2 (one fresh confirmation verifier, lens A+B)**: _to be filled after the round-2 result._

## Checks (commands run)

- `git clone --depth 1 --branch v0.31.0 https://github.com/vllm-project/vllm.git` into the lane scratch;
  `git log --oneline -1` → `db9527a [Misc] Add Transformers version upper bound in requirements (#59614)`.
- `grep -rn -i "embeddinggemma\|embedding_gemma" vllm/` in the v0.31.0 clone → no hit; registry lines listed above.
- `curl` of the Hub files at the pinned revision (`config.json`, `sentence_bert_config.json`,
  `config_sentence_transformers.json`, `modules.json`, `processor_config.json`, `1_Pooling/config.json`,
  `2_Normalize/config.json`, `tokenizer_config.json`, `chat_template.jinja`, `README.md`) and of the Hub API
  (`gated`, `license`, siblings).
- `curl` of `transformers-5.17.0`, `5.18.0`, `5.19.0` wheels from PyPI; the 5.19.0 wheel contains
  `transformers/models/embedding_gemma2/{configuration,modeling,processing,video_processing}_embedding_gemma2.py`;
  5.17.0 and 5.18.0 contain none.
- GitHub API: the support commit's file list (+1,142/+106/+178/+1/+687/+3-2), the commit dates, the v0.31.1rc0
  tag and its `requirements/common.txt` (`< 5.19.0`), the main HEAD.
- Docker Hub API: `vllm/vllm-openai:v0.31.1rc0` → 404; the nightly tag digests and the 2026-10-08/09 nightlies'
  `requirements/common.txt` (`>= 5.16.1, < 5.20.0`).
- Base lint repair: `uv run --no-sync ruff check --fix` + `uv run --no-sync ruff format` on the six files below;
  afterwards `ruff format --check .` → "517 files already formatted", `ruff check .` → "All checks passed!".
- `bin/gate lane/rec-egemma2` is run on the final head after the round-2 confirmation; its SUMMARY is reported
  in the lane's final message (the report file itself does not carry a forward-referenced result).

## Open questions

- The owner picks option 1, 2 or 3. If option 2, the switch-to-release note belongs in the recipe's `sources`
  and in the wave's verification list (an image tag alone is not the pin: the digest is).
- The audio encoder is present in the checkpoint and out of scope for 0.0.1 (the brief's scope sentence;
  `handover/10-data-io-and-mteb.md:135`); when the family is
  written it must declare audio refused, and the vLLM `main` class loads the audio tower unconditionally unless
  the checkpoint is loaded with the audio config omitted (the card's selective-loading `config_kwargs`), which
  is another unverified serving fact.
- The card's `title: none | text: ` document prompt and the task-instruction field (decision 33) need the same
  mapping decision every instruction-prefix model got; not settled here because the family was not written.

## CHANGELOG entry

None — no public surface changed.

## Public surface changes

None.

## Files outside scope

The base `lane/rfam` @ `dc6c5986` was red under the lane's own fast checks, so the gate could not pass. This
lane repaired exactly the lint/format failures, with no behaviour change (mechanical; `ruff check --fix` plus
`ruff format`):

- `rcp-ndcg-test/tests/recipes/test_qwen3_reranker.py` — removed the unused `load_recipe` import (F401).
- `rcp-ndcg-test/tests/test_contract_helper.py` — removed the unused `default_recipes_root` import (F401) and
  wrapped the 121-char call (E501).
- `rcp-ndcg-test/tests/test_quality.py` — import block sorted (I001).
- `rcp-ndcg-test/tests/test_recipe.py` — wrapped the 126-char assignment (E501).
- `rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipe.py` — one raise wrapped by the formatter.
- `rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/qwen3-reranker/reference.py` — one call wrapped by the formatter.

These belong to the `lane/rfam` owner; the fix is listed here so the rfam lane can take it over when it commits
its own tree (the same failures are present in `wt-rfam`).

## Docs updated

None — no public name, flag, config key, exit code or behaviour changed, so no page in `docs/`, `README.md`,
`REPRODUCIBILITY.md`, `skills/`, `examples/` or `mkdocs.yml` is now false. The sweep,
`git grep -n -i -e embeddinggemma -e embedding_gemma -- docs README.md REPRODUCIBILITY.md skills examples
experiments mkdocs.yml`, returns no hit: the model does not exist in the repository at all (the lane wrote no
recipe).

## For the next lanes

- If the owner picks the nightly image, the family can be built in the rfam format with the engine block
  `image: vllm/vllm-openai:nightly-8cbd5d03006c33185f402249ff2b448efd594986` (or the `cu129-` variant) plus the
  digest in `sources`, `min_version` naming the nightly's vLLM dev version, and the switch-to-release note; the
  model facts above are the per-variant facts.
- The step-2 items still need the tag source read at the nightly commit (`embedding_gemma2.py`'s processor
  geometry and `video.py`'s `EmbeddingGemma2VideoBackend` frame indices), the 32-bit/at-budget hang checks, and
  the client-side video pin so the client counts the same frames the engine samples (the qwen3-vl lesson).
- The engine's transformers version must be checked on the wave, not assumed from the pin.
