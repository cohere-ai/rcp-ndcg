# MRL model cards: every embedding and multi-vector variant's MRL kind and set

Status: research only, 2026-10-09. This file is the deliverable of lane `mrl-cards`; lane `mrl-core` (decision 39)
owns the product changes it implies. No code, no recipe is changed here.

The spec fixes, for every embedding and multi-vector variant the owner named -- shipped, in flight and the new
sizes -- the Matryoshka kind (`truncation` / `projection` / `none`), the supported dimension set and its source, where
the full-width vector comes from (the backbone's hidden state, a sentence-transformers `Dense`, or a projection
inside the model), and whether a served engine-side cut exists at vLLM v0.31.0: the per-request `dimensions` field
of `/v1/embeddings`, the serve-time `pooler_config.dimensions` of `/pooling`, or neither. Projection kinds carry the
file, the matrices' shapes and the order of application. Every fact carries its evidence: the Hub repo at a pinned
revision (the 40-hex `sha` from the Hub API and a `file:key`), or the vLLM tag and a `file:line`. Section 4 names
every gap, including cards that claim MRL without a set and sets the engine path cannot serve.

Sources and method:

- Hub facts were read anonymously through the public Hub API on 2026-10-09 (`https://huggingface.co/api/models/<repo>`
  for the `sha`, the repo tree at that revision, and the raw files `config.json`,
  `config_sentence_transformers.json`, `sentence_bert_config.json`, `modules.json`, `1_Pooling/config.json`,
  `1_Dense/config.json`, `2_Normalize/config.json`, `2_MultiVectorMask/config.json`, `README.md`, the remote
  `modeling_*.py` / `custom_st.py` code, and the safetensors headers). For the eight shipped recipes the revision is
  the recipe's own `revision:` pin, re-verified against the Hub API; the other variants are pinned at the API `sha`
  on 2026-10-09. Every `sha` below is the full 40-hex value.
- vLLM facts are `file:line` at tag `v0.31.0` (commit `db9527a46873454610df6dbedf79a36d6bf1a7f6`), the image
  `vllm/vllm-openai:v0.31.0` the recipes serve on. The tag's own transformers pin is
  `requirements/common.txt`: `transformers >= 5.10.4, < 5.18.0`.
- Product facts (the two MRL paths, the records, the fake engine, the harness) are `file:line` in this tree.
- "Card" below means the model card `README.md` at the pinned revision; "reference code" means the remote `.py`
  files the checkpoint ships (or, for the ST-metadata models, the modules the card's own loading path runs).

---

## 1. The engine facts at vLLM v0.31.0 (one home)

### 1.1 Two routes, one head order

| Route | Models | Head order at v0.31.0 | Per-request `dimensions` |
|---|---|---|---|
| `/v1/embeddings` (task `embed`) | dense, one vector per item | projector (ST `Dense` from `modules.json`, else none) -> slice -> activation (`PoolerNormalize`, L2) | accepted: `vllm/entrypoints/pooling/base/protocol.py:345-348` (`EmbedRequestMixin.dimensions`) -> `vllm/entrypoints/pooling/embed/protocol.py:41-44` -> `vllm/model_executor/layers/pooler/seqwise/heads.py:78-109` (slice at `:97`, activation at `:109`) |
| `/pooling`, task `token_embed` | multi-vector, one vector per kept token | projector (the model's `custom_text_proj`) -> slice -> activation | **refused**: `vllm/entrypoints/pooling/pooling/serving.py:62-66` raises "dimensions is currently not supported" for a request that carries it; the serve-time `pooler_config.dimensions` is copied into the request params instead: `vllm/pooling_params.py:120-127`, reaching `vllm/model_executor/layers/pooler/tokwise/heads.py:87-96` (slice at `:96`, activation after) |

Both orders are **cut before L2**; a client-side cut must mirror that (`slice` then `normalize`).

### 1.2 The gates on a cut

`PoolingParams._set_default_parameters` (`vllm/pooling_params.py:164-193`) applies three checks to any non-`None`
`dimensions` on a task that lists it (`embed` and `token_embed` both do: `:80-84`):

1. `model_config.is_matryoshka` must be true (`:175`), i.e. the HF config carries a truthy `matryoshka_dimensions`
   or `is_matryoshka` (`vllm/config/model.py:2023-2025`). **No checkpoint in this spec declares either key in
   `config.json`** (checked for all 22); a recipe must add it through `serve.hf_overrides`.
2. `1 <= k <= embedding_size` (`:174`, `:182-186`), where `embedding_size` is the HF config's `embedding_size`
   override, else the last ST `Dense` module's `out_features`, else the hidden size
   (`vllm/config/model.py:2076-2083`).
3. Membership in `model_config.matryoshka_dimensions` when that list is non-`None` (`:188-192`); a checkpoint that
   declares only `is_matryoshka: true` accepts any integer in range.

The serve-time field is `PoolerConfig.dimensions` (`vllm/config/pooler.py:67-71`), already in the recipe schema's
pinned field list (`rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipe.py:51-64`).

### 1.3 Which architectures v0.31.0 serves

Native registry entries (`vllm/model_executor/models/registry.py`): `Qwen3ForCausalLM:201`, `Qwen3_5ForCausalLM:203`,
`Gemma3TextModel:228`, `JinaEmbeddingsV5Model:232`, `ColQwen3_5:284`, `Qwen3VLForConditionalGeneration:591`. The
`Model` suffix rule maps `Qwen3Model` / `Qwen3_5Model` to the `ForCausalLM` class under `runner_type: pooling`,
`convert_type: embed` (`vllm/config/model.py:2284-2291`, `:2296-2313`). `JinaEmbeddingsV5Model.__new__` dispatches
`is_decoder: false` checkpoints to the bidirectional EuroBERT encoder class
(`vllm/model_executor/models/jina.py:305-355`), so `jina-embeddings-v5-text-nano` is native too. No
`embedding_gemma2` symbol exists anywhere in the tag.

The plugin models are outside the registry: `topk-embed` (`TopkEmbedModel` subclasses `ColQwen3_5Model`, whose
`pooler_for_token_embed` builds the projector -> slice -> L2 chain), and the pplx plugin's
`PplxContextualForPooling` / late class (both build vLLM's `TokenEmbeddingPoolerHead`, so the slice is the
engine's own). `PPLXQwen3Model` (pplx-embed-v1) has **no** engine class at v0.31.0.

### 1.4 The product's MRL paths and records today (what `mrl-core` builds on)

- Dense: `EmbeddingEndpoint.dimensions` (`rcp-ndcg/src/rcp_ndcg/inference/config.py:344`), sent as the request's
  `dimensions` (`inference/adapters/embeddings.py:344-345`), reply width checked (`:214-217`); the hosted profiles
  refuse it (`SUPPORTS_DIMENSIONS = False` at `:368`, `:423`, `:456`). No client-side cut exists on the dense path.
- Multi-vector: `PoolingEndpoint.mrl_dim` (`config.py:437`), applied client-side as cut-then-renormalise in
  `inference/clients/pool.py:370-386` through the one home `data/postprocess.py:114-131` (`mrl_cut` = slice then
  `l2_normalize`); `PoolingEndpoint.dimensions` is refused as inert (`config.py:465-477`) and `mrl_dim >= dim` is
  refused (`:453-462`).
- No `matryoshka_dims` field exists on either endpoint (whole-tree `git grep`: none). Identity roles:
  `dimensions` CONTENT (`config.py:324`), `mrl_dim` CONTENT (`:429`). Fingerprint:
  `dimensions: "request"` (`rcp-ndcg-test/src/rcp_ndcg_test/fingerprint.py:93`), `mrl_dim: "post_processing"`
  (`:114`). `ChangeMechanism` has no MRL member (`data/text_budget.py:94-101`), so the cut is recorded nowhere
  except the identity.
- The fake engine generates a fresh `dimensions`-wide surrogate for `/embeddings` (`engines.py:1026-1035`) and
  validates no `is_matryoshka`/range/set; `/pooling` ignores `dimensions` entirely (`:1063-1120`) where real vLLM
  refuses it. The observation set has one `dimensions` probe (`observe/requests.py:803-804`, falling back to a
  hard-coded 32); no `mrl_dim` probe, no MRL control, no MRL in conformance or the equivalence stages
  (`git grep -i mrl` over `rcp_ndcg_test/equivalence/` and `observe/controls.py`: no matches).

---

## 2. The variants at a glance

`set` is the card-supported set; `-` means the card declares none. `full width` names where the served vector's
width comes from: `backbone` (hidden state + pooling head), `Dense` (an ST `Dense` module), or `projection` (a
learned matrix inside the model). The engine columns state the capability **if the recipe declares the
`is_matryoshka` gate**; the shipped recipe's current declaration is in section 3.

| Variant (repo @ revision) | Role | MRL kind | Set (source) | Full width | `/v1/embeddings` per-request | `/pooling` serve-time `pooler_config.dimensions` | Projection file / shapes / order |
|---|---|---|---|---|---|---|---|
| `Qwen/Qwen3-Embedding-0.6B` @ `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` | embed | truncation | 32..1024 (card prose) | backbone 1024 | yes (gate) | n/a | - |
| `Qwen/Qwen3-Embedding-4B` @ `5cf2132abc99cad020ac570b19d031efec650f2b` | embed | truncation | 32..2560 (card prose) | backbone 2560 | yes (gate) | n/a | - |
| `Qwen/Qwen3-Embedding-8B` @ `1d8ad4ca9b3dd8059ad90a75d4983776a23d44af` | embed | truncation | 32..4096 (card prose) | backbone 4096 | yes (gate) | n/a | - |
| `Qwen/Qwen3-VL-Embedding-2B` @ `9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda` | embed | truncation | 64..2048 (card prose) | backbone 2048 | yes (gate) | n/a | - |
| `Qwen/Qwen3-VL-Embedding-8B` @ `2c4565515e0f265c6511776e7193b22c0968ddc7` | embed | truncation | 64..4096 (card prose) | backbone 4096 | yes (gate) | n/a | - |
| `jinaai/jina-embeddings-v5-text-small` @ `dd76d535f5447ca3897a9c893fb1e612ead98192` | embed | truncation | 32, 64, 128, 256, 512, 768, 1024 (card table) | backbone 1024 | yes (gate; recipe declares it) | n/a | - |
| `jinaai/jina-embeddings-v5-text-nano` @ `8a7f00aac812071b69403df470f1038ec85f8925` | embed | truncation | 32, 64, 128, 256, 512, 768 (card table) | backbone 768 | yes (gate) | n/a | - |
| `Octen/Octen-Embedding-0.6B` @ `d715b32ee68f057b54dff09fc93c23485bc403d3` | embed | none | - | backbone 1024 | would need a gate; not card-supported | n/a | - |
| `Octen/Octen-Embedding-4B` @ `fea468fae3f0caffbae8a12ba792d1c394b6277d` | embed | none | - | backbone 2560 | would need a gate; not card-supported | n/a | - |
| `Octen/Octen-Embedding-8B` @ `5adcfa292e712091dfc30f0e97f0b2282e6cc66c` | embed | none | - | backbone 4096 | would need a gate; not card-supported | n/a | - |
| `zeroentropy/zembed-1-embedding` @ `cf13c81f3274394053d166740294f7eea4586f7a` | embed | **projection** | 2560, 1280, 640, 320, 160, 80, 40 (card prose) | backbone 2560 (served) | yes as a slice, but a slice is WRONG; no engine path applies the projections | n/a | `projections.safetensors`, six F32 matrices, chained `2560->1280->640->320->160->80->40` |
| `topk-io/topk-embed-v1-small` @ `e54485ebab921f2c18c4d092b3f4c40dcca26781` | multi_vector | truncation after projection | 64, 128, 256, 512, 1024, 2048 (card tables); any `1..2048` (code) | `head.weight` projection 2048 | refused | yes (gate; slice inherited) | `head.weight` `[2048, 2048]`; projection -> slice -> L2 |
| `topk-io/topk-embed-v1-xsmall` @ `210ebf2a25fb7128480f9b9c8f228e8d65c433a7` | multi_vector | truncation after projection | 64, 128, 256, 512, 1024 (card tables); any `1..1024` (code) | `head.weight` projection 1024 | refused | yes (gate; slice inherited) | `head.weight` `[1024, 1024]`; projection -> slice -> L2 |
| `perplexity-ai/pplx-embed-v1-0.6b` @ `2c4d510dd4a732063c31a0f70193e35067b51fd8` | embed | truncation (card-only claim) | - | backbone 1024 + int8 quantizer | **no v0.31.0 class** | **no v0.31.0 class** | - |
| `perplexity-ai/pplx-embed-v1-4b` @ `06456497a00540a582918fe8dcd3a5eabb207772` | embed | truncation (card-only claim) | - | backbone 2560 + int8 quantizer | **no v0.31.0 class** | **no v0.31.0 class** | - |
| `perplexity-ai/pplx-embed-v2-context-9b-preview` @ `b667039ee8b438a6350fbc91bbcecd86f9d363ba` | multi_vector | truncation of the projected output | 1024, 2048 (card Matryoshka section) | projection 2048 | refused | yes (gate; plugin head) | `contextual_head.safetensors` `contextual_projection.weight` `[2048, 4096]` F32; span-mean -> projection -> slice -> L2 |
| `perplexity-ai/pplx-embed-v2-late-0.6b` @ `8fc2de24534aa3610d85fa59c463313a5f096455` | multi_vector | none | - | `1_Dense` 128 | refused | technically yes; not card-supported | `1_Dense/model.safetensors` `linear.weight` `[128, 1024]` F32; Dense -> mask -> L2 |
| `perplexity-ai/pplx-embed-v2-late-9b` @ `0f49a9977fe06b83377d598094c5c0204ce18ad9` | multi_vector | none | - | `1_Dense` 128 | refused | technically yes; not card-supported | `1_Dense/model.safetensors` `linear.weight` `[128, 4096]` F32; Dense -> mask -> L2 |
| `microsoft/harrier-oss-v1-270m` @ `31de22b673913c7d658c0f03f792d77c2dcf8ebd` | embed | none | - | backbone 640 | would need a gate; not card-supported | n/a | - |
| `microsoft/harrier-oss-v1-0.6b` @ `f9b9dc8d367d443f2479d27aa5d8d2850c0774ee` | embed | none | - | backbone 1024 | would need a gate; not card-supported | n/a | - |
| `microsoft/harrier-oss-v1-27b` @ `0c0fc62f6d8af9e8604cb818c412301b103a0093` | embed | none | - | backbone 5376 | would need a gate; not card-supported | n/a | - |
| `google/embeddinggemma-2` @ `914f7f89142e33e77833254d9c9b90c3cef7303b` | embed | truncation | 768, 512, 256, 128 (card MRL section) | projection 768 | **no v0.31.0 class** | **no v0.31.0 class** | `language_model.embedding_projection.weight` `[768, 512]` BF16; per-token projection -> mean pool -> slice -> L2 |

The shipped recipe pins (section 3) are identical to the API `sha` except `pplx-embed-v2-context-9b-preview`, whose
recipe pins `b667039ee8b438a6350fbc91bbcecd86f9d363ba` while Hub `main` is now
`e2c06a779d45e8db53066735be1b98ffe3412ca3` (drift, section 4 G10).

---

## 3. Per-variant evidence

### 3.1 Qwen3-Embedding 0.6B / 4B / 8B (`embed`)

Pinned: 0.6B `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`, 4B `5cf2132abc99cad020ac570b19d031efec650f2b`, 8B
`1d8ad4ca9b3dd8059ad90a75d4983776a23d44af` (all three match the Hub API `sha`; 0.6B is the shipped recipe pin).

- **Kind: truncation.** The card's model table marks `MRL Support: Yes` (0.6B `README.md:44`, 4B `:45`, 8B
  `:44`), and the model-details bullet gives the card's set as a range: "Embedding Dimension: Up to 1024, supports
  user-defined output dimensions ranging from **32 to 1024**" (0.6B `README.md:36`; 4B `:36` "32 to 2560"; 8B `:34`
  "32 to 4096"). The note at `README.md:52` (8B `:50`) defines the column as "supports custom dimensions for the
  final embedding".
- **Set: card prose only.** `config.json` declares no `matryoshka_dimensions` and no `is_matryoshka` (checked at all
  three revisions); `config_sentence_transformers.json` carries only `prompts` and `similarity_fn_name` (no
  `truncate_dim`); the reference code has no cut — the card's `truncation=True` is tokenizer truncation
  (`README.md:157`), not an MRL cut. So the "set" is the card's continuous range; there is no discrete list the
  engine could enforce and no 32 floor in the engine (section 1.2 gate 2 starts at 1).
- **Full width: backbone.** `modules.json` = `Transformer`, `1_Pooling`, `2_Normalize`; `1_Pooling/config.json`
  `pooling_mode_lasttoken: true`, `word_embedding_dimension` 1024/2560/4096. The served head is the seqwise
  last-token + `PoolerNormalize` (no projector).
- **Engine path.** `/v1/embeddings` per-request `dimensions` is accepted after the `is_matryoshka` gate. The shipped
  0.6B recipe declares `serve.hf_overrides: {}` (`rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/qwen3-embedding-0.6b/
  recipe.yaml:26`) and `client.dimensions: null` (`:75`), so as shipped the engine would **refuse** any
  `dimensions` (HTTP 400 on the first request). The recipe's note "MRL 32..1024 is available per request via the
  OpenAI dimensions field" (`recipe.yaml:75-76`) is the engine's capability, not a declaration this recipe makes;
  see G3. Serve-time `pooler_config.dimensions` is the same head and the same gate.
- The 4B/8B checkpoints are structurally identical (same `modules.json`, same `1_Pooling` shape, same card prose);
  they are not in the tree yet (the `rfam` family layout and the new sizes are in flight).

### 3.2 Qwen3-VL-Embedding 2B / 8B (`embed`)

Pinned: 2B `9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda` (shipped recipe pin, matches the API `sha`), 8B
`2c4565515e0f265c6511776e7193b22c0968ddc7` (API `sha`).

- **Kind: truncation.** The card table marks `MRL Support: Yes` (2B `README.md:52`, 8B `:54`) and the details bullet
  gives the range: "Embedding Dimension: Up to 2048, supports user-defined output dimensions ranging from **64 to
  2048**" (2B `README.md:44`; 8B `:45` "64 to 4096"; the column note is `:59`/`:60`).
- **Set: card prose only.** `config.json` has no `matryoshka_dimensions`/`is_matryoshka`; the
  `config_sentence_transformers.json` has `default_prompt_name: default`, `prompts.default`, `similarity_fn_name:
  cosine` and no cut key; the card's reference script `scripts/qwen3_vl_embedding.py` (both revisions) has no
  `truncate_dim`/MRL code — the only `truncate` is `_truncate_tokens` (`:178-179`), tokenizer truncation.
- **Full width: backbone.** `modules.json` = `Transformer`, `1_Pooling`, `2_Normalize`; `1_Pooling/config.json`
  `embedding_dimension` 2048/4096, `pooling_mode: lasttoken`; the recipe pins
  `pooler_config: {seq_pooling_type: LAST}` (`recipes/qwen3-vl-embedding-2b/recipe.yaml:18`) and the head is the
  default `PoolerNormalize`.
- **Engine path.** Native `Qwen3VLForConditionalGeneration` (`registry.py:591`); `/v1/embeddings` per-request
  `dimensions` gated as above. The shipped recipe declares `hf_overrides: {}` and `dimensions: null`; the same
  G3 note applies.

### 3.3 jina-embeddings-v5-text small / nano (`embed`)

Pinned: small `dd76d535f5447ca3897a9c893fb1e612ead98192` (shipped recipe pin, matches the API `sha`), nano
`8a7f00aac812071b69403df470f1038ec85f8925` (API `sha`).

- **Kind: truncation.** The card's model table declares the discrete set: small `README.md:40`
  "Matryoshka Dimensions | 32, 64, 128, 256, 512, 768, 1024", nano `README.md:38` "32, 64, 128, 256, 512, 768".
  The set includes the full width as its last member.
- **Set: card + recipe `hf_overrides`.** Neither `config.json` declares `matryoshka_dimensions`/`is_matryoshka`
  (the comment in the small recipe says so: `recipe.yaml:30`). The small recipe mirrors the card into the serve
  block: `hf_overrides: {is_matryoshka: true, matryoshka_dimensions: [32, 64, 128, 256, 512, 768, 1024]}`
  (`recipe.yaml:33-34`), which is what makes the engine's gate pass for exactly the card's set. Nano has no recipe
  yet.
- **Full width: backbone.** `modules.json` = one `custom_st.Transformer` with kwargs `task`, `truncate_dim`;
  the card's reference code cuts in `modeling_jina_embeddings_v5.py` `encode`: `pooled = pooled[:, :truncate_dim]`
  then `F.normalize` (`:110-112`), and `custom_st.py:106-108` does the same — **cut then renormalise**, the order
  of the engine head.
- **Engine path.** Native `JinaEmbeddingsV5Model` (`registry.py:232`); nano is dispatched to
  `JinaEmbeddingsV5EncoderModel` because its `config.json` has `is_decoder: false`
  (`vllm/model_executor/models/jina.py:305-355`). `/v1/embeddings` per-request `dimensions` with the
  `is_matryoshka` + membership gates; the small recipe's declaration enables exactly its set. Serve-time
  `pooler_config.dimensions` is the same head.

### 3.4 Octen-Embedding 0.6B / 4B / 8B (`embed`)

Pinned: 0.6B `d715b32ee68f057b54dff09fc93c23485bc403d3`, 4B `fea468fae3f0caffbae8a12ba792d1c394b6277d`, 8B
`5adcfa292e712091dfc30f0e97f0b2282e6cc66c` (8B is the shipped recipe pin; all match the API `sha`).

- **Kind: none.** The cards contain no `Matryoshka`/`MRL`/`projection` sentence (grep over each `README.md`); the
  only "truncation" is the context note ">32K tokens require truncation" (8B `README.md:183`). `config.json`
  declares no `matryoshka_dimensions`/`is_matryoshka`; `config_sentence_transformers.json` has prompts and cosine
  only.
- **Full width: backbone.** `modules.json` = `Transformer`, `1_Pooling`, `2_Normalize`; `1_Pooling/config.json`
  `pooling_mode_lasttoken: true`, `word_embedding_dimension` 1024/2560/4096; `2_Normalize/config.json`
  `normalize_embeddings: true`.
- **Engine path.** `Qwen3Model` -> `Qwen3ForCausalLM` via the suffix rule; `/v1/embeddings` per-request
  `dimensions` would pass only if a recipe set `is_matryoshka`, which the cards do not support. The shipped 8B
  recipe keeps `hf_overrides: {}` and no `dimensions`, and its note says exactly that: "the checkpoint declares no
  matryoshka_dimensions, so vLLM refuses a dimensions parameter outright; do not set client.dimensions"
  (`recipes/octen-embedding-8b/recipe.yaml:117-118`).

### 3.5 zembed-1-embedding (`embed`) -- the projection kind

Pinned: `cf13c81f3274394053d166740294f7eea4586f7a` (shipped recipe pin, matches the API `sha`).

- **Kind: projection.** The card: "The model supports flexible dimension projections (2560, 1280, 640, 320, 160,
  80, 40) and quantization down to binary ... See our Technical Report (Coming soon!) for details on the
  projection method" (`README.md:33`). The 2560 is the full width; the six smaller values are learned projections.
- **Set and file.** `projections.safetensors` at the pinned revision holds six F32 tensors whose names are their
  output widths; the shapes show a chain, not independent `2560 -> k` maps:

  | Tensor | Shape | Reads |
  |---|---|---|
  | `1280` | `[2560, 1280]` | 2560 -> 1280 |
  | `640` | `[1280, 640]` | 1280 -> 640 |
  | `320` | `[640, 320]` | 640 -> 320 |
  | `160` | `[320, 160]` | 320 -> 160 |
  | `80` | `[160, 80]` | 160 -> 80 |
  | `40` | `[80, 40]` | 80 -> 40 |

  Order of application: to reach `k`, apply every matrix from 1280 down to `k` in sequence (e.g. `k=640`:
  `W1280` then `W640`); there is no direct `2560 -> 640` matrix. No public code loads them: the checkpoint's own
  `modeling_zembed.py` has no `projection`/`matryoshka`/`dimension` reference (grep: none), and the recipe note
  says "not Matryoshka slices, and no public code loads them" (`recipes/zembed-1-embedding/recipe.yaml:116-118`).
- **Full width: backbone 2560.** Served: last-token pooling + `Normalize` only (`modules.json` `Transformer`,
  `1_Pooling` lasttoken, `2_Normalize`); `hf_to_vllm`-style ST-metadata resolution reads pooling only, so the
  projection file is not part of the served head.
- **Engine path.** `Qwen3ForCausalLM` native; the checkpoint declares no `matryoshka_dimensions`, so the engine
  refuses any `dimensions`. A slice would be **wrong** for this model: the smaller vectors are `W_k x`, not
  `x[:k]`. Any MRL adoption needs a plugin that loads the chain, not a cut.

### 3.6 topk-embed-v1 small / xsmall (`multi_vector`)

Pinned: small `e54485ebab921f2c18c4d092b3f4c40dcca26781` (shipped recipe pin, matches the API `sha`), xsmall
`210ebf2a25fb7128480f9b9c8f228e8d65c433a7` (API `sha`).

- **Kind: truncation after a projection.** Card: "Embeddings have 2048 dimensions. For smaller Matryoshka (MRL)
  embeddings, pass `config_kwargs={"output_dim": 256}` when loading the model" (small `README.md:146-147`); the
  evaluation tables use "Matryoshka (MRL) prefix dimensions; 2048 is the full embedding width"
  (`README.md:152`) with columns 64, 128, 256, 512, 1024, 2048. xsmall is the same text with 1024 and columns 64,
  128, 256, 512, 1024.
- **Set: card tables + code.** The card's tested set is the table's columns; the code accepts any integer:
  `TopkEmbedConfig.__post_init__` defaults `output_dim` to `dim` and refuses only outside `1..dim`
  (`modeling_topk_embed.py:23-29`), and `topk_embed_st.py:53-54` returns `output_dim or dim` as the embedding
  dimension. No `config.json` `matryoshka_dimensions`/`is_matryoshka`; no `config_sentence_transformers.json` cut
  key.
- **Full width: the `head` projection.** `config.json` `dim`/`output_dim` = 2048 (small) / 1024 (xsmall);
  `model.safetensors` `head.weight` `[2048, 2048]` / `[1024, 1024]`. Reference order:
  `vectors = self.head(hidden).float()`, `vectors = vectors[..., :dim]`, `F.normalize`
  (`modeling_topk_embed.py:76-82`) -- projection, slice, L2.
- **Engine path.** `/pooling` task `token_embed`; per-request `dimensions` is refused
  (`pooling/serving.py:62-66`). The plugin inherits vLLM's `ColQwen3_5Model` pooler (`custom_text_proj` ->
  tokwise head slice -> L2), so a serve-time `pooler_config.dimensions` reaches the same slice -- but only with
  the `is_matryoshka` gate. The shipped recipe declares `hf_overrides: {}` and no `pooler_config.dimensions`
  (`recipes/topk-embed-v1-small/recipe.yaml` serve block), so the slice is not enabled. The plugin's
  `TopkEmbedConfig` restates `output_dim` but the served class does not consume it (only the CPU equivalence
  helper `models/topk/pooling.py:token_embed_pool` takes it); `vllm`'s `ColQwen3_5Model` reads `config.dim` for
  the projection width. `embedding_size` for the range gate resolves to the hidden size (2048/1024) because the
  checkpoint has no ST `Dense`.

### 3.7 pplx-embed-v1 0.6B / 4B (`embed`)

Pinned: 0.6B `2c4d510dd4a732063c31a0f70193e35067b51fd8`, 4B `06456497a00540a582918fe8dcd3a5eabb207772` (API `sha`;
not in the tree).

- **Kind: truncation, card claim only.** The card table marks `MRL: Yes` for both rows (`README.md:33-36`), but the
  card never gives a set or a method (the only other `truncation` is the tokenizer example at `:117`), and the
  reference `modeling.py` is a plain `PPLXQwen3Model(Qwen3Model)` with bidirectional attention and no
  projection/truncation code (91 lines, grep: no matches). `config.json` has no `matryoshka_dimensions`; there is
  no `config_sentence_transformers.json`.
- **Full width: backbone + int8 quantizer.** `1_Pooling/config.json` `pooling_mode_mean_tokens: true`,
  `word_embedding_dimension` 1024/2560; `modules.json` = `Transformer`, `Pooling`,
  `st_quantize.FlexibleQuantizer`; the card notes the outputs are unnormalized int8 and must be compared by
  cosine (`README.md:25-26`). No `Dense`, no projection file.
- **Engine path: none at v0.31.0.** `PPLXQwen3Model`/`bidirectional_pplx_qwen3` are not in the tag's registry or
  config registry; the checkpoint's `auto_map` points at its remote `modeling.py`/`configuration.py`. A recipe
  would need a plugin (as the context/late siblings have) or a newer engine. G1.

### 3.8 pplx-embed-v2-context-9b-preview (`multi_vector`)

Pinned: recipe pin `b667039ee8b438a6350fbc91bbcecd86f9d363ba`; Hub `main` is now
`e2c06a779d45e8db53066735be1b98ffe3412ca3` (drift, G10). All facts here are at the recipe pin.

- **Kind: truncation of the projected output.** Card, "Matryoshka dimensions" section (`README.md:80-88`): "The
  model was trained with Matryoshka losses at **1024** and **2048** dimensions. To use 1024-dimensional
  embeddings, take the first 1024 values of each unnormalized embedding and normalize afterwards" (code sample
  `e[:, :1024]` then `e / norm`). Set: `{1024, 2048}`.
- **Full width: the `contextual_projection`.** `modeling_pplx_contextual.py:21-27`:
  `contextual_projection = nn.Linear(config.text_config.hidden_size, config.embedding_dim, bias=False)`, i.e.
  `[2048, 4096]` F32; the tensor ships in `contextual_head.safetensors`
  (`contextual_projection.weight` `[2048, 4096]` F32, header read at the pin) and the checkpoint's
  `config.json:114` declares `embedding_dim: 2048` (its `text_config.hidden_size` is 4096); applied as
  `F.linear(pooled, self.contextual_projection.weight.float())` (`:167`) after the span-mean pooling (`:164-166`).
  The plugin loads the same tensor into `PplxInt8Projection` and passes it as the tokwise head's projector
  (`rcp-ndcg-vllm/src/rcp_ndcg_vllm/models/pplx/model.py`, `pooler.py`).
- **Order:** span-mean -> projection -> slice -> L2, matching the card's "first 1024 of the unnormalised values,
  normalize afterwards" and vLLM's `TokenEmbeddingPoolerHead` (`tokwise/heads.py:87-96`).
- **Engine path.** `/pooling` per-request `dimensions` refused; the plugin's head is vLLM's own, so a serve-time
  `pooler_config.dimensions: 1024` reaches the slice -- gated by `is_matryoshka`, which the shipped recipe does
  not declare. The recipe note says so: "MRL {1024, 2048} is NOT declared ... adopting MRL is one field away when
  the wave wants the 1024-dim run" (`recipes/pplx-embed-v2-context-9b-preview/recipe.yaml:225-228`).

### 3.9 pplx-embed-v2-late 0.6B / 9B (`multi_vector`)

Pinned: 0.6B `8fc2de24534aa3610d85fa59c463313a5f096455` (shipped recipe pin, matches the API `sha`), 9B
`0f49a9977fe06b83377d598094c5c0204ce18ad9` (API `sha`).

- **Kind: none.** Neither card mentions Matryoshka/MRL/projection (grep: none); the embedding is a fixed
  128-dim per-token vector (`README.md`, `MultiVectorEncoder` usage; `sentence_bert_config.json`
  `module_output_name: token_embeddings`).
- **Full width: the ST `Dense`.** `modules.json` = `Transformer`, `1_Dense`, `2_MultiVectorMask`, `3_Normalize`;
  `1_Dense/config.json` `in_features` 1024/4096, `out_features` 128, `bias: false`,
  `activation_function: Identity`; `1_Dense/model.safetensors` one tensor `linear.weight` `[128, 1024]` /
  `[128, 4096]` F32. The plugin serves it as `hf_overrides: {embed_dim: 128}` into `custom_text_proj`.
- **Engine path.** `/pooling` per-request `dimensions` refused; a serve-time `pooler_config.dimensions` could
  slice the 128-wide output, but the card declares no set and the shipped recipe does not declare one.

### 3.10 harrier-oss-v1 270M / 0.6B / 27B (`embed`, in flight)

Pinned: 270M `31de22b673913c7d658c0f03f792d77c2dcf8ebd`, 0.6B `f9b9dc8d367d443f2479d27aa5d8d2850c0774ee`,
27B `0c0fc62f6d8af9e8604cb818c412301b103a0093` (API `sha`; not in the tree).

- **Kind: none.** No card mentions Matryoshka/MRL/projection (grep over each `README.md`); `config.json` declares
  no `matryoshka_dimensions`/`is_matryoshka`; `config_sentence_transformers.json` carries the three query prompts
  (`web_search_query`, `sts_query`, `bitext_query`) and cosine.
- **Full width: backbone.** `modules.json` = `Transformer`, `1_Pooling`, `2_Normalize`; `1_Pooling/config.json`
  `pooling_mode_lasttoken: true`, `word_embedding_dimension` 640/1024/5376. Architectures: `Gemma3TextModel` for
  270M/27B, `Qwen3Model` for 0.6B.
- **Engine path.** Both architectures are native at v0.31.0 (`registry.py:228`; suffix rule
  `config/model.py:2291`), so the `/v1/embeddings` per-request `dimensions` path exists with the `is_matryoshka`
  gate -- but the cards give no set, so a recipe must not enable it.

### 3.11 embeddinggemma-2 (`embed`, in flight)

Pinned: `914f7f89142e33e77833254d9c9b90c3cef7303b` (API `sha`; not in the tree).

- **Kind: truncation.** Card bullet: "**Matryoshka Representation Learning (MRL):** Native support for truncated
  embeddings across 128d, 256d, 512d, and 768d" (`README.md:44`); section "3. Matryoshka Dimension Truncation":
  "the 768-dimensional output vector can be shortened by keeping only its leading dimensions. The supported
  dimensions are 768, 512, 256, and 128" (`:168`), with the runtime rules "Re-normalize after truncating"
  (`:172`) and `truncate_dim=128, # or 512, 256` (`:175-179`). Set: `{768, 512, 256, 128}`.
- **Full width: a projection inside the model.** `config.json` `text_config.hidden_size` 512,
  `text_config.embedding_dim` 768; `model.safetensors` `language_model.embedding_projection.weight` `[768, 512]`
  BF16; `1_Pooling/config.json` `embedding_dimension: 768`, `pooling_mode: mean`; `modules.json` =
  `Transformer`, `Pooling`, `Normalize` -- **no ST `Dense`**, so the 768 comes from the model's own forward. The
  reference `EmbeddingGemma2TextModel` owns `embedding_projection = nn.Linear(hidden_size, embedding_dim,
  bias=False)` and applies it per token before pooling ("Projecting per token is equivalent to projecting after
  mean pooling").
- **Order:** per-token projection -> mean pool -> slice to `k` -> L2 normalise.
- **Engine path: none at v0.31.0.** No `embedding_gemma2` symbol anywhere in the tag (registry, config,
  transformers utils); the card's `config_sentence_transformers.json` declares `transformers 5.18.0.dev0` while
  v0.31.0 pins `transformers >= 5.10.4, < 5.18.0`. A plugin (like topk/pplx) or a newer engine is required;
  once served, the seqwise head would provide the per-request `dimensions` path with the `is_matryoshka` gate.

---

## 4. Gaps

**G1 -- a card that claims MRL without a set.** `pplx-embed-v1-0.6B`/`4B` mark `MRL: Yes` in the card table but
declare no set, no method and no code path (section 3.7); `PPLXQwen3Model` is not served at v0.31.0 anyway. The
card must not be read as a supported set.

**G2 -- a set the engine path cannot serve.** `zembed-1-embedding` is a projection kind (section 3.5): the engine's
`dimensions`/`pooler_config.dimensions` can only slice, and a slice of the 2560-d vector is not any of the card's
sizes; the chain `2560->1280->...->40` is not loaded by any served path or by the checkpoint's own code. The
engine must refuse it (and today does, because the checkpoint declares no `matryoshka_dimensions`), but nothing in
the product records *why*.

**G3 -- a recipe claim the shipped declaration does not back.** The Qwen3-Embedding recipe note says "MRL 32..1024
is available per request via the OpenAI dimensions field" while `serve.hf_overrides` is `{}` and the checkpoint
declares no `matryoshka_dimensions`/`is_matryoshka`; a `client.dimensions` set by a user would load and then fail
as an HTTP 400 on the first request. The same holds for Qwen3-VL-Embedding. Either the recipe declares the gate
(`hf_overrides: {is_matryoshka: true}`, optionally with the card's range as a discrete set) or the note must say
the engine path is gated off.

**G4 -- the card's range vs the engine's range.** Qwen3-Embedding (32..W) and Qwen3-VL-Embedding (64..W) give a
floor the engine does not enforce (gate 2 starts at 1) and a continuous range, not a discrete set. Declaring
`matryoshka_dimensions` with the card's range is possible but makes every integer a member; there is no
card-faithful way to express "any integer in 32..W" in vLLM's membership check except `is_matryoshka: true` alone
(which loses the floor).

**G5 -- card sets not declared to the engine.** `topk-embed-v1` (the tables' columns 64, 128, 256, 512, 1024,
2048 / 64, 128, 256, 512, 1024), `pplx-embed-v2-context-9b-preview` ({1024, 2048}), and `embeddinggemma-2`
({768, 512, 256, 128}) all have a card set but no `config.json` declaration; the engine's set-membership check only
runs when a recipe passes `matryoshka_dimensions` in `hf_overrides`/`pooler_config`. Without it the engine accepts
any integer in range.

**G6 -- the topk `output_dim` knob is not the served knob.** The card tells users to pass
`config_kwargs={"output_dim": k}`; the plugin restates `output_dim` in `TopkEmbedConfig`, but the served class
(`ColQwen3_5Model` subclass) reads `config.dim`, not `output_dim`, and the inherited tokwise slice is driven by
`pooling_param.dimensions`. So the card's knob and the served MRL path are different mechanisms; the recipe must
use `is_matryoshka` + `pooler_config.dimensions`, not `output_dim`.

**G7 -- projection kinds need a product field, not prose.** `zembed` (projection) and `embeddinggemma-2`
(truncation of a projected output) both have learned matrices in the full-width path, but the product has no field
that says a checkpoint's MRL kind. Today only recipe prose (`zembed` note 6) keeps a user from cutting a
projection model. The `mrl-core` lane's `matryoshka_kind` (or equivalent) is what turns that prose into a
validator.

**G8 -- product-side gaps the study recorded (G1-G9 there).** No `matryoshka_dims` field; `EmbeddingEndpoint` has
no `mrl_dim`; `ChangeMechanism` has no MRL member, so the cut is recorded nowhere but the identity; the fake engine
generates a fresh cut-width vector for `/embeddings` (it cannot catch a wrong cut/normalise order) and ignores
`/pooling` `dimensions` where vLLM refuses; the observation set has one bare `dimensions` probe and no MRL control;
the equivalence stages gate full width only. `mrl-core` owns these.

**G9 -- in-flight variants without a recipe.** `harrier-oss-v1` (none), `jina-v5-text-nano` (truncation, native
encoder dispatch), `embeddinggemma-2` (truncation, no v0.31.0 class), `pplx-embed-v1` (card-only claim, no class),
and the new sizes of Qwen3-Embedding / Qwen3-VL-Embedding / Octen / topk / pplx-late have no recipe yet; the
family lane (`rfam` and the recipe lanes) owns them. This spec is the card-side input to that work.

**G10 -- pin drift on `pplx-embed-v2-context-9b-preview`.** The shipped recipe pins
`b667039ee8b438a6350fbc91bbcecd86f9d363ba`; Hub `main` is `e2c06a779d45e8db53066735be1b98ffe3412ca3` on
2026-10-09. Every fact in section 3.8 is at the recipe pin; the Matryoshka section and `embedding_dim: 2048` were
re-checked at `main` and are unchanged, but the recipe's `revision:` (and any MRL adoption) must be re-read there
before the wave.

**G11 -- the card's tested sets are not the code's bounds.** `topk` accepts any `1..dim`, `zembed`'s projections
accept exactly six sizes, `embeddinggemma-2`'s card names four. A set declared to the engine should be the card's
tested set, not the code's bounds; the spec records both so the recipe lane can choose.

---

## 5. Open questions for the owner / `mrl-core`

1. For the range cards (Qwen3-Embedding, Qwen3-VL-Embedding): declare `is_matryoshka: true` with no set (engine
   accepts 1..W) or mirror the card's range as a discrete `matryoshka_dimensions` (loses the floor but pins the
   tested grid)? The no-silent-default rule argues for a declared set; the cards give none.
2. For `zembed`: refuse MRL in the recipe validator (projection kind, no loaded chain) or build the plugin that
   loads `projections.safetensors`? The latter is a model-specific plugin; the former is one field.
3. For `pplx-embed-v1`: treat the card's `MRL: Yes` as unbacked (kind `none`) until a set and a served path
   exist, or add a plugin and adopt prefix truncation?
4. Does the engine-side path become the one home for MRL cuts where the card's set exists (dense models), with
   `mrl_dim` reserved for `/pooling` (whose route refuses `dimensions`) and for the ex-post sweep? The spec's
   table shows both routes are needed; the product decision is which one the identity/records key on.
5. For `pplx-embed-v2-context-9b-preview` and `topk`: adopt MRL now (one field: `is_matryoshka` +
   `pooler_config.dimensions`, or `mrl_dim`) or leave it to the wave? The card's set is explicit in both cases.
