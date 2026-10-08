# rcp-ndcg-vllm-pplx

A pure-Python vLLM plugin that serves the two perplexity-ai pplx embedding checkpoints on the
stock `vllm/vllm-openai:v0.31.0` image:

- [perplexity-ai/pplx-embed-v2-context-9b-preview](https://huggingface.co/perplexity-ai/pplx-embed-v2-context-9b-preview)
  (revision `b667039ee8b438a6350fbc91bbcecd86f9d363ba`): one embedding per document chunk. Stock vLLM cannot serve
  it (the architecture `PplxContextualModel` is not in the model registry, the weights are saved under `language_model.*`
  where vLLM's Qwen3.5 classes expect `language_model.model.*`, the `contextual_projection.weight` head has no module to
  land in, and no pooler expresses the span-mean-then-int8-tanh head), so the plugin registers one out-of-tree model
  class (`PplxContextualModel`) through the `vllm.general_plugins` entry point, modelled on vLLM's own
  `ColQwen3_5Model` / `ColBERTLfm2Model` pooling precedents.
- [perplexity-ai/pplx-embed-v2-late-0.6b](https://huggingface.co/perplexity-ai/pplx-embed-v2-late-0.6b)
  (revision `8fc2de24534aa3610d85fa59c463313a5f096455`): one L2-normalized 128-dim vector per kept token, MaxSim
  scored client-side. Stock vLLM cannot serve it either (the checkpoint's `architectures[0]` `Qwen3_5Model` is not in
  the registry, and the trained Dense head ships as a separate `1_Dense/model.safetensors` the stock weight discovery
  never reads), so the same wheel registers a second out-of-tree model class (`Qwen3_5Model` ->
  `rcp_vllm_pplx.late.PplxLateMultiVectorModel`, a `ColQwen3_5Model` subclass that loads the head itself and marks the
  zero-initialised bias loaded). That checkpoint needs no config registration: its `model_type qwen3_5` is native to
  the engine's transformers line, parsed through vLLM's own config registry.

After `pip install --no-deps <wheel>` the unmodified image loads either checkpoint as a pooling model instead of
failing on weight loading.

## Install and serve (the contextual chunk model)

```bash
pip install --no-deps rcp_ndcg_vllm_pplx-0.0.1-py3-none-any.whl
vllm serve perplexity-ai/pplx-embed-v2-context-9b-preview \
  --revision b667039ee8b438a6350fbc91bbcecd86f9d363ba \
  --runner pooling \
  --dtype bfloat16 \
  --pooler-config '{"task": "token_embed"}'
```

No `--trust-remote-code`: the plugin registers the checkpoint's configuration class with transformers'
`AutoConfig` (see `src/rcp_vllm_pplx/hf_config.py`), so the engine parses `config.json` locally and never
executes the checkpoint's remote config code; `--dtype bfloat16` is the served dtype (vLLM v0.31.0's GDN
kernels refuse float32 — see the recipe's notes for the two kernel floors and what the GPU wave measures).

## Install and serve (the late-interaction model)

```bash
pip install --no-deps rcp_ndcg_vllm_pplx-0.0.1-py3-none-any.whl
vllm serve perplexity-ai/pplx-embed-v2-late-0.6b \
  --revision 8fc2de24534aa3610d85fa59c463313a5f096455 \
  --runner pooling \
  --dtype bfloat16 \
  --hf-overrides '{"embed_dim": 128}' \
  --mm-processor-kwargs '{"images_kwargs": {"min_pixels": 3136, "max_pixels": 1800964}}'
```

No `--trust-remote-code` either: `model_type qwen3_5` is native, and vLLM's config registry parses
`config.json` (vllm/transformers_utils/config.py registers its own `Qwen3_5Config` with transformers'
`AutoConfig`). The `--hf-overrides` names the head's width (`1_Dense/config.json` out_features), and the
pixel pin carries the shipped processor's effective budget — the two serve facts the
`pplx-embed-v2-late-0.6b` recipe declares.

The plugin is picked up automatically: every vLLM process loads `vllm.general_plugins` entry points
(`VLLM_PLUGINS=rcp_vllm_pplx` to pin it by name). At import it refuses any vLLM outside the range it was
validated against (`>=0.31,<0.32`) with a clear message.

## What the client must send (the contract)

The engine must receive **token ids**, not text — a server-side tokenization of the document text diverges from
the reference implementation (the `[D] ` prefix tokenizes as one special id there and as the two literal tokens
`[D`, `]` in the reference). `POST /pooling` with `input: [[ids...]]` carries them. Per role:

- **Query**: `[248077] + tokenized query text` (248077 is the `[Q] ` prefix as one added token). The pooler
  returns one vector: the mean over **all** tokens, the `[Q] ` token included — exactly the reference's
  masked-mean pool.
- **Document**: `tokenize("[D] " + chunk0 + "<|chunk_sep|>" + chunk1 + …, split_special_tokens=True)` — the
  prefix renders as the two literal tokens `[62724, 60]` and each boundary marker as the single id `248079`.
  The pooler segments on the boundary markers and returns **one span-mean vector per chunk**, prefix and
  markers excluded. A one-chunk document therefore has no marker and is one span; a query is identified by its
  leading `248077` — that leading id is the role disambiguation rule. An input whose first id is neither role
  prefix is refused with a message naming the contract (the all-zero token grid of vLLM's pooler warm-up is the
  one exception: it is pooled as a single span so engine warm-up succeeds; its output is discarded).
- An **empty chunk** (two consecutive markers, or a trailing marker) yields a zero vector for that chunk — the
  reference's own behavior for a chunk with no tokens.
- A chunk's text must not contain the literal `<|chunk_sep|>`: the marker is an added token (not a special
  token), so a client rendering that text produces its id `248079` inside the chunk, and the pooler — which sees
  only ids — cannot distinguish it from a boundary marker (the chunk would split there, the token would be
  excluded, and one extra vector would come back). The reference implementation, which pools by character spans,
  keeps such a token inside its chunk; the token-id wire contract cannot reproduce that, so the recipe's chunker
  must not emit the marker string as chunk content.

The head is fixed by the model: fp32 cast, `Linear(4096→2048, bias=False)` in fp32, then
`round(tanh(x)·127).clamp(-128, 127)` — the card's "unnormalized int8-quantized embeddings". The optional L2
normalisation is the request's `use_activation` (on by default, as the card's `normalize_embeddings=True`);
`use_activation: false` returns the raw int8-valued vectors. Matryoshka (1024 of 2048) is a client-side cut of
the unnormalized values, per the card.

The role prefixes and the boundary marker are constants of the pinned revision's tokenizer
(`query_prefix` / `document_prefix` / `boundary_marker` in the checkpoint config); the pooler requires the
ids to begin with exactly `[62724, 60]` and refuses anything else loudly. If the Hub revision changes
its tokenizer, the ids change and this plugin must be re-pinned — by design, never silently.

## What the plugin registers

| Registered | What | Modelled on |
|---|---|---|
| `PplxContextualModel` → `PplxContextualForPooling` | the contextual model class (lazy `"module:Class"` string) | `ColQwen3_5Model` (vLLM `models/colqwen3_5.py`) |
| `AutoConfig[pplx_contextual_qwen3_5]` → `hf_config.PplxContextualConfig` | the contextual checkpoint's transformers config class, restated locally (the explicit-local-code path: no `--trust-remote-code`, no remote code execution) | `rcp_ndcg_vllm_topk.config.TopkEmbedConfig` (the topk plugin's registration) |
| `MODELS_CONFIG_MAP["PplxContextualModel"]` → `config.PplxModelConfigHandler` | forces `is_causal = False` on the HF config and the text config (the model's bidirectional contract; also the path `ModelConfig.attn_type` reads) | `ColQwen3_5Config` (vLLM `models/config.py`) |
| `Qwen3_5Model` → `late.PplxLateMultiVectorModel` | the late-interaction sibling's architecture (a `ColQwen3_5Model` subclass): loads the checkpoint's separate `1_Dense/model.safetensors` head into `custom_text_proj`, shape-checked, and marks the zero-initialised bias loaded | `rcp_ndcg_vllm_topk.model.TopkEmbedModel` (the topk plugin's subclass) |

The contextual class deliberately registers **no multimodal processor** (the ColQwen3.5 precedent does; copying it
would make the server silently accept images the embedding path never uses) and constructs no vision tower and no
`lm_head` (the checkpoint's 333 `visual.*` tensors are skipped, and the 4 GB fp32 unused vocabulary head is replaced
by a `StageMissingLayer`, following vLLM's own `_create_pooling_model_cls`). The late-interaction class is the
opposite case on the first point: it **inherits** `ColQwen3_5Model`'s processor registration (its checkpoint's own
`Qwen3VLProcessor` drives image documents through the checkpoint's chat template), and the tied `lm_head` loads as
the skipped alias of `embed_tokens` (`tie_word_embeddings: true`).

## What the late-interaction class changes

`PplxLateMultiVectorModel` (in `src/rcp_vllm_pplx/late.py`, with the checkpoint facts as data in
`late_data.py`) changes exactly one behaviour against the inherited `ColQwen3_5Model` — weight loading:

- the checkpoint's backbone (`language_model.*` top-level, the ColPali convention) loads through the inherited
  mapper; `visual.*` matches as-is; the absent `lm_head` is the tied alias of `embed_tokens`
  (`tie_word_embeddings: true`); no `mtp.*` weights exist (the census is pinned in `late_data.py` and verified by
  the tests).
- the trained Dense head ships as a SEPARATE sentence-transformers module file, `1_Dense/model.safetensors` (one
  tensor, `linear.weight` [128, 1024], fp32). The stock loader's discovery globs `*.safetensors` in the snapshot
  root non-recursively (default_loader.py:226-233 at v0.31.0), so the head never enters the weight iterator: the
  class fetches that file itself (local directory, else the same Hub snapshot the engine already downloaded),
  shape-checks it against the projector module, and renames it onto `custom_text_proj.weight`, the name the
  inherited loader's projection branch intercepts.
- the checkpoint's head is bias-less while the inherited constructor builds `custom_text_proj` with a
  zero-initialised bias (score-equivalent); the returned loaded set is annotated under both qualnames so vLLM's
  load tracker accepts it, exactly as the in-tree projection loader marks a shipped bias.

## Verification story

- `tests/` proves, on CPU: the pure-torch pooling core reproduces the reference math on a tiny random
  configuration of the same architecture (weights shared, fp32 tolerance), the version guard, the entry-point
  registration (against a stub registry where vLLM is absent — now both architectures), the late-interaction
  class's weight contract as data (the checkpoint census, the head file and tensor names, the zero-bias marking),
  and the `--no-deps` freeze behaviour (simulated). Tests that need `vllm` importable (the wired pooler against a
  real `PoolingMetadata`, the tiny-config equivalence through the real model class) skip themselves with a clear
  reason where vLLM cannot import on CPU, and run on the GPU wave, which has the engine image's vLLM. The
  transformers config restatement is pinned field for field against the remote class's constants where
  transformers imports.
- `scripts/check_no_deps_freeze.sh` is the check the GPU wave runs: in a venv over the engine environment,
  `pip freeze` must change by exactly the one wheel.
- On real weights (GPU wave): engine loads, `/pooling` returns `(n_chunks, 2048)` per document and one vector
  per query, and served vectors match the reference implementation within the recipe's tolerance gate — served
  at `bfloat16` (vLLM v0.31.0's GDN kernels refuse float32 on every prefill path; see the recipe's notes),
  against the fp32 reference, so the tolerance gate covers the cast as well as the kernels.
