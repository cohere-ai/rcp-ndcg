# rcp-ndcg-vllm-pplx

A pure-Python vLLM plugin that serves
[perplexity-ai/pplx-embed-v2-context-9b-preview](https://huggingface.co/perplexity-ai/pplx-embed-v2-context-9b-preview)
(revision `b667039ee8b438a6350fbc91bbcecd86f9d363ba`) on the stock `vllm/vllm-openai:v0.31.0` image: after
`pip install --no-deps <wheel>` the unmodified image loads the checkpoint as a pooling model and returns **one
embedding per document chunk** (the model's whole point) instead of failing on weight loading.

Stock vLLM v0.31.0 cannot serve this checkpoint: the architecture `PplxContextualModel` is not in the model
registry, the weights are saved under `language_model.*` where vLLM's Qwen3.5 classes expect
`language_model.model.*`, the `contextual_projection.weight` head has no module to land in, and no pooler
expresses the span-mean-then-int8-tanh head. This plugin registers one out-of-tree model class
(`PplxContextualModel`) through the `vllm.general_plugins` entry point, modelled on vLLM's own
`ColQwen3_5Model` / `ColBERTLfm2Model` pooling precedents.

## Install and serve

```bash
pip install --no-deps rcp_ndcg_vllm_pplx-0.0.1-py3-none-any.whl
vllm serve perplexity-ai/pplx-embed-v2-context-9b-preview \
  --revision b667039ee8b438a6350fbc91bbcecd86f9d363ba \
  --runner pooling \
  --pooler-config '{"task": "token_embed"}' \
  --trust-remote-code
```

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

The head is fixed by the model: fp32 cast, `Linear(4096→2048, bias=False)` in fp32, then
`round(tanh(x)·127).clamp(-128, 127)` — the card's "unnormalized int8-quantized embeddings". The optional L2
normalisation is the request's `use_activation` (on by default, as the card's `normalize_embeddings=True`);
`use_activation: false` returns the raw int8-valued vectors. Matryoshka (1024 of 2048) is a client-side cut of
the unnormalized values, per the card.

The role prefixes and the boundary marker are constants of the pinned revision's tokenizer
(`query_prefix` / `document_prefix` / `boundary_marker` in the checkpoint config); the pooler asserts the
document prefix renders as exactly `[62724, 60]` and refuses anything else loudly. If the Hub revision changes
its tokenizer, the ids change and this plugin must be re-pinned — by design, never silently.

## What the plugin registers

| Registered | What | Modelled on |
|---|---|---|
| `PplxContextualModel` → `PplxContextualForPooling` | the model class (lazy `"module:Class"` string) | `ColQwen3_5Model` (vLLM `models/colqwen3_5.py`) |
| `MODELS_CONFIG_MAP["PplxContextualModel"]` → `PplxContextualConfig` | forces `is_causal = False` on the HF config and the text config (the model's bidirectional contract; also the path `ModelConfig.attn_type` reads) | `ColQwen3_5Config` (vLLM `models/config.py`) |

It deliberately registers **no multimodal processor** (the ColQwen3.5 precedent does; copying it would make the
server silently accept images the embedding path never uses) and constructs no vision tower and no `lm_head`
(the checkpoint's 333 `visual.*` tensors are skipped, and the 4 GB fp32 unused vocabulary head is replaced by a
`StageMissingLayer`, following vLLM's own `_create_pooling_model_cls`).

## Verification story

- `tests/` proves, on CPU: the pure-torch pooling core reproduces the reference math on a tiny random
  configuration of the same architecture (weights shared, fp32 tolerance), the version guard, the entry-point
  registration (against a stub registry where vLLM is absent), and the `--no-deps` freeze behaviour
  (simulated). Tests that need `vllm` importable (the wired pooler against a real `PoolingMetadata`, the
  tiny-config equivalence through the real model class) skip themselves with a clear reason where vLLM cannot
  import on CPU, and run on the GPU wave, which has the engine image's vLLM.
- `scripts/check_no_deps_freeze.sh` is the check the GPU wave runs: in a venv over the engine environment,
  `pip freeze` must change by exactly the one wheel.
- On real weights (GPU wave): engine loads, `/pooling` returns `(n_chunks, 2048)` per document and one vector
  per query, and served vectors match the reference implementation within the recipe's tolerance gate — the
  dtype rung is `float32` first (the checkpoint is all-fp32; 33.6 GB), then `bfloat16` against a measured
  tolerance.
