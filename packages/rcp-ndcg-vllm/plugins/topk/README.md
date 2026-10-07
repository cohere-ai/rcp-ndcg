# rcp-ndcg-vllm-topk

A [vLLM](https://docs.vllm.ai) general plugin that serves
`topk-io/topk-embed-v1-small` — a multimodal late-interaction (multi-vector)
retriever on a Qwen3.5-2B backbone — on the **stock**
`vllm/vllm-openai:v0.31.0` image. One pure-Python wheel; no compiled
extensions, no downloads, no changes to the image beyond
`pip install --no-deps`.

The model is a late-interaction (multi-vector) embedder: one L2-normalised
2048-dim vector per kept prompt token, scored client-side by fp32 MaxSim. It
is served through `/pooling` with `task: token_embed`.

## Install (engine image)

```bash
pip install --no-deps rcp_ndcg_vllm_topk-0.0.1-py3-none-any.whl
```

`pip freeze` before and after must differ by exactly this one distribution
(`tests/check_no_deps_install.py` is the reusable check; the GPU wave wraps it
around the real install). The wheel declares no dependencies: it imports only
torch, vLLM, transformers and the stdlib, all of which the image ships.

## Serve

```bash
vllm serve topk-io/topk-embed-v1-small \
  --runner pooling \
  --max-model-len 8448 \
  --dtype bfloat16 \
  --limit-mm-per-prompt '{"image": 1}' \
  --mm-processor-kwargs '{"min_pixels": 65536, "max_pixels": 1310720}'
```

No `--trust-remote-code`: the plugin registers the checkpoint's configuration
class with transformers' `AutoConfig`, so the engine resolves `config.json`
locally and never executes the checkpoint's remote code (whose module imports
`flash-linear-attention`, which the image does not ship — with
`--trust-remote-code` on the stock image, the config load dies with
`ModuleNotFoundError: No module named 'fla'` before the model loads; without
the flag it dies with the "contains custom code" error; with the plugin,
transformers' explicit-local-code path wins under either flag value). The
registered class restates the remote one field for line
(`modeling_topk_embed.py:12-29` at the pinned revision), so the config object
the engine consumes is identical for every field vLLM reads.

`--max-model-len 8448` is the 8192-token document cap plus headroom (the
model's default 262144 would over-reserve KV). The flat `--mm-processor-kwargs`
shape is the one the v0.31.0 image's own ColQwen3.5 example documents and
reproduces the model card's 1280-patch image budget
(`max_pixels = 1280*(16*2)^2`); the GPU wave confirms the cap on a live
server. Queries are text-only; text prompts are raw `"Query: "` /
`"Document: "` strings (no chat template); image documents go through the
repo's chat template as an image-only user message. The client
keeps every query vector, drops document vectors at `scoring_skip_ids`
positions (41 ids), and applies the 1024/8192-token caps with right
truncation — none of that lives in the engine (the recipe carries it).

## What the plugin is

Two registrations, one model class:

- **The configuration class.** `rcp_ndcg_vllm_topk.config.TopkEmbedConfig`
  restates the checkpoint's remote `TopkEmbedConfig` (a subclass of
  transformers' `Qwen3_5Config`, plus the retrieval knobs) and
  `rcp_ndcg_vllm_topk.plugin.register` adds it to transformers' `AutoConfig`
  with `model_type: topk_embed`. Without it, the checkpoint's own `auto_map`
  makes the engine execute remote code whose module imports
  `flash-linear-attention` — absent from the image — and the config load
  fails before any weight loads. With it, the remote code is never executed
  (no `fla` requirement, no runtime code download) and the parse is the
  same: the class is a line-for-line restatement and transformers' own
  Qwen3.5 machinery builds the sub-configs in both cases.
- **The model class.** `rcp_ndcg_vllm_topk.model.TopkEmbedModel` subclasses
  vLLM's native `ColQwen3_5Model` (the stock late-interaction model on the
  same Qwen3.5 backbone) and overrides two things: the checkpoint-name
  mapping (the Qwen3-VL convention plus `head.` → `custom_text_proj.`,
  replacing the stock ColPali-convention mapper) and `load_weights`, which
  marks the projection's zero bias as loaded after the stock loader (the
  checkpoint's head is bias-less, and vLLM v0.31.0's load tracker refuses a
  parameter the checkpoint never supplied). Every forward-affecting behaviour
  is inherited:

- **Non-causal full attention.** The checkpoint's `text_config` carries
  `is_causal: false`, which the stock `Qwen3NextAttention` constructor reads
  (vLLM v0.31.0, `qwen3_next.py:335-343`) to build bidirectional
  ENCODER_ONLY attention on the six full-attention layers — the same state
  the in-tree ColQwen3.5 models reach via a config handler. The plugin adds
  nothing.
- **Projection and pooling.** `ColQwen3_5Model` builds `custom_text_proj`
  (zero-initialised bias, score-equivalent to the checkpoint's bias-less
  head) and hands it to `pooler_for_token_embed`: per-token vectors, float32
  head arithmetic (the pooling default), optional `dimensions` MRL slice,
  L2 normalisation. The reference computes the head matmul on bf16 operands
  and casts the result; the served fp32 head differs only in that rounding,
  which the GPU equivalence measures (rcp-ndcg transfers float16 on the
  client, which dominates it).
- **Multimodal.** The inherited registration drives the checkpoint's own
  `Qwen3VLProcessor` for image documents.

The `vllm.general_plugins` entry point (`rcp_ndcg_vllm_topk.plugin:register`)
registers the configuration class (transformers `AutoConfig`) and the model
architecture (`ModelRegistry`) in every engine process at startup; the model
class is imported lazily by the registry.

## Version guard

The plugin refuses any vLLM outside `>=0.31,<0.32` with a message naming the
tested range — at entry-point load (engine startup) and again at the model
module import. vLLM logs both paths with the traceback (entry-point load
failures are swallowed at `logger.exception` level and the serve then fails
on the unregistered architecture; the guard's message is written to be
readable in that log).

## Tests

```bash
pytest tests/
```

Three environments, with skips stated by name:

- **CPU dev box** (torch, no vLLM, no transformers): everything except the
  vLLM- and transformers-gated checks — the entry-point declaration, the
  version guard, the weight mapping against the checkpoint's full 618-name
  census, the tiny-config equivalence of the pooling chain against the
  reference chain (shared random weights, float32 tolerance, the bf16-head
  rounding bound), and the simulated `--no-deps` freeze check (builds the
  wheel with uv and installs it into a fresh offline venv; forged wheels with
  a declared dependency, a compiled artifact or a platform tag must fail it).
- **Engine image** (vLLM and transformers importable): additionally the
  registry effects of the entry point (architecture and config class), the
  config-class parse of a same-architecture config with the checkpoint's
  `auto_map` (remote code never executes), a cross-check of the served
  class's real `WeightsMapper` against the restated table, and the
  served-class shape checks.
- **GPU wave** (real weights): the served-vs-reference equivalence against
  the model card's implementation — outside this suite. It must check: the
  engine loads with the plugin (T0 `/v1/models` shows the model), `/pooling`
  `token_embed` shapes (one vector per prompt token, 2048 dims, unit-norm),
  the image-token budget (a 4-Mpx page stays within 1280 patch tokens), and
  the per-token cosine + MaxSim equivalence against the card's
  `MultiVectorEncoder` reference on real weights (the non-causal layers, the
  probe-image count, the fp32-vs-bf16 head rounding).

Build: `uv build packages/rcp-ndcg-vllm/plugins/topk` produces one
`py3-none-any` wheel; `twine check` passes.
