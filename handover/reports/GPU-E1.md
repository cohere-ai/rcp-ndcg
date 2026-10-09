# Report GPU-E1: the first GPU wave on the M3 tip (18 retrieval recipes)

**Status:** DONE as evidence; no recipe status changed (statuses move only in E2, from evidence). Run on 8 x B200
pods at `dev-high`, stock image `vllm/vllm-openai:v0.31.0`, the reference environment over the image's torch 2.13.0
and transformers 5.17.0. Each recipe: serve, smoke, stage 1 (renders vs the reference and `/tokenize`), stage 2
(scores and vectors vs the reference), the media gate, the recorder. Diagnosis runs used a throwaway branch (serve
flags and one reference guard changed there, never merged).

## Outcome per recipe
| Recipe | Result | Cause (class) |
|---|---|---|
| zerank-1, zerank-1-small, zerank-2 | verified | |
| jina-embeddings-v5-text-small | verified with `--max-num-batched-tokens 32768` | engine hang at exactly max_model_len tokens without it (vLLM bug, fixed upstream) |
| zembed-1-embedding | hang (as jina); with the flag every vector differs (cosine 0.08-0.4) | pooled vector is not the card's (recipe) |
| qwen3-reranker-0.6b, -8b | reference fails | the reference requires flash-attn (reference) |
| qwen3-reranker-4b, ctxl-rerank-v2 x3 | stage 2 out of bound | bf16-quantised reference scores (reference precision) |
| ctxl-rerank-v2-instruct-multilingual-1b | stage 1 | engine `/tokenize` +2..10 tokens vs the client (recipe) |
| jina-reranker-v3 | stage 2 | empty-document policy differs from the reference (recipe) |
| qwen3-embedding-0.6b | stage 2 min cosine 0.99885 (bound 0.999) | bf16 near-miss (precision) |
| qwen3-vl-embedding-2b | stage 2 0.99897; media: video token count 98 vs 512, 458 vs 1952 | precision; video policy unpinned (recipe) |
| qwen3-vl-reranker-2b | reference crash (`list.to`); guarded: 93 % of documents within 0.02 (bound 99 %), tau 1.0 | reference bug; precision |
| octen-embedding-8b | stage 2 min cosine 0.9936 (tiny query; special tokens; long base64) | precision (bf16 engine vs reference) |
| topk-embed-v1-small | reference crash | the checkpoint's remote code needs an older transformers (`layer_type`) |
| pplx-embed-v2-context-9b-preview | engine dies at startup | 262144-token warmup overflows 32-bit offsets; the plugin rejects the warmup input (recipe/plugin) |

Images: every image row of the three media recipes matched the engine's media-token count; the query-image and
mixed-batch rows passed. Video is the open media item.

## Root causes established
- **Engine hang (jina, zembed).** Any pooling prompt of exactly `max_model_len` tokens that is chunked never completes
  (32768 tokens hang, 32767 return in 0.5 s; content-independent). vLLM v0.31.0's scheduler caps a running request
  at `max_model_len - num_computed_tokens - num_sampled_tokens_per_step` (`vllm/v1/core/sched/scheduler.py:687-692`)
  and reserves one sampled-token slot for pooling runners too (`scheduler.py:146-148`), so the last prompt token is
  never scheduled. The harness's over-cap probe is cut to exactly the budget, so one request per run stuck; the
  product client's 600 s timeout and retry made each one cost 10 minutes. Fixed upstream by vllm-project/vllm#48039
  (commit e6fc81bc78, 2026-10-07; not in v0.31.1rc0); the recipe-fix lane backports it as an opt-in, self-retiring
  patch module.
- **pplx-embed-v2-context-9b-preview.** Pooling runs without chunked prefill, so vLLM's warmup is one sequence of
  `max_model_len` = 262144 tokens; q is 262144 x 16 x 256 x 2 B = 2^31 bytes and the kernels fault
  (`cudaErrorIllegalAddress` in FlashAttention 4's encoder path; with the Triton backend it surfaces later, in
  inductor). At 32768 the warmup passes and the plugin then raises on the engine's dummy input (its role-prefix
  check fires on token ids `[0, 1]`).
- **Precision class.** The engine serves bf16; the references run at other precisions or quantise scores to bf16.
  The bound decisions (like-for-like dtype, or declared bf16 bounds) follow decision 9.
- **Every reference ran on CPU.** The harness's `equivalence.run` defaults to `device="cpu"` and the wave runner
  passes none. The Qwen3.5-based references cannot run there at all (pplx-embed-v2-late-0.6b, added after the
  wave: its engine served and passed smoke; its reference failed in flash-linear-attention's Triton kernel on a CPU
  tensor), and every precision-class verdict above compared a CPU reference with the bf16 GPU engine. E2 re-judges
  them with references on their own GPU.

## Harness findings (lane harness-fix)
The runner serialises steps across recipes (one stuck request blocked two other recipes for 40 minutes); results
upload only when the pod ends; the pod log is silent; one engine crash killed the whole pod; the reference
environment ignored PEP 508 markers; `rc_build.sh` built the unpublished workspace member.

## Next
The recipe-fix lane takes the recipe, reference and plugin causes above (per family, after decision 34's family
layout); the harness-fix lane takes the harness findings; E2 re-runs every recipe and variant from the RC.
