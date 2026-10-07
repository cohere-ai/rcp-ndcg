# Report 03: the six recipe families (integrated at M2)

**Status:** DONE. The families ran as three lanes (qwen3-rerank + zerank; ctxl + dense; vl + late with both plugins),
each with two independent verifiers and a fix round; all merged into the recipe line, which landed on `rfc-0001` at M2.
Every recipe `status` stays `unverified` (GPU validation is the owner's).

## Decision 9 (the main finding)
Six references were porting the client's cut (query share, settle-once, prefix cut) so that over-cap rows matched:
qwen3-reranker-8b and zerank-1-small (recipe-sweep `f1d1e0a`), zerank-1 and zerank-2 (`ecbf3f8`, `d79a1a8`), all three
ctxl references (`68644be`), qwen3-embedding-0.6b, qwen3-vl-reranker-2b (`ae300f4`, not format-only) and the
qwen3-vl-embedding reference. All now render the paper's or the model card's own cut (character offsets, never decode),
in the required span format, with the over-cap deviation declared: `over_cap_cut_differs` where the anchors survive
(qwen3-reranker family, qwen3-embedding, octen, jina-v5, jina-reranker-v3, the qwen3-vl recipes, topk),
`anchor_drop_over_cap` where they do not (zerank family, ctxl family, zembed). The topk reference now reproduces the
card's 1024/8192 cut, including decomposed-Unicode boundaries.

## Per family (highlights)
- qwen3-rerank: one over-cap policy; 4b's mislabelled deviation fixed; `empty_query: send`.
- zerank: one declared `normalize: [strip]` rule; `serve.convert` removed (the scorer is declared through
  `hf_overrides`).
- ctxl: `instruction: none` for all three sizes; the 6b template and reference no longer append an instruction.
- dense: per-shape `query_max_tokens` where the reference caps queries (jina-reranker-v3 512); jina-v5 `anchor:
  last_content`; zembed's reference pinned to sentence-transformers `>=5.3,<5.4` (5.4 drops the suffix) with
  `transformers>=4.51`.
- vl: the R20 pixel pin is nested `mm_processor_kwargs.images_kwargs` (decided from the vLLM v0.31.0 source and
  measured; flat keys also resize every video clip); the embedding recipe gained an explicit `query` shape following
  the card; the client media policy carries the serve numbers.
- late: topk plugin (`load_weights` override marking the zero projection bias initialised; plugin distribution name in
  `serve.plugin`; no remote code); pplx plugin-registered config class (no remote code), served in bfloat16 (vLLM's
  kernels refuse float32; deviation documented).
- Deep review per recipe against the model card at the pinned revision and the vLLM v0.31.0 source (load path, pooler,
  template, overrides), with file:line evidence; internal labels removed from shipped files (two tests refuse them).

## Stage 1 on real tokenizers (network-gated, at M2)
All 18 recipe test files pass (pplx's token_ids xfails resolved after the harness fix; the strict xfails for the
`last_content` audit and `empty_doc` order were removed when lane H fixed them).
