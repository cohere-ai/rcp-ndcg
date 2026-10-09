# Recipe families (owner decision 34, 2026-10-08): one family, many sizes, every size its own tested recipe id

## Decision (owner)
"Where models have shared abstractions we should have a generalized recipe ... we would want to avoid code duplication
(e.g. also the various ctxl sizes) ... But it's good to test all individually." Mechanism chosen by the owner: one
family directory per model family with `family.yaml` (the shared client/serve/reference blocks, ONE shared
`reference.py`, one template) plus a `variants` table holding only per-size facts; every variant resolves to a full
recipe id that is served, contract-tested, stage-1-tested and GPU-validated on its own. Scope (owner): every public
size on the Hub of every family we carry, plus pplx-embed-v1.

## Today (M3)
18 standalone directories; sizes of one family duplicate everything: e.g. qwen3-reranker-0.6b vs -4b `reference.py`
421 vs 288 lines with ~600 differing lines, `recipe.yaml` ~200 differing lines; the same for ctxl and zerank.

## Target shape (after the layout move: recipes are package data of rcp-ndcg-vllm)
```
recipes/<family>/
  family.yaml            # schema_version; family id; shared role/input/client/serve/engine/reference/gates/notes
  reference.py           # ONE reference implementation, parameterised by the variant (model id, revision, dims...)
  template.jinja         # shared, when the model needs one
  requirements-reference.txt
  variants:              # inside family.yaml
    - id: qwen3-reranker-0.6b
      model: Qwen/Qwen3-Reranker-0.6B
      revision: <40-hex>
      overrides: {resources: {gpus: 1}, engine: {max_model_len: ...}, client: {...}}   # only per-size facts
```
- `load_recipe("<variant id>")` and `iter_recipes()` expand a family into full `Recipe` objects (one per variant) by a
  deep merge of the family blocks and the variant's `overrides`, then validate the result with today's `Recipe`
  model (the merged recipe is exactly what a standalone recipe would have been: the JSON Schema of a resolved recipe
  is unchanged; the family file gets its own exported schema). Overrides may only touch declared per-size fields
  (model, revision, resources, engine limits, dims, max lengths, per-size notes, status); a shape or content field
  that differs between sizes is a modelling error the loader refuses unless the family declares it per-variant.
- `serve_argv`, `client_config`, `rcp-ndcg-vllm serve <variant id>`, `recipe:<variant id>` in rcp-ndcg, the
  fingerprint, the harness, the wave lists and the catalog all work on variant ids, never on family ids.
- The reference receives the variant through its existing CLI contract (`--recipe` / environment), so stage 1/2 run
  per variant; the reference code exists once.
- A single-size model is a family with one variant (uniform; no second loader path).
- Tests: one contract test module per family, parametrized over its variants (every field pinned per variant; mutants
  per family); stage-1 network tests per variant (one file per family, parametrized); the pairs generator writes one
  pairs file per variant id.

## Variants to carry (Hub API, 2026-10-08; quantized repackagings excluded)
| Family | Variants (new in bold) |
|---|---|
| qwen3-embedding | 0.6b, **4b**, **8b** |
| qwen3-vl-embedding | 2b, **8b** |
| qwen3-reranker | 0.6b, 4b, 8b |
| qwen3-vl-reranker | 2b, **8b** |
| ctxl-rerank-v2-instruct-multilingual | 1b, 2b, 6b |
| zerank | 1, 1-small, 2 (same family only if the architecture and reference are shared; else separate families) |
| octen-embedding | **0.6b**, **4b**, 8b |
| jina-embeddings-v5-text | **nano**, small |
| topk-embed-v1 | **xsmall**, small |
| pplx-embed-v1 | **0.6b**, **4b** |
| pplx-embed-v2-late | 0.6b (lane rec-pplx-late), **9b** |
| pplx-embed-v2-context | 9b-preview |
| zembed-1, jina-reranker-v3 | one variant each |
Variant ids keep today's naming rule (the lowercased canonical Hub repo name).
