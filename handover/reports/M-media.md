# Report M-media: the media equivalence gate and the messages route (integrated)

**Status:** DONE, merged into `rfc-0001` at M3 after one independent verifier round. Every fix is test-first.

## What landed
- **Media equivalence stage** (`equivalence/media.py`): for every pairs row with media, each media side goes through
  the product's role client; the stage compares the parts actually sent (order, each image's prepared size decoded
  from the sent bytes, the client's token count) with the reference's `--mode media`. With an engine it re-sends each
  request without its media; the `usage.prompt_tokens` difference is the engine's own media count and must equal the
  client's. `run()` adds the stage for every recipe with image or video input; stages 1 and 2 compare text rows only.
- **Product fix found by the stage**: the role client put the fitted text before every image; it now keeps the given
  part order.
- **Control (f)** unpins the nested `mm_processor_kwargs.images_kwargs` pin (before, it looked for flat keys only and
  found nothing on the real recipes); it is not applicable when the client prepares images inside the family's stock
  range. A test shows the unpinned budget fails the media stage.
- **Media request set** (`observe/media_set.py`, versioned on its own so no text rows were re-sampled): one row per
  image size bucket plus a captioned page; too-many-images and a corrupt image are sent as bare requests; video is
  recorded absent with the reason.
- **`add_generation_prompt`** on `EmbeddingEndpoint` (messages route only; vLLM v0.31.0 defaults it to false at
  `vllm/entrypoints/pooling/base/protocol.py:230-237`; `false` is stored as `None`, so no existing identity changes;
  refused on other routes and the pooling role). qwen3-vl-embedding-2b uses the messages route with the flag.
- **`checkpoint_chat_template`** reads `chat_template.jinja`, then `chat_template.json`, then `tokenizer_config.json`
  at the pinned revision; unreadable means the check fails as `unresolved`, never passes.
- **Stub engine** handles messages bodies like vLLM's chat path (template render, image resize, refusals, usage);
  stage 2 runs on the messages route.
- **Recipes**: jina-reranker-v3 `document_max_tokens: 2048`; current notes for jina-v5 and qwen3-vl-embedding;
  `engine_pixel_pinning` plus `image_processor: qwen3_vl` for qwen3-vl-embedding-2b and qwen3-vl-reranker-2b (without
  it the reranker's images were refused; its text-only corpus was re-keyed); the qwen3-vl-reranker reference accepts a
  whitespace-only query as the card's `format_mm_content` does (only the empty query is refused, as the client does).
  Pairs regenerated for the three media recipes and jina-reranker-v3 (text rows unchanged; media rows added).
- **Hygiene**: public `stored_tokenizer` and `bare_exchange` replace private imports; `runs.execution.stage_run`
  replaces the e2e driver's private call; `changes.py` and `tests/_engines.py` read corpora through the corpus reader;
  the recipe loader refuses duplicate YAML keys; re-keyed corpora keep the old manifest digest; `stale.json` must be
  empty under `RCP_NDCG_RELEASE=1` (a test enforces it; it fails today by design). NOTICE attributes the restated
  `smart_resize` (qwen-vl-utils 0.0.14, transformers).

## Verifier round (one independent verifier; all confirmed findings fixed test-first)
Control (f) now decides its applicability from the checkpoint's own pixel budget at the pinned revision (not
applicable to qwen3-vl-reranker-2b and topk, whose pins lie inside it; served and caught for qwen3-vl-embedding-2b;
unreadable is a blocker); the recorder records a client's media refusal instead of failing the whole corpus step;
the media stage names reference refusals first and fails uncounted image tokens; the checkpoint reader falls through
only on a file the Hub (or the cache's record of it) says is absent, never on one that is merely uncached offline;
the generator's index remap is pinned by a test; NOTICE credits the fixture reference's `smart_resize`.

## Verified
The full quality bar on the merged tip (see the master section 3). Network-gated stage 1 per recipe file, one at a
time: qwen3-vl-embedding, qwen3-vl-reranker, jina-reranker-v3, jina-v5, topk, octen, qwen3-embedding, zembed, pplx all
green.

## Open (in the master's section 9)
topk cannot send images; video is not gated; the Hub dependency of
the qwen3-vl-embedding stage 1; `find_corpora` still parses manifests itself. The re-key stays a documented procedure,
not a script: the release empties `stale.json` by re-recording.
