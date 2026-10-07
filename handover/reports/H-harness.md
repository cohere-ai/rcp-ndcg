# Report H: harness and product follow-ups found by the recipe families (integrated)

**Status:** DONE, merged into `rfc-0001` after one verifier round (three majors fixed) at M3.

## What landed (each test-first)
- **Per-row `ProcessingRecord`** (`client.processing`): every input row the client changed relative to the uncut
  input, with named causes (`budget_cut`, `query_share`, `document_share`, `empty_doc`, `media_resize`, `media_drop`),
  the uncut and kept request totals (frame and media included) and the budget. Built from data the client already has
  (the census rows carry the same facts); nothing is measured twice.
- **Per-text gating in the equivalence harness** (decision 9): a text the client changed is reported non-gating under
  the recipe's declared deviation; every other text gates exactly, including uncut siblings in the same pairs row.
  A settled query makes every pair of its row non-gating (decision 26); a document's own cut affects only its pair.
  Declared normalisation is never a change; only actual removals and budget-driven media changes are.
- `anchor: last_content` audit; `RerankEndpoint.document_max_tokens` (a per-document cap applied client-side, anchors
  kept, in the identity and the census); `ImagePolicy.engine_pixel_pinning` (a budget below a processor's default
  floor when the engine is pinned identically; the recipe validator checks the serve pin, nested or flat, including HF
  `size` pins); a pinned shrink that cannot fit raises a typed error.
- The embed `messages` route sends the content and lets the engine's chat template frame it once, one conversation per
  item (vLLM v0.31.0 embeds a multi-turn conversation as one vector), with the declared `add_special_tokens`.
- `empty_doc` applies to the content before the prompt and the template.
- The offline fake counts tokens with the config's declared tokenizer (whitespace words remain the documented
  fallback).
