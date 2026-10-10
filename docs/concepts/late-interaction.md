# Late interaction: multi-vector encoding and MaxSim

A single-vector model asks "how close are these two points"; a late-interaction
(ColBERT-style) model keeps one vector per token and asks, for each query token,
"what is the best thing in this document to match it against", and sums the
answers:

$$
s(Q, D) = \sum_{i \in Q} \max_{j \in D} q_i \cdot d_j
$$

That is not expressible as an inner product between two pooled vectors, which is
why a late-interaction model needs its own wire format (one ragged buffer of
token vectors, not one vector per text) and its own scorer
(`rcp_ndcg.retrieval.maxsim.maxsim_topk`).

## The pooling wire: `vllm_pooling`

A multi-vector encoder is served with vLLM's pooling API and reached with the
`vllm_pooling` wire adapter (the `api` field of a `PoolingEndpoint`), which sends
`POST {base_url}/pooling` with `task: "token_embed"` — the field names and values
below are verified against the vLLM entrypoints (`vllm/entrypoints/pooling/` and
`vllm/utils/serial_utils.py` in the vLLM checkout):

| Field | Sent | Why |
| --- | --- | --- |
| `model` | the config's `model` | the served model name |
| `input` | the batch's texts (or, under `request_shape: token_ids`, the fitted id lists) | one list per request |
| `task` | `"token_embed"` | one vector per token, the late-interaction task |
| `encoding_format` | `"base64"` | JSON floats are what made corpus indexing expensive |
| `embed_dtype` | the config's `embed_dtype` | the transfer precision (below) |
| `endianness` | `"little"` | explicit, so the frame decodes on any server platform |

Two wire forms exist, and only one of them runs the server's chat template:
a text-only batch travels as one `input` list (the rendered strings, or the
client-fitted id lists under `request_shape: token_ids`), while a media item (a
page image, for the ColPali and ColQwen3 checkpoints) always travels as its own
`messages` request — the only form in which the template reaches the image
placeholders, which is why the pooling wire lowers it itself and a config
declaring `request_shape: messages` is refused. A `PoolingEndpoint` refuses `dimensions` at
construction (`/pooling` has no such field).

A media batch of one page and one caption therefore becomes two requests, and the
client reassembles the vectors in input order. The `interpret` side accepts three
reply shapes, because the layout of the answer — never configuration — says which
pooling task ran:

* **base64 frames** (what the adapter asks for): the flat `embed_dtype` array,
  reshaped to `(tokens, dim)` from the declared `dim`;
* **nested float lists** (what a server that ignores `encoding_format` sends):
  decoded as they arrive;
* **the framed `bytes` encoding**: per-item `start`/`end`/`shape` metadata from
  the response header (`bytes_only` sends no framing and is refused).

A reply that reports one vector per item and a `usage` line whose token counts
contradict it — the shape a pooled (not `token_embed`) server answers — is
refused; only a usage-less reply passes through as one vector per item. When the
config declares `document_skip_engine_side` the engine's reply carries only the
kept vectors, so the check is the declared kept count instead (`kept_vector_count`
per item, carried on the request as `PoolRequest.kept_counts`): the reply's
decoded count must equal it, and a reply that ignored the rule (the full prompt
count) is a typed `ProviderError`.

```python
import numpy as np

from rcp_ndcg.inference.adapters.pooling import VllmPooling
from rcp_ndcg.inference.types import EncodeRole, PoolRequest, Reply
from rcp_ndcg_core.content import Content

import base64

request = PoolRequest(
    contents=(Content.from_text("evidence"),),
    role=EncodeRole.DOCUMENT,
    embed_dtype="float16",
    dim=2,
)
calls = VllmPooling().calls(request, model="colbert")
assert calls[0].json == {
    "model": "colbert",
    "input": ["evidence"],
    "task": "token_embed",
    "encoding_format": "base64",
    "embed_dtype": "float16",
    "endianness": "little",
}

frame = base64.b64encode(np.array([[1.0, 0.0], [0.5, 0.5]], dtype="<f2").tobytes()).decode("ascii")
reply = Reply(200, {"data": [{"index": 0, "data": frame}]}, {})
embeddings = VllmPooling().interpret(request, [reply])
assert embeddings.is_multi_vector
assert embeddings.vectors.dtype == np.float16
```

## The transfer precision: float16 by default

The vectors cross the wire in the config's `embed_dtype`: `float16` by default
(2 bytes per token vector — a 40k-page corpus at 1,030 vectors of dim 128 is
about 10.6 GB), `float32` opt-in (4 bytes, the lossless choice). The ragged
buffer keeps that dtype end to end — an index built from float16 vectors stores
float16 — and MaxSim computes in float32 either way, so float16 costs only its
quantisation (~1e-3 relative against float64 on real documents) and nothing in
the scoring.

The flat base64 frame carries no shape, so the config declares `dim`, the
checkpoint's token-vector width. What keeps a mistyped `dim` from silently
mis-shaping a corpus is the reply's own `usage`: a `token_embed` answer has one
vector per prompt token, so the decoded token counts must sum to
`usage.prompt_tokens`, and a mismatch is a loud `ProviderError` naming the dim.

## What the client decides about content

A `PoolingClient` owns the content decisions the config states, and nothing else, on the shared client base
(`rcp_ndcg.inference.clients.RoleClient` -- the adapter lookup, the transport or an injected `Sender` with the
sync bridge, `close()`/`await aclose()`, and the fan-out under one `asyncio.TaskGroup`):

* the role's prompt (`query_prompt`/`doc_prompt`) is prepended by `_prepare`, and then -- when the config
  declares a budget (`tokenizer` + `max_tokens`, explicit for a self-hosted role) -- every item's text is
  fitted through the one text-budget mechanism: only the content span is cut, the template re-attached with
  its anchor, every cut recorded in the census. A hosted profile that declares no limit sends every item whole;
  a self-hosted config must declare its budget (`tokenizer` + `max_tokens`). Per-shape
  budgets: `query_max_tokens` caps the query shape whole (a late-interaction embedder caps its two sides
  differently), `max_tokens` caps the document shape, and a query budget above `max_tokens` is refused. Media
  items keep their parts beside the fitted text; a config with `dim` unset is refused at construction (the
  base64 frame needs the width -- a refusal at construction keeps the GPU idle-time free), and so is
  `on_overflow: chunk` (chunks pool scores by max, and token vectors have none to pool -- a
  late-interaction document is chunked at the corpus layer, one slice per chunk in the index);
* `normalize` (the default) L2-normalises every token vector, in float32,
  stored back in the transfer dtype;
* `document_skip_token_ids` drops document vectors at the positions whose token
  id is listed (the topk reference scores nothing by 41 punctuation/special
  ids; queries keep all their vectors) -- the positions are the ids the client
  sent, and a count mismatch is a typed error. The skip rule at image
  positions: a media document's positions are the server's chat-template
  render, which the client cannot tokenise, so the media vectors are never
  skipped -- the client keeps them all and records the deviation on the row's
  processing record (`skip_unapplied`). (The engine's `/pooling` route returns
  no per-position token ids: `PoolingParams.returned_token_ids` is a step
  pooler's output slice and `requires_token_ids` is the worker's internal CPU
  copy, while the response carries vectors only -- so the client cannot compute
  a checkpoint's own image-position mask from the reply and keeps the media
  document whole, on record.)
* `document_skip_engine_side` moves the rule above into the served plugin
  instead: the recipe declares the same ids for the engine
  (`serve.hf_overrides.document_skip_token_ids`, which the recipe loader
  cross-checks against the client's list) together with the document ROLE GATE
  (`serve.hf_overrides.document_skip_prefix_token_id`, the leading token id a
  document prompt opens with -- the checkpoint's mask is document-side, so a
  query prompt keeps every position), and the plugin's pooler drops the rule's
  positions from the token ids it sees -- the render's own ids, a media
  document's head and vision markers included (the head is in the render when
  the recipe sends it, `media_head_as_system`; the shipped pplx-late family
  does) -- so the wire carries only kept vectors. The client cannot recompute
  the kept set from the reply:
  it counts the declared kept vectors instead (`kept_vector_count`: the sent
  render's ids outside the rule, or a media document's sent head plus its
  prepared media block) and refuses a reply whose per-item count disagrees (a
  typed `ProviderError`, never a silent misalignment). No `skip_unapplied`
  record is written then -- the engine applied the rule. The engine's
  `usage.prompt_tokens` counts the *prompt*, so under this rule it no longer
  describes the vectors: the declared counts are what the reply is checked
  against. vLLM v0.31.0's pooling route still returns no per-position token ids,
  which is why the rule's home is the engine-side plugin; a recipe without
  this flag keeps the client-side rule above.
* `media_keep_token_ids` is the MEDIA allowlist, for a checkpoint whose image
  documents keep only a subset of the render's positions (topk-embed-v1's
  image-patch token: its reference keeps `ids == image_token_id` for an image
  document). The served plugin applies it engine-side through the same path
  (the recipe declares the same ids for the engine in
  `serve.hf_overrides.document_keep_token_ids`, which the loader cross-checks),
  and the allowlist is its own gate: a row carrying one of its ids is a media
  document and keeps only those positions -- the chat template's wrapper, the
  trained head and a caption drop, exactly what the reference keeps. The
  client counts the media block's patch run (the block's counted tokens minus
  the vision wrapper) and refuses a reply whose per-item count disagrees; no
  `skip_unapplied` record is written for a media item, because the engine
  applied the allowlist. A recipe without it keeps the media-render rule
  above: a media document's vectors are kept whole, on record.
* `media_head_as_system` (a pooling config with a template and media) sends the
  side's leading fixed template segments -- the trained role prefix, e.g.
  `[D] ` -- as a leading `system` message for a media item, instead of inside
  the user turn: a checkpoint whose engine chat template injects no frame of
  its own (a pass-through template, pplx-embed-v2-late) would otherwise render
  an image-only document without the prefix the card's sentence-transformers
  path sends as a system message. The user turn then carries only the content
  span (the image), the fixed overhead the budget reserves already includes the
  head, and the startup media probe's baseline carries the same head so the
  engine's media delta cancels it;
* `mrl_dim` applies the declared Matryoshka head client-side ([matryoshka heads](matryoshka.md)): the
  full-width reply is normalised when `normalize`, then the head cuts and renormalises (or applies the
  checkpoint's learned projection), and every changed item carries an `mrl_cut` `ProcessingRecord`. The
  config refuses a `k` outside `mrl_dims`/`mrl_range` and an `mrl_dim` at or over `dim`; `dimensions` is refused on this
  wire (`/pooling` has no such field). `outputs: per_chunk` accepts a per-chunk multi-output model --
  several outputs per input -- where the per-token usage cross-check cannot apply;
* `batch_size` items per request, at most `concurrency` requests in flight,
  reassembled in input order.

## MaxSim scoring

`maxsim_topk(documents, queries, k)` blocks both axes by token count: the peak
working set is one query block (budgeted at 4,096 rows by the mean token
counts), one document block and one score tile, the float32 copies bounded by
the 64 MiB tile budget for near-uniform token counts (larger in proportion to
the skew within a block) — never a float32 copy of the whole corpus. Vectors may
be float16 or float32; every dot product and per-query sum is computed in
float32, upcast block by block (upcasting is exact, so a float32 corpus is
scored exactly as before and the per-token L2 norms the client stores survive
the round trip). Ties break toward the lower document index, so a run is
reproducible across machines.
