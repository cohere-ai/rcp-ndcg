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
refused; only a usage-less reply passes through as one vector per item.

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
  processing record (`skip_unapplied`);
* `mrl_dim` applies the Matryoshka cut client-side as cut-then-renormalise
  (the card's order; `/pooling` refuses per-request `dimensions`), and
  `outputs: per_chunk` accepts a per-chunk multi-output model -- several
  outputs per input -- where the per-token usage cross-check cannot apply;
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
