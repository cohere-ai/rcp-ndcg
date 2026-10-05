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
| `input` | the batch's texts | one list per request |
| `task` | `"token_embed"` | one vector per token, the late-interaction task |
| `encoding_format` | `"base64"` | JSON floats are what made corpus indexing expensive |
| `embed_dtype` | the config's `embed_dtype` | the transfer precision (below) |
| `endianness` | `"little"` | explicit, so the frame decodes on any server platform |

Two request shapes exist, and only one of them runs the server's chat template:
a text-only batch travels as one `input` list (tokenised raw), while a media
item (a page image, for the ColPali and ColQwen3 checkpoints) travels as its own
`messages` request — the only shape in which the template reaches the image
placeholders. `dimensions` is never sent:
`/pooling` refuses it.

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

A reply that reports one vector per item and a `usage` line — the shape a
pooled (not `token_embed`) server answers — is refused, because one vector per
item contradicts one vector per prompt token; without a `usage` line it passes
through as one vector per item.

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

A `PoolingClient` owns the content decisions the config states, and nothing else:

* the role's prompt (`query_prompt`/`doc_prompt`) is prepended by `_prepare`,
  the one seam the text budget will join. Until that mechanism is wired the
  client cuts nothing and sends every content as given — a config that sets
  `max_tokens` is refused (`ConfigError`), because a budget silently ignored
  would change the vectors;
* `normalize` (the default) L2-normalises every token vector, in float32,
  stored back in the transfer dtype;
* `batch_size` items per request, at most `concurrency` requests in flight,
  reassembled in input order.

## MaxSim scoring

`maxsim_topk(documents, queries, k)` blocks both axes by token count: the peak
working set is one query block, one document block and one score tile, each
bounded by the 64 MiB tile budget — never a float32 copy of the whole corpus.
Vectors may be float16 or float32; every dot product and per-query sum is
computed in float32, upcast block by block (upcasting is exact, so a float32
corpus is scored exactly as before and the per-token L2 norms the client stores
survive the round trip). Ties break toward the lower document index, so a run is
reproducible across machines.
