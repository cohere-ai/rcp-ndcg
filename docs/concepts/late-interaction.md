# Late interaction: multi-vector encoding and MaxSim

A single-vector model asks: how close are these two points? A late-interaction (ColBERT-style) model keeps one
vector per token. It asks, for each query token: what is the best thing in this document to match it against?
It then sums the answers:

$$
s(Q, D) = \sum_{i \in Q} \max_{j \in D} q_i \cdot d_j
$$

This score is not an inner product between two pooled vectors. A late-interaction model therefore needs its
own wire format: one ragged buffer of token vectors, not one vector per text. It also needs its own scorer
(`rcp_ndcg.retrieval.maxsim.maxsim_topk`).

## The pooling wire: `vllm_pooling`

A multi-vector encoder is served with vLLM's pooling API. The `vllm_pooling` wire adapter reaches it; that is
the `api` field of a `PoolingEndpoint`. The adapter sends `POST {base_url}/pooling` with
`task: "token_embed"`. `base_url` is the engine's root, because vLLM registers `/pooling` at the root, not
under `/v1`. A `.../v1` base is refused where the retriever config is read, never as a 404 at the first
request. The field names and values below are verified against the vLLM entrypoints. Those are
`vllm/entrypoints/pooling/` and `vllm/utils/serial_utils.py` in the vLLM checkout:

| Field | Sent | Why |
| --- | --- | --- |
| `model` | the config's `model` | the served model name |
| `input` | the batch's texts (or, under `request_shape: token_ids`, the fitted id lists) | one list per request |
| `task` | `"token_embed"` | one vector per token, the late-interaction task |
| `encoding_format` | `"base64"` | JSON floats are what made corpus indexing expensive |
| `embed_dtype` | the config's `embed_dtype` | the transfer precision (below) |
| `endianness` | `"little"` | explicit, so the frame decodes on any server platform |

Two wire forms exist, and only one of them runs the server's chat template. A text-only batch travels as one
`input` list: the rendered strings, or the client-fitted id lists under `request_shape: token_ids`. A media
item always travels as its own `messages` request. That item is a page image for the ColPali and ColQwen3
checkpoints. The `messages` form is the only one in which the template reaches the image placeholders. The
pooling wire therefore lowers a media item itself, and a config declaring `request_shape: messages` is
refused. A `PoolingEndpoint` refuses `dimensions` at construction, because `/pooling` has no such field.

A media batch of one page and one caption therefore becomes two requests. The client reassembles the vectors
in input order. The `interpret` side accepts three reply shapes. It checks the layout against the requested
`token_embed` contract rather than trusting it:

* **base64 frames** (what the adapter asks for): the flat `embed_dtype` array,
  reshaped to `(tokens, dim)` from the declared `dim`;
* **nested float lists** (what a server that ignores `encoding_format` sends):
  decoded as they arrive, and a 2-D frame's width must match the declared `dim`;
* **the framed `bytes` encoding**: per-item `start`/`end`/`shape` metadata from
  the response header (`bytes_only` sends no framing and is refused).

An answer of one vector per item is a *pooled* task, not `token_embed`. It is refused outright, because a
pooled answer scored as a late-interaction index would be a silent function change. There is one exception: a
config that declares `outputs: per_chunk` for a per-chunk multi-output model. A reply is refused too when its
`usage` token counts contradict one vector per prompt token. A frame holding a non-finite value is refused as
well, as the `/embeddings` wire refuses one. The config can declare `document_skip_engine_side`, or a media
`media_keep_token_ids`. The engine's reply then carries only the kept vectors, so the check is the declared
kept count instead. That count is `kept_vector_count` per item, carried on the request as
`PoolRequest.kept_counts`. The reply's decoded count must equal it. A reply that ignored the rule carries the
full prompt count, and it is a typed `ProviderError`.

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

The vectors cross the wire in the config's `embed_dtype`. The default is `float16`, at 2 bytes per token
vector. A 40k-page corpus at 1,030 vectors of dim 128 is about 10.6 GB. `float32` is opt-in, at 4 bytes, and
it is the lossless choice. The ragged buffer keeps that dtype end to end, so an index built from float16
vectors stores float16. MaxSim computes in float32 either way. float16 therefore costs only its quantisation,
about 1e-3 relative against float64 on real documents, and nothing in the scoring.

The flat base64 frame carries no shape, so the config declares `dim`, the checkpoint's token-vector width.
The reply's own `usage` keeps a mistyped `dim` from silently mis-shaping a corpus. A `token_embed` answer has
one vector per prompt token, so the decoded token counts must sum to `usage.prompt_tokens`. A mismatch is a
loud `ProviderError` naming the dim.

## What the client decides about content

A `PoolingClient` owns the content decisions that the config states, and nothing else. It sits on the shared
client base, `rcp_ndcg.inference.clients.RoleClient`. That base holds the adapter lookup, the transport or an
injected `Sender` with the sync bridge. It also holds `close()` /`await aclose()`, and the fan-out under one
`asyncio.TaskGroup`:

* `_prepare` prepends the role's prompt (`query_prompt`/`doc_prompt`). The config can declare a budget (`tokenizer`
  + `max_tokens`, explicit for a self-hosted role). Every item's text is then fitted through the one text-budget
  mechanism. Only the content span is cut, the template is re-attached with its anchor, and every cut is recorded in
  the census. A hosted profile that declares no limit sends every item whole. A self-hosted config must declare its
  budget (`tokenizer` + `max_tokens`). The budgets are per shape. `query_max_tokens` caps the query shape whole,
  because a late-interaction embedder caps its two sides differently. `max_tokens` caps the document shape, and a
  query budget above `max_tokens` is refused. Media items keep their parts beside the fitted text. A config with
  `dim` unset is refused at construction, because the base64 frame needs the width; a refusal at construction keeps
  the GPU idle-time free. `on_overflow: chunk` is refused too: chunks pool scores by max, and token vectors have
  none to pool. A late-interaction document is chunked at the corpus layer instead, one slice per chunk in the
  index;
* `normalize` (the default) L2-normalises every token vector. The normalisation computes in float32, and the result
  is stored back in the transfer dtype;
* `document_split_special_tokens` tokenises the DOCUMENT side with the reference's split parse. The split parse is
  transformers' `split_special_tokens`, the raw tokenizers' `encode_special_tokens` toggle. The field is only
  meaningful on `request_shape: token_ids`, and it is refused beside `text`/`messages`. An added SPECIAL token is
  textified instead of matched as one id. A `[D] ` render therefore opens with the reference's own two literal
  prefix ids, which is what the pplx-embed-v2-context plugin's role check reads. Unset is the default, and it is
  the file's own parse, so a recipe that does not declare the field is unchanged. The fit's budget counts the file's
  own parse. A recipe that declares the split parse therefore reserves its measured prefix delta in `max_tokens`,
  and pplx-embed-v2-context reserves the measured two tokens. The sent ids are what the engine reads, and the
  render-text comparison tokenises with the same declared parse.
* `query_split_special_tokens` sends the QUERY side as the reference's own construction. The fixed head tokenises
  with the file's own parse, so its `[Q] ` crosses as the reference's one prefix id. The content span tokenises with
  the split parse. A query carrying a literal added-token spelling therefore reads the same ids on the engine as in
  the reference. pplx-embed-v2-context's remote `prepare_inputs` split-tokenises the query body too. The field is
  only meaningful on `request_shape: token_ids`. It is refused beside `text`/`messages`, and it is refused without a
  template whose query shape is a fixed head followed by one content span. Unset is the default, and it is one parse
  of the whole render, so a recipe that does not declare the field is unchanged.
* `document_skip_token_ids` drops document vectors at the positions whose token id is listed. The topk reference
  scores nothing by 41 punctuation/special ids, and queries keep all their vectors. The positions are the ids that
  the client sent, and a count mismatch is a typed error. The rule does not apply at image positions. A media
  document's positions are the server's chat-template render, which the client cannot tokenise, so the media
  vectors are never skipped. The client keeps them all, and it records the deviation on the row's processing record
  (`skip_unapplied`). The engine's `/pooling` route returns no per-position token ids.
  `PoolingParams.returned_token_ids` is a step pooler's output slice, and `requires_token_ids` is the worker's
  internal CPU copy, while the response carries vectors only. The client therefore cannot compute a checkpoint's
  own image-position mask from the reply, and it keeps the media document whole, on record.
* `document_skip_engine_side` moves the rule above into the served plugin. The recipe declares the same ids for the
  engine in `serve.hf_overrides.document_skip_token_ids`, and the recipe loader cross-checks them against the
  client's list. The recipe also declares the document ROLE GATE in
  `serve.hf_overrides.document_skip_prefix_token_id`. That is the leading token id that a document prompt opens
  with, because the checkpoint's mask is document-side and a query prompt keeps every position. The plugin's pooler
  drops the rule's positions from the token ids that it sees. Those ids are the render's own ids, a media
  document's head and vision markers included. The head is in the render when the recipe sends it,
  `media_head_as_system`, and the shipped pplx-late family does. The wire then carries only kept vectors. The
  client cannot recompute the kept set from the reply, so it counts the declared kept vectors instead. That count
  is `kept_vector_count`: the sent render's ids outside the rule, or a media document's sent head plus its prepared
  media block. The client refuses a reply whose per-item count disagrees, as a typed `ProviderError`, never a
  silent misalignment. No `skip_unapplied` record is written then, because the engine applied the rule. The
  engine's `usage.prompt_tokens` counts the *prompt*, so under this rule it no longer describes the vectors. The
  declared counts are what the reply is checked against. vLLM v0.33.0's pooling route still returns no
  per-position token ids, which is why the rule's home is the engine-side plugin. A recipe without this flag keeps
  the client-side rule above.
* `media_keep_token_ids` is the MEDIA allowlist. It serves a checkpoint whose image documents keep only a subset of
  the render's positions, such as topk-embed-v1's image-patch token: its reference keeps `ids == image_token_id` for
  an image document. The served plugin applies the allowlist engine-side through the same path. The recipe declares
  the same ids for the engine in `serve.hf_overrides.document_keep_token_ids`, and the loader cross-checks them. The
  allowlist is its own gate. A row carrying one of its ids is a media document, and it keeps only those positions:
  the chat template's wrapper, the trained head and a caption drop. That is exactly what the reference keeps. The
  client counts the media block's patch run, which is the block's counted tokens minus the vision wrapper. It
  refuses a reply whose per-item count disagrees. No `skip_unapplied` record is written for a media item, because
  the engine applied the allowlist. A recipe without the allowlist keeps the media-render rule above, and a media
  document's vectors are kept whole, on record. A TEXT render whose own ids carry an allowlist id would be misread
  as a media document, because the allowlist is its own gate. The client refuses that collision by name before
  sending.
* `media_head_as_system` applies to a pooling config with a template and media. It sends the side's leading fixed
  template segments as a leading `system` message for a media item, instead of inside the user turn. Those segments
  are the trained role prefix, e.g. `[D] `. Some checkpoints have an engine chat template that injects no frame of
  its own, such as a pass-through template like pplx-embed-v2-late. Such a checkpoint would otherwise render an
  image-only document without the prefix that the card's sentence-transformers path sends as a system message. The
  user turn then carries only the content span, which is the image. The fixed overhead that the budget reserves
  already includes the head. The startup media probe's baseline carries the same head, so the engine's media delta
  cancels it;
* `mrl_dim` applies the declared Matryoshka head client-side ([matryoshka heads](matryoshka.md)). The full-width
  reply is normalised when `normalize` is set. The head then cuts and renormalises, or it applies the checkpoint's
  learned projection. Every changed item carries an `mrl_cut` `ProcessingRecord`. The config refuses a `k` outside
  `mrl_dims`/`mrl_range`, and it refuses an `mrl_dim` at or over `dim`. `dimensions` is refused on this wire,
  because `/pooling` has no such field. `outputs: per_chunk` accepts a per-chunk multi-output model, with several
  outputs per input, where the per-token usage cross-check cannot apply;
* `batch_size` items per request, at most `concurrency` requests in flight,
  reassembled in input order.

## MaxSim scoring

`maxsim_topk(documents, queries, k)` blocks both axes by token count. The peak working set is one query block,
one document block and one score tile. The query block is budgeted at 4,096 rows by the mean token counts. The
float32 copies are bounded by the 64 MiB tile budget for near-uniform token counts. That budget is larger in
proportion to the skew within a block. There is never a float32 copy of the whole corpus. Vectors may be
float16 or float32. Every dot product and per-query sum is computed in float32, upcast block by block.
Upcasting is exact, so a float32 corpus is scored exactly as before. The per-token L2 norms that the client
stores survive the round trip. Ties break toward the lower document index, so a run is reproducible across
machines.
