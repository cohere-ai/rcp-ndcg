# Embedding endpoints

Dense embeddings cross one wire shape: OpenAI `POST {base_url}/embeddings`. The self-hosted engines (vLLM,
TEI, Infinity) and the OpenAI API answer it as is; the hosted APIs (Cohere, Voyage, Gemini) are profiles
of the same adapters. A config selects the wire with `api`, and one role client owns every content decision --
the prompts, the normalisation, the batching -- so no engine's defaults (silent truncation, unprompted
pooling changes) ever reach your vectors. This is the one embedding path: the retrieval commands
(`rcp-ndcg retrieval index|search`, a run's `retrieve` step) build their `EmbeddingClient` or `PoolingClient`
from a retriever config's `encoder` on exactly this layer.

## The config

An embedding endpoint is an `EmbeddingEndpoint` (`rcp_ndcg.inference.config`): the endpoint fields
(`model`, `revision`, `api_key_env`, `headers_env`, `concurrency`, the timeouts, retries and
`wait_on_outage_s`) plus the fields its wire protocol needs.

| Field | What it does |
|---|---|
| `api` | The wire adapter, from the embed role's registry: `openai_embeddings` (default), `cohere`, `voyage`, `gemini`, or a third party's from the `rcp_ndcg.adapters` entry-point group (entries named `embed.<name>`) |
| `base_url` | The endpoint; `null` for a hosted API, which then uses the profile's public URL |
| `api_key_env` | The variable holding the key, resolved by the transport; when unset, the wire adapter profile's own variables are tried in the profile's header -- but only when the request goes to the profile's own default host. Any other `base_url` receives a key only from an explicit `api_key_env` ([the credential rule](../reference/cli.md#credentials)) |
| `query_prompt`, `doc_prompt` | Text prepended to every query / document (an asymmetric embedder's instruction prefix); refused beside a `template` -- the prefix then goes in as the template's fixed segment, so declaring both would double it |
| `normalize` | L2-normalise the vectors (the default); normalising twice is harmless |
| `dimensions` | The Matryoshka cut, sent only when set |
| `batch_size` | Texts per request, refused above a hosted profile's published cap (Cohere 96, Voyage 128, Gemini 100). No `openai_embeddings` batch is ever refused client-side (its shape serves engines too): an over-count engine answers its own refusal (`HTTP 413`, mapped to a typed `CapabilityError` naming `batch_size`) -- no stale client-side cap turns one away |
| `concurrency` | Batch requests in flight at once |
| `recipe`, `tokenizer`, `max_tokens`, `query_max_tokens` | Declared for the served engine's settings and the client-side text budget (below): a self-hosted config must declare both `tokenizer` and `max_tokens`, and the client cuts the content spans itself (`on_overflow: cut`, the default; `chunk` pools scores by max, and an embedding has none to pool, so it is refused). The hosted profiles take no `dimensions` (their APIs fix the output dimension); a config that sets `dimensions` on one is refused when the request is built (the API has no such parameter). `max_tokens` caps the document shape; `query_max_tokens` (per-shape budgets) caps the query shape whole -- an asymmetric or late-interaction embedder caps queries and documents differently -- and must not exceed `max_tokens` |

Two hosted shortcuts: a config with no `base_url` points at the profile's public URL
(`https://api.cohere.com/v2` for Cohere, and so on), and a profile that requires a key raises a
`CredentialsError` naming its variables when none is set. The profile's variables travel only to the
profile's own default host: any other `base_url` (a served engine, a third party, a gateway) carries a key
only through an explicit `api_key_env` -- a variable set for one vendor must never authenticate a request
somewhere else.

## The wire adapters

An adapter turns one role's request into HTTP calls and the replies back into vectors. The embedding adapters
are registered under their names and are stateless:

* `openai_embeddings` sends `{"model", "input": [texts], "encoding_format": "float"}` plus `dimensions` when
  the config sets one, and reads `data[].embedding` in `data[].index` order (float lists or base64 float32).
  `request_shape: messages` sends the chat-style embeddings input instead -- each item its own conversation
  of one user message (one conversation alone, a list of them for a batch: vLLM reads one conversation as one
  embedding), lowered with image parts and video parts (sampled frames as image parts, a `video_url` container
  per the role's `video_policy`) for the vision-language embedders. The engine renders every chat-shaped
  request through its chat template, which places each media part inside the user turn, so the item carries
  its content (the prompt and the cut content span), never the framed render: the served chat template must
  render the declared template's frame around it, once, and the request carries the declared
  `add_special_tokens` (the chat route's own default is false). A frame that ends with the chat template's
  generation prompt (the assistant header, Qwen3-VL-Embedding's) declares `add_generation_prompt: true`, which
  every `messages` request then carries (the chat route's default is false, and the header would be missing);
  the flag is refused on any other route. `request_shape: token_ids` sends the ids the
  client fitted; the hosted profiles implement text only.
* `cohere` sends `{"model", "texts", "input_type", "embedding_types": ["float"]}` to `POST {base_url}/embed`,
  with `input_type` `search_query` / `search_document`, and reads `embeddings.float`.
* `voyage` sends the OpenAI body with an `input_type` of `query` / `document`, and reads the OpenAI
  `data[].embedding` shape.
* `gemini` sends `POST {base_url}/models/{model}:batchEmbedContents` with one `requests` entry per text
  (`models/<model>`, `taskType` `RETRIEVAL_QUERY` / `RETRIEVAL_DOCUMENT`), puts the key in `x-goog-api-key`,
  and reads `embeddings[].values`.

Every adapter refuses media on the text and token-ids routes with a `CapabilityError` naming the media type
(the `messages` route carries it), maps an over-length HTTP 400 ("maximum context length") onto a `CapabilityError` whose hint names
`max_tokens` and `batch_size`, and a batch-cap HTTP 413 onto a `CapabilityError` naming `batch_size`. Any other
HTTP 400 or 422 is a `RequestRejectedError` for that one request.

Names are scoped by role: the rerank role registers its own `cohere` and `voyage` wires, and a config selects
only among its own role's names. The adapters are usable on their own -- build the role's request, read the
calls, parse the replies:

```python
from rcp_ndcg.inference import EncodeRole, get_adapter
from rcp_ndcg.inference.types import EmbedRequest
from rcp_ndcg_core.content import Content

request = EmbedRequest(
    contents=(Content.from_text("a query about foxes"),),
    role=EncodeRole.QUERY,
)
call = get_adapter("cohere", role="embed")().calls(request, model="embed-v4.0")[0]
print(call.path, sorted(call.json))
```

## The client

`EmbeddingClient` (`rcp_ndcg.inference.clients`) applies the config's content decisions and sends the batches
through a transport -- its own, built from the config, or one you pass as `sender` (any `Sender` with the sync
bridge `run`; a transport's is used as is and pointed at the profile's credential facts). The config declares
the content; the client applies it, on the shared client base (`RoleClient` -- the adapter lookup, the hosted
profile's URL, the transport, the sync bridge, `close()`/`await aclose()` and the context managers, and the
fan-out under one `asyncio.TaskGroup`):

```python
from rcp_ndcg.inference.config import EmbeddingEndpoint

config = EmbeddingEndpoint(
    base_url="http://127.0.0.1:8000/v1",
    model="octen-embedding-8b",
    tokenizer="tok",
    max_tokens=8192,  # a self-hosted config declares its text budget: the package cuts, never the engine
    query_prompt="query: ",
    doc_prompt="- ",
    normalize=True,
    batch_size=32,
    concurrency=8,
)
print(config.api, config.batch_size, config.concurrency)
```

`client.encode(contents, role)` is synchronous and `client.aencode(contents, role)` asynchronous; both return
an `Embeddings` with one vector per content, in the input's order. The client:

* prepends `query_prompt` / `doc_prompt` per side, through one private seam, and then fits every request into
  the declared budget: the side's shape (`query` or `document`), only content spans cut, the template
  re-attached with its anchors, every cut recorded in the census (a hosted profile that declares no limit sends
  every item whole; a self-hosted config must declare its budget);
* slices the items into `batch_size`-sized requests and keeps at most `concurrency` in flight under one
  `asyncio.TaskGroup` (a failing request cancels its siblings), reassembling in the input's order whatever
  order the replies arrive in;
* L2-normalises when `normalize`;
* resolves nothing credential-wise: the key is the transport's, from `api_key_env` (else the profile's own
  variables, which apply only to the profile's own default host), sent in the profile's header.

The vectors are raw float32 from the adapter -- the normalisation is the client's content decision, not the
wire's. Each adapter's `usage()` reports the input tokens its API names (OpenAI's `usage.prompt_tokens`,
Cohere's billed units; Gemini reports none), for the transport's accounting. `dimensions` changes what the
server computes and is content; `base_url`, `batch_size`, `concurrency`
and the credentials change where and how fast, and are runtime.

## Identity

Two runs share an index only if they computed the same vectors. Which model, checkpoint and wire adapter
computed them (`api`, `model`, `revision`, `recipe`, the prompts, `normalize`, `dimensions`) is content and
enters the identity; where and how fast (`base_url`, `batch_size`, `concurrency`, the timeouts) is runtime and
never does. The tokenizer's name is runtime and its digest is content
([the tokenizer's digest](text-budgets.md#the-tokenizers-digest)): the `retrieve`/`rerank` step identities and the
index identity key on it.

## Text budgets

A self-hosted role config declares its budget explicitly (`tokenizer` + `max_tokens`; a config with one and
not the other is refused): the client fits every request through the one text-budget mechanism
(`rcp_ndcg.data.preprocess.fit`), cutting each text at token boundaries of the declared `tokenizer`, with the
template re-attached around the cut so every anchor survives, and every cut recorded -- each census row
naming its shape's budget (`budget_tokens`: the query rows a declared `query_max_tokens`, the document rows
`max_tokens`). The template can also declare a per-shape content normalisation (`normalize`:
`"strip"`, `"lowercase"` -- the reference and the engine see the same text). Never engine-side:
no request asks for truncation, and an over-length HTTP 400 from an engine maps to a `CapabilityError` whose
hint names `max_tokens` and `batch_size`. A hosted profile may declare only `max_tokens` (its documented
limit): the content is sent uncut and the limit is recorded as the effective budget (`budget_source:
vendor`). See [text budgets for served roles](text-budgets.md) for the mechanism.
