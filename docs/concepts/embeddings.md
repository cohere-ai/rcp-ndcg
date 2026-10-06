# Embedding endpoints

Dense embeddings cross one wire shape: OpenAI `POST {base_url}/embeddings`. The self-hosted engines (vLLM,
SGLang, TEI, Infinity) and the OpenAI API answer it as is; the hosted APIs (Cohere, Voyage, Gemini) are profiles
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
| `api_key_env` | The variable holding the key; when unset, a hosted profile reads its own (e.g. `CO_API_KEY` or `COHERE_API_KEY`) |
| `query_prompt`, `doc_prompt` | Text prepended to every query / document (an asymmetric embedder's instruction prefix) |
| `normalize` | L2-normalise the vectors (the default); normalising twice is harmless |
| `dimensions` | The Matryoshka cut, sent only when set |
| `batch_size` | Texts per request, refused above the profile's published cap (Cohere 96, Voyage 128, Gemini 100, the OpenAI route 128) |
| `concurrency` | Batch requests in flight at once |
| `recipe`, `max_tokens`, `tokenizer` | Declared for the served engine's settings and the client-side text budget (see below). The hosted profiles take no `dimensions` (their APIs fix the output dimension); a config that sets `dimensions` on one is refused |

Two hosted shortcuts: a config with no `base_url` points at the profile's public URL
(`https://api.cohere.com/v2` for Cohere, and so on), and a profile that requires a key raises a
`CredentialsError` naming its variables when none is set. A served engine (`base_url` set, `openai_embeddings`)
takes no key unless `api_key_env` names a variable.

## The wire adapters

An adapter turns one role's request into HTTP calls and the replies back into vectors. The embedding adapters
are registered under their names and are stateless:

* `openai_embeddings` sends `{"model", "input": [texts], "encoding_format": "float"}` plus `dimensions` when
  the config sets one, and reads `data[].embedding` in `data[].index` order (float lists or base64 float32).
* `cohere` sends `{"model", "texts", "input_type", "embedding_types": ["float"]}` to `POST {base_url}/embed`,
  with `input_type` `search_query` / `search_document`, and reads `embeddings.float`.
* `voyage` sends the OpenAI body with an `input_type` of `query` / `document`, and reads the OpenAI
  `data[].embedding` shape.
* `gemini` sends `POST {base_url}/models/{model}:batchEmbedContents` with one `requests` entry per text
  (`models/<model>`, `taskType` `RETRIEVAL_QUERY` / `RETRIEVAL_DOCUMENT`), puts the key in `x-goog-api-key`,
  and reads `embeddings[].values`.

Every adapter refuses media with a `CapabilityError` naming the media type (these endpoints are text-only for
now), maps an over-length HTTP 400 ("maximum context length") onto a `CapabilityError` whose hint names
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
through a transport -- its own, built from the config, or one you pass as `sender` (until the built-in
transport is wired, construct the client with an explicit `sender`). The config declares the
content; the client applies it:

```python
from rcp_ndcg.inference.config import EmbeddingEndpoint

config = EmbeddingEndpoint(
    base_url="http://127.0.0.1:8000/v1",
    model="octen-embedding-8b",
    tokenizer="fixtures/tokenizer.json",
    max_tokens=4096,  # a self-hosted config declares its text budget: the package cuts, never the engine
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

* prepends `query_prompt` / `doc_prompt` per side, through one private seam the text-budget mechanism plugs
  into when it lands (until then a config that sets `max_tokens` is refused with a `ConfigError`, never
  silently ignored);
* slices the items into `batch_size`-sized requests and keeps at most `concurrency` in flight, reassembling in
  the input's order whatever order the replies arrive in;
* L2-normalises when `normalize`;
* resolves the API key from `api_key_env` (else the profile's own variables) and sends it in the profile's
  header, so the transport never adds a second one.

The vectors are raw float32 from the adapter -- the normalisation is the client's content decision, not the
wire's. Each adapter's `usage()` reports the input tokens its API names (OpenAI's `usage.prompt_tokens`,
Cohere's billed units; Gemini reports none), for the transport's accounting. `dimensions` changes what the
server computes and is content; `base_url`, `batch_size`, `concurrency`
and the credentials change where and how fast, and are runtime.

## Identity

Two runs share an index only if they computed the same vectors. Which model, checkpoint and wire adapter
computed them (`api`, `model`, `revision`, `recipe`, the prompts, `normalize`, `dimensions`) is content and
enters the identity; where and how fast (`base_url`, `batch_size`, `concurrency`, the timeouts) is runtime and
never does. The tokenizer's name is runtime: the config inherits `Endpoint.identity_extra()`, which returns the SHA-256 of
its `tokenizer.json` (`{"tokenizer_sha256": ...}`) under that one key. The `retrieve`/`rerank` step identities
splice the digest in at the encoder and the reranker -- the same rule the judge applies to its `tokenizer` --
and the index identity carries it too. Every role config with a `tokenizer` (the judge's, the embedding, pooling and rerank configs)
carries the digest under this one key, from the one helper in `rcp_ndcg.data.tokenizer`.

## Text limits

A config that sets `max_tokens` is refused today: cutting to a token budget is the text-budget mechanism's job
(`rcp_ndcg.data.preprocess`), which will wire into this client and cut each text at token boundaries of the
declared `tokenizer`, never by engine-side truncation. Until then, send text that already fits -- a request an
engine cannot take fails with a `CapabilityError`, never a silent cut.
