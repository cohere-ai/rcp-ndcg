# Embedding endpoints

Dense embeddings cross one wire shape: OpenAI `POST {base_url}/embeddings`. The self-hosted engines (vLLM,
SGLang, TEI, Infinity) and the OpenAI API answer it as is; the hosted APIs (Cohere, Voyage, Gemini) are profiles
of the same adapters. A config selects the wire with `api`, and one role client owns every content decision --
the prompts, the normalisation, the batching -- so no engine's defaults (silent truncation, unprompted
pooling changes) ever reach your vectors.

This page describes the inference layer's embedding path as it ships today. The retrieval commands still use
the older hosted path (`provider:` in the retrieval config); they move onto this layer with the retrieval
rewiring.

## The config

An embedding endpoint is an `EmbeddingEndpoint` (`rcp_ndcg.inference.config`): the endpoint fields
(`model`, `revision`, `api_key_env`, `headers_env`, `concurrency`, the timeouts, retries and
`wait_on_outage_s`) plus the fields its wire protocol needs.

| Field | What it does |
|---|---|
| `api` | The wire adapter: `openai_embeddings` (default), `cohere_embed`, `voyage_embed`, `gemini_embed`, or a third party's from the `rcp_ndcg.adapters` entry-point group |
| `base_url` | The endpoint; `null` for a hosted API, which then uses the profile's public URL |
| `api_key_env` | The variable holding the key; when unset, a hosted profile reads its own (e.g. `CO_API_KEY` or `COHERE_API_KEY`) |
| `query_prompt`, `doc_prompt` | Text prepended to every query / document (an asymmetric embedder's instruction prefix) |
| `normalize` | L2-normalise the vectors (the default); normalising twice is harmless |
| `dimensions` | The Matryoshka cut, sent only when set |
| `batch_size` | Texts per request, refused above the profile's published cap (Cohere 96, Voyage 128, Gemini 100, the OpenAI route 128) |
| `concurrency` | Batch requests in flight at once |
| `recipe`, `max_tokens`, `tokenizer` | Declared for the served engine's settings and the client-side text budget (see below) |

Two hosted shortcuts: a config with no `base_url` points at the profile's public URL
(`https://api.cohere.com/v2` for Cohere, and so on), and a profile that requires a key raises a
`CredentialsError` naming its variables when none is set. A served engine (`base_url` set, `openai_embeddings`)
takes no key unless `api_key_env` names a variable.

## The wire adapters

An adapter turns one role's request into HTTP calls and the replies back into vectors. The embedding adapters
are registered under their names and are stateless:

* `openai_embeddings` sends `{"model", "input": [texts], "encoding_format": "float"}` plus `dimensions` when
  the config sets one, and reads `data[].embedding` in `data[].index` order (float lists or base64 float32).
* `cohere_embed` sends `{"model", "texts", "input_type", "embedding_types": ["float"]}` to `POST {base_url}/embed`,
  with `input_type` `search_query` / `search_document`, and reads `embeddings.float`.
* `voyage_embed` sends the OpenAI body with an `input_type` of `query` / `document`, and reads the OpenAI
  `data[].embedding` shape.
* `gemini_embed` sends `POST {base_url}/models/{model}:batchEmbedContents` with one `requests` entry per text
  (`models/<model>`, `taskType` `RETRIEVAL_QUERY` / `RETRIEVAL_DOCUMENT`), puts the key in `x-goog-api-key`,
  and reads `embeddings[].values`.

Every adapter refuses media with a `CapabilityError` naming the media type (these endpoints are text-only for
now), maps an over-length HTTP 400 ("maximum context length") onto a `CapabilityError` whose hint names
`max_tokens` and `batch_size`, and a batch-cap HTTP 413 onto a `CapabilityError` naming `batch_size`. Any other
HTTP 400 or 422 is a `RequestRejectedError` for that one request.

The adapters are usable on their own -- build the role's request, read the calls, parse the replies:

```python
from rcp_ndcg.inference import EncodeRole, get_adapter
from rcp_ndcg.inference.types import EmbedRequest
from rcp_ndcg_core.content import Content

request = EmbedRequest(
    contents=(Content.from_text("a query about foxes"),),
    role=EncodeRole.QUERY,
)
call = get_adapter("cohere_embed")().calls(request, model="embed-v4.0")[0]
print(call.path, sorted(call.json))
```

## The client

`EmbeddingClient` (`rcp_ndcg.inference.clients`) applies the config's content decisions and sends the batches
through a transport (its own, built from the config, or one you pass as `sender`). The config declares the
content; the client applies it:

```python
from rcp_ndcg.inference.config import EmbeddingEndpoint

config = EmbeddingEndpoint(
    base_url="http://127.0.0.1:8000/v1",
    model="octen-embedding-8b",
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

* prepends `query_prompt` / `doc_prompt` per side, through `_prepare` -- the one seam the text-budget mechanism
  plugs into when it lands (until then a config that sets `max_tokens` is refused with a `ConfigError`, never
  silently ignored);
* slices the items into `batch_size`-sized requests and keeps at most `concurrency` in flight, reassembling in
  the input's order whatever order the replies arrive in;
* L2-normalises when `normalize`;
* resolves the API key from `api_key_env` (else the profile's own variables) and sends it in the profile's
  header, so the transport never adds a second one.

The vectors are raw float32 from the adapter -- the normalisation is the client's content decision, not the
wire's. `dimensions` changes what the server computes and is content; `base_url`, `batch_size`, `concurrency`
and the credentials change where and how fast, and are runtime.

## Identity

Two runs share an index only if they computed the same vectors. Which model, checkpoint and wire adapter
computed them (`api`, `model`, `revision`, `recipe`, the prompts, `normalize`, `dimensions`) is content and
enters the identity; where and how fast (`base_url`, `batch_size`, `concurrency`, the timeouts) is runtime and
never does. The tokenizer's name is runtime too: `EmbeddingEndpoint.identity_extra()` returns the SHA-256 of
its `tokenizer.json` (`{"tokenizer_sha256": ...}`), which is what a step identity carries instead of the name
-- the same rule the judge applies to its `tokenizer`.

## Text limits

A config that sets `max_tokens` is refused today: cutting to a token budget is the text-budget mechanism's job
(`rcp_ndcg.data.preprocess`), which will wire into this client and cut each text at token boundaries of the
declared `tokenizer`, never by engine-side truncation. Until then, send text that already fits -- a request an
engine cannot take fails with a `CapabilityError`, never a silent cut.
