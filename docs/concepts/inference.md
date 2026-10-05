# The inference layer: one transport for every role

The judge, the dense encoders, the multi-vector poolers and the rerankers all send HTTP requests to a served
model. They share one transport (`rcp_ndcg.inference`): the replica routing, the bounded concurrency, the
retries, the outage parking, the credentials, the usage and the provenance probe are written once, so every role
gets the judge's robustness. A *wire adapter* turns a role's request into HTTP calls and reads the replies back;
the transport does everything around that; a *role client* ties the two together for its API.

## The endpoint

`rcp_ndcg.inference.Endpoint` is the one home of the fields every role shares; a role's config subclasses it and
adds only the fields its wire protocol needs (the judge: `temperature`, `decoding`, the media limits; the
encoders: the prompts, `max_tokens`; [serving](serving.md) describes the judge's fields). Which model, which
checkpoint and which wire adapter decide what is computed, and do; where and how fast a model is asked are
runtime fields that never enter an identity.

| Field | Meaning |
|---|---|
| `api` | the wire adapter (below); each role's config sets its default |
| `base_url` | the endpoint (`.../v1`), one URL or a list of replica URLs of the same served model; `None` for a hosted API whose URL its profile resolves |
| `model` | the served model name, sent as the request's `model` |
| `revision` | the checkpoint commit the served weights resolved to; recorded in identities |
| `api_key_env` | the environment variable that holds the API key, sent as `Authorization: Bearer`; `None` sends no key. A missing variable is a `CredentialsError` naming it |
| `headers_env` | extra headers (e.g. `{"X-Gateway-Key": "GATEWAY_KEY"}`), each value read from its environment variable at send time, never from a config, and never logged |
| `concurrency` | requests in flight at once, over all replicas; the HTTP pool is sized to it |
| `timeout_s`, `connect_timeout_s` | the per-request and connect timeouts, seconds |
| `max_retries` | retries of a transient failure on the same replica before it counts as an outage |
| `wait_on_outage_s` | how long a request waits while every replica is down before `BackendUnavailableError`; `None` waits indefinitely |

## Sending: the adapter and the transport

An adapter turns one request into `Call` objects (method, path, JSON body, headers) and reads the `Reply`
objects (status, decoded body, headers) back into the role's result, raising the role's typed errors. The
transport routes the calls to one replica, retries what the status map calls unavailable, parks while every
replica is down, and counts the calls. A role client sends one request like this:

```python
from rcp_ndcg.inference import Call, TokenCount, Transport, register_adapter
from rcp_ndcg.inference.adapters.base import get_adapter
from rcp_ndcg.inference.config import EmbeddingEndpoint


@register_adapter
class DemoAdapter:
    """A minimal wire adapter, to show the pattern; the shipped ones live in rcp_ndcg.inference.adapters."""

    name = "demo"
    role = "embed"

    def calls(self, request, *, model):
        return [Call("POST", "/embeddings", {"model": model, "input": ["hello"]})]

    def interpret(self, request, replies):
        return [entry["embedding"] for entry in replies[0].body["data"]]

    def usage(self, reply):
        tokens = (reply.body or {}).get("usage") or {}
        return TokenCount(input_tokens=tokens.get("prompt_tokens")) if tokens else None


endpoint = EmbeddingEndpoint(base_url="fake://embed?dim=8", model="demo-encoder")
transport = Transport(endpoint)
adapter = get_adapter("demo", role="embed")()
replies = transport.run(transport.send(adapter.calls(None, model=endpoint.model)))
for reply in replies:
    transport.add_usage(adapter.usage(reply))
vectors = adapter.interpret(None, replies)
assert len(vectors[0]) == 8 and transport.usage.calls == 1
transport.aclose()
```

The snippet sends through the [offline fakes](#the-offline-fakes) (`fake://`), so it runs with no server.
Calling `send` without `run` awaits it directly (async code); `run` is the sync bridge, below.

### The status map

One table, in `rcp_ndcg.errors`, shared by every role's transport:

| Status or failure | What the transport does |
|---|---|
| connection error, timeout, HTTP 408, 429, 5xx | *unavailable*: retried up to `max_retries` with an exponential backoff (1 s doubling, capped at 60 s) or the server's `Retry-After`, then the replica is set aside; when every replica is set aside, the request parks until `wait_on_outage_s` passes, then `BackendUnavailableError` |
| HTTP 401, 403 | `CredentialsError` (the credentials were refused) |
| HTTP 404 | a non-retryable `ProviderError` naming the URL and the model (no such route or model) |
| every other 4xx (400, 413, 422, ...) | returned to the adapter as a `Reply`, for its `interpret` to raise its role-specific errors (`RequestRejectedError`, `CapabilityError`) |

Bodies are decoded as JSON; an `application/octet-stream` body stays `bytes`.

### Replicas and outages

Each request goes to the live replica with the fewest requests in flight from this transport, then the fewest
sent; the concurrency is shared over all replicas, and the HTTP pool is sized to it. A replica that fails is set
aside for a backoff that doubles while it keeps failing (from 5 to 60 seconds, the judge's numbers), and the
request moves at once to another live replica. A replica that answers again is used again. While every replica
is down, requests wait and are re-sent until one answers, or until `wait_on_outage_s` passes
(`BackendUnavailableError`, whose message states how long the endpoint was unavailable): a run against dead
servers parks instead of turning the outage into missing results. A request that keeps failing on a replica
that answers other requests is that request's failure: it is refused (`RequestRejectedError`).

## The sync bridge

The retrieval API is synchronous. `Transport.run(coroutine)` runs a coroutine to completion on a private event
loop the transport owns, reusing one event loop and one HTTP pool across calls; called while another loop is
running in the thread (a notebook), it runs on a private background thread instead of failing. `aclose()` (and
its `close()` twin, and the `with` block) closes the pool; a later `run` builds a fresh one.

## Usage

`transport.usage` counts the calls, the failed calls and the input and output tokens. The transport counts the
calls and the failed calls itself; the tokens cross the adapter, which is where the API's field names are
known: the role client calls `transport.add_usage(adapter.usage(reply))` once per reply.

## The provenance probe

`transport.probe()` asks each replica `GET {base_url}/models`, best effort, and records what it says into an
`EngineInfo` per replica: the served model id, `owned_by`, `max_model_len` when reported, and the `server` and
any version header. An endpoint that cannot be read is recorded with its `error`, never raised; a server that
does not list the endpoint's model is named in a warning. Nothing engine-specific is asked, and none of it
enters an identity.

## The offline fakes

A `fake://` base URL makes the transport send through an in-process `httpx.MockTransport` that speaks each
role's wire, deterministically (every draw is a hash of the endpoint's seed and the item's text):

| Route | Wire |
|---|---|
| `GET /models` | names the endpoint's model |
| `POST /chat/completions` | the judge's fake (registered by `rcp_ndcg.llm._fake`): it reads the documents out of the real rendered prompt and answers in the JSON the real parsers read, so `JudgeConfig.fake(seed)` runs a real client over the real transport |
| `POST /embeddings` | OpenAI shape; hash-seeded unit vectors, dimension from the URL's `?dim=` query (default 64), cut to the request's `dimensions` when it carries one |
| `POST /pooling` | vLLM `task: token_embed`; ragged per-token vectors, as floats or base64-packed in the request's `embed_dtype` (default `float16`) |
| `POST /rerank` | Cohere shape; each document scored by the same hidden ability the fake judge reads, so a tiny run's rerank and judge agree |

The fakes sit *below* the transport, so routing, retries, parking and usage run in every offline test. A seed
comes from the URL's numeric path tail (`fake://seed/3`); the vector dimension from its `?dim=` query. Extra
routes (a third party's, or another role's) register with `register_fake_route(method, path, handler)`.
