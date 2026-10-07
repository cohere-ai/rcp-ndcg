# The inference layer: one transport for every role

The judge, the dense encoders, the multi-vector poolers and the rerankers all send HTTP requests to a served
model. They share one transport (`rcp_ndcg.inference`): the replica routing, the bounded concurrency, the
retries, the outage parking, the credentials, the usage and the provenance probe are written once, so every role
gets the judge's robustness. A *wire adapter* turns a role's request into HTTP calls and reads the replies back;
the transport does everything around that; a *role client* ties the two together for its API.

## The endpoint

`rcp_ndcg.inference.Endpoint` is the one home of the fields every role shares; a role's config subclasses it and
adds only the fields its wire protocol needs (the judge: `temperature`, `decoding`, the media limits; the
encoders: the prompts, the text budget; [judges](judges.md) describes the judge's fields). Which model, which
checkpoint and which wire adapter decide what is computed, and do; where and how fast a model is asked are
runtime fields that never enter an identity.

| Field | Meaning |
|---|---|
| `api` | the wire adapter (below); each role's config sets its default |
| `base_url` | the endpoint (`.../v1`), one URL or a list of replica URLs of the same served model; `None` for a hosted API whose URL its profile resolves |
| `model` | the served model name, sent as the request's `model` |
| `revision` | the checkpoint commit the served weights resolved to; recorded in identities |
| `api_key_env` | the environment variable that holds the key; `None` (the default) uses the adapter profile's own variables -- but only when the request goes to the profile's own default host. Any other `base_url` receives a key only from an explicit `api_key_env` ([the credential rule](../reference/cli.md#credentials)); an unset variable a config named is a `CredentialsError` naming it |
| `headers_env` | extra headers (e.g. `{"X-Gateway-Key": "GATEWAY_KEY"}`), each value read from its environment variable at send time, never from a config, and never logged |
| `concurrency` | requests in flight at once, over all replicas; the HTTP pool is sized to it |
| `timeout_s`, `connect_timeout_s` | the per-request and connect timeouts, seconds |
| `max_retries` | retries of a transient failure on the same replica before it counts as an outage |
| `wait_on_outage_s` | how long a request waits while every replica is down before `BackendUnavailableError`; `None` waits indefinitely |

## Sending: the adapter and the transport

An adapter turns one request into `Call` objects (method, path, JSON body, headers) and reads the `Reply`
objects (status, decoded body, headers) back into the role's result, raising the role's typed errors. The
transport routes the calls to one replica, retries what the status map calls unavailable, parks while every
replica is down, and counts the calls. The shipped adapters register per role at import of
`rcp_ndcg.inference.adapters`: the judge's `openai_chat` (the `JudgeConfig.api` default), the embed role's
`openai_embeddings`, `cohere`, `voyage` and `gemini`, the rerank role's `rerank`, `cohere` and `voyage`, and the
multi-vector role's `vllm_pooling`. A role client sends one request like this:

```python
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers

from rcp_ndcg.inference import Call, TokenCount, Transport, register_adapter
from rcp_ndcg.inference.adapters.base import get_adapter
from rcp_ndcg.inference.config import EmbeddingEndpoint


@register_adapter
class DemoAdapter:
    """A minimal wire adapter, to show the pattern; the shipped ones live in rcp_ndcg.inference.adapters."""

    name = "demo"
    role = "embed"
    HOSTED = False
    API_KEY_ENV = ()
    KEY_REQUIRED = False
    AUTH_HEADER = None
    DEFAULT_BASE_URL = None

    def calls(self, request, *, model):
        return [Call("POST", "/embeddings", {"model": model, "input": ["hello"]})]

    def interpret(self, request, replies):
        return [entry["embedding"] for entry in replies[0].body["data"]]

    def usage(self, reply):
        tokens = (reply.body or {}).get("usage") or {}
        return TokenCount(input_tokens=tokens.get("prompt_tokens")) if tokens else None


# A self-hosted role config declares its text budget: the tokenizer the budget counts in, and the cap
# (the role clients cut the content spans themselves -- see the text budgets page).
backend = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
backend.pre_tokenizer = pre_tokenizers.Whitespace()
Path("tok").mkdir()
(Path("tok") / "tokenizer.json").write_text(backend.to_str())

endpoint = EmbeddingEndpoint(base_url="fake://embed?dim=8", model="demo-encoder", tokenizer="tok", max_tokens=8192)
transport = Transport(endpoint)
adapter = get_adapter("demo", role="embed")()
replies = transport.run(transport.send(adapter.calls(None, model=endpoint.model)))
for reply in replies:
    transport.add_usage(adapter.usage(reply))
vectors = adapter.interpret(None, replies)
assert len(vectors[0]) == 8 and transport.usage.requests == 1
transport.close()  # a sync caller; an async one awaits transport.aclose()
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

## Third-party adapters (C2)

Every role's registry accepts a third-party adapter, shipped in the `rcp_ndcg.adapters` entry-point group with
entries named `<role>.<name>` (e.g. `embed.bedrock`, one entry per role; a name may repeat across roles). A
config selects one with `api: <name>`. The seam's contract is declared: an adapter subclasses
`rcp_ndcg.inference.adapters.base.AdapterBase` (the credential facts with their declared defaults, and the
constructor convention -- built with the role config), registration refuses a class without the three members
(`calls`, `interpret`, `usage`) or the five credential facts -- `HOSTED` declared by the class itself, never
inherited from the base's `False` (what a wire is decides its published cap and its served-only refusals) --
and `rcp_ndcg.testing.adapter_contract` checks a
wire adapter as one listed failure set (a third party's test suite calls it):

* the retrieval roles resolve a non-shipped `api` against the role's registry where the config is read (the
  CLI's YAML loading, a run config, `validate_retriever`/`validate_reranker`): it builds the role's generic
  endpoint config, exposed from `rcp_ndcg.retrieval` as `PluginEmbedding`, `PluginPooling` and
  `PluginReranker` (the shipped names -- `openai_embeddings`, `cohere`, `voyage`, `gemini`, `vllm_pooling`,
  `rerank` -- keep selecting their own classes). A name that is not registered for that role -- unknown, or
  registered for another role -- is refused with the registry's hint, and the adapters the retrieval steps
  build (`EmbeddingClient`, `PoolingClient`, `RerankClient`) then run the third-party wire like a shipped one.
* the judge resolves its `api` lazily, at the first judging call (the refusal still precedes any wire
  traffic); its config validates any name.
* the adapter name is content: a step (and an index) identity keys on it, so two wires never share an index or
  a cache. Everything else about the identity (model, revision, recipe, prompts, budgets, the tokenizer's
  SHA-256) is unchanged.

## The sync bridge

The retrieval API is synchronous. `Transport.run(coroutine)` runs a coroutine to completion on a private event
loop the transport owns, reusing one event loop and one HTTP pool across calls; called while another loop is
running in the thread (a notebook), it runs on a private background thread instead of failing. The bridge is
one loop, one caller at a time: concurrent synchronous callers (a thread pool of retrievals sharing one
transport) queue on the bridge lock, and a `close()` from another thread waits for the in-flight call instead
of pulling its feet out from under it. `aclose()` is the awaitable close; `close()` (and the `with` block)
runs the same close synchronously, and a later `run` builds a fresh pool.

## The role clients

One client per role (the embedder, the reranker, the multi-vector pooler and the judge),
all derived from `rcp_ndcg.inference.clients.RoleClient`, which owns the shared ground once:

- **the adapter lookup within the client's role** -- `get_adapter(config.api, role=...)`, so a config whose
  `api` names another role's adapter is refused at construction;
- **the hosted profile's default base URL** -- the config's `base_url`, else the adapter's `DEFAULT_BASE_URL`
  (a hosted profile's public root); a config with neither is refused;
- **the transport** -- built from the resolved config unless the caller passes a `Sender` (the test fakes, a
  recording one, a third party's); an injected `Transport` is pointed at the adapter profile's credential
  facts, so the key resolution follows the adapter either way;
- **the sync bridge, one rule** -- the sender's `run`: a `Transport` always has it, and any other sender must
  provide `run`, or the constructor raises `ConfigError` (no `asyncio.run` fallback: a fresh loop per call
  would give a non-transport sender no pool reuse and would fail inside a running loop);
- **the lifecycle** -- `close()` synchronous, `await aclose()` asynchronous, and both context managers
  (`with` and `async with`); safe to call twice;
- **the fan-out, one rule** -- every client's requests run in one `asyncio.TaskGroup`: a failing request
  cancels its siblings, no callback (a rerank `checkpoint`) lands after the failure, and no task is left
  pending; a group carrying exactly one failure is raised as that failure, so the typed errors surface;
- **the text budget and the media** -- the `TextBudget` resolved once from the role config's fields, the
  tokenizer it names loaded once, and the shared `rcp_ndcg.data.preprocess.fit` called from each client's
  `_prepare` (see [text budgets](text-budgets.md)); a census of every cut is
  at `client.census`. Media preparation runs through the judge's own path (`prepare_request`) -- sized exactly as
  the role's `image_policy`/`image_processor` would -- with the tokens counted and reserved whole (never
  cut), the budget's fit applied per wire request (a vision block is atomic -- shrink to the policy minimum,
  drop whole items with `dropped=True` in `client.media_census`, or refuse under `fail`/`chunk`), and the
  `max_images`/`max_videos` gates over what one wire request carries. Which SIDES may carry media is the
  config's `media_sides` (both by default): media on a side it does not name is refused before preparation,
  with the error naming the field (the topk reference rejects image queries -- media is documents-only
  there). The pool and rerank clients' startup `probe()` runs `check_engine_media()` when the role declares
  an `image_processor` (the embed client's `probe()` is the transport's replica probe only): the media check
  sends one prepared probe image AND the same request without its media, and the DELTA of the engine's two
  prompt-token reports -- the template and the text cancel -- is compared with the counted media tokens. A
  mismatch is refused, a reply without usage recorded `not_checked`, never silent: a served chat template
  does not fail a correct engine, because it cancels in the delta.

The two role vocabularies meet in one written mapping, `ENGINE_ADAPTER_ROLES`
(`rcp_ndcg.inference.adapters.base`): a `judge` engine speaks `judge` adapters, an `encoder` engine `embed` or
`multi_vector` ones, a `reranker` engine `rerank` ones. Where the runners resolve an engine onto a config (the
engine overlay), `check_engine_api` refuses a config whose `api` selects an adapter of a different engine
role.

## Usage

`transport.usage` counts the requests, the failed requests and the input and output tokens, in the run
manifest's `Usage` shape (one type for every role). The transport counts the requests and the failed requests
itself; the tokens cross the adapter, which is where the API's field names are
known: the role client calls `transport.add_usage(adapter.usage(reply))` once per reply -- the embed, the
pool and the rerank clients all do, at send time (whatever way the calls go out, including a paused
profile's per-call loop), and `client.usage` reads the same accounting.

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
comes from the URL's numeric path tail (`fake://seed/3`); the vector dimension from its `?dim=` query. An item's
token count -- `/pooling`'s vectors per item and its `prompt_token_ids`, and both routes' `usage` -- follows the
request's tokenization as an engine's would: a token-ids input is its ids, and a text is counted in the
tokenizer the endpoint's config declares (`tokenizer`), under the request's `add_special_tokens` (default true).
Without a declared tokenizer, or for a chat conversation (whose template the fake does not render), the fallback
counts whitespace words. Extra routes (a third party's, or another role's) register with
`register_fake_route(method, path, handler)`.

A URL whose host names an engine and its version (`fake://vllm-0.31.0/<recipe>`) selects a **verified
fake engine** instead of these hash-seeded fakes: an emulator of a recorded observation corpus
(protocol emulated with the recipe's real tokenizer, model outputs replayed for observed inputs and
marked `replayed`/`surrogate` in every reply). See
[Use the verified fake engines](../how-to/use-verified-fake-engines.md).
