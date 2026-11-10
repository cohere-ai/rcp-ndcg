# The inference layer: one transport for every role

The judge, the dense encoders, the multi-vector poolers and the rerankers all send HTTP requests to a served
model. They share one transport (`rcp_ndcg.inference`). The replica routing, the bounded concurrency, the
retries and the outage parking are written once. So are the credentials, the usage and the provenance probe.
Every role therefore gets the judge's robustness. A *wire adapter* turns a role's request into HTTP calls, and
it reads the replies back. The transport does everything around that. A *role client* ties the two together
for its API.

## The endpoint

`rcp_ndcg.inference.Endpoint` is the one home of the fields that every role shares. A role's config subclasses
it, and it adds only the fields that its wire protocol needs. The judge adds `temperature`, `decoding` and
the media limits, and the encoders add the prompts and the text budget. [Judges](judges.md) describes the
judge's fields. The model, the checkpoint and the wire adapter decide what is computed, and they enter the
identity. Where and how fast a model is asked are runtime fields, and they never enter an identity.

| Field | Meaning |
|---|---|
| `api` | the wire adapter (below); each role's config sets its default |
| `base_url` | the endpoint (`.../v1`), one URL or a list of replica URLs of the same served model; the scheme must be `http://`, `https://` or `fake://` (a schemeless or mistyped URL is a typed `ConfigError`, not an outage); `None` for a hosted API whose URL its profile resolves |
| `model` | the served model name, sent as the request's `model` |
| `revision` | the checkpoint commit the served weights resolved to; recorded in identities |
| `api_key_env` | the environment variable that holds the key; `None` (the default) uses the adapter profile's own variables -- but only when the request goes to the profile's own default host. Any other `base_url` receives a key only from an explicit `api_key_env` ([the credential rule](../reference/cli.md#credentials)); an unset variable a config named is a `CredentialsError` naming it |
| `headers_env` | extra headers (e.g. `{"X-Gateway-Key": "GATEWAY_KEY"}`), each value read from its environment variable at send time, never from a config, and never logged |
| `concurrency` | requests in flight at once, over all replicas; the HTTP pool is sized to it |
| `timeout_s`, `connect_timeout_s` | the per-request and connect timeouts, seconds |
| `max_retries` | retries of a transient failure on the same replica before it counts as an outage |
| `wait_on_outage_s` | how long a request waits while every replica is down before `BackendUnavailableError`; the default is 1800 s (an engine restart plus a large model's load), and `None` waits indefinitely |

## Sending: the adapter and the transport

An adapter turns one request into `Call` objects: method, path, JSON body and headers. It reads the `Reply`
objects back into the role's result: status, decoded body and headers. It raises the role's typed errors. The
transport routes the calls to one replica, and it retries what the status map calls unavailable. It parks
while every replica is down, and it counts the calls. The shipped adapters register per role at import of
`rcp_ndcg.inference.adapters`. They are the judge's `openai_chat` (the `JudgeConfig.api` default), and the
embed role's `openai_embeddings`, `cohere`, `voyage` and `gemini`. The rerank role has `rerank`, `cohere`
and `voyage`, and the multi-vector role has `vllm_pooling`.

The rerank role's adapters speak one Cohere-shaped wire: `POST {base_url}/rerank` with `model`, `query` and
`documents`. The served-engine and Cohere wires set `top_n` to the number of documents sent, so every
document sent is scored. Voyage's return-limit field is `top_k`, and Voyage returns every document by
default, so no such field is sent there. vLLM, Infinity and Cohere answer
`{"results": [{"index", "relevance_score"}]}`, and Voyage answers `{"data": [...]}`. One adapter family
builds the request and parses both answer shapes. It realigns every score to its document by `index`, because
the answers come back ranked. Reading them positionally would silently permute scores and documents. An answer
that is a bare list of `{"index", "score"}` rows is refused by name. The hint says to serve the model on vLLM.
SGLang and TEI answer that shape. A missing, duplicated or out-of-range index is a non-retryable error naming
the server. A candidate set can be larger than the wire's declared cap, or than a set `batch_size`, whichever
is smaller. It is then split into requests, and the chunks' scores are merged back into one aligned result. A
`listwise` config refuses the split. It scores the whole candidate set in one prompt, and a split would change
the scores. A role client sends one request like this:

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
Calling `send` without `run` awaits it directly, in async code. `run` is the sync bridge, below.

### The status map

One table, in `rcp_ndcg.errors`, shared by every role's transport:

| Status or failure | What the transport does |
|---|---|
| connection error, timeout, HTTP 408, 429, 5xx | *unavailable*: retried up to `max_retries` with an exponential backoff (1 s doubling, capped at 60 s) or the server's `Retry-After`, then the replica is set aside; when every replica is set aside, the request parks until `wait_on_outage_s` passes (1800 s by default; `None` waits indefinitely), then `BackendUnavailableError` |
| HTTP 401, 403 | `CredentialsError` (the credentials were refused) |
| HTTP 404 | a non-retryable `ProviderError` naming the URL and the model (no such route or model) |
| every other 4xx (400, 413, 422, ...) | returned to the adapter as a `Reply`, for its `interpret` to raise its role-specific errors (`RequestRejectedError`, `CapabilityError`) |

Bodies are decoded as JSON; an `application/octet-stream` body stays `bytes`.

### Replicas and outages

Each request goes to the live replica with the fewest requests in flight from this transport. Ties go to the
replica with the fewest requests sent. The concurrency is shared over all replicas, and the HTTP pool is sized
to it. A replica that fails is set aside for a backoff. The backoff doubles while it keeps failing, from 5 to
60 seconds; those are the judge's numbers. Its request moves at once to another live replica. A replica that
answers again is used again. While every replica is down, requests wait and are re-sent until one answers, or
until `wait_on_outage_s` passes. That raises `BackendUnavailableError`, whose message states how long the
endpoint was unavailable. A run against dead servers therefore parks, instead of turning the outage into
missing results. The default wait is 1800 s: an engine restart plus a large model's load.
`wait_on_outage_s: null` is the explicit "wait indefinitely" choice. A run step's `step_budget_s` bounds the
step on top. The transport checks it before each request and after every park. A step over budget therefore
stops with `StepBudgetExceededError`, instead of working through a long outage ([runs](runs.md)). A request
that keeps failing on a replica that answers other requests is that request's failure. It is refused
(`RequestRejectedError`). A 408 or 429 is never that. It is the endpoint's load, and it stays on the outage
path. The replica is set aside for the server's `Retry-After` when it gives a usable one. A usable value is a
finite, positive number of seconds. A zero, a negative or an unparseable header is no wait at all, so the
doubling backoff applies instead. The backoff is jittered, so parked requests do not all return at once. The
request parks and retries, instead of a rate-limited window aborting a whole encode.

## Third-party adapters (C2)

Every role's registry accepts a third-party adapter. Such an adapter ships in the `rcp_ndcg.adapters`
entry-point group, with entries named `<role>.<name>`. One example is `embed.bedrock`, with one entry per
role. A name may repeat across roles. A config selects one with `api: <name>`. The seam's contract is
declared. An adapter subclasses `rcp_ndcg.inference.adapters.base.AdapterBase`. That base holds the
credential facts with their declared defaults. It also holds the constructor convention, which builds the
adapter with the role config. Registration refuses a class without the three members `calls`, `interpret` and
`usage`, or without the five credential facts. `HOSTED` is declared by the class itself, never inherited from
the base's `False`. What a wire is decides its published cap and its served-only refusals.
`rcp_ndcg.testing.adapter_contract` checks a wire adapter as one listed failure set, and a third party's test
suite calls it:

* the retrieval roles resolve a non-shipped `api` against the role's registry where the config is read. That is the
  CLI's YAML loading, a run config, or `validate_retriever`/`validate_reranker`. The lookup builds the role's generic
  endpoint config, exposed from `rcp_ndcg.retrieval` as `PluginEmbedding`, `PluginPooling` and `PluginReranker`. The
  shipped names keep selecting their own classes: `openai_embeddings`, `cohere`, `voyage`, `gemini`, `vllm_pooling`
  and `rerank`. A name that is not registered for that role is refused with the registry's hint. The name can be
  unknown, or registered for another role. The adapters that the retrieval steps build (`EmbeddingClient`,
  `PoolingClient`, `RerankClient`) then run the third-party wire like a shipped one.
* the judge resolves its `api` lazily, at the first judging call. The refusal still precedes any wire traffic, and
  its config validates any name.
* the adapter name is content. A step (and an index) identity keys on it, so two wires never share an index or a
  cache. Everything else about the identity is unchanged: model, revision, recipe, prompts, budgets, the tokenizer's
  SHA-256.

## The sync bridge

The retrieval API is synchronous. `Transport.run(coroutine)` runs a coroutine to completion on a private event
loop that the transport owns. It reuses one event loop and one HTTP pool across calls. A call can come from a
thread that already runs another loop, such as a notebook. It runs on a private background thread instead of
failing. The bridge is one loop with one caller at a time. Concurrent synchronous callers queue on the bridge
lock; one example is a thread pool of retrievals that share one transport. A `close()` from another thread
waits for the in-flight call instead of interrupting it. `aclose()` is the awaitable close. `close()`, and
the `with` block, runs the same close synchronously, and a later `run` builds a fresh pool.

## The role clients

There is one client per role: the embedder, the reranker, the multi-vector pooler and the judge. All derive
from `rcp_ndcg.inference.clients.RoleClient`, which owns the shared ground once:

- **the adapter lookup within the client's role**: `get_adapter(config.api, role=...)`, so a config whose
  `api` names another role's adapter is refused at construction;
- **the hosted profile's default base URL**: the config's `base_url`, else the adapter's `DEFAULT_BASE_URL`
  (a hosted profile's public root); a config with neither is refused;
- **the transport**: built from the resolved config unless the caller passes a `Sender`. Examples are the test fakes, a
  recording one or a third party's. An injected `Transport` is pointed at the adapter profile's credential facts, so the
  key resolution follows the adapter either way;
- **the sync bridge, one rule**: the sender's `run`. A `Transport` always has it, and any other sender must
  provide `run`, or the constructor raises `ConfigError`. There is no `asyncio.run` fallback, because a fresh loop
  per call would give a non-transport sender no pool reuse and would fail inside a running loop;
- **the lifecycle**: `close()` synchronous, `await aclose()` asynchronous, and both context managers
  (`with` and `async with`); safe to call twice;
- **the fan-out, one rule**: every client's requests run in one `asyncio.TaskGroup`. A failing request cancels its
  siblings. No callback (a rerank `checkpoint`) lands after the failure, and no task is left pending. A group carrying
  exactly one failure is raised as that failure, so the typed errors surface;
- **the text budget and the media**: the `TextBudget` is resolved once from the role config's fields. The tokenizer it
  names is loaded once. The shared fit is called from the one declared preparation pipeline. The pipeline is `STAGES`:
  normalise, empty, media, render, budget, lower. One order serves every role, with the per-row record at
  `client.processing` (see [text budgets](text-budgets.md)). A census of every cut is at `client.census`. Media
  preparation runs through the judge's own path (`prepare_request`), sized exactly as the role's
  `image_policy`/`image_processor` would. The tokens are counted and reserved whole, never cut. The budget's fit is
  applied per wire request, because a vision block is atomic. It shrinks to the policy minimum, drops whole items with
  `dropped=True` in `client.media_census`, or refuses under `fail`/`chunk`. The `max_images`/`max_videos` gates apply to
  what one wire request carries. The client that packs a batch splits a media-bearing one at them. It sends one image
  per request where the endpoint declares `max_images: 1`, and a text-only batch packs by `batch_size` as before. Which
  SIDES may carry media is the config's `media_sides`, both by default. Media on a side that it does not name is refused
  before preparation, with the error naming the field. The topk reference rejects image queries, because media is
  documents-only there. The pool and rerank clients' startup `probe()` runs `check_engine_media()` when the role
  declares an `image_processor`. The embed client's `probe()` is the transport's replica probe only. The media check
  sends one prepared probe image and the same request without its media. The DELTA of the engine's two prompt-token
  reports is compared with the counted media tokens. The template and the text cancel in that delta. A mismatch is
  refused. A passing check is recorded `engine_media_check:ok`, and a reply without usage is recorded `not_checked`,
  never silent. A served chat template does not fail a correct engine, because it cancels in the delta.

The two role vocabularies meet in one written mapping, `ENGINE_ADAPTER_ROLES`
(`rcp_ndcg.inference.adapters.base`). A `judge` engine speaks `judge` adapters. An `encoder` engine speaks
`embed` or `multi_vector` ones, and a `reranker` engine speaks `rerank` ones. The runners resolve an engine
onto a config, in the engine overlay. There, `check_engine_api` refuses a config whose `api` selects an
adapter of a different engine role.

## Usage

`transport.usage` counts the requests, the failed requests and the input and output tokens, in the run
manifest's `Usage` shape. That shape is one type for every role. The transport counts the requests and the
failed requests itself. The tokens cross the adapter, because the adapter knows the API's field names. The
role client calls `transport.add_usage(adapter.usage(reply))` once per reply, at send time. The embed, the
pool and the rerank clients all do. That holds whatever way the calls go out, including a paused profile's
per-call loop. `client.usage` reads the same accounting.

## The provenance probe

`transport.probe()` asks each replica `GET {base_url}/models`, as a best effort. It records the answer into
an `EngineInfo` per replica. That covers the served model id, `owned_by`, `max_model_len` when reported, and
the `server` and any version header. An endpoint that cannot be read is recorded with its `error`, never
raised. A server that does not list the endpoint's model is named in a warning. Nothing engine-specific is
asked, and none of it enters an identity.

## The offline fakes

A `fake://` base URL makes the transport send through an in-process `httpx.MockTransport`. It speaks each
role's wire, deterministically: every draw is a hash of the endpoint's seed and the item's text.

| Route | Wire |
|---|---|
| `GET /models` | names the endpoint's model |
| `POST /chat/completions` | the judge's fake (registered by `rcp_ndcg.judging._fake`): it reads the documents out of the real rendered prompt and answers in the JSON the real parsers read, so `JudgeConfig.fake(seed)` runs a real client over the real transport |
| `POST /embeddings` | OpenAI shape; hash-seeded unit vectors, dimension from the URL's `?dim=` query (default 64), cut to the request's `dimensions` when it carries one |
| `POST /pooling` | vLLM `task: token_embed`; ragged per-token vectors, as floats or base64-packed in the request's `embed_dtype` (default `float16`) |
| `POST /rerank` | Cohere shape; each document scored by the same hidden ability the fake judge reads, so a tiny run's rerank and judge agree |

The fakes sit *below* the transport, so routing, retries, parking and usage run in every offline test. A seed
comes from the URL's numeric path tail (`fake://seed/3`), and the vector dimension comes from its `?dim=`
query. An item's token count follows the request's tokenization, as an engine's would. That covers `/pooling`
's vectors per item and its `prompt_token_ids`, and both routes' `usage`. A token-ids input is its ids. A
text is counted in the tokenizer that the endpoint's config declares (`tokenizer`), under the request's
`add_special_tokens` (default true). The fallback counts whitespace words. It applies without a declared
tokenizer, or for a chat conversation whose template the fake does not render. Extra routes, a third party's
or another role's, register with `register_fake_route(method, path, handler)`.

A URL whose host names an engine and its version (`fake://vllm-0.33.0/<recipe>`) selects a **verified fake
engine**. It does not select these hash-seeded fakes. It is an emulator of a recorded observation corpus. The
protocol is emulated with the recipe's real tokenizer, and model outputs are replayed for observed inputs.
Every reply marks them `replayed` or `surrogate`. See [Use the verified fake engines](../how-to/use-verified-fake-engines.md).
