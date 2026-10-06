# `rcp_ndcg.inference`

One inference layer for every model the package calls: where a model is served, the wire types its requests
and answers travel in, the adapter that speaks its protocol, and one client per role. This page documents the
rerank role: its wire adapter, its hosted profiles and its client. The transport's behaviour (routing, retries,
parking, the credentials) is described on [the inference page](../concepts/inference.md); a role client used
without a sender builds its own transport, and a test or a third party can inject any `Sender` that provides
the sync bridge (`run`).

## The wire adapter: the Cohere-shaped rerank

Every engine and API of the rerank role speaks one request shape, `POST {base_url}/rerank` with `model`,
`query`, `documents` and `top_n` (the number of documents, so every document sent is scored). vLLM, Infinity
and Cohere answer `{"results": [{"index", "relevance_score"}]}`; Voyage answers `{"data": [...]}`; SGLang
answers a bare list of `{"index", "score"}` rows. One adapter family builds the request and parses all three
answer shapes, realigning every score to the request's documents by `index` -- the answers come back ranked,
so reading them positionally would silently permute the association between scores and documents.

A config's `api` field selects the wire by its registered name within the rerank role's registry (the same
registry the other roles use, scoped by role: the embed role registers its own `cohere` and `voyage`, and one
role's lookup never reaches another role's adapters):

| `api` | Wire | Cap per request | Notes |
|---|---|---|---|
| `rerank` (default) | a served engine's `POST {base_url}/rerank` | none | the engine's `instruction` and `use_activation` travel only when the config sets them |
| `cohere` | `https://api.cohere.com/v2/rerank` | 1000 documents | Cohere's recommendation, declared policy |
| `voyage` | `https://api.voyageai.com/v1/rerank` | 1000 documents | requests of one query spaced half a second apart; no `top_n` sent -- Voyage's return-limit field is `top_k`, and it returns every document by default |

A candidate set larger than the cap (or than a set `batch_size`) is split into requests of that many documents
and the chunks' scores are merged back into one aligned result; for a `listwise` config splitting is refused
with a `CapabilityError`, because a listwise model scores the whole candidate set in one prompt and a split
would change the scores. An unusable answer is refused, never repaired: a missing, duplicated or
out-of-range index raises a non-retryable `ProviderError` naming the server. An endpoint that refuses the
request as too long raises a `CapabilityError` whose hint names `max_tokens`; any other refusal is a
`RequestRejectedError`.

The pauses pace *one query's* requests. `rerank_many` still runs `concurrency` queries in flight, so an
`api: voyage` endpoint sees about `concurrency` requests every half second; the transport bounds requests, not
queries per second, so set `concurrency` low (2--4) for Voyage, which enforces strict rate limits.

## The client

`RerankClient(config)` sends one query's whole candidate set per request and reads the scores back aligned to
the documents, in the order they were given. It derives from the shared client base
(`rcp_ndcg.inference.clients.RoleClient`), which owns the adapter lookup within the rerank role's registry,
the hosted profiles' public roots, the transport (or an injected `Sender` with the sync bridge `run`), and the
lifecycle: `close()` synchronous, `await aclose()` asynchronous, both context managers. `rerank_many` runs
`concurrency` queries in flight under one `asyncio.TaskGroup`: a failing query cancels its siblings, and no
`checkpoint` lands after the failure.

- `rerank(query, documents, *, instruction=None) -> RerankResult` -- one query's scores (and `arerank`, the
  async half). Empty documents are sent as given and score whatever the server returns; an empty candidate
  set makes no request and scores nothing.
- `rerank_many(examples, *, checkpoint=None) -> list[RerankResult]` -- every example, `concurrency` queries
  in flight, results in input order. The `checkpoint` callable is called once per query as it lands, with the
  query id and its (pooled) scores aligned to the example's `doc_ids`: write the record and flush there, and a
  crash costs at most the queries in flight.
- `close()` / `await aclose()` -- closes the sender, when it closes (the client's own transport, or an
  injected one that defines `close`); safe to call twice.

The query text follows one rule for every path, decided by the config's `instruction` mode (`fold` by
default): `fold` sends `Task: <instruction>\nQuery: <text>` (the served path's render, byte for byte),
`field` sends the bare query plus the engine's `instruction` request field (served vLLM only -- a hosted
profile has no such field and refuses the mode), `none` sends the bare query. A served config (`api:
rerank`) sets `use_activation` explicitly (`true`: the score is a probability; `false`: the raw logit is
stored) -- `None` would send nothing and let the engine's default apply, and two engines with different
defaults would then share an identity; a hosted profile keeps it unset (its scale is fixed).

**Preparation.** Every input passes through one seam, `_prepare(contents, role)`, where the instruction mode
applies, and then the pair budget: when the config declares one, every request is fitted through the shared
text-budget mechanism (`rcp_ndcg.data.preprocess.fit`, the `pair` shape) -- the query cut to
`query_max_tokens` when it is set, each document cut to what remains, the template's fixed segments
re-attached so the anchors survive, every cut recorded in the census, and a chunked document sent as one
candidate-set row per chunk, scored in the query's request(s), with the chunks' scores pooled back by
`max`. The wire carries the cut spans (the engine
renders the template itself), and no `truncate_prompt_tokens` or `max_tokens_per_doc` is ever sent: the client
cut already, so there is nothing left for the engine to truncate. A config without `max_tokens` sends every
pair whole.

## Identity

A rerank endpoint keys on its `api`, `model` and `revision` (content); where and how fast it is asked
(`base_url`, `concurrency`, timeouts, `batch_size`) never enters an identity. `max_tokens` is content, and
with it the tokenizer's SHA-256: `identity_extra()` (inherited from `Endpoint`) returns `{"tokenizer_sha256": ...}` of the
named `tokenizer.json` (the name itself stays runtime), so two passes whose tokenizers differ never pool.
Every role config with a `tokenizer` -- the judge's, the embedding, pooling and rerank configs -- carries the
digest under this one key (`Endpoint.identity_extra()`), computed by the one helper in
`rcp_ndcg.data.tokenizer`; the judge's own identity payload keeps its existing keys and is unchanged.

```python
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers
from rcp_ndcg.inference import RerankClient, RerankEndpoint
from rcp_ndcg.inference.types import Call, Reply, Usage


# A served rerank config declares its text budget (the tokenizer the pair budget counts in, the cap) and
# sets use_activation explicitly (true: probability, false: raw logit). The tokenizer here is a tiny
# in-memory one, saved to disk so the
# client can load it; a recipe names the model's own.
backend = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
backend.pre_tokenizer = pre_tokenizers.Whitespace()
Path("tok").mkdir()
(Path("tok") / "tokenizer.json").write_text(backend.to_str())


class FakeRerankServer:
    """A test sender: one row per document, scored by the text, answered in arrival order shuffled."""

    async def send(self, calls: list[Call]) -> list[Reply]:
        replies = []
        for call in calls:
            documents = call.json["documents"]
            rows = [{"index": i, "relevance_score": ((len(d) * 7) % 10) / 10} for i, d in enumerate(documents)]
            replies.append(Reply(200, {"results": rows[::-1]}, {}))
        return replies

    async def probe(self) -> list[object]:
        return []

    @property
    def usage(self) -> Usage:
        return Usage()

    def run(self, coroutine):
        import asyncio

        return asyncio.run(coroutine)


config = RerankEndpoint(
    base_url="http://127.0.0.1:8000/v1",
    model="qwen3-reranker-8b",
    tokenizer="tok",
    max_tokens=8192,
    use_activation=False,
)
client = RerankClient(config, sender=FakeRerankServer())
result = client.rerank("what does rcp-ndcg measure", ["a metric", "a fruit"], instruction="Find the relevant passage")
print(result.scores)  # aligned to the input documents, whatever order the server answered in
client.close()
```


A served endpoint's config must declare its text budget (`tokenizer` and `max_tokens`: the package cuts
itself, never the engine) -- the client fits every request through `rcp_ndcg.data.preprocess.fit` and
records the cuts in the census; the engine never truncates.
