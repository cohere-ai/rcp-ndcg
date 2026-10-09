# Matryoshka heads: output dimensions, the full-width store and the sweep

A Matryoshka Representation Learning (MRL) model ships a family of usable output widths: one forward pass
computes a full-width vector, and a smaller width is derived from it. RCP-nDCG makes the derivation a
first-class, declared decision, so a run never guesses a width and a sweep over widths never pays for more
than one forward pass.

## Two kinds of head

A checkpoint's smaller sizes come from one of exactly two mechanisms, and the recipe declares which:

* **truncation** -- the checkpoint was trained so that the first `k` components of the full-width vector are
  a valid `k`-dimensional embedding (the Matryoshka property). The head slices the vector to `k` and
  renormalises it. The slice is taken before the L2 normalisation of the cut: cutting a normalised vector
  and not renormalising would ship non-unit vectors, so the head always renormalises (`rcp_ndcg.data.mrl.mrl_cut`).
* **projection** -- the checkpoint's smaller sizes are computed by its own learned matrices (a
  `*.safetensors` file at the pinned revision), so a slice of the full-width vector is *not* the right
  input. The head loads the file (through `rcp_ndcg.storage`, so a Hub path
  `hf://org/model@revision/projections.safetensors` or a local file both work), applies the chain of
  matrices and renormalises the result. One naming convention: **the file's tensor names are their target
  widths**, and the declared `mrl_dims` are the projected sizes (the full width is served without a head
  and is not declared). To reach `k`, the head applies the tensors named for every declared dimension at
  or above `k`, widest first -- for `mrl_dims: (1280, 640, 320)` and `k=640`, the `1280` tensor
  (full width -> 1280) then the `640` tensor (1280 -> 640). The chain computes in float32 and returns
  float32, even over a float16 store (the learned matrices are F32); a truncation cut keeps a float16
  input float16 and returns float32 otherwise (the normalisation computes in float32).

A model without an MRL head declares none, and nothing is ever cut. The kind and the card-supported output
dimensions are declared once, on the endpoint config (a recipe's client block): `mrl_kind` is `truncation`
or `projection`, and the declaration is either `mrl_dims` (the card's discrete table) or `mrl_range` (the
card's prose range, e.g. `[32, 1024]`; every `k` in the closed interval is selectable, and the client
enforces the floor). A run selects `k` from that declaration, and anything else is refused at load:

```python
from rcp_ndcg.inference.config import EmbeddingEndpoint

config = EmbeddingEndpoint(
    base_url="http://127.0.0.1:8000/v1",
    model="some-embedding-model",
    tokenizer="org/model@revision",
    max_tokens=8192,
    mrl_kind="truncation",
    mrl_dims=(64, 128, 256, 512),  # or mrl_range=(32, 1024) for a card whose prose gives a range
    mrl_dim=128,  # the run's selection, inside the declaration
)
print(config.mrl_kind, config.mrl_dim)
```

The selection is refused when it is outside `mrl_dims`/`mrl_range`, when the kind is undeclared, when both
`mrl_dims` and `mrl_range` are declared, and when a projection
kind has no `mrl_projection` source; `dimensions` (the engine-side cut) is refused beside `mrl_dim`, is
allowed only for the truncation kind, and only on the dense `/embeddings` wire -- `/pooling` has no
per-request `dimensions` field, so the pooling route is always cut client-side. Each refusal names the
field and the fix.

## The order

A client applies the declared kind to the reply through the one head home (`rcp_ndcg.data.mrl.MrlHead`):

1. the declared `normalize` (the full-width embedding is L2-normalised when the config says so);
2. the head: truncation slices then renormalises, projection applies the learned chain then renormalises.

Renormalising before or after the cut gives the same direction (the head renormalises the cut, and the
learned projection is linear), and this order is what makes the ex-post sweep over a stored full-width
vector bit-identical to a direct run at the same `k`. Every row the head changed carries a
`ProcessingRecord` with the mechanism `mrl_cut` and the kind, the selected `k` and the full width -- a run
with `mrl_dim` is recorded, never silently cut.

The engine-side `dimensions` path is different: the request carries `dimensions: k`, the engine slices the
raw output before its own normalisation (vLLM's order), and the reply arrives already `k`-wide. The two
selections are mutually exclusive on one config; one cut, one home.

## The full-width store

Evaluating several `k` values from one forward pass needs the full-width vectors, not a cut. `rcp-ndcg
retrieval store` (`rcp_ndcg.retrieval.build_store`) encodes a dataset's corpus **and** its queries once at
the checkpoint's full width (the configured selection is stripped for that pass) and writes an
`EmbeddingStore`:

```
<store>/store.json            the typed record: schema, provenance, ids
<store>/corpus.npy            (count, full_width) or the flat ragged buffer
<store>/corpus_offsets.npy    (count + 1,) for a late-interaction store
<store>/queries.npy           (query_count, full_width)
<store>/queries_offsets.npy   (query_count + 1,) for a late-interaction store
```

The record carries the schema (`rcp-ndcg.embedding-store.v1`), the model, its revision, the recipe, the
digest of the request-shaping fields (prompts, template, budget, shapes), the tokenizer and its digest, the
text budget, the full width, the stored dtype, the declared MRL kind, set and projection source, and the
corpus and query ids in stored order. Its identity is the retrieval identity of the full-width encoder plus
a full-width marker, so a cut store can never be mistaken for a full-width one; the reader refuses vectors
narrower than the record's `full_width`.

`k` never reaches the engine request and never enters the store's identity: the store is the same for every
`k`, and the selection enters only the identity of the per-`k` artifacts it derives.

## The sweep

`rcp-ndcg retrieval sweep` (`rcp_ndcg.retrieval.sweep`) applies the declared head per `k` to the stored
vectors, scores each cut with the retrieval scorer (dense inner product or MaxSim), and writes one rankings
file per `k` whose system is `<model>@<k>`. With `--dataset`, it also runs `evaluate` and `compare` across
the `k` systems, so one store yields the whole width/quality curve:

```
rcp-ndcg retrieval store  --dataset beir:data/nfcorpus --retriever encoder.yaml --out store/
rcp-ndcg retrieval sweep  --store store/ --dims 64 --dims 128 --dims 256 \
                          --dataset beir:data/nfcorpus --out-dir sweep/
```

`--dims` selects a subset of the declared set (the default is every declared dimension; a store that
declares only `mrl_range` needs explicit `--dims`, because a range cannot be enumerated). Each `k` costs one
head application and one scoring pass; no model is called again. A per-`k` run through the ordinary
`retrieval index`/`search` commands computes the same vectors and the same scores -- the sweep is the same
arithmetic, from one store.

## Gating every declared k

The equivalence harness (`rcp-ndcg-test`) gates every declared dimension from one full-width run: stage 2
builds the recipe's client with its selection stripped (`dimensions`/`mrl_dim`), so the served engine and the
reference subprocess both answer at full width, and then applies the declared head to both sides per `k`
before the ordinary per-vector (or per-token) cosine gate -- one gate row per `k` in `equivalence.json` and
the report, beside the full-width row. `k` never enters the engine request: the one pass is full width, and
the head derives every `k` ex-post, so the head's derivation semantics are versioned by the gating code
(`rcp_ndcg_test.equivalence.MRL_GATE_VERSION`). A recipe that declares only `mrl_range` gates its two
endpoints and the run's selection: a range cannot be enumerated, and the interior is not silently claimed.
The observation request set records the engine's own side of each declared `k` (a bare `dimensions=k` probe,
one per declared `k`) so the fake engines replay it.
