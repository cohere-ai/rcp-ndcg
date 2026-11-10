# Matryoshka heads: output dimensions, the full-width store and the sweep

A Matryoshka Representation Learning (MRL) model ships a family of usable output widths. One forward pass
computes a full-width vector, and a smaller width is derived from it. RCP-nDCG makes the derivation a
first-class, declared decision. A run therefore never guesses a width, and a sweep over widths pays for only
one forward pass.

## Two kinds of head

A checkpoint's smaller sizes come from one of exactly two mechanisms. The recipe declares which:

* **truncation**: the checkpoint was trained so that the first `k` components of the full-width vector are a valid
  `k`-dimensional embedding. This is the Matryoshka property. The head slices the vector to `k` and renormalises
  it. The slice is taken before the L2 normalisation of the cut. Cutting a normalised vector and not renormalising
  would ship non-unit vectors, so the head always renormalises (`rcp_ndcg.data.mrl.mrl_cut`).
* **projection**: the checkpoint's smaller sizes are computed by its own learned matrices. Those matrices are a
  `*.safetensors` file at the pinned revision. A slice of the full-width vector is therefore *not* the right
  input. The head loads the file through `rcp_ndcg.storage`, so a Hub path
  `hf://org/model@revision/projections.safetensors` or a local file both work. It applies the chain of matrices and
  renormalises the result. One naming convention applies: **the file's tensor names are their target widths**. The
  declared `mrl_dims` are the projected sizes. The full width is served without a head, and it is not declared. To
  reach `k`, the head applies the tensors named for every declared dimension at or above `k`, widest first. For
  `mrl_dims: (1280, 640, 320)` and `k=640`, it applies the `1280` tensor (full width -> 1280), then the `640`
  tensor (1280 -> 640). The chain computes in float32 and returns float32, even over a float16 store, because the
  learned matrices are F32. A truncation cut keeps a float16 input float16, and it returns float32 otherwise,
  because the normalisation computes in float32.

A model without an MRL head declares none, and nothing is ever cut. The kind and the card-supported output
dimensions are declared once, on the endpoint config, which is a recipe's client block. `mrl_kind` is
`truncation` or `projection`. The declaration is either `mrl_dims`, the card's discrete table, or
`mrl_range`, the card's prose range, e.g. `[32, 1024]`. Every `k` in the closed interval is selectable, and
the client enforces the floor. A run selects `k` from that declaration. Anything else is refused at load:

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

The selection is refused in four cases. It is outside `mrl_dims` /`mrl_range`. The kind is undeclared. Both
`mrl_dims` and `mrl_range` are declared. A projection kind has no `mrl_projection` source. `dimensions`, the
engine-side cut, is refused beside `mrl_dim`. It is allowed only for the truncation kind, and only on the
dense `/embeddings` wire. `/pooling` has no per-request `dimensions` field, so the pooling route is always cut
client-side. Each refusal names the field and the fix.

## The recipes' declarations

Every shipped embedding and multi-vector variant declares its head once in its recipe (`family.yaml`). It
declares `mrl_kind` (`truncation`, `projection` or `none`). It declares the model card's set with it:
`mrl_dims` for a discrete table, `mrl_range` for a prose range. A projection kind also declares the
`mrl_projection` source. The recipes ship the checkpoint's full width, and a run selects `k` from the
declaration. A `recipe: <id>` config may set `mrl_dim` (the client head) or `dimensions` (the engine-side
cut). A `k` inside the declared set is a selection, not a content disagreement. A `k` outside it is refused,
and the refusal names the set.

Some cards support a cut, and the engine serves it. For those, the recipe also declares the engine's own gate
in `serve.hf_overrides`. It sets `is_matryoshka: true` for a card whose prose gives a range. It sets
`is_matryoshka: true` with `matryoshka_dimensions` for a card whose table gives a discrete set. vLLM accepts a
per-request `dimensions` only when the checkpoint's config carries one of those two keys. No card declares
either, so the recipe adds it. The loader enforces one rule between the two blocks. A discrete engine list and
the client's `mrl_dims` are the same set. An open gate still needs the client's `mrl_dims` /`mrl_range`. A
serve gate beside a projection kind is refused, because the engine can only slice. Some cards the engine
cannot serve a per-request cut for. One is a checkpoint with no served class. Another is one whose smaller
sizes are learned matrices that the engine never loads (zembed). A third is a multi-vector card whose
`/pooling` route refuses the per-request `dimensions` (topk, pplx-context). Such a card declares its head
without the serve gate, and the client-side head is the one cut.

## The order

A client applies the declared kind to the reply through the one head home (`rcp_ndcg.data.mrl.MrlHead`):

1. the declared `normalize` (the full-width embedding is L2-normalised when the config says so);
2. the head: truncation slices then renormalises, projection applies the learned chain then renormalises.

Renormalising before or after the cut gives the same direction. The head renormalises the cut, and the learned
projection is linear. This order makes the ex-post sweep over a stored full-width vector bit-identical to a
direct run at the same `k`. Every row that the head changed carries a `ProcessingRecord`. The record holds
the mechanism `mrl_cut`, the kind, the selected `k` and the full width. A run with `mrl_dim` is therefore
recorded, never silently cut.

A `k` equal to the checkpoint's own width is the **identity selection**. No head is applied, and no `mrl_cut`
record is written. A card's full-width member therefore stays selectable: a truncation table whose last member
is the width, e.g. topk's 2048, or a range whose ceiling is the width. The selection still enters the config's
identity, because it is a declared CONTENT field. The ex-post sweep's `k == full_width` artifact is the stored
full-width vectors themselves. A `k` wider than the vectors is still refused, on every route.

The engine-side `dimensions` path is different. The request carries `dimensions: k`, and the engine slices
the raw output before its own normalisation, which is vLLM's order. The reply arrives already `k` -wide. The
two selections are mutually exclusive on one config. One cut, one home.

## The full-width store

Evaluating several `k` values from one forward pass needs the full-width vectors, not a cut.
`rcp-ndcg retrieval store` (`rcp_ndcg.retrieval.build_store`) encodes a dataset's corpus **and** its queries
once, at the checkpoint's full width. The configured selection is stripped for that pass. It writes an
`EmbeddingStore`:

```
<store>/store.json            the typed record: schema, provenance, ids
<store>/corpus.npy            (count, full_width) or the flat ragged buffer
<store>/corpus_offsets.npy    (count + 1,) for a late-interaction store
<store>/queries.npy           (query_count, full_width)
<store>/queries_offsets.npy   (query_count + 1,) for a late-interaction store
```

The record carries the schema (`rcp-ndcg.embedding-store.v1`), the model and its revision. It carries the
recipe and the digest of the request-shaping fields: prompts, template, budget and shapes. It carries the
dataset's document- and query-side task instructions when it declares them. The corpus and query vectors were
encoded with them, the same inputs that `retrieval index` /`search` use. The record also carries the tokenizer
and its digest, the text budget and the full width. It carries the stored dtype and the declared MRL kind, set
and projection source. It carries the corpus and query ids in stored order. Its identity is the retrieval
identity of the full-width encoder plus a full-width marker. A cut store can therefore never be mistaken for a
full-width one. The reader refuses vectors narrower than the record's `full_width`.

`k` never reaches the engine request, and it never enters the store's identity. The store is the same for
every `k`. The selection enters only the identity of the per-`k` artifacts that it derives.

## The sweep

`rcp-ndcg retrieval sweep` (`rcp_ndcg.retrieval.sweep`) applies the declared head per `k` to the stored
vectors. It scores each cut with the retrieval scorer, dense inner product or MaxSim. It writes one rankings
file per `k`, whose system is `<model>@<k>`. With `--dataset`, it also runs `evaluate` and `compare` across
the `k` systems. One store therefore yields the whole width/quality curve:

```
rcp-ndcg retrieval store  --dataset beir:data/nfcorpus --retriever encoder.yaml --out store/
rcp-ndcg retrieval sweep  --store store/ --dims 64 --dims 128 --dims 256 \
                          --dataset beir:data/nfcorpus --out-dir sweep/
```

`--dims` selects a subset of the declared set. The default is every declared dimension. A store that declares
only `mrl_range` needs explicit `--dims`, because a range cannot be enumerated. Each `k` costs one head
application and one scoring pass, and no model is called again. A per-`k` run through the ordinary
`retrieval index` /`search` commands computes the same vectors and the same scores. The sweep is the same
arithmetic, from one store.

`--k` is repeatable. With two or more cutoffs, the comparison is written at the first one, and one report
holds every cutoff. `--metrics` defaults to `rcp_ndcg` and `qrel_ndcg` when the dataset carries released
gains. It defaults to `qrel_ndcg` alone when the dataset does not. RCP-nDCG needs a calibration, which the
sweep does not take yet. The command says so in its log. The documented `beir:data/nfcorpus` example is such a
dataset. It therefore scores qrel-nDCG alone, unless `--metrics rcp_ndcg` is passed with a gains source.

## Gating every declared k

The equivalence harness (`rcp-ndcg-test`) gates every declared dimension from one full-width run. Stage 2
builds the recipe's client with its selection stripped (`dimensions`/`mrl_dim`). The served engine and the
reference subprocess therefore both answer at full width. It then applies the declared head to both sides per
`k`, before the ordinary per-vector (or per-token) cosine gate. That gives one gate row per `k` in
`equivalence.json` and the report, beside the full-width row. `k` never enters the engine request. The one
pass is full width, and the head derives every `k` ex-post. The gating code therefore versions the head's
derivation semantics (`rcp_ndcg_test.equivalence.MRL_GATE_VERSION`). A recipe that declares only `mrl_range`
gates its two endpoints and the run's selection. A range cannot be enumerated, and the interior is not
silently claimed. The observation request set records the engine's own side of each declared `k`. That is a
bare `dimensions=k` probe: one per set member, or one per range endpoint and the run's selection. The fake
engines replay it.
