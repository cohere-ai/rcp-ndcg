<!-- Handover copy of the operator's working note `drafts/case-format.md` (sanitized: private names, the private repository, the job platform and storage locations removed; `<operator-notes>/...` paths point to files that are NOT available — the content you need is in handover/). Treat it as the specification it was for its lane. -->

# The `rcp-ndcg-test` case format (operator-defined; binding for `test-pkg` and every `cases-*` lane)

One case per file: `packages/rcp-ndcg-test/cases/<recipe-id>/<case-slug>.yaml` (the recipe id is the canonical one
from `<operator-notes>/drafts/recipes.tsv`). Media files a case uses live next to it under
`packages/rcp-ndcg-test/cases/<recipe-id>/media/` (only with a licence that allows redistribution in an Apache-2.0
repository; otherwise generate a synthetic image or video and say so).

```yaml
id: <recipe-id>/<case-slug>
recipe: <recipe-id>
role: embed | multi_vector | rerank
source:
  kind: model_card | generated          # model_card: taken from the model's Hub README; generated: built for a stratum
  url: https://huggingface.co/<repo>    # model_card only
  revision: <40-hex commit of the README>
  section: "<heading the example sits under>"
  quote: |                              # the exact lines of the card's example the case reproduces (inputs and
    ...                                 # printed outputs), verbatim
strata:
  modality: text | image | video | mixed
  length: short | long_under | long_over | mixed   # long_under: within 5% under the recipe's max_tokens (no cut);
                                                    # long_over: over it (the client cuts; anchors must survive)
  batch: single | uniform | mixed_length | mixed_modality
inputs:
  instruction: <string or null>         # the task instruction the card uses, verbatim
  queries:   [{id: q1, text: "..."}]
  documents: [{id: d1, text: "..."}, {id: d2, image: media/<file>}, {id: d3, video: media/<file>, text: "..."}]
expected:
  kind: similarity_matrix | scores | ranking | none   # what is compared; none: the case only exercises a path
  values: <matrix rows = queries, columns = documents | list of scores per query | ranked document ids per query>
  tolerance: {abs: <float>} | {rank_exact: true} | {spearman_min: <float>}
  origin: published | reference | engine              # published: printed in the card; reference: the recipe's
                                                      # reference implementation; engine: a recorded vLLM run
  status: published_unverified | reproduced | pending_gpu
notes: "<anything a reader needs: rounding in the card, fp32 vs bf16, a card example that does not run as written>"
```

Rules:
- A `model_card` case copies inputs and printed outputs verbatim from the card at the pinned README revision; the
  tolerance reflects the card's rounding (e.g. printed to 4 decimals -> `abs: 5e-4`) plus a declared bf16 margin
  stated in `notes`. If the card prints no outputs, `expected.kind: none` or `ranking` derived from the card's text
  (say which).
- `generated` cases cover the strata the cards do not: every recipe gets at least one case per applicable cell of
  modality x length x batch: `short`, `long_under`, `long_over`, `mixed_length` batches, and for vision-language
  recipes `image`, `video` (if the model takes video) and `mixed_modality` batches. Their `expected` starts as
  `origin: reference, status: pending_gpu` with `values: null`; the GPU waves fill them from the reference
  implementation and the engine, and record both.
- Long inputs are built deterministically (a seeded generator recorded in `notes`, or a public-domain text with its
  source), never random at test time; lengths are measured with the recipe's tokenizer at its pinned revision.
- No network at test time: everything a case needs is in the file or its `media/`.
