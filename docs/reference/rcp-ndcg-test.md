# The `rcp-ndcg-test` package

Reference cases, one conformance suite, model-level fakes and the GPU job tooling for RCP-nDCG's served
models. Audience: contributors to the repository and authors of serving recipes -- users of the metric need
none of it.

**Unpublished on purpose.** `rcp-ndcg-test` is never uploaded to PyPI and no published package names it: the
repository's test runs install it from the checkout (the workspace's `dev` dependency group) or from a staged
wheelhouse. It depends on `rcp-ndcg` and `rcp-ndcg-vllm` at the release's version -- never the other way
round. Its own README has the runnable commands; this page maps what lives where.

## What it holds

- **Reference cases.** One case per model-card example and per generated stratum (`cases/<recipe-id>/`): the
  card's exact inputs and quote (model card cases, each pinned to the card's commit), and generated cases over
  the strata the budgets care about -- multimodal, short, long-under and long-over the budget, and mixed
  batches.
- **Conformance.** One suite, two targets -- a live engine and a recipe-level fake engine -- both driven
  through the product's role clients (never raw HTTP, never a copy of the client), on CPU.
- **Fakes.** Model-level fakes rebuilt from the GPU observation recordings. The generic `fake://` transport
  stays in the product ([the offline fakes](../concepts/inference.md#the-offline-fakes)); the model-faithful
  ones live here.
- **Equivalence and recording.** The equivalence harness (served output against the reference implementation)
  and the recorder that captures an observation corpus. Both send through the product's role clients.
- **Jobs.** The release-candidate build, the node bootstrap and the wave runner of [validate a recipe on
  GPUs](../how-to/validate-a-recipe.md).

Conformance and the golden replays run on CPU and offline; GPU work produces observation corpora and never
runs under pytest.
