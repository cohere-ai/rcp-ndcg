# The `rcp-test` package

This package holds the reference cases, one conformance suite, model-level fakes and the GPU job tooling. The tooling
serves the models of RCP-nDCG. The audience is the contributors to the repository and the authors of serving recipes.
A user of the metric needs none of it.

**The package is unpublished on purpose.** `rcp-test` is never uploaded to PyPI, and no published package names
it. The repository's test runs install it from the checkout, as the `dev` dependency group of the workspace. The
client environment of a GPU node installs the staged harness wheel (`rc_build.sh` builds it into `<stage>/harness/`).
Anywhere else, such as a downstream's own suite, installs it **from a git subdirectory** of the repository:

```bash
pip install "rcp-test @ git+https://github.com/cohere-ai/rcp-ndcg.git@v0.0.1#subdirectory=packages/rcp-test"
```

It depends on `cohere-rcp` and `cohere-rcp-vllm` at the version of the release. The dependency never points the other way.
Its own README holds the runnable commands. This page maps what lives where.

## What it holds

- **Reference cases.** A reference case is one input set together with the output that the reference implementation
  gives. There is one case per model-card example and per generated stratum (`cases/<recipe-id>/`). A model-card case
  holds the exact inputs and the quote of the card. Each case is pinned to the commit of its card. The generated cases
  cover the strata that the budgets care about. The strata are multimodal, short, long-under the budget, long-over the
  budget, and mixed batches.
- **Filling the cases.**
  `python -m rcp_test.case_fill --recipe <id> --reference-python <the family's reference venv>` fills the cases
  of one recipe. It runs the reference implementation of the recipe over the inputs of every pending case. It writes
  the outputs of the reference into the YAML of the case. The file gets the `values`, the `status: reproduced` and the
  fill in the note. The `media/` files of a case with media enter the `media` field of the pairs row. The reference's
  own mode therefore reads them, and the case fills like a text case. The run-level `instruction` of a case enters the
  pairs row only when the client of the recipe places it. A recipe that declares `instruction: none` drops the
  instruction by declaration. The reference therefore reads exactly the inputs that the served request carries. The
  fill retries on its own a case that the reference refuses. An example is a container that it cannot sample at the
  pin of the recipe. It records that case as an exception. The reason goes into the report and into the note of the
  case, and the case stays `pending_gpu`. The fill skips a pending case that the mode cannot fill (`kind: none`, or
  the kind of another stage). It records the reason for each skipped case and leaves the file untouched. The cuda
  families fill their cases with the `submit.sh --script casefill` job.
- **Conformance.** One conformance suite has two targets: a live engine and a fake engine at recipe level. The suite
  drives both through the role clients of the product. It never uses raw HTTP, and it never uses a copy of the client.
  It runs on CPU.
- **Fakes.** A fake replaces a model in a test. The model-level fakes are rebuilt from the GPU observation recordings.
  The generic `fake://` transport stays in the product ([the offline
  fakes](../concepts/inference.md#the-offline-fakes)). The model-faithful fakes live here.
- **Equivalence and recording.** The equivalence harness compares the served output against the reference
  implementation. The recorder captures an observation corpus, which is the recorded set of exchanges with a model.
  Both send their requests through the role clients of the product.
- **Jobs.** The package holds the release-candidate build, the node bootstrap and the wave runner of
  [validate a recipe on GPUs](../how-to/validate-a-recipe.md).

Conformance and the golden replays run on CPU and offline. GPU work produces observation corpora, and it never runs
under pytest.
