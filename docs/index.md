# RCP-nDCG

RCP-nDCG is nDCG with calibrated relevance gains from an LLM judge. The judge orders each query's candidate pool in
a listwise tournament and answers five binary rubric criteria for every document. A two-parameter item-response
model then puts the tournament scores of all queries on one scale, and the gain of a document is its
discrimination-weighted probability of passing the criteria. The method and its validation are in the paper,
[Rubric-Calibrated Preferences: Cross-Query Calibration of LLM Judgments via Item Response Theory](https://arxiv.org/abs/2609.35739).

Start with the [quickstart](quickstart.md).

## Concepts

- [The metric](concepts/metric.md): nDCG, the RCP gain, qrel-nDCG and Count-nDCG.
- [Scoring protocols and tie rules](concepts/protocols.md): candidates, excluded documents, ties, ideal rankings
  and aggregation, per suite.
- [Stage A: the tournament](concepts/tournament.md): listwise windows and the Bradley-Terry fit.
- [Stage B: the rubric](concepts/rubric.md): the criteria C1 to C5 and their window schedule.
- [The calibration](concepts/calibration.md): the 2PL model, its fit, its two modes and its artifacts.
- [Primitives](concepts/primitives.md): re-annotation, insertion into a tournament, and several judges.
- [Preprocessing and chunking](concepts/preprocessing.md): what the judge reads, and the record of every cut.
- [Text budgets for served roles](concepts/text-budgets.md): what an embedder, pooler or reranker reads: the
  template anchors, the budget and the fit, and every recorded cut.
- [The inference layer](concepts/inference.md): one transport for every role -- wire adapters, role clients,
  routing, retries and credentials.
- [Retrieval and reranking](concepts/retrieval.md): the candidate pools, the retriever kinds, and the index and
  checkpoint identities.
- [Embedding endpoints](concepts/embeddings.md): the embed wire and its client.
- [Late interaction](concepts/late-interaction.md): per-token vectors, pooling and MaxSim.
- [Judges, the judgement store and estimates](concepts/judges.md): judge configs, serving a judge, the store's
  identity and cost estimates.
- [Runs, engines and runners](concepts/runs.md): run configs and steps, serve-by-role engines, job phases and
  durability.

## Data and how-to guides

- [Data](data.md): the released datasets and how to load them.
- [Calibrate your benchmark](how-to/calibrate-your-benchmark.md).
- [MTEB integration](how-to/mteb-integration.md).
- [Reproduce the paper](how-to/reproduce-the-paper.md).
- [Serve a retrieval model](how-to/serve-a-model.md).
- [Add a serving recipe](how-to/add-a-model.md).
- [Validate a recipe on GPUs](how-to/validate-a-recipe.md).
- [Release candidates and the GPU waves](how-to/release-candidates.md).

## Reference

- [Command line](reference/cli.md).
- [Recipes and serving models](reference/recipes.md): the catalog of the 44 recipes (24 families) and the
  `rcp-ndcg-vllm` surface.
- [The results record](reference/results-record.md): the versioned evaluation record and its sink contract.
- [Compatibility and versioning](reference/versioning.md): what is public, the `0.0.x` rules, the recipe
  `schema_version`, the behaviour fingerprint and the artifact tags.
- [The `rcp-ndcg-test` package](reference/rcp-ndcg-test.md): reference cases, conformance and the GPU job
  tooling for contributors.
- [`rcp_ndcg_core`: metric, gains and protocols](api/metric.md).
- [`rcp_ndcg.eval`: evaluation](api/evaluate.md).
- [`rcp_ndcg.inference`: the rerank wire and client](api/inference.md).
