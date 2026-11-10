# RCP-nDCG

RCP-nDCG is nDCG (normalized discounted cumulative gain) with calibrated relevance gains from a large language
model (LLM) judge. The judge orders the candidate pool of each query in a listwise tournament. Listwise means that
the judge compares all documents of the pool as one list. The judge also answers five binary rubric criteria
(yes/no quality checks) for each document. A two-parameter item-response model (2PL) then places the tournament
scores of all queries on one scale. The gain of a document is its discrimination-weighted probability of passing
the criteria. The paper,
[Rubric-Calibrated Preferences: Cross-Query Calibration of LLM Judgments via Item Response Theory](https://arxiv.org/abs/2609.35739),
gives the method and its validation.

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
- [Text budgets for served roles](concepts/text-budgets.md): what an embedder, a pooler or a reranker reads. It
  covers the template anchors, the budget, the fit and every recorded cut.
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
- [Compatibility and versioning](reference/versioning.md): what is public, the `0.0.x` rules and the recipe
  `schema_version`. It also covers the behaviour fingerprint and the artifact tags.
- [The `rcp-ndcg-test` package](reference/rcp-ndcg-test.md): reference cases, conformance and the GPU job
  tooling for contributors.
- [`rcp_ndcg_core`: metric, gains and protocols](reference/api/rcp_ndcg_core.md).
- [`rcp_ndcg.eval`: evaluation](reference/api/rcp_ndcg.eval.md).
- [`rcp_ndcg.inference`: the rerank wire and client](reference/api/rcp_ndcg.inference.md).
