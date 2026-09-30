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
- [Judges, serving and runners](concepts/serving.md): judge configs, vLLM and SGLang, cost estimates, and the
  local, SLURM and Kubernetes runners.

## Data and tutorials

- [Data](data.md): the released datasets and how to load them.
- [Calibrate your benchmark](tutorials/calibrate-your-benchmark.md).
- [MTEB integration](tutorials/mteb-integration.md).
- [Reproduce the paper](tutorials/reproduce-the-paper.md).

## Reference

- [Command line](reference/cli.md).
- [`rcp_ndcg_core`: metric, gains and protocols](api/metric.md).
- [`rcp_ndcg.eval`: evaluation](api/evaluate.md).
