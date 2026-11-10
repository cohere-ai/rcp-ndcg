# RCP-nDCG

RCP-nDCG is nDCG (normalized discounted cumulative gain) with calibrated gains. A large language model (LLM) judge
supplies those gains through its judgements, not through sparse human relevance labels. The judge orders the
candidate pool of each query in a listwise tournament. Listwise means that the judge compares all documents of the
pool as one list. The judge also answers five binary rubric criteria (yes/no quality checks) for each document. A
two-parameter item-response model (2PL) then places the tournament scores of all queries on one scale. The gain of a
document is its discrimination-weighted probability of passing the criteria. The paper,
[Rubric-Calibrated Preferences: Cross-Query Calibration of LLM Judgments via Item Response Theory](https://arxiv.org/abs/2609.35739),
gives the method and its validation.

## The four distributions

| Distribution | What it is | README |
|---|---|---|
| `rcp-ndcg` | The pipeline and the command line: storage, data, retrieval, judging, calibration, evaluation, runs | [rcp-ndcg/README.md](rcp-ndcg/README.md) |
| `rcp-ndcg-core` | The metric: nDCG, the gains, the scoring protocols, the public records, the IRT (item response theory) estimators (numpy + pydantic) | [rcp-ndcg-core/README.md](rcp-ndcg-core/README.md) |
| `rcp-ndcg-vllm` | The lean serving package: the GPU-validated serving recipes and `rcp-ndcg-vllm serve <recipe-id>` | [rcp-ndcg-vllm/README.md](rcp-ndcg-vllm/README.md) |
| `rcp-ndcg-test` | The validation tooling (unpublished, [installed from a git subdirectory](rcp-ndcg-test/README.md)): reference cases, the conformance suite, the equivalence harness, the recorder, the verified emulators, the wave runner | [rcp-ndcg-test/README.md](rcp-ndcg-test/README.md) |

## Install

```bash
pip install rcp-ndcg                  # the pipeline and the CLI
pip install "rcp-ndcg[calibrate]"     # + torch, for the IRT fits
pip install rcp-ndcg-vllm             # serving recipes; resolves recipe:<id> beside rcp-ndcg
```

## Documentation

The rendered pages are at [the documentation site](https://cohere-ai.github.io/rcp-ndcg/). The site shows
the concepts, the how-to guides and the command line. It also holds the recipe catalog and the [compatibility and
versioning policy](docs/reference/versioning.md). The pages are in `docs/`. `mkdocs build --strict` renders them.
[The quickstart](docs/quickstart.md) is a 30-minute introduction from the install to a scored run.
[CHANGELOG.md](CHANGELOG.md) states the public surface and the changes in 0.0.1. `experiments/run_all.py`
reproduces the tables of the paper from the public data. [REPRODUCIBILITY.md](REPRODUCIBILITY.md) is the record.
Contributors and coding agents change this repository under [AGENTS.md](AGENTS.md). [CITATION.cff](CITATION.cff)
holds the paper reference.

## License

Apache-2.0. Every distribution ships [LICENSE](LICENSE) and [NOTICE](NOTICE). Send security reports through
[SECURITY.md](SECURITY.md) (GitHub private vulnerability reporting).
