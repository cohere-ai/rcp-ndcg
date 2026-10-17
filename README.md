# RCP-nDCG

RCP-nDCG is nDCG whose gains come from calibrated LLM judgements instead of sparse human relevance labels. An LLM
judge orders each query's candidate pool in a listwise tournament and answers five binary rubric criteria for every
document; a two-parameter item-response model then puts the tournament scores of all queries on one shared scale, and
a document's gain is its discrimination-weighted probability of passing the criteria. The method and its validation
are in the paper
[Rubric-Calibrated Preferences: Cross-Query Calibration of LLM Judgments via Item Response Theory](https://arxiv.org/abs/2609.35739).

## The four distributions

| Distribution | What it is | README |
|---|---|---|
| `rcp-ndcg` | The pipeline and the command line: storage, data, retrieval, judging, calibration, evaluation, runs | [rcp-ndcg/README.md](rcp-ndcg/README.md) |
| `rcp-ndcg-core` | The metric: nDCG, the gains, the scoring protocols, the public records, the IRT estimators (numpy + pydantic) | [rcp-ndcg-core/README.md](rcp-ndcg-core/README.md) |
| `rcp-ndcg-vllm` | The lean serving package: the GPU-validated serving recipes and `rcp-ndcg-vllm serve <recipe-id>` | [rcp-ndcg-vllm/README.md](rcp-ndcg-vllm/README.md) |
| `rcp-ndcg-test` | The validation tooling (unpublished, [installed from a git subdirectory](rcp-ndcg-test/README.md)): reference cases, the conformance suite, the equivalence harness, the recorder, the verified emulators, the wave runner | [rcp-ndcg-test/README.md](rcp-ndcg-test/README.md) |

## Install

```bash
pip install rcp-ndcg                  # the pipeline and the CLI
pip install "rcp-ndcg[calibrate]"     # + torch, for the IRT fits
pip install rcp-ndcg-vllm             # serving recipes; resolves recipe:<id> beside rcp-ndcg
```

## Documentation

The rendered docs, the concepts and the command-line reference live at
[the documentation site](https://cohere-ai.github.io/rcp-ndcg/) (`docs/` in the repository;
`mkdocs build --strict` renders them): the concepts, the how-to guides, the command line, the recipe catalog and
the [compatibility and versioning policy](docs/reference/versioning.md). [The quickstart](docs/quickstart.md) is
the 30-minute tour from install to a scored run; [CHANGELOG.md](CHANGELOG.md) states the public surface and what
0.0.1 changes. The paper's tables reproduce from the public data with `experiments/run_all.py`
([REPRODUCIBILITY.md](REPRODUCIBILITY.md)). Contributors and coding agents change this repository under
[AGENTS.md](AGENTS.md); [CITATION.cff](CITATION.cff) holds the paper reference.

## License

Apache-2.0: [LICENSE](LICENSE) and [NOTICE](NOTICE) ship in every distribution.
Security reports: [SECURITY.md](SECURITY.md) (GitHub private vulnerability reporting).
