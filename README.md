<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/cohere-ai/rcp-ndcg/main/docs/assets/cohere-logo-dark.svg">
    <img src="https://raw.githubusercontent.com/cohere-ai/rcp-ndcg/main/docs/assets/cohere-logo.svg" alt="Cohere" height="36">
  </picture>
</p>

# RCP-nDCG

RCP-nDCG is nDCG whose gains come from calibrated LLM judgements instead of sparse human relevance labels (the
paper: [Rubric-Calibrated Preferences](https://arxiv.org/abs/2609.35739)). The repository holds four
distributions; the full documentation is at [docs/](docs/quickstart.md).

| Directory | Distribution | What it is |
|---|---|---|
| [`rcp-ndcg-core/`](rcp-ndcg-core/README.md) | `rcp-ndcg-core` | The metric, the gains, the scoring protocols and the IRT estimators (numpy + pydantic). |
| [`rcp-ndcg/`](rcp-ndcg/README.md) | `rcp-ndcg` | The pipeline and the CLI: storage, data, retrieval, judging, calibration, evaluation, runs. Start here. |
| [`rcp-ndcg-vllm/`](rcp-ndcg-vllm/README.md) | `rcp-ndcg-vllm` | The lean serving package: 18 GPU-validated recipes and `rcp-ndcg-vllm serve <recipe-id>`. |
| [`rcp-ndcg-test/`](rcp-ndcg-test/README.md) | *(unpublished)* | The validation tooling: the equivalence harness, the engine recorder and the GPU wave runner. |

```bash
pip install "rcp-ndcg[hf,calibrate]" --extra-index-url https://download.pytorch.org/whl/cpu
```

[Quickstart](docs/quickstart.md) · [API reference](docs/api/) · [REPRODUCIBILITY](REPRODUCIBILITY.md) ·
[CHANGELOG](CHANGELOG.md) · [Contributing](AGENTS.md) · [Citation](CITATION.cff)

Licensed under [Apache-2.0](LICENSE); third-party code is credited in [NOTICE](NOTICE).
