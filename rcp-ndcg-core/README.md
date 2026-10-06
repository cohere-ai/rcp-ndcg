# rcp-ndcg-core

The dependency-light core of [RCP-nDCG](https://github.com/cohere-ai/rcp-ndcg): nDCG with float gains and explicit
tie rules, the RCP gain and the other gains, the paper's scoring protocols, the public records, and the IRT
estimators. It needs numpy and pydantic only; the `irt` extra adds torch and scipy for fitting.

Install it from PyPI (Python 3.12):

```bash
pip install rcp-ndcg-core            # metric, gains, protocols, schemas
pip install "rcp-ndcg-core[irt]"     # + torch and scipy, for the Bradley-Terry and 2PL fits
```

```python
from rcp_ndcg_core import PROTOCOLS, count_gain, gain, ndcg, pass_probabilities, qrel_gain, score_query

items = {"gamma": [1.2, 1.0, 1.1, 0.9, 0.8], "beta": [-2.0, -0.5, 0.0, 0.8, 1.7]}  # per criterion C1..C5
thetas = {"d1": 1.4, "d2": -0.3, "d3": 0.6}  # calibrated abilities, in logits
gains = {doc_id: gain(theta, items) for doc_id, theta in thetas.items()}  # RCP gains in (0, 1)

scores = {"d1": 0.91, "d2": 0.75, "d3": 0.20}  # one system's scores for the query
print(ndcg(scores, gains, k=10))  # RCP-nDCG@10, group-mean ties
print(score_query(scores, gains, protocol=PROTOCOLS["nanobeir"], candidates=["d1", "d2", "d3"]))
print(pass_probabilities(1.4, items), count_gain([3, 2, 2, 1, 0], placements=3), qrel_gain(2))
```

RCP-, qrel- and Count-nDCG differ only in their gains. `PROTOCOLS` holds the paper's per-suite rules (candidates,
excluded documents, tie rule, rounding), `score_query` scores one query under one, and `aggregate` averages like
the paper. The pipeline that produces calibrated abilities (judging, calibration, evaluation, the CLI) is the
`rcp-ndcg` package in the same repository; see its [documentation](https://github.com/cohere-ai/rcp-ndcg/blob/main/docs/index.md).
