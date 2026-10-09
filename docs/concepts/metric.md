# The metric: RCP-nDCG

RCP-nDCG is nDCG@k whose gains come from calibrated LLM judgements instead of human relevance labels. This page
defines the metric and its gain. The other pages explain where the calibrated abilities come from:
[the tournament](tournament.md), [the rubric](rubric.md) and [the calibration](calibration.md).

## nDCG

For a ranking $\pi$ of a query's documents, with $d_{\pi(r)}$ at rank $r$, the discounted cumulative gain and
its normalised form at cutoff $k$ are

$$
\mathrm{DCG@}k(\pi) = \sum_{r=1}^{k} \frac{G\bigl(d_{\pi(r)}\bigr)}{\log_2(r+1)},
\qquad
\mathrm{nDCG@}k(\pi) = \frac{\mathrm{DCG@}k(\pi)}{\mathrm{DCG@}k(\pi^*)}.
$$

The ideal ranking $\pi^*$ sorts the query's documents by decreasing gain. The gain $G(d)$ enters linearly: a
document with twice the gain counts twice as much. Documents without a gain count 0.

Three metrics in this package share this formula and differ only in the gain $G$:

| Metric | Gain $G(d)$ | Ideal ranking over |
|---|---|---|
| qrel-nDCG | the human grade (qrel) of $d$ | every positive qrel of the query |
| RCP-nDCG | the calibrated gain $g(\tilde\theta_{ij})$ below | the query's calibrated pool |
| Count-nDCG | the share of rubric criteria $d$ passes | the query's judged pool |

## The gain

Stage B asks the judge $C = 5$ binary criteria about each document ([the rubric](rubric.md)). The calibration
fits, for every criterion $c$, a discrimination $\gamma_c > 0$ and a difficulty $\beta_c$ that all queries
share, and it gives document $d_i$ of query $q_j$ a calibrated ability $\tilde\theta_{ij}$ in logits. With
the logistic link $\sigma(x) = 1 / (1 + e^{-x})$, the probability that the document passes criterion $c$ is

$$
P_c(\tilde\theta_{ij}) = \sigma\bigl(\gamma_c\,(\tilde\theta_{ij} - \beta_c)\bigr).
$$

The gain of RCP-nDCG is the discrimination-weighted mean of these pass probabilities:

$$
g(\tilde\theta_{ij}) = \frac{\sum_c \gamma_c\, \sigma\bigl(\gamma_c\,(\tilde\theta_{ij} - \beta_c)\bigr)}{\sum_c \gamma_c}.
$$

The gain rises smoothly from 0 to 1 as the ability grows. Weighting by $\gamma_c$ makes it rise fastest where
the criteria measure most precisely. RCP-nDCG@k is nDCG@k with $G(d_i) = g(\tilde\theta_{ij})$.

The calibrated ability is a positive affine map of the query's tournament score,
$\tilde\theta_{ij} = \tau_j \hat\theta^{\mathrm{BT}}_i + \alpha_j$ with $\tau_j > 0$ ([the calibration](calibration.md)).
Sorting a query's documents by $\tilde\theta_{ij}$, or by $g(\tilde\theta_{ij})$, therefore reproduces their
Stage A order, and the ideal ranking of RCP-nDCG is the Stage A order of the calibrated pool. A document outside
the calibrated pool has no ability and gets gain 0.

## Count-nDCG

Count-nDCG is the rubric-only baseline. Let $d_i$ pass criterion $c$ in $S_{ijc}$ of its $n_{ij}$ Stage B
windows for query $q_j$. Its gain is the pass share

$$
G_{\mathrm{Count}}(d_i) = \frac{\sum_c S_{ijc}}{C\, n_{ij}}.
$$

Documents with equal pass shares tie. Count-nDCG needs no tournament and no calibration: its gains come from the
rubric windows alone. `rcp_ndcg.calibration.count_gains(judgements)` is the one derivation (the windows'
per-criterion pass counts through `count_gain`); pass it to `rcp_ndcg.eval.evaluate(..., count_gains=...)`, or
score it on the command line with
`rcp-ndcg eval score --metrics count_ndcg --judgements <rubric store>`.

## qrel-nDCG

qrel-nDCG uses the human grade as the gain, linearly (grade 2 counts twice grade 1). The exponential gain
$2^{\mathrm{grade}} - 1$ is available as an option (`qrel_gain(grade, "exponential")`). Its ideal ranking takes
every positive qrel of the query, including positives outside the scored candidates, as trec_eval does.

## In code

The formulas live in the core package `rcp_ndcg_core`, which needs only numpy and pydantic:

```python
from rcp_ndcg_core import count_gain, gain, ndcg, pass_probabilities

items = {"gamma": [1.2, 1.0, 1.1, 0.9, 0.8], "beta": [-2.0, -0.5, 0.0, 0.8, 1.7]}
thetas = {"d1": 1.4, "d2": -0.3, "d3": 0.6}  # calibrated abilities, in logits
gains = {doc_id: gain(theta, items) for doc_id, theta in thetas.items()}

system = {"d1": 0.91, "d2": 0.75, "d3": 0.20}  # one system's scores for the query
print(ndcg(system, gains, k=10))  # RCP-nDCG@10 of this query
print(pass_probabilities(1.4, items))  # P_c for C1..C5
print(count_gain([3, 3, 2, 1, 0], placements=3))  # Count-nDCG gain of a document seen in 3 windows
```

- `ndcg(scores, gains, *, k=10, ties="group_mean", ideal=None)` takes either a score mapping, which it ranks under
  a [tie rule](protocols.md), or an ordered list of document ids. `ideal` gives the gains the ideal ranking is
  built from; it defaults to all values of `gains`.
- `gain(theta, items)` and `pass_probabilities(theta, items)` accept a scalar or an array of abilities. `items` is
  an `ItemParams`, a `Calibration`'s `items`, or any mapping with `gamma` and `beta`.
- `count_gain(passes, placements)` and `qrel_gain(grade, scheme="linear")` give the other two gains.

To score whole runs on a benchmark, with the paper's per-suite rules, use `rcp_ndcg.eval.evaluate`
([scoring protocols](protocols.md) and [the API page](../api/evaluate.md)).
