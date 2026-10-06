# `rcp_ndcg_core.metric`, `rcp_ndcg_core.gain` and `rcp_ndcg_core.protocol`

Dependency-light nDCG with float gains, the gains, and the paper's scoring protocol. Pure Python and numpy: no
torch, no pandas, no LLM client, no cloud storage.

## Metric (`rcp_ndcg_core.metric`)

- `ndcg(scores, gains, *, k=10, ties="group_mean", ideal=None)`: the one nDCG@k, with linear gains. `scores` is
  a mapping `{doc_id: score}` (ranked under the tie rule) or an ordered list of doc ids. `ideal` gives the gains of
  the ideal ranking (default: all values of `gains`). RCP-, qrel- and Count-nDCG differ only in the gains. Scores,
  gains and ideal gains must be finite numbers: a non-finite one is refused, not NaN'd through the sums.
- Tie rules (`TieRule`): `"group_mean"` (every tied document gets its tie group's mean gain: the expected nDCG over
  tie orders, as mteb's `ndcg_float_at_k`), `"doc_id_desc"` (trec_eval), `"input_order"` (the order the scores
  come in).
- `rank_by_score(scores, *, ties)`, `dcg(gains, k)`, `ideal_dcg(gains, k)`, `discount(rank)`: building blocks.

## Gains (`rcp_ndcg_core.gain`)

- `gain(theta, items)`: the RCP gain; `pass_probabilities(theta, items)`: the per-criterion pass
  probabilities. `items` has `gamma` and `beta` sequences.
- `count_gain(passes, placements)`: the paper's rubric-only **Count-nDCG** gain: the share of Stage B criteria a
  document passes, $\sum_c S_c / (C\, n)$, with no tournament and no calibration. The pass counts are counts:
  a fractional or out-of-range one is refused, not read as a share.
- `qrel_gain(grade, scheme="linear")`: the gain of a grade (`"linear"`, the paper) or `2**grade - 1`
  (`"exponential"`, which refuses a grade whose `2**grade` would overflow, i.e. grades >= 1024).

The gain of the paper is the discrimination-weighted criterion pass probability

$$
g(\tilde\theta) = \frac{\sum_c \gamma_c \, \sigma\big(\gamma_c(\tilde\theta - \beta_c)\big)}{\sum_c \gamma_c},
$$

which is monotone in $\tilde\theta$ and lies in $(0, 1)$. The alternatives compared in the
paper's gain-function ablation live in `experiments/gain_variants.py`, outside the library.

## Protocol (`rcp_ndcg_core.protocol`)

- `Protocol`: a suite's rules: `ties`, `qrel_gain`, `restrict_to_candidates` (rank only the judged pool),
  `drop_identical_ids`, `round_digits` (BEIR rounds qrel-nDCG to 5).
- `PROTOCOLS`: presets `nanobeir`, `bright` (doc-id ties, BEIR rounding), `vidore`, `mteb` (group-mean ties),
  `trecdl` (input-order ties over the judged pool in pool order) and `plain`.
- `score_query(scores, gains, *, protocol, k, metric, candidates, excluded, query_id)`: one query's nDCG@k.
  Excluded documents leave the ranking and the ideal; the RCP (and Count) ideal takes all gains of the query, the
  qrel ideal all positive grades.
- `candidate_docs(...)` and `aggregate(per_dataset)` (mean over queries per dataset, then over datasets).

`rcp_ndcg.data.load_dataset` loads the released HuggingFace datasets, and `rcp_ndcg.eval.evaluate` scores rankings
on them.

See [`metric.py`](https://github.com/cohere-ai/rcp-ndcg/blob/main/packages/rcp-ndcg-core/src/rcp_ndcg_core/metric.py),
[`gain.py`](https://github.com/cohere-ai/rcp-ndcg/blob/main/packages/rcp-ndcg-core/src/rcp_ndcg_core/gain.py) and
[`protocol.py`](https://github.com/cohere-ai/rcp-ndcg/blob/main/packages/rcp-ndcg-core/src/rcp_ndcg_core/protocol.py)
for the full implementation.
