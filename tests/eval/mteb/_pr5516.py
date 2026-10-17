"""The float-gain metric and the `load_float_gains` reader of mteb PR #5516, vendored as a test oracle.

Source: https://github.com/embeddings-benchmark/mteb/pull/5516 ("feat: Add optional float relevance gains for
retrieval (ndcg_float_at_k)"), at head commit ``595c8ecccca2d5553df4652543bc4206dcbabbe2`` (Apache-2.0,
Copyright the mteb contributors; attributed in NOTICE). ``_dcg`` and ``ndcg_float_scores`` are copied verbatim
from ``mteb/_evaluators/retrieval_metrics.py``; ``load_float_gains`` is copied verbatim from the PR's
``mteb/tasks/reranking/eng/nano_beir_rcp_reranking.py`` (the same function in the BRIGHT and ViDoRe v3 task
files). The tests compare ``rcp_ndcg.eval.mteb.ndcg_float_scores`` against the vendored function and load the
written layout with the vendored ``load_float_gains``, so the repository's copy is checked against the PR's own
code rather than against itself.

Three test-only adaptations, all noted where they occur: ``datasets`` is imported inside ``load_float_gains``
(the test environment without the ``[mteb]`` extra must still import this module for the pure metric), and
``evaluate_abstention`` falls back to an empty mapping when mteb is absent. A third is cosmetic:
``load_float_gains`` annotates its metadata parameter ``Any`` instead of importing mteb's ``TaskMetadata``, so
the metric functions import without the extra.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from itertools import groupby
from typing import Any

try:  # the PR imports this from mteb; the metric itself is pure Python
    from mteb._evaluators.retrieval_metrics import evaluate_abstention
except ImportError:  # pragma: no cover - the fallback is for environments without the [mteb] extra

    def evaluate_abstention(results: Mapping[str, Any], per_query: Mapping[str, Any]) -> dict[str, float]:  # noqa: ANN401
        return {}


def _dcg(gains: Sequence[float], k: int) -> float:
    """DCG@k = sum_{r=1}^{k} gain(r) / log2(r + 1)."""
    return sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains[:k], start=1))


def ndcg_float_scores(
    gains: Mapping[str, Mapping[str, float]],
    results: Mapping[str, Mapping[str, float]],
    k_values: Sequence[int],
) -> dict[str, float]:
    """Computes NDCG@k over continuous (float) relevance gains, bypassing pytrec_eval.

    pytrec_eval only accepts integer relevance labels. Here each document's gain is
    its float value, used as-is (linear gain, no ``2**g - 1``); gains must be finite
    and non-negative. Equal model scores form one equivalence class: every tied
    document is credited the group-mean gain, the expectation over all tie
    resolutions. A stable sort instead would let the candidate-pool order (which
    is relevance-ordered for reranking pools) leak ground truth into tied scores.

    A query ID in `results` that is missing from `gains` (e.g. its qrels gain
    entries are all null) is scored 0.0. A non-finite or negative
    gain, and a NaN model score, raise `ValueError` rather than being scored --
    each would otherwise reach the mean as a silent `nan` or an extra tie class.
    Unlike the integer-qrels metrics, `skip_first_result` is not applied to the
    float metric.

    Args:
        gains: Continuous gains for each query, `{query_id: {doc_id: gain}}`.
            Should cover every query ID in `results` (missing queries score 0.0).
        results: Retrieval scores for each query, `{query_id: {doc_id: score}}`.
        k_values: The k values for which to compute the scores.

    Returns:
        A dictionary with the mean `ndcg_float_at_{k}` scores and the nAUC
        variants of the per-query scores.
    """
    for query_id, doc_gains in gains.items():
        # NaN passes every comparison, so finiteness is checked before the sign
        if any(not math.isfinite(gain) or gain < 0 for gain in doc_gains.values()):
            raise ValueError(
                f"Non-finite or negative gain for query {query_id}. Gains must be finite and non-negative."
            )

    for query_id, doc_scores in results.items():
        if any(math.isnan(score) for score in doc_scores.values()):
            raise ValueError(
                f"NaN model score for query {query_id}. NDCG_float is undefined "
                "for NaN model scores (infinities are ranked as usual)."
            )

    per_query: dict[str, list[float]] = defaultdict(list)
    for query_id, doc_scores in results.items():
        query_gains = gains.get(query_id, {})
        ranking = sorted(doc_scores, key=lambda doc_id: doc_scores[doc_id], reverse=True)

        tie_mean: dict[str, float] = {}
        for _, tie_group in groupby(ranking, key=doc_scores.__getitem__):
            tie_docs = list(tie_group)
            mean = sum(query_gains.get(doc_id, 0.0) for doc_id in tie_docs) / len(tie_docs)
            tie_mean.update(dict.fromkeys(tie_docs, mean))

        ideal_gains = sorted(query_gains.values(), reverse=True)
        for k in k_values:
            ideal_dcg = _dcg(ideal_gains, k)
            if ideal_dcg == 0.0:
                per_query[f"NDCG_float@{k}"].append(0.0)
                continue
            actual_dcg = _dcg([tie_mean[doc_id] for doc_id in ranking[:k]], k)
            per_query[f"NDCG_float@{k}"].append(actual_dcg / ideal_dcg)

    summary = {
        f"ndcg_float_at_{key.split('@')[1]}": round(sum(values) / len(values), 5) for key, values in per_query.items()
    }
    naucs = evaluate_abstention(results, per_query)
    return {
        **summary,
        **{key.replace("@", "_at_").lower(): value for key, value in naucs.items()},
    }


def load_float_gains(metadata: Any, hf_subset: str, split: str) -> dict[str, dict[str, float]]:  # noqa: ANN401
    """Load the float `gain` column of a subset's qrels.

    The standard loader keeps only the integer `score`. Rows whose `gain` is null are skipped.
    """
    import datasets  # test-only adaptation: the [mteb] extra is not needed to import the metric

    qrels = datasets.load_dataset(
        metadata.dataset["path"],
        f"{hf_subset}-qrels",
        split=split,
        revision=metadata.dataset["revision"],
    )
    gains: dict[str, dict[str, float]] = defaultdict(dict)
    for query_id, doc_id, gain in zip(qrels["query-id"], qrels["corpus-id"], qrels["gain"], strict=True):
        if gain is not None:
            gains[str(query_id)][str(doc_id)] = float(gain)
    return dict(gains)
