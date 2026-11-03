"""Stdlib-only helpers shared by the core metric and the core records (kept in core, which depends on nothing
above it)."""

from __future__ import annotations

import math


def descending_score_order(scores: list[float]) -> list[int]:
    """Indices that sort ``scores`` descending, ties keeping their original order.

    The one implementation of the ``"input_order"`` tie rule:
    :func:`rcp_ndcg_core.metric.rank_by_score` (and through it ``ndcg`` and ``tie_groups``) and
    ``RankingExample``'s auto-sort both order by it, so the rule cannot fork into two behaviours on a tie.

    Raises:
        ValueError: a score is not a finite number. A NaN compares False with everything, so
            the sort would silently degenerate to the input order -- for a judged pool, often
            the relevance order the tie rules exist to keep out of the metric.
    """
    bad = [index for index, value in enumerate(scores) if not math.isfinite(value)]
    if bad:
        raise ValueError(f"scores must be finite; got non-finite score(s) at index {bad[:5]}")
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)


__all__ = ["descending_score_order"]
