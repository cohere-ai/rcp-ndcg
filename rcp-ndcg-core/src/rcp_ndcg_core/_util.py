"""Stdlib-only helpers of the core records (kept in core, which depends on nothing above it)."""

from __future__ import annotations

import math


def descending_score_order(scores: list[float]) -> list[int]:
    """Indices that sort ``scores`` descending, ties keeping their original order.

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
