"""Stdlib-only helpers of the core records (kept in core, which depends on nothing above it)."""

from __future__ import annotations


def descending_score_order(scores: list[float]) -> list[int]:
    """Indices that sort ``scores`` descending, ties keeping their original order."""
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)


__all__ = ["descending_score_order"]
