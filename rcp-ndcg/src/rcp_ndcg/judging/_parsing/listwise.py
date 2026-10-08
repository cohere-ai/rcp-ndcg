"""The tournament parser: a window's ranking and logit-scale scores, and the comparisons they make.

:func:`parse_calibrated_listwise` reads one tournament answer (a complete
ranking and a score per document, decoded by
:func:`~rcp_ndcg.judging._parsing.common.decode_answer`); :func:`window_comparisons` turns a window's
scores into the all-pairs Bradley-Terry observations the calibration fits, and
:func:`judgement_comparisons` does so for a stored window record.
"""

from __future__ import annotations

import math

from rcp_ndcg_core.gain import sigmoid
from rcp_ndcg_core.schemas import Judgement

from rcp_ndcg.judging._parsing.common import UnparseableAnswer, decode_answer, document_position
from rcp_ndcg.judging.client import Completion

# ---------------------------------------------------------------------------
# Calibrated listwise: ranking + logit-scale scores -> all-pairs BT soft labels
# ---------------------------------------------------------------------------


def _all_pairs_from_scores(
    scores: dict[str, float],
) -> list[tuple[str, str, float]]:
    """Derive all-pairs soft labels from logit-scale scores.

    For each pair (a, b) where score_a > score_b, produces
    ``(a, b, sigmoid(score_a - score_b))``, clipped to ``[0.01, 0.99]``.  The
    scores live on a Bradley-Terry logit scale, so before the clip the sigmoid
    of the difference is P(a ≻ b) under the BT model; the clip keeps one
    extreme gap from acting as a certain win.
    """
    ids = list(scores.keys())
    pairs: list[tuple[str, str, float]] = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            diff = scores[a] - scores[b]
            winner, loser = (a, b) if diff >= 0 else (b, a)
            pairs.append((winner, loser, max(0.01, min(0.99, sigmoid(abs(diff))))))
    return pairs


def window_pair_weight(window_size: int) -> float:
    """Weight of each all-pairs soft comparison from one listwise window of ``window_size`` documents.

    A window yields ``w(w-1)/2`` pairs but only ``w-1`` degrees of freedom
    (``w`` scalar scores up to location), so each pair is weighted ``2/w``:
    ``w(w-1)/2 * 2/w = w-1``.  Used wherever tournament windows become
    Bradley-Terry observations, live or replayed from a saved output.
    """
    return 2.0 / max(window_size, 2)


def window_comparisons(scores: dict[str, float]) -> list[tuple[str, str, float, float]]:
    """The Bradley-Terry observations one scored listwise window contributes.

    Every pair of the window's documents as ``(winner, loser, weight,
    soft_label)``: the winner is the higher-scored document, ``soft_label`` is
    ``sigmoid(score gap)`` clipped to ``[0.01, 0.99]``, and ``weight`` is
    :func:`window_pair_weight` of the window size.

    Args:
        scores: ``{doc_id: score}`` of every document of the window, in logits.
    """
    weight = window_pair_weight(len(scores))
    return [(winner, loser, weight, prob) for winner, loser, prob in _all_pairs_from_scores(scores)]


def judgement_comparisons(judgement: Judgement) -> list[tuple[str, str, float, float]]:
    """The Bradley-Terry observations one tournament window record contributes: the tournament's grammar.

    The live tournament, the calibration refit and document insertion all read a
    window through this function, so they fit the same observations: a valid
    window, which scores every placement, is :func:`window_comparisons` of its
    scores; an invalid window contributes none. Ids are the ids the judge saw
    (chunk ids when chunked).

    Args:
        judgement: A tournament judgement record.

    Returns:
        ``[(winner, loser, weight, soft_label), ...]``.
    """
    if judgement.stage != "tournament" or not judgement.valid:
        return []
    return window_comparisons({p.unit_id: p.score for p in judgement.placements if p.score is not None})


def _check_coverage(what: str, positions: list[int], window_size: int) -> None:
    """Refuse positions that are not each window slot exactly once: ``schema`` for an unknown or repeated slot,
    ``incomplete`` when the named slots are valid but some are missing."""
    expected = set(range(1, window_size + 1))
    unknown = sorted(set(positions) - expected)
    repeated = sorted({position for position in positions if positions.count(position) > 1})
    if unknown or repeated:
        raise UnparseableAnswer(
            f"the {what} names unknown documents {unknown} or repeats {repeated} (window of {window_size})", "schema"
        )
    missing = sorted(expected - set(positions))
    if missing:
        raise UnparseableAnswer(f"the {what} leaves out documents {missing} (window of {window_size})", "incomplete")


def _score(key: str, value: object) -> float:
    """A finite score: a JSON number, or a string holding one (``"2.5"``)."""
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise UnparseableAnswer(f"the score of {key!r} must be a number, got {value!r}", "schema")
    try:
        score = float(value)
    except ValueError:
        raise UnparseableAnswer(f"the score of {key!r} must be a number, got {value!r}", "schema") from None
    if not math.isfinite(score):
        raise UnparseableAnswer(f"the score of {key!r} must be finite, got {value!r}", "schema")
    return score


def _read_listwise(text: str | None, window_size: int) -> tuple[list[int], dict[int, float]]:
    """``(ranking positions, {position: score})`` of a complete tournament answer, exactly as the judge wrote them."""
    parsed = decode_answer(text)
    for key in ("ranking", "scores"):
        if key not in parsed:
            raise UnparseableAnswer(f"the answer has no {key!r} key", "schema")
    raw_ranking, raw_scores = parsed["ranking"], parsed["scores"]
    if not isinstance(raw_ranking, list):
        raise UnparseableAnswer(f"'ranking' must be a list, got {type(raw_ranking).__name__}", "schema")
    if not isinstance(raw_scores, dict):
        raise UnparseableAnswer(f"'scores' must be an object, got {type(raw_scores).__name__}", "schema")
    ranking = [document_position(value) for value in raw_ranking]
    scores: dict[int, float] = {}
    positions: list[int] = []
    for key, value in raw_scores.items():
        position = document_position(key)
        score = _score(key, value)
        positions.append(position)
        scores[position] = score
    _check_coverage("ranking", ranking, window_size)
    _check_coverage("scores", positions, window_size)
    return ranking, scores


def parse_calibrated_listwise(
    query_id: str,
    completion: Completion,
    original_ids: list[str],
) -> tuple[list[str], dict[str, float]]:
    """Parse one tournament answer: a complete ranking and a logit-scale score per document.

    Args:
        query_id: The query, for the error message.
        completion: The judge's answer.
        original_ids: The ids the window showed, in prompt order (``doc_1`` is the first).

    Returns:
        ``(ranking, scores)``: the ids best first, and ``{id: score}`` in logits, mapped to ``original_ids``.

    Raises:
        UnparseableAnswer: the answer does not hold a complete ranking and score set (with its category).
        ValueError: ``original_ids`` is empty or repeats an id (a caller's defect).
    """
    window_size = len(original_ids)
    if window_size == 0 or len(set(original_ids)) != window_size:
        raise ValueError(f"a window shows at least one document, each once; got {original_ids}")
    try:
        ranking, scores = _read_listwise(completion.response, window_size)
    except UnparseableAnswer as exc:
        raise exc.in_context(query_id, completion.finish_reason, completion.response) from exc
    return (
        [original_ids[position - 1] for position in ranking],
        {original_ids[position - 1]: score for position, score in sorted(scores.items())},
    )


__all__ = [
    "judgement_comparisons",
    "parse_calibrated_listwise",
    "window_comparisons",
    "window_pair_weight",
]
