"""The rubric parser: binary criteria C1..CK per document of a window.

The rubric prompt asks the judge to apply ``num_criteria`` binary criteria to
each document independently. The answer is a JSON object mapping ``doc_<i>``
keys to ``{"criteria": {"C1": 0|1, ...}}``, decoded by
:func:`~rcp_ndcg.llm._parsing.common.decode_answer`.

:func:`parse_rubric_criteria` reads one answer into
``{global_doc_id: {"C1": 0|1, ...}}``, refusing an answer that leaves out a
document (``incomplete``) or a criterion, or names an unknown one (``schema``).
"""

from __future__ import annotations

from typing import Any

from rcp_ndcg.llm._parsing.common import UnparseableAnswer, decode_answer, document_position
from rcp_ndcg.llm.client import Completion


def _binary_criterion(value: object, criterion: str) -> int:
    """A verdict: the integer 0 or 1, or the string ``"0"`` or ``"1"``."""
    if type(value) is int and value in (0, 1):
        return value
    if isinstance(value, str) and value in {"0", "1"}:
        return int(value)
    raise UnparseableAnswer(f"{criterion} must be binary 0/1, got {value!r}", "schema")


def _read_rubric(parsed: dict[str, Any], window_doc_ids: list[str], num_criteria: int) -> dict[str, dict[str, int]]:
    expected_criteria = [f"C{k}" for k in range(1, num_criteria + 1)]
    result: dict[str, dict[str, int]] = {}
    for key, value in parsed.items():
        position = document_position(key)
        if not 1 <= position <= len(window_doc_ids):
            raise UnparseableAnswer(f"unknown document {key!r} (window of {len(window_doc_ids)})", "schema")
        doc_id = window_doc_ids[position - 1]
        if doc_id in result:
            raise UnparseableAnswer(f"document {key!r} is answered twice", "schema")
        criteria = value.get("criteria") if isinstance(value, dict) else None
        if not isinstance(criteria, dict):
            raise UnparseableAnswer(f"document {key!r} has no criteria object", "schema")
        if set(criteria) != set(expected_criteria):
            missing = sorted(set(expected_criteria) - set(criteria))
            unknown = sorted(set(criteria) - set(expected_criteria))
            raise UnparseableAnswer(
                f"document {key!r} has invalid criterion coverage; missing={missing}, unknown={unknown}", "schema"
            )
        result[doc_id] = {label: _binary_criterion(criteria[label], label) for label in expected_criteria}
    missing_docs = [doc_id for doc_id in window_doc_ids if doc_id not in result]
    if missing_docs:
        raise UnparseableAnswer(f"the answer leaves out documents {missing_docs}", "incomplete")
    return result


def parse_rubric_criteria(
    query_id: str,
    completion: Completion,
    window_doc_ids: list[str],
    num_criteria: int,
) -> dict[str, dict[str, int]]:
    """Parse one rubric answer: criteria ``C1..C<num_criteria>`` for every document of the window.

    Args:
        query_id: The query, for the error message.
        completion: The judge's answer.
        window_doc_ids: The ids the window showed, in prompt order (``doc_1`` is the first).
        num_criteria: The number of criteria the prompt asks for.

    Returns:
        ``{doc_id: {"C1": 0|1, ...}}`` for every id of the window.

    Raises:
        UnparseableAnswer: the answer does not hold every criterion of every document (with its category).
        ValueError: an empty or repeating window, or ``num_criteria`` below 1 (a caller's defect).
    """
    if num_criteria <= 0:
        raise ValueError(f"num_criteria must be positive, got {num_criteria}")
    if not window_doc_ids or len(set(window_doc_ids)) != len(window_doc_ids):
        raise ValueError(f"a window shows at least one document, each once; got {window_doc_ids}")
    try:
        return _read_rubric(decode_answer(completion.response), list(window_doc_ids), num_criteria)
    except UnparseableAnswer as exc:
        raise exc.in_context(query_id, completion.finish_reason, completion.response) from exc


__all__ = ["parse_rubric_criteria"]
