"""The JSON schemas of the two stages' answers, sent as ``response_format`` when the endpoint constrains output.

A judge configured with ``decoding: json_schema`` (vLLM and SGLang do, through
their structured-output backends; so does the OpenAI API) is asked for the
OpenAI-standard ``response_format``::

    {"type": "json_schema", "json_schema": {"name": ..., "schema": {...}, "strict": true}}

The schema is the shape the shipped prompts ask for, sized to the window: the
tournament answer is ``reasoning``, a ``ranking`` of exactly the window's
``doc_<i>`` identifiers and a score for each of them; the rubric answer has one
object per document with its ``reasoning`` and every criterion ``C1..CK`` as 0 or
1. Every object is closed (``additionalProperties: false``) and every key
required, as strict mode needs. A ranking that repeats a document still fits the
schema (``uniqueItems`` is not honoured by every backend); the parser refuses it.

A change to these schemas changes what the judge returns, so it bumps
:data:`~rcp_ndcg.judging._parsing.common.PARSE_VERSION`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from rcp_ndcg_core.schemas import Stage


def _document_keys(window_size: int) -> list[str]:
    return [f"doc_{index}" for index in range(1, window_size + 1)]


def _closed(properties: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def answer_schema(stage: Stage, window_size: int, criteria: Sequence[str] = ()) -> dict[str, Any]:
    """The JSON schema of one answer of ``stage`` for a window of ``window_size`` documents.

    Args:
        stage: ``"tournament"`` or ``"rubric"``.
        window_size: The documents the window shows (``doc_1`` .. ``doc_<window_size>``).
        criteria: The rubric's criterion labels (``C1..CK``); unused by the tournament.
    """
    documents = _document_keys(window_size)
    if stage == "tournament":
        return _closed(
            {
                "reasoning": {"type": "string"},
                "ranking": {
                    "type": "array",
                    "items": {"type": "string", "enum": documents},
                    "minItems": window_size,
                    "maxItems": window_size,
                },
                "scores": _closed({document: {"type": "number"} for document in documents}),
            }
        )
    verdicts = _closed({criterion: {"type": "integer", "enum": [0, 1]} for criterion in criteria})
    document_answer = _closed({"reasoning": {"type": "string"}, "criteria": verdicts})
    return _closed(dict.fromkeys(documents, document_answer))


def response_format(stage: Stage, window_size: int, criteria: Sequence[str] = ()) -> dict[str, Any]:
    """The OpenAI-standard ``response_format`` that constrains one answer to :func:`answer_schema`."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": f"rcp_ndcg_{stage}_window",
            "schema": answer_schema(stage, window_size, criteria),
            "strict": True,
        },
    }


__all__ = ["answer_schema", "response_format"]
