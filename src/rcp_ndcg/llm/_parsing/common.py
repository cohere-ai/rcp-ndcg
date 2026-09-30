"""The decoder the tournament and rubric parsers share, and the error they raise.

:func:`decode_answer` finds the judge's JSON object in an answer. It is tolerant
of what surrounds the object and of nothing inside it:

1. it strips ``<think>...</think>`` blocks, and an orphaned leading ``...</think>``
   (an engine may drop the opening tag and keep the closing one);
2. it strips one code fence around the whole answer;
3. it decodes the first complete JSON object that starts at the first ``{`` and
   ignores whatever follows it (prose, a stray ``}``, a second object);
4. when a string of that object holds an invalid escape (LaTeX such as ``\\pi``,
   ``\\{`` or ``\\underline`` in the reasoning), it doubles that backslash, so the
   string holds the backslash as text, and decodes again, up to
   :data:`MAX_ESCAPE_REPAIRS` times;
5. when the object still does not decode, it retries once with a stray quote after
   a number before a closing brace removed (``"doc_6": -4.5"}``).

The escape repair changes the content of a string and nothing else: an invalid
escape is only ever reported inside a string, and the doubled backslash neither
ends that string nor starts another, so every key, number and array of the
object is the one the judge wrote. The decoder does nothing else. It never reads
digits out of prose and never completes a missing document: every ranking,
score and criterion of a parsed judgement is a value of the judge's own JSON
object, and an answer without one is refused with an
:data:`~rcp_ndcg_core.schemas.InvalidCategory`.
"""

from __future__ import annotations

import json
import re
from typing import Any

from rcp_ndcg_core.schemas import InvalidCategory

#: Bumped whenever parsing of judge answers changes in a way that can alter the
#: observations read from identical text, including the JSON schemas that
#: :mod:`rcp_ndcg.llm._parsing.schema` sends the endpoint. Part of the judgement
#: family: observations parsed by different versions are never pooled.
PARSE_VERSION = 2

_THINK_BLOCK = re.compile(r"<think>[\s\S]*?</think>")
_ORPHANED_THINK_END = re.compile(r"^[\s\S]*?</think>")
_CODE_FENCE = re.compile(r"^```(?:\w+)?\s*(.*?)\s*```$", flags=re.DOTALL)
_STRAY_QUOTE = re.compile(r'(-?\d+\.?\d*)"(\s*\n?\s*})')

#: Invalid escapes repaired in one answer at most; an answer with more is refused as ``invalid_json``.
MAX_ESCAPE_REPAIRS = 1000

#: The finish reason of an answer cut at the token limit.
LENGTH_FINISH_REASON = "length"

#: A document key or identifier of an answer: ``doc_3``, ``Document 3``, ``document-03`` or ``3``.
_DOCUMENT_KEY_RE = re.compile(r"^(?:(?:doc(?:ument)?)[\s_-]*)?0*([1-9]\d*)$", flags=re.IGNORECASE)


class UnparseableAnswer(ValueError):
    """A judge's answer that does not parse into a complete observation of its window.

    What the judging loop catches and records as an invalid judgement; any other
    exception from parsing is a defect and propagates.

    Attributes:
        reason: What is wrong with the answer, in one line.
        category: The :data:`~rcp_ndcg_core.schemas.InvalidCategory` of the reason.
    """

    def __init__(self, reason: str, category: InvalidCategory, *, context: str | None = None) -> None:
        super().__init__(reason if context is None else f"{context}\n{reason}")
        self.reason = reason
        self.category: InvalidCategory = category

    def in_context(self, query_id: str, finish_reason: str | None, response: str | None) -> UnparseableAnswer:
        """This error for one query's answer; an answer cut at the token limit is ``truncated`` whatever else failed."""
        category: InvalidCategory = "truncated" if finish_reason == LENGTH_FINISH_REASON else self.category
        reason = self.reason if category == self.category else f"{self.reason} (the answer hit the token limit)"
        excerpt = (response or "")[:500]
        return UnparseableAnswer(
            reason, category, context=f"query {query_id}, finish_reason {finish_reason!r}, answer {excerpt!r}"
        )


class _DuplicateKey(ValueError):
    pass


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKey(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


_DECODER = json.JSONDecoder(object_pairs_hook=_object_without_duplicate_keys)


def strip_reasoning(text: str) -> str:
    """``text`` without ``<think>...</think>`` blocks and without an orphaned leading ``...</think>``."""
    text = _THINK_BLOCK.sub("", text)
    text = _ORPHANED_THINK_END.sub("", text)
    return text.strip()


def strip_code_fence(text: str) -> str:
    """The content of one markdown code fence (```lang ... ```) around the whole of ``text``, else ``text``."""
    match = _CODE_FENCE.match(text)
    return match.group(1).strip() if match else text


def _invalid_escape(text: str, error: json.JSONDecodeError) -> int | None:
    """The index of the backslash of the invalid string escape ``error`` reports, else ``None``."""
    if error.msg.startswith("Invalid \\escape") and text[error.pos : error.pos + 1] == "\\":
        return error.pos
    if error.msg.startswith("Invalid \\uXXXX escape") and text[error.pos - 1 : error.pos] == "\\":
        return error.pos - 1
    return None


def _decode_object(candidate: str) -> Any:
    """The JSON object at the start of ``candidate``, with the escape and stray-quote repairs (module docstring)."""
    text, escapes, quote_repaired = candidate, 0, False
    while True:
        try:
            value, _ = _DECODER.raw_decode(text)
            return value
        except json.JSONDecodeError as error:
            backslash = _invalid_escape(text, error)
            if backslash is not None and escapes < MAX_ESCAPE_REPAIRS:
                text, escapes = text[:backslash] + "\\" + text[backslash:], escapes + 1
                continue
            if not quote_repaired:
                repaired = _STRAY_QUOTE.sub(r"\1\2", text)
                if repaired != text:
                    text, quote_repaired = repaired, True
                    continue
            raise UnparseableAnswer(f"the JSON object does not decode: {error}", "invalid_json") from error


def decode_answer(text: str | None) -> dict[str, Any]:
    """The judge's JSON object in an answer (see the module docstring for what is tolerated).

    Raises:
        UnparseableAnswer: ``no_json`` when the answer holds no ``{``; ``invalid_json`` when the object at the
            first ``{`` does not decode; ``schema`` when it repeats a key.
    """
    body = strip_code_fence(strip_reasoning(text or ""))
    start = body.find("{")
    if start < 0:
        raise UnparseableAnswer("the answer holds no JSON object", "no_json")
    try:
        return _decode_object(body[start:])
    except _DuplicateKey as exc:
        raise UnparseableAnswer(str(exc), "schema") from exc


def document_position(value: object) -> int:
    """The 1-based prompt slot a document key or identifier names (``doc_3``, ``Document 3``, ``"3"`` or ``3``).

    Raises:
        UnparseableAnswer: not a document identifier (category ``schema``).
    """
    if isinstance(value, bool):
        raise UnparseableAnswer(f"invalid document identifier {value!r}", "schema")
    if isinstance(value, int):
        if value > 0:
            return value
        raise UnparseableAnswer(f"document positions must be positive, got {value}", "schema")
    if isinstance(value, str):
        match = _DOCUMENT_KEY_RE.fullmatch(value.strip())
        if match:
            return int(match.group(1))
    raise UnparseableAnswer(f"invalid document identifier {value!r}", "schema")


__all__ = [
    "LENGTH_FINISH_REASON",
    "MAX_ESCAPE_REPAIRS",
    "PARSE_VERSION",
    "UnparseableAnswer",
    "decode_answer",
    "document_position",
    "strip_code_fence",
    "strip_reasoning",
]
