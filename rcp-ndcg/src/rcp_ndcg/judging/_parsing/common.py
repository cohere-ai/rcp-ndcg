"""The decoder the tournament and rubric parsers share, and the error they raise.

:func:`decode_answer` finds the judge's JSON object in an answer. It tolerates
what surrounds the object, and makes exactly two named repairs inside it (items 4
and 5):

1. it takes the answer as the text after the last ``</think>``: everything before it is reasoning, a whole
   ``<think>...</think>`` block or reasoning whose opening tag the engine dropped, so an object drafted in
   the reasoning is never read, and an object that only precedes a ``</think>`` (a stray one, or a later
   reasoning block) is not the answer; an **unclosed** opener before the answer's object makes everything
   from it to the end of the answer reasoning (cut at the token limit, or never closed), so a truncated
   answer's draft is never the judgement;
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
#: :mod:`rcp_ndcg.judging._parsing.schema` sends the endpoint. Part of the judgement
#: family: observations parsed by different versions are never pooled.
PARSE_VERSION = 4

_THINK_END = "</think>"
_THINK_OPEN = re.compile(r"<think\b")
_FENCE = "```"
_FENCE_LANGUAGE = re.compile(r"\w*")
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
    """The answer of ``text``: what follows its last ``</think>``, without reasoning that never closed.

    Everything up to the last think-end is the judge's reasoning: one or more ``<think>...</think>`` blocks,
    or reasoning whose opening tag the engine dropped. Reasoning routinely holds braces and drafts of the
    answer object, so the cut is at the tag, never at the first ``{``. An object that only precedes a
    think-end (a stray one, or a later reasoning block) is therefore not read: from the text alone it cannot
    be told apart from a draft in the reasoning, and reading a draft would put an observation in the fit that
    the judge did not answer. The cut is at the last think-end wherever it stands, even inside a string of
    an object: nothing before it is ever read, and the text after it is never rewritten.

    An **unclosed** ``<think`` opener before the answer's first ``{`` means the object lies inside the
    reasoning (an answer cut at the token limit while the judge was still thinking, or reasoning the judge
    never closed): the answer is what precedes that tag, so the draft is not read. An opener after the first
    ``{`` is left to the decoder, which reads the object and ignores what follows it.
    """
    end = text.rfind(_THINK_END)
    answer = text[end + len(_THINK_END) :] if end >= 0 else text
    opening, brace = _THINK_OPEN.search(answer), answer.find("{")
    if opening is not None and (brace < 0 or opening.start() < brace):
        answer = answer[: opening.start()]
    return answer.strip()


def strip_code_fence(text: str) -> str:
    """The content of one markdown code fence (```lang ... ```) around the whole of ``text``, else ``text``.

    The text must open with the fence (an optional language word after it) and close with one at its end, or
    before one final newline; the content between them is returned stripped. This is what the pattern
    ``^```(?:\\w+)?\\s*(.*?)\\s*```$`` (DOTALL) matches, located in linear time: that pattern backtracked
    quadratically or worse on a fence followed by a long whitespace run.
    """
    if not text.startswith(_FENCE):
        return text
    language = _FENCE_LANGUAGE.match(text, len(_FENCE))  # \w* always matches, possibly empty
    after_language = language.end() if language else len(_FENCE)
    if text.endswith(_FENCE):
        closing = len(text) - len(_FENCE)
    elif text.endswith(_FENCE + "\n"):
        closing = len(text) - len(_FENCE) - 1
    else:
        return text
    if closing < after_language:  # the closing fence would overlap the opening one
        return text
    return text[after_language:closing].strip()


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
        except _DuplicateKey:
            raise  # a repeated key is a schema refusal (decode_answer), not a decoding failure
        except (ValueError, RecursionError) as error:
            # What json raises past its own grammar: an integer literal beyond int's digit limit (4,300 digits) is a
            # ValueError, nesting beyond the recursion limit a RecursionError. Either is an answer that does not
            # decode, never an exception that escapes the judging loop.
            raise UnparseableAnswer(f"the JSON object does not decode: {error}", "invalid_json") from error


def decode_answer(text: str | None) -> dict[str, Any]:
    """The judge's JSON object in an answer (see the module docstring for what is tolerated).

    Raises:
        UnparseableAnswer: ``no_json`` when the answer holds no ``{``; ``invalid_json`` when the object at the
            first ``{`` does not decode (an integer literal beyond the digit limit and nesting beyond the
            recursion limit included); ``schema`` when it repeats a key.
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
            try:
                return int(match.group(1))
            except ValueError:  # beyond int's digit limit (4,300 digits): no window holds that slot
                raise UnparseableAnswer(
                    f"invalid document identifier of {len(value)} characters ({value[:40]!r}...)", "schema"
                ) from None
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
