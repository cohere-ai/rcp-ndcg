"""The two answer parsers: a tournament window (ranking and scores) and a rubric window (criteria per document).

The invariant: every ranking, score and criterion of a parsed answer is a value of the judge's own JSON object.
The decoder tolerates what surrounds the object (reasoning, a code fence, prose, a stray brace, a second object)
and a stray quote after a number; it never reads digits out of prose and never completes a missing document. An
answer it cannot read is refused with a category from a closed set.
"""

from __future__ import annotations

import json
import math
import random
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from rcp_ndcg_core.schemas import Judgement, Placement

from rcp_ndcg.llm._parsing.common import MAX_ESCAPE_REPAIRS, PARSE_VERSION, UnparseableAnswer, decode_answer
from rcp_ndcg.llm._parsing.listwise import judgement_comparisons, parse_calibrated_listwise, window_comparisons
from rcp_ndcg.llm._parsing.rubric import parse_rubric_criteria
from rcp_ndcg.llm._parsing.schema import answer_schema, response_format
from rcp_ndcg.llm.client import Completion

IDS = ["a", "b", "c"]

#: The judge's object in the golden answers: doc_2 first, then doc_1, then doc_3.
ANSWER = {
    "reasoning": "doc_2 answers it.",
    "ranking": ["doc_2", "doc_1", "doc_3"],
    "scores": {"doc_1": 1.5, "doc_2": 3.0, "doc_3": -2.0},
}
OBJECT = json.dumps(ANSWER)
PARSED_RANKING = ["b", "a", "c"]
PARSED_SCORES = {"a": 1.5, "b": 3.0, "c": -2.0}


def _tournament(text: str, ids: list[str] = IDS, finish_reason: str | None = "stop"):  # noqa: B006
    ranking, scores = parse_calibrated_listwise("q1", Completion(response=text, finish_reason=finish_reason), ids)
    return SimpleNamespace(ranking=ranking, scores=scores)


def _rubric(payload: object, ids: list[str], num_criteria: int = 2, finish_reason: str | None = "stop"):
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return parse_rubric_criteria("q1", Completion(response=text, finish_reason=finish_reason), ids, num_criteria)


def _category(parse, *args, **kwargs) -> str:
    with pytest.raises(UnparseableAnswer) as caught:
        parse(*args, **kwargs)
    return caught.value.category


# ---------------------------------------------------------------------------
# The golden corpus of malformed answers
# ---------------------------------------------------------------------------

#: Answers that hold the judge's object: each parses to exactly that object.
RECOVERED = {
    "stray-trailing-brace": OBJECT + "}",
    "stray-trailing-braces-and-newline": OBJECT + "\n}}",
    "stray-quote-after-a-number": OBJECT.replace("-2.0}", '-2.0"}'),
    "think-block-with-numbered-prose": "<think>1. doc_3\n2. doc_1\n3. doc_2 {draft}</think>\n" + OBJECT,
    "orphaned-think-end-with-numbered-prose": "Ranking: 3, 1, 2.\n1. doc_3 is best\n</think>\n\n" + OBJECT,
    "orphaned-think-end-after-the-object": OBJECT + "\n</think>",  # a stray tag never erases the object
    "orphaned-think-end-after-the-object-with-prose": OBJECT + "\n</think>\n(its reasoning was cut)",
    "code-fence": "```json\n" + OBJECT + "\n```",
    "code-fence-inside-prose": "Here is the ranking:\n```json\n" + OBJECT + "\n```\nI ranked 3 documents.",
    "prose-before-the-json": "Sure. Of documents 1, 2 and 3, doc_3 is weakest:\n" + OBJECT,
    "prose-after-the-json": OBJECT + "\nNote: 1. doc_3 2. doc_1 3. doc_2",
    "a-second-object-after-the-first": OBJECT + '\n{"ranking": ["doc_3", "doc_1", "doc_2"]}',
    "latex-escapes-in-the-reasoning": OBJECT.replace(
        "doc_2 answers it.", r"area \pi r^2, set \{x\}, \frac12\n\underline{a}"
    ),
    "latex-escapes-and-a-stray-quote": OBJECT.replace("answers it.", r"has \sum_i x_i \le 1").replace(
        "-2.0}", '-2.0"}'
    ),
}


@pytest.mark.parametrize("text", RECOVERED.values(), ids=RECOVERED.keys())
def test_an_answer_holding_the_judges_object_parses_to_exactly_that_object(text: str) -> None:
    response = _tournament(text)
    assert response.ranking == PARSED_RANKING
    assert response.scores == PARSED_SCORES


_TRUNCATED = OBJECT[: OBJECT.index('"doc_3": -2.0')]

#: Answers without the judge's complete object: (answer, finish_reason, category).
REFUSED = {
    "numbered-prose-only": ("1. doc_3\n2. doc_1\n3. doc_2", "stop", "no_json"),
    "think-leftovers-with-numbered-prose-only": ("<think>doc_3 > doc_1</think>Ranking: 3, 1, 2", "stop", "no_json"),
    "empty": ("", "stop", "no_json"),
    "truncated-at-the-token-limit": (_TRUNCATED, "length", "truncated"),
    "truncated-reasoning-at-the-token-limit": ("<think>1. doc_3 2. doc_1", "length", "truncated"),
    "truncated-without-a-length-finish": (_TRUNCATED, "stop", "invalid_json"),
    "unquoted-identifiers": ('{"ranking": [doc_2, doc_1, doc_3], "scores": {}}', "stop", "invalid_json"),
    "prose-with-a-brace-before-the-json": ("Scores {doc_1 high}: " + OBJECT, "stop", "invalid_json"),
    "duplicate-key": (OBJECT.replace('"doc_1": 1.5', '"doc_1": 1.5, "doc_1": 0.5'), "stop", "schema"),
    "no-scores-key": (json.dumps({"ranking": ANSWER["ranking"]}), "stop", "schema"),
    "ranking-not-a-list": (json.dumps({**ANSWER, "ranking": "doc_2, doc_1, doc_3"}), "stop", "schema"),
    "score-not-a-number": (json.dumps({**ANSWER, "scores": {**ANSWER["scores"], "doc_3": "low"}}), "stop", "schema"),
    "unknown-document": (json.dumps({**ANSWER, "ranking": ["doc_2", "doc_1", "doc_4"]}), "stop", "schema"),
    "repeated-document": (json.dumps({**ANSWER, "ranking": ["doc_2", "doc_2", "doc_3"]}), "stop", "schema"),
    "a-ranking-missing-a-document": (json.dumps({**ANSWER, "ranking": ["doc_2", "doc_1"]}), "stop", "incomplete"),
    "scores-missing-a-document": (
        json.dumps({**ANSWER, "scores": {"doc_1": 1.5, "doc_2": 3.0}}),
        "stop",
        "incomplete",
    ),
    "incomplete-and-cut-at-the-token-limit": (
        json.dumps({**ANSWER, "scores": {"doc_1": 1.5, "doc_2": 3.0}}),
        "length",
        "truncated",
    ),
}


@pytest.mark.parametrize(("text", "finish_reason", "category"), REFUSED.values(), ids=REFUSED.keys())
def test_an_answer_without_the_judges_complete_object_is_refused_with_its_category(
    text: str, finish_reason: str, category: str
) -> None:
    assert _category(_tournament, text, finish_reason=finish_reason) == category


def test_an_invalid_escape_is_kept_as_text_and_a_valid_one_as_json_defines_it() -> None:
    """LaTeX in a string: an invalid escape (``\\pi``, ``\\{``, ``\\underline``) becomes a literal backslash; a
    valid one (``\\n``, and ``\\f`` of ``\\frac``) decodes as JSON defines it. Only that string changes."""
    text = OBJECT.replace("doc_2 answers it.", r"area \pi r^2, set \{x\}, \frac12\n\underline{a}")
    decoded = decode_answer(text)
    assert decoded["reasoning"] == "area \\pi r^2, set \\{x\\}, \x0crac12\n\\underline{a}"
    assert {key: value for key, value in decoded.items() if key != "reasoning"} == {
        key: value for key, value in ANSWER.items() if key != "reasoning"
    }


def test_the_escape_repair_is_bounded() -> None:
    within = json.dumps({**ANSWER, "reasoning": "§" * MAX_ESCAPE_REPAIRS}, ensure_ascii=False).replace("§", "\\q")
    assert _tournament(within).ranking == PARSED_RANKING
    beyond = json.dumps({**ANSWER, "reasoning": "§" * (MAX_ESCAPE_REPAIRS + 1)}, ensure_ascii=False).replace("§", "\\q")
    assert _category(_tournament, beyond) == "invalid_json"


def test_prose_that_states_a_ranking_is_never_read_as_one() -> None:
    """The paper-era fallback read digits out of the whole answer when the JSON failed; no path does that now."""
    for text in ("Final ranking: doc_2, doc_1, doc_3", "2 > 1 > 3", "<think>x</think>\n[2, 1, 3]"):
        with pytest.raises(UnparseableAnswer):
            _tournament(text)


def test_the_error_names_the_query_and_the_answer() -> None:
    with pytest.raises(UnparseableAnswer, match="query q1") as caught:
        _tournament("no json here")
    assert caught.value.reason == "the answer holds no JSON object"


class TestTournament:
    def test_positions_numbers_and_strings_name_documents(self) -> None:
        text = json.dumps({"ranking": [2, "1", "Document 3"], "scores": {"1": 1.5, "doc_2": "3.0", "doc_3": -2}})
        response = _tournament(text)
        assert response.ranking == PARSED_RANKING
        assert response.scores == PARSED_SCORES

    @pytest.mark.parametrize("score", [float("nan"), float("inf"), True], ids=["nan", "inf", "bool"])
    def test_a_non_finite_or_boolean_score_is_refused(self, score: object) -> None:
        text = json.dumps({**ANSWER, "scores": {**ANSWER["scores"], "doc_3": score}})
        assert _category(_tournament, text) == "schema"

    @pytest.mark.parametrize("finish_reason", [None, "abort"])
    def test_a_complete_answer_parses_whatever_finish_reason_the_endpoint_reports(self, finish_reason) -> None:
        assert _tournament(OBJECT, finish_reason=finish_reason).ranking == PARSED_RANKING

    def test_a_complete_object_before_the_token_limit_is_the_judges_answer(self) -> None:
        assert _tournament(OBJECT + "\nAnd one more thing abo", finish_reason="length").ranking == PARSED_RANKING

    def test_a_window_is_every_pair_at_weight_two_over_w_winner_first(self) -> None:
        comparisons = window_comparisons({"a": 2.0, "b": -1.0, "c": 0.5, "d": 0.5})
        assert len(comparisons) == 6 and all(weight == 2 / 4 for _, _, weight, _ in comparisons)
        (a_over_b,) = [c for c in comparisons if c[:2] == ("a", "b")]
        assert a_over_b[3] == pytest.approx(1 / (1 + math.exp(-3.0)))
        assert all(0.01 <= label <= 0.99 for *_, label in comparisons)


# ---------------------------------------------------------------------------
# Property: the parsed ranking is the embedded one, whatever surrounds it
# ---------------------------------------------------------------------------

_PROSE = ["Ranking:", "1.", "2)", "doc_3", "doc_1 >", "best", "3, 1, 2", "\n", "- doc_2", "score 4.5", "#"]
#: What may follow the object, an orphaned ``</think>`` included: only what sits before the
#: object is reasoning, so a tag after it must never erase the object.
_TAIL = ["}", "}}", "]", "{", '{"ranking": [1]}', "doc_1", "4 3 2 1", "```", "\n", "<think>", "</think>", '"', "..."]


def _junk(rng: random.Random, words: list[str], n: int) -> str:
    return " ".join(rng.choice(words) for _ in range(n))


def _embedded(rng: random.Random) -> tuple[str, list[int], dict[int, float], int]:
    w = rng.randint(2, 10)
    ranking = rng.sample(range(1, w + 1), w)
    scores = {position: round(rng.uniform(-5, 5), 2) for position in range(1, w + 1)}
    obj = {
        "reasoning": _junk(rng, _PROSE, rng.randint(0, 6)),
        "ranking": [f"doc_{position}" if rng.random() < 0.7 else position for position in ranking],
        "scores": {f"doc_{position}": score for position, score in rng.sample(sorted(scores.items()), w)},
    }
    text = json.dumps(obj, indent=rng.choice([None, 2]))
    if rng.random() < 0.3:
        text = "```json\n" + text + "\n```"
    prefix = _junk(rng, _PROSE, rng.randint(0, 8))
    if rng.random() < 0.4:
        prefix = "<think>" + _junk(rng, _PROSE + ["{", "}", '{"ranking": [1, 2]}'], 6) + "</think>" + prefix
    return prefix + "\n" + text + _junk(rng, _TAIL, rng.randint(0, 6)), ranking, scores, w


def test_the_parsed_ranking_is_the_embedded_one_whatever_surrounds_it() -> None:
    rng = random.Random(20260930)
    for _ in range(400):
        text, ranking, scores, w = _embedded(rng)
        ids = [f"u{i}" for i in range(1, w + 1)]
        response = _tournament(text, ids)
        assert response.ranking == [ids[position - 1] for position in ranking], text
        assert response.scores == {ids[position - 1]: score for position, score in scores.items()}, text


def test_an_answer_whose_object_lacks_a_document_is_never_given_an_order() -> None:
    """Remove one document from the embedded object: the parser refuses rather than returning any order."""
    rng = random.Random(7)
    for _ in range(200):
        text, ranking, _, w = _embedded(rng)
        dropped = f'"doc_{ranking[-1]}"'
        broken = text.replace(f", {dropped}]", "]").replace(f",\n    {dropped}\n  ]", "\n  ]")
        if broken == text:
            continue
        with pytest.raises(UnparseableAnswer):
            _tournament(broken, [f"u{i}" for i in range(1, w + 1)])


#: LaTeX that is an invalid JSON escape when written raw into a string (``\\u`` followed by a non-hex letter too).
_LATEX = [
    r"\pi",
    r"\{",
    r"\}",
    r"\alpha",
    r"\sum",
    r"\le",
    r"\cdot",
    r"\underline",
    r"\uparrow",
    r"\,",
    r"\%",
    r"\$",
    r"\ ",
]
#: Text a string may hold besides the LaTeX, including what JSON escapes validly (quote, backslash, newline, tab).
_WORDS = ["doc_3", "x", "1.5", "}", "{", "]", ",", ":", '"', "\\", "\n", "\t", "a b", "-2"]


def _value(rng: random.Random, depth: int, latex: list[str]) -> object:
    kind = rng.choice(["int", "float", "bool", "null", "string", "string"] + (["list", "object"] if depth < 3 else []))
    if kind == "int":
        return rng.randint(-10, 10)
    if kind == "float":
        return round(rng.uniform(-5, 5), 3)
    if kind == "bool":
        return rng.random() < 0.5
    if kind == "null":
        return None
    if kind == "list":
        return [_value(rng, depth + 1, latex) for _ in range(rng.randint(0, 4))]
    if kind == "object":
        return {f"k{i}": _value(rng, depth + 1, latex) for i in range(rng.randint(0, 4))}
    parts = []
    for _ in range(rng.randint(0, 6)):
        if rng.random() < 0.4:
            latex.append(rng.choice(_LATEX))
            parts.append(f"\u00a7{len(latex) - 1}\u00a7")
        else:
            parts.append(rng.choice(_WORDS))
    return " ".join(parts)


def _with_latex(value: object, latex: list[str]) -> object:
    """``value`` with every placeholder replaced by its LaTeX, as text: what the repaired decode must return."""
    if isinstance(value, str):
        for index, command in enumerate(latex):
            value = value.replace(f"\u00a7{index}\u00a7", command)
        return value
    if isinstance(value, list):
        return [_with_latex(item, latex) for item in value]
    if isinstance(value, dict):
        return {key: _with_latex(item, latex) for key, item in value.items()}
    return value


def test_the_escape_repair_changes_string_content_only() -> None:
    """Raw LaTeX written into the strings of a random object decodes to exactly that object with the LaTeX as text:
    every key, number, boolean, null, array and object is the one written, of the same type and in the same order."""
    rng = random.Random(314)
    for _ in range(300):
        latex: list[str] = []
        obj = {"reasoning": _value(rng, 3, latex), **{f"k{i}": _value(rng, 0, latex) for i in range(rng.randint(1, 5))}}
        text = json.dumps(obj, ensure_ascii=False, indent=rng.choice([None, 2]))
        for index, command in enumerate(latex):
            text = text.replace(f"\u00a7{index}\u00a7", command)
        decoded = decode_answer(text + rng.choice(["", "}", "\nDone."]))
        assert json.dumps(decoded) == json.dumps(_with_latex(obj, latex)), text


# ---------------------------------------------------------------------------
# Stored window records -> Bradley-Terry observations
# ---------------------------------------------------------------------------


class TestWindowRecords:
    """How a stored tournament window enters the Bradley-Terry fit."""

    @staticmethod
    def _record(scores: list[float | None], **fields) -> Judgement:
        recorded_at = datetime.now(UTC)
        placements = tuple(Placement(position=i, doc_id=f"d{i}", score=s) for i, s in enumerate(scores, 1))
        return Judgement(
            record_id="r",
            dataset="ds",
            query_id="q",
            stage="tournament",
            family_key="f",
            window_seq=0,
            placements=placements,
            recorded_at=recorded_at,
            **fields,
        )

    def test_a_fully_scored_window_is_its_window_comparisons(self) -> None:
        record = self._record([2.0, -1.0, 0.5])
        assert judgement_comparisons(record) == window_comparisons({"d1": 2.0, "d2": -1.0, "d3": 0.5})

    @pytest.mark.parametrize("scores", [[2.0, None, 0.5], [None, None, None]], ids=["partial", "ranking-only"])
    def test_a_window_without_a_score_on_every_document_is_not_a_valid_record(self, scores) -> None:
        with pytest.raises(ValidationError, match="score on every placement"):
            self._record(scores, ranking=(1, 3, 2))

    def test_an_invalid_window_gives_none(self) -> None:
        invalid = self._record([1.0, 0.0], valid=False, invalid_reason="garbled", invalid_category="invalid_json")
        assert judgement_comparisons(invalid) == []


# ---------------------------------------------------------------------------
# Rubric
# ---------------------------------------------------------------------------

RUBRIC = {"doc_1": {"reasoning": "on topic", "criteria": {"C1": 1, "C2": 0}}, "doc_2": {"criteria": {"C1": 0, "C2": 1}}}
RUBRIC_PARSED = {"ga": {"C1": 1, "C2": 0}, "gb": {"C1": 0, "C2": 1}}


class TestRubric:
    def test_positions_map_to_global_ids_and_string_bits_are_read(self) -> None:
        data = {"doc_1": {"criteria": {"C1": 1, "C2": "0"}}, "Document 2": {"criteria": {"C1": "0", "C2": 1}}}
        assert _rubric(data, ["ga", "gb"]) == RUBRIC_PARSED

    @pytest.mark.parametrize(
        "text",
        [
            json.dumps(RUBRIC) + "}",
            "<think>1. doc_2 passes C2</think>```json\n" + json.dumps(RUBRIC) + "\n```",
            "Verdicts for documents 1 and 2:\n" + json.dumps(RUBRIC) + "\nDone.",
            json.dumps(RUBRIC).replace('"C2": 1}', '"C2": 1"}'),
            json.dumps({key: {**value, "reasoning": "§"} for key, value in RUBRIC.items()}, ensure_ascii=False).replace(
                "§", r"meets \{C1\} since \alpha \le \beta\n"
            ),
        ],
        ids=["stray-brace", "think-and-fence", "prose", "stray-quote", "latex-escapes-in-the-reasoning"],
    )
    def test_an_answer_holding_the_judges_object_parses_to_exactly_that_object(self, text: str) -> None:
        assert _rubric(text, ["ga", "gb"]) == RUBRIC_PARSED

    @pytest.mark.parametrize(
        ("data", "category"),
        [
            ({"doc_1": {"criteria": {"C1": 1, "C2": 0}}}, "incomplete"),
            ({}, "incomplete"),
            ({"doc_1": {"criteria": {"C1": 1, "C2": 0}}, "doc_2": {"criteria": {"C1": 0}}}, "schema"),
            ({"doc_1": {"criteria": {"C1": 1, "C2": 0}}, "doc_2": {"criteria": {"C1": 0, "C2": 1, "C3": 0}}}, "schema"),
            ({"doc_1": {"criteria": {"C1": 1, "C2": 0}}, "doc_3": {"criteria": {"C1": 0, "C2": 1}}}, "schema"),
            ({"doc_1": {"criteria": {"C1": 1, "C2": 0}}, "Document 1": {"criteria": {"C1": 0, "C2": 1}}}, "schema"),
            ({"doc_1": {"C1": 1, "C2": 0}, "doc_2": {"criteria": {"C1": 0, "C2": 1}}}, "schema"),
            ('{"doc_1": {"criteria": {"C1": 1, "C2": 0}}, "doc_1": {"criteria": {"C1": 1, "C2": 0}}}', "schema"),
            ("1. doc_1 passes C1\n2. doc_2 passes C2", "no_json"),
            ('{"doc_1": {"criteria": {"C1": 1, "C2": 0}}, "doc_2": {"crit', "invalid_json"),
        ],
        ids=[
            "missing-document",
            "empty-object",
            "missing-criterion",
            "extra-criterion",
            "unknown-document",
            "duplicate-document",
            "no-criteria-object",
            "duplicate-key",
            "prose-only",
            "truncated",
        ],
    )
    def test_incomplete_or_unknown_observations_are_refused_with_their_category(self, data, category) -> None:
        assert _category(_rubric, data, ["ga", "gb"]) == category

    def test_an_answer_cut_at_the_token_limit_is_truncated(self) -> None:
        cut = '{"doc_1": {"criteria": {"C1": 1, "C2": 0}}, "doc_2": {"crit'
        assert _category(_rubric, cut, ["ga", "gb"], finish_reason="length") == "truncated"

    @pytest.mark.parametrize("value", [2, -1, True, "yes", "", 0.0])
    def test_a_verdict_must_be_zero_or_one(self, value: object) -> None:
        with pytest.raises(UnparseableAnswer, match="must be binary 0/1"):
            _rubric({"doc_1": {"criteria": {"C1": value}}}, ["ga"], num_criteria=1)


# ---------------------------------------------------------------------------
# The answer schemas sent for structured output
# ---------------------------------------------------------------------------


def _conforms(value: object, schema: dict) -> bool:
    """The subset of JSON Schema the answer schemas use: object, array, string, number, integer and enum."""
    if "enum" in schema and value not in schema["enum"]:
        return False
    kind = schema.get("type")
    if kind == "object":
        properties = schema["properties"]
        return (
            isinstance(value, dict)
            and set(value) == set(schema["required"]) == set(properties)
            and all(_conforms(value[key], properties[key]) for key in value)
        )
    if kind == "array":
        return (
            isinstance(value, list)
            and schema["minItems"] <= len(value) <= schema["maxItems"]
            and all(_conforms(item, schema["items"]) for item in value)
        )
    if kind == "string":
        return isinstance(value, str)
    if kind == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    return kind == "integer" and type(value) is int


def test_the_answer_schemas_are_the_prompts_answers_and_closed() -> None:
    assert _conforms(ANSWER, answer_schema("tournament", 3))
    assert not _conforms({**ANSWER, "extra": 1}, answer_schema("tournament", 3))
    assert not _conforms({**ANSWER, "ranking": ["doc_2", "doc_1"]}, answer_schema("tournament", 3))
    assert not _conforms({**ANSWER, "ranking": ["doc_2", "doc_1", "doc_4"]}, answer_schema("tournament", 3))
    rubric = {
        "doc_1": {"reasoning": "", "criteria": {"C1": 1, "C2": 0}},
        "doc_2": {"reasoning": "", "criteria": {"C1": 0, "C2": 1}},
    }
    assert _conforms(rubric, answer_schema("rubric", 2, ("C1", "C2")))
    assert not _conforms({"doc_1": rubric["doc_1"]}, answer_schema("rubric", 2, ("C1", "C2")))
    assert _rubric(rubric, ["ga", "gb"]) == RUBRIC_PARSED


def test_the_response_format_is_the_openai_standard_json_schema_form() -> None:
    fmt = response_format("rubric", 4, ("C1", "C2", "C3", "C4", "C5"))
    assert fmt["type"] == "json_schema"
    assert set(fmt["json_schema"]) == {"name", "schema", "strict"} and fmt["json_schema"]["strict"] is True
    assert fmt["json_schema"]["schema"] == answer_schema("rubric", 4, ("C1", "C2", "C3", "C4", "C5"))


def test_the_parse_version_is_pinned() -> None:
    """The parse version is part of the judgement family: a change that can alter what identical text parses
    to must bump it (M5's trailing-orphan fix did -- pre-fix answers and post-fix ones never pool), and the
    bump is deliberate policy, pinned as a literal."""
    assert PARSE_VERSION == 3
