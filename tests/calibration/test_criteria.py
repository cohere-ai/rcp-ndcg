"""Rubric criteria are checked against the family's declared criteria before a fit reads them."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from rcp_ndcg_core.schemas import Judgement, JudgementSet, criterion_labels

from rcp_ndcg.calibration import calibrate
from rcp_ndcg.errors import DataError


def _edited(judgements: JudgementSet, edit: Callable[[dict[str, int]], dict[str, int]]) -> tuple[JudgementSet, str]:
    """``judgements`` with the first valid rubric placement's verdicts edited; returns the set and that record's id."""
    records = list(judgements.judgements)
    index = next(i for i, j in enumerate(records) if j.stage == "rubric" and j.valid)
    first = records[index]
    placements = (first.placements[0].model_copy(update={"criteria": edit(dict(first.placements[0].criteria))}),)
    records[index] = first.model_copy(update={"placements": placements + first.placements[1:]})
    return JudgementSet(judgements=tuple(records), families=judgements.families), first.record_id


def _drop(label: str) -> Callable[[dict[str, int]], dict[str, int]]:
    return lambda verdicts: {key: value for key, value in verdicts.items() if key != label}


@pytest.mark.parametrize(
    ("edit", "named"),
    [
        (lambda verdicts: {**verdicts, "clarity": 1}, "clarity"),  # a criterion the family does not declare
        (_drop("C3"), "C3"),  # a declared criterion without a verdict
        (lambda verdicts: {**_drop("C5")(verdicts), "relevance": verdicts["C5"]}, "relevance"),  # a renamed one
    ],
    ids=["unknown", "missing", "renamed"],
)
def test_verdicts_that_do_not_match_the_declared_criteria_are_refused(
    judgements: JudgementSet, edit: Callable, named: str
) -> None:
    edited, record_id = _edited(judgements, edit)
    with pytest.raises(DataError, match=named) as refused:
        calibrate(edited)
    assert record_id in refused.value.message
    assert refused.value.details["declared"] == list(criterion_labels(5))


def _with_criteria(judgements: JudgementSet, labels: tuple[str, ...]) -> JudgementSet:
    """The rubric judgements answered under a family that declares ``labels``, the verdicts cut to match."""
    rubric = judgements.of_stage("rubric")
    (family,) = rubric.families.values()
    declared = family.model_copy(update={"criteria": labels})
    records = []
    for judgement in rubric.judgements:
        placements = tuple(
            p.model_copy(update={"criteria": dict(zip(labels, (p.criteria or {}).values(), strict=False))})
            for p in judgement.placements
        )
        update = {"family_key": declared.key, "placements": placements if judgement.valid else judgement.placements}
        records.append(Judgement.model_validate({**judgement.model_dump(), **update}))
    return JudgementSet(judgements=tuple(records), families={declared.key: declared})


def test_a_family_that_declares_three_criteria_is_fitted_with_three(judgements: JudgementSet) -> None:
    calibration = calibrate(_with_criteria(judgements, criterion_labels(3)), mode="rubric_only")
    assert calibration.items.criteria == criterion_labels(3)


def test_criteria_not_named_as_a_ladder_are_refused(judgements: JudgementSet) -> None:
    with pytest.raises(DataError, match="C1"):
        calibrate(_with_criteria(judgements, ("relevance", "clarity", "depth")), mode="rubric_only")
