"""Adding documents to a frozen calibration: score_documents, insert_documents, select_opponents."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from rcp_ndcg_core.irt import score_document
from rcp_ndcg_core.schemas import EstimateFlags, JudgementSet, QueryParams

from rcp_ndcg.calibration import (
    Calibration,
    calibrate,
    insert_documents,
    read_judgements,
    score_documents,
    select_opponents,
)
from rcp_ndcg.calibration.fit import population_prior
from rcp_ndcg.errors import DataError, IdentityError, RcpNdcgWarning
from rcp_ndcg.testing import TinyWorld


@pytest.fixture(scope="module")
def inserted(world: TinyWorld, fitted: Calibration):
    return insert_documents(fitted, read_judgements(world.judgements, world.insertion))


class TestScoreDocuments:
    def test_a_document_scored_from_its_own_fit_evidence_gets_its_fit_ability_back(self, world: TinyWorld) -> None:
        """Rubric-only: scoring uses the fit's own ability prior, so the same answers give the same ability."""
        import warnings

        rubric = read_judgements(world.judgements).of_stage("rubric")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = calibrate(rubric, mode="rubric_only")
        for doc in ("q1-d00", "q1-d05", "q1-d09"):
            row = next(r for r in fitted.thetas if r.doc_id == doc)
            lacking = dataclasses.replace(fitted, thetas=tuple(r for r in fitted.thetas if r.doc_id != doc))
            own = rubric.select(lambda j, d=doc: any(p.doc_id == d for p in j.placements))
            (record,) = [r for r in score_documents(lacking, own).records if r.doc_id == doc]
            # The fit's grid, mapped to the reported scale, differs from the scoring grid: 1e-4 logits of quadrature,
            # against the tenths of a logit another prior moves a document by.
            assert record.estimate.theta == pytest.approx(row.theta, abs=1e-4), doc
            assert record.estimate.se == pytest.approx(row.theta_se, abs=1e-4), doc

    def test_scores_the_rejudged_documents_with_frozen_items(self, fitted: Calibration, judgements) -> None:
        extension = score_documents(fitted, judgements)
        assert sorted(r.doc_id for r in extension.records) == sorted(fitted_lacks(fitted, judgements))
        assert extension.skipped  # the fit's own documents in those windows keep their ability
        mean, sd = population_prior(fitted)
        rubric = judgements.of_stage("rubric")
        for record in extension.records:
            rows = [
                [p.criteria[c] for c in fitted.items.criteria]
                for j in rubric.judgements
                for p in j.placements
                if j.query_id == record.query_id and p.doc_id == record.doc_id
            ]
            expected = score_document(
                fitted.items, len(rows), np.sum(rows, axis=0), prior_mean=mean, prior_sd=sd, se_target=0.5
            )
            assert (record.source, record.estimate) == ("scored", expected)

    def test_extending_adds_rows_and_moves_nothing(self, fitted: Calibration, judgements, tmp_path) -> None:
        extension = score_documents(fitted, judgements)
        extended = fitted.extended(extension)
        assert extended.fingerprint == fitted.fingerprint
        assert extended.thetas[: len(fitted.thetas)] == fitted.thetas
        assert {r.source for r in extended.thetas[len(fitted.thetas) :]} == {"scored"}
        assert extended.extended(extension) == extended  # applying twice is a no-op
        assert Calibration.load(extended.save(tmp_path / "cal")) == extended

    def test_scoring_offers_no_anchor_check_it_cannot_fail(self, fitted: Calibration, judgements) -> None:
        """No row of the calibration is recomputed when scoring, so there is nothing to compare."""
        import inspect

        from rcp_ndcg.cli.calibration import CalibrationScoreRequest

        assert score_documents(fitted, judgements).anchor_report is None
        assert "max_gain_shift" not in inspect.signature(score_documents).parameters
        assert "max_gain_shift" not in CalibrationScoreRequest.model_fields

    def test_two_records_for_one_document_are_refused(self, fitted: Calibration, judgements) -> None:
        extension = score_documents(fitted, judgements)
        twin = extension.records[0].model_copy(update={"theta": 0.0, "record_key": "twin"})
        with pytest.raises(IdentityError, match="already in the calibration"):
            fitted.extended(extension.model_copy(update={"records": (*extension.records, twin)}))

    def test_another_judge_is_refused(self, fitted: Calibration, world: TinyWorld) -> None:
        with pytest.raises(IdentityError, match="not from the instrument") as caught:
            score_documents(fitted, read_judgements(world.lenient))
        assert caught.value.details["foreign_families"]

    def test_nothing_new_is_refused(self, fitted: Calibration, judgements, world: TinyWorld) -> None:
        only_fit = judgements.select(lambda j: not any(p.doc_id in world.rejudged_docs for p in j.placements))
        with pytest.raises(DataError, match="no rubric judgement of a document the calibration lacks"):
            score_documents(fitted, only_fit)


def fitted_lacks(fitted: Calibration, judgements: JudgementSet) -> set[str]:
    known = {row.doc_id for row in fitted.thetas}
    return {p.doc_id for j in judgements.of_stage("rubric").judgements for p in j.placements} - known


class TestInsertDocuments:
    def test_opponents_are_one_window_of_calibrated_documents(self, fitted: Calibration, world: TinyWorld) -> None:
        (window,) = select_opponents(fitted, "q1", world.inserted_doc, n=9)
        assert window[0] == world.inserted_doc and len(window) == 10 == len(set(window))
        assert set(window[1:]) <= set(fitted.theta_map()["q1"])
        with pytest.raises(DataError, match="no tournament query"):
            select_opponents(fitted, "q9", "x")

    def test_a_window_size_splits_the_opponents_into_windows_that_each_hold_the_new_document(
        self, fitted: Calibration, world: TinyWorld
    ) -> None:
        (whole,) = select_opponents(fitted, "q1", world.inserted_doc, n=7)
        windows = select_opponents(fitted, "q1", world.inserted_doc, n=7, window=4)
        assert [len(window) for window in windows] == [4, 4, 2]
        assert all(window[0] == world.inserted_doc for window in windows)
        assert [doc for window in windows for doc in window[1:]] == whole[1:]
        with pytest.raises(DataError, match="window"):
            select_opponents(fitted, "q1", world.inserted_doc, window=1)

    def test_inserts_on_the_calibrated_scale(self, inserted, fitted: Calibration, world: TinyWorld) -> None:
        (record,) = inserted.records
        params = fitted.queries["dataset||q1"]
        assert (record.source, record.doc_id) == ("inserted", world.inserted_doc)
        estimate = record.estimate
        assert estimate.se > 0 and estimate.information > 0 and estimate.flags == EstimateFlags()
        # The conditional MLE's SE is 1/sqrt(information) on its own scale; mapping the SE by tau and the
        # information by 1/tau^2 keeps that on the calibrated scale.
        assert estimate.se == pytest.approx(1 / math.sqrt(estimate.information))
        assert params.to_tournament(estimate.theta) != estimate.theta  # the map was applied (tau, alpha != 1, 0)
        report = inserted.anchor_report
        assert report.ok and report.documents_checked == len(fitted.theta_map()["q1"])
        thetas = fitted.theta_map()["q1"]
        assert min(thetas.values()) < record.estimate.theta < max(thetas.values())

    def test_a_calibration_the_windows_do_not_reproduce_is_refused(self, fitted: Calibration, world: TinyWorld):
        queries = dict(fitted.queries)
        queries["dataset||q1"] = QueryParams(tau=queries["dataset||q1"].tau * 1.3, alpha=0.5)
        moved = dataclasses.replace(fitted, queries=queries)
        with pytest.raises(IdentityError, match="no longer reproduce") as caught:
            insert_documents(moved, read_judgements(world.judgements, world.insertion))
        assert not caught.value.details["anchor_report"]["ok"]

    def test_the_fitted_windows_are_required(self, fitted: Calibration, world: TinyWorld) -> None:
        with pytest.raises(DataError, match="windows the calibration was fitted on are missing"):
            insert_documents(fitted, read_judgements(world.insertion))

    def test_a_rubric_only_calibration_takes_scoring(self, judgements, world: TinyWorld) -> None:
        rubric_only = calibrate(judgements, mode="rubric_only")
        with pytest.raises(DataError, match="rubric-only"):
            insert_documents(rubric_only, read_judgements(world.insertion))
        with pytest.raises(DataError, match="rubric-only"):
            select_opponents(rubric_only, "q1", "x")

    def test_scoring_and_inserting_one_document_conflict(self, inserted, fitted: Calibration) -> None:
        record = inserted.records[0].model_copy(update={"source": "scored", "record_key": "other"})
        with pytest.raises(IdentityError, match="already in the calibration"):
            fitted.extended(inserted).extended(inserted.model_copy(update={"records": (record,)}))
        stale = inserted.model_copy(update={"calibration": "0" * 16})
        with pytest.raises(IdentityError, match="computed against calibration"):
            fitted.extended(stale)


def test_insertion_refits_with_the_l2_the_calibration_was_fitted_with(tmp_path) -> None:
    from rcp_ndcg_core.irt import Priors

    from rcp_ndcg.judging import TournamentSchedule, judge
    from rcp_ndcg.testing import TINY_RUBRIC, TINY_TOURNAMENT, FakeJudge, tiny_rows

    rows, ability = tiny_rows()
    fake = FakeJudge(lambda text: ability[text.split()[-1]])
    pools = {row.id: [d for d in row.doc_ids if d != "q1-new"] for row in rows}
    store, insertion = tmp_path / "judgements", tmp_path / "insertion"
    judge(rows, pools, fake, stage="tournament", out=store, schedule=TINY_TOURNAMENT)
    judge(rows, pools, fake, stage="rubric", out=store, schedule=TINY_RUBRIC)
    calibration = calibrate(read_judgements(store), priors=Priors(bt_l2=5e-2))
    (window,) = select_opponents(calibration, "q1", "q1-new", n=9)
    schedule = TournamentSchedule(window=10, random_placements=12.0, stratified_placements=0.0, adaptive_batches=0)
    judge(rows, None, fake, stage="tournament", out=insertion, docs={"q1": window}, schedule=schedule)

    extension = insert_documents(calibration, read_judgements(store, insertion))

    assert extension.anchor_report is not None and extension.anchor_report.ok


def test_a_pooled_calibration_scores_each_document_once_from_every_judge(world: TinyWorld, tmp_path) -> None:
    import shutil

    from rcp_ndcg.judging import judge
    from rcp_ndcg.testing import TINY_RUBRIC, FakeJudge

    lenient_store = tmp_path / "lenient"
    shutil.copytree(world.lenient, lenient_store)
    lenient = FakeJudge(lambda text: world.abilities[text.split()[-1]], name="fake-lenient", severity=-0.6)
    rejudged = {"q2": list(world.rejudged_docs)}
    schedule = TINY_RUBRIC.model_copy(update={"seed": 0})
    judge(world.dataset, None, lenient, stage="rubric", out=lenient_store, schedule=schedule, docs=rejudged)
    both = read_judgements(world.judgements, lenient_store)
    with pytest.warns(RcpNdcgWarning, match="rubric verdicts but no tournament ability"):
        pooled = calibrate(both, judges="pooled")  # the re-judged q2 documents have no tournament ability

    extension = score_documents(pooled, both)

    assert sorted(r.doc_id for r in extension.records) == sorted(world.rejudged_docs)
    assert len({r.record_key for r in extension.records}) == len(extension.records)
    scored = [(r.query_id, r.doc_id) for r in pooled.extended(extension).thetas if r.source == "scored"]
    assert sorted(scored) == [("q2", doc) for doc in sorted(world.rejudged_docs)]


def test_the_record_key_digest_recipe_is_pinned(inserted: object) -> None:
    """``record_key`` names a record in the calibration's own fit evidence (it is re-checked on a refit); its
    digest recipe is pinned so it cannot drift silently under a changed payload."""
    from rcp_ndcg.support.identity import hash_payload, short

    (record,) = sorted(inserted.records, key=lambda r: (r.dataset, r.query_id, r.doc_id))
    expected = short(
        hash_payload(
            {
                "source": record.source,
                "slot": [record.dataset, record.query_id, record.doc_id],
                "calibration": record.calibration,
                "judgements": record.judgements,
                "estimate": record.estimate.model_dump(mode="json"),
            }
        ),
        16,
    )
    assert record.record_key == expected, "the recipe: source, slot, calibration, judgements and the estimate"
