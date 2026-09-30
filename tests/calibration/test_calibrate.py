"""calibrate(): the fit, its refusals, and the one artifact layout."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from rcp_ndcg_core.schemas import JudgementSet

from rcp_ndcg.calibration import Calibration, calibrate, read_judgements
from rcp_ndcg.calibration._projection import rubric_observations
from rcp_ndcg.calibration.fit import (
    COVERAGE_FILE,
    DIAGNOSTICS_FILE,
    EXTENSIONS_FILE,
    IDENTITY_FILE,
    ITEMS_FILE,
    QUERIES_FILE,
    THETAS_FILE,
)
from rcp_ndcg.errors import DataError, IdentityError, MissingInputError, RcpNdcgWarning
from rcp_ndcg.testing import TinyWorld

from .conftest import rubric_set


def _fit_only(judgements: JudgementSet, world: TinyWorld) -> JudgementSet:
    """The judgements the saved calibration was fitted on (the re-judged subset left out)."""
    return judgements.select(lambda j: not any(p.doc_id in world.rejudged_docs for p in j.placements))


class TestTournamentFit:
    def test_calibrated_ability_is_the_query_transform_of_its_tournament_order(self, fitted: Calibration) -> None:
        assert fitted.mode == "tournament"
        assert fitted.items.criteria == ("C1", "C2", "C3", "C4", "C5")
        assert sum(fitted.items.gamma) == pytest.approx(5.0, abs=1e-4)
        assert list(fitted.items.beta) == sorted(fitted.items.beta)  # the fake's criteria get harder C1 -> C5
        assert set(fitted.queries) == {"dataset||q1", "dataset||q2"}
        assert all(row.source == "fit" and row.theta_se and row.theta_se > 0 for row in fitted.thetas)

    def test_abilities_follow_the_hidden_ones(self, fitted: Calibration, world: TinyWorld) -> None:
        for thetas in fitted.theta_map().values():
            ordered = sorted(thetas, key=thetas.__getitem__)
            truth = sorted(thetas, key=world.abilities.__getitem__)
            agree = sum(a == b for a, b in zip(ordered, truth, strict=True))
            assert agree >= len(ordered) // 3
            assert thetas[truth[-1]] > thetas[truth[0]]

    def test_gains_are_monotone_probabilities(self, fitted: Calibration) -> None:
        gains, thetas = fitted.gains(), fitted.theta_map()
        assert set(gains) == {"q1", "q2"}  # one dataset: raw query ids
        for query, per_doc in gains.items():
            assert all(0.0 <= g <= 1.0 for g in per_doc.values())
            by_theta = sorted(per_doc, key=thetas[query].__getitem__)
            assert [per_doc[d] for d in by_theta] == sorted(per_doc.values())
        with pytest.raises(DataError, match="no dataset"):
            fitted.gains("elsewhere")

    def test_refit_is_deterministic(self, judgements: JudgementSet, world: TinyWorld, fitted: Calibration) -> None:
        assert calibrate(_fit_only(judgements, world)).fingerprint == fitted.fingerprint


class TestModes:
    def test_rubric_only_has_no_query_transform(self, judgements: JudgementSet) -> None:
        fit = calibrate(judgements, mode="rubric_only")
        assert fit.mode == "rubric_only" and not fit.queries
        assert list(fit.items.beta) == sorted(fit.items.beta)
        assert all(row.theta_se is not None for row in fit.thetas)

    def test_auto_without_a_tournament_is_rubric_only(self, judgements: JudgementSet) -> None:
        assert calibrate(judgements.of_stage("rubric")).mode == "rubric_only"

    def test_mixed_coverage_is_refused_with_per_query_counts(self, judgements: JudgementSet) -> None:
        mixed = judgements.select(lambda j: not (j.stage == "tournament" and j.query_id == "q2"))
        with pytest.raises(DataError, match="1 of 2 rubric queries have no tournament") as caught:
            calibrate(mixed)
        details = caught.value.details
        assert details["queries_without_tournament"] == ["dataset||q2"]
        counts = details["windows_per_query"]
        assert counts["dataset||q2"]["tournament"] == 0 and counts["dataset||q2"]["rubric"] > 0
        assert counts["dataset||q1"]["tournament"] > 0
        # The same judgements calibrate rubric-only when asked to.
        assert calibrate(mixed, mode="rubric_only").mode == "rubric_only"

    def test_no_rubric_is_refused(self, judgements: JudgementSet) -> None:
        with pytest.raises(DataError, match="no rubric judgements"):
            calibrate(judgements.of_stage("tournament"))


class TestJudges:
    def test_single_with_two_judges_is_refused(self, world: TinyWorld) -> None:
        with pytest.raises(IdentityError, match="2 families") as caught:
            calibrate(read_judgements(world.judgements, world.lenient))
        assert len(caught.value.details["families"]) == 2

    def test_pooled_fit_recovers_the_lenient_judge(self, world: TinyWorld) -> None:
        fit = calibrate(read_judgements(world.judgements, world.lenient), judges="pooled")
        assert fit.judge_severity["fake"] > fit.judge_severity["fake-lenient"]
        assert fit.judge_severity["fake"] + fit.judge_severity["fake-lenient"] == pytest.approx(0.0, abs=1e-6)
        assert len(fit.family_of("rubric")) == 2
        assert set(fit.diagnostics.reliability.per_family) == {f.key for f in fit.family_of("rubric")}

    def test_pooled_judges_who_share_no_document_are_refused(self, world: TinyWorld) -> None:
        judgements = read_judgements(world.judgements, world.lenient)
        lenient = {key for key, family in judgements.families.items() if family.judge_model == "fake-lenient"}
        disjoint = judgements.select(
            lambda j: j.stage == "tournament" or (j.query_id == "q2") == (j.family_key in lenient)
        )
        with pytest.raises(DataError, match="share") as caught:
            calibrate(disjoint, judges="pooled")
        assert caught.value.details["pairs"] == {"fake|fake-lenient": 0}
        assert set(caught.value.details["items_per_judge"]) == {"fake", "fake-lenient"}

    def test_pooling_needs_two_judges_and_the_tournament(self, world: TinyWorld) -> None:
        with pytest.raises(IdentityError, match="two or more judges"):
            calibrate(read_judgements(world.judgements), judges="pooled")
        with pytest.raises(DataError, match="pooled judges need the tournament"):
            calibrate(read_judgements(world.judgements, world.lenient), judges="pooled", mode="rubric_only")


def _relabelled(judgements: JudgementSet, **family_fields) -> JudgementSet:
    """``judgements`` answered under another family: the same records under a family with ``family_fields`` changed."""
    (family,) = judgements.families.values()
    other = family.model_copy(update=family_fields)
    relabelled = tuple(
        j.model_copy(update={"family_key": other.key, "record_id": f"{j.record_id}-other"})
        for j in judgements.judgements
    )
    return JudgementSet(judgements=relabelled, families={other.key: other})


class TestFamilies:
    def test_two_tournament_families_are_refused(self, judgements: JudgementSet) -> None:
        second = _relabelled(judgements.of_stage("tournament"), judge_model="another-judge")
        with pytest.raises(IdentityError, match="2 tournament families") as caught:
            calibrate(JudgementSet.merge([judgements, second]))
        assert len(caught.value.details["families"]) == 2

    def test_pooled_judges_who_answered_different_rubrics_are_refused(self, world: TinyWorld) -> None:
        lenient = read_judgements(world.lenient)
        other_rubric = _relabelled(lenient, prompt_hash="f" * 64)
        with pytest.raises(IdentityError, match="different rubrics") as caught:
            calibrate(JudgementSet.merge([read_judgements(world.judgements), other_rubric]), judges="pooled")
        assert len(caught.value.details["rubric_keys"]) == 2


class TestLayout:
    def test_save_load_round_trips_every_file(self, fitted: Calibration, tmp_path: Path) -> None:
        out = fitted.save(tmp_path / "cal")
        names = {ITEMS_FILE, QUERIES_FILE, THETAS_FILE, COVERAGE_FILE, DIAGNOSTICS_FILE, EXTENSIONS_FILE}
        assert {p.name for p in out.iterdir()} == names | {IDENTITY_FILE}
        loaded = Calibration.load(out)
        assert loaded == fitted
        assert loaded.fingerprint == json.loads((out / ITEMS_FILE).read_text())["fingerprint"]

    def test_coverage_and_identity_say_what_was_fitted(self, fitted: Calibration) -> None:
        assert fitted.coverage.queries.model_dump() == {"rubric": 2, "tournament": 2, "calibrated": 2}
        assert fitted.coverage.uncalibrated_queries == []
        assert fitted.identity.mode == "tournament" and fitted.identity.judges == "single"
        assert sorted(fitted.identity.families) == sorted(fitted.families)

    def test_a_directory_without_the_layout_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(MissingInputError, match="not a calibration"):
            Calibration.load(tmp_path)

    def test_a_directory_without_a_store_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(MissingInputError, match="not a judgement store"):
            read_judgements(tmp_path)


def test_rubric_chunks_pool_within_a_window_never_across() -> None:
    rows = rubric_set(
        {
            ("ds", "q"): [
                [("a", "a#0", [1, 0, 0, 0, 0]), ("a", "a#1", [0, 1, 0, 0, 0]), ("b", None, [1, 1, 1, 0, 0])],
                [("a", "a#1", [0, 0, 0, 0, 1])],
            ]
        }
    )
    observations = rubric_observations(rows, tag_judges=True)["ds||q"]
    assert observations == [
        ("a", {"C1": 1, "C2": 1, "C3": 0, "C4": 0, "C5": 0}, "hand"),
        ("b", {"C1": 1, "C2": 1, "C3": 1, "C4": 0, "C5": 0}, "hand"),
        ("a", {"C1": 0, "C2": 0, "C3": 0, "C4": 0, "C5": 1}, "hand"),
    ]


def test_an_unidentifiable_fit_is_a_data_error(judgements: JudgementSet, monkeypatch: pytest.MonkeyPatch) -> None:
    """The estimators refuse a degenerate fit with a bare ValueError, which once reached users as exit 1."""

    def degenerate(*args, **kwargs):
        raise ValueError("degenerate 2PL fit: criterion C5 holds 0.9900 of the gain weight")

    monkeypatch.setattr("rcp_ndcg.calibration.fit.fit_calibration", degenerate)

    with pytest.raises(DataError, match="not identifiable.*C5"):
        calibrate(judgements, mode="rubric_only")


class TestTheBradleyTerryPenalty:
    """The refit's L2 is a fit option; the judgement store records the one the live tournament used."""

    def test_the_store_records_the_live_fits_penalty_and_the_fit_its_own(self, world: TinyWorld, tmp_path) -> None:
        from rcp_ndcg_core.irt import Priors

        from rcp_ndcg.llm import JudgementStore

        live = JudgementStore(world.judgements).identities()["tournament"]["identity"]["bt_l2"]
        stiff = calibrate(read_judgements(world.judgements), priors=Priors(bt_l2=0.05))
        default = calibrate(read_judgements(world.judgements))

        assert live == Priors().bt_l2 == default.identity.priors.bt_l2
        assert stiff.identity.priors.bt_l2 == 0.05
        assert stiff.theta_map() != default.theta_map()

    def test_a_penalty_other_than_the_live_fits_is_warned_not_refused(self, world: TinyWorld, tmp_path) -> None:
        import json

        from click.testing import CliRunner
        from rcp_ndcg_core.irt import Priors

        from rcp_ndcg.calibration import judged_bt_l2
        from rcp_ndcg.cli.calibration import calibration_group

        live = judged_bt_l2(world.judgements)
        with pytest.warns(RcpNdcgWarning, match="0.05"):
            stiff = calibrate(read_judgements(world.judgements), priors=Priors(bt_l2=0.05), judged_bt_l2=live)
        assert [w["code"] for w in stiff.warnings] == ["BT_L2_MISMATCH"]
        assert calibrate(read_judgements(world.judgements), judged_bt_l2=live).warnings == []

        stiff.save(tmp_path / "cal")
        shown = CliRunner().invoke(calibration_group, ["show", "--calibration", str(tmp_path / "cal"), "--json"])
        assert [w["code"] for w in json.loads(shown.stdout)["data"]["warnings"]] == ["BT_L2_MISMATCH"]


def test_theta_map_and_gains_share_one_key_rule(fitted: Calibration) -> None:
    """One dataset (or dataset= given): bare query ids for both views; the queries table stays namespaced."""
    thetas = fitted.theta_map()
    assert set(thetas) == set(fitted.gains()) == {"q1", "q2"}
    assert fitted.theta_map(dataset="dataset") == thetas
    assert set(fitted.queries) == {"dataset||q1", "dataset||q2"}
    with pytest.raises(DataError, match="no dataset"):
        fitted.theta_map(dataset="other")


def test_the_calibration_tables_come_out_as_frames_with_gains(fitted: Calibration) -> None:
    from rcp_ndcg_core import gain

    thetas = fitted.to_pandas()
    assert list(thetas.columns) == ["dataset", "query_id", "doc_id", "theta", "theta_se", "gain", "source"]
    row = thetas.iloc[0]
    assert row["gain"] == pytest.approx(gain(row["theta"], fitted.items))
    assert list(fitted.to_pandas("queries").columns) == ["dataset", "query_id", "tau", "alpha"]
    assert len(fitted.to_pandas("queries")) == len(fitted.queries)
    assert fitted.to_pandas("items")["criterion"].tolist() == [f"C{c}" for c in range(1, 6)]
    assert repr(fitted).startswith("Calibration(mode=tournament, 1 datasets, 2 queries, ")
