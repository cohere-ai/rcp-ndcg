"""The paper-reproduction rules in ``experiments/`` on small synthetic inputs (no network, no data download).

The scoring protocol itself is tested in ``tests/core/test_protocol.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- human study


def test_verdict_class_names_the_metric_that_matches_the_verdict(experiment):
    human_study = experiment("human_study")
    assert human_study.verdict_class("A", "A", "B") == "rcp"
    assert human_study.verdict_class("B", "A", "B") == "qrel"
    assert human_study.verdict_class("A", "A", "tie") == "rcp"
    assert human_study.verdict_class("A", "A", "A") == "both"
    assert human_study.verdict_class("tie", "A", "B") == "tie"


def test_clustered_rate_widens_the_interval_for_correlated_clusters(experiment):
    human_study = experiment("human_study")
    hits = np.array([1, 1, 0, 0] * 5)
    k, n, pct, lo, hi = human_study.clustered_rate(hits, np.arange(20))
    _, _, _, lo_c, hi_c = human_study.clustered_rate(hits, np.arange(20) // 2)  # pairs share a query
    assert (k, n, pct) == (10, 20, 50.0)
    assert hi_c - lo_c > hi - lo


# ---------------------------------------------------------------- external judges


def test_a_judge_decides_only_when_both_display_orders_agree_and_majority_needs_two(experiment):
    external_judges = experiment("external_judges")
    rows = []
    for case, picks in {
        "c1": [("rcp", "rcp"), ("rcp", "rcp"), ("count", "count")],
        "c2": [("rcp", "count"), ("rcp", None), ("count", "count")],
    }.items():
        for judge, (r1, r2) in zip(("j1", "j2", "j3"), picks, strict=True):
            rows += [
                {"case_id": case, "judge": judge, "rep": 1, "side": r1},
                {"case_id": case, "judge": judge, "rep": 2, "side": r2},
            ]
    dec = external_judges.judge_decisions(pd.DataFrame(rows), ["case_id"], "side")
    assert dec.loc["c1"].tolist() == ["rcp", "rcp", "count"]
    assert dec.loc["c2"].isna().tolist() == [True, True, False]
    cons = external_judges.majority(dec, ("rcp", "count"))
    assert cons["c1"] == "rcp"
    assert pd.isna(cons["c2"])


# ---------------------------------------------------------------- checker and paper values


def test_checker_fails_outside_tolerance_and_accepts_documented_deviations(capsys, experiment):
    checks = experiment("checks")
    chk = checks.Checker("t")
    assert chk.compare("exact", 87.2, 87.23, 0.05)
    assert chk.compare("known", 87.9, 87.848, 0.05, known=(0.08, "reason"))
    assert chk.finish() == 0
    assert not chk.compare("off", 87.9, 87.7, 0.05, known=(0.08, "reason"))
    assert chk.finish() == 1
    assert "FAIL" in capsys.readouterr().out


def test_paper_values_cover_the_three_leaderboard_tables(experiment):
    checks = experiment("checks")
    lb = checks.paper_values()["leaderboards"]
    for suite, n_datasets in (("nanobeir", 13), ("bright", 12), ("vidore", 8)):
        for metric in ("qrel_ndcg10", "rcp_ndcg10"):
            table = lb[suite][metric]
            assert len(table) == 14
            assert all(len(cells) == n_datasets + 1 for cells in table.values())
