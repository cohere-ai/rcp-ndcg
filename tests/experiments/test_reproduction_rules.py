"""The paper-reproduction rules in ``experiments/`` on small synthetic inputs (no network, no data download).

The scoring protocol itself is tested in ``tests/core/test_protocol.py``.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]

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
    dev = checks.KnownDeviation(0.08, "reason", cells=1)
    chk = checks.Checker("t", (dev,))
    assert chk.compare("exact", 87.2, 87.23, 0.05)
    assert chk.compare("known", 87.9, 87.848, 0.05, known=dev)
    assert chk.finish() == 0
    assert not chk.compare("off", 87.9, 87.7, 0.05, known=dev)
    assert chk.finish() == 1
    assert "FAIL" in capsys.readouterr().out


def test_a_documented_deviation_is_bounded_to_its_population(experiment):
    """The known-deviation mechanism bounds the population, not just each cell: a documented deviation that
    ``cells`` compared values may take may not cover a 31st drifting cell (the sweep's reproduction, case 1),
    and a population that does not materialise is drift too."""
    checks = experiment("checks")
    dev = checks.KnownDeviation(0.05, "documented", cells=2)
    ok = checks.Checker("t", (dev,))
    assert ok.compare("within tolerance", 10.0, 10.01, 0.05, known=dev), "a row that matched took nothing"
    for i in range(2):
        assert ok.compare(f"c{i}", 10.0, 10.04, 0.001, known=dev)
    assert ok.finish() == 0
    crowded = checks.Checker("t", (dev,))
    for i in range(3):
        crowded.compare(f"c{i}", 10.0, 10.04, 0.001, known=dev)
    assert crowded.finish() == 1, "three drifting cells may not hide inside a documented population of two"
    thin = checks.Checker("t", (dev,))
    thin.compare("c0", 10.0, 10.04, 0.001, known=dev)
    assert thin.finish() == 1, "a population that did not materialise means the documentation is stale"
    undocumented = checks.Checker("t")
    undocumented.compare("c0", 10.0, 10.04, 0.001, known=dev)
    assert undocumented.finish() == 1, "a deviation outside every documented population must fail"


def test_the_known_deviation_populations_are_the_documented_ones(experiment):
    """The code table is ``experiments/README.md``'s \"Known deviations\": NanoBEIR 5 FEVER + 2 Quora + 2
    NFCorpus + 1 HotpotQA cells within 0.08, BRIGHT 13 TheoremQA Theorems cells within 2.6 and 12 of the 14 qrel
    means within 0.25 -- 35 rows in total, and no wildcard entry that would cover anything else."""
    lb = experiment("leaderboards")
    populations = {(s, m, col): (dev.bound, dev.cells) for (s, m, col), dev in lb.KNOWN.items()}
    assert populations == {
        ("nanobeir", "rcp_ndcg10", "NanoFEVERRetrieval"): (0.08, 5),
        ("nanobeir", "rcp_ndcg10", "NanoQuoraRetrieval"): (0.08, 2),
        ("nanobeir", "rcp_ndcg10", "NanoNFCorpusRetrieval"): (0.08, 2),
        ("nanobeir", "rcp_ndcg10", "NanoHotpotQARetrieval"): (0.08, 1),
        ("bright", "qrel_ndcg10", "theoremqa_theorems"): (2.6, 13),
        ("bright", "qrel_ndcg10", "mean"): (0.25, 12),
    }


def test_check_trecdl_refuses_a_ragged_reranker_matrix(experiment):
    """A (query, reranker) score missing from the pool panel is an error naming the suite, not a NaN that
    "not separated" and "disagrees" swallow before a later count drifts (the sweep's reproduction, case 2)."""
    lb = experiment("leaderboards")
    rows = []
    for dataset in ("trec_dl_2019", "trec_dl_2020"):
        for query in ("q1", "q2", "q3"):
            for model in ("a", "b"):
                if (dataset, query, model) == ("trec_dl_2019", "q3", "b"):
                    continue  # the ragged hole
                rows.append(
                    {
                        "suite": "trecdl",
                        "dataset": dataset,
                        "reranker": model,
                        "query_id": query,
                        "qrel_ndcg10": 0.5,
                        "rcp_ndcg10": 0.4,
                    }
                )
    paper = {
        year: {
            "queries": 3,
            "pairs": 1,
            "separated_qrel_linear": 0,
            "separated_rcp": 0,
            "agree_with_nist": "0/0",
            "kendall_rcp_vs_nist_14": 0.0,
        }
        for year in ("dl19", "dl20")
    }
    checks = experiment("checks")
    with pytest.raises(ValueError, match="trec_dl_2019"):
        lb.check_trecdl(pd.DataFrame(rows), paper, checks.Checker("t"))


def test_the_bright_exclusion_tasks_are_one_concept_with_one_definition(experiment):
    """``BRIGHT_WITH_EXCLUSIONS`` (the subset ids) is defined once, in ``fetch_data``; the external-judges script
    holds the display labels of the same three tasks and they correspond one to one."""
    fetch_data = experiment("fetch_data")
    external_judges = experiment("external_judges")
    definitions = 0
    for name in ("fetch_data", "external_judges"):
        source = (ROOT / "experiments" / f"{name}.py").read_text(encoding="utf-8")
        definitions += len(re.findall(r"^BRIGHT_WITH_EXCLUSIONS\s*=", source, re.M))
    assert definitions == 1, "BRIGHT_WITH_EXCLUSIONS must be defined exactly once"
    ids = fetch_data.BRIGHT_WITH_EXCLUSIONS
    labels = external_judges.BRIGHT_EXCLUSION_LABELS
    assert [re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_") for label in labels] == ids


def test_the_two_setup_snippets_give_the_same_install_commands():
    """Experiments install the checkout's own core first (``pip install -e .`` alone would resolve
    ``rcp-ndcg-core==0.0.1`` from PyPI): ``experiments/README.md`` and ``REPRODUCIBILITY.md`` say the same."""

    def install_commands(text: str) -> list[str]:
        return [
            line.split("#", 1)[0].strip()
            for block in re.findall(r"```bash\n(.*?)```", text, re.S)
            for line in block.splitlines()
            if line.strip().startswith("pip install")
        ]

    readme = install_commands((ROOT / "experiments" / "README.md").read_text(encoding="utf-8"))
    repro = install_commands((ROOT / "REPRODUCIBILITY.md").read_text(encoding="utf-8"))
    assert readme and readme == repro


def test_paper_values_cover_the_three_leaderboard_tables(experiment):
    checks = experiment("checks")
    lb = checks.paper_values()["leaderboards"]
    for suite, n_datasets in (("nanobeir", 13), ("bright", 12), ("vidore", 8)):
        for metric in ("qrel_ndcg10", "rcp_ndcg10"):
            table = lb[suite][metric]
            assert len(table) == 14
            assert all(len(cells) == n_datasets + 1 for cells in table.values())
