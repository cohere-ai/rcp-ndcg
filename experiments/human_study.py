"""Reproduce the headline numbers of the blind human study from the released tables.

Reads ``human_study/{contests,reviews,grades}.parquet`` of the ``rcp-ndcg-external-validation`` dataset and
recomputes, for the 311 analysed contests (two rerankers' top-five lists of one query, three annotators each):

* the descriptive counts (contests, queries, annotators, reviews, counted grades, distinct documents);
* the verdict table: the share of contests won by RCP-nDCG among those where exactly one metric names the
  majority verdict (main measure), on disagreements only, split by margin (both >= 0.02 or not) and without
  the zerank rerankers;
* per-metric hit rates (the verdict matches the metric's winner), overall and in the margin bands of the
  margin figure, and the margin AUCs.

The metric winners and margins (``rcp_winner_shown``, ``d_rcp_shown``, ...) are nDCG@5 over the displayed
documents with a shared ideal, as released; the majority verdict is ``human_verdict``. Rates are shares of
decided contests, with normal 95% CIs clustered by query.

Usage::

    python experiments/human_study.py
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd
from checks import Checker, paper_values
from fetch_data import fetch_group, local_path
from scipy.stats import rankdata

REPO = "rcp-ndcg-external-validation"
DECISIVE_MARGIN = 0.02


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The analysed contests, their valid reviews and their counted grades."""
    contests = pd.read_parquet(local_path(REPO, "human_study/contests.parquet"))
    contests = contests[contests.in_analysis].copy()
    contests["cluster"] = contests.suite + "|" + contests.dataset + "|" + contests.query_id
    reviews = pd.read_parquet(local_path(REPO, "human_study/reviews.parquet"))
    reviews = reviews[reviews.valid_review & reviews.contest_id.isin(contests.contest_id)]
    grades = pd.read_parquet(local_path(REPO, "human_study/grades.parquet"))
    grades = grades[grades.counted & grades.contest_id.isin(contests.contest_id)]
    return contests, reviews, grades


def verdict_class(verdict: str, rcp_winner: str, qrel_winner: str) -> str:
    """Which metric names the majority verdict: ``rcp`` (RCP-nDCG only), ``qrel`` (qrel-nDCG only), ``both``,
    ``neither``, or ``tie`` when the verdict itself is a tie."""
    if verdict not in ("A", "B"):
        return "tie"
    rcp, qrel = verdict == rcp_winner, verdict == qrel_winner
    return "both" if rcp and qrel else "rcp" if rcp else "qrel" if qrel else "neither"


def clustered_rate(hits: np.ndarray, clusters: np.ndarray) -> tuple[int, int, float, float, float]:
    """``(k, n, rate, lo, hi)`` in %: share of hits with a query-clustered normal 95% CI."""
    hits = np.asarray(hits, float)
    n = len(hits)
    if n == 0:
        return 0, 0, math.nan, math.nan, math.nan
    p = hits.mean()
    resid = pd.Series(hits - p).groupby(np.asarray(clusters)).sum().to_numpy()
    g = len(resid)
    half = 1.96 * math.sqrt(g / (g - 1) * float((resid**2).sum()) / n**2) if g > 1 else 0.0
    return int(hits.sum()), n, 100 * p, 100 * max(0.0, p - half), 100 * min(1.0, p + half)


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """Mann-Whitney AUC with tie-averaged ranks."""
    label = np.asarray(label, bool)
    n1, n0 = int(label.sum()), int((~label).sum())
    ranks = rankdata(score)
    return float((ranks[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args(argv)
    fetch_group("validation")
    paper = paper_values()["human_study"]
    c, rv, gr = load()
    chk = Checker("human study")

    print("== descriptives")
    d = paper["descriptives"]
    desc = {
        "contests": len(c),
        "queries": c.cluster.nunique(),
        "datasets": c[["suite", "dataset"]].drop_duplicates().shape[0],
        "annotators": rv.annotator.nunique(),
        "reviews": len(rv),
        "grades": len(gr),
        "distinct-docs": gr.merge(c[["contest_id", "cluster"]])[["cluster", "doc_id"]].drop_duplicates().shape[0],
        "verdict-a": int((c.human_verdict == "A").sum()),
        "verdict-b": int((c.human_verdict == "B").sum()),
        "verdict-tie": int((c.human_verdict == "tie").sum()),
        "pairs": c.apply(lambda r: tuple(sorted((r.system_a, r.system_b))), axis=1).nunique(),
        "rerankers": len(set(c.system_a) | set(c.system_b)),
    }
    rvs = rv.merge(c[["contest_id", "suite"]])
    for s in ("nanomteb", "bright", "vidore"):
        desc[f"contests.{s}"] = int((c.suite == s).sum())
        desc[f"annotators.{s}"] = rvs[rvs.suite == s].annotator.nunique()
        desc[f"reviews.{s}"] = int((rvs.suite == s).sum())

    tw, nw = c.rcp_winner_shown.to_numpy(), c.ndcg_winner_shown.to_numpy()
    v = c.human_verdict.to_numpy()
    cls = np.array([verdict_class(*t) for t in zip(v, tw, nw, strict=True)])
    decided = np.isin(v, ["A", "B"])
    flip = np.isin(tw, ["A", "B"]) & np.isin(nw, ["A", "B"]) & (tw != nw)
    desc["decided-flips"] = int((flip & decided).sum())
    for key, val in desc.items():
        chk.count(f"{key}", d[key], val)

    print("\n== verdicts (rates in %, 95% CIs clustered by query)")
    cluster = c.cluster.to_numpy()
    zerank = (c.system_a.str.startswith("zerank") | c.system_b.str.startswith("zerank")).to_numpy()
    cell = c.design_cell.to_numpy()
    # "exactly one metric names the verdict" slices
    slices = {
        "main": np.ones(len(c), bool),
        "main_without_zerank": ~zerank,
        "main_zerank_contests": zerank,
        "disagreements": flip & decided,
        "disagreements_without_zerank": flip & decided & ~zerank,
        "disagreements_both_margins_ge_0.02": cell == "discordant_decisive",
        "disagreements_a_margin_lt_0.02": cell == "discordant_marginal",
    }
    rates = {}
    for key, mask in slices.items():
        m = mask & np.isin(cls, ["rcp", "qrel"])
        rates[key] = clustered_rate(cls[m] == "rcp", cluster[m])
    # per-metric hit rates: verdict == the metric's winner, where the metric names one
    rcp_margin, qrel_margin = c.d_rcp_shown.abs().to_numpy(), c.d_ndcg_shown.abs().to_numpy()
    hit = {
        "hit_rcp": (tw, np.ones(len(c), bool)),
        "hit_qrel": (nw, np.ones(len(c), bool)),
        "band_rcp_margin_lt_0.02": (tw, rcp_margin < DECISIVE_MARGIN),
        "band_rcp_margin_ge_0.20": (tw, rcp_margin >= 0.20),
        "band_qrel_margin_ge_0.20": (nw, qrel_margin >= 0.20),
        "rcp_margin_ge_0.02": (tw, rcp_margin >= DECISIVE_MARGIN),
    }
    for key, (winner, mask) in hit.items():
        m = mask & decided & np.isin(winner, ["A", "B"])
        rates[key] = clustered_rate(v[m] == winner[m], cluster[m])

    for key, ref in paper["rates"].items():
        k, n, pct, lo, hi = rates[key]
        print(f"  {key:<36} {pct:5.1f}% ({k}/{n}) [{lo:.1f}, {hi:.1f}]")
        chk.compare(f"{key} rate (%)", ref["pct"], pct, 0.05)
        if "k" in ref:
            chk.count(f"{key} k", ref["k"], k)
        if "n" in ref:
            chk.count(f"{key} n", ref["n"], n)
        if "lo" in ref:
            chk.compare(f"{key} CI low (%)", ref["lo"], lo, 0.05)
            chk.compare(f"{key} CI high (%)", ref["hi"], hi, 0.05)

    print("\n== margin AUC (does a metric's margin predict each annotator's pick?)")
    rr = rv.merge(c[["contest_id", "d_rcp_shown", "d_ndcg_shown"]])
    rr = rr[rr.pick.isin(["A", "B"]) & rr.d_rcp_shown.notna() & rr.d_ndcg_shown.notna()]
    picks_a = (rr.pick == "A").to_numpy()
    ref = paper["margin_auc"]
    chk.count("margin AUC picks", ref["picks"], len(rr))
    chk.compare("margin AUC, RCP-nDCG", ref["rcp"], auc(rr.d_rcp_shown.to_numpy(), picks_a), 5e-4)
    chk.compare("margin AUC, qrel-nDCG", ref["qrel"], auc(rr.d_ndcg_shown.to_numpy(), picks_a), 5e-4)
    return chk.finish()


if __name__ == "__main__":
    raise SystemExit(main())
