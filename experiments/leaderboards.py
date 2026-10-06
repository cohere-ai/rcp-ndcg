"""Reproduce the reranker leaderboards (qrel-nDCG@10 and RCP-nDCG@10) from the public data.

* NanoBEIR, BRIGHT and ViDoRe v3 (native-language questions): every cell of the paper's three leaderboard tables,
  14 rerankers x (datasets + mean), for both metrics. Judge: Qwen3.5-397B.
* TREC-DL 2019 and 2020 (pool panel, judge Qwen3.6-27B, calibration CAL-A): the reranker-pair table (pairs that
  each metric separates, agreement with NIST on the NIST-decisive pairs) and the Kendall correlation of the two
  leaderboards over the 14 rerankers.

Inputs are the benchmark datasets' ``qrels`` (human grades plus RCP gains), ``top_ranked`` (judged pools),
``excluded`` (BRIGHT) and ``provenance/runs`` (the stored reranker scores); no model is called.

Usage::

    python experiments/leaderboards.py [--suite nanobeir bright vidore trecdl] [--out per_query.csv]

Prints paper vs reproduced for every compared value and exits 1 if any differs beyond its tolerance.
"""

from __future__ import annotations

import argparse
import itertools
from collections.abc import Iterator

import numpy as np
import pandas as pd
from checks import Checker, KnownDeviation, paper_values
from fetch_data import BRIGHT_SUBSETS, NANOBEIR_SUBSETS, TRECDL_SUBSETS, VIDORE_NATIVE, fetch_group, local_path
from rcp_ndcg_core.protocol import candidate_docs, score_query
from scipy.stats import kendalltau, ttest_rel

# Hub model id -> the paper's short name.
RERANKERS = {
    "ctxl-rerank-v2-instruct-multilingual-1b": "CTXL-RR-1B",
    "ctxl-rerank-v2-instruct-multilingual-2b": "CTXL-RR-2B",
    "ctxl-rerank-v2-instruct-multilingual-6b": "CTXL-RR-6B",
    "jina-reranker-v3": "Jina-RR-v3",
    "Qwen3-Reranker-0.6B": "Qwen3-RR-0.6B",
    "Qwen3-Reranker-4B": "Qwen3-RR-4B",
    "Qwen3-Reranker-8B": "Qwen3-RR-8B",
    "rerank-v4.0-fast": "RR4-Fast",
    "rerank-v4.0-pro": "RR4-Pro",
    "rerank-2.5-lite": "Voyage2.5-lt",
    "rerank-2.5": "Voyage2.5",
    "zerank-1-small": "zerank-1-sm",
    "zerank-1": "zerank-1",
    "zerank-2": "zerank-2",
}

# Tolerance for a table cell, in nDCG points: the printed rounding (0.05) plus one unit of float noise.
CELL_TOL = 0.05 + 1e-6

_NANOBEIR_GAINS = (
    "hub gains come from a refit that differs from the paper's by <= 0.0065 in gain; 10 cells move <= 0.07"
)
_THQA_GOLD_IDS = "paper used the gold ids of a later BRIGHT revision on 27 of 58 ThQA-T queries; hub keeps a75a0eb4"
_MEAN_CARRIES = "carries the ThQA-T qrel difference into the mean"

# Documented deviations: (suite, metric, dataset or "mean") -> (bound in points, reason, documented cells).
# The cell populations are ``experiments/README.md``'s "Known deviations": NanoBEIR 5 FEVER + 2 Quora + 2
# NFCorpus + 1 HotpotQA cells of the 196, BRIGHT 13 of the 14 TheoremQA Theorems qrel cells and 12 of the 14 qrel
# means -- 35 rows in total (`run_all.py`). No other cell may deviate at all, and every count here is exact:
# ``Checker.finish`` fails the run when the populations do not materialise as documented.
KNOWN = {
    ("nanobeir", "rcp_ndcg10", "NanoFEVERRetrieval"): KnownDeviation(0.08, _NANOBEIR_GAINS, 5),
    ("nanobeir", "rcp_ndcg10", "NanoQuoraRetrieval"): KnownDeviation(0.08, _NANOBEIR_GAINS, 2),
    ("nanobeir", "rcp_ndcg10", "NanoNFCorpusRetrieval"): KnownDeviation(0.08, _NANOBEIR_GAINS, 2),
    ("nanobeir", "rcp_ndcg10", "NanoHotpotQARetrieval"): KnownDeviation(0.08, _NANOBEIR_GAINS, 1),
    ("bright", "qrel_ndcg10", "theoremqa_theorems"): KnownDeviation(2.6, _THQA_GOLD_IDS, 13),
    ("bright", "qrel_ndcg10", "mean"): KnownDeviation(0.25, _MEAN_CARRIES, 12),
}


def _runs(repo: str, subset: str, keep_queries: set[str] | None = None) -> pd.DataFrame:
    """Stored reranker scores of one subset (judge strategies dropped, NaN scores removed)."""
    cols = ["model", "query-id", "corpus-id", "score"]
    filters = [("query-id", "in", sorted(keep_queries))] if keep_queries is not None else None
    df = pd.read_parquet(local_path(repo, f"provenance/runs/{subset}.parquet"), columns=cols, filters=filters)
    df = df[df.model.isin(list(RERANKERS)) & df.score.notna()]
    return df


def _qrels(repo: str, subset: str) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """``(positives, gains)`` per query: human grades > 0, and RCP gains of the judged documents."""
    q = pd.read_parquet(local_path(repo, f"{subset}/qrels.parquet"))
    pos = q[q.score > 0]
    judged = q[q.theta.notna()]
    positives = {k: dict(zip(g["corpus-id"], g.score.astype(float), strict=True)) for k, g in pos.groupby("query-id")}
    gains = {k: dict(zip(g["corpus-id"], g.gain, strict=True)) for k, g in judged.groupby("query-id")}
    return positives, gains


def _pools(repo: str, subset: str) -> dict[str, list[str]]:
    t = pd.read_parquet(local_path(repo, f"{subset}/top_ranked.parquet"))
    return {q: list(ids) for q, ids in zip(t["query-id"], t["corpus-ids"], strict=True)}


def _score_subset(suite, dataset, runs, positives, gains, pools=None, excluded=None) -> Iterator[tuple]:
    excluded = excluded or {}
    for (model, qid), g in runs.groupby(["model", "query-id"], sort=False):
        if qid not in gains:
            continue
        scored = dict(zip(g["corpus-id"], g.score, strict=True))
        pool = pools[qid] if pools is not None else None
        cands = candidate_docs(suite, scored, candidates=pool, excluded=excluded.get(qid, ()))
        q_gains = gains[qid]
        if suite == "nanobeir":  # the RCP ideal ranking is taken over the candidates (self-docs removed)
            q_gains = {d: v for d, v in q_gains.items() if d in set(pool)}
        ranked = {d: scored[d] for d in cands}
        qrel = score_query(ranked, positives.get(qid, {}), protocol=suite, metric="qrel_ndcg", candidates=pool)
        rcp = score_query(ranked, q_gains, protocol=suite, candidates=pool)
        yield suite, dataset, RERANKERS[model], qid, qrel, rcp


def per_query(suite: str) -> pd.DataFrame:
    """Per-query qrel-nDCG@10 and RCP-nDCG@10 of the 14 rerankers on one suite."""
    rows: list[tuple] = []
    if suite == "nanobeir":
        repo = "rcp-ndcg-nanobeir"
        for s in NANOBEIR_SUBSETS:
            positives, gains = _qrels(repo, s)
            rows += _score_subset(suite, s, _runs(repo, s), positives, gains, pools=_pools(repo, s))
    elif suite == "bright":
        repo = "rcp-ndcg-bright"
        for s in BRIGHT_SUBSETS:
            positives, gains = _qrels(repo, s)
            excluded = {}
            path = local_path(repo, f"{s}/excluded.parquet")
            if path.is_file():
                e = pd.read_parquet(path)
                excluded = {q: set(ids) for q, ids in zip(e["query-id"], e["excluded-corpus-ids"], strict=True)}
            rows += _score_subset(suite, s, _runs(repo, s), positives, gains, excluded=excluded)
    elif suite == "vidore":
        repo = "rcp-ndcg-vidore-v3"
        for domain, lang in VIDORE_NATIVE.items():
            positives, gains = _qrels(repo, f"{domain}__{lang}")
            pools = _pools(repo, f"{domain}__{lang}")
            runs = _runs(repo, domain, keep_queries=set(gains))
            rows += _score_subset(suite, domain, runs, positives, gains, pools=pools)
    elif suite == "trecdl":
        repo = "rcp-ndcg-trecdl"
        for s in TRECDL_SUBSETS:
            positives, gains = _qrels(repo, s)
            rows += _score_subset(suite, s, _runs(repo, s), positives, gains, pools=_pools(repo, s))
    else:
        raise ValueError(suite)
    return pd.DataFrame(rows, columns=["suite", "dataset", "reranker", "query_id", "qrel_ndcg10", "rcp_ndcg10"])


def check_table(suite: str, pq: pd.DataFrame, paper: dict, chk: Checker) -> None:
    """Compare every cell (per-dataset means and the macro mean, in %) with the paper's table."""
    per_ds = pq.groupby(["reranker", "dataset"])[["qrel_ndcg10", "rcp_ndcg10"]].mean() * 100
    print(f"\n== {suite}: {pq.query_id.groupby(pq.dataset).nunique().sum()} queries, mean over datasets (%)")
    print(f"  {'reranker':<14} {'qrel paper':>10} {'reproduced':>10}   {'RCP paper':>9} {'reproduced':>10}")
    for metric in ("qrel_ndcg10", "rcp_ndcg10"):
        for rr, cells in paper[suite][metric].items():
            for col, pv in cells.items():
                val = per_ds.loc[rr, metric].mean() if col == "mean" else per_ds.loc[(rr, col), metric]
                chk.compare(
                    f"{suite} {metric} {rr} {col}",
                    pv,
                    float(val),
                    CELL_TOL,
                    known=KNOWN.get((suite, metric, col)),
                    quiet=True,
                )
    for rr in paper[suite]["rcp_ndcg10"]:
        q, r = per_ds.loc[rr].mean()
        pq_, pr_ = paper[suite]["qrel_ndcg10"][rr]["mean"], paper[suite]["rcp_ndcg10"][rr]["mean"]
        print(f"  {rr:<14} {pq_:>10.1f} {q:>10.2f}   {pr_:>9.1f} {r:>10.2f}")


def check_trecdl(pq: pd.DataFrame, paper: dict, chk: Checker) -> None:
    """Reranker pairs separated by a paired t-test (p < .05) and agreement with NIST; Kendall tau of the means."""
    print("\n== trecdl (pool panel, 14 rerankers)")
    for year, ds in (("dl19", "trec_dl_2019"), ("dl20", "trec_dl_2020")):
        sub = pq[pq.dataset == ds]
        mats = {m: sub.pivot(index="query_id", columns="reranker", values=m) for m in ("qrel_ndcg10", "rcp_ndcg10")}
        for metric, mat in mats.items():
            if not mat.notna().all().all():
                # A doubled (query, reranker) run or a dropped score leaves NaN, which the t-test counts as "not
                # separated", sign(NaN) == NaN counts as disagreement, and DataFrame.mean() skips silently.
                holes = sorted(
                    (str(mat.index[i]), str(mat.columns[j]))
                    for i, j in zip(*np.where(mat.isna().to_numpy()), strict=True)
                )
                raise ValueError(
                    f"{ds}: the {metric} (query, reranker) matrix is ragged; missing scores for {holes}: "
                    "every pool query must carry one score per reranker"
                )
        systems = sorted(mats["rcp_ndcg10"].columns)
        sep = {"qrel_ndcg10": 0, "rcp_ndcg10": 0}
        decisive = agree = 0
        for a, b in itertools.combinations(systems, 2):
            sign, separated = {}, {}
            for m, mat in mats.items():
                x, y = mat[a].to_numpy(), mat[b].to_numpy()
                sign[m] = np.sign(np.mean(x - y))
                separated[m] = ttest_rel(x, y).pvalue < 0.05
                sep[m] += separated[m]
            if separated["qrel_ndcg10"]:  # NIST-decisive pair
                decisive += 1
                agree += sign["rcp_ndcg10"] == sign["qrel_ndcg10"]
        means = {m: mat.mean() for m, mat in mats.items()}
        tau = kendalltau(means["rcp_ndcg10"][systems], means["qrel_ndcg10"][systems]).statistic
        p = paper[year]
        n_pairs = len(systems) * (len(systems) - 1) // 2
        chk.count(f"trecdl {year} queries", p["queries"], sub.query_id.nunique())
        chk.count(f"trecdl {year} reranker pairs", p["pairs"], n_pairs)
        chk.count(
            f"trecdl {year} pairs separated by qrel-nDCG@10 (NIST)", p["separated_qrel_linear"], sep["qrel_ndcg10"]
        )
        chk.count(f"trecdl {year} pairs separated by RCP-nDCG@10", p["separated_rcp"], sep["rcp_ndcg10"])
        k_paper, n_paper = (int(x) for x in p["agree_with_nist"].split("/"))
        chk.count(f"trecdl {year} NIST-decisive pairs", n_paper, decisive)
        chk.count(f"trecdl {year} RCP-nDCG@10 agrees with NIST", k_paper, int(agree))
        chk.compare(f"trecdl {year} Kendall tau (RCP vs NIST, 14 rerankers)", p["kendall_rcp_vs_nist_14"], tau, 5e-4)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", nargs="*", default=["nanobeir", "bright", "vidore", "trecdl"])
    ap.add_argument("--out", help="write the per-query scores to this CSV")
    args = ap.parse_args(argv)
    fetch_group("leaderboards")
    pv = paper_values()
    chk = Checker("leaderboards", tuple(KNOWN.values()))
    frames = []
    for suite in args.suite:
        pq = per_query(suite)
        frames.append(pq)
        if suite == "trecdl":
            check_trecdl(pq, pv["trecdl_pairs"], chk)
        else:
            check_table(suite, pq, pv["leaderboards"], chk)
    if args.out:
        pd.concat(frames).to_csv(args.out, index=False)
    return chk.finish()


if __name__ == "__main__":
    raise SystemExit(main())
