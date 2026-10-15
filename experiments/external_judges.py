"""Reproduce the external LLM-judge results from the released verdicts (no model is called).

Three judges from other model families (GLM-5.3-flash, DeepSeek-4.1-flash, Kimi-K3) compared two blinded lists
or sets, each case shown twice with the two sides swapped. A judge **decides** a case when it picks the same
side in both orders; the **majority** decides it when at least two of the three judges decide it the same way.

* Tournament-order test (``blind_comparisons/external_judges_*``): share of decided cases that favour the
  RCP-nDCG side over the Count-nDCG side, per set (S1 per-query reranker disagreements and agreement controls,
  S2 ideal top-10 lists, S3 minimal pairs), per judge and by majority; S2 also per suite and judge.
* Study 1 (``blind_comparisons/study1_*``): share of decided cases in which the judges prefer the set selected by
  the reference judge over the benchmark's qrel set, on NanoBEIR and BRIGHT.

arXiv v2 applies one exclusion rule to every judged case: a case enters an analysis only if, under the v2 fit, it
keeps the property it was selected for. Release 2 of the external-validation dataset records it per case in a column
``excluded_under_fit``: in ``external_judges_cases`` (an S1 disagreement on which both metrics now favour the same
list, an S1 control on which they now disagree, an S2 or S3 case whose RCP side changed) and in ``study1_cases`` (a
case whose qrel set has at least as much total RCP gain as the reference judge's set under the v2 fit). The script
leaves the flagged cases out. A table without the column (release 1) keeps every case.

Usage::

    python experiments/external_judges.py
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from checks import Checker, paper_values
from fetch_data import fetch_group, local_path

REPO = "rcp-ndcg-external-validation"
JUDGES = {"glm-5-3-flash": "GLM-5.3-flash", "deepseek-4-1-flash": "DeepSeek-4.1-flash", "kimi-k3": "Kimi-K3"}
# The display labels of the three BRIGHT tasks with excluded ids (fetch_data.BRIGHT_WITH_EXCLUSIONS, whose
# subset ids these labels normalise to -- pinned in tests/experiments/test_reproduction_rules.py).
BRIGHT_EXCLUSION_LABELS = ["AoPS", "LeetCode", "TheoremQA Questions"]


def kept_under_fit(cases: pd.DataFrame) -> pd.Series:
    """The cases that enter the analyses: those not flagged ``excluded_under_fit`` (release 2's exclusion rule:
    the case no longer has, under the v2 fit, the property it was selected for). A table without the column
    (release 1) keeps every case."""
    if "excluded_under_fit" not in cases.columns:
        return pd.Series(True, index=cases.index)
    return ~cases["excluded_under_fit"].fillna(False).astype(bool)


def judge_decisions(verdicts: pd.DataFrame, keys: list[str], side: str) -> pd.DataFrame:
    """Per case and judge: the side picked in both display orders, else missing.

    Args:
        verdicts: one row per (case, judge, rep) with the chosen side in column ``side``.
        keys: columns that identify a case.
        side: column holding the chosen side (a tie or an unparsed answer never decides).

    Returns:
        A frame indexed by ``keys`` with one column per judge.
    """
    wide = verdicts.pivot_table(index=[*keys, "judge"], columns="rep", values=side, aggfunc="first")
    same = wide[1].notna() & (wide[1] == wide[2])
    return wide[1].where(same).unstack("judge")


def majority(decisions: pd.DataFrame, sides: tuple[str, str]) -> pd.Series:
    """Side decided by at least two judges (a strict majority of three), else missing."""
    counts = {s: (decisions == s).sum(axis=1) for s in sides}
    out = pd.Series(None, index=decisions.index, dtype=object)
    for s in sides:
        out[counts[s] >= 2] = s
    return out


def share(picks: pd.Series, win: str) -> tuple[int, float]:
    """``(n decided, % of them equal to win)`` over the non-null picks."""
    p = picks.dropna()
    return len(p), 100 * float((p == win).mean()) if len(p) else float("nan")


def tournament_order_test(chk: Checker, paper: dict) -> None:
    cases = pd.read_parquet(local_path(REPO, "blind_comparisons/external_judges_cases.parquet"))
    verdicts = pd.read_parquet(local_path(REPO, "blind_comparisons/external_judges_verdicts.parquet"))
    verdicts["side"] = verdicts.choice_side.map({"rcp_side": "rcp", "count_side": "count"})
    keep = kept_under_fit(cases)
    if not keep.all():
        left = cases[~keep].groupby(["set", cases.kind.fillna("")]).size()
        print(
            f"== tournament-order test: {int((~keep).sum())} of {len(cases)} cases left out (excluded_under_fit: "
            + ", ".join(f"{s} {k}".strip() + f" {n}" for (s, k), n in left.items())
            + ")"
        )
    cases = cases[keep].reset_index(drop=True)
    dec = judge_decisions(verdicts, ["case_id"], "side").reindex(cases.case_id)
    cons = majority(dec, ("rcp", "count"))
    unit = np.where(cases.set == "S1", np.where(cases.kind == "control", "s1ctl", "s1"), cases.set.str.lower())
    print("== tournament-order test: % of decided cases favouring RCP-nDCG (n decided)")
    for u, p in paper.items():
        m = unit == u
        n, pct = share(cons[m], "rcp")
        print(f"  {u:<6} majority {pct:5.1f}% ({n} of {int(m.sum())} cases)")
        chk.count(f"{u} cases", p["consensus"]["n_cases"], int(m.sum()))
        chk.count(f"{u} majority n decided", p["consensus"]["n_decided"], n)
        chk.compare(f"{u} majority share (%)", p["consensus"]["rcp_share"], pct, 0.05)
        for key, ref in p.items():
            if key == "consensus":
                continue
            judge, _, suite = key.partition(".")
            mm = m & (cases.suite == suite).to_numpy() if suite else m
            n, pct = share(dec.loc[mm, JUDGES[judge]], "rcp")
            chk.count(f"{u} {key} n decided", ref["n_decided"], n)
            chk.compare(f"{u} {key} share (%)", ref["rcp_share"], pct, 0.05)


def study1(chk: Checker, paper: dict) -> None:
    cases = pd.read_parquet(local_path(REPO, "blind_comparisons/study1_cases.parquet"))
    keep = kept_under_fit(cases)
    # the original cases the rebuilt BRIGHT set replaces, counted as selected (before the exclusion rule)
    n_original = int(((cases.set == "bright_qwen3.5-397b") & cases.dataset.isin(BRIGHT_EXCLUSION_LABELS)).sum())
    if not keep.all():
        left = cases[~keep].groupby("set").size()
        print(
            f"== Study 1: {int((~keep).sum())} of {len(cases)} cases left out (excluded_under_fit: "
            + ", ".join(f"{s} {n}" for s, n in left.items())
            + ")"
        )
    cases = cases[keep].reset_index(drop=True)
    verdicts = pd.read_parquet(local_path(REPO, "blind_comparisons/study1_external_panel_verdicts.parquet"))
    verdicts["side"] = verdicts.choice_role.map({"LLM": "rcp", "GT": "qrel"})
    keys = ["set", "dataset", "query_id"]
    dec = judge_decisions(verdicts, keys, "side").reindex(pd.MultiIndex.from_frame(cases[keys]))
    cons = majority(dec, ("rcp", "qrel"))
    unanimous = dec.apply(lambda r: r.iloc[0] if r.notna().all() and r.nunique() == 1 else None, axis=1)
    s, ds = dec.index.get_level_values("set"), dec.index.get_level_values("dataset")
    affected = np.isin(ds, BRIGHT_EXCLUSION_LABELS)
    selections = {
        "nanobeir_qwen": s == "nanomteb_qwen3.5-397b",
        "nanobeir_oss": s == "nanomteb_gpt-oss-120b",
        # the rebuilt cases replace the original ones on the three BRIGHT tasks with excluded ids
        "bright_qwen_12": ((s == "bright_qwen3.5-397b") & ~affected) | (s == "bright_qwen3.5-397b_rebuilt"),
        "bright_qwen_rebuilt": s == "bright_qwen3.5-397b_rebuilt",
        "bright_oss_unaffected": (s == "bright_gpt-oss-120b") & ~affected,
    }
    print("\n== Study 1: % of decided cases preferring the reference judge's set over the qrel set")
    for key, m in selections.items():
        ref = paper[key]
        n, pct = share(cons[m], "rcp")
        print(f"  {key:<22} majority {pct:5.1f}% ({n} of {int(m.sum())} cases)")
        chk.count(f"study1 {key} majority n decided", ref["n_decided"], n)
        chk.compare(f"study1 {key} majority share (%)", ref["rcp_share"], pct, 0.05)
        if "k" in ref:
            chk.count(f"study1 {key} majority k", ref["k"], int((cons[m] == "rcp").sum()))
        if "n_cases" in ref:
            chk.count(f"study1 {key} cases", ref["n_cases"], int(m.sum()))
        if "unanimous_n" in ref:
            n_u, pct_u = share(unanimous[m], "rcp")
            chk.count(f"study1 {key} unanimous n", ref["unanimous_n"], n_u)
            chk.compare(f"study1 {key} unanimous share (%)", ref["unanimous_share"], pct_u, 0.05)
    chk.count(
        "study1 bright_qwen_rebuilt original cases",
        paper["bright_qwen_rebuilt"]["n_original"],
        n_original,
    )
    nano = selections["nanobeir_qwen"]
    per_ds = pd.DataFrame({"ds": ds[nano], "pick": cons[nano].to_numpy()}).dropna()
    rate = per_ds.groupby("ds").pick.apply(lambda x: (x == "rcp").mean())
    ref = paper["nanobeir_qwen"]
    chk.count("study1 nanobeir datasets", ref["datasets"], len(rate))
    chk.count("study1 nanobeir datasets with an RCP majority", ref["datasets_rcp_majority"], int((rate > 0.5).sum()))
    hotpot = per_ds[per_ds.ds.str.contains("HotpotQA")]
    chk.count("study1 NanoHotpotQA decided cases", ref["hotpotqa_decided_all_qrel"], len(hotpot))
    chk.count("study1 NanoHotpotQA decided cases won by the RCP set", 0, int((hotpot.pick == "rcp").sum()))


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args(argv)
    fetch_group("validation")
    pv = paper_values()
    chk = Checker("external LLM judges")
    tournament_order_test(chk, pv["tournament_order_test"])
    study1(chk, pv["study1"])
    return chk.finish()


if __name__ == "__main__":
    raise SystemExit(main())
