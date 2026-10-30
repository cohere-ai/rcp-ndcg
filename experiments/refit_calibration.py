"""Refit RCP's 2PL calibration from a pinned public NanoBEIR judgment subset.

This is a bounded reproduction *driver*, not the paper's full-suite fit. It
calls the library's existing fit rather than implementing a second estimator.
No model endpoint or credential is used.

Usage::

    python -m experiments.refit_calibration --subset NanoSciFactRetrieval

The default subset is intentionally small enough to check data joins and fit
diagnostics locally. ``--all-nanobeir`` fits one shared model over all 13
subsets, the population needed to compare against the paper's suite-level fit.
Fresh fits may differ numerically from the released gains because the paper's
fit and this library are not bit-deterministic across all environments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
from rcp_ndcg_core.irt import fit_calibration
from scipy.optimize import minimize
from scipy.special import expit

from experiments.fetch_data import DATASETS, NANOBEIR_SUBSETS, fetch, local_path

__all__ = ["load_subset", "split_documents", "main"]

REPO = "rcp-ndcg-nanobeir"
TOURNAMENT_MODEL = "Qwen3.5-397B-A17B-NVFP4 (tournament, judge)"
CRITERIA = tuple(f"C{k}" for k in range(1, 6))


def load_subset(subset: str) -> tuple[dict[str, list[tuple[str, dict[str, int]]]], dict[str, dict[str, float]]]:
    """Read one pinned NanoBEIR subset as (rubric observations, tournament scores).

    Query ids are prefixed with the subset name so ids shared between datasets
    cannot be pooled by mistake. Each Stage B row is one placement; repeated
    documents are kept because the fit consumes every judgment placement.
    """
    if subset not in NANOBEIR_SUBSETS:
        raise ValueError(f"unknown NanoBEIR subset {subset!r}")
    run_file = f"provenance/runs/{subset}.parquet"
    rubric_file = f"provenance/judge_criteria/{subset}.parquet"
    fetch(REPO, [run_file, rubric_file])
    runs = pd.read_parquet(local_path(REPO, run_file))
    rubric = pd.read_parquet(local_path(REPO, rubric_file))
    tournament = runs.loc[(runs["model"] == TOURNAMENT_MODEL) & runs["is_judge"]]
    if tournament.empty:
        raise ValueError(f"{subset}: tournament judge scores missing")
    if tournament.duplicated(["query-id", "corpus-id"]).any():
        raise ValueError(f"{subset}: duplicate tournament scores")
    if not tournament["score"].map(math.isfinite).all():
        raise ValueError(f"{subset}: non-finite tournament score")
    if rubric.empty or rubric[list(CRITERIA)].isna().any().any():
        raise ValueError(f"{subset}: missing rubric verdicts")
    for criterion in CRITERIA:
        if not rubric[criterion].isin((0, 1)).all():
            raise ValueError(f"{subset}: {criterion} has a verdict outside 0/1")

    bt: dict[str, dict[str, float]] = defaultdict(dict)
    for query, doc, score in tournament[["query-id", "corpus-id", "score"]].itertuples(index=False, name=None):
        bt[f"{subset}||{query}"][str(doc)] = float(score)

    observations: dict[str, list[tuple[str, dict[str, int]]]] = defaultdict(list)
    for row in rubric[["query-id", "corpus-id", *CRITERIA]].itertuples(index=False, name=None):
        query, doc, *values = row
        observations[f"{subset}||{query}"].append((str(doc), dict(zip(CRITERIA, map(int, values), strict=True))))
    if not set(observations).issubset(bt):
        missing = sorted(set(observations) - set(bt))
        raise ValueError(f"{subset}: rubric queries without tournament scores: {missing[:3]}")
    missing_docs = [(query, doc) for query, rows in observations.items() for doc, _ in rows if doc not in bt[query]]
    if missing_docs:
        raise ValueError(f"{subset}: rubric documents without tournament scores: {missing_docs[:3]}")
    return dict(observations), dict(bt)


def split_documents(
    observations: dict[str, list[tuple[str, dict[str, int]]]],
    *,
    fraction: float = 0.2,
) -> tuple[dict[str, list[tuple[str, dict[str, int]]]], dict[str, list[tuple[str, dict[str, int]]]]]:
    """Hold out complete query-document identities, including all repeated Stage B placements.

    The split is deterministic and keeps at least one document in each side of
    every query. The estimand is new documents within queries seen during the
    fit, with the Stage A tournament scores already known.
    """
    if not 0 < fraction < 1:
        raise ValueError("holdout fraction must be between zero and one")
    train, heldout = {}, {}
    for query, rows in observations.items():
        documents = {doc for doc, _ in rows}
        if len(documents) < 2:
            raise ValueError(f"{query}: need two or more judged documents for a holdout")
        ordered = sorted(
            documents,
            key=lambda doc: (hashlib.sha256(f"{query}\x1f{doc}".encode()).digest(), doc),
        )
        heldout_count = max(1, min(len(ordered) - 1, round(len(ordered) * fraction)))
        selected = set(ordered[:heldout_count])
        train[query] = [(doc, verdicts) for doc, verdicts in rows if doc not in selected]
        heldout[query] = [(doc, verdicts) for doc, verdicts in rows if doc in selected]
    return train, heldout


def _reliability_summary(fit: Any, rows: dict[str, list[tuple[str, dict[str, int]]]]) -> dict[str, Any]:
    """Report the existing library's per-criterion Brier/ECE on Stage B verdicts."""
    from rcp_ndcg.calibration.diagnostics import fit_diagnostics

    reliability = fit_diagnostics(
        fit.items,
        fit.thetas,
        rows,
        family_of_judge={"public-qwen": "nanobeir"},
        single_judge="public-qwen",
    )

    def compact(value: Any) -> dict[str, float | int | None]:
        return {"count": value.count, "brier": value.brier, "ece_10_bins": value.ece}

    return {
        "overall": compact(reliability.overall),
        "per_criterion": {criterion: compact(value) for criterion, value in reliability.per_criterion.items()},
    }


def _constant_baseline(
    train: dict[str, list[tuple[str, dict[str, int]]]],
    heldout: dict[str, list[tuple[str, dict[str, int]]]],
) -> dict[str, Any]:
    """Weak reference: each criterion's training pass rate, with no Stage A score."""
    from rcp_ndcg.calibration.diagnostics import reliability

    train_rows = [verdicts for rows in train.values() for _, verdicts in rows]
    heldout_rows = [verdicts for rows in heldout.values() for _, verdicts in rows]
    rates = {criterion: sum(row[criterion] for row in train_rows) / len(train_rows) for criterion in CRITERIA}
    per_criterion = {
        criterion: reliability(
            [rates[criterion]] * len(heldout_rows),
            [row[criterion] for row in heldout_rows],
        )
        for criterion in CRITERIA
    }
    pooled = reliability(
        [rates[criterion] for row in heldout_rows for criterion in CRITERIA],
        [row[criterion] for row in heldout_rows for criterion in CRITERIA],
    )
    return {
        "description": "training-set criterion pass rates; no Stage A score",
        "overall": {"count": pooled.count, "brier": pooled.brier, "ece_10_bins": pooled.ece},
        "per_criterion": {
            criterion: {
                "train_pass_rate": rates[criterion],
                "count": value.count,
                "brier": value.brier,
                "ece_10_bins": value.ece,
            }
            for criterion, value in per_criterion.items()
        },
    }


def _logistic_baseline(
    train: dict[str, list[tuple[str, dict[str, int]]]],
    heldout: dict[str, list[tuple[str, dict[str, int]]]],
    bt_scores: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Fit a pooled per-criterion logistic curve on raw Stage A scores only.

    Unlike the 2PL fit this has no per-query affine parameters. It uses the
    same training placements and held-out documents as the 2PL diagnostic.
    """
    from rcp_ndcg.calibration.diagnostics import reliability

    train_pairs = [(bt_scores[query][doc], verdicts) for query, rows in train.items() for doc, verdicts in rows]
    heldout_pairs = [(bt_scores[query][doc], verdicts) for query, rows in heldout.items() for doc, verdicts in rows]
    x_train = np.asarray([score for score, _ in train_pairs], dtype=float)
    x_heldout = np.asarray([score for score, _ in heldout_pairs], dtype=float)
    report = {}
    pooled_predictions: list[float] = []
    pooled_observed: list[int] = []
    for criterion in CRITERIA:
        y_train = np.asarray([verdicts[criterion] for _, verdicts in train_pairs], dtype=float)
        y_heldout = [verdicts[criterion] for _, verdicts in heldout_pairs]
        rate = float(np.clip(y_train.mean(), 1e-6, 1 - 1e-6))
        initial = np.array([math.log(rate / (1 - rate)), 1.0])

        def loss_and_gradient(parameters: np.ndarray, y: np.ndarray = y_train) -> tuple[float, np.ndarray]:
            logits = parameters[0] + parameters[1] * x_train
            residual = expit(logits) - y
            loss = float(np.mean(np.logaddexp(0.0, logits) - y * logits))
            gradient = np.array([residual.mean(), (residual * x_train).mean()])
            return loss, gradient

        result = minimize(loss_and_gradient, initial, jac=True, method="L-BFGS-B")
        if not result.success:
            raise ValueError(f"raw-score logistic baseline failed for {criterion}: {result.message}")
        predicted = expit(result.x[0] + result.x[1] * x_heldout)
        summary = reliability(predicted, y_heldout)
        pooled_predictions.extend(predicted.tolist())
        pooled_observed.extend(y_heldout)
        report[criterion] = {
            "intercept": float(result.x[0]),
            "raw_bt_slope": float(result.x[1]),
            "count": summary.count,
            "brier": summary.brier,
            "ece_10_bins": summary.ece,
        }
    pooled = reliability(pooled_predictions, pooled_observed)
    return {
        "description": "one pooled logistic model per criterion on raw Stage A score; no per-query affine map",
        "overall": {"count": pooled.count, "brier": pooled.brier, "ece_10_bins": pooled.ece},
        "per_criterion": report,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--subset", choices=NANOBEIR_SUBSETS)
    source.add_argument("--all-nanobeir", action="store_true")
    parser.add_argument("--holdout", action="store_true", help="fit 80% of judged documents per query and score 20%")
    args = parser.parse_args(argv)

    subsets = NANOBEIR_SUBSETS if args.all_nanobeir else [args.subset]
    observations: dict[str, list[tuple[str, dict[str, int]]]] = {}
    bt_scores: dict[str, dict[str, float]] = {}
    for subset in subsets:
        rows, scores = load_subset(subset)
        observations.update(rows)
        bt_scores.update(scores)

    train, heldout = split_documents(observations) if args.holdout else (observations, None)
    fit = fit_calibration(train, mode="tournament", bt_scores=bt_scores)
    if fit.diagnostics.skipped_observations or fit.diagnostics.skipped_queries:
        raise ValueError(f"fit skipped input unexpectedly: {fit.diagnostics.model_dump()}")
    report: dict[str, Any] = {
        "source": REPO,
        "revision": DATASETS[REPO],
        "subsets": subsets,
        "fit_mode": fit.mode,
        "queries": len(bt_scores),
        "documents_with_tournament_score": sum(map(len, bt_scores.values())),
        "rubric_observations": sum(map(len, observations.values())),
        "items": {
            "criteria": fit.items.criteria,
            "gamma": list(fit.items.gamma),
            "beta": list(fit.items.beta),
        },
        "fit_diagnostics": fit.diagnostics.model_dump(mode="json"),
    }
    if heldout is not None:
        report["evaluation"] = {
            "scope": "new judged documents within fitted queries; fixed public Stage A scores",
            "split": "deterministic SHA-256 by (query-id, corpus-id); all placements for a document stay together",
            "train_documents": sum(len({doc for doc, _ in rows}) for rows in train.values()),
            "heldout_documents": sum(len({doc for doc, _ in rows}) for rows in heldout.values()),
            "train_placements": sum(map(len, train.values())),
            "heldout_placements": sum(map(len, heldout.values())),
            "train_in_sample": _reliability_summary(fit, train),
            "heldout": _reliability_summary(fit, heldout),
            "constant_baseline_heldout": _constant_baseline(train, heldout),
            "raw_bt_logistic_baseline_heldout": _logistic_baseline(train, heldout, bt_scores),
        }
    else:
        report["train_in_sample"] = _reliability_summary(fit, observations)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
