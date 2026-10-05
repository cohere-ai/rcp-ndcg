"""``explain``: one query, side by side -- each system's top-k with gains, grades, thetas and criteria.

The per-criterion view comes from :func:`rcp_ndcg_core.pass_probabilities`: at a calibrated ability ``theta``
criterion ``c`` passes with probability ``sigmoid(gamma_c (theta - beta_c))`` and contributes
``gamma_c p_c / sum(gamma)`` to the gain. Between two systems the RCP-nDCG@k gap (B minus A) splits into
**selection** (which documents reach the top k) and **ordering** (how the chosen ones are arranged).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict
from rcp_ndcg_core import ndcg, pass_probabilities
from rcp_ndcg_core.gain import item_arrays
from rcp_ndcg_core.metric import rank_by_score
from rcp_ndcg_core.protocol import candidate_docs

from rcp_ndcg.data.dataset import Dataset
from rcp_ndcg.errors import DataError
from rcp_ndcg.eval.evaluate import EvalReport, _system_queries


class CriterionParams(BaseModel):
    """One criterion's calibrated item parameters.

    Attributes:
        criterion: ``"C1"`` ... ``"C5"``.
        gamma: Discrimination (dimensionless).
        beta: Difficulty, in logits.
    """

    model_config = ConfigDict(frozen=True)

    criterion: str
    gamma: float
    beta: float


class CriterionContribution(BaseModel):
    """One criterion at a document's calibrated ability (its ``gamma`` and ``beta``: :attr:`QueryExplanation.items`).

    Attributes:
        criterion: ``"C1"`` ... ``"C5"``.
        p_pass: ``sigmoid(gamma * (theta - beta))``, in ``(0, 1)``.
        contribution: ``gamma * p_pass / sum(gamma)``; the contributions sum to the gain.
    """

    model_config = ConfigDict(frozen=True)

    criterion: str
    p_pass: float
    contribution: float


class RankedDocument(BaseModel):
    """A document in a system's top k."""

    model_config = ConfigDict(frozen=True)

    rank: int
    doc_id: str
    score: float
    gain: float | None
    grade: float | None
    theta: float | None
    criteria: list[CriterionContribution] = []


class SystemExplanation(BaseModel):
    """One system on the query: its metric values and its top k."""

    model_config = ConfigDict(frozen=True)

    system: str
    values: dict[str, float | None]
    top: list[RankedDocument]


class ScoreDelta(BaseModel):
    """The RCP-nDCG gap between two systems' displayed orders (B minus A), split into selection and ordering.

    The cutoff is the explanation's ``k`` (the documents shown), which may differ from the report's cutoffs in
    :attr:`SystemExplanation.values`. ``selection`` is what choosing other documents for the top k changes, and
    ``ordering`` what arranging them differently changes; the two sum to ``total``.
    """

    model_config = ConfigDict(frozen=True)

    system_a: str
    system_b: str
    total: float
    selection: float
    ordering: float


class QueryExplanation(BaseModel):
    """The result of :func:`explain`.

    Attributes:
        dataset: The dataset (subset) of the query.
        query_id: The query.
        k: The documents shown per system, and the cutoff of ``deltas``.
        protocol: The name of the report's scoring protocol.
        items: The criteria's item parameters behind every document's ``criteria`` (empty without a calibration).
        systems: Each system's values (at the report's cutoffs) and its top k.
        deltas: Every system against the first, at cutoff ``k``.
    """

    model_config = ConfigDict(frozen=True)

    dataset: str
    query_id: str
    k: int
    protocol: str
    items: list[CriterionParams] = []
    systems: list[SystemExplanation]
    deltas: list[ScoreDelta]

    def to_json(self, **kwargs: Any) -> str:
        """The explanation as JSON."""
        return self.model_dump_json(**kwargs)


def per_criterion(theta: float, items: Any) -> list[CriterionContribution]:
    """The per-criterion decomposition of the gain at a calibrated ability.

    Args:
        theta: Calibrated ability, in logits.
        items: Item parameters with ``gamma`` and ``beta`` sequences (an ``ItemParams``, or a mapping).

    Returns:
        One :class:`CriterionContribution` per criterion; the contributions sum to ``gain(theta, items)``.
    """
    gammas, _ = item_arrays(items)
    total = sum(gammas)
    return [
        CriterionContribution(
            criterion=f"C{c + 1}", p_pass=float(p), contribution=float(g * p / total) if total > 0 else 0.0
        )
        for c, (g, p) in enumerate(zip(gammas, pass_probabilities(theta, items), strict=True))
    ]


def _criterion_params(items: Any) -> list[CriterionParams]:
    gammas, betas = item_arrays(items)
    return [
        CriterionParams(criterion=f"C{c + 1}", gamma=float(g), beta=float(b))
        for c, (g, b) in enumerate(zip(gammas, betas, strict=True))
    ]


def score_delta(
    order_a: Sequence[str], order_b: Sequence[str], gains: Mapping[str, float], *, k: int = 10
) -> tuple[float, float, float]:
    """Split the nDCG@k gap between two orders (B minus A) into selection and ordering.

    Ordering is B's score minus the score of B's top-k documents arranged in A's order (those A lacks appended in
    B's order); selection is the rest.

    Returns:
        ``(total, selection, ordering)``.
    """
    top_b = list(order_b[:k])
    in_b = set(top_b)
    b_in_a_order = [d for d in order_a if d in in_b]
    b_in_a_order += [d for d in top_b if d not in set(b_in_a_order)]
    score_a, score_b = ndcg(list(order_a), gains, k=k), ndcg(list(order_b), gains, k=k)
    ordering = score_b - ndcg(b_in_a_order, gains, k=k)
    total = score_b - score_a
    return total, total - ordering, ordering


def explain(
    report: EvalReport,
    query_id: str,
    *,
    calibration: Any = None,
    k: int = 10,
    dataset: str | None = None,
) -> QueryExplanation:
    """Explain one query of a report: each system's top k, and the gaps between systems.

    Args:
        report: A report from :func:`~rcp_ndcg.eval.evaluate` (it keeps the rankings and data it scored).
        query_id: The query.
        calibration: A :class:`~rcp_ndcg.calibration.Calibration`, for the thetas and the per-criterion
            probabilities (a calibration of several datasets gives the query's dataset's thetas). Without it, the
            dataset's released thetas are shown and no criteria.
        k: How many documents per system.
        dataset: The dataset (subset) of the query, when the report spans several.

    Returns:
        The :class:`QueryExplanation`; deltas compare every system against the first.
    """
    rankings = report._inputs.get("rankings")
    data: Dataset | None = report._inputs.get("dataset")
    if rankings is None or data is None:
        raise DataError("explain needs the rankings and data behind the report; use a report returned by evaluate()")
    part = _part(data, dataset, query_id)
    gains = report._inputs["labels"].get("rcp_ndcg", {}).get(part.name, {}).get(query_id, {})
    items, thetas = _calibration(calibration, part, query_id, suite=bool(data.subsets))
    rules = report.protocol
    excluded = set(part.excluded.get(query_id, ()))
    ideal_gains = {d: g for d, g in gains.items() if d not in excluded}

    systems, orders = [], {}
    scored = report.systems  # the systems the report scored (systems= may have restricted the rankings' file)
    if not scored:
        raise DataError(
            f"the report scored no systems (its metrics matched no labelled queries of {data.name!r})",
            hint="score the query with a metric that has labels for it: RCP gains for rcp_ndcg, qrels for qrel_ndcg",
        )
    for system in scored:
        scores = _system_queries(rankings, system, part.name).get(query_id, {})
        entering = candidate_docs(
            rules, scores, candidates=(part.candidates or {}).get(query_id), excluded=excluded, query_id=query_id
        )
        ranked = {d: scores[d] for d in entering}
        order = rank_by_score(ranked, ties="input_order" if rules.ties == "input_order" else "doc_id_desc")
        orders[system] = order
        values = {
            f"{row.metric}@{row.k}": row.value
            for row in report.per_query
            if row.system == system and row.dataset == part.name and row.query_id == query_id
        }
        top = [
            RankedDocument(
                rank=rank,
                doc_id=doc_id,
                score=ranked[doc_id],
                gain=gains.get(doc_id),
                grade=part.qrels.get(query_id, {}).get(doc_id),
                theta=thetas.get(doc_id),
                criteria=per_criterion(thetas[doc_id], items) if items is not None and doc_id in thetas else [],
            )
            for rank, doc_id in enumerate(order[:k], start=1)
        ]
        systems.append(SystemExplanation(system=system, values=values, top=top))

    first = scored[0]
    deltas = []
    for other in scored[1:] if ideal_gains else []:
        total, selection, ordering = score_delta(orders[first], orders[other], ideal_gains, k=k)
        deltas.append(ScoreDelta(system_a=first, system_b=other, total=total, selection=selection, ordering=ordering))
    return QueryExplanation(
        dataset=part.name,
        query_id=query_id,
        k=k,
        protocol=rules.name,
        items=_criterion_params(items) if items is not None else [],
        systems=systems,
        deltas=deltas,
    )


def _part(data: Dataset, dataset: str | None, query_id: str) -> Dataset:
    scope = [p for p in data.parts if dataset is None or p.name == dataset]
    parts = [p for p in scope if query_id in p.qrels | (p.gains or {})]
    if not parts:
        known = sorted({q for p in scope for q in p.qrels | (p.gains or {})})
        shown = ", ".join(known[:10]) + (f" (+{len(known) - 10} more)" if len(known) > 10 else "")
        where = f"dataset {dataset!r}" if dataset is not None else "the report"
        raise DataError(
            f"query {query_id!r} is not in {where}; known: {shown or 'none'}",
            details={"query_id": query_id, "known": known[:50], "num_known": len(known)},
        )
    if len(parts) > 1:
        names = [p.name for p in parts]
        raise DataError(
            f"query {query_id!r} is in {len(parts)} datasets of the report: {names}",
            hint="name the query's dataset with dataset=",
            cli_hint="name the query's dataset with --subset",
            details={"query_id": query_id, "datasets": names},
        )
    return parts[0]


def _calibration(calibration: Any, part: Dataset, query_id: str, *, suite: bool) -> tuple[Any, dict[str, float]]:
    """The item parameters and ``{doc_id: theta}`` of ``query_id``: the calibration's, else the released thetas.

    The calibration's thetas are those of its dataset named like ``part`` when the report spans a suite or the
    calibration holds several datasets (as :func:`~rcp_ndcg.eval.evaluate` reads its gains).
    """
    if calibration is None:
        return None, dict((part.thetas or {}).get(query_id, {}))
    rows = getattr(calibration, "thetas", None)
    if getattr(calibration, "items", None) is None or rows is None or isinstance(rows, Mapping):
        raise DataError(
            f"calibration must be a Calibration, got {type(calibration).__name__}",
            hint="pass rcp_ndcg.calibration.Calibration.load(<calibration dir>)",
        )
    by_name = suite or len({row.dataset for row in rows}) > 1
    thetas = {
        row.doc_id: row.theta for row in rows if row.query_id == query_id and (not by_name or row.dataset == part.name)
    }
    return calibration.items, thetas


__all__ = [
    "CriterionContribution",
    "CriterionParams",
    "QueryExplanation",
    "RankedDocument",
    "ScoreDelta",
    "SystemExplanation",
    "explain",
    "per_criterion",
    "score_delta",
]
