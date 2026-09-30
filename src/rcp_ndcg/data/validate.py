"""``validate``: check a dataset, and optionally rankings, against the scoring protocol before scoring them.

The checks (each a :class:`ValidationCheck` with a code, a severity, a count and up to five examples):

* ``GAIN_RANGE`` (error): released gains outside ``[0, 1]``.
* ``EXCLUDED_RELEVANT`` (error): excluded documents that carry a positive label.
* ``UNJUDGED_CANDIDATES`` (warning): pool documents without a gain (scored as 0).
* ``QUERIES_WITHOUT_POOL`` (warning): labelled queries without a candidate pool.
* ``NO_RANKINGS`` (error): no ranking row names the dataset, and none names no dataset.
* ``UNKNOWN_QUERIES`` (error): ranked queries the dataset does not have.
* ``OUTSIDE_POOL`` (warning): ranked documents outside the judged pool (scored as 0).
* ``EXCLUDED_RANKED`` (warning): ranked documents the protocol removes.

An error means the numbers computed from this input would be wrong; a warning means they hold, with the stated
consequence. The rankings checks read the rows that rank the dataset, as :func:`~rcp_ndcg.eval.evaluate` does: the
rows whose ``dataset`` names it, else the rows that name no dataset (:meth:`Rankings.resolve_dataset`).
``rcp-ndcg data validate`` is this function behind a command line.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from rcp_ndcg.data.dataset import Dataset
from rcp_ndcg.data.rankings import Rankings
from rcp_ndcg.errors import ConfigError

Severity = Literal["error", "warning"]


class ValidationCheck(BaseModel):
    """One problem found; ``error`` means the numbers computed from this input would be wrong.

    Attributes:
        code: The check's code (see the module docstring).
        severity: ``"error"`` or ``"warning"``.
        message: What the check found.
        count: How many items it found.
        examples: Up to five of them, as ``query_id/doc_id`` (or ``query_id``), sorted.
    """

    code: str
    severity: Severity
    message: str
    count: int
    examples: list[str] = Field(default_factory=list)


class ValidationReport(BaseModel):
    """The problems found; ``ok`` when none is an error.

    Attributes:
        dataset: The dataset's name.
        uri: The URI it was loaded from (``None`` for a dataset built in memory).
        systems: The systems of the checked rankings (``None`` without rankings).
        ok: Whether no check is an error.
        checks: The problems found.
    """

    dataset: str
    uri: str | None = None
    systems: list[str] | None = None
    ok: bool
    checks: list[ValidationCheck]

    @property
    def errors(self) -> list[str]:
        """The codes of the failed error checks."""
        return [check.code for check in self.checks if check.severity == "error"]


def validate(dataset: Dataset, rankings: Rankings | None = None) -> ValidationReport:
    """Check a dataset, and optionally rankings, against the scoring protocol.

    Args:
        dataset: One dataset (a suite's subset, not the suite).
        rankings: Rankings to check against the dataset (every system; the rows that rank this dataset).

    Returns:
        The :class:`ValidationReport`; ``ok`` is false when a check is an error.

    Raises:
        ConfigError: ``dataset`` is a suite.
    """
    if dataset.subsets:
        raise ConfigError(f"{dataset.name!r} is a suite; validate one of its subsets (Dataset.subsets)")
    checks: list[ValidationCheck] = []
    positive = _positive(dataset)
    gains = dataset.gains or {}
    candidates = dataset.candidates or {}
    _add(
        checks,
        "GAIN_RANGE",
        "error",
        "gains outside [0, 1]",
        [f"{q}/{d}" for q, docs in gains.items() for d, g in docs.items() if not 0 <= g <= 1],
    )
    _add(
        checks,
        "EXCLUDED_RELEVANT",
        "error",
        "excluded documents that carry a positive label",
        [f"{q}/{d}" for q, ids in dataset.excluded.items() for d in ids if d in positive.get(q, {})],
    )
    if dataset.gains is not None:
        unjudged = [f"{q}/{d}" for q, pool in candidates.items() for d in pool if d not in gains.get(q, {})]
        _add(checks, "UNJUDGED_CANDIDATES", "warning", "pool documents without a gain (scored as 0)", unjudged)
    if dataset.candidates is not None:
        without_pool = [q for q in positive if q not in candidates]
        _add(checks, "QUERIES_WITHOUT_POOL", "warning", "labelled queries without a candidate pool", without_pool)
    if rankings is not None:
        checks += _rankings_checks(dataset, rankings, positive)
    return ValidationReport(
        dataset=dataset.name,
        uri=dataset.uri,
        systems=rankings.systems if rankings is not None else None,
        ok=not any(check.severity == "error" for check in checks),
        checks=checks,
    )


def _positive(dataset: Dataset) -> dict[str, dict[str, float]]:
    return {q: {d: g for d, g in docs.items() if g > 0} for q, docs in dataset.qrels.items()}


def _add(checks: list[ValidationCheck], code: str, severity: Severity, message: str, items: list[str]) -> None:
    if items:
        examples = sorted(set(items))[:5]
        checks.append(
            ValidationCheck(code=code, severity=severity, message=message, count=len(items), examples=examples)
        )


def _rankings_checks(
    dataset: Dataset, rankings: Rankings, positive: dict[str, dict[str, float]]
) -> list[ValidationCheck]:
    pools = {query: set(pool) for query, pool in (dataset.candidates or {}).items()}
    dropped = {query: set(ids) for query, ids in dataset.excluded.items()}
    known = set(pools) | set(positive)
    unknown: set[str] = set()
    outside, excluded = [], []
    checks: list[ValidationCheck] = []
    key = rankings.resolve_dataset(dataset.name)
    if key is None:
        message = f"no ranking row names {dataset.name!r} or no dataset; the rows name {rankings.datasets}"
        _add(checks, "NO_RANKINGS", "error", message, [dataset.name])
        return checks
    frame = rankings.to_pandas()
    frame = frame[frame["dataset"] == key]
    for query_id, doc_id in zip(frame["query_id"], frame["doc_id"], strict=True):
        if query_id not in known:
            unknown.add(query_id)
            continue
        pool = pools.get(query_id)
        if doc_id in dropped.get(query_id, ()):
            excluded.append(f"{query_id}/{doc_id}")
        elif pool and doc_id not in pool:
            outside.append(f"{query_id}/{doc_id}")
    _add(checks, "UNKNOWN_QUERIES", "error", "ranked queries that the dataset does not have", sorted(unknown))
    _add(checks, "OUTSIDE_POOL", "warning", "ranked documents outside the judged pool (scored as 0)", outside)
    _add(checks, "EXCLUDED_RANKED", "warning", "ranked documents the protocol removes", excluded)
    return checks


__all__ = ["ValidationCheck", "ValidationReport", "validate"]
