"""The metric seam: the ``rcp.metrics`` entry-point group, the shipped nDCGs and the plugin contract.

The three nDCGs the package ships (``rcp_ndcg``, ``qrel_ndcg``, ``count_ndcg``) are the registry's first members,
registered at import, so selecting one behaves exactly as it did before the seam. Another metric is one class
in a package that publishes it under the group::

    [project.entry-points."rcp.metrics"]
    map = "my_package.metrics:MeanAveragePrecision"

The class subclasses :class:`Metric`, sets ``name`` and ``k_required``, and implements
:meth:`Metric.score`. It is scored **per query** with the dataset's qrels, the protocol (so the tie rule and the
pool restriction stay :mod:`rcp_core`'s) and the judged pool; it returns a value in the metric's own range, or
``None`` for an undefined query. ``k_required`` says whether the metric takes the requested cutoff: a metric
that does is scored at every requested ``k``; one that does not is scored once, at ``k=0`` ("no cutoff").

The lookup is :mod:`rcp.support.entrypoints`'s one policy: a name provided by more than one installed
distribution is refused, naming the providers. The shipped three are entries of the group too (synthesised here,
because they ship inside this package), so a plugin that publishes ``rcp_ndcg`` is refused by the same policy rather
than silently replacing the built-in.
"""

from __future__ import annotations

import abc
from collections.abc import Collection, Mapping, Sequence
from importlib.metadata import EntryPoint, entry_points
from typing import ClassVar

from rcp_core.protocol import MetricName, Protocol, score_query

from rcp.errors import ConfigError
from rcp.support.entrypoints import load_entry_point, provider_of, registered_names

METRICS_GROUP = "rcp.metrics"
"""The entry-point group a metric plugin publishes its class under."""

METRICS: tuple[MetricName, ...] = ("rcp_ndcg", "qrel_ndcg", "count_ndcg")
"""The shipped metrics, and the registry's first members (``rcp.eval.METRICS``)."""


class Metric(abc.ABC):
    """The contract a metric plugin implements: a name, whether it takes a cutoff, and one query's value.

    A plugin's entry point names the class; the class is instantiated once per :func:`~rcp.eval.evaluate` call.
    ``score`` receives the query id, the system's ``{doc_id: score}``, the dataset's ``{doc_id: grade}`` qrels,
    the scoring :class:`~rcp_core.protocol.Protocol`, the cutoff and the query's judged pool and exclusions. It
    returns a finite float, or ``None`` when the metric is undefined for the query (the value drops out of the
    means, as qrel-nDCG does without a positive grade).
    """

    #: The metric name: the ``rcp.metrics`` entry-point name (the two must agree).
    name: ClassVar[str]
    #: Whether the metric takes the requested cutoff (default true). A false metric is scored once, at ``k=0``.
    k_required: ClassVar[bool] = True

    @abc.abstractmethod
    def score(
        self,
        query: str,
        ranking: Mapping[str, float],
        labels: Mapping[str, float],
        *,
        protocol: Protocol,
        k: int,
        candidates: Sequence[str] | None = None,
        excluded: Collection[str] = (),
    ) -> float | None:
        """One query's value.

        Args:
            query: The query id (for a protocol's ``drop_identical_ids``).
            ranking: The system's ``{doc_id: score}`` for the query.
            labels: The query's ``{doc_id: grade}`` qrels.
            protocol: The dataset's scoring protocol: :func:`rcp_core.protocol.candidate_docs` is the one
                ranking rule (pool restriction, exclusions, tie-independent order).
            k: The cutoff (``0`` for a metric whose ``k_required`` is false).
            candidates: The query's judged pool in pool order, or ``None``.
            excluded: The document ids the benchmark removes for the query.

        Returns:
            The value, or ``None`` when it is undefined for the query.
        """


class _ShippedNdcg(Metric):
    """One of the three nDCGs: the arithmetic is :func:`rcp_core.protocol.score_query`'s, unchanged."""

    metric: ClassVar[MetricName]

    def score(
        self,
        query: str,
        ranking: Mapping[str, float],
        labels: Mapping[str, float],
        *,
        protocol: Protocol,
        k: int,
        candidates: Sequence[str] | None = None,
        excluded: Collection[str] = (),
    ) -> float | None:
        return score_query(
            ranking,
            labels,
            protocol=protocol,
            k=k,
            metric=self.metric,
            candidates=candidates,
            excluded=excluded,
            query_id=query,
        )


class RcpNdcg(_ShippedNdcg):
    """RCP-nDCG: the calibrated gains in ``[0, 1]``."""

    name = "rcp_ndcg"
    metric = "rcp_ndcg"


class QrelNdcg(_ShippedNdcg):
    """qrel-nDCG: the human grades (positive grades are the relevant documents)."""

    name = "qrel_ndcg"
    metric = "qrel_ndcg"


class CountNdcg(_ShippedNdcg):
    """Count-nDCG: the share of passed rubric criteria per document."""

    name = "count_ndcg"
    metric = "count_ndcg"


_SHIPPED: tuple[type[Metric], ...] = (RcpNdcg, QrelNdcg, CountNdcg)


def _shipped_entries() -> tuple[EntryPoint, ...]:
    """The shipped three as entries of the group, so one lookup covers built-ins and plugins alike."""
    return tuple(
        EntryPoint(name=metric.name, value=f"{metric.__module__}:{metric.__qualname__}", group=METRICS_GROUP)
        for metric in _SHIPPED
    )


def _entries() -> tuple[EntryPoint, ...]:
    """The group's entries: the shipped three, then the installed plugins."""
    return (*_shipped_entries(), *entry_points(group=METRICS_GROUP))


def registered_metrics() -> tuple[str, ...]:
    """The metric names: the shipped three first, then the installed plugins, sorted."""
    plugins = registered_names(entry_points(group=METRICS_GROUP), METRICS_GROUP)
    return (*METRICS, *(name for name in plugins if name not in METRICS))


def metric_class(name: str) -> type[Metric]:
    """The metric class registered under ``name`` in the ``rcp.metrics`` entry-point group.

    Args:
        name: The metric name (a shipped one, or a plugin's entry-point name).

    Returns:
        The plugin class, checked against :class:`Metric` and its two declarations.

    Raises:
        ConfigError: No metric has that name (the installed ones are named), the name is provided by more than
            one installed distribution (the one duplicate-name policy), the entry point does not load, what it
            names is not a :class:`Metric`, or the class's ``name`` disagrees with its entry-point name.
    """
    entries = _entries()
    available = registered_names(entries, METRICS_GROUP)
    if name not in available:
        raise ConfigError(
            f"unknown metric {name!r}; available: {list(registered_metrics())}",
            hint=f"a metric is a {METRICS_GROUP!r} entry point; install the package that provides it, or name "
            f"one of {list(METRICS)}",
        )
    metric = load_entry_point(provider_of(entries, METRICS_GROUP, name, kind="metric"), Metric, kind="metric")
    if not isinstance(getattr(metric, "name", None), str) or not metric.name:
        raise ConfigError(f"the metric {name!r} declares no name")
    if not isinstance(getattr(metric, "k_required", None), bool):
        raise ConfigError(f"the metric {name!r} declares no k_required bool")
    if metric.name != name:
        raise ConfigError(
            f"the metric entry point {name!r} names a class that calls itself {metric.name!r}: the entry-point "
            "name is the metric name, so the two must agree"
        )
    return metric


__all__ = [
    "METRICS",
    "METRICS_GROUP",
    "Metric",
    "metric_class",
    "registered_metrics",
]
