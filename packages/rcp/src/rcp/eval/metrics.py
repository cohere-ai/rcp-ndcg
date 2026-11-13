"""The metric seam: the ``rcp.metrics`` entry-point group, the shipped metrics and the plugin contract.

The seam scores **one query**. :meth:`Metric.score` receives one query's ``{doc_id: score}``, that query's qrels
and the dataset's protocol, and returns the query's value; :func:`rcp.eval.evaluate` averages those values over
the dataset's labelled queries, so a corpus value is that mean. A metric belongs here when its value is one
query's; the denominator is the dataset's labelled queries, and a labelled query the run does not rank is scored
``0.0`` (the report warns ``UNRANKED_QUERIES``).

The ten metrics the package ships are the registry's first members, registered at import, so selecting one
behaves exactly as it did before the seam:

* the three nDCGs -- ``rcp_ndcg``, ``qrel_ndcg``, ``count_ndcg`` -- are scored by
  :func:`rcp_core.protocol.score_query`, so the protocol's tie rule reaches them;
* the seven corpus metrics of :mod:`rcp_core.metric` -- ``ndcg_float``, ``map``, ``recall``, ``precision``,
  ``success``, ``mrr``, ``recall_cap`` -- are scored by their own function, so each keeps the one tie rule it
  publishes and the protocol's tie rule does not reach it.

Another metric is one class in a package that publishes it under the group::

    [project.entry-points."rcp.metrics"]
    toy_metric = "my_package.metrics:MeanAveragePrecision"

The class subclasses :class:`Metric`, sets ``name`` and ``k_required``, and implements
:meth:`Metric.score`. It is scored **per query** with the dataset's qrels, the protocol (so the tie rule and the
pool restriction stay :mod:`rcp_core`'s) and the judged pool; it returns a value in the metric's own range, or
``None`` for an undefined query. ``k_required`` says whether the metric takes the requested cutoff: a metric
that does is scored at every requested ``k``; one that does not is scored once, at ``k=0`` ("no cutoff").

The lookup is :mod:`rcp.support.entrypoints`'s one policy: a name provided by more than one installed
distribution is refused, naming the providers. The shipped ten are entries of the group too (synthesised here,
because they ship inside this package), so a plugin that publishes ``rcp_ndcg`` is refused by the same policy rather
than silently replacing the built-in.

Three metrics of :mod:`rcp_core.metric` are **not** members of this registry, because the seam has no input for
them. They are public there and reachable by import, and no ``--metric`` names them:

* :func:`rcp_core.metric.hole` reads the corpus-wide annotated set (the union of every qrel's document ids),
  which one query's labels do not carry;
* :func:`rcp_core.metric.paired_accuracy` reads ``pair_groups``, the two queries of each group, which
  :func:`rcp.eval.evaluate` does not take;
* :func:`rcp_core.metric.coverage` reads ``{item_id: chosen_probability}``, neither a qrels nor a ranking.

:func:`rcp_core.metric.per_query_ndcg` is not a fourth: it is the batch form of ``ndcg_float`` -- one list of
per-query values per dataset, and those values are the rows ``ndcg_float`` produces.

A caller reads this module through :mod:`rcp.eval`, which re-exports :data:`METRICS`, :data:`METRICS_GROUP`,
:class:`Metric`, :func:`metric_class` and :func:`registered_metrics`. It is not a module of its own in
:data:`rcp.support.api_docs.PUBLIC_MODULES`, so its generated page is ``rcp.eval``'s.
"""

from __future__ import annotations

import abc
from collections.abc import Callable, Collection, Mapping, Sequence
from functools import partial
from importlib.metadata import EntryPoint, entry_points
from typing import ClassVar

from rcp_core import metric as core_metric
from rcp_core.protocol import MetricName, Protocol, candidate_docs, ranked_scores, score_query

from rcp.errors import ConfigError
from rcp.support.entrypoints import load_entry_point, provider_of, registered_names

METRICS_GROUP = "rcp.metrics"
"""The entry-point group a metric plugin publishes its class under."""

METRICS: tuple[str, ...] = (
    "rcp_ndcg",
    "qrel_ndcg",
    "count_ndcg",
    "ndcg_float",
    "map",
    "recall",
    "precision",
    "success",
    "mrr",
    "recall_cap",
)
"""The shipped metrics, and the registry's first members (``rcp.eval.METRICS``): the three nDCGs, then the
seven corpus metrics of :mod:`rcp_core.metric` whose value is one query's."""


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
            The value, or ``None`` when it is undefined for the query. ``nan`` is one spelling of undefined: the
            seam drops such a row as ``None``, with no warning, the way qrel-nDCG drops a query without a positive
            grade. ``inf`` and a value that is not a number are refused (:class:`~rcp.errors.DataError`).
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


class _CorpusMetric(Metric):
    """One corpus metric of :mod:`rcp_core.metric`, scored on one query.

    The arithmetic is the corpus function's, called on a one-query corpus, so a metric has one body and the seam
    cannot disagree with :mod:`rcp_core.metric` about a query. ``protocol`` supplies the *set* of entering
    documents (:func:`rcp_core.protocol.candidate_docs`: the pool restriction and the exclusions);
    :func:`rcp_core.protocol.ranked_scores` keeps the run's insertion order among them, because that order is
    the ``input_order`` tie rule of ``mrr`` and ``recall_cap`` -- the pool's order is often a relevance order,
    and is never a ranking.

    The mean is over the dataset's labelled queries: a labelled query the run does not rank is scored ``0.0``,
    where the standalone function divides by the queries it evaluates (``mrr`` and ``recall_cap``: by every
    qrels query, refusing a run whose query ids differ). A query is never undefined for these metrics -- one
    with no relevant document scores ``0.0`` and stays in the mean.
    """

    function: ClassVar[Callable[..., float]]
    """The ``rcp_core.metric`` function: ``function(qrels, results, k=k)``, one query on each side.

    A subclass declares it through ``staticmethod``. Python 3.14's ``functools.partial`` implements the
    descriptor protocol, so a bare ``partial`` class attribute is read through the instance as
    ``partial(function, self)``, and :meth:`score` would pass ``self`` as a third positional argument (every
    one of the seven raised ``TypeError`` on 3.14); ``staticmethod`` returns the partial unchanged from both an
    instance and the class.
    """

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
        entering = candidate_docs(protocol, ranking, candidates=candidates, excluded=excluded, query_id=query)
        ordered = ranked_scores(ranking, entering)
        return self.function({query: labels}, {query: ordered}, k=k)


class NdcgFloat(_CorpusMetric):
    """``ndcg_float``: nDCG@k over continuous qrels, averaged over the evaluated queries.

    Equal scores are credited the mean gain of their equal-score class (``group_mean``), the gain is the float
    qrel itself -- finite and at least 0 -- and the ideal DCG is the query's whole float-qrel set. A query with
    no positive qrel scores 0.0 and stays in the mean.

    Raises:
        ValueError: ``k`` is not positive, a grade is not finite or is negative, or a score is not finite.
    """

    name = "ndcg_float"
    function = staticmethod(partial(core_metric.ndcg_float))


class Map(_CorpusMetric):
    """``map``: MAP@k, averaged over the evaluated queries.

    Equal scores are ordered by document id, descending (``doc_id_desc``), and a document is relevant when its
    qrel grade is above 0. The per-query denominator is ``|R_q|``, the positives in the query's whole qrels,
    never capped at ``k``: MAP@k is below 1.0 when the query has more positives than ``k``. A query with no
    relevant document scores 0.0 and stays in the mean.

    Raises:
        ValueError: ``k`` is not positive, a grade is not finite or is negative, or a score is not finite.
    """

    name = "map"
    function = staticmethod(partial(core_metric.map))


class Recall(_CorpusMetric):
    """``recall``: Recall@k, averaged over the evaluated queries.

    Equal scores are ordered by document id, descending (``doc_id_desc``), and a document is relevant when its
    qrel grade is above 0. The denominator is ``|R_q|``, the positives in the query's whole qrels, so a run
    shorter than the positives cannot reach 1.0. A query with no relevant document scores 0.0 and stays in the
    mean.

    Raises:
        ValueError: ``k`` is not positive, a grade is not finite or is negative, or a score is not finite.
    """

    name = "recall"
    function = staticmethod(partial(core_metric.recall))


class Precision(_CorpusMetric):
    """``precision``: P@k, averaged over the evaluated queries.

    Equal scores are ordered by document id, descending (``doc_id_desc``), and the denominator is the requested
    ``k``, not the ranking's length, so a ranking shorter than ``k`` cannot reach 1.0 (the ``pytrec_eval``
    ``P_cut`` definition). A query with no relevant document scores 0.0 and stays in the mean.

    Raises:
        ValueError: ``k`` is not positive, a grade is not finite or is negative, or a score is not finite.
    """

    name = "precision"
    function = staticmethod(partial(core_metric.precision))


class Success(_CorpusMetric):
    """``success``: success@k, averaged over the evaluated queries.

    Equal scores are ordered by document id, descending (``doc_id_desc``), and a query scores 1.0 when any
    document in its top k has a qrel grade above 0. The per-query value is binary, so no partial credit can hide
    a wrong tie order: the rule matters only when a tie straddles the cutoff. A query with no relevant document
    scores 0.0 and stays in the mean.

    Raises:
        ValueError: ``k`` is not positive, a grade is not finite or is negative, or a score is not finite.
    """

    name = "success"
    function = staticmethod(partial(core_metric.success))


class Mrr(_CorpusMetric):
    """``mrr``: MRR@k, the mean of ``1 / rank`` of the first relevant document.

    Equal scores keep the run's insertion order (``input_order``), and a document is relevant when its qrel
    grade is above 0. This is the metric the tie rule moves first: a tie at the top moves the first hit from
    rank 1 to rank 2. A hit is found or it is not, so a ``k`` deeper than the run changes nothing.

    The mean is over the dataset's labelled queries, as for every metric here (a labelled query the run does not
    rank scores 0.0); the standalone function divides by every qrels query, refusing a run whose query ids
    differ.

    Raises:
        ValueError: ``k`` is not positive, a grade is not finite or is negative, or a score is not finite.
    """

    name = "mrr"
    function = staticmethod(partial(core_metric.mrr))


class RecallCap(_CorpusMetric):
    """``recall_cap``: R_cap@k, the depth-capped recall of BEIR's ``custom_metrics``.

    Equal scores keep the run's insertion order (``input_order``), and the denominator is the smaller of the
    positive count and ``k``: a run of ``k`` documents cannot retrieve more than ``k`` positives, so the value
    stays at 1.0 when the query has more positives than ``k`` -- the one difference from :class:`Recall`. A
    query with no relevant document has no denominator, scores 0.0 and stays in the mean.

    The mean is over the dataset's labelled queries, as for every metric here (a labelled query the run does not
    rank scores 0.0); the standalone function divides by every qrels query, refusing a run whose query ids
    differ.

    Raises:
        ValueError: ``k`` is not positive, a grade is not finite or is negative, or a score is not finite.
    """

    name = "recall_cap"
    function = staticmethod(partial(core_metric.recall_cap))


_SHIPPED: tuple[type[Metric], ...] = (
    RcpNdcg,
    QrelNdcg,
    CountNdcg,
    NdcgFloat,
    Map,
    Recall,
    Precision,
    Success,
    Mrr,
    RecallCap,
)


def _shipped_entries() -> tuple[EntryPoint, ...]:
    """The shipped ten as entries of the group, so one lookup covers built-ins and plugins alike."""
    return tuple(
        EntryPoint(name=metric.name, value=f"{metric.__module__}:{metric.__qualname__}", group=METRICS_GROUP)
        for metric in _SHIPPED
    )


def _entries() -> tuple[EntryPoint, ...]:
    """The group's entries: the shipped ten, then the installed plugins."""
    return (*_shipped_entries(), *entry_points(group=METRICS_GROUP))


def registered_metrics() -> tuple[str, ...]:
    """The metric names: the shipped ten first, then the installed plugins, sorted.

    A plugin's name that a shipped metric already holds is left out of this listing, so every name appears
    once: choosing that name is refused by :func:`metric_class`'s duplicate-name policy instead.
    """
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
