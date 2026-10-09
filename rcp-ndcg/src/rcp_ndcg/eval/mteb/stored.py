"""Scoring inside mteb: a `SearchProtocol` that serves stored :class:`~rcp_ndcg.data.Rankings` to
``mteb.evaluate``.

Embedders, late-interaction models, rerankers and LLM judges all produce ``Rankings``, so one route covers all
four: the scores are already computed, and mteb scores them against its tasks, writes its own predictions file
(``{Task}_predictions.json``) and records genuine ``TaskResult`` files in its ``ResultCache`` layout, ready for
``ResultCache.submit_results``.

::

    from rcp_ndcg.eval.mteb import get_tasks, model_meta, stored_rankings_model

    model = stored_rankings_model(rankings, model_meta("org/model", revision))
    mteb.evaluate(model, get_tasks("nanobeir", ["NanoFiQA2018Retrieval"]), encode_kwargs={})

The leaderboard itself also needs the model's ``ModelMeta`` merged into mteb, which lives outside this
repository (mteb's model registry).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from rcp_ndcg.data.rankings import MTEB_MAX_DOCS, Rankings
from rcp_ndcg.errors import ConfigError

if TYPE_CHECKING:
    from rcp_ndcg_core._records import ID

__all__ = ["StoredRankings", "model_meta", "stored_rankings_model"]

_META_DEFAULTS: dict[str, Any] = {
    "loader": None,
    "release_date": None,
    "languages": None,
    "n_parameters": None,
    "memory_usage_mb": None,
    "max_tokens": None,
    "embed_dim": None,
    "license": None,
    "open_weights": None,
    "public_training_code": None,
    "public_training_data": None,
    "framework": [],
    "similarity_fn_name": None,
    "use_instructions": None,
    "training_datasets": None,
}
"""The undeclared ModelMeta fields: unknown (``None``), except ``framework``, which mteb requires as a list."""


def model_meta(name: str, revision: str, **fields: Any) -> Any:
    """mteb's ``ModelMeta`` for our model identity: the name (``org/model``) and revision, plus the fields the
    caller declares; everything undeclared is unknown (``None``).

    Args:
        name: The model name in mteb's ``org/model`` form (the served checkpoint or the hosted profile).
        revision: The model revision the scores were produced with.
        **fields: Any other ``ModelMeta`` field the caller declares (``n_parameters``, ``license``, ...).

    Returns:
        The ``ModelMeta`` (the ``[mteb]`` extra).

    Raises:
        ConfigError: The name has no ``org/`` prefix (mteb requires it), a field is not a ``ModelMeta`` field,
            or the declared values fail mteb's own validation.
    """
    from mteb.models.model_meta import ModelMeta

    if "/" not in name:
        raise ConfigError(
            f"the model name {name!r} is not in mteb's 'org/model' form",
            hint="mteb keys the results repository by it (results/{org__model}/...)",
        )
    unknown = sorted(set(fields) - set(ModelMeta.model_fields))
    if unknown:
        raise ConfigError(
            f"unknown ModelMeta fields {unknown}",
            hint=(
                "declared fields must be mteb's ModelMeta fields: "
                f"{sorted(set(ModelMeta.model_fields) - set(_META_DEFAULTS))}"
            ),
        )
    try:
        return ModelMeta(name=name, revision=revision, **{**_META_DEFAULTS, **fields})
    except Exception as exc:
        raise ConfigError(f"mteb refused the ModelMeta fields: {exc}") from exc


def stored_rankings_model(rankings: Rankings, meta: Any, *, system: str | None = None) -> StoredRankings:
    """mteb's ``SearchProtocol`` over one system's stored :class:`~rcp_ndcg.data.Rankings`.

    Args:
        rankings: The scores to serve; several systems require ``system=``.
        meta: The :func:`model_meta` of the model that produced them.
        system: The system to serve, for rankings of several.

    Returns:
        The model object for ``mteb.evaluate`` (the ``[mteb]`` extra).

    Raises:
        DataError: The rankings hold several systems without ``system=``, or ``system`` is unknown.
    """
    return StoredRankings(rankings, meta, system=system)


class StoredRankings:
    """mteb's ``SearchProtocol`` (duck-typed) over stored rankings: nothing to index, nothing to encode.

    ``search`` returns the stored scores of the queries mteb asks about -- and only those, because mteb raises
    on a result for a query that has no qrels. When the task carries ``top_ranked`` (the reranking view), the
    scores are restricted to the pool, exactly what a reranker sees; they are capped at ``top_k``, ties by
    document id descending (the cap order of :meth:`Rankings.top`, and mteb's own tie rule). A query the run
    did not rank returns ``{}``: mteb scores it 0, the same semantics our evaluator reports an unranked
    labelled query with.
    """

    def __init__(self, rankings: Rankings, meta: Any, *, system: str | None = None) -> None:
        self.mteb_model_meta = meta
        self._rankings = rankings
        self._system = rankings._one_system(system)

    def index(
        self,
        corpus: Any,
        *,
        task_metadata: Any,
        hf_split: str,
        hf_subset: str,
        encode_kwargs: Mapping[str, Any] | None = None,
        num_proc: int | None = None,
    ) -> None:
        """Nothing to index: the scores are already computed and stored."""

    def search(
        self,
        queries: Any,
        *,
        task_metadata: Any,
        hf_split: str,
        hf_subset: str,
        top_k: int,
        encode_kwargs: Mapping[str, Any] | None = None,
        top_ranked: Mapping[str, list[ID]] | None = None,
        num_proc: int | None = None,
    ) -> dict[str, dict[str, float]]:
        """The stored scores of the asked queries, restricted to the pool when the task carries one.

        Args:
            queries: mteb's query table (the qrels-filtered queries; a ``Dataset`` with an ``id`` column, or a
                mapping).
            task_metadata: The mteb task metadata (unused; the scores are stored).
            hf_split: The task split being evaluated (unused; the rankings carry no split).
            hf_subset: The task subset: the rankings' ``dataset`` value (``default`` for unnamed ones).
            top_k: The cap per query (mteb's own ``top_k``, 1000 for the retrieval k values).
            encode_kwargs: Unused: no encoder runs.
            top_ranked: The pool per query, when the task reranks: scores outside it are dropped.
            num_proc: Unused.

        Returns:
            ``{query_id: {doc_id: score}}``, every asked query present.

        Raises:
            DataError: No rows of the rankings rank ``hf_subset`` (the datasets the rows do name are listed).
        """
        scores = self._rankings.queries(system=self._system, dataset=_dataset_key(self._rankings, hf_subset))
        return {
            query_id: _of_query(scores.get(query_id, {}), top_ranked.get(query_id) if top_ranked else None, top_k)
            for query_id in _ids(queries)
        }


def _dataset_key(rankings: Rankings, hf_subset: str) -> str | None:
    """The rankings' dataset value of the task subset: the rows naming it, else the unnamed rows."""
    if hf_subset in rankings.datasets:
        return hf_subset
    if "" in rankings.datasets:
        return ""
    return None  # Rankings.queries raises the no-rankings error naming what the rows do name


def _ids(queries: Any) -> list[str]:
    """The asked query ids, from mteb's query table or a plain mapping."""
    if isinstance(queries, Mapping):
        return [str(query_id) for query_id in queries]
    return [str(query_id) for query_id in queries["id"]]


def _of_query(scores: dict[str, float], pool: list[ID] | None, top_k: int) -> dict[str, float]:
    """One query's scores, inside its pool when the task carries one, capped at ``top_k`` (mteb's own cap is
    1000; :data:`MTEB_MAX_DOCS` is the predictions file's); ties by document id descending."""
    if pool is not None:
        inside = set(pool)
        scores = {doc_id: value for doc_id, value in scores.items() if doc_id in inside}
    if len(scores) > min(top_k, MTEB_MAX_DOCS):
        ordered = sorted(scores.items(), key=lambda item: (item[1], item[0]), reverse=True)
        scores = dict(ordered[: min(top_k, MTEB_MAX_DOCS)])
    return scores
