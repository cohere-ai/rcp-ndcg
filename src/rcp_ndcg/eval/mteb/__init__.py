"""The public suites as mteb tasks: ``get_tasks(suite)`` (extra ``mteb``, mteb >= 2.0.1).

Each task reports mteb's usual integer-qrels metrics plus ``ndcg_float_at_k``: nDCG over the released continuous
gains (the ``gain`` column of ``{subset}-qrels``) with linear gains and group-mean ties, the metric of mteb PR
5516. The main score is ``ndcg_float_at_10``.

The task names and metadata are the ones each suite's dataset repository publishes in its
``rcp_ndcg_tasks.py``; they are read from that file as data, never executed. When the installed mteb already
ships a task (PR 5516 merged) at the same data revision, ``get_tasks`` returns mteb's own.

Views:

* ``"reranking"`` (default): each query is scored over its ``{subset}-top_ranked`` pool, exactly the documents
  that have gains.
* ``"retrieval"``: full-corpus search minus ``{subset}-excluded`` where present; only the integer-qrels metrics
  are reported, because gains exist for the pooled documents only.
"""

from __future__ import annotations

import functools
import json
import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from rcp_ndcg_core.metric import ndcg

from rcp_ndcg.data.dataset import SUITES
from rcp_ndcg.errors import CapabilityError, ConfigError, DataError, MissingInputError

K_VALUES = (1, 3, 5, 10, 20, 100, 1000)

Mode = Literal["reranking", "retrieval"]


def ndcg_float_scores(
    gains: Mapping[str, Mapping[str, float]],
    results: Mapping[str, Mapping[str, float]],
    k_values: Sequence[int] = K_VALUES,
) -> dict[str, float]:
    """Mean nDCG@k over continuous gains with group-mean ties: ``{"ndcg_float_at_<k>": value}``, 5 decimals.

    With mteb installed, the abstention nAUCs of the per-query values are added (``nauc_ndcg_float_at_<k>_*``).

    Args:
        gains: ``{query_id: {doc_id: gain}}``, finite and non-negative; the ideal DCG sorts all of a query's gains
            (a calibration's table: ``Calibration.gains(dataset)``).
        results: ``{query_id: {doc_id: score}}`` of the model; every query needs gains. An empty ranking scores 0.
        k_values: The cutoffs.

    Returns:
        ``{"ndcg_float_at_<k>": value}``, the mean nDCG@k over the queries (5 decimals), plus mteb's abstention
        nAUCs of the per-query values when mteb is installed.

    Raises:
        DataError: A non-finite or negative gain, a non-finite score, or a query of ``results`` without gains.
    """
    per_k: dict[int, list[float]] = defaultdict(list)
    for query_id, scores in results.items():
        try:
            query_gains = gains[query_id]
        except KeyError:
            raise DataError(
                f"query {query_id!r} has no gains, and every scored query needs them",
                hint="pass gains covering every query of results (a calibration's per-dataset gains for a suite)",
            ) from None
        if any(not math.isfinite(g) or g < 0 for g in query_gains.values()):
            raise DataError(
                f"Non-finite or negative gain for query {query_id!r}.",
                hint="the float-gain metric scores gains in [0, 1]; grades belong in the integer-qrels metrics",
            )
        for k in k_values:
            per_k[k].append(ndcg(scores, query_gains, k=k, ties="group_mean"))
    summary = {f"ndcg_float_at_{k}": round(math.fsum(v) / len(v), 5) for k, v in per_k.items() if v}
    try:  # mteb's abstention nAUCs of the per-query values, as mteb PR 5516 reports them
        from mteb._evaluators.retrieval_metrics import evaluate_abstention
    except ImportError:
        return summary
    naucs = evaluate_abstention(dict(results), {f"NDCG_float@{k}": v for k, v in per_k.items() if v})
    return {**summary, **{key.replace("@", "_at_").lower(): value for key, value in naucs.items()}}


def get_tasks(
    suite: str,
    names: Sequence[str] | None = None,
    *,
    mode: Mode = "reranking",
    revision: str | None = None,
) -> list[Any]:
    """The mteb tasks of a public suite.

    Args:
        suite: ``"nanobeir"``, ``"bright"``, ``"vidore"`` or ``"trecdl"``.
        names: Subsets (for ViDoRe v3: domains) to return; ``None`` returns all.
        mode: ``"reranking"`` (the recommended view) or ``"retrieval"``.
        revision: The data revision; ``None`` is the revision the suite's task file was released with.

    Returns:
        ``mteb`` task objects, ready for ``mteb.evaluate``.
    """
    if suite not in SUITES:
        raise ConfigError(f"unknown suite {suite!r}; expected one of {sorted(SUITES)}")
    if mode not in ("reranking", "retrieval"):
        raise ConfigError("mode must be 'reranking' or 'retrieval'")
    if names is not None and not names:
        raise ConfigError(
            "names is empty; pass subset names, or None for all of them",
            hint="available subsets: task_metadata on the suite's rcp_ndcg_tasks.py, or pass no names for all of them",
        )
    if names is not None:
        repeated = sorted({name for name in names if list(names).count(name) > 1})
        if repeated:
            raise ConfigError(
                f"names {repeated} appear twice; pass each subset once",
                hint="drop the repeated names, or pass no names for all of them",
            )
    repo = SUITES[suite].repo
    released, metadata = task_metadata(_hub_text(repo, "rcp_ndcg_tasks.py"))
    revision = revision or released
    chosen = list(names) if names else sorted(metadata)
    unknown = sorted(set(chosen) - set(metadata))
    if unknown:
        raise ConfigError(f"unknown subsets {unknown}; available: {sorted(metadata)}")
    return [_make_task(repo, metadata[name], mode, revision, shared_corpus=suite == "vidore") for name in chosen]


def task_metadata(source: str) -> tuple[str, dict[str, dict[str, Any]]]:
    """``(released revision, {subset: TaskMetadata fields})`` from the text of a suite's ``rcp_ndcg_tasks.py``."""
    revision = re.search(r'^_REVISION = "([0-9a-f]+)"', source, re.M)
    table = re.search(r'_TASK_METADATA[^=]*= json\.loads\(r"""(.*?)"""\)', source, re.S)
    if revision is None or table is None:
        raise DataError("not an rcp_ndcg_tasks.py: no _REVISION or _TASK_METADATA")
    return revision.group(1), json.loads(table.group(1))


def _hub_text(repo: str, path: str) -> str:
    from rcp_ndcg.data.dataset import _hub_file

    local = _hub_file(repo, path, None)
    if local is None:
        raise MissingInputError(
            f"hf://{repo}: {path} does not exist",
            hint="the suite's published task file is missing: check the suite name, or the revision",
        )
    return local.read_text(encoding="utf-8")


def _make_task(repo: str, fields: dict[str, Any], mode: Mode, revision: str, *, shared_corpus: bool) -> Any:
    from mteb.abstasks.task_metadata import TaskMetadata

    fields = json.loads(json.dumps(fields))  # deep copy
    if mode == "reranking":
        in_tree = _in_tree_task(fields["name"])
        if in_tree is not None and in_tree.metadata.dataset["revision"] == revision:
            return in_tree
    fields["dataset"] = {"path": repo, "revision": revision}
    if mode == "retrieval":
        fields["name"] += ".retrieval"
        fields["main_score"] = "ndcg_at_10"
        fields["description"] += " Full-corpus retrieval view: only the integer-qrels metrics are reported."
    if isinstance(fields.get("date"), list):
        fields["date"] = tuple(fields["date"])
    known = set(TaskMetadata.model_fields)
    meta = TaskMetadata(**{k: v for k, v in fields.items() if k in known})  # tolerate older mteb
    attributes = {"metadata": meta, "rcp_mode": mode, "shared_corpus": shared_corpus}
    return type(fields["name"].replace(".", "_"), (_task_base(),), attributes)()


def _in_tree_task(name: str) -> Any | None:
    """mteb's own task of this name, if the installed mteb ships it."""
    try:
        from mteb.get_tasks import _TASKS_REGISTRY

        return _TASKS_REGISTRY[name]() if name in _TASKS_REGISTRY else None
    except Exception:  # noqa: BLE001 - any mteb version quirk means "not available"
        return None


@functools.cache
def _task_base() -> type:
    """The task class, built on first use so importing this module needs no mteb."""
    from datasets import get_dataset_config_names, load_dataset
    from mteb.abstasks.retrieval import AbsTaskRetrieval

    class RCPRetrieval(AbsTaskRetrieval):
        """Loads the gain column of ``{subset}-qrels`` and reports ``ndcg_float_at_k``."""

        rcp_mode: Mode = "reranking"
        shared_corpus: bool = False
        k_values = K_VALUES

        def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
            if self.data_loaded:
                return
            if not self.shared_corpus:
                super().load_data(num_proc=num_proc, **kwargs)
                return
            # ViDoRe v3: a domain's language subsets share one page-image corpus; load it once.
            from mteb.abstasks.retrieval_dataset_loaders import RetrievalDatasetLoader

            path, rev = self.metadata.dataset["path"], self.metadata.dataset["revision"]
            shared: dict[str, Any] = {}

            class SharedCorpusLoader(RetrievalDatasetLoader):
                def _load_corpus(self, *args: Any, **kw: Any):  # noqa: N805 - mteb's signature
                    if "corpus" not in shared:
                        shared["corpus"] = super()._load_corpus(*args, **kw)
                    return shared["corpus"]

            self.dataset = {}
            for hf_subset in self.hf_subsets:
                for split in self.metadata.eval_splits:
                    loader = SharedCorpusLoader(hf_repo=path, revision=rev, split=split, config=hf_subset)
                    self.dataset.setdefault(hf_subset, {})[split] = loader.load(num_proc=num_proc)
            self.dataset_transform(num_proc=num_proc)
            self.data_loaded = True

        def dataset_transform(self, num_proc: int | None = None, **kwargs: Any) -> None:
            path, rev = self.metadata.dataset["path"], self.metadata.dataset["revision"]
            configs = set(get_dataset_config_names(path, revision=rev))
            self._gains: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
            for hf_subset, splits in self.dataset.items():
                self._gains[hf_subset] = {}
                for split, data in splits.items():
                    if self.rcp_mode == "reranking":
                        qrels = load_dataset(path, f"{hf_subset}-qrels", split=split, revision=rev)
                        gains: dict[str, dict[str, float]] = defaultdict(dict)
                        for q, d, g in zip(qrels["query-id"], qrels["corpus-id"], qrels["gain"], strict=True):
                            if g is not None:
                                gains[str(q)][str(d)] = float(g)
                        self._gains[hf_subset][split] = dict(gains)
                        # encode only the candidates (a bi-encoder would otherwise embed the whole corpus)
                        keep = {d for docs in data["top_ranked"].values() for d in docs}
                        corpus = data["corpus"]
                        data["corpus"] = corpus.select([i for i, d in enumerate(corpus["id"]) if d in keep])
                    elif f"{hf_subset}-excluded" in configs:
                        # the full corpus minus each query's excluded documents
                        excluded_rows = load_dataset(path, f"{hf_subset}-excluded", split=split, revision=rev)
                        excluded = {r["query-id"]: set(r["excluded-corpus-ids"]) for r in excluded_rows}
                        ids = list(data["corpus"]["id"])
                        data["top_ranked"] = {
                            q: [d for d in ids if d not in excluded[q]] if q in excluded else ids
                            for q in data["queries"]["id"]
                        }
                    else:
                        data["top_ranked"] = None

        def evaluate(self, model: Any, split: str = "test", *args: Any, **kwargs: Any) -> Any:
            if self.rcp_mode == "retrieval":
                from mteb.models import CrossEncoderProtocol, SearchProtocol

                if isinstance(model, CrossEncoderProtocol) and not isinstance(model, SearchProtocol):
                    raise CapabilityError(
                        f"{self.metadata.name}: cross-encoders cannot do full-corpus retrieval; "
                        "use the reranking view (mode='reranking')."
                    )
            return super().evaluate(model, split, *args, **kwargs)

        def task_specific_scores(self, scores, qrels, results, hf_split: str, hf_subset: str) -> dict[str, float]:
            if self.rcp_mode != "reranking":
                return {}
            if getattr(self, "skip_first_result", False):
                raise CapabilityError("skip_first_result is not supported by the float-gains metric.")
            gains = self._gains[hf_subset][hf_split]
            # the queries mteb's integer metrics average over: those present in `results`
            scored = {q: results[q] for q in qrels if q in results}
            if getattr(self, "ignore_identical_ids", False):
                gains = {q: {d: g for d, g in docs.items() if d != q} for q, docs in gains.items()}
                scored = {q: {d: s for d, s in docs.items() if d != q} for q, docs in scored.items()}
            return ndcg_float_scores(gains, scored, self.k_values)

    return RCPRetrieval


__all__ = ["K_VALUES", "get_tasks", "ndcg_float_scores", "task_metadata"]
