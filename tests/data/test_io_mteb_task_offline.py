"""Offline tests of the ``mteb:<Task>`` reader: a stub task object, no Hub and no network.

``tests/data/test_io_mteb_task.py`` is network-gated and loads real tasks; it is the reader's only
other test. These tests build the smallest ``RetrievalSplitData`` the reader accepts and patch
``mteb.get_task``, so the conversion and the refusals run in the gated CI job (the ``mteb`` extra)
without a network.
"""

from __future__ import annotations

from typing import Any

import pytest

mteb = pytest.importorskip("mteb")

from datasets import Dataset  # noqa: E402
from mteb.abstasks import AbsTask  # noqa: E402
from mteb.abstasks.retrieval import AbsTaskRetrieval  # noqa: E402
from mteb.abstasks.task_metadata import TaskMetadata  # noqa: E402

from rcp_ndcg.data import load_dataset  # noqa: E402
from rcp_ndcg.data.io.mteb_task import MtebTaskReader  # noqa: E402
from rcp_ndcg.errors import ConfigError  # noqa: E402


def _split_data() -> dict[str, Any]:
    return {
        "corpus": Dataset.from_list([{"id": "d1", "text": "doc one"}, {"id": "d2", "text": "doc two"}]),
        "queries": Dataset.from_list([{"id": "q1", "text": "query one"}]),
        "relevant_docs": {"q1": {"d1": 1}},
        "top_ranked": {"q1": ["d1", "d2"]},
    }


class _StubRetrievalTask(AbsTaskRetrieval):
    """The smallest retrieval task mteb's loader contract accepts (no Hub, no remote code)."""

    metadata = TaskMetadata(
        name="StubRetrieval",
        description="an offline stub task",
        dataset={"path": "stub/stub", "revision": "0" * 40},
        type="Retrieval",
        eval_langs={"default": ["eng-Latn"]},
        main_score="ndcg_at_10",
        category="t2t",
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        self.dataset = {"default": {"test": _split_data()}}
        self.data_loaded = True


class _NotARetrievalTask(AbsTask):
    """What ``mteb.get_task`` returns for a classification task: an ``AbsTask``, not a retrieval one.

    It carries the metadata, the filters and ``load_data`` the reader walks before the conversion, so
    without the guard the reader reaches ``convert_v1_dataset_format_to_v2`` and raises a bare
    ``AttributeError`` instead of a typed refusal.
    """

    metadata = TaskMetadata(
        name="StubClassification",
        description="an offline stub classification task",
        dataset={"path": "stub/classification", "revision": "0" * 40},
        type="Classification",
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        category="t2t",
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        self.data_loaded = True

    def _evaluate_subset(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("the stub is never evaluated")

    def _calculate_descriptive_statistics_from_split(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("the stub has no statistics")


def test_a_stub_retrieval_task_reads_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    task = _StubRetrievalTask()
    monkeypatch.setattr(mteb, "get_task", lambda name: task)

    dataset = load_dataset("mteb:StubRetrieval")

    assert dataset.task == "StubRetrieval"
    assert (dataset.subset, dataset.split) == ("default", "test")
    assert {d.id for d in dataset.corpus.values()} == {"d1", "d2"}
    assert {q.id for q in dataset.queries.values()} == {"q1"}
    assert dataset.qrels["q1"]["d1"] == 1


def test_a_non_retrieval_task_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """A classification task has no ``convert_v1_dataset_format_to_v2``: the reader must refuse it with a
    typed error and a hint, not reach mteb's loader and raise a bare ``AttributeError``."""
    monkeypatch.setattr(mteb, "get_task", lambda name: _NotARetrievalTask())

    with pytest.raises(ConfigError, match="not a retrieval task") as caught:
        list(MtebTaskReader("SomeClassificationTask").documents())

    assert "AbsTaskRetrieval" in str(caught.value.hint)
