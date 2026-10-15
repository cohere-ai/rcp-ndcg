"""Offline tests of the ``mteb:<Task>`` reader: a stub task object, no Hub and no network.

``tests/data/test_io_mteb_task.py`` is network-gated and loads real tasks; it is the reader's only
other test. These tests patch ``mteb.get_task`` and run in the gated CI job (the ``mteb`` extra)
without a network.
"""

from __future__ import annotations

from typing import Any

import pytest

mteb = pytest.importorskip("mteb")

from mteb.abstasks import AbsTask  # noqa: E402
from mteb.abstasks.task_metadata import TaskMetadata  # noqa: E402

from rcp_ndcg.data.io.mteb_task import MtebTaskReader  # noqa: E402
from rcp_ndcg.errors import ConfigError  # noqa: E402


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


def test_a_non_retrieval_task_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """A classification task has no ``convert_v1_dataset_format_to_v2``: the reader must refuse it with a
    typed error and a hint, not reach mteb's loader and raise a bare ``AttributeError``."""
    monkeypatch.setattr(mteb, "get_task", lambda name: _NotARetrievalTask())

    with pytest.raises(ConfigError, match="not a retrieval task") as caught:
        list(MtebTaskReader("SomeClassificationTask").documents())

    assert "AbsTaskRetrieval" in str(caught.value.hint)
