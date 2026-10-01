"""The mteb tasks of the public suites: the float-gain metric offline, the tasks themselves with mteb and the Hub."""

from __future__ import annotations

import math
import os
import sys

import pytest

from rcp_ndcg.errors import ConfigError, DataError
from rcp_ndcg.eval.mteb import get_tasks, ndcg_float_scores, task_metadata

LOG2_3 = math.log2(3)


def test_ndcg_float_credits_tied_documents_their_group_mean(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "mteb._evaluators.retrieval_metrics", None)  # the metric alone
    gains = {"q1": {"a": 1.0, "b": 0.0, "c": 0.5}, "q2": {"a": 1.0}}
    results = {"q1": {"a": 1.0, "b": 1.0, "c": 0.0}, "q2": {}}

    scores = ndcg_float_scores(gains, results, k_values=(2,))

    ideal = 1.0 + 0.5 / LOG2_3
    assert scores == {"ndcg_float_at_2": round(((0.5 + 0.5 / LOG2_3) / ideal + 0.0) / 2, 5)}
    with pytest.raises(ValueError, match="negative gain"):
        ndcg_float_scores({"q1": {"a": -1.0}}, {"q1": {"a": 1.0}}, k_values=(2,))
    with pytest.raises(KeyError):
        ndcg_float_scores({}, {"q9": {"a": 1.0}}, k_values=(2,))


def test_task_metadata_is_read_from_the_published_task_file_as_data() -> None:
    source = '_REVISION = "abc123"\n_TASK_METADATA = json.loads(r"""{"aops": {"name": "BrightAopsRCPRetrieval"}}""")\n'

    assert task_metadata(source) == ("abc123", {"aops": {"name": "BrightAopsRCPRetrieval"}})
    with pytest.raises(DataError, match="not an rcp_ndcg_tasks.py"):
        task_metadata("import os\n")


def test_unknown_suites_and_modes_are_refused() -> None:
    with pytest.raises(ConfigError, match="unknown suite"):
        get_tasks("msmarco")
    with pytest.raises(ConfigError, match="mode"):
        get_tasks("bright", mode="dense")  # type: ignore[arg-type]


@pytest.mark.network
@pytest.mark.skipif(not os.environ.get("RCP_NDCG_NETWORK_TESTS"), reason="set RCP_NDCG_NETWORK_TESTS=1 (HF Hub)")
def test_the_bright_tasks_carry_the_published_names() -> None:
    pytest.importorskip("mteb")

    tasks = get_tasks("bright", ["aops"])

    assert [task.metadata.name for task in tasks] == ["BrightAopsRCPRetrieval"]
    assert tasks[0].metadata.main_score == "ndcg_float_at_10"
