"""The mteb tasks of the public suites: the float-gain metric offline, the tasks themselves with mteb and the Hub."""

from __future__ import annotations

import math
import os
import sys

import pytest

import rcp_ndcg.eval.mteb as mteb_module
from rcp_ndcg.errors import ConfigError, DataError
from rcp_ndcg.eval.mteb import get_tasks, ndcg_float_scores, task_metadata, task_subsets

LOG2_3 = math.log2(3)

# The published files that key their task table by task name (the 2026-10 rename): the table's keys are the
# published task names and `_SUBSETS` maps the subset aliases to them.
_NEW_FORMAT_FILE = (
    '_REVISION = "abc123"\n'
    '_TASK_METADATA = json.loads(r"""{'
    '"BrightAopsRCPReranking": {'
    '"name": "BrightAopsRCPReranking", "type": "Reranking", '
    '"description": "Reranking over a 150-document candidate pool, scored with NDCG over continuous relevance '
    'gains (`ndcg_float_at_10`).", '
    '"dataset": {"path": "fabianschmidt-cohere/rcp-ndcg-bright", "revision": "abc123"}, '
    '"eval_langs": ["eng-Latn"], "eval_splits": ["test"], "main_score": "ndcg_float_at_10", '
    '"modalities": ["text"], "category": "t2t"'
    '}}""")\n'
    '_SUBSETS = json.loads(r"""{"aops": {"task": "BrightAopsRCPReranking"}}""")\n'
)


def test_ndcg_float_credits_tied_documents_their_group_mean(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "mteb._evaluators.retrieval_metrics", None)  # the metric alone
    gains = {"q1": {"a": 1.0, "b": 0.0, "c": 0.5}, "q2": {"a": 1.0}}
    results = {"q1": {"a": 1.0, "b": 1.0, "c": 0.0}, "q2": {}}

    scores = ndcg_float_scores(gains, results, k_values=(2,))

    ideal = 1.0 + 0.5 / LOG2_3
    assert scores == {"ndcg_float_at_2": round(((0.5 + 0.5 / LOG2_3) / ideal + 0.0) / 2, 5)}
    with pytest.raises(DataError, match="negative gain"):
        ndcg_float_scores({"q1": {"a": -1.0}}, {"q1": {"a": 1.0}}, k_values=(2,))
    with pytest.raises(DataError, match="q9") as caught:  # the docstring once promised ValueError, the code KeyError
        ndcg_float_scores({}, {"q9": {"a": 1.0}}, k_values=(2,))
    assert caught.value.hint or ""
    with pytest.raises(DataError, match="finite") as caught:  # the score came from the model, not the gains
        ndcg_float_scores({"q1": {"a": 1.0}}, {"q1": {"a": float("nan")}}, k_values=(2,))
    assert caught.value.hint or ""


def test_task_metadata_is_read_from_the_published_task_file_as_data() -> None:
    source = '_REVISION = "abc123"\n_TASK_METADATA = json.loads(r"""{"aops": {"name": "BrightAopsRCPRetrieval"}}""")\n'

    assert task_metadata(source) == ("abc123", {"aops": {"name": "BrightAopsRCPRetrieval"}})
    with pytest.raises(DataError, match="not an rcp_ndcg_tasks.py"):
        task_metadata("import os\n")


def test_task_subsets_reads_the_published_alias_map_as_data() -> None:
    """`_SUBSETS` of a task-keyed file maps the subset aliases to the published task names; an older file has
    none and its table is keyed by the subset names themselves."""
    assert task_subsets(_NEW_FORMAT_FILE) == {"aops": "BrightAopsRCPReranking"}
    old = '_TASK_METADATA = json.loads(r"""{"aops": {"name": "BrightAopsRCPRetrieval"}}""")\n'
    assert task_subsets(old) == {}


def test_unknown_suites_and_modes_are_refused() -> None:
    with pytest.raises(ConfigError, match="unknown suite"):
        get_tasks("msmarco")
    with pytest.raises(ConfigError, match="mode"):
        get_tasks("bright", mode="dense")  # type: ignore[arg-type]


def test_empty_or_repeated_subset_names_are_refused() -> None:
    """``names=[]`` used to mean "all subsets" and a repeated name built the task twice: both are refused."""
    with pytest.raises(ConfigError, match="names is empty"):
        get_tasks("bright", [])
    with pytest.raises(ConfigError, match="twice") as caught:
        get_tasks("bright", ["aops", "aops"])
    assert "aops" in caught.value.message


def test_get_tasks_accepts_the_published_names_and_the_subset_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The published files that key their task table by task name ship `_SUBSETS` for the subset aliases:
    `get_tasks` accepts both spellings (the same task either way), refuses an unknown name naming the
    spellings it has, and derives the retrieval view's published name."""
    pytest.importorskip("mteb")
    monkeypatch.setattr(mteb_module, "_hub_text", lambda repo, path: _NEW_FORMAT_FILE)

    by_alias = get_tasks("bright", ["aops"])
    by_name = get_tasks("bright", ["BrightAopsRCPReranking"])

    assert [task.metadata.name for task in by_alias] == ["BrightAopsRCPReranking"]
    assert [task.metadata for task in by_alias] == [task.metadata for task in by_name]
    with pytest.raises(ConfigError, match="unknown subsets") as caught:
        get_tasks("bright", ["nope"])
    assert "aops" in caught.value.message and "BrightAopsRCPReranking" in caught.value.message
    with pytest.raises(ConfigError, match="twice") as caught:
        get_tasks("bright", ["aops", "BrightAopsRCPReranking"])
    assert "twice" in caught.value.message
    retrieval = get_tasks("bright", ["aops"], mode="retrieval")
    assert [task.metadata.name for task in retrieval] == ["BrightAopsRCPRetrieval"]
    assert retrieval[0].metadata.main_score == "ndcg_at_10"


@pytest.mark.network
@pytest.mark.skipif(not os.environ.get("RCP_NDCG_NETWORK_TESTS"), reason="set RCP_NDCG_NETWORK_TESTS=1 (HF Hub)")
def test_the_bright_tasks_carry_the_published_names() -> None:
    pytest.importorskip("mteb")

    by_alias = get_tasks("bright", ["aops"])
    by_name = get_tasks("bright", ["BrightAopsRCPReranking"])

    assert [task.metadata.name for task in by_alias] == ["BrightAopsRCPReranking"]
    assert [task.metadata for task in by_alias] == [task.metadata for task in by_name]
