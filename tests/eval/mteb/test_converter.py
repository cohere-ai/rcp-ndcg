"""`tools/republish_mteb.py`: the write-and-validate half runs offline (a records-built suite stands in for the
published repositories; the Hub load is the owner's network run)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

MTEB = pytest.importorskip("mteb")

TOOLS = Path(__file__).resolve().parents[3] / "tools" / "republish_mteb.py"


def a_suite() -> Any:
    from rcp_ndcg.data.dataset import Dataset

    def subset(
        name: str, query: str, corpus: list[dict[str, str]], qrels: list[dict[str, Any]], **extra: Any
    ) -> Dataset:
        return Dataset.from_records(
            name=name,
            queries=[{"query_id": "q1", "text": query}],
            corpus=corpus,
            qrels=qrels,
            **extra,
        )

    return Dataset(
        name="nanobeir",
        subsets=(
            subset(
                "NanoArguAnaRetrieval",
                "what is a tortoise",
                [{"doc_id": "d1", "text": "a tortoise is a reptile"}, {"doc_id": "d2", "text": "a tortoise"}],
                [{"query_id": "q1", "doc_id": "d1", "grade": 2, "gain": 0.9, "theta": 1.1}],
                candidates={"q1": ["d1", "d2"]},
            ),
            subset(
                "NanoFEVERRetrieval",
                "second",
                [{"doc_id": "d3", "text": "fever text"}],
                [{"query_id": "q1", "doc_id": "d3", "grade": 1}],
                excluded={"q1": ["d3"]},
            ),
        ),
    )


@pytest.fixture
def module(monkeypatch: pytest.MonkeyPatch) -> Any:
    import sys

    sys.path.insert(0, str(TOOLS.parent))
    import republish_mteb

    monkeypatch.setattr(republish_mteb, "OWNER", "org")
    return republish_mteb


def test_republish_validates_the_written_layout_with_mteb_s_own_loader(
    tmp_path: Path, module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "_load", lambda uri, revision: a_suite())
    monkeypatch.setattr(module, "_card", lambda repo, dataset: None)

    summary = module.republish("rcp-ndcg-nanobeir", tmp_path)

    assert summary["repo"] == "rcp-ndcg-nanobeir"
    assert set(summary["subsets"]) == {"NanoArguAnaRetrieval", "NanoFEVERRetrieval"}
    assert (tmp_path / "rcp-ndcg-nanobeir" / "NanoArguAnaRetrieval-qrels" / "test-00000-of-00001.parquet").is_file()
    readme = (tmp_path / "rcp-ndcg-nanobeir" / "README.md").read_text()
    assert "config_name: NanoFEVERRetrieval-qrels" in readme  # one README over all subsets
    assert "path: NanoFEVERRetrieval-qrels/test-*" in readme  # the eval split, not the published train


def test_a_failing_subset_is_reported_with_the_pair_named(
    tmp_path: Path, module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite = a_suite()
    broken = suite.model_copy(
        update={"subsets": suite.subsets[:1] + (suite.subsets[1].model_copy(update={"qrels": {"q1": {"d9": 1}}}),)}
    )
    monkeypatch.setattr(module, "_load", lambda uri, revision: broken)
    monkeypatch.setattr(module, "_card", lambda repo, dataset: None)

    with pytest.raises(Exception, match="d9"):
        module.republish("rcp-ndcg-nanobeir", tmp_path)


def test_the_exclusion_folds_into_the_pool_mteb_reads(
    tmp_path: Path, module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mteb.abstasks.retrieval_dataset_loaders import RetrievalDatasetLoader

    monkeypatch.setattr(module, "_load", lambda uri, revision: a_suite())
    monkeypatch.setattr(module, "_card", lambda repo, dataset: None)

    module.republish("rcp-ndcg-nanobeir", tmp_path)

    loaded = RetrievalDatasetLoader(
        hf_repo=str(tmp_path / "rcp-ndcg-nanobeir"), revision="main", split="test", config="NanoFEVERRetrieval"
    ).load()
    assert loaded["top_ranked"] == {"q1": []}  # the excluded document left the pool
    assert loaded["relevant_docs"] == {"q1": {"d3": 1}}
