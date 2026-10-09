"""`tools/republish_mteb.py`: the write-and-validate half runs offline (a records-built suite stands in for the
published repositories; the Hub load is the owner's network run)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

MTEB = pytest.importorskip("mteb")

TOOLS = Path(__file__).resolve().parents[3] / "tools" / "republish_mteb.py"


def a_suite(*, split: str = "test") -> Any:
    from rcp_ndcg.data.dataset import Dataset

    def subset(
        name: str, query: str, corpus: list[dict[str, str]], qrels: list[dict[str, Any]], **extra: Any
    ) -> Dataset:
        return Dataset.from_records(
            name=name,
            queries=[{"query_id": "q1", "text": query}],
            corpus=corpus,
            qrels=qrels,
            split=split,
            **extra,
        )

    return Dataset(
        name="nanobeir",
        revision="abc123",
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


def a_task_source(splits: dict[str, str]) -> str:
    """A published `rcp_ndcg_tasks.py` in the 2026-10 rename format: the table keyed by task name, `_SUBSETS`
    mapping each subset to it, the prompt and the real split in the metadata."""
    table: dict[str, dict[str, Any]] = {}
    aliases: dict[str, dict[str, str]] = {}
    for subset, split in splits.items():
        name = f"{subset}RCPReranking"
        table[name] = {
            "name": name,
            "type": "Reranking",
            "description": f"Reranking over {subset}'s candidate pool.",
            "dataset": {"path": "org/rcp-ndcg-nanobeir", "revision": "abc"},
            "eval_langs": {subset: ["eng-Latn"]},
            "eval_splits": [split],
            "main_score": "ndcg_float_at_10",
            "prompt": {"query": "find documents that answer"},
        }
        aliases[subset] = {"task": name}
    return (
        '_REVISION = "abc"\n'
        f'_TASK_METADATA = json.loads(r"""{json.dumps(table)}""")\n'
        f'_SUBSETS = json.loads(r"""{json.dumps(aliases)}""")\n'
    )


def a_nanobeir_task_source(*, split: str = "test") -> str:
    return a_task_source({"NanoArguAnaRetrieval": split, "NanoFEVERRetrieval": split})


def a_two_task_source(*, ocr_split: str = "test") -> str:
    """A ViDoRe-v3-like file: one language subset maps to two tasks (the page-image one and its OCR view)."""
    subset = "computer_science__english"
    base: dict[str, Any] = {
        "name": "Vidore3ComputerScienceRCPReranking",
        "type": "DocumentUnderstanding",
        "description": "pages",
        "dataset": {"path": "org/rcp-ndcg-vidore-v3", "revision": "abc"},
        "eval_langs": {subset: ["eng-Latn"]},
        "eval_splits": ["test"],
        "main_score": "ndcg_float_at_10",
        "prompt": {"query": "find a screenshot"},
    }
    ocr = {**base, "name": "Vidore3ComputerScienceRCPRerankingOCR", "eval_splits": [ocr_split]}
    table = {"Vidore3ComputerScienceRCPReranking": base, "Vidore3ComputerScienceRCPRerankingOCR": ocr}
    aliases = {subset: {"task": "Vidore3ComputerScienceRCPReranking", "ocr": "Vidore3ComputerScienceRCPRerankingOCR"}}
    return (
        '_REVISION = "abc"\n'
        f'_TASK_METADATA = json.loads(r"""{json.dumps(table)}""")\n'
        f'_SUBSETS = json.loads(r"""{json.dumps(aliases)}""")\n'
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
    monkeypatch.setattr(module, "_task_source", lambda repo: a_nanobeir_task_source())
    monkeypatch.setattr(module, "_card", lambda repo, dataset, source: None)

    summary = module.republish("rcp-ndcg-nanobeir-layout", tmp_path)

    assert summary["repo"] == "rcp-ndcg-nanobeir-layout"
    assert set(summary["subsets"]) == {"NanoArguAnaRetrieval", "NanoFEVERRetrieval"}
    assert (
        tmp_path / "rcp-ndcg-nanobeir-layout" / "NanoArguAnaRetrieval-qrels" / "test-00000-of-00001.parquet"
    ).is_file()
    readme = (tmp_path / "rcp-ndcg-nanobeir-layout" / "README.md").read_text()
    assert "config_name: NanoFEVERRetrieval-qrels" in readme  # one README over all subsets
    assert "path: NanoFEVERRetrieval-qrels/test-*" in readme


def test_the_converter_writes_the_task_definition_split(
    tmp_path: Path, module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Decision 40: the writer keeps the PR's split names (NanoBEIR `train`, BRIGHT `standard`, ViDoRe v3
    `test`) instead of re-laying every repository to `test`."""
    monkeypatch.setattr(module, "_load", lambda uri, revision: a_suite(split="train"))
    monkeypatch.setattr(module, "_task_source", lambda repo: a_nanobeir_task_source(split="train"))
    monkeypatch.setattr(module, "_card", lambda repo, dataset, source: None)

    module.republish("rcp-ndcg-nanobeir-train", tmp_path)

    assert (
        tmp_path / "rcp-ndcg-nanobeir-train" / "NanoArguAnaRetrieval-qrels" / "train-00000-of-00001.parquet"
    ).is_file()
    assert not (
        tmp_path / "rcp-ndcg-nanobeir-train" / "NanoArguAnaRetrieval-qrels" / "test-00000-of-00001.parquet"
    ).exists()
    assert "split: train" in (tmp_path / "rcp-ndcg-nanobeir-train" / "README.md").read_text()


def test_the_converter_refuses_a_task_definition_with_a_different_split(
    tmp_path: Path, module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "_load", lambda uri, revision: a_suite(split="standard"))
    monkeypatch.setattr(module, "_task_source", lambda repo: a_nanobeir_task_source(split="test"))
    monkeypatch.setattr(module, "_card", lambda repo, dataset, source: None)

    with pytest.raises(Exception, match="standard.*test|test.*standard") as caught:
        module.republish("rcp-ndcg-nanobeir-split", tmp_path)
    assert "NanoArguAnaRetrieval" in str(caught.value)
    assert not (tmp_path / "rcp-ndcg-nanobeir-split").exists()


def test_the_converter_refuses_a_task_definition_missing_a_subset(
    tmp_path: Path, module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "_load", lambda uri, revision: a_suite())
    monkeypatch.setattr(module, "_task_source", lambda repo: a_task_source({"NanoArguAnaRetrieval": "test"}))
    monkeypatch.setattr(module, "_card", lambda repo, dataset, source: None)

    with pytest.raises(Exception, match="NanoFEVERRetrieval") as caught:
        module.republish("rcp-ndcg-nanobeir-subset", tmp_path)
    assert "task" in str(caught.value)
    assert not (tmp_path / "rcp-ndcg-nanobeir-subset").exists()


def test_the_converter_checks_every_task_definition_of_a_subset(
    tmp_path: Path, module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ViDoRe v3 maps one subset to two tasks (the page-image one and its OCR view): a split mismatch on either
    is refused, not only on the first match."""
    from rcp_ndcg.data.dataset import Dataset

    part = Dataset.from_records(
        name="computer_science__english",
        queries=[{"query_id": "q1", "text": "q"}],
        corpus=[{"doc_id": "d1", "text": "page"}],
        qrels=[{"query_id": "q1", "doc_id": "d1", "grade": 1}],
        split="test",
    )
    suite = Dataset(name="vidore", revision="abc123", subsets=(part,))
    monkeypatch.setattr(module, "_load", lambda uri, revision: suite)
    monkeypatch.setattr(module, "_task_source", lambda repo: a_two_task_source(ocr_split="train"))
    monkeypatch.setattr(module, "_card", lambda repo, dataset, source: None)

    with pytest.raises(Exception, match="OCR.*train|train.*OCR") as caught:
        module.republish("rcp-ndcg-vidore-v3", tmp_path)
    assert "computer_science__english" in str(caught.value)


def test_the_card_carries_the_published_prompt_and_the_task_split(
    tmp_path: Path, module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R7: the prompt travels where mteb reads it (`TaskMetadata.prompt`), and the card renders from the
    published task metadata."""
    monkeypatch.setattr(module, "_load", lambda uri, revision: a_suite(split="train"))
    source = a_nanobeir_task_source(split="train")
    monkeypatch.setattr(module, "_task_source", lambda repo: source)

    card = module._card("rcp-ndcg-nanobeir", a_suite(split="train"), source)

    assert card["prompt"] == {"query": "find documents that answer"}
    assert card["eval_splits"] == ["train"]
    assert card["dataset"]["path"] == "org/rcp-ndcg-nanobeir"

    module.republish("rcp-ndcg-nanobeir-card", tmp_path)
    readme = (tmp_path / "rcp-ndcg-nanobeir-card" / "README.md").read_text()
    assert "NanoArguAnaRetrievalRCPReranking" in readme  # mteb's own card template, from the task metadata


def test_a_failing_subset_is_reported_with_the_pair_named(
    tmp_path: Path, module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite = a_suite()
    broken = suite.model_copy(
        update={"subsets": suite.subsets[:1] + (suite.subsets[1].model_copy(update={"qrels": {"q1": {"d9": 1}}}),)}
    )
    monkeypatch.setattr(module, "_load", lambda uri, revision: broken)
    monkeypatch.setattr(module, "_task_source", lambda repo: a_nanobeir_task_source())
    monkeypatch.setattr(module, "_card", lambda repo, dataset, source: None)

    with pytest.raises(Exception, match="d9"):
        module.republish("rcp-ndcg-nanobeir-broken", tmp_path)


def test_the_published_revisions_resolve_under_script_invocation(tmp_path: Path) -> None:
    """The documented invocation is `python tools/republish_mteb.py` from the repository root, where sys.path[0]
    is tools/ and the experiments package is not importable: the revisions load by file path."""
    code = (
        f"import sys; sys.path.insert(0, {str(TOOLS.parent)!r}); import republish_mteb; "
        "print(republish_mteb._published_revision('rcp-ndcg-nanobeir'))"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=tmp_path, timeout=120)
    assert done.returncode == 0, done.stderr
    assert len(done.stdout.strip()) == 40  # the pinned commit sha, not a traceback


def test_the_exclusion_folds_into_the_pool_mteb_reads(
    tmp_path: Path, module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mteb.abstasks.retrieval_dataset_loaders import RetrievalDatasetLoader

    monkeypatch.setattr(module, "_load", lambda uri, revision: a_suite())
    monkeypatch.setattr(module, "_task_source", lambda repo: a_nanobeir_task_source())
    monkeypatch.setattr(module, "_card", lambda repo, dataset, source: None)

    module.republish("rcp-ndcg-nanobeir-excluded", tmp_path)

    loaded = RetrievalDatasetLoader(
        hf_repo=str(tmp_path / "rcp-ndcg-nanobeir-excluded"), revision="main", split="test", config="NanoFEVERRetrieval"
    ).load()
    assert loaded["top_ranked"] == {"q1": []}  # the excluded document left the pool
    assert loaded["relevant_docs"] == {"q1": {"d3": 1}}
