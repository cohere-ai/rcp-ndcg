"""`tools/republish_mteb.py`: the write-and-validate half runs offline (a records-built suite stands in for the
published repositories; the Hub load is the owner's network run)."""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

MTEB = pytest.importorskip("mteb")

TOOLS = Path(__file__).resolve().parents[3] / "tools" / "republish_mteb.py"


def a_subset(name: str, query: str, corpus: list[dict[str, Any]], qrels: list[dict[str, Any]], **extra: Any) -> Any:
    from rcp_ndcg.data.dataset import Dataset

    return Dataset.from_records(
        name=name,
        queries=[{"query_id": "q1", "text": query}],
        corpus=corpus,
        qrels=qrels,
        subset=name,
        **extra,
    ).model_copy(update={"revision": "abc123"})


def a_suite(*, split: str = "test") -> Any:
    from rcp_ndcg.data.dataset import Dataset

    def subset(name: str, query: str, corpus: list[dict[str, Any]], qrels: list[dict[str, Any]], **extra: Any) -> Any:
        return a_subset(name, query, corpus, qrels, **extra).model_copy(update={"split": split})

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


def a_loader(suite: Any) -> Any:
    """The per-subset `_load` the converter now uses: the URI names one subset."""

    def load(uri: str, revision: str | None) -> Any:
        subset = uri.rsplit("/", 1)[-1]
        for part in suite.parts:
            if part.name == subset:
                return part
        raise AssertionError(f"no part for {uri}")

    return load


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


def stub_repo(monkeypatch: pytest.MonkeyPatch, module: Any, *names: str) -> None:
    monkeypatch.setattr(module, "_repo_subsets", lambda repo, revision: tuple(names))


def test_republish_validates_the_written_layout_with_mteb_s_own_loader(
    tmp_path: Path, module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "_load", a_loader(a_suite()))
    monkeypatch.setattr(module, "_task_source", lambda repo: a_nanobeir_task_source())
    stub_repo(monkeypatch, module, "NanoArguAnaRetrieval", "NanoFEVERRetrieval")
    monkeypatch.setattr(module, "_card", lambda repo, dataset, source: None)

    summary = module.republish("rcp-ndcg-nanobeir", tmp_path)

    assert summary["repo"] == "rcp-ndcg-nanobeir"
    assert set(summary["subsets"]) == {"NanoArguAnaRetrieval", "NanoFEVERRetrieval"}
    assert (tmp_path / "rcp-ndcg-nanobeir" / "NanoArguAnaRetrieval-qrels" / "test-00000-of-00001.parquet").is_file()
    readme = (tmp_path / "rcp-ndcg-nanobeir" / "README.md").read_text()
    assert "config_name: NanoFEVERRetrieval-qrels" in readme  # one README over all subsets
    assert "path: NanoFEVERRetrieval-qrels/test-*" in readme


def test_the_converter_writes_the_task_definition_split(
    tmp_path: Path, module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Decision 40: the writer keeps the PR's split names (NanoBEIR `train`, BRIGHT `standard`, ViDoRe v3
    `test`) instead of re-laying every repository to `test`."""
    monkeypatch.setattr(module, "_load", a_loader(a_suite(split="train")))
    monkeypatch.setattr(module, "_task_source", lambda repo: a_nanobeir_task_source(split="train"))
    stub_repo(monkeypatch, module, "NanoArguAnaRetrieval", "NanoFEVERRetrieval")
    monkeypatch.setattr(module, "_card", lambda repo, dataset, source: None)

    module.republish("rcp-ndcg-nanobeir-train", tmp_path)

    assert (
        tmp_path / "rcp-ndcg-nanobeir-train" / "NanoArguAnaRetrieval-qrels" / "train-00000-of-00001.parquet"
    ).is_file()
    assert not (
        tmp_path / "rcp-ndcg-nanobeir-train" / "NanoArguAnaRetrieval-qrels" / "test-00000-of-00001.parquet"
    ).exists()
    assert "split: train" in (tmp_path / "rcp-ndcg-nanobeir-train" / "README.md").read_text()


def test_the_converter_writes_every_subset_the_task_file_defines(
    tmp_path: Path, module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ViDoRe v3's task file defines 48 language subsets; the converter must write all of them, not only the
    eight native-language ones `SUITES` scores (the verifier's blocker)."""
    from rcp_ndcg.data.dataset import Dataset

    names = ["computer_science__english", "computer_science__french", "computer_science__german"]
    parts = tuple(
        a_subset(name, "q", [{"doc_id": "d1", "text": "page"}], [{"query_id": "q1", "doc_id": "d1", "grade": 1}])
        for name in names
    )
    suite = Dataset(name="vidore", revision="abc123", subsets=parts)
    monkeypatch.setattr(module, "_load", a_loader(suite))
    monkeypatch.setattr(module, "_task_source", lambda repo: a_task_source(dict.fromkeys(names, "test")))
    stub_repo(monkeypatch, module, "computer_science__english", "computer_science__french", "computer_science__german")
    monkeypatch.setattr(module, "_card", lambda repo, dataset, source: None)

    summary = module.republish("rcp-ndcg-vidore-v3", tmp_path)

    assert set(summary["subsets"]) == set(names)
    readme = (tmp_path / "rcp-ndcg-vidore-v3" / "README.md").read_text()
    for name in names:
        assert f"config_name: {name}-qrels" in readme


def test_a_shared_corpus_is_written_once_and_read_by_every_language(
    tmp_path: Path, module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The published ViDoRe v3 stores one corpus per domain; `corpus_group` keeps the republished layout from
    holding six copies, and every language's `-corpus` config still reads the shared rows."""
    from datasets import load_dataset

    from rcp_ndcg.data.dataset import Dataset

    names = ["computer_science__english", "computer_science__french"]
    parts = tuple(
        a_subset(name, "q", [{"doc_id": "d1", "text": "page"}], [{"query_id": "q1", "doc_id": "d1", "grade": 1}])
        for name in names
    )
    suite = Dataset(name="vidore", revision="abc123", subsets=parts)
    monkeypatch.setattr(module, "_load", a_loader(suite))
    monkeypatch.setattr(module, "_task_source", lambda repo: a_task_source(dict.fromkeys(names, "test")))
    stub_repo(monkeypatch, module, "computer_science__english", "computer_science__french")
    monkeypatch.setattr(module, "_card", lambda repo, dataset, source: None)

    module.republish("rcp-ndcg-vidore-shared", tmp_path)

    root = tmp_path / "rcp-ndcg-vidore-shared"
    assert (root / "computer_science-corpus" / "test-00000-of-00001.parquet").is_file()
    for name in names:
        assert not (root / f"{name}-corpus").exists()
        loaded = load_dataset(str(root), f"{name}-corpus", split="test")
        assert loaded["id"] == ["d1"]
        assert loaded[0]["text"] == "page"


def test_the_converter_refuses_a_task_definition_with_a_different_split(
    tmp_path: Path, module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "_load", a_loader(a_suite(split="standard")))
    monkeypatch.setattr(module, "_task_source", lambda repo: a_nanobeir_task_source(split="test"))
    stub_repo(monkeypatch, module, "NanoArguAnaRetrieval", "NanoFEVERRetrieval")
    monkeypatch.setattr(module, "_card", lambda repo, dataset, source: None)

    with pytest.raises(Exception, match="standard.*test|test.*standard") as caught:
        module.republish("rcp-ndcg-nanobeir-split", tmp_path)
    assert "NanoArguAnaRetrieval" in str(caught.value)
    assert not (tmp_path / "rcp-ndcg-nanobeir-split").exists()


def test_the_converter_refuses_a_data_subset_without_a_task_definition(
    tmp_path: Path, module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "_load", a_loader(a_suite()))
    monkeypatch.setattr(module, "_task_source", lambda repo: a_task_source({"NanoArguAnaRetrieval": "test"}))
    stub_repo(monkeypatch, module, "NanoArguAnaRetrieval", "NanoFEVERRetrieval")
    monkeypatch.setattr(module, "_card", lambda repo, dataset, source: None)

    with pytest.raises(Exception, match="NanoFEVERRetrieval") as caught:
        module.republish("rcp-ndcg-nanobeir-subset", tmp_path)
    assert "no task" in str(caught.value)
    assert not (tmp_path / "rcp-ndcg-nanobeir-subset").exists()


def test_the_converter_refuses_a_task_definition_subset_the_data_lacks(
    tmp_path: Path, module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "_load", a_loader(a_suite()))
    source = a_task_source({"NanoArguAnaRetrieval": "test", "NanoFEVERRetrieval": "test", "NanoGhostRetrieval": "test"})
    monkeypatch.setattr(module, "_task_source", lambda repo: source)
    stub_repo(monkeypatch, module, "NanoArguAnaRetrieval", "NanoFEVERRetrieval")
    monkeypatch.setattr(module, "_card", lambda repo, dataset, source: None)

    with pytest.raises(Exception, match="NanoGhostRetrieval") as caught:
        module.republish("rcp-ndcg-nanobeir-ghost", tmp_path)
    assert "does not have" in str(caught.value)
    assert not (tmp_path / "rcp-ndcg-nanobeir-ghost").exists()


def test_the_converter_checks_every_task_definition_of_a_subset(
    tmp_path: Path, module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ViDoRe v3 maps one subset to two tasks (the page-image one and its OCR view): a split mismatch on either
    is refused, not only on the first match."""
    from rcp_ndcg.data.dataset import Dataset

    part = a_subset(
        "computer_science__english",
        "q",
        [{"doc_id": "d1", "text": "page"}],
        [{"query_id": "q1", "doc_id": "d1", "grade": 1}],
    )
    suite = Dataset(name="vidore", revision="abc123", subsets=(part,))
    monkeypatch.setattr(module, "_load", a_loader(suite))
    monkeypatch.setattr(module, "_task_source", lambda repo: a_two_task_source(ocr_split="train"))
    stub_repo(monkeypatch, module, "computer_science__english")
    monkeypatch.setattr(module, "_card", lambda repo, dataset, source: None)

    with pytest.raises(Exception, match="OCR.*train|train.*OCR") as caught:
        module.republish("rcp-ndcg-vidore-ocr", tmp_path)
    assert "computer_science__english" in str(caught.value)


def test_the_card_fields_carry_the_published_prompt_and_the_task_split(
    tmp_path: Path, module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R7: the prompt travels in the task metadata mteb reads (`TaskMetadata.prompt`); mteb's dataset card
    template does not render it, so the test checks the fields the converter hands to the template."""
    monkeypatch.setattr(module, "_load", a_loader(a_suite(split="train")))
    source = a_nanobeir_task_source(split="train")
    monkeypatch.setattr(module, "_task_source", lambda repo: source)
    stub_repo(monkeypatch, module, "NanoArguAnaRetrieval", "NanoFEVERRetrieval")

    card = module._card("rcp-ndcg-nanobeir", a_suite(split="train"), source)

    assert card["prompt"] == {"query": "find documents that answer"}
    assert card["eval_splits"] == ["train"]
    assert card["dataset"]["path"] == "org/rcp-ndcg-nanobeir"

    module.republish("rcp-ndcg-nanobeir-card", tmp_path)
    readme = (tmp_path / "rcp-ndcg-nanobeir-card" / "README.md").read_text()
    assert "NanoArguAnaRetrievalRCPReranking" in readme  # mteb's own card template, from the task metadata


def test_a_second_run_validates_its_own_layout_not_the_first_run_s_cache(
    tmp_path: Path, module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`datasets` keys a local build by the directory basename, so a second republish into the same
    `out/<repo>` must not validate against the first run's cached build (the unique symlink view)."""
    from rcp_ndcg.data.dataset import Dataset

    monkeypatch.setattr(module, "_task_source", lambda repo: a_task_source({"NanoArguAnaRetrieval": "test"}))
    stub_repo(monkeypatch, module, "NanoArguAnaRetrieval")
    monkeypatch.setattr(module, "_card", lambda repo, dataset, source: None)

    def suite(text: str) -> Dataset:
        part = a_subset(
            "NanoArguAnaRetrieval",
            "q",
            [{"doc_id": "d1", "text": text}],
            [{"query_id": "q1", "doc_id": "d1", "grade": 1}],
        )
        return Dataset(name="nanobeir", revision="abc123", subsets=(part,))

    monkeypatch.setattr(module, "_load", a_loader(suite("first")))
    module.republish("rcp-ndcg-nanobeir-rerun", tmp_path)
    monkeypatch.setattr(module, "_load", a_loader(suite("second")))
    summary = module.republish("rcp-ndcg-nanobeir-rerun", tmp_path)

    assert summary["subsets"]["NanoArguAnaRetrieval"]["documents"] == 1


def test_the_converter_validates_media_bytes(tmp_path: Path, module: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """The media comparison is live: corrupting the written image bytes makes the validation fail."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image
    from rcp_ndcg_core.content import Content, ImagePart, TextPart

    from rcp_ndcg.data.dataset import Dataset
    from rcp_ndcg.data.media import store_media

    buffer = io.BytesIO()
    Image.new("RGB", (2, 3), "red").save(buffer, "PNG")
    ref = store_media(buffer.getvalue(), ".png", root=str(tmp_path / "media"))
    content = Content.from_parts([TextPart(text="page"), ImagePart(ref=ref)])
    part = a_subset(
        "NanoArguAnaRetrieval",
        "q",
        [{"doc_id": "d1", "text": "page", "content": content}],
        [{"query_id": "q1", "doc_id": "d1", "grade": 1}],
    )
    suite = Dataset(name="nanobeir", revision="abc123", subsets=(part,))
    monkeypatch.setattr(module, "_load", a_loader(suite))
    monkeypatch.setattr(module, "_task_source", lambda repo: a_task_source({"NanoArguAnaRetrieval": "test"}))
    stub_repo(monkeypatch, module, "NanoArguAnaRetrieval")
    monkeypatch.setattr(module, "_card", lambda repo, dataset, source: None)

    real = module.MtebWriter.write_dataset

    def corrupting(self: Any, dataset: Any, uri: str, **kwargs: Any) -> int:
        written = real(self, dataset, uri, **kwargs)
        path = Path(uri) / "NanoArguAnaRetrieval-corpus" / "test-00000-of-00001.parquet"
        table = pq.read_table(path)
        other = io.BytesIO()
        Image.new("RGB", (2, 3), "blue").save(other, "PNG")
        image_type = table.schema.field("image").type
        table = table.set_column(
            table.schema.get_field_index("image"),
            "image",
            pa.array([{"bytes": other.getvalue(), "path": None}], type=image_type),
        )
        pq.write_table(table, path)
        return written

    monkeypatch.setattr(module.MtebWriter, "write_dataset", corrupting)
    with pytest.raises(Exception, match="image") as caught:
        module.republish("rcp-ndcg-nanobeir-media", tmp_path)
    assert "NanoArguAnaRetrieval" in str(caught.value)


def test_a_failing_subset_is_reported_with_the_pair_named(
    tmp_path: Path, module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite = a_suite()
    broken = suite.model_copy(
        update={"subsets": suite.subsets[:1] + (suite.subsets[1].model_copy(update={"qrels": {"q1": {"d9": 1}}}),)}
    )
    monkeypatch.setattr(module, "_load", a_loader(broken))
    monkeypatch.setattr(module, "_task_source", lambda repo: a_nanobeir_task_source())
    stub_repo(monkeypatch, module, "NanoArguAnaRetrieval", "NanoFEVERRetrieval")
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

    monkeypatch.setattr(module, "_load", a_loader(a_suite()))
    monkeypatch.setattr(module, "_task_source", lambda repo: a_nanobeir_task_source())
    stub_repo(monkeypatch, module, "NanoArguAnaRetrieval", "NanoFEVERRetrieval")
    monkeypatch.setattr(module, "_card", lambda repo, dataset, source: None)

    module.republish("rcp-ndcg-nanobeir-excluded", tmp_path)

    loaded = RetrievalDatasetLoader(
        hf_repo=str(tmp_path / "rcp-ndcg-nanobeir-excluded"),
        revision="main",
        split="test",
        config="NanoFEVERRetrieval",
    ).load()
    assert loaded["top_ranked"] == {"q1": []}  # the excluded document left the pool
    assert loaded["relevant_docs"] == {"q1": {"d3": 1}}
