"""``rcp-ndcg retrieval``: a dataset URI in, rankings files out, over the retrieval API."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from rcp_ndcg.cli.retrieval import retrieval_group as retrieval
from rcp_ndcg.data import Rankings, load_rankings


def _invoke(*args: str) -> dict:
    result = CliRunner().invoke(retrieval, [*args, "--json"])
    return {"exit_code": result.exit_code, **json.loads(result.stdout)}


@pytest.fixture
def dataset(tmp_path: Path) -> str:
    rows = [
        {"query_id": "q1", "query": "tortoise shell", "doc_ids": ["d1", "d2", "d3"],
         "docs": ["the tortoise shell", "a hare", "a tortoise"], "qrels": {"d1": 1}},
        {"query_id": "q2", "query": "hare speed", "doc_ids": ["d1", "d2", "d3"],
         "docs": ["the tortoise shell", "a hare", "a tortoise"], "qrels": {"d2": 1}},
    ]  # fmt: skip
    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return f"jsonl:{path}"


@pytest.fixture
def bm25(tmp_path: Path) -> str:
    path = tmp_path / "bm25.yaml"
    path.write_text(yaml.safe_dump({"kind": "bm25"}), encoding="utf-8")
    return str(path)


def test_search_retrieves_into_a_rankings_file(dataset: str, bm25: str, tmp_path: Path) -> None:
    out = tmp_path / "bm25.parquet"

    document = _invoke("search", "--dataset", dataset, "--retriever", bm25, "--out", str(out), "--depth", "2")

    assert document["exit_code"] == 0, document
    assert document["data"] == {
        "schema": "rcp-ndcg.rankings-file.v1",
        "out": str(out),
        "systems": ["bm25"],
        "queries": 2,
    }
    top = load_rankings(out).for_query("q2")
    assert max(top, key=top.get) == "d2"


def test_index_then_search_with_the_index(dataset: str, bm25: str, tmp_path: Path) -> None:
    built = _invoke("index", "--dataset", dataset, "--retriever", bm25, "--out", str(tmp_path / "index"))
    searched = _invoke(
        "search", "--dataset", dataset, "--index", built["data"]["index"], "--out", str(tmp_path / "r.jsonl")
    )

    assert built["data"]["num_documents"] == 3
    assert searched["exit_code"] == 0, searched
    assert load_rankings(tmp_path / "r.jsonl").systems == ["bm25"]


def test_search_needs_exactly_one_of_retriever_and_index(dataset: str, tmp_path: Path) -> None:
    document = _invoke("search", "--dataset", dataset, "--out", str(tmp_path / "r.parquet"))

    assert document["exit_code"] == 2
    assert document["error"]["code"] == "USAGE"


def test_a_malformed_retriever_is_a_config_error(dataset: str, tmp_path: Path) -> None:
    path = tmp_path / "old.yaml"
    path.write_text(yaml.safe_dump({"type": "bm25", "name": "bm25"}), encoding="utf-8")

    document = _invoke("search", "--dataset", dataset, "--retriever", str(path), "--out", str(tmp_path / "r.parquet"))

    assert document["exit_code"] == 3
    assert "kind" in document["error"]["hint"]


def test_rerank_rescores_the_top_candidates(dataset: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class _Served:
        def __init__(self, *_: object, **__: object) -> None: ...

        def rerank(self, query: object, documents: list[object], **_: object) -> list[float]:
            return [float(len(str(document))) for document in documents]

    monkeypatch.setattr("rcp_ndcg.retrieval.vllm_http.VllmPoolingClient", _Served)
    first = tmp_path / "first.parquet"
    Rankings.from_orders({"q1": ["d2", "d3", "d1"]}, system="first").save(first)
    reranker = tmp_path / "reranker.yaml"
    reranker.write_text(yaml.safe_dump({"provider": "openai_compatible", "model": "stub"}), encoding="utf-8")

    document = _invoke(
        "rerank", "--dataset", dataset, "--rankings", str(first), "--reranker", str(reranker),
        "--set", "base_url=http://stub:8000", "--depth", "2", "--out", str(tmp_path / "reranked.parquet"),
    )  # fmt: skip

    assert document["exit_code"] == 0, document
    scores = load_rankings(tmp_path / "reranked.parquet").for_query("q1")
    assert sorted(scores) == ["d2", "d3"]
    assert scores["d3"] > scores["d2"]


def test_fuse_combines_every_system(tmp_path: Path) -> None:
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    Rankings.from_orders({"q1": ["d1", "d2", "d3"]}, system="a").save(a)
    Rankings.from_orders({"q1": ["d3", "d2", "d1"]}, system="b").save(b)

    document = _invoke("fuse", "--rankings", str(a), "--rankings", str(b), "--out", str(tmp_path / "f.parquet"))

    assert document["data"]["systems"] == ["rrf"]
    assert set(load_rankings(tmp_path / "f.parquet").for_query("q1")) == {"d1", "d2", "d3"}


def test_fuse_needs_two_inputs(tmp_path: Path) -> None:
    a = tmp_path / "a.jsonl"
    Rankings.from_orders({"q1": ["d1"]}).save(a)

    document = _invoke("fuse", "--rankings", str(a), "--out", str(tmp_path / "f.parquet"))

    assert document["exit_code"] == 2
    assert "at least 2" in document["error"]["message"]


def test_set_is_refused_with_an_index_whose_retriever_is_already_built(dataset: str, tmp_path: Path) -> None:
    document = _invoke(
        "search", "--dataset", dataset, "--index", str(tmp_path / "index"), "--set", "implementation=bm25s",
        "--out", str(tmp_path / "out.jsonl"),
    )  # fmt: skip
    assert document["exit_code"] == 2 and document["error"]["code"] == "USAGE", document
    assert "--set" in document["error"]["message"]
