"""What a document reads as, at the retrieval steps: MTEB's title join, or the title separately.

Decision 27: a model reads ``(title + " " + body).strip()`` (the body alone without a title), the one
join, applied where the model's text is formatted -- the corpus materialisation of index, search and
rerank. A model or recipe that takes the title separately declares ``title: separate`` on its endpoint
config. Decision 33: the task instruction (``Dataset.task_instruction``) is placed by the config's mode
and the per-query instruction (``Query.instruction``) is appended.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from rcp_ndcg.data import Dataset, Rankings
from rcp_ndcg.retrieval import DenseConfig, ServedReranker, index, rerank, search
from tests.conftest import SESSION_TOKENIZER

_SERVED_BUDGET: dict[str, Any] = {"tokenizer": str(SESSION_TOKENIZER), "max_tokens": 8192}


def _corpus(**extra: Any) -> Dataset:
    return Dataset.from_records(
        name="sample",
        corpus=[
            {"doc_id": "d1", "title": "Tortoises", "text": "a tortoise is a reptile"},
            {"doc_id": "d2", "title": "  ", "text": "  a haiku about ponds  "},
        ],
        queries=[{"query_id": "q1", "text": "find docs", "instruction": "about turtles"}],
        qrels=[{"query_id": "q1", "doc_id": "d1", "grade": 1.0}],
        candidates={"q1": ["d1", "d2"]},
        task_instruction="Given a claim, find documents that refute the claim",
    )


@pytest.fixture
def wire(monkeypatch: pytest.MonkeyPatch) -> list[httpx.Request]:
    """The hosted wire, patched under the clients: records every request and answers the shapes used here."""
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        body = json.loads(request.content)
        texts = [str(text) for text in body.get("input", body.get("texts", body.get("documents", [])))]
        if request.url.path.endswith("/rerank"):
            return httpx.Response(
                200, json={"results": [{"index": i, "relevance_score": 1.0} for i in range(len(texts))]}
            )
        vectors = [[float(len(text) % 7 + 1), 1.0] for text in texts]
        if request.url.path.endswith("/embed"):
            return httpx.Response(200, json={"embeddings": {"float": vectors}})
        return httpx.Response(
            200, json={"data": [{"index": i, "embedding": vector} for i, vector in enumerate(vectors)]}
        )

    from rcp_ndcg.inference import transport as transport_module

    real = transport_module.Transport

    def patched(endpoint: Any, *, auth: Any = None, httpx_transport: httpx.Any = None) -> Any:
        return real(endpoint, auth=auth, httpx_transport=httpx.MockTransport(handler))

    monkeypatch.setattr("rcp_ndcg.inference.clients._base.Transport", patched)
    monkeypatch.setenv("CO_API_KEY", "test-key")
    return sent


def _retriever(**title: Any) -> DenseConfig:
    return DenseConfig.model_validate({"kind": "dense", "encoder": {"api": "cohere", "model": "m", **title}})


def _sent_documents(sent: list[httpx.Request]) -> list[str]:
    """The texts every recorded request carried (the embed routes' ``input``/``texts`` shapes)."""
    out: list[str] = []
    for request in sent:
        body = json.loads(request.content)
        out.extend(str(text) for text in body.get("input", body.get("texts", [])))
    return out


def test_index_joins_the_title_like_mteb(wire: list[httpx.Request], tmp_path: Any) -> None:
    index(_corpus(), _retriever(), out=tmp_path / "index")

    assert _sent_documents(wire) == ["Tortoises a tortoise is a reptile", "a haiku about ponds"]


def test_index_can_take_the_title_separately(wire: list[httpx.Request], tmp_path: Any) -> None:
    """A declared ``title: separate`` sends the title as its own part: the text route joins the parts."""
    index(_corpus(), _retriever(title="separate"), out=tmp_path / "index")

    assert _sent_documents(wire) == ["Tortoises\na tortoise is a reptile", "  a haiku about ponds  "]


def test_search_folds_the_task_instruction_and_appends_the_per_query_one(
    wire: list[httpx.Request], tmp_path: Any
) -> None:
    dataset = _corpus()
    built = index(dataset, _retriever(), out=tmp_path / "index")
    wire.clear()

    search(built, dataset)

    assert [str(text) for text in json.loads(wire[0].content).get("texts", [])] == [
        "Task: Given a claim, find documents that refute the claim\nQuery: find docs about turtles"
    ]


def test_rerank_joins_the_title_and_places_both_instructions(wire: list[httpx.Request], tmp_path: Any) -> None:
    dataset = _corpus()
    rankings = Rankings.from_scores({"q1": {"d1": 2.0, "d2": 1.0}}, system="bm25")
    reranker = ServedReranker(
        base_url="http://rerank.test/v1", model="stub-reranker", use_activation=False, **_SERVED_BUDGET
    )
    wire.clear()

    rerank(dataset, rankings, reranker, out=tmp_path / "rerank")

    body = json.loads(wire[0].content)
    assert body["query"] == "Task: Given a claim, find documents that refute the claim\nQuery: find docs about turtles"
    assert body["documents"] == ["Tortoises a tortoise is a reptile", "a haiku about ponds"]
