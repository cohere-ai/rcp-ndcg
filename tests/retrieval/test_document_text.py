"""What a document reads as, at the retrieval steps: MTEB's title join, or the title separately.

Decision 27: a model reads ``(title + " " + body).strip()`` (the body alone without a title), the one
join, applied where the model's text is formatted -- the corpus materialisation of index, search and
rerank. A model or recipe that takes the title separately declares ``title: separate`` on its endpoint
config. Decision 33: the task instruction (``Dataset.task_instruction``) is placed by the config's mode
and the per-query instruction (``Query.instruction``) is appended. The sparse path reads neither: mteb's
own BM25 is not a served model, joins a corpus row with a newline and takes no task instruction, and the
sparse path follows it byte for byte.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from rcp_ndcg.data import Dataset, Rankings
from rcp_ndcg.errors import DataError, IdentityError
from rcp_ndcg.retrieval import BM25Config, DenseConfig, ServedReranker, index, rerank, search
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
        **extra,
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

    def patched(endpoint: Any, *, auth: Any = None, httpx_transport: Any = None) -> Any:
        return real(endpoint, auth=auth, httpx_transport=httpx.MockTransport(handler))

    monkeypatch.setattr("rcp_ndcg.inference.clients._base.Transport", patched)
    monkeypatch.setenv("CO_API_KEY", "test-key")
    return sent


def _retriever(**encoder: Any) -> DenseConfig:
    """The dense retriever: ``instruction: fold`` is declared (an undeclared policy refuses a task
    instruction), and the rest are the test's overrides."""
    settings: dict[str, Any] = {"api": "cohere", "model": "m", "instruction": "fold"}
    settings.update(encoder)
    return DenseConfig.model_validate({"kind": "dense", "encoder": settings})


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


def test_rerank_refuses_a_document_side_task_instruction(wire: list[httpx.Request], tmp_path: Any) -> None:
    """A reranker reads a (query, document) pair: its template's instruction slot is the query's, so a
    document-side task instruction has no place on its wire -- refused, never dropped silently."""
    from rcp_ndcg.errors import ConfigError

    dataset = _corpus().model_copy(
        update={"task_instruction": {"query": "a query instruction", "document": "a passage instruction"}}
    )
    rankings = Rankings.from_scores({"q1": {"d1": 1.0}}, system="bm25")
    reranker = ServedReranker(
        base_url="http://rerank.test/v1", model="stub-reranker", use_activation=False, **_SERVED_BUDGET
    )

    with pytest.raises(ConfigError, match="document-side task instruction"):
        rerank(dataset, rankings, reranker, out=tmp_path / "rerank")


def test_the_index_identity_covers_the_document_side_instruction(wire: list[httpx.Request], tmp_path: Any) -> None:
    """The document-side task instruction changes the indexed text: two builds that differ only in it never
    share an index identity, and an index of the other instruction is not silently reused. (A document-side
    instruction reaches the encoder through the template's ``instruction`` span: the generic default frames
    the query side only.)"""
    from rcp_ndcg.data.templates import Segment, TemplateSpec

    template = TemplateSpec(
        query=(Segment(fixed="Query: "), Segment(content="query")),
        document=(
            Segment(fixed="Instruct: "),
            Segment(content="instruction"),
            Segment(fixed="\nDocument: "),
            Segment(content="document"),
        ),
    )
    first = _corpus()
    second = first.model_copy(
        update={"task_instruction": {"query": first.task_instruction, "document": "a passage instruction"}}
    )
    retriever = DenseConfig.model_validate(
        {
            "kind": "dense",
            "encoder": {
                "api": "openai_embeddings",
                "model": "m",
                "base_url": "http://engine:8000/v1",
                "instruction": "fold",
                "template": template,
                **_SERVED_BUDGET,
            },
        }
    )

    one = index(first, retriever, out=tmp_path / "one")
    two = index(second, retriever, out=tmp_path / "two")

    assert one.identity != two.identity, "the identity must cover the text the instruction changes"
    with pytest.raises(IdentityError, match="another corpus or retriever"):
        search(one, second)


def test_a_rebuild_that_pooled_to_single_vectors_is_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """A late-interaction rebuild whose encoder answers one vector per document is refused: a pooled answer is
    not a multi-vector index, and ``search`` would slice the new vectors by the previous build's offsets. The
    refusal comes before anything is written, so the previous index stays intact (stronger than clearing the
    stale offsets after the fact)."""
    import numpy as np

    from rcp_ndcg.inference.types import Embeddings
    from rcp_ndcg.retrieval import LateInteractionConfig
    from rcp_ndcg.retrieval import _api as retrieval_api

    retriever = LateInteractionConfig.model_validate(
        {
            "kind": "late_interaction",
            "encoder": {
                "api": "vllm_pooling",
                "model": "m",
                "base_url": "http://engine:8000/v1",
                "dim": 2,
                **_SERVED_BUDGET,
            },
        }
    )
    calls = {"n": 0}

    def fake(config: Any, contents: Any, role: Any, *, instruction: Any = None) -> Embeddings:
        calls["n"] += 1
        if calls["n"] == 1:
            return Embeddings.ragged([np.ones((3, 2), dtype=np.float16)] * len(contents), dtype=np.float16)
        return Embeddings.single(np.ones((len(contents), 2), dtype=np.float32))

    monkeypatch.setattr(retrieval_api, "_encode", fake)
    out = tmp_path / "index"

    index(_corpus(), retriever, out=out)
    assert (out / "offsets.npy").exists()

    with pytest.raises(DataError, match="one vector per document"):
        index(_corpus(), retriever, out=out)

    assert (out / "offsets.npy").exists(), "the refused rebuild left the previous index intact"
    assert (out / "vectors.npy").exists()


def test_a_rebuild_of_another_kind_clears_the_old_kind(wire: list[httpx.Request], tmp_path: Any) -> None:
    """An index directory holds the current build's files only: a sparse rebuild drops the vector files and a
    vector rebuild drops ``bm25s/`` (the index record gates the branch, so nothing stale is ever read -- this
    is disk hygiene: the directory must not carry a whole corpus's vectors beside its ``bm25s/`` index)."""
    dataset = _corpus()
    out = tmp_path / "index"

    index(dataset, BM25Config(), out=out)
    assert (out / "bm25s").is_dir()

    index(dataset, _retriever(), out=out)
    assert not (out / "bm25s").exists(), "a vector rebuild must drop the sparse index"

    index(dataset, BM25Config(), out=out)
    assert not (out / "vectors.npy").exists(), "a sparse rebuild must drop the vector index"
    assert not (out / "offsets.npy").exists()


def test_the_sparse_corpus_reads_a_content_carrying_rows_body(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """A row whose ``content`` is set is authoritative (``DocumentRow.as_content``): the sparse path indexes
    the part's text, not the raw ``text`` field (which a media row leaves empty) -- the body of an OCR row or
    a caption must not vanish from the BM25 index."""
    from rcp_ndcg_core.content import Content, TextPart

    from rcp_ndcg.retrieval import sparse

    dataset = Dataset.from_records(
        name="content-rows",
        corpus=[
            {"doc_id": "d1", "title": "T", "content": Content.from_parts([TextPart(text="the real body")])},
            {"doc_id": "d2", "text": "plain"},
        ],
        queries=[{"query_id": "q1", "text": "find docs"}],
        qrels=[{"query_id": "q1", "doc_id": "d1", "grade": 1.0}],
    )
    seen: list[list[str]] = []
    real = sparse.build_bm25_index

    def recording(corpus: Any, dataset_dir: Any, *, stemmer: Any) -> None:
        seen.append([item if isinstance(item, str) else item.text for item in corpus])
        real(corpus, dataset_dir, stemmer=stemmer)

    monkeypatch.setattr("rcp_ndcg.retrieval.sparse.build_bm25_index", recording)

    index(dataset, BM25Config(), out=tmp_path / "index")

    assert seen == [["T\nthe real body", "\nplain"]]


def test_bm25_indexes_mtebs_own_corpus_join(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """The sparse path's declared join is mteb's own BM25 one, byte for byte: ``title + "\\n" + body``, both
    as given -- not the retrieval dataloader's ``(title + " " + body).strip()`` the dense and rerank paths
    read (mteb's BM25 is not a served model and reads no dataloader)."""
    from rcp_ndcg.retrieval import sparse

    seen: list[list[str]] = []
    real = sparse.build_bm25_index

    def recording(corpus: Any, dataset_dir: Any, *, stemmer: Any) -> None:
        seen.append([item if isinstance(item, str) else item.text for item in corpus])
        real(corpus, dataset_dir, stemmer=stemmer)

    monkeypatch.setattr("rcp_ndcg.retrieval.sparse.build_bm25_index", recording)

    index(_corpus(), BM25Config(), out=tmp_path / "index")

    assert seen == [["Tortoises\na tortoise is a reptile", "  \n  a haiku about ponds  "]]


def test_bm25_sends_no_task_instruction(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """mteb's BM25 takes no task instruction (its ``search`` only appends the per-query one): the sparse
    path sends the per-query append alone, never the ``Task:`` frame."""
    from rcp_ndcg.retrieval import sparse

    seen: list[list[str]] = []
    real = sparse.search_bm25

    def recording(dataset_dir: Any, queries: Any, *, k: Any) -> Any:
        seen.append(list(queries))
        return real(dataset_dir, queries, k=k)

    monkeypatch.setattr("rcp_ndcg.retrieval.sparse.search_bm25", recording)
    dataset = _corpus()
    built = index(dataset, BM25Config(), out=tmp_path / "index")

    search(built, dataset)

    assert seen == [["find docs about turtles"]]
