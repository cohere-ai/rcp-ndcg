"""The retrieval API: role configs by ``api``, and index/search/retrieve/rerank/fuse from a Dataset to Rankings.

Every model runs behind the offline fakes (``fake://``, the served wire) or an ``httpx.MockTransport`` wired
under the role clients (a hosted profile's public shape), so the tests exercise the real clients over the real
transport.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
import yaml
from pydantic import TypeAdapter, ValidationError

from rcp_ndcg.data import Rankings, load_dataset
from rcp_ndcg.errors import ConfigError, CredentialsError
from rcp_ndcg.retrieval import (
    BM25Config,
    CohereEmbedding,
    CohereReranker,
    DenseConfig,
    GeminiEmbedding,
    LateInteractionConfig,
    RerankerConfig,
    RetrieverConfig,
    ServedEmbedding,
    ServedPooling,
    ServedReranker,
    VoyageEmbedding,
    VoyageReranker,
    fuse,
    index,
    rerank,
    retrieve,
    search,
    validate_reranker,
    validate_retriever,
)
from rcp_ndcg.retrieval import _api as retrieval_api
from rcp_ndcg.retrieval.config import _OLD_SHAPE_HINT

DOCS = {
    "d1": "tortoises move slowly across the sand",
    "d2": "hares run fast in open fields",
    "d3": "the sand dunes of the desert",
}

_ENCODER = TypeAdapter(DenseConfig.model_fields["encoder"].annotation)
_RERANKER = TypeAdapter(RerankerConfig)


@pytest.fixture
def dataset(tmp_path: Path):
    root = tmp_path / "beir"
    (root / "qrels").mkdir(parents=True)
    (root / "corpus.jsonl").write_text("".join(json.dumps({"_id": d, "text": t}) + "\n" for d, t in DOCS.items()))
    (root / "queries.jsonl").write_text(
        json.dumps({"_id": "q1", "text": "slow tortoises"})
        + "\n"
        + json.dumps({"_id": "q2", "text": "fast hares"})
        + "\n"
    )
    (root / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td1\t1\nq2\td2\t1\n")
    return load_dataset(f"beir:{root}")


@pytest.fixture
def hosted_wire(monkeypatch: pytest.MonkeyPatch) -> list[httpx.Request]:
    """The role clients' transports send through one recording ``httpx.MockTransport``.

    The retrieval API builds its clients from configs alone, so the wire is patched under the clients (the
    endpoint they hand the transport already carries the profile's public URL). The handler answers the
    profiles' public shapes: Cohere ``/embed`` and ``/v2/rerank``, Voyage ``/embeddings`` and ``/v1/rerank``,
    Gemini ``batchEmbedContents``.
    """
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        body = json.loads(request.content)
        texts = _texts(body)
        if request.url.path.endswith("/rerank"):
            scores = [{"index": i, "relevance_score": 1.0 / (i + 1)} for i in range(len(texts))]
            if body.get("results") is not None or "cohere" in str(request.url):
                return httpx.Response(200, json={"results": scores})
            return httpx.Response(200, json={"data": scores})
        vectors = [[float(len(text) % 7 + 1), 1.0] for text in texts]
        if "batchEmbedContents" in str(request.url):
            return httpx.Response(200, json={"embeddings": [{"values": vector} for vector in vectors]})
        if request.url.path.endswith("/embed"):
            return httpx.Response(200, json={"embeddings": {"float": vectors}})
        return httpx.Response(
            200, json={"data": [{"index": i, "embedding": vector} for i, vector in enumerate(vectors)]}
        )

    from rcp_ndcg.inference import transport as transport_module

    real = transport_module.Transport

    def patched(endpoint: Any, *, httpx_transport: httpx.AsyncBaseTransport | None = None) -> Any:
        return real(endpoint, httpx_transport=httpx.MockTransport(handler))

    monkeypatch.setattr("rcp_ndcg.inference.clients.embed.Transport", patched)
    monkeypatch.setattr("rcp_ndcg.inference.clients.rerank.Transport", patched)
    monkeypatch.setenv("CO_API_KEY", "test-key")
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    return sent


def _texts(body: dict) -> list[str]:
    """The texts one hosted request carries, whatever shape it arrives in."""
    if isinstance(body.get("input"), list):
        return [str(text) for text in body["input"]]
    if isinstance(body.get("texts"), list):
        return [str(text) for text in body["texts"]]
    if isinstance(body.get("documents"), list):
        return [str(text) for text in body["documents"]]
    if isinstance(body.get("requests"), list):
        return [str(row["content"]["parts"][0]["text"]) for row in body["requests"]]
    return [str(body.get("query", ""))]


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------


def test_the_retriever_config_is_a_union_discriminated_on_kind() -> None:
    adapter = TypeAdapter(RetrieverConfig)
    assert adapter.validate_python({"kind": "bm25"}).kind == "bm25"
    assert (
        adapter.validate_python({"kind": "dense", "encoder": {"api": "cohere", "model": "m"}}).encoder.api == "cohere"
    )
    with pytest.raises(ValidationError, match="kind"):
        adapter.validate_python({"provider": "local", "model": "m"})


@pytest.mark.parametrize(
    ("encoder", "api_type"),
    [
        ({"api": "openai_embeddings", "model": "m", "base_url": "http://h:8000/v1"}, ServedEmbedding),
        ({"api": "cohere", "model": "m"}, CohereEmbedding),
        ({"api": "voyage", "model": "m"}, VoyageEmbedding),
        ({"api": "gemini", "model": "m"}, GeminiEmbedding),
    ],
)
def test_an_encoder_config_is_its_role_endpoint(encoder: dict, api_type: type) -> None:
    assert isinstance(_ENCODER.validate_python(encoder), api_type)


@pytest.mark.parametrize(
    ("reranker", "api_type"),
    [
        ({"api": "rerank", "model": "m", "base_url": "http://h:8000/v1"}, ServedReranker),
        ({"api": "cohere", "model": "rerank-v4.0-pro"}, CohereReranker),
        ({"api": "voyage", "model": "rerank-2.5", "batch_size": 50}, VoyageReranker),
    ],
)
def test_a_reranker_config_is_its_role_endpoint(reranker: dict, api_type: type) -> None:
    assert isinstance(_RERANKER.validate_python(reranker), api_type)


def test_an_encoder_takes_only_what_its_api_uses() -> None:
    with pytest.raises(ValidationError):
        _ENCODER.validate_python({"api": "openai_embeddings", "model": "m", "pooling": "token"})
    with pytest.raises(ValidationError):
        _ENCODER.validate_python({"api": "cohere", "model": "m", "engine": "hf"})


def test_an_old_config_shape_is_refused_with_the_new_shape_hint() -> None:
    for data in (
        {"provider": "local", "model": "m", "pooling": "last"},
        {"provider": "cohere", "model": "m"},
        {"provider": "openai_compatible", "model": "m", "base_url": "http://h:8000"},
        {"engine": "vllm", "model": "m", "pooling": "token"},
    ):
        with pytest.raises(ConfigError, match="api, not by provider|api, not by engine") as caught:
            validate_retriever({"kind": "dense", "encoder": data})
        assert "openai_embeddings" in (caught.value.hint or "")
    with pytest.raises(ConfigError, match="api, not by provider"):
        validate_reranker({"provider": "local", "model": "Qwen/Qwen3-Reranker-8B"})
    assert "api: rerank" in _OLD_SHAPE_HINT


def test_the_old_shapes_load_through_the_cli_as_a_config_error(tmp_path: Path) -> None:
    from click.testing import CliRunner

    from rcp_ndcg.cli.retrieval import retrieval_group

    path = tmp_path / "old_shape.yaml"
    path.write_text(yaml.safe_dump({"kind": "dense", "encoder": {"provider": "local", "model": "m"}}))
    result = CliRunner().invoke(
        retrieval_group,
        [
            "index",
            "--dataset",
            "beir:/nope",
            "--retriever",
            str(path),
            "--out",
            str(tmp_path / "out.parquet"),
            "--json",
        ],
    )

    assert result.exit_code == 3, result.output
    error = json.loads(result.stdout)["error"]
    assert error["code"] == "CONFIG"
    assert "api: openai_embeddings" in error["hint"]


def test_single_and_multi_vector_retrievers_are_different_kinds() -> None:
    adapter = TypeAdapter(RetrieverConfig)
    pooling = {"api": "vllm_pooling", "model": "colqwen", "base_url": "http://h:8000/v1"}
    served = {"api": "openai_embeddings", "model": "colqwen", "base_url": "http://h:8000/v1"}

    with pytest.raises(ValidationError, match="vllm_pooling"):
        adapter.validate_python({"kind": "dense", "encoder": pooling})
    with pytest.raises(ValidationError, match="vllm_pooling"):
        adapter.validate_python({"kind": "late_interaction", "encoder": served})
    assert adapter.validate_python({"kind": "late_interaction", "encoder": pooling}).kind == "late_interaction"


def test_where_an_encoder_runs_is_not_part_of_what_an_index_is() -> None:
    """The endpoint's timeouts, retries and batch size are runtime; the model, revision and prompts are content."""
    first = _ENCODER.validate_python({"api": "cohere", "model": "embed-v4.0", "batch_size": 8})
    second = _ENCODER.validate_python({"api": "cohere", "model": "embed-v4.0", "timeout_s": 5, "max_retries": 0})
    other_model = _ENCODER.validate_python({"api": "cohere", "model": "embed-v3.0"})

    identity = retrieval_api._identity
    assert identity(DenseConfig(encoder=first), ["d"], []) == identity(DenseConfig(encoder=second), ["d"], [])
    assert identity(DenseConfig(encoder=first), ["d"], []) != identity(DenseConfig(encoder=other_model), ["d"], [])


def test_a_listwise_reranker_takes_no_batch_size_and_a_hosted_one_no_engine_fields() -> None:
    with pytest.raises(ValidationError, match="listwise"):
        _RERANKER.validate_python({"api": "rerank", "model": "jina-reranker-v3", "listwise": True, "batch_size": 4})
    with pytest.raises(ValidationError, match="hosted"):
        _RERANKER.validate_python({"api": "voyage", "model": "rerank-2.5", "use_activation": True})
    with pytest.raises(ValidationError, match="hosted"):
        _RERANKER.validate_python({"api": "cohere", "model": "rerank-v4.0-pro", "instruction": "field"})


def test_a_reranker_refuses_max_tokens_where_its_client_is_built(dataset) -> None:
    """A budget is never silently ignored: the clients refuse it until the text-budget mechanism wires the cut."""
    from rcp_ndcg.inference.clients import RerankClient

    config = validate_reranker({"api": "rerank", "model": "m", "base_url": "http://h:8000/v1", "max_tokens": 8192})
    with pytest.raises(ConfigError, match="max_tokens"):
        RerankClient(config)


# ---------------------------------------------------------------------------
# index, search, retrieve (the offline fakes: the served wire)
# ---------------------------------------------------------------------------


def test_bm25_indexes_and_searches_a_dataset(dataset, tmp_path: Path) -> None:
    built = index(dataset, BM25Config(), out=tmp_path / "idx")

    rankings = search(built, dataset, depth=2)

    assert rankings.systems == ["bm25"]
    for query_id, expected in (("q1", "d1"), ("q2", "d2")):
        scores = rankings.for_query(query_id)
        assert max(scores, key=scores.__getitem__) == expected
        assert len(scores) <= 2


def test_dense_retrieval_ranks_by_inner_product_and_reuses_an_index(dataset, tmp_path: Path) -> None:
    """A fake served encoder, the real client and transport; the same identity reuses the index."""
    config = DenseConfig(encoder=ServedEmbedding(base_url="fake://seed/7?dim=8", model="stub"))

    first = retrieve(dataset, config, depth=3, out=tmp_path / "idx")
    stamp = (tmp_path / "idx" / "index.json").stat().st_mtime_ns
    again = retrieve(dataset, config, depth=3, out=tmp_path / "idx")

    vectors = np.load(tmp_path / "idx" / "vectors.npy")
    assert vectors.shape == (3, 8)
    assert again.for_query("q1") == first.for_query("q1")
    assert (tmp_path / "idx" / "index.json").stat().st_mtime_ns == stamp, "the same identity reuses the index"
    assert first.systems == ["stub"]
    assert all(np.isclose(np.linalg.norm(row), 1.0) for row in vectors), "the client normalises"


def test_late_interaction_indexes_and_searches_a_ragged_index(dataset, tmp_path: Path) -> None:
    """A pooling endpoint builds the ragged index (vectors + offsets) and searches it by MaxSim."""
    config = LateInteractionConfig(encoder=ServedPooling(base_url="fake://seed/3?dim=4", model="colqwen", dim=4))

    built = index(dataset, config, out=tmp_path / "idx")
    rankings = search(built, dataset, depth=3)

    vectors = np.load(tmp_path / "idx" / "vectors.npy")
    offsets = np.load(tmp_path / "idx" / "offsets.npy")
    assert vectors.dtype == np.float16, "the transfer dtype reaches the index"
    assert len(offsets) == 4 and offsets[0] == 0 and offsets[-1] == len(vectors)
    assert set(rankings.for_query("q1")) <= set(DOCS)


def test_a_served_encoder_without_a_url_is_refused_where_its_client_is_built(dataset, tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="no base_url") as caught:
        retrieve(dataset, DenseConfig(encoder=ServedEmbedding(model="m")), out=tmp_path / "idx")
    assert "serve.encoder" in (caught.value.hint or "")
    with pytest.raises(ConfigError, match="no base_url"):
        retrieve(dataset, LateInteractionConfig(encoder=ServedPooling(model="m")), out=tmp_path / "idx")


# ---------------------------------------------------------------------------
# Hosted profiles through the role clients over a mock wire
# ---------------------------------------------------------------------------


def test_a_hosted_encoder_embeds_through_its_public_profile(dataset, tmp_path: Path, hosted_wire) -> None:
    config = DenseConfig(encoder=CohereEmbedding(model="embed-v4.0", batch_size=2))

    built = index(dataset, config, out=tmp_path / "idx")
    rankings = search(built, dataset, depth=2)

    assert built.num_documents == 3
    assert rankings.systems == ["embed-v4.0"]
    cohere_calls = [request for request in hosted_wire if request.url.host == "api.cohere.com"]
    assert cohere_calls, "the profile's public URL was used"
    body = json.loads(cohere_calls[0].content)
    assert body["input_type"] == "search_document" and len(body["texts"]) == 2, "batch_size splits the corpus"
    vectors = np.load(tmp_path / "idx" / "vectors.npy")
    assert all(np.isclose(np.linalg.norm(row), 1.0) for row in vectors), "the client normalises"


def test_a_hosted_encoder_key_is_read_from_its_profile_variables(
    dataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CO_API_KEY", raising=False)
    monkeypatch.delenv("COHERE_API_KEY", raising=False)

    with pytest.raises(CredentialsError, match="CO_API_KEY"):
        index(dataset, DenseConfig(encoder=CohereEmbedding(model="embed-v4.0")), out=tmp_path / "idx")


# ---------------------------------------------------------------------------
# rerank and fuse
# ---------------------------------------------------------------------------


def test_rerank_rescores_the_top_candidates_through_the_fake_endpoint(dataset, tmp_path: Path) -> None:
    """Scores align to the candidates, and the checkpoint records each scored query."""
    config = ServedReranker(base_url="fake://seed/1", model="stub-reranker", instruction="fold")
    candidates = Rankings.from_scores({"q1": {"d1": 3.0, "d2": 2.0, "d3": 1.0}}, system="bm25")

    rescored = rerank(dataset, candidates, config, depth=2, out=tmp_path / "rerank")

    assert rescored.systems == ["stub-reranker"]
    top = rescored.for_query("q1")
    assert set(top) == {"d1", "d2"}
    assert (tmp_path / "rerank" / "rank000.jsonl").exists(), "the checkpoint records each scored query"


def test_the_rerank_client_receives_the_raw_query_and_folds_it_once(dataset) -> None:
    """The example's query and instruction are folded by the client, never by the caller: one fold, not two."""
    from rcp_ndcg_core._records import RankingExample

    from rcp_ndcg.inference.clients import RerankClient
    from rcp_ndcg.inference.types import Reply

    sent: list[dict] = []

    class RecordingSender:
        """A sender that records the bodies it is given and scores by position."""

        async def send(self, calls: list[Any]) -> list[Reply]:
            replies = []
            for call in calls:
                sent.append(call.json)
                documents = call.json["documents"]
                rows = [{"index": i, "relevance_score": float(i)} for i in range(len(documents))]
                replies.append(Reply(200, {"results": rows}, {}))
            return replies

    config = ServedReranker(base_url="http://rerank.test/v1", model="stub-reranker")
    client = RerankClient(config, sender=RecordingSender())
    example = RankingExample(
        query_id="q1", query="base query", instruction="Find relevant passages", doc_ids=["d1"], docs=["a document"]
    )
    try:
        client.rerank_many([example])
    finally:
        client.close()

    assert sent[0]["query"] == "Task: Find relevant passages\nQuery: base query", "the fold happens exactly once"
    assert sent[0]["documents"] == ["a document"]


def test_a_hosted_reranker_runs_through_its_public_profile(dataset, tmp_path: Path, hosted_wire) -> None:
    config = CohereReranker(model="rerank-v4.0-fast", api_key_env="CO_API_KEY")
    candidates = Rankings.from_scores({"q1": {"d1": 3.0, "d2": 2.0, "d3": 1.0}}, system="bm25")

    rescored = rerank(dataset, candidates, config, depth=3)

    assert rescored.systems == ["rerank-v4.0-fast"]
    cohere_calls = [request for request in hosted_wire if request.url.host == "api.cohere.com"]
    assert cohere_calls and json.loads(cohere_calls[0].content)["top_n"] == 3


def test_fuse_sums_reciprocal_ranks() -> None:
    one = Rankings.from_scores({"q1": {"d1": 1.0, "d2": 2.0}}, system="a")
    two = Rankings.from_scores({"q1": {"d2": 1.0, "d1": 2.0}}, system="b")

    fused = fuse([one, two], depth=2)

    scores = fused.for_query("q1", system="rrf")
    assert scores["d1"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["d2"] == pytest.approx(1 / 62 + 1 / 61)


def test_fuse_keeps_the_datasets_of_a_suite_apart() -> None:
    one = Rankings.from_records(
        [
            {"system": "a", "dataset": "one", "query_id": "q1", "doc_id": "d1", "score": 1.0},
            {"system": "a", "dataset": "two", "query_id": "q1", "doc_id": "d2", "score": 1.0},
        ]
    )
    two = Rankings.from_records(
        [
            {"system": "b", "dataset": "one", "query_id": "q1", "doc_id": "d2", "score": 1.0},
            {"system": "b", "dataset": "two", "query_id": "q1", "doc_id": "d1", "score": 1.0},
        ]
    )

    fused = fuse([one, two])

    assert sorted(fused.datasets) == ["one", "two"]
