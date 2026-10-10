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

from rcp_ndcg.data import Rankings
from rcp_ndcg.errors import ConfigError, CredentialsError, DataError
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
    load_index,
    rerank,
    retrieve,
    search,
    validate_reranker,
    validate_retriever,
)
from rcp_ndcg.retrieval import _api as retrieval_api
from rcp_ndcg.retrieval.config import _OLD_SHAPE_HINT
from tests.conftest import SESSION_TOKENIZER
from tests.retrieval.conftest import DOCS

_SERVED_BUDGET: dict[str, Any] = {"tokenizer": str(SESSION_TOKENIZER), "max_tokens": 8192}
_SERVED_RERANK_BUDGET: dict[str, Any] = {**_SERVED_BUDGET, "use_activation": False}

_ENCODER = TypeAdapter(DenseConfig.model_fields["encoder"].annotation)
_RERANKER = TypeAdapter(RerankerConfig)


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

    def patched(
        endpoint: Any,
        *,
        auth: Any = None,
        httpx_transport: httpx.AsyncBaseTransport | None = None,
    ) -> Any:
        return real(endpoint, auth=auth, httpx_transport=httpx.MockTransport(handler))

    # the role clients read Transport from the client base (which builds it for a config)
    monkeypatch.setattr("rcp_ndcg.inference.clients._base.Transport", patched)
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
        (
            {"api": "openai_embeddings", "model": "m", "base_url": "http://h:8000/v1", **_SERVED_BUDGET},
            ServedEmbedding,
        ),
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
        (
            {"api": "rerank", "model": "m", "base_url": "http://h:8000/v1", **_SERVED_RERANK_BUDGET},
            ServedReranker,
        ),
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
    pooling = {
        "api": "vllm_pooling",
        "model": "colqwen",
        "base_url": "http://h:8000/v1",
        **_SERVED_BUDGET,
    }
    served = {
        "api": "openai_embeddings",
        "model": "colqwen",
        "base_url": "http://h:8000/v1",
        **_SERVED_BUDGET,
    }

    # A cross-role api is refused where the config is read, with the registry's own message.
    with pytest.raises(ConfigError, match="unknown embed adapter 'vllm_pooling'") as caught:
        adapter.validate_python({"kind": "dense", "encoder": pooling})
    assert "multi_vector" in (caught.value.hint or "")
    with pytest.raises(ConfigError, match="unknown multi_vector adapter 'openai_embeddings'") as caught:
        adapter.validate_python({"kind": "late_interaction", "encoder": served})
    assert "embed" in (caught.value.hint or "")
    assert adapter.validate_python({"kind": "late_interaction", "encoder": pooling}).kind == "late_interaction"


def test_where_an_encoder_runs_is_not_part_of_what_an_index_is() -> None:
    """The endpoint's timeouts and retries are runtime; the model, revision, prompts and request packing
    are content (a bf16 batch's numbers can depend on its composition)."""
    first = _ENCODER.validate_python({"api": "cohere", "model": "embed-v4.0", "batch_size": 8})
    second = _ENCODER.validate_python(
        {"api": "cohere", "model": "embed-v4.0", "batch_size": 8, "timeout_s": 5, "max_retries": 0}
    )
    other_model = _ENCODER.validate_python({"api": "cohere", "model": "embed-v3.0", "batch_size": 8})
    packed = _ENCODER.validate_python({"api": "cohere", "model": "embed-v4.0", "batch_size": 16})

    identity = retrieval_api._identity
    assert identity(DenseConfig(encoder=first), ["d"], []) == identity(DenseConfig(encoder=second), ["d"], [])
    assert identity(DenseConfig(encoder=first), ["d"], []) != identity(DenseConfig(encoder=other_model), ["d"], [])
    assert identity(DenseConfig(encoder=first), ["d"], []) != identity(DenseConfig(encoder=packed), ["d"], []), (
        "request packing is content: two batch sizes never share an index"
    )


def test_a_cached_index_is_not_reused_across_media_caps(dataset, tmp_path: Path) -> None:
    """A media cap decides how much one request carries: an index built with one ``max_images`` is rebuilt
    when the next run declares another (the identity carries it, so the cache check refuses to reuse it)."""
    one = DenseConfig(
        encoder=ServedEmbedding(base_url="fake://seed/7?dim=8", model="stub", max_images=1, **_SERVED_BUDGET)
    )
    two = DenseConfig(
        encoder=ServedEmbedding(base_url="fake://seed/7?dim=8", model="stub", max_images=2, **_SERVED_BUDGET)
    )
    assert retrieval_api._identity(one, ["d"], []) != retrieval_api._identity(two, ["d"], [])

    out = tmp_path / "idx"
    first = index(dataset, one, out=out)
    first_rankings = search(first, dataset, depth=3)
    stamp = (out / "index.json").stat().st_mtime_ns
    second = retrieve(dataset, two, depth=3, out=out)
    assert (out / "index.json").stat().st_mtime_ns != stamp, "the other media cap must rebuild the index"
    assert first_rankings.for_query("q1") == second.for_query("q1")  # same fake engine, same corpus
    assert load_index(out).identity != first.identity


def test_a_listwise_reranker_takes_no_batch_size_and_a_hosted_one_no_engine_fields() -> None:
    from rcp_ndcg.errors import ConfigError

    # A ConfigError with a hint escapes pydantic un-wrapped (one error shape for the config family).
    with pytest.raises(ConfigError, match="listwise"):
        _RERANKER.validate_python(
            {
                "api": "rerank",
                "model": "jina-reranker-v3",
                "listwise": True,
                "batch_size": 4,
                **_SERVED_RERANK_BUDGET,
            }
        )
    with pytest.raises(ValidationError, match="hosted"):
        _RERANKER.validate_python({"api": "voyage", "model": "rerank-2.5", "use_activation": True})
    with pytest.raises(ValidationError, match="hosted"):
        _RERANKER.validate_python({"api": "cohere", "model": "rerank-v4.0-pro", "instruction": "field"})


def test_a_served_reranker_config_builds_its_client(dataset) -> None:
    """The budget is wired: a served paper-style config (budget declared) builds its client, and the config
    that omits it is refused at the config (never silently ignored)."""
    from rcp_ndcg.inference.clients import RerankClient

    config = validate_reranker({"api": "rerank", "model": "m", "base_url": "http://h:8000/v1", **_SERVED_RERANK_BUDGET})
    client = RerankClient(config, sender=_recording_sender())
    try:
        assert client.config.max_tokens == 8192
    finally:
        client.close()
    with pytest.raises(ConfigError, match="tokenizer"):
        validate_reranker({"api": "rerank", "model": "m", "base_url": "http://h:8000/v1", "max_tokens": 8192})


def _recording_sender() -> Any:
    """A sender with the sync bridge, answering one score per document."""
    from rcp_ndcg.inference.types import Reply

    class _Recording:
        async def send(self, calls: list[Any]) -> list[Reply]:
            documents = calls[0].json["documents"]
            rows = [{"index": i, "relevance_score": float(i)} for i in range(len(documents))]
            return [Reply(200, {"results": rows}, {}) for _ in calls]

        async def probe(self) -> list[Any]:
            return []

        @property
        def usage(self) -> Any:
            from rcp_ndcg.inference.types import Usage

            return Usage()

        def run(self, coroutine: Any) -> Any:
            import asyncio

            return asyncio.run(coroutine)

    return _Recording()


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
    config = DenseConfig(encoder=ServedEmbedding(base_url="fake://seed/7?dim=8", model="stub", **_SERVED_BUDGET))

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
    config = LateInteractionConfig(
        encoder=ServedPooling(base_url="fake://seed/3?dim=4", model="colqwen", dim=4, **_SERVED_BUDGET)
    )

    built = index(dataset, config, out=tmp_path / "idx")
    rankings = search(built, dataset, depth=3)

    vectors = np.load(tmp_path / "idx" / "vectors.npy")
    offsets = np.load(tmp_path / "idx" / "offsets.npy")
    assert vectors.dtype == np.float16, "the transfer dtype reaches the index"
    assert len(offsets) == 4 and offsets[0] == 0 and offsets[-1] == len(vectors)
    assert set(rankings.for_query("q1")) <= set(DOCS)


def test_a_served_encoder_without_a_url_is_refused_where_its_client_is_built(dataset, tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="no base_url") as caught:
        retrieve(dataset, DenseConfig(encoder=ServedEmbedding(model="m", **_SERVED_BUDGET)), out=tmp_path / "idx")
    assert "serve.encoder" in (caught.value.hint or "")
    with pytest.raises(ConfigError, match="no base_url"):
        retrieve(
            dataset, LateInteractionConfig(encoder=ServedPooling(model="m", **_SERVED_BUDGET)), out=tmp_path / "idx"
        )


# ---------------------------------------------------------------------------
# Hosted profiles through the role clients over a mock wire
# ---------------------------------------------------------------------------


def test_an_index_from_the_previous_release_is_a_config_error_with_the_hint(dataset, tmp_path: Path) -> None:
    """An index.json written before the api rewiring (provider:-shaped) is refused by name, not as a raw error."""
    root = tmp_path / "idx"
    root.mkdir()
    (root / "index.json").write_text(
        json.dumps(
            {
                "schema": "rcp-ndcg.index.v1",
                "path": str(root),
                "dataset": "beir",
                "retriever": {"kind": "dense", "encoder": {"provider": "local", "model": "m", "pooling": "last"}},
                "identity": "0" * 64,
                "num_documents": 3,
            }
        )
    )

    with pytest.raises(ConfigError, match="api, not by provider") as caught:
        load_index(root)
    assert "api: openai_embeddings" in (caught.value.hint or "")


def test_retrieve_rebuilds_an_index_of_an_unreadable_shape(dataset, tmp_path: Path) -> None:
    """``retrieve(out=...)`` rebuilds over an index.json it cannot read (the IdentityError hint sends users
    there to rebuild); refusing them left a stale directory the call could never get past."""
    root = tmp_path / "idx"
    root.mkdir()
    (root / "index.json").write_text(
        json.dumps(
            {
                "schema": "rcp-ndcg.index.v1",
                "path": str(root),
                "dataset": "beir",
                "retriever": {"kind": "dense", "encoder": {"provider": "local", "model": "m", "pooling": "last"}},
                "identity": "0" * 64,
                "num_documents": 3,
            }
        )
    )
    config = DenseConfig(encoder=ServedEmbedding(base_url="fake://seed/7?dim=8", model="stub", **_SERVED_BUDGET))

    rankings = retrieve(dataset, config, depth=3, out=root)

    assert rankings.systems == ["stub"], "the old shape was rebuilt over, not refused"
    assert isinstance(load_index(root).retriever, DenseConfig), "the index was rewritten in the current shape"


def test_the_index_identity_carries_the_tokenizer_digest(dataset, tmp_path: Path) -> None:
    """The encoder's tokenizer digest enters the index identity (as it enters the step identities): the same
    bytes under another path share it, different bytes do not, and a URL change never does."""
    from tests._tokenizers import byte_bpe_tokenizer, save, word_tokenizer

    for directory in ("one", "two", "other"):
        (tmp_path / directory).mkdir()
    first = save(word_tokenizer(), tmp_path / "one")
    second = save(word_tokenizer(), tmp_path / "two")  # same bytes, different path
    other = save(byte_bpe_tokenizer(), tmp_path / "other")

    def identity(encoder: dict) -> str:
        config = DenseConfig(
            encoder=ServedEmbedding(model="stub", **{**_SERVED_BUDGET, "base_url": "fake://seed/7?dim=8", **encoder})
        )
        doc_ids, contents = retrieval_api._corpus(dataset)
        return retrieval_api._identity(config, doc_ids, contents)

    # The served role always declares a tokenizer (the explicit budget), so the digest is content: the
    # same bytes under any path share the identity, different bytes re-key, and the URL never does.
    base = identity({})
    assert identity({"tokenizer": str(first)}) == base, "the same bytes (any path) share the identity"
    assert identity({"tokenizer": str(second)}) == base, "the same bytes (any path) share the identity"
    assert identity({"tokenizer": str(other)}) != base, "different tokenizer bytes re-key"
    assert identity({"base_url": "http://elsewhere.test:8000/v1"}) == identity({"base_url": "fake://seed/7?dim=8"}), (
        "a URL is runtime, never identity"
    )


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
    config = ServedReranker(
        base_url="fake://seed/1", model="stub-reranker", instruction="fold", **_SERVED_RERANK_BUDGET
    )
    candidates = Rankings.from_scores({"q1": {"d1": 3.0, "d2": 2.0, "d3": 1.0}}, system="bm25")

    rescored = rerank(dataset, candidates, config, depth=2, out=tmp_path / "rerank")

    assert rescored.systems == ["stub-reranker"]
    top = rescored.for_query("q1")
    assert set(top) == {"d1", "d2"}
    assert (tmp_path / "rerank" / "rank000.jsonl").exists(), "the checkpoint records each scored query"


def test_the_rerank_client_receives_the_raw_query_and_appends_the_per_query_instruction(dataset) -> None:
    """The example's query and instruction go to the client raw: the instruction is the PER-QUERY one, and
    the client appends it as mteb appends it -- once, never folded as a task instruction."""
    from rcp_ndcg_core.records import RankingExample

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

        def run(self, coroutine: Any) -> Any:
            import asyncio

            return asyncio.run(coroutine)

    config = ServedReranker(base_url="http://rerank.test/v1", model="stub-reranker", **_SERVED_RERANK_BUDGET)
    client = RerankClient(config, sender=RecordingSender())
    example = RankingExample(
        query_id="q1", query="base query", instruction="Find relevant passages", doc_ids=["d1"], docs=["a document"]
    )
    try:
        client.rerank_many([example])
    finally:
        client.close()

    assert sent[0]["query"] == "base query Find relevant passages", "the per-query instruction is appended once"
    assert sent[0]["documents"] == ["a document"]


def test_a_hosted_reranker_runs_through_its_public_profile(dataset, tmp_path: Path, hosted_wire) -> None:
    config = CohereReranker(model="rerank-v4.0-fast", api_key_env="CO_API_KEY")
    candidates = Rankings.from_scores({"q1": {"d1": 3.0, "d2": 2.0, "d3": 1.0}}, system="bm25")

    rescored = rerank(dataset, candidates, config, depth=3)

    assert rescored.systems == ["rerank-v4.0-fast"]
    cohere_calls = [request for request in hosted_wire if request.url.host == "api.cohere.com"]
    assert cohere_calls and json.loads(cohere_calls[0].content)["top_n"] == 3


def test_rerank_refuses_a_depth_like_search_does(dataset) -> None:
    """``depth`` is validated in ``rerank`` like in ``search``: zero collapses the candidates to nothing (and
    raised the unrelated 'rankings hold 0 systems'), and a negative one silently cut by pandas' ``head(-n)``."""
    candidates = Rankings.from_scores({"q1": {"d1": 3.0, "d2": 2.0, "d3": 1.0}}, system="bm25")
    config = ServedReranker(base_url="fake://seed/1", model="stub-reranker", **_SERVED_RERANK_BUDGET)

    for depth in (0, -1):
        with pytest.raises(ConfigError, match="depth must be positive"):
            rerank(dataset, candidates, config, depth=depth)


def test_tied_candidates_reach_the_reranker_in_the_rankings_order(dataset, monkeypatch: pytest.MonkeyPatch) -> None:
    """The candidates go over the wire in the first stage's order (score descending, then the *lower*
    document id -- the one tie rule the top-k, the BM25 cut and ``Rankings.top`` share): a listwise model's
    scores depend on the batch composition, so it is pinned. The sweep's M14 mutation reversed the order and
    nothing failed."""
    sent: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        sent.append(body)
        documents = body["documents"]
        return httpx.Response(
            200, json={"results": [{"index": i, "relevance_score": float(i)} for i in range(len(documents))]}
        )

    from rcp_ndcg.inference import transport as transport_module

    real = transport_module.Transport

    def patched(endpoint: Any, **kwargs: Any) -> Any:
        return real(endpoint, httpx_transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr("rcp_ndcg.inference.clients._base.Transport", patched)
    tied = Rankings.from_scores({"q1": {"d1": 1.0, "d2": 1.0, "d3": 1.0}}, system="bm25")

    rerank(
        dataset,
        tied,
        ServedReranker(base_url="http://rerank.test/v1", model="stub-reranker", **_SERVED_RERANK_BUDGET),
        depth=3,
    )

    assert sent[0]["documents"] == [DOCS["d1"], DOCS["d2"], DOCS["d3"]], "ties: the lower document id"


def test_the_candidate_cut_keeps_the_lowest_ids_of_a_tie() -> None:
    """A9: the rerank depth cut applies the same tie rule as the first stage (score descending, then the
    lower document id), so a tie class straddling the cut keeps the documents ``search`` would return."""
    tied = Rankings.from_scores({"q1": {"d1": 1.0, "d2": 1.0, "d3": 1.0, "d4": 1.0}}, system="bm25")

    assert sorted(tied.top(2).for_query("q1")) == ["d1", "d2"]


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


def test_fuse_fuses_per_subset_files_and_refuses_nothing_at_all_first() -> None:
    """One file per (system, subset), as a per-subset fan-out writes them: a ranking with no rows for a
    subset does not enter that subset's fusion (the old path raised 'no rankings of dataset' naming the wrong
    datasets), and no rankings at all is refused before the loop."""
    one = Rankings.from_records([{"system": "a", "dataset": "one", "query_id": "q1", "doc_id": "d1", "score": 1.0}])
    two = Rankings.from_records([{"system": "a", "dataset": "two", "query_id": "q1", "doc_id": "d2", "score": 1.0}])

    fused = fuse([one, two])

    assert sorted(fused.datasets) == ["one", "two"], "each subset fused from the files that name it"
    assert fused.for_query("q1", system="rrf", dataset="one")["d1"] == pytest.approx(1 / 61)
    assert fused.for_query("q1", system="rrf", dataset="two")["d2"] == pytest.approx(1 / 61)
    with pytest.raises(DataError, match="needs rankings"):
        fuse([])


def test_fuse_fuses_every_system_of_a_concatenated_file() -> None:
    """V5: a file whose systems cover different subsets must fuse (the docstrings promise "every system of
    each file"); the per-file filter appended an empty run for the system that does not cover the subset,
    and the core's coverage guard then refused the whole fusion."""
    a_x = Rankings.from_scores({"q1": {"d1": 2.0, "d2": 1.0}}, system="A", dataset="x")
    b_y = Rankings.from_scores({"q1": {"d3": 1.0}}, system="B", dataset="y")
    c_x = Rankings.from_scores({"q1": {"d2": 2.0, "d1": 1.0}}, system="C", dataset="x")

    fused = fuse([Rankings.concat([a_x, b_y]), c_x], depth=2)

    assert sorted(fused.datasets) == ["x", "y"]
    assert fused.for_query("q1", dataset="x") == {
        "d1": pytest.approx(1 / 61 + 1 / 62),
        "d2": pytest.approx(1 / 62 + 1 / 61),
    }
    assert fused.for_query("q1", dataset="y") == {"d3": pytest.approx(1 / 61)}


def test_fuse_refuses_a_rankings_file_with_no_rows() -> None:
    """An empty file silently contributed nothing, and the result looked like a real fusion of the systems
    that did load (every overlapping score halved)."""
    nonempty = Rankings.from_scores({"q1": {"d1": 1.0}}, system="a")

    with pytest.raises(DataError, match="holds no rows"):
        fuse([Rankings.from_records([]), nonempty])


def test_fuse_validates_depth_and_rrf_k_by_their_public_names() -> None:
    """``fuse(depth=0)`` said "top_k" (the core's own argument) and ``fuse(rrf_k=0)`` passed the CLI schema
    before failing at runtime."""
    a = Rankings.from_scores({"q1": {"d1": 1.0}}, system="a")
    b = Rankings.from_scores({"q1": {"d1": 1.0}}, system="b")

    with pytest.raises(ConfigError, match="depth must be positive") as caught:
        fuse([a, b], depth=0)
    assert "rrf_k" not in str(caught.value)
    with pytest.raises(ConfigError, match="rrf_k must be positive"):
        fuse([a, b], rrf_k=0)


def test_fuse_ranks_a_tied_pair_by_the_lower_id() -> None:
    """A9/V5: within one system's rows a tie breaks by the lower document id (the retrieval stack's one
    rule), so a tied pair contributes the same ranks here as the first stage gave it."""
    # d2 first in the mapping's order: the old path kept that order at a tie (higher id first).
    a = Rankings.from_scores({"q1": {"d2": 1.0, "d1": 1.0}}, system="a")
    b = Rankings.from_scores({"q1": {"d1": 1.0, "d2": 1.0}}, system="b")

    fused = fuse([a, b], depth=2)

    assert fused.for_query("q1") == {
        "d1": pytest.approx(1 / 61 + 1 / 61),
        "d2": pytest.approx(1 / 62 + 1 / 62),
    }
