"""The retrieval API: typed configs, and index/search/retrieve/rerank/fuse from a Dataset to Rankings."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from pydantic import TypeAdapter, ValidationError

from rcp_ndcg.data import Rankings, load_dataset
from rcp_ndcg.errors import ConfigError, CredentialsError, IdentityError
from rcp_ndcg.retrieval import (
    BM25Config,
    Cohere,
    DenseConfig,
    EncoderConfig,
    Local,
    LocalEncoder,
    OpenAICompatibleReranker,
    RerankerConfig,
    RetrieverConfig,
    fuse,
    index,
    rerank,
    retrieve,
    search,
)
from rcp_ndcg.retrieval import _api as retrieval_api
from rcp_ndcg.retrieval.encoder import Embeddings

DOCS = {
    "d1": "tortoises move slowly across the sand",
    "d2": "hares run fast in open fields",
    "d3": "the sand dunes of the desert",
}


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


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------


_ENCODER = TypeAdapter(EncoderConfig)
_RERANKER = TypeAdapter(RerankerConfig)


def test_the_retriever_config_is_a_union_discriminated_on_kind_and_provider() -> None:
    adapter = TypeAdapter(RetrieverConfig)

    bm25 = adapter.validate_python({"kind": "bm25"})
    dense = adapter.validate_python(
        {"kind": "dense", "encoder": {"provider": "cohere", "model": "embed-v4.0", "api_key_env": "CO_API_KEY"}}
    )

    assert isinstance(bm25, BM25Config) and isinstance(dense, DenseConfig)
    assert isinstance(dense.encoder, Cohere)
    with pytest.raises(ValidationError):
        adapter.validate_python({"kind": "splade"})
    with pytest.raises(ValidationError, match="implementation"):
        adapter.validate_python({"kind": "bm25", "implementation": "numpy"})


@pytest.mark.parametrize(
    ("encoder", "match"),
    [
        ({"provider": "local", "model": "m"}, "pooling"),
        ({"provider": "local", "model": "m", "pooling": "mean"}, "last token"),
        ({"provider": "cohere", "model": "m", "pooling": "token"}, "pooling"),
        ({"provider": "local", "model": "m", "pooling": "last", "query_prompt": "q: "}, "no query prompt"),
        ({"provider": "openai_compatible", "model": "m", "base_url": "http://h", "concurrency": 4}, "one at a time"),
        ({"provider": "gemini", "model": "m", "concurrency": 4}, "one at a time"),
    ],
)
def test_an_encoder_takes_only_what_its_provider_uses(encoder: dict, match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        _ENCODER.validate_python(encoder)


def test_single_and_multi_vector_retrievers_are_different_kinds() -> None:
    adapter = TypeAdapter(RetrieverConfig)
    token = {"provider": "openai_compatible", "model": "colqwen", "base_url": "http://h:8000", "pooling": "token"}
    local = {"provider": "local", "model": "colqwen", "pooling": "token", "engine": "vllm"}

    with pytest.raises(ValidationError, match="late interaction"):
        adapter.validate_python({"kind": "dense", "encoder": token})
    with pytest.raises(ValidationError, match="late interaction needs"):
        adapter.validate_python({"kind": "late_interaction", "encoder": {**local, "pooling": "last"}})
    assert adapter.validate_python({"kind": "late_interaction", "encoder": token}).kind == "late_interaction"
    assert adapter.validate_python({"kind": "late_interaction", "encoder": local}).kind == "late_interaction"


def test_where_an_encoder_runs_is_not_part_of_what_an_index_is() -> None:
    """The endpoint's timeouts, retries and batch size are runtime; the model, revision and prompts are content."""
    first = _ENCODER.validate_python({"provider": "cohere", "model": "embed-v4.0", "batch_size": 8})
    second = _ENCODER.validate_python({"provider": "cohere", "model": "embed-v4.0", "timeout_s": 5, "max_retries": 0})
    other_model = _ENCODER.validate_python({"provider": "cohere", "model": "embed-v3.0"})

    identity = retrieval_api._identity
    assert identity(DenseConfig(encoder=first), ["d"], []) == identity(DenseConfig(encoder=second), ["d"], [])
    assert identity(DenseConfig(encoder=first), ["d"], []) != identity(DenseConfig(encoder=other_model), ["d"], [])


# ---------------------------------------------------------------------------
# index, search, retrieve
# ---------------------------------------------------------------------------


def test_bm25_indexes_and_searches_a_dataset(dataset, tmp_path: Path) -> None:
    built = index(dataset, BM25Config(), out=tmp_path / "idx")

    rankings = search(built, dataset, depth=2)

    assert rankings.systems == ["bm25"]
    for query_id, expected in (("q1", "d1"), ("q2", "d2")):
        scores = rankings.for_query(query_id)
        assert max(scores, key=scores.__getitem__) == expected
        assert len(scores) <= 2


class _Encoder:
    """Documents and queries as fixed vectors: q1 is closest to d3, q2 to d2."""

    vectors = {"tortoises": [0.0, 0.1, 1.0], "hares": [0.0, 1.0, 0.0], "sand": [0.0, 0.0, 1.0]}

    def encode(self, contents, *, role, batch_size=None) -> Embeddings:
        rows = [next(v for word, v in self.vectors.items() if word in c.text) for c in contents]
        return Embeddings.single(np.asarray(rows, dtype=np.float32))


def test_dense_retrieval_ranks_by_inner_product_and_reuses_an_index(dataset, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(retrieval_api, "_encoder", lambda config: _Encoder())
    config = DenseConfig(encoder=LocalEncoder(model="stub", pooling="last"))

    first = retrieve(dataset, config, depth=3, out=tmp_path / "idx")
    stamp = (tmp_path / "idx" / "index.json").stat().st_mtime_ns
    again = retrieve(dataset, config, depth=3, out=tmp_path / "idx")

    scores = first.for_query("q2", system="stub")
    assert sorted(scores, key=scores.__getitem__, reverse=True)[0] == "d2"
    assert again == first
    assert (tmp_path / "idx" / "index.json").stat().st_mtime_ns == stamp, "the same identity reuses the index"


class _TiedEncoder:
    """Every document and query as the same vector: every score ties."""

    def encode(self, contents, *, role, batch_size=None) -> Embeddings:
        return Embeddings.single(np.ones((len(contents), 3), dtype=np.float32))


def test_dense_ties_break_toward_the_lower_document_id_whatever_the_corpus_order(tmp_path, monkeypatch) -> None:
    """The corpus lists d3 first; with every score tied, depth 2 keeps d1 and d2, on every run and platform."""
    root = tmp_path / "beir"
    (root / "qrels").mkdir(parents=True)
    corpus = [("d3", "c"), ("d1", "a"), ("d2", "b")]
    (root / "corpus.jsonl").write_text("".join(json.dumps({"_id": d, "text": t}) + "\n" for d, t in corpus))
    (root / "queries.jsonl").write_text(json.dumps({"_id": "q1", "text": "q"}) + "\n")
    (root / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td1\t1\n")
    monkeypatch.setattr(retrieval_api, "_encoder", lambda config: _TiedEncoder())
    config = DenseConfig(encoder=LocalEncoder(model="stub", pooling="last"))

    rankings = retrieve(load_dataset(f"beir:{root}"), config, depth=2, out=tmp_path / "idx")

    assert sorted(rankings.for_query("q1", system="stub")) == ["d1", "d2"]


def test_an_index_refuses_a_different_corpus(dataset, tmp_path: Path) -> None:
    built = index(dataset, BM25Config(), out=tmp_path / "idx")
    other_root = tmp_path / "other"
    (other_root / "qrels").mkdir(parents=True)
    (other_root / "corpus.jsonl").write_text(json.dumps({"_id": "x", "text": "another corpus"}) + "\n")
    (other_root / "queries.jsonl").write_text(json.dumps({"_id": "q1", "text": "q"}) + "\n")
    (other_root / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\nq1\tx\t1\n")

    with pytest.raises(IdentityError, match="another corpus or retriever"):
        search(built, load_dataset(f"beir:{other_root}"))


# ---------------------------------------------------------------------------
# rerank and fuse
# ---------------------------------------------------------------------------


def test_rerank_rescores_the_top_candidates_with_the_reranker(dataset, monkeypatch) -> None:
    seen = {}

    def fake_rerank(examples, settings, **kwargs):
        seen["settings"], seen["orders"] = settings, {e.id: list(e.doc_ids) for e in examples}
        return [e.model_copy(update={"scores": [float(i) for i in range(len(e.doc_ids))]}) for e in examples]

    monkeypatch.setattr("rcp_ndcg.retrieval.cross_encoder.rerank_examples", fake_rerank)
    candidates = Rankings.from_scores({"q1": {"d1": 3.0, "d2": 2.0, "d3": 1.0}}, system="bm25")

    reranked = rerank(dataset, candidates, OpenAICompatibleReranker(model="rr", base_url="http://h:8000"), depth=2)

    assert seen["orders"] == {"q1": ["d1", "d2"]}, "the top `depth` candidates, best first"
    assert (seen["settings"].framework, seen["settings"].api_base) == ("vllm", "http://h:8000")
    assert reranked.for_query("q1", system="rr") == {"d1": 0.0, "d2": 1.0}


def test_reranker_backends_resolve_to_their_implementations(monkeypatch) -> None:
    monkeypatch.setenv("CO_KEY", "k")

    cohere = retrieval_api._rerank_settings(Cohere(model="rerank-v4.0-pro", api_key_env="CO_KEY"))
    qwen = retrieval_api._rerank_settings(Local(model="Qwen/Qwen3-Reranker-4B"))
    jina = retrieval_api._rerank_settings(Local(model="jinaai/jina-reranker-v3"))

    assert (cohere.framework, cohere.api_key, qwen.framework, jina.framework) == ("cohere", "k", "qwen_og", "jina_hf")
    with pytest.raises(CredentialsError, match="NOPE_KEY"):
        retrieval_api._rerank_settings(
            _RERANKER.validate_python({"provider": "voyage", "model": "r", "api_key_env": "NOPE_KEY"})
        )
    with pytest.raises(ConfigError, match="no in-process reranker"):
        retrieval_api._rerank_settings(Local(model="someone/else"))


def test_cohere_rerank_scores_each_document_through_the_public_api(monkeypatch) -> None:
    from rcp_ndcg.retrieval.external_rerankers import CohereRerank

    class Response:
        status_code = 200

        def raise_for_status(self) -> None: ...

        def json(self) -> dict:
            return {"results": [{"index": 1, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.2}]}

    calls = []
    monkeypatch.setattr("httpx.post", lambda url, json, headers, timeout: calls.append(json) or Response())

    scores = CohereRerank(api_key="k", model_name="rerank-v4.0-fast").predict("q", ["a", "", "b"])

    assert scores == [0.2, 0.0, 0.9], "empty documents are not sent and score 0"
    assert calls == [{"model": "rerank-v4.0-fast", "query": "q", "documents": ["a", "b"], "top_n": 2}]
    monkeypatch.delenv("CO_API_KEY", raising=False)
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    with pytest.raises(CredentialsError):
        CohereRerank()


def test_fuse_sums_reciprocal_ranks() -> None:
    first = Rankings.from_scores({"q1": {"a": 3.0, "b": 2.0, "c": 1.0}}, system="bm25")
    second = Rankings.from_scores({"q1": {"b": 3.0, "c": 2.0, "a": 1.0}}, system="dense")

    fused = fuse([first, second], rrf_k=60, depth=2).for_query("q1", system="rrf")

    assert list(fused) == ["b", "a"]
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)


def test_fuse_of_one_system_gives_its_rrf_scores() -> None:
    one = Rankings.from_scores({"q1": {"a": 3.0, "b": 2.0}}, system="bm25")

    assert fuse([one]).for_query("q1") == {"a": 1 / 61, "b": 1 / 62}


def test_fuse_keeps_the_datasets_of_a_suite_apart() -> None:
    """Two subsets share the query id "0": each is fused on its own and keeps its dataset."""
    rows = [
        {"system": s, "dataset": ds, "query_id": "0", "doc_id": f"{ds}-{d}", "score": score}
        for s, order in (("bm25", "xy"), ("dense", "yx"))
        for ds in ("biology", "earth_science")
        for d, score in zip(order, (2.0, 1.0), strict=True)
    ]

    fused = fuse([Rankings.from_records(rows)])

    assert fused.datasets == ["biology", "earth_science"]
    assert set(fused.for_query("0", dataset="biology")) == {"biology-x", "biology-y"}
    assert set(fused.for_query("0", dataset="earth_science")) == {"earth_science-x", "earth_science-y"}


# ---------------------------------------------------------------------------
# The in-process (hf) encoder: what the config says reaches the model
# ---------------------------------------------------------------------------


class _FakeTransformers:
    """``transformers`` with ``AutoModel``/``AutoTokenizer`` that record how they were loaded."""

    def __init__(self) -> None:
        self.loads: list[tuple[str, str, dict]] = []
        fake = self

        class _Model:
            config = type("Config", (), {"hidden_size": 4})()

            def to(self, device):
                return self

            def eval(self):
                return self

        class _Auto:
            def __init__(self, kind: str) -> None:
                self.kind = kind

            def from_pretrained(self, name, **kwargs):
                fake.loads.append((self.kind, name, kwargs))
                return _Model()

        self.module = type(sys)("transformers")
        self.module.AutoModel = _Auto("model")
        self.module.AutoTokenizer = _Auto("tokenizer")


@pytest.fixture
def fake_transformers(monkeypatch: pytest.MonkeyPatch) -> _FakeTransformers:
    pytest.importorskip("torch")
    fake = _FakeTransformers()
    monkeypatch.setitem(sys.modules, "transformers", fake.module)
    return fake


def _encoded_texts(monkeypatch: pytest.MonkeyPatch, encoder, role) -> list[str]:
    seen: list[str] = []

    def fake_encode(model, tokenizer, texts, **kwargs):
        seen.extend(texts)
        return np.zeros((len(texts), 4), dtype=np.float32)

    monkeypatch.setattr("rcp_ndcg.retrieval.hf_dense.encode_text_batches", fake_encode)
    encoder.encode_texts(["tortoises"], role=role)
    return seen


def test_the_octen_config_prefixes_documents_and_not_queries(fake_transformers, monkeypatch) -> None:
    """The paper encoded Octen documents as ``"- " + text`` and queries bare; the shipped config says so."""
    from rcp_ndcg.retrieval.encoder import EncodeRole

    octen = yaml.safe_load((_PAPER / "retrieval" / "octen.yaml").read_text(encoding="utf-8"))
    config = TypeAdapter(RetrieverConfig).validate_python(octen)
    encoder = retrieval_api._encoder(config.encoder)

    assert config.encoder.doc_prompt == "- "
    assert _encoded_texts(monkeypatch, encoder, EncodeRole.DOCUMENT) == ["- tortoises"]
    assert _encoded_texts(monkeypatch, encoder, EncodeRole.QUERY) == ["tortoises"]


def test_the_hf_encoder_has_no_prefix_the_config_does_not_state(fake_transformers, monkeypatch) -> None:
    from rcp_ndcg.retrieval.encoder import EncodeRole

    encoder = retrieval_api._encoder(LocalEncoder(model="some/model", pooling="last"))

    assert _encoded_texts(monkeypatch, encoder, EncodeRole.DOCUMENT) == ["tortoises"]


def test_the_encoder_revision_is_the_revision_loaded(fake_transformers) -> None:
    retrieval_api._encoder(LocalEncoder(model="some/model", pooling="last", revision="abc"))

    assert {(kind, name, kwargs.get("revision")) for kind, name, kwargs in fake_transformers.loads} == {
        ("model", "some/model", "abc"),
        ("tokenizer", "some/model", "abc"),
    }


# ---------------------------------------------------------------------------
# Rerankers: every config field reaches the backend it applies to, or is refused
# ---------------------------------------------------------------------------

_PAPER = Path(__file__).parents[2] / "experiments" / "paper"


def test_the_cohere_configs_send_100_documents_per_request() -> None:
    """Cohere bills per request (a search unit is 100 documents), so the batch is the API's, not 8."""
    for name in ("cohere_rerank_v4_pro", "cohere_rerank_v4_fast"):
        config = yaml.safe_load((_PAPER / f"rerankers/{name}.yaml").read_text(encoding="utf-8"))

        assert retrieval_api._rerank_settings(_RERANKER.validate_python(config)).request_size == 100


def test_the_paper_cohere_configs_read_either_key_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    """They once pinned api_key_env: CO_API_KEY, so a key in COHERE_API_KEY alone (as documented) exited 5."""
    from rcp_ndcg.retrieval.external_rerankers import CohereRerank

    monkeypatch.delenv("CO_API_KEY", raising=False)
    monkeypatch.setenv("COHERE_API_KEY", "k")
    for name in ("cohere_rerank_v4_pro", "cohere_rerank_v4_fast"):
        config = _RERANKER.validate_python(yaml.safe_load((_PAPER / f"rerankers/{name}.yaml").read_text()))
        settings = retrieval_api._rerank_settings(config)
        assert CohereRerank(api_key=settings.api_key, model_name=settings.model_name)._api_key == "k"
    embed = yaml.safe_load((_PAPER / "retrieval/cohere_embed_v4.yaml").read_text())
    retriever = TypeAdapter(RetrieverConfig).validate_python(embed)
    assert retrieval_api._encoder(retriever.encoder).client._key() == "k"


@pytest.mark.parametrize(
    ("config", "match"),
    [
        ({"provider": "openai_compatible", "model": "m", "base_url": "http://h", "batch_size": 4}, "batch_size"),
        ({"provider": "cohere", "model": "rerank-v4.0-pro", "concurrency": 4}, "concurrency"),
        ({"provider": "local", "model": "m", "base_url": "http://h"}, "base_url"),
        ({"provider": "gemini", "model": "m"}, "provider"),
    ],
)
def test_a_field_the_provider_does_not_use_is_refused(config: dict, match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        _RERANKER.validate_python(config)


def test_a_served_model_without_a_url_is_refused_where_its_client_is_built() -> None:
    """``base_url`` is omitted only when the run's job starts the model's engine (``serve.``); building the
    retrieval client without one is refused, never silently pointed at a vendor's public API."""
    from rcp_ndcg.retrieval import _api as retrieval_api
    from rcp_ndcg.retrieval.config import OpenAICompatibleEncoder

    with pytest.raises(ConfigError, match="has no base_url"):
        retrieval_api._encoder(OpenAICompatibleEncoder(model="m"))


@pytest.mark.parametrize("model", ["jinaai/jina-reranker-v3", "zeroentropy/zerank-2"])
def test_a_local_reranker_that_batches_on_its_own_refuses_a_batch_size(model: str) -> None:
    with pytest.raises(ConfigError, match="batch_size"):
        retrieval_api._rerank_settings(Local(model=model, batch_size=4))


def test_a_hosted_rerankers_endpoint_reaches_its_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    """base_url (a proxy), timeout_s and max_retries of a Cohere config are what the requests use."""
    from rcp_ndcg.retrieval import external_rerankers

    seen: list = []
    monkeypatch.setattr(
        "rcp_ndcg.retrieval._http.post_json",
        lambda url, payload, **kwargs: seen.append((url, kwargs)) or {"results": [{"index": 0, "relevance_score": 1}]},
    )
    config = Cohere(
        model="rerank-v4.0-pro", api_key_env="CO_KEY", base_url="http://proxy/v2", timeout_s=7, max_retries=1
    )
    monkeypatch.setenv("CO_KEY", "k")

    external_rerankers.load_external_model(retrieval_api._rerank_settings(config)).predict("q", ["a"])

    ((url, kwargs),) = seen
    assert url == "http://proxy/v2/rerank"
    assert (kwargs["timeout_s"], kwargs["max_retries"]) == (7, 1)


def test_the_reranker_revision_is_the_revision_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    from rcp_ndcg.retrieval import external_rerankers

    seen: dict = {}
    monkeypatch.setattr(external_rerankers, "QwenOGRerank", lambda **kwargs: seen.update(kwargs))
    settings = retrieval_api._rerank_settings(Local(model="Qwen/Qwen3-Reranker-4B", revision="abc"))

    external_rerankers.load_external_model(settings, device="cpu")

    assert (seen["model_name_or_path"], seen["revision"]) == ("Qwen/Qwen3-Reranker-4B", "abc")


def test_jina_loads_the_revision(fake_transformers: _FakeTransformers) -> None:
    from rcp_ndcg.retrieval.external_rerankers import JinaRerank

    fake_transformers.module.AutoModelForSequenceClassification = fake_transformers.module.AutoModel
    JinaRerank("jinaai/jina-reranker-v3", revision="abc", device="cpu")

    assert {(kind, kwargs.get("revision")) for kind, _, kwargs in fake_transformers.loads} == {
        ("model", "abc"),
        ("tokenizer", "abc"),
    }


def test_voyage_reranks_over_http_without_its_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    from rcp_ndcg.retrieval.external_rerankers import VoyageRerank

    class Response:
        status_code = 200

        def raise_for_status(self) -> None: ...

        def json(self) -> dict:
            return {"data": [{"index": 1, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.2}]}

    calls = []
    monkeypatch.setitem(sys.modules, "voyageai", None)
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    monkeypatch.setattr("httpx.post", lambda url, json, headers, timeout: calls.append((url, json)) or Response())

    scores = VoyageRerank(api_key="k", model_name="rerank-2.5").predict("q", ["a", "", "b"])

    assert scores == [0.2, 0.0, 0.9], "empty documents are not sent and score 0"
    assert calls == [
        ("https://api.voyageai.com/v1/rerank", {"model": "rerank-2.5", "query": "q", "documents": ["a", "b"]})
    ]
