"""Plugin adapters reach retrieval (C2): a registered adapter's name selects the role's generic config.

The shipped ``api`` names select their own config classes; any other ``api`` is resolved against the role's
registry where the config is read (an unregistered or wrong-role name is refused with the registry's hint) and
builds the role's generic endpoint config (:class:`~rcp_ndcg.retrieval.config.PluginEmbedding`,
:class:`~rcp_ndcg.retrieval.config.PluginReranker`), which ``index``/``search``/``retrieve``/``rerank`` run like
any other. The test adapters below speak the offline fakes' wire, so the round trips run over ``fake://``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from rcp_ndcg.data import Rankings, load_dataset
from rcp_ndcg.errors import ConfigError
from rcp_ndcg.inference.adapters.base import register_adapter
from rcp_ndcg.inference.types import Call, Embeddings, Reply, RerankResult
from rcp_ndcg.retrieval import (
    DenseConfig,
    ServedEmbedding,
    index,
    validate_reranker,
    validate_retriever,
)
from rcp_ndcg.retrieval import _api as retrieval_api
from rcp_ndcg.retrieval.config import PluginEmbedding, PluginReranker
from tests.conftest import SESSION_TOKENIZER

DOCS = {
    "d1": "tortoises move slowly across the sand",
    "d2": "hares run fast in open fields",
    "d3": "the sand dunes of the desert",
}


@register_adapter
class SlowEmbed:
    """A third-party embed adapter (the shipped OpenAI wire, under its own name): the test's C2 stand-in."""

    name = "slow_embed"
    role = "embed"
    HOSTED = False
    API_KEY_ENV = ()
    KEY_REQUIRED = False
    AUTH_HEADER = None
    DEFAULT_BASE_URL = None

    def __init__(self, config: Any | None = None) -> None:
        self.config = config

    def calls(self, request: Any, *, model: str) -> list[Call]:
        """One `POST /embeddings` for the batch, the shape the offline fake answers."""
        return [Call("POST", "/embeddings", {"model": model, "input": [content.text for content in request.contents]})]

    def interpret(self, request: Any, replies: list[Reply]) -> Embeddings:
        """One float32 row per item, in the request's order."""
        vectors = np.asarray([item["embedding"] for item in replies[0].body["data"]], dtype=np.float32)
        return Embeddings.single(vectors)

    def usage(self, reply: Reply) -> None:
        """The fake's shape reports no tokens."""
        return None


@register_adapter
class SlowPooling:
    """A third-party multi-vector adapter: only the `/pooling` wire, so a wrong dispatch cannot answer it."""

    name = "slow_pooling"
    role = "multi_vector"
    HOSTED = False
    API_KEY_ENV = ()
    KEY_REQUIRED = False
    AUTH_HEADER = None
    DEFAULT_BASE_URL = None

    def __init__(self, config: Any = None) -> None:
        self.config = config
        self.paths: list[str] = []

    def calls(self, request: Any, *, model: str) -> list[Call]:
        """One `POST /pooling` per batch, the shape the offline fake's pooling route answers."""
        self.paths.append("/pooling")
        return [
            Call(
                "POST",
                "/pooling",
                {
                    "model": model,
                    "input": [content.text for content in request.contents],
                    "task": "token_embed",
                    "encoding_format": "base64",
                    "embed_dtype": request.embed_dtype,
                    "endianness": "little",
                },
            )
        ]

    def interpret(self, request: Any, replies: list[Reply]) -> Embeddings:
        """The ragged rows of the base64 frames, one slice per item."""
        from rcp_ndcg.inference.adapters.pooling import VllmPooling

        return VllmPooling().interpret(request, replies)

    def usage(self, reply: Reply) -> None:
        return None


@register_adapter
class SlowRerank:
    """A third-party rerank adapter: the Cohere-shaped wire, under its own name."""

    name = "slow_rerank"
    role = "rerank"
    HOSTED = False
    API_KEY_ENV = ()
    KEY_REQUIRED = False
    AUTH_HEADER = None
    DEFAULT_BASE_URL = None

    def __init__(self, config: Any = None) -> None:
        self.config = config

    def calls(self, request: Any, *, model: str) -> list[Call]:
        """One `POST /rerank` for the query's candidate set."""
        return [
            Call(
                "POST",
                "/rerank",
                {
                    "model": model,
                    "query": request.query.text,
                    "documents": [document.text for document in request.documents],
                    "top_n": len(request.documents),
                },
            )
        ]

    def interpret(self, request: Any, replies: list[Reply]) -> RerankResult:
        """The rows realigned to the request's documents."""
        rows = sorted(replies[0].body["results"], key=lambda row: int(row["index"]))
        return RerankResult.aligned(request, [float(row["relevance_score"]) for row in rows])

    def usage(self, reply: Reply) -> None:
        return None


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


def test_a_registered_embed_adapter_runs_retrieve_end_to_end(dataset, tmp_path: Path) -> None:
    """`api: slow_embed` builds the generic endpoint config and `retrieve` runs it over the offline fake wire."""
    from rcp_ndcg.retrieval import retrieve

    config = validate_retriever(
        {
            "kind": "dense",
            "encoder": {"api": "slow_embed", "base_url": "fake://seed/7?dim=8", "model": "stub", "batch_size": 2},
        }
    )
    assert isinstance(config.encoder, PluginEmbedding), "a non-shipped api selects the generic endpoint config"

    rankings = retrieve(dataset, config, depth=3, out=tmp_path / "idx")

    assert rankings.systems == ["stub"]
    assert set(rankings.for_query("q1")) <= set(DOCS)


def test_a_registered_rerank_adapter_runs_rerank_end_to_end(dataset, tmp_path: Path) -> None:
    from rcp_ndcg.retrieval import rerank

    config = validate_reranker(
        {
            "api": "slow_rerank",
            "base_url": "fake://seed/1",
            "model": "stub-reranker",
            "instruction": "none",
            "use_activation": False,
            "tokenizer": str(SESSION_TOKENIZER),
            "max_tokens": 8192,
        }
    )
    assert isinstance(config, PluginReranker), "the generic rerank config carries the third-party api"

    rescored = rerank(
        dataset,
        Rankings.from_scores({"q1": {"d1": 3.0, "d2": 2.0, "d3": 1.0}}, system="bm25"),
        config,
        depth=2,
        out=tmp_path / "rerank",
    )

    assert rescored.systems == ["stub-reranker"]
    assert set(rescored.for_query("q1")) == {"d1", "d2"}


def test_a_registered_pooling_adapter_runs_late_interaction_end_to_end(
    dataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A plugin multi_vector config dispatches to PoolingClient: the adapter only ever sees `/pooling`."""
    from rcp_ndcg.retrieval import retrieve
    from rcp_ndcg.retrieval.config import PluginPooling

    config = validate_retriever(
        {
            "kind": "late_interaction",
            "encoder": {"api": "slow_pooling", "base_url": "fake://seed/3?dim=4", "model": "colqwen", "dim": 4},
        }
    )
    assert isinstance(config.encoder, PluginPooling)

    # The dispatch is by the pooling type: were _encode to route this config to the embedding client, the run
    # would construct one and this bomb fires.
    def _wrong_dispatch(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a pooling config reached the embedding client")

    monkeypatch.setattr("rcp_ndcg.retrieval._api.EmbeddingClient", _wrong_dispatch)

    built = index(dataset, config, out=tmp_path / "idx")
    rankings = retrieve(dataset, config, depth=3, out=tmp_path / "idx")

    assert built.num_documents == 3
    assert set(rankings.for_query("q1")) <= set(DOCS)
    assert set(np.load(tmp_path / "idx" / "offsets.npy")) >= {0.0}, "the ragged layout is stored"


def test_the_identity_keys_on_the_adapter_name(dataset) -> None:
    """The adapter name is content: the config payload carries it, so two plugins never share a key."""
    from rcp_ndcg.support.identity import identity_payload
    from tests.conftest import SESSION_TOKENIZER

    budget = {"tokenizer": str(SESSION_TOKENIZER), "max_tokens": 8192}
    plugin = validate_retriever(
        {
            "kind": "dense",
            "encoder": {"api": "slow_embed", "base_url": "fake://seed/7?dim=8", "model": "stub", **budget},
        }
    )
    shipped = DenseConfig(encoder=ServedEmbedding(base_url="fake://seed/7?dim=8", model="stub", **budget))

    doc_ids, contents = retrieval_api._corpus(dataset)

    assert identity_payload(plugin)["encoder"]["api"] == "slow_embed"
    assert retrieval_api._identity(plugin, doc_ids, contents) != retrieval_api._identity(shipped, doc_ids, contents), (
        "two different wires never share an index"
    )


def test_an_unregistered_name_is_refused_with_the_registrys_message() -> None:
    with pytest.raises(ConfigError, match="unknown embed adapter 'never_registered'") as caught:
        validate_retriever(
            {"kind": "dense", "encoder": {"api": "never_registered", "model": "m", "base_url": "http://h/v1"}}
        )
    assert "openai_embeddings" in (caught.value.hint or ""), "the hint lists the role's names"
    with pytest.raises(ConfigError, match="unknown rerank adapter 'never_registered'"):
        validate_reranker({"api": "never_registered", "model": "m", "base_url": "http://h/v1"})


def test_a_wrong_role_name_is_refused_with_the_registrys_message() -> None:
    """`slow_rerank` is registered for the rerank role only: an encoder config carrying it names the role."""
    with pytest.raises(ConfigError, match="unknown embed adapter 'slow_rerank'") as caught:
        validate_retriever(
            {"kind": "dense", "encoder": {"api": "slow_rerank", "model": "m", "base_url": "http://h/v1"}}
        )
    assert "rerank" in (caught.value.hint or ""), "the hint says the name is registered for the rerank role"
    with pytest.raises(ConfigError, match="unknown rerank adapter 'slow_embed'"):
        validate_reranker({"api": "slow_embed", "model": "m", "base_url": "http://h/v1"})


def test_a_plugin_config_survives_a_yaml_round_trip(tmp_path: Path) -> None:
    """A plugin config reads back from YAML: the api resolves against this process's registry."""
    path = tmp_path / "retriever.yaml"
    path.write_text(
        yaml.safe_dump(
            {"kind": "dense", "encoder": {"api": "slow_embed", "base_url": "fake://seed/7?dim=8", "model": "stub"}}
        )
    )
    config = validate_retriever(yaml.safe_load(path.read_text(encoding="utf-8")))
    assert isinstance(config.encoder, PluginEmbedding)
    assert config.encoder.api == "slow_embed"
