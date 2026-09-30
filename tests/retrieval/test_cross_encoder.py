"""Cross-encoder reranking: the served path and the vendor-``predict`` path.

The served path is tested against a fake HTTP client rather than a live engine, so
what is under test is what this module is actually responsible for: request
formatting (instructions, truncation budgets, media parts), score-to-doc_id alignment,
sharding, and checkpoint resume. The engine's own scoring is not ours to verify.
"""

from __future__ import annotations

from typing import Any
from unittest import mock

import pytest
from rcp_ndcg_core._records import RankingExample
from rcp_ndcg_core.content import Content, MediaRef

from rcp_ndcg.retrieval.cross_encoder import RerankSettings, rerank_examples


class _FakeRerankClient:
    """Stands in for the vLLM ``/rerank`` endpoint.

    Scores by query/document term overlap, which is deterministic and
    content-sensitive, so a mis-assembled request shows up as a wrong *order*
    rather than as an assertion on a payload shape nobody reads.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def rerank(self, query: Any, documents: list[Any], **kwargs: Any) -> list[float]:
        self.calls.append({"query": query, "documents": documents, **kwargs})
        query_text = query if isinstance(query, str) else str(query)
        terms = set(query_text.lower().split())
        return [float(len(terms & set(str(document).lower().split()))) for document in documents]


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch) -> _FakeRerankClient:
    client = _FakeRerankClient()
    monkeypatch.setattr(
        "rcp_ndcg.retrieval.vllm_http.VllmPoolingClient",
        lambda *args, **kwargs: client,
    )
    return client


def _examples() -> list[RankingExample]:
    return [
        RankingExample(
            query="capital of france",
            id="q1",
            doc_ids=["d1", "d2", "d3"],
            docs=["paris is the capital of france", "lyon is a city", "unrelated text"],
        ),
        RankingExample(
            query="who wrote moby dick",
            id="q2",
            doc_ids=["d4", "d5"],
            docs=["herman melville wrote moby dick", "ishmael narrates"],
        ),
    ]


class TestServedRerank:
    def test_scores_land_on_the_right_documents(self, fake_client: _FakeRerankClient) -> None:
        examples = _examples()
        out = rerank_examples(examples, RerankSettings(model_name="stub", api_base="http://h:8000"))

        assert out is not None
        for original, rescored in zip(examples, out, strict=True):
            assert rescored.scores is not None
            assert list(rescored.doc_ids) == list(original.doc_ids)
            assert all(score != float("-inf") for score in rescored.scores)

        # The relevant document must hold the top score, which only holds if each
        # response index was mapped back to its own doc_id.
        by_doc = dict(zip(out[0].doc_ids, out[0].scores or [], strict=True))
        assert by_doc["d1"] == max(by_doc.values())
        assert by_doc["d3"] == min(by_doc.values())

    def test_document_order_is_left_alone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Scores come back aligned with ``doc_ids``, not sorted.

        ``/rerank`` returns results ranked, with an ``index`` per result. Reading
        them in arrival order would attach each score to the wrong document while
        looking entirely plausible, so the alignment is restored and the candidate
        order is preserved for the caller to sort if it wants to.
        """

        class _RankedClient:
            def rerank(self, query: Any, documents: list[Any], **kwargs: Any) -> list[float]:
                return [0.1, 0.9]

        monkeypatch.setattr(
            "rcp_ndcg.retrieval.vllm_http.VllmPoolingClient",
            lambda *args, **kwargs: _RankedClient(),
        )
        example = RankingExample(query="q", id="q1", doc_ids=["a", "b"], docs=["first", "second"])
        out = rerank_examples([example], RerankSettings(model_name="stub", api_base="http://h:8000"))

        assert out is not None
        assert list(out[0].doc_ids) == ["a", "b"]
        assert out[0].scores == [0.1, 0.9]

    def test_truncation_budgets_are_sent_not_applied_locally(self, fake_client: _FakeRerankClient) -> None:
        """Truncation belongs where the tokenizer is, so it travels as request fields."""
        rerank_examples(_examples()[:1], RerankSettings(model_name="stub", api_base="http://h:8000"))

        call = fake_client.calls[0]
        assert (call["truncate_prompt_tokens"], call["max_tokens_per_query"]) == (8192, 4096)
        assert "max_tokens_per_doc" not in call

    def test_the_example_instruction_is_folded_into_the_query(self, fake_client: _FakeRerankClient) -> None:
        example = RankingExample(
            query="base query", id="q1", doc_ids=["d1"], docs=["doc"], instruction="Find relevant passages"
        )
        rerank_examples([example], RerankSettings(model_name="stub", api_base="http://h:8000"))
        assert fake_client.calls[0]["query"] == "Task: Find relevant passages\nQuery: base query"

    def test_image_documents_are_sent_as_content_not_as_empty_text(
        self,
        fake_client: _FakeRerankClient,
    ) -> None:
        """An image-only document's ``.text`` is ``""``; sending that scores nothing."""
        example = RankingExample(
            query="which page shows the revenue chart",
            id="q1",
            doc_ids=["page-1"],
            contents=[Content.from_image("file:///tmp/page1.png", sha256="a" * 64)],
        )
        rerank_examples([example], RerankSettings(model_name="stub", api_base="http://h:8000"))

        sent = fake_client.calls[0]["documents"][0]
        assert isinstance(sent, Content)
        assert sent.media[0].uri == "file:///tmp/page1.png"

    def test_a_query_with_no_documents_is_not_a_request(self, fake_client: _FakeRerankClient) -> None:
        example = RankingExample(query="q", id="q1", doc_ids=[], docs=[])
        out = rerank_examples([example], RerankSettings(model_name="stub", api_base="http://h:8000"))
        assert out is not None
        assert out[0].scores == []
        assert fake_client.calls == []

    def test_unscored_document_sorts_last_rather_than_disappearing(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The candidate set is part of the run's identity, so nothing is dropped."""

        class _PartialClient:
            def rerank(self, query: Any, documents: list[Any], **kwargs: Any) -> list[float]:
                return [1.0] * len(documents)

        monkeypatch.setattr(
            "rcp_ndcg.retrieval.vllm_http.VllmPoolingClient",
            lambda *args, **kwargs: _PartialClient(),
        )
        examples = [RankingExample(query="q", id="q1", doc_ids=["a"], docs=["x"])]
        out = rerank_examples(examples, RerankSettings(model_name="stub", api_base="http://h:8000"))
        assert out is not None

        # A query absent from the results map (e.g. a rank that never scored it)
        # still yields a full-length score list.
        from rcp_ndcg.retrieval.cross_encoder import _apply_scores

        recovered = _apply_scores(examples, {})
        assert recovered[0].scores == [float("-inf")]

    def test_non_main_rank_returns_none(self, fake_client: _FakeRerankClient) -> None:
        from rcp_ndcg.retrieval.accel import AccelState

        state = AccelState()
        with (
            mock.patch.object(type(state), "is_main_process", new_callable=mock.PropertyMock, return_value=False),
            mock.patch.object(type(state), "num_processes", new_callable=mock.PropertyMock, return_value=2),
            mock.patch.object(state, "wait_for_everyone"),
            mock.patch.object(state, "gather_object", return_value=[{}, {}]),
            mock.patch("rcp_ndcg.retrieval.cross_encoder.AccelState", return_value=state),
        ):
            out = rerank_examples(_examples(), RerankSettings(model_name="stub", api_base="http://h:8000"))
            assert out is None


class TestCheckpointing:
    def test_scores_survive_a_restart(self, fake_client: _FakeRerankClient, tmp_path: Any) -> None:
        examples = _examples()
        cfg = RerankSettings(model_name="stub", api_base="http://h:8000")
        ckpt = tmp_path / "partial"

        first = rerank_examples(examples, cfg, checkpoint_dir=ckpt)
        assert first is not None
        calls_after_first = len(fake_client.calls)

        second = rerank_examples(examples, cfg, checkpoint_dir=ckpt)
        assert second is not None
        # Nothing was re-scored, and the scores came back from disk.
        assert len(fake_client.calls) == calls_after_first
        assert [example.scores for example in second] == [example.scores for example in first]

    def test_a_truncated_checkpoint_line_does_not_lose_the_rest(
        self,
        fake_client: _FakeRerankClient,
        tmp_path: Any,
    ) -> None:
        """The usual crash signature is a process that died mid-flush."""
        from rcp_ndcg.retrieval.cross_encoder import _checkpoint_scores

        ckpt = tmp_path / "partial"
        ckpt.mkdir()
        (ckpt / "rank000.jsonl").write_text('{"q": "q1", "s": {"d1": 0.5}}\n{"q": "q2", "s": {"d4"\n')
        assert _checkpoint_scores(ckpt) == {"q1": {"d1": 0.5}}


def test_an_unknown_framework_is_refused_before_any_scoring() -> None:
    with pytest.raises(ValueError, match=r"unknown reranker framework 'hf'.*cohere.*vllm serve"):
        rerank_examples(_examples(), RerankSettings(model_name="stub", framework="hf"))


class _StubExternalReranker:
    """Stub vendor reranker: score = query/document term overlap."""

    def __init__(self) -> None:
        self.device: str | None = None

    def to(self, device: str) -> _StubExternalReranker:
        self.device = device
        return self

    def predict(self, query: str, docs: list[str]) -> list[float]:
        terms = set(query.lower().split())
        return [float(len(terms & set(doc.lower().split()))) for doc in docs]


class TestExternalFrameworks:
    def test_predict_path_is_used(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from rcp_ndcg.retrieval import external_rerankers as ext_mod

        calls: dict[str, object] = {}

        def fake_load(cfg, device=None):
            calls["framework"] = cfg.framework
            calls["device"] = device
            return _StubExternalReranker()

        monkeypatch.setattr(ext_mod, "load_external_model", fake_load)

        examples = [
            RankingExample(
                query="herman melville moby dick",
                id="q1",
                doc_ids=["d1", "d2"],
                docs=["herman melville wrote moby dick", "unrelated text about cooking"],
            ),
            RankingExample(query="empty", id="q2", doc_ids=[], docs=[]),
        ]
        out = rerank_examples(
            examples,
            RerankSettings(model_name="stub", framework="zerank"),
        )
        assert out is not None
        assert calls["framework"] == "zerank"
        assert list(out[0].doc_ids) == ["d1", "d2"]
        assert out[0].scores is not None and out[0].scores[0] > out[0].scores[1]
        assert out[1].scores == []


class TestConfigResolution:
    def test_shipped_configs_load_as_reranker_configs(self) -> None:
        """Every shipped reranker config is a ``RerankerConfig`` the retrieval API can run."""
        import yaml
        from pydantic import TypeAdapter

        from rcp_ndcg.retrieval import RerankerConfig
        from rcp_ndcg.retrieval._api import _rerank_settings
        from tests import REPO_ROOT

        paths = sorted((REPO_ROOT / "experiments" / "paper" / "rerankers").glob("*.yaml"))
        assert paths
        for path in paths:
            config = TypeAdapter(RerankerConfig).validate_python(yaml.safe_load(path.read_text(encoding="utf-8")))
            if getattr(config, "api_key_env", None) is None:
                assert _rerank_settings(config).model_name == config.model

    def test_media_ref_documents_reach_the_client_untouched(self, fake_client: _FakeRerankClient) -> None:
        ref = MediaRef(uri="gs://bucket/page.png", sha256="b" * 64, mime="image/png")
        example = RankingExample(
            query="q",
            id="q1",
            doc_ids=["p1"],
            contents=[Content.from_parts([*Content.from_image(ref.uri, sha256=ref.sha256).parts])],
        )
        rerank_examples([example], RerankSettings(model_name="stub", api_base="http://h:8000"))
        sent = fake_client.calls[0]["documents"][0]
        assert isinstance(sent, Content)
        assert sent.media[0].sha256 == ref.sha256
