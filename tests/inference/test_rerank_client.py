"""The rerank role client: the instruction modes, the per-query checkpoint, the concurrency and the refusals.

The client owns the one rule every rerank path shares -- how the instruction reaches the model -- and the
per-query checkpoint of today's served path. The sender is a fake; the transport's behaviour is lane L1's.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any, ClassVar

import pytest
from rcp_ndcg_core._records import RankingExample

from rcp_ndcg.errors import ConfigError
from rcp_ndcg.inference.clients import RerankClient
from rcp_ndcg.inference.config import RerankEndpoint
from rcp_ndcg.inference.types import Call, Reply, RerankResult, Usage

# ---------------------------------------------------------------------------------------------------------------
# The fake server: a Sender answering /rerank over the three body shapes
# ---------------------------------------------------------------------------------------------------------------


class _FakeRerankServer:
    """A :class:`~rcp_ndcg.inference.transport.Sender` that answers each ``/rerank`` call with one row per
    document, scored by the text so that every document scores differently, in arrival order shuffled -- so a
    positional reading of the answers shows."""

    shape: ClassVar[str] = "results"  # "results" | "data" | "list"

    def __init__(self, *, refuse: int | None = None, body: dict[str, Any] | None = None) -> None:
        self.calls: list[Call] = []
        self.refuse = refuse
        self.body = body

    def score(self, text: str) -> float:
        return (len(text) * 37 % 97) / 97

    def rows(self, documents: list[str]) -> list[dict[str, Any]]:
        rows = [
            {"index": index, "relevance_score": self.score(str(document))} for index, document in enumerate(documents)
        ]
        return rows[::-1]  # answered ranked

    async def send(self, calls: Sequence[Call]) -> list[Reply]:
        replies: list[Reply] = []
        for call in calls:
            self.calls.append(call)
            if self.refuse is not None:
                replies.append(Reply(self.refuse, self.body or {"error": {"message": "refused"}}, {}))
                continue
            documents = (call.json or {}).get("documents", [])
            if self.shape == "results":
                replies.append(Reply(200, {"results": self.rows(documents)}, {}))
            elif self.shape == "data":
                replies.append(Reply(200, {"data": self.rows(documents)}, {}))
            else:
                bare = [{"index": row["index"], "score": row["relevance_score"]} for row in self.rows(documents)]
                replies.append(Reply(200, bare, {}))
        return replies

    async def probe(self) -> list[Any]:
        return []

    @property
    def usage(self) -> Usage:
        return Usage()


class _CountingServer(_FakeRerankServer):
    """The same server, tracking how many sends are in flight at once (the concurrency probe)."""

    def __init__(self) -> None:
        super().__init__()
        self.in_flight = 0
        self.max_in_flight = 0

    async def send(self, calls: Sequence[Call]) -> list[Reply]:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(0.001)  # yield, so concurrent queries pile up under the semaphore
            return await super().send(calls)
        finally:
            self.in_flight -= 1


def _server(**kwargs: Any) -> _FakeRerankServer:
    return _FakeRerankServer(**kwargs)


def _config(**kwargs: Any) -> RerankEndpoint:
    kwargs.setdefault("base_url", "http://engine:8000/v1")
    kwargs.setdefault("model", "qwen3-reranker-8b")
    return RerankEndpoint(**kwargs)


def _examples(*queries: tuple[str, str | None], doc: str = "relevant passage") -> list[RankingExample]:
    return [
        RankingExample(query=text, id=f"q{index}", doc_ids=["d1"], docs=[doc], instruction=instruction)
        for index, (text, instruction) in enumerate(queries)
    ]


# ---------------------------------------------------------------------------------------------------------------
# The instruction modes: the one rule for every path
# ---------------------------------------------------------------------------------------------------------------


class TestInstructionModes:
    def test_fold_sends_exactly_the_served_paths_query_text(self) -> None:
        """``Task: <instruction>\\nQuery: <text>`` -- the served path's render, byte for byte."""
        server = _server()
        client = RerankClient(_config(), sender=server)
        result = client.rerank("base query", ["doc"], instruction="Find relevant passages")

        assert server.calls[0].json["query"] == "Task: Find relevant passages\nQuery: base query"
        assert "instruction" not in server.calls[0].json
        assert result.scores == (server.score("doc"),)

    def test_fold_without_an_instruction_sends_the_bare_query(self) -> None:
        server = _server()
        RerankClient(_config(), sender=server).rerank("base query", ["doc"])

        assert server.calls[0].json["query"] == "base query"

    def test_field_sends_the_bare_query_plus_the_instruction_field(self) -> None:
        server = _server()
        RerankClient(_config(instruction="field"), sender=server).rerank(
            "base query", ["doc"], instruction="Find relevant passages"
        )

        assert server.calls[0].json["query"] == "base query"
        assert server.calls[0].json["instruction"] == "Find relevant passages"

    def test_none_sends_the_bare_query_and_no_field(self) -> None:
        server = _server()
        RerankClient(_config(instruction="none"), sender=server).rerank(
            "base query", ["doc"], instruction="Find relevant passages"
        )

        assert server.calls[0].json["query"] == "base query"
        assert "instruction" not in server.calls[0].json

    def test_rerank_many_folds_the_examples_instruction_once(self) -> None:
        """The example's raw query and instruction go in; the folded format_content() text would fold twice."""
        server = _server()
        RerankClient(_config(), sender=server).rerank_many(
            [
                RankingExample(
                    query="base query", id="q1", doc_ids=["d1"], docs=["doc"], instruction="Find relevant passages"
                )
            ]
        )

        assert server.calls[0].json["query"] == "Task: Find relevant passages\nQuery: base query"

    def test_arerank_is_the_async_half(self) -> None:
        async def run() -> tuple[str, Any]:
            server = _server()
            result = await RerankClient(_config(), sender=server).arerank("q", ["a", "b"], instruction="i")
            return server.calls[0].json["query"], result.scores

        query, scores = asyncio.run(run())
        assert query == "Task: i\nQuery: q"
        assert scores == (_server().score("a"), _server().score("b"))


# ---------------------------------------------------------------------------------------------------------------
# The per-query checkpoint and the concurrency
# ---------------------------------------------------------------------------------------------------------------


class TestRerankMany:
    def test_the_checkpoint_is_called_once_per_query_as_each_lands(self) -> None:
        server = _server()
        client = RerankClient(_config(), sender=server)
        examples = _examples(("first", "i1"), ("second", None), ("third", "i3"))
        seen: list[tuple[str, tuple[float, ...]]] = []

        results = client.rerank_many(examples, checkpoint=lambda query_id, scores: seen.append((query_id, scores)))

        assert [str(example.id) for example in examples] == [query_id for query_id, _ in seen]
        assert [result.scores for result in results] == [scores for _, scores in seen]
        assert all(scores == (server.score("relevant passage"),) for _, scores in seen)

    def test_concurrency_bounds_the_queries_in_flight(self) -> None:
        server = _CountingServer()
        client = RerankClient(_config(concurrency=2), sender=server)
        examples = _examples(*[(f"query {index}", None) for index in range(6)])

        client.rerank_many(examples)

        assert server.max_in_flight == 2, "at most `concurrency` queries in flight, and they do run together"

    def test_an_example_with_no_documents_is_checkpointed_without_a_request(self) -> None:
        server = _server()
        client = RerankClient(_config(), sender=server)
        empty = RankingExample(query="q", id="q0", doc_ids=[], docs=[])
        seen: list[tuple[str, tuple[float, ...]]] = []

        results = client.rerank_many([empty], checkpoint=lambda query_id, scores: seen.append((query_id, scores)))

        assert server.calls == []
        assert seen == [("q0", ())]
        assert results[0].scores == ()


# ---------------------------------------------------------------------------------------------------------------
# What the client refuses, and what it sends as given
# ---------------------------------------------------------------------------------------------------------------


class TestRefusalsAndPassthrough:
    def test_max_tokens_needs_the_text_budget_mechanism(self) -> None:
        """A budget is never silently ignored: until the mechanism lands, the config is refused."""
        with pytest.raises(ConfigError, match=r"^max_tokens needs the text-budget mechanism, which is not wired yet$"):
            RerankClient(_config(max_tokens=8192), sender=_server())

    def test_a_served_endpoint_needs_a_base_url(self) -> None:
        with pytest.raises(ConfigError, match="base_url"):
            RerankClient(RerankEndpoint(model="m"), sender=_server())

    def test_a_hosted_profile_defaults_to_its_public_root(self) -> None:
        server = _server()
        client = RerankClient(RerankEndpoint(api="voyage", model="rerank-2.5"), sender=server)

        assert client.config.base_url == "https://api.voyageai.com/v1"
        client.rerank("q", ["a"])
        assert server.calls[0].json == {"model": "rerank-2.5", "query": "q", "documents": ["a"]}

    def test_empty_documents_are_sent_as_given(self) -> None:
        """Nothing is filtered: an empty document scores whatever the server returns."""
        server = _server()
        result = RerankClient(_config(), sender=server).rerank("q", ["a", "", "b"])

        assert server.calls[0].json["documents"] == ["a", "", "b"]
        assert result.scores == (server.score("a"), server.score(""), server.score("b"))

    def test_a_response_with_shuffled_indices_comes_back_aligned(self) -> None:
        server = _server()
        result = RerankClient(_config(), sender=server).rerank("q", ["a", "b", "c"])

        assert result.scores == (server.score("a"), server.score("b"), server.score("c"))

    def test_the_voyage_profile_sleeps_between_its_requests(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The pause of today's ``VoyageRerank``: one half-second sleep before each request."""
        sleeps: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            sleeps.append(seconds)

        monkeypatch.setattr("rcp_ndcg.inference.clients.rerank.asyncio.sleep", fake_sleep)
        server = _server()
        RerankClient(RerankEndpoint(api="voyage", model="rerank-2.5"), sender=server).rerank("q", ["a"])

        assert sleeps == [0.5]

    def test_a_refusal_travels_as_its_typed_error(self) -> None:
        from rcp_ndcg.errors import CapabilityError

        server = _server(refuse=400, body={"error": {"message": "maximum context length exceeded"}})
        with pytest.raises(CapabilityError, match="too long"):
            RerankClient(_config(), sender=server).rerank("q", ["a"])


# ---------------------------------------------------------------------------------------------------------------
# Identity: the tokenizer's SHA-256 is content
# ---------------------------------------------------------------------------------------------------------------


class TestTokenizerIdentity:
    def test_the_tokenizers_sha256_is_the_content_identity(self, tmp_path: Any) -> None:
        import hashlib

        from tests._tokenizers import save, word_tokenizer

        directory = tmp_path / "tok"
        directory.mkdir()
        file = save(word_tokenizer(), directory)
        config = RerankEndpoint(base_url="http://a:8000/v1", model="m", tokenizer=str(file))

        assert config.identity_extra() == {"tokenizer_sha256": hashlib.sha256(file.read_bytes()).hexdigest()}

    def test_without_a_tokenizer_the_identity_is_empty(self) -> None:
        assert RerankEndpoint(base_url="http://a:8000/v1", model="m").identity_extra() == {}

    def test_the_name_stays_runtime(self) -> None:
        """The sha enters the identity; the tokenizer's name never does (as the judge's already works)."""
        from rcp_ndcg.support.identity import identity_payload

        config = RerankEndpoint(base_url="http://a:8000/v1", model="m", tokenizer="Qwen/Qwen3-Reranker-8B@abc")
        assert "tokenizer" not in identity_payload(config)


def test_the_adapter_seam_re_exports_the_family_and_its_base() -> None:
    """The third-party seam: the concrete wires and the base they share, one import away."""
    from rcp_ndcg.inference import adapters
    from rcp_ndcg.inference.adapters.rerank import RerankAdapter, RerankWire

    assert adapters.RerankWire is RerankWire
    assert issubclass(RerankAdapter, RerankWire)


def test_an_incomplete_rerank_wire_subclass_is_refused_at_construction() -> None:
    """A subclass that omits the wire facts fails with a typed error naming them, not an AttributeError at
    first use."""
    from rcp_ndcg.inference.adapters.rerank import RerankWire

    class _HalfWire(RerankWire):
        name = "half_wire"

    with pytest.raises(ConfigError, match="without its wire facts.*REQUEST_CAP"):
        _HalfWire(_config())


def test_an_adapter_of_another_role_is_refused_by_the_client() -> None:
    """A registered adapter of the wrong role (a judge's, say, selected by typo) is a config error, not a
    silent no-op: it is unknown in the rerank registry, and the error names where the name does live."""
    from rcp_ndcg.inference.adapters import register_adapter
    from rcp_ndcg.inference.adapters.base import AdapterRole

    class _JudgeShaped:
        name = "wrong_role_probe"
        role: ClassVar[AdapterRole] = "judge"

    register_adapter(_JudgeShaped)
    with pytest.raises(ConfigError, match="unknown rerank adapter 'wrong_role_probe'"):
        RerankClient(RerankEndpoint(api="wrong_role_probe", base_url="http://a:8000/v1", model="m"), sender=_server())


def test_a_client_tolerates_an_adapter_without_the_profile_facts() -> None:
    """A third-party rerank adapter that shape-matches only the Adapter protocol still constructs: no default
    base URL (the config must set one) and no pause."""
    from rcp_ndcg.inference.adapters import register_adapter
    from rcp_ndcg.inference.adapters.base import AdapterRole

    class _ThirdParty:
        name = "third_party_rerank"
        role: ClassVar[AdapterRole] = "rerank"

        def __init__(self, config: RerankEndpoint) -> None:
            self.config = config

        def calls(self, request: Any, *, model: str) -> list[Call]:
            return [Call("POST", "/rerank", {"model": model, "query": "q", "documents": ["a"], "top_n": 1})]

        def interpret(self, request: Any, replies: Sequence[Reply]) -> RerankResult:
            return RerankResult(scores=(0.5,))

        def usage(self, reply: Reply) -> None:
            return None

    register_adapter(_ThirdParty)
    server = _server()
    client = RerankClient(
        RerankEndpoint(api="third_party_rerank", base_url="http://a:8000/v1", model="m"), sender=server
    )

    assert client.rerank("q", ["a"]).scores == (0.5,)
    # The client read the profile facts defensively: no default base URL was invented, no pause applied.
    assert server.calls[0].json == {"model": "m", "query": "q", "documents": ["a"], "top_n": 1}
