"""The rerank wire adapters: the served Cohere-shaped wire and the hosted ``cohere`` and ``voyage`` profiles.

The request bodies and the answer parsing are the specification of what an engine or a hosted API receives
and returns: today's ``CohereRerank``/``VoyageRerank`` and served ``/rerank`` bodies, with the three answer
shapes (``results``, Voyage's ``data``, SGLang's bare list) realigned by ``index``.
"""

from __future__ import annotations

from typing import Any

import pytest
from rcp_ndcg_core.content import Content

from rcp_ndcg.errors import CapabilityError, ConfigError, ProviderError, RequestRejectedError
from rcp_ndcg.inference.adapters.rerank import CohereRerankAdapter, RerankAdapter, VoyageRerankAdapter
from rcp_ndcg.inference.config import RerankEndpoint
from rcp_ndcg.inference.types import Reply, RerankRequest, RerankResult, TokenCount
from tests.inference import _budget

# ---------------------------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------------------------


def _request(
    *documents: str,
    query: str = "q",
    instruction: str | None = None,
    content: Content | None = None,
) -> RerankRequest:
    """One rerank request over text documents (or one media ``content`` as the query)."""
    return RerankRequest(
        query=content if content is not None else Content.from_text(query),
        documents=tuple(Content.from_text(document) for document in documents),
        instruction=instruction,
    )


def _config(**kwargs: Any) -> RerankEndpoint:
    """A served config; hosted profiles pass ``api`` and drop ``base_url``. A served one declares its
    explicit budget (tokenizer + max_tokens) and ``use_activation`` (F10)."""
    kwargs.setdefault("base_url", "http://engine:8000/v1")
    kwargs.setdefault("model", "qwen3-reranker-8b")
    if kwargs.get("api", "rerank") == "rerank":
        kwargs.setdefault("tokenizer", _budget.DEFAULT_TOKENIZER or "test/tokenizer")
        kwargs.setdefault("max_tokens", 8192)
        kwargs.setdefault("use_activation", False)
    return RerankEndpoint(**kwargs)


def _reply(status: int, body: Any) -> Reply:
    return Reply(status, body, {})


def _scored_rows(count: int, *, key: str = "relevance_score", offset: int = 0) -> list[dict[str, Any]]:
    """One row per document, scored so that every document's score differs and arrival order is shuffled."""
    rows = [{"index": index + offset, key: ((index + offset) * 37 % 97) / 97} for index in range(count)]
    return rows[::-1]  # the server answers ranked, not in the request's order


# ---------------------------------------------------------------------------------------------------------------
# The served wire: request bodies
# ---------------------------------------------------------------------------------------------------------------


class TestServedBody:
    def test_a_request_is_the_cohere_shape_with_top_n_equal_to_the_documents(self) -> None:
        adapter = RerankAdapter(_config())
        (call,) = adapter.calls(_request("first", "second", "third"), model="qwen3-reranker-8b")

        assert call.method == "POST" and call.path == "/rerank"
        assert call.json == {
            "model": "qwen3-reranker-8b",
            "query": "q",
            "documents": ["first", "second", "third"],
            "top_n": 3,
            "use_activation": False,  # F10: a served config sets it explicitly, and it travels
        }

    def test_the_engine_extensions_are_absent_until_the_config_sets_them(self) -> None:
        """A plain Cohere-shaped server receives only the fields the config and request set: the
        instruction field waits for ``instruction: field``, and no engine-side truncation field is ever sent."""
        adapter = RerankAdapter(_config())
        call = adapter.calls(_request("a", "b"), model="m")[0]

        assert set(call.json) == {"model", "query", "documents", "top_n", "use_activation"}
        assert not {"instruction", "max_tokens_per_doc", "truncate_prompt_tokens"} & set(call.json)

    def test_the_instruction_field_travels_only_when_the_request_carries_one(self) -> None:
        adapter = RerankAdapter(_config(instruction="field"))
        (call,) = adapter.calls(_request("a", instruction="Find relevant passages"), model="m")

        assert call.json["query"] == "q"
        assert call.json["instruction"] == "Find relevant passages"

    def test_use_activation_travels_only_when_the_config_sets_it(self) -> None:
        """Raw logit or probability is a different stored score, so False is sent as False, never dropped."""
        (call,) = RerankAdapter(_config(use_activation=False)).calls(_request("a"), model="m")
        assert call.json["use_activation"] is False

        (call,) = RerankAdapter(_config(use_activation=True)).calls(_request("a"), model="m")
        assert call.json["use_activation"] is True

    def test_a_text_query_is_a_string_and_a_media_query_the_content_parts(self, tmp_path: Any) -> None:
        import hashlib

        from rcp_ndcg.data.media import content_parts_payload

        text = RerankAdapter(_config()).calls(_request("a"), model="m")[0].json
        assert isinstance(text["query"], str)

        page = tmp_path / "page1.png"
        page.write_bytes(b"png-bytes")
        parts = Content.from_parts(
            [
                Content.from_text("which page shows the revenue chart").parts[0],
                Content.from_image(page.as_uri(), sha256=hashlib.sha256(page.read_bytes()).hexdigest()).parts[0],
            ]
        )
        media = RerankAdapter(_config()).calls(_request("d", query="", content=parts), model="m")[0].json
        assert media["query"] == {"content": content_parts_payload(parts)}

    def test_a_whole_candidate_set_goes_in_one_request_without_a_cap(self) -> None:
        """The served engine reuses the query's prefix across the documents: no split."""
        adapter = RerankAdapter(_config())
        documents = [f"doc-{index}" for index in range(2500)]
        (call,) = adapter.calls(_request(*documents), model="m")

        assert len(call.json["documents"]) == 2500 and call.json["top_n"] == 2500

    def test_batch_size_asks_for_that_many_documents_per_request(self) -> None:
        adapter = RerankAdapter(_config(batch_size=2))
        calls = adapter.calls(_request("a", "b", "c", "d", "e"), model="m")

        assert [call.json["documents"] for call in calls] == [["a", "b"], ["c", "d"], ["e"]]
        assert [call.json["top_n"] for call in calls] == [2, 2, 1]

    def test_an_adapter_refuses_a_served_config_without_a_base_url_at_the_client(self) -> None:
        """The adapter itself builds bodies; the client refuses the missing ``base_url`` (its default is none)."""
        assert RerankAdapter.DEFAULT_BASE_URL is None


# ---------------------------------------------------------------------------------------------------------------
# The hosted profiles: request bodies
# ---------------------------------------------------------------------------------------------------------------


class TestHostedProfiles:
    def test_the_cohere_body_is_todays_shape(self) -> None:
        """``model``, ``query``, ``documents``, ``top_n`` -- the body today's ``CohereRerank`` sends."""
        adapter = CohereRerankAdapter(
            _config(api="cohere", base_url="https://api.cohere.com/v2", model="rerank-v4.0-fast")
        )
        (call,) = adapter.calls(_request("a", "", "b"), model="rerank-v4.0-fast")

        assert call.path == "/rerank"
        # An empty document is sent as it is and scores whatever the server returns; nothing is filtered.
        assert call.json == {
            "model": "rerank-v4.0-fast",
            "query": "q",
            "documents": ["a", "", "b"],
            "top_n": 3,
        }

    def test_the_cohere_profile_caps_at_1000_documents_and_merges_the_chunks(self) -> None:
        adapter = CohereRerankAdapter(_config(api="cohere", base_url="https://api.cohere.com/v2"))
        documents = [f"doc-{index}" for index in range(1001)]
        request = _request(*documents)
        calls = adapter.calls(request, model="rerank-v4.0-fast")

        assert [len(call.json["documents"]) for call in calls] == [1000, 1]
        assert [call.json["top_n"] for call in calls] == [1000, 1]
        assert calls[0].json["documents"][0] == "doc-0" and calls[1].json["documents"][0] == "doc-1000"

        replies = [_reply(200, {"results": _scored_rows(size)}) for size in (1000, 1)]
        result = adapter.interpret(request, replies)
        assert len(result.scores) == 1001
        assert result.scores[999] == (999 * 37 % 97) / 97  # chunk one's last document, realigned by index
        assert result.scores[1000] == (0 * 37 % 97) / 97  # chunk two's only document

    @pytest.mark.parametrize("adapter_cls", [CohereRerankAdapter, VoyageRerankAdapter])
    def test_the_declared_request_cap_is_1000_documents(self, adapter_cls: type[RerankAdapter]) -> None:
        """The hosted profiles' declared cap is a boundary: 1000 documents make one request, 1001 split
        into [1000, 1]. Both profiles declare the same cap; the test pins each profile's own attribute."""
        adapter = adapter_cls(_config(api=adapter_cls.name, base_url=adapter_cls.DEFAULT_BASE_URL))
        at_cap = adapter.calls(_request(*[f"doc-{index}" for index in range(1000)]), model="m")
        assert [len(call.json["documents"]) for call in at_cap] == [1000]
        over = adapter.calls(_request(*[f"doc-{index}" for index in range(1001)]), model="m")
        assert [len(call.json["documents"]) for call in over] == [1000, 1]

    def test_an_unreadable_body_is_truncated_at_300_characters(self) -> None:
        """``_short`` keeps 300 characters of a body it cannot read and marks the cut: the 301st
        character (the repr's closing quote, after 299 s) must not surface."""
        adapter = VoyageRerankAdapter(_config(api="voyage", base_url="https://api.voyageai.com/v1"))
        with pytest.raises(RequestRejectedError) as caught:
            adapter.interpret(_request("doc"), [_reply(503, "s" * 299)])  # repr: quote + 299 s + quote
        message = str(caught.value)
        assert "'" + "s" * 299 in message  # the repr's first 300 characters
        assert message.endswith("...")  # the 301st character (the closing quote) is gone

    def test_the_voyage_body_is_todays_shape(self) -> None:
        """``model``, ``query``, ``documents`` -- no ``top_n``: Voyage's return-limit field is ``top_k``, and
        it returns every document by default."""
        adapter = VoyageRerankAdapter(_config(api="voyage", base_url="https://api.voyageai.com/v1", model="rerank-2.5"))
        (call,) = adapter.calls(_request("a", "", "b"), model="rerank-2.5")

        assert call.path == "/rerank"
        assert call.json == {"model": "rerank-2.5", "query": "q", "documents": ["a", "", "b"]}

    def test_voyage_paces_its_requests_by_half_a_second(self) -> None:
        assert VoyageRerankAdapter.PAUSE_S == 0.5
        assert CohereRerankAdapter.PAUSE_S == 0.0
        assert RerankAdapter.PAUSE_S == 0.0

    def test_a_hosted_profile_refuses_an_instruction_field(self) -> None:
        with pytest.raises(ConfigError, match="no instruction field"):
            VoyageRerankAdapter(_config(api="voyage", instruction="field"))
        with pytest.raises(ConfigError, match="no instruction field"):
            CohereRerankAdapter(_config(api="cohere", instruction="field"))

    def test_a_hosted_profile_refuses_use_activation(self) -> None:
        with pytest.raises(ConfigError, match="use_activation"):
            CohereRerankAdapter(_config(api="cohere", use_activation=False))

    def test_a_listwise_config_refuses_to_split_an_oversized_candidate_set(self) -> None:
        """A listwise model scores the whole set in one prompt; splitting would change the scores."""
        adapter = CohereRerankAdapter(_config(api="cohere", listwise=True))
        request = _request(*[f"doc-{index}" for index in range(1001)])

        with pytest.raises(CapabilityError, match="listwise.*1001"):
            adapter.calls(request, model="m")

    def test_a_listwise_config_at_or_under_the_cap_makes_no_split(self) -> None:
        adapter = VoyageRerankAdapter(_config(api="voyage", listwise=True))
        (call,) = adapter.calls(_request(*[f"doc-{index}" for index in range(1000)]), model="m")

        assert len(call.json["documents"]) == 1000


# ---------------------------------------------------------------------------------------------------------------
# The three answer shapes, realigned by index
# ---------------------------------------------------------------------------------------------------------------


class TestAnswerShapes:
    def test_results_rows_come_back_ranked_and_are_realigned_by_index(self) -> None:
        """Reading the ranked rows positionally would attach each score to the wrong document."""
        adapter = RerankAdapter(_config())
        request = _request("d0", "d1", "d2")
        reply = _reply(
            200,
            {
                "results": [
                    {"index": 2, "relevance_score": 0.1},
                    {"index": 0, "relevance_score": 0.9},
                    {"index": 1, "relevance_score": 0.5},
                ]
            },
        )

        result = adapter.interpret(request, [reply])

        assert isinstance(result, RerankResult)
        assert result.scores == (0.9, 0.5, 0.1)

    def test_voyages_data_rows_are_read_the_same_way(self) -> None:
        adapter = VoyageRerankAdapter(_config(api="voyage"))
        reply = _reply(200, {"data": [{"index": 1, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.2}]})

        assert adapter.interpret(_request("a", "b"), [reply]).scores == (0.2, 0.9)

    def test_sglangs_bare_list_is_read_with_the_score_key(self) -> None:
        adapter = RerankAdapter(_config())
        reply = _reply(200, [{"index": 1, "score": 0.9}, {"index": 0, "score": 0.2}])

        assert adapter.interpret(_request("a", "b"), [reply]).scores == (0.2, 0.9)

    def test_scores_pass_through_untransformed(self) -> None:
        """The scores are the server's, whatever scale it uses (a raw logit with ``use_activation: false``)."""
        adapter = RerankAdapter(_config(use_activation=False))
        reply = _reply(
            200, {"results": [{"index": 0, "relevance_score": -3.75}, {"index": 1, "relevance_score": 12.5}]}
        )

        assert adapter.interpret(_request("a", "b"), [reply]).scores == (-3.75, 12.5)

    def test_chunked_replies_merge_in_document_order(self) -> None:
        adapter = RerankAdapter(_config(batch_size=2))
        request = _request("d0", "d1", "d2")
        replies = [
            _reply(200, {"results": [{"index": 1, "relevance_score": 0.3}, {"index": 0, "relevance_score": 0.4}]}),
            _reply(200, {"results": [{"index": 0, "relevance_score": 0.9}]}),
        ]

        assert adapter.interpret(request, replies).scores == (0.4, 0.3, 0.9)


# ---------------------------------------------------------------------------------------------------------------
# Refusals and unusable answers
# ---------------------------------------------------------------------------------------------------------------


class TestUnusableAnswers:
    def test_a_missing_score_is_refused_not_made_up(self) -> None:
        adapter = RerankAdapter(_config())
        reply = _reply(200, {"results": [{"index": 0, "relevance_score": 0.9}]})  # index 1 never answered

        with pytest.raises(ProviderError, match="no score for 1 of 2 documents") as caught:
            adapter.interpret(_request("a", "b"), [reply])
        assert caught.value.retryable is False

    def test_a_duplicated_index_is_refused(self) -> None:
        adapter = RerankAdapter(_config())
        reply = _reply(200, {"results": [{"index": 0, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.8}]})

        with pytest.raises(ProviderError, match="index 0 twice"):
            adapter.interpret(_request("a", "b"), [reply])

    def test_an_out_of_range_index_is_refused(self) -> None:
        adapter = RerankAdapter(_config())
        reply = _reply(200, {"results": [{"index": 2, "relevance_score": 0.9}]})

        with pytest.raises(ProviderError, match="index 2 for a request of 2 documents"):
            adapter.interpret(_request("a", "b"), [reply])

    def test_a_row_without_an_integer_index_is_refused(self) -> None:
        adapter = RerankAdapter(_config())
        reply = _reply(200, {"results": [{"relevance_score": 0.9}]})

        with pytest.raises(ProviderError, match="without an integer index"):
            adapter.interpret(_request("a", "b"), [reply])

    def test_a_row_without_a_score_is_refused(self) -> None:
        adapter = RerankAdapter(_config())
        reply = _reply(200, {"results": [{"index": 0}]})

        with pytest.raises(ProviderError, match="without a numeric relevance_score"):
            adapter.interpret(_request("a", "b"), [reply])

    def test_an_unrecognised_body_is_refused(self) -> None:
        adapter = RerankAdapter(_config())

        with pytest.raises(ProviderError, match="cannot read"):
            adapter.interpret(_request("a", "b"), [_reply(200, {"score": []})])

    def test_a_row_that_is_not_an_object_is_refused(self) -> None:
        adapter = RerankAdapter(_config())

        with pytest.raises(ProviderError, match="not an object"):
            adapter.interpret(_request("a", "b"), [_reply(200, ["nope"])])

    def test_an_over_length_refusal_is_a_capability_error_hinting_max_tokens(self) -> None:
        adapter = RerankAdapter(_config())
        reply = _reply(400, {"error": {"message": "This model's maximum context length is 8192 tokens"}})

        with pytest.raises(CapabilityError, match="too long") as caught:
            adapter.interpret(_request("a", "b"), [reply])
        assert "max_tokens" in (caught.value.hint or "")

    def test_another_refusal_is_a_request_rejection(self) -> None:
        adapter = RerankAdapter(_config())
        reply = _reply(422, {"detail": "unknown field use_activation"})

        with pytest.raises(RequestRejectedError, match="refused the rerank request") as caught:
            adapter.interpret(_request("a", "b"), [reply])
        assert caught.value.retryable is False

    def test_a_reply_count_that_does_not_match_the_chunks_is_refused(self) -> None:
        adapter = CohereRerankAdapter(_config(api="cohere"))
        request = _request(*[f"doc-{index}" for index in range(1001)])

        with pytest.raises(ProviderError, match="0 reply/replies for the 2 request"):
            adapter.interpret(request, [])

    def test_errors_name_the_server(self) -> None:
        adapter = VoyageRerankAdapter(_config(api="voyage", base_url="https://api.voyageai.com/v1"))
        with pytest.raises(ProviderError, match="Voyage"):
            adapter.interpret(_request("a", "b"), [_reply(200, {"results": []})])


# ---------------------------------------------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------------------------------------------


class TestUsage:
    def test_openai_style_usage_is_read(self) -> None:
        adapter = RerankAdapter(_config())
        reply = _reply(200, {"results": [], "usage": {"prompt_tokens": 5, "completion_tokens": 0}})
        assert adapter.usage(reply) == TokenCount(5, 0)

    def test_a_body_without_tokens_reports_none(self) -> None:
        adapter = RerankAdapter(_config())
        assert adapter.usage(_reply(200, {"results": []})) is None
        assert adapter.usage(_reply(200, {"results": [], "usage": {"total_tokens": 9}})) is None

    def test_a_boolean_token_count_is_no_count(self) -> None:
        """JSON booleans are ints to ``isinstance``; a non-conforming body reports no tokens."""
        adapter = RerankAdapter(_config())
        reply = _reply(200, {"results": [], "usage": {"prompt_tokens": True, "completion_tokens": False}})

        assert adapter.usage(reply) is None


# ---------------------------------------------------------------------------------------------------------------
# The registry: the three adapters are the role's shipped names
# ---------------------------------------------------------------------------------------------------------------


def test_the_rerank_adapters_register_under_their_api_names() -> None:
    from rcp_ndcg.inference.adapters import get_adapter, known_adapters

    assert {"rerank", "cohere", "voyage"} <= set(known_adapters("rerank"))
    for name, adapter in (
        ("rerank", RerankAdapter),
        ("cohere", CohereRerankAdapter),
        ("voyage", VoyageRerankAdapter),
    ):
        assert get_adapter(name, role="rerank") is adapter


def test_an_empty_candidate_set_makes_no_call() -> None:
    """There is no score to ask for: no request, an empty result."""
    adapter = RerankAdapter(_config())

    assert adapter.calls(_request(), model="m") == []
    assert adapter.interpret(_request(), []) == RerankResult(scores=())


class TestTheServerIsNamedWithoutItsSecrets:
    """Every rerank error names the server by its URL: as :func:`~rcp_ndcg.support.urls.safe_url` writes it,
    in the message and in the details -- credentials embedded in the URL never ride along."""

    SECRET_URL = "https://user:fake-secret-pw@gw.example/v1?key=fake-secret-q"

    def test_a_missing_score_names_the_server_safely(self) -> None:
        adapter = RerankAdapter(_config(base_url=self.SECRET_URL))
        request = _request("d0", "d1")
        with pytest.raises(ProviderError) as caught:
            adapter.interpret(request, [_reply(200, {"results": _scored_rows(1)})])
        said = f"{caught.value} {caught.value.details} {caught.value.hint}"
        assert "gw.example" in said
        assert "fake-secret" not in said

    def test_a_duplicated_index_names_the_server_safely(self) -> None:
        adapter = RerankAdapter(_config(base_url=self.SECRET_URL))
        request = _request("d0", "d1")
        rows = [{"index": 0, "relevance_score": 0.1}, {"index": 0, "relevance_score": 0.2}]
        with pytest.raises(ProviderError) as caught:
            adapter.interpret(request, [_reply(200, {"results": rows})])
        said = f"{caught.value} {caught.value.details} {caught.value.hint}"
        assert "fake-secret" not in said
