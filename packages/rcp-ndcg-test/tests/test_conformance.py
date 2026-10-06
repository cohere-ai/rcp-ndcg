"""The conformance runner, one role and one target at a time.

The packaged fixture recipe (``fake-embed``) and the shipped fake cover the embed path end to end; the
tests' fixture recipes and fakes cover the rerank and the multi-vector (MaxSim) paths; a real HTTP
socket proves the engine target is the same code path with the network in it.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from rcp_ndcg_test.cases import CaseTolerance, load_case, load_cases
from rcp_ndcg_test.conformance import CaseResult, ConformanceReport, run_case, run_suite
from rcp_ndcg_test.errors import CaseError, ConformanceError
from rcp_ndcg_test.fakes import FakeEmbedEngine, FakeReply, fixture_path
from rcp_ndcg_vllm.recipe import load_recipe

from tests.fakes_fixture import FakePoolEngine, FakeRerankEngine

PACKAGED = fixture_path("cases")
PACKAGED_RECIPES = fixture_path("recipes")
RECIPES = Path(__file__).resolve().parent / "fixtures" / "recipes"
CASES = Path(__file__).resolve().parent / "fixtures" / "cases"


def packaged_recipe() -> object:
    return load_recipe(PACKAGED_RECIPES / "fake-embed")


def packaged_cases() -> tuple:
    return load_cases(PACKAGED, recipes_root=PACKAGED_RECIPES).cases


def pending_case():
    """The packaged case whose values are still pending (the GPU wave fills it)."""
    return next(case for case in packaged_cases() if case.expected.values is None)


def filled_case():
    """The packaged case whose values are recorded."""
    return next(case for case in packaged_cases() if case.expected.values is not None)


# ---------------------------------------------------------------------------
# The fake target: all three roles, end to end
# ---------------------------------------------------------------------------


def test_the_packaged_suite_passes_against_the_shipped_fake() -> None:
    report = run_suite(packaged_recipe(), packaged_cases(), target="fake")
    assert isinstance(report, ConformanceReport)
    assert report.ok
    assert len(report.results) == 4
    assert {result.case_id.split("/")[-1] for result in report.skipped} == {"long-under"}
    for result in report.results:
        if result.passed and result.compared:
            assert result.detail is not None and "max abs delta" in result.detail
    assert report.base_url == "http://rcp-ndcg-test.fake/v1"
    assert "4 case(s)" in report.summary() and "ok" in report.summary()


def test_a_wrong_expected_fails_with_the_worst_delta() -> None:
    case = filled_case()
    tolerance = case.expected.tolerance
    assert tolerance is not None and tolerance.abs is not None
    wrong_values = [[value + 0.5 for value in row] for row in case.expected.values]
    wrong = case.model_copy(update={"expected": case.expected.model_copy(update={"values": wrong_values})})
    result = run_case(packaged_recipe(), wrong, target="fake")
    assert result.failed
    assert result.skipped is None and result.compared
    assert "exceeds abs" in (result.detail or "")
    assert "[query q1, document d" in (result.detail or "")


def test_a_rank_exact_breach_names_the_derived_order() -> None:
    recipe = load_recipe(RECIPES / "fake-rerank")
    case = load_case(CASES / "fake-rerank" / "short-ranking.yaml")
    assert case.expected.kind == "ranking" and case.expected.values is not None
    swapped = [[*reversed(row)] for row in case.expected.values]
    wrong = case.model_copy(update={"expected": case.expected.model_copy(update={"values": swapped})})
    result = run_case(recipe, wrong, target="fake", fake_engine=FakeRerankEngine())
    assert result.failed and "rank_exact" in (result.detail or "")


def test_a_spearman_tolerance_passes_the_order_and_fails_the_reversal() -> None:
    recipe = load_recipe(RECIPES / "fake-rerank")
    case = load_case(CASES / "fake-rerank" / "short-single.yaml")

    def with_tolerance(value: float):
        return case.model_copy(
            update={"expected": case.expected.model_copy(update={"tolerance": CaseTolerance(spearman_min=value)})}
        )

    assert run_case(recipe, with_tolerance(-1.0), target="fake", fake_engine=FakeRerankEngine()).passed
    assert run_case(recipe, with_tolerance(0.99), target="fake", fake_engine=FakeRerankEngine()).passed
    spearman_case = with_tolerance(0.99)
    reversed_values = [list(reversed(row)) for row in case.expected.values]
    reversed_case = spearman_case.model_copy(
        update={"expected": spearman_case.expected.model_copy(update={"values": reversed_values})}
    )
    failing = run_case(recipe, reversed_case, target="fake", fake_engine=FakeRerankEngine())
    assert failing.failed and "spearman" in (failing.detail or "")


def test_the_rerank_fixture_runs_end_to_end() -> None:
    recipe = load_recipe(RECIPES / "fake-rerank")
    bundle = load_cases(CASES, recipe, recipes_root=RECIPES)
    report = run_suite(recipe, bundle.cases, target="fake", fake_engine=FakeRerankEngine())
    assert report.ok, report.summary()
    assert len(report.results) == 5
    assert {result.case_id.split("/")[-1] for result in report.skipped} == {"long-under"}


def test_the_multi_vector_fixture_scores_maxsim_end_to_end() -> None:
    recipe = load_recipe(RECIPES / "fake-pool")
    bundle = load_cases(CASES, recipe, recipes_root=RECIPES)
    report = run_suite(recipe, bundle.cases, target="fake", fake_engine=FakePoolEngine())
    assert report.ok
    assert {result.case_id.split("/")[-1] for result in report.skipped} == {"long-under"}
    compared = [result for result in report.results if result.compared]
    assert compared and all("max abs delta" in (result.detail or "") for result in compared)


# ---------------------------------------------------------------------------
# The engine target: the same path over a real HTTP socket
# ---------------------------------------------------------------------------


class _HTTPFake:
    """The shipped fake served over a real ephemeral HTTP socket (the engine target's transport)."""

    def __init__(self, engine: FakeEmbedEngine) -> None:

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802  # stdlib name
                self._answer({})

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("content-length", 0))
                body = json.loads(self.rfile.read(length)) if length else {}
                self._answer(body)

            def _answer(self, body: dict) -> None:
                reply = engine.handle(self.command, self.path, body)
                payload = json.dumps(reply.json_body).encode() if reply.json_body is not None else b""
                self.send_response(reply.status)
                self.send_header("content-type", reply.content_type)
                self.send_header("content-length", str(len(payload)))
                self.end_headers()
                if payload:
                    self.wfile.write(payload)

            def log_message(self, format: str, *args: object) -> None:  # noqa: A002
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}/v1"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def test_the_engine_target_runs_the_same_suite_over_http() -> None:
    engine = _HTTPFake(FakeEmbedEngine())
    try:
        fake_report = run_suite(packaged_recipe(), packaged_cases(), target="fake")
        engine_report = run_suite(packaged_recipe(), packaged_cases(), target="engine", base_url=engine.base_url)
    finally:
        engine.stop()
    assert engine_report.ok and fake_report.ok
    assert [(r.case_id, r.passed, r.skipped is not None) for r in engine_report.results] == [
        (r.case_id, r.passed, r.skipped is not None) for r in fake_report.results
    ]
    assert engine_report.base_url.startswith("http://127.0.0.1:")


# ---------------------------------------------------------------------------
# The target rules and the error paths
# ---------------------------------------------------------------------------


def test_the_engine_target_needs_a_base_url() -> None:
    with pytest.raises(ConformanceError, match="needs base_url"):
        run_suite(packaged_recipe(), packaged_cases(), target="engine")


def test_the_fake_target_refuses_a_base_url() -> None:
    with pytest.raises(ConformanceError, match="not a base_url"):
        run_suite(packaged_recipe(), packaged_cases(), target="fake", base_url="http://127.0.0.1:8000/v1")


def test_the_engine_target_refuses_a_fake_engine() -> None:
    with pytest.raises(ConformanceError, match="takes a base_url"):
        run_suite(packaged_recipe(), packaged_cases(), target="engine", base_url="http://x/v1", fake_engine=object())


def test_an_unregistered_fake_is_refused_with_a_hint() -> None:
    from rcp_ndcg_test import fakes

    fakes.unregister_fake_engine("fake-embed")
    try:
        with pytest.raises(ConformanceError, match="register_fake_engine"):
            run_suite(packaged_recipe(), packaged_cases(), target="fake")
    finally:
        fakes.register_fake_engine(fakes.FakeEmbedEngine())
        assert "fake-embed" in fakes.registered_fake_engines()


def test_a_product_error_is_a_failure_with_the_products_message() -> None:
    class BrokenEngine:
        recipe_id = "fake-embed"
        name = "broken"

        def handle(self, method: str, path: str, body: object) -> FakeReply:
            return FakeReply(status=404, json_body={"error": {"message": f"no route {path}"}})

    result = run_case(packaged_recipe(), filled_case(), target="fake", fake_engine=BrokenEngine())
    assert result.failed and result.skipped is None and not result.compared
    assert "ProviderError" in (result.detail or "")


def test_kind_none_exercises_the_path_without_comparing() -> None:
    case = filled_case()
    none_case = case.model_copy(
        update={"expected": case.expected.model_copy(update={"kind": "none", "values": None, "tolerance": None})}
    )
    result = run_case(packaged_recipe(), none_case, target="fake")
    assert result.passed and not result.compared and result.skipped is None
    assert result.detail is not None and "exercises the path" in result.detail


def test_a_recipe_with_side_prompts_is_refused() -> None:
    recipe = packaged_recipe()
    prompted = recipe.model_copy(update={"client": recipe.client.model_copy(update={"query_prompt": "Q: "})})
    with pytest.raises(ConformanceError, match="query_prompt/doc_prompt"):
        run_suite(prompted, packaged_cases(), target="fake")


def test_a_pending_case_skips_and_never_sends() -> None:
    calls: list[str] = []

    class CountingEngine(FakeEmbedEngine):
        def handle(self, method: str, path: str, body: object):
            calls.append(path)
            return super().handle(method, path, body)

    result = run_case(packaged_recipe(), pending_case(), target="fake", fake_engine=CountingEngine())
    assert result.skipped is not None and "values is null" in result.skipped
    assert result.passed is False and result.compared is False
    assert calls == [], "a pending case must not reach the engine"


def test_the_report_verdicts() -> None:
    report = run_suite(packaged_recipe(), packaged_cases(), target="fake")
    assert not report.failures
    assert len(report.skipped) == 1
    assert isinstance(report.results[0], CaseResult)
    assert report.ok


# ---------------------------------------------------------------------------
# The fold parity: the runner's fit input is the client's own fold
# ---------------------------------------------------------------------------


class _SpySender:
    """A sender that records the wire bodies and answers one fixed rerank reply per call."""

    def __init__(self) -> None:
        self.recorded: list[dict] = []

    async def send(self, calls):
        from rcp_ndcg.inference.types import Reply

        for call in calls:
            assert call.json is not None
            self.recorded.append(call.json)
            documents = len(call.json["documents"])
            return [
                Reply(
                    status=200,
                    body={"results": [{"index": index, "relevance_score": float(index)} for index in range(documents)]},
                    headers={},
                )
            ]
        raise AssertionError("the spy was called with no calls")

    async def probe(self):
        from rcp_ndcg.inference.types import EngineInfo

        return [EngineInfo(url="spy://")]

    @property
    def usage(self):
        from rcp_ndcg.inference.types import Usage

        return Usage()


def test_the_rerank_fit_measures_the_clients_own_fold() -> None:
    from rcp_ndcg_test.conformance import _fold_query

    from rcp_ndcg.inference.clients import RerankClient

    recipe = load_recipe(RECIPES / "fake-rerank")
    query, instruction = "who wrote the republic", "Find the document that answers the question."
    spy = _SpySender()
    client = RerankClient(
        recipe.client.model_copy(update={"max_tokens": None, "query_max_tokens": None, "base_url": "http://spy/v1"}),
        sender=spy,
    )
    client.rerank(query, ["plato wrote the republic"], instruction=instruction)
    [body] = spy.recorded
    assert body["query"] == _fold_query(recipe, query, instruction)
    assert body["query"].startswith("Task: ")


def test_a_malformed_case_fails_the_plugins_collection(tmp_path: Path) -> None:
    from rcp_ndcg_test.plugin import conformance_params

    (tmp_path / "fake-embed").mkdir()
    (tmp_path / "fake-embed" / "bad.yaml").write_text("not: a: case\n", encoding="utf-8")
    with pytest.raises(CaseError):
        conformance_params("fake", cases_root=tmp_path, recipes_root=PACKAGED_RECIPES)
