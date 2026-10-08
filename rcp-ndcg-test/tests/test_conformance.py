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
from rcp_ndcg_test.conformance import ConformanceReport, run_case, run_suite
from rcp_ndcg_test.errors import CaseError, ConformanceError
from rcp_ndcg_test.fakes import (
    FakeEmbedEngine,
    FakeReply,
    fixture_path,
    register_fake_engine,
    unregister_fake_engine,
)
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


def test_the_run_refuses_a_case_whose_recipe_disagrees() -> None:
    """(round-2 F4) ``run_case(recipe, case)`` refuses a case belonging to a different recipe: a run
    under the wrong scorer is an identity lie that can pass as a numbers check."""
    from rcp_ndcg_test.errors import ConformanceError

    recipe = load_recipe(RECIPES / "fake-pool")
    case = load_case(PACKAGED / "fake-embed" / "short-single.yaml")
    assert case.recipe == "fake-embed"
    with pytest.raises(ConformanceError, match="belongs to recipe"):
        run_case(recipe, case, target="fake", fake_engine=FakePoolEngine())


def test_a_spearman_prefix_expectation_is_not_a_free_pass_when_displaced() -> None:
    """(v1-F5) A ``kind: ranking`` row under ``spearman_min`` names the derived **top-k prefix** (its
    documents must hold the top-k positions exactly); correlating absolute ranks is shift-invariant
    and would pass a displaced expectation free -- the case format promises a prefix, never a
    placement-free shuffle. A one-document row must never pass free either.

    The comparator (``_compare_ranking``) is the unit here: the matrix is crafted so the derived
    ranking is [d3, d2, d1] while the case expects the top-2 prefix [d1, d2] -- d1/d2 sit at the
    bottom two, so the expectation must FAIL.
    """
    from rcp_ndcg_test.cases import CaseExpected, CaseTolerance
    from rcp_ndcg_test.conformance import _compare_ranking

    case = load_case(CASES / "fake-rerank" / "short-ranking.yaml")
    ids = [document.id for document in case.inputs.documents]
    assert len(ids) == 3, ids  # three documents: the expected top-2 can be displaced below a third
    displaced = case.model_copy(
        update={
            "expected": CaseExpected(
                kind="ranking",
                values=[[ids[0], ids[1]]],  # claims d1 d2 are the derived top-2; the matrix below disagrees
                tolerance=CaseTolerance(spearman_min=0.5),
                origin="engine",
                status="reproduced",
            )
        }
    )
    matrix = [[0.5, 0.1, 1.0]]  # derived: [d3, d1, d2] -- the expected top-2 shifted one slot down
    result = _compare_ranking(displaced, matrix, displaced.expected.tolerance)
    assert result.passed is False, result.detail
    assert "prefix" in (result.detail or ""), result.detail

    # a one-document expectation is refused (the load forbids it; an unvalidated case must not pass free)
    one_row = case.model_copy(
        update={
            "expected": CaseExpected(
                kind="ranking",
                values=[["d1"]],
                tolerance=CaseTolerance(spearman_min=0.5),
                origin="engine",
                status="reproduced",
            )
        }
    )
    one = _compare_ranking(one_row, matrix, one_row.expected.tolerance)
    assert one.passed is False, one.detail


def test_an_image_type_outside_the_products_mime_table_is_refused_not_silently_defaulted(
    tmp_path: Path,
) -> None:
    """(v2-F3) A media file whose type the product's media table does not know is refused at the
    send-preparation: a bare MediaRef (no mime, no dimensions) would silently inflate the media token
    count at the policy's pixel ceiling -- a silent default this package forbids. The tests write to
    ``tmp_path`` only, so the case tree is copied there with the odd-named file beside its media."""
    import shutil

    shutil.copytree(CASES, tmp_path / "cases")
    media = tmp_path / "cases" / "fake-pool" / "media"
    (media / "pixel.xyz").write_bytes(_tiny_png())  # bytes are never parsed blindly (checks the type gate)
    case = load_case(tmp_path / "cases" / "fake-pool" / "image-pixel.yaml")
    document = case.inputs.documents[0].model_copy(update={"image": "media/pixel.xyz"})
    mutated = case.model_copy(update={"inputs": case.inputs.model_copy(update={"documents": [document]})})
    recipe = load_recipe(RECIPES / "fake-pool")
    result = run_case(recipe, mutated, target="fake", fake_engine=FakePoolEngine())
    assert result.failed and "unknown image type" in (result.detail or ""), (result.skipped, result.detail)


def test_the_rerank_fixture_runs_end_to_end() -> None:
    recipe = load_recipe(RECIPES / "fake-rerank")
    bundle = load_cases(CASES, recipe, recipes_root=RECIPES)
    report = run_suite(recipe, bundle.cases, target="fake", fake_engine=FakeRerankEngine())
    assert report.ok, report.summary()
    assert len(report.results) == 7
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


def test_a_chunking_recipe_runs_and_pools_through_the_client() -> None:
    """The wired client pools a chunked document's scores by max: the case runs to a real comparison
    (a regression of the bridge era, when the runner sent chunks as if they were documents)."""
    from rcp_ndcg.data.preprocess import ChunkPolicy

    recipe = load_recipe(RECIPES / "fake-rerank")
    chunked = recipe.model_copy(
        update={
            "client": recipe.client.model_copy(
                update={"on_overflow": "chunk", "chunk": ChunkPolicy(max_tokens=48, overlap_tokens=8)}
            )
        }
    )
    cases = load_cases(CASES, chunked, recipes_root=RECIPES, check_lengths=False).cases
    case = next(c for c in cases if c.id.endswith("short-single"))
    result = run_case(chunked, case, target="fake", fake_engine=FakeRerankEngine())
    assert result.compared, (result.skipped, result.detail)
    # the pooled verdict: the recorded values (recorded on the cut render) are not the pooled scores, so
    # the comparison may fail -- but it is a comparison of the case's own documents, never a shape error
    assert result.skipped is None
    assert "matrix for" not in (result.detail or "")


def test_a_recipe_with_side_prompts_runs_through_the_client() -> None:
    """The wired embed client prepends the per-side prompts before its fit: a prompt-carrying recipe runs."""
    recipe = load_recipe(PACKAGED_RECIPES / "fake-embed")
    prompted = recipe.model_copy(update={"client": recipe.client.model_copy(update={"query_prompt": "Q: "})})
    case = load_case(PACKAGED / "fake-embed" / "short-single.yaml")

    recorded: list[list[str]] = []

    class RecordingEmbedEngine(FakeEmbedEngine):
        def handle(self, method: str, path: str, body: object):
            if method == "POST" and path.endswith("/embeddings"):
                recorded.append(list(body["input"]))
            return super().handle(method, path, body)

    run_case(prompted, case, target="fake", fake_engine=RecordingEmbedEngine())
    assert recorded and all(isinstance(text, str) and text.startswith("query: Q: ") for text in recorded[0])


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


def test_the_rerank_client_cuts_a_long_query_to_its_declared_share() -> None:
    """A rerank case with a declared query_max_tokens and a long query: the client cuts the query span
    on the wire (the runner pre-fits nothing)."""
    from rcp_ndcg_core._records import Query
    from rcp_ndcg_test.cases import CaseDocument, CaseInputs, CaseQuery
    from rcp_ndcg_test.cases import _recipe_fitter as recipe_fitter

    from rcp_ndcg.data.preprocess import fit

    recipe = load_recipe(RECIPES / "fake-rerank")
    assert recipe.client.query_max_tokens is not None
    long_query = " ".join(["overrun"] * 30)  # the fold measures far over the 48-token query share
    case = load_case(CASES / "fake-rerank" / "short-single.yaml").model_copy(
        update={
            "inputs": CaseInputs(
                instruction="Find the document that answers the question.",
                queries=[CaseQuery(id="q1", text=long_query)],
                documents=[CaseDocument(id="d1", text="plato wrote the republic around 375 bc")],
            )
        }
    )
    raw_fold = str(Query(query_id="", query=long_query, instruction=case.inputs.instruction).format_query())

    recorded: list[dict] = []

    class RecordingRerankEngine(FakeRerankEngine):
        def handle(self, method: str, path: str, body: object):
            if method == "POST" and path.endswith("/rerank"):
                recorded.append(body)
            return super().handle(method, path, body)

    run_case(recipe, case, target="fake", fake_engine=RecordingRerankEngine())
    [body] = recorded
    assert body["query"] != raw_fold, "the wire must carry the client's cut query, not the raw fold"
    # the client's own fit: the folded query cut to its declared share (settled once for the batch)
    tokenizer, budget = recipe_fitter(recipe)
    folded = str(Query(query_id="", query=long_query, instruction=case.inputs.instruction).format_query())
    fitted = fit([(folded, "x")], "pair", budget, tokenizer, ids=["d"], instruction=case.inputs.instruction)
    cut_query = fitted.contents[0][0]
    assert body["query"] == cut_query and len(body["query"]) < len(raw_fold)


def test_the_rerank_client_folds_the_instruction_itself() -> None:
    """The raw query and the instruction go in; the client's own fold renders the wire query.

    The probe query's fold fits the declared share, so the budget binds on overflow only and the wire
    query is the fold, uncut."""
    from rcp_ndcg_core._records import Query
    from rcp_ndcg_test.cases import CaseQuery

    recipe = load_recipe(RECIPES / "fake-rerank")
    query, instruction = "a query", "Find the document that answers the question."
    case = load_case(CASES / "fake-rerank" / "short-single.yaml")
    inputs = case.inputs.model_copy(update={"queries": [CaseQuery(id="q1", text=query)]})
    case = case.model_copy(update={"inputs": inputs})
    recorded: list[dict] = []

    class RecordingRerankEngine(FakeRerankEngine):
        def handle(self, method: str, path: str, body: object):
            if method == "POST" and path.endswith("/rerank"):
                recorded.append(body)
            return super().handle(method, path, body)

    run_case(recipe, case, target="fake", fake_engine=RecordingRerankEngine())
    [body] = recorded
    expected = str(Query(query_id="", query=query, instruction=instruction).format_query())
    assert body["query"] == expected and body["query"].startswith("Task: ")


def test_a_malformed_case_fails_the_plugins_collection(tmp_path: Path) -> None:
    from rcp_ndcg_test.plugin import conformance_params

    (tmp_path / "fake-embed").mkdir()
    (tmp_path / "fake-embed" / "bad.yaml").write_text("not: a: case\n", encoding="utf-8")
    with pytest.raises(CaseError):
        conformance_params("fake", cases_root=tmp_path, recipes_root=PACKAGED_RECIPES)


# ---------------------------------------------------------------------------
# Media: an image case runs the pool and rerank routes (content parts on the wire); video media
# cases and the embed route are declared skips (checked before every send), never a silent empty send
# ---------------------------------------------------------------------------


def test_a_kind_none_case_with_media_skips_too() -> None:
    """A path-exercise case is a skip, never a silent empty-text send, when it carries media."""
    from rcp_ndcg_test.cases import CaseDocument

    case = filled_case()
    documents = list(case.inputs.documents)
    documents[0] = CaseDocument(id=documents[0].id, image="media/pixel.png")
    inputs = case.inputs.model_copy(update={"documents": documents})
    media_case = case.model_copy(
        update={
            "inputs": inputs,
            "expected": case.expected.model_copy(update={"kind": "none", "values": None, "tolerance": None}),
        }
    )
    calls: list[str] = []

    class CountingEngine(FakeEmbedEngine):
        def handle(self, method: str, path: str, body: object):
            calls.append(path)
            return super().handle(method, path, body)

    result = run_case(packaged_recipe(), media_case, target="fake", fake_engine=CountingEngine())
    assert result.skipped is not None and "text only" in result.skipped
    assert not result.passed and not result.compared
    assert calls == [], "a media case must not reach the engine on the kind-none path either"


def _tiny_png() -> bytes:
    """A 1x1 red PNG the wired client can inline (PIL-decodable)."""
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (1, 1), (255, 0, 0)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_an_image_case_runs_on_the_rerank_and_pool_routes(tmp_path: Path) -> None:
    """A media case RUNS on the pool and rerank routes: the wired clients prepare the media and the
    image reaches the wire as image_url content parts (the bridge-era skip is gone).

    The media document carries text beside its image: the pool fixture declares
    ``empty_doc: omit_zero``, which omits an empty-text document outright (the product's own rule) --
    a caption keeps the media document in the request where its image parts can be seen on the wire.
    """
    import shutil

    # the fixture recipe + case sets, copied into tmp so the cases' media/ dirs are writable (tests
    # never write into the checkout)
    shutil.copytree(RECIPES, tmp_path / "recipes")
    (tmp_path / "tokenizer.json").write_bytes(
        (Path(__file__).resolve().parent / "fixtures" / "tokenizer.json").read_bytes()
    )
    shutil.copytree(CASES, tmp_path / "cases")
    for name in ("fake-rerank", "fake-pool"):
        (tmp_path / "cases" / name / "media").mkdir(exist_ok=True)
        (tmp_path / "cases" / name / "media" / "pixel.png").write_bytes(_tiny_png())

    from rcp_ndcg_test.cases import CaseDocument

    def text_case(bundle):
        return next(case for case in bundle.cases if case.id.endswith("short-single"))

    def with_image(case):
        documents = list(case.inputs.documents)
        documents[0] = CaseDocument(id=documents[0].id, text="the page above the image", image="media/pixel.png")
        return case.model_copy(update={"inputs": case.inputs.model_copy(update={"documents": documents})})

    # the rerank route: one /rerank request carries the query and the media document's content parts
    recorded: list[dict] = []

    class RecordingRerankEngine(FakeRerankEngine):
        def handle(self, method: str, path: str, body: object):
            if method == "POST" and path.endswith("/rerank"):
                recorded.append(body)
            return super().handle(method, path, body)

    recipe = load_recipe(tmp_path / "recipes" / "fake-rerank")
    bundle = load_cases(tmp_path / "cases", recipe, recipes_root=tmp_path / "recipes", check_lengths=False)
    result = run_case(recipe, with_image(text_case(bundle)), target="fake", fake_engine=RecordingRerankEngine())
    assert recorded, "the send must reach the fake"
    assert result.compared, (result.skipped, result.detail)  # a real comparison, never a skip
    document = recorded[0]["documents"][0]
    assert isinstance(document, dict) and "content" in document, document  # the OpenAI content-parts shape
    assert any(part["type"] == "image_url" for part in document["content"])
    image_part = next(part for part in document["content"] if part["type"] == "image_url")
    assert image_part["image_url"]["url"].startswith("data:image/")

    # the pooling route: the media item goes its own messages request (one media item per call)
    recorded.clear()

    class RecordingPoolEngine(FakePoolEngine):
        def handle(self, method: str, path: str, body: object):
            if method == "POST" and path.endswith("/pooling"):
                recorded.append(body)
            return super().handle(method, path, body)

    pool_recipe = load_recipe(tmp_path / "recipes" / "fake-pool")
    pool_bundle = load_cases(tmp_path / "cases", pool_recipe, recipes_root=tmp_path / "recipes", check_lengths=False)
    pool_result = run_case(
        pool_recipe, with_image(text_case(pool_bundle)), target="fake", fake_engine=RecordingPoolEngine()
    )
    assert pool_result.compared, (pool_result.skipped, pool_result.detail)  # a comparison, never a product error
    messages_bodies = [body for body in recorded if "messages" in body]
    assert messages_bodies, f"the media item must go out as a messages request; recorded {sorted(recorded[0])}"
    parts = messages_bodies[0]["messages"][0]["content"]
    assert any(part["type"] == "image_url" for part in parts)


def test_an_image_case_skips_on_the_embed_route(tmp_path: Path) -> None:
    """The embed role's wires are text-only: the product's client refuses media, so the case is a
    declared skip (never a silent empty-text send)."""
    from rcp_ndcg_test.cases import CaseDocument

    media = tmp_path / "media"
    media.mkdir()
    (media / "pixel.png").write_bytes(_tiny_png())
    case = filled_case()
    documents = list(case.inputs.documents)
    documents[0] = CaseDocument(id=documents[0].id, image="media/pixel.png")
    inputs = case.inputs.model_copy(update={"documents": documents})
    image_case = case.model_copy(update={"inputs": inputs})
    result = run_case(packaged_recipe(), image_case, target="fake", fake_engine=FakeEmbedEngine())
    assert result.skipped is not None and "text only" in result.skipped
    assert not result.passed and not result.compared


def test_the_plugin_skips_recipes_without_a_registered_fake(tmp_path: Path) -> None:
    """A recipe with no fake for the target is silently absent (the params stay collectable)."""
    from rcp_ndcg_test.plugin import conformance_params

    register_fake_engine(FakeRerankEngine())
    register_fake_engine(FakePoolEngine())
    unregister_fake_engine("fake-rerank")
    try:
        params = conformance_params("fake", recipes_root=RECIPES, cases_root=CASES)
    finally:
        unregister_fake_engine("fake-rerank")  # restore the registry this test found
        unregister_fake_engine("fake-pool")
    assert {param.id.split("/")[0] for param in params} == {"fake-pool"}
