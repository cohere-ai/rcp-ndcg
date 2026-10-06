"""The conformance suite: one runner, two targets, one typed report.

A case is sent through the **product's role client** built from the recipe's ``client`` block -- the
product's endpoint config, wire adapter, transport and interpretation, never raw HTTP and never a copy
of the client -- and its answer is compared with the case's ``expected`` under the declared tolerance.
Two targets:

- ``target="engine"``: ``base_url`` of a live engine serving the recipe (a vLLM on a GPU slot, or any
  process speaking the role's routes). The runner's own
  :class:`~rcp_ndcg.inference.transport.Transport` sends through real HTTP.
- ``target="fake"``: a recipe-level fake engine (:mod:`rcp_ndcg_test.fakes`), resolved through the
  registry by recipe id. The fake sits *below* the transport: the runner hands it to the product's
  :class:`~rcp_ndcg.inference.transport.Transport` as an ``httpx`` transport, so routing, retries and
  the adapters all run in the product's code path.

The report is per case (:class:`CaseResult`: compared, passed, skipped with a reason) and per run
(:class:`ConformanceReport`). ``expected.values: null`` is a skip, never a pass; ``expected.kind: none``
runs the path and compares nothing; a tolerance breach is a failure carrying the worst delta.

One declared bridge, until the role clients carry the text budget (the ``clients-final`` wiring): the
runner pre-fits every input with the product's :func:`rcp_ndcg.data.preprocess.fit` -- the recipe's own
tokenizer, template and budget -- and sends the fitted contents through a client whose budget fields are
cleared (the client refuses a budget until the wiring lands). The recipe's per-side prompts must be
empty: the client would prepend them onto the fitted render a second time. When the wiring lands, drop
``_fit*`` here and send the raw contents; the client then fits and cuts itself.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from .cases import Case
from .errors import ConformanceError

if TYPE_CHECKING:
    from rcp_ndcg_vllm.recipe import Recipe

__all__ = [
    "CaseResult",
    "ConformanceReport",
    "Target",
    "run_case",
    "run_suite",
]

Target = Literal["engine", "fake"]
"""What answers the case: a live engine at ``base_url``, or a fake engine from the registry."""

_FAKE_BASE_URL = "http://rcp-ndcg-test.fake/v1"
"""The placeholder base URL of a fake target: the fake transport answers every request, so the URL is
only what the transport prepends to the route paths."""


@dataclass(frozen=True)
class CaseResult:
    """The outcome of one case under one target.

    Attributes:
        case_id: The case's id (``<recipe>/<slug>``).
        compared: Whether a comparison against ``expected.values`` ran (a skipped case and a
            ``kind: none`` case compare nothing).
        passed: Whether the case passed (a skipped case has ``passed=False`` and a reason).
        skipped: Why the case did not run to a comparison, or ``None``. ``values: null`` (pending
            GPU), a media input the product's role clients cannot send yet, and -- once filled -- a
            video container are skips; a tolerance breach never is.
        detail: The worst delta against the tolerance on a comparison, the skip's reason, or the
            product error a failed send raised.
    """

    case_id: str
    compared: bool
    passed: bool
    skipped: str | None
    detail: str | None

    @property
    def failed(self) -> bool:
        """The case ran and failed (a tolerance breach, a refused request, an unusable answer)."""
        return not self.passed and self.skipped is None


@dataclass(frozen=True)
class ConformanceReport:
    """The typed report of one suite run: every case's outcome, and the run's verdict.

    Attributes:
        recipe_id: The recipe the cases belong to.
        target: What answered (``engine`` or ``fake``).
        base_url: The endpoint the cases were sent to (the engine's URL, or the fake's placeholder).
        results: One :class:`CaseResult` per case, in the suite's order.
    """

    recipe_id: str
    target: Target
    base_url: str
    results: tuple[CaseResult, ...]

    @property
    def ok(self) -> bool:
        """Every case passed or was skipped; nothing failed. A suite with pending values is ok."""
        return all(result.passed or result.skipped is not None for result in self.results)

    @property
    def failures(self) -> tuple[CaseResult, ...]:
        """The results that ran and failed."""
        return tuple(result for result in self.results if result.failed)

    @property
    def skipped(self) -> tuple[CaseResult, ...]:
        """The results that did not run to a comparison, reasons attached."""
        return tuple(result for result in self.results if result.skipped is not None)

    def summary(self) -> str:
        """One line for a log: the counts and the failing case ids."""
        state = "ok" if self.ok else f"FAILED ({', '.join(result.case_id for result in self.failures)})"
        return (
            f"{self.recipe_id} [{self.target} @ {self.base_url}]: {len(self.results)} case(s), "
            f"{sum(result.passed for result in self.results)} passed, {len(self.skipped)} skipped, "
            f"{len(self.failures)} failed -- {state}"
        )


def run_suite(
    recipe: Recipe,
    cases: Sequence[Case],
    *,
    target: Target,
    base_url: str | None = None,
    fake_engine: Any | None = None,
) -> ConformanceReport:
    """Run every case of one recipe against one target and return the report.

    Inputs: a loaded recipe, the recipe's cases, the ``target`` and -- for ``target="engine"`` -- the
    engine's ``base_url`` (for ``target="fake"``, the fake engine itself or a registry lookup by recipe
    id). Outputs: the :class:`ConformanceReport`; the suite never raises for a failing case, the
    failure is in the report.

    Raises:
        ConformanceError: ``target="engine"`` without a ``base_url``, a fake target with no registered
            fake for the recipe, or a recipe whose client config the role client refuses (see the
            module docstring for the pre-fit bridge).
    """
    resolved = _resolve(recipe, target, base_url, fake_engine)
    try:
        results = tuple(_run_case(resolved, case) for case in cases)
    finally:
        resolved.close()
    return ConformanceReport(recipe_id=recipe.id, target=target, base_url=resolved.base_url, results=results)


def run_case(
    recipe: Recipe,
    case: Case,
    *,
    target: Target,
    base_url: str | None = None,
    fake_engine: Any | None = None,
) -> CaseResult:
    """Run one case against one target (see :func:`run_suite` for the arguments and the errors)."""
    resolved = _resolve(recipe, target, base_url, fake_engine)
    try:
        return _run_case(resolved, case)
    finally:
        resolved.close()


# ---------------------------------------------------------------------------
# The resolved target: the endpoint config and the transport every case sends through
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Resolved:
    """One target's arrangement: the endpoint config and the transport (owned, closed by the caller)."""

    recipe: Recipe
    endpoint: Any
    sender: Any | None
    base_url: str
    transport: Any | None

    def close(self) -> None:
        """Close the transport the runner built (a fake target's, or one built for the client)."""
        if self.transport is not None:
            self.transport.aclose()


def _resolve(recipe: Recipe, target: Target, base_url: str | None, fake_engine: Any | None) -> _Resolved:
    """The endpoint config and sender of one target.

    The config is the recipe's ``client`` block -- the product's own endpoint model -- with ``base_url``
    and the recipe id filled in. On this branch the runner pre-fits every input with the product's
    ``fit``, so the budget fields are cleared (the client refuses a budget until the wiring lands) and
    the recipe's per-side prompts must be empty (they would be prepended onto the fitted render a
    second time). The transport's retries and outage parking are switched off: a conformance run fails
    fast on a refusal instead of parking for minutes on a dead engine.
    """
    client = recipe.client
    if getattr(client, "query_prompt", "") or getattr(client, "doc_prompt", ""):
        raise ConformanceError(
            f"recipe {recipe.id}: the runner pre-fits with the product's fit and sends the fitted render "
            "through the role client, which would prepend query_prompt/doc_prompt a second time; a recipe "
            "with per-side prompts needs the budget wired into the client (clients-final) first"
        )
    runtime = {"recipe": recipe.id, "max_retries": 0, "wait_on_outage_s": 0.0}
    if target == "engine":
        if base_url is None:
            raise ConformanceError(
                f"recipe {recipe.id}: target 'engine' needs base_url (the engine's URL, e.g. "
                "http://127.0.0.1:8000/v1); run against a fake without one"
            )
        if fake_engine is not None:
            raise ConformanceError(f"recipe {recipe.id}: target 'engine' takes a base_url, not a fake engine")
        endpoint = client.model_copy(update={**runtime, "base_url": base_url})
        from rcp_ndcg.inference.transport import Transport

        transport = Transport(endpoint)
        return _Resolved(recipe=recipe, endpoint=endpoint, sender=transport, base_url=base_url, transport=transport)
    if base_url is not None:
        raise ConformanceError(f"recipe {recipe.id}: target 'fake' takes a fake engine, not a base_url")
    engine = fake_engine if fake_engine is not None else _registered_fake(recipe)
    endpoint = client.model_copy(update={**runtime, "base_url": _FAKE_BASE_URL})
    from rcp_ndcg.inference.transport import Transport

    from .fakes import fake_http_transport

    transport = Transport(endpoint, httpx_transport=fake_http_transport(engine))
    return _Resolved(recipe=recipe, endpoint=endpoint, sender=transport, base_url=_FAKE_BASE_URL, transport=transport)


def _registered_fake(recipe: Recipe) -> Any:
    """The registry's fake for the recipe id (importing :mod:`rcp_ndcg_test.fakes` registers the shipped one)."""
    from .fakes import fake_engine_for

    return fake_engine_for(recipe.id)


# ---------------------------------------------------------------------------
# One case
# ---------------------------------------------------------------------------


def _run_case(resolved: _Resolved, case: Case) -> CaseResult:
    """One case: the early skips, the send through the product's role client, the comparison.

    A product error (an engine refusal, an unusable answer) is the case's failure, carrying the
    product's own typed error as its detail -- it never raises out of the runner.
    """
    try:
        return _run_or_skip(resolved, case)
    except Exception as error:
        return CaseResult(
            case_id=case.id, compared=False, passed=False, skipped=None, detail=f"{type(error).__name__}: {error}"
        )


def _run_or_skip(resolved: _Resolved, case: Case) -> CaseResult:
    """The case's path: the early skips, then the send and the comparison."""
    expected = case.expected
    if expected.kind == "none":
        _send(resolved, case)  # the path is the exercise; nothing is compared
        return CaseResult(
            case_id=case.id,
            compared=False,
            passed=True,
            skipped=None,
            detail="no comparison: expected.kind is none (the case exercises the path)",
        )
    if expected.values is None:
        return CaseResult(
            case_id=case.id,
            compared=False,
            passed=False,
            skipped=(
                f"expected.values is null (status={expected.status}, origin={expected.origin}): the GPU wave "
                "fills it from the reference implementation and the engine"
            ),
            detail=None,
        )
    blocked = _media_skip(resolved.recipe, case)
    if blocked is not None:
        return CaseResult(case_id=case.id, compared=False, passed=False, skipped=blocked, detail=None)
    return _compare(case, _matrix(_send(resolved, case)))


def _media_skip(recipe: Recipe, case: Case) -> str | None:
    """Why the runner cannot send this case's media yet, or ``None`` when it can.

    The product's embed adapters take text only (media embedding is wired with the media-preparation
    mechanism), and the media lowering sends image parts: a video container needs the frames reader,
    which is an ingest step. Images on the pooling and rerank routes run.
    """
    if not any(document.image or document.video for document in case.inputs.documents):
        return None
    if recipe.role == "embed":
        return (
            "the embed adapters take text only; media embedding is wired with the media-preparation "
            "mechanism (the product's adapter names it in its refusal)"
        )
    if any(document.video for document in case.inputs.documents):
        return (
            "the product's media lowering sends images; a video container is refused by "
            "content_parts_payload until the clip is ingested as frames"
        )
    return None


# ---------------------------------------------------------------------------
# The send: the product's fit, then the product's role client
# ---------------------------------------------------------------------------


def _send(resolved: _Resolved, case: Case) -> Any:
    """The role client's answer for the case.

    For ``embed`` and ``multi_vector``: the fitted queries and documents through the product's
    :class:`~rcp_ndcg.inference.clients.EmbeddingClient` / ``PoolingClient``, as a pair of
    :class:`~rcp_ndcg.inference.types.Embeddings`. For ``rerank``: the fitted pairs through
    :class:`~rcp_ndcg.inference.clients.RerankClient`, as one
    :class:`~rcp_ndcg.inference.types.RerankResult` per query.
    """
    endpoint = _budget_fields_cleared(resolved.endpoint)
    if resolved.recipe.role == "rerank":
        return _send_rerank(resolved, case, endpoint)
    from rcp_ndcg.inference.clients import EmbeddingClient, PoolingClient
    from rcp_ndcg.inference.types import EncodeRole

    queries = _fit_side(resolved.recipe, [query.text for query in case.inputs.queries], "query", case)
    documents = _fit_side(
        resolved.recipe, [document.text or "" for document in case.inputs.documents], "document", case
    )
    client: Any = (PoolingClient if resolved.recipe.role == "multi_vector" else EmbeddingClient)(
        endpoint, sender=resolved.sender
    )
    try:
        query_side = client.encode(_as_contents(queries), EncodeRole.QUERY)
        document_side = client.encode(_as_contents(documents), EncodeRole.DOCUMENT)
    finally:
        client_close = getattr(client, "close", None)
        if client_close is not None:
            client_close()
    return query_side, document_side


def _send_rerank(resolved: _Resolved, case: Case, endpoint: Any) -> Any:
    """The case's queries through :meth:`RerankClient.rerank_many`, the documents cut to the budget.

    The query goes in raw with the case's instruction, so the client applies its own fold
    (``instruction: fold``) -- exactly the text the fit measured. When the fit had to cut the query
    span (over its declared share), the folded-and-cut text goes in directly with no instruction: the
    client cannot fold and cut in one step on this branch, and folding a folded query would fold twice.
    """
    from rcp_ndcg_core._records import RankingExample

    from rcp_ndcg.inference.clients import RerankClient

    client = RerankClient(endpoint, sender=resolved.sender)
    doc_ids = [document.id for document in case.inputs.documents]
    examples = []
    for query in case.inputs.queries:
        folded, documents, query_cut = _fit_pair(resolved.recipe, query.text, case)
        examples.append(
            RankingExample(
                query_id=str(query.id),
                query=query.text if not query_cut else folded,
                docs=documents,
                doc_ids=doc_ids,
                instruction=None if query_cut else case.inputs.instruction,
            )
        )
    try:
        results = client.rerank_many(examples)
    finally:
        client.close()
    if len(results) != len(examples):
        raise ConformanceError(
            f"case {case.id}: the rerank client returned {len(results)} result(s) for {len(examples)} query(ies)"
        )
    return results


def _fit_side(recipe: Recipe, texts: list[str], shape: str, case: Case) -> list[str]:
    """One side's rendered (and cut) strings, through the product's :func:`fit`."""
    from rcp_ndcg.data.preprocess import fit

    tokenizer, budget = _fitter(recipe)
    result = fit(
        texts,
        shape,  # type: ignore[arg-type]
        budget,
        tokenizer,
        ids=[str(index) for index in range(len(texts))],
        instruction=case.inputs.instruction,
    )
    return list(result.texts)


def _fit_pair(recipe: Recipe, query: str, case: Case) -> tuple[str, list[str], bool]:
    """The pair fit of one query against the case's documents.

    Returns the folded query text, each document's fitted (cut) text, and whether the query span was
    cut (the send then uses the cut text unfolded -- see :func:`_send_rerank`).
    """
    from rcp_ndcg.data.preprocess import fit

    tokenizer, budget = _fitter(recipe)
    folded = _fold_query(recipe, query, case.inputs.instruction)
    inputs = [(folded, document.text or "") for document in case.inputs.documents]
    result = fit(
        inputs,
        "pair",
        budget,
        tokenizer,
        ids=[document.id for document in case.inputs.documents],
        instruction=case.inputs.instruction,
    )
    documents = [document for _, document in result.contents]
    query_cut = any(query_text != folded for query_text, _ in result.contents)
    return folded, documents, query_cut


def _fitter(recipe: Recipe) -> tuple[Any, Any]:
    """The recipe's (tokenizer, budget) pair, loaded once per process: the product's tokenizer loader
    and the budget its client block declares (the harness's own bridge, reused, not copied)."""
    cached = _FITTERS.get(recipe.id)
    if cached is not None:
        return cached
    from rcp_ndcg_vllm.equivalence.fitting import budget_of, tokenizer_of

    tokenizer = tokenizer_of(recipe)
    cached = (tokenizer, budget_of(recipe).model_copy(update={"tokenizer": tokenizer.name}))
    _FITTERS[recipe.id] = cached
    return cached


_FITTERS: dict[str, tuple[Any, Any]] = {}


def _fold_query(recipe: Recipe, query: str, instruction: str | None) -> str:
    """The query as the client folds it for ``instruction: fold`` (the product's own render)."""
    from rcp_ndcg.inference.config import RerankEndpoint

    if not isinstance(recipe.client, RerankEndpoint) or recipe.client.instruction != "fold" or not instruction:
        return query
    from rcp_ndcg_core._records import Query

    return str(Query(query_id="", query=query, instruction=instruction).format_query())


def _budget_fields_cleared(endpoint: Any) -> Any:
    """The endpoint config without the budget fields (the client refuses them until the wiring lands)."""
    updates: dict[str, Any] = {"max_tokens": None}
    if getattr(endpoint, "query_max_tokens", None) is not None:
        updates["query_max_tokens"] = None
    return endpoint.model_copy(update=updates)


def _as_contents(texts: list[str]) -> list[Any]:
    """The fitted strings as the wire contents the role clients take."""
    from rcp_ndcg_core.content import Content

    return [Content.from_text(text) for text in texts]


# ---------------------------------------------------------------------------
# The answer: one query x document score matrix, however the role produced it
# ---------------------------------------------------------------------------


def _matrix(answer: Any) -> Any:
    """The ``len(queries) x len(documents)`` matrix of the role's answer.

    ``embed``: the cosine of every query/document pair (the client L2-normalised when the config asked;
    the matrix is computed on unit vectors either way). ``multi_vector``: MaxSim, through the product's
    exact scorer. ``rerank``: the relevance scores, aligned to the documents by position.
    """
    from rcp_ndcg.inference.types import Embeddings, RerankResult

    if isinstance(answer, tuple) and len(answer) == 2 and all(isinstance(item, Embeddings) for item in answer):
        query_vectors, document_vectors = answer
        return _similarity_matrix(query_vectors, document_vectors)
    if isinstance(answer, list) and all(isinstance(item, RerankResult) for item in answer):
        return np.asarray([result.scores for result in answer], dtype=np.float64)
    raise ConformanceError(f"the runner cannot read the role's answer type {type(answer).__name__}")


def _similarity_matrix(query_vectors: Any, document_vectors: Any) -> Any:
    """The query x document similarity matrix of the two sides' embeddings.

    A single-vector pair scores cosine; a late-interaction pair scores MaxSim with the product's
    blocked scorer and its tie rule (ties toward the lower document index).
    """
    from rcp_ndcg.inference.types import l2_normalize

    if query_vectors.num_items == 0 or document_vectors.num_items == 0:
        raise ConformanceError("the case sent no queries or no documents; there is nothing to compare")
    if query_vectors.is_multi_vector or document_vectors.is_multi_vector:
        if not (query_vectors.is_multi_vector and document_vectors.is_multi_vector):
            raise ConformanceError("one side is multi-vector and the other is not; the roles do not mix")
        return _maxsim_matrix(document_vectors, query_vectors)
    queries = np.asarray(l2_normalize(np.asarray(query_vectors.vectors, dtype=np.float64)))
    documents = np.asarray(l2_normalize(np.asarray(document_vectors.vectors, dtype=np.float64)))
    return queries @ documents.T


def _maxsim_matrix(document_vectors: Any, query_vectors: Any) -> Any:
    """The full query x document MaxSim matrix from the product's top-k scorer (``k`` = all documents)."""
    from rcp_ndcg.retrieval.maxsim import maxsim_topk

    n_documents = document_vectors.num_items
    scores, indices = maxsim_topk(document_vectors, query_vectors, k=n_documents)
    matrix = np.full((query_vectors.num_items, n_documents), np.nan)
    rows = np.repeat(np.arange(query_vectors.num_items), n_documents)
    matrix[rows, np.asarray(indices).reshape(-1)] = np.asarray(scores).reshape(-1)
    return matrix


# ---------------------------------------------------------------------------
# The comparison
# ---------------------------------------------------------------------------


def _compare(case: Case, matrix: Any) -> CaseResult:
    """The matrix against ``expected`` under the tolerance: the case's verdict and its detail."""
    expected = case.expected
    shape = (len(case.inputs.queries), len(case.inputs.documents))
    if matrix.shape != shape:
        return CaseResult(
            case_id=case.id,
            compared=True,
            passed=False,
            skipped=None,
            detail=f"the runner produced a {matrix.shape[0]}x{matrix.shape[1]} matrix for a {shape[0]}x{shape[1]} case",
        )
    tolerance = expected.tolerance
    if tolerance is None:  # pragma: no cover - validated at load
        raise ConformanceError(f"case {case.id} carries no tolerance")
    if expected.kind == "ranking":
        return _compare_ranking(case, matrix, tolerance)
    if tolerance.abs is not None:
        return _compare_abs(case, matrix, tolerance.abs)
    return _compare_spearman(case, matrix, tolerance.spearman_min)  # type: ignore[arg-type]


def _compare_abs(case: Case, matrix: Any, bound: float) -> CaseResult:
    """Every cell's absolute difference within ``abs``; the detail names the worst cell."""
    delta = np.abs(matrix - np.asarray(case.expected.values, dtype=np.float64))
    worst = float(np.max(delta))
    if worst <= bound:
        return CaseResult(
            case_id=case.id,
            compared=True,
            passed=True,
            skipped=None,
            detail=f"max abs delta {worst:.3g} <= abs {bound:.3g}",
        )
    row, column = (int(index) for index in np.unravel_index(int(np.argmax(delta)), delta.shape))
    return CaseResult(
        case_id=case.id,
        compared=True,
        passed=False,
        skipped=None,
        detail=(
            f"max abs delta {worst:.6g} exceeds abs {bound:.3g} at "
            f"[query {case.inputs.queries[row].id}, document {case.inputs.documents[column].id}]: "
            f"got {matrix[row, column]:.6g}, expected {case.expected.values[row][column]:.6g}"
        ),
    )


def _compare_spearman(case: Case, matrix: Any, threshold: float) -> CaseResult:
    """Every row's rank correlation against the expected row, bounded from below."""
    worst: tuple[float, int] | None = None
    for index, (got_row, want_row) in enumerate(zip(matrix, case.expected.values, strict=True)):
        rho = _spearman(np.asarray(got_row, dtype=np.float64), np.asarray(want_row, dtype=np.float64))
        if worst is None or rho < worst[0]:
            worst = (rho, index)
    if worst is None:  # pragma: no cover - the load requires at least one query
        raise ConformanceError(f"case {case.id} holds no queries")
    rho, index = worst
    if rho >= threshold:
        return CaseResult(
            case_id=case.id,
            compared=True,
            passed=True,
            skipped=None,
            detail=f"worst spearman {rho:.4g} >= spearman_min {threshold:.4g}",
        )
    return CaseResult(
        case_id=case.id,
        compared=True,
        passed=False,
        skipped=None,
        detail=f"spearman {rho:.4g} < spearman_min {threshold:.4g} on query {case.inputs.queries[index].id}",
    )


def _compare_ranking(case: Case, matrix: Any, tolerance: Any) -> CaseResult:
    """The ranking kind: the derived order per query against the case's expected order.

    ``rank_exact``: the derived order (all documents, ties toward the lower document index -- the
    package's rule) must equal the expected row exactly. ``spearman_min``: the expected row may be a
    top-k prefix; its documents' positions in the derived order correlate with ``1..k`` from below.
    """
    doc_ids = [document.id for document in case.inputs.documents]
    if tolerance.rank_exact:
        for index, (row, expected_row) in enumerate(zip(matrix, case.expected.values, strict=True)):
            derived = _ranking_of(row, doc_ids)
            if derived != expected_row:
                return CaseResult(
                    case_id=case.id,
                    compared=True,
                    passed=False,
                    skipped=None,
                    detail=(
                        f"the ranking of query {case.inputs.queries[index].id} is {derived}, expected "
                        f"{expected_row} (rank_exact)"
                    ),
                )
        return CaseResult(
            case_id=case.id,
            compared=True,
            passed=True,
            skipped=None,
            detail="every query's ranking matches rank-exactly",
        )
    threshold = tolerance.spearman_min
    worst: tuple[float, int] | None = None
    for index, (row, expected_row) in enumerate(zip(matrix, case.expected.values, strict=True)):
        derived = _ranking_of(row, doc_ids)
        position = {doc_id: rank for rank, doc_id in enumerate(derived)}
        if len(expected_row) < 2:
            rho = 1.0
        else:
            got = [float(position[doc_id]) for doc_id in expected_row]
            rho = _spearman(np.asarray(got), np.arange(len(expected_row), dtype=np.float64))
        if worst is None or rho < worst[0]:
            worst = (rho, index)
    if worst is None:  # pragma: no cover - the load requires at least one query
        raise ConformanceError(f"case {case.id} holds no queries")
    rho, index = worst
    if rho >= threshold:
        return CaseResult(
            case_id=case.id,
            compared=True,
            passed=True,
            skipped=None,
            detail=f"worst expected-prefix spearman {rho:.4g} >= spearman_min {threshold:.4g}",
        )
    return CaseResult(
        case_id=case.id,
        compared=True,
        passed=False,
        skipped=None,
        detail=(
            f"spearman {rho:.4g} < spearman_min {threshold:.4g} on the ranking of query {case.inputs.queries[index].id}"
        ),
    )


def _ranking_of(row: Any, doc_ids: list[str]) -> list[str]:
    """The document ids of one score row, best first, ties toward the lower document index."""
    order = sorted(range(len(row)), key=lambda index: (-float(row[index]), index))
    return [doc_ids[index] for index in order]


def _spearman(got: Any, want: Any) -> float:
    """The Spearman rank correlation of two score rows, ties averaged.

    The product has Kendall's tau only (:func:`rcp_ndcg_vllm.equivalence.gates.kendall_tau_b`); the
    case format's ``spearman_min`` needs the rank correlation, so it is implemented here and reported
    as a product gap (a shared helper belongs in the product's eval surface). Two constant rows carry
    no ranking information: the correlation is ``nan`` and the comparison fails, never silently passes.
    """
    if len(got) != len(want):
        raise ConformanceError(f"spearman needs two rows of one length, got {len(got)} and {len(want)}")
    if len(got) < 2:
        return 1.0
    if float(np.ptp(got)) == 0.0 or float(np.ptp(want)) == 0.0:
        return float("nan")
    return float(np.corrcoef(_average_ranks(got), _average_ranks(want))[0, 1])


def _average_ranks(values: Any) -> Any:
    """Zero-based ranks with ties averaged (the rank correlation's convention, before correlating)."""
    order = np.argsort(np.asarray(values, dtype=np.float64), kind="stable")
    sorted_values = np.asarray(values, dtype=np.float64)[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranks
