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
runs the path and compares nothing; a tolerance breach is a failure carrying the worst delta. Under the
recipe's declared over-cap deviation, the cells the client changed relative to the case's raw input are
**reported, not compared** -- stage 2's rule (``reference.known_deviations``: ``anchor_drop_over_cap`` or
``over_cap_cut_differs``): the expectation was filled from the uncut input, so a correctly served engine
must not fail on the reference's own over-cap cut.

"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from .cases import Case
from .errors import ConformanceError
from .ranks import spearman

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


@dataclass(frozen=True)
class _Sent:
    """One case's answer and the cells the client changed relative to the case's raw input.

    ``changed`` is a ``len(queries) x len(documents)`` boolean array: a query the client settled or cut
    changes every cell of its row, a changed document every cell of its column (read from the client's own
    processing records, the same referent stage 2 uses). Under the recipe's declared over-cap deviation those
    cells are reported instead of compared.
    """

    answer: Any
    changed: Any


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
        ConformanceError: ``target="engine"`` without a ``base_url``, or a fake target with no
            registered fake for the recipe.
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
    if case.recipe != recipe.id:
        raise ConformanceError(
            f"case {case.id!r} belongs to recipe {case.recipe!r} but the run plans recipe "
            f"{recipe.id!r}: a run under the wrong scorer is an identity lie"
        )
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
            self.transport.close()


def _resolve(recipe: Recipe, target: Target, base_url: str | None, fake_engine: Any | None) -> _Resolved:
    """The endpoint config and sender of one target.

    The config is the recipe's ``client`` block -- the product's own endpoint model, budget and media
    fields included -- with ``base_url`` and the recipe id filled in: the wired role client fits, cuts,
    folds and prepares the media itself, and the runner sends the case's raw contents. The transport's
    retries and outage parking are switched off: a conformance run fails fast on a refusal instead of
    parking for minutes on a dead engine.
    """
    client = recipe.client
    # A recipe-relative tokenizer path resolves against the recipe directory (the harness's own rule);
    # the wired client loads the tokenizer from the config, so the spec must be absolute here.
    spec = client.get("tokenizer")
    directory = recipe._dir
    if spec is not None and directory is not None:
        from pathlib import Path

        candidate = Path(spec)
        if not candidate.is_absolute() and (directory / candidate).exists():
            client = {**client, "tokenizer": str(directory / candidate)}
    runtime = {"recipe": recipe.id, "max_retries": 0, "wait_on_outage_s": 0.0}
    if target == "engine":
        if base_url is None:
            raise ConformanceError(
                f"recipe {recipe.id}: target 'engine' needs base_url (the engine's URL, e.g. "
                "http://127.0.0.1:8000/v1); run against a fake without one"
            )
        if fake_engine is not None:
            raise ConformanceError(f"recipe {recipe.id}: target 'engine' takes a base_url, not a fake engine")
        from rcp_ndcg.inference.config import EmbeddingEndpoint, PoolingEndpoint, RerankEndpoint
        from rcp_ndcg.inference.transport import Transport

        classes = {"embed": EmbeddingEndpoint, "multi_vector": PoolingEndpoint, "rerank": RerankEndpoint}
        endpoint = classes[recipe.role].model_validate({**client, **runtime, "base_url": base_url})
        transport = Transport(endpoint)
        return _Resolved(recipe=recipe, endpoint=endpoint, sender=transport, base_url=base_url, transport=transport)
    if base_url is not None:
        raise ConformanceError(f"recipe {recipe.id}: target 'fake' takes a fake engine, not a base_url")
    engine = fake_engine if fake_engine is not None else _registered_fake(recipe)
    from rcp_ndcg.inference.config import EmbeddingEndpoint, PoolingEndpoint, RerankEndpoint
    from rcp_ndcg.inference.transport import Transport

    classes = {"embed": EmbeddingEndpoint, "multi_vector": PoolingEndpoint, "rerank": RerankEndpoint}
    endpoint = classes[recipe.role].model_validate({**client, **runtime, "base_url": _FAKE_BASE_URL})

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
        blocked = _media_skip(resolved.recipe, case)
        if blocked is not None:
            return CaseResult(case_id=case.id, compared=False, passed=False, skipped=blocked, detail=None)
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
    sent = _send(resolved, case)
    return _compare(case, _matrix(sent.answer), report=_over_cap_report(resolved.recipe, sent.changed))


def _media_skip(recipe: Recipe, case: Case) -> str | None:
    """Why the runner cannot send this case's media, or ``None`` when it can.

    The wired pooling and rerank clients prepare the media themselves (image parts on the wire, tokens
    reserved whole), so an image case runs on every role that reads media. What still cannot run: the
    embed role (its adapters are text-only -- the product's client refuses media before preparation)
    and a video container on any role (the OpenAI content-parts lowering sends images only; a
    container needs the frames reader). A skip is recorded with its reason, never a silent pass.
    """
    if not any(document.image or document.video for document in case.inputs.documents):
        return None
    if recipe.role == "embed":
        return (
            "the product's embed client refuses media (its adapters take text only; the product raises "
            "the refusal itself); media embedding is wired with the media-preparation mechanism"
        )
    if any(document.video for document in case.inputs.documents):
        return (
            "the product's media lowering sends images; a video container is refused by "
            "content_parts_payload until the clip is ingested as frames"
        )
    return None


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# The send: the case's raw contents through the product's wired role client
# ---------------------------------------------------------------------------


def _send(resolved: _Resolved, case: Case) -> _Sent:
    """The role client's answer for the case: the case's raw contents through the product's role client.

    The client does every content decision -- the per-side prompts, the text budget's fit and cut, the
    instruction fold, the media preparation and the empty-document rule -- from the recipe's ``client``
    block; the runner pre-fits nothing. For ``embed`` and ``multi_vector``: the raw queries and documents
    through :class:`~rcp_ndcg.inference.clients.EmbeddingClient` / ``PoolingClient``, as a pair of
    :class:`~rcp_ndcg.inference.types.Embeddings`. For ``rerank``: the raw queries, documents and
    instruction through :class:`~rcp_ndcg.inference.clients.RerankClient` (``rerank_many``), as one
    :class:`~rcp_ndcg.inference.types.RerankResult` per query, aligned to the case's documents.

    The client's processing records are read beside the answer: which cells it changed relative to the raw
    input, the referent of the over-cap report-not-compare rule.
    """
    if resolved.recipe.role == "rerank":
        return _send_rerank(resolved, case)
    from rcp_ndcg.inference.clients import EmbeddingClient, PoolingClient
    from rcp_ndcg.inference.types import EncodeRole

    client: Any = (PoolingClient if resolved.recipe.role == "multi_vector" else EmbeddingClient)(
        resolved.endpoint, sender=resolved.sender
    )
    try:
        # The case's instruction is the run's TASK instruction, placed by the recipe's own mode (the client's
        # `instruction: fold` prefix, or the template's own span); a recipe that declares no policy carries the
        # case's instruction in its template instead (the case loader's check).
        instruction = case.inputs.instruction if resolved.recipe.client.get("instruction") is not None else None
        query_side = client.encode(_query_contents(case), EncodeRole.QUERY, instruction=instruction)
        document_side = client.encode(_document_contents(case), EncodeRole.DOCUMENT)
        changed = _embed_changed(client, len(case.inputs.queries), len(case.inputs.documents))
    finally:
        client.close()
    return _Sent(answer=(query_side, document_side), changed=changed)


def _embed_changed(client: Any, n_queries: int, n_documents: int) -> Any:
    """The query x document cells the embed/pool client changed, from its own processing records.

    A record names the row's position within its call (``input_id``) and the shape it was prepared as: a
    changed query settles every cell of its row, a changed document every cell of its column. A record that
    cannot be attributed to one input (a media fit's owner fallback) changes every cell -- never a crash,
    never a silent gate.
    """
    flags = np.zeros((n_queries, n_documents), dtype=bool)
    for record in client.processing:
        if not record.changed:
            continue  # pragma: no cover - a client emits records for changed rows only
        if not record.input_id.isdigit():
            return np.ones((n_queries, n_documents), dtype=bool)
        position = int(record.input_id)
        if record.shape == "query" and 0 <= position < n_queries:
            flags[position, :] = True
        elif record.shape == "document" and 0 <= position < n_documents:
            flags[:, position] = True
        else:
            return np.ones((n_queries, n_documents), dtype=bool)
    return flags


def _send_rerank(resolved: _Resolved, case: Case) -> _Sent:
    """The case's queries through :meth:`RerankClient.rerank`, raw.

    The raw query goes in with the case's instruction as the run's TASK instruction (the client folds or
    sends it per its ``instruction`` mode) and the documents as the content parts the case declares (media
    included): the client fits the pairs, cuts the query to its declared share, chunks on overflow and pools
    by max -- every content decision is the product's.  One call per query (the same call ``rerank_many``
    makes under its semaphore), so each query's processing records are attributable to its own row.
    """
    from rcp_ndcg_core.records import RankingExample

    from rcp_ndcg.inference.clients import RerankClient

    client = RerankClient(resolved.endpoint, sender=resolved.sender)
    doc_ids = [document.id for document in case.inputs.documents]
    documents = _document_contents(case)
    from .cases import text_of

    examples = [
        RankingExample(
            query_id=str(query.id),
            query=text_of(query) or "",
            docs=[content.text for content in documents],
            contents=documents,
            doc_ids=doc_ids,
        )
        for query in case.inputs.queries
    ]
    results: list[Any] = []
    changed = np.zeros((len(examples), len(doc_ids)), dtype=bool)
    try:
        for index, example in enumerate(examples):
            start = len(client.processing)
            results.append(
                client.rerank(
                    example.as_content,
                    example.doc_contents,
                    instruction=case.inputs.instruction,
                    query_instruction=example.instruction,
                    query_id=str(example.id),
                )
            )
            changed[index] = _rerank_changed(client, start, len(doc_ids))
    finally:
        client.close()
    if len(results) != len(examples):
        raise ConformanceError(
            f"case {case.id}: the rerank client returned {len(results)} result(s) for {len(examples)} query(ies)"
        )
    return _Sent(answer=results, changed=changed)


def _rerank_changed(client: Any, start: int, n_documents: int) -> list[bool]:
    """Per document of one rerank call, whether the client changed the pair, from its processing records.

    The one home of the rule stage 2 uses (``stages._changed_rows``): a record of the document's position
    changes that pair; the reranker's shared query (``QUERY_DOC_ID``) or a record under no position changes
    every pair of the call.
    """
    from rcp_ndcg.inference.clients.rerank import QUERY_DOC_ID

    flags = [False] * n_documents
    for record in client.processing[start:]:
        if not record.changed:
            continue  # pragma: no cover - a client emits records for changed rows only
        if record.input_id == QUERY_DOC_ID or not record.input_id.isdigit():
            return [True] * n_documents
        position = int(record.input_id)
        if 0 <= position < n_documents:
            flags[position] = True
    return flags


def _over_cap_report(recipe: Any, changed: Any) -> tuple[str, Any] | None:
    """The cells to report instead of comparing, or ``None`` when the recipe declares no over-cap deviation.

    Stage 2's rule: only a recipe whose ``reference.known_deviations`` declares ``anchor_drop_over_cap`` or
    ``over_cap_cut_differs`` exempts the cells its client changed; every other recipe gates them exactly.
    """
    reference = getattr(recipe, "reference", None)
    deviation = reference.over_cap_deviation if reference is not None else None
    if deviation is None or not bool(np.asarray(changed).any()):
        return None
    return deviation, changed


def _query_contents(case: Case) -> list[Any]:
    """The case's queries as the wire contents the role clients take (text literal or text_ref)."""
    from rcp_ndcg_core.content import Content

    from .cases import text_of

    return [Content.from_text(text_of(query) or "") for query in case.inputs.queries]


def _document_contents(case: Case) -> list[Any]:
    """The case's documents as the wire contents: the text part plus the declared media parts.

    The media references resolve against the case's recipe directory (the format's ``media/<file>``
    paths), which :func:`load_case` pins on the case; the wired clients size and prepare them.
    """
    from rcp_ndcg_core.content import Content, ImagePart, TextPart, VideoPart

    from .cases import media_ref, text_of

    contents: list[Any] = []
    for document in case.inputs.documents:
        parts: list[Any] = []
        text = text_of(document)
        if text is not None:
            parts.append(TextPart(text=text))
        if document.image is not None:
            parts.append(ImagePart(ref=media_ref(case, document.image)))
        if document.video is not None:
            parts.append(VideoPart(ref=media_ref(case, document.video)))
        contents.append(Content.from_parts(parts) if parts else Content.from_text(""))
    return contents


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


def _compare(case: Case, matrix: Any, *, report: tuple[str, Any] | None = None) -> CaseResult:
    """The matrix against ``expected`` under the tolerance: the case's verdict and its detail.

    ``report`` is the over-cap report-not-compare rule (``(deviation, cells)`` from
    :func:`_over_cap_report`): the named cells are excluded from every comparison and named in the detail;
    when every cell is reported the case compared nothing (like ``kind: none``), never a false failure.
    """
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
        return _compare_ranking(case, matrix, tolerance, report=report)
    if tolerance.abs is not None:
        return _compare_abs(case, matrix, tolerance.abs, report=report)
    return _compare_spearman(case, matrix, tolerance.spearman_min, report=report)  # type: ignore[arg-type]


def _over_cap_suffix(report: tuple[str, Any] | None) -> str:
    """The detail's tail naming the over-cap cells reported instead of compared (empty without a report)."""
    if report is None:
        return ""
    deviation, cells = report
    return f"; {int(np.asarray(cells).sum())} over-cap cell(s) reported, not compared (recipe declares {deviation})"


def _all_over_cap(case: Case, report: tuple[str, Any]) -> CaseResult:
    """Every cell is over-cap: the case ran the path and compared nothing (like ``kind: none``)."""
    deviation, cells = report
    return CaseResult(
        case_id=case.id,
        compared=False,
        passed=True,
        skipped=None,
        detail=(
            f"no comparison: all {int(np.asarray(cells).sum())} cell(s) are over-cap (the recipe declares "
            f"{deviation}): the client changed these inputs relative to the case's, so the expectation was "
            "not filled from what was sent (stage 2's report-not-compare rule)"
        ),
    )


def _compare_abs(case: Case, matrix: Any, bound: float, *, report: tuple[str, Any] | None = None) -> CaseResult:
    """Every cell's absolute difference within ``abs``; the detail names the worst cell.

    Under ``report`` the over-cap cells are excluded from the worst-delta reduction, never compared against
    an expectation the client's changed input did not produce.
    """
    delta = np.abs(matrix - np.asarray(case.expected.values, dtype=np.float64))
    if report is not None and bool(np.asarray(report[1]).all()):
        return _all_over_cap(case, report)
    if report is not None:
        delta = np.where(np.asarray(report[1]), -np.inf, delta)
    worst = float(np.max(delta))
    suffix = _over_cap_suffix(report)
    if worst <= bound:
        return CaseResult(
            case_id=case.id,
            compared=True,
            passed=True,
            skipped=None,
            detail=f"max abs delta {worst:.3g} <= abs {bound:.3g}{suffix}",
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
            f"got {matrix[row, column]:.6g}, expected {case.expected.values[row][column]:.6g}{suffix}"
        ),
    )


def _compare_spearman(
    case: Case, matrix: Any, threshold: float, *, report: tuple[str, Any] | None = None
) -> CaseResult:
    """Every row's rank correlation against the expected row, bounded from below.

    A constant row (a degenerate output) gives a NaN correlation, which is never a pass: the worst-row
    tracking selects it wherever it sits (``rho < worst[0]`` alone never replaced a finite minimum). Under
    ``report`` a row's over-cap cells are dropped from the correlation; a row with fewer than two under-cap
    cells is reported, not correlated.
    """
    cells = np.asarray(report[1]) if report is not None else None
    if cells is not None and bool(cells.all()):
        assert report is not None
        return _all_over_cap(case, report)
    worst: tuple[float, int] | None = None
    for index, (got_row, want_row) in enumerate(zip(matrix, case.expected.values, strict=True)):
        if cells is not None:
            keep = ~cells[index]
            got_row = np.asarray(got_row)[keep]
            want_row = np.asarray(want_row, dtype=np.float64)[keep]
        if len(got_row) < 2:
            continue  # every cell of this row (or all but one) is over-cap: no order to compare
        rho = spearman(list(got_row), list(want_row))
        if worst is None or math.isnan(rho) or rho < worst[0]:
            worst = (rho, index)
    if worst is None:
        if report is not None:
            return CaseResult(
                case_id=case.id,
                compared=False,
                passed=True,
                skipped=None,
                detail=(f"no comparison: every query holds fewer than two under-cap cells{_over_cap_suffix(report)}"),
            )
        raise ConformanceError(f"case {case.id} holds no queries")  # pragma: no cover - the load requires one
    rho, index = worst
    suffix = _over_cap_suffix(report)
    if rho >= threshold:
        return CaseResult(
            case_id=case.id,
            compared=True,
            passed=True,
            skipped=None,
            detail=f"worst spearman {rho:.4g} >= spearman_min {threshold:.4g}{suffix}",
        )
    return CaseResult(
        case_id=case.id,
        compared=True,
        passed=False,
        skipped=None,
        detail=(f"spearman {rho:.4g} < spearman_min {threshold:.4g} on query {case.inputs.queries[index].id}{suffix}"),
    )


def _compare_ranking(case: Case, matrix: Any, tolerance: Any, *, report: tuple[str, Any] | None = None) -> CaseResult:
    """The ranking kind: the derived order per query against the case's expected order.

    ``rank_exact``: the derived order (ties toward the lower document index -- the package's rule) must
    equal the expected row exactly. ``spearman_min``: the expected row may be a top-k prefix; its documents'
    positions in the derived order correlate with ``1..k`` from below. Under ``report`` the over-cap
    documents are dropped from the derived order and the expectation; a query with fewer than two under-cap
    documents is reported, not ranked.
    """
    doc_ids = [document.id for document in case.inputs.documents]
    cells = np.asarray(report[1]) if report is not None else None
    if cells is not None and bool(cells.all()):
        assert report is not None
        return _all_over_cap(case, report)
    suffix = _over_cap_suffix(report)
    if tolerance.rank_exact:
        compared = 0
        for index, (row, expected_row) in enumerate(zip(matrix, case.expected.values, strict=True)):
            keep = [position for position in range(len(doc_ids)) if cells is None or not cells[index, position]]
            if not keep:
                continue
            compared += 1
            derived = _ranking_of(np.asarray(row)[keep], [doc_ids[position] for position in keep])
            under_cap = set(derived)
            want = [doc_id for doc_id in expected_row if doc_id in under_cap]
            if derived != want:
                return CaseResult(
                    case_id=case.id,
                    compared=True,
                    passed=False,
                    skipped=None,
                    detail=(
                        f"the ranking of query {case.inputs.queries[index].id} is {derived}, expected "
                        f"{want} (rank_exact){suffix}"
                    ),
                )
        if not compared:
            return CaseResult(
                case_id=case.id,
                compared=False,
                passed=True,
                skipped=None,
                detail=(
                    f"no comparison: every query holds fewer than two under-cap documents{_over_cap_suffix(report)}"
                ),
            )
        return CaseResult(
            case_id=case.id,
            compared=True,
            passed=True,
            skipped=None,
            detail=f"every query's ranking matches rank-exactly{suffix}",
        )
    threshold = tolerance.spearman_min
    worst: tuple[float, int] | None = None
    for index, (row, expected_row) in enumerate(zip(matrix, case.expected.values, strict=True)):
        keep = [position for position in range(len(doc_ids)) if cells is None or not cells[index, position]]
        if not keep:
            continue
        derived = _ranking_of(np.asarray(row)[keep], [doc_ids[position] for position in keep])
        under_cap = set(derived)
        want = [doc_id for doc_id in expected_row if doc_id in under_cap]
        k = len(want)
        if k < 2:
            if cells is None:
                return CaseResult(
                    case_id=case.id,
                    compared=True,
                    passed=False,
                    skipped=None,
                    detail=(
                        f"the expected ranking row {expected_row} of query {case.inputs.queries[index].id} "
                        "holds fewer than two documents; a one-document expectation compares rank_exact, "
                        "not spearman_min (it would pass free)"
                    ),
                )
            continue  # fewer than two under-cap documents left: reported, not ranked
        if set(want) != set(derived[:k]):
            return CaseResult(
                case_id=case.id,
                compared=True,
                passed=False,
                skipped=None,
                detail=(
                    f"the expected top-{k} prefix {want} is not the derived top-{k} "
                    f"{derived[:k]} on the ranking of query {case.inputs.queries[index].id} "
                    "(a ranking row names the derived top-k: its documents hold the top-k positions)"
                    f"{suffix}"
                ),
            )
        position = {doc_id: rank for rank, doc_id in enumerate(derived)}
        got = [float(position[doc_id]) for doc_id in want]
        rho = spearman(got, list(range(k)))
        if worst is None or rho < worst[0]:
            worst = (rho, index)
    if worst is None:
        if report is not None:
            return CaseResult(
                case_id=case.id,
                compared=False,
                passed=True,
                skipped=None,
                detail=(
                    f"no comparison: every query holds fewer than two under-cap documents{_over_cap_suffix(report)}"
                ),
            )
        raise ConformanceError(f"case {case.id} holds no queries")  # pragma: no cover - the load requires one
    rho, index = worst
    if rho >= threshold:
        return CaseResult(
            case_id=case.id,
            compared=True,
            passed=True,
            skipped=None,
            detail=f"worst expected-prefix spearman {rho:.4g} >= spearman_min {threshold:.4g}{suffix}",
        )
    return CaseResult(
        case_id=case.id,
        compared=True,
        passed=False,
        skipped=None,
        detail=(
            f"spearman {rho:.4g} < spearman_min {threshold:.4g} on the ranking of query "
            f"{case.inputs.queries[index].id}{suffix}"
        ),
    )


def _ranking_of(row: Any, doc_ids: list[str]) -> list[str]:
    """The document ids of one score row, best first, ties toward the lower document index."""
    order = sorted(range(len(row)), key=lambda index: (-float(row[index]), index))
    return [doc_ids[index] for index in order]
