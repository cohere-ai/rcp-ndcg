"""The rerank wire adapter: the Cohere-shaped ``POST {base_url}/rerank``, its hosted profiles and its answer shapes.

Every engine and API of the rerank role speaks one request shape: ``model``, ``query``, ``documents`` and
``top_n`` (the number of documents, so every document sent is scored -- a rerank request is one query's whole
candidate set, which lets an engine reuse the query's prefix across the documents and is what a listwise model
needs). vLLM, Infinity and Cohere answer ``{"results": [{"index", "relevance_score"}]}`` as is; Voyage answers
``{"data": [...]}`` with ``top_k`` as its return-limit field instead of ``top_n``; SGLang answers a bare list of
``{"index", "score"}`` rows. One adapter family builds the request and parses all three answer shapes,
realigning every score to the request's documents by ``index``: the answers come back ranked, so reading them
positionally would silently permute the association between scores and documents.

The adapters are stateless apart from the role config they are built with, and a config's ``api`` field selects
one by its registered name:

* ``api: rerank`` (the default of :class:`~rcp_ndcg.inference.config.RerankEndpoint`) -- a served engine
  (vLLM ``/rerank``) at the endpoint's own ``base_url``. Carries the engine's extensions when the config sets
  them: ``instruction`` (the engine's own request field, for ``instruction: field``) and ``use_activation``.
  No per-request document cap: the engine scores the whole candidate set.
* ``api: cohere`` -- Cohere's hosted rerank (``https://api.cohere.com/v2/rerank``), at most 1000 documents per
  request (the vendor's recommendation, declared policy: a longer candidate set is split into requests and
  merged).
* ``api: voyage`` -- Voyage's hosted rerank (``https://api.voyageai.com/v1/rerank``), at most 1000 documents
  per request, and its requests spaced by half a second (Voyage enforces strict rate limits).

The hosted profiles have no instruction field on the wire; the config's ``instruction`` mode decides how the
query text is built, and that rule is the client's (:class:`~rcp_ndcg.inference.clients.rerank.RerankClient`):
``fold`` folds the instruction into the query text, ``field`` needs the engine's field (served only), ``none``
sends the bare query. Text budgets are a client concern (:meth:`RerankClient._prepare`); an adapter sends its
request bodies exactly as built and cuts nothing.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, ClassVar, NoReturn

from rcp_ndcg_core.content import Content

from rcp_ndcg.errors import CapabilityError, ConfigError, ProviderError, RequestRejectedError
from rcp_ndcg.inference.adapters.base import AdapterRole, register_adapter
from rcp_ndcg.inference.config import RerankEndpoint
from rcp_ndcg.inference.types import Call, Reply, RerankRequest, RerankResult, TokenCount

#: A 400/422 body naming the thing a smaller client-side budget would fix. The endpoint's own wording varies
#: (vLLM: "This model's maximum context length is ... tokens"; TEI and the hosted APIs word it differently), so
#: the marker list is broad; a false positive costs nothing -- the request is refused either way.
_TOO_LONG = re.compile(r"maximum context length|context length|too long|token limit|max_tokens|truncat", re.IGNORECASE)


#: The class attributes that make a subclass a complete wire (validated at construction, so an incomplete
#: third-party profile fails with a typed error instead of an AttributeError at first use).
_WIRE_FACTS = (
    "SERVER",
    "REQUEST_CAP",
    "PAUSE_S",
    "SENDS_TOP_N",
    "HAS_INSTRUCTION_FIELD",
    "DEFAULT_BASE_URL",
    "HOSTED",
    "API_KEY_ENV",
    "KEY_REQUIRED",
    "AUTH_HEADER",
)


def _score_input(content: Content) -> str | dict[str, Any]:
    """One side of a scoring request, in the shape the served rerank engines take.

    A plain string when there is no media, so a text-only server receives exactly the request it would without
    content parts; ``{"content": [parts]}`` otherwise, the one OpenAI-shaped lowering every engine accepts.

    Args:
        content: The query or document as content parts, as the caller gave it.

    Returns:
        The wire value: a plain string, or the OpenAI content-parts object for a media part.
    """
    if not content.has_media:
        return content.text
    from rcp_ndcg.data.media import content_parts_payload

    return {"content": content_parts_payload(content)}


def _short(body: Any, limit: int = 300) -> str:
    """The first ``limit`` characters of a body's repr, for an error message (never a whole payload)."""
    text = repr(body)
    return text if len(text) <= limit else f"{text[:limit]}..."


class RerankWire:
    """Everything the Cohere-shaped rerank wires share: the body, the split at the profile's cap, the answer.

    Subclasses are the profiles: the class attributes below are their wire facts, and a config's ``api``
    selects one by its registered name. An adapter holds the config it serves (the config's fields decide the
    request), so it is instantiated per client, not shared. A third party's rerank adapter is selectable for
    the role by subclassing (or matching) this shape alongside :class:`Adapter`.
    """

    name: ClassVar[str]
    """The adapter's name, the value a config's ``api`` field holds."""

    role: ClassVar[AdapterRole] = "rerank"
    """The role the adapter serves: one query with its documents, one score per document."""

    SERVER: ClassVar[str]
    """How error messages name the server (``"Cohere"``, ``"the rerank endpoint at http://..."``)."""

    REQUEST_CAP: ClassVar[int | None]
    """Documents per request; ``None`` sends the whole candidate set (a served engine has no count cap)."""

    PAUSE_S: ClassVar[float]
    """Seconds between a query's requests (Voyage enforces strict rate limits); ``0.0``: no pause."""

    SENDS_TOP_N: ClassVar[bool]
    """Whether the body carries ``top_n`` (Cohere-shaped). Voyage's field is ``top_k``, and it returns
    every document by default, so its profile sends no field at all -- the client wants one score per
    document."""

    HAS_INSTRUCTION_FIELD: ClassVar[bool]
    """Whether the wire has the engine's own ``instruction`` request field (the vLLM extension)."""

    DEFAULT_BASE_URL: ClassVar[str | None]
    """The hosted profile's public API root, used when the config sets no ``base_url``; ``None``: ``base_url``
    is required (a served endpoint has no public root)."""

    HOSTED: ClassVar[bool]
    """Whether this wire is a hosted vendor profile (its public API root is its default ``base_url``; its
    score scale is the vendor's own). Declared (R8), never inferred from whether a default URL happens to be
    set: a served wire's ``use_activation`` is refused on a hosted profile, where the field does not exist."""

    API_KEY_ENV: ClassVar[tuple[str, ...]]
    """The environment variables that may hold the API key, most preferred first; the config's
    ``api_key_env`` names one instead. The transport resolves the key and sends it in :attr:`AUTH_HEADER`
    (R6): an adapter never touches a key itself. Empty: the endpoint takes no key (a served engine)."""

    KEY_REQUIRED: ClassVar[bool]
    """Whether the API refuses to answer without a key (the hosted profiles) or takes none."""

    AUTH_HEADER: ClassVar[str | None]
    """The header the key goes in; ``None`` is the OpenAI-standard ``Authorization: Bearer <key>``."""

    def __init__(self, config: RerankEndpoint) -> None:
        """Build the adapter for ``config``.

        Args:
            config: The role config whose ``api`` selected this adapter; its content fields (the instruction
                mode, ``use_activation``) decide what the request bodies carry.

        Raises:
            ConfigError: The config asks for something this wire cannot carry: an ``instruction: field`` on a
                hosted API with no such field, or ``use_activation`` on a hosted API (their score scale is
                their own and cannot be switched off).
        """
        self.config = config
        missing = [fact for fact in _WIRE_FACTS if not hasattr(type(self), fact)]
        if missing:
            raise ConfigError(
                f"{type(self).__name__} subclasses RerankWire without its wire facts: {', '.join(missing)}",
                hint="every RerankWire subclass declares SERVER, REQUEST_CAP, PAUSE_S, SENDS_TOP_N, "
                "HAS_INSTRUCTION_FIELD and DEFAULT_BASE_URL as class attributes",
            )
        if config.instruction == "field" and not self.HAS_INSTRUCTION_FIELD:
            raise ConfigError(
                f"the {self.name!r} rerank API has no instruction field on its wire",
                hint=(
                    "set instruction: fold to fold the instruction into the query text, or instruction: none to drop it"
                ),
            )
        if config.use_activation is not None and self.HOSTED:
            raise ConfigError(
                f"the {self.name!r} rerank API has no use_activation field on its wire: it scores on its own "
                "scale, which the package records as returned",
                hint="leave use_activation unset for a hosted profile; it is a served engine's (vLLM) extension",
            )
        where = f" at {config.base_url}" if config.base_url is not None else ""
        self._server = f"{self.SERVER}{where}"

    # -- the adapter seam ---------------------------------------------------
    def calls(self, request: RerankRequest, *, model: str) -> Sequence[Call]:
        """The HTTP calls ``request`` becomes: one per request-sized chunk of the candidate set.

        Args:
            request: One query with its candidate documents (the query text already prepared per the config's
                instruction mode).
            model: The endpoint's model name, sent as the request's ``model``.

        Returns:
            One call per chunk, in document order; a candidate set at or below the profile's cap is one call.
        """
        sizes = self._chunk_sizes(request)
        calls: list[Call] = []
        start = 0
        for size in sizes:
            chunk = request.documents[start : start + size]
            body: dict[str, Any] = {
                "model": model,
                "query": _score_input(request.query),
                "documents": [_score_input(document) for document in chunk],
            }
            if self.SENDS_TOP_N:
                body["top_n"] = size
            # The engine's extensions, only when the request carries them: a plain Cohere-shaped server never
            # receives an unknown field. The client sets `instruction` only for instruction: field.
            if request.instruction is not None:
                body["instruction"] = request.instruction
            if self.config.use_activation is not None:
                body["use_activation"] = self.config.use_activation
            calls.append(Call("POST", "/rerank", body))
            start += size
        return calls

    def interpret(self, request: RerankRequest, replies: Sequence[Reply]) -> RerankResult:
        """The scores of ``request``, aligned to its documents by position, from the replies of :meth:`calls`.

        Each reply's rows are realigned by their ``index`` (the answers come back ranked, so arrival order is
        not the request's document order) and the chunks' scores are concatenated back into the request's
        document order.

        Args:
            request: The request the replies answer.
            replies: One reply per call of :meth:`calls`, in order.

        Returns:
            One relevance score per document of ``request``, in the request's order, as the server returned
            them (a probability when the endpoint activates its score, a raw logit when it does not).

        Raises:
            CapabilityError: The endpoint refused the request as too long (HTTP 400/422 naming the length).
            RequestRejectedError: The endpoint refused this one request another way.
            ProviderError: An answer is unusable (an unrecognised body, an index missing, duplicated or out of
                range); never retried, since the same request would fail the same way.
        """
        sizes = self._chunk_sizes(request)
        if len(replies) != len(sizes):
            raise ProviderError(
                f"{self._server} returned {len(replies)} reply/replies for the {len(sizes)} request(s) the "
                f"candidate set of {len(request.documents)} document(s) splits into",
                retryable=False,
            )
        scores: list[float] = []
        for size, reply in zip(sizes, replies, strict=True):
            scores.extend(self._chunk_scores(reply, size))
        return RerankResult.aligned(request, scores)

    def usage(self, reply: Reply) -> TokenCount | None:
        """The tokens one reply reports, or ``None``: the rerank APIs report no per-role token split.

        A body that carries OpenAI-style ``usage.prompt_tokens`` / ``usage.completion_tokens`` (a served
        engine's) is read; anything else -- a hosted API's totals, or nothing -- reports no tokens.
        """
        body = reply.body
        usage = body.get("usage") if isinstance(body, dict) else None
        if not isinstance(usage, dict):
            return None
        prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
        # JSON booleans are ints to isinstance, but "prompt_tokens": true is no count.
        prompt = prompt if type(prompt) is int else None
        completion = completion if type(completion) is int else None
        if prompt is None and completion is None:
            return None
        return TokenCount(input_tokens=prompt, output_tokens=completion)

    # -- the shared wire logic ----------------------------------------------
    def _chunk_sizes(self, request: RerankRequest) -> list[int]:
        """One size per request the candidate set splits into: chunks of the config's ``batch_size`` documents
        (a pointwise model's request size, runtime), never more than the profile's cap.

        Raises:
            CapabilityError: The set needs splitting and the config is ``listwise``: a listwise model scores
                the whole candidate set in one prompt, and splitting it would change the scores.
        """
        total = len(request.documents)
        limit = self.REQUEST_CAP
        if self.config.batch_size is not None:
            limit = self.config.batch_size if limit is None else min(limit, self.config.batch_size)
        if total == 0:
            return []  # no candidate, no request
        if limit is None or total <= limit:
            return [total]
        if self.config.listwise:
            raise CapabilityError(
                f"{self._server} takes at most {limit} documents per request, but the reranker is listwise: "
                f"its candidate set of {total} documents must be scored in one request",
                hint="rerank against a pointwise server, or serve the listwise model behind an engine that "
                "takes the whole candidate set (vLLM /rerank does)",
                details={"documents": total, "cap": limit, "listwise": True},
            )
        return [min(limit, total - start) for start in range(0, total, limit)]

    def _chunk_scores(self, reply: Reply, size: int) -> list[float]:
        """One chunk's scores, in the chunk's document order, realigned from the reply's ranked rows.

        Args:
            reply: The reply of one call.
            size: The number of documents the call carried.

        Returns:
            One score per document of the chunk, in the chunk's order.

        Raises:
            ProviderError: The answer is unusable: an unrecognised body, a row without a usable index or
                score, an index duplicated or out of range, or a document with no score at all. Non-retryable
                in every case, and each names the server.
        """
        rows = self._rows(reply)
        scores: list[float | None] = [None] * size
        for row in rows:
            index = self._index(row, size)
            if scores[index] is not None:
                raise ProviderError(
                    f"{self._server} returned index {index} twice in one answer "
                    f"({len(rows)} rows for {size} documents)",
                    retryable=False,
                    details={"server": self._server, "index": index},
                )
            scores[index] = self._score(row)
        missing = [index for index, score in enumerate(scores) if score is None]
        if missing:
            raise ProviderError(
                f"{self._server} returned no score for {len(missing)} of {size} documents "
                f"(first: index {missing[0]}); every document sent must be scored",
                retryable=False,
                details={"server": self._server, "missing": missing[:5]},
            )
        return [float(score) for score in scores]  # type: ignore[arg-type]  # every slot is a float now

    def _rows(self, reply: Reply) -> list[dict[str, Any]]:
        """The answer's rows, in whatever of the three shapes the server answered.

        Raises:
            CapabilityError: The endpoint refused the request as too long.
            RequestRejectedError: The endpoint refused this one request another way.
            ProviderError: The answer is a 2xx body the adapter cannot read.
        """
        if reply.status != 200:
            self._refuse(reply)
        body = reply.body
        if isinstance(body, list):  # SGLang's (and TEI's) bare list of rows
            rows = body
        elif isinstance(body, dict) and isinstance(body.get("results"), list):  # Cohere v2, vLLM, Infinity
            rows = body["results"]
        elif isinstance(body, dict) and isinstance(body.get("data"), list):  # Voyage
            rows = body["data"]
        else:
            raise ProviderError(
                f"{self._server} answered the rerank request with a body the adapter cannot read: {_short(body)}",
                retryable=False,
                details={"server": self._server, "status": reply.status},
            )
        return rows

    def _refuse(self, reply: Reply) -> NoReturn:
        """Map a non-200 reply onto the role's errors: too long is a capability, everything else is a refusal.

        Raises:
            CapabilityError: HTTP 400/422 whose body names a length: the request needs a smaller budget
                (``max_tokens``), which is the one knob that bounds it.
            RequestRejectedError: Any other refusal; specific to this request, never retried.
        """
        message = _reply_message(reply)
        if reply.status in (400, 422) and _TOO_LONG.search(message):
            raise CapabilityError(
                f"{self._server} refused the rerank request as too long (HTTP {reply.status}): {message}",
                hint=(
                    "the query and its documents exceed what the endpoint scores in one request; declare the "
                    "endpoint config's text budget (tokenizer and max_tokens), which cuts the pair spans on "
                    "the client, or shorten the inputs"
                ),
                details={"server": self._server, "status": reply.status},
            )
        raise RequestRejectedError(
            f"{self._server} refused the rerank request (HTTP {reply.status}): {message}",
            details={"server": self._server, "status": reply.status},
        )

    def _index(self, row: dict[str, Any], size: int) -> int:
        """The ``index`` of one result row, validated against the request's size.

        Raises:
            ProviderError: The row is not an object, carries no integer ``index``, or one out of range.
        """
        if not isinstance(row, dict):
            raise ProviderError(
                f"{self._server} returned a result row that is not an object: {_short(row)}",
                retryable=False,
                details={"server": self._server},
            )
        index = row.get("index")
        if not isinstance(index, int) or isinstance(index, bool):
            raise ProviderError(
                f"{self._server} returned a result row without an integer index: {_short(row)}",
                retryable=False,
                details={"server": self._server},
            )
        if not 0 <= index < size:
            raise ProviderError(
                f"{self._server} returned index {index} for a request of {size} documents",
                retryable=False,
                details={"server": self._server, "index": index, "documents": size},
            )
        return index

    def _score(self, row: dict[str, Any]) -> float:
        """One row's relevance score: ``relevance_score``, or ``score`` where SGLang and TEI name it that.

        Raises:
            ProviderError: The row carries neither key, or the value is not a number.
        """
        raw = row.get("relevance_score", row.get("score"))
        if not isinstance(raw, int | float) or isinstance(raw, bool):
            raise ProviderError(
                f"{self._server} returned a result row without a numeric relevance_score (or score): {_short(row)}",
                retryable=False,
                details={"server": self._server},
            )
        return float(raw)


def _reply_message(reply: Reply) -> str:
    """The server's own words for a refusal: the body's message field, else the body's repr."""
    body = reply.body
    for key in ("message", "error", "detail"):
        if isinstance(body, dict) and isinstance(body.get(key), str):
            return str(body[key])
        if isinstance(body, dict) and isinstance(body.get(key), dict):
            nested = body[key]
            if isinstance(nested.get("message"), str):
                return str(nested["message"])
    return _short(body)


@register_adapter
class RerankAdapter(RerankWire):
    """The served rerank wire: the Cohere-shaped ``POST {base_url}/rerank`` of a vLLM (or Infinity) engine.

    One query's whole candidate set per request by default, whatever its size (the engine reuses the query's
    prefix across the documents, and a listwise model needs the set together); a config that sets
    ``batch_size`` asks for that many documents per request instead. The engine's extensions
    (``instruction``, ``use_activation``) travel only when the config or request sets them. The budgets are a
    client concern (:meth:`RerankClient._prepare` fits every request to the declared one), so no
    ``max_tokens_per_doc``, ``max_tokens_per_query`` or ``truncate_prompt_tokens`` is ever sent: a rendered
    prompt is never truncated by the engine.
    """

    name: ClassVar[str] = "rerank"
    SERVER = "the rerank endpoint"
    REQUEST_CAP = None
    PAUSE_S = 0.0
    SENDS_TOP_N = True
    HAS_INSTRUCTION_FIELD = True
    DEFAULT_BASE_URL = None
    HOSTED = False
    API_KEY_ENV = ()
    KEY_REQUIRED = False
    AUTH_HEADER = None


@register_adapter
class CohereRerankAdapter(RerankWire):
    """Cohere's hosted rerank: ``https://api.cohere.com/v2/rerank`` (``model``, ``query``, ``documents``,
    ``top_n``), answered as ``results``.

    At most 1000 documents per request (Cohere's recommendation, declared policy), or the config's
    ``batch_size`` when it is smaller: a longer candidate set is split into requests and the chunks' scores
    are merged back into one aligned result. The server cuts long documents itself (its
    ``max_tokens_per_doc`` defaults to 4096); a client-side budget for it is the text-budget mechanism's, not
    this adapter's.
    """

    name: ClassVar[str] = "cohere"
    SERVER = "the Cohere rerank API"
    REQUEST_CAP = 1000
    PAUSE_S = 0.0
    SENDS_TOP_N = True
    HAS_INSTRUCTION_FIELD = False
    DEFAULT_BASE_URL = "https://api.cohere.com/v2"
    HOSTED = True
    API_KEY_ENV = ("CO_API_KEY", "COHERE_API_KEY")
    KEY_REQUIRED = True
    AUTH_HEADER = None


@register_adapter
class VoyageRerankAdapter(RerankWire):
    """Voyage's hosted rerank: ``https://api.voyageai.com/v1/rerank`` (``model``, ``query``, ``documents``),
    answered as ``data``.

    At most 1000 documents per request (or the config's ``batch_size`` when smaller), and the requests of one
    query spaced by half a second (the pause of today's hosted Voyage path): Voyage enforces strict rate limits. Its
    return-limit field is ``top_k``; the client wants every document it sends scored, and Voyage returns all
    of them by default, so no field is sent.
    """

    name: ClassVar[str] = "voyage"
    SERVER = "the Voyage rerank API"
    REQUEST_CAP = 1000
    PAUSE_S = 0.5
    SENDS_TOP_N = False
    HAS_INSTRUCTION_FIELD = False
    DEFAULT_BASE_URL = "https://api.voyageai.com/v1"
    HOSTED = True
    API_KEY_ENV = ("VOYAGE_API_KEY",)
    KEY_REQUIRED = True
    AUTH_HEADER = None


__all__ = ["CohereRerankAdapter", "RerankAdapter", "RerankWire", "VoyageRerankAdapter"]
