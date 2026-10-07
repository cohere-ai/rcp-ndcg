"""The recorder: the product's role clients, observed through an ``httpx`` transport hook, for the contract
fixtures.

Against a served recipe it drives the product's role client
(:class:`~rcp_ndcg.inference.clients.EmbeddingClient`, ``PoolingClient`` or ``RerankClient``) built from
:func:`~rcp_ndcg_vllm.recipe.client_config` with the recipe's real budget -- the client prompts, fits and
settles, the adapter renders, the transport sends -- and records the engine behaviour the adapters must map:
the provenance ``GET /v1/models``, and on the role route an over-length prompt (measured against the engine's
own ``serve.max_model_len`` cap, sent bare: the client would cut it before the engine saw it) and an unknown
request field.  Written under ``<out>/<engine>-<version>/<recipe-id>/``::

    {"route": "http://engine/v1/embeddings", "request": {"url": ..., "body": {...}}, "status": 200,
     "headers": {"content-type": ..., "server": ...}, "body": ...}

Bytes bodies are base64-encoded with their framing headers kept.  No secret and no hostname is written: the URL
carries the placeholder host ``http://engine``.  The role route's request is the product's, byte for byte: it
crosses the capturing transport the client's own transport wraps (the product's injection point), never a
hand-built copy of the adapter's body.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import httpx

from .equivalence.wire import role_client
from .errors import HarnessError
from .recipe import Recipe

__all__ = ["bare_exchange", "entry_from_exchange", "record", "record_corpus"]

_PLACEHOLDER = "http://engine"
_TIMEOUT_S = 120.0
_SNIPPET_TEXT = "What is the capital of France?"
_SNIPPET_DOCUMENTS = ["Paris is the capital of France.", "Berlin is the capital of Germany."]

_ROLE_ROUTES = {"embed": "/v1/embeddings", "multi_vector": "/pooling", "rerank": "/rerank"}


def _engine_root(base_url: str) -> str:
    """The engine's root: a base URL ending in the version segment loses it (the routes are root-relative)."""
    root = base_url.rstrip("/")
    if root.endswith(("/v1", "/v2")):
        root = root.rsplit("/", 1)[0]
    return root


def record(
    recipe: Recipe,
    base_url: str,
    out_dir: str | Path,
    *,
    timeout_s: float = _TIMEOUT_S,
) -> list[Path]:
    """Record the engine's request set for ``recipe``; return the written file paths.

    Every exchange becomes ``<out>/<engine>-<version>/<recipe-id>/<method>-<route>-<status>.json``; a failed
    exchange is recorded like any other (its status and body are the fixture), except a connection error, which
    stops the recording with :class:`HarnessError`.  The set: ``GET /v1/models``, the role route (the product's
    request, through the role client), the over-length 400 and the unknown-field 400 on the role route.
    """
    root = _engine_root(base_url)
    out = Path(out_dir) / f"{recipe.engine.name}-{_version(recipe.engine.image)}" / recipe.id
    out.mkdir(parents=True, exist_ok=True)
    exchanges: list[dict[str, Any]] = []
    with httpx.Client(base_url=root, timeout=timeout_s) as bare:
        _record_one(bare, exchanges, "GET", "/v1/models", None)
        _record_role_request(recipe, base_url, exchanges)
        _record_errors(bare, recipe, exchanges)
    return [
        _write_exchange(out / f"{index:02d}-{_slug(exchange)}.json", exchange)
        for index, exchange in enumerate(exchanges)
    ]


def _record_one(http: httpx.Client, exchanges: list[dict[str, Any]], method: str, route: str | None, body: Any) -> None:
    """One bare exchange against ``route``; a connection error stops the recording, a status does not.

    The bare client records the engine-behaviour probes only (the provenance route and the error bodies): the
    role route's request comes from the product's role client, never from a hand-built copy.
    """
    if route is None:
        return
    try:
        response = http.request(method, route, json=body)
    except httpx.HTTPError as error:
        raise HarnessError(f"recording {method} {route} failed: {error}") from error
    exchanges.append(_exchange(f"{_PLACEHOLDER}{route}", method, body, response))


def _record_role_request(recipe: Recipe, base_url: str, exchanges: list[dict[str, Any]]) -> None:
    """The role route through the product's role client: the captured exchange is the product's request."""
    client, capture = role_client(recipe, base_url)
    if recipe.role == "rerank":
        client.rerank(_SNIPPET_TEXT, _SNIPPET_DOCUMENTS, instruction=_default_instruction(recipe))
    else:
        from rcp_ndcg_core.content import Content

        from rcp_ndcg.inference.types import EncodeRole

        role = EncodeRole.QUERY if "query" in _declared(recipe) else EncodeRole.DOCUMENT
        text = _SNIPPET_TEXT if role is EncodeRole.QUERY else _SNIPPET_DOCUMENTS[0]
        client.encode([Content.from_text(text)], role)
    for exchange in capture.exchanges:
        exchange["url"] = _placeholder(exchange["url"])
        exchanges.append(exchange)


def _default_instruction(recipe: Recipe) -> str | None:
    """The recipe's default instruction, sent as the request field when the mode sends one."""
    instruction = getattr(recipe.client, "default_instruction", None)
    mode = getattr(recipe.client, "instruction", None)
    return instruction if (mode == "field" and instruction) else None


def _declared(recipe: Recipe) -> list[str]:
    """The recipe's declared shapes (the side the recorder probes follows them)."""
    template = recipe.client.template
    return [str(shape) for shape in template.shapes()] if template is not None else ["document"]


def _record_errors(http: httpx.Client, recipe: Recipe, exchanges: list[dict[str, Any]]) -> None:
    """The error bodies the adapters map, on the recipe's role route: an over-length prompt and an unknown field.

    The over-length input is measured against the engine's own cap, ``serve.max_model_len`` -- the number the
    engine enforces -- so the probe genuinely crosses it and records the engine's 400.  The probes go bare (a
    deliberate refusal shape): the product's clients cut before the engine would refuse, and these fixtures
    document exactly what the adapters must map.
    """
    over_length = "a " * (recipe.serve.max_model_len * 2)
    route, over_length_body = _role_request(recipe.role, recipe.id, over_length)
    _record_one(http, exchanges, "POST", route, over_length_body)
    _, in_budget_body = _role_request(recipe.role, recipe.id, _SNIPPET_TEXT)
    _record_one(http, exchanges, "POST", route, {**in_budget_body, "unknown_field": "map-the-error"})


def _role_request(role: str, model: str, text: str) -> tuple[str, dict[str, Any]]:
    """The role route and its request with ``text`` as the input (the route the role's adapter speaks)."""
    routes: dict[str, tuple[str, dict[str, Any]]] = {
        "embed": (
            "/v1/embeddings",
            {"model": model, "input": [text], "encoding_format": "float"},
        ),
        "multi_vector": (
            "/pooling",
            {"model": model, "input": [text], "task": "token_embed", "encoding_format": "float"},
        ),
        "rerank": ("/rerank", {"model": model, "query": text, "documents": _SNIPPET_DOCUMENTS}),
    }
    return routes[role]


def _exchange(url: str, method: str, request_body: Any, response: httpx.Response) -> dict[str, Any]:
    """One recorded exchange from the raw httpx response."""
    response.read()
    raw = response.content
    try:
        response_json: Any = response.json()
    except ValueError:
        response_json = None
    return {
        "url": _placeholder(url),
        "method": method,
        "request_body": request_body,
        "status": response.status_code,
        "headers": {key: response.headers.get(key, "") for key in ("content-type", "server")},
        "response_bytes": base64.b64encode(raw).decode("ascii"),
        "response_json": response_json,
    }


def _placeholder(url: str) -> str:
    """The engine URL with the placeholder host (no hostname is ever recorded); the path is kept."""
    for scheme in ("https://", "http://"):
        if url.startswith(scheme):
            return _PLACEHOLDER + url[url.find("/", len(scheme)) :]
    return _PLACEHOLDER + "/" + url.removeprefix("/")


def _version(image: str) -> str:
    """The engine version from the image tag: ``vllm/vllm-openai:v0.31.0`` -> ``0.31.0``."""
    _, _, tag = image.rpartition(":")
    return tag.removeprefix("v") or "unknown"


def _write_exchange(path: Path, exchange: dict[str, Any]) -> Path:
    """One exchange file; a JSON body is kept decoded, a binary body base64 with its framing headers."""
    body: Any
    if exchange["response_json"] is not None:
        body = exchange["response_json"]
    else:
        framing = {
            key: value
            for key, value in exchange["headers"].items()
            if key.lower() in ("content-type", "content-length")
        }
        body = {"base64": exchange["response_bytes"], "framing_headers": framing}
    document = {
        "route": exchange["url"],
        "request": {"url": exchange["url"], "body": exchange["request_body"]},
        "status": exchange["status"],
        "headers": exchange["headers"],
        "body": body,
    }
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _slug(exchange: dict[str, Any]) -> str:
    """The exchange's file stem: the method and the URL path (``post-v1-embeddings-400``)."""
    path = exchange["url"].removeprefix(_PLACEHOLDER).strip("/")
    return f"{exchange['method'].lower()}-{path.replace('/', '-')}-{exchange['status']}"


# ---------------------------------------------------------------------------
# The observation corpus (OBSERVATIONS-SPEC sections 1-6): raw-first records, provenance, the checks.
# ---------------------------------------------------------------------------

_ENTRY_HEADERS = ("content-type", "server", "x-vllm-version", "metadata")
"""The response headers that matter: the content type, the server and its version header, and ``metadata`` (a
``/pooling`` ``bytes`` reply's framing -- its vectors do not decode without it).  Anything else, authorisation
and gateway headers included, is stripped before a record is written."""

_REQUEST_HEADERS = ("content-type",)


def _entry_body(raw: bytes, parsed: Any) -> dict[str, Any]:
    """One body as sent or received: raw (UTF-8 text, else ``{"base64": ...}``) and its parsed JSON."""
    try:
        body_raw: Any = raw.decode("utf-8")
    except UnicodeDecodeError:
        body_raw = {"base64": base64.b64encode(raw).decode("ascii")}
    return {"body_raw": body_raw, "body_parsed": parsed}


def entry_from_exchange(exchange: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """One captured exchange (:class:`~rcp_ndcg_vllm.equivalence.wire.CapturingTransport`'s or a bare probe's)
    as a corpus record's ``request`` and ``response`` (OBSERVATIONS-SPEC section 3).

    Only the headers that matter cross; authorisation and gateway headers never reach a record, and the URL
    becomes a root-relative path (no hostname is recorded).
    """
    url = str(exchange["url"])
    path = "/" + url.split("://", 1)[-1].split("/", 1)[-1] if "://" in url else url
    raw_request = base64.b64decode(exchange.get("request_bytes", ""))
    request = {
        "method": exchange["method"],
        "path": path.split("?", 1)[0],
        "headers": {key: "application/json" for key in _REQUEST_HEADERS if raw_request},
        **_entry_body(raw_request, exchange.get("request_body")),
    }
    headers = exchange.get("headers", {})
    response = {
        "status": exchange["status"],
        "headers": {key: value for key, value in headers.items() if key in _ENTRY_HEADERS and value},
        **_entry_body(base64.b64decode(exchange["response_bytes"]), exchange.get("response_json")),
        "latency_s": exchange.get("latency_s"),
    }
    return request, response


def bare_exchange(
    http: httpx.Client, method: str, route: str, body: Any, *, raw: bytes | None = None
) -> dict[str, Any]:
    """One bare probe (deliberate refusals, framing variants, the readiness edge) as a captured exchange.

    Inputs: an ``httpx`` client on the engine's root, the method and route, the JSON body (``None`` for
    none) or the exact ``raw`` bytes to send.  Output: the exchange in the capture's shape (request and
    reply bytes base64, the parsed bodies, ``latency_s``); a connection error is recorded as a status-less
    exchange (the request set's answer was "no answer"), never raised."""
    import time

    request_bytes = raw if raw is not None else json.dumps(body).encode("utf-8") if body is not None else b""
    headers = {"content-type": "application/json"} if request_bytes else {}
    started = time.monotonic()
    try:
        response = http.request(method, route, content=request_bytes or None, headers=headers)
    except httpx.HTTPError as error:
        return {
            "url": f"{_PLACEHOLDER}{route}",
            "method": method,
            "request_bytes": base64.b64encode(request_bytes).decode("ascii"),
            "request_body": body,
            "status": None,
            "headers": {},
            "response_bytes": base64.b64encode(f"{type(error).__name__}".encode()).decode("ascii"),
            "response_json": None,
            "latency_s": time.monotonic() - started,
        }
    return {
        "url": f"{_PLACEHOLDER}{route}",
        "method": method,
        "request_bytes": base64.b64encode(request_bytes).decode("ascii"),
        "request_body": body,
        "status": response.status_code,
        "headers": {key: response.headers.get(key, "") for key in _ENTRY_HEADERS},
        "response_bytes": base64.b64encode(response.content).decode("ascii"),
        "response_json": _safe_json(response.content),
        "latency_s": time.monotonic() - started,
    }


def _safe_json(payload: bytes) -> Any:
    try:
        return json.loads(payload)
    except (ValueError, UnicodeDecodeError):
        return None


class _Collector:
    """One corpus recording: the records in sending order, the pass being sent and its engine run."""

    def __init__(self, recipe: Recipe, tokenizer: Any) -> None:
        self.recipe = recipe
        self.tokenizer = tokenizer
        self.records: list[dict[str, Any]] = []
        self.repetition = "same_process_1"
        self.server_run_id = ""

    def add(self, exchange: dict[str, Any], *, batch_context: dict[str, Any], inputs: dict[str, Any]) -> None:
        """One exchange as the next record of the current pass."""
        from .observe.corpus import build_record

        request, response = entry_from_exchange(exchange)
        self.records.append(
            build_record(
                sequence=len(self.records),
                repetition=self.repetition,
                batch_context=batch_context,
                server_run_id=self.server_run_id,
                request=request,
                response=response,
                inputs=inputs,
            )
        )

    def inputs(self, row: dict[str, Any], **extra: Any) -> dict[str, Any]:
        """What the request set meant by one row: ids, strata, source, layer, token counts, media."""
        return {
            "request_id": str(row.get("request_id", "")),
            "stratum": str(row.get("stratum", "")),
            "probe": "ok",
            "layer": "model",
            "source": row.get("_source") or row.get("source"),
            "strata": row.get("_strata") or row.get("strata") or [],
            "token_counts": {
                "query_tokens": self.tokenizer.count(row["query"]),
                "document_tokens": [self.tokenizer.count(document) for document in row["documents"]],
            },
            "media": row.get("media"),
            **extra,
        }


def record_corpus(
    recipe: Recipe,
    base_url: str,
    rows: list[dict[str, Any]],
    out_dir: str | Path,
    *,
    server_run_id: str,
    engine_facts: dict[str, Any],
    collector: dict[str, Any] | None = None,
    after_restart: tuple[str, str] | None = None,
    restart: Any | None = None,
    batch_sizes: tuple[int, ...] = (1, 2, 8, 32),
    plan_ids: list[str] | None = None,
    equivalence_exchanges: list[dict[str, Any]] | None = None,
    while_loading: list[dict[str, Any]] | None = None,
    hub_cache: str | Path | None = None,
    plugin_wheel: str | Path | None = None,
    timeout_s: float = _TIMEOUT_S,
) -> dict[str, Any]:
    """Record one observation corpus for ``recipe`` over the request plan's rows (OBSERVATIONS-SPEC 1-6).

    The whole set is sent twice in the same engine process (``same_process_1``, ``same_process_2``, both under
    ``server_run_id``) and once after an engine restart (``after_restart``, under the restarted engine's own
    run id): ``after_restart`` names the restarted engine as ``(base_url, server_run_id)``, or ``restart()``
    stops and restarts it and returns that pair (``None``: the pass is recorded absent, with the reason).  The
    model layer is the product's role client on every row -- for an embedder each input text alone and in
    batches of ``batch_sizes`` per side, for a reranker each row with its candidates in the given and the
    reversed order --, captured at the product's transport seam.  Every pass also sends the protocol probes
    (routes and error bodies) and the engine's ``/tokenize`` of each sampled input.

    Inputs beyond the rows: the engine block (:func:`rcp_ndcg_vllm.observe.provenance.engine_facts`), the
    collector block (default: :func:`~rcp_ndcg_vllm.observe.provenance.collector_facts` without wave or job
    ids), the plan's request ids (default: the rows'), the equivalence stage's captured exchanges of the same
    wave (checked for consistency), and where the model facts read from (the Hub cache, the plugin wheel).

    Output: ``{"passed", "checks", "corpus_dir", "records"}``; the corpus directory holds ``records.jsonl``,
    ``nondeterminism.json`` and the hash-chained ``manifest.json`` with the section-4 provenance and the plan,
    and :func:`~rcp_ndcg_vllm.observe.corpus.verify_corpus` has accepted or refused it before this returns.
    """
    from .equivalence.fitting import tokenizer_of
    from .observe.corpus import summarise_nondeterminism, verify_corpus, write_corpus
    from .observe.provenance import collector_facts, model_facts, recipe_facts, unavailable
    from .observe.requests import CORPUS_PLAN_VERSION, corpus_plan

    started = _now()
    if not rows:
        raise HarnessError("record_corpus needs at least one request row")
    tokenizer = tokenizer_of(recipe)
    plan = corpus_plan(recipe, tokenizer, rows)
    collected = _Collector(recipe, tokenizer)
    passes: list[dict[str, Any]] = []

    def one_pass(repetition: str, url: str, run_id: str) -> None:
        collected.repetition, collected.server_run_id = repetition, run_id
        sent = _model_layer(recipe, url, plan.rows, collected, batch_sizes)
        _bare_probes(recipe, url, plan.rows, plan.bare, collected, timeout_s=timeout_s)
        _tokenize_probes(recipe, url, sent, collected, timeout_s=timeout_s)
        passes.append({"repetition": repetition, "server_run_id": run_id})

    one_pass("same_process_1", base_url, server_run_id)
    one_pass("same_process_2", base_url, server_run_id)
    restarted = after_restart if after_restart is not None else (restart() if restart is not None else None)
    strata = dict(plan.strata)
    if while_loading:
        collected.repetition = "after_restart"
        collected.server_run_id = restarted[1] if restarted else "loading"
        for exchange in while_loading:
            request_id = "edge:while_loading"
            collected.add(
                exchange,
                batch_context={"size": 1, "request_ids": [request_id], "positions": [0]},
                inputs={"request_id": request_id, "stratum": request_id, "probe": request_id, "layer": "protocol"},
            )
        strata["edge:while_loading"] = {"present": True}
    else:
        strata["edge:while_loading"] = {
            "present": False,
            "reason": "no request was sent while the engine loaded (the wave's restart sends it)",
        }
    if restarted:
        one_pass("after_restart", restarted[0], restarted[1])
    else:
        passes.append(
            {"repetition": "after_restart", "absent": "the engine did not restart (no restart given, or it failed)"}
        )

    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    nondeterminism = summarise_nondeterminism(collected.records, dim=getattr(recipe.client, "dim", None))
    (directory / "nondeterminism.json").write_text(
        json.dumps(nondeterminism, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    collector_block = collector or collector_facts(wave_id=None, job_id=None, started=started, finished=None)
    collector_block = {**collector_block, "finished": _now(), "module": "rcp_ndcg_vllm.record.record_corpus"}
    manifest = {
        "engine": engine_facts,
        "model": model_facts(recipe, hub_cache=hub_cache, plugin_wheel=plugin_wheel),
        "recipe": recipe_facts(recipe),
        "collector": {**collector_block, "batch_sizes": list(batch_sizes), "passes": passes},
        "plan": {
            "corpus_plan_version": CORPUS_PLAN_VERSION,
            "request_ids": list(plan_ids) if plan_ids is not None else plan.request_ids(),
            "strata": strata,
        },
    }
    if not restarted:
        manifest["collector"]["after_restart"] = unavailable("the engine did not restart; the pass is absent")
    write_corpus(directory, manifest, collected.records)
    report = verify_corpus(directory, equivalence_exchanges=equivalence_exchanges)
    return {
        "passed": report["passed"],
        "checks": report["checks"],
        "corpus_dir": str(directory),
        "records": len(collected.records),
    }


def _now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat(timespec="seconds")


def _model_layer(
    recipe: Recipe, url: str, rows: list[dict[str, Any]], collected: _Collector, batch_sizes: tuple[int, ...]
) -> list[tuple[dict[str, Any], str, str, str]]:
    """The model layer of one pass: the product's role client on every row, as the served path sends it.

    Returns the texts the client actually put on the wire for each input sent alone, as ``(row, item, shape,
    text)`` -- the rendered prompt for an embedder, the settled query span and the document spans for a
    reranker -- for the engine's ``/tokenize`` of exactly what the client sends.  A row carrying inline media
    (the media request set) sends each media side alone, with its media
    (:func:`~rcp_ndcg_vllm.equivalence.media.side_contents`).
    """
    from rcp_ndcg_core.content import Content

    from rcp_ndcg.inference.types import EncodeRole

    from .equivalence.fitting import declared_shapes
    from .equivalence.media import media_rows, side_contents

    client, capture = role_client(recipe, url)

    sent: list[tuple[dict[str, Any], str, str, str]] = []

    def captured_since(start: int, batch_context: dict[str, Any], inputs: dict[str, Any]) -> None:
        for exchange in capture.exchanges[start:]:
            collected.add(exchange, batch_context=batch_context, inputs=inputs)

    # The media rows (the media request set's inline images): each media side alone, as the media stage
    # sends it -- one item per request, a reranker's pair per request; their texts are not /tokenize-probed
    # (the engine's chat render of media is not a text the client sends).
    media = media_rows(rows)
    for _, row in media:
        query, documents = side_contents(row)
        context = {"size": 1, "request_ids": [row["request_id"]], "positions": [0], "media": True}
        for position, document in enumerate(documents):
            if not (document.has_media or (recipe.role == "rerank" and query.has_media)):
                continue
            start = len(capture.exchanges)
            if recipe.role == "rerank":
                client.rerank(query, [document], instruction=row.get("instruction"))
            else:
                client.encode([document], EncodeRole.DOCUMENT)
            captured_since(start, context, collected.inputs(row, item=f"document:{position}", side="document"))
        if query.has_media and recipe.role != "rerank":
            start = len(capture.exchanges)
            client.encode([query], EncodeRole.QUERY)
            captured_since(start, context, collected.inputs(row, item="query:0", side="query"))
    media_ids = {id(row) for _, row in media}
    rows = [row for row in rows if id(row) not in media_ids]

    if recipe.role == "rerank":
        # One query per /rerank request: the candidate set is the batch, sent in the given and the reversed order.
        for row in rows:
            for order in ("given", "reversed"):
                documents = list(row["documents"]) if order == "given" else list(reversed(row["documents"]))
                start = len(capture.exchanges)
                client.rerank(row["query"], documents, instruction=row.get("instruction"))
                context = {"size": 1, "request_ids": [row["request_id"]], "positions": [0], "order": order}
                captured_since(start, context, collected.inputs(row, order=order))
                if order == "given":
                    for exchange in capture.exchanges[start:]:
                        texts = capture.texts(exchange)
                        if texts.get("query"):
                            sent.append((row, "query", "pair", str(texts["query"])))
                        sent.extend((row, f"document:{k}", "pair", str(d)) for k, d in enumerate(texts["documents"]))
        return sent
    declared = set(declared_shapes(recipe))
    sides = [("query", EncodeRole.QUERY)] if "query" in declared else []
    if declared & {"document", "pair"} or not sides:
        sides.append(("document", EncodeRole.DOCUMENT))
    for side, role in sides:
        items: list[tuple[dict[str, Any], str, str]] = []
        for row in rows:
            texts = [row["query"]] if side == "query" else list(row["documents"])
            items.extend((row, f"{side}:{index}", text) for index, text in enumerate(texts))
        for size in batch_sizes:
            for start_index in range(0, len(items), size):
                group = items[start_index : start_index + size]
                context = {
                    "size": len(group),
                    "request_ids": [row["request_id"] for row, _, _ in group],
                    "items": [item for _, item, _ in group],
                    "positions": list(range(len(group))),
                }
                row, item, _ = group[0]
                inputs = collected.inputs(row, item=item, side=side)
                if len(group) > 1:
                    inputs["request_id"] = "batch:" + "+".join(f"{r['request_id']}/{i}" for r, i, _ in group)
                start = len(capture.exchanges)
                client.encode([Content.from_text(text) for _, _, text in group], role)
                captured_since(start, context, inputs)
                if size == batch_sizes[0] and len(group) == 1:
                    for exchange in capture.exchanges[start:]:
                        sent.extend((row, item, side, str(text)) for text in capture.texts(exchange)["input"])
    return sent


def _bare_probes(
    recipe: Recipe,
    base_url: str,
    rows: list[dict[str, Any]],
    planned: list[dict[str, Any]],
    collected: _Collector,
    *,
    timeout_s: float,
) -> None:
    """The bare requests of one pass: the collector's standing protocol probes (routes and error bodies) and
    the plan's bare rows (the uncut ladder and content kinds, the wire variants, the protocol edges).

    An unknown request field is answered 200 by vLLM v0.31.0 on every role route (measured in the shakedown);
    the corpus records what the engine answered and the acceptance check's status table expects the
    measurement.  ``wrong_model``, ``empty_input`` and the plan's rows carry no measured expectation.
    """
    route = _ROLE_ROUTES[recipe.role]
    over_length = "a " * (recipe.serve.max_model_len * 2)
    first = dict(rows[0])
    probes: list[tuple[str, str, str, Any, bytes | None]] = [
        ("ok", "GET", "/v1/models", None, None),
        ("ok", "GET", "/health", None, None),
        ("unknown_field", "POST", route, _role_body(recipe, first, unknown_field=True), None),
        ("malformed_json", "POST", route, None, b"{not json"),
        ("wrong_model", "POST", route, _role_body(recipe, first, model="not-a-model"), None),
        ("empty_input", "POST", route, _role_body(recipe, first, empty=True), None),
        (
            "over_length",
            "POST",
            route,
            _role_body(recipe, {**first, "query": over_length, "documents": [over_length]}),
            None,
        ),
    ]
    with httpx.Client(base_url=_engine_root(base_url), timeout=timeout_s) as http:
        for probe, method, path, body, raw in probes:
            request_id = f"probe:{probe}" if path == route or probe != "ok" else f"probe:{path}"
            inputs = collected.inputs(first, probe=probe, request_id=request_id, stratum="protocol", layer="protocol")
            collected.add(
                bare_exchange(http, method, path, body, raw=raw),
                batch_context={"size": 1, "request_ids": [request_id], "positions": [0]},
                inputs=inputs,
            )
        for row in planned:
            inputs = {
                "request_id": row["request_id"],
                "stratum": row["stratum"],
                "probe": row["probe"],
                "layer": row["layer"],
            }
            collected.add(
                bare_exchange(http, row["method"], row["path"], row.get("body"), raw=row.get("raw")),
                batch_context={"size": 1, "request_ids": [row["request_id"]], "positions": [0]},
                inputs=inputs,
            )


def _tokenize_probes(
    recipe: Recipe,
    base_url: str,
    sent: list[tuple[dict[str, Any], str, str, str]],
    collected: _Collector,
    *,
    timeout_s: float,
) -> None:
    """The engine's ``/tokenize`` of every text the client put on the wire (OBSERVATIONS-SPEC section 1): the
    exact prompt each input was sent as, with the special-tokens flag the shape is sent with, recorded as the
    ground truth for the client's ``fit`` and the fake engines' token counting."""
    from .equivalence.stages import _add_specials_flag, tokenize_url

    route = tokenize_url(base_url).removeprefix(_engine_root(base_url))
    seen: set[tuple[str, bool]] = set()
    with httpx.Client(base_url=_engine_root(base_url), timeout=timeout_s) as http:
        for row, item, shape, text in sent:
            flag = _add_specials_flag(recipe, shape) if recipe.role != "rerank" else False
            if (text, flag) in seen:
                continue
            seen.add((text, flag))
            request_id = f"tokenize:{row['request_id']}/{item}"
            inputs = collected.inputs(row, probe="tokenize", request_id=request_id, stratum="tokenize", item=item)
            collected.add(
                bare_exchange(http, "POST", route, {"model": recipe.id, "prompt": text, "add_special_tokens": flag}),
                batch_context={"size": 1, "request_ids": [request_id], "positions": [0]},
                inputs=inputs,
            )


def _role_body(
    recipe: Recipe,
    row: dict[str, Any],
    *,
    model: str | None = None,
    unknown_field: bool = False,
    empty: bool = False,
) -> dict[str, Any]:
    """One bare request body on the recipe's role route (the wire shape the adapter speaks)."""
    body: dict[str, Any]
    if recipe.role == "rerank":
        body = {
            "model": model or recipe.id,
            "query": "" if empty else row["query"],
            "documents": [] if empty else list(row["documents"]),
        }
    elif recipe.role == "embed":
        body = {"model": model or recipe.id, "input": [""] if empty else [row["query"]], "encoding_format": "float"}
    else:
        body = {
            "model": model or recipe.id,
            "input": [""] if empty else [row["query"]],
            "task": "token_embed",
            "encoding_format": "float",
        }
    if unknown_field:
        body["unknown_field"] = "map-the-error"
    return body
