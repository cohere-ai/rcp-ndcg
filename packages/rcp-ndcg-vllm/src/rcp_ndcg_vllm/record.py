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
from .observe.corpus import (
    build_record,
    summarise_nondeterminism,
    verify_corpus,
    write_corpus,
)
from .recipe import Recipe

__all__ = ["record", "record_corpus"]

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
# The observation corpus (OBSERVATIONS-SPEC sections 1-3): raw-first records.
# ---------------------------------------------------------------------------


def _entry_body(raw: bytes, parsed: Any) -> dict[str, Any]:
    """One request/response body as sent or received: raw bytes (base64 when not UTF-8) and its parsed JSON."""
    try:
        text = raw.decode("utf-8")
        body_raw: Any = text
    except UnicodeDecodeError:
        body_raw = {"base64": base64.b64encode(raw).decode("ascii")}
    return {"body_raw": body_raw, "body_parsed": parsed}


_ENTRY_HEADERS = ("content-type", "server", "x-vllm-version")
"""The response headers that matter (content type, the server's version header); anything else --
authorisation and gateway headers included -- is stripped before a record is written."""


def _entry_from_exchange(exchange: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """One captured exchange as the corpus record's ``request`` and ``response`` (section 3).

    Only the headers that matter cross (:data:`_ENTRY_HEADERS` on the response, content type on the
    request); authorisation and gateway headers never reach a record, and the URL is a path (no
    hostname is recorded).
    """
    raw_request = base64.b64decode(exchange.get("request_bytes", ""))
    request = {
        "method": exchange["method"],
        "path": exchange["url"].removeprefix(_PLACEHOLDER),
        "headers": {key: value for key, value in exchange.get("headers", {}).items() if key == "content-type"},
        **_entry_body(raw_request, exchange.get("request_body")),
    }
    response_bytes = base64.b64decode(exchange["response_bytes"])
    response = {
        "status": exchange["status"],
        "headers": {key: value for key, value in exchange.get("headers", {}).items() if key in _ENTRY_HEADERS},
        **_entry_body(response_bytes, exchange.get("response_json")),
        "latency_s": exchange.get("latency_s"),
    }
    return request, response


def _bare_exchange(
    http: httpx.Client, method: str, route: str, body: Any, *, raw: bytes | None = None
) -> dict[str, Any]:
    """One bare probe (deliberate refusals and framing variants) as a captured exchange."""
    request_bytes = raw if raw is not None else json.dumps(body).encode("utf-8") if body is not None else b""
    headers = {"content-type": "application/json"} if body is not None or raw is not None else {}
    payload = request_bytes if (body is not None or raw is not None) else None
    try:
        response = http.request(method, route, content=payload, headers=headers)
    except httpx.HTTPError as error:
        raise HarnessError(f"recording {method} {route} failed: {error}") from error
    return {
        "url": f"{_PLACEHOLDER}{route}",
        "method": method,
        "request_bytes": base64.b64encode(request_bytes).decode("ascii"),
        "request_body": body,
        "status": response.status_code,
        "headers": {key: response.headers.get(key, "") for key in ("content-type", "server")},
        "response_bytes": base64.b64encode(response.content).decode("ascii"),
        "response_json": _safe_json(response.content),
    }


def _safe_json(payload: bytes) -> Any:
    try:
        return json.loads(payload)
    except (ValueError, UnicodeDecodeError):
        return None


def record_corpus(
    recipe: Recipe,
    base_url: str,
    rows: list[dict[str, Any]],
    out_dir: str | Path,
    *,
    server_run_id: str,
    engine_facts: dict[str, Any] | None = None,
    after_restart_base_url: str | None = None,
    restart: Any | None = None,
    batch_sizes: tuple[int, ...] = (1, 2, 8, 32),
    timeout_s: float = _TIMEOUT_S,
) -> dict[str, Any]:
    """Record one observation corpus for ``recipe`` over the plan's rows (OBSERVATIONS-SPEC 1-3, 5-6).

    Every request is sent **twice in the same server process** and the whole set once more **after the
    engine restart** -- ``after_restart_base_url`` names the restarted engine, or ``restart()`` stops
    and restarts it and returns the URL (the wave's closure; ``None`` records the pass's absence
    with a note); each sampled input travels alone and inside batches of the declared sizes (bf16
    kernels can change numbers with batch composition), the rerank candidate set also in reverse
    order, and every sampled input's ``/tokenize`` reply (ids and count) is recorded as ground truth.
    The request/response record is
    raw-first (:data:`~rcp_ndcg_vllm.observe.corpus.RECORD_SCHEMA`; the role requests are the product
    role clients', captured at the product's transport seam); the bare probes cover the routes and the
    error bodies (over-length, unknown field, malformed JSON, wrong model name, empty input) with the
    status table that shake1c measured on vLLM v0.31.0.

    Output: the corpus report ``{"passed": ..., "corpus_dir": ...}``; the corpus directory holds
    ``records.jsonl``, ``nondeterminism.json`` and the hash-chained ``manifest.json``, and is checked
    by :func:`~rcp_ndcg_vllm.observe.corpus.verify_corpus` before this returns (a failing corpus is
    never silently accepted).
    """
    from .equivalence.fitting import tokenizer_of

    rows = [row for row in rows]
    if not rows:
        raise HarnessError("record_corpus needs at least one request row")
    tokenizer = tokenizer_of(recipe)
    collected: list[dict[str, Any]] = []
    pass_names: list[str] = []
    restarted_note = None
    sequence = 0

    def one_pass(repetition: str, url: str) -> None:
        """Send the whole set once against ``url`` (rows alone and batched, probes, /tokenize)."""
        nonlocal sequence
        client, capture = role_client(recipe, url)
        for size in batch_sizes:
            for group_start in range(0, len(rows), size):
                group = rows[group_start : group_start + size]
                request_ids = [str(row.get("request_id", group_start + offset)) for offset, row in enumerate(group)]
                batch_context = {
                    "size": len(group),
                    "request_ids": request_ids,
                    "positions": list(range(len(group))),
                }
                sequence = _send_row_group(
                    recipe,
                    client,
                    capture,
                    group,
                    collected,
                    sequence=sequence,
                    repetition=repetition,
                    batch_context=batch_context,
                    server_run_id=server_run_id,
                    tokenizer=tokenizer,
                )
                if recipe.role == "rerank" and size == 1:
                    reversed_row = {**group[0], "documents": list(reversed(list(group[0]["documents"])))}
                    sequence = _send_row_group(
                        recipe,
                        client,
                        capture,
                        [reversed_row],
                        collected,
                        sequence=sequence,
                        repetition=repetition,
                        batch_context={**batch_context, "probe": "order_reversed"},
                        server_run_id=server_run_id,
                        tokenizer=tokenizer,
                    )
        collected.extend(
            _bare_probes(
                recipe,
                url,
                rows,
                sequence,
                repetition=repetition,
                server_run_id=server_run_id,
                tokenizer=tokenizer,
                timeout_s=timeout_s,
            )
        )
        pass_names.append(repetition)

    # Twice in the same server process, THEN (and only then) the restart and the after-restart pass.
    one_pass("same_process", base_url)
    one_pass("same_process", base_url)
    if after_restart_base_url:
        one_pass("after_restart", after_restart_base_url)
    elif restart is not None:
        restarted_url = restart()
        if restarted_url:
            one_pass("after_restart", restarted_url)
        else:
            restarted_note = "the engine did not restart in time; the after_restart pass is absent"

    nondeterminism = summarise_nondeterminism(collected)
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "nondeterminism.json").write_text(
        json.dumps(nondeterminism, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "engine": engine_facts or {"version": "unknown"},
        "recipe": {"id": recipe.id, "model": recipe.model, "revision": recipe.revision},
        "collector": {
            "module": "rcp_ndcg_vllm.record.record_corpus",
            "server_run_id": server_run_id,
            "batch_sizes": list(batch_sizes),
            "repetitions": pass_names,
            **({"note": restarted_note} if restarted_note else {}),
        },
    }
    write_corpus(directory, manifest, collected)
    report = verify_corpus(directory)
    return {
        "passed": report["passed"],
        "checks": report["checks"],
        "corpus_dir": str(directory),
        "records": len(collected),
    }


def _send_row_group(
    recipe: Recipe,
    client: Any,
    capture: Any,
    group: list[dict[str, Any]],
    collected: list[dict[str, Any]],
    *,
    sequence: int,
    repetition: str,
    batch_context: dict[str, Any],
    server_run_id: str,
    tokenizer: Any,
) -> int:
    """One client call for one request group; the captured requests become corpus records in place."""
    started = __import__("time").monotonic()
    start = len(capture.exchanges)
    _drive_client(recipe, client, group)
    latency = __import__("time").monotonic() - started
    for offset, exchange in enumerate(capture.exchanges[start:]):
        request, response = _entry_from_exchange(exchange)
        response["latency_s"] = latency
        if len(group) > 1:
            joined = "+".join(str(row.get("request_id", "")) for row in group)
            row = {**group[0], "request_id": joined}
        else:
            row = group[min(offset, len(group) - 1)]
        collected.append(
            build_record(
                sequence=sequence + offset,
                repetition=repetition,
                batch_context=batch_context,
                server_run_id=server_run_id,
                request=request,
                response=response,
                inputs=_inputs_of(row, tokenizer),
            )
        )
    return sequence + max(len(capture.exchanges) - start, 1)


def _drive_client(recipe: Recipe, client: Any, group: list[dict[str, Any]]) -> None:
    """The product's role client sends one row group exactly as the served path would."""
    from rcp_ndcg_core.content import Content

    from rcp_ndcg.inference.types import EncodeRole

    if recipe.role == "rerank":
        for row in group:
            client.rerank(row["query"], list(row["documents"]), instruction=row.get("instruction"))
        return
    queries = [row["query"] for row in group if row["query"]]
    documents = [document for row in group for document in row["documents"]]
    from .equivalence.fitting import declared_shapes

    declared = set(declared_shapes(recipe))
    if queries and "query" in declared:
        client.encode([Content.from_text(text) for text in queries], EncodeRole.QUERY)
    if documents and "document" in declared:
        client.encode([Content.from_text(text) for text in documents], EncodeRole.DOCUMENT)
    if queries and not ("query" in declared or "document" in declared):
        client.encode([Content.from_text(text) for text in queries], EncodeRole.DOCUMENT)


def _inputs_of(row: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    """What the generator meant for one row: source ids, strata, token counts and declared media."""
    return {
        "request_id": str(row.get("request_id", "")),
        "stratum": str(row.get("stratum", "")),
        "probe": str(row.get("probe", "ok")),
        "source": row.get("_source") or row.get("source"),
        "strata": row.get("_strata") or row.get("strata") or [],
        "token_counts": {
            "query_tokens": tokenizer.count(row["query"]),
            "document_tokens": [tokenizer.count(document) for document in row["documents"]],
        },
        "media": row.get("media"),
    }


def _bare_probes(
    recipe: Recipe,
    base_url: str,
    rows: list[dict[str, Any]],
    sequence: int,
    *,
    repetition: str,
    server_run_id: str,
    tokenizer: Any,
    timeout_s: float,
) -> list[dict[str, Any]]:
    """The protocol probes (routes and error bodies) as bare calls: the refusals the adapters map.

    An unknown request field is answered 200 by vLLM v0.31.0 on every role route (measured, shake1c);
    the corpus records what the engine answered, and the acceptance check's status table expects the
    measurement.  ``wrong_model`` and ``empty_input`` are recorded with no measured expectation.
    """
    route = _ROLE_ROUTES[recipe.role]
    over_length = "a " * (recipe.serve.max_model_len * 2)
    template_row = dict(rows[0])
    probes: list[dict[str, Any]] = [
        {"probe": "ok", "method": "GET", "path": "/v1/models", "body": None, "raw": None},
        {"probe": "ok", "method": "GET", "path": "/health", "body": None, "raw": None},
        {
            "probe": "unknown_field",
            "method": "POST",
            "path": route,
            "body": _role_body(recipe, template_row, unknown_field=True),
            "raw": None,
        },
        {"probe": "malformed_json", "method": "POST", "path": route, "body": None, "raw": b"{not json"},
        {
            "probe": "wrong_model",
            "method": "POST",
            "path": route,
            "body": _role_body(recipe, template_row, model="not-a-model"),
            "raw": None,
        },
        {
            "probe": "empty_input",
            "method": "POST",
            "path": route,
            "body": _role_body(recipe, template_row, empty=True),
            "raw": None,
        },
        {
            "probe": "over_length",
            "method": "POST",
            "path": route,
            "body": _role_body(recipe, {**template_row, "query": over_length, "documents": [over_length]}),
            "raw": None,
        },
    ]
    out: list[dict[str, Any]] = []
    with httpx.Client(base_url=_engine_root(base_url), timeout=timeout_s) as http:
        for probe_row in probes:
            exchange = _bare_exchange(
                http,
                str(probe_row["method"]),
                str(probe_row["path"]),
                probe_row["body"],
                raw=probe_row["raw"],
            )
            request, response = _entry_from_exchange(exchange)
            inputs = _inputs_of(template_row, tokenizer)
            inputs.update(
                {
                    "probe": probe_row["probe"],
                    "request_id": f"probe:{probe_row['probe']}",
                    "stratum": "protocol",
                }
            )
            out.append(
                build_record(
                    sequence=sequence + len(out),
                    repetition=repetition,
                    batch_context={"size": 1, "request_ids": [inputs["request_id"]], "positions": [0]},
                    server_run_id=server_run_id,
                    request=request,
                    response=response,
                    inputs=inputs,
                )
            )
    out.extend(
        _tokenize_probes(
            recipe,
            base_url,
            rows,
            sequence + len(out),
            server_run_id=server_run_id,
            tokenizer=tokenizer,
            repetition=repetition,
        )
    )
    return out


def _role_body(
    recipe: Recipe,
    row: dict[str, Any],
    *,
    model: str | None = None,
    unknown_field: bool = False,
    empty: bool = False,
) -> dict[str, Any]:
    """One bare request body on the recipe's role route (the exact wire shape the adapter speaks)."""
    body: dict[str, Any]
    if recipe.role == "rerank":
        body = {
            "model": model or recipe.id,
            "query": "" if empty else row["query"],
            "documents": [] if empty else list(row["documents"]),
        }
    elif recipe.role == "embed":
        body = {
            "model": model or recipe.id,
            "input": [""] if empty else ([row["query"]] or [row["documents"][0]]),
            "encoding_format": "float",
        }
    else:
        body = {
            "model": model or recipe.id,
            "input": [""] if empty else ([row["query"]] or [row["documents"][0]]),
            "task": "token_embed",
            "encoding_format": "float",
        }
    if unknown_field:
        body["unknown_field"] = "map-the-error"
    return body


def _tokenize_probes(
    recipe: Recipe,
    base_url: str,
    rows: list[dict[str, Any]],
    sequence: int,
    *,
    repetition: str,
    server_run_id: str,
    tokenizer: Any,
) -> list[dict[str, Any]]:
    """The engine's own ``/tokenize`` reply per sampled input (ids and count): the token ground truth."""
    from .equivalence.stages import tokenize_url

    out: list[dict[str, Any]] = []
    with httpx.Client(base_url=_engine_root(base_url), timeout=_TIMEOUT_S) as http:
        for row_index, row in enumerate(rows):
            for side, text in (("query", row["query"]), ("document", row["documents"][0])):
                body = {"model": recipe.id, "prompt": text, "add_special_tokens": True}
                route = tokenize_url(base_url).removeprefix(_engine_root(base_url))
                exchange = _bare_exchange(http, "POST", route, body)
                request, response = _entry_from_exchange(exchange)
                inputs = _inputs_of(row, tokenizer)
                inputs.update(
                    {
                        "probe": "tokenize",
                        "request_id": f"tokenize:{row_index}:{side}",
                        "stratum": "tokenize",
                        "side": side,
                    }
                )
                out.append(
                    build_record(
                        sequence=sequence + len(out),
                        repetition=repetition,
                        batch_context={"size": 1, "request_ids": [inputs["request_id"]], "positions": [0]},
                        server_run_id=server_run_id,
                        request=request,
                        response=response,
                        inputs=inputs,
                    )
                )
    return out
