"""The media stage: what the served client sends for image and video inputs, against what the reference consumes.

Stages 1 and 2 compare texts and numbers; neither sends media, so nothing there proves that a vision-language
recipe's served path shows the model the same images as its reference.  This stage does, for every pairs row
that carries media (the ``media`` field, :func:`media_rows`), side by side:

- **the client** -- the recipe's role client (the product: its media preparation, ``rcp_ndcg.data.prepare``
  under the declared policy, ``rcp_ndcg.data.resolution``, and its budget fit) sends each media side, captured
  through the product's injection point; from the captured request the stage reads what crossed the wire:
  the parts in order (the **placement**), every image's prepared **geometry** (decoded from the sent bytes),
  every video's frame count, and the **token count** the client reserved for each item (the product's
  :func:`~rcp_ndcg.data.resolution.content_media_tokens` under the client's effective policies);
- **the reference** -- the recipe's reference subprocess in ``--mode media`` reports the same facts for the
  same rows as the model card consumes them (its own processor's resize, its own placement, its own count);
- **the engine** (with ``--base-url``) -- each captured media request is sent again without its media parts,
  and the difference of the two ``usage.prompt_tokens`` reports is the engine's own media count (everything
  else cancels): an engine whose pixel pin is missing re-resizes the prepared image and counts other tokens.

Every media item gates exactly: the count, the placement, the geometry and the tokens must be equal on every
side; a video container's token count is reported, not gated (the client counts a container at its family's
declared timestamp bound).  A media recipe whose pairs carry no media row fails the stage (it checked
nothing); a recipe that declares no media input has no media stage.

The pairs file's ``media`` field: ``{"query": [entry, ...], "documents": [[entry, ...], ...]}``, each entry a
:class:`~rcp_ndcg_core.content.MediaRef` object plus its ``kind`` (``image`` or ``video``) -- the bytes
inline (a ``data:`` URI) or at a resolvable URI.  A side's content is its media in entry order, then its text
(:func:`side_content`): the order the vision-language cards build their inputs in.
"""

from __future__ import annotations

import base64
import io
import json
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rcp_ndcg_vllm.recipe import Recipe

from rcp_ndcg_test.errors import HarnessError

from .reference import run_reference
from .wire import role_client

__all__ = ["MEDIA_KINDS", "media_rows", "side_content", "side_contents", "stage_media", "takes_media", "text_rows"]

MEDIA_KINDS = ("image", "video")
"""The media kinds a pairs entry names."""

_REF_FIELDS = ("uri", "sha256", "mime", "width", "height", "num_bytes", "num_frames", "duration_s", "fps")


def takes_media(recipe: Recipe) -> bool:
    """Whether the recipe declares an image or video input (it then has a media stage)."""
    return bool({"image", "video"} & set(recipe.input))


def _entries(row: dict[str, Any]) -> list[dict[str, Any]]:
    media = row.get("media") or {}
    documents = [entry for entries in media.get("documents") or [] for entry in entries or []]
    return [entry for entry in [*(media.get("query") or []), *documents] if isinstance(entry, dict)]


def media_rows(rows: Sequence[dict[str, Any]]) -> list[tuple[int, dict[str, Any]]]:
    """The pairs rows that carry inline media (an entry with its bytes' ``uri``), with their positions in the
    file: this stage's rows.  A row whose media are source coordinates only (a suite page, resolved at
    ingest) stays a text row of stages 1 and 2 and is counted here as unresolved."""
    return [(index, row) for index, row in enumerate(rows) if any(entry.get("uri") for entry in _entries(row))]


def text_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The pairs rows without inline media: what stages 1 and 2 compare (the media rows are this stage's)."""
    media = {index for index, _ in media_rows(rows)}
    return [row for index, row in enumerate(rows) if index not in media]


def _part(entry: dict[str, Any]) -> Any:
    """One pairs media entry as the product's content part."""
    from rcp_ndcg_core.content import ImagePart, MediaRef, VideoPart

    kind = entry.get("kind", "image")
    if kind not in MEDIA_KINDS:
        raise HarnessError(f"a media entry's kind must be one of {list(MEDIA_KINDS)}, got {kind!r}")
    if not entry.get("uri"):
        raise HarnessError(
            "a media entry needs its bytes (a uri: inline data: or a resolvable path); a source coordinate "
            f"(suite/subset/doc_id) is resolved at ingest, not here: {sorted(entry)}"
        )
    ref = MediaRef.model_validate({key: entry[key] for key in _REF_FIELDS if entry.get(key) is not None})
    return ImagePart(ref=ref) if kind == "image" else VideoPart(ref=ref)


def side_content(text: str, entries: Sequence[dict[str, Any]]) -> Any:
    """One side of a media row as the product's :class:`~rcp_ndcg_core.content.Content`: its media parts in
    entry order, then its text when it has one."""
    from rcp_ndcg_core.content import Content, TextPart

    parts: list[Any] = [_part(entry) for entry in entries]
    if text:
        parts.append(TextPart(text=text))
    return Content.from_parts(parts) if parts else Content.from_text("")


def side_contents(row: dict[str, Any]) -> tuple[Any, list[Any]]:
    """A media row's query and documents as content (:func:`side_content` per side)."""
    media = row.get("media") or {}
    documents_media = list(media.get("documents") or [])
    query = side_content(str(row["query"]), list(media.get("query") or []))
    documents = [
        side_content(str(text), documents_media[position] if position < len(documents_media) else [])
        for position, text in enumerate(row["documents"])
    ]
    return query, documents


# ---------------------------------------------------------------------------
# The client's side: what crossed the wire.
# ---------------------------------------------------------------------------


def _image_size(url: str) -> tuple[int, int]:
    """The ``(width, height)`` of an image part's bytes, as sent."""
    from PIL import Image

    if not url.startswith("data:") or "," not in url:
        raise HarnessError(f"the client sent an image that is not inline ({url[:40]}...): the stage reads sent bytes")
    with Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))) as handle:
        return handle.size


def _parts_of(value: Any) -> list[dict[str, Any]]:
    """A sent side as its content parts: a messages user turn's content, a rerank side's ``{"content": [...]}``
    or a plain string (one text part)."""
    if isinstance(value, str):
        return [{"type": "text", "text": value}]
    if isinstance(value, dict) and isinstance(value.get("content"), list):
        value = value["content"]
    if not isinstance(value, list):
        return []
    return [{"type": "text", "text": part} if isinstance(part, str) else part for part in value]


def _sent_side(parts: list[dict[str, Any]], client: Any) -> dict[str, Any]:
    """The facts of one sent side: the placement (part kinds in order), and per media item its kind, geometry
    or frame count and the tokens the client counted for it (the product's own count)."""
    from rcp_ndcg_core.content import Content, ImagePart, MediaRef, VideoPart

    from rcp_ndcg.data.prepare import media_policies_for
    from rcp_ndcg.data.resolution import ImagePolicy, content_media_tokens

    image_policy, video_policy = media_policies_for(client.config)

    def tokens_of(content: Any) -> int | None:
        try:
            return content_media_tokens(content, image_policy or ImagePolicy.native(), video_policy).tokens
        except Exception:  # noqa: BLE001 - a policy that cannot count: reported as None, compared as a failure
            return None

    placement: list[str] = []
    items: list[dict[str, Any]] = []
    for part in parts:
        kind = part.get("type")
        if kind == "text":
            if part.get("text"):
                placement.append("text")
        elif kind == "image_url":
            width, height = _image_size(str((part.get("image_url") or {}).get("url", "")))
            placement.append("image")
            content = Content.from_parts([ImagePart(ref=MediaRef(uri="data:,", width=width, height=height))])
            items.append({"kind": "image", "width": width, "height": height, "tokens": tokens_of(content)})
        elif kind == "video_url":
            placement.append("video")
            frames = video_policy.num_frames if video_policy is not None else None
            content = Content.from_parts([VideoPart(ref=MediaRef(uri="data:,", mime="video/mp4"))])
            items.append({"kind": "video", "frames": frames, "tokens": tokens_of(content)})
    return {"placement": placement, "media": items}


def _client_facts(
    recipe: Recipe, rows: list[tuple[int, dict[str, Any]]], base_url: str | None
) -> tuple[dict[tuple[int, str], dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Send every media side through the role client; return the sent facts per ``(row, side)``, the
    refusals, and the captured media requests (each with the ``(row, side)`` keys it carries)."""
    from rcp_ndcg.errors import RcpNdcgError
    from rcp_ndcg.inference.types import EncodeRole

    client, capture = role_client(recipe, base_url)
    facts: dict[tuple[int, str], dict[str, Any]] = {}
    refusals: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    for index, row in rows:
        query, documents = side_contents(row)
        calls: list[tuple[list[str], Any]] = []
        if recipe.role == "rerank":
            # One pair per request: the engine's prompt-token report is then that pair's alone.
            for position, document in enumerate(documents):
                if query.has_media or document.has_media:
                    keys = ["query"] if query.has_media else []
                    keys += [f"document {position}"] if document.has_media else []
                    calls.append((keys, (query, [document], row.get("instruction"))))
        else:
            if query.has_media:
                calls.append((["query"], ([query], EncodeRole.QUERY)))
            for position, document in enumerate(documents):
                if document.has_media:
                    calls.append(([f"document {position}"], ([document], EncodeRole.DOCUMENT)))
        for keys, call in calls:
            start = len(capture.exchanges)
            try:
                if recipe.role == "rerank":
                    client.rerank(call[0], call[1], instruction=call[2])
                else:
                    client.encode(*call)
            except RcpNdcgError as error:
                refusals.append({"row": index, "sides": keys, "error": f"{type(error).__name__}: {error}"})
                continue
            for exchange in capture.exchanges[start:]:
                body = exchange.get("request_body") or {}
                if recipe.role == "rerank":
                    document_parts = _parts_of((body.get("documents") or [""])[0])
                    sent = {key: _parts_of(body.get("query")) if key == "query" else document_parts for key in keys}
                else:
                    conversation = body.get("messages") or []
                    conversation = (
                        conversation[0] if conversation and isinstance(conversation[0], list) else conversation
                    )
                    turn = next((m for m in conversation if isinstance(m, dict) and m.get("role") == "user"), {})
                    sent = {keys[0]: _parts_of(turn.get("content") if turn else body.get("input"))}
                counted = 0
                for side, parts in sent.items():
                    side_facts = _sent_side(parts, client)
                    facts[(index, side)] = side_facts
                    counted += sum(int(item["tokens"] or 0) for item in side_facts["media"])
                requests.append({"row": index, "sides": keys, "exchange": exchange, "counted": counted})
    return facts, refusals, requests


# ---------------------------------------------------------------------------
# The engine's side: the media count it reports.
# ---------------------------------------------------------------------------


def _without_media(body: dict[str, Any]) -> dict[str, Any]:
    """The request without its media parts -- the baseline whose prompt-token report the media count is the
    difference to (a side left with no part keeps one empty text part, as the product's lowering does)."""

    def strip(value: Any) -> Any:
        if isinstance(value, dict) and isinstance(value.get("content"), list):
            return {**value, "content": strip(value["content"])}
        if isinstance(value, list) and value and all(isinstance(part, dict) and "type" in part for part in value):
            kept = [part for part in value if part.get("type") not in ("image_url", "video_url")]
            return kept or [{"type": "text", "text": ""}]
        return value

    out = json.loads(json.dumps(body))
    if isinstance(out.get("messages"), list):
        conversations = out["messages"]
        nested = conversations and isinstance(conversations[0], list)
        for conversation in conversations if nested else [conversations]:
            for message in conversation:
                if isinstance(message, dict):
                    message["content"] = strip(message.get("content"))
    for key in ("query",):
        if key in out:
            out[key] = strip(out[key])
    if isinstance(out.get("documents"), list):
        out["documents"] = [strip(document) for document in out["documents"]]
    return out


def _prompt_tokens(reply: Any) -> int | None:
    usage = reply.get("usage") if isinstance(reply, dict) else None
    value = (usage or {}).get("prompt_tokens") if isinstance(usage, dict) else None
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _engine_check(requests: list[dict[str, Any]], base_url: str | None) -> dict[str, Any]:
    """Per captured media request: the engine's media count (the prompt-token difference to the same request
    without its media) against the client's count; ``not_run`` without an engine, never passed."""
    if base_url is None:
        return {"status": "not_run", "passed": None, "reason": "no engine URL: the engine's media count is the wave's"}
    import httpx

    failures: list[dict[str, Any]] = []
    checked = 0
    with httpx.Client(timeout=120.0) as http:
        for request in requests:
            exchange = request["exchange"]
            with_media = _prompt_tokens(exchange.get("response_json"))
            reply = http.post(exchange["url"], json=_without_media(exchange.get("request_body") or {}))
            try:
                without = _prompt_tokens(reply.json())
            except ValueError:
                without = None
            checked += 1
            if with_media is None or without is None:
                failures.append(
                    {"row": request["row"], "sides": request["sides"], "note": "the engine reported no prompt tokens"}
                )
                continue
            if with_media - without != request["counted"]:
                failures.append(
                    {
                        "row": request["row"],
                        "sides": request["sides"],
                        "engine_media_tokens": with_media - without,
                        "client_media_tokens": request["counted"],
                    }
                )
    return {
        "status": "run",
        "checked": checked,
        "passed": bool(checked) and not failures,
        "failures": failures,
        "referent": "per media request, the engine's usage.prompt_tokens minus the same request's without its "
        "media parts must equal the media tokens the client counted (an unpinned or re-resizing engine differs)",
    }


# ---------------------------------------------------------------------------
# The stage.
# ---------------------------------------------------------------------------


def _reference_facts(
    recipe: Recipe, reference_python: str, rows: list[tuple[int, dict[str, Any]]]
) -> dict[tuple[int, str], dict[str, Any]]:
    """The reference subprocess's media facts for the media rows (``--mode media``), keyed by the rows'
    positions in the pairs file and the side."""
    from .fitting import resolved_tokenizer_spec

    recipe_dir = recipe._dir
    if recipe_dir is None:  # pragma: no cover - load_recipe sets it
        raise HarnessError(f"recipe {recipe.id} was not loaded from a directory")
    with tempfile.TemporaryDirectory() as work:
        pairs_path = Path(work) / "pairs.jsonl"
        public = [{key: value for key, value in row.items() if not key.startswith("_")} for _, row in rows]
        pairs_path.write_text("".join(json.dumps(row) + "\n" for row in public), encoding="utf-8")
        document = run_reference(
            reference_python,
            str(recipe_dir / recipe.reference.entry),
            mode="media",
            pairs_path=pairs_path,
            out_path=Path(work) / "reference.json",
            tokenizer_spec=resolved_tokenizer_spec(recipe),
        )
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for entry in document.get("rows", []):
        position = int(entry["index"])
        if not 0 <= position < len(rows):
            raise HarnessError(f"the reference reported media for row {position}, which it was not given")
        out[(rows[position][0], str(entry["side"]))] = {
            "placement": list(entry.get("placement") or []),
            "media": list(entry.get("media") or []),
            **({"refused": str(entry["refused"])} if entry.get("refused") else {}),
        }
    return out


def _compare(key: tuple[int, str], client: dict[str, Any], reference: dict[str, Any]) -> list[dict[str, Any]]:
    """Every difference between the client's and the reference's facts of one side (each item gates)."""
    where = {"row": key[0], "side": key[1]}
    if reference.get("refused"):
        # The card cannot consume this input (e.g. an image and a text in one late-interaction document):
        # what the client sends here reaches the model as an input its reference never defines.
        return [{**where, "check": "reference_refused", "reason": reference["refused"]}]
    failures: list[dict[str, Any]] = []
    if client["placement"] != reference["placement"]:
        failures.append(
            {**where, "check": "placement", "client": client["placement"], "reference": reference["placement"]}
        )
    if len(client["media"]) != len(reference["media"]):
        failures.append(
            {**where, "check": "count", "client": len(client["media"]), "reference": len(reference["media"])}
        )
        return failures
    for position, (sent, consumed) in enumerate(zip(client["media"], reference["media"], strict=True)):
        names = ("kind", "width", "height", "frames") + (("tokens",) if sent["kind"] == "image" else ())
        for name in names:
            counted = name != "tokens" or all(
                isinstance(value, int) and not isinstance(value, bool) for value in (sent.get(name), consumed.get(name))
            )  # an image's tokens gate as counts: uncounted on either side is a failure, never None == None
            if not counted or sent.get(name) != consumed.get(name):
                failures.append(
                    {
                        **where,
                        "item": position,
                        "check": name,
                        "client": sent.get(name),
                        "reference": consumed.get(name),
                    }
                )
    return failures


def stage_media(
    recipe: Recipe, pairs_path: str | Path, reference_python: str | None, *, base_url: str | None = None
) -> dict[str, Any] | None:
    """The media stage of one recipe (see the module docstring); ``None`` for a recipe without media input.

    Inputs: the recipe, the pairs file (its media rows), the reference interpreter and the engine's URL.
    Output: ``{"status", "rows", "sides", "items", "unresolved_rows", "failures", "refusals", "engine_check",
    "passed", "referent"}`` (``unresolved_rows``: rows whose media are source coordinates, reported, not
    compared): ``passed`` only when every media item of every side matched the reference (count, placement,
    geometry or frames, tokens), the client refused no media row and -- with an engine -- the engine counted
    what the client counted.  Without ``reference_python`` the stage is ``not_run`` (neutral, as stage 1's
    render check); a media recipe whose pairs carry no media row fails.
    """
    from .fitting import load_pairs

    if not takes_media(recipe):
        return None
    loaded = load_pairs(pairs_path)
    rows = media_rows(loaded)
    unresolved = sum(1 for row in text_rows(loaded) if _entries(row))
    if not rows:
        return {
            "status": "no_media_rows",
            "passed": False,
            "reason": f"the recipe declares input {list(recipe.input)} and the pairs file carries no media row: "
            "the media gate checked nothing",
        }
    if not reference_python:
        return {"status": "not_run", "passed": None, "reason": "no --reference-python: the reference's media facts"}
    client, refusals, requests = _client_facts(recipe, rows, base_url)
    reference = _reference_facts(recipe, reference_python, rows)
    failures: list[dict[str, Any]] = []
    for key in sorted(set(client) | set(reference)):
        if key not in reference:
            failures.append(
                {"row": key[0], "side": key[1], "check": "reference", "note": "the reference reported no media facts"}
            )
        elif reference[key].get("refused"):
            # The card does not define this input: that decides the row (the generator prunes it), whether or
            # not the client sent it.
            failures.append(
                {"row": key[0], "side": key[1], "check": "reference_refused", "reason": reference[key]["refused"]}
            )
        elif key not in client:
            failures.append({"row": key[0], "side": key[1], "check": "client", "note": "the client sent no such side"})
        else:
            failures.extend(_compare(key, client[key], reference[key]))
    engine = _engine_check(requests, base_url)
    items = sum(len(facts["media"]) for facts in client.values())
    return {
        "status": "run",
        "rows": len(rows),
        "sides": len(client),
        "items": items,
        "unresolved_rows": unresolved,
        "failures": failures,
        "refusals": refusals,
        "engine_check": engine,
        "passed": bool(items) and not failures and not refusals and engine.get("passed") is not False,
        "referent": "per media row and side, what the client sent (the parts in order, each image's prepared "
        "geometry and each video's frames, the tokens it counted) must equal what the reference consumes; with "
        "an engine, the engine's own media count must equal the client's",
    }
