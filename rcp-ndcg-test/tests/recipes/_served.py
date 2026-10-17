"""What the SERVED path actually sends, read through the product's role clients.

Every helper here builds the product's own role client from ``client_config(recipe)`` (the harness's
:func:`rcp_ndcg_test.equivalence.wire.role_client` seam) and reads the captured request bodies -- so a
test sees the product's decisions (the prompt, the text budget cut, the reranker's settle-once query)
exactly as the engine would.  Nothing here re-derives a render, a cut or a settlement (R30): the texts
come from the capture, the cut counts from the client's own census slice per row, and the fixed frame
overhead from the product's ``TemplateSpec.overhead``.

The one sampling loop is the harness's own (:func:`rcp_ndcg_test.equivalence.stages._sampled_rows`,
:func:`rcp_ndcg_test.equivalence.stages._probe`), the same code stage 1 audits.
"""

from __future__ import annotations

import os
import tempfile
import urllib.request
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from rcp_ndcg_test.equivalence import fitting, stages

__all__ = [
    "client_template",
    "fetch_tokenizer",
    "served_pair",
    "served_rows",
    "served_texts",
    "stage1_facts",
    "tokenizer_cache",
]


def client_template(recipe: Any) -> Any:
    """The recipe's client template as the product's :class:`~rcp_ndcg.data.templates.TemplateSpec`.

    The recipe's ``client.template`` is plain data (the lean package validates no endpoint model); the
    product's model validates it at the read, so a test sees exactly what ``rcp-ndcg`` would construct (R30).
    """
    from rcp_ndcg.data.templates import TemplateSpec

    data = recipe.client.get("template")
    return None if data is None else TemplateSpec.model_validate(data)


CACHE_VARIABLE = "RCP_NDCG_VLLM_TOKENIZER_CACHE"
"""The environment variable naming the tokenizer download cache (the lane's scratch dir)."""


def tokenizer_cache(fallback: Path) -> Path:
    """The tokenizer cache directory: ``$RCP_NDCG_VLLM_TOKENIZER_CACHE`` when set, else ``fallback``."""
    override = os.environ.get(CACHE_VARIABLE)
    root = Path(override) if override else fallback
    root.mkdir(parents=True, exist_ok=True)
    return root


def fetch_tokenizer(url: str, name: str, tmp_path: Path, *, sha256: str | None = None) -> Path:
    """Download one tokenizer file into the tokeniser cache (or ``tmp_path`` without the variable).

    Inputs: the file's URL, the file name to cache it under (``<name>``), the test's ``tmp_path`` as
    fallback root, and the pinned SHA-256 the download must match when given.  Output: the local path
    (a cached copy with the pinned hash is reused, offline runs included).  Skips with a clear reason
    when the file is needed and neither cached nor downloadable.  A cached copy that does NOT match the
    pin while the download fails is refused with :class:`~rcp_ndcg_test.errors.HarnessError` naming the
    pin, the cached hash and the fix (clear the cache entry, or restore the network): a stale tokenizer
    would silently move every token count, which the tests must never accept.  The write is atomic (a
    unique temporary file renamed into place), a download that fails the pin yields to a pinned file a
    concurrent worker wrote (one retry first), and a failed write leaves no temporary behind: several
    workers share one cache under ``-n 4``.
    """
    import hashlib

    from rcp_ndcg_test.errors import HarnessError

    root = tokenizer_cache(tmp_path / "tokenizer-cache")
    target = root / name

    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()

    if target.is_file() and (sha256 is None or _sha256(target) == sha256):
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                data = response.read()
        except OSError as error:
            if target.is_file():
                # The cached copy exists but its hash does not match the pin (a matching one returned above):
                # it is stale or corrupt, and with the download failing there is no way to know what the
                # caller would be counting with.  Refuse -- never a silent fallback (AGENTS.md).
                cached = _sha256(target)
                raise HarnessError(
                    f"the cached {target} does not match the pinned sha256 and the download failed ({error}): "
                    f"cached {cached}, pinned {sha256}; clear the cache entry (rm {target}) and retry, or "
                    "restore the network"
                ) from error
            import pytest

            pytest.skip(f"offline: cannot fetch {url} ({error}); the stage-1 checks need the real tokenizer")
        digest = hashlib.sha256(data).hexdigest()
        if sha256 is None or digest == sha256:
            break
        # A concurrent worker may have written the pinned file while this download ran (the `-n 4`
        # shared-cache race): keep its bytes, never clobber them.  One retry covers a transient
        # truncation; a second mismatch is a real failure.
        if target.is_file() and _sha256(target) == sha256:
            return target
    else:
        raise AssertionError(f"the downloaded {name} does not match the pinned sha256")
    # Through a unique temporary file, renamed into place: a reader (another worker's cached check) can
    # never see a partial download, and two concurrent fetches of the same file never share a temporary.
    handle, temporary = tempfile.mkstemp(dir=target.parent, prefix=f"{target.name}.", suffix=".tmp")
    os.close(handle)
    try:
        Path(temporary).write_bytes(data)
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return target


def served_rows(recipe: Any, rows: Sequence[dict[str, Any]], tokenizer: Any) -> dict[str, Any]:
    """What the role client sent for each row of ``rows``, with the cut counts and the frame overhead.

    Inputs: the recipe and its sampled rows (``stages._sampled_rows``' shape is fine: the probe audits
    over-length rows too -- pass the rows exactly, no sampling here).  Output::

        {"probe": <stages._probe output>,         # per-row shapes and texts, the raw audit seam
         "per_shape": {shape: {"texts": [...],        # embed roles: one rendered prompt per row (the row's
                               "row_indexes": [...],  #   query / its first document), in row order
                               "all_texts": {...},    #   per row index: every rendered text of the row
                               "spans": [...],        # rerank: {"query", "documents"} per row, as shipped
                               "cuts": n,             # census cut records (one per cut span)
                               "cut_rows": m,         # rows with at least one cut (per-shape attribution
                               "overhead": m}}}       # TemplateSpec.overhead (None without a template)

    A pairs-file row probed under every declared shape counts its shared cuts under each of its shapes.
    """
    probed = stages._probe(recipe, list(rows), None, tokenizer)
    template = client_template(recipe)
    per_shape: dict[str, dict[str, Any]] = {}

    def body(shape: str) -> dict[str, Any]:
        return per_shape.setdefault(
            shape,
            {
                "texts": [],
                "row_indexes": [],
                "all_texts": {},
                "spans": [],
                "cuts": 0,
                "cut_rows": 0,
                "overhead": _overhead(template, shape, tokenizer),
            },
        )

    for index, entry in enumerate(probed["rows"]):
        for shape, shape_body in entry["shapes"].items():
            record = body(shape)
            record["cuts"] += entry["cuts"]
            record["cut_rows"] += int(entry["cuts"] > 0)
            if "texts" in shape_body:
                texts = shape_body["texts"]
                record["texts"].append(texts[0])
                record["row_indexes"].append(index)
                record["all_texts"][index] = list(texts)
            else:
                record["spans"].append(
                    {"query": shape_body.get("query"), "documents": list(shape_body.get("documents", []))}
                )
    return {"probe": probed, "per_shape": per_shape}


def _overhead(template: Any, shape: str, tokenizer: Any) -> int | None:
    """The declared frame's fixed overhead in tokens (the product's own measurement), or ``None``."""
    if template is None:
        return None
    return template.overhead(fitting.cast_shape(shape), tokenizer)


def stage1_facts(
    recipe: Any, rows: Sequence[dict[str, Any]], tokenizer: Any, over_length_per_shape: int = 5
) -> dict[str, Any]:
    """What the role client sent for a stage-1 pairing of ``rows``: the pairs rows plus the harness
    sampler's over-length padding (the same rows ``stage1_prompts`` probes)."""
    sampled = stages._sampled_rows(recipe, list(rows), tokenizer, over_length_per_shape)
    return served_rows(recipe, sampled, tokenizer)


def served_texts(recipe: Any, texts: Iterable[str], shape: str) -> list[str]:
    """What the role client SHIPS for each input text at ``shape`` (``query`` or ``document``): the request text
    on the text route, and on the messages route the engine's chat-template render of the sent conversation.

    One capture per input (the client fans out concurrently; per-call captures keep position
    attribution exact), read from the request bodies of the product's own client.
    """
    from rcp_ndcg_core.content import Content
    from rcp_ndcg_test.equivalence.wire import role_client

    if recipe.role == "rerank":
        raise ValueError("served_texts is for the embed roles; a rerank pair is served_pair")
    client, capture = role_client(recipe, None)
    role = _encode_role(shape)
    shipped: list[str] = []
    template: str | None = None
    for text in texts:
        start = len(capture.exchanges)
        client.encode([Content.from_text(text)], role)
        captured = capture.texts(capture.exchanges[start])
        if "conversations" not in captured:
            shipped.append(captured["input"][0])
            continue
        # The messages route ships the content; the engine's chat template frames it -- the prompt the engine
        # reads is that render (the served template, else the checkpoint's own), under the request's flag.
        if template is None:
            template = stages.served_chat_template(recipe)[1]
        flag = bool(captured["add_generation_prompt"])
        shipped.append(stages.render_chat(template, captured["conversations"][0], add_generation_prompt=flag))
    return shipped


def served_pair(recipe: Any, query: str, documents: Sequence[str], *, instruction: str | None = None) -> dict[str, Any]:
    """What the role client SHIPS for one rerank request: the settled query span and the document spans.

    Inputs: the recipe, the query and its candidate documents (plus the run-level instruction).  Output:
    ``{"query": str, "documents": [str, ...]}`` -- the captured request bodies (the caller's fit and the
    settle-once query are the product's own, under the recipe's declared budget).
    """
    from rcp_ndcg_test.equivalence.wire import role_client

    if recipe.role != "rerank":
        raise ValueError("served_pair is for rerank recipes; the embed roles ship served_texts")
    client, capture = role_client(recipe, None)
    start = len(capture.exchanges)
    client.rerank(query, list(documents), instruction=instruction)
    queries: list[str] = []
    shipped: list[str] = []
    for exchange in capture.exchanges[start:]:
        texts = capture.texts(exchange)
        if texts.get("query") is not None:
            queries.append(texts["query"])
        shipped.extend(texts.get("documents", []))
    settled = queries[0] if queries else ""
    if len(set(queries)) > 1:
        raise AssertionError(f"the client shipped {len(set(queries))} query spans for one request: settle-once broken")
    return {"query": settled, "documents": shipped}


def _encode_role(shape: str) -> Any:
    from rcp_ndcg.inference.types import EncodeRole

    return EncodeRole.QUERY if shape == "query" else EncodeRole.DOCUMENT
