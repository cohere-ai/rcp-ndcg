"""The synthetic adversarial set, stored as text (OBSERVATIONS-SPEC section 1, content kinds).

Every string here is committed as text and never fetched from anywhere, so the generator asks the same
questions for years.  The content kinds are the spec's: ASCII prose, code, CJK, a right-to-left script,
emoji and ZWJ sequences, combining marks, whitespace-only, the empty string and very long single tokens
(URLs and base64 blobs).

The one content kind that cannot live in a file is the model's special tokens' literal spellings: those
are built at generation time from the recipe's own tokenizer (:func:`special_token_spellings`), never
typed by hand and never spelled in a source file, and they reach the pairs rows through the generator
only.

Public surface:

- :data:`CONTENT_KINDS` — the kind names, in the spec's order.
- :data:`SYNTHETIC_TEXTS` — ``{kind: text}`` for every kind a file can hold.
- :func:`synthetic_text` — one kind's text, with the tokenizer-derived kinds filled in.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = [
    "CONTENT_KINDS",
    "SYNTHETIC_TEXTS",
    "checkpoint_vocab_size",
    "omitted_spellings",
    "special_token_spellings",
    "synthetic_text",
]

ASCII_PROSE = (
    "A deterministic passage for the observation corpus. The engine reads requests, not intentions; "
    "every deviation between two implementations shows up as bytes crossing a wire."
)
CODE = (
    "def digest(payload: bytes) -> str:\n"
    '    """The SHA-256 of payload, hex."""\n'
    "    import hashlib\n"
    "    return hashlib.sha256(payload).hexdigest()\n"
)
CJK = (
    "\u65e5\u672c\u8a9e\u306e\u30c6\u30ad\u30b9\u30c8\u3002\u4e2d\u6587\u7684\u53e5\u5b50\u4e5f\u5728\u8fd9\u91cc\u3002"
    "\u3053\u306e\u884c\u306f\u8a8d\u8b58\u306e\u305f\u3081\u306b\u3042\u308b\u3002"
)
RTL = (
    "\u0647\u0630\u0627 \u0646\u0635 \u0639\u0631\u0628\u064a \u062f\u0648\u0646 \u0623\u064a \u062e\u0644\u0644. "
    "\u05d6\u05d4\u05d9 \u05e2\u05d1\u05e8\u05d9\u05ea \u05de\u05d9\u05de\u05d9\u05df \u05dc\u05de\u05e4\u05d9\u05df."
)
EMOJI_ZWJ = (
    "\U0001f469\u200d\U0001f4bb\U0001f3f3\ufe0f\u200d\U0001f308"
    "\U0001f468\U0001f3fd\u200d\U0001f469\U0001f3ff\u200d\U0001f466 "
    "\U0001f1e6\U0001f1e8\U0001f44d\U0001f3fc"
)
COMBINING_MARKS = "e\u0301le\u0301ve\u0301 \u03b1\u0313\u03b9\u0301 \u0628\u0650\u0633\u0652\u0645\u0650"
WHITESPACE_ONLY = "   \t\t\n   \u00a0 "
EMPTY_STRING = ""
LONG_URL = "https://example.com/synthetic/adversarial/very/long/single/token/" + "/".join(
    f"segment-{index:04d}" for index in range(64)
)
BASE64_BLOB = ("QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVphYmNkZWZnaGlqa2xtbm9wcXJzdHV2d3h5ejAxMjM0" * 8)[:2048]

SYNTHETIC_TEXTS: dict[str, str] = {
    "ascii_prose": ASCII_PROSE,
    "code": CODE,
    "cjk": CJK,
    "rtl": RTL,
    "emoji_zwj": EMOJI_ZWJ,
    "combining_marks": COMBINING_MARKS,
    "whitespace_only": WHITESPACE_ONLY,
    "empty": EMPTY_STRING,
    "long_token_url": LONG_URL,
    "long_token_base64": BASE64_BLOB,
}
"""Every content kind a committed file can hold; ``special_token_spellings`` is built from the tokenizer."""

CONTENT_KINDS: tuple[str, ...] = (*SYNTHETIC_TEXTS.keys(), "special_token_spellings")
"""Every content kind of OBSERVATIONS-SPEC section 1, in the spec's order."""


def special_token_spellings(tokenizer: Any, *, vocab_size: int | None = None) -> str:
    """The tokenizer's special tokens' literal spellings, joined as one adversarial text.

    Inputs: the recipe's loaded :class:`~rcp_ndcg.data.tokenizer.TextTokenizer`, and the checkpoint's
    declared vocabulary size when the caller resolved it (:func:`checkpoint_vocab_size`).  Output: one
    string holding the literal spelling of every added token the tokenizer declares (the spellings are
    read from the tokenizer at generation time -- never typed into a file or a source).  An empty string
    when the tokenizer exposes no added-token spellings (a plain BPE with a byte-level alphabet).

    ``vocab_size`` bounds the set: an added token whose id is not below the checkpoint's declared
    vocabulary size is left out, because the checkpoint's embedding table has no such row and any text
    carrying its spelling kills the engine (harrier-oss-v1-270m: the tokenizer declares
    ``<image_soft_token>`` at id 262144 while its ``config.json`` declares ``vocab_size: 262144`` and its
    embedding is [262144, 640]; the gather asserts device-side -- E2 round 4).  The 27b sibling declares
    ``vocab_size: 262208`` for the same token, so its spelling stays.  The caller records what was left
    out (:func:`omitted_spellings`), never silently.
    """
    decoder = getattr(getattr(tokenizer, "backend", None), "get_added_tokens_decoder", None)
    if decoder is None:
        return ""
    tokens = decoder()
    if vocab_size is not None:
        tokens = {token_id: token for token_id, token in tokens.items() if 0 <= int(token_id) < vocab_size}
    spellings = sorted({token.content for token in tokens.values()})
    return "\n".join(spellings)


def omitted_spellings(tokenizer: Any, *, vocab_size: int | None) -> list[dict[str, Any]]:
    """The added tokens :func:`special_token_spellings` leaves out, with their ids.

    Inputs: the recipe's tokenizer and the resolved vocabulary size (``None``: nothing is bounded).
    Output: ``[{"content", "id"}]``, sorted by id, for every added token whose id is not below the
    bound -- what the plan records in the ``content:special_token_spellings`` stratum so a generation
    run's manifest says which spelling was left out and why.  Units: none.
    """
    if vocab_size is None:
        return []
    decoder = getattr(getattr(tokenizer, "backend", None), "get_added_tokens_decoder", None)
    if decoder is None:
        return []
    out_of_range = {
        int(token_id): token for token_id, token in decoder().items() if not 0 <= int(token_id) < vocab_size
    }
    return sorted(
        ({"content": token.content, "id": token_id} for token_id, token in out_of_range.items()),
        key=lambda entry: entry["id"],
    )


def checkpoint_vocab_size(recipe: Any) -> int | None:
    """The checkpoint's declared vocabulary size, read from its ``config.json`` at the pinned revision.

    Inputs: the resolved recipe (its ``model``/``revision``, the same pair the tokenizer spec comes from).
    Output: the top-level ``vocab_size``, else the text config's (a composite checkpoint declares its
    vocabulary there), else ``None`` when the config cannot be read (offline, uncached or a fixture
    recipe): the bound is a refinement of the spelling set, never a requirement, and a run without it
    keeps every spelling.  Units: none.
    """
    model = str(getattr(recipe, "model", "") or "")
    if not model or "/" not in model:
        return None
    import json

    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(model, "config.json", revision=getattr(recipe, "revision", None) or None)
        config = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - an unreadable config leaves the spelling set unbounded
        return None
    for candidate in (config.get("vocab_size"), (config.get("text_config") or {}).get("vocab_size")):
        if isinstance(candidate, int) and candidate > 0:
            return candidate
    return None


def synthetic_text(kind: str, tokenizer: Any, *, vocab_size: int | None = None) -> str:
    """One content kind's adversarial text.

    Inputs: the kind name (:data:`CONTENT_KINDS`), the recipe's tokenizer (only
    ``special_token_spellings`` reads it) and the checkpoint's declared vocabulary size (only that kind
    uses it).  Output: the text.  Raises ``KeyError`` on an unknown kind.
    """
    if kind == "special_token_spellings":
        return special_token_spellings(tokenizer, vocab_size=vocab_size)
    return SYNTHETIC_TEXTS[kind]
