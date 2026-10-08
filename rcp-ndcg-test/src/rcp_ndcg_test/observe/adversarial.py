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

from typing import Any

__all__ = ["CONTENT_KINDS", "SYNTHETIC_TEXTS", "special_token_spellings", "synthetic_text"]

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


def special_token_spellings(tokenizer: Any) -> str:
    """The tokenizer's special tokens' literal spellings, joined as one adversarial text.

    Inputs: the recipe's loaded :class:`~rcp_ndcg.data.tokenizer.TextTokenizer`.  Output: one string
    holding the literal spelling of every added token the tokenizer declares (the spellings are read
    from the tokenizer at generation time -- never typed into a file or a source).  An empty string when
    the tokenizer exposes no added-token spellings (a plain BPE with a byte-level alphabet).
    """
    decoder = getattr(getattr(tokenizer, "backend", None), "get_added_tokens_decoder", None)
    if decoder is None:
        return ""
    spellings = sorted({token.content for token in decoder().values()})
    return "\n".join(spellings)


def synthetic_text(kind: str, tokenizer: Any) -> str:
    """One content kind's adversarial text.

    Inputs: the kind name (:data:`CONTENT_KINDS`) and the recipe's tokenizer (only
    ``special_token_spellings`` reads it).  Output: the text.  Raises ``KeyError`` on an unknown kind.
    """
    if kind == "special_token_spellings":
        return special_token_spellings(tokenizer)
    return SYNTHETIC_TEXTS[kind]
