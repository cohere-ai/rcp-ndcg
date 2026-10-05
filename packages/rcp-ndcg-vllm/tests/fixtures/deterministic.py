"""Deterministic numbers shared by the fixture references and the stub engine.

The stub engine answers from :func:`score` and :func:`vector`, and each fixture ``reference.py`` answers from the
same functions, so a clean engine equals its reference exactly and the gates pass with zero slack; the stub's
``--noise`` flag adds seeded noise above the gates.  Everything is derived from SHA-256 digests, so the numbers are
identical on any platform and any run order.
"""

from __future__ import annotations

import hashlib

import numpy as np

DIM = 8


def _digest(tag: str, *parts: str) -> bytes:
    payload = "\x00".join((tag, *parts)).encode("utf-8")
    return hashlib.sha256(payload).digest()


def score(query: str, doc: str, *, noise: float = 0.0) -> float:
    """A probability-scale score in [0, 1) for one (query, document) pair."""
    value = int.from_bytes(_digest("score", query, doc)[:4], "big") / 2**32
    if noise:
        value += _noise("score-noise", f"{query}\x00{doc}", noise)
    return float(min(max(value, 0.0), 1.0))


def vector(text: str, tag: str, *, noise: float = 0.0) -> np.ndarray:
    """A unit L2 float32 vector of :data:`DIM` dimensions for one text; ``noise`` perturbs it deterministically."""
    raw = np.frombuffer(_digest("vector", tag, text)[:DIM], dtype=np.uint8).astype(np.float32) - 127.5
    if noise:
        perturbation = (
            np.frombuffer(_digest("vector-noise", tag, text)[:DIM], dtype=np.uint8).astype(np.float32) - 127.5
        )
        raw = raw + perturbation * noise
    norm = float(np.linalg.norm(raw))
    return raw / norm if norm else raw


def token_id(word: str) -> int:
    """The deterministic id of one word: the stub's tokenizer and the fixture references agree on it."""
    return int.from_bytes(_digest("token", word)[:4], "big") % 50000


def tokens(text: str) -> list[str]:
    """The stub's tokenisation: whitespace words, lowercased, deterministic and dependency-free."""
    return [word for word in text.lower().replace(".", " ").split() if word]


class FixtureTokenizer:
    """The fixture tokenizer: whitespace words with deterministic ids (the harness's TokenizerAdapter protocol).

    Recipe references may provide ``tokenizer()`` to check stage 1 without a Hub download; a production reference
    omits it and the harness loads ``client.tokenizer`` with transformers.
    """

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        """Token ids of ``text``; the fixture tokenizer adds no specials, so the flag changes nothing."""
        del add_special_tokens
        return [self.id_to_token_value(word) for word in tokens(text)]

    @staticmethod
    def id_to_token_value(word: str) -> int:
        """The id of one word: specials keep their reserved ids, content words hash."""
        specials = {"<<cls>>": 49999, "<<sep>>": 49998, "<<end>>": 49997}
        return specials.get(word, token_id(word))

    def truncate(self, text: str, max_tokens: int) -> str:
        """The words of the first ``max_tokens`` tokens of ``text``, joined (a right token-boundary cut)."""
        words = tokens(text)
        if len(words) <= max_tokens:
            return text
        return " ".join(words[:max_tokens])

    def special_tokens(self) -> dict[str, int]:
        """The fixture's special tokens by name, resolved like a real tokenizer's added-vocab map."""
        return {"cls": 49999, "sep": 49998, "end": 49997}

    def id_to_token(self, token_id_value: int) -> str:
        """The token string of one id; special ids resolve to their literal framing, content ids to the report's."""
        specials = {49999: "<<cls>>", 49998: "<<sep>>", 49997: "<<end>>"}
        return specials.get(token_id_value, f"tok:{token_id_value}")

    def decode(self, token_ids: list[int]) -> str:
        """A stand-in text for a kept prefix of ids: the ids rendered like the report's token strings.

        The fixture only compares ids, so any stable text does; the ids round-trip through encode only for
        whitespace tokenisations, and the fixture templates cut whole words, so this is exact for them.
        """
        return " ".join(self.id_to_token(value) for value in token_ids)


def token_vectors(text: str, tag: str, *, noise: float = 0.0) -> np.ndarray:
    """One unit vector per whitespace token of ``text``, shape ``(n_tokens, DIM)`` in float16."""
    words = tokens(text) or ["<empty>"]
    return np.stack([vector(word, tag, noise=noise) for word in words]).astype(np.float16)


def _noise(tag: str, key: str, scale: float) -> float:
    """One seeded noise value in ``(-scale, scale)``, deterministic per (tag, key)."""
    return (int.from_bytes(_digest(tag, key)[:4], "big") / 2**31 - 1.0) * scale


def reserve_and_append(
    prefix_ids: list[int], content_ids: list[int], suffix_ids: list[int], max_tokens: int
) -> list[int]:
    """The anchor-preserving cut: reserve the fixed segments, cut only the content, re-attach.

    This is the fixture references' implementation of the design's cut rule; the harness assembles the same
    ids from the declared shapes, and stage 1 requires the two to agree exactly.
    """
    budget = max_tokens - len(prefix_ids) - len(suffix_ids)
    if budget < 0:
        raise ValueError(f"the fixed segments ({len(prefix_ids) + len(suffix_ids)}) exceed the budget {max_tokens}")
    return prefix_ids + content_ids[:budget] + suffix_ids
