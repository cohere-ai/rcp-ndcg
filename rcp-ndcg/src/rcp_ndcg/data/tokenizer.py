"""The judge's tokenizer: text limits are counted in the tokens the judge reads.

A judge names its tokenizer (``JudgeConfig.tokenizer``): a Hugging Face repository id with an optional
``@revision`` (``Qwen/Qwen3.5-397B-A17B-FP8``, ``org/model@<commit>``), or a local path to a ``tokenizer.json``
(or to a directory holding one). :func:`load_tokenizer` loads it once per process with the ``tokenizers`` library
(no torch), and :class:`TextTokenizer` counts tokens and reports where each token sits in the original text, so
that text is cut at token boundaries of the document itself (:mod:`rcp_ndcg.data.preprocess`), never by decoding
tokens back to text.

The tokenizer's identity is the SHA-256 of its ``tokenizer.json`` -- extended with the sidecar content exactly when
that content changes the effective vocabulary (an added or special token ``tokenizer.json`` does not already carry
as added): two passes whose judges tokenize differently never pool (it is recorded in the judgement family and the
pass's preprocessing identity). The sidecars are the ones ``AutoTokenizer`` honours beside ``tokenizer.json``
(``tokenizer_config.json``, ``added_tokens.json``, ``special_tokens_map.json``): the engine tokenizes with
``AutoTokenizer``, so a bare load that ignored them would count -- and cut -- differently than the engine reads
(GPU-E1's ctxl-1b: ``tokenizer_config.json``'s ``pad_token: "+"``).
"""

from __future__ import annotations

import functools
import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rcp_ndcg.errors import ConfigError, MissingInputError, dependency_error

#: The file a tokenizer is read from, locally or in a Hub repository.
TOKENIZER_FILE = "tokenizer.json"

#: The sidecar files ``AutoTokenizer`` also honours beside ``tokenizer.json``; each is optional.
SIDECAR_FILES: tuple[str, ...] = ("tokenizer_config.json", "added_tokens.json", "special_tokens_map.json")

_SINGLE_TOKEN_FIELDS: tuple[str, ...] = (
    "bos_token",
    "eos_token",
    "unk_token",
    "sep_token",
    "pad_token",
    "cls_token",
    "mask_token",
)
_LIST_TOKEN_FIELDS: tuple[str, ...] = ("additional_special_tokens", "extra_special_tokens")


def _backend_class() -> Any:
    try:
        from tokenizers import Tokenizer
    except ModuleNotFoundError as exc:
        raise dependency_error("tokenizers", needed_for="counting text in the judge's tokens") from exc
    return Tokenizer


#: The added vocabulary per tokenizer file (keyed by its SHA-256), resolved once per file per process.
_ADDED_TOKENS_CACHE: dict[str, dict[str, str]] = {}


def _added_tokens(backend: Any) -> dict[str, str]:
    """The backend's added vocabulary, resolved once at load: name -> literal text (see ``added_tokens``)."""
    tokens: dict[str, str] = {}
    for token in backend.get_added_tokens_decoder().values():
        content = token.content
        name = content[2:-2] if content.startswith("<|") and content.endswith("|>") else content
        tokens.setdefault(name, content)
        tokens.setdefault(content, content)
    return tokens


def _token_contents(value: Any) -> Iterator[str]:
    """The token contents a sidecar value names: a string, a ``{"content": ...}`` entry, or a list of either."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        content = value.get("content")
        if isinstance(content, str):
            yield content
    elif isinstance(value, Sequence):
        for item in value:
            yield from _token_contents(item)


def _sidecar_json(payloads: Mapping[str, bytes], filename: str) -> Mapping[str, Any] | None:
    """One sidecar's parsed object, or ``None`` when it is absent; a malformed sidecar is refused, never
    skipped (ignoring it would silently tokenize differently than the engine's ``AutoTokenizer``)."""
    data = payloads.get(filename)
    if data is None:
        return None
    try:
        parsed = json.loads(data)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"the tokenizer sidecar {filename!r} is not valid JSON: {exc}") from exc
    if not isinstance(parsed, Mapping):
        raise ConfigError(f"the tokenizer sidecar {filename!r} is not a JSON object")
    return parsed


def _sidecar_token_contents(payloads: Mapping[str, bytes]) -> list[str]:
    """The token contents the sidecar files name, in a deterministic order, deduplicated."""
    contents: list[str] = []

    def add(value: Any) -> None:
        contents.extend(_token_contents(value))

    config = _sidecar_json(payloads, "tokenizer_config.json")
    if config is not None:
        decoder = config.get("added_tokens_decoder")
        if isinstance(decoder, Mapping):
            for _, entry in sorted(decoder.items(), key=lambda item: str(item[0])):
                add(entry)
        for field_name in _SINGLE_TOKEN_FIELDS + _LIST_TOKEN_FIELDS:
            add(config.get(field_name))
    special = _sidecar_json(payloads, "special_tokens_map.json")
    if special is not None:
        for field_name in _SINGLE_TOKEN_FIELDS + _LIST_TOKEN_FIELDS:
            add(special.get(field_name))
    added = _sidecar_json(payloads, "added_tokens.json")
    if added is not None:
        for name in added:
            add(name)
    return list(dict.fromkeys(contents))


def _apply_sidecar_tokens(backend: Any, payloads: Mapping[str, bytes]) -> list[str]:
    """Add the sidecars' tokens to ``backend`` the way ``AutoTokenizer`` adds them; returns the effective ones.

    A token ``tokenizer.json`` already carries as *added* is skipped (adding it again is a no-op); a token that
    exists in the base vocabulary but is not an added token (ctxl-1b's ``+``) *is* added, because the added-token
    matcher then keeps it whole -- the engine's own behaviour.
    """
    existing = {token.content for token in backend.get_added_tokens_decoder().values()}
    effective = [content for content in _sidecar_token_contents(payloads) if content not in existing]
    if effective:
        backend.add_special_tokens(effective)
    return effective


@dataclass(frozen=True)
class TextTokenizer:
    """A loaded tokenizer: counts tokens and locates them in the original text.

    Special tokens (a BOS marker, a chat template's role tokens) are never added: the counts are of the text alone.

    Attributes:
        name: The tokenizer as the judge names it (repository id with its revision, or a path).
        sha256: SHA-256 of its ``tokenizer.json`` (for a tokenizer built in memory, of its JSON serialisation),
            extended with the applied sidecar tokens when they change the effective vocabulary.
    """

    name: str
    sha256: str
    backend: Any = field(repr=False, compare=False)

    @classmethod
    def from_json(cls, data: bytes, *, name: str, sidecars: Mapping[str, bytes] | None = None) -> TextTokenizer:
        """The tokenizer serialised in ``data`` (the bytes of a ``tokenizer.json``).

        ``sidecars`` maps the optional sidecar file names to their bytes (see :data:`SIDECAR_FILES`); the tokens
        they add are applied the way ``AutoTokenizer`` applies them, and only the effective ones enter the
        identity (a sidecar that repeats ``tokenizer.json``'s added vocabulary adds nothing). ``None`` (the
        default) applies none: the caller that has them is the one that passes them -- ``load_tokenizer``
        reads the files beside ``tokenizer.json`` (or fetches them beside it from the Hub) and passes them
        here, so a caller that wants the engine's tokenization must come through it or read the sidecars
        itself; ``from_json`` never guesses them from the ``name``.

        The backend's embedded truncation and padding are reset at load (G5): a ``tokenizer.json`` can ship
        ``truncation: {max_length: 1024}`` (topk-embed-v1-small does) or fixed-length padding, and an
        un-reset backend silently tops every count and id list at those lengths -- no budget above them could
        ever cut, and no failure would name the cause. transformers resets the same caps per call; this is
        the load-time equivalent, the one construction site in the package (``load_tokenizer``,
        ``from_backend`` and every caller funnel through here).
        """
        backend = _backend_class().from_str(data.decode("utf-8"))
        backend.no_truncation()
        backend.no_padding()
        effective = _apply_sidecar_tokens(backend, sidecars or {})
        digest = hashlib.sha256(data)
        if effective:
            digest.update(b"\x00")
            digest.update(json.dumps(effective, ensure_ascii=False).encode("utf-8"))
        return cls(name=name, sha256=digest.hexdigest(), backend=backend)

    @classmethod
    def from_backend(cls, backend: Any, *, name: str) -> TextTokenizer:
        """Wrap a ``tokenizers.Tokenizer`` built in memory; its identity is the hash of its serialisation."""
        return cls.from_json(backend.to_str().encode("utf-8"), name=name)

    def count(self, text: str, *, add_special_tokens: bool = False, split_special_tokens: bool = False) -> int:
        """The number of tokens of ``text``; with ``add_special_tokens``, as the engine counts it
        (the tokenizer's post-processor tokens included, e.g. an appended end-of-text anchor).

        ``split_special_tokens`` selects the reference's parse (transformers' ``split_special_tokens``,
        the raw tokenizers ``encode_special_tokens`` toggle): an added SPECIAL token is textified instead
        of matched as one id, so a token-ids wire that must carry the reference's ids (the
        pplx-embed-v2-context document contract) can count what it sends. The default is the file's own
        parse, unchanged.
        """
        return len(
            self._encode(text, add_special_tokens=add_special_tokens, split_special_tokens=split_special_tokens).ids
        )

    def ids(self, text: str, *, add_special_tokens: bool = False, split_special_tokens: bool = False) -> list[int]:
        """The token ids of ``text``, optionally with the post-processor's (as the engine reads them).

        ``split_special_tokens`` is :meth:`count`'s parse: added SPECIAL tokens are textified, the
        reference's ``split_special_tokens`` ids. The default is the file's own parse.
        """
        return list(
            self._encode(text, add_special_tokens=add_special_tokens, split_special_tokens=split_special_tokens).ids
        )

    def _encode(self, text: str, *, add_special_tokens: bool, split_special_tokens: bool) -> Any:
        """One encode under the requested parse; the parse toggle is a backend property, so it is set for
        the call and restored after it (the tokenizer is shared and cached: a flag that stuck would change
        every later caller's ids)."""
        backend = self.backend
        previous = backend.encode_special_tokens
        if previous == split_special_tokens:
            return backend.encode(text, add_special_tokens=add_special_tokens)
        backend.encode_special_tokens = split_special_tokens
        try:
            return backend.encode(text, add_special_tokens=add_special_tokens)
        finally:
            backend.encode_special_tokens = previous

    def offsets(self, text: str, *, split_special_tokens: bool = False) -> list[tuple[int, int]]:
        """``(start, end)`` character offsets in ``text`` of each of its tokens, in order.

        ``split_special_tokens`` is :meth:`ids`' parse (the offsets belong to the ids it returns); the
        default is the file's own parse.
        """
        encoded = self._encode(text, add_special_tokens=False, split_special_tokens=split_special_tokens)
        return [tuple(offset) for offset in encoded.offsets]  # type: ignore[misc]

    def identity(self) -> dict[str, str]:
        """``{"name", "sha256"}``: what a preprocessing record names the tokenizer by."""
        return {"name": self.name, "sha256": self.sha256}

    def added_tokens(self) -> dict[str, str]:
        """The added vocabulary: a special token's name (its literal text without the ``<|...|>`` wrapper)
        to that literal text, e.g. the end-of-turn marker's name to its literal form. Both the bare name and
        the literal text are accepted as lookup keys by :meth:`special_text`; the literal text is what a
        template's fixed segment is rendered with. The mapping is read-only and shared per tokenizer file
        (keyed by its SHA-256, so a re-load of the same file reuses it); treat it as immutable.
        """
        cached = _ADDED_TOKENS_CACHE.get(self.sha256)
        if cached is None:
            cached = _added_tokens(self.backend)
            if len(_ADDED_TOKENS_CACHE) >= 32:
                _ADDED_TOKENS_CACHE.clear()
            _ADDED_TOKENS_CACHE[self.sha256] = cached
        return cached

    def special_text(self, name: str) -> str:
        """The literal text of the tokenizer's added token named ``name`` (the bare name or the wrapped
        form), for splicing into a rendered template; the engine's tokenizer then matches it back to one id.
        The name is taken exactly as written: an added token may carry significant whitespace (``"[Q] "``
        ships on pplx-embed-v2-contextual), and a stripped lookup could never resolve it.

        Raises:
            ConfigError: the tokenizer has no added token by that name; the hint names the nearest ones it
                has, then all of them.
        """
        tokens = self.added_tokens()
        try:
            return tokens[name]
        except KeyError:
            import difflib

            known = sorted({key for key in tokens if not key.startswith("<|")})
            nearest = difflib.get_close_matches(name, sorted(tokens), n=3, cutoff=0.6)
            raise ConfigError(
                f"the tokenizer {self.name!r} has no special token named {name!r}",
                hint=("did you mean " + ", ".join(repr(match) for match in nearest) + "? " if nearest else "")
                + f"its added tokens are named {known}; write one as {{special:<name>}} in the template",
            ) from None

    def special_id(self, name: str) -> int:
        """The token id of the added token named ``name`` (as :meth:`special_text` resolves it).

        Raises:
            ConfigError: the tokenizer has no added token by that name.
        """
        return self.backend.token_to_id(self.special_text(name))


def tokenizer_identity(spec: str) -> dict[str, str]:
    """The tokenizer's content identity as every role config carries it in ``identity_extra()``.

    The one tokenizer-identity helper: the SHA-256 comes from :attr:`TextTokenizer.sha256` (the one hashing
    site, over the ``tokenizer.json`` the spec names), under the one key ``tokenizer_sha256`` -- never the
    name the spec spells, which is runtime and only recorded beside the identity as a source.

    Args:
        spec: A Hugging Face repository id with an optional ``@revision``, or a local path to a
            ``tokenizer.json`` or to a directory holding one -- the value a role config's ``tokenizer`` field
            holds.

    Returns:
        ``{"tokenizer_sha256": <sha>}``.

    Raises:
        DependencyError: ``tokenizers`` (or, for a Hub id, ``huggingface_hub``) is not installed.
        MissingInputError: the local file, or the repository's ``tokenizer.json``, does not exist.
    """
    return {"tokenizer_sha256": load_tokenizer(spec).sha256}


def _local_path(spec: str) -> Path | None:
    """The local path ``spec`` names, or ``None`` for a Hub repository id.

    A path is anything that exists on disk, is absolute or relative (``/``, ``./``, ``../``, ``~``), or ends in
    ``.json``.
    """
    if spec.startswith(("/", "./", "../", "~")) or spec.endswith(".json") or Path(spec).exists():
        return Path(spec).expanduser()
    return None


@functools.lru_cache(maxsize=8)
def load_tokenizer(spec: str) -> TextTokenizer:
    """Load the tokenizer ``spec`` names, once per process.

    Args:
        spec: A Hugging Face repository id with an optional ``@revision`` (``org/model@<commit>``), or a local path
            to a ``tokenizer.json`` or to a directory holding one.

    Returns:
        The :class:`TextTokenizer`, named ``spec``.

    Raises:
        DependencyError: ``tokenizers`` (or, for a Hub id, ``huggingface_hub``) is not installed (``[hf]`` extra).
        MissingInputError: the local file, or the repository's ``tokenizer.json``, does not exist.
    """
    path = _local_path(spec)
    if path is not None:
        file = path / TOKENIZER_FILE if path.is_dir() else path
        if not file.is_file():
            raise MissingInputError(
                f"no tokenizer file at {file}",
                hint=f"judge.tokenizer takes a {TOKENIZER_FILE} path, a directory holding one, or a Hub repository id",
            )
        sidecars = {
            filename: (file.parent / filename).read_bytes()
            for filename in SIDECAR_FILES
            if (file.parent / filename).is_file()
        }
        return TextTokenizer.from_json(file.read_bytes(), name=spec, sidecars=sidecars)
    _backend_class()  # a missing library fails before the download
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError
    except ModuleNotFoundError as exc:
        raise dependency_error("huggingface_hub", needed_for="loading the judge's tokenizer from the Hub") from exc
    repo, _, revision = spec.partition("@")
    try:
        file = hf_hub_download(repo, TOKENIZER_FILE, revision=revision or None)
        sidecars = {}
        for filename in SIDECAR_FILES:
            try:
                sidecars[filename] = Path(hf_hub_download(repo, filename, revision=revision or None)).read_bytes()
            except EntryNotFoundError:
                # The sidecar is optional: the repository ships no such file, or (offline) it is not in the
                # local cache and the two cannot be told apart -- either way the load proceeds without it.
                # The required ``tokenizer.json`` itself stays loud: its uncached-offline case re-raises
                # below, before this branch could catch it.
                continue
    except LocalEntryNotFoundError:
        raise  # offline and not cached: classify() says so
    except EntryNotFoundError as exc:
        raise MissingInputError(
            f"the Hub repository {repo!r} has no {TOKENIZER_FILE}" + (f" at revision {revision}" if revision else ""),
            hint="name a repository that ships a tokenizer.json, or a local tokenizer.json path",
        ) from exc
    return TextTokenizer.from_json(Path(file).read_bytes(), name=spec, sidecars=sidecars)


__all__ = ["SIDECAR_FILES", "TOKENIZER_FILE", "TextTokenizer", "load_tokenizer", "tokenizer_identity"]
