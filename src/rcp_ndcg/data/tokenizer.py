"""The judge's tokenizer: text limits are counted in the tokens the judge reads.

A judge names its tokenizer (``JudgeConfig.tokenizer``): a Hugging Face repository id with an optional
``@revision`` (``Qwen/Qwen3.5-397B-A17B-FP8``, ``org/model@<commit>``), or a local path to a ``tokenizer.json``
(or to a directory holding one). :func:`load_tokenizer` loads it once per process with the ``tokenizers`` library
(no torch), and :class:`TextTokenizer` counts tokens and reports where each token sits in the original text, so
that text is cut at token boundaries of the document itself (:mod:`rcp_ndcg.data.preprocess`), never by decoding
tokens back to text.

The tokenizer's identity is the SHA-256 of its ``tokenizer.json``: two passes whose judges tokenize differently never
pool (it is recorded in the judgement family and the pass's preprocessing identity).
"""

from __future__ import annotations

import functools
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rcp_ndcg.errors import ConfigError, MissingInputError, dependency_error

#: The file a tokenizer is read from, locally or in a Hub repository.
TOKENIZER_FILE = "tokenizer.json"


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


@dataclass(frozen=True)
class TextTokenizer:
    """A loaded tokenizer: counts tokens and locates them in the original text.

    Special tokens (a BOS marker, a chat template's role tokens) are never added: the counts are of the text alone.

    Attributes:
        name: The tokenizer as the judge names it (repository id with its revision, or a path).
        sha256: SHA-256 of its ``tokenizer.json`` (for a tokenizer built in memory, of its JSON serialisation).
    """

    name: str
    sha256: str
    backend: Any = field(repr=False, compare=False)

    @classmethod
    def from_json(cls, data: bytes, *, name: str) -> TextTokenizer:
        """The tokenizer serialised in ``data`` (the bytes of a ``tokenizer.json``)."""
        backend = _backend_class().from_str(data.decode("utf-8"))
        return cls(name=name, sha256=hashlib.sha256(data).hexdigest(), backend=backend)

    @classmethod
    def from_backend(cls, backend: Any, *, name: str) -> TextTokenizer:
        """Wrap a ``tokenizers.Tokenizer`` built in memory; its identity is the hash of its serialisation."""
        return cls.from_json(backend.to_str().encode("utf-8"), name=name)

    def count(self, text: str, *, add_special_tokens: bool = False) -> int:
        """The number of tokens of ``text``; with ``add_special_tokens``, as the engine counts it
        (the tokenizer's post-processor tokens included, e.g. an appended end-of-text anchor)."""
        return len(self.backend.encode(text, add_special_tokens=add_special_tokens).ids)

    def ids(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        """The token ids of ``text``, optionally with the post-processor's (as the engine reads them)."""
        return list(self.backend.encode(text, add_special_tokens=add_special_tokens).ids)

    def offsets(self, text: str) -> list[tuple[int, int]]:
        """``(start, end)`` character offsets in ``text`` of each of its tokens, in order."""
        return [tuple(offset) for offset in self.backend.encode(text, add_special_tokens=False).offsets]  # type: ignore[misc]

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

        Raises:
            ConfigError: the tokenizer has no added token by that name; the message lists the names it has.
        """
        tokens = self.added_tokens()
        try:
            return tokens[name]
        except KeyError:
            known = sorted({key for key in tokens if not key.startswith("<|")})
            raise ConfigError(
                f"the tokenizer {self.name!r} has no special token named {name!r}",
                hint=f"its added tokens are named {known}; write one as {{special:<name>}} in the template",
            ) from None

    def special_id(self, name: str) -> int:
        """The token id of the added token named ``name`` (as :meth:`special_text` resolves it).

        Raises:
            ConfigError: the tokenizer has no added token by that name.
        """
        return self.backend.token_to_id(self.special_text(name))


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
        return TextTokenizer.from_json(file.read_bytes(), name=spec)
    _backend_class()  # a missing library fails before the download
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError
    except ModuleNotFoundError as exc:
        raise dependency_error("huggingface_hub", needed_for="loading the judge's tokenizer from the Hub") from exc
    repo, _, revision = spec.partition("@")
    try:
        file = hf_hub_download(repo, TOKENIZER_FILE, revision=revision or None)
    except LocalEntryNotFoundError:
        raise  # offline and not cached: classify() says so
    except EntryNotFoundError as exc:
        raise MissingInputError(
            f"the Hub repository {repo!r} has no {TOKENIZER_FILE}" + (f" at revision {revision}" if revision else ""),
            hint="name a repository that ships a tokenizer.json, or a local tokenizer.json path",
        ) from exc
    return TextTokenizer.from_json(Path(file).read_bytes(), name=spec)


__all__ = ["TOKENIZER_FILE", "TextTokenizer", "load_tokenizer"]
