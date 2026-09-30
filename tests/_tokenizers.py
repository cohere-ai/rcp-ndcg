"""Tiny tokenizers built in memory, for the token-limit tests (no download, no model files)."""

from __future__ import annotations

import functools
from pathlib import Path

from rcp_ndcg.data.tokenizer import TextTokenizer

#: The words the word-level tokenizer knows; any other word or punctuation run is one ``[UNK]`` token.
WORDS = "the a of to and in is it evidence query document passage relevant answer page one two three four five"


@functools.cache
def word_tokenizer(name: str = "test/word-level") -> TextTokenizer:
    """One token per word or punctuation run (``Whitespace`` pre-tokenizer); whitespace is no token."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers

    vocab = {"[UNK]": 0, **{word: index + 1 for index, word in enumerate(WORDS.split())}}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.decoder = decoders.WordPiece()  # joins tokens with single spaces
    return TextTokenizer.from_backend(backend, name=name)


@functools.cache
def byte_bpe_tokenizer(name: str = "test/byte-bpe") -> TextTokenizer:
    """A byte-level BPE (the GPT-2 / Qwen kind) trained on a few sentences: words split into sub-word tokens.

    Trained once per process (a third of a second).
    """
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=320, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False
    )
    corpus = [
        "The relevant passage answers the query with evidence.",
        "Unbelievable results: the document is relevant & precise.",
        "Ünïcödé text — 日本語 and emoji 🙂 appear in documents.",
    ] * 3
    backend.train_from_iterator(corpus, trainer)
    return TextTokenizer.from_backend(backend, name=name)


def save(tokenizer: TextTokenizer, directory: Path) -> Path:
    """Write ``tokenizer`` as ``directory/tokenizer.json`` and return the file."""
    path = directory / "tokenizer.json"
    path.write_text(tokenizer.backend.to_str(), encoding="utf-8")
    return path
