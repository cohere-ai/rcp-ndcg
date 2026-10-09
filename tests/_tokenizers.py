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


@functools.cache
def framed_bpe_tokenizer(name: str = "test/framed-bpe") -> TextTokenizer:
    """A byte-level BPE with two named special tokens and a post-processor anchor, for template tests.

    The two added tokens are the way a chat tokenizer ships them: an end-of-turn marker (a template's
    suffix anchor, written in a template as ``{special:end_turn}``) and an end-of-text marker appended
    by a post-processor when ``add_special_tokens=True`` (the way a pooling route's engine appends the
    checkpoint's end-of-text marker after the client's text). The post-processor makes
    ``count(text, add_special_tokens=True)`` exactly one token longer than ``count(text)``.
    """
    from tokenizers import Tokenizer, processors

    backend = Tokenizer.from_str(byte_bpe_tokenizer().backend.to_str())
    backend.add_special_tokens(["<|end_turn|>", "<|end_of_text|>"])
    eos_id = backend.token_to_id("<|end_of_text|>")
    backend.post_processor = processors.TemplateProcessing(
        single="$A <|end_of_text|>",
        pair="$A $B <|end_of_text|>",
        special_tokens=[("<|end_of_text|>", eos_id)],
    )
    return TextTokenizer.from_backend(backend, name=name)


@functools.cache
def spaced_special_tokenizer(name: str = "test/spaced-special") -> TextTokenizer:
    """A byte-level BPE whose added tokens carry significant whitespace: ``[Q] `` and ``[D] `` end in a
    space, the way pplx-embed-v2-contextual's added tokens ship (one id each; a stripped name cannot be
    written in a template)."""
    from tokenizers import Tokenizer

    backend = Tokenizer.from_str(byte_bpe_tokenizer().backend.to_str())
    backend.add_special_tokens(["[Q] ", "[D] "])
    return TextTokenizer.from_backend(backend, name=name)


def save(tokenizer: TextTokenizer, directory: Path) -> Path:
    """Write ``tokenizer`` as ``directory/tokenizer.json`` and return the file."""
    path = directory / "tokenizer.json"
    path.write_text(tokenizer.backend.to_str(), encoding="utf-8")
    return path


@functools.cache
def vendored_qwen3_vl_tokenizer() -> TextTokenizer:
    """The Qwen3-VL-Embedding checkpoint's own tokenizer from the committed store (no network).

    The E1 video reproduction and the exact-timestamp fit test need the checkpoint's own timestamp
    tokenisation; a fake tokenizer would only test the fake. The store is the unpublished test
    distribution's (``rcp-ndcg-test/corpora/vllm-0.31.0/_tokenizers``).
    """
    import gzip
    import json

    store = Path(__file__).resolve().parents[1] / "rcp-ndcg-test/corpora/vllm-0.31.0/_tokenizers"
    index = json.loads((store / "index.json").read_text(encoding="utf-8"))
    entry = index["Qwen/Qwen3-VL-Embedding-2B@9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda"]
    data = gzip.decompress((store / entry["file"]).read_bytes())
    return TextTokenizer.from_json(data, name="qwen3-vl-embedding")
