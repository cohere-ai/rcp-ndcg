"""The per-query rerank checkpoint: the record format, the fsync, and the resume.

The checkpoint is what makes a crashed rerank cheap: every scored query is one JSON record (``{"q", "k", "s"}``)
appended to ``rank000.jsonl`` and fsynced, and a rerun with the same reranker over the same candidates skips the
queries the checkpoint already holds. The key is the reranker's content identity plus the exact texts sent (the
record format is still the served path's), so a rerun after any content change -- the config's, the query's, or
the candidates' -- is scored again instead of resuming stale scores; a checkpoint written before that key
existed (the earlier release keyed on ids and historical budget constants only) is re-scored too.
"""

from __future__ import annotations

import json
import os
from typing import Any
from unittest import mock

import pytest
from rcp_ndcg_core.records import RankingExample

from rcp_ndcg.data import media
from rcp_ndcg.retrieval import _api as retrieval_api
from rcp_ndcg.retrieval import rerank
from rcp_ndcg.retrieval._api import _checkpoint_key
from rcp_ndcg.retrieval.config import CohereReranker, ServedReranker
from rcp_ndcg.support.identity import hash_payload, short
from tests.conftest import SESSION_TOKENIZER

_SERVED_BUDGET: dict[str, Any] = {"tokenizer": str(SESSION_TOKENIZER), "max_tokens": 8192}
_SERVED_RERANK_BUDGET: dict[str, Any] = {**_SERVED_BUDGET, "use_activation": False}


def _config(**kwargs: Any) -> ServedReranker:
    return ServedReranker(base_url="fake://seed/1", **{"model": "stub-reranker", **_SERVED_RERANK_BUDGET, **kwargs})


def _examples() -> list[RankingExample]:
    return [
        RankingExample(
            query_id="q1",
            query="capital of france",
            doc_ids=["d1", "d2", "d3"],
            docs=["paris is the capital of france", "lyon is a city", "unrelated text"],
        ),
        RankingExample(
            query_id="q2",
            query="who wrote moby dick",
            doc_ids=["d4", "d5"],
            docs=["herman melville wrote moby dick", "ishmael narrates"],
        ),
    ]


def _records(path: Any) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_records_are_appended_flushed_and_fsynced_per_query(tmp_path: Any, monkeypatch: Any) -> None:
    fsyncs: list[int] = []
    real_fsync = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (fsyncs.append(fd), real_fsync(fd))[1])

    retrieval_api._rerank_examples(_examples(), _config(), checkpoint_dir=tmp_path / "ckpt")

    records = _records(tmp_path / "ckpt" / "rank000.jsonl")
    assert [record["q"] for record in records] == ["q1", "q2"]
    for record in records:
        assert set(record) == {"q", "k", "s"}, "the served path's record format"
        assert record["k"] and isinstance(record["s"], dict)
    assert len(fsyncs) >= len(records), "one fsync per scored query, so a crash costs the in-flight ones"


def test_a_resumed_rerank_skips_the_scored_queries_and_keeps_their_scores(tmp_path: Any) -> None:
    ckpt = tmp_path / "ckpt"
    first = retrieval_api._rerank_examples(_examples(), _config(), checkpoint_dir=ckpt)
    assert len(_records(ckpt / "rank000.jsonl")) == 2

    # A second run over the same examples scores nothing: every query is in the checkpoint.
    with mock.patch.object(retrieval_api.RerankClient, "arerank", side_effect=AssertionError("no request")):
        second = retrieval_api._rerank_examples(_examples(), _config(), checkpoint_dir=ckpt)

    assert [e.scores for e in second] == [e.scores for e in first]


def test_the_key_covers_every_content_field_and_the_exact_texts() -> None:
    """The key is what a checkpointed query's scores are valid for: a rerun after the instruction mode, the
    activation switch, the recipe, the wire adapter, the query's text or the documents' contents changed must
    not resume the old scores. Only model, revision and budgets used to re-key it."""
    example = _examples()[0]
    base = _checkpoint_key(_config(), example)

    changed_configs = {
        "instruction": _config(instruction="none"),
        "use_activation": _config(use_activation=True),
        "recipe": _config(recipe="qwen3-v2"),
        "api": CohereReranker(model="stub-reranker"),
    }
    for label, config in changed_configs.items():
        assert _checkpoint_key(config, example) != base, f"{label} is content"

    changed_texts = {
        "another query": _examples()[1],
        "query text": example.model_copy(update={"text": "a different question entirely"}),
        "query instruction": example.model_copy(update={"instruction": "Find the passage"}),
        "document texts": example.model_copy(update={"docs": ["changed text", "other text", "third text"]}),
    }
    for label, changed in changed_texts.items():
        assert _checkpoint_key(_config(), changed) != base, f"{label} is content"

    # The run's task instruction is what the model was asked to do: another one scores the query again (the
    # example carries no task instruction of its own, so the None case differs from every declared one too).
    assert _checkpoint_key(_config(), example, task_instruction="Find relevant passages") != base
    assert _checkpoint_key(_config(), example, task_instruction=None) != _checkpoint_key(
        _config(), example, task_instruction="Find relevant passages"
    )
    assert _checkpoint_key(_config(), example) == base, "the same content keys the same"


def test_the_key_covers_a_media_query_s_parts_not_only_its_text(tmp_path: Any) -> None:
    """Two image queries with the same (empty) text but different images are different queries, and a
    replaced image at the same URI re-keys (an unhashed reference's size and change stamp are content)."""
    from rcp_ndcg_core.content import Content, ImagePart, MediaRef

    def image_query(uri: str) -> RankingExample:
        fields = _examples()[0].model_dump(by_alias=True, exclude={"content"})
        fields["query"] = ""
        return RankingExample.model_validate(
            {**fields, "content": Content.from_parts([ImagePart(ref=MediaRef(uri=uri, mime="image/png"))])}
        )

    one_path, other_path = tmp_path / "one.png", tmp_path / "other.png"
    one_path.write_bytes(b"one image")
    other_path.write_bytes(b"other image")
    one, other = image_query(str(one_path)), image_query(str(other_path))
    assert one.text == other.text
    assert _checkpoint_key(_config(), one) != _checkpoint_key(_config(), other)

    before = _checkpoint_key(_config(), one)
    one_path.write_bytes(b"new image")  # the same length, changed bytes and mtime
    media._OBJECT_INFO_CACHE.clear()  # the next run's view: the object-info memo lives for one process
    assert _checkpoint_key(_config(), one) != before, "the replaced image's bytes are content"


def test_a_budget_change_re_keys_the_checkpoint(tmp_path: Any) -> None:
    """``max_tokens`` and ``query_max_tokens`` decide what text reaches the model, so they re-key; the
    tokenizer's digest does too (same bytes under another path never does)."""
    from tests._tokenizers import byte_bpe_tokenizer, save, word_tokenizer

    example = _examples()[0]
    base = _checkpoint_key(_config(), example)

    assert _checkpoint_key(_config(max_tokens=512), example) != base
    assert _checkpoint_key(_config(query_max_tokens=128), example) != base

    for directory in ("one", "two", "other"):
        (tmp_path / directory).mkdir()
    first = save(word_tokenizer(), tmp_path / "one")
    second = save(word_tokenizer(), tmp_path / "two")  # same bytes, different path
    other = save(byte_bpe_tokenizer(), tmp_path / "other")

    # The base config already declares the session's word tokenizer (a served config must): the same bytes
    # under another path keep the key, other bytes re-key it.
    assert _checkpoint_key(_config(tokenizer=str(first)), example) == base, "the digest, not the path"
    assert _checkpoint_key(_config(tokenizer=str(second)), example) == base, "the digest, not the path"
    assert _checkpoint_key(_config(tokenizer=str(other)), example) != base, "different tokenizer bytes re-key"


def test_an_old_format_checkpoint_is_scored_again(tmp_path: Any) -> None:
    """A checkpoint written by the earlier release (its key payload: model, the ``vllm`` framework name, the
    historical budget constants, the ids) does not match the content key: the query is scored again instead of
    resuming scores computed for other texts. Nothing is released yet, so no checkpoint in the wild carries the
    old key (CHANGELOG); the old record stays where it is and a fresh one lands beside it."""
    example = _examples()[0]
    config = _config(revision="cafe1234")
    # The served path's historical key: model, the "vllm" framework name, the budgets as constants.
    payload = {
        "model": config.model,
        "framework": "vllm",
        "revision": config.revision,
        "max_seq_length": 8192,
        "max_query_length": 4096,
        "query_id": str(example.id),
        "doc_ids": [str(doc_id) for doc_id in example.doc_ids],
    }
    old_key = short(hash_payload(payload), 16)

    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "rank000.jsonl").write_text(
        json.dumps({"q": str(example.id), "k": old_key, "s": {"d1": 0.5, "d2": 0.25, "d3": 0.125}}) + "\n"
    )

    scored = retrieval_api._rerank_examples([example], config, checkpoint_dir=ckpt)
    fresh = retrieval_api._rerank_examples([example], config, checkpoint_dir=None)  # what a run without it scores

    assert len(_records(ckpt / "rank000.jsonl")) == 2, "the query was scored again: a record was appended"
    assert scored[0].scores == fresh[0].scores, "the fresh scores, not the old record's"
    assert _checkpoint_key(config, example) != old_key, "the content key is not the historical one"


def test_a_checkpoint_record_missing_a_document_is_scored_again(tmp_path: Any) -> None:
    """A record that misses one of its example's documents (historical or hand-edited) is dropped and the
    query scored again: resuming it would fail with a refusal whose hint -- rerun the rerank -- replayed the
    identical failure forever."""
    example = _examples()[0]
    config = _config()
    key = _checkpoint_key(config, example)
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "rank000.jsonl").write_text(json.dumps({"q": "q1", "k": key, "s": {"d1": 0.5, "d2": 0.25}}) + "\n")

    first = retrieval_api._rerank_examples([example], config, checkpoint_dir=ckpt)
    second = retrieval_api._rerank_examples([example], config, checkpoint_dir=ckpt)

    assert first[0].scores == second[0].scores, "the second run resumes the fresh complete record"
    assert len(_records(ckpt / "rank000.jsonl")) == 2, "exactly one re-score: the partial record was dropped"


def test_a_record_with_an_unparseable_score_is_dropped_not_a_crash(tmp_path: Any) -> None:
    """A value that is not a number poisons only its own record (the reader's tolerance for foreign files):"""
    example = _examples()[0]
    config = _config()
    key = _checkpoint_key(config, example)
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "rank000.jsonl").write_text(json.dumps({"q": "q1", "k": key, "s": {"d1": "oops"}}) + "\n")

    scored = retrieval_api._rerank_examples([example], config, checkpoint_dir=ckpt)

    assert len(scored[0].scores or []) == 3, "the record was dropped and the query scored"


def test_another_reranker_or_a_deeper_pool_is_scored_again(tmp_path: Any) -> None:
    """The checkpoint once reused reranker A's scores for reranker B, and crashed on a deeper rerun."""
    ckpt = tmp_path / "shared"
    shallow = [
        example.model_copy(update={"doc_ids": example.doc_ids[:1], "docs": example.docs[:1]}) for example in _examples()
    ]
    retrieval_api._rerank_examples(shallow, _config(model="A"), checkpoint_dir=ckpt)

    other = retrieval_api._rerank_examples(shallow, _config(model="B"), checkpoint_dir=ckpt)
    deeper = retrieval_api._rerank_examples(_examples(), _config(model="B"), checkpoint_dir=ckpt)

    assert other is not None and deeper is not None
    assert [len(e.scores or []) for e in deeper] == [3, 2]


def test_a_truncated_checkpoint_line_does_not_lose_the_rest(tmp_path: Any) -> None:
    """The usual crash signature is a process that died mid-flush."""
    ckpt = tmp_path / "partial"
    ckpt.mkdir()
    (ckpt / "rank000.jsonl").write_text('{"q": "q1", "k": "k1", "s": {"d1": 0.5}}\n{"q": "q2", "k": "k2", "s": {"d4"\n')

    assert retrieval_api._checkpoint_scores(ckpt) == {"k1": {"d1": 0.5}}


def test_an_unscored_document_is_refused_rather_than_given_a_score() -> None:
    with pytest.raises(retrieval_api.DataError, match="no score"):
        retrieval_api._apply_scores(_examples(), ["k1", "k2"], {"k1": {"d1": 0.5, "d2": 0.25, "d3": 0.125}})


def test_the_rerank_step_folds_the_instruction_exactly_once(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """The example's raw query and instruction reach the client, which folds them once: never a double fold."""
    import httpx

    from rcp_ndcg.data import Rankings, load_dataset

    sent: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        documents = sent[-1]["documents"]
        return httpx.Response(
            200, json={"results": [{"index": i, "relevance_score": float(i)} for i in range(len(documents))]}
        )

    from rcp_ndcg.inference import transport as transport_module

    real = transport_module.Transport

    def patched(endpoint: Any, **kwargs: Any) -> Any:
        return real(endpoint, httpx_transport=httpx.MockTransport(handler))

    monkeypatch.setattr("rcp_ndcg.inference.clients._base.Transport", patched)

    root = tmp_path / "beir"
    (root / "qrels").mkdir(parents=True)
    (root / "corpus.jsonl").write_text(
        "".join(json.dumps({"_id": d, "text": t}) + "\n" for d, t in (("d1", "paris"), ("d2", "lyon")))
    )
    (root / "queries.jsonl").write_text(
        json.dumps({"_id": "q1", "text": "capital of france", "instruction": "Find the relevant passage"}) + "\n"
    )
    (root / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td1\t1\n")
    dataset = load_dataset(f"beir:{root}")
    rankings = Rankings.from_scores({"q1": {"d1": 3.0, "d2": 2.0}}, system="bm25")

    rerank(dataset, rankings, _config(), depth=2, out=tmp_path / "rerank")

    assert [call["query"] for call in sent] == ["capital of france Find the relevant passage"], (
        "the per-query instruction is appended exactly once"
    )


def test_the_rerank_step_checkpoints_and_resumes(tmp_path: Any) -> None:
    """``rerank()`` writes the same records the served path did, and its resume appends nothing."""
    from rcp_ndcg.data import Rankings, load_dataset

    root = tmp_path / "beir"
    (root / "qrels").mkdir(parents=True)
    (root / "corpus.jsonl").write_text(
        "".join(
            json.dumps({"_id": d, "text": t}) + "\n"
            for d, t in (("d1", "paris"), ("d2", "lyon"), ("d3", "nice"), ("d4", "moby dick"), ("d5", "whale"))
        )
    )
    (root / "queries.jsonl").write_text(
        json.dumps({"_id": "q1", "text": "capital of france"})
        + "\n"
        + json.dumps({"_id": "q2", "text": "moby dick author"})
        + "\n"
    )
    (root / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td1\t1\nq2\td4\t1\n")
    dataset = load_dataset(f"beir:{root}")
    rankings = Rankings.from_scores(
        {"q1": {"d1": 3.0, "d2": 2.0, "d3": 1.0}, "q2": {"d4": 1.0, "d5": 2.0}}, system="bm25"
    )

    rerank(dataset, rankings, _config(), depth=2, out=tmp_path / "rerank")
    first_records = _records(tmp_path / "rerank" / "rank000.jsonl")

    rerank(dataset, rankings, _config(), depth=2, out=tmp_path / "rerank")
    second_records = _records(tmp_path / "rerank" / "rank000.jsonl")

    assert len(first_records) == 2
    assert len(second_records) == 2, "every query was already scored: nothing appended on the rerun"
