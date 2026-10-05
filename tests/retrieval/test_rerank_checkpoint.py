"""The per-query rerank checkpoint: the record format, the fsync, and the resume.

The checkpoint is what makes a crashed rerank cheap: every scored query is one JSON record (``{"q", "k", "s"}``)
appended to ``rank000.jsonl`` and fsynced, and a rerun with the same reranker over the same candidates skips the
queries the checkpoint holds. The key and the record format are the served path's historical ones, so a rerun
also resumes a checkpoint an earlier release wrote -- that is what ``test_an_old_format_checkpoint_resumes``
pins, writing the file with the key payload the release's served path computed.
"""

from __future__ import annotations

import json
import os
from typing import Any
from unittest import mock

import pytest
from rcp_ndcg_core._records import RankingExample

from rcp_ndcg.retrieval import _api as retrieval_api
from rcp_ndcg.retrieval import rerank
from rcp_ndcg.retrieval.config import ServedReranker
from rcp_ndcg.support.identity import hash_payload, short


def _config(**kwargs: Any) -> ServedReranker:
    return ServedReranker(base_url="fake://seed/1", **{"model": "stub-reranker", **kwargs})


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


def test_an_old_format_checkpoint_resumes(tmp_path: Any) -> None:
    """A checkpoint written by the earlier release (its key payload, its file name) is read, not rescored."""
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
    key = short(hash_payload(payload), 16)

    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    scores = {"d1": 0.5, "d2": 0.25, "d3": 0.125}
    (ckpt / "rank000.jsonl").write_text(json.dumps({"q": str(example.id), "k": key, "s": scores}) + "\n")

    scored = retrieval_api._rerank_examples([example], config, checkpoint_dir=ckpt)

    assert scored[0].scores == [0.5, 0.25, 0.125], "the scores come back from the old checkpoint"
    assert retrieval_api._checkpoint_key(config, example) == key, "the same key the old writer computed"


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
