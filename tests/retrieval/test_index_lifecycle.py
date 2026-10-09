"""The index lifecycle: the payload a record describes, the directory it is read from, and the version.

A2/A3/D3/D4: ``index()`` publishes atomically under a lock and records the sha256 of every payload file;
``search`` refuses a payload the record does not describe (a killed or concurrent build) and a missing one is
a typed error; ``load_index(path)`` reads the payload from *path* -- the record's own ``path`` is provenance,
not the I/O root -- and a remote ``out`` is refused instead of becoming a local directory named ``gs:/...``.
A5: an explicit behaviour version enters the index identity.
"""

from __future__ import annotations

import json
import shutil
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from rcp_ndcg.data import load_dataset
from rcp_ndcg.errors import ConfigError, DataError, IdentityError, MissingInputError
from rcp_ndcg.retrieval import DenseConfig, ServedEmbedding, index, load_index, retrieve, search
from rcp_ndcg.retrieval import _api as retrieval_api
from tests.conftest import SESSION_TOKENIZER

_SERVED_BUDGET: dict[str, Any] = {"tokenizer": str(SESSION_TOKENIZER), "max_tokens": 8192}
_ENCODER: dict[str, Any] = {"api": "openai_embeddings", "model": "stub", "base_url": "fake://seed/7?dim=8"}


def _beir(root: Path, texts: dict[str, str], *, query: str = "slow tortoises") -> Any:
    """A one-query BEIR dataset over *texts* (the same ids, different bodies per corpus)."""
    (root / "qrels").mkdir(parents=True)
    (root / "corpus.jsonl").write_text("".join(json.dumps({"_id": d, "text": t}) + "\n" for d, t in texts.items()))
    (root / "queries.jsonl").write_text(json.dumps({"_id": "q1", "text": query}) + "\n")
    (root / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td1\t1\n")
    return load_dataset(f"beir:{root}")


@pytest.fixture
def dense() -> DenseConfig:
    return DenseConfig(encoder=ServedEmbedding(**_ENCODER, **_SERVED_BUDGET))


class TestThePayloadTheRecordDescribes:
    """A2/D4: nothing is scored that the record does not describe; a missing payload is a typed error."""

    @pytest.mark.parametrize("damage", ["other-corpus", "truncated", "nan", "missing"])
    def test_a_payload_the_record_does_not_describe_is_refused(
        self, tmp_path: Path, dense: DenseConfig, damage: str
    ) -> None:
        dataset = _beir(tmp_path / "ds", {"d1": "tortoises move slowly", "d2": "hares run fast"})
        built = index(dataset, dense, out=tmp_path / "idx")
        vectors = tmp_path / "idx" / "vectors.npy"
        if damage == "other-corpus":
            np.save(vectors, np.ones((2, 8), dtype=np.float32))
        elif damage == "truncated":
            vectors.write_bytes(vectors.read_bytes()[:40])
        elif damage == "nan":
            array = np.load(vectors)
            array[0] = np.nan
            np.save(vectors, array)
        else:
            vectors.unlink()

        expected = MissingInputError if damage == "missing" else DataError
        with pytest.raises(expected, match="the index at|is missing"):
            search(built, dataset, depth=2)

    def test_a_record_without_a_payload_digest_is_refused(self, tmp_path: Path, dense: DenseConfig) -> None:
        """An index.json written before the digest existed describes nothing: it is rebuilt, never scored."""
        dataset = _beir(tmp_path / "ds", {"d1": "tortoises move slowly"})
        built = index(dataset, dense, out=tmp_path / "idx")
        record = json.loads((tmp_path / "idx" / "index.json").read_text(encoding="utf-8"))
        record.pop("payload")
        (tmp_path / "idx" / "index.json").write_text(json.dumps(record), encoding="utf-8")

        with pytest.raises(DataError, match="describes no payload"):
            search(load_index(tmp_path / "idx"), dataset, depth=1)
        assert built.payload, "the build records the digest"

    def test_retrieve_rebuilds_a_missing_payload(self, tmp_path: Path, dense: DenseConfig) -> None:
        """D4: a half-written cache directory is a cache miss for ``retrieve``, not a permanent wedge."""
        dataset = _beir(tmp_path / "ds", {"d1": "tortoises move slowly", "d2": "hares run fast"})
        first = retrieve(dataset, dense, depth=2, out=tmp_path / "idx")
        (tmp_path / "idx" / "vectors.npy").unlink()

        again = retrieve(dataset, dense, depth=2, out=tmp_path / "idx")

        assert again.for_query("q1") == first.for_query("q1")
        assert (tmp_path / "idx" / "vectors.npy").is_file(), "the payload is rebuilt"

    def test_retrieve_rebuilds_a_payload_of_another_corpus(self, tmp_path: Path, dense: DenseConfig) -> None:
        """A killed rebuild leaves the old record beside the new payload: the identity matches the dataset,
        the digest does not, and retrieve rebuilds instead of scoring the mixed pair."""
        dataset = _beir(tmp_path / "ds", {"d1": "tortoises move slowly", "d2": "hares run fast"})
        first = retrieve(dataset, dense, depth=2, out=tmp_path / "idx")
        other = _beir(tmp_path / "other", {"d1": "quantum chromodynamics", "d2": "lattice gauge theory"})
        index(other, dense, out=tmp_path / "scratch")
        shutil.copyfile(tmp_path / "scratch" / "vectors.npy", tmp_path / "idx" / "vectors.npy")

        again = retrieve(dataset, dense, depth=2, out=tmp_path / "idx")

        assert again.for_query("q1") == first.for_query("q1"), "the mixed pair was refused and rebuilt"

    def test_a_concurrent_build_never_leaves_a_pair_that_is_scored(self, tmp_path: Path, dense: DenseConfig) -> None:
        """Two builds of different corpora into one directory, under the lock: whatever the final state,
        ``search`` either matches a record's own corpus or refuses -- never scores one corpus's vectors under
        the other's identity."""
        first = _beir(tmp_path / "one", {"d1": "tortoises move slowly", "d2": "hares run fast"})
        second = _beir(tmp_path / "two", {"d1": "quantum chromodynamics", "d2": "lattice gauge theory"})
        out = tmp_path / "idx"
        alone = search(index(first, dense, out=tmp_path / "alone"), first, depth=2).for_query("q1")

        errors: list[BaseException] = []

        def build(dataset: Any) -> None:
            try:
                index(dataset, dense, out=out)
            except BaseException as exc:  # noqa: BLE001 - reported below
                errors.append(exc)

        threads = [threading.Thread(target=build, args=(dataset,)) for dataset in (first, second)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert not errors

        loaded = load_index(out)
        try:
            got = search(loaded, first, depth=2).for_query("q1")
        except (DataError, MissingInputError, IdentityError):
            return  # refused: the pair is not one build's, or the record is the other corpus's
        assert got == alone, "a scored answer must be the record's own corpus"


class TestTheDirectoryTheIndexIsReadFrom:
    """A3: the record's ``path`` is provenance; the payload is read from the directory ``load_index`` got."""

    def test_a_copied_index_is_read_where_it_now_is(self, tmp_path: Path, dense: DenseConfig) -> None:
        first = _beir(tmp_path / "a", {"d1": "tortoises move slowly", "d2": "hares run fast"})
        second = _beir(tmp_path / "b", {"d1": "quantum chromodynamics", "d2": "lattice gauge theory"})
        index(first, dense, out=tmp_path / "idxA")
        shutil.copytree(tmp_path / "idxA", tmp_path / "idxB")
        index(second, dense, out=tmp_path / "idxA")  # the recorded path now holds another corpus's payload
        expected = search(index(first, dense, out=tmp_path / "fresh"), first, depth=2).for_query("q1")

        loaded = load_index(tmp_path / "idxB")
        assert loaded.path == str(tmp_path / "idxB"), "the record reads from the directory it was loaded from"
        assert search(loaded, first, depth=2).for_query("q1") == expected

        shutil.rmtree(tmp_path / "idxA")
        assert search(load_index(tmp_path / "idxB"), first, depth=2).for_query("q1") == expected

    def test_a_remote_directory_is_refused_and_never_created(self, tmp_path: Path, monkeypatch, dense) -> None:
        """D3: a remote ``out`` is refused with a hint, never a local directory named ``gs:/...``."""
        monkeypatch.chdir(tmp_path)
        dataset = _beir(tmp_path / "ds", {"d1": "tortoises move slowly"})

        with pytest.raises(ConfigError, match="local or shared-filesystem") as caught:
            index(dataset, dense, out="gs://YOUR-BUCKET/idx")
        assert "mirror" in (caught.value.hint or "")
        with pytest.raises(ConfigError, match="local or shared-filesystem"):
            load_index("gs://YOUR-BUCKET/idx")
        assert not Path("gs:/YOUR-BUCKET/idx").exists(), "no local directory named after the bucket"


class TestTheBehaviourVersion:
    """A5: an explicit version enters the identity of each output-producing step, bumped deliberately."""

    def test_the_index_version_is_part_of_the_index_identity(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from rcp_ndcg.retrieval import BM25Config

        before = retrieval_api._identity(BM25Config(), ["d1"], [])
        monkeypatch.setattr(retrieval_api, "INDEX_BEHAVIOUR_VERSION", "999")

        assert retrieval_api._identity(BM25Config(), ["d1"], []) != before


class TestTheMediaBytesInTheIdentity:
    """A6: a media reference without ``sha256`` (the reader default) records its object's size and change
    stamp, so replacing the bytes at the same URI moves the index identity."""

    @staticmethod
    def _content(path: Path) -> Any:
        from rcp_ndcg_core.content import Content, ImagePart, MediaRef

        return Content.from_parts([ImagePart(ref=MediaRef(uri=str(path), mime="image/png"))])

    def test_replacing_the_bytes_moves_the_identity(self, tmp_path: Path) -> None:
        from rcp_ndcg.retrieval import BM25Config

        page = tmp_path / "page.png"
        page.write_bytes(b"first")
        content = self._content(page)
        before = retrieval_api._identity(BM25Config(), ["d1"], [content])

        page.write_bytes(b"other")  # same size, changed bytes and mtime

        assert retrieval_api._identity(BM25Config(), ["d1"], [content]) != before

    def test_a_hashed_reference_keys_by_its_hash(self, tmp_path: Path) -> None:
        from rcp_ndcg_core.content import Content, ImagePart, MediaRef

        from rcp_ndcg.retrieval import BM25Config

        page = tmp_path / "page.png"
        page.write_bytes(b"first")
        first = Content.from_parts([ImagePart(ref=MediaRef(uri=str(page), sha256="a" * 64))])
        second = Content.from_parts([ImagePart(ref=MediaRef(uri=str(page), sha256="b" * 64))])

        assert retrieval_api._identity(BM25Config(), ["d1"], [first]) != retrieval_api._identity(
            BM25Config(), ["d1"], [second]
        )


class TestTheLateInteractionLayout:
    """A7: a pooled answer must never become a late-interaction index (it scored as a dense inner product,
    silently, and the poisoned index was then reused by identity)."""

    def test_index_refuses_a_single_vector_document_buffer(self, tmp_path: Path, monkeypatch) -> None:
        from rcp_ndcg.inference.types import Embeddings
        from rcp_ndcg.retrieval import LateInteractionConfig, ServedPooling

        dataset = _beir(tmp_path / "ds", {"d1": "tortoises move slowly", "d2": "hares run fast"})
        config = LateInteractionConfig(
            encoder=ServedPooling(api="vllm_pooling", model="colqwen", base_url="fake://seed/3?dim=4", dim=4,
                                  **_SERVED_BUDGET)
        )
        pooled = Embeddings.single(np.ones((2, 4), dtype=np.float32))
        monkeypatch.setattr(retrieval_api, "_encode", lambda *args, **kwargs: pooled)

        with pytest.raises(DataError, match="one vector per document"):
            index(dataset, config, out=tmp_path / "idx")

    def test_search_refuses_a_late_interaction_record_without_offsets(self, tmp_path: Path) -> None:
        from rcp_ndcg.retrieval import Index, LateInteractionConfig, ServedPooling

        dataset = _beir(tmp_path / "ds", {"d1": "tortoises move slowly", "d2": "hares run fast"})
        config = LateInteractionConfig(
            encoder=ServedPooling(api="vllm_pooling", model="colqwen", base_url="fake://seed/3?dim=4", dim=4,
                                  **_SERVED_BUDGET)
        )
        root = tmp_path / "idx"
        root.mkdir()
        np.save(root / "vectors.npy", np.ones((2, 4), dtype=np.float32))
        doc_ids, contents = retrieval_api._corpus(dataset)
        record = Index(
            path=str(root),
            dataset=dataset.name,
            retriever=config,
            identity=retrieval_api._identity(config, doc_ids, contents),
            num_documents=len(doc_ids),
            payload=retrieval_api._payload_digest(root),
        )
        (root / "index.json").write_text(record.model_dump_json(by_alias=True), encoding="utf-8")

        with pytest.raises(DataError, match="no offsets.npy") as caught:
            search(record, dataset, depth=2)

        assert "rebuild" in (caught.value.hint or "")


class TestTheEmptyEdges:
    """A11/A12/E: an all-empty side under ``omit_zero`` is refused by name, a zero-query dataset is refused
    by name, an empty candidate set is refused by name, and duplicate query ids are refused (a query's
    scores are checkpointed under its id)."""

    def test_an_all_empty_document_side_under_omit_zero_is_refused_by_name(
        self, tmp_path: Path, dense: DenseConfig
    ) -> None:
        """A12: every document empty and ``empty_doc: omit_zero`` builds a zero-width index that used to
        fail on the next search with a late-interaction message."""
        dataset = _beir(tmp_path / "ds", {"d1": "", "d2": ""})
        config = DenseConfig(encoder=ServedEmbedding(empty_doc="omit_zero", **_ENCODER, **_SERVED_BUDGET))

        with pytest.raises(DataError, match="zero-width document vectors") as caught:
            index(dataset, config, out=tmp_path / "idx")

        assert "empty_doc" in (caught.value.hint or "")

    def test_a_zero_query_dataset_is_refused_by_name(self, tmp_path: Path, dense: DenseConfig) -> None:
        """A zero-query dataset used to die inside numpy_topk with "embeddings must be aligned 2D matrices"."""
        from rcp_ndcg.data import Dataset

        dataset = Dataset.from_records(
            name="no-queries", corpus=[{"doc_id": "d1", "text": "tortoises move slowly"}], queries=[]
        )
        built = index(dataset, dense, out=tmp_path / "idx")

        with pytest.raises(DataError, match="holds no queries") as caught:
            search(built, dataset, depth=1)

        assert "query source" in (caught.value.hint or "")

    def test_an_empty_candidate_set_is_refused_by_name(self, tmp_path: Path, dense: DenseConfig) -> None:
        """E: ``rerank`` returned an empty Rankings (no system at all) for a system with no candidates,
        contradicting its docstring; the mismatch is refused instead."""
        from rcp_ndcg.data import Rankings
        from rcp_ndcg.retrieval import ServedReranker, rerank

        dataset = _beir(tmp_path / "ds", {"d1": "tortoises move slowly"})
        rankings = Rankings.from_records(
            [
                {"system": "a", "dataset": dataset.name, "query_id": "q1", "doc_id": "d1", "score": 1.0},
                {"system": "b", "dataset": "elsewhere", "query_id": "q1", "doc_id": "d1", "score": 1.0},
            ]
        )
        config = ServedReranker(
            api="rerank",
            model="stub-reranker",
            base_url="fake://seed/1",
            instruction="fold",
            tokenizer=str(SESSION_TOKENIZER),
            max_tokens=8192,
            use_activation=False,
        )

        with pytest.raises(DataError, match="no candidates for"):
            rerank(dataset, rankings, config, system="b", depth=2)

    def test_duplicate_query_ids_are_refused(self, tmp_path: Path, dense: DenseConfig) -> None:
        """A11: the checkpoint callback keys on the query id, so duplicates would be attributed to one
        another."""
        from rcp_ndcg_core._records import RankingExample

        from rcp_ndcg.retrieval import ServedReranker

        config = ServedReranker(
            api="rerank",
            model="stub-reranker",
            base_url="fake://seed/1",
            instruction="fold",
            tokenizer=str(SESSION_TOKENIZER),
            max_tokens=8192,
            use_activation=False,
        )
        examples = [
            RankingExample(query_id="q1", query="a", doc_ids=["d1"], docs=["one"]),
            RankingExample(query_id="q1", query="b", doc_ids=["d1"], docs=["one"]),
        ]

        with pytest.raises(DataError, match="more than once") as caught:
            retrieval_api._rerank_examples(examples, config, checkpoint_dir=None)

        assert "unique" in (caught.value.hint or "")
