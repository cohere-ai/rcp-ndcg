"""The full-width embedding store and the ex-post MRL sweep.

The store persists corpus and query vectors at full width (one forward pass), and the sweep applies the
declared MRL head per k and scores from the stored vectors: the rankings must equal a direct per-k run
through the same fake engine, because the cut is deterministic and the fake's draw is.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from rcp_ndcg.data.embedding_store import (
    EmbeddingStore,
    StoredVectors,
    load_embedding_store,
    save_embedding_store,
)
from rcp_ndcg.errors import ConfigError, DataError
from rcp_ndcg.retrieval import (
    DenseConfig,
    LateInteractionConfig,
    ServedEmbedding,
    ServedPooling,
    build_store,
    load_store,
    retrieve,
    sweep,
)
from tests._safetensors import write_safetensors
from tests.conftest import SESSION_TOKENIZER

_BUDGET: dict[str, Any] = {"tokenizer": str(SESSION_TOKENIZER), "max_tokens": 8192}


def _dense(**overrides: Any) -> DenseConfig:
    settings: dict[str, Any] = {
        "base_url": "fake://seed/7?dim=8",
        "model": "stub",
        "mrl_kind": "truncation",
        "mrl_dims": (2, 4),
        **_BUDGET,
    }
    settings.update(overrides)
    return DenseConfig(encoder=ServedEmbedding(**settings))


def _late(**overrides: Any) -> LateInteractionConfig:
    settings: dict[str, Any] = {
        "base_url": "fake://seed/3?dim=4",
        "model": "colqwen",
        "dim": 4,
        "mrl_kind": "truncation",
        "mrl_dims": (2,),
        **_BUDGET,
    }
    settings.update(overrides)
    return LateInteractionConfig(encoder=ServedPooling(**settings))


class TestStoreRoundTrip:
    def test_the_store_holds_full_width_corpus_and_query_vectors(self, dataset: Any, tmp_path: Path) -> None:
        """The config selects k=2, and the store is still 8-wide: k never enters the engine request."""
        record = build_store(dataset, _dense(mrl_dim=2), out=tmp_path / "store")

        assert record.full_width == 8
        assert record.layout == "single"
        assert record.count == 3 and record.query_count == 2
        assert record.doc_ids == ["d1", "d2", "d3"]
        assert record.query_ids == ["q1", "q2"]
        assert record.mrl_kind == "truncation" and record.mrl_dims == (2, 4)
        loaded, corpus, queries = load_store(tmp_path / "store")
        assert loaded == record
        assert corpus.as_matrix().shape == (3, 8)
        assert queries.as_matrix().shape == (2, 8)
        np.testing.assert_array_equal(np.load(tmp_path / "store" / "corpus.npy"), corpus.as_matrix())

    def test_the_identity_ignores_the_selected_k(self, dataset: Any, tmp_path: Path) -> None:
        first = build_store(dataset, _dense(mrl_dim=2), out=tmp_path / "a")
        second = build_store(dataset, _dense(mrl_dim=4), out=tmp_path / "b")
        assert first.identity == second.identity

    def test_the_identity_carries_the_full_width_marker(self, dataset: Any, tmp_path: Path) -> None:
        """A full-width store's identity is not a k-index's identity: the marker is part of it."""
        from rcp_ndcg.retrieval import index

        record = build_store(dataset, _dense(mrl_dim=2), out=tmp_path / "store")
        built = index(dataset, _dense(mrl_dim=2), out=tmp_path / "index")
        assert record.identity != built.identity

    def test_a_late_interaction_store_round_trips_ragged_offsets(self, dataset: Any, tmp_path: Path) -> None:
        record = build_store(dataset, _late(mrl_dim=2), out=tmp_path / "store")
        assert record.layout == "ragged" and record.full_width == 4
        loaded, corpus, queries = load_store(tmp_path / "store")
        assert loaded == record
        assert corpus.offsets is not None and queries.offsets is not None
        assert int(corpus.offsets[0]) == 0
        assert int(corpus.offsets[-1]) == len(corpus.vectors)
        assert np.all(np.diff(corpus.offsets) >= 0)

    def test_a_bm25_retriever_has_no_vectors_to_store(self, dataset: Any, tmp_path: Path) -> None:
        from rcp_ndcg.retrieval import BM25Config

        with pytest.raises(ConfigError, match="BM25") as caught:
            build_store(dataset, BM25Config(), out=tmp_path / "store")
        assert "vectors" in (caught.value.hint or "")

    def test_a_store_rebuilt_in_place_across_layouts_still_reads(self, dataset: Any, tmp_path: Path) -> None:
        """A late-interaction store then a dense store in one directory leaves the ragged offsets behind;
        the record's layout is what the reader follows, so the stale files are never mistaken for data."""
        root = tmp_path / "store"
        build_store(dataset, _late(mrl_dim=2), out=root)
        record = build_store(dataset, _dense(mrl_dim=2), out=root)

        assert record.layout == "single"
        loaded, corpus, queries = load_store(root)
        assert loaded.layout == "single"
        assert corpus.offsets is None and queries.offsets is None


class TestCutStoreRefusal:
    def test_the_saver_refuses_a_cut_store_as_full_width(self, tmp_path: Path) -> None:
        record = EmbeddingStore(
            path=str(tmp_path / "store"),
            dataset="d",
            identity="a" * 64,
            layout="single",
            model="m",
            prompt_digest="b" * 16,
            full_width=8,
            dtype="float32",
            mrl_kind="truncation",
            mrl_dims=(2,),
            count=1,
            query_count=1,
            doc_ids=["d1"],
            query_ids=["q1"],
        )
        cut = StoredVectors(vectors=np.ones((1, 2), dtype=np.float32))
        with pytest.raises(DataError, match="full_width") as caught:
            save_embedding_store(tmp_path / "store", record, cut, cut)
        assert "full width" in (caught.value.hint or "") or "full_width" in (caught.value.hint or "")

    def test_the_reader_refuses_a_store_whose_vectors_are_narrower_than_its_record(self, tmp_path: Path) -> None:
        root = tmp_path / "store"
        root.mkdir()
        record = EmbeddingStore(
            path=str(root),
            dataset="d",
            identity="a" * 64,
            layout="single",
            model="m",
            prompt_digest="b" * 16,
            full_width=8,
            dtype="float32",
            mrl_kind="truncation",
            mrl_dims=(2,),
            count=1,
            query_count=1,
            doc_ids=["d1"],
            query_ids=["q1"],
        )
        (root / "store.json").write_text(record.model_dump_json(by_alias=True), encoding="utf-8")
        np.save(root / "corpus.npy", np.ones((1, 2), dtype=np.float32))
        np.save(root / "queries.npy", np.ones((1, 2), dtype=np.float32))
        with pytest.raises(DataError, match="full_width"):
            load_embedding_store(root)


class TestSweep:
    def test_the_sweep_equals_per_k_direct_runs(self, dataset: Any, tmp_path: Path) -> None:
        record = build_store(dataset, _dense(mrl_dim=2), out=tmp_path / "store")
        _, corpus, queries = load_store(tmp_path / "store")

        ranked = sweep(record, corpus, queries, dims=(2, 4), depth=3)

        assert [row.systems for row in ranked] == [["stub@2"], ["stub@4"]]
        for k, rows in zip((2, 4), ranked, strict=True):
            direct = retrieve(dataset, _dense(mrl_dim=k), depth=3, out=tmp_path / f"direct-{k}")
            assert rows.queries() == direct.queries(), f"k={k}"

    def test_a_late_interaction_sweep_equals_per_k_direct_runs(self, dataset: Any, tmp_path: Path) -> None:
        record = build_store(dataset, _late(mrl_dim=2), out=tmp_path / "store")
        _, corpus, queries = load_store(tmp_path / "store")

        ranked = sweep(record, corpus, queries, dims=(2,), depth=3)

        direct = retrieve(dataset, _late(mrl_dim=2), depth=3, out=tmp_path / "direct")
        assert ranked[0].queries() == direct.queries()

    def test_the_sweep_identity_k_equals_the_direct_full_width_run(self, dataset: Any, tmp_path: Path) -> None:
        """A declared set that keeps the card's full-width member (8 here): sweeping ``k == full_width``
        uses the stored full-width vectors unchanged (the identity selection applies no head), and a
        direct ``k == full_width`` run agrees."""
        record = build_store(dataset, _dense(mrl_dims=(2, 8), mrl_dim=8), out=tmp_path / "store")
        _, corpus, queries = load_store(tmp_path / "store")

        ranked = sweep(record, corpus, queries, dims=(2, 8), depth=3)

        assert [row.systems for row in ranked] == [["stub@2"], ["stub@8"]]
        for k in (2, 8):
            direct = retrieve(dataset, _dense(mrl_dims=(2, 8), mrl_dim=k), depth=3, out=tmp_path / f"direct-{k}")
            assert ranked[0 if k == 2 else 1].queries() == direct.queries(), f"k={k}"

    def test_the_default_sweep_covers_every_declared_dim(self, dataset: Any, tmp_path: Path) -> None:
        record = build_store(dataset, _dense(mrl_dim=2), out=tmp_path / "store")
        _, corpus, queries = load_store(tmp_path / "store")
        assert [row.systems for row in sweep(record, corpus, queries, depth=2)] == [["stub@2"], ["stub@4"]]

    def test_a_dim_outside_the_declared_set_is_refused(self, dataset: Any, tmp_path: Path) -> None:
        record = build_store(dataset, _dense(mrl_dim=2), out=tmp_path / "store")
        _, corpus, queries = load_store(tmp_path / "store")
        with pytest.raises(ConfigError, match="mrl_dims") as caught:
            sweep(record, corpus, queries, dims=(3,))
        assert "mrl_dims" in (caught.value.hint or "")

    def test_a_store_without_a_declared_set_cannot_sweep(self, dataset: Any, tmp_path: Path) -> None:
        record = build_store(dataset, _dense(mrl_kind="none", mrl_dims=None), out=tmp_path / "store")
        _, corpus, queries = load_store(tmp_path / "store")
        with pytest.raises(ConfigError, match="mrl_dims") as caught:
            sweep(record, corpus, queries)
        assert "mrl_dims" in (caught.value.hint or "")

    def test_a_range_store_sweeps_the_explicit_dims(self, dataset: Any, tmp_path: Path) -> None:
        """A range card (Qwen3-Embedding's prose range) has no enumerable set: the sweep takes the k values
        explicitly, and the cut from the store still equals the direct run."""
        ranged = _dense(mrl_kind="truncation", mrl_dims=None, mrl_range=(2, 4), mrl_dim=3)
        record = build_store(dataset, ranged, out=tmp_path / "store")
        assert record.mrl_dims == () and record.mrl_range == (2, 4)
        _, corpus, queries = load_store(tmp_path / "store")

        ranked = sweep(record, corpus, queries, dims=(2, 4), depth=3)

        assert [row.systems for row in ranked] == [["stub@2"], ["stub@4"]]
        direct = retrieve(
            dataset,
            _dense(mrl_kind="truncation", mrl_dims=None, mrl_range=(2, 4), mrl_dim=2),
            depth=3,
            out=tmp_path / "direct",
        )
        assert ranked[0].queries() == direct.queries()

    def test_a_range_store_refuses_a_default_sweep(self, dataset: Any, tmp_path: Path) -> None:
        ranged = _dense(mrl_kind="truncation", mrl_dims=None, mrl_range=(2, 4), mrl_dim=3)
        record = build_store(dataset, ranged, out=tmp_path / "store")
        _, corpus, queries = load_store(tmp_path / "store")
        with pytest.raises(ConfigError, match="range") as caught:
            sweep(record, corpus, queries)
        assert "dims" in (caught.value.hint or "")

    def test_a_k_outside_the_declared_range_is_refused(self, dataset: Any, tmp_path: Path) -> None:
        ranged = _dense(mrl_kind="truncation", mrl_dims=None, mrl_range=(2, 4), mrl_dim=3)
        record = build_store(dataset, ranged, out=tmp_path / "store")
        _, corpus, queries = load_store(tmp_path / "store")
        with pytest.raises(ConfigError, match="mrl_range") as caught:
            sweep(record, corpus, queries, dims=(1,))
        assert "mrl_range" in (caught.value.hint or "")

    def test_an_explicitly_empty_dims_is_refused(self, dataset: Any, tmp_path: Path) -> None:
        record = build_store(dataset, _dense(mrl_dim=2), out=tmp_path / "store")
        _, corpus, queries = load_store(tmp_path / "store")
        with pytest.raises(ConfigError, match="empty"):
            sweep(record, corpus, queries, dims=())

    def test_a_projection_store_sweeps_its_learned_matrix(self, dataset: Any, tmp_path: Path) -> None:
        """The projection head through the store: the file's tensor names are the widths (one convention),
        the identity hashes, and the sweep equals a direct run of the same head."""
        matrix = np.eye(8, 2, dtype=np.float32)
        source = write_safetensors(tmp_path / "projections.safetensors", {"2": matrix})
        encoder: dict[str, Any] = {
            "base_url": "fake://seed/7?dim=8",
            "model": "stub",
            "mrl_kind": "projection",
            "mrl_dims": (2,),
            "mrl_projection": {"source": str(source)},
            "mrl_dim": 2,
            **_BUDGET,
        }
        record = build_store(dataset, DenseConfig(encoder=ServedEmbedding(**encoder)), out=tmp_path / "store")
        assert record.mrl_kind == "projection" and record.mrl_projection is not None
        _, corpus, queries = load_store(tmp_path / "store")

        ranked = sweep(record, corpus, queries, dims=(2,), depth=3)

        direct = retrieve(dataset, DenseConfig(encoder=ServedEmbedding(**encoder)), depth=3, out=tmp_path / "direct")
        assert ranked[0].queries() == direct.queries()


class TestSweepCommand:
    def test_the_sweep_writes_per_k_rankings_and_a_report(self, dataset: Any, tmp_path: Path) -> None:
        import json

        from click.testing import CliRunner

        from rcp_ndcg.cli.retrieval import retrieval_group

        store = tmp_path / "store"
        build_store(dataset, _dense(mrl_dim=2), out=store)
        out_dir = tmp_path / "out"

        result = CliRunner().invoke(
            retrieval_group,
            [
                "sweep",
                "--store",
                str(store),
                "--out-dir",
                str(out_dir),
                "--dims",
                "2",
                "--dims",
                "4",
                "--dataset",
                dataset.uri,
                "--metrics",
                "qrel_ndcg",
                "--json",
            ],
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.stdout)["data"]
        assert payload["systems"] == ["stub@2", "stub@4"]
        assert payload["rankings"] == [str(out_dir / "stub@2.parquet"), str(out_dir / "stub@4.parquet")]
        assert payload["report"] == str(out_dir / "report.json")
        assert payload["comparison"] == str(out_dir / "comparison.json")
        assert (out_dir / "stub@2.parquet").exists() and (out_dir / "report.json").exists()

    def test_the_sweep_writes_rankings_only_without_a_dataset(self, dataset: Any, tmp_path: Path) -> None:
        import json

        from click.testing import CliRunner

        from rcp_ndcg.cli.retrieval import retrieval_group

        store = tmp_path / "store"
        build_store(dataset, _dense(mrl_dim=2), out=store)
        result = CliRunner().invoke(
            retrieval_group,
            ["sweep", "--store", str(store), "--out-dir", str(tmp_path / "out"), "--json"],
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.stdout)["data"]
        assert payload["systems"] == ["stub@2", "stub@4"]
        assert payload["report"] is None and payload["comparison"] is None

    def test_the_store_command_reports_a_range(self, dataset: Any, tmp_path: Path) -> None:
        """`retrieval store` maps the record's declaration into its result, range included, so a range
        store's refusal of the default sweep is visible from the command output."""
        import json

        import yaml
        from click.testing import CliRunner

        from rcp_ndcg.cli.retrieval import retrieval_group

        retriever = tmp_path / "retriever.yaml"
        retriever.write_text(
            yaml.safe_dump(
                {
                    "kind": "dense",
                    "encoder": {
                        "api": "openai_embeddings",
                        "base_url": "fake://seed/7?dim=8",
                        "model": "stub",
                        "mrl_kind": "truncation",
                        "mrl_range": [2, 4],
                        "mrl_dim": 3,
                        **_BUDGET,
                    },
                }
            ),
            encoding="utf-8",
        )
        result = CliRunner().invoke(
            retrieval_group,
            [
                "store",
                "--dataset",
                dataset.uri,
                "--retriever",
                str(retriever),
                "--out",
                str(tmp_path / "store"),
                "--json",
            ],
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.stdout)["data"]
        assert payload["mrl_dims"] == [] and payload["mrl_range"] == [2, 4]
