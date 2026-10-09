"""The MTEB dataset writer (`data/io/mteb.py`): the layout `push_dataset_to_hub` writes, plus the extras mteb
ignores, and the round trip through mteb's own `RetrievalDatasetLoader` (the `[mteb]` extra)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from rcp_ndcg.data.dataset import Dataset
from rcp_ndcg.data.io import available_writers, get_writer
from rcp_ndcg.data.io.mteb import MtebWriter
from rcp_ndcg.errors import ConfigError, DataError

MTEB = pytest.importorskip("mteb")

from datasets import get_dataset_config_names  # noqa: E402 (mteb's dependency)


def a_dataset(**overrides: Any) -> Dataset:
    records: dict[str, Any] = dict(
        name="NanoArguAnaRetrieval",
        queries=[
            {"query_id": "q1", "text": "what is a tortoise"},
            {"query_id": "q2", "text": "how long do they live"},
        ],
        corpus=[
            {"doc_id": "d1", "text": "a tortoise is a reptile"},
            {"doc_id": "d2", "text": "unrelated passage"},
            {"doc_id": "d3", "text": "tortoises can live over a century"},
        ],
        qrels=[
            {"query_id": "q1", "doc_id": "d1", "grade": 2},
            {"query_id": "q1", "doc_id": "d2", "grade": 0},
            {"query_id": "q2", "doc_id": "d3", "grade": 1},
        ],
        candidates={"q1": ["d1", "d2", "d3"], "q2": ["d3"]},
    )
    records.update(overrides)
    return Dataset.from_records(**records)


def a_png() -> bytes:
    """A 2x3 red PNG, small enough to inline in the tests."""
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (2, 3), "red").save(buffer, "PNG")
    return buffer.getvalue()


def a_media_dataset(tmp_path: Path, kind: str, payload: bytes) -> Dataset:
    """One document whose content is its text plus one media part of *kind*."""
    from rcp_ndcg_core.content import Content, ImagePart, TextPart, VideoPart

    from rcp_ndcg.data.media import store_media

    if kind == "image":
        ref = store_media(payload, ".png", root=str(tmp_path / "media"))
        part = ImagePart(ref=ref)
    else:
        ref = store_media(payload, ".mp4", root=str(tmp_path / "media"), mime="video/mp4")
        part = VideoPart(ref=ref)
    content = Content.from_parts([TextPart(text="page one"), part])
    return Dataset.from_records(
        name="media",
        queries=[{"query_id": "q1", "text": "q"}],
        corpus=[{"doc_id": "d1", "text": "page one", "content": content}],
        qrels=[{"query_id": "q1", "doc_id": "d1", "grade": 1, "gain": 0.9}],
        candidates={"q1": ["d1"]},
    )


def write(dataset: Dataset, tmp_path: Path, **kwargs: Any) -> str:
    # `datasets` keys its built-dataset cache for a local directory by the directory's basename (its content is
    # not in the key), so the layout goes under the test's own unique name: two tests' layouts never alias.
    out = str(tmp_path / tmp_path.name)
    written = MtebWriter().write_dataset(dataset, out, **kwargs)
    assert written == len(dataset.corpus)
    return out


def read_parquet(root: str, config: str, split: str = "test") -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    return pq.read_table(f"{root}/{config}/{split}-00000-of-00001.parquet").to_pylist()


# -- the pure data layout (no mteb import) -----------------------------------


def test_the_writer_is_registered() -> None:
    assert "mteb" in available_writers()
    assert get_writer("mteb").name == "mteb"


def test_the_layout_is_what_push_dataset_to_hub_writes(tmp_path: Path) -> None:
    out = write(a_dataset(), tmp_path)
    assert sorted(get_dataset_config_names(out)) == ["corpus", "qrels", "queries", "top_ranked"]
    assert read_parquet(out, "corpus") == [
        {"id": "d1", "title": "", "text": "a tortoise is a reptile"},
        {"id": "d2", "title": "", "text": "unrelated passage"},
        {"id": "d3", "title": "", "text": "tortoises can live over a century"},
    ]
    assert read_parquet(out, "queries") == [
        {"id": "q1", "text": "what is a tortoise"},
        {"id": "q2", "text": "how long do they live"},
    ]
    assert read_parquet(out, "qrels") == [
        {"query-id": "q1", "corpus-id": "d1", "score": 2},
        {"query-id": "q1", "corpus-id": "d2", "score": 0},
        {"query-id": "q2", "corpus-id": "d3", "score": 1},
    ]


def test_a_named_subset_prefixes_every_config(tmp_path: Path) -> None:
    out = write(a_dataset(), tmp_path, subset="NanoArguAnaRetrieval")
    assert sorted(get_dataset_config_names(out)) == [
        "NanoArguAnaRetrieval-corpus",
        "NanoArguAnaRetrieval-qrels",
        "NanoArguAnaRetrieval-queries",
        "NanoArguAnaRetrieval-top_ranked",
    ]
    assert read_parquet(out, "NanoArguAnaRetrieval-corpus")[0]["id"] == "d1"


def test_the_score_column_is_int64_and_the_extras_ride_on_the_qrels(tmp_path: Path) -> None:
    import pyarrow.parquet as pq

    dataset = a_dataset(
        qrels=[
            {"query_id": "q1", "doc_id": "d1", "grade": 2, "gain": 0.9, "theta": 1.2},
            {"query_id": "q1", "doc_id": "d2", "grade": 0},
            {"query_id": "q2", "doc_id": "d3", "grade": 1},
        ]
    )
    out = write(dataset, tmp_path)
    schema = pq.read_schema(f"{out}/qrels/test-00000-of-00001.parquet")
    assert str(schema.field("score").type) == "int64"
    rows = read_parquet(out, "qrels")
    assert rows[0] == {"query-id": "q1", "corpus-id": "d1", "score": 2, "gain": 0.9, "theta": 1.2}
    assert rows[1] == {"query-id": "q1", "corpus-id": "d2", "score": 0, "gain": None, "theta": None}


def test_the_candidates_become_top_ranked(tmp_path: Path) -> None:
    out = write(a_dataset(), tmp_path)
    assert sorted(get_dataset_config_names(out)) == ["corpus", "qrels", "queries", "top_ranked"]
    assert read_parquet(out, "top_ranked") == [
        {"query-id": "q1", "corpus-ids": ["d1", "d2", "d3"]},
        {"query-id": "q2", "corpus-ids": ["d3"]},
    ]


def test_the_readme_declares_the_configs_so_load_dataset_finds_them(tmp_path: Path) -> None:
    out = write(a_dataset(), tmp_path)
    front = (Path(out) / "README.md").read_text().split("---")[1]
    assert "config_name: corpus" in front and "config_name: qrels" in front
    assert "path: corpus/test-*" in front


def test_a_card_from_mteb_s_template(tmp_path: Path) -> None:
    out = write(
        a_dataset(),
        tmp_path,
        card={
            "name": "NanoArguAnaRetrieval",
            "description": "NanoArguAna, a smaller subset of ArguAna.",
            "type": "Retrieval",
            "dataset": {"path": "org/rcp-ndcg-nanobeir", "revision": "abc123"},
            "eval_langs": ["eng-Latn"],
            "main_score": "ndcg_float_at_10",
            "license": "cc-by-4.0",
        },
    )
    readme = (Path(out) / "README.md").read_text()
    assert "NanoArguAnaRetrieval" in readme
    assert "config_name: corpus" in readme.split("---")[1]  # the configs stay in the front matter
    assert "- mteb" in readme  # the card body is mteb's template, tags and all


def test_a_card_missing_a_required_task_field_is_refused(tmp_path: Path) -> None:
    fields = {"description": "no name", "type": "Retrieval", "dataset": {"path": "o/r", "revision": "abc"}}
    with pytest.raises(ConfigError, match="name"):
        write(a_dataset(), tmp_path, card=fields)
    with pytest.raises(ConfigError, match="dataset"):
        write(a_dataset(), tmp_path, card={"name": "T", "description": "d", "type": "Retrieval"})


def test_exclusions_travel_in_the_ignored_config_and_leave_the_pool(tmp_path: Path) -> None:
    out = write(a_dataset(excluded={"q1": ["d2"]}), tmp_path)
    assert sorted(get_dataset_config_names(out)) == ["corpus", "excluded", "qrels", "queries", "top_ranked"]
    assert read_parquet(out, "excluded") == [{"query-id": "q1", "excluded-corpus-ids": ["d2"]}]
    assert read_parquet(out, "top_ranked") == [
        {"query-id": "q1", "corpus-ids": ["d1", "d3"]},  # d2 folded out of the pool
        {"query-id": "q2", "corpus-ids": ["d3"]},
    ]


def test_without_candidates_the_pool_is_the_corpus_minus_the_exclusions(tmp_path: Path) -> None:
    """The BRIGHT derivation: the task builds `top_ranked` as the corpus minus its exclusions."""
    out = write(a_dataset(candidates=None, excluded={"q1": ["d2"]}), tmp_path)
    assert read_parquet(out, "top_ranked") == [
        {"query-id": "q1", "corpus-ids": ["d1", "d3"]},
        {"query-id": "q2", "corpus-ids": ["d1", "d2", "d3"]},
    ]


def test_no_pool_and_no_exclusions_write_no_top_ranked(tmp_path: Path) -> None:
    out = write(a_dataset(candidates=None), tmp_path)
    assert sorted(get_dataset_config_names(out)) == ["corpus", "qrels", "queries"]


def test_an_instruction_column_applies_only_when_a_query_carries_one(tmp_path: Path) -> None:
    dataset = a_dataset(
        queries=[
            {"query_id": "q1", "text": "q", "instruction": "find documents that refute"},
            {"query_id": "q2", "text": "q2"},
        ]
    )
    out = write(dataset, tmp_path)
    assert read_parquet(out, "queries") == [
        {"id": "q1", "text": "q", "instruction": "find documents that refute"},
        {"id": "q2", "text": "q2", "instruction": None},
    ]
    assert sorted(get_dataset_config_names(out)) == ["corpus", "qrels", "queries", "top_ranked"]


def test_fractional_grades_are_refused(tmp_path: Path) -> None:
    dataset = Dataset.from_records(
        name="x",
        queries=[{"query_id": "q1", "text": "q"}],
        corpus=[{"doc_id": "d1", "text": "t"}],
        qrels=[{"query_id": "q1", "doc_id": "d1", "grade": 1.5}],
    )
    with pytest.raises(DataError, match="1.5") as caught:
        write(dataset, tmp_path)
    assert caught.value.hint and "integer" in caught.value.hint


def test_qrels_for_an_unknown_query_or_document_are_refused(tmp_path: Path) -> None:
    dataset = a_dataset()
    broken = dict(dataset.qrels)
    broken["q9"] = {"d1": 1}
    with pytest.raises(DataError, match="q9"):
        MtebWriter().write_corpus(dataset.corpus.values(), dataset.queries.values(), broken, str(tmp_path / "a"))


def test_a_pool_naming_an_unknown_document_is_refused(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="d9"):
        write(a_dataset(candidates={"q1": ["d1", "d9"], "q2": ["d3"]}), tmp_path)


def test_an_image_corpus_writes_mteb_s_image_column(tmp_path: Path) -> None:
    """The vidore v3 shape: `image` as a `datasets.Image` struct, decoded by mteb's own dataloader."""
    import pyarrow.parquet as pq
    from datasets import Image, load_dataset
    from mteb._create_dataloaders import create_dataloader
    from mteb.abstasks.task_metadata import TaskMetadata

    payload = a_png()
    dataset = a_media_dataset(tmp_path, "image", payload)
    out = write(dataset, tmp_path)

    schema = pq.read_schema(f"{out}/corpus/test-00000-of-00001.parquet")
    assert str(schema.field("image").type) == "struct<bytes: binary, path: string>"
    features = json.loads(schema.metadata[b"huggingface"])["info"]["features"]
    assert features["image"] == {"_type": "Image"}

    loaded = load_dataset(out, "corpus", split="test")
    assert isinstance(loaded.features["image"], Image)
    assert loaded[0]["image"].size == (2, 3)  # the page image mteb hands a model
    meta = TaskMetadata(
        name="Probe",
        description="probe",
        type="DocumentUnderstanding",
        category="t2it",
        modalities=["text", "image"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_float_at_10",
        dataset={"path": "probe", "revision": "main"},
    )
    batch = next(iter(create_dataloader(loaded, task_metadata=meta)))
    assert batch["image"][0].size == (2, 3)
    assert batch["text"] == ["page one"]


def test_a_video_corpus_writes_mteb_s_video_column(tmp_path: Path) -> None:
    """`video` as a `datasets.Video` struct; decoding it needs torchcodec (mteb's own requirement), so the
    round trip checked here is the feature and the bytes mteb's `RetrievalDatasetLoader` reads."""
    import pyarrow.parquet as pq
    from datasets import Video, load_dataset

    payload = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom" + b"\x00" * 64
    dataset = a_media_dataset(tmp_path, "video", payload)
    out = write(dataset, tmp_path)

    schema = pq.read_schema(f"{out}/corpus/test-00000-of-00001.parquet")
    assert str(schema.field("video").type) == "struct<bytes: binary, path: string>"
    features = json.loads(schema.metadata[b"huggingface"])["info"]["features"]
    assert features["video"] == {"_type": "Video"}

    loaded = load_dataset(out, "corpus", split="test")
    assert isinstance(loaded.features["video"], Video)
    raw = loaded.cast_column("video", Video(decode=False))[0]["video"]
    assert raw == {"bytes": payload, "path": None}


def test_two_images_in_one_document_are_refused_by_name(tmp_path: Path) -> None:
    from rcp_ndcg_core.content import Content, ImagePart, TextPart

    from rcp_ndcg.data.media import store_media

    ref = store_media(a_png(), ".png", root=str(tmp_path / "media"))
    content = Content.from_parts([TextPart(text="p"), ImagePart(ref=ref), ImagePart(ref=ref)])
    dataset = Dataset.from_records(
        name="media",
        queries=[{"query_id": "q1", "text": "q"}],
        corpus=[{"doc_id": "d1", "text": "p", "content": content}],
        qrels=[{"query_id": "q1", "doc_id": "d1", "grade": 1}],
    )
    with pytest.raises(ConfigError, match="d1.*2 image|2 image.*d1") as caught:
        write(dataset, tmp_path)
    assert "jsonl" in (caught.value.hint or "")


def test_a_video_of_frames_is_refused_with_or_without_a_container(tmp_path: Path) -> None:
    """mteb's Video column holds a container; a part whose frames are extracted (with a container too -- the
    model allows both) is refused by name rather than silently dropping the frames."""
    from rcp_ndcg_core.content import Content, TextPart, VideoPart

    from rcp_ndcg.data.media import store_media

    frame = store_media(a_png(), ".png", root=str(tmp_path / "media"))
    container = store_media(
        b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom",
        ".mp4",
        root=str(tmp_path / "media"),
        mime="video/mp4",
    )
    for part in (VideoPart(frames=[frame]), VideoPart(ref=container, frames=[frame])):
        content = Content.from_parts([TextPart(text="p"), part])
        dataset = Dataset.from_records(
            name="media",
            queries=[{"query_id": "q1", "text": "q"}],
            corpus=[{"doc_id": "d1", "text": "p", "content": content}],
            qrels=[{"query_id": "q1", "doc_id": "d1", "grade": 1}],
        )
        with pytest.raises(ConfigError, match="container") as caught:
            write(dataset, tmp_path)
        assert "frames" in (caught.value.hint or "")


def test_the_written_qrels_load_with_the_pr_s_load_float_gains(tmp_path: Path) -> None:
    """mteb PR #5516's own `load_float_gains` reads the written `gain` column (the round trip the brief asks
    for): the gains our `Dataset` holds are exactly what the PR's reader returns."""
    from mteb.abstasks.task_metadata import TaskMetadata

    from ._pr5516 import load_float_gains

    dataset = a_dataset(
        qrels=[
            {"query_id": "q1", "doc_id": "d1", "grade": 2, "gain": 0.9, "theta": 1.2},
            {"query_id": "q1", "doc_id": "d2", "grade": 0, "gain": 0.1, "theta": -0.4},
            {"query_id": "q2", "doc_id": "d3", "grade": 1},
        ]
    )
    out = write(dataset, tmp_path, subset="NanoArguAnaRetrieval")
    meta = TaskMetadata(
        name="NanoArguAnaRCPReranking",
        description="probe",
        type="Reranking",
        dataset={"path": out, "revision": "main"},
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_float_at_10",
    )

    assert load_float_gains(meta, "NanoArguAnaRetrieval", "test") == dataset.gains


def test_empty_qrels_or_queries_are_refused(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="qrels"):
        write(a_dataset(qrels=[]), tmp_path)
    with pytest.raises(DataError, match="quer"):
        write(a_dataset(queries=[]), tmp_path)


def test_a_non_finite_grade_is_refused(tmp_path: Path) -> None:
    dataset = Dataset.from_records(
        name="x",
        queries=[{"query_id": "q1", "text": "q"}],
        corpus=[{"doc_id": "d1", "text": "t"}],
        qrels=[{"query_id": "q1", "doc_id": "d1", "grade": 1}],
    )
    qrels = {"q1": {"d1": float("nan")}}
    with pytest.raises(DataError, match="finite"):
        MtebWriter().write_corpus(dataset.corpus.values(), dataset.queries.values(), qrels, str(tmp_path / "a"))


def test_an_explicit_subset_on_a_suite_is_refused(tmp_path: Path) -> None:
    suite = Dataset(name="nanobeir", subsets=(a_dataset(),))
    with pytest.raises(ConfigError, match="suite"):
        MtebWriter().write_dataset(suite, str(tmp_path / "out"), subset="one")


# -- the round trip through mteb's own loader --------------------------------


def test_the_writer_output_loaded_by_mteb_equals_our_dataset(tmp_path: Path) -> None:
    from mteb.abstasks.retrieval_dataset_loaders import RetrievalDatasetLoader

    dataset = a_dataset()
    out = write(dataset, tmp_path)

    loaded = RetrievalDatasetLoader(hf_repo=out, revision="main", split="test", config=None).load()

    assert loaded["relevant_docs"] == {q: {d: int(g) for d, g in docs.items()} for q, docs in dataset.qrels.items()}
    assert loaded["queries"]["id"] == ["q1", "q2"]
    assert loaded["queries"]["text"] == ["what is a tortoise", "how long do they live"]
    assert loaded["corpus"]["id"] == ["d1", "d2", "d3"]
    assert loaded["corpus"]["text"] == [
        "a tortoise is a reptile",
        "unrelated passage",
        "tortoises can live over a century",
    ]
    assert loaded["top_ranked"] == {"q1": ["d1", "d2", "d3"], "q2": ["d3"]}


def test_the_named_subset_round_trips(tmp_path: Path) -> None:
    from mteb.abstasks.retrieval_dataset_loaders import RetrievalDatasetLoader

    out = write(a_dataset(), tmp_path, subset="NanoArguAnaRetrieval")
    loaded = RetrievalDatasetLoader(hf_repo=out, revision="main", split="test", config="NanoArguAnaRetrieval").load()
    assert loaded["queries"]["id"] == ["q1", "q2"]
    assert loaded["top_ranked"] == {"q1": ["d1", "d2", "d3"], "q2": ["d3"]}


def test_data_convert_writes_the_layout(tmp_path: Path) -> None:
    """The advertised command: `data convert --to mteb` writes a layout load_dataset reads back."""
    from click.testing import CliRunner

    from rcp_ndcg.cli.data import data_group

    source = tmp_path / "rows"
    source.mkdir()
    (source / "corpus.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in [{"doc_id": "d1", "text": "a tortoise is a reptile"}])
    )
    (source / "queries.jsonl").write_text(json.dumps({"query_id": "q1", "text": "what is a tortoise"}) + "\n")
    (source / "qrels.jsonl").write_text(json.dumps({"query_id": "q1", "qrels": {"d1": 2}}) + "\n")
    done = CliRunner().invoke(
        data_group,
        ["convert", "--format", "jsonl", "--to", "mteb", "--source", str(source), "--out", str(tmp_path / "out")],
    )
    assert done.exit_code == 0, done.output
    assert sorted(get_dataset_config_names(str(tmp_path / "out"))) == ["corpus", "qrels", "queries"]
