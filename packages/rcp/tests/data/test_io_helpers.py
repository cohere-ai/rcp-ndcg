"""The reader helpers two formats share: one home each, and the rule that home states.

``rcp.data.io``'s readers carried a copy of three helpers per format: the merge of a table's duplicate counts
into a reader's provenance, the ``text`` column rule, and the media-column loop. A copy that has diverged is
a correctness bug waiting for one of the two readers to be fixed first, and ``AGENTS.md`` treats a second
implementation of one concept as a merge blocker.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from rcp.data import load_dataset, media
from rcp.data.io import base, beir, get_reader, hub, mteb_task
from rcp.data.io.mteb_task import MtebTaskReader
from rcp.data.media import media_parts
from rcp.errors import DataError
from tests._helpers.hub_stubs import SHA, hub_fixture


@pytest.fixture(autouse=True)
def media_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Media the helper under test writes goes to a content-addressed cache under tmp_path."""
    cache = tmp_path / "media-cache"
    monkeypatch.setenv("RCP_NDCG_MEDIA_CACHE", str(cache))
    return cache


def _png() -> bytes:
    """A real PNG payload, 8x6, so the dimensions the helper reads from the bytes are checkable."""
    image = pytest.importorskip("PIL.Image")
    buffer = io.BytesIO()
    image.new("RGB", (8, 6), (10, 20, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


def _beir_with_a_folded_label(tmp_path: Path) -> str:
    """A BEIR directory whose one label is read twice with the same grade (an exact duplicate)."""
    root = tmp_path / "beir"
    (root / "qrels").mkdir(parents=True)
    (root / "corpus.jsonl").write_text(json.dumps({"_id": "d1", "text": "a body"}) + "\n")
    (root / "queries.jsonl").write_text(json.dumps({"_id": "q1", "text": "a query"}) + "\n")
    (root / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td1\t1\nq1\td1\t1\n")
    return str(root)


# ---------------------------------------------------------------------------
# A table's duplicate counts: the merge into the provenance and the log line
# ---------------------------------------------------------------------------


def test_the_duplicates_totals_have_one_home() -> None:
    """Recording one table's duplicate counts is one function (``base.record_duplicate_counts``): the BEIR and
    the Hub reader each carried a copy, identical apart from the locator they logged."""
    assert "record_duplicate_counts" in base.__all__
    assert not hasattr(beir.BeirReader, "_note")
    assert not hasattr(hub.HubReader, "_note_row_duplicates")


def test_a_table_read_twice_is_counted_once(tmp_path: Path) -> None:
    """The one home keeps the first-pass rule both copies had: a second read of one table is logged, never
    added to the provenance's counts twice."""
    reader = get_reader("beir", uri=_beir_with_a_folded_label(tmp_path))

    assert reader.qrels() == {"q1": {"d1": 1.0}}
    assert reader.qrels() == {"q1": {"d1": 1.0}}, "a re-read yields the same labels"

    counts = reader.provenance.duplicates
    assert counts is not None and counts.folded == 1, "the folded label is counted once, however often it is read"


# ---------------------------------------------------------------------------
# A row's text column: a string, anything else refused
# ---------------------------------------------------------------------------


def test_one_text_rule_for_every_reader() -> None:
    """A row's ``text`` cell is read one way, in one place. The mteb reader's copy named the row alone (no
    source) and cited an internal tracking id in its hint."""
    assert "text_of" in base.__all__
    assert not hasattr(hub, "_text_of")
    assert not hasattr(mteb_task, "_text_of")

    with pytest.raises(DataError) as caught:
        MtebTaskReader("StubTask")._document({"id": "d1", "text": [{"role": "user", "content": "hi"}]})

    assert "mteb:StubTask" in str(caught.value), "the refusal names the source it read"
    assert "the corpus row's 'text'" in str(caught.value)
    assert "workstream" not in str(caught.value.hint), "no internal tracking id in a user-facing hint"


def test_the_hub_reader_refuses_the_same_cell_with_the_same_message(tmp_path: Path) -> None:
    """The one rule reaches the Hub reader too: the same shape of message (source, row, column), with the
    remedy that reader's own source offers."""
    with hub_fixture("mteb-nfcorpus", tmp_path) as root:
        conversation = json.dumps({"_id": "MED-10", "title": "", "text": [{"role": "user"}]}) + "\n"
        (root / "corpus.jsonl").write_text(conversation)
        with pytest.raises(DataError) as caught:
            _ = load_dataset(f"hf://mteb/nfcorpus@{SHA}").corpus

    assert "hf://mteb/nfcorpus/default" in str(caught.value)
    assert "the corpus row's 'text'" in str(caught.value)


# ---------------------------------------------------------------------------
# A row's media columns: encoded, dimensioned, stored once
# ---------------------------------------------------------------------------


def test_the_media_columns_have_one_home() -> None:
    """The media-column loop (encode the cell, read an image's dimensions from its bytes, store it once) is
    one function in ``rcp.data.media``: the Hub reader and the mteb reader each carried a copy of it."""
    assert "encode_media" in media.__all__ and "media_parts" in media.__all__
    assert not hasattr(hub, "_encode_media")
    assert not hasattr(hub.HubReader, "_persist")
    assert not hasattr(mteb_task.MtebTaskReader, "_media_of")


def test_the_one_media_helper_serves_both_readers(tmp_path: Path) -> None:
    """``root`` is the only thing the two callers differ on -- the Hub reader writes media where it is told,
    the mteb reader into the cache -- and the cell encodes the same way either way."""
    cell = {"bytes": _png(), "path": "page.png"}

    (part,) = media_parts({"image": cell}, root=str(tmp_path / "pages"))
    assert (part.ref.width, part.ref.height) == (8, 6), "an image's dimensions come from its own bytes"
    assert Path(part.ref.uri).is_relative_to(tmp_path / "pages")

    document = MtebTaskReader("StubTask")._document({"id": "d1", "image": cell})
    assert document.as_content.media[0].sha256 == part.ref.sha256, "one encoding, one hash, both readers"
    assert document.as_content.media[0].mime == "image/png"
