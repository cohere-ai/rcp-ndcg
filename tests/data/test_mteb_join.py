"""The join gate: our document and query text against mteb's own dataloader, byte for byte.

Decision 27: a model reads ``(title + " " + body).strip()`` (the body alone without a title),
byte-identical to mteb's ``_create_dataloaders._corpus_to_dict``, and decision 33: the per-query
instruction is appended as mteb appends it (``query + " " + instruction``). This file compares our
formatting with mteb's *own functions* over a sample of documents and queries -- never a copied rule --
in the ``[mteb]`` test environment (offline: mteb and ``datasets`` are imported, no Hub is read).
"""

from __future__ import annotations

import inspect

import pytest
from rcp_ndcg_core.records import Document, Query

from rcp_ndcg.data import Dataset

mteb = pytest.importorskip("mteb")
dataloaders = pytest.importorskip("mteb._create_dataloaders")

#: (title, body) samples: a plain pair, a title-only document, a body-only document, blank and whitespace
#: titles (mteb's ``len(row["title"]) > 0`` rule), interior whitespace (only the ends are stripped), unicode.
DOCUMENTS: list[tuple[str, str]] = [
    ("Tortoises", "a tortoise is a reptile"),
    ("", "a body without a title"),
    ("A title only", ""),
    ("   ", "a blank-looking title"),
    ("  T  ", "  body  "),
    ("", "  body  "),
    ("Über die Kräfte", "  die Kräfte wirken  "),
    ("tab\tsep", "line\nbreak"),
]

#: (query, instruction) samples: no instruction, an instruction, an empty instruction (mteb appends a
#: space), whitespace, unicode.
QUERIES: list[tuple[str, str | None]] = [
    ("find docs", None),
    ("find docs", "Given a claim, find documents that refute the claim"),
    ("find docs", ""),
    ("  find docs  ", "  about turtles  "),
    ("", "an instruction with no query"),
    ("find docs", "Über Kräfte"),
]


def mteb_query_text(text: str, instruction: str | None) -> str:
    """mteb's own query formatting for one row: the row function of mteb 2.10.x, the ``Dataset`` function
    of 2.21.x (the shape changed, the rule did not: ``text + " " + instruction`` when the instruction is
    not ``None``).
    """
    combine = dataloaders._combine_queries_with_instruction_text
    if "row" in inspect.signature(combine).parameters:  # mteb 2.10.x: one row
        return combine({"text": text, "instruction": instruction})["text"]
    datasets = pytest.importorskip("datasets")  # mteb 2.21.x: the whole queries dataset
    frame = datasets.Dataset.from_dict({"text": [text], "instruction": [instruction]})
    return str(combine(frame)["text"][0])


@pytest.mark.parametrize(("title", "body"), DOCUMENTS)
def test_a_document_reads_byte_identically_to_mtebs_dataloader(title: str, body: str) -> None:
    ours = Document(doc_id="d", title=title or None, text=body).model_content().text
    theirs = dataloaders._corpus_to_dict({"id": "d", "title": title, "text": body})["text"]
    assert ours == theirs


def test_a_none_title_is_no_title_where_mteb_would_raise() -> None:
    """Our readers normalise a blank or NaN title to ``None``; mteb's loader raises on a ``None`` title.

    The one divergence, and it is the safe direction: a document our reader could not give a title reads
    as its stripped body, exactly as mteb reads an empty-string title (which is what its loaders store).
    """
    assert Document(doc_id="d", title=None, text="  body  ").model_content().text == "body"
    with pytest.raises(TypeError):
        dataloaders._corpus_to_dict({"id": "d", "title": None, "text": "body"})


@pytest.mark.parametrize(("query", "instruction"), QUERIES)
def test_a_query_reads_byte_identically_to_mtebs_dataloader(query: str, instruction: str | None) -> None:
    ours = Query(query_id="q", query=query, instruction=instruction).format_query()
    assert ours == mteb_query_text(query, instruction)


def test_a_fixture_corpus_row_reads_like_mteb(tmp_path) -> None:
    """A real corpus row (the nfcorpus fixture's first jsonl line), title and body through both rules."""
    import json

    fixture = __import__("pathlib").Path(__file__).resolve().parent / "fixtures" / "mteb-nfcorpus" / "corpus.jsonl"
    row = json.loads(fixture.read_text(encoding="utf-8").splitlines()[0])
    document = Document(doc_id=str(row["_id"]), title=row["title"], text=row["text"])
    assert (
        document.model_content().text
        == dataloaders._corpus_to_dict({"id": row["_id"], "title": row["title"], "text": row["text"]})["text"]
    )


def test_the_dataset_row_formatting_matches_the_core_record() -> None:
    """The data layer's row is the same rule (one home): a ``Dataset``'s document and query format like the
    core records they materialise.
    """
    dataset = Dataset.from_records(
        name="sample",
        corpus=[{"doc_id": "d", "title": "Tortoises", "text": "a tortoise is a reptile"}],
        queries=[{"query_id": "q", "text": "find docs", "instruction": "about turtles"}],
        qrels=[{"query_id": "q", "doc_id": "d", "grade": 1.0}],
        task_instruction="Given a claim, find documents that refute the claim",
    )
    assert (
        dataset.corpus["d"].model_content().text
        == dataloaders._corpus_to_dict({"id": "d", "title": "Tortoises", "text": "a tortoise is a reptile"})["text"]
    )
    assert (
        dataset.queries["q"].format_query(task_instruction=dataset.task_instruction)
        == "Task: Given a claim, find documents that refute the claim\nQuery: find docs about turtles"
    )
