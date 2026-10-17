"""Live loads of the canonical repositories at pinned revisions (network-gated).

The offline fixtures in ``tests/data/fixtures`` hold these repositories' cards plus a few sampled rows; here the
real repositories are read at the revisions the fixtures were sampled from, so a moved upstream cannot silently
change what the offline fixtures say the layout is. Run one file at a time, online, with the per-test hang guard
raised (a multilingual corpus is hundreds of megabytes)::

    RCP_NDCG_NETWORK_TESTS=1 RCP_NDCG_TEST_TIMEOUT=600 \
        uv run pytest tests/data/test_io_hub_live.py -q -p no:cacheprovider -o faulthandler_timeout=300
"""

from __future__ import annotations

import os

import pytest

from rcp_ndcg.data import load_dataset
from rcp_ndcg.data.io.hub import hub_subsets
from rcp_ndcg.errors import ConfigError

pytestmark = [
    pytest.mark.network,
    pytest.mark.skipif(os.environ.get("RCP_NDCG_NETWORK_TESTS") != "1", reason="needs the public Hugging Face Hub"),
]

PINS = {
    "mteb/nfcorpus": "52ac3f19d3449632d9f00aab0ad34a110fc03816",
    "mteb/MIRACLRetrieval": "9c09abc13478308c27598f350e31d8f06b9b5481",
    "mteb/AskUbuntuDupQuestions": "74be56d07b2def3cd1275eb27bb55c6f53bbbcbd",
    "mteb/Core17InstructionRetrieval": "67d8733dd0691f08f34ed9dda5941578cba8d91c",
    "mteb/blink-it2i": "f399ecd9713a91c97d7e26ca3917d97b2d98b22a",
    "vidore/vidore_v3_finance_en_mteb_format": "fa78cb14152b3dde8c5defdc4e3ddf50de69dfeb",
    "mteb/arxivqa_test_subsampled_beir": "179762404662be6bf48c30b4fb2b2ef8c6891290",
    "mteb/BRIGHT": "23697a955adedade1098da06fcc393818573005d",
    "fabianschmidt-cohere/rcp-ndcg-nanobeir": "aaab69b45c90ab3e7dc6db55f6cbcaac5aeac866",
}


@pytest.fixture(autouse=True)
def media_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("RCP_NDCG_MEDIA_CACHE", str(tmp_path / "media-cache"))
    return tmp_path / "media-cache"


def test_the_v1_beir_mirror_loads_pinned() -> None:
    dataset = load_dataset(f"hf://mteb/nfcorpus@{PINS['mteb/nfcorpus']}")
    assert dataset.name == "nfcorpus" and dataset.split == "test"
    assert len(dataset.qrels) > 0 and len(dataset.corpus) > 0
    assert dataset.provenance.revision == PINS["mteb/nfcorpus"]


def test_the_multilingual_mirror_loads_one_language_pinned() -> None:
    dataset = load_dataset(f"hf://mteb/MIRACLRetrieval/en@{PINS['mteb/MIRACLRetrieval']}")
    assert dataset.split == "dev"
    assert dataset.qrels and dataset.queries, "the en subset's labels and queries read from the card's configs"


def test_the_v2_reranking_mirror_loads_pools_pinned() -> None:
    dataset = load_dataset(f"hf://mteb/AskUbuntuDupQuestions@{PINS['mteb/AskUbuntuDupQuestions']}")
    assert dataset.candidates is not None
    assert set(dataset.qrels) == set(dataset.candidates)


def test_the_instruction_task_loads_pinned() -> None:
    dataset = load_dataset(f"hf://mteb/Core17InstructionRetrieval@{PINS['mteb/Core17InstructionRetrieval']}")
    assert all(query.instruction for query in dataset.queries.values())


def test_the_any2any_task_loads_media_pinned() -> None:
    dataset = load_dataset(f"hf://mteb/blink-it2i@{PINS['mteb/blink-it2i']}")
    assert next(iter(dataset.queries.values())).as_content.has_media
    assert next(iter(dataset.corpus.values())).as_content.has_media


def test_the_multilingual_image_repository_loads_pinned() -> None:
    pin = PINS["vidore/vidore_v3_finance_en_mteb_format"]
    dataset = load_dataset(f"hf://vidore/vidore_v3_finance_en_mteb_format/english@{pin}")
    assert next(iter(dataset.corpus.values())).as_content.has_media
    assert dataset.qrels


def test_the_rcp_ndcg_layout_loads_gains_pinned() -> None:
    pin = PINS["fabianschmidt-cohere/rcp-ndcg-nanobeir"]
    dataset = load_dataset(f"hf://fabianschmidt-cohere/rcp-ndcg-nanobeir/NanoArguAnaRetrieval@{pin}")
    assert dataset.protocol == "nanobeir" and dataset.gains is not None


def test_a_raw_layout_repository_is_refused_pinned() -> None:
    with pytest.raises(ConfigError, match="name the subset"):
        load_dataset(f"hf://mteb/BRIGHT@{PINS['mteb/BRIGHT']}")


def test_the_subsets_come_from_the_cards_pinned() -> None:
    assert "en" in hub_subsets("mteb/MIRACLRetrieval", PINS["mteb/MIRACLRetrieval"])
    pin = PINS["fabianschmidt-cohere/rcp-ndcg-nanobeir"]
    assert "NanoArguAnaRetrieval" in hub_subsets("fabianschmidt-cohere/rcp-ndcg-nanobeir", pin)


# ---------------------------------------------------------------------------
# Decision 30's measurement: how often the canonical repositories repeat a row
# ---------------------------------------------------------------------------

MEASURED = (
    # (repo, subset, read the corpus too: the multilingual mirror's and ViDoRe's corpora are hundreds of
    # megabytes, so their labels and queries are measured and their corpora are not)
    ("mteb/nfcorpus", "", True),
    ("mteb/AskUbuntuDupQuestions", "", True),
    ("mteb/Core17InstructionRetrieval", "", True),
    ("mteb/blink-it2i", "", True),
    ("vidore/vidore_v3_finance_en_mteb_format", "/english", False),
    ("fabianschmidt-cohere/rcp-ndcg-nanobeir", "/NanoArguAnaRetrieval", True),
)


def _duplicate_ids(rows, key: str) -> int:
    """How many rows of a table repeat an id (the measurement behind decision 30)."""
    ids = [str(row.get(key) or row.get("id") or row.get("_id")) for row in rows]
    return len(ids) - len(set(ids))


def test_the_canonical_repositories_hold_no_duplicate_labels() -> None:
    """Decision 30's measurement, reported in the lane report: how often the canonical repositories repeat a
    row. The labels read with ``duplicates="last"`` (mteb's behaviour), and the counts it took are recorded in
    the provenance; the corpus and query ids are counted from the raw rows (the reader folds them, so the
    folded output cannot show them)."""
    from rcp_ndcg.data.io.hub import HubReader

    measured: list[str] = []
    for repo, subset, corpus_too in MEASURED:
        dataset = load_dataset(f"hf://{repo}{subset}@{PINS[repo]}", duplicates="last")
        counts = dataset.provenance.duplicates
        assert counts is not None
        reader = HubReader(f"{repo}{subset}", revision=PINS[repo])
        qrels = list(reader._rows("qrels"))
        pairs = [(str(row["query-id"]), str(row["corpus-id"])) for row in qrels]
        duplicates = {
            "labels": len(pairs) - len(set(pairs)),
            "queries": _duplicate_ids(reader._rows("queries"), "id"),
            "corpus": _duplicate_ids(reader._rows("corpus"), "id") if corpus_too else -1,
        }
        assert counts.folded + counts.resolved == duplicates["labels"], "the provenance count is the raw count"
        measured.append(
            f"{repo}{subset}: labels={duplicates['labels']} queries={duplicates['queries']} "
            f"corpus={duplicates['corpus'] if corpus_too else 'not measured (too large)'}"
        )
    print("\n".join(measured))  # the numbers the lane report carries
