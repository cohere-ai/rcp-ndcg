"""The Hub reader (``hf://``) against the canonical repositories' real cards.

The fixtures under ``tests/data/fixtures/`` hold each canonical repository's dataset card plus a few rows
sampled from the repositories' own files (media cells synthesized: the layout is what is under test, not the
pixels); the loader serves them offline by faking the two boundaries the reader talks to (see
``hub_stubs``). The network-gated counterparts in ``test_io_hub_live.py`` load the real repositories at pinned
revisions.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

from rcp_ndcg.data import load_dataset
from rcp_ndcg.data.io.hub import HubReader, hub_subsets
from rcp_ndcg.data.revisions import resolve_revision
from rcp_ndcg.errors import ConfigError, DataError, RcpNdcgWarning
from tests.data.hub_stubs import SHA, hub_fixture

PIL = pytest.importorskip("PIL.Image")

NANOBEIR = "hf://fabianschmidt-cohere/rcp-ndcg-nanobeir/NanoArguAnaRetrieval"


@pytest.fixture(autouse=True)
def media_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Media written by the hub tests goes to a content-addressed cache under tmp_path."""
    cache = tmp_path / "media-cache"
    monkeypatch.setenv("RCP_NDCG_MEDIA_CACHE", str(cache))
    return cache


@pytest.fixture(autouse=True)
def clean_revision_cache():
    resolve_revision.cache_clear()
    yield
    resolve_revision.cache_clear()


@pytest.fixture
def hub(tmp_path: Path):
    """The hub fixture factory, bound to this test's tmp_path: the fixtures are copied before a test that
    rewrites a table (a duplicates case, an instruction the reader refuses) runs, so nothing is written into
    the checkout."""

    def bind(name: str):
        return hub_fixture(name, tmp_path)

    return bind


# ---------------------------------------------------------------------------
# The layout rules, one canonical repository per rule
# ---------------------------------------------------------------------------


def test_the_v1_beir_layout_reads_jsonl_at_the_root(hub) -> None:
    """``mteb/nfcorpus``: the qrels config is named ``default`` (with three splits), the corpus and the
    queries are root jsonl files at their own splits."""
    with hub("mteb-nfcorpus"):
        dataset = load_dataset(f"hf://mteb/nfcorpus@{SHA}")

        assert dataset.name == "nfcorpus", "a repository with one subset loads it, named after the repository"
        assert (dataset.subset, dataset.split) == ("default", "test")
        assert dataset.qrels == {"PLAIN-2": {"MED-2427": 2.0, "MED-10": 2.0}}, "jsonl scores read as floats"
        assert dataset.provenance.revision == SHA
        assert set(dataset.queries) == {"PLAIN-2"}, "the queries are cut to those with qrels"
        assert dataset.queries["PLAIN-2"].text == "Do Cholesterol Statin Drugs Cause Breast Cancer?"
        documents = dataset.corpus
        assert set(documents) == {"MED-10", "MED-14"}
        assert documents["MED-10"].title.startswith("Statin Use and Breast Cancer")
        assert documents["MED-10"].text.startswith("Recent studies"), "the body is the text; the title stays a field"


def test_the_prefixed_subset_layout_reads_one_language(hub) -> None:
    """``mteb/MIRACLRetrieval``: 54 ``{lang}-{corpus,qrels,queries}`` configs at split ``dev``."""
    with hub("mteb-miraclretrieval"):
        dataset = load_dataset(f"hf://mteb/MIRACLRetrieval/en@{SHA}")

        assert (dataset.name, dataset.split) == ("en", "dev")
        assert dataset.qrels == {"0": {"462221#4": 1.0, "29810#23": 1.0}}
        assert set(dataset.queries) == {"0"}
        assert dataset.queries["0"].text == "Is Creole a pidgin of French?"
        documents = dataset.corpus
        assert documents["53417369#5"].title == "Alma Thorpe"
        assert dataset.provenance.source_uri == "hf://mteb/MIRACLRetrieval/en"


def test_a_multilingual_repository_without_a_subset_names_them(hub) -> None:
    with hub("mteb-miraclretrieval"):
        with pytest.raises(ConfigError, match="name the subset") as caught:
            load_dataset(f"hf://mteb/MIRACLRetrieval@{SHA}")
        assert "ar" in str(caught.value.hint) and "en" in str(caught.value.hint)


def test_a_v2_reranking_repository_reads_top_ranked_as_pools(hub) -> None:
    """``mteb/AskUbuntuDupQuestions``: v2 reranking is retrieval plus ``top_ranked``; the qrels config is
    named ``default`` with its files under ``data/``."""
    with hub("mteb-askubutudupquestions"):
        dataset = load_dataset(f"hf://mteb/AskUbuntuDupQuestions@{SHA}")

        assert dataset.candidates is not None
        pool = dataset.candidates["test_query0"]
        assert pool[0] == "apositive_test_query0_00000"
        assert dataset.qrels["test_query0"]["apositive_test_query0_00000"] == 1.0


def test_the_instruction_config_wins_over_the_column(hub) -> None:
    """``mteb/Core17InstructionRetrieval``: an instruction config and an instruction column (plus an ignored
    ``qrel_diff`` config); the config wins over the column, as in mteb."""
    with hub("mteb-core17instructionretrieval") as root:
        # The two sources carry the same instruction in the real repository; make the column differ, so the
        # precedence is observable.
        queries = root / "queries/test-00000-of-00001.parquet"
        frame = pd.read_parquet(queries)
        frame["instruction"] = "the column's own instruction"
        frame.to_parquet(queries)

        dataset = load_dataset(f"hf://mteb/Core17InstructionRetrieval@{SHA}")

        assert set(dataset.queries) == {"677-changed"}, "the queries are cut to those with qrels"
        instruction = dataset.queries["677-changed"].instruction
        assert instruction != "the column's own instruction", "the config is the only source when it exists"


def test_the_instruction_config_instructs_every_query(hub) -> None:
    """As in mteb: a repository with an instruction config instructs every query."""
    with hub("mteb-core17instructionretrieval") as root:
        instructions = root / "instruction/test-00000-of-00001.parquet"
        frame = pd.read_parquet(instructions)
        frame[frame["query-id"] != frame["query-id"].iloc[0]].to_parquet(instructions)

        dataset = load_dataset(f"hf://mteb/Core17InstructionRetrieval@{SHA}")
        with pytest.raises(DataError, match="no row in the instruction config"):
            _ = dataset.queries


def test_the_qrel_diff_config_is_ignored(hub) -> None:
    """mteb's loader ignores every config it does not name (``qrel_diff`` is the FollowIR case)."""
    with hub("mteb-core17instructionretrieval"):
        dataset = load_dataset(f"hf://mteb/Core17InstructionRetrieval@{SHA}")
        assert set(dataset.qrels) == {"677-changed"}, "only the qrels config's rows are labels"


def test_the_query_config_always_wins_and_media_become_parts(hub) -> None:
    """``mteb/blink-it2i``: Any2Any with a ``query`` config (image+text queries), a ``qrels`` config, and an
    image corpus -- the ``query`` override ignores the subset, exactly as mteb's loader does."""
    with hub("mteb-blink-it2i"):
        dataset = load_dataset(f"hf://mteb/blink-it2i@{SHA}")

        (query,) = dataset.queries.values()
        assert query.text == "Which image shares the same style as the reference image?"
        assert query.as_content.has_media
        assert query.as_content.media[0].sha256, "the query config's image column is a part"
        (document,) = dataset.corpus.values()
        assert document.text == "", "an image corpus row has no text"
        assert document.as_content.has_media


def test_a_multilingual_image_repository_reads_its_subsets(hub) -> None:
    """``vidore/vidore_v3_finance_en_mteb_format``: prefixed subsets, an image corpus, a ``language`` column
    the reader ignores."""
    with hub("vidore-vidore-v3-finance-en"):
        dataset = load_dataset(f"hf://vidore/vidore_v3_finance_en_mteb_format/english@{SHA}")

        assert dataset.name == "english"
        (document,) = dataset.corpus.values()
        assert document.doc_id == "corpus-test-0" and document.as_content.has_media
        assert document.text.startswith("JPMorganChase"), "the corpus's OCR text column is the body"
        assert dataset.qrels == {"query-test-0": {"corpus-test-10": 2.0, "corpus-test-346": 1.0}}


def test_a_raw_layout_repository_is_refused_with_the_mteb_hint(hub) -> None:
    """``mteb/BRIGHT`` is a raw layout whose task code builds ``top_ranked`` itself: the card reader finds no
    corpus config, and the failure says which reader serves it."""
    with hub("mteb-bright"):
        with pytest.raises(ConfigError, match="name the subset") as caught:
            load_dataset(f"hf://mteb/BRIGHT@{SHA}")
        assert "mteb:<Task>" in str(caught.value.hint)


def test_the_rcp_ndcg_layout_reads_gains_thetas_and_exclusions_from_the_card(hub) -> None:
    """rcp-ndcg's own layout is simply this reader: the qrels ``gain``/``theta`` columns and the
    ``-excluded`` config are read where present, and everything else is mteb's layout."""
    with hub("rcp-ndcg-nanobeir"):
        dataset = load_dataset(f"{NANOBEIR}@{SHA}")

        assert dataset.name == "NanoArguAnaRetrieval"
        assert dataset.gains is not None
        assert all(0 <= gain <= 1 for docs in dataset.gains.values() for gain in docs.values())
        assert dataset.thetas is not None
        assert dataset.excluded, "the -excluded config is read"
        assert dataset.candidates is not None


def test_hub_subsets_come_from_the_config_prefixes(hub) -> None:
    with hub("mteb-miraclretrieval"):
        subsets = hub_subsets("mteb/MIRACLRetrieval", SHA)
        assert "ar" in subsets and "en" in subsets, "the 54 prefixed configs name 18 language subsets"
    with hub("rcp-ndcg-nanobeir"):
        assert "NanoArguAnaRetrieval" in hub_subsets("fabianschmidt-cohere/rcp-ndcg-nanobeir", SHA)
        assert "provenance" not in hub_subsets("fabianschmidt-cohere/rcp-ndcg-nanobeir", SHA)


def test_a_layout_the_loader_cannot_read_stays_a_data_error(hub) -> None:
    """``mteb/arxivqa_test_subsampled_beir``: ids in ``corpus-id``/``query-id`` columns (its task code remaps
    them); the card reader, like mteb's own loader, reads ``_id``/``id`` -- and refuses what it cannot read."""
    with hub("mteb-arxivqa-test-subsampled-beir"):
        dataset = load_dataset(f"hf://mteb/arxivqa_test_subsampled_beir@{SHA}")
        with pytest.raises(DataError, match="carries no id"):
            _ = dataset.corpus


# ---------------------------------------------------------------------------
# document_parts, media persistence, the split option
# ---------------------------------------------------------------------------


def test_document_parts_reads_the_ids_the_same_whichever_parts_are_read(hub) -> None:
    """ViDoRe v3 ships OCR text and page images in one row: the parts selector changes what is read, never
    the ids (what makes the text and vision runs comparable at all)."""
    with hub("vidore-vidore-v3-finance-en"):

        def corpus(parts):
            return load_dataset(
                f"hf://vidore/vidore_v3_finance_en_mteb_format/english@{SHA}", document_parts=parts
            ).corpus

        ids = {parts: [doc.doc_id for doc in corpus(parts).values()] for parts in ("auto", "text", "image", "both")}
        assert len(set(map(tuple, ids.values()))) == 1, "the ids are the same whichever parts are read"
        (document,) = corpus("auto").values()
        assert document.as_content.has_media and document.text, "a multi-column row reads both"


def test_document_parts_image_drops_the_ocr_text(hub) -> None:
    """The visual run: what the judge sees is the page, not a transcription (ViDoRe v3 ships both in one
    row; they are different experiments over the same judgements)."""
    with hub("vidore-vidore-v3-finance-en"):
        dataset = load_dataset(f"hf://vidore/vidore_v3_finance_en_mteb_format/english@{SHA}", document_parts="image")
        (document,) = dataset.corpus.values()
        assert document.as_content.has_media and document.text == ""


def test_document_parts_text_drops_the_pages(hub) -> None:
    """The OCR baseline: comparable to the visual run because the ids and the qrels are identical."""
    with hub("vidore-vidore-v3-finance-en"):
        dataset = load_dataset(f"hf://vidore/vidore_v3_finance_en_mteb_format/english@{SHA}", document_parts="text")
        (document,) = dataset.corpus.values()
        assert not document.as_content.has_media and document.text.startswith("JPMorganChase")


def test_document_parts_refuses_a_corpus_with_either_absent(hub) -> None:
    with hub("mteb-nfcorpus"):
        dataset = load_dataset(f"hf://mteb/nfcorpus@{SHA}", document_parts="both")
        with pytest.raises(DataError, match="needs a media column"):
            _ = dataset.corpus
    with hub("mteb-blink-it2i"):
        dataset = load_dataset("hf://mteb/blink-it2i", revision=SHA, document_parts="both")
        with pytest.raises(DataError, match="needs a text column"):
            _ = dataset.corpus


def test_an_unknown_document_parts_selector_is_refused_at_construction() -> None:
    with pytest.raises(ConfigError, match="document_parts must be one of"):
        HubReader("mteb/nfcorpus", document_parts="pixels-please")  # type: ignore[arg-type]


def test_identical_pages_share_one_file(hub) -> None:
    """Content addressing keeps a corpus with repeated pages from writing the same bytes twice."""
    with hub("vidore-vidore-v3-finance-en") as root:
        path = root / "english-corpus/test-00000-of-00001.parquet"
        frame = pd.read_parquet(path)
        pd.concat([frame, frame], ignore_index=True).to_parquet(path)  # the same page twice

        dataset = load_dataset(f"hf://vidore/vidore_v3_finance_en_mteb_format/english@{SHA}")
        refs = [ref for document in dataset.corpus.values() for ref in document.as_content.media]
        assert len({ref.uri for ref in refs}) == 1
        assert all(ref.sha256 and ref.sha256 in ref.uri for ref in refs)


def test_the_split_is_the_requested_one_or_the_configs_only_one(hub) -> None:
    with hub("mteb-nfcorpus"):
        dataset = load_dataset(f"hf://mteb/nfcorpus@{SHA}", split="test")
        assert dataset.split == "test", "the requested split is in the qrels config's declared splits"
        dataset = load_dataset(f"hf://mteb/nfcorpus@{SHA}", split="dev")
        assert dataset.qrels != {}, "the dev qrels read when asked"
        assert dataset.split == "dev"


def test_an_unresolvable_split_is_an_error_naming_the_splits(hub) -> None:
    with hub("mteb-nfcorpus"):
        with pytest.raises(ConfigError, match="declares the splits"):
            load_dataset(f"hf://mteb/nfcorpus@{SHA}", split="validation")


def test_the_queries_are_cut_to_those_with_qrels(hub) -> None:
    """mteb cuts the queries at load; the fixture's query table holds PLAIN-3, which the test qrels lack."""
    with hub("mteb-nfcorpus"):
        dataset = load_dataset(f"hf://mteb/nfcorpus@{SHA}")
        assert set(dataset.queries) == {"PLAIN-2"} <= set(dataset.qrels)


# ---------------------------------------------------------------------------
# The duplicates policy over the labels
# ---------------------------------------------------------------------------


def _labelled_twice(root: Path, *, grades: tuple[float, float]) -> None:
    path = root / "NanoArguAnaRetrieval/qrels.parquet"
    frame = pd.read_parquet(path)
    first, second = frame.iloc[0].copy(), frame.iloc[0].copy()
    first["score"], second["score"] = grades
    pd.DataFrame([first, second]).to_parquet(path)


def test_a_pair_labelled_twice_with_the_same_grade_folds_with_a_count(hub) -> None:
    """Decision 30: the same pair with the same grade folds and is counted in the provenance."""
    with hub("rcp-ndcg-nanobeir") as root:
        _labelled_twice(root, grades=(1.0, 1.0))
        dataset = load_dataset(f"{NANOBEIR}@{SHA}")
        assert len(next(iter(dataset.qrels.values()))) == 1
        assert dataset.provenance.duplicates is not None
        assert (dataset.provenance.duplicates.folded, dataset.provenance.duplicates.resolved) == (1, 0)
        assert dataset.provenance.duplicates.policy == "error"


def test_a_conflicting_duplicate_is_refused_naming_the_rows(hub) -> None:
    with hub("rcp-ndcg-nanobeir") as root:
        _labelled_twice(root, grades=(1.0, 2.0))
        with pytest.raises(DataError, match="appears twice with different content"):
            load_dataset(f"{NANOBEIR}@{SHA}")


def test_duplicates_last_takes_the_last_row_as_mteb_does(hub) -> None:
    with hub("rcp-ndcg-nanobeir") as root:
        _labelled_twice(root, grades=(1.0, 2.0))
        dataset = load_dataset(f"{NANOBEIR}@{SHA}", duplicates="last")
        (judged,) = dataset.qrels.values()
        assert list(judged.values()) == [2.0]
        assert dataset.provenance.duplicates is not None
        assert dataset.provenance.duplicates.resolved == 1 and dataset.provenance.duplicates.policy == "last"


def test_an_unknown_duplicates_policy_is_refused() -> None:
    with pytest.raises(ConfigError, match="duplicates"):
        HubReader("mteb/nfcorpus", duplicates="first")


# ---------------------------------------------------------------------------
# The card-less degradation and the conventional path layout
# ---------------------------------------------------------------------------


def _staged_without_card(root: Path, *, offline: bool = False):
    """Serve the fixture repository without its card: ``_hub_file`` returns ``None`` where the Hub would 404
    (or raises the offline miss, with ``offline``, the shape an uncached offline read raises)."""
    listing = sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and p.name != "README.md")

    def download(repo: str, path: str, revision: str | None = None, **_: object):
        target = root / path
        if path == "README.md":
            if offline:
                from huggingface_hub.errors import LocalEntryNotFoundError

                raise LocalEntryNotFoundError(f"offline and uncached: {path}")
            return None
        return target if target.is_file() else None

    return {"_hub_file": download, "_hub_listing": lambda repo, revision: listing}


def test_a_card_less_repository_serves_the_path_layout(tmp_path: Path) -> None:
    """The released rcp-ndcg layout (``{subset}/qrels.parquet`` and friends) reads without a card."""
    from rcp_ndcg.data.io import hub as hub_module

    root = Path(__file__).parent / "fixtures" / "rcp-ndcg-nanobeir"
    with mock.patch.multiple(hub_module, **_staged_without_card(root)):
        dataset = load_dataset(f"{NANOBEIR}@{SHA}")
        assert dataset.qrels and dataset.candidates is not None


def test_an_uncached_card_falls_back_with_a_typed_warning(tmp_path: Path) -> None:
    """Offline without the card, the reader says it is serving the path layout -- never silently."""
    from rcp_ndcg.data.io import hub as hub_module

    root = Path(__file__).parent / "fixtures" / "rcp-ndcg-nanobeir"
    with mock.patch.multiple(hub_module, **_staged_without_card(root, offline=True)):
        with pytest.warns(RcpNdcgWarning, match="path layout") as seen:
            dataset = load_dataset(f"{NANOBEIR}@{SHA}")
        assert seen[0].message.code == "CARD_UNCACHED"
        assert dataset.qrels, "the conventional paths serve the tables"
