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
        assert dataset.qrels == {"PLAIN-2": {"MED-10": 2.0}}, "jsonl scores read as floats"
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
    ``qrel_diff`` config); the config wins over the column, as in mteb.

    The fixture's one kept query carries one instruction, so the reader lifts it to the task instruction
    (decision 33: a uniform instruction is a task instruction); the column's own value never reaches the
    dataset.
    """
    with hub("mteb-core17instructionretrieval") as root:
        # The two sources carry the same instruction in the real repository; make the column differ, so the
        # precedence is observable.
        queries = root / "queries/test-00000-of-00001.parquet"
        frame = pd.read_parquet(queries)
        frame["instruction"] = "the column's own instruction"
        frame.to_parquet(queries)

        dataset = load_dataset(f"hf://mteb/Core17InstructionRetrieval@{SHA}")

        assert set(dataset.queries) == {"677-changed"}, "the queries are cut to those with qrels"
        assert dataset.task_instruction != "the column's own instruction", "the config wins when it exists"
        assert all(query.instruction is None for query in dataset.queries.values()), "the lift clears it"


def test_a_differing_instruction_config_stays_per_query(hub) -> None:
    """Instructions that differ per query are mteb's InstructionRetrieval data: no lift, every query keeps
    its own."""
    with hub("mteb-core17instructionretrieval") as root:
        instructions = root / "instruction/test-00000-of-00001.parquet"
        frame = pd.read_parquet(instructions)
        frame = pd.concat([frame, frame.assign(**{"query-id": "another-query", "instruction": "another instruction"})])
        frame.to_parquet(instructions)

        dataset = load_dataset(f"hf://mteb/Core17InstructionRetrieval@{SHA}")

        assert dataset.task_instruction is None
        assert dataset.queries["677-changed"].instruction is not None


def test_a_uniform_instruction_column_is_lifted_when_the_card_declares_it(hub) -> None:
    """The column form (no instruction config): the card's ``dataset_info`` declares the queries' features,
    so the reader knows the column exists without reading the table -- and lifts a uniform one."""
    with hub("mteb-core17instructionretrieval") as root:
        _drop_the_instruction_config(root)
        queries = root / "queries/test-00000-of-00001.parquet"
        frame = pd.read_parquet(queries)
        frame["instruction"] = "one instruction for the whole subset"
        frame.to_parquet(queries)

        dataset = load_dataset(f"hf://mteb/Core17InstructionRetrieval@{SHA}")

        assert dataset.task_instruction == "one instruction for the whole subset"
        assert all(query.instruction is None for query in dataset.queries.values())


def test_the_column_completeness_is_decided_over_the_kept_queries(hub) -> None:
    """A query the qrels cut drops is never read: its missing instruction cannot make a coherent subset look
    mixed (mteb keeps only the queries with qrels, and the lift must decide over the same set)."""
    with hub("mteb-core17instructionretrieval") as root:
        _drop_the_instruction_config(root)
        queries = root / "queries/test-00000-of-00001.parquet"
        frame = pd.read_parquet(queries)
        frame["instruction"] = ""
        frame.loc[frame["id"] == "677-changed", "instruction"] = "one instruction for the kept query"
        frame.to_parquet(queries)

        dataset = load_dataset(f"hf://mteb/Core17InstructionRetrieval@{SHA}")

        assert dataset.task_instruction == "one instruction for the kept query"
        assert all(query.instruction is None for query in dataset.queries.values())


def test_a_dataset_info_only_card_falls_back_to_the_conventional_paths(tmp_path: Path) -> None:
    """A card that lists a config only under ``dataset_info`` (its features, no ``data_files``) names no file:
    the conventional ``{subset}/{part}.parquet`` path still serves it -- a phantom config must not shadow it
    (it would raise "the 'queries' config names no file for the split" instead of reading the table)."""
    from rcp_ndcg.data.io import hub as hub_module
    from tests.data.hub_stubs import _fake_hub_file, _fake_listing

    root = tmp_path / "repo"
    (root / "default").mkdir(parents=True)
    (root / "README.md").write_text(
        "---\ndataset_info:\n"
        "- config_name: queries\n  features:\n  - name: id\n    dtype: string\n"
        "- config_name: corpus\n  features:\n  - name: id\n    dtype: string\n"
        "- config_name: qrels\n  features:\n  - name: query-id\n    dtype: string\n---\n",
        encoding="utf-8",
    )
    pd.DataFrame({"query-id": ["q1"], "corpus-id": ["d1"], "score": [1]}).to_parquet(root / "default/qrels.parquet")
    pd.DataFrame({"id": ["q1"], "text": ["find docs"]}).to_parquet(root / "default/queries.parquet")
    pd.DataFrame({"id": ["d1"], "title": ["T"], "text": ["a body"]}).to_parquet(root / "default/corpus.parquet")

    with (
        mock.patch.object(hub_module, "_hub_file", _fake_hub_file(root)),
        mock.patch.object(hub_module, "_hub_listing", _fake_listing(root)),
    ):
        dataset = load_dataset(f"hf://owner/repo@{SHA}")
        # The reads are lazy: keep them inside the fake Hub.
        assert dataset.qrels == {"q1": {"d1": 1.0}}
        assert set(dataset.queries) == {"q1"}
        assert set(dataset.corpus) == {"d1"}


def test_a_column_that_instructs_only_some_queries_is_refused(hub) -> None:
    """A mixed subset -- the kept queries instructed, others not (or the other way round) -- is neither a
    task instruction nor coherent per-query data: refused when the queries are read, never half-lifted."""
    with hub("mteb-core17instructionretrieval") as root:
        _drop_the_instruction_config(root)
        # Two KEPT queries (a label for the second one too), one instructed and one not: a mixed subset.
        qrels = root / "qrels/test-00000-of-00001.parquet"
        labels = pd.read_parquet(qrels)
        labels = pd.concat([labels, labels.iloc[[0]].assign(**{"query-id": "310-og"})])
        labels.to_parquet(qrels)
        queries = root / "queries/test-00000-of-00001.parquet"
        frame = pd.read_parquet(queries)
        frame["instruction"] = ""
        frame.loc[frame["id"] == "677-changed", "instruction"] = "only one of the kept queries carries one"
        frame.to_parquet(queries)

        dataset = load_dataset(f"hf://mteb/Core17InstructionRetrieval@{SHA}")
        assert dataset.task_instruction is None
        with pytest.raises(DataError, match="mixed subset"):
            _ = dataset.queries


def _drop_the_instruction_config(root: Path) -> None:
    """The fixture card's instruction config, removed from the ``configs`` and the ``dataset_info`` lists
    (the queries' own ``instruction`` column becomes the only source)."""
    import yaml

    card = root / "README.md"
    text = card.read_text(encoding="utf-8")
    header = yaml.safe_load(text.split("---", 2)[1]) or {}
    header["configs"] = [entry for entry in header["configs"] if entry["config_name"] != "instruction"]
    header["dataset_info"] = [entry for entry in header["dataset_info"] if entry["config_name"] != "instruction"]
    card.write_text(
        "---\n" + yaml.safe_dump(header, sort_keys=False) + "---" + text.split("---", 2)[2], encoding="utf-8"
    )


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
        documents = dataset.corpus
        assert documents["corpus-test-10"].as_content.has_media
        assert documents["corpus-test-10"].text == "page text", "the corpus's OCR text column is the body"
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
        document = corpus("auto")["corpus-test-10"]
        assert document.as_content.has_media and document.text, "a multi-column row reads both"


def test_document_parts_image_drops_the_ocr_text(hub) -> None:
    """The visual run: what the judge sees is the page, not a transcription (ViDoRe v3 ships both in one
    row; they are different experiments over the same judgements)."""
    with hub("vidore-vidore-v3-finance-en"):
        dataset = load_dataset(f"hf://vidore/vidore_v3_finance_en_mteb_format/english@{SHA}", document_parts="image")
        document = dataset.corpus["corpus-test-10"]
        assert document.as_content.has_media and document.text == ""


def test_document_parts_text_drops_the_pages(hub) -> None:
    """The OCR baseline: comparable to the visual run because the ids and the qrels are identical."""
    with hub("vidore-vidore-v3-finance-en"):
        dataset = load_dataset(f"hf://vidore/vidore_v3_finance_en_mteb_format/english@{SHA}", document_parts="text")
        document = dataset.corpus["corpus-test-10"]
        assert not document.as_content.has_media and document.text == "page text"


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
        assert len({ref.uri for ref in refs}) == 1, "the same page twice writes one file"
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


def test_a_raw_binary_media_cell_is_read_by_its_magic_numbers(hub) -> None:
    """mteb's Any2Any repositories store the page bytes directly (a binary parquet column), not the
    ``{"bytes", "path"}`` struct: the reader sniffs the format from the bytes instead of guessing from the
    column's name."""
    import io

    from PIL import Image

    with hub("vidore-vidore-v3-finance-en") as root:
        path = root / "english-corpus/test-00000-of-00001.parquet"
        frame = pd.read_parquet(path)
        buffer = io.BytesIO()
        Image.new("RGB", (16, 12), (7, 8, 9)).save(buffer, format="JPEG")
        frame["image"] = [buffer.getvalue() for _ in frame.index]  # raw JPEG bytes, no struct
        frame.to_parquet(path)

        dataset = load_dataset(f"hf://vidore/vidore_v3_finance_en_mteb_format/english@{SHA}")
        refs = [ref for document in dataset.corpus.values() for ref in document.as_content.media]
        assert refs and all(ref.mime == "image/jpeg" and ref.uri.endswith(".jpg") for ref in refs)
        assert all((ref.width, ref.height) == (16, 12) for ref in refs)


def test_a_raw_binary_media_cell_of_no_known_format_is_refused(hub) -> None:
    with hub("vidore-vidore-v3-finance-en") as root:
        path = root / "english-corpus/test-00000-of-00001.parquet"
        frame = pd.read_parquet(path)
        frame["image"] = [b"\x00\x01\x02not-a-format" for _ in frame.index]
        frame.to_parquet(path)

        dataset = load_dataset(f"hf://vidore/vidore_v3_finance_en_mteb_format/english@{SHA}")
        with pytest.raises(DataError, match="no magic number names"):
            _ = dataset.corpus


@pytest.mark.parametrize(
    ("fixture", "uri"),
    [
        ("mteb-nfcorpus", "mteb/nfcorpus"),
        ("mteb-askubutudupquestions", "mteb/AskUbuntuDupQuestions"),
        ("mteb-core17instructionretrieval", "mteb/Core17InstructionRetrieval"),
        ("mteb-blink-it2i", "mteb/blink-it2i"),
        ("vidore-vidore-v3-finance-en", "vidore/vidore_v3_finance_en_mteb_format/english"),
        ("rcp-ndcg-nanobeir", "fabianschmidt-cohere/rcp-ndcg-nanobeir/NanoArguAnaRetrieval"),
    ],
)
def test_the_hub_reader_passes_the_shared_conformance(fixture: str, uri: str, tmp_path: Path) -> None:
    """The built-in Hub reader runs through the same ``rcp_ndcg.testing.io_conformance`` a plugin does; the
    fixture repositories' sampled rows are internally consistent (their pools and qrels name the sampled
    documents). The multilingual mirror is left out: its corpus is 32.9M rows across 28 shards, so a sampled
    corpus cannot cover the sampled qrels (its layout is covered by its own tests and the live load)."""
    from rcp_ndcg.testing import io_conformance

    with hub_fixture(fixture, tmp_path):
        io_conformance(HubReader(uri, revision=SHA))


def test_a_video_cell_in_the_struct_form_keeps_its_container_mime(hub) -> None:
    """A media column may hold a container: the cell's suffix (or the magic numbers) states the MIME, and
    ``store_media`` must not refuse a video suffix for want of one."""
    with hub("vidore-vidore-v3-finance-en") as root:
        path = root / "english-corpus/test-00000-of-00001.parquet"
        frame = pd.read_parquet(path)
        payload = b"RIFF" + (0).to_bytes(4, "little") + b"AVI " + b"\x00" * 64
        frame["video"] = [{"bytes": payload, "path": "clip.avi"} for _ in frame.index]
        frame = frame.drop(columns=["image"])
        frame.to_parquet(path)

        dataset = load_dataset(f"hf://vidore/vidore_v3_finance_en_mteb_format/english@{SHA}")
        refs = [ref for document in dataset.corpus.values() for ref in document.as_content.media]
        assert refs and all(ref.mime == "video/x-msvideo" and ref.uri.endswith(".avi") for ref in refs)


def test_a_decoded_video_object_is_refused_by_name() -> None:
    """``datasets``' Video feature hands back a decoder object; this reader cannot encode one without ffmpeg,
    and says so instead of crashing with an AttributeError."""
    from rcp_ndcg.data.io.hub import _encode_media

    class VideoDecoder:
        """The shape of a decoded video object: no ``save``, no bytes."""

    with pytest.raises(DataError, match="decoded VideoDecoder"):
        _encode_media(VideoDecoder(), column="video")


def test_an_unknown_hub_option_is_a_typed_config_error() -> None:
    """The retired reader's ``corpus_split=`` once reached HubReader as a bare TypeError; the reader table's
    option check keeps the typed contract."""
    with hub_fixture("mteb-nfcorpus"):
        with pytest.raises(ConfigError, match="takes no option"):
            load_dataset(f"hf://mteb/nfcorpus@{SHA}", corpus_split="corpus")


def test_document_parts_is_a_corpus_setting_only(hub) -> None:
    """An Any2Any query's image is the query, not a corpus-column choice: ``document_parts='text'`` strips
    corpus media, never a query's."""
    with hub("mteb-blink-it2i") as root:
        path = root / "corpus-00000-of-00001.parquet"
        frame = pd.read_parquet(path)
        frame["text"] = ["an OCR transcription" for _ in frame.index]  # a multi-column corpus row
        frame.to_parquet(path)

        dataset = load_dataset("hf://mteb/blink-it2i", revision=SHA, document_parts="text")
        (query,) = dataset.queries.values()
        assert query.as_content.has_media, "the query keeps its image"
        (document,) = dataset.corpus.values()
        assert not document.as_content.has_media and document.text == "an OCR transcription"


def test_the_hub_provenance_records_every_eager_tables_duplicates(hub) -> None:
    """The provenance's counts cover the tables every load reads -- the labels, the pools and the exclusions."""
    with hub("mteb-askubutudupquestions") as root:
        path = root / "top_ranked/test-00000-of-00001.parquet"
        frame = pd.read_parquet(path)
        pd.concat([frame, frame], ignore_index=True).to_parquet(path)  # the same pool twice

        dataset = load_dataset(f"hf://mteb/AskUbuntuDupQuestions@{SHA}")
        counts = dataset.provenance.duplicates
        assert counts is not None
        assert counts.folded == 1, "the repeated pool row folded and is counted"


def test_duplicates_last_cannot_resolve_a_streamed_corpus_row(hub) -> None:
    """A corpus streams: an earlier row is already yielded, so a conflicting duplicate refuses even under
    ``duplicates='last'`` -- with the option's scope named, never a resolution the data does not carry."""
    with hub("vidore-vidore-v3-finance-en") as root:
        path = root / "english-corpus/test-00000-of-00001.parquet"
        frame = pd.read_parquet(path)
        conflicting = frame.iloc[0].copy()
        conflicting["text"] = "a different page"
        pd.concat([frame, pd.DataFrame([conflicting])], ignore_index=True).to_parquet(path)

        dataset = load_dataset(f"hf://vidore/vidore_v3_finance_en_mteb_format/english@{SHA}", duplicates="last")
        with pytest.raises(DataError, match="appears twice with different content") as caught:
            _ = dataset.corpus
        assert "cannot replace a row it has already yielded" in str(caught.value.hint)
        assert "labels, pools and exclusions" in str(caught.value.hint)


def test_duplicates_last_resolves_a_pool_conflict(hub) -> None:
    """A pool is materialised into a dict, so the last row can win there: the resolution is recorded."""
    with hub("mteb-askubutudupquestions") as root:
        path = root / "top_ranked/test-00000-of-00001.parquet"
        frame = pd.read_parquet(path)
        conflicting = frame.iloc[0].copy()
        conflicting["corpus-ids"] = ["apositive_test_query0_00001"]
        pd.concat([frame, pd.DataFrame([conflicting])], ignore_index=True).to_parquet(path)

        dataset = load_dataset(f"hf://mteb/AskUbuntuDupQuestions@{SHA}", duplicates="last")
        assert dataset.candidates["test_query0"] == ["apositive_test_query0_00001"], "the last pool wins"
        assert dataset.provenance.duplicates is not None
        assert dataset.provenance.duplicates.resolved == 1
