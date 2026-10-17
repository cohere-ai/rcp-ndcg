"""The ``mteb:<Task>`` reader: a task loaded through mteb itself (the ``[mteb]`` extra, network-gated).

The reader runs mteb's own ``load_data`` and converts ``task.dataset[subset][split]``, so the custom-loaded
tasks (ViDoRe v1's id prefixes, BRIGHT's exclusion-as-``top_ranked``) read through the same contract as every
other source. The tasks here are public and pinned by mteb's own task metadata. Run one file at a time, online::

    RCP_NDCG_NETWORK_TESTS=1 RCP_NDCG_TEST_TIMEOUT=600 \\
        uv run pytest tests/data/test_io_mteb_task.py -q -p no:cacheprovider -o faulthandler_timeout=300
"""

from __future__ import annotations

import os

import pytest

from rcp_ndcg.data import load_dataset
from rcp_ndcg.data.io.mteb_task import MtebTaskReader
from rcp_ndcg.errors import ConfigError

mteb = pytest.importorskip("mteb")

pytestmark = [
    pytest.mark.network,
    pytest.mark.skipif(os.environ.get("RCP_NDCG_NETWORK_TESTS") != "1", reason="needs the public Hugging Face Hub"),
]


def test_a_monolingual_task_loads_through_mteb() -> None:
    dataset = load_dataset("mteb:NFCorpus")

    assert dataset.task == "NFCorpus"
    assert dataset.name == "NFCorpus", "the default subset takes the task's name"
    assert (dataset.subset, dataset.split) == ("default", "test")
    assert dataset.qrels and dataset.queries and dataset.corpus
    assert dataset.provenance.source_uri == "mteb:NFCorpus/default@test"
    assert dataset.provenance.revision, "the task metadata pins the dataset revision"


def test_a_task_instruction_comes_from_the_task_metadata() -> None:
    """``NanoArguAnaRetrieval`` sets ``TaskMetadata.prompt``: one instruction for the whole task, a field of
    its own (the recipe places it; nothing merges it into a query's text at load)."""
    dataset = load_dataset("mteb:NanoArguAnaRetrieval")

    assert dataset.task_instruction == {"query": "Given a claim, find documents that refute the claim"}
    assert all(query.instruction is None for query in dataset.queries.values()), (
        "the task instruction is not a per-query instruction"
    )


def test_a_multilingual_task_needs_its_subset_named() -> None:
    with pytest.raises(ConfigError, match="name the subset") as caught:
        load_dataset("mteb:MIRACLRetrieval")
    assert "en" in str(caught.value.hint) and "de" in str(caught.value.hint)


def test_a_multilingual_task_loads_one_subset() -> None:
    dataset = load_dataset("mteb:MIRACLRetrieval/en")

    assert dataset.task == "MIRACLRetrieval"
    assert (dataset.name, dataset.subset, dataset.split) == ("en", "en", "dev")
    assert dataset.qrels and dataset.queries


def test_an_unknown_task_subset_or_split_is_refused() -> None:
    with pytest.raises(ConfigError, match="could not be loaded|not found"):
        load_dataset("mteb:NoSuchTaskAnywhere")

    with pytest.raises(ConfigError, match="has no subset"):
        load_dataset("mteb:MIRACLRetrieval/xx")

    with pytest.raises(ConfigError, match="declares the evaluation splits"):
        load_dataset("mteb:NFCorpus@validation")


def test_the_split_can_come_from_the_uri() -> None:
    reader = MtebTaskReader("NFCorpus@test")
    assert reader.split == "test"


def test_two_splits_are_refused() -> None:
    with pytest.raises(ConfigError, match="two splits"):
        MtebTaskReader("NFCorpus@test", split="dev")


def test_an_empty_task_name_is_refused() -> None:
    with pytest.raises(ConfigError, match="mteb:<TaskName>"):
        MtebTaskReader("/en")


def test_the_mteb_reader_passes_the_shared_conformance() -> None:
    """The built-in ``mteb:`` reader runs through the same ``rcp_ndcg.testing.io_conformance`` a plugin does,
    against the real task (NFCorpus's full corpus, mteb's own ids and texts)."""
    from rcp_ndcg.testing import io_conformance

    io_conformance(MtebTaskReader("NFCorpus"))


def test_a_v1_style_task_converts_like_mteb_does() -> None:
    """A task whose loader fills ``corpus``/``queries``/``relevant_docs`` (BRIGHT's own loader) converts to
    ``task.dataset`` the way mteb's ``evaluate`` does; without the conversion the read crashed with an
    ``AttributeError`` on ``task.dataset is None``."""
    dataset = load_dataset("mteb:BrightBiologyRetrieval")

    assert dataset.task == "BrightBiologyRetrieval"
    assert (dataset.subset, dataset.split) == ("default", "standard")
    assert dataset.qrels and dataset.queries and dataset.corpus
    assert dataset.candidates is None, "this task builds no top_ranked (the task's own loader says so)"
    assert dataset.provenance.revision, "the task metadata pins the dataset revision"


def test_a_media_query_keeps_its_instruction() -> None:
    """The per-query instruction is a field in both branches: a query with media keeps it too."""
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (1, 2, 3)).save(buffer, format="PNG")
    reader = MtebTaskReader("NFCorpus")  # no task load: the row conversion is what is under test
    row = {
        "id": "q1",
        "text": "a query",
        "instruction": "per-query instruction",
        "image": {"bytes": buffer.getvalue(), "path": "p.png"},
    }
    query = reader._query(row)

    assert query.as_content.has_media
    assert query.instruction == "per-query instruction"


def test_an_mteb_uri_records_the_tasks_pinned_revision() -> None:
    """``dataset_uri_revision`` resolves an ``mteb:`` task's metadata revision, like an ``hf://`` URI's commit."""
    from rcp_ndcg.data.revisions import dataset_uri_revision

    payload = dataset_uri_revision("mteb:NFCorpus")

    assert payload is not None
    assert payload["repo"] == "mteb/nfcorpus"
    assert payload["verified"] and payload["commit"]
