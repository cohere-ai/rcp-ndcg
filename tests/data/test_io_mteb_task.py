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
