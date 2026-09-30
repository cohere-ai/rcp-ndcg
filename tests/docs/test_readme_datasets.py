"""Every released dataset is reachable from the repository: the README and docs/data.md link each one.

The list is the release manifest. The final release gate also compares it with the live Hugging Face listing
(the ``rcp-ndcg-*`` ids of the owner's namespace), so a dataset added later cannot go missing here.
"""

from __future__ import annotations

import importlib.util

import pytest

from rcp_ndcg.data import SUITES
from tests.docs._markdown import ROOT

RELEASED = (
    "fabianschmidt-cohere/rcp-ndcg-nanobeir",
    "fabianschmidt-cohere/rcp-ndcg-bright",
    "fabianschmidt-cohere/rcp-ndcg-vidore-v3",
    "fabianschmidt-cohere/rcp-ndcg-trecdl",
    "fabianschmidt-cohere/rcp-ndcg-external-validation",
)
URL = "https://huggingface.co/datasets/{})"


@pytest.mark.parametrize("page", ["README.md", "docs/data.md"])
def test_the_page_links_every_released_dataset(page: str) -> None:
    text = (ROOT / page).read_text(encoding="utf-8")
    missing = [name for name in RELEASED if URL.format(name) not in text]
    assert not missing, f"{page} must link every released dataset: {missing}"


def test_the_release_list_matches_the_fetch_script_and_the_suites() -> None:
    spec = importlib.util.spec_from_file_location("_fetch_data", ROOT / "experiments" / "fetch_data.py")
    assert spec is not None and spec.loader is not None
    fetch_data = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fetch_data)
    assert {f"{fetch_data.OWNER}/{name}" for name in fetch_data.DATASETS} == set(RELEASED)
    assert {suite.repo for suite in SUITES.values()} <= set(RELEASED)
