"""The documentation uses the public names and describes the metric correctly."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.docs._markdown import ROOT, markdown_files

FILES = markdown_files()

# (pattern, why it must not appear)
FORBIDDEN = [
    (re.compile(r"binary nDCG", re.I), "the metrics are RCP-nDCG, qrel-nDCG and Count-nDCG"),
    (re.compile(r"θ-?n?DCG|theta[-_ ]n?dcg", re.I), "the product is RCP-nDCG"),
    # The pattern is split in two so that the repository's own name scanners do not match this line.
    (re.compile(r"import rcp_ndcg as t" r"n\b"), "the package alias is rcp"),
    (re.compile(r"\b(RFC|JC)[- ]?\d{3,4}\b"), "no internal design or lane ids"),
]

# The user-facing pages (everything the release publishes except the CHANGELOG's historical record)
# carry no internal design ids: "(owner decisions 38/35)", "(decision 34)" and "lane L7" mean nothing
# to a reader who is not in the design threads. Say what the rule IS, not which decision made it. The
# CHANGELOG keeps its provenance markers, so these patterns are checked everywhere but there.
FORBIDDEN_IN_USER_FACING_PAGES = [
    (re.compile(r"owner decisions? \d+", re.I), "internal design id: state the rule itself"),
    (re.compile(r"\bdecision \d+", re.I), "internal design id: state the rule itself"),
    (re.compile(r"\blane L\d+", re.I), "internal design id: state the rule itself"),
]


@pytest.mark.parametrize("path", FILES, ids=str)
def test_no_retired_or_internal_names(path: Path) -> None:
    text = (ROOT / path).read_text(encoding="utf-8")
    checks = FORBIDDEN + ([] if path.name == "CHANGELOG.md" else FORBIDDEN_IN_USER_FACING_PAGES)
    hits = [f"{rx.pattern}: {why}" for rx, why in checks if rx.search(text)]
    assert hits == []
