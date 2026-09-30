"""Paper anchor: the released gains follow from the released abilities and item parameters.

Five rows of the public NanoBEIR release (``fixtures/paper_label_rows.json``, one per subset, gains from 0.02
to 0.97) pin the gain formula: a document's released ``gain`` is ``gain(theta)`` of its released ``theta``
under the item parameters the release ships. The release stores both as float64 and they agree to rounding
over all of its rows, so the tolerance is 1e-12.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from rcp_ndcg_core import gain

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "paper_label_rows.json").read_text())


@pytest.mark.parametrize("row", FIXTURE["rows"], ids=lambda row: f"{row['subset']}:{row['doc_id'][:16]}")
def test_a_released_gain_is_the_gain_of_the_released_ability(row: dict) -> None:
    assert gain(row["theta"], FIXTURE["item_params"]) == pytest.approx(row["gain"], abs=1e-12)
