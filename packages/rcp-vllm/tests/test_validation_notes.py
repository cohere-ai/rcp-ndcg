"""Every recipe note's claim about its GPU validation wave is the ledger's row.

``notes:`` is prose, but the claim it carries ("verified by its GPU validation wave (<wave>, <date>)") is
the one a release gate reads as evidence, and the generated ledger is the record of what actually ran. Eight
families once claimed ``rc0-202610121840`` / ``2026-10-12`` while the ledger recorded ``rc2`` / ``2026-11-09``
for every one of their variants: a validated state the evidence did not hold. The sweep below reads the
claim out of every family's notes and holds it to the ledger rows of that family's variants, so the next
stale note reds here instead of in a release.
"""

from __future__ import annotations

import re
from pathlib import Path

from rcp_vllm.recipe import iter_families

LEDGER = Path(__file__).resolve().parents[3] / "docs" / "reference" / "gpu-validation.md"
"""The generated ledger (``python -m rcp_test.jobs.ledger --write``): one row per shipped variant."""

_CLAIM = re.compile(r"verified by its GPU validation wave \(([^,]+), (\d{4}-\d{2}-\d{2})\)")
"""The notes' claim: ``(<wave id>, <YYYY-MM-DD>)``."""

_ROW = re.compile(r"^\| `([^`]+)` \| [^|]+ \| [^|]+ \| `([^`]+)` \| (\d{4}-\d{2}-\d{2}) \|", re.MULTILINE)
"""A ledger row: ``| `recipe` | role | image | `wave` | date | fingerprint | state |``."""


def test_every_notes_wave_claim_is_the_ledger_row_of_every_variant() -> None:
    """A note may only claim the wave and date the ledger records for the family's variants -- checked for
    every family that makes the claim, not for the ones a report happened to name."""
    rows = {recipe: (wave, date) for recipe, wave, date in _ROW.findall(LEDGER.read_text(encoding="utf-8"))}
    assert rows, f"the ledger at {LEDGER} parsed no rows"

    stale: dict[str, tuple[tuple[str, str], list[tuple[str, tuple[str, str] | None]]]] = {}
    for family in iter_families():
        for match in _CLAIM.finditer(family.notes):
            claim = (match.group(1), match.group(2))
            recorded = [(variant.id, rows.get(variant.id)) for variant in family.variants]
            if any(row is not None and row != claim for _, row in recorded):
                stale[family.id] = (claim, recorded)

    assert not stale, f"notes claiming a wave the ledger does not record (claim, variant rows): {stale}"
