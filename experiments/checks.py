"""Compare reproduced numbers with the values printed in the paper and set the exit code."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["Checker", "paper_values"]

_PAPER_VALUES = Path(__file__).resolve().parent / "paper_values.json"


def paper_values() -> dict:
    """The paper's printed values (``paper_values.json``, transcribed from arXiv:2609.35739)."""
    return json.loads(_PAPER_VALUES.read_text())


@dataclass
class _Row:
    label: str
    paper: float
    value: float
    tol: float
    status: str
    note: str = ""


@dataclass
class Checker:
    """Collects ``paper vs reproduced`` comparisons and prints them.

    A value **matches** when it is within ``tol`` of the printed value (``tol`` is stated per check and is at
    least the printed rounding). A **known deviation** is a documented difference with its own bound and reason;
    it does not fail the run while it stays inside that bound. Everything else **fails**.
    """

    title: str
    rows: list[_Row] = field(default_factory=list)

    def compare(
        self,
        label: str,
        paper: float,
        value: float,
        tol: float,
        *,
        known: tuple[float, str] | None = None,
        quiet: bool = False,
    ) -> bool:
        """Record one comparison; return whether it passed. ``known = (bound, reason)`` marks a documented
        deviation. ``quiet`` hides a passing row from the printed table (it still counts)."""
        diff = abs(value - paper) if not (math.isnan(value) or math.isnan(paper)) else math.inf
        if diff <= tol + 1e-9:
            status, note = "ok", ""
        elif known is not None and diff <= known[0] + 1e-9:
            status, note = "known", known[1]
        else:
            status, note = "FAIL", known[1] if known else ""
        self.rows.append(_Row(label, paper, value, tol, status, note))
        if not quiet or status != "ok":
            self._print(self.rows[-1])
        return status != "FAIL"

    def count(self, label: str, paper: int, value: int) -> bool:
        """Exact comparison of a count."""
        return self.compare(label, float(paper), float(value), 0.0)

    @staticmethod
    def _print(r: _Row) -> None:
        extra = f"  [{r.note}]" if r.note else ""
        print(
            f"  {r.status:<5} {r.label:<58} paper {r.paper:>9.4g}  reproduced {r.value:>11.5g}  (tol {r.tol:g}){extra}"
        )

    def finish(self) -> int:
        """Print the summary; return the process exit code (1 if any comparison failed)."""
        n = len(self.rows)
        by = {s: sum(r.status == s for r in self.rows) for s in ("ok", "known", "FAIL")}
        print(f"\n{self.title}: {n} checks, {by['ok']} match, {by['known']} known deviations, {by['FAIL']} failed")
        return 1 if by["FAIL"] else 0
