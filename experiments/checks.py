"""Compare reproduced numbers with the values printed in the paper and set the exit code."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["Checker", "KnownDeviation", "paper_values"]

_PAPER_VALUES = Path(__file__).resolve().parent / "paper_values.json"


def paper_values() -> dict:
    """The paper's printed values (``paper_values.json``, transcribed from arXiv:2609.35739)."""
    return json.loads(_PAPER_VALUES.read_text())


@dataclass(frozen=True)
class KnownDeviation:
    """A documented difference between the paper's printed values and the reproduction.

    Fields:
        bound: The largest difference (in the compared unit, e.g. nDCG points) the documentation covers.
        reason: Why the reproduction differs; printed with every row that takes the deviation.
        cells: Exactly how many compared values may take this deviation -- the documented population, not a cap;
            a different count (either way) means the documentation or the data moved, and both are failures.
        label: Where the population lives (e.g. the table position), for the failure messages.

    Declarations are matched by value: a ``KnownDeviation`` with the same fields is the same documented
    population, whoever constructed the instance. Two populations at different positions carry different
    labels, so they never merge.
    """

    bound: float
    reason: str
    cells: int
    label: str = ""


@dataclass
class _Row:
    label: str
    paper: float
    value: float
    tol: float
    status: str
    note: str = ""
    dev: KnownDeviation | None = None


@dataclass
class Checker:
    """Collects ``paper vs reproduced`` comparisons and prints them.

    A value **matches** when it is within ``tol`` of the printed value (``tol`` is stated per check and is at
    least the printed rounding). A **known deviation** is a documented difference with its own bound and reason;
    it does not fail the run while it stays inside that bound -- and while it stays inside its documented
    population (``KnownDeviation.cells``, declared in ``deviations``). Everything else **fails**.
    """

    title: str
    deviations: tuple[KnownDeviation, ...] = ()
    rows: list[_Row] = field(default_factory=list)

    def compare(
        self,
        label: str,
        paper: float,
        value: float,
        tol: float,
        *,
        known: KnownDeviation | None = None,
        quiet: bool = False,
    ) -> bool:
        """Record one comparison; return whether it passed. ``known`` marks a documented deviation (its
        ``(bound, reason, cells)`` population applies). ``quiet`` hides a passing row from the printed table
        (it still counts)."""
        diff = abs(value - paper) if not (math.isnan(value) or math.isnan(paper)) else math.inf
        if diff <= tol + 1e-9:
            status, note = "ok", ""
        elif known is not None and diff <= known.bound + 1e-9:
            status, note = "known", known.reason
        else:
            status, note = "FAIL", known.reason if known else ""
        self.rows.append(_Row(label, paper, value, tol, status, note, known))
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
        """Print the summary; return the process exit code (1 if any comparison failed).

        Also fails when the known deviations did not materialise exactly as documented: every declared
        deviation must be taken by exactly its ``cells`` rows (a row takes it when it prints ``known``), and no
        row may deviate outside every declaration -- the documented populations, not just the per-cell bounds.
        Declarations are matched by value, so a re-created instance with the same fields is no "stray".
        """
        n = len(self.rows)
        by = {s: sum(r.status == s for r in self.rows) for s in ("ok", "known", "FAIL")}
        population_failures = []
        for dev in self.deviations:
            taken = sum(r.status == "known" and r.dev == dev for r in self.rows)
            if taken != dev.cells:
                population_failures.append(
                    f"{taken} row(s) took the documented deviation [{dev.label or dev.reason}]: {dev.cells}"
                )
        strays = sum(
            r.status == "known" and (r.dev is None or all(r.dev != d for d in self.deviations)) for r in self.rows
        )
        if strays:
            population_failures.append(f"{strays} row(s) deviated outside every documented population")
        for failure in population_failures:
            print(f"  FAIL  known-deviation population: {failure}")
        print(f"\n{self.title}: {n} checks, {by['ok']} match, {by['known']} known deviations, {by['FAIL']} failed")
        return 1 if by["FAIL"] or population_failures else 0
