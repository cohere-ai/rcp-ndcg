"""The rank correlation the case format's ``spearman_min`` tolerance bounds: one home, tie-corrected.

The Spearman rank correlation of two score rows is the Pearson correlation of their ranks, with ties
averaged -- the general form (the closed form ``1 - 6*sum(d^2)/(n(n^2-1))`` is exact only without ties).
Implemented here on numpy alone (no SciPy), because the case format's ``spearman_min`` tolerance bounds
this correlation and the package must not grow a dependency for it. Two constant rows carry no ranking
information: the correlation is ``nan`` and a comparison against it fails, never silently passes.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .errors import ConformanceError

__all__ = ["average_ranks", "spearman"]


def spearman(got: Sequence[float], want: Sequence[float]) -> float:
    """The Spearman rank correlation of two score rows, ties averaged.

    Inputs: two rows of one length (at least one element). Outputs: the correlation in [-1, 1] computed
    as the Pearson correlation of the average ranks; ``1.0`` for a single pair (nothing to correlate);
    ``nan`` when either row is constant (no ranking information).

    Raises:
        ConformanceError: the rows hold different numbers of values.
    """
    if len(got) != len(want):
        raise ConformanceError(f"spearman needs two rows of one length, got {len(got)} and {len(want)}")
    if len(got) < 2:
        return 1.0
    got_values, want_values = np.asarray(got, dtype=np.float64), np.asarray(want, dtype=np.float64)
    if float(np.ptp(got_values)) == 0.0 or float(np.ptp(want_values)) == 0.0:
        return float("nan")
    return float(np.corrcoef(average_ranks(list(got_values)), average_ranks(list(want_values)))[0, 1])


def average_ranks(values: Sequence[float]) -> np.ndarray:
    """Zero-based ranks with ties averaged (the rank correlation's convention, before correlating).

    Inputs: one row of numbers. Outputs: one rank per value -- the position each value would take in the
    sorted order, with equal values sharing the average of their positions.
    """
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="stable")
    sorted_values = array[order]
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        stop = start + 1
        while stop < len(array) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranks
