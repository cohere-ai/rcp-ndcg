"""Window schedules of the two judging stages, with the paper's settings as defaults.

:class:`TournamentSchedule` (Stage A) and :class:`RubricSchedule` (Stage B) are
the single source of the judging schedule: the judging functions, the cost
estimate and the run config all read these models, and nothing redeclares their
numbers.

**Tournament** (per query, over the candidate pool of ``n`` documents):

1. *Random windows* -- ``random_windows`` balanced random groups of ``window``
   documents (every document drawn equally often), each also sent reversed when
   ``mirror`` is set, to cancel position bias;
2. *Stratified windows* -- ``stratified_windows`` groups of documents that share a
   tier of the preliminary Bradley-Terry ability, again mirrored;
3. *Adaptive windows* -- ``adaptive_batches`` batches of
   ``adaptive_windows_per_batch`` contiguous windows over the top
   ``adaptive_depth`` documents, chosen by the information of their adjacent
   boundaries (Fisher information x nDCG discount x novelty, covered boundaries
   discounted by ``overlap_discount``), with a Bradley-Terry refit between batches.

The paper's schedule is 53 + 27 mirrored windows of 10 (160 calls) and 7 x 8 = 56
adaptive calls: 216 calls per query of 150 candidates.

**Rubric** (per query): ``windows_per_query`` windows of ``window`` documents,
the first ``random_windows`` balanced random, the rest stratified by a preliminary
Rasch ability. The paper's schedule is 100 windows of 10, 50 of them random.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator
from rcp_ndcg_core.metric import discount

#: The document modalities a prompt is written for.
Modality = Literal["text", "image", "video"]

#: Default share of a covered boundary's value kept when a window over it is selected.
OVERLAP_DISCOUNT = 0.3


class TournamentSchedule(BaseModel):
    """The Stage A schedule; the defaults are the paper's.

    Attributes:
        window: Documents per random and stratified window.
        random_windows: Balanced random windows per query.
        stratified_windows: Windows over tiers of the preliminary ability, per query.
        mirror: Also send every random and stratified window in reverse order.
        adaptive_window: Documents per adaptive window.
        adaptive_depth: Adaptive windows are drawn from the top ``adaptive_depth`` documents.
        adaptive_batches: Adaptive batches, with a Bradley-Terry refit after each.
        adaptive_windows_per_batch: Adaptive windows per batch.
        overlap_discount: Factor applied to a boundary's value once a selected window covers it.
        seed: Seed of the window draws (each query draws from its own stream).
        prompt: A prompt name or file; ``None`` uses the shipped tournament prompt for the corpus's modality.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    window: int = Field(default=10, ge=2)
    random_windows: int = Field(default=53, ge=0)
    stratified_windows: int = Field(default=27, ge=0)
    mirror: bool = True
    adaptive_window: int = Field(default=10, ge=2)
    adaptive_depth: int = Field(default=150, ge=2)
    adaptive_batches: int = Field(default=7, ge=0)
    adaptive_windows_per_batch: int = Field(default=8, ge=0)
    overlap_discount: float = Field(default=OVERLAP_DISCOUNT, ge=0, le=1)
    seed: int = 42
    prompt: str | None = None

    @property
    def adaptive_calls(self) -> int:
        """Adaptive calls per query at most: ``adaptive_batches * adaptive_windows_per_batch`` (56 in the paper)."""
        return self.adaptive_batches * self.adaptive_windows_per_batch

    def calls_per_query(self, n_docs: int) -> int:
        """The most judge calls one query of ``n_docs`` candidates takes.

        A pool no larger than a window is one stratified window and one adaptive
        window per batch; the adaptive phase can also stop early when no
        informative boundary is left, so this is an upper bound.
        """
        if n_docs < 2:
            return 0
        stratified = self.stratified_windows if n_docs > self.window else min(self.stratified_windows, 1)
        adaptive = self.adaptive_calls if n_docs > self.adaptive_window else self.adaptive_batches
        return (self.random_windows + stratified) * (2 if self.mirror else 1) + adaptive

    @classmethod
    def for_modality(cls, modality: Modality) -> TournamentSchedule:
        """The shipped schedule for a corpus of text, page images or videos.

        Images and videos cost many tokens per document, so their windows are
        smaller and there are more of them, keeping judgements per document close
        to the text schedule's.
        """
        if modality == "text":
            return cls()
        return cls(
            window=5,
            random_windows=106,
            stratified_windows=54,
            adaptive_window=5,
            adaptive_batches=7,
            adaptive_windows_per_batch=17,
        )


class RubricSchedule(BaseModel):
    """The Stage B schedule; the defaults are the paper's.

    Attributes:
        window: Documents per window.
        windows_per_query: Windows per query (raised to ``ceil(n / window)`` so every document, or every
            chunk of a chunked document, is seen).
        random_windows: How many of them are balanced random windows; the rest are stratified.
        seed: Seed of the window draws (each query draws from its own stream).
        prompt: A prompt name or file; ``None`` uses the shipped rubric prompt (criteria C1..C5) for
            the corpus's modality.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    window: int = Field(default=10, ge=1)
    windows_per_query: int = Field(default=100, ge=1)
    random_windows: int = Field(default=50, ge=1)
    seed: int = 42
    prompt: str | None = None

    @model_validator(mode="after")
    def _random_within_total(self) -> Self:
        if self.random_windows > self.windows_per_query:
            raise ValueError(
                f"random_windows ({self.random_windows}) cannot exceed windows_per_query ({self.windows_per_query})"
            )
        return self

    def windows_for(self, n_docs: int, *, pool_size: int | None = None, n_units: int | None = None) -> tuple[int, int]:
        """``(random, stratified)`` windows for ``n_docs`` documents.

        ``pool_size`` is the query's full candidate pool when only ``n_docs`` of
        it are judged (a re-judged subset): the window count then scales with the
        share of the pool, so each document gets the placements the full schedule
        would give it. ``n_units`` is what the windows show when documents are
        judged in chunks (default ``n_docs``): at least ``ceil(n_units / window)``
        windows are asked, so every chunk is seen. The random share of the full
        schedule is kept.
        """
        if n_docs < 1:
            return (0, 0)
        units = n_docs if n_units is None else n_units
        w = min(self.window, units)
        base = self.windows_per_query
        if pool_size is not None and pool_size > n_docs:
            base = max(1, round(self.windows_per_query * n_docs / pool_size))
        total = max(base, math.ceil(units / w))
        n_random = max(1, int(total * self.random_windows / self.windows_per_query))
        return (n_random, total - n_random)

    def calls_per_query(self, n_docs: int, *, pool_size: int | None = None, n_units: int | None = None) -> int:
        """Judge calls one query takes (a pool no larger than a window has one stratified window)."""
        n_random, n_stratified = self.windows_for(n_docs, pool_size=pool_size, n_units=n_units)
        units = n_docs if n_units is None else n_units
        return n_random + (n_stratified if units > self.window else min(n_stratified, 1))

    @classmethod
    def for_modality(cls, modality: Modality) -> RubricSchedule:
        """The shipped schedule for text, page images or videos (see :meth:`TournamentSchedule.for_modality`)."""
        if modality == "text":
            return cls()
        if modality == "image":
            return cls(window=8, windows_per_query=125, random_windows=62)
        return cls(window=5, windows_per_query=200, random_windows=100)


def schedule_for(stage: Literal["tournament", "rubric"], modality: Modality) -> TournamentSchedule | RubricSchedule:
    """The shipped schedule of ``stage`` for a corpus of ``modality``: what a pass without a schedule runs."""
    return TournamentSchedule.for_modality(modality) if stage == "tournament" else RubricSchedule.for_modality(modality)


# ---------------------------------------------------------------------------
# Window selection (the tournament's numerics; the rubric reuses the first two)
# ---------------------------------------------------------------------------


def _canonical_pair(a: str, b: str) -> tuple[str, str]:
    """A pair of ids in a stable canonical order."""
    return (a, b) if a <= b else (b, a)


def _balanced_groups(n: int, window_size: int, num_groups: int, rng: random.Random) -> list[list[int]]:
    """Balanced random groups of indices with good pair coverage."""
    if window_size >= n:
        return [list(range(n)) for _ in range(num_groups)]

    indices = list(range(n))
    counts: dict[int, int] = {i: 0 for i in indices}
    groups: list[list[int]] = []

    for _ in range(num_groups):
        sorted_by_count = sorted(indices, key=lambda i: (counts[i], rng.random()))
        group = sorted_by_count[:window_size]
        rng.shuffle(group)  # randomise presentation order to reduce position bias
        for i in group:
            counts[i] += 1
        groups.append(group)

    return groups


def _stratified_groups(
    doc_ids: list[str], theta: dict[str, float], window_size: int, num_groups: int, rng: random.Random
) -> list[list[str]]:
    """Groups of documents that share a tier of ``theta``.

    Documents are sorted by theta and each group is the contiguous window with
    the lowest coverage so far, so every document is observed about equally
    often; presentation order within a group is shuffled. Judging peers
    together avoids the contrast effect of a dominant document in a mixed window.
    """
    n = len(doc_ids)
    sorted_ids = sorted(doc_ids, key=lambda d: theta.get(d, 0.0), reverse=True)

    if window_size >= n:
        group = list(sorted_ids)
        rng.shuffle(group)
        return [group] * min(num_groups, 1)

    counts = [0] * n
    groups: list[list[str]] = []

    for _ in range(num_groups):
        best_start = 0
        best_score = float("inf")
        for s in range(n - window_size + 1):
            score = sum(counts[s : s + window_size])
            if score < best_score or (score == best_score and rng.random() < 0.3):
                best_score = score
                best_start = s

        group = list(sorted_ids[best_start : best_start + window_size])
        rng.shuffle(group)  # randomise presentation order
        for i in range(best_start, best_start + window_size):
            counts[i] += 1
        groups.append(group)

    return groups


def _compute_boundary_values(
    theta: dict[str, float], obs_counts: Counter[tuple[str, str]], top_k: int
) -> list[tuple[float, str, str]]:
    """Value of each adjacent boundary in the theta-sorted order (top ``top_k`` documents only).

    ``value = fisher(p) * ndcg_weight(position) * novelty(observations of the pair)``;
    returned as ``(value, doc_above, doc_below)`` in rank order.
    """
    sorted_ids = sorted(theta, key=lambda d: theta[d], reverse=True)[:top_k]
    values: list[tuple[float, str, str]] = []

    for pos in range(len(sorted_ids) - 1):
        a, b = sorted_ids[pos], sorted_ids[pos + 1]
        diff = theta[a] - theta[b]
        p_ab = 1.0 / (1.0 + math.exp(-diff))

        fisher = p_ab * (1.0 - p_ab)  # max at p=0.5
        ndcg_weight = discount(pos + 1)  # top positions matter more
        novelty = 1.0 / (1.0 + obs_counts[_canonical_pair(a, b)])

        values.append((fisher * ndcg_weight * novelty, a, b))

    return values


def _greedy_select_windows(
    theta: dict[str, float],
    boundary_values: list[tuple[float, str, str]],
    window_size: int,
    num_windows: int,
    overlap_discount: float,
) -> list[list[str]]:
    """Greedily select contiguous theta-sorted windows of the highest total boundary value.

    After a window is selected, the boundaries it covers are multiplied by
    ``overlap_discount`` so the next window prefers new ground.
    """
    sorted_ids = sorted(theta, key=lambda d: theta[d], reverse=True)
    n = len(sorted_ids)
    if window_size >= n:
        return [sorted_ids[:]] * min(num_windows, 1)

    id_to_pos = {d: i for i, d in enumerate(sorted_ids)}

    v = [0.0] * (n - 1)
    for val, a, _b in boundary_values:
        pos = id_to_pos.get(a)
        if pos is not None and pos < n - 1:
            v[pos] = val

    windows: list[list[str]] = []

    for _ in range(num_windows):
        best_start = 0
        best_score = -1.0
        for s in range(n - window_size + 1):
            score = sum(v[s : s + window_size - 1])
            if score > best_score:
                best_score = score
                best_start = s

        if best_score <= 0:
            break

        windows.append(sorted_ids[best_start : best_start + window_size])

        for p in range(best_start, best_start + window_size - 1):
            v[p] *= overlap_discount

    return windows


def query_rng(seed: int, dataset: str, query_id: str) -> random.Random:
    """The window-draw stream of one query: deterministic, and independent of other queries and of order."""
    return random.Random(f"{seed}/{dataset}/{query_id}")


__all__ = ["OVERLAP_DISCOUNT", "Modality", "RubricSchedule", "TournamentSchedule", "query_rng", "schedule_for"]
