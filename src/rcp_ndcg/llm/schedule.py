"""Window schedules of the two judging stages, with the paper's settings as defaults.

:class:`TournamentSchedule` (Stage A) and :class:`RubricSchedule` (Stage B) are
the single source of the judging schedule: the judging functions, the cost
estimate and the run config all read these models, and nothing redeclares their
numbers.

A schedule is specified in **placements per document**: how often, on average, a
document of the pool is shown to the judge in a phase. A query's window counts
follow from its pool of ``n`` documents, ``round(placements * n / w)``, so the
calls scale with the pool, and with a subset judged on its own (``docs=``). The
effective window ``w = min(window, n)`` is what a window of the pool can hold, so
the placements per document hold for any pool: a pool of 4 gets
``round(3.53) = 4`` random windows of all 4 documents, each in its own order.

**Tournament** (per query, over the candidate pool of ``n`` documents):

1. *Random windows* -- ``round(random_placements * n / w)`` balanced random
   groups of ``w`` documents (every document drawn equally often), each also
   sent reversed when ``mirror`` is set, to cancel position bias;
2. *Stratified windows* -- ``round(stratified_placements * n / w)`` groups of
   documents that share a tier of the preliminary Bradley-Terry ability, again mirrored;
3. *Adaptive windows* -- ``adaptive_batches`` batches of
   ``round(adaptive_placements * n / (w_a * adaptive_batches))`` contiguous
   windows of ``w_a = min(adaptive_window, n)`` documents over the top
   ``adaptive_depth`` documents, chosen by the information of their adjacent
   boundaries (Fisher information x nDCG discount x novelty, covered boundaries
   discounted by ``overlap_discount``), with a Bradley-Terry refit between batches.
   A pool no larger than ``adaptive_window`` has one adaptive window per batch:
   every adaptive window would be the whole pool in its current order, so a
   second one in the same batch would repeat the first.

The defaults give the paper's schedule at its pool of 150: 53 + 27 mirrored windows
of 10 (160 calls) and 7 x 8 = 56 adaptive calls, 216 calls per query.

**Rubric** (per query): ``max(ceil(n / w), round(placements_per_doc * n / w))``
windows of ``w = min(window, n)`` documents, so every document is seen;
``round(random_share * windows)`` of them (at least one) balanced random, the rest
stratified by a preliminary Rasch ability. At a pool of 150 the defaults give the paper's 100
windows of 10, 50 of them random.

Rounding is Python's ``round`` (half to even).
"""

from __future__ import annotations

import math
import random
from collections import Counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from rcp_ndcg_core.gain import sigmoid
from rcp_ndcg_core.metric import discount

from rcp_ndcg.support.identity import hash_payload, short

#: The document modalities a prompt is written for.
Modality = Literal["text", "image", "video"]

#: Default share of a covered boundary's value kept when a window over it is selected.
OVERLAP_DISCOUNT = 0.3

#: The paper's pool: the schedules' default placements are its window counts at this pool size.
PAPER_POOL = 150


def _windows(placements: float, n_docs: int, window: int) -> int:
    """``round(placements * n_docs / window)``: the windows that give ``n_docs`` documents ``placements`` each."""
    return round(placements * n_docs / window)


class TournamentSchedule(BaseModel):
    """The Stage A schedule, in placements per document; the defaults are the paper's.

    Attributes:
        window: Documents per random and stratified window.
        random_placements: Placements per document in balanced random windows (53 windows of 10 at a pool of 150).
        stratified_placements: Placements per document in windows over tiers of the preliminary ability (27 windows
            of 10 at a pool of 150).
        mirror: Also send every random and stratified window in reverse order.
        adaptive_window: Documents per adaptive window.
        adaptive_depth: Adaptive windows are drawn from the top ``adaptive_depth`` documents.
        adaptive_batches: Adaptive batches, with a Bradley-Terry refit after each.
        adaptive_placements: Placements per document over all adaptive batches (7 batches of 8 windows of 10 at a
            pool of 150).
        overlap_discount: Factor applied to a boundary's value once a selected window covers it.
        seed: Seed of the window draws (each query draws from its own stream).
        prompt: A prompt name or file; ``None`` uses the shipped tournament prompt for the corpus's modality.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    window: int = Field(default=10, ge=2)
    random_placements: float = Field(default=53 * 10 / PAPER_POOL, ge=0)
    stratified_placements: float = Field(default=27 * 10 / PAPER_POOL, ge=0)
    mirror: bool = True
    adaptive_window: int = Field(default=10, ge=2)
    adaptive_depth: int = Field(default=150, ge=2)
    adaptive_batches: int = Field(default=7, ge=0)
    adaptive_placements: float = Field(default=7 * 8 * 10 / PAPER_POOL, ge=0)
    overlap_discount: float = Field(default=OVERLAP_DISCOUNT, ge=0, le=1)
    seed: int = 42
    prompt: str | None = None

    def windows_for(self, n_docs: int) -> tuple[int, int, int]:
        """``(random, stratified, adaptive per batch)`` windows the tournament asks for a pool of ``n_docs``.

        Each count is ``round(placements * n_docs / w)`` with the effective window ``w = min(window, n_docs)``
        (``min(adaptive_window, n_docs)`` for the adaptive phase, whose count is shared by the batches). A pool no
        larger than ``adaptive_window`` has at most one adaptive window per batch. A pool of fewer than two
        documents has no windows.
        """
        if n_docs < 2:
            return (0, 0, 0)
        per_batch = 0
        if self.adaptive_batches:
            adaptive_window = min(self.adaptive_window, n_docs)
            per_batch = _windows(self.adaptive_placements, n_docs, adaptive_window * self.adaptive_batches)
            if n_docs <= self.adaptive_window:
                per_batch = min(per_batch, 1)
        window = min(self.window, n_docs)
        return (
            _windows(self.random_placements, n_docs, window),
            _windows(self.stratified_placements, n_docs, window),
            per_batch,
        )

    def phase_calls(self, n_docs: int) -> tuple[int, int]:
        """``(calls in random and stratified windows, calls in adaptive windows)`` for a pool of ``n_docs``.

        The first are the random and stratified windows (twice when mirrored), the second the adaptive ones, at
        most (see :meth:`windows_for`).
        """
        random_windows, stratified, per_batch = self.windows_for(n_docs)
        return (random_windows + stratified) * (2 if self.mirror else 1), self.adaptive_batches * per_batch

    def calls_per_query(self, n_docs: int) -> int:
        """The most judge calls one query of ``n_docs`` candidates takes (:meth:`phase_calls` summed).

        The adaptive phase can stop early when no informative boundary is left, so this is an upper bound.
        """
        return sum(self.phase_calls(n_docs))

    @classmethod
    def for_modality(cls, modality: Modality) -> TournamentSchedule:
        """The shipped schedule for a corpus of text, page images or videos.

        Images and videos cost many tokens per document, so their windows hold 5
        documents instead of 10; the placements per document are the text
        schedule's, so there are twice as many windows (106 random and 54
        stratified at a pool of 150, and 16 adaptive windows per batch).
        """
        if modality == "text":
            return cls()
        return cls(window=5, adaptive_window=5)


class RubricSchedule(BaseModel):
    """The Stage B schedule, in placements per document; the defaults are the paper's.

    Attributes:
        window: Documents per window.
        placements_per_doc: Placements per document (100 windows of 10 at a pool of 150). The windows are raised to
            ``ceil(n / window)`` so every document, or every chunk of a chunked document, is seen.
        random_share: The share of the windows that are balanced random (at least one); the rest are stratified.
        seed: Seed of the window draws (each query draws from its own stream).
        prompt: A prompt name or file; ``None`` uses the shipped rubric prompt (criteria C1..C5) for
            the corpus's modality.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    window: int = Field(default=10, ge=1)
    placements_per_doc: float = Field(default=100 * 10 / PAPER_POOL, gt=0)
    random_share: float = Field(default=0.5, gt=0, le=1)
    seed: int = 42
    prompt: str | None = None

    def windows_for(self, n_docs: int, *, n_units: int | None = None) -> tuple[int, int]:
        """``(random, stratified)`` windows the rubric asks for ``n_docs`` documents.

        ``max(ceil(n_units / w), round(placements_per_doc * n_docs / w))`` windows of ``w = min(window, n_units)``,
        of which ``round(random_share * windows)`` (at least one) are random. ``n_units`` is what the windows show
        when documents are judged in chunks (default ``n_docs``), so every chunk is seen.
        """
        if n_docs < 1:
            return (0, 0)
        units = n_docs if n_units is None else n_units
        window = min(self.window, units)
        total = max(math.ceil(units / window), _windows(self.placements_per_doc, n_docs, window))
        n_random = min(total, max(1, round(self.random_share * total)))
        return (n_random, total - n_random)

    def calls_per_query(self, n_docs: int, *, n_units: int | None = None) -> int:
        """Judge calls one query takes: the windows of :meth:`windows_for`."""
        return sum(self.windows_for(n_docs, n_units=n_units))

    @classmethod
    def for_modality(cls, modality: Modality) -> RubricSchedule:
        """The shipped schedule for text, page images or videos: windows of 10, 8 and 5, the same placements."""
        if modality == "text":
            return cls()
        return cls(window=8 if modality == "image" else 5)


def schedule_key(schedule: TournamentSchedule | RubricSchedule) -> str:
    """16-hex digest of a schedule's numbers (its prompt left out: the judgement family holds the prompt's hash).

    Planned windows are keyed by it (:func:`~rcp_ndcg_core.schemas.judgement_record_id`).
    """
    return short(hash_payload(schedule.model_dump(mode="json", exclude={"prompt"})), 16)


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
    """Balanced random groups of indices with good pair coverage.

    A window of at least ``n`` holds the whole pool, each group in its own shuffled order.
    """
    window_size = min(window_size, n)
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
    often; presentation order within a group is shuffled. A window of at least
    ``n`` holds the whole pool, each group in its own shuffled order. Judging peers
    together avoids the contrast effect of a dominant document in a mixed window.
    """
    n = len(doc_ids)
    sorted_ids = sorted(doc_ids, key=lambda d: theta.get(d, 0.0), reverse=True)
    window_size = min(window_size, n)

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
        p_ab = sigmoid(diff)

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
    ``overlap_discount`` so the next window prefers new ground. A window of at
    least the pool is the whole pool in its current order, selected once.
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


__all__ = [
    "OVERLAP_DISCOUNT",
    "PAPER_POOL",
    "Modality",
    "RubricSchedule",
    "TournamentSchedule",
    "query_rng",
    "schedule_for",
    "schedule_key",
]
