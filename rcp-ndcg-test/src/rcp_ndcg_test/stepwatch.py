"""One harness step's declared wall-clock budget, enforced where the requests happen.

GPU-E1: the wave runner worked through the recipes' steps serially, one stuck engine request held
every other recipe's steps behind it, and the pod log said nothing.  The runner now gives every step a
budget (from the recipe's request count; :attr:`rcp_ndcg_vllm.recipe.EngineSpec.step_budget_s` raises
it) and runs each recipe's steps in a worker of its own; this module is the budget's enforcement seam:

- :class:`StepWatch` tracks the step's elapsed time and the request currently in flight
  (``method path (request <index>)``), and raises :class:`StepBudgetExceeded` at the next request
  begin once the budget is out -- the runner then fails that step with that exact message, stops the
  engine and continues with the other recipes.  Nothing waits silently.
- :func:`watched` / :func:`current_watch` carry the watch to the request seams as a context-local: the
  capturing transport (:mod:`rcp_ndcg_test.equivalence.wire`), the recorder's bare probes and the
  reference subprocess read the thread's watch without every call site growing a parameter.  A thread
  that did not set a watch sees none (the tests, the equivalence CLI on its own).
"""

from __future__ import annotations

import contextlib
import contextvars
import threading
import time
from collections.abc import Iterator

__all__ = ["StepBudgetExceeded", "StepWatch", "current_watch", "watched"]


class StepBudgetExceeded(RuntimeError):
    """A harness step outlived its declared wall-clock budget; the message names the step, the budget
    and the request that was in flight."""


class StepWatch:
    """One step's budget: elapsed seconds, the in-flight request, the overrun check.

    Inputs: the step's name (for the error) and its budget in seconds.  The transport and the bare
    probes call :meth:`begin`/:meth:`end` around each request; :meth:`check` raises
    :class:`StepBudgetExceeded` when the budget is out, with ``step <name> exceeded <budget>s; in
    flight: <method path (request <index>)>``.  The runner also calls :meth:`remaining_s` to bound a
    subprocess (the reference) by the budget's remainder.  Units: seconds (monotonic clock).
    """

    def __init__(self, step: str, budget_s: float) -> None:
        self.step = step
        self.budget_s = float(budget_s)
        self.requests = 0
        self.started = time.monotonic()
        self._in_flight = ""
        self._lock = threading.Lock()

    def elapsed_s(self) -> float:
        """Seconds since the step began (monotonic)."""
        return time.monotonic() - self.started

    def remaining_s(self) -> float:
        """The budget's unspent seconds (never negative): a subprocess's own timeout."""
        return max(self.budget_s - self.elapsed_s(), 0.0)

    def begin(self, method: str, path: str) -> None:
        """One request begins: it becomes the in-flight one (its 1-based index within the step), and an
        already-over budget raises before it is sent."""
        with self._lock:
            self.requests += 1
            self._in_flight = f"{method} {path} (request {self.requests})"
        self.check()

    def end(self) -> None:
        """The request is done; between requests the step still checks nothing (the caller does)."""
        with self._lock:
            self._in_flight = ""

    def check(self) -> None:
        """Raise :class:`StepBudgetExceeded` when the step is over budget, naming the in-flight request."""
        if self.elapsed_s() > self.budget_s:
            raise StepBudgetExceeded(f"step {self.step} exceeded {self.budget_s:.0f}s; in flight: {self.describe()}")

    def describe(self) -> str:
        """The in-flight request, or ``between requests`` when the step is between two of them."""
        with self._lock:
            return self._in_flight or "between requests"


_WATCH: contextvars.ContextVar[StepWatch | None] = contextvars.ContextVar("rcp_ndcg_test_step_watch", default=None)


@contextlib.contextmanager
def watched(watch: StepWatch | None) -> Iterator[None]:
    """Run the block with ``watch`` as the thread's step watch (the request seams read it); restores the
    previous one afterwards.  A ``None`` watch clears it."""
    token = _WATCH.set(watch)
    try:
        yield
    finally:
        _WATCH.reset(token)


def current_watch() -> StepWatch | None:
    """The calling thread's step watch, or ``None`` outside a watched step."""
    return _WATCH.get()
