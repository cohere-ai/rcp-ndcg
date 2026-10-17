"""One run step's wall-clock budget, enforced where the requests happen.

The run config's ``step_budget_s`` gives every step a budget (``None`` leaves it unbudgeted). The budget is
checked before each request: the shared transport checks it before a request is queued and after every park,
and the judging pass checks it before a phase's windows. A step over budget stops with
:class:`~rcp_ndcg.errors.StepBudgetExceededError`; what it wrote stays (a judging store's last complete line),
and ``run resume`` continues from there. A request already in flight is not interrupted: it finishes, or hits
the endpoint's ``timeout_s``/``wait_on_outage_s``, and the next check stops the step.

The watch travels as a context-local (:func:`step_budgeted` / :func:`current_step_budget`), so the request
seams read the calling step's budget without every call site growing a parameter; a caller that set none sees
none (the library used directly, the tests).
"""

from __future__ import annotations

import contextlib
import contextvars
import time
from collections.abc import Iterator

from rcp_ndcg.errors import StepBudgetExceededError

__all__ = ["StepBudget", "current_step_budget", "step_budgeted"]


class StepBudget:
    """One step's budget: the elapsed seconds and the overrun check.

    Inputs: the step's name (for the error) and its budget in seconds. Units: seconds (monotonic clock).
    """

    def __init__(self, step: str, budget_s: float) -> None:
        self.step = step
        self.budget_s = float(budget_s)
        self.started = time.monotonic()

    def elapsed_s(self) -> float:
        """Seconds since the step began (monotonic)."""
        return time.monotonic() - self.started

    def remaining_s(self) -> float:
        """The budget's unspent seconds, never negative (a park is bounded by it)."""
        return max(self.budget_s - self.elapsed_s(), 0.0)

    def check(self) -> None:
        """Raise :class:`~rcp_ndcg.errors.StepBudgetExceededError` when the step is over its budget."""
        if self.elapsed_s() > self.budget_s:
            raise StepBudgetExceededError(
                f"step {self.step} exceeded its step_budget_s of {self.budget_s}s",
                hint="raise step_budget_s (or set it to null to leave the step unbudgeted); the store keeps what "
                "the step wrote, and a resume continues from there",
            )


_BUDGET: contextvars.ContextVar[StepBudget | None] = contextvars.ContextVar("rcp_ndcg_step_budget", default=None)


@contextlib.contextmanager
def step_budgeted(budget: StepBudget | None) -> Iterator[None]:
    """Run the block with ``budget`` as the calling context's step budget; restores the previous one."""
    token = _BUDGET.set(budget)
    try:
        yield
    finally:
        _BUDGET.reset(token)


def current_step_budget() -> StepBudget | None:
    """The calling context's step budget, or ``None`` outside a budgeted step."""
    return _BUDGET.get()
