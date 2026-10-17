"""The harness step's wall-clock budget watch (GPU-E1: one stuck request held a node for hours).

The watch is the runner's step budget made enforceable where the requests happen: the capturing
transport, the bare probes and the reference subprocess all consult it, so an overrunning step fails
with the request that was in flight -- nothing waits silently.
"""

from __future__ import annotations

import threading
import time

import pytest
from rcp_ndcg_test.stepwatch import StepBudgetExceeded, StepWatch, current_watch, watched


def test_the_watch_names_the_in_flight_request_on_overrun() -> None:
    """The overrun error is `step <name> exceeded <budget>s; in flight: <method path, request index>`."""
    watch = StepWatch("equivalence", 0.05)
    watch.begin("POST", "/rerank")
    time.sleep(0.08)
    with pytest.raises(
        StepBudgetExceeded, match=r"step equivalence exceeded 0s; in flight: POST /rerank \(request 1\)"
    ):
        watch.check()


def test_the_watch_tracks_the_request_index_and_clears_between_requests() -> None:
    watch = StepWatch("record", 60.0)
    watch.begin("POST", "/v1/embeddings")
    watch.end()
    watch.begin("POST", "/v1/embeddings")
    try:
        assert "(request 2)" in watch.describe()
    finally:
        watch.end()
    assert watch.describe() == "between requests"
    assert watch.requests == 2


def test_a_step_over_budget_fails_at_the_next_request_begin() -> None:
    """The cooperative enforcement point: a request begun after the budget trips raises, naming the step."""
    watch = StepWatch("corpus", 0.05)
    watch.begin("GET", "/v1/models")  # begun within the budget: fine
    watch.end()
    time.sleep(0.06)
    with pytest.raises(StepBudgetExceeded, match="step corpus exceeded 0s"):
        watch.begin("POST", "/rerank")


def test_the_watch_is_a_context_local_of_the_thread_that_sets_it() -> None:
    """The watch follows the step's thread (one worker per recipe): other threads see none."""
    watch = StepWatch("smoke", 60.0)
    seen: list[StepWatch | None] = []

    def read() -> None:
        seen.append(current_watch())

    with watched(watch):
        assert current_watch() is watch
        thread = threading.Thread(target=read)
        thread.start()
        thread.join()
    assert seen == [None]
    assert current_watch() is None


def test_remaining_and_elapsed_are_seconds_for_the_reference_subprocess() -> None:
    """The reference subprocess gets the budget's remainder as its own timeout (its overrun fails the
    step, it never hangs past it)."""
    watch = StepWatch("equivalence", 120.0)
    assert 0 < watch.remaining_s() <= 120.0
    time.sleep(0.01)
    assert watch.elapsed_s() > 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
