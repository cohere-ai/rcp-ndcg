"""The per-test timeout guard of this tree's conftest: a hang fails its own test.

``faulthandler_timeout`` only dumps a traceback, and the gate caller's outer timeout bounds the whole run,
so without this guard a wedged test in this ~10-minute tree stalls until CI's job timeout. The timer is
armed around every phase of a test (setup, call, teardown); this test pins that it is armed, that the
default is a real bound, and that a test sizes its own bound with ``timeout_s`` and nothing else moves.
"""

from __future__ import annotations

import signal
from typing import Any

import pytest

from tests import conftest

#: A bound no tree's default reaches, so a test marked with it can tell the mark from the default.
_OWN_BOUND_S = 10_000.0


def test_the_per_test_timeout_is_armed() -> None:
    """The SIGALRM timer is running inside a test body: removing the guard turns this red."""
    if conftest.TEST_TIMEOUT_S <= 0:
        pytest.skip("the per-test timeout is deliberately off (RCP_NDCG_TEST_TIMEOUT=0)")
    remaining, _ = signal.getitimer(signal.ITIMER_REAL)
    assert remaining > 0, "no per-test timer is armed; a hang would run to the outer timeout"


def test_the_default_timeout_bounds_a_hang() -> None:
    """The default is a positive, finite bound in seconds: a tree whose default is 0 (or negative) has no
    guard at all, and the failure would be silent."""
    assert 0 < conftest._DEFAULT_TIMEOUT_S < 3600


@pytest.mark.timeout_s(_OWN_BOUND_S)
def test_a_tests_own_bound_is_the_armed_one() -> None:
    """``timeout_s`` replaces the tree's bound for the test that marks it: the armed timer is the mark's."""
    if conftest.TEST_TIMEOUT_S <= 0:
        pytest.skip("the per-test timeout is deliberately off (RCP_NDCG_TEST_TIMEOUT=0)")
    remaining, _ = signal.getitimer(signal.ITIMER_REAL)
    assert remaining > conftest.TEST_TIMEOUT_S, f"the mark did not raise the bound: {remaining:g}s armed"


def test_an_unmarked_test_keeps_the_trees_bound(request: pytest.FixtureRequest) -> None:
    """The mark is local: an unmarked test is still timed by the tree's own bound, never by a neighbour's."""
    assert conftest.budget_s(request.node) == conftest.TEST_TIMEOUT_S
    remaining, _ = signal.getitimer(signal.ITIMER_REAL)
    assert 0 < remaining <= conftest.TEST_TIMEOUT_S, f"an unmarked test was armed with {remaining:g}s"


class _Marked:
    """The only part of a test item :func:`tests.conftest.budget_s` reads: the mark the test carries."""

    def __init__(self, *args: object) -> None:
        self._mark = pytest.mark.timeout_s(*args)

    def get_closest_marker(self, name: str) -> Any:
        return self._mark if name == "timeout_s" else None


def test_a_malformed_bound_is_refused_not_ignored() -> None:
    """A mark that names no single positive number is a usage error: a typo must not leave a test unbounded."""
    assert conftest.budget_s(_Marked(180)) == 180.0
    for args in ((0,), (-5.0,), (), (180, 60)):
        with pytest.raises(pytest.UsageError):
            conftest.budget_s(_Marked(*args))
