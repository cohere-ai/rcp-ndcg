"""The per-test timeout guard of this tree's conftest: a hang fails its own test.

``faulthandler_timeout`` only dumps a traceback, and the gate caller's outer timeout bounds the whole run,
so without this guard a wedged test in this ~10-minute tree stalls until CI's job timeout. The timer is
armed around every phase of a test (setup, call, teardown); this test pins that it is armed and that the
default is a real bound.
"""

from __future__ import annotations

import signal

import pytest

from tests import conftest


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
