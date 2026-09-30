"""Tests for shared logger naming."""

from rcp_ndcg.support.logging import get_logger


def test_get_logger_does_not_duplicate_package_prefix() -> None:
    assert get_logger("rcp_ndcg.calibration.calibrate").name == "rcp_ndcg.calibration.calibrate"


def test_get_logger_prefixes_short_names() -> None:
    assert get_logger("worker").name == "rcp_ndcg.worker"
