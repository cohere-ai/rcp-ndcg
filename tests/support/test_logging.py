"""Tests for shared logger naming and configuration."""

import pytest

from rcp_ndcg.errors import ConfigError
from rcp_ndcg.support.logging import configure_logging, get_logger


def test_get_logger_does_not_duplicate_package_prefix() -> None:
    assert get_logger("rcp_ndcg.calibration.calibrate").name == "rcp_ndcg.calibration.calibrate"


def test_get_logger_prefixes_short_names() -> None:
    assert get_logger("worker").name == "rcp_ndcg.worker"


def test_an_unknown_level_is_a_config_error_naming_the_known_ones() -> None:
    """paths.log_level() raises ConfigError for a bad env var; the direct call raised a bare
    ValueError from three frames inside logging -- two entry points, one typed, one not."""
    with pytest.raises(ConfigError, match="SPAM"):
        configure_logging("SPAM")
