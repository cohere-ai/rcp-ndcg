"""Markers for the recipe-lane tests: the network mark from the root suite's convention.

The root package registers ``network`` in its own pytest ini; this package (outside the root workspace)
registers its own copy here so the recipe tests can name their Hub dependency without an unknown-mark
warning. The mark is documentation: the guard is the offline skip in the download helper (the recipe
brief mandates run-when-online, skip-when-offline, the inverse of the root suite's env gate).
"""


def pytest_configure(config: "object") -> None:
    """Register the ``network`` marker for the recipe tests."""
    config.addinivalue_line("markers", "network: the test reaches the Hugging Face Hub (skips offline)")  # type: ignore[attr-defined]
