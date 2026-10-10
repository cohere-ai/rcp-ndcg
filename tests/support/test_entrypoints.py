"""The one entry-point registry: the listing, the ambiguity refusal and the load check.

The readers/writers, the result sinks and the job runners all resolve through this module, so the tests here
are the seam's contract; the per-seam tests (``tests/data/test_io_contract.py``, ``tests/results``,
``tests/runners``) exercise it through their own groups.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from rcp_ndcg.errors import ConfigError
from rcp_ndcg.support.entrypoints import load_entry_point, provider_of, registered_names


@dataclass(frozen=True)
class FakeEntryPoint:
    """An entry point with the four fields the registry reads (the real one needs a distribution to load)."""

    name: str
    value: str
    group: str = "rcp_ndcg.test"
    distribution: str | None = "one-dist"
    loaded: Any = None
    failure: Exception | None = None

    @property
    def dist(self) -> Any:
        return None if self.distribution is None else _Dist(self.distribution)

    def load(self) -> Any:
        if self.failure is not None:
            raise self.failure
        return self.loaded


@dataclass(frozen=True)
class _Dist:
    name: str


class Base:
    """The base class a plugin of the seam must subclass."""


class Impl(Base):
    pass


def test_registered_names_lists_each_name_once_sorted() -> None:
    entries = [FakeEntryPoint("b", "m:B"), FakeEntryPoint("a", "m:A"), FakeEntryPoint("a", "m:A2")]
    assert registered_names(entries, "rcp_ndcg.test") == ("a", "b")


def test_provider_of_returns_the_entry_point_of_a_sole_provider() -> None:
    entry = FakeEntryPoint("a", "m:A")
    assert provider_of([entry], "rcp_ndcg.test", "a", kind="thing") is entry


def test_two_entry_points_of_one_distribution_and_target_are_one_provider() -> None:
    """A distribution declaring the same name twice with the same target is not an ambiguity (the entry
    point metadata repeats); two distributions, or two targets, are."""
    entries = [FakeEntryPoint("a", "m:A"), FakeEntryPoint("a", "m:A")]
    assert provider_of(entries, "rcp_ndcg.test", "a", kind="thing") is entries[0]


def test_an_ambiguous_name_is_refused_naming_every_provider() -> None:
    """The one duplicate-name policy: a name two distributions provide is refused, never silently resolved
    to the last one (a plugin publishing a built-in's name would replace it unseen)."""
    entries = [
        FakeEntryPoint("a", "m:A", distribution="built-in"),
        FakeEntryPoint("a", "other:A", distribution="plugin"),
    ]
    with pytest.raises(ConfigError) as raised:
        provider_of(entries, "rcp_ndcg.test", "a", kind="runner")
    message = str(raised.value)
    assert "the runner name 'a' is provided by more than one installed distribution" in message
    assert "built-in (m:A)" in message and "plugin (other:A)" in message


def test_a_name_that_is_not_registered_is_refused_naming_the_group() -> None:
    with pytest.raises(ConfigError, match="no dataset format named 'nope' is installed"):
        provider_of([], "rcp_ndcg.readers", "nope", kind="dataset format")


def test_load_entry_point_checks_the_base_class_and_names_a_broken_plugin() -> None:
    assert load_entry_point(FakeEntryPoint("a", "m:Impl", loaded=Impl), Base, kind="thing") is Impl
    with pytest.raises(ConfigError, match="names .*not a Base"):
        load_entry_point(FakeEntryPoint("a", "m:Base", loaded=Base()), Base, kind="thing")
    with pytest.raises(ConfigError, match="could not be loaded: ImportError: no module"):
        load_entry_point(FakeEntryPoint("a", "m:X", failure=ImportError("no module")), Base, kind="thing")
