"""The network gate's own test: the classification rule, pinned offline.

The rule lives in ``conftest.py`` (one place): a test whose requested fixtures include one of
``NETWORK_FIXTURES`` -- or that carries an explicit ``@pytest.mark.network`` -- is network-gated and, with
``RCP_NDCG_NETWORK_TESTS`` unset, skipped; an explicit ``@pytest.mark.offline`` opts a test out. These tests
drive the collection hook with a stand-in item, so the rule's behaviour (the marker, the skip, both
overrides, the declared set) is pinned without any Hub access.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.recipes import conftest

#: A module-level write to the hub client's offline flag: huggingface_hub reads it once, so the write leaks
#: into every other test module a worker imports (the gate runs this whole directory with ``-n 4``), turning
#: a sibling's Hub reads into offline-mode failures. A module that needs the offline behaviour passes
#: ``local_files_only`` to its own call instead. Every write spelling counts -- ``setdefault``, ``update``
#: (keyword, dict literal, key not first, ``dict(...)``), ``putenv``, plain ``os.putenv`` and subscript
#: assignment; a read (``os.environ.get`` or a subscript comparison) does not.
_HUB_FLAG_WRITE = re.compile(
    r"os\.environ(?:\.setdefault|\.putenv)?\(\s*[\"']?HF_HUB_OFFLINE"
    r"|os\.environ\.update\([^)]*HF_HUB_OFFLINE"
    r"|(?:\bos\.)?environ\[\s*[\"']HF_HUB_OFFLINE[\"']\s*\]\s*=(?!=)"
    r"|os\.putenv\(\s*[\"']HF_HUB_OFFLINE"
)


def test_no_recipe_test_module_flips_a_process_wide_hub_flag() -> None:
    """No recipe test module writes ``HF_HUB_OFFLINE`` at import time: the flag is process-wide, so it would
    decide every sibling module's Hub reads in the same pytest worker."""
    offenders = sorted(
        path.name
        for path in Path(__file__).parent.glob("test_*.py")
        if _HUB_FLAG_WRITE.search(path.read_text(encoding="utf-8"))
    )
    assert offenders == [], (
        "these recipe test modules write HF_HUB_OFFLINE at import time, leaking the offline mode into every "
        f"sibling test in the same worker: {offenders}"
    )


class _Item:
    """A stand-in collection item: the three attributes the hook reads."""

    def __init__(self, *, fixturenames: tuple[str, ...] = (), markers: tuple[object, ...] = ()) -> None:
        self.path = Path(conftest.__file__).parent / "test_network_gate.py"
        self.fixturenames = list(fixturenames)
        self.markers = list(markers)
        self.added: list[pytest.MarkDecorator] = []

    def get_closest_marker(self, name: str) -> pytest.MarkDecorator | None:
        return next((marker for marker in self.markers if marker.name == name), None)  # type: ignore[attr-defined]

    def add_marker(self, marker: pytest.MarkDecorator) -> None:
        self.added.append(marker)


def _gated(item: _Item, monkeypatch: pytest.MonkeyPatch, *, online: bool) -> _Item:
    """Run the hook over one stand-in item, with the network variable set or unset."""
    if online:
        monkeypatch.setenv(conftest.VARIABLE, "1")
    else:
        monkeypatch.delenv(conftest.VARIABLE, raising=False)
    conftest.pytest_collection_modifyitems(None, [item])
    return item


def test_the_rule_names_every_hub_backed_fixture() -> None:
    """The declared set is pinned: a renamed fixture must fail here instead of silently running offline."""
    assert conftest.NETWORK_FIXTURES == frozenset(
        {
            "tokenizer",
            "tokenizer_dir",
            "zerank_tokenizer",
            "chat_template",
            "_seed_checkpoint_template",
            "recipe_cpu",
            "checkpoint",
            "snapshot",
            "recipe",
            "pairs_path",
        }
    )


def test_an_item_without_a_hub_fixture_is_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """The offline pins (contract, mutants, goldens) get no marker and no skip."""
    item = _gated(_Item(fixturenames=("tmp_path", "variant_id")), monkeypatch, online=False)
    assert item.added == []


def test_a_hub_fixture_marks_and_skips_an_offline_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Hub-backed fixture carries the network marker and, with the variable unset, the skip naming it."""
    item = _gated(_Item(fixturenames=("tmp_path", "tokenizer")), monkeypatch, online=False)
    assert [marker.name for marker in item.added] == ["network", "skip"]
    assert conftest.VARIABLE in str(item.added[1].kwargs["reason"])


def test_a_hub_fixture_runs_when_the_variable_is_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the variable set the item is marked network but never skipped."""
    item = _gated(_Item(fixturenames=("tokenizer",)), monkeypatch, online=True)
    assert [marker.name for marker in item.added] == ["network"]


def test_an_explicit_network_marker_gates_a_fixture_less_item(monkeypatch: pytest.MonkeyPatch) -> None:
    """A test that downloads inside its body (no Hub fixture) gates itself with the marker."""
    item = _gated(_Item(markers=(pytest.mark.network,)), monkeypatch, online=False)
    assert [marker.name for marker in item.added] == ["network", "skip"]


def test_an_explicit_offline_marker_wins_over_a_hub_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    """The override in the other direction: a Hub fixture whose test needs no download stays offline."""
    item = _gated(_Item(fixturenames=("tokenizer",), markers=(pytest.mark.offline,)), monkeypatch, online=False)
    assert item.added == []
