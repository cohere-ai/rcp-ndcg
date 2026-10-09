"""The network gate itself: an unset variable skips every recipe test, a set one lets them run."""

from __future__ import annotations

import re
from pathlib import Path

#: A module-level write to the hub client's offline flag: huggingface_hub reads it once, so the write leaks
#: into every other test module a worker imports (the gate runs this whole directory with ``-n 4``), turning
#: a sibling's Hub reads into offline-mode failures. A module that needs the offline behaviour passes
#: ``local_files_only`` to its own call instead. Every write form counts -- ``setdefault``, ``update``
#: (keyword or dict literal), ``putenv``, plain ``os.putenv`` and subscript assignment; a read
#: (``os.environ.get``) does not.
_HUB_FLAG_WRITE = re.compile(
    r"os\.environ(?:\.setdefault|\.update|\.putenv)?\(\s*[\{(]?\s*(?:dict\()?\s*[\"']?HF_HUB_OFFLINE"
    r"|os\.environ\[\s*[\"']HF_HUB_OFFLINE[\"']\s*\]\s*="
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


def test_the_recipe_gate_marks_this_test_network() -> None:
    """Collected under tests/recipes/, this test carries the network marker and the offline skip."""
    from tests.recipes.conftest import VARIABLE

    assert VARIABLE == "RCP_NDCG_NETWORK_TESTS"


def test_a_recipe_test_runs_only_when_the_variable_is_set() -> None:
    """A stand-in recipe test: the same gating every recipe lane's test gets."""
    import os

    assert os.environ.get("RCP_NDCG_NETWORK_TESTS") in ("1", None), "the gate decides; the test just runs"
