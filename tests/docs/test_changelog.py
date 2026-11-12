"""The release notes agree with the shipped recipes, name modules that exist, and carry each change once.

The 0.0.1 entry is the release record. A reader deciding how over-cap rows were gated, or how empty
documents score, takes the recipe facts from its bullets, so every ``status``/``known_deviations``/
``empty_doc`` claim a bullet makes about a shipped recipe is resolved against the recipe here. The same
entry names the harness modules; they moved to the unpublished ``rcp-test`` at the layout move, so the
entry must name their shipped homes (``rcp_test.*``, ``packages/rcp-test/schema/``) while the modules
themselves live on. Above its ``### Changed — the rename`` section the entry describes the tree that
ships, so it names the shipped spellings there; below that section the entries keep the names of the day
they were written, and that history is not checked.
"""

from __future__ import annotations

import importlib
import re
from collections import Counter

import pytest
from rcp_vllm.recipe import Recipe, iter_recipes

from tests.docs._markdown import ROOT

CHANGELOG = ROOT / "CHANGELOG.md"

_BULLET = re.compile(r"(?m)^(?=- )")
_ID = re.compile(r"`([a-z0-9][a-z0-9.-]*)`|recipes/([a-z0-9][a-z0-9.-]*)/")
_UNVERIFIED = re.compile(r"(?i)(?:status|state)\s*:?\s*`?unverified`?")
_KNOWN_DEVIATIONS = re.compile(r"known_deviations:?\s*\[([^\]]*)\]")
_EMPTY_DOC = re.compile(r"empty_doc:?\s*`?([a-z_]+)`?")


@pytest.fixture(scope="module")
def recipes() -> dict[str, Recipe]:
    """Every shipped recipe, by id."""
    return {recipe.id: recipe for recipe in iter_recipes()}


def _bullets() -> list[str]:
    return [bullet for bullet in _BULLET.split(CHANGELOG.read_text(encoding="utf-8")) if bullet.startswith("- ")]


def _named_recipes(bullet: str, recipes: dict[str, Recipe]) -> list[tuple[int, str]]:
    """The shipped recipe ids a bullet names, with the offset each name appears at."""
    named: list[tuple[int, str]] = []
    for match in _ID.finditer(bullet):
        recipe_id = match.group(1) or match.group(2)
        if recipe_id in recipes:
            named.append((match.start(), recipe_id))
    return named


def _subject(bullet: str, offset: int, recipes: dict[str, Recipe]) -> str | None:
    """The recipe a claim at ``offset`` is about: the nearest shipped recipe named before it."""
    before = [recipe_id for start, recipe_id in _named_recipes(bullet, recipes) if start < offset]
    return before[-1] if before else None


def test_no_bullet_claims_a_shipped_recipe_ships_unverified(recipes: dict[str, Recipe]) -> None:
    """`unverified` does not ship in a tag: a bullet that says so about a shipped recipe is stale."""
    stale = [
        f"{recipe_id} (ships {recipes[recipe_id].status.state}): {bullet.splitlines()[0]}"
        for bullet in _bullets()
        if _UNVERIFIED.search(bullet)
        for _, recipe_id in _named_recipes(bullet, recipes)
        if recipes[recipe_id].status.state != "unverified"
    ]
    assert stale == [], f"CHANGELOG.md claims shipped recipes ship unverified: {stale}"


def test_known_deviations_claims_match_the_shipped_recipes(recipes: dict[str, Recipe]) -> None:
    """A bullet's `known_deviations: [...]` about a shipped recipe is the recipe's declared value."""
    stale: list[str] = []
    for bullet in _bullets():
        for match in _KNOWN_DEVIATIONS.finditer(bullet):
            claimed = [value.strip().strip("`") for value in match.group(1).split(",") if value.strip()]
            recipe_id = _subject(bullet, match.start(), recipes)
            if recipe_id is None:
                continue
            recipe = recipes[recipe_id]
            if recipe.reference is None or claimed == list(recipe.reference.known_deviations):
                continue
            declared = list(recipe.reference.known_deviations)
            stale.append(f"{recipe_id}: bullet says {claimed}, recipe declares {declared}")
    assert stale == [], f"CHANGELOG.md known_deviations claims disagree with the recipes: {stale}"


def test_empty_doc_claims_match_the_shipped_recipes(recipes: dict[str, Recipe]) -> None:
    """A bullet's `empty_doc: value` about a shipped recipe is the recipe's declared policy."""
    stale: list[str] = []
    for bullet in _bullets():
        for match in _EMPTY_DOC.finditer(bullet):
            recipe_id = _subject(bullet, match.start(), recipes)
            if recipe_id is None:
                continue
            shipped = recipes[recipe_id].client.get("empty_doc")
            if shipped is not None and match.group(1) != shipped:
                stale.append(f"{recipe_id}: bullet says {match.group(1)!r}, recipe declares {shipped!r}")
    assert stale == [], f"CHANGELOG.md empty_doc claims disagree with the recipes: {stale}"


MOVED_HARNESS = (
    "fingerprint",
    "observe",
    "observe.corpus",
    "observe.media_set",
    "equivalence",
    "equivalence.media",
    "equivalence.stages",
    "equivalence.checkpoint",
    "record",
    "changes",
    "e2e",
)
PRE_MOVE_PREFIXES = ("rcp_vllm.", "rcp.testing.")

RENAME_HEADING = "### Changed — the rename"

# Spellings that shipped with an artifact or a dataset and did not move with the rename: the run layout's
# own tag (not a schema id), the published dataset ids, and the emulator's extension keys.
RECORDED = ("rcp-ndcg.run-layout.v1", "rcp-ndcg-vidore-v3", "x-rcp-ndcg-emulator-")
_MOVED_FROM = re.compile(r"moved from `[^`]*`")
_OLD_SPELLING = re.compile(r"rcp-ndcg(?:-core|-vllm|-test)?(?![\w.-])|rcp_ndcg_(?:core|vllm|test)\b|rcp_ndcg\.")


def _current_notes() -> str:
    """The release notes above the rename section, flattened: the part that describes the tree that ships."""
    head, separator, _ = CHANGELOG.read_text(encoding="utf-8").partition(f"\n{RENAME_HEADING}\n")
    assert separator, f"CHANGELOG.md has no {RENAME_HEADING!r} section"
    return " ".join(head.split())


def test_the_release_notes_above_the_rename_use_the_shipped_names() -> None:
    """0.0.1 is the release record: above the rename section it names `rcp`, not the pre-rename spellings."""
    current = _MOVED_FROM.sub("", _current_notes())
    for spelling in RECORDED:
        current = current.replace(spelling, "")
    stale = sorted(set(_OLD_SPELLING.findall(current)))
    assert stale == [], f"CHANGELOG.md describes the shipped tree with pre-rename names: {stale}"


def test_the_harness_modules_are_named_at_their_shipped_home() -> None:
    """The notes name the harness under `rcp_test` (its home since the layout move), not the old one."""
    current = _current_notes()
    stale: list[str] = []
    for name in MOVED_HARNESS:
        module = f"rcp_test.{name}"
        assert module in current, f"the release notes never name the shipped home {module}"
        importlib.import_module(module)
        for prefix in PRE_MOVE_PREFIXES:
            old = f"{prefix}{name}"
            for match in re.finditer(re.escape(old) + r"(?![\w.])", current):
                if "moved from `" in current[max(0, match.start() - 16) : match.start()]:
                    continue  # the notes state where it moved from; that is history, not a current home
                stale.append(old)
                break
    assert stale == [], f"CHANGELOG.md names moved harness modules at their old home: {stale}"


def test_no_change_appears_twice_in_the_release_notes() -> None:
    bullets = [line for line in CHANGELOG.read_text(encoding="utf-8").splitlines() if line.startswith("- ")]
    repeated = sorted(text for text, count in Counter(bullets).items() if count > 1)
    assert repeated == []


def test_the_text_limits_section_keeps_only_its_prose() -> None:
    lines = CHANGELOG.read_text(encoding="utf-8").splitlines()
    start = next(index for index, line in enumerate(lines) if line.startswith("### Changed from the paper's code"))
    body: list[str] = []
    for line in lines[start + 1 :]:
        if line.startswith("#"):
            break
        body.append(line)
    assert [line for line in body if line.startswith("- ")] == []
