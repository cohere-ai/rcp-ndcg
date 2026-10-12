"""The repo-wide hygiene guards over the shipped tree.

A shipped recipe is package data of a public distribution: every line must read as a self-contained public
statement. Internal process shorthand (a lane or sweep name, a private review id), a private work directory or
a local machine path is not a source a reader of the wheel can resolve, and the standing rule forbids them
(the ctxl family's own guard is the pattern; this module is its one repo-wide home, so a family that ships a
new file cannot pass unnoticed). The same rule covers the rest of the shipped tree: no tracked file outside
the scaffolding (``handover/``, ``experiments/``) cites the internal design document or a lane branch -- the
docs wording test forbids those ids in Markdown and the gate's public-names step forbids operator paths, so
this scan is their one home everywhere else. Offline by construction: it reads the tracked tree, nothing else.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from rcp_ndcg_vllm.recipe import default_recipes_root

INTERNAL_LABELS = re.compile(
    r"p1-tail|fam-(?:dense|ctxl|vl|late|zerank|qwen3)"
    r"|\bsweep|lanes' base|audit-synth"
    r"|\br-(?:ctxl|jina[35]|octen|zembed1|qwen3-emb|qwen3vl-emb|qwen3vl-rer|topk|pplx|zerank[12]?|qwen3-rer)\b"
    r"|\bresearch\b|\blanes?\b|REVIEW-LOG|ANCHOR-FINDING|\bR(?!29\b)\d{1,2}\b|\bG[1-5]\b|clients-final"
    r"|\boperator\b|\b09x\b|\.refs/|recipe-common|corrections table|\bfinding #?\d|shake"
    r"|\bworkstream\b|\brecipe-fix\b|\bhandover/"
    r"|(?:^|[\s\"'(=])/(?:root|home|Users|private|tmp)/"
)
"""The internal-label and private-path pattern every shipped recipe file is scanned with."""

#: The internal design-document and lane-branch citations no shipped file may carry: the design document is
#: operator scaffolding and a lane branch is unknown outside this workspace. Case-insensitive: a lowercased
#: citation of the same design id is the same id. The scan skips :data:`SCAFFOLDING` (see the module docstring).
DESIGN_AND_LANE_IDS = re.compile(r"\bRFC[- ]?\d{3,4}\b|\blane/", re.IGNORECASE)

#: The scaffolding directories the shipped-tree scan skips: working notes and paper reproduction, deleted at
#: release.
SCAFFOLDING = ("handover", "experiments")


#: The per-column media fields the harness retired in favour of one ``media`` field (W2). Every family's
#: ``reference.py`` refuses them, because a row carrying one would otherwise be encoded or scored as text --
#: a different prompt, never compared. The refusal is five copies **by design** (a reference subprocess
#: imports nothing from the product, so it cannot share the product's helper); this module is the one place
#: that names the list, so a sixth retired column cannot be added to four of the five.
RETIRED_MEDIA_COLUMNS = ("query_image", "query_video", "documents_images", "documents_videos")

_COLUMN_TUPLE = re.compile(r"for key in \(([^)]*)\) if row\.get\(key\)")


def _repository_root() -> Path:
    """The checkout root: this module lives three directories below it."""
    return Path(__file__).resolve().parents[3]


def _shipped_files(root: Path) -> list[Path]:
    """Every tracked file under ``root`` except the scaffolding (see :data:`SCAFFOLDING`).

    The tracked tree is what a release ships; an untracked file is not part of it.
    """
    listed = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], check=True, capture_output=True).stdout
    return [
        root / name for name in listed.decode("utf-8").split("\0") if name and name.split("/", 1)[0] not in SCAFFOLDING
    ]


def test_no_shipped_file_cites_an_internal_design_or_lane_id() -> None:
    """No tracked file outside the scaffolding cites the internal design document or a lane branch.

    ``handover/`` and ``experiments/`` are scaffolding, deleted at release; the docs wording test covers
    Markdown and the gate's public-names step operator paths, so this scan is the one home for the rest of the
    shipped tree (the product's code and tests, the test package's cases and goldens).
    """
    root = _repository_root()
    hits = [
        f"{path.relative_to(root)}:{number}: {line.strip()[:140]}"
        for path in _shipped_files(root)
        for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1)
        if DESIGN_AND_LANE_IDS.search(line)
    ]
    assert not hits, "internal design or lane ids in the shipped tree:\n" + "\n".join(hits)


def test_every_reference_refuses_the_same_retired_media_columns() -> None:
    """The five copies of the retired-column refusal name the same columns (see the constant above)."""
    root = default_recipes_root()
    found: dict[str, tuple[str, ...]] = {}
    for path in sorted(root.glob("*/reference.py")):
        text = path.read_text(encoding="utf-8")
        if "_refuse_old_media_columns" not in text:
            continue
        match = _COLUMN_TUPLE.search(text)
        assert match is not None, f"{path.parent.name}: the refusal no longer names its columns in the pinned shape"
        found[path.parent.name] = tuple(re.findall(r'"([a-z_]+)"', match.group(1)))
    assert found, "no family's reference refuses the retired media columns"
    assert len(set(found.values())) == 1, f"the retired-column copies disagree: {found}"
    assert set(next(iter(found.values()))) == set(RETIRED_MEDIA_COLUMNS), found


def test_every_shipped_recipe_file_reads_as_a_public_statement() -> None:
    """Every shipped recipe file and template reads as a public statement, judge families included."""
    root = default_recipes_root()
    hits = [
        f"{path.relative_to(root)}:{number}: {line.strip()[:140]}"
        for path in sorted(root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
        for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1)
        if INTERNAL_LABELS.search(line)
    ]
    assert not hits, "internal labels in shipped recipe files:\n" + "\n".join(hits)
