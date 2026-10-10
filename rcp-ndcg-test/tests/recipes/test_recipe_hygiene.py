"""One repo-wide hygiene guard over every shipped recipe file and template.

A shipped recipe is package data of a public distribution: every line must read as a self-contained public
statement. Internal process shorthand (a lane or sweep name, a private review id), a private work directory or
a local machine path is not a source a reader of the wheel can resolve, and the standing rule forbids them
(the ctxl family's own guard is the pattern; this module is its one repo-wide home, so a family that ships a
new file cannot pass unnoticed). Offline by construction: it reads the package data, nothing else.
"""

from __future__ import annotations

import re

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


#: The per-column media fields the harness retired in favour of one ``media`` field (W2). Every family's
#: ``reference.py`` refuses them, because a row carrying one would otherwise be encoded or scored as text --
#: a different prompt, never compared. The refusal is five copies **by design** (a reference subprocess
#: imports nothing from the product, so it cannot share the product's helper); this module is the one place
#: that names the list, so a sixth retired column cannot be added to four of the five.
RETIRED_MEDIA_COLUMNS = ("query_image", "query_video", "documents_images", "documents_videos")

_COLUMN_TUPLE = re.compile(r"for key in \(([^)]*)\) if row\.get\(key\)")


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
    """Every shipped retrieval recipe file and template reads as a public statement.

    The judge families (merged after this guard's brief) cite the handover judge-catalog spec and their
    lane's report by section; their notes are the judge lane's / docs-final's cleanup when handover is
    deleted, and this guard scopes itself to the retrieval families (the lane that owns it).
    """
    import yaml

    root = default_recipes_root()
    judge_families = {
        family_path.parent
        for family_path in root.glob("*/family.yaml")
        if (yaml.safe_load(family_path.read_text(encoding="utf-8")) or {}).get("role") == "judge"
    }
    hits = [
        f"{path.relative_to(root)}:{number}: {line.strip()[:140]}"
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and "__pycache__" not in path.parts
        and not any(parent in path.parents for parent in judge_families)
        for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1)
        if INTERNAL_LABELS.search(line)
    ]
    assert not hits, "internal labels in shipped recipe files:\n" + "\n".join(hits)
