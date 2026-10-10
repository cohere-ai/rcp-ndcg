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
    r"|\bworkstream \d|\bhandover/"
    r"|(?:^|[\s\"'(=])/(?:root|home|Users|private|tmp)/"
)
"""The internal-label and private-path pattern every shipped recipe file is scanned with."""


def test_every_shipped_recipe_file_reads_as_a_public_statement() -> None:
    root = default_recipes_root()
    hits = [
        f"{path.relative_to(root)}:{number}: {line.strip()[:140]}"
        for path in sorted(root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
        for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1)
        if INTERNAL_LABELS.search(line)
    ]
    assert not hits, "internal labels in shipped recipe files:\n" + "\n".join(hits)
