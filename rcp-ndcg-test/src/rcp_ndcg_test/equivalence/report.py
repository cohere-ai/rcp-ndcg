"""The harness report: ``equivalence.json`` (every number with its referent) and a short ``EQUIVALENCE.md``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = ["write_report"]

_SECTIONS = {"stage1": "Stage 1 - prompts", "stage2": "Stage 2 - scores and vectors", "stage3": "Stage 3 - metrics"}


def write_report(out_dir: str | Path, document: dict[str, Any]) -> tuple[Path, Path]:
    """Write ``equivalence.json`` and ``EQUIVALENCE.md`` under ``out_dir``; return both paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "equivalence.json"
    md_path = out / "EQUIVALENCE.md"
    json_path.write_text(json.dumps(document, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(document), encoding="utf-8")
    return json_path, md_path


def _markdown(document: dict[str, Any]) -> str:
    """The short EQUIVALENCE.md section: header, one table per stage and the verdict."""
    lines = [
        "# Equivalence report",
        "",
        f"- recipe: `{document.get('recipe', '?')}`",
        f"- engine image: `{document.get('image', '?')}`",
        f"- base URL: `{document.get('base_url', 'http://engine')}` (replaced; no hostname is recorded)",
        f"- pairs: {document.get('pairs', '?')}",
        "",
    ]
    for stage, title in _SECTIONS.items():
        body = document.get(stage)
        if not isinstance(body, dict):
            continue
        lines.append(f"## {title}")
        lines.append("")
        if stage == "stage1":
            lines.append(f"- checked pairs: {body.get('checked')}/{body.get('pairs')}")
            lines.append(f"- passed: **{body.get('passed')}**")
            mismatch = body.get("mismatch")
            if mismatch:
                lines.append(f"- first mismatch at query {mismatch['query_index']} document "
                             f"{mismatch['document_index']}")  # fmt: skip
                lines.append(f"  - served tokens: `{mismatch['served_tokens']}`")
                lines.append(f"  - reference tokens: `{mismatch['reference_tokens']}`")
        else:
            for row in body.get("gates", []):
                value = row.get("value")
                bound = row.get("bound")
                lines.append(f"- `{row['gate']}`: {value} vs bound {bound} -> **{row['passed']}**")
                lines.append(f"  - referent: {row.get('referent', '')}")
        lines.append("")
    lines.append(f"## Verdict: {'PASS' if document.get('passed') else 'FAIL'}")
    return "\n".join(lines) + "\n"
