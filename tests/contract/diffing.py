"""Classify a snapshot diff: BREAKING (something removed or changed) or ADDITIVE (something new)."""

from __future__ import annotations

from typing import Any


def classify(surface: str, old: Any, new: Any, path: str = "") -> list[str]:
    """One line per difference, ``BREAKING``/``ADDITIVE``, with the JSON path that differs."""
    lines: list[str] = []
    where = path or "<root>"
    if isinstance(old, dict) and isinstance(new, dict):
        for key in sorted(set(old) | set(new), key=str):
            sub = f"{path}/{key}" if path else str(key)
            if key not in new:
                lines.append(f"BREAKING  {surface:<11} {sub}: removed")
            elif key not in old:
                lines.append(f"ADDITIVE  {surface:<11} {sub}: added")
            else:
                lines.extend(classify(surface, old[key], new[key], sub))
        return lines
    if old != new:
        lines.append(f"BREAKING  {surface:<11} {where}: {_short(old)} -> {_short(new)}")
    return lines


def _short(value: Any, limit: int = 100) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."
