"""The one vLLM version guard of the folded model plugins (layout-move item 3): topk and pplx share it.

The plugin is developed and tested against one vLLM line (``>=0.31,<0.32``,
the ``vllm/vllm-openai:v0.31.0`` image).  It refuses to register on any other
version with a message that names the tested range: subclassing
``ColQwen3_5Model`` and the pooler wiring below are vLLM-internals contracts
that other releases may break silently, which would corrupt scores rather
than fail loudly.  The guard runs

- when the entry-point callable executes (engine startup), and
- when vLLM lazily imports the model module (``rcp_ndcg_vllm_topk.model``),

both without importing vLLM (``importlib.metadata`` reads the installed
distribution's version).

The stock loader swallows entry-point exceptions with a logged traceback
(``load_plugins_by_group`` in vllm/plugins/__init__.py) and a lazy model
import failure becomes a "not supported" error, so the guard's message is
designed to be readable in both paths' logs.
"""

from __future__ import annotations

import importlib.metadata
import re

__all__ = [
    "SUPPORTED_VLLM_MAX",
    "SUPPORTED_VLLM_MIN",
    "TESTED_VLLM_MAX",
    "TESTED_VLLM_MIN",
    "checked_vllm_version",
    "ensure_vllm_version",
    "installed_vllm_version",
    "parse_vllm_minor_version",
    "require_vllm_version",
]

# The tested range, as (major, minor) bounds: [0.31, 0.32).
TESTED_VLLM_MIN = (0, 31)
TESTED_VLLM_MAX = (0, 32)

_LEADING_NUMBERS = re.compile(r"\d+")


def parse_vllm_minor_version(version: str) -> tuple[int, int] | None:
    """Return the ``(major, minor)`` prefix of a PEP 440 version string.

    >>> parse_vllm_minor_version("0.31.0")
    (0, 31)
    >>> parse_vllm_minor_version("0.31.1rc2")
    (0, 31)
    >>> parse_vllm_minor_version("garbage") is None
    True

    Any pre-release/post/dev suffix after the second number is ignored, so a
    hypothetical ``0.32.0.dev0`` still counts as 0.32 and is refused.
    """
    numbers = _LEADING_NUMBERS.findall(version)
    if len(numbers) < 2:
        return None
    return (int(numbers[0]), int(numbers[1]))


def installed_vllm_version() -> str:
    """Return the installed vLLM distribution's version string.

    Reads importlib.metadata, so this never imports vLLM itself.  Raises
    ``RuntimeError`` naming the distribution when vLLM is absent.
    """
    try:
        return importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(
            "rcp-ndcg-vllm-topk: no installed vLLM distribution found; the "
            "plugin registers a model class for the vLLM engine and has "
            "nothing to register into."
        ) from error


def ensure_vllm_version() -> tuple[int, int]:
    """Refuse any vLLM outside the tested range ``[0.31, 0.32)``.

    Returns the parsed ``(major, minor)`` of the installed vLLM.  Raises
    ``RuntimeError`` with the tested range when the version is outside it (or
    unreadable), because the plugin's internals contracts (registry, model
    class hierarchy, pooler) are only verified for that line.
    """
    version = installed_vllm_version()
    parsed = parse_vllm_minor_version(version)
    lo, hi = TESTED_VLLM_MIN, TESTED_VLLM_MAX
    if parsed is None or not (lo <= parsed < hi):
        raise RuntimeError(
            f"rcp-ndcg-vllm: refusing to register TopkEmbedModel on "
            f"vLLM {version!r}. The plugin is written against "
            f">={lo[0]}.{lo[1]},<{hi[0]}.{hi[1]} (the vllm/vllm-openai:v0.31.0 "
            "image) and its vLLM-internals contracts "
            "(ColQwen3_5Model, pooler_for_token_embed, the is_causal config "
            "hook) are untested on any other version. Pin the image's vLLM "
            "to the tested line, or review the plugin against the new "
            "version and widen the range deliberately."
        )
    return parsed


#: The pplx spelling of the same range (the folded plugins share one guard).
SUPPORTED_VLLM_MIN = TESTED_VLLM_MIN
SUPPORTED_VLLM_MAX = TESTED_VLLM_MAX

_VERSION_RE = re.compile(r"^(\d+)\.(\d+)")


def checked_vllm_version() -> tuple[int, int]:
    """The installed vLLM's ``(major, minor)`` (the pplx spelling; :func:`parse_vllm_minor_version` reads it)."""
    return parse_vllm_minor_version(installed_vllm_version()) or (-1, -1)


def require_vllm_version() -> tuple[int, int]:
    """The pplx spelling of :func:`ensure_vllm_version` (one guard, both folded models).

    Raises:
        RuntimeError: the running vLLM is outside ``>=0.31,<0.32``.
    """
    version = checked_vllm_version()
    if not (SUPPORTED_VLLM_MIN <= version < SUPPORTED_VLLM_MAX):
        raise RuntimeError(
            f"rcp-ndcg-vllm's pplx model was validated against vLLM >= {SUPPORTED_VLLM_MIN[0]}."
            f"{SUPPORTED_VLLM_MIN[1]}, < {SUPPORTED_VLLM_MAX[0]}.{SUPPORTED_VLLM_MAX[1]} "
            f"({version} is installed; the pinned image is vllm/vllm-openai:v0.31.0); "
            "the model subclasses vLLM internals whose shapes vLLM does not "
            "promise across releases. Pin the engine image's vLLM to the validated line, or review the "
            "model against the new version and widen the range deliberately."
        )
    return version
