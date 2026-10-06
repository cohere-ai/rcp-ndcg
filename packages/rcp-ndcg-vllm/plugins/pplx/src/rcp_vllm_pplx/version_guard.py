"""The version guard of the plugin, importable without vLLM.

The plugin subclasses vLLM internals (``Qwen3_5ForCausalLMBase``, the pooler constructors)
whose shapes vLLM does not promise across releases; it is therefore validated against one
range — ``>=0.31,<0.32``, the pinned engine image ``vllm/vllm-openai:v0.31.0`` — and
refuses anything outside it at the two import moments vLLM actually exercises: the
``register()`` call of the ``vllm.general_plugins`` entry point, and the lazy import of the
model class the registry resolves.
"""

from __future__ import annotations

__all__ = ["SUPPORTED_VLLM_MAX", "SUPPORTED_VLLM_MIN", "checked_vllm_version", "require_vllm_version"]

import importlib.metadata
import re

#: The inclusive lower bound of the validated vLLM range (the image's release).
SUPPORTED_VLLM_MIN = (0, 31)
#: The exclusive upper bound of the validated vLLM range.
SUPPORTED_VLLM_MAX = (0, 32)

_VERSION_RE = re.compile(r"^(\d+)\.(\d+)")


def checked_vllm_version() -> tuple[int, int]:
    """The running vLLM's ``(major, minor)``, from installed distribution metadata.

    Falls back to ``vllm.__version__`` when the distribution metadata is unreadable (a
    from-source checkout without installation). Raises ``ImportError`` when neither is
    available, so a caller never silently skips the guard.
    """
    try:
        raw = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        import vllm  # noqa: PLC0415  (fallback only; the guard must stay cheap)

        raw = getattr(vllm, "__version__", None)
        if raw is None:
            raise ImportError(
                "cannot read the running vLLM's version (no installed distribution "
                "metadata and no vllm.__version__); rcp-ndcg-vllm-pplx refuses to load "
                "unverified"
            ) from None
    match = _VERSION_RE.match(raw)
    if match is None:
        raise ImportError(f"cannot parse the running vLLM version {raw!r}")
    return int(match.group(1)), int(match.group(2))


def require_vllm_version() -> tuple[int, int]:
    """Refuse any vLLM outside the validated range, with the range and the image named.

    Returns:
        The running vLLM's ``(major, minor)`` when it is inside the range.

    Raises:
        RuntimeError: The running vLLM is older or newer than ``[0.31, 0.32)``; the
            message names both the range and the pinned image so an operator sees the fix.
    """
    version = checked_vllm_version()
    if not (SUPPORTED_VLLM_MIN <= version < SUPPORTED_VLLM_MAX):
        raise RuntimeError(
            f"rcp-ndcg-vllm-pplx was validated against vLLM >= {SUPPORTED_VLLM_MIN[0]}."
            f"{SUPPORTED_VLLM_MIN[1]}, < {SUPPORTED_VLLM_MAX[0]}.{SUPPORTED_VLLM_MAX[1]} "
            "(the pinned engine image vllm/vllm-openai:v0.31.0) but is running on vLLM "
            f"{version[0]}.{version[1]}; refusing to register. Re-validate the plugin "
            "against the new engine version before serving with it."
        )
    return version
