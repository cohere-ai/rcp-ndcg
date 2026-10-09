"""Engine-side patch modules, opted in per engine process through one environment variable.

The engine process opts in by carrying the patch names in :data:`PATCHES_ENV` (comma-separated);
:func:`rcp_ndcg_vllm.models.register` -- the one ``vllm.general_plugins`` entry point vLLM loads in every
engine process -- calls :func:`apply_opted_in_patches` there. Nothing here imports vLLM at module level: each
patch imports the vLLM surface it wraps inside its own ``apply``, so importing this package stays clean.

The opt-in contract: the recipe's declared patch names (``serve.patches``, validated against
:data:`PATCH_NAMES`) are rendered into the engine process's value of :data:`PATCHES_ENV` by
``rcp-ndcg-vllm serve``, comma-separated, e.g. ``RCP_NDCG_VLLM_PATCHES=pooling-full-context``. A name
outside that set is ignored with a warning. The engine's environment is recorded in the corpus provenance,
so the value the engine actually ran with is visible beside the fingerprint that keys it.
"""

from __future__ import annotations

__all__ = [
    "PATCHES_ENV",
    "PATCH_MODULES",
    "PATCH_NAMES",
    "apply_opted_in_patches",
    "opted_in_patch_names",
    "patches_env_value",
]

import logging
import os
from collections.abc import Iterable, Mapping

from . import pooling_full_context

logger = logging.getLogger(__name__)

#: The engine process's opt-in variable: a comma-separated list of patch names. ``rcp-ndcg-vllm serve``
#: sets it from the recipe's declared ``serve.patches`` before it execs the engine, so the value the engine
#: reads is exactly what the recipe declared (and what the fingerprint keys); a hand-set value is overridden.
#: vLLM loads this package's entry point in every engine process (process 0, the engine core and the
#: workers), so the variable is read there, never in the client.
PATCHES_ENV = "RCP_NDCG_VLLM_PATCHES"

#: Every patch this package ships, by name, and the module that implements it -- the one home of the
#: name-to-code mapping. A recipe may declare a name (the recipe schema validates it against
#: :data:`PATCH_NAMES`), and the behaviour fingerprint hashes the module's bytes for every declared patch
#: (``plugin_sha256.<module>``), so a patch fix moves the fingerprint of exactly the recipes that opt in.
PATCH_MODULES: dict[str, str] = {
    pooling_full_context.PATCH_NAME: "rcp_ndcg_vllm.patches.pooling_full_context",
}

#: Every patch this package ships, by name -- what a recipe may declare (``serve.patches``).
PATCH_NAMES: tuple[str, ...] = tuple(PATCH_MODULES)


def patches_env_value(names: Iterable[str]) -> str:
    """The engine process's :data:`PATCHES_ENV` value for a recipe's declared patch names.

    One home for the rendering every engine-start path uses: the ``rcp-ndcg-vllm serve`` console, the wave
    runner's engine starts and the e2e driver's serve configs.  The names are comma-separated in declaration
    order; no name gives the empty string, which opts into nothing (an engine started with it runs no patch).
    """
    return ",".join(names)


def opted_in_patch_names(environ: Mapping[str, str] | None = None) -> tuple[str, ...]:
    """The patch names opted in through :data:`PATCHES_ENV`, in declaration order, deduplicated.

    Args:
        environ: The mapping to read (default :data:`os.environ`).

    Returns:
        The stripped, non-empty names of the comma-separated value; ``()`` when it is unset or empty.
    """
    raw = (os.environ if environ is None else environ).get(PATCHES_ENV, "")
    return tuple(dict.fromkeys(name.strip() for name in raw.split(",") if name.strip()))


def apply_opted_in_patches(
    *, environ: Mapping[str, str] | None = None, scheduler_cls: type | None = None
) -> tuple[str, ...]:
    """Apply every opted-in engine patch, in this package's order.

    Args:
        environ: The mapping to read the opt-in from (default :data:`os.environ`).
        scheduler_cls: The scheduler class to wrap; vLLM's own
            (``vllm.v1.core.sched.scheduler.Scheduler``) is imported when omitted. A seam for the CPU
            tests, which run without vLLM.

    Returns:
        The opted-in patch names that this package knows (empty when none were named). An unknown name is
        ignored with a warning: the recipe schema restricts the field to :data:`PATCH_NAMES`, so only a
        hand-set variable can carry one, and a silent no-op would hide a typo.
    """
    names = opted_in_patch_names(environ)
    unknown = [name for name in names if name not in PATCH_NAMES]
    if unknown:
        logger.warning("ignoring unknown %s names: %s", PATCHES_ENV, ", ".join(unknown))
    if pooling_full_context.PATCH_NAME in names:
        pooling_full_context.apply(scheduler_cls=scheduler_cls)
    return tuple(name for name in names if name in PATCH_NAMES)
