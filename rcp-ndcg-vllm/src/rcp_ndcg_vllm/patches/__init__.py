"""Engine-side patch modules, opted in per engine process through one environment variable.

The engine process opts in by carrying the patch names in :data:`PATCHES_ENV` (comma-separated);
:func:`rcp_ndcg_vllm.models.register` -- the one ``vllm.general_plugins`` entry point vLLM loads in every
engine process -- calls :func:`apply_opted_in_patches` there. Nothing here imports vLLM at module level: each
patch imports the vLLM surface it wraps inside its own ``apply``, so importing this package stays clean.

The opt-in contract (what the recipe schema's opt-in field will call when it lands): the recipe's declared
patch names become the value of :data:`PATCHES_ENV` in the engine process, comma-separated, e.g.
``RCP_NDCG_VLLM_PATCHES=pooling-full-context``; until that field lands, the caller sets the variable (the
``rcp-ndcg-vllm serve`` console execs the engine with its environment inherited). The names a recipe may
declare are :data:`PATCH_NAMES`; a name outside that set is ignored with a warning.
"""

from __future__ import annotations

__all__ = ["PATCHES_ENV", "PATCH_NAMES", "apply_opted_in_patches", "opted_in_patch_names"]

import logging
import os
from collections.abc import Mapping

from . import pooling_full_context

logger = logging.getLogger(__name__)

#: The engine process's opt-in variable: a comma-separated list of patch names. The caller sets it in the
#: engine's environment (``rcp-ndcg-vllm serve`` execs ``vllm serve`` with its environment inherited); the
#: recipe-side field that renders a recipe's declared patches into it is not shipped yet. vLLM loads this
#: package's entry point in every engine process (process 0, the engine core and the workers), so the
#: variable is read there, never in the client.
PATCHES_ENV = "RCP_NDCG_VLLM_PATCHES"

#: Every patch this package ships, by name -- what a recipe may declare (the recipe-side opt-in field is not
#: shipped yet; this tuple is the engine-side half of the contract).
PATCH_NAMES: tuple[str, ...] = (pooling_full_context.PATCH_NAME,)


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
