"""Backport of vllm-project/vllm#48039 (commit ``e6fc81bc78``): pooling chunked prefill uses the full context.

At vLLM v0.31.0, ``Scheduler.__init__`` reserves one sampled-token slot for every runner that is not a
diffusion model (``vllm/v1/core/sched/scheduler.py:146-148``), and the per-step scheduling cap subtracts that
reservation (``scheduler.py:687-692``)::

    num_new_tokens = min(
        num_new_tokens,
        self.max_model_len - request.num_computed_tokens - self.num_sampled_tokens_per_step,
    )

A pooling request appends no sampled token after prefill, so a prompt of exactly ``max_model_len`` tokens
under chunked prefill never schedules its last token and the request hangs (measured on the GPU wave: a
32768-token prompt hangs, 32767 returns in 0.5 s).

The upstream fix, vllm-project/vllm#48039 (commit ``e6fc81bc78``, "[Bugfix][Scheduler] Let pooling chunked
prefill use full context", 2026-10-07), stores ``num_sampled_tokens_per_step = 0`` for a pooling runner
inside ``Scheduler.__init__``. This module backports exactly that semantics by wrapping the constructor:
after the original ``__init__``, when ``vllm_config.model_config.runner_type == "pooling"`` and the
attribute is not already 0, it sets the attribute to 0 and logs one line; an engine whose constructor
already stores 0 (a release that carries the fix) logs one inert line instead. Nothing else is touched: a
generate runner is left exactly as the original constructor left it.

Opt-in: the patch is inert unless the engine process's ``RCP_NDCG_VLLM_PATCHES`` contains
``pooling-full-context`` (the serve path renders a recipe's declared patches into that variable; see
:mod:`rcp_ndcg_vllm.patches`). It is applied by :func:`rcp_ndcg_vllm.models.register`, the one
``vllm.general_plugins`` entry point vLLM loads in every engine process.

Removal condition: delete this module, its name in :data:`rcp_ndcg_vllm.patches.PATCH_NAMES` and the
recipes' opt-in when ``engine.image`` moves to the first vLLM release that carries ``e6fc81bc78`` -- the
inert log line names that moment.
"""

from __future__ import annotations

__all__ = ["PATCH_NAME", "apply"]

import functools
import logging
from typing import Any

logger = logging.getLogger(__name__)

#: The patch's name in ``RCP_NDCG_VLLM_PATCHES`` (and, later, in a recipe's declared patches).
PATCH_NAME = "pooling-full-context"

#: The marker on the wrapped ``__init__``: what makes a second :func:`apply` a no-op.
_WRAPPED_MARKER = "_rcp_ndcg_pooling_full_context_wrapped"


def apply(*, scheduler_cls: type | None = None) -> bool:
    """Install the ``Scheduler.__init__`` wrapper once (idempotent).

    Args:
        scheduler_cls: The scheduler class to wrap; when omitted,
            ``vllm.v1.core.sched.scheduler.Scheduler`` is imported (the engine's own class). The CPU
            tests pass a fake.

    Returns:
        True when this call installed the wrapper, False when it was already installed.
    """
    if scheduler_cls is None:
        from vllm.v1.core.sched.scheduler import Scheduler  # noqa: PLC0415

        scheduler_cls = Scheduler
    if getattr(scheduler_cls.__init__, _WRAPPED_MARKER, False):
        return False
    original_init: Any = scheduler_cls.__init__

    @functools.wraps(original_init)
    def patched_init(self: Any, vllm_config: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, vllm_config, *args, **kwargs)
        if vllm_config.model_config.runner_type != "pooling":
            return
        if getattr(self, "num_sampled_tokens_per_step", None) == 0:
            logger.info(
                "%s is inert: this vLLM already sets num_sampled_tokens_per_step=0 for the pooling "
                "runner (it carries vllm-project/vllm#48039, commit e6fc81bc78); remove the patch",
                PATCH_NAME,
            )
            return
        self.num_sampled_tokens_per_step = 0
        logger.info(
            "%s applied: num_sampled_tokens_per_step set to 0 for the pooling runner "
            "(backport of vllm-project/vllm#48039, commit e6fc81bc78)",
            PATCH_NAME,
        )

    setattr(patched_init, _WRAPPED_MARKER, True)
    scheduler_cls.__init__ = patched_init  # type: ignore[method-assign]  (the class attribute vLLM calls)
    return True
