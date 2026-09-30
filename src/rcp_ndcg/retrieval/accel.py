"""Accelerate-driven primitives for single-/multi-GPU retrieval workloads.

A thin shim over :class:`accelerate.PartialState` that lets the same code
path run **both** in a plain Python process (single GPU / CPU) **and**
under ``accelerate launch --num_processes N`` (one rank per visible GPU).

Every helper is no-op-safe outside of an ``accelerate launch`` context:
``num_processes == 1`` and ``process_index == 0`` mean "single rank, run
locally, gather is identity".  That extends to accelerate not being installed at
all -- the CPU-only clients (served rerank, hosted embedding APIs) get a world of
one rather than an import error, since a world of one is arithmetic rather than a
distributed system.

Public surface:

* :class:`AccelState` -- wraps :class:`PartialState` with extra
  conveniences (``shard``, ``gather``, ``reorder_global``).
* :func:`resolve_torch_dtype` -- a dtype name as a ``torch.dtype``.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from datetime import timedelta
from typing import Any, TypeVar

from rcp_ndcg.support.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

# Default NCCL collective (watchdog) timeout for the multi-GPU retrieval /
# rerank process group.  The stock PyTorch default is 10 min, which aborts the
# *entire* job at the rerank tail: heavy, length-skewed corpora finish their
# per-rank shards far apart, so the first rank to reach the end-of-loop barrier
# (``wait_for_everyone`` -> a 1-element ALLREDUCE) waits on the slow ranks and
# trips the watchdog -> ``DistBackendError`` -> SIGABRT at ~99% done.  A few
# hours comfortably covers observed tail skew; override with
# ``RCP_NDCG_PG_TIMEOUT_MIN`` (minutes) without a rebuild.
_DEFAULT_PG_TIMEOUT_MIN = 240.0


def _pg_timeout() -> timedelta:
    raw = os.environ.get("RCP_NDCG_PG_TIMEOUT_MIN")
    try:
        minutes = float(raw) if raw else _DEFAULT_PG_TIMEOUT_MIN
    except ValueError:
        logger.warning("Invalid RCP_NDCG_PG_TIMEOUT_MIN=%r; using default %.0f min", raw, _DEFAULT_PG_TIMEOUT_MIN)
        minutes = _DEFAULT_PG_TIMEOUT_MIN
    return timedelta(minutes=minutes)


# ---------------------------------------------------------------------------
# AccelState -- the single source of truth for "where am I in the world?"
# ---------------------------------------------------------------------------


class _SingleProcessState:
    """The ``PartialState`` surface for a world of one, without accelerate.

    Deliberately the same attribute names rather than an adapter: every helper on
    :class:`AccelState` then works unchanged, and there is no second code path
    whose behaviour could drift from the real one.
    """

    process_index = 0
    num_processes = 1
    is_main_process = True

    @property
    def device(self) -> Any:
        try:
            import torch
        except ImportError:
            return "cpu"
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def wait_for_everyone(self) -> None:
        """Nothing to wait for."""


class AccelState:
    """Wrapper around :class:`accelerate.PartialState` with helpers.

    Use it as a context-manager-free singleton: ``state = AccelState()``
    inside any code that wants to be multi-GPU-aware.  Repeated calls return
    instances that share the underlying ``PartialState`` (which is itself a
    process-wide singleton in accelerate).

    Attributes:
        process_index: 0-based rank in the current launch group.
        num_processes: World size.  ``1`` outside of ``accelerate launch``.
        is_main_process: ``True`` on rank 0.
        device: ``torch.device`` for this rank (e.g. ``cuda:0``, ``cuda:3``,
            or ``cpu``).
    """

    def __init__(self) -> None:
        try:
            from accelerate import PartialState
        except ImportError:
            # Not an error: a world of one is arithmetic, not a distributed
            # system. The served rerank and hosted-API paths are CPU-only clients
            # that legitimately run without the `[local]` extra, and demanding
            # accelerate from them would make the dependency-light spine
            # un-runnable for no gain. Multi-rank launches go through
            # `accelerate launch`, which by definition has accelerate.
            self._state = _SingleProcessState()
            return

        # Initialise the process group with an extended collective timeout so
        # rerank-tail stragglers don't trip the 10-min NCCL watchdog (see
        # ``_DEFAULT_PG_TIMEOUT_MIN``).  ``PartialState`` is a process-wide
        # singleton initialised on first construction; this is the only place
        # the package builds it, so the timeout always takes effect.  Older
        # accelerate builds that don't forward the ``timeout`` kwarg fall back
        # to the default.
        try:
            self._state = PartialState(timeout=_pg_timeout())
        except TypeError:
            self._state = PartialState()

    @property
    def process_index(self) -> int:
        return int(self._state.process_index)

    @property
    def num_processes(self) -> int:
        return int(self._state.num_processes)

    @property
    def is_main_process(self) -> bool:
        return bool(self._state.is_main_process)

    @property
    def device(self) -> Any:
        """Underlying ``torch.device`` for this rank (cuda:N, cpu, ...)."""
        return self._state.device

    def wait_for_everyone(self) -> None:
        """Cross-rank barrier; no-op when single-process."""
        self._state.wait_for_everyone()

    # -- sharding ----------------------------------------------------------

    def shard(self, items: Sequence[T]) -> tuple[list[T], list[int]]:
        """Return this rank's slice of ``items`` plus the **global indices**.

        **Round-robin (strided)** assignment: rank ``r`` gets positions
        ``r, r + W, r + 2W, ...`` (``W = num_processes``).  This interleaving
        is deliberate -- the alternative contiguous block split gives one rank
        an entire run of *adjacent* items, so when heavy items cluster (e.g. a
        corpus where long documents sit together) that rank does far more work
        and finishes the rerank tail minutes after the others.  The first rank
        to reach the end-of-loop ``wait_for_everyone`` barrier then waited on
        the straggler and tripped the NCCL watchdog (job aborted at ~99%).
        Striding spreads clustered heavy items evenly across ranks so per-rank
        finish times converge.

        The assignment is a pure function of ``(len(items), rank, world)`` --
        independent of item content -- so call sites that shard the *same*
        input twice get an identical rank->item mapping without coordinating.

        Counts stay balanced to within one item; the returned indices let the
        caller :meth:`reorder_global` the gathered results back to the original
        order.  For ``num_processes == 1`` this is the identity.

        Args:
            items: Anything supporting ``len()`` and indexing.

        Returns:
            ``(local_items, local_indices)`` where ``local_indices[i]`` is
            the original 0-based position of ``local_items[i]``.
        """
        n = len(items)
        if self.num_processes == 1:
            return list(items), list(range(n))

        local_indices = list(range(self.process_index, n, self.num_processes))
        return [items[i] for i in local_indices], local_indices

    # -- gather ------------------------------------------------------------

    def gather_object(self, obj: T) -> list[T]:
        """Gather a Python object from every rank to every rank.

        Returns a list of length ``num_processes``.  In single-process mode
        this returns ``[obj]``.

        Wraps :func:`accelerate.utils.gather_object` so callers don't have
        to import accelerate themselves.
        """
        if self.num_processes == 1:
            return [obj]
        from accelerate.utils import gather_object

        return list(gather_object([obj]))

    def reorder_global(
        self,
        per_rank_items: Iterable[Sequence[T]],
        per_rank_indices: Iterable[Sequence[int]],
        total_size: int,
    ) -> list[T]:
        """Reconstruct an N-item list from per-rank gathered shards.

        ``per_rank_items[r]`` and ``per_rank_indices[r]`` come from
        :meth:`gather_object`-ing a ``(local_items, local_indices)`` pair
        that was originally produced by :meth:`shard`.  This puts every
        item back in its original global slot.
        """
        out: list[T | None] = [None] * total_size
        for items, indices in zip(per_rank_items, per_rank_indices, strict=True):
            for item, idx in zip(items, indices, strict=True):
                if idx < 0 or idx >= total_size:
                    raise IndexError(f"index {idx} out of range for size={total_size}")
                out[idx] = item
        missing = [i for i, v in enumerate(out) if v is None]
        if missing:
            raise RuntimeError(
                f"reorder_global missing {len(missing)} positions (e.g. {missing[:5]}); "
                "ranks did not collectively cover the full input."
            )
        return [v for v in out if v is not None]  # all positions filled


_DTYPE_ALIASES: dict[str, str] = {
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
    "float16": "float16",
    "fp16": "float16",
    "float32": "float32",
    "fp32": "float32",
}


def resolve_torch_dtype(name: str) -> Any:
    """Resolve a dtype name to a ``torch.dtype``.

    Accepts canonical names and the common ``bf16`` / ``fp16`` / ``fp32``
    aliases.  Unknown names fall back to ``torch.bfloat16``.
    """
    import torch

    canonical = _DTYPE_ALIASES.get(name, name)
    return getattr(torch, canonical, torch.bfloat16)


__all__ = ["AccelState", "resolve_torch_dtype"]
