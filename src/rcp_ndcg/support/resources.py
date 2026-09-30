"""What a job asks of a scheduler: the one home of :class:`Resources`.

A leaf model, so the run config (which declares a run's resources) and the job runners (which request them) share
it without the run layer importing the runners.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Resources(BaseModel):
    """What one task of a job needs from the scheduler. ``None`` leaves it to the scheduler's default."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    gpus: int = Field(default=0, ge=0)
    """GPUs per task."""
    cpus: int | None = Field(default=None, ge=1)
    """CPU cores per task."""
    memory_gb: float | None = Field(default=None, gt=0)
    """Memory per task, in GiB."""
    time_limit_s: int | None = Field(default=None, ge=1)
    """Wall-clock limit per task, in seconds."""


__all__ = ["Resources"]
