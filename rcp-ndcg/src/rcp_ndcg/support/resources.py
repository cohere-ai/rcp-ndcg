"""What a job asks of a scheduler: the one home of :class:`Resources` and of a job's :data:`Environment`.

Leaf models, so the run config (which declares a run's resources) and the job runners (which request them) share
it without the run layer importing the runners.
"""

from __future__ import annotations

import re
from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

_SHELL_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _shell_name(name: str) -> str:
    if not _SHELL_NAME.fullmatch(name):
        raise ValueError(
            f"{name!r} is not an environment variable name: letters, digits and _, not starting with a digit"
        )
    return name


#: An environment variable's name, as a job's shell exports it (``[A-Za-z_][A-Za-z0-9_]*``).
EnvName = Annotated[str, AfterValidator(_shell_name)]
#: Environment variables a job or an engine sets: ``{name: value}``; each name a shell identifier.
Environment = dict[EnvName, str]


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


__all__ = ["EnvName", "Environment", "Resources"]
