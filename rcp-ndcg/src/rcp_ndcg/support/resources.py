"""What a job asks of a scheduler: the one home of :class:`Resources` and of a job's :data:`Environment`.

Leaf models, so the run config (which declares a run's resources) and the job runners (which request them) share
it without the run layer importing the runners.
"""

from __future__ import annotations

import re
from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

_SHELL_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

#: C0 controls and DEL: a newline ends a heredoc body or an ``#SBATCH`` directive line, and a rendered script
#: is line-based, so a name or directive value that reaches a renderer may not hold one.
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f]")

#: The marker recorded in place of a secret's value (the value itself never reaches ``run.yaml`` or a mirror).
REDACTED = "<redacted>"

#: An environment variable name that names a credential (``*_TOKEN``, ``*_KEY``, ``*SECRET*``, ``*PASSWORD*``,
#: ``*_AUTH``, ``*CREDENTIAL*``). Word-bounded, so ``MONKEY`` and ``KEYSTONE`` are ordinary names.
_SECRET_NAME = re.compile(
    r"(?i)(?:^|_)(?:token|key|secret|password|passwd|credential|auth)(?:_|$)|secret|password|passwd|credential"
)


def no_control_characters(value: str) -> str:
    """``value``, refused when it holds a control character (``\\x00``-``\\x1f`` or DEL).

    For a name, an id, a queue, a namespace or an ``#SBATCH`` directive value, where no control character is ever
    meaningful: a quoted shell word keeps a newline, and the job script is assembled line by line (a heredoc
    body, an ``#SBATCH`` directive, a shell line built from config), so a value with one can end the construct
    and turn the rest of the value into script lines.

    Raises:
        ValueError: ``value`` holds a control character; the message names it.
    """
    found = _CONTROL_CHARACTERS.search(value)
    if found:
        raise ValueError(
            f"{value!r} contains a control character (U+{ord(found.group()):04X}); a newline or a NUL in a "
            "config value would end a heredoc or a scheduler directive line, and the rendered script is line-based"
        )
    return value


def no_nul_byte(value: str) -> str:
    """``value``, refused when it holds a NUL (the one character bash cannot carry through a script).

    A free-form value (an argv word, an environment value, an engine command) may hold a newline: it is quoted
    as one shell word and round-trips inert, and :func:`~rcp_ndcg.runners.script.heredoc` refuses the one body
    that would forge its terminator. A NUL would truncate the rendered script itself, so it is refused.

    Raises:
        ValueError: ``value`` holds a NUL.
    """
    if "\x00" in value:
        raise ValueError(f"{value!r} contains a NUL (U+0000), which a rendered script cannot carry")
    return value


def looks_like_secret(name: str) -> bool:
    """Whether an environment variable *name* looks like a credential (``*_TOKEN``, ``*_KEY``, ``*SECRET*``, ...)."""
    return _SECRET_NAME.search(name) is not None


def refuse_secret_value(name: str, value: str) -> str:
    """``value``, refused when ``name`` looks like a credential and the value is not the redaction marker.

    The value would be recorded in ``run.yaml``, the manifest and every mirror copy. A credential belongs in the
    environment the job runs in (inherited by the local and SLURM runners) or in a Kubernetes secret reference
    (``runner.options.secrets``), never in a config.

    Raises:
        ValueError: ``name`` looks like a credential and ``value`` is a literal value.
    """
    if looks_like_secret(name) and value != REDACTED:
        raise ValueError(
            f"env.{name} looks like a credential (a *_TOKEN, *_KEY, *SECRET* or *PASSWORD* name), and its value "
            "would be written into run.yaml, the manifest and the mirror; export it in the environment the job "
            "runs in, or reference a Kubernetes secret (runner.options.secrets), and leave it out of the config"
        )
    return value


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


__all__ = [
    "Environment",
    "EnvName",
    "REDACTED",
    "Resources",
    "looks_like_secret",
    "no_control_characters",
    "no_nul_byte",
    "refuse_secret_value",
]
