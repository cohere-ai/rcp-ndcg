"""The suite file format: a benchmark as data, so a downstream adds one without a code edit.

A suite is a directory with a ``suite.yaml`` (or the YAML file itself)::

    kind: suite
    name: my-suite
    repo: org/my-dataset
    subsets: [task-a, task-b]
    protocol: plain
    revision: null
    gains: true

The four shipped suites are data files of this package (``rcp/data/suites/<name>.yaml``) and are read by
:class:`SuiteReader` too, so the shipped set exercises the same path a downstream suite takes.
:func:`rcp.data.load_dataset` resolves ``suite:<name>`` through the shipped view first and then reads
``suite:<path>`` as a suite directory or file.

The fields:

* ``kind`` is the discriminator, ``suite``; a file for another kind is refused.
* ``name`` is the suite's name: what reports call it, and the key of the shipped view.
* ``repo`` is the HuggingFace dataset repository holding the subsets (``owner/name``).
* ``subsets`` are the repository's subset names, in scoring order; at least one, each once.
* ``protocol`` is the :data:`~rcp_core.protocol.PROTOCOLS` preset every subset is scored with.
* ``revision`` is the Hub revision (commit, tag or branch) the data is read at; ``null`` is ``main``. A full
  commit pins the data exactly, and the suite refuses a part read at another revision.
* ``gains`` records whether the released data carries calibrated gains (the qrels ``gain``/``theta`` columns);
  the four released suites do.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from rcp.errors import ConfigError
from rcp.support.config import config_error

SUITE_FILE = "suite.yaml"
"""The file a suite directory carries; a suite may also be that file's path directly."""

SHIPPED_SUITE_PACKAGE = "rcp.data"
"""The package whose ``suites/`` directory holds the four shipped suite files."""


class SuiteSpec(BaseModel):
    """One suite definition, as a ``suite.yaml`` carries it.

    Attributes:
        kind: The file's discriminator; always ``suite``.
        name: The suite's name: what reports call it and the key of the shipped view.
        repo: The HuggingFace dataset repository (``owner/name``) holding the subsets.
        subsets: The repository's subset names, in scoring order; at least one, each once.
        protocol: The :data:`~rcp_core.protocol.PROTOCOLS` preset the subsets are scored with.
        revision: The Hub revision (commit, tag or branch) the data is read at; ``None`` is ``main``.
        gains: Whether the released data carries calibrated gains (the qrels ``gain``/``theta`` columns).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["suite"] = "suite"
    name: str
    repo: str
    subsets: tuple[str, ...]
    protocol: str
    revision: str | None = None
    gains: bool = True

    @field_validator("name", "protocol")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        """A name and a protocol are non-empty strings."""
        if not value.strip():
            raise ValueError("must not be blank")
        if value != value.strip():
            raise ValueError("must not carry leading or trailing whitespace")
        return value

    @field_validator("revision")
    @classmethod
    def _revision_not_blank(cls, value: str | None) -> str | None:
        """A declared revision is a branch, tag or commit; a blank string is not ``None``, it is a typo."""
        if value is not None and not value.strip():
            raise ValueError("must not be blank; omit it (or write null) for main")
        return value

    @field_validator("repo")
    @classmethod
    def _owner_and_repo(cls, value: str) -> str:
        """The repository is ``owner/name``: two non-empty path segments, no surrounding whitespace."""
        parts = value.split("/")
        if len(parts) != 2 or not all(part.strip() for part in parts) or value != value.strip():
            raise ValueError("must be '<owner>/<repo>'")
        return value

    @field_validator("subsets")
    @classmethod
    def _subsets_are_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """At least one subset, each named once, in scoring order."""
        if not value:
            raise ValueError("must name at least one subset")
        if len(set(value)) != len(value):
            repeated = sorted({subset for subset in value if value.count(subset) > 1})
            raise ValueError(f"names the subsets {repeated} twice")
        if any(not subset.strip() for subset in value):
            raise ValueError("a subset name must not be blank")
        return value


class SuiteReader:
    """Reads one suite definition: a directory with a ``suite.yaml``, or the YAML file itself.

    Args:
        uri: The suite directory, or the ``suite.yaml`` file.

    Raises:
        ConfigError: There is no such directory or file, the directory has no ``suite.yaml``, the YAML is not
            a mapping, or the file does not validate as a :class:`SuiteSpec` (the file and field are named).
    """

    name = "suite"

    def __init__(self, uri: str) -> None:
        self.uri = uri
        self.path = suite_file(uri)
        self.spec = read_suite_spec(self.path)


def suite_file(uri: str) -> Path:
    """The ``suite.yaml`` *uri* names, directly or inside the suite directory it names.

    Args:
        uri: A suite directory, or a YAML file.

    Returns:
        The suite file's path.

    Raises:
        ConfigError: The path does not exist, or the directory carries no ``suite.yaml``.
    """
    path = Path(uri)
    if path.is_dir():
        candidate = path / SUITE_FILE
        if not candidate.is_file():
            raise ConfigError(
                f"{path} has no {SUITE_FILE}",
                hint=f"a suite directory carries a {SUITE_FILE} (name, repo, subsets, protocol, revision, gains)",
            )
        return candidate
    if path.is_file():
        return path
    raise ConfigError(
        f"no suite directory or {SUITE_FILE} at {uri!r}",
        hint="point at a suite directory (with a suite.yaml) or at the YAML file itself",
    )


def read_suite_spec(path: str | Path) -> SuiteSpec:
    """Read one suite file into a :class:`SuiteSpec`.

    Args:
        path: The suite directory or YAML file.

    Returns:
        The validated spec.

    Raises:
        ConfigError: The file is missing, not a YAML mapping, or does not validate (field and value named).
    """
    file = suite_file(str(path))
    try:
        data = yaml.safe_load(file.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{file}: not valid YAML: {exc}") from exc
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"{file} must be a YAML mapping, got {type(data).__name__}")
    try:
        return SuiteSpec.model_validate(data)
    except ValidationError as exc:
        raise config_error(exc, model=SuiteSpec, source=str(file)) from None


def shipped_suite_names() -> tuple[str, ...]:
    """The names of the suites shipped as data files of this package, sorted."""
    root = files(SHIPPED_SUITE_PACKAGE).joinpath("suites")
    return tuple(sorted(item.name.removesuffix(".yaml") for item in root.iterdir() if item.name.endswith(".yaml")))


def shipped_suite_path(name: str) -> Path:
    """The shipped ``suite.yaml`` file of suite *name*.

    Raises:
        ConfigError: No shipped suite has that name (the shipped names are listed).
    """
    item = files(SHIPPED_SUITE_PACKAGE).joinpath("suites", f"{name}.yaml")
    if not item.is_file():
        raise ConfigError(
            f"no shipped suite {name!r}",
            hint=f"the shipped suites: {list(shipped_suite_names())}",
        )
    return Path(str(item))


def shipped_suite_specs() -> tuple[SuiteSpec, ...]:
    """Every shipped suite, read through :class:`SuiteReader` (the shipped view's source)."""
    return tuple(SuiteReader(str(shipped_suite_path(name))).spec for name in shipped_suite_names())


def resolve_suite_spec(name_or_path: str) -> SuiteSpec | None:
    """The suite a ``suite:`` URI names: a shipped suite first, then the suite file at the path.

    The one home of the resolution order. :func:`rcp.data.load_dataset` reads it, and so does the identity
    layer (:func:`rcp.data.revisions.dataset_uri_revision`).

    Args:
        name_or_path: The ``suite:`` URI's remainder: a shipped suite name (``trecdl``), or a path to a suite
            directory or to its ``suite.yaml``.

    Returns:
        The validated :class:`SuiteSpec`, or ``None`` when the argument names neither a shipped suite nor an
        existing path.

    Raises:
        ConfigError: The path exists but is no suite (a directory without ``suite.yaml``, or a file that does not
            validate).
    """
    if name_or_path in shipped_suite_names():
        return read_suite_spec(shipped_suite_path(name_or_path))
    if not Path(name_or_path).exists():
        return None
    return SuiteReader(name_or_path).spec


__all__ = [
    "SHIPPED_SUITE_PACKAGE",
    "SUITE_FILE",
    "SuiteReader",
    "SuiteSpec",
    "read_suite_spec",
    "resolve_suite_spec",
    "shipped_suite_names",
    "shipped_suite_path",
    "shipped_suite_specs",
    "suite_file",
]
