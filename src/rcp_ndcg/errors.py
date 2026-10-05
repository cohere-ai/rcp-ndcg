"""Typed errors, typed warnings and exit codes.

An agent driving the CLI or the MCP server has to decide what to do about a failure: retry the call, fix the
config, ask for a credential, or give up. Every failure this package raises deliberately is therefore an
:class:`RcpNdcgError` subclass that carries an exit code, a message, a ``hint`` naming the next step, ``details``
an automated caller can read, and whether retrying can help (``retryable``). :func:`classify` maps every other
exception onto the same classes, so the CLI and MCP report one shape whatever the source.

==== ================= ========================== =======================================================
Code Name              Class                      What a caller should do
==== ================= ========================== =======================================================
0    ``SUCCESS``                                  proceed (a resume with nothing left to do is a success)
1    ``INTERNAL``      :class:`RcpNdcgError`      a bug: report it (the traceback is in ``--log-file``)
2    ``USAGE``         :class:`UsageError`        fix the command line or the tool arguments
3    ``CONFIG``        :class:`ConfigError`       fix the config file or the ``--set`` override
4    ``MISSING_INPUT`` :class:`MissingInputError` produce the input the hint names
5    ``CREDENTIALS``   :class:`CredentialsError`  set the variable the hint names
6    ``PROVIDER``      :class:`ProviderError`     an endpoint failed after retries: resume later if retryable
7    (retired)                                    never returned; the number is not reused
8    ``CAPABILITY``    :class:`CapabilityError`   the endpoint cannot take the request: change one
9    ``INTERRUPTED``   :class:`Interrupted`       SIGINT or SIGTERM; the state is consistent: resume
10   ``DEPENDENCY``    :class:`DependencyError`   install the extra the hint names
11   ``IDENTITY``      :class:`IdentityError`     refusing to mix: write to a new output, or ``--force``
12   ``DATA``          :class:`DataError`         the input would produce wrong numbers: fix it
==== ================= ========================== =======================================================

Warnings that a caller may want to branch on are :class:`RcpNdcgWarning` instances with a ``code`` from the
closed list :data:`WarningCode`; the CLI puts them into the ``warnings`` of its ``--json`` envelope.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from enum import IntEnum
from typing import Any, ClassVar, Literal, get_args


class ExitCode(IntEnum):
    """Process exit codes, stable across releases."""

    SUCCESS = 0
    INTERNAL = 1
    USAGE = 2
    CONFIG = 3
    MISSING_INPUT = 4
    CREDENTIALS = 5
    PROVIDER = 6
    # 7 is retired: never returned, never reused.
    CAPABILITY = 8
    INTERRUPTED = 9
    DEPENDENCY = 10
    IDENTITY = 11
    DATA = 12


class RcpNdcgError(Exception):
    """Base class of every deliberate failure; on its own it means an unexpected failure (exit 1).

    Args:
        message: What went wrong, for a person.
        hint: The next step (a flag, a command, a variable name); never a secret. Worded for a Python caller
            (keyword arguments, functions).
        details: JSON-serialisable facts a program can act on (paths, counts, differing fields).
        retryable: Whether the same call can succeed later unchanged. Defaults to the class's value.
        cli_hint: The same next step worded for the command line (flags, commands), when it differs from
            ``hint``. The CLI and the MCP server show it in place of ``hint`` (:meth:`for_cli`).
    """

    exit_code: ClassVar[ExitCode] = ExitCode.INTERNAL
    retryable: bool = False

    def __init__(
        self,
        message: str,
        *,
        hint: str | None = None,
        details: Mapping[str, Any] | None = None,
        retryable: bool | None = None,
        cli_hint: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.cli_hint = cli_hint
        self.details: dict[str, Any] = dict(details or {})
        if retryable is not None:
            self.retryable = retryable

    @property
    def code(self) -> str:
        """The exit code's name (``"CONFIG"``), what the ``--json`` envelope calls ``error.code``."""
        return self.exit_code.name

    def for_cli(self) -> RcpNdcgError:
        """This error as the CLI and the MCP server report it: ``hint`` is the command-line wording, if any."""
        if self.cli_hint is not None:
            self.hint = self.cli_hint
        return self

    def to_dict(self) -> dict[str, Any]:
        """The ``error`` object of the ``--json`` envelope and of an MCP tool error."""
        return {
            "code": self.code,
            "exit_code": int(self.exit_code),
            "message": self.message,
            "hint": self.hint,
            "retryable": self.retryable,
            "details": self.details,
        }


class UsageError(RcpNdcgError):
    """The command line or the tool arguments are malformed: an unknown flag, a missing or ill-typed value."""

    exit_code = ExitCode.USAGE


class ConfigError(RcpNdcgError):
    """A config value is invalid: an unknown key, a value out of range, a floating model alias."""

    exit_code = ExitCode.CONFIG


class MissingInputError(RcpNdcgError):
    """A file, run, dataset or artifact the call needs does not exist; the hint names what produces it."""

    exit_code = ExitCode.MISSING_INPUT


class CredentialsError(RcpNdcgError):
    """A credential is absent or rejected. The hint names the variable, never its value."""

    exit_code = ExitCode.CREDENTIALS


class ProviderError(RcpNdcgError):
    """An endpoint or a scheduler failed after its retries (unreachable, timing out, rate limiting, an empty answer).

    Retryable unless the failure cannot change by waiting (a route or model the endpoint does not have).
    """

    exit_code = ExitCode.PROVIDER
    retryable = True


class BackendUnavailableError(ProviderError):
    """Every replica of an endpoint stayed unavailable for longer than its ``wait_on_outage_s``.

    The public outage type of the inference layer (the API key, routing, parking and retries are the
    transport's): a run against dead servers parks instead of turning the outage into
    missing judgements, and this error says the parking gave up. Retryable: the endpoint may come back.
    """


class RequestRejectedError(ProviderError):
    """An endpoint refused this one request (e.g. HTTP 400 for a prompt over the context) or answered it empty.

    Specific to the request, unlike :class:`BackendUnavailableError`: the endpoint serves other requests, so a
    judging pass records the window as invalid and goes on, and a resumed pass asks it again.
    """

    retryable = False


class CapabilityError(RcpNdcgError):
    """The judge or endpoint cannot take what a request carries.

    Raised by the first such request: an answer schema the endpoint refuses, images or videos beyond the judge's
    ``max_images`` / ``max_videos``, a window whose media exceed the context, media for a text-only encoder.
    """

    exit_code = ExitCode.CAPABILITY


class Interrupted(RcpNdcgError):
    """The process was interrupted (SIGINT or SIGTERM). The state on disk is consistent; resume it."""

    exit_code = ExitCode.INTERRUPTED
    retryable = True


class DependencyError(RcpNdcgError):
    """An optional extra is needed and not installed; the hint is the exact install command."""

    exit_code = ExitCode.DEPENDENCY


class IdentityError(RcpNdcgError):
    """Refusing to mix artifacts whose identity differs.

    Raised for a resume with a changed config, judgements from another family, or an insertion whose anchor
    check failed. ``details`` names the differing fields.
    """

    exit_code = ExitCode.IDENTITY


class DataError(RcpNdcgError):
    """Input that would produce wrong numbers, or does not parse.

    Raised for malformed or non-finite rankings, qrels or gains, gains that match no labelled query, ids that do
    not join, a document over its text cap under ``on_overflow: fail``, a new document the evidence cannot
    identify, a query with invalid windows under ``strict``, and a damaged mirror.
    """

    exit_code = ExitCode.DATA


WarningCode = Literal[
    "APPROXIMATE_IMAGE_TOKENS",
    "BT_L2_MISMATCH",
    "INVALID_WINDOWS",
    "UNCALIBRATED_DOCUMENTS",
    "UNPINNED_REVISION",
    "UNREADABLE_RUN",
]
"""The closed list of warning codes. Adding a code is an additive change; renaming one is breaking."""

WARNING_CODES: tuple[str, ...] = get_args(WarningCode)


class RcpNdcgWarning(UserWarning):
    """A condition worth a caller's attention that does not stop the call.

    Raise it with ``warnings.warn(RcpNdcgWarning("UNREADABLE_RUN", "..."))``; the CLI collects it into the
    ``warnings`` of its ``--json`` envelope (and prints it on stderr otherwise), and the MCP server logs it.

    Args:
        code: One of :data:`WarningCode`.
        message: What happened, for a person.
    """

    def __init__(self, code: WarningCode, message: str) -> None:
        if code not in WARNING_CODES:
            raise ValueError(f"unknown warning code {code!r}; known: {', '.join(WARNING_CODES)}")
        super().__init__(message)
        self.code = code
        self.message = message

    def to_dict(self) -> dict[str, str]:
        """The warning as the ``--json`` envelope carries it."""
        return {"code": self.code, "message": self.message}


#: Which extra provides which import: the one table of the optional extras (``rcp-ndcg doctor`` checks them from it).
#: A caller that hits one of these gets the install command, not a bare ``No module named 'torch'``: the difference
#: between a fixable situation and an apparent crash.
EXTRA_FOR_MODULE: dict[str, str] = {
    "torch": "calibrate",
    "accelerate": "local",
    "transformers": "local",
    "mteb": "mteb",
    "datasets": "data",
    "pypdfium2": "data",
    "huggingface_hub": "hf",
    "tokenizers": "hf",
    "s3fs": "s3",
    "adlfs": "azure",
    "aiohttp": "http",
    "vllm": "vllm",
}

_INSTALL_MARKER = "rcp-ndcg["


def dependency_error(module: str, *, needed_for: str = "this step") -> DependencyError:
    """The :class:`DependencyError` of a missing ``module``: its hint installs the extra that provides it.

    Args:
        module: The import name that is missing (``"s3fs"``; a dotted name counts by its top-level package).
        needed_for: What needs it, for the message.
    """
    root = module.split(".")[0]
    extra = EXTRA_FOR_MODULE.get(root)
    return DependencyError(
        f"{needed_for} needs {root}, which is not installed",
        hint=f'pip install "rcp-ndcg[{extra}]"' if extra else f"pip install {root}",
        details={"module": root, "extra": extra},
    )


def error_class(exit_code: int) -> type[RcpNdcgError] | None:
    """The error class whose exit code is ``exit_code`` (``None`` for a code no class uses, e.g. a signal's).

    A caller that ran an ``rcp-ndcg`` command as a child process raises the child's failure as its own class.
    """
    classes = (RcpNdcgError, *_subclasses(RcpNdcgError))
    return next((cls for cls in classes if int(cls.exit_code) == exit_code and exit_code != 0), None)


def _subclasses(cls: type[RcpNdcgError]) -> list[type[RcpNdcgError]]:
    """The public error classes below ``cls`` (a class keeps its parent's code only through inheritance)."""
    found = []
    for sub in cls.__subclasses__():
        if sub.__module__ == __name__:
            found += [sub, *_subclasses(sub)]
    return found


def _named(exc: BaseException | None, module: str, *names: str) -> bool:
    """Whether *exc* is an instance of one of ``module.names``, without importing *module*.

    A foreign exception can only come from a module that is already loaded, so an absent module means "no";
    so does a ``None`` cause (an exception raised with nothing chained under it).
    """
    loaded = sys.modules.get(module)
    if loaded is None:
        return False
    classes = tuple(cls for cls in (getattr(loaded, name, None) for name in names) if isinstance(cls, type))
    return bool(classes) and isinstance(exc, classes)


def classify(exc: BaseException) -> RcpNdcgError:
    """Map any exception onto the typed surface.

    Deliberate failures (:class:`RcpNdcgError`) pass through unchanged. Foreign exceptions are mapped only where
    their type says unambiguously what the caller should do:

    * a missing file is :class:`MissingInputError`; a missing optional module is :class:`DependencyError`;
    * a config that does not parse or validate (``yaml``, ``pydantic``) is :class:`ConfigError`;
    * a malformed JSON input is :class:`DataError`;
    * connection failures, timeouts and rate limits (``httpx``, ``openai``, built-in) are :class:`ProviderError`;
      rejected credentials (``openai``, gated Hub repositories) are :class:`CredentialsError`; a Hub repository,
      file or revision that does not exist is :class:`MissingInputError`;
    * a Hub download that found neither the file nor a usable cache entry (``LocalEntryNotFoundError``) is read
      from its cause (``__cause__`` / ``__context__``): offline is a non-retryable :class:`MissingInputError`
      (the dataset surface adds the ``--revision`` wording where it knows the revision is not resolved); a Hub
      that cannot be reached — connection failure, timeout, or answering 5xx or 429 — is a retryable
      :class:`ProviderError`;
    * ``KeyboardInterrupt`` is :class:`Interrupted`.

    Everything else, including a bare ``ValueError`` or ``TypeError``, is an unexpected failure (exit 1): code
    that refuses a user's input deliberately raises a typed error, so an untyped one is a bug to report, not a
    config to fix.

    Args:
        exc: Any exception.

    Returns:
        The typed error; ``exc`` itself when it already is one.
    """
    if isinstance(exc, RcpNdcgError):
        return exc
    name = f"{type(exc).__name__}: {exc}"
    if isinstance(exc, KeyboardInterrupt):
        return Interrupted("interrupted", hint="the state on disk is consistent: re-run the command to resume")
    if isinstance(exc, ModuleNotFoundError) and exc.name and exc.name.split(".")[0] in EXTRA_FOR_MODULE:
        return dependency_error(exc.name)
    if isinstance(exc, ImportError) and _INSTALL_MARKER in str(exc):
        text = str(exc)
        return DependencyError(text, hint=text[text.find("pip install") :] if "pip install" in text else None)

    # before FileNotFoundError, which it subclasses: an offline cache miss is not a missing file.
    # The library raises LocalEntryNotFoundError for every Hub failure it cannot answer from the local cache,
    # chaining the real cause; the cause says what a caller should do, the message alone does not.
    # A Hub that is down (5xx) or rate-limiting (429) is unreachable for now: the same retryable provider failure
    # huggingface_hub itself buckets with the transport errors.
    def offline_miss() -> MissingInputError:
        """The offline Hub failure: retrying cannot help; the fix is online once, or pinning what the cache has."""
        return MissingInputError(
            name,
            hint="the Hub is unreachable offline (HF_HUB_OFFLINE); run once online to download the file — or, if "
            "the revision is not already pinned, pin the revision the cache was filled at",
        )

    def hub_down(cause: BaseException | None) -> bool:
        """Whether *cause* is the Hub answering with a failure a later retry can survive."""
        if not _named(cause, "huggingface_hub.errors", "HfHubHTTPError"):
            return False
        status = getattr(getattr(cause, "response", None), "status_code", 0)
        return status >= 500 or status == 429

    if _named(exc, "huggingface_hub.errors", "OfflineModeIsEnabled"):
        return offline_miss()
    if _named(exc, "huggingface_hub.errors", "LocalEntryNotFoundError"):
        cause = exc.__cause__ or exc.__context__
        # Offline with nothing to resolve: OfflineModeIsEnabled (a ConnectionError subclass, so told apart
        # before any transport check) with no resolvable commit, or the offline flag with no cause at all.
        offline = _named(cause, "huggingface_hub.errors", "OfflineModeIsEnabled") or (
            cause is None
            # the values revisions.hub_offline() accepts; importing it would point errors below data
            and os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in {"1", "true", "yes", "on"}
        )
        if not offline and (
            _named(cause, "httpx", "TransportError")
            or _named(cause, "requests", "ConnectionError", "Timeout", "ConnectTimeout", "ReadTimeout")
            or isinstance(cause, ConnectionError | TimeoutError)
            or hub_down(cause)
        ):
            return ProviderError(
                name,
                hint="the Hugging Face Hub could not be reached; check connectivity and HF_ENDPOINT, then retry",
            )
        if offline:
            return offline_miss()
        if _named(cause, "huggingface_hub.errors", "FileMetadataError"):
            # the Hub answered, but without its headers: a mirror or proxy that is not a Hub endpoint
            return MissingInputError(
                name,
                hint="the endpoint answered without the Hub's metadata: check HF_ENDPOINT points to a Hub-compatible "
                "endpoint, and the proxy settings",
            )
        return MissingInputError(
            name,
            hint="the file is not in the local Hub cache and Hub access is off (HF_HUB_OFFLINE); "
            "unset HF_HUB_OFFLINE or download the file first",
        )
    if _named(exc, "huggingface_hub.errors", "HfHubHTTPError") and not _named(
        exc,
        "huggingface_hub.errors",
        "GatedRepoError",
        "RepositoryNotFoundError",
        "RevisionNotFoundError",
        "EntryNotFoundError",
    ):
        status = getattr(getattr(exc, "response", None), "status_code", 0)
        if status == 401:
            return CredentialsError(
                name, hint="check the token the request carries (HF_TOKEN) and the repository's access terms"
            )
        if status >= 500 or status == 429:
            # the Hub answered but is down or rate-limiting: the same retryable provider failure as a connection one
            return ProviderError(
                name,
                hint="the Hugging Face Hub could not be reached; check connectivity and HF_ENDPOINT, then retry",
            )
    if isinstance(exc, FileNotFoundError):
        details = {"path": str(exc.filename)} if exc.filename is not None else None
        return MissingInputError(str(exc), hint="check the path, or run the step that produces it", details=details)
    if _named(exc, "pydantic", "ValidationError"):
        from rcp_ndcg.support.config import config_error

        return config_error(exc)
    if _named(exc, "yaml", "YAMLError"):
        return ConfigError(name, hint="fix the config value the message names")
    if _named(exc, "json", "JSONDecodeError"):
        return DataError(name, hint="the input file is not valid JSON; check the file the command read")
    if _named(exc, "openai", "AuthenticationError", "PermissionDeniedError"):
        return CredentialsError(name, hint="check the API key variable the judge config names")
    if _named(exc, "huggingface_hub.errors", "GatedRepoError"):
        return CredentialsError(name, hint="accept the dataset's terms on the Hub and set HF_TOKEN")
    if _named(exc, "huggingface_hub.errors", "RepositoryNotFoundError", "RevisionNotFoundError", "EntryNotFoundError"):
        return MissingInputError(name, hint="check the dataset id, subset and revision")
    if (
        _named(exc, "httpx", "TransportError")
        or _named(exc, "requests", "ConnectionError", "Timeout", "ConnectTimeout", "ReadTimeout")
        or _named(exc, "openai", "APIConnectionError", "RateLimitError", "InternalServerError")
        or isinstance(exc, ConnectionError | TimeoutError)
    ):
        return ProviderError(
            name, hint="the endpoint failed; retry later, or lower judge.concurrency (--set judge.concurrency=N)"
        )
    return RcpNdcgError(name, hint="this is a bug: please report it with the traceback from --log-file")


__all__ = [
    "EXTRA_FOR_MODULE",
    "WARNING_CODES",
    "BackendUnavailableError",
    "CapabilityError",
    "ConfigError",
    "CredentialsError",
    "DataError",
    "DependencyError",
    "ExitCode",
    "IdentityError",
    "Interrupted",
    "MissingInputError",
    "ProviderError",
    "RequestRejectedError",
    "RcpNdcgError",
    "RcpNdcgWarning",
    "UsageError",
    "WarningCode",
    "classify",
    "dependency_error",
    "error_class",
]
