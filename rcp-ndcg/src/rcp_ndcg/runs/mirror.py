"""Mirror a local run directory (or judgement store) to an object store while it grows, and restore it from there.

A run writes locally (:func:`rcp_ndcg.storage.local_dir`): a job that is preempted loses nothing its disk held. A
mirror makes what it wrote durable beyond that disk, continuously, so a new job on another node resumes where the
preempted one stopped. It is the one mechanism that copies a run to a bucket.

**Any fsspec filesystem is a mirror target** (``s3://``, ``gs://``, ``az://``, ``memory://``, a local or shared
path or ``file://``, or one your own package registers with ``fsspec.register_implementation`` or the
``fsspec.specs`` entry point), because the mirror uses exactly three of its operations: write an object
(``pipe_file`` on a remote target; a local or shared target publishes whole files atomically through
:func:`rcp_ndcg.storage.publish_bytes`), read an object (``cat_file``) and list a prefix (``ls``). A remote
target never asks whether an object exists and never appends; a local target creates the directories it writes
into and publishes each whole file with one temp-file rename. ``hf://`` works
but warns: every write to the Hub is a commit, and its rate limits make it a place to publish a finished run, not
to mirror a running one.

Layout under the mirror URI, relative paths as in the local directory:

* an **append-only JSONL** (a judgement store's ``tournament.jsonl`` and ``rubric.jsonl``, the text census
  ``preprocessing.jsonl``) is uploaded in immutable parts, ``<file>.parts/<start>-<end>`` with zero-padded byte
  offsets: every flush uploads the bytes appended since the last one, up to the last complete line, and never
  rewrites an uploaded part;
* every **other file** (``manifest.json``, ``identity.json``, ``run.yaml``, the calibration, the reports, the log) is
  uploaded whole whenever it changed, overwriting.

``work/`` (indices and caches the steps can recompute) and temporary files are not mirrored.

:func:`mirrored` flushes every ``interval_s`` seconds in a background thread, and once more when the block ends,
also on ``SIGTERM`` or ``SIGINT``; a failed flush is recorded in the state file (``run status`` shows it).
:func:`restore` rebuilds a missing, shorter or older local copy: a JSONL from its parts in offset order (their
contiguity checked), a missing file from its copy, and, when the mirror's ``manifest.json`` is newer than the local
one (another host ran the run further), every whole file that differs from the mirror's. The run's resume logic
then does the rest.
"""

from __future__ import annotations

import json
import os
import re
import signal
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from rcp_ndcg import storage
from rcp_ndcg.errors import DataError, Interrupted
from rcp_ndcg.runs.layout import MANIFEST_NAME, WORK_DIR
from rcp_ndcg.support.logging import get_logger
from rcp_ndcg.support.urls import redact_urls, safe_url

logger = get_logger(__name__)

#: Seconds between two flushes of a running mirror.
DEFAULT_INTERVAL_S = 60.0
#: The JSONL files the package only ever appends to: mirrored in parts.
APPEND_ONLY = frozenset({"tournament.jsonl", "rubric.jsonl", "preprocessing.jsonl"})
PARTS_SUFFIX = ".parts"
_PART = re.compile(r"^(\d{12})-(\d{12})$")
_SKIPPED_DIRS = (WORK_DIR,)
#: How many bytes before a JSONL's mirrored end are compared to detect a rewritten file.
_TAIL = 256


class MirrorState(BaseModel):
    """What a mirror last did (``run status`` shows it).

    ``remote`` is the display form of the mirror URI (:func:`~rcp_ndcg.support.urls.safe_url`): userinfo, query
    and fragment never reach the state file, ``run status --json`` or a log line. ``last_error`` is redacted the
    same way: a state file written before this rule holds the full URI in a flush error's text.
    """

    remote: str
    last_upload_at: datetime | None = None
    last_error: str | None = None
    lag_s: float | None = Field(default=None, description="Seconds since the last successful upload.")

    @field_validator("remote")
    @classmethod
    def _safe_remote(cls, value: str) -> str:
        return safe_url(value)

    @field_validator("last_error")
    @classmethod
    def _safe_last_error(cls, value: str | None) -> str | None:
        return None if value is None else redact_urls(value)


class Mirror:
    """One local directory and the URI it is mirrored to.

    Args:
        root: The local directory (a run directory or a judgement store).
        remote: The mirror URI (``gs://``, ``s3://``, ``memory://``, ...).
        state_file: Where the mirror records its last upload (default ``<root>/.mirror.json``).
    """

    def __init__(self, root: str | Path, remote: str, *, state_file: str | Path | None = None) -> None:
        self.root = Path(root)
        self.remote = remote.rstrip("/")
        if safe_url(self.remote) != self.remote:
            logger.warning(
                "[mirror] %s carries credentials in its URI; they are never recorded (run.yaml, the manifest, the "
                "state file, a log line) and the job must read them from the environment",
                safe_url(self.remote),
            )
        self._target = _Target(self.remote)
        self.state_file = Path(state_file) if state_file is not None else self.root / ".mirror.json"
        self._ends: dict[str, int] = {}
        self._tails: dict[str, bytes] = {}
        self._seen: dict[str, tuple[int, int]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Upload
    # ------------------------------------------------------------------

    def flush(self) -> None:
        """Upload what changed since the last flush, then record the upload time.

        Raises:
            DataError: an append-only file was rewritten (it no longer extends what the mirror holds).
        """
        with self._lock:
            for path in self._files():
                relative = path.relative_to(self.root).as_posix()
                if path.name in APPEND_ONLY:
                    self._append(path, relative)
                else:
                    self._upload(path, relative)
            self._record(error=None)

    def _files(self) -> list[Path]:
        if not self.root.is_dir():
            return []
        files = []
        for path in sorted(self.root.rglob("*")):
            parts = path.relative_to(self.root).parts
            if not path.is_file() or parts[0] in _SKIPPED_DIRS or path == self.state_file or path.suffix == ".tmp":
                continue
            files.append(path)
        return files

    def _append(self, path: Path, relative: str) -> None:
        start = self._ends.get(relative)
        if start is None:  # first contact: what the mirror holds, and its last bytes to compare the file with
            parts = _parts(self._target, relative)
            start = _end(parts)
            self._tails[relative] = self._target.read(parts[-1][2])[-_TAIL:] if parts else b""
        size = path.stat().st_size
        tail = self._tails.get(relative, b"")
        if size < start or _read(path, start - len(tail), start) != tail:
            raise DataError(
                f"{path} no longer extends what the mirror holds ({start} bytes): it was rewritten or superseded",
                hint="mirror the rewritten store to a new URI",
            )
        chunk = _read(path, start, size)
        end = start + chunk.rfind(b"\n") + 1  # up to the last complete line; a torn line waits
        if end > start:
            self._target.write(f"{relative}{PARTS_SUFFIX}/{start:012d}-{end:012d}", chunk[: end - start])
            self._tails[relative] = (tail + chunk[: end - start])[-_TAIL:]
        self._ends[relative] = end

    def _upload(self, path: Path, relative: str) -> None:
        stat = path.stat()
        signature = (stat.st_size, stat.st_mtime_ns)
        if self._seen.get(relative) == signature:
            return
        self._target.write(relative, path.read_bytes())
        self._seen[relative] = signature

    def _record(self, *, error: str | None) -> None:
        state = self.state()
        state = state.model_copy(
            update={
                "last_error": redact_urls(error) if error else None,
                **({} if error else {"last_upload_at": datetime.now(UTC)}),
            }
        )
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        # A temp file and a rename (the one storage helper): `run status` reads the state while a flush
        # rewrites it, and a rewrite in place would serve a partial JSON. Owner-only: the record names the
        # mirror and its errors.
        storage.publish_bytes(self.state_file, state.model_dump_json(exclude={"lag_s"}).encode("utf-8"), mode=0o600)

    def state(self) -> MirrorState:
        """The last recorded state, with the lag since the last upload."""
        return read_state(self.state_file) or MirrorState(remote=self.remote)

    # ------------------------------------------------------------------
    # Restore
    # ------------------------------------------------------------------

    def restore(self) -> list[str]:
        """Rebuild what the local directory lacks from the mirror; return the relative paths restored.

        A local append-only file shorter than its parts is rebuilt from them; a file missing locally is
        downloaded. When the local directory is behind the mirror (the mirror's ``manifest.json`` was updated
        later than the local one: a job elsewhere ran the run further), every other whole file that differs from
        the mirror's copy is replaced by it. Nothing the local directory holds beyond the mirror is touched.

        Raises:
            DataError: the parts of a file are not contiguous from offset 0 (a gap or an overlap).
        """
        restored = []
        remote = _remote_files(self._target)
        behind = self._behind(remote)
        for relative, kind in sorted(remote.items()):
            target = self.root / relative
            if kind == "parts":
                parts = _parts(self._target, relative)
                end = _end(parts)
                if target.exists() and target.stat().st_size >= end:
                    continue
                payload = b"".join(self._target.read(name) for _, _, name in parts)
            else:
                if target.exists() and not behind:
                    continue
                payload = self._target.read(relative)
                if target.exists() and target.read_bytes() == payload:
                    continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f"{target.name}.restore.tmp")
            # Created owner-only (no window at 0644 on a shared filesystem, and a stale temp from a killed
            # restore cannot hand its old mode to the restored file), then renamed over the target.
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
            temporary.replace(target)
            restored.append(relative)
        if restored:
            logger.info("[mirror] restored %d files of %s from %s", len(restored), self.root, safe_url(self.remote))
        return restored

    def read(self, relative: str) -> bytes | None:
        """The mirror's copy of the whole file ``relative`` (e.g. ``manifest.json``), or ``None`` when it has none."""
        try:
            return self._target.read(relative)
        except FileNotFoundError:
            return None

    def _behind(self, remote: dict[str, str]) -> bool:
        """Whether the mirror's manifest was updated after the local one (a directory without one is never)."""
        local = self.root / MANIFEST_NAME
        if remote.get(MANIFEST_NAME) != "file" or not local.is_file():
            return False
        mirrored_at = _updated_at(self._target.read(MANIFEST_NAME))
        local_at = _updated_at(local.read_bytes())
        return mirrored_at is not None and local_at is not None and mirrored_at > local_at


def read_state(state_file: str | Path) -> MirrorState | None:
    """A mirror's recorded state (``None`` when it never ran), with the lag since its last upload.

    An unparseable state file is a torn write (a reader racing a flush, or a writer the kernel killed): it is
    treated as "never ran" with a warning, as the store treats a torn identity -- ``run status`` must not crash
    on its own state file.
    """
    path = Path(state_file)
    if not path.is_file():
        return None
    try:
        state = MirrorState.model_validate_json(path.read_text(encoding="utf-8"))
    except ValueError as exc:  # ValidationError and UnicodeDecodeError are both ValueError
        # The pydantic text renders the input value; a legacy state file holds the full URI, credentials included.
        logger.warning(
            "%s does not parse (a torn mirror state write); treating the mirror as never run: %s",
            path,
            redact_urls(str(exc)),
        )
        return None
    if state.last_upload_at is None:
        return state
    return state.model_copy(update={"lag_s": (datetime.now(UTC) - state.last_upload_at).total_seconds()})


@contextmanager
def mirrored(
    root: str | Path,
    remote: str,
    *,
    interval_s: float = DEFAULT_INTERVAL_S,
    state_file: str | Path | None = None,
) -> Iterator[Mirror]:
    """Restore ``root`` from ``remote``, then mirror it while the block runs.

    A background thread flushes every ``interval_s`` seconds; the block's end flushes once more, also when
    ``SIGTERM`` (raised as :class:`~rcp_ndcg.errors.Interrupted`) or ``SIGINT`` stops it.
    """
    mirror = Mirror(root, remote, state_file=state_file)
    mirror.restore()
    stop = threading.Event()

    def loop() -> None:
        while not stop.wait(interval_s):
            try:
                mirror.flush()
            except Exception as exc:  # the run goes on; the next flush retries, run status shows the error
                logger.warning("[mirror] flush to %s failed: %s", safe_url(remote), redact_urls(str(exc)))
                mirror._record(error=f"{type(exc).__name__}: {exc}")

    thread = threading.Thread(target=loop, name="rcp-ndcg-mirror", daemon=True)
    previous = _on_sigterm()
    thread.start()
    try:
        yield mirror
    finally:
        stop.set()
        thread.join()
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)
        try:
            mirror.flush()
        except Exception as exc:  # recorded for run status, then raised: the mirror does not hold the end
            mirror._record(error=f"{type(exc).__name__}: {exc}")
            raise


def _on_sigterm():
    """Turn SIGTERM into :class:`Interrupted` (so the final flush runs); the previous handler, or ``None``."""
    if threading.current_thread() is not threading.main_thread():
        return None

    def terminated(*_: object) -> None:
        raise Interrupted("terminated (SIGTERM)", hint="the mirror holds everything written: resume the run")

    return signal.signal(signal.SIGTERM, terminated)


def restore(root: str | Path, remote: str) -> list[str]:
    """Rebuild what ``root`` lacks from the mirror at ``remote`` (:meth:`Mirror.restore`)."""
    return Mirror(root, remote).restore()


def check_target(remote: str) -> None:
    """Refuse a mirror URI no installed filesystem serves, before anything is written.

    A URI that carries credentials (userinfo, a query or a fragment) is accepted -- the job must reach the store
    -- and warned about: they are never recorded (``run.yaml``, the manifest, the state file, a log line) and the
    job must read them from the environment instead.

    Raises:
        DependencyError: no fsspec filesystem is installed or registered for the URI's protocol.
    """
    if safe_url(remote) != remote:
        logger.warning(
            "[mirror] %s carries credentials in its URI; they are never recorded (run.yaml, the manifest, the "
            "state file, a log line) and the job must read them from the environment",
            safe_url(remote),
        )
    storage.filesystem(remote, **_options(remote))


def _options(remote: str) -> dict[str, bool]:
    """The filesystem options of a mirror URI: a local (or shared POSIX) target creates its directories."""
    return {"auto_mkdir": True} if storage.protocol_of(remote) in ("file", "local") else {}


def _updated_at(payload: bytes) -> datetime | None:
    """A manifest's ``updated_at`` (``None`` when it records none)."""
    value = json.loads(payload).get("updated_at")
    return datetime.fromisoformat(value) if isinstance(value, str) else None


class _Target:
    """The mirror URI's filesystem, used through exactly three operations: write, read and list."""

    def __init__(self, remote: str) -> None:
        if storage.protocol_of(remote) == "hf":
            logger.warning(
                "[mirror] %s: every write to the Hugging Face Hub is a commit, and its rate limits apply; mirror to "
                "an object store and publish the finished run to the Hub instead",
                safe_url(remote),
            )
        self.fs = storage.filesystem(remote, **_options(remote))  # a missing or unknown one: DependencyError
        self.root = type(self.fs)._strip_protocol(remote).rstrip("/")
        # The protocol of the *mirror URI*, not of the stripped root: `memory://` and a custom filesystem's
        # `_strip_protocol` drop their scheme, and the stripped path would read as local.
        self._remote = storage.is_remote(remote)

    def write(self, relative: str, payload: bytes) -> None:
        uri = f"{self.root}/{relative}"
        if self._remote:
            self.fs.pipe_file(uri, payload)
        else:
            # A local or shared POSIX target: publish (temp file + rename), so a concurrent restore() never
            # reads a partial file. An object store publishes each object whole anyway.
            storage.publish_bytes(uri, payload)

    def read(self, relative: str) -> bytes:
        return self.fs.cat_file(f"{self.root}/{relative}")

    def list(self, relative: str = "") -> list[tuple[str, bool]]:
        """``(relative path, is a directory)`` of the entries under ``relative`` (none when it does not exist)."""
        base = f"{self.root}/{relative}".rstrip("/")
        try:
            entries = self.fs.ls(base, detail=True)
        except FileNotFoundError:
            return []
        prefix = f"{self.root}/"
        out = []
        for entry in entries:
            name = str(entry["name"]).rstrip("/")
            relative_name = name[len(prefix) :] if name.startswith(prefix) else name.lstrip("/")
            out.append((relative_name, entry.get("type") == "directory"))
        return out


def _remote_files(target: _Target) -> dict[str, str]:
    """``{relative path: "parts" | "file"}`` of everything the mirror holds (a walk of ``ls``)."""
    files: dict[str, str] = {}
    pending = [""]
    while pending:
        for relative, directory in target.list(pending.pop()):
            if relative.endswith(".tmp"):
                continue  # a publish temp (`.storage.publish`) is never part of the run
            if relative.endswith(PARTS_SUFFIX):
                files[relative[: -len(PARTS_SUFFIX)]] = "parts"
            elif directory:
                pending.append(relative)
            else:
                files[relative] = "file"
    return files


def _parts(target: _Target, relative: str) -> list[tuple[int, int, str]]:
    """``(start, end, relative path)`` of a file's parts in offset order.

    Raises:
        DataError: the parts leave a gap or overlap.
    """
    parts = []
    for name, _ in target.list(f"{relative}{PARTS_SUFFIX}"):
        match = _PART.match(name.rpartition("/")[2])
        if match:
            parts.append((int(match.group(1)), int(match.group(2)), name))
    parts.sort()
    expected = 0
    for start, end, name in parts:
        if start != expected or end <= start:
            raise DataError(
                f"the mirror of {relative} is not contiguous: part {name} starts at {start}, expected {expected}",
                hint="the mirror is damaged; restore from another copy or start a new run",
            )
        expected = end
    return parts


def _end(parts: list[tuple[int, int, str]]) -> int:
    return parts[-1][1] if parts else 0


def _read(path: Path, start: int, end: int) -> bytes:
    with path.open("rb") as handle:
        handle.seek(start)
        return handle.read(max(end - start, 0))


__all__ = [
    "APPEND_ONLY",
    "DEFAULT_INTERVAL_S",
    "Mirror",
    "MirrorState",
    "check_target",
    "mirrored",
    "read_state",
    "restore",
]
