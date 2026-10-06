"""The append-only judgement store: one JSONL file per stage and the identity it answers for.

Layout of a store directory::

    <out>/
      identity.json      {"schema": "rcp-ndcg.judgement-store.v1", "stages": {stage: {...}}}
      tournament.jsonl   one Judgement per line (schema rcp-ndcg.judgement.v1)
      rubric.jsonl
      preprocessing.jsonl  every text cut the judging passes made: each distinct load-time cut of a
                           document (doc_policy) once, and each window presentation shortened to
                           the window's text budget (window_budget)
      prompts/<sha256>.txt the text of every prompt the store was judged with, by its hash
                           (the family's prompt_hash), so the instrument survives its file

Each record is keyed by its ``record_id`` (a digest of the family, query,
stage, the dataset's identity key, the window position and the ids shown; a
planned window by the schedule instead of the position), and a record is only
ever appended:
re-running a judging pass reads the records present and asks the judge only for
the windows that are missing, so a resumed or re-judged pass needs no merge step.

``identity.json`` records, per stage, the identity of the judging pass that
writes into the file (the family, the judge's content fields, the schedule, the
dataset and the preprocessing). A pass with a different identity is refused
with :class:`~rcp_ndcg.errors.IdentityError` naming the differing fields, unless
forced: then the old stage file moves to ``<out>/.superseded/<timestamp>/``.
Each stage's entry also records its family key, when it was claimed, the
package version and ``engines``, what the endpoints said they serve
(:class:`~rcp_ndcg.llm.client.EngineInfo`, one entry per distinct report), for a
person reading the store; no code reads those four.
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from rcp_ndcg_core.schemas import Family, Judgement, JudgementSet, Stage, supersedes

from rcp_ndcg.errors import DataError, IdentityError
from rcp_ndcg.llm.client import EngineInfo
from rcp_ndcg.storage import atomic_write
from rcp_ndcg.support.identity import identity_differences
from rcp_ndcg.support.logging import get_logger

if TYPE_CHECKING:
    from rcp_ndcg.llm.schedule import RubricSchedule, TournamentSchedule

logger = get_logger(__name__)

STORE_SCHEMA = "rcp-ndcg.judgement-store.v1"
IDENTITY_FILE = "identity.json"
SUPERSEDED_DIR = ".superseded"
#: The directory of the prompts' texts, by hash.
PROMPTS_DIR = "prompts"
STAGES: tuple[Stage, ...] = ("tournament", "rubric")


class StageEntry(BaseModel):
    """One stage's entry in a store's ``identity.json``: the identity its judgements answer for, and when and by
    what it was claimed.

    Attributes:
        identity: The identity of the judging pass that writes the stage; a later pass into the store is compared
            with it field by field. Its keys: ``stage``, ``family`` (the Family), ``judge`` (the judge config's
            content fields), ``schedule`` (its numbers; the prompt is the family's ``prompt_hash``), ``dataset``
            (its name, and for a loaded dataset its URI, local paths absolute, and resolved revision),
            ``preprocessing`` (the effective policy, with the tokenizer's SHA-256), and for the tournament
            ``bt_l2`` (the penalty of the live Bradley-Terry fit that chose its adaptive windows). Only content
            enters it: the names the inputs were given are ``sources``.
        family: The family of the stage's judgements.
        family_key: Its key.
        created_at: When the stage was claimed (ISO 8601, UTC).
        package_version: The rcp-ndcg version that claimed it.
        engines: What the endpoints said they serve, one entry per distinct report (runtime information, never part
            of the identity).
        sources: How the claiming pass named its inputs: ``prompt`` (the prompt's name or path),
            ``schedule_prompt`` (the schedule's ``prompt``, which :meth:`JudgementStore.schedule` gives back) and
            ``tokenizer`` (as the judge config named it). Runtime information, never compared.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    identity: dict[str, Any]
    family: Family
    family_key: str
    created_at: str
    package_version: str
    engines: list[EngineInfo] = Field(default_factory=list)
    sources: dict[str, Any] = Field(default_factory=dict)


class StoreIdentity(BaseModel):
    """``identity.json`` of a judgement store (``rcp-ndcg.judgement-store.v1``): an entry per stage it holds.

    Attributes:
        stages: ``{stage: StageEntry}`` for each stage written into the store.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", validate_by_name=True, serialize_by_alias=True)

    schema_name: Literal["rcp-ndcg.judgement-store.v1"] = Field(default=STORE_SCHEMA, alias="schema")
    stages: dict[Stage, StageEntry]


class JudgementStore:
    """A judgement store directory (see the module docstring)."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._checked_tail: set[Path] = set()

    def path(self, stage: Stage) -> Path:
        """The JSONL file of one stage."""
        return self.root / f"{stage}.jsonl"

    @property
    def identity_path(self) -> Path:
        return self.root / IDENTITY_FILE

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    def identities(self) -> dict[str, dict[str, Any]]:
        """``{stage: entry}`` of ``identity.json`` (empty when the store is new)."""
        if not self.identity_path.exists():
            return {}
        payload = json.loads(self.identity_path.read_text(encoding="utf-8"))
        tag = payload.get("schema") if isinstance(payload, dict) else None
        if tag != STORE_SCHEMA:
            raise DataError(
                f"{self.identity_path} is not a judgement store identity ({tag!r})",
                hint="point at the directory judge() wrote",
                cli_hint="point at the --out directory of `rcp-ndcg judge tournament|rubric`",
            )
        try:
            StoreIdentity.model_validate(payload)
        except ValidationError as exc:
            raise DataError(
                f"{self.identity_path} does not match {STORE_SCHEMA}: {exc.error_count()} problem(s)",
                hint="compare it with `rcp-ndcg schema show judgement-store`",
                details={"errors": exc.errors(include_url=False, include_context=False, include_input=False)},
            ) from exc
        return dict(payload["stages"])

    def check(self, stage: Stage, identity: dict[str, Any], *, force: bool = False) -> bool:
        """Whether a pass of ``identity`` may write ``stage`` here, and must claim it first; writes nothing.

        Returns:
            ``False`` when the stage is already claimed under ``identity``, ``True`` when it is new (or held under
            another identity and ``force`` supersedes it).

        Raises:
            IdentityError: the stage already holds judgements of another identity (and ``force`` is false).
        """
        present = self.identities().get(stage)
        if present is None:
            return True
        differences = identity_differences(present["identity"], identity)
        if not differences:
            return False
        if not force:
            raise IdentityError(
                f"{self.path(stage)} holds {stage} judgements of another identity: {'; '.join(differences[:8])}",
                hint="write to a new output directory, or pass force=True to supersede them",
                cli_hint="write to a new --out directory, or pass --force to supersede them",
                details={"stage": stage, "differences": differences},
            )
        return True

    def claim(
        self,
        stage: Stage,
        identity: dict[str, Any],
        family: Family,
        *,
        force: bool = False,
        sources: dict[str, Any] | None = None,
    ) -> None:
        """Record that ``stage`` of this store is written under ``identity`` (``sources``: how its inputs were named).

        Raises:
            IdentityError: the stage already holds judgements of another identity (and ``force`` is false).
        """
        import rcp_ndcg

        with self._identity_lock():
            if not self.check(stage, identity, force=force):
                return
            entries = self.identities()
            if stage in entries:
                self._supersede(stage)
            entries[stage] = {
                "identity": identity,
                "family": family.model_dump(mode="json"),
                "family_key": family.key,
                "created_at": datetime.now(UTC).isoformat(),
                "package_version": rcp_ndcg.__version__,
                **({"sources": sources} if sources else {}),
            }
            self._write_identities(entries)

    def note_engines(self, stage: Stage, engines: Sequence[Any]) -> None:
        """Add what the endpoints reported (:class:`~rcp_ndcg.llm.client.EngineInfo`) to ``stage``'s entry.

        Runtime information: it sits beside the identity and never enters it, so a resumed pass against another
        engine version adds a report instead of being refused. A report already recorded is not added again, and one
        that completes a recorded report replaces it.
        """
        with self._identity_lock():
            entries = self.identities()
            entry = entries.get(stage)
            if entry is None or not engines:
                return
            recorded = list(entry.get("engines", []))
            for engine in engines:
                report = engine.model_dump(mode="json", exclude_none=True, exclude_defaults=True)
                if any(report.items() <= known.items() for known in recorded):
                    continue
                # A report that completes one recorded before (the fingerprint of a first answer) replaces it.
                recorded = [known for known in recorded if not known.items() <= report.items()] + [report]
            if recorded == entry.get("engines"):
                return
            entries[stage] = {**entry, "engines": recorded}
            self._write_identities(entries)

    @contextmanager
    def _identity_lock(self) -> Iterator[None]:
        """Serialize the identity file's read-modify-write between processes.

        Two passes claiming the two stages of one fresh store at the same time would otherwise lose one
        stage's entry (the last full-file write clobbers the other), and the losing pass crashes on ``read()``;
        the store's resume design invites overlapping passes, so the read and the write of every claim or
        engine note happen under one advisory lock. It is an ``flock`` on the store directory itself: no lock
        file joins the layout, and it is released by closing, so a killed process leaves nothing behind.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.root, os.O_RDONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)

    def _write_identities(self, entries: dict[str, dict[str, Any]]) -> None:
        payload = {"schema": STORE_SCHEMA, "stages": entries}
        StoreIdentity.model_validate(payload)  # the file is what the exported schema describes
        # A temp file and a rename (the one storage helper): `run status` counts a running pass's windows while
        # the pass claims its stages, and a rewrite in place would serve it an empty or partial file
        # (tests/llm/test_store.py races a claim against a reader). The temp name carries the process id and a
        # random suffix, so two writers never share one.
        atomic_write(
            self.identity_path,
            lambda tmp: tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"),
        )

    def schedule(self, stage: Stage) -> TournamentSchedule | RubricSchedule | None:
        """The schedule ``stage`` of this store was judged with (``None``: unclaimed): its numbers from the identity,
        and its ``prompt`` as the claiming pass named it."""
        from rcp_ndcg.llm.schedule import RubricSchedule, TournamentSchedule

        entry = self.identities().get(stage)
        if entry is None:
            return None
        kind = TournamentSchedule if stage == "tournament" else RubricSchedule
        prompt = (entry.get("sources") or {}).get("schedule_prompt")
        return kind.model_validate({**entry["identity"]["schedule"], "prompt": prompt})

    def keep_prompt(self, text: str) -> Path:
        """Store a prompt's text as ``prompts/<sha256>.txt`` (once) and return that path.

        The stored text is verified against its name: a torn write (a process killed mid-write) left a file
        whose content contradicted its filename forever, silently corrupting the store's provenance claim; one
        whose hash disagrees with its stem is rewritten.
        """
        from rcp_ndcg.support.identity import hash_text

        digest = hash_text(text)
        path = self.root / PROMPTS_DIR / f"{digest}.txt"
        if path.exists() and path.read_text(encoding="utf-8") == text:
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(path, lambda tmp: tmp.write_text(text, encoding="utf-8"))
        return path

    def _supersede(self, stage: Stage) -> None:
        target = self.root / SUPERSEDED_DIR / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        target.mkdir(parents=True, exist_ok=True)
        if self.path(stage).exists():
            shutil.move(str(self.path(stage)), target / self.path(stage).name)
        shutil.copyfile(self.identity_path, target / IDENTITY_FILE)
        logger.warning("superseded %s judgements of %s -> %s", stage, self.root, target)

    # ------------------------------------------------------------------
    # Records
    # ------------------------------------------------------------------

    def records(self, stage: Stage) -> dict[str, Judgement]:
        """``{record_id: Judgement}`` of one stage, in file order.

        Of several copies of one id, :func:`~rcp_ndcg_core.schemas.supersedes` picks the one kept, as
        :meth:`~rcp_ndcg_core.schemas.JudgementSet.merge` does: a valid copy beats an invalid one, then the later
        (a window a pass resumed asks again only while its record is invalid).
        """
        path = self.path(stage)
        records: dict[str, Judgement] = {}
        if not path.exists():
            return records
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    judgement = Judgement.model_validate_json(line)
                except ValueError as exc:
                    if not line.endswith("\n"):
                        # A torn last line (the process died mid-write): the window is asked again.
                        logger.warning("%s:%d: ignoring a torn last record", path, number)
                        continue
                    raise DataError(f"{path}:{number}: not a judgement record: {exc}") from exc
                present = records.get(judgement.record_id)
                if present is None or supersedes(judgement, present):
                    records[judgement.record_id] = judgement
        return records

    def append(self, judgement: Judgement) -> None:
        """Append one record (one line, flushed).

        Overlapping passes append to one file: the first append's torn-tail repair truncates to the last
        complete line, and a peer's in-flight line is exactly what that truncation would cut -- so the tail
        repair and the append hold the store's advisory lock, like every other writer of this directory."""
        with self._identity_lock():
            path = self.path(judgement.stage)
            path.parent.mkdir(parents=True, exist_ok=True)
            if path not in self._checked_tail:
                self._checked_tail.add(path)
                _drop_torn_tail(path)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(judgement.model_dump_json() + "\n")
                handle.flush()

    def read(self, stage: Stage | None = None) -> JudgementSet:
        """The store's judgements (of one stage, or of every stage) with their families.

        Raises:
            DataError: a stage file without its identity entry.
        """
        entries = self.identities()
        judgements: list[Judgement] = []
        families: dict[str, Family] = {}
        for name in STAGES if stage is None else (stage,):
            if not self.path(name).exists():
                continue
            entry = entries.get(name)
            if entry is None:
                raise DataError(f"{self.path(name)} has no entry in {self.identity_path}; it cannot be attributed")
            family = Family.model_validate(entry["family"])
            families[family.key] = family
            judgements.extend(self.records(name).values())
        return JudgementSet(judgements=tuple(judgements), families=families)


def records_stored(path: str | Path) -> int:
    """The records a stage file holds (its non-empty lines): the progress an estimate and ``run status`` report.

    The one count of a store file's lines: an estimate's note and a run's progress used to count twice, and one
    copy drifting (skipping comments, say) would report different progress for the same file.
    """
    path = Path(path)
    if not path.is_file():
        return 0
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _drop_torn_tail(path: Path) -> None:
    """Cut a last line the writer did not finish (a process killed mid-write), so appends start on a fresh line.

    The discipline's one home is :func:`rcp_ndcg.data.preprocess.drop_torn_last_line`; this wrapper keeps the
    store's message and only logs when there is something to cut.
    """
    from rcp_ndcg.data.preprocess import drop_torn_last_line

    if not path.exists():
        return
    size = path.stat().st_size
    with path.open("r+b") as handle:
        handle.seek(size - 1)
        if handle.read(1) == b"\n":
            return
    drop_torn_last_line(path)
    logger.warning("%s: dropping a torn last record (%d bytes)", path, size)


__all__ = [
    "IDENTITY_FILE",
    "PROMPTS_DIR",
    "STORE_SCHEMA",
    "SUPERSEDED_DIR",
    "JudgementStore",
    "StageEntry",
    "StoreIdentity",
    "records_stored",
]
