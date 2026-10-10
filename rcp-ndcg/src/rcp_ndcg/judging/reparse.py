"""Re-read a judgement store's stored answers with the current parser, into a new store; the judge is not called.

A store keeps every answer's raw text (``Judgement.response``). When the parser
changes, :func:`reparse` reads each stored answer again with
:data:`~rcp_ndcg.judging._parsing.common.PARSE_VERSION` and writes the result to a
new store: the family is the source's with the new parse version, so the new
records have their own family key and record ids and never pool with the old
ones. Records without an answer (a request the endpoint refused) are carried
over as they are, under the new family.

A source already at the current parse version has nothing to re-parse (the copy
would share its family key and record ids), and a source from a newer checkout
cannot be downgraded to this parser's version: both are refused with an
:class:`~rcp_ndcg.errors.IdentityError` before anything is written.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from rcp_ndcg_core.schemas import InvalidCategory, Judgement, JudgementFamily, JudgementSet, Stage, judgement_record_id

from rcp_ndcg.errors import DataError, IdentityError, MissingInputError
from rcp_ndcg.judging._parsing.common import PARSE_VERSION, UnparseableAnswer
from rcp_ndcg.judging.client import Completion
from rcp_ndcg.judging.judging import PREPROCESSING_RECORD, WindowAnswer, parse_window, window_record
from rcp_ndcg.judging.schedule import schedule_key
from rcp_ndcg.judging.store import IDENTITY_FILE, PROMPTS_DIR, STAGES, JudgementStore
from rcp_ndcg.storage import local_dir
from rcp_ndcg.support.logging import get_logger

logger = get_logger(__name__)


def _reparsed(
    judgement: Judgement, family: JudgementFamily, schedule_key: str, dataset_key: str, example: dict[str, Any] | None
) -> Judgement:
    """``judgement`` read again under ``family`` (its answer re-parsed when it has one).

    ``schedule_key`` is the digest of the stage's schedule, which keys a planned window; ``dataset_key`` is the
    digest of the store identity's dataset entry, which every record id names the corpus by (the source's, so a
    re-parsed record keeps its window). ``example`` is the source prompt's own worked example, when the store
    kept its text: an answer equal to it is refused exactly as the judging pass refuses it.
    """
    units = [placement.unit_id for placement in judgement.placements]
    record_id = judgement_record_id(
        family.key,
        judgement.query_id,
        judgement.stage,
        judgement.window_seq,
        units,
        dataset=dataset_key,
        schedule_key=schedule_key,
    )
    if judgement.response is None:
        return judgement.model_copy(update={"record_id": record_id, "family_key": family.key})
    completion = Completion(
        response=judgement.response,
        finish_reason=judgement.finish_reason,
        input_tokens=judgement.input_tokens,
        output_tokens=judgement.output_tokens,
    )
    answer: WindowAnswer | None = None
    failure: tuple[str, InvalidCategory] | None = None
    try:
        answer = parse_window(
            judgement.stage, judgement.query_id, completion, units, family.num_criteria, example=example
        )
    except UnparseableAnswer as exc:
        failure = (f"{type(exc).__name__}: {exc.reason}", exc.category)
    return window_record(
        record_id=record_id,
        dataset=judgement.dataset,
        query_id=judgement.query_id,
        stage=judgement.stage,
        family=family,
        window_seq=judgement.window_seq,
        phase=judgement.phase,
        placements=[(placement.unit_id, placement.doc_id) for placement in judgement.placements],
        answer=answer,
        completion=completion,
        failure=failure,
        recorded_at=judgement.recorded_at,
    )


def _check_target(source: JudgementStore, out: Path) -> None:
    if out.resolve() == source.root.resolve():
        raise IdentityError(
            f"reparse writes a new store and never into its source ({source.root})",
            hint="pass another --out directory",
        )
    present = [name for name in (IDENTITY_FILE, *(f"{stage}.jsonl" for stage in STAGES)) if (out / name).exists()]
    if present:
        raise IdentityError(
            f"{out} already holds a judgement store ({', '.join(present)}); reparse writes a new one",
            hint="pass an empty or new --out directory",
            details={"out": str(out), "present": present},
        )


def _check_parse_version(source: JudgementStore, stage: str, entry: dict[str, Any]) -> None:
    """Refuse a source this parser cannot usefully re-read: the same version (a same-key copy) or a newer one
    (a silent downgrade of answers produced under a newer schema).

    Raises:
        IdentityError: the source's parse version is the current one (nothing to re-parse) or newer.
    """
    source_version = int(entry["family"]["parse_version"])
    if source_version > PARSE_VERSION:
        raise IdentityError(
            f"{source.identity_path} was written by a newer parser (parse_version {source_version}); this "
            f"checkout parses version {PARSE_VERSION}",
            hint="upgrade rcp-ndcg, or judge the stage again into a new store",
            details={"stage": stage, "source_parse_version": source_version, "parse_version": PARSE_VERSION},
        )
    if source_version == PARSE_VERSION:
        raise IdentityError(
            f"{stage} of {source.root} is already parsed with the current parser (version {PARSE_VERSION}): "
            "re-parsing it would produce the same family key and record ids",
            hint="there is nothing to re-parse; use the store as it is",
            details={"stage": stage, "parse_version": PARSE_VERSION},
        )


def _prompt_example(root: Path, prompt_hash: str) -> dict[str, Any] | None:
    """The source prompt's own worked example, read from the store's ``prompts/<sha256>.txt``, or ``None``.

    The store keeps the prompt's text by its hash, so a reparse applies the same example check the judging pass
    applied; a store without the file (an interrupted copy) re-parses without the check rather than failing.
    """
    path = root / PROMPTS_DIR / f"{prompt_hash}.txt"
    if not path.is_file():
        return None
    from rcp_ndcg.judging.prompts import Prompt

    return Prompt(name=str(path), text=path.read_text(encoding="utf-8")).worked_example


def reparse(store: str | Path, out: str | Path) -> JudgementSet:
    """Re-parse every stored answer of a judgement store with the current parser, into a new store at ``out``.

    Every record keeps its window (query, stage, sequence, placements, phase) and its answer; its observation,
    validity and invalid category come from parsing the answer again. The new store's identity is the
    source's with the family's ``parse_version`` set to the current one. No judge is called.

    Args:
        store: The source judgement store (a directory ``judge()`` wrote).
        out: The new store's directory; it must not hold a store already.

    Returns:
        The new store's judgements, with their families.

    Raises:
        MissingInputError: ``store`` is not a judgement store.
        IdentityError: ``out`` is the source store, or already holds a store.
        ConfigError: ``out`` is a remote URI (a store is written locally).
    """
    local_dir(out, "a judgement store")
    source = JudgementStore(store)
    if not source.identity_path.exists():
        raise MissingInputError(
            f"{store} is not a judgement store (no {IDENTITY_FILE})",
            hint="point at the directory judge() or `rcp-ndcg judge` wrote",
            cli_hint="point at the --out directory of `rcp-ndcg judge tournament|rubric`",
        )
    target_root = Path(out)
    _check_target(source, target_root)
    target = JudgementStore(target_root)
    entries = source.identities()
    stages: list[Stage] = [stage for stage in STAGES if entries.get(stage) is not None and source.path(stage).exists()]
    # Every stage's parse version is checked before anything is written: a partially written target store
    # would be refused by its own `_check_target` on a retry.
    for stage in stages:
        _check_parse_version(source, stage, entries[stage])
    for stage in stages:
        entry = entries[stage]
        family = JudgementFamily.model_validate(entry["family"]).model_copy(update={"parse_version": PARSE_VERSION})
        identity = {**entry["identity"], "family": family.model_dump(mode="json")}
        target.claim(stage, identity, family, sources=entry.get("sources"))
        schedule = source.schedule(stage)
        if schedule is None:
            raise DataError(
                f"{source.identity_path} has no schedule for {stage}; the store's records cannot be re-keyed",
                hint="the store's identity entry is incomplete; judge the stage again into a new store",
            )
        from rcp_ndcg.support.identity import hash_payload, short

        dataset_key = short(hash_payload(entry["identity"]["dataset"]), 16)
        example = _prompt_example(source.root, family.prompt_hash)
        records = source.records(stage)
        for judgement in records.values():
            target.append(_reparsed(judgement, family, schedule_key(schedule), dataset_key, example))
        logger.info("reparsed %d %s records of %s into %s", len(records), stage, source.root, target_root)
    if (source.root / PROMPTS_DIR).is_dir():
        shutil.copytree(source.root / PROMPTS_DIR, target_root / PROMPTS_DIR, dirs_exist_ok=True)
    census = source.root / PREPROCESSING_RECORD
    if census.exists():
        # Re-serialized through the one census-row reader: a torn last row is cut, a complete row that is not a
        # census row is refused here instead of being copied into the new store.
        target_root.mkdir(parents=True, exist_ok=True)
        from rcp_ndcg.storage.census import read_census_rows

        with (target_root / PREPROCESSING_RECORD).open("w", encoding="utf-8") as handle:
            for row in read_census_rows(census):
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    return target.read()


__all__ = ["PREPROCESSING_RECORD", "reparse"]
