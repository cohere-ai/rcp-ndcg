"""Re-read a judgement store's stored answers with the current parser, into a new store; the judge is not called.

A store keeps every answer's raw text (``Judgement.response``). When the parser
changes, :func:`reparse` reads each stored answer again with
:data:`~rcp_ndcg.llm._parsing.common.PARSE_VERSION` and writes the result to a
new store: the family is the source's with the new parse version, so the new
records have their own family key and record ids and never pool with the old
ones. Records without an answer (a request the endpoint refused) are carried
over as they are, under the new family.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from rcp_ndcg_core.schemas import Family, InvalidCategory, Judgement, JudgementSet, judgement_record_id

from rcp_ndcg.errors import DataError, IdentityError, MissingInputError
from rcp_ndcg.llm._parsing.common import PARSE_VERSION, UnparseableAnswer
from rcp_ndcg.llm.client import Completion
from rcp_ndcg.llm.judging import PREPROCESSING_RECORD, WindowAnswer, parse_window, window_record
from rcp_ndcg.llm.schedule import schedule_key
from rcp_ndcg.llm.store import IDENTITY_FILE, PROMPTS_DIR, STAGES, JudgementStore
from rcp_ndcg.storage import local_dir
from rcp_ndcg.support.logging import get_logger

logger = get_logger(__name__)


def _reparsed(judgement: Judgement, family: Family, schedule_key: str, dataset_key: str) -> Judgement:
    """``judgement`` read again under ``family`` (its answer re-parsed when it has one).

    ``schedule_key`` is the digest of the stage's schedule, which keys a planned window; ``dataset_key`` is the
    digest of the store identity's dataset entry, which every record id names the corpus by (the source's, so a
    re-parsed record keeps its window).
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
        answer = parse_window(judgement.stage, judgement.query_id, completion, units, family.num_criteria)
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
    for stage in STAGES:
        entry = entries.get(stage)
        if entry is None or not source.path(stage).exists():
            continue
        family = Family.model_validate(entry["family"]).model_copy(update={"parse_version": PARSE_VERSION})
        identity = {**entry["identity"], "family": family.model_dump(mode="json")}
        target.claim(stage, identity, family, sources=entry.get("sources"))
        schedule = source.schedule(stage)
        if schedule is None:
            raise DataError(
                f"{source.identity_path} has no schedule for {stage}; the store's records cannot be re-keyed",
                hint="the store's identity entry is incomplete; judge the stage again into a new store",
            )
        from rcp_ndcg_core._hashing import hash_payload, short

        dataset_key = short(hash_payload(entry["identity"]["dataset"]), 16)
        records = source.records(stage)
        for judgement in records.values():
            target.append(_reparsed(judgement, family, schedule_key(schedule), dataset_key))
        logger.info("reparsed %d %s records of %s into %s", len(records), stage, source.root, target_root)
    if (source.root / PROMPTS_DIR).is_dir():
        shutil.copytree(source.root / PROMPTS_DIR, target_root / PROMPTS_DIR, dirs_exist_ok=True)
    census = source.root / PREPROCESSING_RECORD
    if census.exists():
        target_root.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(census, target_root / PREPROCESSING_RECORD)
    return target.read()


__all__ = ["PREPROCESSING_RECORD", "reparse"]
