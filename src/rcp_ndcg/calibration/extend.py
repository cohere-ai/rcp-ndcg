"""Adding documents to a calibration without refitting it.

Refitting after a failed window, a hydration miss or a new candidate would move
every published ability. The two primitives here add documents with the
calibration frozen instead, and return an :class:`Extension`; nothing already in
the calibration moves (:meth:`Calibration.extended` adds the records):

* :func:`score_documents` -- documents judged again on the rubric alone, each scored
  from its own answers with the item parameters frozen (EAP under the
  calibration's ability prior, posterior SD as the standard error);
* :func:`insert_documents` -- new documents judged in tournament windows against
  documents the calibration already scores, each estimated with every existing
  ability frozen (conditional MLE), then mapped onto the calibrated scale by its
  query's ``(tau, alpha)``. :func:`select_opponents` says whom to judge it against.
  Insertion refits the calibration's own tournament windows first, and its anchor
  report says whether they still reproduce the calibration.

The judgements must come from the instrument the calibration was fitted with
(same family); anything else is refused, because an ability measured with another
rubric or judge is on another scale.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from datetime import UTC, datetime
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from rcp_ndcg_core.irt import (
    DEFAULT_SE_TARGET,
    MIN_OPPONENTS,
    ScaleMovementError,
    UnidentifiableInsertion,
    insert_document,
    score_document,
)
from rcp_ndcg_core.irt._insertion import select_opponents as _opponent_kernel
from rcp_ndcg_core.schemas import DocumentEstimate, ItemParams, JudgementSet

from rcp_ndcg.calibration._projection import (
    check_criteria,
    namespace,
    rubric_observations,
    split_namespace,
    tournament_comparisons,
)
from rcp_ndcg.calibration.fit import Calibration, population_prior
from rcp_ndcg.errors import DataError, IdentityError
from rcp_ndcg.support.identity import hash_payload, short

#: Default tolerance of the anchor check, in gain units ([0, 1]).
MAX_GAIN_SHIFT = 0.01


class ExtensionRecord(BaseModel):
    """One document added to a calibration: one line of a calibration's ``extensions.jsonl``
    (``rcp-ndcg.extension-record.v1``).

    Attributes:
        source: ``"scored"`` (from its own rubric answers, items frozen: the posterior mean and SD) or
            ``"inserted"`` (from tournament comparisons, opponents frozen: the conditional MLE).
        dataset: The dataset of the query.
        query_id: The query (not namespaced).
        doc_id: The document.
        estimate: The ability on the calibration's scale (logits), its standard error and information, and
            its flags. An inserted document has no flag set: an insertion that fails a check is refused.
        calibration: The fingerprint of the calibration it was computed against.
        judgements: Digest of the judgement records it was computed from.
        created_at: When it was computed.
        record_key: Digest of the slot, the source, the calibration, the judgements and the estimate.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, validate_by_name=True, serialize_by_alias=True)

    schema_name: Literal["rcp-ndcg.extension-record.v1"] = Field(default="rcp-ndcg.extension-record.v1", alias="schema")
    source: Literal["scored", "inserted"]
    dataset: str
    query_id: str
    doc_id: str
    estimate: DocumentEstimate
    calibration: str
    judgements: str
    created_at: datetime
    record_key: str = ""


class AnchorReport(BaseModel):
    """Do the fitted tournament windows, refitted, reproduce the calibration's abilities? Old against new.

    Attributes:
        documents_checked: Documents compared.
        max_abs_theta_shift: Largest ability movement, logits.
        max_abs_gain_shift: Largest gain movement, gain units.
        max_gain_shift: The tolerance, gain units.
        ok: Whether ``max_abs_gain_shift <= max_gain_shift``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    documents_checked: int
    max_abs_theta_shift: float
    max_abs_gain_shift: float
    max_gain_shift: float
    ok: bool


class Extension(BaseModel):
    """The documents :func:`score_documents` or :func:`insert_documents` added, and the anchor report.

    Apply it with :meth:`Calibration.extended`. ``anchor_report`` is ``None`` for scored documents:
    scoring recomputes no row of the calibration, so there is nothing to compare.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, validate_by_name=True, serialize_by_alias=True)

    schema_name: Literal["rcp-ndcg.extension.v1"] = Field(default="rcp-ndcg.extension.v1", alias="schema")
    calibration: str
    """The fingerprint of the calibration the records are anchored to."""
    records: tuple[ExtensionRecord, ...]
    skipped: tuple[str, ...] = ()
    """Documents judged but already in the calibration (``<query>/<doc>``); they keep their ability."""
    anchor_report: AnchorReport | None = None


def _keyed(record: ExtensionRecord) -> ExtensionRecord:
    payload = {
        "source": record.source,
        "slot": [record.dataset, record.query_id, record.doc_id],
        "calibration": record.calibration,
        "judgements": record.judgements,
        "estimate": record.estimate.model_dump(mode="json"),
    }
    return record.model_copy(update={"record_key": short(hash_payload(payload), 16)})


def _anchor(pairs: list[tuple[float, float]], items: ItemParams, max_gain_shift: float) -> AnchorReport:
    from rcp_ndcg_core.gain import gain

    theta_shift = max((abs(after - before) for before, after in pairs), default=0.0)
    gain_shift = max((abs(gain(after, items) - gain(before, items)) for before, after in pairs), default=0.0)  # type: ignore[arg-type]
    return AnchorReport(
        documents_checked=len(pairs),
        max_abs_theta_shift=theta_shift,
        max_abs_gain_shift=gain_shift,
        max_gain_shift=max_gain_shift,
        ok=gain_shift <= max_gain_shift,
    )


def _judgements_digest(judgements: JudgementSet) -> str:
    return short(hash_payload(sorted(j.record_id for j in judgements.judgements)), 16)


def _same_instrument(
    calibration: Calibration, judgements: JudgementSet, stage: Literal["tournament", "rubric"]
) -> None:
    fitted = {family.key for family in calibration.family_of(stage)}
    foreign = sorted({j.family_key for j in judgements.judgements if j.stage == stage} - fitted)
    if foreign:
        raise IdentityError(
            f"{stage} judgements of family {foreign} are not from the instrument the calibration was fitted "
            f"with ({sorted(fitted) or 'none'}); an ability measured with another rubric or judge is on another scale",
            hint="judge the documents with the calibration's judge, prompt and preprocessing, or refit",
            details={"foreign_families": foreign, "fitted_families": sorted(fitted)},
        )


# ---------------------------------------------------------------------------
# Stand-alone scoring
# ---------------------------------------------------------------------------


def score_documents(
    calibration: Calibration,
    judgements: JudgementSet,
    *,
    se_target: float = DEFAULT_SE_TARGET,
) -> Extension:
    """Score the documents a calibration lacks from their own rubric judgements, items frozen.

    Args:
        calibration: The calibration to extend.
        judgements: Rubric judgements of the documents (e.g. ``judge(..., stage="rubric", docs=...)``).
            Documents the calibration already holds keep their ability and are listed as skipped.
        se_target: The standard error (logits) the evidence should reach (flagged, not enforced).

    Returns:
        The :class:`Extension`; ``theta`` is the EAP under the calibration's ability prior (:func:`_scoring_prior`),
        ``theta_se`` the posterior SD. It has no anchor report: nothing is refitted.

    Raises:
        IdentityError: the judgements come from another rubric family.
        DataError: no rubric judgement of a document the calibration lacks.
    """
    rubric = judgements.of_stage("rubric")
    _same_instrument(calibration, rubric, "rubric")
    check_criteria(rubric)
    prior_mean, prior_sd = _scoring_prior(calibration)
    known = calibration._namespaced_thetas()
    severity = dict(calibration.judge_severity)
    evidence: dict[tuple[str, str], dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    skipped: set[str] = set()
    criteria = calibration.items.criteria
    for query, rows in rubric_observations(rubric, tag_judges=True).items():
        for doc_id, verdicts, judge in rows:  # type: ignore[misc]
            if doc_id in known.get(query, {}):
                skipped.add(f"{query}/{doc_id}")
                continue
            evidence[(query, doc_id)][judge].append(np.array([verdicts[label] for label in criteria], dtype=float))
    if not evidence:
        raise DataError(
            "no rubric judgement of a document the calibration lacks",
            hint="judge the documents to add with stage='rubric' (docs={query: [doc ids]})",
            cli_hint="judge the documents to add with `rcp-ndcg judge rubric --docs QUERY_ID:DOC_ID --out <store>`",
            details={"skipped": sorted(skipped)},
        )
    unscaled = sorted({judge for by_judge in evidence.values() for judge in by_judge} - set(severity))
    if severity and unscaled:
        raise DataError(f"the pooled calibration has no severity for judges {unscaled}")
    digest = _judgements_digest(rubric)
    records = []
    for (query, doc_id), by_judge in sorted(evidence.items()):
        judges = sorted(by_judge)
        # One document, one ability: every judge's answers enter one likelihood, each judge's logits shifted by
        # its severity in a pooled calibration.
        score = score_document(
            calibration.items,
            [len(by_judge[judge]) for judge in judges],
            [np.sum(by_judge[judge], axis=0).tolist() for judge in judges],
            prior_mean=prior_mean,
            prior_sd=prior_sd,
            se_target=se_target,
            severity=[severity.get(judge, 0.0) for judge in judges],
        )
        dataset, query_id = split_namespace(query)
        records.append(
            _keyed(
                ExtensionRecord(
                    source="scored",
                    dataset=dataset,
                    query_id=query_id,
                    doc_id=doc_id,
                    estimate=score,
                    calibration=calibration.fingerprint,
                    judgements=digest,
                    created_at=datetime.now(UTC),
                )
            )
        )
    return Extension(calibration=calibration.fingerprint, records=tuple(records), skipped=tuple(sorted(skipped)))


def _scoring_prior(calibration: Calibration) -> tuple[float, float]:
    """``(mean, sd)`` of the ability prior a document scored against ``calibration`` gets, in logits.

    A rubric-only calibration's own fit prior (``diagnostics.fit.ability_prior``, on the reported scale): the fit's
    abilities are posterior means under it, so a document scored from exactly its fit evidence gets its fit ability
    back. A tournament calibration has no rubric prior of its own (its abilities come from the tournament), so the
    mean and SD of its fitted abilities stand in (:func:`~rcp_ndcg.calibration.fit.population_prior`).

    Raises:
        DataError: a tournament calibration with fewer than two distinct abilities.
    """
    prior = calibration.diagnostics.fit.ability_prior if calibration.mode == "rubric_only" else None
    return (prior.mean, prior.sd) if prior is not None else population_prior(calibration)


# ---------------------------------------------------------------------------
# Tournament insertion
# ---------------------------------------------------------------------------


def _published_bt(calibration: Calibration, query: str) -> dict[str, float]:
    """The query's fitted abilities mapped back onto its Bradley-Terry scale: ``(theta - alpha) / tau``."""
    params = calibration.queries[query]
    return {
        doc: params.to_tournament(theta) for doc, theta in calibration._namespaced_thetas(source="fit")[query].items()
    }


def insert_documents(
    calibration: Calibration,
    judgements: JudgementSet,
    *,
    max_gain_shift: float = MAX_GAIN_SHIFT,
    se_target: float = DEFAULT_SE_TARGET,
    min_opponents: int = MIN_OPPONENTS,
) -> Extension:
    """Insert new documents into a tournament calibration, every existing ability frozen.

    Args:
        calibration: A tournament calibration.
        judgements: Tournament judgements of the queries: the windows the calibration was
            fitted on (they anchor the check) together with windows that pair each new
            document with documents of the calibration (see :func:`select_opponents`).
        max_gain_shift: The anchor tolerance, gain units: the Bradley-Terry fit of the fitted
            windows must reproduce the calibration's abilities within it.
        se_target: The standard error (logits, tournament scale) the evidence must reach.
        min_opponents: Distinct opponents each new document must face.

    Returns:
        The :class:`Extension`, with the anchor report.

    Raises:
        IdentityError: the judgements come from another tournament family, or the anchor check fails.
        DataError: a rubric-only calibration, a query without its fitted windows, or a document
            whose windows cannot identify it (too few opponents, disconnected, too little information).
    """
    from rcp_ndcg.calibration._projection import bradley_terry

    if calibration.mode != "tournament":
        raise DataError(
            "the calibration is rubric-only: a document is added to it by scoring its rubric judgements "
            "(score_documents), not by tournament insertion"
        )
    tournament = judgements.of_stage("tournament")
    _same_instrument(calibration, tournament, "tournament")
    if any(j.placements and any(p.chunk_id for p in j.placements) for j in tournament.judgements):
        raise DataError("insertion needs document-level tournament windows; these were judged in chunks")
    fitted = calibration._namespaced_thetas(source="fit")
    known = calibration._namespaced_thetas()
    l2 = calibration.identity.priors.bt_l2  # the refit must be the fit's own Bradley-Terry
    digest = _judgements_digest(tournament)
    pairs: list[tuple[float, float]] = []
    records = []
    skipped: set[str] = set()
    for query, comparisons in sorted(tournament_comparisons(tournament).items()):
        docs = {d for pair in comparisons for d in pair[:2]}
        new = sorted(docs - set(known.get(query, {})))
        skipped.update(f"{query}/{d}" for d in docs - set(fitted.get(query, {})) - set(new))
        if not new:
            continue
        if query not in calibration.queries:
            raise DataError(f"query {query!r} has no calibrated tournament; score its documents instead")
        published = _published_bt(calibration, query)
        # The windows and comparisons of the fit are those among its fitted documents only: a document inserted
        # earlier (in the calibration, but not fitted) and its windows are neither anchor nor opponent evidence.
        fitted_docs = set(fitted[query])
        old_windows = tournament.select(
            lambda j, q=query, f=fitted_docs: (
                namespace(j.dataset, j.query_id) == q and all(p.doc_id in f for p in j.placements)
            )
        )
        if not old_windows.judgements:
            raise DataError(
                f"query {query!r}: the windows the calibration was fitted on are missing",
                hint="pass the judgements of the calibration's tournament store together with the new windows",
            )
        refit, _ = bradley_terry(old_windows, l2=l2)
        params = calibration.queries[query]
        pairs.extend((fitted[query][d], params.calibrated(refit[query][d])) for d in fitted[query] if d in refit[query])
        existing = [c for c in comparisons if c[0] in fitted_docs and c[1] in fitted_docs]
        for doc_id in new:
            incident = [c for c in comparisons if doc_id in c[:2] and (c[1] if c[0] == doc_id else c[0]) in published]
            if not incident:
                raise DataError(f"{doc_id!r} of {query!r} faces no document of the calibration in the new windows")
            try:
                result = insert_document(
                    doc_id,
                    new_comparisons=incident,
                    existing_thetas=published,
                    existing_comparisons=existing,
                    l2_reg=l2,
                    se_target=se_target,
                    min_opponents=min_opponents,
                )
            except UnidentifiableInsertion as exc:
                raise DataError(
                    str(exc),
                    hint="plan more opponents (select_opponents(..., n=...)) and judge the new windows, or accept a "
                    "larger se_target; a query with few documents has no more opponents to plan, so only a larger "
                    "se_target helps there",
                    cli_hint="plan more opponents (`calibration insert --plan --n N`) and judge the new windows with "
                    "`judge tournament --plan`, or accept a larger --se-target; a plan that reports capped: true "
                    "already holds every document of the query, so only a larger --se-target helps",
                    details=exc.report,
                ) from exc
            except ScaleMovementError as exc:
                raise IdentityError(str(exc), details=exc.report) from exc
            dataset, query_id = split_namespace(query)
            records.append(
                _keyed(
                    ExtensionRecord(
                        source="inserted",
                        dataset=dataset,
                        query_id=query_id,
                        doc_id=doc_id,
                        # The tournament-scale estimate on the calibrated scale: theta and SE scale by tau,
                        # the information by 1 / tau^2.
                        estimate=DocumentEstimate(
                            theta=params.calibrated(result.theta),
                            se=params.calibrated_se(result.se),
                            information=result.information / params.tau**2,
                        ),
                        calibration=calibration.fingerprint,
                        judgements=digest,
                        created_at=datetime.now(UTC),
                    )
                )
            )
    report = _anchor(pairs, calibration.items, max_gain_shift)
    if not report.ok:
        raise IdentityError(
            f"the fitted windows no longer reproduce the calibration: max gain shift {report.max_abs_gain_shift:.3g} "
            f"> {max_gain_shift} (max ability shift {report.max_abs_theta_shift:.3g})",
            hint="insert against the calibration fitted on these windows",
            details={"anchor_report": report.model_dump()},
        )
    if not records:
        raise DataError("no new document in the tournament judgements", details={"skipped": sorted(skipped)})
    return Extension(
        calibration=calibration.fingerprint,
        records=tuple(records),
        skipped=tuple(sorted(skipped)),
        anchor_report=report,
    )


def select_opponents(
    calibration: Calibration,
    query_id: str,
    doc_id: str,
    *,
    n: int = 9,
    window: int | None = None,
    dataset: str | None = None,
) -> list[list[str]]:
    """Whom to judge a new document against: windows of the document and ``n`` opponents in all.

    Opponents are picked across the query's ability range (one per quantile bin),
    favouring the documents whose comparison is most informative at the query's
    median ability and whose rank the nDCG discount weighs most.

    Args:
        calibration: A tournament calibration.
        query_id: The query (raw id; ``dataset`` disambiguates when several datasets share it).
        doc_id: The new document.
        n: Opponents in all (fewer when the query has fewer documents).
        window: Documents per window, the new one included: the ``window`` of the calibration's tournament
            schedule (``JudgementStore(store).schedule("tournament").window``). ``None``: one window of them all.
        dataset: The query's dataset, when the calibration holds several.

    Returns:
        The windows, each the new document first and then its share of the opponents, in the opponents' order:
        ``[[doc_id, opponent_1, ...], ...]``. Judge exactly these windows with the schedule of the calibration's
        tournament store, into that store: ``judge(..., stage="tournament", windows={query_id: windows})``, or
        ``rcp-ndcg judge tournament --plan``. A mirroring schedule asks each window twice, once reversed.

    Raises:
        DataError: a rubric-only calibration, a query it does not calibrate, or a ``window`` below 2.
    """
    if window is not None and window < 2:
        raise DataError(f"a window holds the new document and at least one opponent: window={window}")
    if calibration.mode != "tournament":
        raise DataError("opponents are chosen from a tournament calibration; this one is rubric-only")
    datasets = [dataset] if dataset is not None else calibration.datasets
    keys = [namespace(d, query_id) for d in datasets if namespace(d, query_id) in calibration.queries]
    if len(keys) != 1:
        raise DataError(
            f"query {query_id!r} is calibrated in {len(keys)} datasets; name one with dataset=..."
            if keys
            else f"the calibration has no tournament query {query_id!r}"
        )
    published = _published_bt(calibration, keys[0])
    ranked = sorted(published, key=lambda d: published[d], reverse=True)
    ranks = {d: rank for rank, d in enumerate(ranked, start=1)}
    opponents = _opponent_kernel(doc_id, statistics.median(published.values()), published, ranks, k=n)
    size = (window or len(opponents) + 1) - 1
    return [[doc_id, *opponents[start : start + size]] for start in range(0, len(opponents), size)] or [[doc_id]]


__all__ = [
    "MAX_GAIN_SHIFT",
    "AnchorReport",
    "Extension",
    "ExtensionRecord",
    "insert_documents",
    "score_documents",
    "select_opponents",
]
