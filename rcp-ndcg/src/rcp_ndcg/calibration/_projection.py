"""Reading judgement stores and projecting judgements onto what each estimator consumes.

* rubric judgements -> one observation per document per window,
  ``(doc_id, {"C1": 0|1, ...}[, judge])``, sibling chunks max-pooled inside the window;
* tournament judgements -> the query's Bradley-Terry abilities, refitted from every
  window's soft comparisons in the tournament's own grammar: the same observations
  the live tournament fits, but cold (from zero, not warm after every batch), so the
  refit agrees with the live fit to convergence tolerance, not bit for bit. Chunk
  abilities are max-pooled onto their documents.

Queries are namespaced ``<dataset>||<query_id>`` (:data:`QUERY_ID_SEP`) so two
datasets that share raw ids cannot collide in one fit.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from rcp_ndcg_core.schemas import Judgement, JudgementSet

from rcp_ndcg.data.postprocess import max_pool_rubric_window_by_document, max_pool_scores_by_document
from rcp_ndcg.errors import DataError, MissingInputError
from rcp_ndcg.judging.store import JudgementStore

#: Separator of the ``<dataset>||<query_id>`` namespaced query id.
QUERY_ID_SEP = "||"

#: One rubric observation: ``(doc_id, criteria)`` or, pooled, ``(doc_id, criteria, judge)``.
RubricObservation = tuple[str, dict[str, int]] | tuple[str, dict[str, int], str]


def namespace(dataset: str, query_id: str) -> str:
    """The fit's key of a query: ``<dataset>||<query_id>``."""
    return f"{dataset}{QUERY_ID_SEP}{query_id}"


def split_namespace(key: str) -> tuple[str, str]:
    """Invert :func:`namespace`."""
    dataset, _, query_id = key.partition(QUERY_ID_SEP)
    return dataset, query_id


def judged_bt_l2(*stores: str | Path) -> float | None:
    """The Bradley-Terry L2 the tournament's live fit used while judging, as the stores' identities record it.

    Returns:
        The penalty (logits^-2), or ``None`` when no store holds tournament judgements.

    Raises:
        IdentityError: the stores' tournaments were judged with different penalties.
    """
    from rcp_ndcg.errors import IdentityError

    penalties = {}
    for root in stores:
        entry = JudgementStore(root).identities().get("tournament")
        if entry is not None and "bt_l2" in entry["identity"]:
            penalties[str(root)] = float(entry["identity"]["bt_l2"])
    if len(set(penalties.values())) > 1:
        raise IdentityError(
            f"the tournaments were judged with different Bradley-Terry L2 penalties: {penalties}",
            hint="calibrate the stores separately",
            details={"bt_l2": penalties},
        )
    return next(iter(penalties.values()), None)


def read_judgements(*stores: str | Path) -> JudgementSet:
    """Every judgement in the given stores (directories written by ``judge``): per window, its latest valid one.

    A window (``record_id``) held by several stores counts once (:meth:`JudgementSet.merge`).

    Raises:
        MissingInputError: a path is not a judgement store.
    """
    sets = []
    for root in stores:
        store = JudgementStore(root)
        if not store.identity_path.exists():
            raise MissingInputError(
                f"{root} is not a judgement store (no {store.identity_path.name})",
                hint="point at the directory judge() or `rcp-ndcg judge` wrote",
                cli_hint="point at the --out directory of `rcp-ndcg judge tournament|rubric`, or a run directory",
            )
        sets.append(store.read())
    return JudgementSet.merge(sets)


def check_criteria(judgements: JudgementSet) -> None:
    """Refuse rubric verdicts that do not answer exactly the criteria their family declares.

    A rubric family declares its criteria (``Family.criteria``); the fit reads them in order as ``C1..CK``, so they
    must be named that way (any ``K``: a custom rubric declares its own). Every placement of a valid rubric
    judgement must carry a verdict for each declared criterion and for no other: a missing verdict is never read
    as a fail, and an extra one is never ignored.

    Raises:
        DataError: a family whose criteria are not named ``C1..CK``, or a placement with a missing, unknown or
            renamed criterion (the record, the document and the criteria are named in the message and
            ``details``).
    """
    from rcp_ndcg_core.schemas import criterion_labels

    for key, family in judgements.families.items():
        if family.stage == "rubric" and family.criteria != criterion_labels(len(family.criteria)):
            raise DataError(
                f"rubric family {key} declares the criteria {list(family.criteria)}; the fit reads a rubric's "
                "criteria in order as C1..CK",
                hint="declare the criteria as C1..CK (Family.criteria = criterion_labels(K)) and key the verdicts "
                "the same way",
                details={"family": key, "declared": list(family.criteria)},
            )
    for judgement in judgements.judgements:
        if judgement.stage != "rubric" or not judgement.valid:
            continue
        declared = judgements.families[judgement.family_key].criteria
        for placement in judgement.placements:
            given = set(placement.criteria or {})
            if given == set(declared):
                continue
            missing = [label for label in declared if label not in given]
            unknown = sorted(given - set(declared))
            raise DataError(
                f"rubric judgement {judgement.record_id} (query {judgement.query_id!r}, document "
                f"{placement.doc_id!r}) answers criteria that its family does not declare: missing {missing}, "
                f"unknown {unknown}; declared {list(declared)}",
                hint="give every placement exactly one verdict per declared criterion, or declare the criteria the "
                "judge answered in the family",
                details={
                    "record_id": judgement.record_id,
                    "doc_id": placement.doc_id,
                    "missing": missing,
                    "unknown": unknown,
                    "declared": list(declared),
                },
            )


def _ordered(judgements: Iterable[Judgement]) -> list[Judgement]:
    # Scheduled windows in schedule order, then the planned ones (no sequence number) by record id: a planned
    # window carries no position, so its record id is the one total order that does not inherit the order the
    # stores happened to hand it in. The float sums of the refit would otherwise move with that order.
    return sorted(
        judgements,
        key=lambda j: (j.dataset, j.query_id, j.family_key, j.window_seq is None, j.window_seq or 0, j.record_id),
    )


def rubric_observations(judgements: JudgementSet, *, tag_judges: bool) -> dict[str, list[RubricObservation]]:
    """``{query: [(doc_id, criteria[, judge]), ...]}``: one entry per document per valid rubric window."""
    families = judgements.families
    observations: dict[str, list[RubricObservation]] = defaultdict(list)
    for judgement in _ordered(j for j in judgements.judgements if j.stage == "rubric" and j.valid):
        chunk_map = {p.unit_id: p.doc_id for p in judgement.placements}
        pooled = max_pool_rubric_window_by_document(
            {p.unit_id: p.criteria for p in judgement.placements if p.criteria}, chunk_map
        )
        judge = families[judgement.family_key].judge_model
        key = namespace(judgement.dataset, judgement.query_id)
        for doc_id, criteria in pooled.items():
            observations[key].append((doc_id, criteria, judge) if tag_judges else (doc_id, criteria))
    return dict(observations)


def tournament_comparisons(judgements: JudgementSet) -> dict[str, list[tuple[str, str, float, float]]]:
    """``{query: [(winner, loser, weight, soft_label), ...]}`` in window order, chunk ids as the judge saw them.

    Each window is read by :func:`~rcp_ndcg.judging._parsing.listwise.judgement_comparisons`, the grammar the
    live tournament fits.
    """
    from rcp_ndcg.judging._parsing.listwise import judgement_comparisons

    comparisons: dict[str, list[tuple[str, str, float, float]]] = defaultdict(list)
    for judgement in _ordered(j for j in judgements.judgements if j.stage == "tournament" and j.valid):
        comparisons[namespace(judgement.dataset, judgement.query_id)].extend(judgement_comparisons(judgement))
    return dict(comparisons)


def count_gains(
    judgements: JudgementSet | Iterable[JudgementSet], *, dataset: str | None = None
) -> dict[str, dict[str, float]]:
    """``{query key: {doc_id: gain}}``: the Count-nDCG gain of every document with valid rubric verdicts.

    One gain per document per query, the paper's rubric-only baseline ``sum_c S_c / (C * n)``
    (:func:`rcp_ndcg_core.count_gain`): ``S_c`` is the number of the document's rubric windows that passed
    criterion ``c`` and ``n`` its windows -- the window is the unit of evidence, so its verdicts are summed per
    document and sibling chunks pool inside a window, as the fit reads them. No tournament and no calibration is
    involved; the gains feed ``evaluate(..., count_gains=...)`` and ``rcp-ndcg eval score --metrics count_ndcg``.

    Keys follow :meth:`Calibration.gains`' rule: the bare query id when ``dataset`` is given or the judgements
    hold one dataset, else ``"<dataset>||<query_id>"``.

    Args:
        judgements: The judgements (e.g. :func:`read_judgements` of the rubric stores), one set or several.
        dataset: Only this dataset's queries.

    Returns:
        The gains, in ``[0, 1]``.

    Raises:
        DataError: A rubric family whose criteria are not ``C1..CK`` or a placement that does not answer its
            family's criteria (:func:`check_criteria`), a document judged under rubrics of different sizes, or a
            ``dataset`` the judgements hold no rubric verdicts for.
    """
    from rcp_ndcg_core.gain import count_gain

    judgement_set = JudgementSet.merge([judgements] if isinstance(judgements, JudgementSet) else judgements)
    check_criteria(judgement_set)
    observations = rubric_observations(judgement_set, tag_judges=False)
    names = {split_namespace(key)[0] for key in observations}
    if dataset is not None and dataset not in names:
        raise DataError(
            f"the judgements hold no rubric verdicts for dataset {dataset!r}; they hold {sorted(names)}",
            hint="pass the rubric store of the dataset being scored",
        )
    bare = dataset is not None or len(names) == 1
    passes: dict[tuple[str, str], dict[str, int]] = {}
    placements: dict[tuple[str, str], int] = {}
    for key, rows in observations.items():
        if dataset is not None and split_namespace(key)[0] != dataset:
            continue
        for row in rows:
            doc_id, criteria = row[0], row[1]
            slot = (key, doc_id)
            counts = passes.get(slot)
            if counts is None:
                counts = passes[slot] = dict.fromkeys(criteria, 0)
            elif set(counts) != set(criteria):
                raise DataError(
                    f"document {doc_id!r} of query {key!r} was judged under rubrics of different criteria: "
                    f"{sorted(counts)} and {sorted(criteria)}",
                    hint="a Count gain is the share of passed criteria of one rubric; score one rubric at a time",
                )
            for criterion, value in criteria.items():
                counts[criterion] += int(value)
            placements[slot] = placements.get(slot, 0) + 1
    out: dict[str, dict[str, float]] = {}
    for (key, doc_id), counts in passes.items():
        name, query_id = split_namespace(key)
        query_key = query_id if bare else namespace(name, query_id)
        out.setdefault(query_key, {})[doc_id] = count_gain(list(counts.values()), placements[(key, doc_id)])
    return out


def _units(judgements: JudgementSet) -> dict[str, dict[str, str]]:
    """``{query: {unit_id: doc_id}}``: every unit the query's tournament windows showed, valid or invalid."""
    units: dict[str, dict[str, str]] = defaultdict(dict)
    for judgement in _ordered(j for j in judgements.judgements if j.stage == "tournament"):
        key = namespace(judgement.dataset, judgement.query_id)
        for placement in judgement.placements:
            units[key].setdefault(placement.unit_id, placement.doc_id)
    return units


def no_tournament_evidence(judgements: JudgementSet) -> list[str]:
    """``["<dataset>||<query_id>/<doc_id>", ...]``: documents a fitted query showed with no comparison at all.

    A document whose only tournament windows are invalid enters the Bradley-Terry fit with no observations: the
    estimator gives it the query's mean ability (and the ridge's standard error only when the query has other
    comparisons). Those are the model's own numbers (the paper's), but no comparison backs them, so the fit
    reports the documents instead of presenting them as judged. A query whose windows are all invalid is not
    fitted at all -- no ability is written for it -- and its documents are ``uncalibrated_documents``.
    """
    units = _units(judgements)
    evidenced: dict[str, set[str]] = defaultdict(set)
    comparisons = tournament_comparisons(judgements)
    for key, rows in comparisons.items():
        for winner, loser, _weight, _soft_label in rows:
            evidenced[key].add(winner)
            evidenced[key].add(loser)
    out = []
    for key in comparisons:  # only the queries the fit reads: an all-invalid query is uncalibrated instead
        unit_docs = units.get(key, {})
        backed = {unit_docs[unit] for unit in evidenced.get(key, set()) if unit in unit_docs}
        out.extend(f"{key}/{doc_id}" for doc_id in sorted(set(unit_docs.values()) - backed))
    return sorted(out)


def bradley_terry(
    judgements: JudgementSet, *, l2: float
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float | None]]]:
    """Each tournament query's Bradley-Terry abilities and standard errors, per document (logits).

    Refitted from every valid window's comparisons over every unit the query
    showed; a document judged in chunks takes its best chunk's ability (and that
    chunk's standard error). A document with no comparison at all takes the mean
    ability (:func:`no_tournament_evidence` names it): its standard error is the
    ridge's ``1 / sqrt(l2)`` when the query has other comparisons, and ``None``
    when it has none, since then no information matrix was formed.
    """
    from rcp_ndcg_core.irt._bradley_terry import BradleyTerryEstimator

    units = _units(judgements)
    thetas: dict[str, dict[str, float]] = {}
    ses: dict[str, dict[str, float | None]] = {}
    for key, comparisons in tournament_comparisons(judgements).items():
        estimator = BradleyTerryEstimator(doc_ids=list(units[key]), l2_reg=l2)
        for winner, loser, weight, soft_label in comparisons:
            estimator.add_comparison(winner, loser, weight=weight, soft_label=soft_label)
        estimator.fit_lbfgs()
        per_unit = estimator.get_scores_with_se(list(units[key]))
        pooled = max_pool_scores_by_document({u: theta for u, (theta, _) in per_unit.items()}, units[key])
        best_unit = {units[key][u]: u for u in sorted(per_unit, key=lambda u: per_unit[u][0])}
        thetas[key] = dict(pooled)
        ses[key] = {}
        for doc in pooled:
            se = per_unit[best_unit[doc]][1]
            ses[key][doc] = None if se is None else float(se)
    return thetas, ses


__all__ = [
    "QUERY_ID_SEP",
    "RubricObservation",
    "bradley_terry",
    "check_criteria",
    "count_gains",
    "judged_bt_l2",
    "namespace",
    "no_tournament_evidence",
    "read_judgements",
    "rubric_observations",
    "split_namespace",
    "tournament_comparisons",
]
