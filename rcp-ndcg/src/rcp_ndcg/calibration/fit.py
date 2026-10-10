"""The calibration: judgements in, a :class:`Calibration` out; one artifact layout on disk.

:func:`calibrate` fits the 2PL model of the paper to Stage B (rubric) judgements,
anchored on Stage A (tournament) abilities when the queries have them:

* **tournament** -- ``P(C_k = 1) = sigmoid(gamma_k * (tau_q * theta_BT - beta_k + alpha_q))``;
  a document's calibrated ability is ``theta = tau_q * theta_BT + alpha_q``;
* **rubric_only** -- ``P(C_k = 1) = sigmoid(gamma_k * (theta - beta_k))``, ``theta ~ N(0, 1)``,
  fitted by marginal maximum likelihood; a document's ability is its posterior mean.

Several judges who answered one rubric pool into one fit with a severity per
judge (``judges="pooled"``, tournament mode).

A :class:`Calibration` is immutable. It saves to, and loads from, one layout::

    <dir>/
      items.json          item parameters, mode, judge severity, the families fitted (plus the criterion
                          labels and the fingerprint, for a person reading the file)
      queries.parquet     dataset, query_id, tau, alpha      (tournament mode)
      thetas.parquet      dataset, query_id, doc_id, theta, theta_se, source (fit | scored | inserted)
      coverage.json       queries per stage, uncalibrated queries, invalid windows per query (by stage, phase
                          and category), the flagged queries, degenerate documents
      diagnostics.json    the fit summary, its reliability (ECE, Brier) per family, and the warnings
      extensions.jsonl    every scored or inserted document, with its provenance (one
                          rcp-ndcg.extension-record.v1 object per line)
      identity.json       what was fitted: the judgements, the families, the switches and the priors
"""

from __future__ import annotations

import json
import math
import statistics
import warnings
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast, get_args

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from rcp_ndcg_core.irt import JudgeOverlapError, Priors, fit_calibration
from rcp_ndcg_core.schemas import ItemParams, JudgementFamily, JudgementSet, QueryParams

from rcp_ndcg.calibration._projection import (
    QUERY_ID_SEP,
    bradley_terry,
    check_criteria,
    namespace,
    no_tournament_evidence,
    rubric_observations,
    split_namespace,
)
from rcp_ndcg.calibration.coverage import (
    INVALID_WINDOW_SHARE,
    CalibrationCoverage,
    DegenerateCounts,
    QueryCounts,
    StageWindows,
    WindowCount,
    coverage_flags,
    flags_message,
    window_coverage,
)
from rcp_ndcg.calibration.diagnostics import Diagnostics, FitWarning, fit_diagnostics
from rcp_ndcg.errors import DataError, IdentityError, MissingInputError, RcpNdcgWarning
from rcp_ndcg.support.identity import hash_payload, short
from rcp_ndcg.support.logging import get_logger

if TYPE_CHECKING:
    import pandas as pd

    from rcp_ndcg.calibration.extend import ExtensionRecord

logger = get_logger(__name__)

CALIBRATION_SCHEMA = "rcp-ndcg.calibration.v1"
ITEMS_FILE = "items.json"
QUERIES_FILE = "queries.parquet"
THETAS_FILE = "thetas.parquet"
COVERAGE_FILE = "coverage.json"
DIAGNOSTICS_FILE = "diagnostics.json"
EXTENSIONS_FILE = "extensions.jsonl"
IDENTITY_FILE = "identity.json"

Mode = Literal["auto", "tournament", "rubric_only"]
Judges = Literal["single", "pooled"]
Source = Literal["fit", "scored", "inserted"]


@dataclass(frozen=True)
class ThetaRow:
    """One document's calibrated ability.

    Attributes:
        dataset: The dataset of the query.
        query_id: The query (not namespaced).
        doc_id: The document.
        theta: Calibrated ability (logits).
        theta_se: Its standard error (logits), when known: a rubric-only document's posterior standard deviation,
            or a tournament document's Bradley-Terry SE (the diagonal approximation of its own comparisons'
            information, mapped by the query's ``tau``; see ``docs/concepts/calibration.md``).
        source: ``fit`` (estimated by the calibration), ``scored`` (added from its own rubric
            answers, items frozen) or ``inserted`` (added to the tournament, opponents frozen).
    """

    dataset: str
    query_id: str
    doc_id: str
    theta: float
    theta_se: float | None
    source: Source


class CalibrationIdentity(BaseModel):
    """What a calibration was fitted from and how.

    Attributes:
        judgements: 16-hex digest of the fitted judgements' record ids.
        families: The family keys fitted.
        mode: The fit's mode.
        judges: ``"single"`` or ``"pooled"``.
        priors: The priors and penalties of the fit, the Bradley-Terry refit's ``bt_l2`` included.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", validate_by_name=True, serialize_by_alias=True)

    schema_name: Literal["rcp-ndcg.calibration-identity.v1"] = Field(
        default="rcp-ndcg.calibration-identity.v1", alias="schema"
    )
    judgements: str
    families: list[str]
    mode: Literal["tournament", "rubric_only"]
    judges: Judges
    priors: Priors


class CalibrationItems(BaseModel):
    """``items.json``, the file of a calibration directory that names its layout (``rcp-ndcg.calibration.v1``).

    Attributes:
        mode: ``"tournament"`` or ``"rubric_only"``.
        criteria: The criterion labels ``C1..CK``, for a person reading the file; ``K`` is the length of ``gamma``.
        gamma: Discrimination per criterion (dimensionless, > 0).
        beta: Difficulty per criterion (logits).
        judge_severity: ``{judge: logit offset}`` of a pooled fit; empty for one judge.
        families: ``{family_key: JudgementFamily}`` of the judgements fitted (both stages).
        fingerprint: 16-hex digest of the fit (items, queries, the fit's own abilities), for a person reading the
            file; extensions record it and are checked against the fit itself.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", validate_by_name=True, serialize_by_alias=True)

    schema_name: Literal["rcp-ndcg.calibration.v1"] = Field(default=CALIBRATION_SCHEMA, alias="schema")
    mode: Literal["tournament", "rubric_only"]
    criteria: list[str]
    gamma: list[float]
    beta: list[float]
    judge_severity: dict[str, float]
    families: dict[str, JudgementFamily]
    fingerprint: str


@dataclass(frozen=True)
class Calibration:
    """A fitted calibration: item and query parameters, abilities, and what was checked.

    Attributes:
        mode: ``"tournament"`` or ``"rubric_only"``.
        items: The criterion parameters.
        queries: ``{"<dataset>||<query_id>": QueryParams}`` (tournament mode; empty otherwise).
        thetas: Every document's ability, the fit's own and those added later.
        families: ``{family_key: JudgementFamily}`` of the judgements fitted (both stages).
        judge_severity: ``{judge: logit offset}`` of a pooled fit; empty for one judge.
        coverage: Queries per stage, uncalibrated queries, invalid windows per query (by stage, phase and
            category), the flagged queries, degenerate documents.
        diagnostics: The fit summary and its reliability (ECE, Brier), overall, per criterion and per family,
            and the fit's ``warnings``.
        identity: What was fitted (judgements digest, families, switches, priors).
        extensions: The records of every scored or inserted document.
    """

    mode: Literal["tournament", "rubric_only"]
    items: ItemParams
    queries: Mapping[str, QueryParams]
    thetas: tuple[ThetaRow, ...]
    families: Mapping[str, JudgementFamily]
    judge_severity: Mapping[str, float]
    coverage: CalibrationCoverage
    diagnostics: Diagnostics
    identity: CalibrationIdentity
    extensions: tuple[ExtensionRecord, ...] = field(default=())

    # ------------------------------------------------------------------
    # Views
    # ------------------------------------------------------------------

    @property
    def datasets(self) -> list[str]:
        """The datasets calibrated, sorted."""
        return sorted({row.dataset for row in self.thetas})

    @property
    def warnings(self) -> list[dict[str, str]]:
        """The fit's typed warnings, ``[{"code", "message"}, ...]`` (codes from :data:`rcp_ndcg.errors.WarningCode`)."""
        return [warning.model_dump() for warning in self.diagnostics.warnings]

    def family_of(self, stage: Literal["tournament", "rubric"]) -> list[JudgementFamily]:
        """The families fitted for one stage."""
        return [family for family in self.families.values() if family.stage == stage]

    @property
    def fingerprint(self) -> str:
        """16-hex digest of the fit itself (items, queries, the fit's own abilities): what extensions anchor to."""
        payload = {
            "mode": self.mode,
            "items": self.items.model_dump(mode="json"),
            "queries": {key: params.model_dump(mode="json") for key, params in sorted(self.queries.items())},
            "thetas": sorted(
                (row.dataset, row.query_id, row.doc_id, row.theta) for row in self.thetas if row.source == "fit"
            ),
        }
        return short(hash_payload(payload), 16)

    def theta_map(self, *, source: Source | None = None, dataset: str | None = None) -> dict[str, dict[str, float]]:
        """``{query key: {doc_id: theta}}``: the calibrated abilities (logits), of one source or of all.

        Keys follow one rule, shared with :meth:`gains`: the bare query id when ``dataset`` is given or the
        calibration holds one dataset, else ``"<dataset>||<query_id>"`` (the form :attr:`queries` always uses).

        Args:
            source: Only the abilities of this source (``fit``, ``scored`` or ``inserted``).
            dataset: Only this dataset's queries.

        Raises:
            DataError: The calibration holds no dataset ``dataset``.
        """
        return {key: dict(docs) for key, docs in self._keyed(source=source, dataset=dataset).items()}

    def gains(self, dataset: str | None = None) -> dict[str, dict[str, float]]:
        """``{query key: {doc_id: gain}}``: the paper's gain of every calibrated ability, in [0, 1].

        Keys follow the rule of :meth:`theta_map`: the bare query id when ``dataset`` is given or the calibration
        holds one dataset, else ``"<dataset>||<query_id>"``.

        Args:
            dataset: Only this dataset's queries.

        Raises:
            DataError: The calibration holds no dataset ``dataset``.
        """
        from rcp_ndcg_core.gain import gain

        return {
            key: {doc: gain(theta, self.items) for doc, theta in docs.items()}  # type: ignore[arg-type]
            for key, docs in self._keyed(dataset=dataset).items()
        }

    def _keyed(self, *, source: Source | None = None, dataset: str | None = None) -> dict[str, dict[str, float]]:
        if dataset is not None and dataset not in self.datasets:
            raise DataError(f"the calibration holds no dataset {dataset!r}; it holds {self.datasets}")
        bare = dataset is not None or len(self.datasets) == 1
        out: dict[str, dict[str, float]] = {}
        for row in self.thetas:
            if (source is None or row.source == source) and (dataset is None or row.dataset == dataset):
                key = row.query_id if bare else namespace(row.dataset, row.query_id)
                out.setdefault(key, {})[row.doc_id] = row.theta
        return out

    def _namespaced_thetas(self, *, source: Source | None = None) -> dict[str, dict[str, float]]:
        """``{"<dataset>||<query_id>": {doc_id: theta}}`` whatever the number of datasets (the extensions' keys)."""
        out: dict[str, dict[str, float]] = {}
        for row in self.thetas:
            if source is None or row.source == source:
                out.setdefault(namespace(row.dataset, row.query_id), {})[row.doc_id] = row.theta
        return out

    def to_pandas(self, table: Literal["thetas", "queries", "items"] = "thetas") -> pd.DataFrame:
        """One table of the calibration as a DataFrame.

        Args:
            table: ``"thetas"``: ``dataset, query_id, doc_id, theta, theta_se, gain, source`` (the gain in
                ``[0, 1]``, the ability in logits); ``"queries"``: ``dataset, query_id, tau, alpha`` (tournament
                mode; empty otherwise); ``"items"``: ``criterion, gamma, beta``.
        """
        import pandas as pd
        from rcp_ndcg_core.gain import gain

        if table == "thetas":
            columns = ["dataset", "query_id", "doc_id", "theta", "theta_se", "gain", "source"]
            rows = [
                (r.dataset, r.query_id, r.doc_id, r.theta, r.theta_se, gain(r.theta, self.items), r.source)  # type: ignore[arg-type]
                for r in self.thetas
            ]
            return pd.DataFrame(rows, columns=columns)
        if table == "queries":
            rows = [(*key.split("||", 1), p.tau, p.alpha) for key, p in self.queries.items()]
            return pd.DataFrame(rows, columns=["dataset", "query_id", "tau", "alpha"])
        if table == "items":
            rows = [
                (f"C{c + 1}", float(g), float(b))
                for c, (g, b) in enumerate(zip(self.items.gamma, self.items.beta, strict=True))
            ]
            return pd.DataFrame(rows, columns=["criterion", "gamma", "beta"])
        raise DataError(f"unknown table {table!r}; expected thetas, queries or items")

    def __repr__(self) -> str:
        judges = sorted({family.judge_model for family in self.families.values()})
        return (
            f"Calibration(mode={self.mode}, {len(self.datasets)} datasets, {len(self.queries) or len(self._keyed())} "
            f"queries, {len(self.thetas)} thetas, {len(self.items.gamma)} criteria, judges={judges})"
        )

    def extended(self, extension: Any) -> Calibration:
        """This calibration with an :class:`~rcp_ndcg.calibration.extend.Extension`'s documents added.

        Raises:
            IdentityError: the extension was computed against another calibration, or it gives a
                document already in this calibration a different ability.
        """
        if extension.calibration != self.fingerprint:
            raise IdentityError(
                f"the extension was computed against calibration {extension.calibration}, not {self.fingerprint}",
                hint="score or insert against this calibration",
            )
        present = {(r.dataset, r.query_id, r.doc_id): r for r in self.thetas}
        by_key = {record.record_key: record for record in self.extensions}
        rows = list(self.thetas)
        records = list(self.extensions)
        for record in extension.records:
            slot = (record.dataset, record.query_id, record.doc_id)
            if record.record_key in by_key:
                continue
            if slot in present:
                raise IdentityError(
                    f"{record.doc_id!r} of query {record.query_id!r} is already in the calibration "
                    f"({present[slot].source}); an extension never replaces an ability",
                    hint="write the new evidence into a calibration that lacks the document",
                )
            estimate = record.estimate
            row = ThetaRow(record.dataset, record.query_id, record.doc_id, estimate.theta, estimate.se, record.source)
            rows.append(row)
            records.append(record)
            present[slot] = row
        return replace(self, thetas=tuple(rows), extensions=tuple(records))

    # ------------------------------------------------------------------
    # Disk
    # ------------------------------------------------------------------

    def save(self, directory: str | Path) -> Path:
        """Write the calibration layout (module docstring) into ``directory``; return it."""
        import pandas as pd

        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        items = CalibrationItems(
            mode=self.mode,
            criteria=list(self.items.criteria),
            gamma=list(self.items.gamma),
            beta=list(self.items.beta),
            judge_severity=dict(self.judge_severity),
            families=dict(sorted(self.families.items())),
            fingerprint=self.fingerprint,
        )
        _write_json(out / ITEMS_FILE, items.model_dump(mode="json"))
        queries = [
            {"dataset": split_namespace(key)[0], "query_id": split_namespace(key)[1], "tau": q.tau, "alpha": q.alpha}
            for key, q in sorted(self.queries.items())
        ]
        pd.DataFrame(queries, columns=["dataset", "query_id", "tau", "alpha"]).to_parquet(
            out / QUERIES_FILE, index=False
        )
        self.to_pandas().to_parquet(out / THETAS_FILE, index=False)
        _write_json(out / COVERAGE_FILE, self.coverage.model_dump(mode="json"))
        _write_json(out / DIAGNOSTICS_FILE, self.diagnostics.model_dump(mode="json"))
        _write_json(out / IDENTITY_FILE, self.identity.model_dump(mode="json"))
        (out / EXTENSIONS_FILE).write_text(
            "".join(record.model_dump_json() + "\n" for record in self.extensions), encoding="utf-8"
        )
        return out

    @classmethod
    def load(cls, directory: str | Path) -> Calibration:
        """Read a calibration directory written by :meth:`save`.

        Raises:
            MissingInputError: the directory lacks a file of the layout.
            DataError: ``items.json`` is not a calibration.
        """
        import pandas as pd

        from rcp_ndcg.calibration.extend import ExtensionRecord

        root = Path(directory)
        layout = (
            ITEMS_FILE,
            QUERIES_FILE,
            THETAS_FILE,
            COVERAGE_FILE,
            DIAGNOSTICS_FILE,
            IDENTITY_FILE,
            EXTENSIONS_FILE,
        )
        missing = [name for name in layout if not (root / name).exists()]
        if missing:
            raise MissingInputError(
                f"{root} is not a calibration: missing {', '.join(missing)}",
                hint="point at the directory calibrate() saved (or a run's calibration/)",
                cli_hint="point at the --out directory of `rcp-ndcg calibration fit`, or a run directory",
            )
        payload = json.loads((root / ITEMS_FILE).read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema") != CALIBRATION_SCHEMA:
            raise DataError(f"{root / ITEMS_FILE} is not a {CALIBRATION_SCHEMA} file")
        try:
            items = CalibrationItems.model_validate(payload)
        except ValidationError as exc:
            raise DataError(
                f"{root / ITEMS_FILE} does not match {CALIBRATION_SCHEMA}: {exc.error_count()} problem(s)",
                hint="compare it with `rcp-ndcg schema show calibration`",
                details={"errors": exc.errors(include_url=False, include_context=False, include_input=False)},
            ) from exc
        query_rows = pd.read_parquet(root / QUERIES_FILE).to_dict("records")
        queries = {
            namespace(str(row["dataset"]), str(row["query_id"])): QueryParams(
                tau=float(row["tau"]), alpha=float(row["alpha"])
            )
            for row in query_rows
        }
        thetas = tuple(
            _theta_row(root / THETAS_FILE, row) for row in pd.read_parquet(root / THETAS_FILE).to_dict("records")
        )
        extensions = tuple(
            ExtensionRecord.model_validate_json(line)
            for line in (root / EXTENSIONS_FILE).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        return cls(
            mode=items.mode,
            items=ItemParams(gamma=tuple(items.gamma), beta=tuple(items.beta)),
            queries=queries,
            thetas=thetas,
            families=dict(items.families),
            judge_severity=dict(items.judge_severity),
            coverage=CalibrationCoverage.model_validate(_read_json(root / COVERAGE_FILE)),
            diagnostics=Diagnostics.model_validate(_read_json(root / DIAGNOSTICS_FILE)),
            identity=CalibrationIdentity.model_validate(_read_json(root / IDENTITY_FILE)),
            extensions=extensions,
        )


#: The sources a ``thetas.parquet`` row may carry (the one home of the literal: :data:`Source`).
THETA_SOURCES: tuple[Source, ...] = cast("tuple[Source, ...]", get_args(Source))


def _theta_row(path: Path, row: Mapping[str, Any]) -> ThetaRow:
    """One ``thetas.parquet`` row, validated: a known source, a finite theta, a finite or missing SE.

    The parquet rows are the one artifact the layout does not read through a pydantic model, so they are checked
    here instead of letting an unknown source or a NaN ability through.

    Raises:
        DataError: An unknown source, a non-finite theta, or an infinite standard error (a NaN one is the
            missing one, as ``None`` is written).
    """
    import pandas as pd

    source = row.get("source")
    if source not in THETA_SOURCES:
        raise DataError(
            f"{path}: source {source!r} is not one of {list(THETA_SOURCES)}",
            hint="a thetas.parquet row is written by the calibration fit, by scoring or by insertion",
            details={"source": source, "doc_id": row.get("doc_id")},
        )
    try:
        theta = float(row["theta"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DataError(
            f"{path}: theta of {row.get('doc_id')!r} is not a number ({row.get('theta')!r})",
            hint="a theta is a logit; refit the calibration or repair the row",
        ) from exc
    if not math.isfinite(theta):
        raise DataError(
            f"{path}: theta of {row.get('doc_id')!r} is not finite ({theta})",
            hint="a non-finite ability calibrates every gain to NaN; repair the row or refit the calibration",
            details={"doc_id": row.get("doc_id"), "theta": theta},
        )
    se = None
    try:
        if not pd.isna(row["theta_se"]):
            se = float(row["theta_se"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DataError(
            f"{path}: standard error of {row.get('doc_id')!r} is not a number ({row.get('theta_se')!r})",
            hint="a standard error is a logit width; leave it empty when unknown",
        ) from exc
    if se is not None and not math.isfinite(se):
        raise DataError(
            f"{path}: standard error of {row.get('doc_id')!r} is not finite ({se})",
            hint="an infinite standard error is not a width; leave it empty when unknown",
            details={"doc_id": row.get("doc_id"), "theta_se": se},
        )
    return ThetaRow(str(row["dataset"]), str(row["query_id"]), str(row["doc_id"]), theta, se, cast("Source", source))


def _write_json(path: Path, payload: Any) -> None:
    try:
        text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    except ValueError as exc:
        raise DataError(
            f"{path.name} would hold a value JSON cannot carry: {exc}",
            hint="a non-finite number (NaN or infinity) reached the artifact; check the fit's diagnostics",
        ) from exc
    path.write_text(text + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------


def calibrate(
    judgements: JudgementSet | Sequence[JudgementSet],
    *,
    mode: Mode = "auto",
    judges: Judges = "single",
    priors: Priors | None = None,
    judged_bt_l2: float | None = None,
    strict: bool = False,
) -> Calibration:
    """Fit the calibration to rubric judgements, anchored on the tournament when there is one.

    Before fitting, every query's windows are checked (:mod:`rcp_ndcg.calibration.coverage`): a query with more
    than 5% invalid windows in a stage the fit reads, or any invalid adaptive tournament window, is flagged. The
    flags are an ``INVALID_WINDOWS`` warning (a :class:`~rcp_ndcg.errors.RcpNdcgWarning`, also kept in the
    calibration's ``diagnostics`` and ``coverage``), or with ``strict`` a refusal.

    A document whose tournament windows are all invalid carries no comparison: the Bradley-Terry fit gives it the
    query's mean ability (the paper's number; the ridge's standard error only when the query has other
    comparisons), and the fit reports the documents as ``NO_VALID_TOURNAMENT_EVIDENCE``
    (``coverage.no_tournament_evidence_documents``), or refuses them with ``strict``.

    Args:
        judgements: The judgements (e.g. :func:`read_judgements` of the stores), one set or several. A window
            judged more than once (one ``record_id``) counts once, by its latest valid judgement; re-judged
            documents (new windows, new record ids) count with their earlier windows.
        mode: ``"tournament"``, ``"rubric_only"``, or ``"auto"``: the tournament fit when every
            rubric query has tournament judgements, the rubric-only fit when none has.
        judges: ``"single"`` (one rubric judge) or ``"pooled"`` (several judges who answered one
            rubric, with a severity each; needs the tournament).
        priors: The priors and penalties of the fit (the paper's, ``Priors()``, when ``None``), the
            Bradley-Terry refit's ``bt_l2`` included.
        judged_bt_l2: The L2 the tournament's live fit used while judging (:func:`judged_bt_l2` of its
            stores). A fit with another ``priors.bt_l2`` refits abilities the adaptive windows were not
            chosen from; it is fitted anyway, with a ``BT_L2_MISMATCH`` warning. ``None``: not checked.
        strict: Refuse to fit when a query is flagged, instead of warning.

    Returns:
        The immutable :class:`Calibration`.

    Raises:
        DataError: no valid rubric judgement, or some rubric queries have tournament judgements
            and others do not (the per-query counts are in ``details``).
        DataError: with ``strict``, a query is flagged for its invalid windows (the flags are in ``details``),
            or a document's tournament windows are all invalid (the documents are in ``details``).
        IdentityError: ``judges="single"`` with several rubric judges, pooled judges who answered
            different rubrics, or two tournament families.
    """
    # One observation per window: its latest valid judgement (JudgementSet.merge), also within a single set.
    judgement_set = JudgementSet.merge([judgements] if isinstance(judgements, JudgementSet) else judgements)
    check_criteria(judgement_set)
    priors = priors or Priors()
    rubric = judgement_set.of_stage("rubric")
    tournament = judgement_set.of_stage("tournament")
    rubric_families = {key: f for key, f in rubric.families.items()}
    tournament_families = {key: f for key, f in tournament.families.items()}
    if not rubric.judgements:
        raise DataError(
            "no rubric judgements to calibrate",
            hint="judge the candidates with stage='rubric' first",
            cli_hint="judge the candidates with `rcp-ndcg judge rubric` first",
        )
    _check_judges(rubric_families, judges)
    if len(tournament_families) > 1:
        raise IdentityError(
            f"the judgements hold {len(tournament_families)} tournament families; one fit anchors on one tournament",
            details={"families": sorted(tournament_families)},
        )
    resolved = _resolve_mode(mode, rubric, tournament)
    if judges == "pooled" and resolved != "tournament":
        raise DataError(
            "pooled judges need the tournament: the rubric-only model has no judge-severity term",
            hint="judge the tournament, or calibrate each judge alone",
        )
    windows = window_coverage(
        judgement_set.judgements, ("rubric", "tournament") if resolved == "tournament" else ("rubric",)
    )
    flags = coverage_flags(windows)
    fit_warnings = []
    if flags:
        message = flags_message(flags)
        if strict:
            raise DataError(
                f"refusing a strict calibration: {message}",
                hint="a refused window (no answer) is asked again by the same judge() call; an unparseable answer "
                "is kept by a resumed pass: re-read the stored answers with a newer parser (reparse), judge into a "
                "new store, or fit without strict",
                cli_hint="a refused window (no answer) is asked again by the same judge command; an unparseable "
                "answer is kept by a resumed pass: re-read the stored answers with a newer parser (`rcp-ndcg judge "
                "reparse`), judge into a new --out store, or fit without --strict",
                details={
                    "flagged": [flag.model_dump(mode="json") for flag in flags],
                    "invalid_window_share": INVALID_WINDOW_SHARE,
                },
            )
        warning = RcpNdcgWarning("INVALID_WINDOWS", message)
        warnings.warn(warning, stacklevel=2)
        fit_warnings.append(FitWarning.model_validate(warning.to_dict()))
    if resolved == "tournament" and judged_bt_l2 is not None and judged_bt_l2 != priors.bt_l2:
        warning = RcpNdcgWarning(
            "BT_L2_MISMATCH",
            f"the fit refits the tournament's Bradley-Terry abilities with L2 {priors.bt_l2:g}, but its live fit "
            f"chose the adaptive windows with {judged_bt_l2:g}",
        )
        warnings.warn(warning, stacklevel=2)
        fit_warnings.append(FitWarning.model_validate(warning.to_dict()))
    criteria = next(iter(rubric_families.values())).criteria
    observations = rubric_observations(rubric, tag_judges=judges == "pooled")
    bt: dict[str, dict[str, float]] | None = None
    bt_se: dict[str, dict[str, float | None]] = {}
    no_evidence: list[str] = []
    if resolved == "tournament":
        bt, bt_se = bradley_terry(tournament, l2=priors.bt_l2)
        no_evidence = no_tournament_evidence(tournament)
        if no_evidence:
            count = len(no_evidence)
            message = (
                f"{count} {'document has' if count == 1 else 'documents have'} no comparison in a valid "
                f"tournament window: the Bradley-Terry fit gives {'it' if count == 1 else 'them'} the query's "
                f"mean ability, which no comparison backs: {', '.join(no_evidence[:5])}" + (" ..." if count > 5 else "")
            )
            if strict:
                raise DataError(
                    f"refusing a strict calibration: {message}",
                    hint="judge a valid tournament window for them (select_opponents plans the opponents), or fit "
                    "without strict",
                    cli_hint="judge a valid tournament window for them (`rcp-ndcg calibration insert --dry-run` "
                    "plans the opponents), or fit without --strict",
                    details={"documents": no_evidence},
                )
            warning = RcpNdcgWarning("NO_VALID_TOURNAMENT_EVIDENCE", message)
            warnings.warn(warning, stacklevel=2)
            fit_warnings.append(FitWarning.model_validate(warning.to_dict()))
    logger.info(
        "calibrating %d queries (%s, %s): %d rubric observations",
        len(observations),
        resolved,
        judges,
        sum(len(rows) for rows in observations.values()),
    )
    try:
        fit = fit_calibration(
            observations,  # type: ignore[arg-type]
            bt_scores=bt,
            mode=resolved,
            judges=judges,
            num_criteria=len(criteria),
            priors=priors,
        )
    except JudgeOverlapError as exc:
        raise DataError(
            f"the pooled judges share too few documents to be put on one scale: {exc}",
            hint="judge shared documents with every judge, or calibrate the judges separately",
            details=exc.report,
        ) from exc
    except ValueError as exc:  # the estimators refuse an unidentifiable fit (e.g. one criterion takes the weight)
        raise DataError(
            f"the {resolved} fit is not identifiable from these judgements: {exc}",
            hint="judge more windows or documents, or check the rubric's pass rates in the judgements",
        ) from exc
    items, queries = fit.items, fit.queries
    uncalibrated_documents = sorted(
        f"{query}/{doc}"
        for query, observed in observations.items()
        for doc in {row[0] for row in observed}
        if doc not in fit.thetas.get(query, {})
    )
    if uncalibrated_documents:
        warning = RcpNdcgWarning(
            "UNCALIBRATED_DOCUMENTS",
            f"{len(uncalibrated_documents)} documents have rubric verdicts but no tournament ability, so the fit "
            f"leaves them out: {', '.join(uncalibrated_documents[:5])}"
            + (" ..." if len(uncalibrated_documents) > 5 else "")
            + "; score them from their rubric verdicts with score_documents, or insert them into the tournament",
        )
        warnings.warn(warning, stacklevel=2)
        fit_warnings.append(FitWarning.model_validate(warning.to_dict()))
    rows = []
    for key in sorted(fit.thetas):
        dataset, query_id = split_namespace(key)
        params = queries.get(key)
        for doc_id, theta in sorted(fit.thetas[key].items()):
            se = fit.theta_se.get(key, {}).get(doc_id)
            if se is None and params is not None:
                bt_se_value = bt_se.get(key, {}).get(doc_id)
                if bt_se_value is not None:
                    se = params.calibrated_se(bt_se_value)
            rows.append(ThetaRow(dataset, query_id, doc_id, float(theta), se, "fit"))
    judge_of = {f.judge_model: key for key, f in rubric_families.items()}
    diagnostics = Diagnostics(
        fit=fit.diagnostics,
        reliability=fit_diagnostics(
            items,
            fit.thetas,
            observations,
            family_of_judge=judge_of,
            judge_severity=fit.judge_severity,
            single_judge=next(iter(judge_of)) if judges == "single" else None,
        ),
        warnings=fit_warnings,
    )
    if fit.diagnostics.ordinal_only:
        logger.warning(
            "the rubric-only fit is ordinal only (%s): the ordering is usable, "
            "the gains are not calibrated probabilities",
            fit.diagnostics.ordinal_reason,
        )
    families = {**tournament_families, **rubric_families} if resolved == "tournament" else rubric_families
    return Calibration(
        mode=resolved,
        items=items,
        queries=queries,
        thetas=tuple(rows),
        families=families,
        judge_severity=fit.judge_severity,
        coverage=_coverage(
            judgement_set, observations, fit.thetas, resolved, windows, flags, uncalibrated_documents, no_evidence
        ),
        diagnostics=diagnostics,
        identity=CalibrationIdentity(
            judgements=short(hash_payload(sorted(j.record_id for j in judgement_set.judgements)), 16),
            families=sorted(families),
            mode=resolved,
            judges=judges,
            priors=priors,
        ),
    )


def _check_judges(rubric_families: Mapping[str, JudgementFamily], judges: Judges) -> None:
    models = sorted({family.judge_model for family in rubric_families.values()})
    if judges == "single":
        if len(rubric_families) > 1:
            raise IdentityError(
                f"judges='single', but the rubric judgements come from {len(rubric_families)} families "
                f"(judges {models})",
                hint="calibrate each judge alone, or pool them with judges='pooled'",
                cli_hint="calibrate each judge alone, or pool them with --judges pooled",
                details={"families": sorted(rubric_families)},
            )
        return
    if judges != "pooled":
        raise IdentityError(f"judges must be 'single' or 'pooled', got {judges!r}")
    if len(models) < 2:
        raise IdentityError(f"judges='pooled' needs two or more judges, got {models}")
    rubric_keys = {family.rubric_key for family in rubric_families.values()}
    if len(rubric_keys) != 1:
        raise IdentityError(
            "the judges answered different rubrics (prompt, criteria, parser or preprocessing differ); "
            "item parameters belong to one rubric",
            details={"rubric_keys": sorted(rubric_keys)},
        )
    if len(models) != len(rubric_families):
        raise IdentityError(f"a judge appears in several rubric families: {models}")


def _resolve_mode(mode: Mode, rubric: JudgementSet, tournament: JudgementSet) -> Literal["tournament", "rubric_only"]:
    if mode == "rubric_only":
        return "rubric_only"
    if mode not in ("auto", "tournament"):
        raise DataError(f"mode must be 'auto', 'tournament' or 'rubric_only', got {mode!r}")
    rubric_queries = set(rubric.query_ids())
    tournament_queries = {key for key in tournament.query_ids() if key in rubric_queries}
    if mode == "auto" and not tournament_queries:
        return "rubric_only"
    lacking = sorted(rubric_queries - tournament_queries)
    if lacking:
        rubric_windows = Counter((j.dataset, j.query_id) for j in rubric.judgements)
        tournament_windows = Counter((j.dataset, j.query_id) for j in tournament.judgements)
        per_dataset = Counter(dataset for dataset, _ in lacking)
        raise DataError(
            f"{len(lacking)} of {len(rubric_queries)} rubric queries have no tournament judgements "
            f"(per dataset: {dict(sorted(per_dataset.items()))}); one fit is either anchored on the tournament "
            "for every query or rubric-only for every query",
            hint="judge the tournament for those queries, or calibrate with mode='rubric_only'",
            cli_hint="judge the tournament for those queries, or calibrate with --mode rubric_only",
            details={
                "queries_without_tournament": [QUERY_ID_SEP.join(pair) for pair in lacking],
                "rubric_queries": len(rubric_queries),
                "tournament_queries": len(tournament_queries),
                "per_dataset": dict(sorted(per_dataset.items())),
                "windows_per_query": {
                    QUERY_ID_SEP.join(pair): {"rubric": rubric_windows[pair], "tournament": tournament_windows[pair]}
                    for pair in sorted(rubric_queries)
                },
            },
        )
    return "tournament"


def _coverage(
    judgements: JudgementSet,
    observations: Mapping[str, Sequence[tuple]],
    thetas: Mapping[str, Mapping[str, float]],
    mode: Literal["tournament", "rubric_only"],
    windows: Mapping[str, Mapping[str, StageWindows]],
    flags: list[Any],
    uncalibrated_documents: list[str],
    no_tournament_evidence_documents: list[str],
) -> CalibrationCoverage:
    """Queries per stage, the queries without calibrated abilities, invalid windows, flags, degenerate documents."""
    windows_per_stage: dict[str, dict[str, int]] = {}
    for stages in windows.values():
        for stage, entry in stages.items():
            total = windows_per_stage.setdefault(stage, {"windows": 0, "invalid": 0})
            total["windows"] += entry.windows
            total["invalid"] += entry.invalid
    totals: dict[tuple[str, str], list[int]] = {}
    for query, rows in observations.items():
        for row in rows:
            passed = sum(int(v) for v in row[1].values())
            entry = totals.setdefault((query, row[0]), [0, 0])
            entry[0] += passed
            entry[1] += len(row[1])
    all_fail = sum(1 for passed, asked in totals.values() if passed == 0)
    all_pass = sum(1 for passed, asked in totals.values() if passed == asked)
    rubric_queries = {namespace(*pair) for pair in judgements.query_ids("rubric")}
    tournament_queries = {namespace(*pair) for pair in judgements.query_ids("tournament")}
    uncalibrated = sorted((rubric_queries | (tournament_queries if mode == "tournament" else set())) - set(thetas))
    return CalibrationCoverage(
        mode=mode,
        queries=QueryCounts(rubric=len(rubric_queries), tournament=len(tournament_queries), calibrated=len(thetas)),
        uncalibrated_queries=uncalibrated,
        uncalibrated_documents=uncalibrated_documents,
        no_tournament_evidence_documents=no_tournament_evidence_documents,
        windows={stage: WindowCount(**total) for stage, total in sorted(windows_per_stage.items())},
        invalid_windows={
            query: {stage: entry for stage, entry in stages.items() if entry.invalid}
            for query, stages in windows.items()
            if any(entry.invalid for entry in stages.values())
        },
        invalid_window_share=INVALID_WINDOW_SHARE,
        flagged_queries=flags,
        documents=len(totals),
        degenerate_documents=DegenerateCounts(all_fail=all_fail, all_pass=all_pass),
        per_dataset={
            dataset: sum(1 for key in thetas if split_namespace(key)[0] == dataset)
            for dataset in sorted({split_namespace(key)[0] for key in thetas})
        },
    )


def population_prior(calibration: Calibration) -> tuple[float, float]:
    """Mean and standard deviation of the calibration's own abilities: the prior a scored document joins.

    Raises:
        DataError: fewer than two abilities, or all equal.
    """
    values = [row.theta for row in calibration.thetas if row.source == "fit"]
    if len(values) < 2 or statistics.pstdev(values) == 0.0:
        raise DataError(f"the calibration has no spread of abilities to form a prior ({len(values)} abilities)")
    return statistics.fmean(values), statistics.pstdev(values)


__all__ = [
    "CALIBRATION_SCHEMA",
    "Calibration",
    "CalibrationIdentity",
    "CalibrationItems",
    "ThetaRow",
    "calibrate",
    "population_prior",
]
