"""Fill the pending reference cases with the recipe's reference implementation's own outputs.

A generated case ships ``expected: {origin: reference, status: pending_gpu, values: null}``: the
numbers the served engine is compared against are the recipe's **reference implementation's own
outputs** for the case's inputs, never the engine's and never a guess (``handover/specs/case-format.md``;
the release checklist's "Fill every ``pending_gpu`` expected value").  This module is that fill: it
loads the recipe's cases, runs the recipe's reference over the pending cases' inputs (one subprocess
over every case's rows, one model load; the pairs rows are the reference subprocess's own input format),
shapes the outputs into the case's ``expected.kind`` -- a scores matrix for a rerank recipe, a similarity
matrix for an embed or multi_vector one (cosine for a single-vector reference, the product's MaxSim for
a late-interaction one) -- and edits each case YAML: ``values`` replaced, ``status: pending_gpu`` ->
``reproduced``, and the pending note's own sentence retired (a note carrying none of the known
phrasings gets the fill's provenance appended; the report's ``notes_updated`` records the update).

Run from a checkout of the repository::

    python -m rcp_ndcg_test.case_fill --recipe qwen3-reranker-0.6b \
        --reference-python <the family reference venv's python>

The reference runs in its own environment (owner decision 35: this process holds no torch) with the
role's mode (``score`` for rerank, ``embed`` for embed/multi_vector) and the recipe's own tokenizer
spec; ``--budget-s`` bounds the recipe's whole fill (every reference subprocess of the recipe draws on
it), because a CPU reference of a large checkpoint can outlive the one-hour default.  A pending case the
mode cannot fill is **skipped per case**, with the reason recorded in the report and its file left
untouched: ``kind: none`` carries no values by design, and a ``expected.kind`` the mode does not produce
is another stage's values.  A media-bearing case is fillable: the pairs row carries the case's ``media/``
files in the ``media`` field (:func:`_plan_rows`, the media set's own entry shape), which the reference's
mode reads exactly as stage 2's media rows are read.  The case's run-level ``instruction`` enters the
pairs row only when the recipe's client places it (:func:`_places_the_instruction`: the resolved
endpoint config's own ``instruction`` mode -- ``none`` drops it by declaration, so the row must not
carry it either), the same rule the served request follows.  A case the reference refuses (a container it cannot
sample at the recipe's pin, a row it cannot express) is a **recorded exception**: the fill retries the
cases one by one so the refusal cannot take the recipe's other cases down, records the reference's own
reason in the case's note and in the report, and leaves the case ``pending_gpu``.  The reference python's
path is never written into the repository: the note carries the device and the date only.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from rcp_ndcg_test.cases import Case, default_cases_root, load_cases, text_of
from rcp_ndcg_test.equivalence.fitting import resolved_tokenizer_spec
from rcp_ndcg_test.equivalence.reference import reference_entry, reference_of, run_reference
from rcp_ndcg_test.equivalence.wire import recipe_config
from rcp_ndcg_test.errors import CaseError, HarnessError, RecipeError
from rcp_ndcg_test.observe.media_set import image_entry, video_entry
from rcp_ndcg_test.stepwatch import StepWatch, watched

__all__ = [
    "ExceptionCase",
    "FillReport",
    "FilledCase",
    "SkippedCase",
    "expected_values",
    "fill_cases",
    "main",
]

_MODE_OF_ROLE = {"rerank": "score", "embed": "embed", "multi_vector": "embed"}
"""The reference subprocess mode each role's cases are filled from."""

_KIND_OF_MODE = {"score": "scores", "embed": "similarity_matrix"}
"""The case ``expected.kind`` each mode produces."""

_VALUES_SENTENCE = (
    "the outputs of the reference implementation itself for these inputs (filled on {device} before the "
    "wave; the wave conformance run compares the served engine against them)"
)
"""What the values' provenance reads as once a pending sentence is retired (the fill's own wording)."""

_PENDING_NOTE_RETIREMENTS: tuple[tuple[str, str], ...] = (
    (
        "expected.values are filled at the GPU wave from the recipe's reference implementation and the "
        "engine, both recorded,",
        f"expected.values are {_VALUES_SENTENCE},",
    ),
    (
        "Values are filled by the GPU wave (reference + engine recorded).",
        f"Values are {_VALUES_SENTENCE}.",
    ),
    (
        "status pending_gpu.",
        "status reproduced (values filled from the reference implementation on {device}, {filled_at}).",
    ),
    (
        "expected.values stays null until the GPU wave fills it from the reference implementation and "
        "the engine (origin reference, status pending_gpu).",
        f"expected.values are {_VALUES_SENTENCE}.",
    ),
    (
        "GPU wave fills values;",
        f"Values are {_VALUES_SENTENCE};",
    ),
    (
        "expected.values are filled by the GPU wave (reference implementation and engine recorded side by side);",
        f"expected.values are {_VALUES_SENTENCE};",
    ),
    (
        "The wave fills values from the reference (transformers, GPU) and the served engine and records both.",
        f"Values are {_VALUES_SENTENCE}.",
    ),
    (
        "origin reference, filled by the GPU wave.",
        f"origin reference; the values are {_VALUES_SENTENCE}.",
    ),
)
"""The pending-case note sentences this fill retires, and what replaces them (the values' provenance).

The wording is the cases lanes' own pending vocabulary -- the shipped generated cases' phrasings
("GPU wave fills values;", "expected.values stays null until ...", the video and text cases' variants)
and the older families' three.  A note carrying none of them gets :data:`_FILLED_SENTENCE` appended
instead (:func:`_append_note`) -- the fill's own provenance, never a reworded guess.  A phrase matches
across a folded scalar's line breaks (the notes are wrapped); an apostrophe appears doubled exactly as
the single-quoted scalar carries it, and the replacements carry none (a bare ``'`` would end the
scalar)."""

_FILLED_SENTENCE = "status reproduced (values filled from the reference implementation on {device}, {filled_at})."
"""What a filled case's note ends with: the values' provenance and the device and date of the fill."""

_EXCEPTION_SENTENCE = (
    "The fill on {device} ({filled_at}) could not produce this case's values: {reason} "
    "The case stays pending_gpu until the reference can read its inputs (the reason is recorded here "
    "and in the case-fill report)."
)
"""What a refused case's note ends with: the reference's own refusal, and that the case stays pending."""

_DEFAULT_BUDGET_S = 3600.0
"""The recipe's default fill budget, seconds (the first reference subprocess's own default)."""


@dataclass(frozen=True)
class FilledCase:
    """One case the fill wrote.

    Attributes:
        case_id: The case's id (``<recipe>/<case-slug>``).
        path: The case YAML that was edited.
        notes_updated: Whether the fill updated the note: a recognized pending sentence was retired, or
            (the shipped generated cases' notes carry none) the fill's provenance sentence was appended.
    """

    case_id: str
    path: Path
    notes_updated: bool


@dataclass(frozen=True)
class SkippedCase:
    """One pending case the fill cannot produce values for (its file is left untouched).

    Attributes:
        case_id: The case's id (``<recipe>/<case-slug>``).
        reason: Why no values were written: ``kind: none`` carries no values by design, or the case's
            ``expected.kind`` is not what the mode produces.
    """

    case_id: str
    reason: str


@dataclass(frozen=True)
class ExceptionCase:
    """One pending case the reference refused (no values written; the refusal is recorded).

    Attributes:
        case_id: The case's id (``<recipe>/<case-slug>``).
        path: The case YAML the refusal was recorded in.
        reason: The reference's own refusal, one line (its stderr's last non-empty line).
        notes_updated: Whether the refusal was recorded in the note (``False`` when the note already
            carried the reason: a re-run never appends it twice).
    """

    case_id: str
    path: Path
    reason: str
    notes_updated: bool


@dataclass(frozen=True)
class FillReport:
    """What one fill did.

    Attributes:
        recipe_id: The recipe whose cases were filled.
        mode: The reference subprocess mode (``score`` or ``embed``).
        device: The reference's device.
        filled: The cases written, sorted by id.
        pending_before: The pending cases found (including the skipped and refused ones).
        skipped: The pending cases no values were written for, each with its reason (the mode does not
            produce their kind: ``kind: none``, or another stage's kind).
        exceptions: The pending cases the reference refused, each with its reason recorded (its file
            stays ``pending_gpu`` with the reason in its note).
    """

    recipe_id: str
    mode: str
    device: str
    filled: tuple[FilledCase, ...]
    pending_before: int
    skipped: tuple[SkippedCase, ...] = ()
    exceptions: tuple[ExceptionCase, ...] = ()

    def as_document(self) -> dict[str, Any]:
        """The report as JSON-ready data (the CLI's stdout)."""
        return {
            "recipe": self.recipe_id,
            "mode": self.mode,
            "device": self.device,
            "pending_before": self.pending_before,
            "filled": [
                {"case_id": filled.case_id, "path": str(filled.path), "notes_updated": filled.notes_updated}
                for filled in self.filled
            ],
            "skipped": [{"case_id": skipped.case_id, "reason": skipped.reason} for skipped in self.skipped],
            "exceptions": [
                {
                    "case_id": exception.case_id,
                    "path": str(exception.path),
                    "reason": exception.reason,
                    "notes_updated": exception.notes_updated,
                }
                for exception in self.exceptions
            ],
        }

    def summary(self) -> str:
        """One line for a log: the recipe, the mode, the counts."""
        return (
            f"{self.recipe_id} [{self.mode} on {self.device}]: filled {len(self.filled)} of "
            f"{self.pending_before} pending case(s), skipped {len(self.skipped)}, "
            f"refused {len(self.exceptions)}"
        )


def fill_cases(
    recipe: Any,
    cases_root: str | Path | None = None,
    *,
    reference_python: str,
    device: str = "cpu",
    filled_at: str | None = None,
    budget_s: float = _DEFAULT_BUDGET_S,
) -> FillReport:
    """Fill every fillable pending case of ``recipe`` with the reference implementation's own outputs.

    Inputs: the loaded recipe, the cases root (the package's ``cases/`` when ``None``), the python that
    runs the recipe's reference (its own environment; never this process), the reference's device, the
    date the fill records in the notes (today when ``None``) and the recipe's whole fill's wall-clock
    budget in seconds (every reference subprocess of the recipe draws on it).

    Outputs: a :class:`FillReport`.  Every pending case the mode can fill is written (values, status
    and note) after the reference returns; a pending case the mode cannot fill is skipped with its
    reason (see :func:`_unfillable_reason`) and its file left untouched.  The reference runs once over
    every fillable case's rows; when that run fails, every case is retried on its own, so one case the
    reference refuses (a container it cannot sample, a row it cannot express) cannot take the recipe's
    other cases down -- the refused case is recorded as an exception (its reason in the report and in
    its own note) and stays ``pending_gpu``.

    Raises:
        CaseError: the recipe's case root or a case's text does not load (a ``text_ref`` whose hash does
            not match).
        HarnessError: the recipe has no reference (a judge), the reference's answers do not align with
            the planned rows, or the recipe's whole fill outlives its budget.
    """
    mode = _MODE_OF_ROLE.get(str(recipe.role))
    if mode is None:
        raise CaseError(
            f"recipe {recipe.id}: role {recipe.role!r} has no case-fill mode (a judge recipe has no reference)"
        )
    root = cases_root if cases_root is not None else default_cases_root()
    bundle = load_cases(root, recipe=recipe, check_lengths=False)
    pending = [case for case in bundle.cases if case.expected.values is None and case.expected.status == "pending_gpu"]
    fillable: list[Case] = []
    skipped: list[SkippedCase] = []
    for case in pending:
        reason = _unfillable_reason(case, mode)
        if reason is None:
            fillable.append(case)
        else:
            skipped.append(SkippedCase(case_id=case.id, reason=reason))
    if not fillable:
        return FillReport(recipe.id, mode, device, (), pending_before=len(pending), skipped=tuple(skipped))
    watch = StepWatch(f"case fill {recipe.id}", budget_s)
    with watched(watch):
        answers = _answers_per_case(
            recipe, mode, fillable, reference_python=reference_python, device=device, watch=watch
        )
    filled: list[FilledCase] = []
    exceptions: list[ExceptionCase] = []
    for case in fillable:
        answer, refusal = answers[case.id]
        values: list[list[float]] | None = None
        if refusal is None:
            try:
                values = expected_values(case, answer, mode)
            except HarnessError as error:  # the reference answered this case's rows in another shape
                refusal = _refusal_reason(error)
        if values is None:
            exceptions.append(
                _record_exception(
                    case,
                    refusal or "the reference returned no values for this case",
                    device=device,
                    filled_at=filled_at or date.today().isoformat(),
                )
            )
            continue
        filled.append(
            _write_case(
                _case_path(case),
                values,
                device=device,
                filled_at=filled_at or date.today().isoformat(),
            )
        )
    return FillReport(
        recipe.id,
        mode,
        device,
        tuple(filled),
        pending_before=len(pending),
        skipped=tuple(skipped),
        exceptions=tuple(exceptions),
    )


def _answers_per_case(
    recipe: Any,
    mode: str,
    fillable: list[Case],
    *,
    reference_python: str,
    device: str,
    watch: StepWatch,
) -> dict[str, tuple[list[dict[str, Any]], str | None]]:
    """The reference's answers per case: ``{case id: (answers, refusal)}`` (exactly one is set).

    The fast path is one subprocess over every fillable case's rows (one model load).  When that run
    fails -- one case's rows can make the reference exit, and its rows cannot be dropped without
    guessing -- every case is retried on its own, so the refusal lands on the case whose rows carry it
    and the recipe's other cases still fill.  A per-case failure is recorded with the reference's own
    reason; a failure that leaves no budget is the recipe's (its fill cannot continue) and is raised.
    """
    rows, mapping = _plan_rows(recipe, fillable)
    try:
        document = _run_reference(recipe, mode, rows, reference_python=reference_python, device=device)
        answers = _answers_by_case(document, rows, mapping)
        return {case.id: (answers[case.id], None) for case in fillable}
    except HarnessError:
        if watch.remaining_s() <= 0:
            raise  # the budget is spent: a per-case retry would only re-record the same overrun
    per_case: dict[str, tuple[list[dict[str, Any]], str | None]] = {}
    for case in fillable:
        case_rows, case_mapping = _plan_rows(recipe, [case])
        try:
            document = _run_reference(recipe, mode, case_rows, reference_python=reference_python, device=device)
            per_case[case.id] = (_answers_by_case(document, case_rows, case_mapping)[case.id], None)
        except HarnessError as error:
            if watch.remaining_s() <= 0:
                raise
            per_case[case.id] = ([], _refusal_reason(error))
    return per_case


def _refusal_reason(error: HarnessError) -> str:
    """The reference's own refusal, one line (the last non-empty line of its error).

    A reference refuses a row by exiting with its message on stderr (``raise SystemExit``), so the last
    non-empty line of the raised error's text is that message; an answer-shape refusal is the whole
    message (already one line).  Whitespace is folded so the reason reads as one sentence in a note.
    """
    lines = [" ".join(line.split()) for line in str(error).splitlines() if line.strip()]
    return lines[-1] if lines else "the reference failed"


def _unfillable_reason(case: Case, mode: str) -> str | None:
    """Why the fill cannot produce this pending case's values, or ``None`` when it can.

    The one home for the fill's per-case refusals: ``kind: none`` carries no values by design, and a
    ``expected.kind`` the mode does not produce is another stage's values (the fill never guesses).  A
    media-bearing case is fillable -- :func:`_plan_rows` reads the case's ``media/`` files into the pairs
    row's ``media`` field, which the reference's own mode reads like stage 2's media rows -- so it is not
    refused here.  A skipped case's file is never touched.
    """
    if case.expected.kind == "none":
        return "kind: none carries no values by design"
    wanted = _KIND_OF_MODE[mode]
    if case.expected.kind != wanted:
        return (
            f"expected.kind {case.expected.kind!r} is not what the {mode} mode produces ({wanted!r}); "
            "the fill does not guess another stage's values"
        )
    return None


def _case_path(case: Case) -> Path:
    """The case YAML's path (the recipe's case directory, pinned by :func:`load_cases`)."""
    return _case_dir(case) / f"{case.id.split('/', 1)[1]}.yaml"


def _case_dir(case: Case) -> Path:
    """The case's recipe directory: where the case YAML and its ``media/<file>`` assets live."""
    directory = getattr(case, "_dir", None)
    if directory is None:  # pragma: no cover - load_cases always pins it
        raise CaseError(f"case {case.id}: no case directory pinned; load it with load_cases")
    return Path(directory)


def _media_entry(case: Case, relative: str) -> dict[str, Any]:
    """One case document's media file as the pairs ``media`` entry the media stage's rows carry.

    The file is read from the case's own directory (the format's ``media/<file>``) and encoded by the
    media set's own builder (:func:`rcp_ndcg_test.observe.media_set.image_entry` / ``video_entry``: the
    ``data:`` URI plus its kind), so one encoder serves the synthetic media set and the cases.  The type
    and dimensions come from the product's one home (``rcp_ndcg.data.media``: its suffix-to-MIME tables
    and :func:`~rcp_ndcg.data.media.image_dimensions`, a clip's header through
    :func:`~rcp_ndcg.data.media.probe_video_header`): a container whose frame count or rate the probe
    cannot read is refused here, never sent as a factless entry the reference would sample differently.

    Raises:
        CaseError: the file's type is unknown, or its dimensions or (for a container) its frame count and
            rate are unreadable.
    """
    from rcp_ndcg.data.media import (
        IMAGE_MIME_BY_SUFFIX,
        VIDEO_MIME_BY_SUFFIX,
        image_dimensions,
        probe_video_header,
    )

    path = _case_dir(case) / relative
    payload = path.read_bytes()
    suffix = path.suffix.lower()
    if suffix in VIDEO_MIME_BY_SUFFIX:
        header = probe_video_header(payload)
        if (
            header is None
            or header.width is None
            or header.height is None
            or header.num_frames is None
            or header.fps is None
        ):
            raise CaseError(
                f"case {case.id}: the video {relative!r} does not state its own frame count and rate "
                f"({header!r}): the reference samples a container from its recorded facts, and a probe "
                "that cannot read them would leave the reference to guess"
            )
        return video_entry(
            relative,
            header.width,
            header.height,
            header.num_frames,
            fps=header.fps,
            payload=payload,
            mime=VIDEO_MIME_BY_SUFFIX[suffix],
        )
    mime = IMAGE_MIME_BY_SUFFIX.get(suffix)
    if mime is None:
        raise CaseError(
            f"case {case.id}: unknown media type {suffix!r} for {relative!r}; the product's media table "
            f"knows {sorted(IMAGE_MIME_BY_SUFFIX)} and {sorted(VIDEO_MIME_BY_SUFFIX)}"
        )
    width, height = image_dimensions(payload)
    if width is None or height is None:
        raise CaseError(
            f"case {case.id}: the image {relative!r} has no readable dimensions; the product's probe cannot size it"
        )
    return image_entry(relative, width, height, payload=payload, mime=mime)


def _plan_rows(recipe: Any, cases: list[Case]) -> tuple[list[dict[str, Any]], list[tuple[Case, int]]]:
    """The reference's pairs rows and, per row, the case and query index it carries.

    One row per query (the reference's own pairs format): the query's text, the case's documents in the
    case's order, and -- when the case carries media -- the documents' ``media`` entries, one list per
    document position (the shape the media stage's rows use; a document's entries are its image or video,
    built from the case's ``media/`` files by :func:`_media_entry`).  A media-only document's text is the
    empty string (the reference reads the side's content from its media); a document with neither text
    nor media, or a query without text, is refused rather than silently sent as a missing part.  The
    run's instruction travels per row only when the recipe's client places it (:func:`_places_the_instruction`)
    -- the reference must read the same inputs the served engine does, never an instruction the client
    drops by declaration.  Only fillable cases reach here (the partition owns the refusals).
    """
    rows: list[dict[str, Any]] = []
    mapping: list[tuple[Case, int]] = []
    places_instruction = _places_the_instruction(recipe)
    for case in cases:
        documents: list[str] = []
        media_documents: list[list[dict[str, Any]]] = []
        for document in case.inputs.documents:
            text = text_of(document)
            entries = [
                _media_entry(case, relative) for relative in (document.image, document.video) if relative is not None
            ]
            if text is None and not entries:
                raise CaseError(f"case {case.id}: document {document.id!r} carries no text and no media")
            documents.append(text if text is not None else "")
            media_documents.append(entries)
        for index, query in enumerate(case.inputs.queries):
            text = text_of(query)
            if text is None:
                raise CaseError(f"case {case.id}: query {query.id!r} carries no text")
            row: dict[str, Any] = {"query": text, "documents": documents}
            if any(media_documents):
                row["media"] = {"query": [], "documents": media_documents}
            if case.inputs.instruction is not None and places_instruction:
                row["instruction"] = case.inputs.instruction
            rows.append(row)
            mapping.append((case, index))
    return rows, mapping


def _places_the_instruction(recipe: Any) -> bool:
    """Whether the recipe's client places a case's instruction on the wire (the pairs row carries it only then).

    One home with the client rule: the product's resolved endpoint config
    (:func:`rcp_ndcg_test.equivalence.wire.recipe_config`, the same object the role clients are built
    from) decides -- ``instruction: none`` sends none, so the pairs row must not carry the field either
    (the model's own frame already holds its default instruction, and the reference refuses or ignores a
    row that carries one), while ``fold`` (and the rerank role's own default) folds it into the query,
    which the reference subprocess folds the same way.  Reading the resolved mode here rather than the
    raw ``client`` key keeps the role default in the product's config, never a second copy.
    """
    return recipe_config(recipe, None).instruction != "none"


def _run_reference(
    recipe: Any,
    mode: str,
    rows: list[dict[str, Any]],
    *,
    reference_python: str,
    device: str,
) -> dict[str, Any]:
    """Run the recipe's reference subprocess over the rows, in a scratch directory outside the repository.

    The subprocess's timeout is the caller's step watch's remaining budget (``fill_cases`` owns the
    recipe's one watch, so every run of the recipe draws on the same budget).
    """
    reference_of(recipe)  # a judge recipe is refused by name here, before any work
    with tempfile.TemporaryDirectory(prefix="rcp-case-fill-") as work:
        pairs_path = Path(work) / "pairs.jsonl"
        pairs_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return run_reference(
            reference_python,
            reference_entry(recipe),
            mode=mode,
            pairs_path=pairs_path,
            out_path=Path(work) / "reference.json",
            tokenizer_spec=resolved_tokenizer_spec(recipe),
            recipe=recipe,
            device=device,
        )


def _answers_by_case(
    document: dict[str, Any], rows: list[dict[str, Any]], mapping: list[tuple[Case, int]]
) -> dict[str, list[dict[str, Any]]]:
    """The reference's per-row outputs, grouped by case in the case's query order.

    The reference reports each row's ``index`` (its position in the pairs file); a missing, duplicated or
    out-of-range index is refused, so a misaligned answer can never be written as a case's values.
    """
    reported = document.get("rows")
    if not isinstance(reported, list) or len(reported) != len(rows):
        raise HarnessError(
            f"the reference returned {len(reported) if isinstance(reported, list) else 'no'} row(s) for "
            f"{len(rows)} pairs row(s)"
        )
    by_index: dict[int, dict[str, Any]] = {}
    for answer in reported:
        if not isinstance(answer, dict) or not isinstance(answer.get("index"), int):
            raise HarnessError(f"the reference returned a row without an integer index: {answer!r}")
        index = int(answer["index"])
        if not 0 <= index < len(rows) or index in by_index:
            raise HarnessError(f"the reference returned a bad row index {index} (for {len(rows)} rows)")
        by_index[index] = answer
    answers: dict[str, list[dict[str, Any]]] = {}
    for index, (case, _) in enumerate(mapping):
        answers.setdefault(case.id, []).append(by_index[index])
    return answers


def expected_values(case: Case, answers: list[dict[str, Any]], mode: str) -> list[list[float]]:
    """The case's ``expected.values`` from the reference's per-row answers, shaped by the mode's kind.

    ``score``: one row of scores per query, one column per document.  ``embed``: the query x document
    similarity matrix of the reference's own vectors -- cosine of the L2-normalized single vectors, the
    product's MaxSim for a late-interaction reference.  A shape mismatch is a typed refusal, never a
    silently padded or truncated matrix.
    """
    return _score_values(case, answers) if mode == "score" else _similarity_values(case, answers)


def _score_values(case: Case, answers: list[dict[str, Any]]) -> list[list[float]]:
    """One row of scores per query, one column per document (the case's ``scores`` kind)."""
    values: list[list[float]] = []
    for answer in answers:
        scores = answer.get("scores")
        if not isinstance(scores, list) or len(scores) != len(case.inputs.documents):
            raise HarnessError(
                f"case {case.id}: the reference returned "
                f"{len(scores) if isinstance(scores, list) else 'no'} score(s) for "
                f"{len(case.inputs.documents)} document(s)"
            )
        values.append([_finite(float(score)) for score in scores])
    return values


def _similarity_values(case: Case, answers: list[dict[str, Any]]) -> list[list[float]]:
    """One row of cosine similarities per query, one column per document (the ``similarity_matrix`` kind).

    The vectors are the reference's own; the matrix is computed the way the conformance runner computes
    it (``rcp_ndcg_test.conformance._similarity_matrix``): cosine of the L2-normalized single vectors,
    the product's MaxSim (``rcp_ndcg.retrieval.maxsim``) for a late-interaction reference.
    """
    from rcp_ndcg.inference.types import l2_normalize
    from rcp_ndcg.retrieval.maxsim import maxsim_topk

    values: list[list[float]] = []
    for answer in answers:
        queries = answer.get("query_vectors")
        documents = answer.get("document_vectors")
        if not isinstance(queries, list) or len(queries) != 1 or not isinstance(documents, list):
            raise HarnessError(
                f"case {case.id}: the reference returned {len(queries) if isinstance(queries, list) else 'no'} "
                f"query vector(s) for one row (and {len(documents) if isinstance(documents, list) else 'no'} "
                "document vector(s))"
            )
        if len(documents) != len(case.inputs.documents):
            raise HarnessError(
                f"case {case.id}: the reference returned {len(documents)} document vector(s) for "
                f"{len(case.inputs.documents)} document(s)"
            )
        if _is_multi_vector(documents):
            embeddings = [_ragged(queries), _ragged(documents)]
            scores, indices = maxsim_topk(embeddings[1], embeddings[0], k=len(documents))
            row = [math.nan] * len(documents)
            for position, document_index in enumerate(np.asarray(indices).reshape(-1)):
                row[int(document_index)] = _finite(float(np.asarray(scores).reshape(-1)[position]))
            if any(math.isnan(value) for value in row):
                raise HarnessError(f"case {case.id}: the MaxSim matrix left a document unscored: {row}")
            values.append(row)
        else:
            query = np.asarray(l2_normalize(np.asarray(queries, dtype=np.float64)), dtype=np.float64)
            document = np.asarray(l2_normalize(np.asarray(documents, dtype=np.float64)), dtype=np.float64)
            values.append([_finite(float(value)) for value in (query @ document.T)[0]])
    return values


def _is_multi_vector(items: list[Any]) -> bool:
    """Whether a reference's per-item vectors are ragged (one list of vectors per item)."""
    return bool(items) and isinstance(items[0], list) and bool(items[0]) and isinstance(items[0][0], list)


def _ragged(items: list[Any]) -> Any:
    """A reference's per-item vectors as the product's ragged :class:`Embeddings` (one item per row)."""
    from rcp_ndcg.inference.types import Embeddings

    flat = [vector for item in items for vector in item]
    offsets = np.cumsum([0, *(len(item) for item in items)]).astype(np.int64)
    return Embeddings(vectors=np.asarray(flat, dtype=np.float64), offsets=offsets)


def _finite(value: float) -> float:
    """The value, refused when it is not finite (a NaN or infinity is no case's expected value)."""
    if not math.isfinite(value):
        raise HarnessError(f"the reference returned a non-finite value ({value!r})")
    return value


def _write_case(path: Path, values: list[list[float]], *, device: str, filled_at: str) -> FilledCase:
    """Write one case's values and status into its YAML, and record the fill in the note.

    The edit is line-wise (``values: null`` becomes the rendered matrix, ``status: pending_gpu`` becomes
    ``reproduced``), so every other byte of the hand-written case survives.  The note records the fill:
    a sentence from :data:`_PENDING_NOTE_RETIREMENTS` is retired where it stands, and a note carrying
    none of them (the shipped generated cases') gets :data:`_FILLED_SENTENCE` appended (:func:`_append_note`)
    -- a filled case's note always says where its values came from.  The written file is loaded again
    through the case format; a file that no longer loads is restored and the error re-raised, so a bad
    write never stays on disk.
    """
    original = path.read_text(encoding="utf-8")
    text = _replace_line(original, "  values: null", f"  values:\n{_render_values(values)}", path)
    text = _replace_line(text, "  status: pending_gpu", "  status: reproduced", path)
    text, notes_updated = _update_note(text, device=device, filled_at=filled_at, path=path)
    path.write_text(text, encoding="utf-8")
    from rcp_ndcg_test.cases import load_case

    try:
        load_case(path)
    except CaseError as error:
        path.write_text(original, encoding="utf-8")
        raise CaseError(f"{path}: the filled case does not load ({error}); the file was left unchanged") from error
    return FilledCase(case_id=f"{path.parent.name}/{path.stem}", path=path, notes_updated=notes_updated)


def _update_note(text: str, *, device: str, filled_at: str, path: Path) -> tuple[str, bool]:
    """The case text with its note recording the fill: ``(text, notes_updated)``.

    A sentence from :data:`_PENDING_NOTE_RETIREMENTS` is retired where it stands (the shipped generated
    cases' pending vocabulary), and a note carrying none of them gets :data:`_FILLED_SENTENCE` appended
    (:func:`_append_note`) -- a filled case's note always says where its values came from.
    """
    notes_updated = False
    for pending, replacement in _PENDING_NOTE_RETIREMENTS:
        pattern = _pending_pattern(pending)
        if pattern.search(text):
            filled = replacement.format(device=device, filled_at=filled_at)
            text = pattern.sub(lambda _match, filled=filled: filled, text)
            notes_updated = True
    if not notes_updated:
        text = _append_note(text, _FILLED_SENTENCE.format(device=device, filled_at=filled_at), path)
        notes_updated = True
    return text, notes_updated


def _record_exception(case: Case, reason: str, *, device: str, filled_at: str) -> ExceptionCase:
    """Record the reference's refusal of one case in its own note (its values stay null, ``pending_gpu``).

    The sentence names the fill, the device and the date and carries the reference's own reason, so the
    case file itself says why it is still pending; a note already carrying the reason is left as it is
    (re-running the fill never appends it twice).  The written file is loaded again through the case
    format; a file that no longer loads is restored and the error re-raised.
    """
    path = _case_path(case)
    original = path.read_text(encoding="utf-8")
    if reason and reason in case.notes:
        return ExceptionCase(case_id=case.id, path=path, reason=reason, notes_updated=False)
    sentence = _EXCEPTION_SENTENCE.format(device=device, filled_at=filled_at, reason=reason)
    text = _append_note(original, sentence, path)
    path.write_text(text, encoding="utf-8")
    from rcp_ndcg_test.cases import load_case

    try:
        load_case(path)
    except CaseError as error:
        path.write_text(original, encoding="utf-8")
        raise CaseError(
            f"{path}: the case with the recorded refusal does not load ({error}); the file was left unchanged"
        ) from error
    return ExceptionCase(case_id=case.id, path=path, reason=reason, notes_updated=True)


def _append_note(text: str, sentence: str, path: Path) -> str:
    """Append ``sentence`` to the case's ``notes`` scalar, in whatever style that scalar is written.

    The notes are hand-written prose in a folded (``>-``) or literal (``|-``) block, a single-quoted
    scalar or a plain one; the sentence goes at the end of the note's own text, so the note reads as one
    statement and the file's other bytes survive.  The scalar's end comes from the YAML parser's own
    marks, never a line scan: a note may contain lines that look like keys or end in an apostrophe.  The
    sentence is escaped for the scalar's style (a single-quoted scalar doubles its apostrophes; a plain
    one cannot carry a ``: `` or a `` #``, so such a note is re-dumped rather than broken).

    Raises:
        CaseError: the file has no ``notes`` scalar to append to (a filled case's note must record the fill).
    """
    notes = None
    document = yaml.compose(text)
    if isinstance(document, yaml.MappingNode):
        notes = next((value for key, value in document.value if key.value == "notes"), None)
    if not isinstance(notes, yaml.ScalarNode):
        raise CaseError(f"{path}: no notes scalar to record the fill in")
    if notes.style is None and re.search(r":\s|\s#", sentence):
        # A plain scalar cannot carry a ``: `` (it starts a mapping value) or a `` #`` (a comment): the
        # note is re-dumped around the updated value instead of being broken.
        value = str(notes.value) + " " + sentence
        dumped = yaml.safe_dump({"notes": value}, width=110, allow_unicode=True, sort_keys=False)
        return text[: notes.start_mark.index].rstrip(" ") + dumped[len("notes:") :] + text[notes.end_mark.index :]
    if notes.style == "'":  # a single-quoted scalar carries an apostrophe doubled
        sentence = sentence.replace("'", "''")
    at = notes.end_mark.index
    if notes.style in (">", "|"):  # the block's end mark sits on the line after its last content line
        at -= 1 if text[at - 1] == "\n" else 0
    elif notes.style == "'":  # just before the closing quote
        at -= 1
    return text[:at] + " " + sentence + text[at:]


def _pending_pattern(phrase: str) -> re.Pattern[str]:
    """The phrase as a pattern that matches across a folded note scalar's line breaks and indentation.

    The notes are hand-written single-quoted scalars (an apostrophe appears doubled there) and
    ``safe_dump``ed plain ones (a single apostrophe); the pattern accepts either.
    """
    words = [re.escape(word).replace("'", "'{1,2}") for word in phrase.split()]
    return re.compile(r"\s+".join(words))


def _replace_line(text: str, line: str, replacement: str, path: Path) -> str:
    """Replace the one line equal to ``line`` (refusing zero or several matches: no silent no-op)."""
    lines = text.splitlines(keepends=True)
    indexes = [index for index, candidate in enumerate(lines) if candidate.rstrip("\n") == line]
    if len(indexes) != 1:
        raise CaseError(f"{path}: expected exactly one {line.strip()!r} line, found {len(indexes)}")
    index = indexes[0]
    lines[index] = replacement if replacement.endswith("\n") else replacement + "\n"
    return "".join(lines)


def _render_values(values: list[list[float]]) -> str:
    """The values as the case YAML's block sequence of rows (one row per query, one column per document)."""
    if not values or any(not row for row in values):
        raise HarnessError(f"the reference returned no values to write: {values!r}")
    lines: list[str] = []
    for row in values:
        rendered = [_yaml_float(value) for value in row]
        lines.append(f"  - - {rendered[0]}")
        lines.extend(f"    - {value}" for value in rendered[1:])
    return "\n".join(lines) + "\n"


def _yaml_float(value: float) -> str:
    """One float as a YAML float (an exponent needs a decimal point: ``1e-05`` would read as a string)."""
    text = repr(float(value))
    mantissa, separator, exponent = text.partition("e")
    if separator and "." not in mantissa:
        mantissa += ".0"
    return f"{mantissa}e{exponent}" if separator else text


def main(argv: list[str] | None = None) -> int:
    """The CLI: ``python -m rcp_ndcg_test.case_fill --recipe <dir-or-id> --reference-python <python>``."""
    from rcp_ndcg_vllm.recipe import load_recipe

    parser = argparse.ArgumentParser(
        prog="python -m rcp_ndcg_test.case_fill",
        description="Fill a recipe's pending reference cases from its reference implementation's own outputs.",
    )
    parser.add_argument("--recipe", required=True, help="the recipe directory (with family.yaml) or its id")
    parser.add_argument("--cases-root", default=None, help="the cases root (default: the package's cases/)")
    parser.add_argument("--reference-python", required=True, help="the python that runs the recipe's reference")
    parser.add_argument("--device", default="cpu", help="the reference's device (default: cpu)")
    parser.add_argument("--filled-at", default=None, help="the date recorded in the notes (default: today)")
    parser.add_argument(
        "--budget-s",
        type=float,
        default=_DEFAULT_BUDGET_S,
        help=f"the recipe's whole fill's wall-clock budget, seconds (default: {_DEFAULT_BUDGET_S:.0f})",
    )
    args = parser.parse_args(argv)
    try:
        recipe = load_recipe(args.recipe)
        report = fill_cases(
            recipe,
            args.cases_root,
            reference_python=args.reference_python,
            device=args.device,
            filled_at=args.filled_at,
            budget_s=args.budget_s,
        )
    except (CaseError, HarnessError, RecipeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report.as_document(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
