"""Fill the pending reference cases with the recipe's reference implementation's own outputs.

A generated case ships ``expected: {origin: reference, status: pending_gpu, values: null}``: the
numbers the served engine is compared against are the recipe's **reference implementation's own
outputs** for the case's inputs, never the engine's and never a guess (``handover/specs/case-format.md``;
the release checklist's "Fill every ``pending_gpu`` expected value").  This module is that fill: it
loads the recipe's cases, runs the recipe's reference **once** over every pending case's inputs (one
model load; the pairs rows are the reference subprocess's own input format), shapes the outputs into
the case's ``expected.kind`` -- a scores matrix for a rerank recipe, a similarity matrix for an embed
or multi_vector one (cosine for a single-vector reference, the product's MaxSim for a late-interaction
one) -- and edits each case YAML: ``values`` replaced, ``status: pending_gpu`` -> ``reproduced``, and
the pending note's own sentence retired (the report's ``notes_updated`` says whether it matched).

Run from a checkout of the repository::

    python -m rcp_ndcg_test.case_fill --recipe qwen3-reranker-0.6b \
        --reference-python <the family reference venv's python>

The reference runs in its own environment (owner decision 35: this process holds no torch) with the
role's mode (``score`` for rerank, ``embed`` for embed/multi_vector) and the recipe's own tokenizer
spec; ``--budget-s`` bounds the subprocess, because a CPU reference of a large checkpoint can outlive
the one-hour default.  A pending case the mode cannot fill is **skipped per case**, with the reason
recorded in the report and its file left untouched: ``kind: none`` carries no values by design, a
``expected.kind`` the mode does not produce is another stage's values, and a media-bearing case's
values cannot come from the text-only reference (its ``media`` mode reports input facts, not model
outputs), so a media-value fill is a follow-up.  The reference python's path is never written into
the repository: the note carries the device and the date only.
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

from rcp_ndcg_test.cases import Case, default_cases_root, load_cases, text_of
from rcp_ndcg_test.equivalence.fitting import resolved_tokenizer_spec
from rcp_ndcg_test.equivalence.reference import reference_entry, reference_of, run_reference
from rcp_ndcg_test.errors import CaseError, HarnessError, RecipeError
from rcp_ndcg_test.stepwatch import StepWatch, watched

__all__ = ["FillReport", "FilledCase", "SkippedCase", "expected_values", "fill_cases", "main"]

_MODE_OF_ROLE = {"rerank": "score", "embed": "embed", "multi_vector": "embed"}
"""The reference subprocess mode each role's cases are filled from."""

_KIND_OF_MODE = {"score": "scores", "embed": "similarity_matrix"}
"""The case ``expected.kind`` each mode produces."""

_PENDING_NOTE_RETIREMENTS: tuple[tuple[str, str], ...] = (
    (
        "expected.values are filled at the GPU wave from the recipe's reference implementation and the "
        "engine, both recorded,",
        "expected.values are the outputs of the reference implementation itself for these inputs "
        "(filled on {device} before the wave; the wave conformance run compares the served engine "
        "against them),",
    ),
    (
        "Values are filled by the GPU wave (reference + engine recorded).",
        "Values are the outputs of the reference implementation itself for these inputs (filled on "
        "{device} before the wave; the wave conformance run compares the served engine against them).",
    ),
    (
        "status pending_gpu.",
        "status reproduced (values filled from the reference implementation on {device}, {filled_at}).",
    ),
)
"""The pending-case note sentences this fill retires, and what replaces them (the values' provenance).

The wording is the cases lanes' own pending vocabulary; a note matching none of them is left untouched
(the report's ``notes_updated`` is ``False`` then), never reworded by guesswork.  A phrase matches
across a folded scalar's line breaks (the notes are wrapped); an apostrophe appears doubled exactly as
the single-quoted scalar carries it, and the replacements carry none (a bare ``'`` would end the
scalar)."""

_DEFAULT_BUDGET_S = 3600.0
"""The reference subprocess's default wall-clock budget, seconds (``run_reference``'s own default)."""


@dataclass(frozen=True)
class FilledCase:
    """One case the fill wrote.

    Attributes:
        case_id: The case's id (``<recipe>/<case-slug>``).
        path: The case YAML that was edited.
        notes_updated: Whether the note carried one of the pending sentences (and was retired).
    """

    case_id: str
    path: Path
    notes_updated: bool


@dataclass(frozen=True)
class SkippedCase:
    """One pending case the fill cannot produce values for (its file is left untouched).

    Attributes:
        case_id: The case's id (``<recipe>/<case-slug>``).
        reason: Why no values were written: ``kind: none`` carries no values by design, the case's
            ``expected.kind`` is not what the mode produces, or the case carries media the text-only
            reference cannot read (a media-value fill is a follow-up).
    """

    case_id: str
    reason: str


@dataclass(frozen=True)
class FillReport:
    """What one fill did.

    Attributes:
        recipe_id: The recipe whose cases were filled.
        mode: The reference subprocess mode (``score`` or ``embed``).
        device: The reference's device.
        filled: The cases written, sorted by id.
        pending_before: The pending cases found (including the skipped ones).
        skipped: The pending cases no values were written for, each with its reason.
    """

    recipe_id: str
    mode: str
    device: str
    filled: tuple[FilledCase, ...]
    pending_before: int
    skipped: tuple[SkippedCase, ...] = ()

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
        }

    def summary(self) -> str:
        """One line for a log: the recipe, the mode, the counts."""
        return (
            f"{self.recipe_id} [{self.mode} on {self.device}]: filled {len(self.filled)} of "
            f"{self.pending_before} pending case(s), skipped {len(self.skipped)}"
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
    date the fill records in the notes (today when ``None``) and the reference subprocess's wall-clock
    budget in seconds.

    Outputs: a :class:`FillReport`.  Every pending case the mode can fill is written (values, status
    and note) after the reference returns; a pending case the mode cannot fill is skipped with its
    reason (see :func:`_unfillable_reason`) and its file left untouched.  The run is one subprocess per
    recipe, so a reference failure leaves every case untouched.

    Raises:
        CaseError: the recipe's case root or a case's text does not load (a ``text_ref`` whose hash does
            not match).
        HarnessError: the recipe has no reference (a judge), the reference's answers do not align with
            the planned rows, or the reference subprocess fails.
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
    rows, mapping = _plan_rows(fillable)
    document = _run_reference(recipe, mode, rows, reference_python=reference_python, device=device, budget_s=budget_s)
    answers = _answers_by_case(document, rows, mapping)
    filled: list[FilledCase] = []
    for case in fillable:
        values = expected_values(case, answers[case.id], mode)
        filled.append(
            _write_case(
                _case_path(case),
                values,
                device=device,
                filled_at=filled_at or date.today().isoformat(),
            )
        )
    return FillReport(recipe.id, mode, device, tuple(filled), pending_before=len(pending), skipped=tuple(skipped))


def _unfillable_reason(case: Case, mode: str) -> str | None:
    """Why the fill cannot produce this pending case's values, or ``None`` when it can.

    The one home for the fill's per-case refusals: ``kind: none`` carries no values by design, a
    ``expected.kind`` the mode does not produce is another stage's values (the fill never guesses),
    and a media-bearing case's values cannot come from the text-only reference -- its ``media`` mode
    reports input facts, not model outputs -- so it stays ``pending_gpu`` until a media-value fill
    exists.  A skipped case's file is never touched.
    """
    if case.expected.kind == "none":
        return "kind: none carries no values by design"
    wanted = _KIND_OF_MODE[mode]
    if case.expected.kind != wanted:
        return (
            f"expected.kind {case.expected.kind!r} is not what the {mode} mode produces ({wanted!r}); "
            "the fill does not guess another stage's values"
        )
    media = [
        document.id for document in case.inputs.documents if document.image is not None or document.video is not None
    ]
    if media:
        return (
            f"document(s) {', '.join(media)} carry media; the reference's {mode} mode reads text only "
            "and its media mode reports input facts, not model outputs, so a media-value fill is a follow-up"
        )
    return None


def _case_path(case: Case) -> Path:
    """The case YAML's path (the recipe's case directory, pinned by :func:`load_cases`)."""
    directory = getattr(case, "_dir", None)
    if directory is None:  # pragma: no cover - load_cases always pins it
        raise CaseError(f"case {case.id}: no case directory pinned; load it with load_cases")
    return Path(directory) / f"{case.id.split('/', 1)[1]}.yaml"


def _plan_rows(cases: list[Case]) -> tuple[list[dict[str, Any]], list[tuple[Case, int]]]:
    """The reference's pairs rows and, per row, the case and query index it carries.

    One row per query (the reference's own pairs format): the query's text and the case's documents, in
    the case's order.  The run's instruction travels per row when the case declares one.  Only
    fillable cases reach here (the partition owns the refusals), so a document or query without text is
    an internal inconsistency, refused rather than silently sent as a missing part.
    """
    rows: list[dict[str, Any]] = []
    mapping: list[tuple[Case, int]] = []
    for case in cases:
        documents = [text_of(document) for document in case.inputs.documents]
        if any(document is None for document in documents):
            raise CaseError(f"case {case.id}: a document carries no text")
        for index, query in enumerate(case.inputs.queries):
            text = text_of(query)
            if text is None:
                raise CaseError(f"case {case.id}: query {query.id!r} carries no text")
            row: dict[str, Any] = {"query": text, "documents": documents}
            if case.inputs.instruction is not None:
                row["instruction"] = case.inputs.instruction
            rows.append(row)
            mapping.append((case, index))
    return rows, mapping


def _run_reference(
    recipe: Any,
    mode: str,
    rows: list[dict[str, Any]],
    *,
    reference_python: str,
    device: str,
    budget_s: float,
) -> dict[str, Any]:
    """Run the recipe's reference subprocess over the rows, in a scratch directory outside the repository."""
    reference_of(recipe)  # a judge recipe is refused by name here, before any work
    with tempfile.TemporaryDirectory(prefix="rcp-case-fill-") as work:
        pairs_path = Path(work) / "pairs.jsonl"
        pairs_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        with watched(StepWatch(f"case fill {recipe.id}", budget_s)):
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
    """Write one case's values and status into its YAML, and retire the pending note's sentence.

    The edit is line-wise (``values: null`` becomes the rendered matrix, ``status: pending_gpu`` becomes
    ``reproduced``), so every other byte of the hand-written case survives.  The written file is loaded
    again through the case format; a file that no longer loads is restored and the error re-raised, so a
    bad write never stays on disk.
    """
    original = path.read_text(encoding="utf-8")
    text = _replace_line(original, "  values: null", f"  values:\n{_render_values(values)}", path)
    text = _replace_line(text, "  status: pending_gpu", "  status: reproduced", path)
    notes_updated = False
    for pending, replacement in _PENDING_NOTE_RETIREMENTS:
        pattern = _pending_pattern(pending)
        if pattern.search(text):
            filled = replacement.format(device=device, filled_at=filled_at)
            text = pattern.sub(lambda _match, filled=filled: filled, text)
            notes_updated = True
    path.write_text(text, encoding="utf-8")
    from rcp_ndcg_test.cases import load_case

    try:
        load_case(path)
    except CaseError as error:
        path.write_text(original, encoding="utf-8")
        raise CaseError(f"{path}: the filled case does not load ({error}); the file was left unchanged") from error
    return FilledCase(case_id=f"{path.parent.name}/{path.stem}", path=path, notes_updated=notes_updated)


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
        help=f"the reference subprocess's wall-clock budget, seconds (default: {_DEFAULT_BUDGET_S:.0f})",
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
