"""The case fill: the reference implementation's own outputs into the pending cases.

The tool is :mod:`rcp_ndcg_test.case_fill`.  These tests copy a fixture recipe whose ``reference.py``
is a fake subprocess and exercise the tool end to end over case YAMLs in ``tmp_path``: the values
written are the reference's outputs, ``status: pending_gpu`` becomes ``reproduced``, the pending note's
own sentence is retired (or, for the shipped cases' notes, which carry no such sentence, the fill's
provenance is appended), a media-bearing case's files enter the pairs row's ``media`` field so the case
fills like a text one, an un-fillable case (a kind the mode does not produce, or ``kind: none``) is
skipped per case with its reason recorded while the rest of the recipe still fills, a case the
reference refuses is recorded as an exception (its reason in the report and its note) without taking
the recipe's other cases down, a second run finds nothing pending, and the CLI prints the report.
"""

from __future__ import annotations

import base64
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml
from rcp_ndcg_test.cases import load_case
from rcp_ndcg_vllm.recipe import default_recipes_root, load_recipe

from tests.conftest import RECIPES, TOKENIZER

ROOT = Path(__file__).resolve().parent.parent
REAL_CASES = ROOT / "cases"

_POISON_REFERENCE = """
import argparse, json, sys

parser = argparse.ArgumentParser()
parser.add_argument("--mode", required=True)
parser.add_argument("--pairs", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--tokenizer", required=True)
parser.add_argument("--recipe", required=True)
parser.add_argument("--device", default="cpu")
args = parser.parse_args()
rows = [json.loads(line) for line in open(args.pairs, encoding="utf-8") if line.strip()]
for row in rows:
    if any("POISON" in str(document) for document in row["documents"]):
        print("the reference refuses this row: POISON is not an input this reference can score", file=sys.stderr)
        sys.exit(1)
out = [{"index": index, "scores": [0.5] * len(row["documents"])} for index, row in enumerate(rows)]
json.dump({"rows": out}, open(args.out, "w", encoding="utf-8"))
"""

_RERANK_REFERENCE = """
import argparse, json

parser = argparse.ArgumentParser()
parser.add_argument("--mode", required=True)
parser.add_argument("--pairs", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--tokenizer", required=True)
parser.add_argument("--recipe", required=True)
parser.add_argument("--device", default="cpu")
args = parser.parse_args()
rows = [json.loads(line) for line in open(args.pairs, encoding="utf-8") if line.strip()]
out = [
    {"index": index, "scores": [0.25 + 0.5 * position for position in range(len(row["documents"]))]}
    for index, row in enumerate(rows)
]
json.dump({"rows": out}, open(args.out, "w", encoding="utf-8"))
"""

_EMBED_REFERENCE = """
import argparse, json

parser = argparse.ArgumentParser()
parser.add_argument("--mode", required=True)
parser.add_argument("--pairs", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--tokenizer", required=True)
parser.add_argument("--recipe", required=True)
parser.add_argument("--device", default="cpu")
args = parser.parse_args()
rows = [json.loads(line) for line in open(args.pairs, encoding="utf-8") if line.strip()]
vectors = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, -1.0], [1.0, -1.0], [0.5, 0.5], [-1.0, 0.0], [0.0, 0.5]]
out = [
    {"index": index, "query_vectors": [[1.0, 0.0]], "document_vectors": vectors[: len(row["documents"])]}
    for index, row in enumerate(rows)
]
json.dump({"rows": out}, open(args.out, "w", encoding="utf-8"))
"""

_PENDING_NOTE = (
    "Generated for the tool's test. expected.values are filled at the GPU wave from the recipe's reference "
    "implementation and the engine, both recorded, gated at the declared abs 0.02. status pending_gpu."
)


def _recipe_copy(
    tmp_path: Path,
    recipe_id: str,
    reference: str,
    *,
    root: Path | None = None,
    drop_template: bool = False,
    variant: str | None = None,
) -> object:
    """A copy of a recipe whose ``reference.py`` is ``reference`` and whose tokenizer resolves.

    ``root`` is the recipe's family directory (the fixture recipes by default); a shipped family passes
    its own directory, since the family directory name is not the variant id (``variant`` names the
    variant of a multi-variant family).  ``drop_template`` removes the client template: the fixture embed
    recipes declare a document shape only, and the case loader requires a query shape for an embed role
    when a template is declared (the fill's rows are the reference's own pairs format, which the fake
    reads).
    """
    source = root if root is not None else RECIPES / recipe_id
    target = tmp_path / "recipes" / source.name  # the family directory name IS the family id
    shutil.copytree(source, target)
    (target / "reference.py").write_text(reference, encoding="utf-8")
    manifest = (target / "family.yaml").read_text(encoding="utf-8")
    (target / "family.yaml").write_text(
        manifest.replace("tokenizer: ../../tokenizer.json", f"tokenizer: {TOKENIZER}"), encoding="utf-8"
    )
    if drop_template:
        data = yaml.safe_load((target / "family.yaml").read_text(encoding="utf-8"))
        data["client"].pop("template", None)
        (target / "family.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return load_recipe(target, variant=variant)


def _write_case(
    directory: Path,
    recipe_id: str,
    slug: str,
    *,
    kind: str,
    length: str,
    batch: str = "single",
    queries: list[dict],
    documents: list[dict],
    note: str = _PENDING_NOTE,
    role: str | None = None,
    instruction: str | None = None,
) -> Path:
    """One pending generated case, written the way the cases lanes write them (values null, pending).

    ``kind: none`` carries no values and no tolerance by design (the case format refuses a tolerance on
    one), so its ``expected`` block holds nulls where the other kinds hold the declared rule.  A case
    whose run-level instruction is not ``None`` carries it under ``inputs`` (the case format's field).
    """
    directory.mkdir(parents=True, exist_ok=True)
    inputs: dict[str, Any] = {"queries": queries, "documents": documents}
    if instruction is not None:
        inputs["instruction"] = instruction
    case = {
        "id": f"{recipe_id}/{slug}",
        "recipe": recipe_id,
        "role": role or ("embed" if kind == "similarity_matrix" else "rerank"),
        "source": {"kind": "generated"},
        "strata": {"modality": "text", "length": length, "batch": batch},
        "inputs": inputs,
        "expected": {
            "kind": kind,
            "values": None,
            "tolerance": None if kind == "none" else {"abs": 0.02},
            "origin": "reference",
            "status": "pending_gpu",
        },
        "notes": note,
    }
    path = directory / f"{slug}.yaml"
    path.write_text(yaml.safe_dump(case, sort_keys=False), encoding="utf-8")
    return path


def _write_case_with_note(
    directory: Path,
    recipe_id: str,
    slug: str,
    note_block: str,
    *,
    kind: str = "scores",
    length: str = "short",
) -> Path:
    """One pending case whose ``notes`` scalar is the hand-written block ``note_block``.

    The fixture writer dumps a plain scalar; the shipped cases' notes are folded blocks and
    single-quoted scalars, so the block replaces the dumped one verbatim (the block includes its own
    style indicator and its leading space).
    """
    path = _write_case(
        directory,
        recipe_id,
        slug,
        kind=kind,
        length=length,
        queries=[{"id": "q1", "text": "the short query"}],
        documents=[{"id": "d1", "text": "the short document"}],
    )
    head, separator, _ = path.read_text(encoding="utf-8").partition("notes:")
    assert separator, "the fixture writer always writes a notes key"
    path.write_text(head + "notes:" + note_block, encoding="utf-8")
    return path


def _pending_copies(source: Path, destination: Path) -> None:
    """Copy a recipe's shipped cases and rewrite every generated one as pending, keeping its inputs.

    The shipped cases are the cuda fill's outputs (values written, pending notes retired); these tests
    exercise the fill itself over pending cases, the state the cases lanes' generator writes, so the
    generated cases are written back to it -- values null, ``status: pending_gpu``, the fixture's pending
    note -- while every other field (the media refs, the declared tolerances, the inputs) stays.
    """
    shutil.copytree(source, destination)
    for path in sorted(destination.glob("*.yaml")):
        case = yaml.safe_load(path.read_text(encoding="utf-8"))
        if (case.get("source") or {}).get("kind") != "generated":
            continue  # a hand-written card case carries its own values, never the fill's pending state
        case["expected"]["values"] = None
        case["expected"]["status"] = "pending_gpu"
        case["notes"] = _PENDING_NOTE
        path.write_text(yaml.safe_dump(case, sort_keys=False, allow_unicode=True, width=110), encoding="utf-8")


def _pending_cases(tmp_path: Path, recipe_id: str, *, kind: str, note: str = _PENDING_NOTE) -> Path:
    """A complete strata grid (the four required cells) of pending cases for one fixture recipe.

    Returns the cases root (the recipe's directory lives under it, as the loader expects).
    """
    directory = tmp_path / "cases" / recipe_id
    short = [{"id": "q1", "text": "the short query"}]
    one = [{"id": "d1", "text": "the short document"}]
    two = [{"id": "d1", "text": "the short document"}, {"id": "d2", "text": "another short document"}]
    _write_case(directory, recipe_id, "gen-short", kind=kind, length="short", queries=short, documents=one, note=note)
    _write_case(
        directory,
        recipe_id,
        "gen-long-under",
        kind=kind,
        length="long_under",
        queries=short,
        documents=one,
        note=note,
    )
    _write_case(
        directory,
        recipe_id,
        "gen-long-over",
        kind=kind,
        length="long_over",
        queries=short,
        documents=one,
        note=note,
    )
    _write_case(
        directory,
        recipe_id,
        "gen-mixed-lengths",
        kind=kind,
        length="mixed",
        batch="mixed_length",
        queries=[*short, {"id": "q2", "text": "the second query"}],
        documents=two,
        note=note,
    )
    return tmp_path / "cases"


def _case(cases_root: Path, recipe_id: str, slug: str) -> Path:
    return cases_root / recipe_id / f"{slug}.yaml"


def test_the_fill_writes_the_references_scores_and_retires_the_pending_note(tmp_path: Path) -> None:
    """The rerank path: one row per query, the reference's scores in document order, and the note retired."""
    from rcp_ndcg_test.case_fill import fill_cases

    recipe = _recipe_copy(tmp_path, "fixture-rerank-pointwise", _RERANK_REFERENCE)
    cases_root = _pending_cases(tmp_path, "fixture-rerank-pointwise", kind="scores")
    report = fill_cases(recipe, cases_root, reference_python="python", filled_at="2026-10-12")

    assert report.mode == "score" and report.device == "cpu"
    assert [filled.case_id for filled in report.filled] == [
        "fixture-rerank-pointwise/gen-long-over",
        "fixture-rerank-pointwise/gen-long-under",
        "fixture-rerank-pointwise/gen-mixed-lengths",
        "fixture-rerank-pointwise/gen-short",
    ]
    assert all(filled.notes_updated for filled in report.filled)

    short = load_case(_case(cases_root, "fixture-rerank-pointwise", "gen-short"))
    assert short.expected.values == [[0.25]]
    # the edit is line-wise: the values block is followed by the next key, never a blank line
    assert "\n\n  tolerance" not in _case(cases_root, "fixture-rerank-pointwise", "gen-short").read_text()
    assert short.expected.status == "reproduced" and short.expected.origin == "reference"
    assert "status reproduced (values filled from the reference implementation on cpu, 2026-10-12)" in short.notes
    assert "pending_gpu" not in short.notes

    mixed = load_case(_case(cases_root, "fixture-rerank-pointwise", "gen-mixed-lengths"))
    assert mixed.expected.values == [[0.25, 0.75], [0.25, 0.75]]
    # the pending sentence is retired where it stood, the rest of the note survives
    assert "expected.values are the outputs of the reference implementation itself" in mixed.notes
    assert "Generated for the tool's test." in mixed.notes

    # a second run finds nothing pending and writes nothing
    again = fill_cases(recipe, cases_root, reference_python="python", filled_at="2026-10-12")
    assert again.filled == () and again.pending_before == 0


def test_the_fill_writes_the_similarity_matrix_from_the_references_vectors(tmp_path: Path) -> None:
    """The embed path: the matrix is the cosine of the reference's own query and document vectors, and
    the zembed family's pending note sentence is retired."""
    from rcp_ndcg_test.case_fill import fill_cases

    recipe = _recipe_copy(tmp_path, "fixture-embed", _EMBED_REFERENCE, drop_template=True)
    cases_root = _pending_cases(
        tmp_path,
        "fixture-embed",
        kind="similarity_matrix",
        note="Values are filled by the GPU wave (reference + engine recorded).",
    )
    report = fill_cases(recipe, cases_root, reference_python="python", filled_at="2026-10-12")
    assert report.mode == "embed" and len(report.filled) == 4

    short = load_case(_case(cases_root, "fixture-embed", "gen-short"))
    assert short.expected.values == [[1.0]]
    assert "Values are the outputs of the reference implementation itself" in short.notes
    assert "GPU wave (reference + engine recorded)" not in short.notes
    mixed = load_case(_case(cases_root, "fixture-embed", "gen-mixed-lengths"))
    assert mixed.expected.values == [[1.0, 0.0], [1.0, 0.0]]


def test_a_media_case_plans_the_case_files_into_the_pairs_media_field() -> None:
    """The media translation: a media document's file is read from the case's own directory and enters
    the pairs row as the media set's own entry shape (the ``data:`` URI plus its kind), the document's
    text is empty (the reference reads the media as the side's content), the query side stays empty
    (the case format's queries are text-only), and a text-only case's row carries no ``media`` at all."""
    from rcp_ndcg_test.case_fill import _plan_rows

    media_dir = REAL_CASES / "qwen3-vl-embedding-2b" / "media"
    cases = [
        load_case(REAL_CASES / "qwen3-vl-embedding-2b" / "image-short-single.yaml"),
        load_case(REAL_CASES / "qwen3-vl-embedding-2b" / "video-short-single.yaml"),
        load_case(REAL_CASES / "qwen3-vl-embedding-2b" / "mixed-modality-batch.yaml"),
    ]
    rows, mapping = _plan_rows(load_recipe("qwen3-vl-embedding-2b"), cases)
    assert [case.id for case, _ in mapping] == [
        "qwen3-vl-embedding-2b/image-short-single",
        "qwen3-vl-embedding-2b/video-short-single",
        "qwen3-vl-embedding-2b/mixed-modality-batch",
        "qwen3-vl-embedding-2b/mixed-modality-batch",
    ]

    stand_in = (media_dir / "stand_in.png").read_bytes()
    assert rows[0]["documents"] == [""]
    assert rows[0]["media"] == {
        "query": [],
        "documents": [
            [
                {
                    "kind": "image",
                    "uri": "data:image/png;base64," + base64.b64encode(stand_in).decode("ascii"),
                    "sha256": hashlib.sha256(stand_in).hexdigest(),
                    "mime": "image/png",
                    "width": 934,
                    "height": 612,
                    "num_bytes": len(stand_in),
                }
            ]
        ],
    }

    clip = (media_dir / "clip.mp4").read_bytes()
    video = rows[1]["media"]["documents"][0][0]
    assert video["kind"] == "video" and video["mime"] == "video/mp4"
    assert video["uri"] == "data:video/mp4;base64," + base64.b64encode(clip).decode("ascii")
    assert video["sha256"] == hashlib.sha256(clip).hexdigest() and video["num_bytes"] == len(clip)
    # the container's own facts, off its header: 448x256, 4 frames at 2 fps (2 s)
    assert (video["width"], video["height"], video["num_frames"]) == (448, 256, 4)
    assert (video["fps"], video["duration_s"]) == (2.0, 2.0)

    # a batch mixing text-only and media documents: every document keeps its own side, in order
    page = (media_dir / "page.png").read_bytes()
    assert rows[2]["documents"] == [
        "A woman shares a happy moment with her golden retriever on a sunny beach.",
        "",
        "A woman shares a happy moment with her golden retriever on a sunny beach.",
    ]
    documents = rows[2]["media"]["documents"]
    assert documents[0] == []
    assert documents[1][0]["sha256"] == hashlib.sha256(stand_in).hexdigest()
    assert (documents[2][0]["width"], documents[2][0]["height"]) == (1700, 2200)
    assert documents[2][0]["sha256"] == hashlib.sha256(page).hexdigest()

    text_only, _ = _plan_rows(
        load_recipe("qwen3-vl-embedding-2b"),
        [load_case(REAL_CASES / "qwen3-vl-embedding-2b" / "text-short-single.yaml")],
    )
    assert "media" not in text_only[0]


def test_a_case_instruction_is_planned_only_when_the_recipe_places_it(tmp_path: Path) -> None:
    """The recipe's resolved instruction mode decides the pairs row, exactly as it decides the served
    request (the client rule): a case carrying an instruction against a recipe whose client declares
    ``instruction: none`` -- the card's default instruction is fixed frame text -- plans a row WITHOUT
    the field, so the reference is never handed an instruction the served engine never sees (the wave's
    nine qwen3-vl-embedding-2b exceptions); a recipe that folds the instruction carries it, and the
    reference folds it the same way the client does."""
    from rcp_ndcg_test.case_fill import _plan_rows

    card_case = load_case(REAL_CASES / "qwen3-vl-embedding-2b" / "text-short-single.yaml")
    assert card_case.inputs.instruction  # the shipped case carries the card's own default instruction
    rows, _ = _plan_rows(load_recipe("qwen3-vl-embedding-2b"), [card_case])
    assert "instruction" not in rows[0], "the served request carries none: the row must not either"

    fold_recipe = load_recipe(RECIPES / "fixture-rerank-pointwise")
    instruction = "Given a web search query, retrieve relevant passages that answer the query"
    path = _write_case(
        tmp_path / "cases" / "fixture-rerank-pointwise",
        "fixture-rerank-pointwise",
        "gen-instruction",
        kind="scores",
        length="short",
        queries=[{"id": "q1", "text": "the short query"}],
        documents=[{"id": "d1", "text": "the short document"}],
        instruction=instruction,
    )
    folded, _ = _plan_rows(fold_recipe, [load_case(path)])
    assert folded[0]["instruction"] == instruction


def test_a_media_case_is_no_longer_skipped_and_a_kind_none_case_still_is(tmp_path: Path) -> None:
    """The qwen3-vl-reranker-2b cases carry images: the fill now sends them in the pairs row's ``media``
    field, so every media case fills like the text ones (no skip); ``kind: none`` still carries no values
    by design, is skipped with its reason recorded, and its file is left byte-identical."""
    from rcp_ndcg_test.case_fill import fill_cases

    recipe = _recipe_copy(
        tmp_path,
        "qwen3-vl-reranker-2b",
        _RERANK_REFERENCE,
        root=default_recipes_root() / "qwen3-vl-reranker",
        variant="qwen3-vl-reranker-2b",
    )
    cases_root = tmp_path / "cases"
    _pending_copies(REAL_CASES / "qwen3-vl-reranker-2b", cases_root / "qwen3-vl-reranker-2b")
    none_case = _write_case(
        cases_root / "qwen3-vl-reranker-2b",
        "qwen3-vl-reranker-2b",
        "gen-kind-none",
        kind="none",
        length="short",
        queries=[{"id": "q1", "text": "the short query"}],
        documents=[{"id": "d1", "text": "the short document"}],
    )
    before = none_case.read_bytes()

    report = fill_cases(recipe, cases_root, reference_python="python", filled_at="2026-10-12")
    assert report.pending_before == 10
    assert [skipped.case_id.split("/", 1)[1] for skipped in report.skipped] == ["gen-kind-none"]
    assert "kind: none" in report.skipped[0].reason
    assert [filled.case_id.split("/", 1)[1] for filled in report.filled] == [
        "gen-image-long-over",
        "gen-image-short-single",
        "gen-image-uniform",
        "gen-mixed-modality",
        "gen-text-long-over",
        "gen-text-long-under",
        "gen-text-mixed-length",
        "gen-text-short-single",
        "gen-text-short-uniform",
    ]
    assert all(filled.notes_updated for filled in report.filled)
    assert none_case.read_bytes() == before

    filled = load_case(cases_root / "qwen3-vl-reranker-2b" / "gen-image-short-single.yaml")
    assert filled.expected.status == "reproduced" and filled.expected.values == [[0.25]]
    # the case's own tolerance is carried through unchanged: the recipe's measured stage-2 bound
    # (g-cases-3) is 0.035, not the previous 0.01.
    assert filled.expected.tolerance is not None and filled.expected.tolerance.abs == 0.035
    # the note's own pending sentence is retired where it stood (the case's "GPU wave fills values;")
    assert "the outputs of the reference implementation itself for these inputs" in filled.notes
    assert "(filled on cpu before the wave" in filled.notes
    assert "GPU wave fills values" not in filled.notes


def test_an_embed_media_case_fills_and_carries_its_own_tolerance(tmp_path: Path) -> None:
    """The embed path over media: the qwen3-vl-embedding-2b cases (image, video and mixed documents)
    fill like the text ones -- the reference's ``embed`` mode reads the pairs row's ``media`` field --
    and each case's own tolerance (the media cases declare 0.002) is carried through unchanged."""
    from rcp_ndcg_test.case_fill import fill_cases

    recipe = _recipe_copy(
        tmp_path,
        "qwen3-vl-embedding-2b",
        _EMBED_REFERENCE,
        root=default_recipes_root() / "qwen3-vl-embedding",
        variant="qwen3-vl-embedding-2b",
    )
    cases_root = tmp_path / "cases"
    _pending_copies(REAL_CASES / "qwen3-vl-embedding-2b", cases_root / "qwen3-vl-embedding-2b")

    report = fill_cases(recipe, cases_root, reference_python="python", filled_at="2026-10-12")
    assert report.mode == "embed" and report.pending_before == 9
    assert report.skipped == () and report.exceptions == ()
    assert len(report.filled) == 9
    image = load_case(cases_root / "qwen3-vl-embedding-2b" / "image-short-single.yaml")
    assert image.expected.status == "reproduced" and image.expected.values == [[1.0]]
    assert image.expected.tolerance is not None and image.expected.tolerance.abs == 0.002
    video = load_case(cases_root / "qwen3-vl-embedding-2b" / "video-short-single.yaml")
    assert video.expected.status == "reproduced" and video.expected.tolerance is not None
    assert video.expected.tolerance.abs == 0.002


def test_one_cases_refusal_is_recorded_and_leaves_the_other_cases_filling(tmp_path: Path) -> None:
    """The isolation: the reference runs once over every case's rows, and when that run fails every
    case is retried on its own -- so a row the reference refuses (here a poisoned document) is recorded
    as an exception (its reason in the report and in its own note, its values left null, status
    ``pending_gpu``) while the recipe's other cases still fill.  A second run records the same exception
    without appending the reason a second time."""
    from rcp_ndcg_test.case_fill import fill_cases

    recipe = _recipe_copy(tmp_path, "fixture-rerank-pointwise", _POISON_REFERENCE)
    cases_root = _pending_cases(tmp_path, "fixture-rerank-pointwise", kind="scores")
    poisoned = _case(cases_root, "fixture-rerank-pointwise", "gen-long-over")
    poisoned.write_text(
        poisoned.read_text(encoding="utf-8").replace("the short document", "POISONED document text"),
        encoding="utf-8",
    )

    report = fill_cases(recipe, cases_root, reference_python="python", filled_at="2026-10-12")
    assert [filled.case_id.split("/", 1)[1] for filled in report.filled] == [
        "gen-long-under",
        "gen-mixed-lengths",
        "gen-short",
    ]
    assert [exception.case_id.split("/", 1)[1] for exception in report.exceptions] == ["gen-long-over"]
    assert report.exceptions[0].notes_updated
    assert "refuses this row" in report.exceptions[0].reason
    refused = load_case(poisoned)
    assert refused.expected.values is None and refused.expected.status == "pending_gpu"
    assert "could not produce this case's values" in refused.notes
    assert "refuses this row" in refused.notes
    assert refused.notes.count("refuses this row") == 1

    again = fill_cases(recipe, cases_root, reference_python="python", filled_at="2026-10-12")
    assert again.filled == () and again.pending_before == 1
    assert [exception.case_id.split("/", 1)[1] for exception in again.exceptions] == ["gen-long-over"]
    assert again.exceptions[0].notes_updated is False
    assert load_case(poisoned).notes.count("refuses this row") == 1


def test_a_filled_case_records_the_fill_when_its_note_has_no_pending_sentence(tmp_path: Path) -> None:
    """Tonight's gap: the shipped cases' notes carry no sentence the retirement vocabulary knows, so a
    filled case's note was left untouched (``notes_updated: false``).  The fill now appends its own
    provenance sentence to the note -- in whichever scalar style the note is written (a folded block, a
    single-quoted scalar, a plain one) -- so every filled case's note records the fill, and the note's
    own text survives."""
    from rcp_ndcg_test.case_fill import fill_cases

    recipe = _recipe_copy(tmp_path, "fixture-rerank-pointwise", _RERANK_REFERENCE)
    directory = tmp_path / "cases" / "fixture-rerank-pointwise"
    _pending_cases(tmp_path, "fixture-rerank-pointwise", kind="scores")
    note = "A note with no pending sentence at all: the notes of the shipped cases look like this."
    _write_case_with_note(directory, "fixture-rerank-pointwise", "gen-short", f" >-\n  {note}\n")
    _write_case_with_note(directory, "fixture-rerank-pointwise", "gen-long-under", f" '{note}'\n", length="long_under")
    _write_case_with_note(
        directory,
        "fixture-rerank-pointwise",
        "gen-long-over",
        " 'A note with no pending sentence at all: the notes of the shipped\n  cases look like this.'\n",
        length="long_over",
    )

    report = fill_cases(recipe, tmp_path / "cases", reference_python="python", filled_at="2026-10-12")
    assert all(filled.notes_updated for filled in report.filled)
    sentence = "status reproduced (values filled from the reference implementation on cpu, 2026-10-12)."
    for slug in ("gen-short", "gen-long-under", "gen-long-over"):
        filled = load_case(directory / f"{slug}.yaml")
        assert filled.notes.endswith(sentence), filled.notes
        assert "A note with no pending sentence at all: the notes of the shipped cases look like this." in filled.notes


def test_a_note_with_a_known_pending_sentence_is_retired_not_appended(tmp_path: Path) -> None:
    """A note the retirement vocabulary knows is rewritten where the pending sentence stands -- the
    provenance sentence is not appended a second time (the retired sentence already records the fill)."""
    from rcp_ndcg_test.case_fill import fill_cases

    recipe = _recipe_copy(tmp_path, "fixture-rerank-pointwise", _RERANK_REFERENCE)
    cases_root = _pending_cases(tmp_path, "fixture-rerank-pointwise", kind="scores")
    report = fill_cases(recipe, cases_root, reference_python="python", filled_at="2026-10-12")
    assert all(filled.notes_updated for filled in report.filled)
    short = load_case(_case(cases_root, "fixture-rerank-pointwise", "gen-short"))
    assert short.notes.count("status reproduced (values filled from the reference implementation") == 1
    assert "Generated for the tool's test." in short.notes


def test_a_kind_the_mode_does_not_produce_is_skipped_not_fatal(tmp_path: Path) -> None:
    """A pending case whose ``expected.kind`` is another stage's shape (an embed case on a rerank
    recipe) is skipped with its reason recorded; the recipe's other cases still fill."""
    from rcp_ndcg_test.case_fill import fill_cases

    recipe = _recipe_copy(tmp_path, "fixture-rerank-pointwise", _RERANK_REFERENCE)
    cases_root = _pending_cases(tmp_path, "fixture-rerank-pointwise", kind="scores")
    _write_case(
        (cases_root / "fixture-rerank-pointwise"),
        "fixture-rerank-pointwise",
        "gen-similarity",
        kind="similarity_matrix",
        length="short",
        queries=[{"id": "q1", "text": "the short query"}],
        documents=[{"id": "d1", "text": "the short document"}],
        role="rerank",
    )

    report = fill_cases(recipe, cases_root, reference_python="python", filled_at="2026-10-12")
    assert [skipped.case_id for skipped in report.skipped] == ["fixture-rerank-pointwise/gen-similarity"]
    assert "similarity_matrix" in report.skipped[0].reason and "score" in report.skipped[0].reason
    assert len(report.filled) == 4


def test_the_cli_prints_the_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The CLI is the runner: ``--recipe --reference-python`` fills and prints the JSON report."""
    from rcp_ndcg_test.case_fill import main

    recipe = _recipe_copy(tmp_path, "fixture-rerank-pointwise", _RERANK_REFERENCE)
    cases_root = _pending_cases(tmp_path, "fixture-rerank-pointwise", kind="scores")
    code = main(
        [
            "--recipe",
            str(recipe._dir),
            "--cases-root",
            str(cases_root),
            "--reference-python",
            "python",
        ]
    )
    assert code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["recipe"] == "fixture-rerank-pointwise"
    assert report["mode"] == "score"
    assert len(report["filled"]) == 4
    assert report["skipped"] == []
    assert report["exceptions"] == []
    assert load_case(_case(cases_root, "fixture-rerank-pointwise", "gen-short")).expected.values == [[0.25]]


def test_the_retirement_vocabulary_covers_every_shipped_generated_note() -> None:
    """The shipped generated cases' pending notes carry their own phrasings, not the three the fill
    started with (measured: none of the 41 matched, so every fill left the note contradicting the
    values).  Every one of the four cuda recipes' generated cases is run through the same note update
    the fill applies: no note keeps a pending claim (``pending_gpu``, ``stays null``, a ``GPU wave``
    fill) and every note ends up naming the reference implementation as the values' source."""
    import re

    from rcp_ndcg_test.case_fill import _update_note

    sentence = "values filled from the reference implementation on cuda, 2026-10-13"
    generated = 0
    for recipe_id in (
        "topk-embed-v1-small",
        "qwen3-embedding-0.6b",
        "qwen3-vl-embedding-2b",
        "qwen3-vl-reranker-2b",
    ):
        for path in sorted((REAL_CASES / recipe_id).glob("*.yaml")):
            text = path.read_text(encoding="utf-8")
            if "source:\n  kind: generated" not in text:
                continue
            generated += 1
            updated, notes_updated = _update_note(text, device="cuda", filled_at="2026-10-13", path=path)
            assert notes_updated
            notes = yaml.safe_load(updated)["notes"]
            for stale in ("pending_gpu", "stays null", "GPU wave", "wave fills"):
                assert stale not in notes, (path.name, stale, notes)
            assert sentence in notes or "the outputs of the reference implementation itself" in notes, (
                path.name,
                notes,
            )
            assert not re.search(r"\bnull until\b", notes), (path.name, notes)
    assert generated == 41


def test_a_late_interaction_reference_scores_through_the_products_maxsim() -> None:
    """A ragged (multi-vector) reference's answers score through the product's own MaxSim scorer, the
    one the conformance runner uses -- not a cosine of concatenated vectors.  The case is a real
    topk-embed-v1-small one (two queries, two documents), so the matrix's document mapping is checked."""
    from rcp_ndcg_test.case_fill import expected_values

    case = load_case(REAL_CASES / "topk-embed-v1-small" / "text-skip-mask-punctuation.yaml")
    assert len(case.inputs.queries) == 2 and len(case.inputs.documents) == 2
    answer = {
        "index": 0,
        "query_vectors": [[[1.0, 0.0], [0.0, 1.0]]],
        "document_vectors": [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 0.0]]],
    }
    assert expected_values(case, [answer, dict(answer, index=1)], "embed") == [[2.0, 1.0], [2.0, 1.0]]
