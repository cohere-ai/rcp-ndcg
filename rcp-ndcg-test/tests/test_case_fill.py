"""The case fill: the reference implementation's own outputs into the pending cases.

The tool is :mod:`rcp_ndcg_test.case_fill`.  These tests copy a fixture recipe whose ``reference.py``
is a fake subprocess and exercise the tool end to end over case YAMLs in ``tmp_path``: the values
written are the reference's outputs, ``status: pending_gpu`` becomes ``reproduced``, the pending note's
own sentence is retired, a media-bearing case is refused by name **before** the reference runs, a
second run finds nothing pending, and the CLI prints the report.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml
from rcp_ndcg_test.cases import load_case
from rcp_ndcg_test.errors import CaseError
from rcp_ndcg_vllm.recipe import load_recipe

from tests.conftest import RECIPES, TOKENIZER

ROOT = Path(__file__).resolve().parent.parent
REAL_CASES = ROOT / "cases"

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
vectors = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, -1.0]]
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


def _recipe_copy(tmp_path: Path, recipe_id: str, reference: str, *, root: Path | None = None) -> object:
    """A copy of a recipe whose ``reference.py`` is ``reference`` and whose tokenizer resolves.

    ``root`` is the recipe's family directory (the fixture recipes by default); a shipped family passes
    its own directory, since the family directory name is not the variant id.
    """
    source = root if root is not None else RECIPES / recipe_id
    target = tmp_path / "recipes" / source.name  # the family directory name IS the family id
    shutil.copytree(source, target)
    (target / "reference.py").write_text(reference, encoding="utf-8")
    manifest = (target / "family.yaml").read_text(encoding="utf-8")
    (target / "family.yaml").write_text(
        manifest.replace("tokenizer: ../../tokenizer.json", f"tokenizer: {TOKENIZER}"), encoding="utf-8"
    )
    return load_recipe(target)


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
) -> Path:
    """One pending generated case, written the way the cases lanes write them (values null, pending)."""
    directory.mkdir(parents=True, exist_ok=True)
    case = {
        "id": f"{recipe_id}/{slug}",
        "recipe": recipe_id,
        "role": "embed" if kind == "similarity_matrix" else "rerank",
        "source": {"kind": "generated"},
        "strata": {"modality": "text", "length": length, "batch": batch},
        "inputs": {"queries": queries, "documents": documents},
        "expected": {
            "kind": kind,
            "values": None,
            "tolerance": {"abs": 0.02},
            "origin": "reference",
            "status": "pending_gpu",
        },
        "notes": note,
    }
    path = directory / f"{slug}.yaml"
    path.write_text(yaml.safe_dump(case, sort_keys=False), encoding="utf-8")
    return path


def _pending_cases(tmp_path: Path, recipe_id: str, *, kind: str) -> Path:
    """A complete strata grid (the four required cells) of pending cases for one fixture recipe.

    Returns the cases root (the recipe's directory lives under it, as the loader expects).
    """
    directory = tmp_path / "cases" / recipe_id
    short = [{"id": "q1", "text": "the short query"}]
    one = [{"id": "d1", "text": "the short document"}]
    two = [{"id": "d1", "text": "the short document"}, {"id": "d2", "text": "another short document"}]
    _write_case(directory, recipe_id, "gen-short", kind=kind, length="short", queries=short, documents=one)
    _write_case(directory, recipe_id, "gen-long-under", kind=kind, length="long_under", queries=short, documents=one)
    _write_case(directory, recipe_id, "gen-long-over", kind=kind, length="long_over", queries=short, documents=one)
    _write_case(
        directory,
        recipe_id,
        "gen-mixed-lengths",
        kind=kind,
        length="mixed",
        batch="mixed_length",
        queries=[*short, {"id": "q2", "text": "the second query"}],
        documents=two,
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
    """The embed path, over the real zembed-1-embedding cases: the matrix is the cosine of the
    reference's own query and document vectors, and the zembed pending sentence is retired."""
    from rcp_ndcg_test.case_fill import fill_cases
    from rcp_ndcg_vllm.recipe import default_recipes_root

    recipe = _recipe_copy(tmp_path, "zembed-1-embedding", _EMBED_REFERENCE, root=default_recipes_root() / "zembed-1")
    cases_root = tmp_path / "cases"
    shutil.copytree(REAL_CASES / "zembed-1-embedding", cases_root / "zembed-1-embedding")
    report = fill_cases(recipe, cases_root, reference_python="python", filled_at="2026-10-12")
    assert report.mode == "embed" and len(report.filled) == 7  # the card case is published, not pending

    short = load_case(_case(cases_root, "zembed-1-embedding", "short-single"))
    assert short.expected.values == [[1.0]]
    assert "Values are the outputs of the reference implementation itself" in short.notes
    assert "GPU wave (reference + engine recorded)" not in short.notes
    mixed = load_case(_case(cases_root, "zembed-1-embedding", "short-uniform"))
    assert mixed.expected.values == [[1.0, 0.0, 0.7071067690849304, 0.0], [1.0, 0.0, 0.7071067690849304, 0.0]]


def test_a_media_case_is_refused_by_name_before_anything_is_written(tmp_path: Path) -> None:
    """The qwen3-vl-reranker-2b cases carry images: the reference's score mode takes text, so the fill
    refuses the first media case by name and leaves every case file byte-identical."""
    from rcp_ndcg_test.case_fill import fill_cases

    recipe = load_recipe("qwen3-vl-reranker-2b")
    cases_root = tmp_path / "cases"
    shutil.copytree(REAL_CASES / "qwen3-vl-reranker-2b", cases_root / "qwen3-vl-reranker-2b")
    before = {path.name: path.read_bytes() for path in (cases_root / "qwen3-vl-reranker-2b").glob("*.yaml")}

    with pytest.raises(CaseError) as error:
        fill_cases(recipe, cases_root, reference_python="python")
    assert "gen-image-long-over" in str(error.value)
    assert "media" in str(error.value)
    after = {path.name: path.read_bytes() for path in (cases_root / "qwen3-vl-reranker-2b").glob("*.yaml")}
    assert after == before


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
    assert load_case(_case(cases_root, "fixture-rerank-pointwise", "gen-short")).expected.values == [[0.25]]


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
