"""The request generator: determinism, the harness pairs format, the strata record and clean output.

Everything here runs offline on the fixture recipes and fixture tokenizer: the source corpora are built
inline (generation itself reads the Hub at pinned commits, in the operator's one-time run).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from rcp_ndcg_vllm.equivalence.fitting import load_pairs, tokenizer_of
from rcp_ndcg_vllm.observe.adversarial import CONTENT_KINDS
from rcp_ndcg_vllm.observe.requests import (
    GENERATOR_VERSION,
    PINNED_DATASET_COMMITS,
    PlannedRow,
    RecipePlan,
    pairs_jsonl,
    plan_recipe,
    write_manifest,
    write_pairs_file,
)
from rcp_ndcg_vllm.observe.sources import SourceCorpus, SourceDoc, SourceMedia, SourceQuery
from rcp_ndcg_vllm.recipe import load_recipe

from tests.conftest import RECIPES

_CREDENTIAL = re.compile(
    r"(hf_[A-Za-z0-9]{20,}|AIza[0-9A-Za-z_-]{20,}|ya29\.[0-9A-Za-z_-]{10,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)",
)


def _plan(tmp_path: Path | None = None) -> tuple[object, RecipePlan]:
    """The fixture-embed recipe's plan over an inline source catalog (offline)."""
    recipe = load_recipe(RECIPES / "fixture-embed")
    tokenizer = tokenizer_of(recipe)
    corpus = SourceCorpus(
        suite="nanobeir",
        subset="NanoNQRetrieval",
        commit="0" * 40,
        queries={
            "q1": SourceQuery("q1", "what is the capital of france", None, ("d1", "d2")),
            "q2": SourceQuery("q2", "rivers in europe", None, ("d1",)),
        },
        docs={"d1": SourceDoc("d1", "paris is the capital of france"), "d2": SourceDoc("d2", "the seine")},
    )
    vidore = SourceCorpus(
        suite="vidore",
        subset="hr__english",
        commit="1" * 40,
        queries={"v1": SourceQuery("v1", "how many pages", None, ("p1",))},
        docs={
            "p1": SourceDoc(
                "p1",
                "the page text",
                media=(SourceMedia("vidore", "hr__english", "p1", 0, "ab" * 32, "image/png", 1234),),
            )
        },
    )
    plan = plan_recipe(recipe, tokenizer, {"nanobeir": [corpus], "vidore": [vidore]})
    return recipe, plan


def test_content_kinds_are_the_specs_eleven() -> None:
    """OBSERVATIONS-SPEC section 1's content kinds, exactly."""
    assert CONTENT_KINDS == (
        "ascii_prose",
        "code",
        "cjk",
        "rtl",
        "emoji_zwj",
        "combining_marks",
        "whitespace_only",
        "empty",
        "long_token_url",
        "long_token_base64",
        "special_token_spellings",
    )


def test_generation_is_deterministic_in_its_declared_inputs() -> None:
    """The same (version, seed, pinned commits, recipe) plan the same rows, byte for byte."""
    _, first = _plan()
    _, second = _plan()
    assert pairs_jsonl(first) == pairs_jsonl(second)
    assert GENERATOR_VERSION >= 1
    assert set(PINNED_DATASET_COMMITS) == {
        "fabianschmidt-cohere/rcp-ndcg-nanobeir",
        "fabianschmidt-cohere/rcp-ndcg-bright",
        "fabianschmidt-cohere/rcp-ndcg-vidore-v3",
        "fabianschmidt-cohere/rcp-ndcg-trecdl",
    }
    for commit in PINNED_DATASET_COMMITS.values():
        assert commit.isalnum() and len(commit) == 40


def test_pairs_rows_are_the_harness_pairs_format(tmp_path: Path) -> None:
    """The written file is exactly what ``load_pairs`` reads: query str, documents list of strings."""
    _, plan = _plan()
    path = write_pairs_file(plan, tmp_path)
    rows = load_pairs(path)
    assert len(rows) == len(plan.rows)
    for row in rows:
        assert isinstance(row["query"], str)
        assert row["documents"] and all(isinstance(document, str) for document in row["documents"])


def test_every_stratum_is_present_or_absent_with_a_reason() -> None:
    """OBSERVATIONS-SPEC section 1: each stratum recorded present or absent (absent only with a why)."""
    _, plan = _plan()
    for name, record in plan.strata.items():
        assert isinstance(record.get("present"), bool), name
        if not record["present"]:
            assert record.get("reason"), f"{name}: absent without a reason"
    # The fixture budget (128 tokens) is too small for the long-single-token kinds: they are recorded
    # absent with the reason (the corpus request set sends them uncut).  Every small kind is present.
    for kind in ("whitespace_only", "combining_marks", "emoji_zwj"):
        assert plan.strata[f"content:{kind}"]["present"] is True, kind
    # The fixture's empty policy (``empty_doc: omit_zero``) rewrites an empty document: the empty kind
    # is recorded absent with the policy's reason, never planned into a row the client would rewrite.
    empty = plan.strata["content:empty"]
    assert empty["present"] is False and "empty policy" in empty["reason"], empty


def test_media_rows_for_a_media_recipe_carry_the_page_refs() -> None:
    """A media recipe's ViDoRe rows carry ``media`` entries by source coordinates (suite/subset/doc/part)."""
    recipe, _ = _plan()
    media_recipe = recipe.model_copy(update={"input": ["text", "image"]})
    tokenizer = tokenizer_of(recipe)
    corpus = SourceCorpus(
        suite="vidore",
        subset="hr__english",
        commit="1" * 40,
        queries={"v1": SourceQuery("v1", "how many pages", None, ("p1",))},
        docs={
            "p1": SourceDoc(
                "p1",
                "the page text",
                media=(SourceMedia("vidore", "hr__english", "p1", 0, "ab" * 32, "image/png", 1234),),
            )
        },
    )
    plan = plan_recipe(media_recipe, tokenizer, {"vidore": [corpus]})
    media_rows = [row for row in plan.rows if row.media]
    assert media_rows, "no media rows were planned for the media recipe"
    entry = media_rows[0].to_pairs_row()["media"]["documents"][0][0]
    assert entry["suite"] == "vidore" and entry["subset"] == "hr__english" and entry["doc_id"] == "p1"
    assert entry["sha256"] == "ab" * 32 and entry["mime"] == "image/png" and entry["num_bytes"] == 1234
    # No machine-local path is ever recorded beside the source coordinates and the content hash.
    assert "uri" not in entry
    assert plan.strata["media:page_image"]["present"] is True


def test_text_only_recipes_record_the_media_stratum_absent_with_the_reason() -> None:
    _, plan = _plan()
    record = plan.strata["media:page_image"]
    assert record["present"] is False and "text-only" in record["reason"]


def test_length_rows_fit_the_declared_budget_uncut() -> None:
    """Every planned row fits ``client.max_tokens`` without a cut (the product's fit is the judge)."""
    from rcp_ndcg_vllm.equivalence.fitting import resolved_tokenizer_spec

    from rcp_ndcg.data.preprocess import TextBudget, TextTruncationCensus, fit

    recipe, plan = _plan()
    tokenizer = tokenizer_of(recipe)
    client = recipe.client.model_dump()
    budget = TextBudget(
        tokenizer=resolved_tokenizer_spec(recipe),
        max_tokens=client["max_tokens"],
        query_max_tokens=client.get("query_max_tokens"),
        template=client.get("template"),
        on_overflow=client.get("on_overflow", "cut"),
    )
    census = TextTruncationCensus()
    for row in plan.rows:
        fit([row.documents[0]], "document", budget, tokenizer, ids=["x"], census=census)
    assert not census.cuts(), f"the generator planned rows the product's fit cuts: {census.cuts()}"


def test_no_credential_shaped_string_can_reach_a_pairs_file() -> None:
    """A scan of every generated byte for credential shapes (tokens, API keys, private key blocks)."""
    _, plan = _plan()
    blob = pairs_jsonl(plan)
    assert _CREDENTIAL.search(blob) is None


def test_manifest_records_hashes_provenance_and_pruned_rows(tmp_path: Path) -> None:
    _, plan = _plan()
    write_pairs_file(plan, tmp_path)
    path = write_manifest([plan], tmp_path, pruned=[{"recipe": plan.recipe_id, "reason": "stage 1 red"}])
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["schema_version"] == 1
    assert document["generator"]["GENERATOR_VERSION"] == GENERATOR_VERSION
    entry = document["files"][0]
    assert entry["recipe"] == plan.recipe_id and entry["rows"] == len(plan.rows)
    assert entry["sha256"] and len(entry["sha256"]) == 64
    assert entry["provenance"], "every row's source identity must be recorded (the selection manifest)"
    assert document["pruned"][0]["reason"] == "stage 1 red"


def test_rows_carry_only_documented_keys() -> None:
    """The pairs row keys are exactly the documented ones (the reader contract)."""
    _, plan = _plan()
    allowed = {"query", "documents", "shape", "instruction", "media", "_strata", "_source"}
    for row in plan.rows:
        assert set(row.to_pairs_row()) <= allowed


def test_planned_row_without_media_omits_the_key() -> None:
    row = PlannedRow(query="q", documents=("d",), strata=("length:tiny",), source={"suite": "synthetic"})
    assert "media" not in row.to_pairs_row()
