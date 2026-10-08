"""The request generator: determinism, the harness pairs format, the strata record and clean output.

Everything here runs offline on the fixture recipes and fixture tokenizer: the source corpora are built
inline (generation itself reads the Hub at pinned commits, in the operator's one-time run).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
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
    client = dict(recipe.client)
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


# --- the corpus plan: what a recording sends beyond the pairs rows (OBSERVATIONS-SPEC section 1) ----------------


def _corpus_plan(recipe_id: str):
    from rcp_ndcg_vllm.observe.requests import corpus_plan

    recipe = load_recipe(RECIPES / recipe_id)
    tokenizer = tokenizer_of(recipe)
    _, plan = _plan()
    rows = [row.to_pairs_row() for row in plan.rows]
    return recipe, tokenizer, corpus_plan(recipe, tokenizer, rows)


def test_the_corpus_plan_sends_the_over_length_ladder_through_the_client_and_uncut() -> None:
    """One token over, 2x and 10x the budget in the recipe's own tokens -- each through the client (its cut)
    and bare (uncut: the engine's own refusal)."""
    _, tokenizer, plan = _corpus_plan("fixture-embed")
    rows = {row["request_id"]: row for row in plan.rows}
    bare = {row["request_id"]: row for row in plan.bare}
    targets = {name: plan.strata[f"length:{name}"]["content_tokens"] for name in ("over_by_1", "over_2x", "over_10x")}
    assert targets["over_2x"] == 2 * (targets["over_by_1"] - 1) and targets["over_10x"] == 10 * (
        targets["over_by_1"] - 1
    )
    for name, target in targets.items():
        text = rows[f"ladder:{name}"]["documents"][0]
        assert abs(tokenizer.count(text) - target) <= 2, name
        assert bare[f"ladder:{name}:uncut"]["body"]["input"] == [text], "the uncut request carries the full text"
    # The rung one token over the budget is over the client's budget once the template's frame is added.
    assert targets["over_by_1"] > 0


def test_the_content_kinds_too_long_for_a_pairs_row_are_sent_both_ways() -> None:
    """A kind the pairs file had to leave out (it exceeds the budget) is not lost: the corpus sends it."""
    _, _, plan = _corpus_plan("fixture-embed")
    ids = set(plan.request_ids())
    assert "content:long_token_base64" in ids and "content:long_token_base64:uncut" in ids
    assert plan.strata["content:long_token_base64:uncut"]["present"] is True
    assert plan.strata["content:code:uncut"]["present"] is False  # its pairs row carries it


def test_the_wire_variants_and_the_protocol_edges_cover_each_route() -> None:
    """Every encoding of the route and each adapter-sendable field on and off; the edges the set can send."""
    _, _, embed = _corpus_plan("fixture-embed")
    assert {"wire:encoding_format=base64", "wire:dimensions=32"} <= set(embed.strata)
    _, _, pooling = _corpus_plan("fixture-multi-vector")
    variants = {row["request_id"] for row in pooling.bare if row["layer"] == "protocol"}
    for encoding in ("float", "base64", "bytes"):
        for dtype in ("float16", "float32"):
            assert f"wire:encoding_format={encoding},embed_dtype={dtype}" in variants
    assert "edge:invalid_embed_dtype" in variants
    _, _, rerank = _corpus_plan("fixture-rerank-pointwise")
    bodies = {row["request_id"]: row["body"] for row in rerank.bare}
    assert bodies["edge:top_n_over_documents"]["top_n"] > len(bodies["edge:top_n_over_documents"]["documents"])
    assert bodies["wire:use_activation=false"]["use_activation"] is False
    assert "instruction" in bodies["wire:instruction=on"]
    for plan in (embed, pooling, rerank):
        for name, record in plan.strata.items():
            assert record["present"] or record.get("reason"), name


def test_stage1_validation_runs_a_skip_list_recipe_on_the_offline_fake(tmp_path: Path) -> None:
    """A multi-vector recipe that declares ``document_skip_token_ids`` validates on the product's offline fake.

    The pooling client refuses a reply whose vector count is not the count of the ids it sent (the skip
    positions would not align); the fake counts a text in the recipe's declared tokenizer, as the engine does,
    so the recipe validates as shipped (the generation once died here on a whitespace-word count).
    """
    import shutil
    import sys

    from rcp_ndcg_vllm.observe.requests import _validate_and_prune

    source = RECIPES / "fixture-multi-vector"
    target = tmp_path / "recipes" / "fixture-multi-vector-skip"
    shutil.copytree(source, target)
    shutil.copy(RECIPES.parent / "deterministic.py", tmp_path / "deterministic.py")
    manifest = (source / "recipe.yaml").read_text(encoding="utf-8")
    manifest = (
        manifest.replace("id: fixture-multi-vector", "id: fixture-multi-vector-skip")
        .replace("tokenizer: ../../tokenizer.json", f"tokenizer: {RECIPES.parent / 'tokenizer.json'}")
        .replace("  dim: 8\n", "  dim: 8\n  document_skip_token_ids: [2]\n")
    )
    (target / "recipe.yaml").write_text(manifest, encoding="utf-8")
    recipe = load_recipe(target)
    assert tuple(recipe.client.get("document_skip_token_ids")) == (2,)
    corpus = SourceCorpus(
        suite="nanobeir",
        subset="NanoNQRetrieval",
        commit="0" * 40,
        queries={"q1": SourceQuery("q1", "what is the capital of france", None, ("d1",))},
        docs={"d1": SourceDoc("d1", "paris is the capital of france, on the seine")},
    )
    plan = plan_recipe(recipe, tokenizer_of(recipe), {"nanobeir": [corpus]})
    validated, _ = _validate_and_prune(recipe, plan, sys.executable)
    assert validated.rows, "validation pruned every row"
    assert validated.validation["render_check"] == "passed", validated.validation


def test_the_offline_probe_bounds_only_the_pooling_reply_width() -> None:
    """A ``/pooling`` recipe's ``dim`` sizes the reply only (the adapter decodes by it; no request carries it),
    so stage 1's offline probe answers 8-wide vectors: the shipped width at 2 x a long budget would be a
    multi-GiB fake reply per probed text.  Every other client field -- everything a request is built from,
    and the skip list -- is the recipe's own; a non-pooling recipe is probed as it ships."""
    from rcp_ndcg_vllm.observe.requests import _offline_probe

    recipe = load_recipe(RECIPES / "fixture-multi-vector")
    wide = recipe.model_copy(update={"client": {**recipe.client, "dim": 2048, "document_skip_token_ids": (2,)}})
    probe = _offline_probe(wide)
    assert probe.client.get("dim") == 8 and tuple(probe.client.get("document_skip_token_ids")) == (2,)
    unchanged = {"dim"}

    def without(client: dict) -> dict:
        return {k: v for k, v in client.items() if k not in unchanged}

    assert without(probe.client) == without(wide.client)
    assert probe._dir == wide._dir  # noqa: SLF001 - the reference still resolves from the recipe directory
    embed = load_recipe(RECIPES / "fixture-embed")
    assert _offline_probe(embed) is embed


def test_a_per_token_probe_too_long_for_the_offline_fake_is_recorded_blocked(tmp_path: Path) -> None:
    """The offline fake seeds each per-token vector with the item's whole body, so a ``/pooling`` probe costs
    tokens x body bytes; stage 1's over-length samples (twice the budget) make that quadratic cost
    infeasible on CPU for a long-context budget.  The generator records the recipe's render check as blocked
    with the reason (the full-budget stage 1 runs against the engine on the GPU wave), never silently
    passed, and writes the rows unpruned -- in seconds, not hours."""
    import shutil
    import sys
    import time

    from rcp_ndcg_vllm.observe.requests import _validate_and_prune

    source = RECIPES / "fixture-multi-vector"
    target = tmp_path / "recipes" / "fixture-multi-vector-long"
    shutil.copytree(source, target)
    shutil.copy(RECIPES.parent / "deterministic.py", tmp_path / "deterministic.py")
    manifest = (
        (source / "recipe.yaml")
        .read_text(encoding="utf-8")
        .replace("id: fixture-multi-vector", "id: fixture-multi-vector-long")
        .replace("tokenizer: ../../tokenizer.json", f"tokenizer: {RECIPES.parent / 'tokenizer.json'}")
        .replace("max_model_len: 512", "max_model_len: 262144")
        .replace("max_tokens: 64", "max_tokens: 262142")
    )
    (target / "recipe.yaml").write_text(manifest, encoding="utf-8")
    recipe = load_recipe(target)
    corpus = SourceCorpus(
        suite="nanobeir",
        subset="NanoNQRetrieval",
        commit="0" * 40,
        queries={"q1": SourceQuery("q1", "what is the capital of france", None, ("d1",))},
        docs={"d1": SourceDoc("d1", "paris is the capital of france")},
    )
    plan = plan_recipe(recipe, tokenizer_of(recipe), {"nanobeir": [corpus]})
    started = time.monotonic()
    validated, pruned = _validate_and_prune(recipe, plan, sys.executable)
    assert time.monotonic() - started < 30
    assert validated.rows == plan.rows and pruned == []
    assert str(validated.validation["render_check"]).startswith("blocked: "), validated.validation
    assert "GPU wave" in validated.validation["render_check"]


def test_a_run_replaces_every_manifest_entry_of_the_recipes_it_touched(tmp_path: Path) -> None:
    """Runs merge per recipe: a recipe this run generated leaves ``skipped_recipes`` and drops the previous
    run's pruned rows; a recipe this run skipped leaves ``files``.  Recipes the run did not touch are kept."""
    _, plan = _plan()
    write_pairs_file(plan, tmp_path)
    other = {"recipe": "other", "error": "kept"}
    write_manifest([], tmp_path, skipped_recipes=[{"recipe": plan.recipe_id, "error": "cannot load"}, other])
    write_manifest([plan], tmp_path, pruned=[{"recipe": plan.recipe_id, "reason": "old"}])
    write_manifest([plan], tmp_path, pruned=[{"recipe": plan.recipe_id, "reason": "new"}])
    document = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert document["skipped_recipes"] == [other]
    assert [entry["reason"] for entry in document["pruned"]] == ["new"]
    assert [entry["recipe"] for entry in document["files"]] == [plan.recipe_id]
    write_manifest([], tmp_path, skipped_recipes=[{"recipe": plan.recipe_id, "error": "blocked"}])
    document = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert document["files"] == []
    assert {entry["recipe"] for entry in document["skipped_recipes"]} == {plan.recipe_id, "other"}
    assert document["pruned"] == []


def test_a_recipe_whose_validation_pruned_every_row_is_recorded_not_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty pairs file would let stage 2 pass on nothing: a recipe whose stage-1 validation pruned every
    row is a skipped recipe with the first failure named, and a previous run's file for it is removed."""
    import sys

    from rcp_ndcg_vllm.observe import requests as generator

    out = tmp_path / "pairs"
    out.mkdir()
    (out / "fixture-embed.jsonl").write_text('{"query": "stale", "documents": ["stale"]}\n', encoding="utf-8")
    monkeypatch.setattr("rcp_ndcg_vllm.observe.sources.load_corpora", lambda *args, **kwargs: [])

    def prune_all(recipe: object, plan: RecipePlan, reference_python: str) -> tuple[RecipePlan, list[dict]]:
        pruned = [{"row": index, "reason": "tail edge"} for index, _ in enumerate(plan.rows)]
        return RecipePlan(plan.recipe_id, [], plan.strata, plan.skipped_sources, {"render_check": "passed"}), pruned

    monkeypatch.setattr(generator, "_validate_and_prune", prune_all)
    argv = ["--recipes-root", str(RECIPES), "--recipes", "fixture-embed", "--out", str(out)]
    assert generator.main([*argv, "--reference-python", sys.executable, "--suites", "nanobeir"]) == 1
    assert not (out / "fixture-embed.jsonl").exists()
    document = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert document["files"] == []
    (skipped,) = document["skipped_recipes"]
    assert skipped["recipe"] == "fixture-embed" and "pruned every row" in skipped["error"]
    assert "tail edge" in skipped["error"]


def test_the_media_request_set_is_planned_for_a_media_recipe() -> None:
    """A recipe with image input plans the synthetic media request set (``MEDIA_SET_VERSION``): one row per
    image size bucket and a captioned page, the images inline and deterministic, every bucket a present
    stratum; the text rows keep their positions (the media rows come last) and a text-only recipe plans none."""
    from PIL import Image
    from rcp_ndcg_vllm.observe.media_set import MEDIA_BUCKETS

    recipe = load_recipe(RECIPES / "fixture-vl-embed")
    plan = plan_recipe(recipe, tokenizer_of(recipe), {})
    again = plan_recipe(recipe, tokenizer_of(recipe), {})
    assert pairs_jsonl(plan) == pairs_jsonl(again)
    media = [row for row in plan.rows if row.media]
    first_media = next(index for index, row in enumerate(plan.rows) if row.media)
    assert all(row.media for row in plan.rows[first_media:]), "the media rows come after every text row"
    assert len(media) == len(MEDIA_BUCKETS) + 1
    sizes = []
    for row in media:
        (entry,) = row.to_pairs_row()["media"]["documents"][0]
        payload = __import__("base64").b64decode(entry["uri"].split(",", 1)[1])
        with Image.open(__import__("io").BytesIO(payload)) as handle:
            assert handle.size == (entry["width"], entry["height"])
        sizes.append((entry["width"], entry["height"]))
    assert {(width, height) for _, width, height in MEDIA_BUCKETS} <= set(sizes)
    for name, _, _ in MEDIA_BUCKETS:
        assert plan.strata[f"media:image:{name}"]["present"] is True
    _, text_plan = _plan()
    assert all(not row.media or "suite" in str(row.media) for row in text_plan.rows)
    assert text_plan.strata["media:image:tiny"] == {
        "present": False,
        "reason": "the recipe is text-only (recipe.input declares no image)",
    }


def test_the_corpus_plan_carries_the_media_edges() -> None:
    """The media request set's protocol edges: more images than ``max_images`` in one request and an
    undecodable image, each sent bare (the engine's refusal); a recipe without video input says why no clip."""
    recipe = load_recipe(RECIPES / "fixture-vl-embed")
    from rcp_ndcg_vllm.observe.requests import corpus_plan

    plan = plan_recipe(recipe, tokenizer_of(recipe), {})
    corpus = corpus_plan(recipe, tokenizer_of(recipe), [row.to_pairs_row() for row in plan.rows])
    bare = {row["request_id"]: row["body"] for row in corpus.bare}
    parts = bare["edge:too_many_images"]["messages"][0]["content"]
    assert sum(part["type"] == "image_url" for part in parts) == recipe.client.get("max_images") + 1
    corrupt = bare["edge:corrupt_image"]["messages"][0]["content"][0]["image_url"]["url"]
    assert corrupt.startswith("data:image/png;base64,")
    assert corpus.strata["media:request_set"]["present"] is True
    assert corpus.strata["edge:too_many_images"]["present"] is True
    assert corpus.strata["media:video"]["present"] is False and corpus.strata["media:video"]["reason"]
    for name, record in corpus.strata.items():
        assert record["present"] or record.get("reason"), name
    assert "BLOCKED" not in json.dumps(corpus.strata)


def test_the_validation_runs_the_media_stage_on_the_media_rows(tmp_path: Path) -> None:
    """The generator's validation runs stage 1 on the text rows (their positions kept) and the media stage
    offline on the media rows; the manifest's validation records the media check."""
    from rcp_ndcg_vllm.observe.requests import _validate_and_prune

    recipe = load_recipe(RECIPES / "fixture-vl-embed")
    plan = plan_recipe(recipe, tokenizer_of(recipe), {})
    validated, pruned = _validate_and_prune(recipe, plan, __import__("sys").executable)
    assert validated.validation["media_check"] == "passed", validated.validation
    assert len(validated.rows) + len(pruned) == len(plan.rows)
    assert all(row.media for row in validated.rows[-len([r for r in plan.rows if r.media]) :])


def test_the_validation_prunes_the_red_text_row_where_a_media_row_precedes_it(tmp_path: Path) -> None:
    """Stage 1 reports a red row by its position among the TEXT rows; the validation maps it back to the plan.
    With a media row first, a red text row's stage-1 index is one less than its plan index: the over-budget
    row is pruned, the media row and the good text row are kept (without the remap, the good row would go)."""
    import sys

    from rcp_ndcg_vllm.observe.media_set import planned_media_rows
    from rcp_ndcg_vllm.observe.requests import _validate_and_prune

    recipe = load_recipe(RECIPES / "fixture-vl-embed")
    media, _ = planned_media_rows(recipe)
    first = media[0]
    rows = [
        PlannedRow(query=first["query"], documents=tuple(first["documents"]), media=first["media"], strata=("media",)),
        PlannedRow(query="capital of france", documents=("paris is the capital",), strata=("good",)),
        PlannedRow(query="a long one", documents=(" ".join(["cities and rivers in europe"] * 40),), strata=("long",)),
    ]
    plan = RecipePlan(recipe_id=recipe.id, rows=rows)
    validated, pruned = _validate_and_prune(recipe, plan, sys.executable)
    assert [row.strata for row in validated.rows] == [("media",), ("good",)]
    assert [entry["strata"] for entry in pruned] == [["long"]]
