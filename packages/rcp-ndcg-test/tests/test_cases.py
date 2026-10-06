"""The case format's rules, one failing case at a time.

Every rule of `case-format.md` that holds without a GPU has a test here: the file-level schema and
provenance rules, the media existence, the expected shapes and tolerances, and -- against the packaged
fixture recipe -- the recipe-backed role/modality rules, the strata coverage and the measured lengths.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from rcp_ndcg_test.cases import Case, CaseBundle, load_case, load_cases
from rcp_ndcg_test.errors import CaseError
from rcp_ndcg_test.fakes import fixture_path
from rcp_ndcg_vllm.recipe import load_recipe

PACKAGED = fixture_path("cases")
PACKAGED_RECIPES = fixture_path("recipes")
FAKE_EMBED = PACKAGED_RECIPES / "fake-embed"
DOCS_BLOCK = """      documents:
        - {id: d1, text: graded gains and a rank-sensitive metric}
        - {id: d2, text: calibrated judgements}"""


def packaged_recipe() -> object:
    """The packaged fixture recipe (fake-embed, max_tokens 128, template query+document)."""
    return load_recipe(FAKE_EMBED)


def write_case(root: Path, recipe_id: str, slug: str, body: str) -> Path:
    """One case file in a temporary cases root, and its path."""
    directory = root / recipe_id
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{slug}.yaml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


VALID = """\
    id: fake-embed/short
    recipe: fake-embed
    role: embed
    source: {kind: generated}
    strata: {modality: text, length: short, batch: single}
    inputs:
      queries: [{id: q1, text: what is retrieval evaluation}]
      documents:
        - {id: d1, text: graded gains and a rank-sensitive metric}
        - {id: d2, text: calibrated judgements}
    expected:
      kind: similarity_matrix
      values: [[0.5, 0.25]]
      tolerance: {abs: 0.001}
      origin: engine
      status: reproduced
"""

LENGTH_CASE = """\
    id: fake-embed/{slug}
    recipe: fake-embed
    role: embed
    source: {{kind: generated}}
    strata: {{modality: text, length: {length}, batch: single}}
    inputs:
      queries: [{{id: q1, text: a query}}]
      documents: [{{id: d1, text: "{text}"}}]
    expected:
      kind: similarity_matrix
      values: null
      tolerance: {{abs: 0.01}}
      origin: reference
      status: pending_gpu
"""


def valid_for(recipe_id: str, slug: str = "short") -> str:
    """The VALID case, renamed for another recipe id (id and recipe field together)."""
    return VALID.replace("id: fake-embed/", f"id: {recipe_id}/").replace("recipe: fake-embed", f"recipe: {recipe_id}")


def write_length_case(tmp_path: Path, slug: str, length: str, text: str) -> Path:
    """One pending long_* case against the packaged recipe, whose document is ``text``."""
    return write_case(tmp_path, "fake-embed", slug, LENGTH_CASE.format(slug=slug, length=length, text=text))


# ---------------------------------------------------------------------------
# The file-level rules
# ---------------------------------------------------------------------------


def test_a_valid_case_loads(tmp_path: Path) -> None:
    case = load_case(write_case(tmp_path, "fake-embed", "short", VALID))
    assert case.id == "fake-embed/short"
    assert case.expected.tolerance is not None and case.expected.tolerance.abs == 0.001


def test_the_id_must_agree_with_the_path(tmp_path: Path) -> None:
    body = VALID.replace("id: fake-embed/short", "id: fake-embed/other")
    with pytest.raises(CaseError, match="must equal the file's location"):
        load_case(write_case(tmp_path, "fake-embed", "short", body))


def test_the_file_must_be_a_yaml_mapping(tmp_path: Path) -> None:
    path = tmp_path / "fake-embed" / "short.yaml"
    path.parent.mkdir(parents=True)
    path.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(CaseError, match="must contain a mapping"):
        load_case(path)


def test_an_unknown_field_is_refused(tmp_path: Path) -> None:
    with pytest.raises(CaseError, match="cheap"):
        load_case(write_case(tmp_path, "fake-embed", "short", VALID + "\ncheap: true\n"))


def test_a_model_card_case_needs_its_provenance(tmp_path: Path) -> None:
    body = VALID.replace("source: {kind: generated}", "source: {kind: model_card}")
    with pytest.raises(CaseError, match="missing url, revision, section, quote"):
        load_case(write_case(tmp_path, "fake-embed", "short", body))


def test_a_model_card_revision_is_40_hex(tmp_path: Path) -> None:
    body = VALID.replace(
        "source: {kind: generated}",
        'source: {kind: model_card, url: "https://huggingface.co/org/model", revision: "abc123", '
        'section: "Usage", quote: "query = [\'who\']"}',
    )
    with pytest.raises(CaseError, match="revision"):
        load_case(write_case(tmp_path, "fake-embed", "short", body))


def test_a_model_card_url_is_a_huggingface_url(tmp_path: Path) -> None:
    """The format pins the card's provenance to the Hub: a model_card url names huggingface.co/<repo>."""
    body = VALID.replace(
        "source: {kind: generated}",
        'source: {kind: model_card, url: "https://example.com/org/model", revision: "' + "0" * 40 + '", '
        'section: "Usage", quote: "x = 1"}',
    )
    with pytest.raises(CaseError, match="huggingface.co"):
        load_case(write_case(tmp_path, "fake-embed", "short", body))


def test_a_model_card_case_states_its_notes(tmp_path: Path) -> None:
    """The card's rounding, the declared margin and (for none/ranking) the derivation live in notes."""
    provenance = (
        'source: {kind: model_card, url: "https://huggingface.co/org/model", revision: "' + "0" * 40 + '", '
        'section: "Usage", quote: "x = 1"}'
    )
    body = VALID.replace("source: {kind: generated}", provenance)
    with pytest.raises(CaseError, match="notes"):
        load_case(write_case(tmp_path, "fake-embed", "short", body))


def test_a_generated_case_carries_no_card_provenance(tmp_path: Path) -> None:
    body = VALID.replace(
        "source: {kind: generated}",
        'source: {kind: generated, url: "https://huggingface.co/org/model"}',
    )
    with pytest.raises(CaseError, match="no card provenance"):
        load_case(write_case(tmp_path, "fake-embed", "short", body))


def test_a_document_needs_a_part(tmp_path: Path) -> None:
    body = VALID.replace(DOCS_BLOCK, "      documents: [{id: d1}]")
    with pytest.raises(CaseError, match="carries no text, image or video"):
        load_case(write_case(tmp_path, "fake-embed", "short", body))


def test_media_lives_under_media_and_exists(tmp_path: Path) -> None:
    body = (
        VALID.replace(DOCS_BLOCK, "      documents: [{id: d1, image: media/page.png}]")
        .replace("values: [[0.5, 0.25]]", "values: [[0.5]]")
        .replace("modality: text", "modality: image")
    )
    path = write_case(tmp_path, "fake-embed", "short", body)
    with pytest.raises(CaseError, match="does not exist"):
        load_case(path)
    (tmp_path / "fake-embed" / "media").mkdir()
    (tmp_path / "fake-embed" / "media" / "page.png").write_bytes(b"png")
    assert load_case(path).inputs.documents[0].image == "media/page.png"


def test_media_may_not_escape_the_recipe_directory(tmp_path: Path) -> None:
    body = VALID.replace(DOCS_BLOCK, "      documents: [{id: d1, image: ../media/page.png}]")
    with pytest.raises(CaseError, match="media/"):
        load_case(write_case(tmp_path, "fake-embed", "short", body))


def test_query_ids_must_be_unique(tmp_path: Path) -> None:
    body = VALID.replace(
        "queries: [{id: q1, text: what is retrieval evaluation}]",
        "queries: [{id: q1, text: one}, {id: q1, text: two}]",
    )
    with pytest.raises(CaseError, match="duplicate query ids"):
        load_case(write_case(tmp_path, "fake-embed", "short", body))


# ---------------------------------------------------------------------------
# The expected block: kinds, tolerances, values
# ---------------------------------------------------------------------------


def test_values_null_is_pending_gpu_only(tmp_path: Path) -> None:
    body = VALID.replace("status: reproduced", "status: published_unverified").replace(
        "values: [[0.5, 0.25]]", "values: null"
    )
    with pytest.raises(CaseError, match="pending_gpu"):
        load_case(write_case(tmp_path, "fake-embed", "short", body))


def test_filled_values_cannot_be_pending(tmp_path: Path) -> None:
    body = VALID.replace("status: reproduced", "status: pending_gpu")
    with pytest.raises(CaseError, match="pending_gpu means no values"):
        load_case(write_case(tmp_path, "fake-embed", "short", body))


def test_exactly_one_tolerance_rule(tmp_path: Path) -> None:
    body = VALID.replace("tolerance: {abs: 0.001}", "tolerance: {abs: 0.001, spearman_min: 0.9}")
    with pytest.raises(CaseError, match="exactly one tolerance rule"):
        load_case(write_case(tmp_path, "fake-embed", "short", body))


def test_rank_exact_applies_to_rankings_only(tmp_path: Path) -> None:
    body = VALID.replace("tolerance: {abs: 0.001}", "tolerance: {rank_exact: true}")
    with pytest.raises(CaseError, match="rank_exact applies to kind 'ranking'"):
        load_case(write_case(tmp_path, "fake-embed", "short", body))


def test_a_disabled_rank_exact_is_not_a_rule(tmp_path: Path) -> None:
    """rank_exact: false declares nothing; a tolerance whose only entry is disabled is refused."""
    body = (
        VALID.replace("kind: similarity_matrix", "kind: ranking")
        .replace("values: [[0.5, 0.25]]", "values: [[d1, d2]]")
        .replace("tolerance: {abs: 0.001}", "tolerance: {rank_exact: false}")
    )
    with pytest.raises(CaseError, match="no tolerance rule|exactly one"):
        load_case(write_case(tmp_path, "fake-embed", "short", body))


def test_a_top_one_ranking_row_refuses_a_spearman_tolerance(tmp_path: Path) -> None:
    """A declared top-1 expectation carries no order to correlate: refused at load, never a free pass."""
    body = (
        VALID.replace("kind: similarity_matrix", "kind: ranking")
        .replace("values: [[0.5, 0.25]]", "values: [[d1]]")
        .replace("tolerance: {abs: 0.001}", "tolerance: {spearman_min: 0.9}")
    )
    with pytest.raises(CaseError, match="at least two documents"):
        load_case(write_case(tmp_path, "fake-embed", "short", body))


def test_a_ranking_case_rejects_an_abs_tolerance(tmp_path: Path) -> None:
    body = VALID.replace("kind: similarity_matrix", "kind: ranking").replace(
        "values: [[0.5, 0.25]]", "values: [[d1, d2]]"
    )
    with pytest.raises(CaseError, match="compares orders"):
        load_case(write_case(tmp_path, "fake-embed", "short", body))


def test_a_ranking_row_names_known_documents(tmp_path: Path) -> None:
    body = (
        VALID.replace("kind: similarity_matrix", "kind: ranking")
        .replace("values: [[0.5, 0.25]]", "values: [[d1, d9]]")
        .replace("tolerance: {abs: 0.001}", "tolerance: {rank_exact: true}")
    )
    with pytest.raises(CaseError, match="unknown documents"):
        load_case(write_case(tmp_path, "fake-embed", "short", body))


def test_rank_exact_ranks_every_document(tmp_path: Path) -> None:
    body = (
        VALID.replace("kind: similarity_matrix", "kind: ranking")
        .replace("values: [[0.5, 0.25]]", "values: [[d1]]")
        .replace("tolerance: {abs: 0.001}", "tolerance: {rank_exact: true}")
    )
    with pytest.raises(CaseError, match="rank_exact ranks every document"):
        load_case(write_case(tmp_path, "fake-embed", "short", body))


def test_a_constant_expected_row_refuses_a_spearman_tolerance(tmp_path: Path) -> None:
    """An all-tie expected row carries no ranking information: it is refused at load, not at run time."""
    body = VALID.replace("values: [[0.5, 0.25]]", "values: [[0.5, 0.5]]").replace(
        "tolerance: {abs: 0.001}", "tolerance: {spearman_min: 0.9}"
    )
    with pytest.raises(CaseError, match="constant"):
        load_case(write_case(tmp_path, "fake-embed", "short", body))


def test_spearman_needs_two_documents(tmp_path: Path) -> None:
    body = (
        VALID.replace(DOCS_BLOCK, "      documents: [{id: d1, text: one}]")
        .replace("values: [[0.5, 0.25]]", "values: [[0.5]]")
        .replace("tolerance: {abs: 0.001}", "tolerance: {spearman_min: 0.9}")
    )
    with pytest.raises(CaseError, match="at least two documents"):
        load_case(write_case(tmp_path, "fake-embed", "short", body))


def test_values_must_align_to_the_case(tmp_path: Path) -> None:
    body = VALID.replace("values: [[0.5, 0.25]]", "values: [[0.5]]")
    with pytest.raises(CaseError, match="holds 1 value"):
        load_case(write_case(tmp_path, "fake-embed", "short", body))
    body = VALID.replace("values: [[0.5, 0.25]]", "values: [[0.5, 0.25], [0.1, 0.2]]")
    with pytest.raises(CaseError, match="2 row"):
        load_case(write_case(tmp_path, "fake-embed", "short", body))


def test_values_must_be_finite_numbers(tmp_path: Path) -> None:
    for value in ("nan", ".inf", "true"):
        body = VALID.replace("values: [[0.5, 0.25]]", f"values: [[0.5, {value}]]")
        with pytest.raises(CaseError, match="non-"):
            load_case(write_case(tmp_path, "fake-embed", "short", body))


def test_origin_published_requires_a_card(tmp_path: Path) -> None:
    body = VALID.replace("origin: engine", "origin: published")
    with pytest.raises(CaseError, match="source.kind must be model_card"):
        load_case(write_case(tmp_path, "fake-embed", "short", body))


def test_a_kind_none_case_cannot_be_pending(tmp_path: Path) -> None:
    """A path-exercise case runs now: pending_gpu is the generated cases' state, not a none case's."""
    body = VALID.replace(
        """      kind: similarity_matrix
      values: [[0.5, 0.25]]
      tolerance: {abs: 0.001}""",
        "      kind: none",
    ).replace("status: reproduced", "status: pending_gpu")
    with pytest.raises(CaseError, match="pending_gpu"):
        load_case(write_case(tmp_path, "fake-embed", "short", body))


def test_a_kind_none_case_carries_nothing(tmp_path: Path) -> None:
    body = VALID.replace(
        """      kind: similarity_matrix
      values: [[0.5, 0.25]]
      tolerance: {abs: 0.001}""",
        "      kind: none",
    )
    case = load_case(write_case(tmp_path, "fake-embed", "short", body))
    assert case.expected.values is None and case.expected.tolerance is None


# ---------------------------------------------------------------------------
# The recipe-backed rules (the packaged fixture recipe)
# ---------------------------------------------------------------------------


def test_the_role_must_match_the_recipe(tmp_path: Path) -> None:
    write_case(tmp_path, "fake-embed", "short", VALID.replace("role: embed", "role: rerank"))
    with pytest.raises(CaseError, match="serves role"):
        load_cases(tmp_path, packaged_recipe(), recipes_root=PACKAGED_RECIPES, check_lengths=False)


def test_the_media_kinds_must_fit_the_recipe(tmp_path: Path) -> None:
    (tmp_path / "fake-embed" / "media").mkdir(parents=True)
    (tmp_path / "fake-embed" / "media" / "page.png").write_bytes(b"png")
    body = (
        VALID.replace(DOCS_BLOCK, "      documents: [{id: d1, image: media/page.png}]")
        .replace("modality: text", "modality: image")
        .replace("values: [[0.5, 0.25]]", "values: [[0.5]]")
    )
    write_case(tmp_path, "fake-embed", "short", body)
    with pytest.raises(CaseError, match=r"accepts input \["):
        load_cases(tmp_path, packaged_recipe(), recipes_root=PACKAGED_RECIPES, check_lengths=False)


def test_a_mixed_modality_needs_two_input_kinds(tmp_path: Path) -> None:
    write_case(tmp_path, "fake-embed", "short", VALID.replace("modality: text", "modality: mixed"))
    with pytest.raises(CaseError, match="modality 'mixed'"):
        load_cases(tmp_path, packaged_recipe(), recipes_root=PACKAGED_RECIPES, check_lengths=False)


# ---------------------------------------------------------------------------
# The measured lengths and the strata coverage
# ---------------------------------------------------------------------------


def test_the_modality_label_must_match_the_documents(tmp_path: Path) -> None:
    """modality 'image' with no image document (or 'text' with one) is a mislabel, refused at load."""
    body = VALID.replace("modality: text", "modality: image")
    with pytest.raises(CaseError, match="modality 'image'"):
        load_case(write_case(tmp_path, "fake-embed", "short", body))


def test_a_mixed_modality_batch_mixes_text_and_media(tmp_path: Path) -> None:
    body = VALID.replace("batch: single", "batch: mixed_modality")
    with pytest.raises(CaseError, match="mixed_modality"):
        load_case(write_case(tmp_path, "fake-embed", "short", body))


def test_a_mixed_length_case_declares_a_mixed_length_batch(tmp_path: Path) -> None:
    """length 'mixed' means the batch mixes lengths: a single/uniform batch is a mislabel."""
    body = VALID.replace("length: short", "length: mixed")
    with pytest.raises(CaseError, match="mixed"):
        load_case(write_case(tmp_path, "fake-embed", "short", body))


def test_long_under_must_measure_within_five_percent_under(tmp_path: Path) -> None:
    """57 rendered tokens is 45% of the fixture recipe's 128-token budget: not the long_under stratum."""
    write_length_case(tmp_path, "length-long-under", "long_under", " ".join(["budgetpad"] * 7))
    with pytest.raises(CaseError, match="within 5% under"):
        load_cases(tmp_path, packaged_recipe(), recipes_root=PACKAGED_RECIPES)


def test_long_under_must_not_be_cut_by_the_products_fit(tmp_path: Path) -> None:
    """The budget bounds the rendered input the engine sees: a span near the budget whose render (with the
    recipe's template and specials) overflows is cut by the fit and refuses the long_under claim."""
    body = LENGTH_CASE.format(slug="length-long-under", length="long_under", text=" ".join(["budgetpad"] * 14))
    # 126 raw tokens (>= 95% of 128) but the render is 134: the product's fit cuts it, so the case is a
    # mislabel -- the long_under stratum's "(no cut)" is what the render must satisfy.
    write_case(tmp_path, "fake-embed", "length-long-under", body)
    with pytest.raises(CaseError, match="long_under"):
        load_cases(tmp_path, packaged_recipe(), recipes_root=PACKAGED_RECIPES)


def test_a_long_over_case_must_actually_be_cut(tmp_path: Path) -> None:
    """long_over means the fit cuts: a render under the budget is refused as the wrong stratum."""
    body = LENGTH_CASE.format(slug="length-long-over", length="long_over", text=" ".join(["budgetpad"] * 13))
    # 117 raw tokens; the render is 125 <= 128: nothing is cut, so this is long_under, not long_over
    write_case(tmp_path, "fake-embed", "length-long-over", body)
    with pytest.raises(CaseError, match="no input renders over the budget"):
        load_cases(tmp_path, packaged_recipe(), recipes_root=PACKAGED_RECIPES)


def test_a_long_under_input_over_the_budget_is_the_wrong_stratum(tmp_path: Path) -> None:
    write_length_case(tmp_path, "length-long-under", "long_under", " ".join(["budgetpad"] * 15))  # 135 tokens
    with pytest.raises(CaseError, match="the product's fit cuts"):
        load_cases(tmp_path, packaged_recipe(), recipes_root=PACKAGED_RECIPES)


def test_a_short_case_must_not_measure_over_the_budget(tmp_path: Path) -> None:
    """A 'short' case whose inputs measure over the budget is a mislabel (the runner would cut it)."""
    write_length_case(tmp_path, "short-over", "short", " ".join(["budgetpad"] * 15))  # 135 tokens
    with pytest.raises(CaseError, match="short, but document"):
        load_cases(tmp_path, packaged_recipe(), recipes_root=PACKAGED_RECIPES)


def test_a_long_over_input_must_render_over_the_budget(tmp_path: Path) -> None:
    """126 raw tokens render to 125 with the template: whole, so the long_over label is wrong."""
    write_length_case(tmp_path, "length-long-over", "long_over", " ".join(["budgetpad"] * 13))
    with pytest.raises(CaseError, match="no input renders over the budget"):
        load_cases(tmp_path, packaged_recipe(), recipes_root=PACKAGED_RECIPES)


def test_check_lengths_false_records_the_skipped_check() -> None:
    bundle = load_cases(PACKAGED, packaged_recipe(), recipes_root=PACKAGED_RECIPES, check_lengths=False)
    assert bundle.skipped_checks == ("lengths:fake-embed (check_lengths=False)",)


def test_the_packaged_fixture_cases_sit_in_their_strata() -> None:
    """The packaged long_under / long_over cases measure exactly where their strata say."""
    bundle = load_cases(PACKAGED, packaged_recipe(), recipes_root=PACKAGED_RECIPES)
    assert {case.strata.length for case in bundle.cases} >= {"short", "long_under", "long_over"}
    assert bundle.skipped_checks == ()


def test_a_mixed_stratum_batch_also_measures_differing_lengths(tmp_path: Path) -> None:
    """The mixedness measurement covers length 'mixed' too: an all-equal batch is refused either way."""
    long_text = " ".join(["budgetpad"] * 14)  # 126 tokens with the fixture tokenizer
    body = (
        LENGTH_CASE.format(slug="mixed-batch", length="mixed", text=long_text)
        .replace("batch: single", "batch: mixed_length")
        .replace(
            "queries: [{id: q1, text: a query}]",
            f"queries: [{{id: q1, text: '{long_text}'}}, {{id: q2, text: '{long_text}'}}]",
        )
        .replace(
            f'documents: [{{id: d1, text: "{long_text}"}}]',
            f'documents: [{{id: d1, text: "{long_text}"}}, {{id: d2, text: "{long_text}"}}]',
        )
    )
    write_case(tmp_path, "fake-embed", "mixed-batch", body)
    with pytest.raises(CaseError, match="holds no mixed lengths"):
        load_cases(tmp_path, packaged_recipe(), recipes_root=PACKAGED_RECIPES)


def test_a_mixed_length_batch_measures_differing_lengths(tmp_path: Path) -> None:
    """batch 'mixed_length' means the batch holds differing measured lengths, not just the label."""
    long_text = " ".join(["budgetpad"] * 14)  # 126 tokens with the fixture tokenizer
    body = (
        LENGTH_CASE.format(slug="mixed-batch", length="short", text=long_text)
        .replace("batch: single", "batch: mixed_length")
        .replace(
            "queries: [{id: q1, text: a query}]",
            f"queries: [{{id: q1, text: '{long_text}'}}, {{id: q2, text: '{long_text}'}}]",
        )
        .replace(
            f'documents: [{{id: d1, text: "{long_text}"}}]',
            f'documents: [{{id: d1, text: "{long_text}"}}, {{id: d2, text: "{long_text}"}}]',
        )
    )
    # four inputs, all measuring 126 tokens: nothing in the batch is actually mixed
    write_case(tmp_path, "fake-embed", "mixed-batch", body)
    with pytest.raises(CaseError, match="holds no mixed lengths"):
        load_cases(tmp_path, packaged_recipe(), recipes_root=PACKAGED_RECIPES)


def test_the_strata_grid_must_be_complete(tmp_path: Path) -> None:
    write_case(tmp_path, "fake-embed", "short", VALID)
    with pytest.raises(CaseError, match="length:long_under, length:long_over, batch:mixed_length"):
        load_cases(tmp_path, packaged_recipe(), recipes_root=PACKAGED_RECIPES, check_lengths=False)


def test_the_packaged_fixture_cases_cover_the_grid() -> None:
    bundle = load_cases(PACKAGED, packaged_recipe(), recipes_root=PACKAGED_RECIPES)
    assert {case.strata.length for case in bundle.cases} >= {"short", "long_under", "long_over"}


# ---------------------------------------------------------------------------
# load_cases: roots, missing recipes, the bundle
# ---------------------------------------------------------------------------


def test_a_missing_cases_root_is_refused(tmp_path: Path) -> None:
    with pytest.raises(CaseError, match="no cases root"):
        load_cases(tmp_path / "absent")


def test_a_recipe_id_without_cases_is_refused(tmp_path: Path) -> None:
    (tmp_path / "fake-embed").mkdir()
    with pytest.raises(CaseError, match="no case directory"):
        load_cases(tmp_path, "some-other-recipe", recipes_root=PACKAGED_RECIPES)


def test_a_missing_recipe_is_recorded_not_silent(tmp_path: Path) -> None:
    write_case(tmp_path, "not-a-recipe", "short", valid_for("not-a-recipe"))
    bundle = load_cases(tmp_path, recipes_root=PACKAGED_RECIPES, check_lengths=False)
    assert bundle.recipes_missing == ("not-a-recipe",)
    assert len(bundle.cases) == 1


def test_a_broken_recipe_behind_cases_fails_the_load(tmp_path: Path) -> None:
    write_case(tmp_path, "broken-recipe", "short", valid_for("broken-recipe"))
    recipes = tmp_path / "recipes" / "broken-recipe"
    recipes.mkdir(parents=True)
    (recipes / "recipe.yaml").write_text("id: broken-recipe\nmodel: x\nrevision: '0'\n", encoding="utf-8")
    with pytest.raises(CaseError, match="does not load"):
        load_cases(tmp_path, recipes_root=recipes.parent, check_lengths=False)


def test_the_packaged_bundle_is_sorted_and_complete() -> None:
    bundle = load_cases(PACKAGED, recipes_root=PACKAGED_RECIPES)
    assert isinstance(bundle, CaseBundle)
    assert [case.id for case in bundle.cases] == sorted(case.id for case in bundle.cases)
    assert bundle.recipes_missing == ()
    assert bundle.root == PACKAGED
    assert "CaseBundle(" in repr(bundle)


def test_the_case_model_refuses_an_id_that_crosses_recipes() -> None:
    data = {
        "id": "other-recipe/short",
        "recipe": "fake-embed",
        "role": "embed",
        "source": {"kind": "generated"},
        "strata": {"modality": "text", "length": "short", "batch": "single"},
        "inputs": {"queries": [{"id": "q1", "text": "x"}], "documents": [{"id": "d1", "text": "y"}]},
        "expected": {
            "kind": "similarity_matrix",
            "values": [[0.5]],
            "tolerance": {"abs": 0.001},
            "origin": "engine",
            "status": "reproduced",
        },
    }
    with pytest.raises(ValueError, match="must be <recipe>/<case-slug>"):
        Case.model_validate(data)
