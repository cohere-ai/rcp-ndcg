"""The case format's rules, one failing case at a time.

Every rule of `case-format.md` that holds without a GPU has a test here: the file-level schema and
provenance rules, the media existence, the expected shapes and tolerances, and -- against the packaged
fixture recipe -- the recipe-backed role/modality rules, the strata coverage and the measured lengths.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from rcp_ndcg_test.cases import Case, CaseBundle, load_case, load_cases, text_of
from rcp_ndcg_test.errors import CaseError
from rcp_ndcg_test.fakes import fixture_path
from rcp_ndcg_vllm.recipe import load_recipe

PACKAGED = fixture_path("cases")
PACKAGED_RECIPES = fixture_path("recipes")
TEST_RECIPES = Path(__file__).resolve().parent / "fixtures" / "recipes"
FAKE_EMBED = PACKAGED_RECIPES / "fake-embed"
DOCS_BLOCK = """      documents:
        - {id: d1, text: graded gains and a rank-sensitive metric}
        - {id: d2, text: calibrated judgements}"""


def _tiny_png_bytes() -> bytes:
    """One syntactically valid 16x16 RGBA PNG (79 bytes): the same bytes as the committed fixture
    media (`fixtures/cases/*/media/pixel.png`), so a declared hash matches a copy of it."""
    import binascii
    import struct
    import zlib

    def chunk(name: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + name
            + payload
            + struct.pack(">I", binascii.crc32(name + payload) & 0xFFFFFFFF)
        )

    header = b"\x89PNG\r\n\x1a\n"
    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", 16, 16, 8, 6, 0, 0, 0))
    raw = b"".join(b"\x00" + b"\x20\x20\x20\x20" * 16 for _ in range(16))
    idat = chunk(b"IDAT", zlib.compress(raw))
    return header + ihdr + idat + chunk(b"IEND", b"")


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
    with pytest.raises(CaseError, match="carries no text, text_ref, image or video"):
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


def test_a_text_ref_materializes_and_verifies(tmp_path: Path) -> None:
    """text_ref produces the recorded bytes (the loader compares by hash), and any drift refuses."""
    import hashlib

    from rcp_ndcg_test.cases import TextRef

    stored = " ".join(["budgetpad"] * 13)
    digest = hashlib.sha256(stored.encode()).hexdigest()
    ref = TextRef(generator="fake-embed-ledger@1" if False else "zembed_ledger@1", params={"blocks": 0}, sha256=digest)
    # zembed_ledger with blocks=0 is the empty string, so this ref's hash is the empty text's
    empty = hashlib.sha256(b"").hexdigest()
    honest = TextRef(generator="zembed_ledger@1", params={"blocks": 0}, sha256=empty)
    assert honest.materialize() == ""
    with pytest.raises(CaseError, match="produced"):
        ref.materialize()


def test_a_case_with_a_text_ref_loads_and_materializes(tmp_path: Path) -> None:
    """A document may carry text_ref instead of text: the loader verifies the hash at load."""
    import hashlib

    from rcp_ndcg_test.cases import load_case

    text = " ".join(["budgetpad"] * 20)
    digest = hashlib.sha256(text.encode()).hexdigest()
    body = f"""
        id: fake-embed/ref-case
        recipe: fake-embed
        role: embed
        source: {{kind: generated}}
        strata: {{modality: text, length: long_under, batch: single}}
        inputs:
          queries: [{{id: q1, text: a query}}]
          documents:
            - id: d1
              text_ref:
                generator: zembed_ledger@1
                params: {{blocks: 0}}
                sha256: "{digest}"
        expected:
          kind: similarity_matrix
          values: null
          tolerance: {{abs: 0.01}}
          origin: reference
          status: pending_gpu
    """
    path = write_case(tmp_path, "fake-embed", "ref-case", body.replace("text_ref:\n", "x"))  # placeholder
    # the blocks=0 text is the empty string; give the ref the hash of the text the generator makes
    empty = hashlib.sha256(b"").hexdigest()
    body = body.replace(f'"{digest}"', f'"{empty}"')
    path = write_case(tmp_path, "fake-embed", "ref-case", body)
    case = load_case(path)
    document = case.inputs.documents[0]
    assert document.text_ref is not None and document.text is None
    assert text_of(document) == ""


def test_a_mutated_param_fails_the_hash(tmp_path: Path) -> None:
    """The hash is the correctness check: a mutated param (or generator version) refuses the load."""
    import hashlib

    text = "Entry 00000 of the ledger."
    digest = hashlib.sha256(text.encode()).hexdigest()
    body = f"""
        id: fake-embed/ref-case
        recipe: fake-embed
        role: embed
        source: {{kind: generated}}
        strata: {{modality: text, length: short, batch: single}}
        inputs:
          queries: [{{id: q1, text_ref:
              {{generator: zembed_ledger@1, params: {{blocks: 2}}, sha256: "{digest}"}}}}]
          documents: [{{id: d1, text: one}}]
        expected:
          kind: similarity_matrix
          values: null
          tolerance: {{abs: 0.01}}
          origin: reference
          status: pending_gpu
    """
    path = write_case(tmp_path, "fake-embed", "ref-case", body)
    with pytest.raises(CaseError, match="produced"):
        load_case(path)


def test_an_unknown_generator_or_version_is_refused(tmp_path: Path) -> None:
    body = f"""
        id: fake-embed/ref-case
        recipe: fake-embed
        role: embed
        source: {{kind: generated}}
        strata: {{modality: text, length: short, batch: single}}
        inputs:
          queries: [{{id: q1, text_ref:
              {{generator: no_such_gen@1, params: {{blocks: 1}}, sha256: "{"0" * 64}"}}}}]
          documents: [{{id: d1, text: one}}]
        expected:
          kind: similarity_matrix
          values: null
          tolerance: {{abs: 0.01}}
          origin: reference
          status: pending_gpu
    """
    path = write_case(tmp_path, "fake-embed", "ref-case", body)
    with pytest.raises(CaseError, match="does not implement"):
        load_case(path)


def test_a_text_ref_long_input_measures_its_stratum(tmp_path: Path) -> None:
    """A text_ref long_under case is measured through its MATERIALIZED text: the rendered input must
    sit within 5% under the budget and be uncut.

    The raw ``.text`` field of a text_ref item is ``None`` by design -- measuring the placeholder would
    see an empty input and refuse every long_under stratum. The materialized text renders at 127 of
    128 tokens (uncut), so the length rule accepts the stratum and the load moves on to the coverage
    check (which this single-case directory fails, by design: that message is the proof the length
    rule passed).
    """
    body = """
        id: fake-embed/ref-long-under
        recipe: fake-embed
        role: embed
        source: {kind: generated}
        strata: {modality: text, length: long_under, batch: single}
        inputs:
          queries: [{id: q1, text: a query}]
          documents:
            - id: d1
              text_ref:
                generator: qwen3vl_wordlist@1
                params: {seed: 7, words: 25}
                sha256: 0dbc9bde16cf3caa5d2dfb93d77a939205e621b6d7c4679b7b462d0e0bf13b6a
        expected:
          kind: similarity_matrix
          values: null
          tolerance: {abs: 0.01}
          origin: reference
          status: pending_gpu
    """
    write_case(tmp_path, "fake-embed", "ref-long-under", body)
    with pytest.raises(CaseError, match="the strata grid is incomplete"):
        load_cases(tmp_path, packaged_recipe(), recipes_root=PACKAGED_RECIPES)


def test_a_multi_query_rerank_case_measures_every_query(tmp_path: Path) -> None:
    """The length strata measure every query's pair fit, not just the first query's."""
    rerank = load_recipe(TEST_RECIPES / "fake-rerank")
    body = f"""
        id: fake-rerank/multi-query
        recipe: fake-rerank
        role: rerank
        source: {{kind: generated}}
        strata: {{modality: text, length: long_under, batch: single}}
        inputs:
          queries:
            - {{id: q1, text: long document under the budget}}
            - {{id: q2, text: '{" ".join(["overrun"] * 30)}'}}
          documents: [{{id: d1, text: '{" ".join(["budgetpad"] * 13)}'}}]
        expected:
          kind: scores
          values: null
          tolerance: {{abs: 0.01}}
          origin: reference
          status: pending_gpu
    """
    write_case(tmp_path, "fake-rerank", "multi-query", textwrap.dedent(body))
    # q1's pair renders near the budget whole; q2's fold is cut by the fit: the stratum promise is false
    with pytest.raises(CaseError, match="the product's fit cuts"):
        load_cases(tmp_path, rerank, recipes_root=PACKAGED_RECIPES)


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


def test_an_empty_instruction_is_refused_not_silently_dropped(tmp_path: Path) -> None:
    """(round-2 F3) ``instruction: ""`` claims the nothing-to-send state -- the field is dropped by
    its absence; a present-but-empty one is refused (the wire carries no instruction, silently)."""
    body = """
        id: fake-embed/empty-instruction
        recipe: fake-embed
        role: embed
        source: {kind: generated}
        strata: {modality: text, length: short, batch: single}
        inputs:
          instruction: ""
          queries: [{id: q1, text: a query}]
          documents: [{id: d1, text: graded gains and a rank-sensitive metric}]
        expected:
          kind: similarity_matrix
          values: null
          tolerance: {abs: 0.01}
          origin: reference
          status: pending_gpu
    """
    case_file = write_case(tmp_path, "fake-embed", "empty-instruction", body)
    with pytest.raises(CaseError, match="empty instruction"):
        load_case(case_file)


def test_an_image_field_may_name_an_image_only(tmp_path: Path) -> None:
    """(round-2 F5) ``image:`` names an image the product's image table knows (``.mp4`` is a video):
    the field-kind cross-check runs at load, not as an opaque decode failure mid-run."""
    body = """
        id: fake-pool/field-kind
        recipe: fake-pool
        role: multi_vector
        source: {kind: generated}
        strata: {modality: image, length: short, batch: single}
        inputs:
          queries: [{id: q1, text: describe the clip}]
          documents:
            - id: d1
              text: the placeholder square
              image: media/clip.mp4
        expected:
          kind: similarity_matrix
          values: null
          tolerance: {abs: 0.01}
          origin: reference
          status: pending_gpu
    """
    case_file = write_case(tmp_path, "fake-pool", "field-kind", body)
    (case_file.parent / "media").mkdir()
    (case_file.parent / "media" / "clip.mp4").write_bytes(b"not-a-real-clip")
    with pytest.raises(CaseError, match="names a video"):
        load_case(case_file)


def test_a_textless_media_document_does_not_supply_the_mixed_lengths(tmp_path: Path) -> None:
    """(round-2 F2, R1a) A media-only document measures nothing (it is not measured) -- it must NOT
    count as a 0-token "length" that satisfies ``length: mixed``. All text-bearing inputs measuring
    the same is a mislabel (the load refuses it)."""
    body = """
        id: fake-pool/media-supplies-no-mix
        recipe: fake-pool
        role: multi_vector
        source: {kind: generated}
        strata: {modality: mixed, length: mixed, batch: mixed_modality}
        inputs:
          queries:
            - {id: q1, text: aaaa}
            - {id: q2, text: aaaa}
          documents:
            - {id: d1, text: aaaa}
            - id: d2
              image: media/pixel.png
        expected:
          kind: similarity_matrix
          values: null
          tolerance: {abs: 0.01}
          origin: reference
          status: pending_gpu
    """
    case_file = write_case(tmp_path, "fake-pool", "media-supplies-no-mix", body)
    (case_file.parent / "media").mkdir()
    (case_file.parent / "media" / "pixel.png").write_bytes(_tiny_png_bytes())
    recipe = load_recipe(TEST_RECIPES / "fake-pool")
    with pytest.raises(CaseError, match="holds no mixed lengths"):
        load_cases(tmp_path, recipe, recipes_root=TEST_RECIPES, check_lengths=True)


def test_a_mixed_length_label_needs_the_mix_within_one_sent_batch(tmp_path: Path) -> None:
    """(round-2 F2, R1b's structural claim) The queries and the documents leave as two independent
    batches; a mix spread ACROSS those pools exercises no mixed batch on the wire -- queries 4/4 and
    documents 8/8 must refuse (the mix must occur within one sent batch)."""
    body = """
        id: fake-pool/never-mixed-on-the-wire
        recipe: fake-pool
        role: multi_vector
        source: {kind: generated}
        strata: {modality: text, length: mixed, batch: mixed_length}
        inputs:
          queries:
            - {id: q1, text: aaaa}
            - {id: q2, text: aaab}
          documents:
            - {id: d1, text: bbbb bbbb}
            - {id: d2, text: cccc cccc}
        expected:
          kind: similarity_matrix
          values: null
          tolerance: {abs: 0.01}
          origin: reference
          status: pending_gpu
    """
    write_case(tmp_path, "fake-pool", "never-mixed-on-the-wire", body)
    recipe = load_recipe(TEST_RECIPES / "fake-pool")
    with pytest.raises(CaseError, match="holds no mixed lengths"):
        load_cases(tmp_path, recipe, recipes_root=TEST_RECIPES, check_lengths=True)


def test_a_media_path_may_not_escape_the_media_directory_even_when_the_target_exists(tmp_path: Path) -> None:
    """(v1-F3) ``image: media/../outside.png`` must NOT load even though the file exists: the
    documented containment guarantee ("paths live under ``media/``") holds against ``..`` too, and
    the file content is later inlined into wire payloads -- an escape would leak bytes off-tree."""
    import shutil

    shutil.copytree(TEST_RECIPES, tmp_path / "recipes")
    (tmp_path / "tokenizer.json").write_bytes(
        (Path(__file__).resolve().parent / "fixtures" / "tokenizer.json").read_bytes()
    )
    body = """
        id: fake-pool/escape
        recipe: fake-pool
        role: multi_vector
        source: {kind: generated}
        strata: {modality: image, length: short, batch: single}
        inputs:
          queries: [{id: q1, text: describe the image}]
          documents:
            - id: d1
              text: an image of a round shape
              image: media/../outside.png
        expected:
          kind: similarity_matrix
          values: null
          tolerance: {abs: 0.01}
          origin: reference
          status: pending_gpu
    """
    case_file = write_case(tmp_path / "cases", "fake-pool", "escape", body)
    (case_file.parent / "outside.png").write_bytes(_tiny_png_bytes())  # the escape TARGET EXISTS
    (case_file.parent / "media").mkdir()
    with pytest.raises(CaseError, match="escapes the media directory"):
        load_case(case_file)


def test_a_mixed_length_label_is_measured_on_every_batch_label(tmp_path: Path) -> None:
    """(v1-F4) ``length: mixed`` means the batch mixes lengths -- measured -- on EVERY batch label,
    ``mixed_modality`` included: a mislabel cannot satisfy the grid (the CHANGELOG promises this and
    the mixed-length cross-check must not be satisfiable by a lie)."""
    body = """
        id: fake-pool/mixed-lie
        recipe: fake-pool
        role: multi_vector
        source: {kind: generated}
        strata: {modality: mixed, length: mixed, batch: mixed_modality}
        inputs:
          queries:
            - {id: q1, text: aaaa}
            - {id: q2, text: aaaa}
          documents:
            - {id: d1, text: aaaa}
            - id: d2
              text: aaaa
              image: media/pixel.png
        expected:
          kind: similarity_matrix
          values: null
          tolerance: {abs: 0.01}
          origin: reference
          status: pending_gpu
    """
    case_file = write_case(tmp_path, "fake-pool", "mixed-lie", body)
    (case_file.parent / "media").mkdir()
    (case_file.parent / "media" / "pixel.png").write_bytes(_tiny_png_bytes())
    recipe = load_recipe(TEST_RECIPES / "fake-pool")
    with pytest.raises(CaseError, match="holds no mixed lengths"):
        load_cases(tmp_path, recipe, recipes_root=TEST_RECIPES, check_lengths=True)


def test_a_run_level_instruction_must_be_on_the_wire_for_the_embed_side(tmp_path: Path) -> None:
    """(v1-F1) The embed/multi_vector clients have no instruction slot: the run-level ``instruction``
    reaches the wire ONLY through the recipe's ``query_prompt`` (the product's one text-prefix
    mechanism for that side). A case declaring an instruction the recipe's query prompt does not
    carry is refused at the recipe-validated load -- the declared inputs would not be what is sent,
    silently."""
    import shutil

    shutil.copytree(TEST_RECIPES, tmp_path / "recipes")
    (tmp_path / "tokenizer.json").write_bytes(
        (Path(__file__).resolve().parent / "fixtures" / "tokenizer.json").read_bytes()
    )
    body = """
        id: fake-pool/instruction-dropped
        recipe: fake-pool
        role: multi_vector
        source: {kind: generated}
        strata: {modality: text, length: short, batch: single}
        inputs:
          instruction: Given a search query, retrieve the passage
          queries: [{id: q1, text: describe the image}]
          documents: [{id: d1, text: a round shape}]
        expected:
          kind: similarity_matrix
          values: null
          tolerance: {abs: 0.01}
          origin: reference
          status: pending_gpu
    """
    write_case(tmp_path, "fake-pool", "instruction-dropped", body)
    recipe = load_recipe(tmp_path / "recipes" / "fake-pool")
    with pytest.raises(CaseError, match="instruction the recipe's query prompt does not carry"):
        load_cases(tmp_path, recipe, recipes_root=TEST_RECIPES, check_lengths=False)


def test_a_media_case_needs_the_recipe_to_declare_its_media_policy(tmp_path: Path) -> None:
    """(the shakedown's sweep-recipes finding #7) A case naming an image needs a recipe whose client
    declares it reads images (``max_images >= 1`` and an ``image_policy``; ``max_videos`` /
    ``video_policy`` for video): the product's own gate refuses the send otherwise. The load fails
    instead -- the mistake the shakedown hit (``recipe.input`` declares image while the client's
    media policy is empty) is refused here, and only here is the case allowed to name media.
    """
    import shutil

    shutil.copytree(TEST_RECIPES, tmp_path / "recipes")
    (tmp_path / "tokenizer.json").write_bytes(
        (Path(__file__).resolve().parent / "fixtures" / "tokenizer.json").read_bytes()
    )
    # the fixture recipe with its media policy removed: the shakedown's exact mistake
    recipe_yaml = tmp_path / "recipes" / "fake-pool" / "recipe.yaml"
    cfg = recipe_yaml.read_text(encoding="utf-8")
    stripped = "  image_policy: {min_px: 3136, max_px: 1003520, processor: qwen2_vl}\n"
    assert stripped in cfg, "the fixture recipe's media policy moved; fix this test against its shape"
    cfg = cfg.replace("  max_images: 4\n", "  max_images: 0\n").replace(stripped, "")
    recipe_yaml.write_text(cfg, encoding="utf-8")
    assert "image_policy" not in cfg and "max_images: 0" in cfg, "the surgery must apply (nothing silent)"

    body = """
        id: fake-pool/policy-missing
        recipe: fake-pool
        role: multi_vector
        source: {kind: generated}
        strata:
          modality: image
          length: short
          batch: single
        inputs:
          queries: [{id: q1, text: describe the image}]
          documents:
            - id: d1
              text: an image of a round shape
              image: media/pixel.png
        expected:
          kind: similarity_matrix
          values: null
          tolerance: {abs: 0.01}
          origin: reference
          status: pending_gpu
    """
    case_file = write_case(tmp_path / "cases", "fake-pool", "policy-missing", body)
    (case_file.parent / "media").mkdir()
    (case_file.parent / "media" / "pixel.png").write_bytes(_tiny_png_bytes())
    recipe = load_recipe(tmp_path / "recipes" / "fake-pool")
    with pytest.raises(CaseError, match="does not declare it reads images"):
        load_cases(tmp_path / "cases", recipe, recipes_root=tmp_path / "recipes", check_lengths=False)


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
