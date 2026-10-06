"""The zerank-2-reranker recipe: load validation, CPU stage 1, the anchor audit, and the mutation.

Stage 1 runs the harness's own machinery against the committed recipe (the product's fit, the
anchor audit, the served-template check and the reference subprocess). The tokenizer files are
the only download (into this test's scratch directory, never into the checkout); offline, the
tests that need them skip with a clear reason. The reference's score mode needs torch and the
checkpoint weights -- it runs on the GPU wave, never here.
"""

from __future__ import annotations

import json
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest
from rcp_ndcg_vllm import load_recipe, serve_argv
from rcp_ndcg_vllm.equivalence import stage1_prompts
from rcp_ndcg_vllm.recipe import Recipe

TESTS = Path(__file__).resolve().parent  # packages/rcp-ndcg-vllm/tests/recipes
PACKAGE = TESTS.parent.parent  # packages/rcp-ndcg-vllm
RECIPES = PACKAGE / "recipes"
RECIPE_DIR = RECIPES / "zerank-2-reranker"
RECIPE_ID = "zerank-2-reranker"
MODEL = "zeroentropy/zerank-2-reranker"
REVISION = "5eae30d5ee3c6b2df2ef6d723bde45172d761c4c"
TEMPLATE = "zerank2_score_template.jinja"
MAX_TOKENS = 8192
QUERY_MAX_TOKENS = 4096
TOKENIZER_URL = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/tokenizer.json"


def committed() -> Recipe:
    """The committed recipe, loaded and validated as the harness loads it."""
    return load_recipe(RECIPE_DIR)


def with_local_tokenizer(tokenizer_path: Path) -> Recipe:
    """The committed recipe, reading its tokenizer from the downloaded file.

    The committed recipe names the Hub spec (what production resolves); the stage-1 checks run on
    the same tokenizer.json, downloaded into this test's scratch, so they stay offline-capable
    after the one download.
    """
    recipe = committed()
    client = recipe.client.model_copy(update={"tokenizer": str(tokenizer_path)})
    return recipe.model_copy(update={"client": client})


@pytest.fixture(scope="session")
def zerank_tokenizer(tmp_path_factory) -> Path:
    """The recipe tokenizer's file, downloaded into this test's scratch directory.

    Skips with a clear reason when offline (CI): stage 1 on CPU is meaningless without the
    tokenizer the recipe declares.
    """
    target = tmp_path_factory.mktemp("zerank-tokenizer") / "tokenizer.json"
    try:
        with urllib.request.urlopen(TOKENIZER_URL, timeout=120) as response:
            target.write_bytes(response.read())
    except OSError as error:
        pytest.skip(f"offline: cannot fetch {MODEL}@{REVISION} tokenizer.json ({error}); stage 1 on CPU needs it")
    return target


def write_pairs(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def sample_pairs() -> list[dict]:
    """17 pairs rows: 13 of varied length, one instruction-bearing (the recipe folds none), one
    empty document, and two over the pair budget (both sides must cut the content and keep the
    anchor). Deterministic; the long rows are big on purpose (~21k tokens), so the render check
    exercises the reference's anchor-preserving cut against the product's fit."""
    import random

    random.seed(11)
    words = (
        "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi "
        "rho sigma tau upsilon phi chi psi omega"
    ).split()

    def document(words_target: int) -> str:
        pieces: list[str] = []
        total = 0
        while total < words_target:
            piece = " ".join(random.choices(words, k=6))
            pieces.append(piece)
            total += len(piece.split())
        return " ".join(pieces)

    rows = [
        {"query": f"what does the {words[index]} of row {index} mean", "documents": [document(random.randint(15, 150))]}
        for index in range(13)
    ]
    rows.append(
        {"query": "instruction-bearing query", "documents": [document(40)], "instruction": "Rank by relevance."}
    )
    rows.append({"query": "empty document query", "documents": [""]})
    long_document = document(21_000)
    rows.append({"query": "over the pair cap", "documents": [long_document]})
    rows.append({"query": "far over the pair cap", "documents": [f"{long_document} {long_document}"]})
    return rows


# -----------------------------------------------------------------------------------------------
# The recipe validates (offline).
# -----------------------------------------------------------------------------------------------


def test_recipe_loads_and_declares_the_paper_path() -> None:
    recipe = committed()
    assert recipe.id == RECIPE_ID
    assert recipe.model == MODEL
    assert recipe.revision == REVISION  # the 40-hex commit the research pinned (live API re-checked)
    assert recipe.role == "rerank" and recipe.scoring == "pointwise"
    assert recipe.input == ["text"]
    assert recipe.licence == "apache-2.0"
    assert recipe.engine.image == "vllm/vllm-openai:v0.31.0"
    assert recipe.client.tokenizer == f"{MODEL}@{REVISION}"
    assert recipe.client.max_tokens == MAX_TOKENS
    assert recipe.client.query_max_tokens == QUERY_MAX_TOKENS
    assert recipe.client.on_overflow == "cut"
    assert recipe.client.instruction == "none"
    assert recipe.client.use_activation is True  # the served score is the probability, like the reference
    assert recipe.reference.kind == "transformers"
    assert recipe.reference.score_scale == "probability"
    assert recipe.reference.known_deviations == ["anchor_drop_over_cap"]
    assert recipe.status.state == "unverified"
    assert recipe.sources


def test_template_declares_specials_by_name_and_the_anchor_tail() -> None:
    """The frame is data: specials by name (never a literal), the anchor is the assistant header."""
    template = committed().client.template
    assert template.shapes() == ("pair",)
    assert template.anchor == "last"
    assert template.adds_special_tokens("pair") is True
    fixed = [segment.fixed for segment in template.pair if segment.fixed is not None]
    assert len(fixed) == 3
    assert all("{special:" in segment for segment in fixed), fixed
    assert not any("<|" in segment for segment in fixed), "specials are declared by name, never typed literally"
    assert [segment.content for segment in template.pair if segment.content is not None] == ["query", "document"]
    assert template.pair[-1].fixed.endswith("{special:im_start}assistant\n")  # the trailing anchor segment


def test_the_template_file_ships_and_the_argv_carries_the_serving_facts() -> None:
    recipe = committed()
    assert (RECIPE_DIR / TEMPLATE).is_file()  # R10: without the file vLLM warns and concatenates
    argv = serve_argv(recipe, port=8100, served_model_name=recipe.id)
    joined = " ".join(argv)
    assert "--chat-template" in argv and str(RECIPE_DIR / TEMPLATE) in argv
    assert "--runner pooling" in joined
    assert "--max-model-len 8192" in joined
    assert "--pooler-config" in joined and '"logit_sigma": 5' in joined
    assert "--dtype bfloat16" in joined
    assert "--trust-remote-code" not in joined
    overrides = json.loads(argv[argv.index("--hf-overrides") + 1])
    assert overrides["architectures"] == ["Qwen3ForSequenceClassification"]
    assert overrides["classifier_from_token"] == ["Yes"]
    assert overrides["method"] == "no_post_processing"


# -----------------------------------------------------------------------------------------------
# Stage 1 on CPU: the product's fit, the anchor audit, the served-template check, the reference
# subprocess's render, and explicit token-id equality.
# -----------------------------------------------------------------------------------------------


def test_stage1_passes_on_cpu(tmp_path: Path, zerank_tokenizer: Path) -> None:
    recipe = with_local_tokenizer(zerank_tokenizer)
    pairs = write_pairs(tmp_path / "pairs.jsonl", sample_pairs())
    document = stage1_prompts(recipe, pairs, sys.executable, over_length_per_shape=5)
    assert document["passed"] is True
    # At least 20 sampled pairs incl. 5 over-length ones (the pair shape is the declared one).
    assert document["sampled"] == 22
    assert document["anchor_check"]["passed"] is True, document["anchor_check"]["failures"][:1]
    assert document["anchor_check"]["checked"] == 22
    assert document["render_check"]["status"] == "run"
    assert document["render_check"]["passed"] is True, document["render_check"]["failures"][:1]
    assert document["render_check"]["rows"] == 17
    assert document["template_render_check"]["passed"] is True, document["template_render_check"]["failures"][:1]
    assert document["engine_tokenize_check"]["status"] == "not_run"  # no engine on CPU; never reported passed
    # The declared budget: the frame overhead is 13 tokens, measured, not assumed.
    assert document["fit"]["pair"]["overhead"] == 13
    assert document["fit"]["pair"]["cuts"] > 0  # the over-budget rows were cut, not sent whole


def test_stage1_token_ids_are_equal_across_the_reference_the_fit_and_the_served_template(
    tmp_path: Path, zerank_tokenizer: Path
) -> None:
    """Token-id equality: the reference subprocess's render, the product's fit and the served
    template file's render tokenize to the same ids, per sampled row."""
    import jinja2
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    from rcp_ndcg_vllm.equivalence.fitting import budget_of

    from rcp_ndcg.data.preprocess import fit
    from rcp_ndcg.data.tokenizer import load_tokenizer

    recipe = with_local_tokenizer(zerank_tokenizer)
    tokenizer = load_tokenizer(str(zerank_tokenizer))
    rows = sample_pairs()
    budget = budget_of(recipe).model_copy(update={"tokenizer": tokenizer.name})
    template_text = (RECIPE_DIR / TEMPLATE).read_text(encoding="utf-8")
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, undefined=jinja2.StrictUndefined)

    for index, row in enumerate(rows[:4]):
        inputs = [(row["query"], row["documents"][0])]
        result = fit(inputs, "pair", budget, tokenizer, ids=[str(index)])
        fitted = result.texts[0]
        # The served template file renders the same prompt from the plain texts (the harness's check).
        jinja_text = env.from_string(template_text).render(
            query=row["query"], document=row["documents"][0], instruction=""
        )
        assert jinja_text == fitted, index
        # The reference subprocess renders the same prompt text for the row.
        rendered = subprocess.run(
            [
                sys.executable,
                str(RECIPE_DIR / "reference.py"),
                "--mode",
                "render",
                "--pairs",
                write_pairs(tmp_path / f"pairs-{index}.jsonl", [row]),
                "--out",
                str(tmp_path / f"reference-{index}.json"),
                "--tokenizer",
                str(zerank_tokenizer),
                "--device",
                "cpu",
            ],
            capture_output=True,
            text=True,
            timeout=300,
            check=True,
        )
        assert rendered.returncode == 0, rendered.stderr
        document = json.loads((tmp_path / f"reference-{index}.json").read_text(encoding="utf-8"))
        assert document["rows"][0]["shape"] == "pair"
        assert document["rows"][0]["text"] == fitted, index
        # Token-id equality, explicitly: fit's ids == the jinja render's ids == the reference's ids.
        flag = recipe.client.template.adds_special_tokens("pair")
        ids = tokenizer.ids(fitted, add_special_tokens=flag)
        assert ids == tokenizer.ids(jinja_text, add_special_tokens=flag)
        assert ids == tokenizer.ids(document["rows"][0]["text"], add_special_tokens=flag)
        # The anchor: the trailing fixed segment sits at the tail of every rendered id list.
        anchor = recipe.client.template.segments("pair")[-1].render(tokenizer)
        assert ids[-len(tokenizer.ids(anchor)) :] == tokenizer.ids(anchor)


def test_stage1_anchor_check_survives_over_length_inputs(tmp_path: Path, zerank_tokenizer: Path) -> None:
    """The anchor audit samples over-length inputs on purpose (5 here) and every cut keeps the anchor."""
    recipe = with_local_tokenizer(zerank_tokenizer)
    pairs = write_pairs(tmp_path / "pairs.jsonl", sample_pairs()[:2])
    document = stage1_prompts(recipe, pairs, None, over_length_per_shape=5)
    assert document["sampled"] == 7  # 2 pairs rows + 5 over-length samples
    assert document["anchor_check"]["passed"] is True, document["anchor_check"]["failures"][:1]
    assert document["anchor_check"]["checked"] == 7


def test_reference_cli_renders_the_anchor_preserving_prompt_and_refuses_embed(
    tmp_path: Path, zerank_tokenizer: Path
) -> None:
    """The subprocess contract: exit 0, the JSON shape, and the over-cap row's cut prompt byte-equal
    to the product's fit (the reference implements the anchor-preserving render without rcp-ndcg)."""
    from rcp_ndcg.data.preprocess import TextBudget, fit
    from rcp_ndcg.data.tokenizer import load_tokenizer

    recipe = with_local_tokenizer(zerank_tokenizer)
    tokenizer = load_tokenizer(str(zerank_tokenizer))
    rows = sample_pairs()
    over_cap = rows[-1]  # the far-over-cap row: the pair overflows, both sides cut the document
    pairs_path = write_pairs(tmp_path / "pairs.jsonl", [over_cap])
    out_path = tmp_path / "reference.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(RECIPE_DIR / "reference.py"),
            "--mode",
            "render",
            "--pairs",
            str(pairs_path),
            "--out",
            str(out_path),
            "--tokenizer",
            str(zerank_tokenizer),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stderr
    document = json.loads(out_path.read_text(encoding="utf-8"))
    assert set(document["rows"][0]) == {"index", "shape", "text"}
    budget = TextBudget(
        tokenizer=str(zerank_tokenizer),
        max_tokens=recipe.client.max_tokens,
        query_max_tokens=recipe.client.query_max_tokens,
        template=recipe.client.template,
        on_overflow=recipe.client.on_overflow,
    )
    fitted = fit([(over_cap["query"], over_cap["documents"][0])], "pair", budget, tokenizer, ids=["0"])
    assert document["rows"][0]["text"] == fitted.texts[0]
    assert len(tokenizer.ids(fitted.texts[0], add_special_tokens=True)) == recipe.client.max_tokens

    refused = subprocess.run(
        [
            sys.executable,
            str(RECIPE_DIR / "reference.py"),
            "--mode",
            "embed",
            "--pairs",
            str(pairs_path),
            "--out",
            str(tmp_path / "refused.json"),
            "--tokenizer",
            str(zerank_tokenizer),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert refused.returncode != 0
    assert "reranker" in (refused.stderr + refused.stdout)


# -----------------------------------------------------------------------------------------------
# The mutation: dropping the template's trailing anchor segment turns the anchor check red.
# -----------------------------------------------------------------------------------------------


def test_mutation_dropping_the_trailing_anchor_segment_turns_the_anchor_check_red(
    tmp_path: Path, zerank_tokenizer: Path
) -> None:
    """The anchor audit must fail when the pair shape loses its trailing fixed segment (the assistant
    header the score head reads): a cut there would pool the score from an interior token."""
    recipe = with_local_tokenizer(zerank_tokenizer)
    pairs = write_pairs(tmp_path / "pairs.jsonl", sample_pairs()[:2])

    healthy = stage1_prompts(recipe, pairs, None, over_length_per_shape=2)
    assert healthy["anchor_check"]["passed"] is True

    template = recipe.client.template
    mutated_template = template.model_copy(update={"pair": template.pair[:-1]})
    assert mutated_template.pair[-1].fixed is None  # the shape now ends with the raw content
    mutated = recipe.model_copy(update={"client": recipe.client.model_copy(update={"template": mutated_template})})
    document = stage1_prompts(mutated, pairs, None, over_length_per_shape=2)
    assert document["anchor_check"]["passed"] is False
    failures = document["anchor_check"]["failures"]
    assert failures and failures[0]["check"] == "tail"
