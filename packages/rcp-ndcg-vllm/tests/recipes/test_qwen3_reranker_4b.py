"""The qwen3-reranker-4b recipe: schema, stage 1 on CPU with the real tokenizer, and the mutations.

The recipe validates through the product's endpoint config; stage 1 runs the product's ``fit``
with the model's Hub tokenizer (files only, downloaded into a scratch cache — skipped with a
clear reason when the Hub is unreachable, e.g. offline CI) and proves: the anchor-preserving
cut, token-id equality with the reference subprocess's render, and the served chat template
file rendering to the declared shapes' ids. The mutations show what stage 1 catches: a recipe
whose declared template drops the trailing anchor segment goes red on the anchor check, and
the served template file without its trailing newline (the stock vLLM example's defect) goes
red on the template check.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
import yaml
from rcp_ndcg_vllm import load_recipe

RECIPES = Path(__file__).resolve().parents[2] / "recipes"
RECIPE_DIR = RECIPES / "qwen3-reranker-4b"
REVISION = "22e683669bc0f0bd69640a1354a6d0aebcfeede5"

# The last fixed segment of the declared pair shape: the assistant suffix (the model's
# read-out anchor), as the paper's code appends it after the truncated pair.
SUFFIX_TOKENS = [151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271]


def _pairs(count: int) -> list[dict]:
    """In-budget pairs: short queries and documents, no instruction (the recipe drops them)."""
    return [
        {
            "query": f"capital of france sample {index}",
            "documents": [
                f"paris is the capital of france, document {index} about cities and rivers in europe {index}"
            ],
        }
        for index in range(count)
    ]


@pytest.fixture(scope="module")
def recipe():
    """The loaded recipe (its client block constructs the product's RerankEndpoint at load)."""
    return load_recipe(RECIPE_DIR)


@pytest.fixture(scope="module")
def qwen_tokenizer():
    """The recipe's Hub tokenizer, downloaded once into a scratch cache.

    ``RCP_VLLM_TOKENIZER_CACHE`` names the cache directory (the lane's scratch on a
    lane run); without it the tokenizer downloads into a pytest temp dir. Skips when
    the Hub is unreachable and the tokenizer is not cached — stage 1 needs the real
    tokenizer's ids, and there is no stand-in for them.
    """
    cache = os.environ.get("RCP_VLLM_TOKENIZER_CACHE")
    if cache:
        os.environ.setdefault("HF_HOME", cache)
    from rcp_ndcg.data.tokenizer import load_tokenizer

    try:
        tokenizer = load_tokenizer(f"Qwen/Qwen3-Reranker-4B@{REVISION}")
    except Exception as error:  # offline CI, or the Hub refused: a clear skip, never a fake pass
        pytest.skip(f"the Qwen/Qwen3-Reranker-4B tokenizer is unavailable (offline?): {error}")
    return tokenizer


def test_recipe_validates_against_the_product_schema(recipe) -> None:
    """The recipe loads; the client block is the product's RerankEndpoint with the paper's budgets."""
    from rcp_ndcg.inference.config import RerankEndpoint

    assert recipe.id == "qwen3-reranker-4b"
    assert recipe.model == "Qwen/Qwen3-Reranker-4B"
    assert recipe.revision == REVISION
    assert isinstance(recipe.client, RerankEndpoint)
    assert recipe.role == "rerank" and recipe.scoring == "pointwise"
    assert recipe.client.tokenizer == f"Qwen/Qwen3-Reranker-4B@{REVISION}"
    assert recipe.client.max_tokens == 8192  # MAX_SEQ_LENGTH
    assert recipe.client.query_max_tokens == 4096  # MAX_QUERY_LENGTH
    assert recipe.client.on_overflow == "cut"
    assert recipe.client.instruction == "none"
    assert recipe.client.use_activation is True  # probability-scale head
    assert recipe.client.template is not None and recipe.client.template.anchor == "last"
    assert recipe.reference.known_deviations == ["anchor_drop_over_cap"]
    assert recipe.serve.chat_template == "template.jinja"
    assert recipe.serve.hf_overrides["architectures"] == ["Qwen3ForSequenceClassification"]
    assert recipe.serve.hf_overrides["classifier_from_token"] == ["no", "yes"]
    assert recipe.serve.hf_overrides["is_original_qwen3_reranker"] is True
    assert recipe.serve.max_model_len >= recipe.client.max_tokens
    assert recipe.status.state == "unverified"
    assert recipe.sources


def test_serve_argv_names_the_shipped_template(recipe) -> None:
    """The template file ships and the argv names it (REVIEW-LOG R10: without it the engine only
    warns, then concatenates the prompts); the served model name is the recipe id."""
    from rcp_ndcg_vllm.recipe import serve_argv

    argv = serve_argv(recipe, port=8100, served_model_name=recipe.id)
    assert "--chat-template" in argv
    assert str(RECIPE_DIR / "template.jinja") in argv
    assert argv[argv.index("--served-model-name") + 1] == "qwen3-reranker-4b"
    assert (RECIPE_DIR / "template.jinja").is_file()


def test_stage1_on_cpu_with_the_real_tokenizer(tmp_path: Path, recipe, qwen_tokenizer) -> None:
    """Stage 1 on CPU: 20 sampled pairs, 5 of them over-length, all green.

    The product's fit renders every sampled prompt; the anchor audit asserts the 9-token
    assistant suffix (the model's anchor) survives every cut; the reference subprocess's
    render is byte-identical on the in-budget rows; the served template file renders to the
    declared shape's text. Without an engine, the /tokenize check is reported not_run.
    """
    from rcp_ndcg_vllm.equivalence import stage1_prompts

    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text("".join(json.dumps(row) + "\n" for row in _pairs(15)), encoding="utf-8")
    document = stage1_prompts(recipe, pairs, sys.executable, over_length_per_shape=5)
    assert document["sampled"] == 20  # 15 in-budget pairs + 5 over-length samples
    assert document["anchor_check"]["passed"] is True, document["anchor_check"]["failures"][:2]
    assert document["anchor_check"]["checked"] >= 20
    assert document["render_check"]["passed"] is True, document["render_check"]["failures"][:2]
    assert document["render_check"]["rows"] == 15
    assert document["template_render_check"]["passed"] is True, document["template_render_check"]["failures"][:2]
    # R29 hygiene: without an engine the check is not_run, never passed.
    assert document["engine_tokenize_check"]["status"] == "not_run"
    assert document["engine_tokenize_check"]["passed"] is None


def test_over_length_pairs_keep_the_suffix_anchor_ids(tmp_path: Path, recipe, qwen_tokenizer) -> None:
    """The anchor check's substance, asserted directly: an over-cap pair's rendered ids still end
    with the 9 suffix tokens (im_end, newline, im_start, assistant, newline, think-open, blank,
    think-close, blank), and the render fits the 8192 budget exactly."""
    from rcp_ndcg.data.preprocess import TextBudget, fit

    client = recipe.client
    budget = TextBudget(
        tokenizer=qwen_tokenizer.name,
        max_tokens=client.max_tokens,
        query_max_tokens=client.query_max_tokens,
        template=client.template,
        on_overflow=client.on_overflow,
    )
    long_query = ("capital of france part 0 " * 1200).strip()
    long_document = ("the rivers and bridges of paris part 0 " * 1200).strip()
    result = fit([(long_query, long_document)], "pair", budget, qwen_tokenizer)
    ids = qwen_tokenizer.ids(result.texts[0], add_special_tokens=True)
    assert len(ids) <= client.max_tokens
    assert ids[-9:] == SUFFIX_TOKENS
    # and the prefix anchor: the render opens with the system-turn special
    assert qwen_tokenizer.ids(result.texts[0])[0] == 151644


def test_mutation_dropping_the_trailing_anchor_segment_turns_the_anchor_check_red(
    tmp_path: Path, recipe, qwen_tokenizer
) -> None:
    """The mutation the brief asks for: the declared template loses its trailing anchor segment
    (the assistant suffix fixed segment). The recipe still loads (the post-processor is declared
    as the anchor instead) and stage 1's anchor check goes red: the rendered ids no longer end
    with the declared edge ids."""
    mutated_dir = tmp_path / "qwen3-reranker-4b"
    mutated_dir.mkdir()
    for name in ("recipe.yaml", "reference.py", "template.jinja"):
        (mutated_dir / name).write_bytes((RECIPE_DIR / name).read_bytes())
    data = yaml.safe_load((mutated_dir / "recipe.yaml").read_text(encoding="utf-8"))
    segments = data["client"]["template"]["pair"]
    assert len(segments) == 5 and "fixed" in segments[-1]
    del segments[-1]  # the trailing anchor segment: the assistant suffix the score is read from
    data["client"]["template"]["add_special_tokens"] = {"pair": True}
    (mutated_dir / "recipe.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    from rcp_ndcg_vllm.equivalence import stage1_prompts

    mutated = load_recipe(mutated_dir)
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text("".join(json.dumps(row) + "\n" for row in _pairs(3)), encoding="utf-8")
    document = stage1_prompts(mutated, pairs, sys.executable, over_length_per_shape=2)
    assert document["anchor_check"]["passed"] is False
    assert document["anchor_check"]["failures"], "the anchor audit must name the failures"


def test_mutation_template_file_without_its_trailing_newline_turns_the_template_check_red(
    tmp_path: Path, recipe, qwen_tokenizer
) -> None:
    """The served template file without the patch's trailing newline (the stock vLLM example
    file's defect, 685 bytes) renders one newline short of the declared shape: the token-id
    equality with the declared template fails, which is the paper-exactness check the recipe
    rests on."""
    mutated_dir = tmp_path / "qwen3-reranker-4b"
    mutated_dir.mkdir()
    for name in ("recipe.yaml", "reference.py", "template.jinja"):
        (mutated_dir / name).write_bytes((RECIPE_DIR / name).read_bytes())
    template = (mutated_dir / "template.jinja").read_text(encoding="utf-8")
    (mutated_dir / "template.jinja").write_text(template.rstrip("\n"), encoding="utf-8")  # drops both

    from rcp_ndcg_vllm.equivalence import stage1_prompts

    mutated = load_recipe(mutated_dir)
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text("".join(json.dumps(row) + "\n" for row in _pairs(3)), encoding="utf-8")
    document = stage1_prompts(mutated, pairs, sys.executable, over_length_per_shape=1)
    assert document["template_render_check"]["passed"] is False
    assert document["passed"] is False
