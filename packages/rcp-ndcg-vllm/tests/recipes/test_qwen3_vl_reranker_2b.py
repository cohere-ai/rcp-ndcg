"""The recipe ``qwen3-vl-reranker-2b``: it validates against the product schema, its stage 1
passes on CPU, and the anchor audit is red when the template's trailing anchor segment is gone.

Stage 1 here runs with the model's tokenizer files only, downloaded once into the lane's scratch
directory (``RCP_VLLM_RECIPE_SCRATCH`` or the lane's own ``scratch/`` next to the worktree); the
download skips with its reason when the environment is offline. The GPU wave (stages 2-3) runs
the harness's full sampling (>= 20 over-length inputs per shape) on the node.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import pytest
import yaml
from jinja2 import StrictUndefined
from jinja2.sandbox import ImmutableSandboxedEnvironment
from rcp_ndcg_vllm import load_recipe, serve_argv
from rcp_ndcg_vllm.equivalence import stage1_prompts
from rcp_ndcg_vllm.equivalence.fitting import tokenizer_of

from rcp_ndcg.data.tokenizer import load_tokenizer
from rcp_ndcg.inference.config import RerankEndpoint

RECIPE_DIR = Path(__file__).resolve().parents[2] / "recipes" / "qwen3-vl-reranker-2b"
REVISION = "4bd860ac4f15ad1897a214615cccc700f8f71818"
REPO = "Qwen/Qwen3-VL-Reranker-2B"
SCRATCH_ENV = "RCP_VLLM_RECIPE_SCRATCH"
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
)


def recipe() -> object:
    """The recipe as shipped, loaded and validated through the product's endpoint config."""
    return load_recipe(RECIPE_DIR)


# ---------------------------------------------------------------------------
# The lane's scratch: where the tokenizer files live between runs (never the checkout).
# ---------------------------------------------------------------------------


def _scratch_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The scratch directory the tokenizer files download into.

    ``RCP_VLLM_RECIPE_SCRATCH`` wins; otherwise the lane's own scratch directory (a sibling of
    the worktree); otherwise an ephemeral pytest directory (a checkout without the lane layout).
    """
    from_env = os.environ.get(SCRATCH_ENV)
    if from_env:
        path = Path(from_env)
        path.mkdir(parents=True, exist_ok=True)
        return path
    worktree = Path(__file__).resolve().parents[4]
    lane = worktree.parent / "rec-qwen3-vl-reranker-2b" / "scratch"
    if worktree.parent.name == "rcp-ndcg-lanes":
        lane.mkdir(parents=True, exist_ok=True)
        return lane
    return tmp_path_factory.mktemp("recipe-tokenizer-cache")


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The model's tokenizer files at the pinned revision, downloaded once into the scratch dir.

    Skips with the reason when the Hub is unreachable (offline CI): stage 1 needs the tokenizer
    files and nothing else -- no weights, no GPU.
    """
    target = _scratch_root(tmp_path_factory) / f"tokenizer-{REVISION}"
    if not (target / "tokenizer.json").is_file():
        try:
            from huggingface_hub import hf_hub_download

            target.mkdir(parents=True, exist_ok=True)
            for name in TOKENIZER_FILES:
                shutil.copyfile(hf_hub_download(REPO, name, revision=REVISION), target / name)
        except Exception as error:  # offline CI, or the Hub refused: skip with the reason
            pytest.skip(f"offline: cannot download the {REPO} tokenizer files: {error}")
    return target


def stage1_recipe(snapshot: Path):
    """The shipped recipe with its tokenizer pointed at the downloaded snapshot (a test view).

    The shipped YAML keeps the Hub spec ``<repo>@<commit>``; the stage-1 copy reads the same
    bytes from the scratch snapshot, so ``fit`` never needs the network.
    """
    recipe = load_recipe(RECIPE_DIR)
    return recipe.model_copy(update={"client": recipe.client.model_copy(update={"tokenizer": str(snapshot)})})


def write_pairs(path: Path) -> Path:
    """Twenty text pairs: sixteen short rows and four with documents of a few hundred tokens."""
    filler = "the retrieval pipeline scores this document against the query under the declared budget. "
    rows = []
    for index in range(16):
        rows.append(
            {
                "query": f"what is the capital of country {index}",
                "documents": [
                    f"country {index} sits on the coast and its capital hosts the main harbour {index}",
                    f"a short note about country {index}",
                ],
            }
        )
    for index in range(4):
        rows.append(
            {
                "query": f"long document stress test {index}",
                "documents": [f"{filler} passage {index} carries the evidence near the end of the text. " * 12],
            }
        )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# The recipe validates, without any network: the client block IS the product's endpoint config.
# ---------------------------------------------------------------------------


def test_recipe_loads_with_the_product_rerank_endpoint() -> None:
    """The recipe loads; its client block constructs the product's RerankEndpoint."""
    recipe = load_recipe(RECIPE_DIR)
    assert recipe.id == "qwen3-vl-reranker-2b"
    assert recipe.model == REPO
    assert recipe.revision == REVISION
    assert recipe.role == "rerank" and recipe.scoring == "pointwise"
    assert recipe.input == ["text", "image"]
    assert isinstance(recipe.client, RerankEndpoint)
    assert recipe.client.model == recipe.id and recipe.client.revision == REVISION


def test_recipe_declares_the_binding_fields() -> None:
    """The explicit budget, the activation, the instruction policy, the media pin, the template."""
    recipe = load_recipe(RECIPE_DIR)
    client = recipe.client
    assert client.tokenizer == f"{REPO}@{REVISION}"
    assert client.max_tokens == 8192 and client.query_max_tokens == 4096
    assert client.on_overflow == "cut"
    assert client.use_activation is True  # probability scale, matching reference.score_scale
    assert client.instruction == "none"  # the engine's template default is pinned in the frame
    assert client.empty_doc == "send_text" and client.empty_doc_text == "NULL"
    assert recipe.serve.chat_template == "template.jinja"
    assert recipe.serve.mm_processor_kwargs == {"min_pixels": 4096, "max_pixels": 1310720}
    assert recipe.serve.hf_overrides["architectures"] == ["Qwen3VLForSequenceClassification"]
    assert recipe.serve.hf_overrides["classifier_from_token"] == ["no", "yes"]
    assert recipe.serve.hf_overrides["is_original_qwen3_reranker"] is True
    assert recipe.serve.pooler_config == {}
    assert recipe.serve.max_model_len == 32768 >= client.max_tokens
    assert recipe.reference.score_scale == "probability"
    assert recipe.reference.known_deviations == ["anchor_drop_over_cap"]
    assert recipe.status.state == "unverified"
    assert recipe.sources


def test_serve_argv_carries_the_pinned_flags() -> None:
    """The argv the wave runner renders: overrides, template, media kwargs, no pooler config."""
    recipe = load_recipe(RECIPE_DIR)
    argv = serve_argv(recipe, port=8100, served_model_name=recipe.id)
    assert json.loads(argv[argv.index("--hf-overrides") + 1]) == {
        "architectures": ["Qwen3VLForSequenceClassification"],
        "classifier_from_token": ["no", "yes"],
        "is_original_qwen3_reranker": True,
    }
    assert json.loads(argv[argv.index("--mm-processor-kwargs") + 1]) == {
        "min_pixels": 4096,
        "max_pixels": 1310720,
    }
    assert "--pooler-config" in argv and argv[argv.index("--pooler-config") + 1] == "{}"
    assert argv[argv.index("--chat-template") + 1] == str(RECIPE_DIR / "template.jinja")
    assert "--dtype" in argv and argv[argv.index("--dtype") + 1] == "bfloat16"
    assert argv[argv.index("--max-model-len") + 1] == "32768"


# ---------------------------------------------------------------------------
# Stage 1 on CPU: the declared shapes render to the served template file's ids, the anchors
# survive every over-length cut, and fit's render equals the reference subprocess's render.
# ---------------------------------------------------------------------------


def test_stage1_passes_on_cpu_token_ids_anchors_and_reference_render(tmp_path: Path, snapshot: Path) -> None:
    """At least 20 sampled pairs including 5 over-length ones, with every check green."""
    recipe = stage1_recipe(snapshot)
    pairs = write_pairs(tmp_path / "pairs.jsonl")
    document = stage1_prompts(
        recipe, pairs, sys.executable, over_length_per_shape=5
    )  # fmt: skip
    assert document["passed"] is True, json.dumps(document["anchor_check"]["failures"][:1])
    fit = document["fit"]["pair"]
    assert fit["n_texts"] == 20 + 5  # the pairs file's rows + the over-length samples
    assert fit["overhead"] == 66  # the empty render's fixed frame, measured, in tokens
    anchor = document["anchor_check"]
    assert anchor["passed"] is True and anchor["checked"] == 25
    template_check = document["template_render_check"]
    assert template_check is not None and template_check["passed"] is True
    # The token-id form of the same proof: fit's rendered ids equal the served template file's ids.
    tokenizer = tokenizer_of(recipe)
    assert load_tokenizer(str(snapshot)).sha256 == tokenizer.sha256
    row = json.loads(pairs.read_text(encoding="utf-8").splitlines()[0])
    declared = recipe.client.template.render("pair", tokenizer, query=row["query"], document=row["documents"][0])
    env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=False, undefined=StrictUndefined
    )
    jinja_render = env.from_string((RECIPE_DIR / "template.jinja").read_text(encoding="utf-8")).render(
        query=row["query"], document=row["documents"][0], instruction=""
    )
    assert tokenizer.ids(declared) == tokenizer.ids(jinja_render)
    render = document["render_check"]
    assert render["status"] == "run" and render["passed"] is True and render["rows"] == 20
    # No engine on CPU: the /tokenize check is reported not_run, never as passed.
    assert document["engine_tokenize_check"]["status"] == "not_run"
    assert document["engine_tokenize_check"]["passed"] is None


def test_mutation_dropping_the_trailing_anchor_segment_turns_the_anchor_check_red(
    tmp_path: Path, snapshot: Path
) -> None:
    """Drop the pair shape's trailing fixed segment (the assistant tail) and the audit goes red.

    The mutated recipe still loads (its ``add_special_tokens: true`` lets a shape end with the
    document content) and ``fit`` still renders -- which is exactly why the anchor audit exists:
    the rendered ids no longer end with the declared anchor edge.
    """
    mutated_dir = tmp_path / "qwen3-vl-reranker-2b"
    mutated_dir.mkdir()
    for name in ("recipe.yaml", "template.jinja", "reference.py"):
        shutil.copyfile(RECIPE_DIR / name, mutated_dir / name)
    data = yaml.safe_load((mutated_dir / "recipe.yaml").read_text(encoding="utf-8"))
    data["client"]["tokenizer"] = str(snapshot)
    data["client"]["template"]["pair"] = data["client"]["template"]["pair"][:-1]
    (mutated_dir / "recipe.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    mutated = load_recipe(mutated_dir)
    assert mutated.client.template.segments("pair")[-1].content == "document"

    document = stage1_prompts(mutated, write_pairs(tmp_path / "pairs.jsonl"), None, over_length_per_shape=2)
    anchor = document["anchor_check"]
    assert anchor["passed"] is False
    assert anchor["failures"] and anchor["failures"][0]["check"] == "tail"
