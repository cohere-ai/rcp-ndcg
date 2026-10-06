"""The ``qwen3-reranker-0.6b`` recipe: the schema, stage 1 on CPU, and the anchor mutation.

Stage 1 runs on the real Qwen3-Reranker-0.6B tokenizer (``tokenizer.json`` and friends only — no
weights), downloaded into the lane's scratch directory and skipped with a clear reason when
offline. The reference runs as a subprocess (its ``render`` mode needs the tokenizer only; the
harness process never imports torch or transformers).
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

import pytest
import yaml
from rcp_ndcg_vllm import client_config, load_recipe, serve_argv
from rcp_ndcg_vllm.equivalence import stage1_prompts

from rcp_ndcg.data.tokenizer import load_tokenizer
from rcp_ndcg.inference.config import RerankEndpoint

RECIPE_DIR = Path(__file__).resolve().parents[2] / "recipes" / "qwen3-reranker-0.6b"
REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"
TOKENIZER_SHA256 = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
TOKENIZER_ENV = "RCP_NDCG_RECIPE_TOKENIZER_DIR"

# The lane scratch dir that holds the downloaded tokenizer files (the brief's download target),
# derived from this worktree's own location; another checkout falls back to tmp_path.
_LANES_SCRATCH = (
    Path(__file__).resolve().parents[4].parent / "rec-qwen3-reranker-0.6b" / "scratch" / "hf" / "qwen3-reranker-0.6b"
)


def _reference_python() -> str:
    """The interpreter the reference subprocess runs in (a reference env may be named)."""
    return os.environ.get("RCP_NDCG_REFERENCE_PYTHON", sys.executable)


@pytest.fixture
def tokenizer_dir(tmp_path: Path) -> Path:
    """The directory holding the recipe tokenizer's files, downloaded into the scratch dir.

    Only tokenizer files are fetched (no weights). When offline, a previously downloaded copy in
    the scratch dir (or the directory named by ``RCP_NDCG_RECIPE_TOKENIZER_DIR``) is used as is;
    with neither, the test skips — the CPU stage-1 check needs the real tokenizer.
    """
    base = Path(os.environ[TOKENIZER_ENV]) if TOKENIZER_ENV in os.environ else _LANES_SCRATCH
    if not base.parent.parent.parent.is_dir():
        base = tmp_path / "qwen3-reranker-tokenizer"
    if (base / "tokenizer.json").is_file():
        return base
    base.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download

        for name in TOKENIZER_FILES:
            downloaded = hf_hub_download(
                "Qwen/Qwen3-Reranker-0.6B", name, revision=REVISION, cache_dir=tmp_path / "hf-cache"
            )
            shutil.copyfile(downloaded, base / name)
    except Exception as error:  # offline, no [hf] extra, or the Hub refused: a clear skip either way
        pytest.skip(
            f"offline: the Qwen3-Reranker-0.6B tokenizer files are neither cached in {base} nor "
            f"downloadable ({type(error).__name__}: {error}); stage 1 on CPU needs the recipe tokenizer"
        )
    return base


def _reference_constants() -> dict[str, str]:
    """The reference module's prompt constants, imported from the recipe directory.

    The module's top level imports only the standard library (torch and transformers load lazily
    in ``load``/``score``), so importing it here never pulls weights into the harness process.
    """
    spec = importlib.util.spec_from_file_location("qwen3_reranker_reference", RECIPE_DIR / "reference.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {"prefix": module.PREFIX_TEXT, "suffix": module.SUFFIX_TEXT, "instruction": module.DEFAULT_INSTRUCTION}


def _local_recipe(tmp_path: Path, tokenizer_dir: Path):
    """The recipe with its declared Hub tokenizer spec pointed at the downloaded files.

    The copy loads through the same ``load_recipe`` validation; only the tokenizer spec changes
    (same file bytes, same SHA-256 identity), so stage 1 never touches the network.
    """
    target = tmp_path / RECIPE_DIR.name
    shutil.copytree(RECIPE_DIR, target)
    data = yaml.safe_load((target / "recipe.yaml").read_text(encoding="utf-8"))
    data["client"]["tokenizer"] = str(tokenizer_dir)
    (target / "recipe.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return load_recipe(target)


def _pairs(tmp_path: Path) -> tuple[Path, list[dict]]:
    """The stage-1 pairs: 20 in-budget rows plus 5 over-budget ones (no ``shape`` key, so the
    render check compares all 25 against the reference subprocess's paper render)."""
    rows = [
        {
            "query": f"query {index}: what is the capital of country {index}?",
            "documents": [
                f"document {index}: The capital of country {index} is city {index}, a place with "
                "rivers and history." * (1 + index % 3)
            ],
        }
        for index in range(20)
    ]
    # Over the 8192-token budget: english filler, CJK filler, mixed punctuation, a long query with
    # a long document, and a huge single-word run — the anchor must survive every cut.
    rows.append({"query": "q" * 5, "documents": ["word filler sentence number one two three four five six. " * 1500]})
    rows.append({"query": "short", "documents": ["汉字填充句子，用于测试分词边界与截断行为。" * 700]})
    rows.append(
        {
            "query": "a query with punctuation! and numbers 12345 plus UTF-8 emoji balloon",
            "documents": ["Mixed 汉字 and latin words, with punctuation; colons: and quotes - plus balloons. " * 500],
        }
    )
    rows.append(
        {"query": "longer query " * 200, "documents": ["filler text for the document side of the pair. " * 900]}
    )
    rows.append({"query": "q", "documents": ["x " * 12000]})
    # A non-NFC pair (decomposed accents): the engine tokenizes the NFC-normalized form, so the
    # ids match, but the reference's render must keep the raw characters for the render check.
    decomposed = "cafe" + chr(101) + chr(769)  # e + combining acute (not NFC)
    rows.append(
        {"query": f"what about {decomposed}?", "documents": [f"The {decomposed} is served over the river." * 3]}
    )
    path = tmp_path / "pairs.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path, rows


def test_recipe_loads_and_validates() -> None:
    """The recipe loads against the product's RerankEndpoint, with the paper's budgets."""
    recipe = load_recipe(RECIPE_DIR)
    assert recipe.id == "qwen3-reranker-0.6b" == RECIPE_DIR.name
    assert recipe.model == "Qwen/Qwen3-Reranker-0.6B"
    assert recipe.revision == REVISION and len(recipe.revision) == 40
    assert recipe.role == "rerank" and recipe.scoring == "pointwise" and recipe.input == ["text"]
    assert recipe.licence == "apache-2.0"
    assert recipe.engine.image == "vllm/vllm-openai:v0.31.0" and recipe.serve.dtype == "bfloat16"
    assert recipe.serve.max_model_len >= recipe.client.max_tokens  # the engine must not 400 the budget
    assert recipe.serve.chat_template == "qwen3_reranker.jinja"  # R10: the template file ships
    client = recipe.client
    assert client.tokenizer == f"Qwen/Qwen3-Reranker-0.6B@{REVISION}"
    assert client.max_tokens == 8192 and client.query_max_tokens == 4096  # the paper's budgets
    assert client.instruction == "none"  # the paper's served path sends no instruction
    assert client.use_activation is True  # probability-scale head, matching score_scale
    assert client.on_overflow == "cut"
    assert client.template.anchor == "last"
    assert recipe.reference.score_scale == "probability" and recipe.reference.known_deviations == []
    assert recipe.status.state == "unverified"


def test_client_config_round_trips_through_the_product() -> None:
    """The client block is the product's endpoint config: the dump constructs the model unchanged."""
    recipe = load_recipe(RECIPE_DIR)
    config = client_config(recipe, base_url="http://127.0.0.1:8100/v1")
    assert config["model"] == recipe.id and config["recipe"] == recipe.id
    endpoint = RerankEndpoint(**config)
    assert str(endpoint.base_url) == "http://127.0.0.1:8100/v1"
    again = RerankEndpoint.model_validate(config)
    assert again.model == recipe.id


def test_serve_argv_renders_the_paper_engine_command() -> None:
    """The argv a wave runs: pooling runner, the conversion overrides, the shipped template."""
    recipe = load_recipe(RECIPE_DIR)
    argv = serve_argv(recipe, port=8100, served_model_name=recipe.id)
    assert argv[:3] == ["vllm", "serve", "Qwen/Qwen3-Reranker-0.6B"]
    assert argv[argv.index("--revision") + 1] == REVISION
    assert argv[argv.index("--served-model-name") + 1] == recipe.id
    overrides = json.loads(argv[argv.index("--hf-overrides") + 1])
    assert overrides["architectures"] == ["Qwen3ForSequenceClassification"]
    assert overrides["classifier_from_token"] == ["no", "yes"]
    assert overrides["is_original_qwen3_reranker"] is True
    assert argv[argv.index("--chat-template") + 1] == str(RECIPE_DIR / "qwen3_reranker.jinja")
    assert argv[argv.index("--max-model-len") + 1] == "10000"
    assert argv[argv.index("--dtype") + 1] == "bfloat16"
    assert "--trust-remote-code" not in argv and "--convert" not in argv
    assert json.loads(argv[argv.index("--pooler-config") + 1]) == {}


def test_served_template_renders_the_reference_frame_for_both_callers(tokenizer_dir: Path) -> None:
    """The shipped template is dual-mode: the check's query/document render and the engine's
    messages render both produce the paper prompt — byte-identical to the declared shape's render."""
    from jinja2 import StrictUndefined
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    reference = _reference_constants()
    recipe = load_recipe(RECIPE_DIR)
    tokenizer = load_tokenizer(str(tokenizer_dir / "tokenizer.json"))
    assert tokenizer.sha256 == TOKENIZER_SHA256
    query, document = "capital of france", "Paris is the capital of France."
    declared = recipe.client.template.render("pair", tokenizer, query=query, document=document)
    paper = (
        reference["prefix"]
        + f"<Instruct>: {reference['instruction']}\n<Query>: {query}\n<Document>: {document}"
        + reference["suffix"]
    )
    environment = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=False, undefined=StrictUndefined
    )
    template = environment.from_string((RECIPE_DIR / recipe.serve.chat_template).read_text(encoding="utf-8"))
    assert template.render(query=query, document=document, instruction="") == declared == paper
    # The engine branch: only `messages` in scope, exactly vLLM's safe_apply_chat_template call.
    assert (
        template.render(messages=[{"role": "query", "content": query}, {"role": "document", "content": document}])
        == paper
    )


def test_stage1_on_cpu_passes_with_the_reference_render(tmp_path: Path, tokenizer_dir: Path) -> None:
    """Stage 1 on CPU: 20 in-budget + 5 over-budget pairs, token-id equality and the anchor audit.

    The reference subprocess renders the paper prompt (tokenizer only, no weights); every pair's
    fitted render must equal it byte for byte — including the 5 over-budget pairs, whose document
    cut the reference performs at the pair-string token boundary — and every anchor must survive
    the over-length inputs (this recipe's 5, plus the 20 per shape stage 1 samples itself).
    """
    recipe = _local_recipe(tmp_path, tokenizer_dir)
    pairs, _ = _pairs(tmp_path)
    report = stage1_prompts(recipe, pairs, _reference_python(), over_length_per_shape=20)
    assert report["passed"] is True, json.dumps(report)[:2000]
    assert report["render_check"]["status"] == "run" and report["render_check"]["rows"] == 26
    assert report["render_check"]["passed"] is True, report["render_check"]["failures"][:1]
    assert report["anchor_check"]["passed"] is True, report["anchor_check"]["failures"][:1]
    assert report["anchor_check"]["checked"] == 46  # 26 pairs rows + 20 over-length samples
    assert report["template_render_check"]["passed"] is True
    assert report["engine_tokenize_check"]["status"] == "not_run"  # no engine on CPU: never "passed"


def test_stage1_pairs_carry_five_over_budget_rows(tmp_path: Path, tokenizer_dir: Path) -> None:
    """The 5 marked rows of the pairs file are genuinely over the 8192-token budget."""
    recipe = _local_recipe(tmp_path, tokenizer_dir)
    tokenizer = load_tokenizer(str(tokenizer_dir / "tokenizer.json"))
    _, rows = _pairs(tmp_path)
    overhead = recipe.client.template.overhead("pair", tokenizer)
    counts = [overhead + tokenizer.count(row["query"]) + tokenizer.count(row["documents"][0]) for row in rows]
    assert sum(count > recipe.client.max_tokens for count in counts) >= 5


def test_dropping_the_trailing_anchor_segment_turns_the_anchor_check_red(tmp_path: Path, tokenizer_dir: Path) -> None:
    """Mutation: without the trailing anchor segment the anchor check fails on every render.

    The recipe still loads (the shape's ``add_special_tokens`` declares the route's post-processor
    tokens as the anchor), so nothing but the anchor audit stands between the mutation and a
    served prompt that ends mid-document — which is exactly the defect the audit exists to catch.
    """
    target = tmp_path / RECIPE_DIR.name
    shutil.copytree(RECIPE_DIR, target)
    data = yaml.safe_load((target / "recipe.yaml").read_text(encoding="utf-8"))
    data["client"]["tokenizer"] = str(tokenizer_dir)
    tail = data["client"]["template"]["pair"][-1]
    assert tail["fixed"].endswith("\n\n")  # the anchor tail: assistant turn + empty think block
    del data["client"]["template"]["pair"][-1]
    (target / "recipe.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    recipe = load_recipe(target)
    pairs, _ = _pairs(tmp_path)
    report = stage1_prompts(recipe, pairs, _reference_python(), over_length_per_shape=3)
    assert report["anchor_check"]["passed"] is False
    assert report["anchor_check"]["failures"], "the anchor audit must name its failures"
    assert all(failure["check"] == "tail" for failure in report["anchor_check"]["failures"])
