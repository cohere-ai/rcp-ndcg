"""The recipe ``ctxl-rerank-v2-instruct-multilingual-1b``: it validates, and stage 1 passes on CPU.

The recipe declares the served contract of ContextualAI/ctxl-rerank-v2-instruct-multilingual-1b
pointwise: the pair frame (document before query, the fixed trailing " ??" suffix the last-position
score reads), the raw-logit activation (``use_activation: false``), the paper's 8192-token pair
budget with the served path's 4096-token query share, and ``reference.known_deviations:
[anchor_drop_over_cap]`` (the paper's whole-prompt right cut drops the query block over the cap; the
served path never drops an anchor).

Stage 1 here runs without an engine: the product's ``fit`` renders every sampled prompt, the anchor
audit reads ``fit``'s output, the reference subprocess's ``render`` must be byte-identical (and so
token-id identical under the recipe's tokenizer), and the served template file must render to the
same bytes. The tokenizer files (tokenizer.json only, ~11 MB) are downloaded from the Hugging Face
Hub at the recipe's pinned revision into a cache directory -- the lane's scratch dir when it exists
(so the download is fetched once per machine, never into the checkout), else pytest's own temp dir;
set ``RCP_NDCG_VLLM_TOKENIZER_CACHE`` to override the location. Tests that need the download skip
themselves with a clear reason when the Hub is unreachable (offline CI) or ``huggingface_hub`` (the
product's ``[hf]`` extra) is not installed.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from rcp_ndcg_vllm import load_recipe
from rcp_ndcg_vllm.equivalence import stage1_prompts
from rcp_ndcg_vllm.recipe import Recipe, default_recipes_root

RECIPE_ID = "ctxl-rerank-v2-instruct-multilingual-1b"
REPO_ID = "ContextualAI/ctxl-rerank-v2-instruct-multilingual-1b"
REVISION = "8fd1edf6a98564cb712064f884b8ef7df5c1b876"


def _recipe_dir() -> Path:
    """The recipe directory: the package's own ``recipes/<id>`` (default_recipes_root)."""
    return default_recipes_root() / RECIPE_ID


def _tokenizer_cache(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Where the tokenizer download lands (see the module docstring): an explicit env override, the
    lane's scratch dir when it sits beside this checkout, else pytest's own temp dir."""
    override = os.environ.get("RCP_NDCG_VLLM_TOKENIZER_CACHE")
    if override:
        return Path(override)
    lane_scratch = Path(__file__).resolve().parents[4].parent / RECIPE_ID / "scratch"
    if (lane_scratch).is_dir():
        return lane_scratch / "hf-home"
    return tmp_path_factory.mktemp("tokenizer-cache")


@pytest.fixture(scope="module")
def recipe() -> Recipe:
    """The loaded recipe (load_recipe validates it through the product's endpoint config)."""
    return load_recipe(_recipe_dir())


@pytest.fixture(scope="module")
def tokenizer_snapshot(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The recipe's tokenizer.json downloaded at the pinned revision (tokenizer files only).

    Skips with the reason when the Hub is unreachable or ``huggingface_hub`` is absent -- the
    offline-CI path. The bytes are the Hub's at the pinned revision; the product's
    ``load_tokenizer`` is pinned to this file below, so stage 1 counts in the checkpoint's tokens
    without a second download.
    """
    cache = _tokenizer_cache(tmp_path_factory)
    try:
        import huggingface_hub
    except ModuleNotFoundError as error:
        pytest.skip(f"huggingface_hub is not installed (pip install 'rcp-ndcg[hf]'): {error}")
    try:
        path = huggingface_hub.hf_hub_download(REPO_ID, "tokenizer.json", revision=REVISION, cache_dir=str(cache))
    except Exception as error:  # offline (CI without network), rate limit, or a Hub outage
        pytest.skip(f"the Hugging Face Hub is unreachable, so the tokenizer files cannot be fetched: {error}")
    return Path(path)


def _pin_tokenizer(monkeypatch: pytest.MonkeyPatch, snapshot: Path) -> None:
    """Point the product's Hub download at the already-fetched snapshot (same bytes, no re-fetch)."""
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *args, **kwargs: str(snapshot))


def _pairs(path: Path) -> Path:
    """18 short pairs (all under the 8192-token budget): half instruction-bearing, two padded.

    The padded rows pin the fold's two directions against the served path: a no-instruction row
    keeps its raw query (the wire sends it unstripped), an instruction row folds the stripped
    fields (``Task: <instruction>\nQuery: <text>``).
    """
    words = (
        "paris river model retrieval document query score ranking europe capital bank token context "
        "rerank multilingual instruction evidence passage neural archive"
    ).split()
    rows = []
    for index in range(16):
        query = f"what does ranking {index} say about the {words[index]} of europe"
        document = " ".join(words[(index + offset) % len(words)] for offset in range(24))
        if index == 7:  # no instruction: the served path sends the raw query, whitespace and all
            query = f"  {query}  "
        row: dict[str, object] = {"query": query, "documents": [document]}
        if index % 2 == 0:
            row["instruction"] = f"Follow retrieval task {index}."
        if index == 9:  # instruction-bearing: both fields are stripped by the fold
            row["query"] = f"  {query}  "
            row["instruction"] = f"  Follow retrieval task {index}.  "
        rows.append(row)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_the_recipe_loads_and_declares_the_served_contract(recipe: Recipe) -> None:
    """The recipe validates through the product's endpoint config, closed schema, no engine."""
    from rcp_ndcg.inference.config import RerankEndpoint

    assert isinstance(recipe, Recipe)
    assert (recipe.id, recipe.model, recipe.role, recipe.scoring) == (
        RECIPE_ID,
        REPO_ID,
        "rerank",
        "pointwise",
    )
    assert recipe.revision == REVISION and len(recipe.revision) == 40
    assert isinstance(recipe.client, RerankEndpoint)
    assert recipe.client.tokenizer == f"{REPO_ID}@{REVISION}"
    assert recipe.client.max_tokens == 8192 and recipe.client.query_max_tokens == 4096
    assert recipe.client.use_activation is False and recipe.client.instruction == "fold"
    assert recipe.client.template is not None and recipe.client.template.anchor == "last"
    assert recipe.reference.known_deviations == ["anchor_drop_over_cap"]
    assert recipe.serve.chat_template == "template.jinja"
    assert recipe.serve.pooler_config == {"use_activation": False}
    assert recipe.status.state == "unverified"


def test_stage1_passes_on_cpu_with_token_id_equality_and_the_anchor_check(
    recipe: Recipe, tokenizer_snapshot: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stage 1 with no engine: at least 20 sampled pairs (5 of them over-length) all pass.

    The product's ``fit`` renders every sample; the anchor audit asserts the " ??" suffix survived
    every cut; the reference subprocess's render is byte-identical (hence token-id identical under
    the recipe tokenizer); the served template file renders to fit's bytes. The engine /tokenize
    check is reported ``not_run`` (no engine on CPU) and gates nothing.
    """
    _pin_tokenizer(monkeypatch, tokenizer_snapshot)
    pairs = _pairs(tmp_path / "pairs.jsonl")
    document = stage1_prompts(recipe, pairs, sys.executable, over_length_per_shape=5)
    assert document["sampled"] >= 20, document["sampled"]
    assert document["fit"]["pair"]["cuts"] >= 5, document["fit"]["pair"]["cuts"]
    assert document["anchor_check"]["passed"] is True, document["anchor_check"]["failures"][:1]
    assert document["anchor_check"]["checked"] >= 20
    assert document["render_check"]["status"] == "run" and document["render_check"]["passed"] is True, document[
        "render_check"
    ]["failures"][:1]
    assert document["template_render_check"]["passed"] is True, document["template_render_check"]["failures"][:1]
    engine = document["engine_tokenize_check"]
    assert engine["status"] == "not_run" and engine["passed"] is None
    assert document["passed"] is True


def test_dropping_the_trailing_anchor_segment_turns_the_anchor_check_red(
    recipe: Recipe, tokenizer_snapshot: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutation: dropping the trailing `` ??`` segment (the anchor the last-position score reads)
    must turn the anchor check red."""
    _pin_tokenizer(monkeypatch, tokenizer_snapshot)
    mutated = tmp_path / RECIPE_ID
    mutated.mkdir()
    recipe_dir = Path(str(getattr(recipe, "_dir", "")) or (_recipe_dir()))
    for name in ("recipe.yaml", "template.jinja", "reference.py"):
        (mutated / name).write_bytes((recipe_dir / name).read_bytes())
    text = (mutated / "recipe.yaml").read_text(encoding="utf-8")
    tail = '      - {fixed: " ??"}       # the anchor: the score reads the last position, after this suffix\n'
    assert tail in text, "the recipe's trailing anchor segment changed; update the mutation"
    (mutated / "recipe.yaml").write_text(text.replace(tail, "      - {content: query}\n"), encoding="utf-8")
    pairs = _pairs(tmp_path / "pairs.jsonl")
    document = stage1_prompts(load_recipe(mutated), pairs, None, over_length_per_shape=3)
    assert document["anchor_check"]["passed"] is False
    assert document["anchor_check"]["failures"], "the anchor check must name the tail it no longer finds"
