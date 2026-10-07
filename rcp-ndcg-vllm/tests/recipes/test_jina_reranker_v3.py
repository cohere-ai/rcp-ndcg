"""jina-reranker-v3: the recipe validates, and the harness's stage 1 passes on CPU.

The recipe is only loaded (the client block constructs the product's ``RerankEndpoint``) and stage 1
runs on CPU with tokenizer files only: the test downloads ``tokenizer.json`` at the pinned revision
into ``tmp_path`` (the product's own loader reads it), copies the recipe beside it, and runs
:func:`rcp_ndcg_vllm.equivalence.stages.stage1_prompts` with the reference subprocess (render mode is
tokenizer-only, no torch).  Offline CI skips the CPU stage with a clear reason; the recipe still
validates offline (the first test).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from rcp_ndcg_vllm.recipe import load_recipe

from rcp_ndcg.data.templates import TemplateSpec

REPO = "jinaai/jina-reranker-v3"
REVISION = "d7d7e73b6ea138ced340b83865931b5dfb6c97aa"
RECIPES = Path(__file__).resolve().parents[2] / "recipes"
RECIPE_DIR = RECIPES / "jina-reranker-v3"


def _tokenizer_file(tmp: Path) -> Path:
    """``tokenizer.json`` at the pinned revision, downloaded into ``tmp``; skips offline."""
    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as error:  # huggingface_hub is rcp-ndcg's optional [hf] extra, not a
        # dependency of this package: in an env without it (e.g. a bare `rcp-ndcg-vllm[test]` install)
        # every tokenizer-backed test here skips, and stage 1 never runs in that env.
        pytest.skip(f"huggingface_hub is not installed: {error}")
    try:
        return Path(hf_hub_download(REPO, "tokenizer.json", revision=REVISION, cache_dir=str(tmp / "hf")))
    except Exception as error:  # noqa: BLE001 - any fetch failure (offline, DNS, 4xx) skips the stage
        pytest.skip(f"offline: could not fetch {REPO}@{REVISION} tokenizer.json ({type(error).__name__}: {error})")


def _recipe_copy_with_local_tokenizer(tmp_path: Path, tokenizer_file: Path) -> Path:
    """The recipe copied into ``tmp_path``, its client tokenizer pointed at the downloaded file."""
    target = tmp_path / "jina-reranker-v3"
    shutil.copytree(RECIPE_DIR, target)
    path = target / "recipe.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["client"]["tokenizer"] = str(tokenizer_file)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return target


def _pairs(count_short: int = 15, count_long: int = 5) -> list[dict]:
    """Pairs rows: short pairs plus documents near the checkpoint's 2048-token cap (all under budget)."""
    unit = "retrieval models rank passages by relevance to a query. "  # 13 tokens with the pinned tokenizer
    long_doc = unit * 195  # ~1950 tokens: long, under the 2048 per-text cap, render-compared
    rows = [
        {
            "query": f"what ranks passages for query {index}",
            "documents": [f"Passage {index}: Paris is the capital of France and its largest city."],
        }
        for index in range(count_short)
    ]
    rows += [
        {
            "query": f"long document {index} on ranking",
            "documents": [f"{long_doc} tail sentence {index} to break the repetition."],
        }
        for index in range(count_long)
    ]
    return rows


def _reference_render(recipe_dir: Path, pairs_path: Path, tokenizer_file: Path, out_path: Path) -> list[dict]:
    """Run the reference subprocess's render mode and return its rows."""
    completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [
            sys.executable,
            str(recipe_dir / "reference.py"),
            "--mode",
            "render",
            "--pairs",
            str(pairs_path),
            "--out",
            str(out_path),
            "--tokenizer",
            str(tokenizer_file),
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stderr[-500:]
    return json.loads(out_path.read_text(encoding="utf-8"))["rows"]


def test_recipe_loads_and_declares_the_product_endpoint() -> None:
    """The recipe validates at load: the client block constructs the product's RerankEndpoint."""
    recipe = load_recipe(RECIPE_DIR)
    assert recipe.id == "jina-reranker-v3"
    assert recipe.model == "jinaai/jina-reranker-v3"
    assert recipe.revision == REVISION
    assert recipe.role == "rerank" and recipe.scoring == "listwise"
    assert recipe.client.get("listwise") is True
    assert recipe.client.get("tokenizer") == f"{REPO}@{REVISION}"
    assert recipe.client.get("max_tokens") == 3219
    assert recipe.client.get("query_max_tokens") == 512
    assert recipe.client.get("on_overflow") == "cut"
    assert recipe.client.get("empty_doc") == "omit_zero"
    assert recipe.client.get("use_activation") is False
    assert recipe.reference.score_scale == "cosine"
    assert recipe.reference.known_deviations == []
    # The engine serves the model natively: no conversion, no plugin, no chat template file.
    assert recipe.serve.runner == "pooling"
    assert recipe.serve.convert is None
    assert recipe.serve.plugin is None
    assert recipe.serve.chat_template is None
    assert recipe.serve.max_model_len == 131072
    assert recipe.serve.dtype == "bfloat16"
    # The declared anchor: the two marker tokens the pooler reads its states from.
    assert TemplateSpec.model_validate(recipe.client.get("template")).anchor == "marker"
    assert set(TemplateSpec.model_validate(recipe.client.get("template")).anchor_markers) == {
        "embed_token",
        "rerank_token",
    }


def test_declared_pair_shape_renders_the_engine_prompt(tmp_path: Path) -> None:
    """The declared pair shape renders the checkpoint builder's 1-vs-1 prompt byte for byte (tokenizer only)."""
    tokenizer_file = _tokenizer_file(tmp_path)
    from rcp_ndcg.data.tokenizer import load_tokenizer

    recipe = load_recipe(_recipe_copy_with_local_tokenizer(tmp_path, tokenizer_file))
    tokenizer = load_tokenizer(str(tokenizer_file))
    query, document = "capital of france", "Paris is the capital of France and its largest city."
    served = TemplateSpec.model_validate(recipe.client.get("template")).render(
        "pair", tokenizer, query=query, document=document
    )
    # The engine's own builder (vLLM's format_docs_prompts_func == the checkpoint's, measured by the
    # research instrument) produces exactly this text: role turns, one passage, the query block, the
    # no-thinking suffix.  Specials travel by name and resolve from the tokenizer's added tokens.
    expected_segments = [
        "system\nYou are a search relevance expert",
        "I will provide you with 1 passages, each indicated by a numerical identifier. "
        f"Rank the passages based on their relevance to query: {query}\n",
        '<passage id="0">\n',
        document,
        "</passage>\n<query>\n",
        query,
        "</query>",
    ]
    position = 0
    for segment in expected_segments:
        position = served.find(segment, position)
        assert position >= 0, f"missing segment {segment!r} in the served render"
        position += len(segment)
    # and the ids: the rendered prompt tokenizes to the same ids the reference's render tokenizes to
    pairs_path = tmp_path / "pairs.jsonl"
    pairs_path.write_text(json.dumps({"query": query, "documents": [document]}) + "\n", encoding="utf-8")
    rows = _reference_render(tmp_path / "jina-reranker-v3", pairs_path, tokenizer_file, tmp_path / "reference.json")
    assert rows[0]["shape"] == "pair"
    assert rows[0]["text"] == served
    assert tokenizer.ids(rows[0]["text"], add_special_tokens=True) == tokenizer.ids(served, add_special_tokens=True)


def test_stage1_passes_on_cpu(tmp_path: Path) -> None:
    """Stage 1 on CPU: tokenizer files only; token-id equality and the anchor audit over >= 20 sampled
    pairs, >= 5 of them over-length (the harness pads both spans past the declared budget per shape)."""
    tokenizer_file = _tokenizer_file(tmp_path)
    recipe = load_recipe(_recipe_copy_with_local_tokenizer(tmp_path, tokenizer_file))
    from rcp_ndcg_vllm.equivalence.stages import stage1_prompts

    pairs_path = tmp_path / "pairs.jsonl"
    pairs_path.write_text("".join(json.dumps(row) + "\n" for row in _pairs()), encoding="utf-8")

    report = stage1_prompts(recipe, pairs_path, sys.executable)
    assert report["passed"] is True, json.dumps(report, indent=1)[:2000]
    # 20 pairs-file rows + 20 over-length samples for the declared pair shape.
    assert report["pairs"] == 20
    assert report["sampled"] == 40
    anchor = report["anchor_check"]
    assert anchor["passed"] is True and anchor["checked"] == 40
    # The over-length samples were really cut (the fit reports them) and kept every anchor.
    assert report["fit"]["pair"]["cuts"] > 0
    # Token-id equality: the reference subprocess's render matches the product's fit, text and ids.
    render_check = report["render_check"]
    assert render_check["status"] == "run" and render_check["passed"] is True
    assert render_check["rows"] == 20


def test_stage1_mutation_dropping_the_tail_anchor_turns_red(tmp_path: Path) -> None:
    """Mutation: drop the template's trailing anchor segment (the rerank-marker tail) - the anchor
    check must go red (the marker literal disappears from every rendered prompt)."""
    tokenizer_file = _tokenizer_file(tmp_path)
    mutated = tmp_path / "mutated" / "jina-reranker-v3"
    mutated.parent.mkdir()
    shutil.copytree(_recipe_copy_with_local_tokenizer(tmp_path, tokenizer_file), mutated)
    path = mutated / "recipe.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    tail = data["client"]["template"]["pair"][-1]
    assert "rerank_token" in tail["fixed"]
    data["client"]["template"]["pair"] = data["client"]["template"]["pair"][:-1]
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    recipe = load_recipe(mutated)
    from rcp_ndcg_vllm.equivalence.stages import stage1_prompts

    pairs_path = tmp_path / "pairs.jsonl"
    rows = _pairs(count_short=2, count_long=1)
    pairs_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    report = stage1_prompts(recipe, pairs_path, None)  # anchor check only: no reference needed
    anchor = report["anchor_check"]
    assert report["passed"] is False
    assert anchor["passed"] is False
    missing = [failure.get("missing_names") for failure in anchor["failures"]]
    assert any(names and "rerank_token" in names for names in missing)
