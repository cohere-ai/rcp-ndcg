"""jina-reranker-v3: the recipe validates, and the harness's stage 1 passes on CPU.

The recipe is only loaded (the client block constructs the product's ``RerankEndpoint``) and stage 1
runs on CPU with tokenizer files only: the test downloads ``tokenizer.json`` at the pinned revision
through the shared ``_served.fetch_tokenizer`` (the tokenizer cache, sha256-pinned; the product's own
loader reads it), copies the recipe beside it, and runs
:func:`rcp_ndcg_test.equivalence.stages.stage1_prompts` with the reference subprocess (render mode is
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
from rcp_ndcg_test.equivalence.fitting import tokenizer_of
from rcp_ndcg_vllm.recipe import default_recipes_root, load_recipe

from ._contract import assert_recipe_contract
from ._served import client_template, fetch_tokenizer, stage1_facts

REPO = "jinaai/jina-reranker-v3"
REVISION = "d7d7e73b6ea138ced340b83865931b5dfb6c97aa"
RECIPES = default_recipes_root()
RECIPE_DIR = RECIPES / "jina-reranker-v3"
TOKENIZER_URL = f"https://huggingface.co/{REPO}/resolve/{REVISION}/tokenizer.json"
TOKENIZER_SHA256 = "4e95945ab0cef486709f760b81efcc7a6e75747f9165d13ead29159737455803"  # Hub LFS oid at REVISION


def _tokenizer_file(tmp: Path) -> Path:
    """``tokenizer.json`` at the pinned revision, sha256-checked, through the shared tokenizer cache;
    skips offline."""
    return fetch_tokenizer(TOKENIZER_URL, f"jina-reranker-v3@{REVISION}/tokenizer.json", tmp, sha256=TOKENIZER_SHA256)


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
    _assert_contract(recipe)  # every serve, client and reference field pinned, exactly
    assert recipe.id == "jina-reranker-v3"
    assert recipe.model == "jinaai/jina-reranker-v3"
    assert recipe.revision == REVISION
    assert recipe.role == "rerank" and recipe.scoring == "listwise"
    assert recipe.client.get("listwise") is True
    assert recipe.client.get("tokenizer") == f"{REPO}@{REVISION}"
    assert recipe.client.get("max_tokens") == 3219
    assert recipe.client.get("query_max_tokens") == 512
    assert recipe.client.get("document_max_tokens") == 2048  # the checkpoint's max_doc_length, beside the pair budget
    assert recipe.client.get("on_overflow") == "cut"
    assert recipe.client.get("empty_doc") == "omit_zero"
    assert recipe.client.get("use_activation") is False
    assert recipe.reference.score_scale == "cosine"
    assert recipe.reference.known_deviations == ["over_cap_cut_differs"]
    # The engine serves the model natively: no conversion, no plugin, no chat template file.
    assert recipe.serve.runner == "pooling"
    assert recipe.serve.convert is None
    assert recipe.serve.plugin is None
    assert recipe.serve.chat_template is None
    assert recipe.serve.max_model_len == 131072
    assert recipe.serve.dtype == "bfloat16"
    # The declared anchor: the two marker tokens the pooler reads its states from.
    assert client_template(recipe).anchor == "marker"
    assert set(client_template(recipe).anchor_markers) == {"embed_token", "rerank_token"}


def _reference_full_prompt(recipe_dir: Path, query: str, document: str, tokenizer_file: Path) -> str:
    """The reference's prompt-level 1-vs-1 render (the declared pair shape's referent)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("jina_reranker_v3_reference", recipe_dir / "reference.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.render_query_and_document(query, document, str(tokenizer_file))


def test_declared_pair_shape_renders_the_engine_prompt(tmp_path: Path) -> None:
    """The declared pair shape renders the checkpoint builder's 1-vs-1 prompt byte for byte (tokenizer only)."""
    tokenizer_file = _tokenizer_file(tmp_path)
    from rcp_ndcg.data.tokenizer import load_tokenizer

    recipe = load_recipe(_recipe_copy_with_local_tokenizer(tmp_path, tokenizer_file))
    tokenizer = load_tokenizer(str(tokenizer_file))
    query, document = "capital of france", "Paris is the capital of France and its largest city."
    served = client_template(recipe).render("pair", tokenizer, query=query, document=document)
    # The engine's own builder (vLLM's format_docs_prompts_func == the checkpoint's, measured when the
    # recipe was written) produces exactly this text: role turns, one passage, the query block, the
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
    # Stage 1's render contract: the reference emits the wire's cut content spans (no frame -- the
    # frame is the engine's own builder); an under-cap pair ships its spans whole.
    pairs_path = tmp_path / "pairs.jsonl"
    pairs_path.write_text(json.dumps({"query": query, "documents": [document]}) + "\n", encoding="utf-8")
    rows = _reference_render(tmp_path / "jina-reranker-v3", pairs_path, tokenizer_file, tmp_path / "reference.json")
    assert rows[0]["shape"] == "pair"
    assert rows[0]["query"] == query and rows[0]["documents"] == [document]
    # and the prompt-level view stays byte-identical to the declared pair shape, ids included.
    full = _reference_full_prompt(tmp_path / "jina-reranker-v3", query, document, tokenizer_file)
    assert full == served
    assert tokenizer.ids(full, add_special_tokens=True) == tokenizer.ids(served, add_special_tokens=True)


def test_stage1_passes_on_cpu(tmp_path: Path) -> None:
    """Stage 1 on CPU: tokenizer files only; token-id equality and the anchor audit over >= 20 sampled
    pairs, >= 5 of them over-length (the harness pads both spans past the declared budget per shape)."""
    tokenizer_file = _tokenizer_file(tmp_path)
    recipe = load_recipe(_recipe_copy_with_local_tokenizer(tmp_path, tokenizer_file))
    from rcp_ndcg_test.equivalence.stages import stage1_prompts

    pairs_path = tmp_path / "pairs.jsonl"
    pairs_path.write_text("".join(json.dumps(row) + "\n" for row in _pairs()), encoding="utf-8")

    report = stage1_prompts(recipe, pairs_path, sys.executable)
    assert report["passed"] is True, json.dumps(report, indent=1)[:2000]
    # 20 pairs-file rows + 20 over-length samples for the declared pair shape.
    assert report["pairs"] == 20
    assert report["sampled"] == 40
    anchor = report["anchor_check"]
    assert anchor["passed"] is True and anchor["checked"] >= 40  # one audit row per pair row and span
    # The over-length samples were really cut (the client's census reports them) and kept every anchor.
    facts = stage1_facts(recipe, _pairs(), tokenizer_of(recipe))
    assert facts["per_shape"]["pair"]["cuts"] > 0
    # Token-id equality: the reference subprocess's render matches the product's fit, text and ids.
    render_check = report["render_check"]
    assert render_check["status"] == "run" and render_check["passed"] is True
    assert render_check["rows"] == 20


def test_mutation_dropping_the_tail_segment_breaks_the_declared_shape(tmp_path: Path) -> None:
    """Mutation: drop the template's trailing fixed segment (the one carrying the query marker).

    On the rerank wire the request ships content spans only and the ENGINE's own builder assembles
    the frame (the markers included), so the span audit cannot see a marker drop -- the red guard
    for this recipe is the declared shape's byte-equality with the builder's prompt (the stage-1
    fixture test above, against the reference's prompt-level render). The mutation makes the
    declared shape render a DIFFERENT prompt from the builder's, and this test asserts exactly
    that divergence became detectable.
    """
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

    from rcp_ndcg.data.tokenizer import load_tokenizer

    recipe = load_recipe(mutated)
    tokenizer = load_tokenizer(str(tokenizer_file))
    query, document = "capital of france", "Paris is the capital of France."
    template = client_template(recipe)
    assert template is not None
    declared = template.render("pair", tokenizer, query=query, document=document)
    full = _reference_full_prompt(mutated, query, document, tokenizer_file)
    assert full != declared, "dropping the tail segment must diverge from the engine builder's prompt"


# ---------------------------------------------------------------------------
# The declared contract: every serve, client and reference field pinned.
# ---------------------------------------------------------------------------

EXPECTED_TOP = {
    "id": "jina-reranker-v3",
    "input": ["text"],
    "licence": "cc-by-nc-4.0",
    "model": "jinaai/jina-reranker-v3",
    "revision": "d7d7e73b6ea138ced340b83865931b5dfb6c97aa",
    "role": "rerank",
    "scoring": "listwise",
}
EXPECTED_SERVE = {
    "chat_template": None,
    "convert": None,
    "dtype": "bfloat16",
    "extra_args": [],
    "hf_overrides": {},
    "io_processor_plugin": None,
    "limit_mm_per_prompt": None,
    "max_model_len": 131072,
    "mm_processor_kwargs": {},
    "plugin": None,
    "pooler_config": {"use_activation": False},
    "runner": "pooling",
    "trust_remote_code": False,
}
EXPECTED_CLIENT = {
    "api": "rerank",
    "request_shape": "text",
    "template": {
        "pair": [
            {
                "fixed": "{special:im_start}system\n"
                "You are a search relevance expert who can determine a ranking of the "
                "passages based on how relevant they are to the query. If the query is a "
                "question, how relevant a passage is depends on how well it answers the "
                "question. If not, try to analyze the intent of the query and assess how "
                "well each passage satisfies the intent. If an instruction is provided, you "
                "should follow the instruction when determining the "
                "ranking.{special:im_end}\n"
                "{special:im_start}user\n"
                "I will provide you with 1 passages, each indicated by a numerical "
                "identifier. Rank the passages based on their relevance to query: "
            },
            {"content": "query"},
            {"fixed": '\n<passage id="0">\n'},
            {"content": "document"},
            {"fixed": "{special:embed_token}\n</passage>\n<query>\n"},
            {"content": "query"},
            {
                "fixed": "{special:rerank_token}\n"
                "</query>{special:im_end}\n"
                "{special:im_start}assistant\n"
                "{special:<think>}\n"
                "\n"
                "{special:</think>}\n"
                "\n"
            },
        ],
        "anchor": "marker",
        "anchor_markers": ["embed_token", "rerank_token"],
        "add_special_tokens": True,
    },
    "instruction": "none",
    "tokenizer": "jinaai/jina-reranker-v3@d7d7e73b6ea138ced340b83865931b5dfb6c97aa",
    "max_tokens": 3219,
    "query_max_tokens": 512,
    "document_max_tokens": 2048,
    "on_overflow": "cut",
    "empty_doc": "omit_zero",
    "use_activation": False,
    "listwise": True,
    "model": "jina-reranker-v3",
    "revision": "d7d7e73b6ea138ced340b83865931b5dfb6c97aa",
}
EXPECTED_REFERENCE = {
    "entry": "reference.py",
    "kind": "remote_code",
    "known_deviations": ["over_cap_cut_differs"],
    "score_scale": "cosine",
}

# Two mutants per recipe against the contract pin above: each drift must fail, naming the field.
MUTANTS: list[tuple[str, tuple[str, ...], object, str]] = [
    (
        "reference.kind drifts to sentence_transformers",
        ("reference", "kind"),
        "sentence_transformers",
        "reference.kind",
    ),
    ("serve.pooler_config drops the use_activation pin", ("serve", "pooler_config"), {}, "use_activation"),
]


def _mutated_recipe(tmp_path: Path, path: tuple[str, ...], value: object) -> object:
    """The recipe directory copied into ``tmp_path`` with one YAML field set to ``value``."""
    import shutil

    import yaml

    target = tmp_path / RECIPE_DIR.name
    shutil.copytree(RECIPE_DIR, target)
    yaml_path = target / "recipe.yaml"
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    node = data
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return load_recipe(target)


def _assert_contract(recipe: object) -> None:
    assert_recipe_contract(
        recipe,
        serve=EXPECTED_SERVE,
        client=EXPECTED_CLIENT,
        reference=EXPECTED_REFERENCE,
        top=EXPECTED_TOP,
    )


@pytest.mark.parametrize(("label", "path", "value", "needle"), MUTANTS, ids=[m[0] for m in MUTANTS])
def test_two_contract_mutants_are_red(
    label: str, path: tuple[str, ...], value: object, needle: str, tmp_path: Path
) -> None:
    """A drifted field fails the contract assertion naming it (two mutants per recipe)."""
    _assert_contract(load_recipe(RECIPE_DIR))  # the pinned recipe itself is green
    with pytest.raises(AssertionError) as caught:
        _assert_contract(_mutated_recipe(tmp_path, path, value))
    assert needle in str(caught.value), f"{label}: the failure must name {needle}: {caught.value}"


def test_notes_state_the_settle_rule_and_the_query_cap() -> None:
    """Finding #5's wording: the pair fit binds on overflow; the rerank client settles the shared
    query span once per call at its declared share -- read from the merged client -- and
    query_max_tokens 512 is the reference's own per-query cap."""
    notes = load_recipe(RECIPE_DIR).notes
    assert "fit binds on overflow only" in notes
    assert "settles the shared query span once per call" in notes
    assert "query_max_tokens 512 declares exactly it" in notes
    assert "document_max_tokens 2048 declares exactly it" in notes
    assert "no per-document cap beside the pair budget" not in notes
    # The served client's wire, as merged: no per-text request caps are ever sent (the stale claim
    # that it sends max_tokens_per_query=4096 and truncate_prompt_tokens=8192 is gone).
    assert "never sends the engine's per-text request caps" in notes
    assert "max_tokens_per_query=4096" not in notes and "truncate_prompt_tokens=8192" not in notes
