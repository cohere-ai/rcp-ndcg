"""The qwen3-vl-embedding-2b recipe: the recipe validates, stage 1 passes on CPU, the anchor mutation is red.

The CPU checks run against the REAL tokenizer of the pinned revision (tokenizer.json only, fetched from
the Hub into a pytest-managed temporary directory and hash-pinned, so a changed file fails here); offline
runs skip with a clear reason. The stage-1 run exercises the harness's own checks (fit renders, the
anchor audit, the served-template render, the engine /tokenize against the stub engine carrying the same
tokenizer) plus the reference subprocess's render mode, and adds the comparisons the harness defers to
the GPU wave: the over-cap cut is byte-identical with the product's fit, and the query text rides the
declared document shape byte-identically (the card encodes both sides with the same frame).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Any

import pytest
import yaml
from rcp_ndcg_vllm import load_recipe, serve_argv
from rcp_ndcg_vllm.equivalence import stage1_prompts
from rcp_ndcg_vllm.equivalence.reference import run_reference

from tests.conftest import start_stub

RECIPE_DIR = Path(__file__).resolve().parents[2] / "recipes" / "qwen3-vl-embedding-2b"
REVISION = "9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda"
MODEL = "Qwen/Qwen3-VL-Embedding-2B"
TOKENIZER_SHA256 = "def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a"
CARD_SHA256 = "8ffa74a1a6bb759610c57865ea416fd4daf9936cb787520e1112a3e1d547f36a"
DEFAULT_INSTRUCTION = "Represent the user's input."
OVER_LENGTH_PER_SHAPE = 5

_PAIRS: list[dict[str, Any]] = [
    {"query": "what is the capital of France", "documents": ["Paris is the capital of France."]},
    {
        "query": "how do neural networks learn",
        "documents": ["Backpropagation adjusts weights by following the gradient of a loss."],
    },
    {
        "query": "define photosynthesis",
        "documents": ["Photosynthesis turns carbon dioxide and water into glucose with light."],
    },
    {"query": "who wrote Pride and Prejudice", "documents": ["Pride and Prejudice was published in 1813."]},
    {
        "query": "speed of light in vacuum",
        "documents": ["Light travels at roughly 299,792 kilometres per second in vacuum."],
    },
    {"query": "what causes the seasons", "documents": ["The seasons follow the tilt of the Earth's rotation axis."]},
    {
        "query": "boiling point of water",
        "documents": ["Water boils at one hundred degrees Celsius at standard pressure."],
    },
    {
        "query": "largest planet in the solar system",
        "documents": ["Jupiter is the largest planet in the solar system."],
    },
    {
        "query": "what is a cache hit",
        "documents": ["A cache hit serves a request from the fast store, not the slow one."],
    },
    {
        "query": "history of the printing press",
        "documents": ["Movable type spread across Europe after the fifteenth century."],
    },
    {
        "query": "definition of an algorithm",
        "documents": ["An algorithm is a finite sequence of steps that solves a problem."],
    },
    {
        "query": "why is the sky blue",
        "documents": ["Short wavelengths scatter more strongly, which tints the daytime sky."],
    },
    {"query": "first crewed lunar landing", "documents": ["The first crewed lunar landing happened in July 1969."]},
    {"query": "what does DNA encode", "documents": ["DNA encodes proteins as sequences of nucleotide bases."]},
    {
        "query": "how do vaccines work",
        "documents": ["A vaccine trains the immune system to recognise a pathogen."],
    },
    {
        "query": "what is a semaphore",
        "documents": ["A semaphore caps how many workers may hold a resource at once."],
    },
]


def _pairs(path: Path, rows: list[dict[str, Any]]) -> Path:
    path = path / "pairs.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _download_tokenizer(target_dir: Path) -> Path:
    """The pinned revision's tokenizer.json, fetched into a pytest-managed directory and hash-pinned;
    offline (or any fetch failure) skips with a clear reason instead of failing the suite.

    The recipe brief mandates this semantics - run when online, skip with a clear reason when offline -
    the inverse of the root suite's RCP_NDCG_NETWORK_TESTS gate: stage 1 on the pinned tokenizer is the
    test's point, so it must run by default in CI's networked vllm-recipes job and skip, never fail,
    offline. The @pytest.mark.network marks name the network dependency in the root convention's
    vocabulary; the offline skip is the guard."""
    target = target_dir / "tokenizer.json"
    if target.is_file():
        return target
    url = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/tokenizer.json"
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            data = response.read()
    except OSError as error:
        pytest.skip(f"offline: cannot fetch the pinned tokenizer.json from the Hub ({error})")
    digest = hashlib.sha256(data).hexdigest()
    assert digest == TOKENIZER_SHA256, f"the downloaded tokenizer.json is not the pinned revision's: {digest}"
    target.write_bytes(data)
    return target


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The pinned tokenizer.json, downloaded once for the module."""
    return _download_tokenizer(tmp_path_factory.mktemp("qwen3-vl-tokenizer"))


@pytest.fixture(scope="module")
def recipe_cpu(tmp_path_factory: pytest.TempPathFactory, tokenizer: Path) -> Any:
    """The shipped recipe, loaded from a pytest-managed copy whose client.tokenizer names the downloaded
    tokenizer file (the recipe itself pins the Hub repository id and revision; the bytes are hash-equal)."""
    target = tmp_path_factory.mktemp("qwen3-vl-recipe") / "qwen3-vl-embedding-2b"
    shutil.copytree(RECIPE_DIR, target)
    data = yaml.safe_load((target / "recipe.yaml").read_text(encoding="utf-8"))
    data["client"]["tokenizer"] = str(tokenizer)
    (target / "recipe.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return load_recipe(target)


def _reference_render(recipe: Any, rows: list[dict[str, Any]], work: Path) -> list[dict[str, Any]]:
    """The reference subprocess's render mode over ``rows`` (its own python via run_reference; the
    render mode imports no torch, so the harness's interpreter drives it here)."""
    out = work / "reference-render.json"
    run_reference(
        sys.executable,
        str(Path(str(recipe._dir)) / recipe.reference.entry),
        mode="render",
        pairs_path=_pairs(work, rows),
        out_path=out,
        tokenizer_spec=str(recipe.client.tokenizer),
    )
    return json.loads(out.read_text(encoding="utf-8"))["rows"]


def test_recipe_validates() -> None:
    """The shipped recipe loads through the product's endpoint config and the harness's closed schema."""
    recipe = load_recipe(RECIPE_DIR)
    assert recipe.id == "qwen3-vl-embedding-2b"
    assert recipe.model == MODEL
    assert recipe.revision == REVISION
    assert recipe.role == "embed" and recipe.input == ["text", "image", "video"]
    client = recipe.client
    assert type(client).__name__ == "EmbeddingEndpoint"
    assert client.api == "openai_embeddings" and client.request_shape == "text"
    assert client.tokenizer == f"{MODEL}@{REVISION}"
    assert client.max_tokens == 8192 == recipe.serve.max_model_len
    assert client.on_overflow == "cut"
    assert (client.empty_doc, client.empty_doc_text) == ("send_text", "NULL")
    assert client.normalize is True
    template = client.template
    assert template is not None and template.shapes() == ("document",)
    assert template.anchor == "last"
    assert template.adds_special_tokens("document") is True
    head = template.segments("document")[0]
    assert head.fixed is not None and DEFAULT_INSTRUCTION in head.fixed, "the pinned default instruction"
    assert [segment.content for segment in template.segments("document")] == [None, "document", None]
    assert template.segments("document")[-1].fixed is not None, "the anchor's trailing fixed segment"
    assert recipe.serve.chat_template == "template.jinja"
    assert (RECIPE_DIR / recipe.serve.chat_template).is_file()
    assert recipe.serve.mm_processor_kwargs == {"images_kwargs": {"min_pixels": 4096, "max_pixels": 1843200}}
    assert recipe.serve.limit_mm_per_prompt == {"image": 1, "video": 1}
    assert recipe.serve.convert == "embed" and recipe.serve.runner == "pooling"
    assert recipe.serve.pooler_config == {"seq_pooling_type": "LAST"}
    assert recipe.serve.trust_remote_code is False
    assert recipe.reference.known_deviations == ["anchor_drop_over_cap"]
    assert recipe.status.state == "unverified"
    argv = serve_argv(recipe, port=8100, served_model_name=recipe.id)
    assert "--chat-template" in argv and "--mm-processor-kwargs" in argv
    assert "--pooler-config" in argv
    assert json.loads(argv[argv.index("--pooler-config") + 1]) == {"seq_pooling_type": "LAST"}


def test_card_script_is_vendored_verbatim() -> None:
    """The vendored card script is byte-identical to the pinned revision's published script."""
    digest = hashlib.sha256((RECIPE_DIR / "qwen3_vl_embedding.py").read_bytes()).hexdigest()
    assert digest == CARD_SHA256


@pytest.mark.network
def test_stage1_on_cpu(recipe_cpu: Any, tokenizer: Path, tmp_path: Path) -> None:
    """Stage 1 with the real tokenizer: fit's renders, the anchor audit (21 sampled rows, 5 over cap),
    the reference render, the template file and the stub engine's /tokenize all agree."""
    recipe = recipe_cpu
    engine = start_stub("--tokenizer", str(tokenizer))
    try:
        document = stage1_prompts(
            recipe,
            _pairs(tmp_path, _PAIRS),
            sys.executable,
            base_url=engine.base_url,
            over_length_per_shape=OVER_LENGTH_PER_SHAPE,
        )
    finally:
        engine.stop()
    assert document["anchor_check"]["passed"] is True, document["anchor_check"]["failures"][:1]
    assert document["anchor_check"]["checked"] >= 20
    body = document["fit"]["document"]
    assert body["n_texts"] == len(_PAIRS) + OVER_LENGTH_PER_SHAPE
    assert body["cuts"] >= OVER_LENGTH_PER_SHAPE, "the over-length rows are cut, the pairs rows are not"
    render = document["render_check"]
    assert render["status"] == "run" and render["passed"] is True, render["failures"][:1]
    template_check = document["template_render_check"]
    assert template_check["passed"] is True, template_check["failures"][:1]
    engine_check = document["engine_tokenize_check"]
    assert engine_check["status"] == "run" and engine_check["passed"] is True, engine_check["failures"][:1]
    assert document["passed"] is True


@pytest.mark.network
def test_reference_cut_matches_fit_under_and_over_cap(recipe_cpu: Any, tmp_path: Path) -> None:
    """The reference's anchor-preserving render is byte-identical with the product's fit, under the cap
    (no cut) and over it (the content cut, the frame and the anchor re-attached) - including a query text
    riding the declared document shape, which is the query side's served prompt."""
    from rcp_ndcg_vllm.equivalence import fitting

    from rcp_ndcg.data.preprocess import fit

    recipe = recipe_cpu
    tokenizer = fitting.tokenizer_of(recipe)
    budget = fitting.budget_of(recipe).model_copy(update={"tokenizer": tokenizer.name})
    long_document = "Island biogeography studies the species richness of isolated habitats. " * 60
    over_cap_query = "cache invalidation strategies for read-mostly workloads part " * 900
    rows = [
        {
            "query": "what is the capital of France",
            "documents": ["Paris is the capital of France."],
        },
        # The query side: the reference renders the same frame for the query text (card semantics).
        {"query": over_cap_query, "documents": [over_cap_query]},
        {
            "query": "a short one",
            "documents": [(long_document + " Shelves creaked under the weight of ages.")],
        },
    ]
    for reference_row in _reference_render(recipe, rows, tmp_path):
        index = int(reference_row["index"])
        content = rows[index]["documents"][0]
        result = fit([content], "document", budget, tokenizer, ids=[str(index)])
        assert reference_row["shape"] == "document"
        assert reference_row["text"] == result.texts[0], f"row {index}: the reference cut differs from fit"
        ids = tokenizer.ids(reference_row["text"], add_special_tokens=True)
        assert len(ids) <= recipe.client.max_tokens
        assert ids[-1] == tokenizer.special_id("endoftext"), "the anchor sits at the tail of every render"


@pytest.mark.network
def test_stage1_anchor_mutation_is_red(recipe_cpu: Any, tmp_path: Path) -> None:
    """Dropping the template's trailing anchor segment turns the anchor check red: the rendered ids no
    longer end with the tail fixed segment plus the post-processor's anchor."""
    mutated_dir = tmp_path / "qwen3-vl-embedding-2b"  # the id must equal the directory name
    shutil.copytree(Path(str(recipe_cpu._dir)), mutated_dir)
    data = yaml.safe_load((mutated_dir / "recipe.yaml").read_text(encoding="utf-8"))
    segments = data["client"]["template"]["document"]
    assert segments[-1]["fixed"] is not None
    data["client"]["template"]["document"] = segments[:-1]  # the trailing assistant header, gone
    (mutated_dir / "recipe.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    mutated = load_recipe(mutated_dir)

    rows = [_PAIRS[0]]
    control = stage1_prompts(recipe_cpu, _pairs(tmp_path, rows), None, over_length_per_shape=1)
    document = stage1_prompts(mutated, _pairs(tmp_path, rows), None, over_length_per_shape=1)
    assert control["anchor_check"]["passed"] is True
    assert document["anchor_check"]["passed"] is False, document["anchor_check"]
    assert document["anchor_check"]["failures"], "the red check names what moved"
