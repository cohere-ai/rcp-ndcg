"""The qwen3-vl-embedding-2b recipe: it validates and pins its declared contract, stage 1 passes on CPU.

The CPU checks run against the REAL tokenizer of the pinned revision (tokenizer.json only, fetched
through the shared :func:`._served.fetch_tokenizer` into ``RCP_NDCG_VLLM_TOKENIZER_CACHE`` (or
``tmp_path``) and hash-pinned, so a changed file fails here); offline runs skip with a clear reason
(the conftest's network gate: every test here needs ``RCP_NDCG_NETWORK_TESTS=1``). The contract test
pins EVERY field of the resolved ``serve``, ``client`` and ``reference`` blocks through the shared
:func:`._contract.assert_recipe_contract`, and two drift mutants are shown red. The stage-1 run
exercises the harness's own checks (fit renders, the anchor audit, the served-template render, the
engine /tokenize against the stub engine carrying the same tokenizer) plus the reference subprocess's
render mode, and adds the comparisons the harness defers to the GPU wave: the over-cap cut is
byte-identical with the product's fit, and the query text rides the declared document shape
byte-identically (the card encodes both sides with the same frame).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
from rcp_ndcg_vllm import load_recipe, serve_argv
from rcp_ndcg_vllm.equivalence import stage1_prompts
from rcp_ndcg_vllm.equivalence.reference import run_reference

from tests.conftest import start_stub

from ._contract import assert_recipe_contract
from ._served import fetch_tokenizer, stage1_facts

RECIPE_DIR = Path(__file__).resolve().parents[2] / "recipes" / "qwen3-vl-embedding-2b"
REVISION = "9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda"
MODEL = "Qwen/Qwen3-VL-Embedding-2B"
TOKENIZER_SHA256 = "def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a"
CARD_SHA256 = "8ffa74a1a6bb759610c57865ea416fd4daf9936cb787520e1112a3e1d547f36a"
DEFAULT_INSTRUCTION = "Represent the user's input."
OVER_LENGTH_PER_SHAPE = 5

#: The resolved blocks the contract pins (the product's ``model_dump(mode="json")`` shape): every
#: field of ``serve``, ``client`` (minus the runtime ``base_url``) and ``reference``, defaults
#: included, so a schema default that moves reds here and is re-pinned deliberately.
SERVE = {
    "runner": "pooling",
    "convert": "embed",
    "hf_overrides": {},
    "chat_template": "template.jinja",
    "pooler_config": {"seq_pooling_type": "LAST"},
    "trust_remote_code": False,
    "max_model_len": 8192,
    "dtype": "bfloat16",
    "plugin": None,
    "io_processor_plugin": None,
    "mm_processor_kwargs": {"images_kwargs": {"min_pixels": 4096, "max_pixels": 1843200}},
    "limit_mm_per_prompt": {"image": 1, "video": 1},
    "extra_args": ["--media-io-kwargs", '{"video": {"num_frames": 64}}'],
}
CLIENT = {
    "api": "openai_embeddings",
    "model": "qwen3-vl-embedding-2b",
    "revision": REVISION,
    "api_key_env": None,
    "headers_env": {},
    "concurrency": 64,
    "timeout_s": 600.0,
    "connect_timeout_s": 5.0,
    "max_retries": 2,
    "wait_on_outage_s": None,
    "image_processor": None,
    "image_policy": None,
    "video_policy": {
        "num_frames": 64,
        "wire": "video_url",
        "engine_video_pinning": True,
        "max_duration_s": None,
    },
    "max_images": 1,
    "max_videos": 1,
    "media_sides": ["query", "document"],
    "recipe": (
        "vLLM 0.31.0 pooling runner (--convert embed), seq_pooling_type LAST with the default "
        "PoolerNormalize head; media: nested images_kwargs min_pixels=4096 max_pixels=1843200 "
        "(serve.mm_processor_kwargs, the R20 one shape), one media item per request "
        "(serve.limit_mm_per_prompt image=1 video=1 = client.max_images/max_videos 1/1), "
        "video_policy 64 uniform frames per clip as video_url with --media-io-kwargs video "
        "num_frames 64 pinned (engine_video_pinning); request_shape text: fit's rendered frame "
        "goes out as the input string and the engine's post-processor appends the end anchor"
    ),
    "tokenizer": f"{MODEL}@{REVISION}",
    "max_tokens": 8192,
    "query_max_tokens": None,
    "template": {
        "query": None,
        "document": [
            {
                "fixed": (
                    "{special:im_start}system\nRepresent the user's input.{special:im_end}\n{special:im_start}user\n"
                ),
                "content": None,
            },
            {"fixed": None, "content": "document"},
            {"fixed": "{special:im_end}\n{special:im_start}assistant\n", "content": None},
        ],
        "pair": None,
        "anchor": "last",
        "anchor_markers": [],
        "add_special_tokens": True,
        "normalize": [],
    },
    "on_overflow": "cut",
    "chunk": None,
    "aggregation": "max",
    "empty_doc": "send_text",
    "empty_doc_text": "NULL",
    "request_shape": "text",
    "query_prompt": "",
    "doc_prompt": "",
    "normalize": True,
    "dimensions": None,
    "batch_size": 32,
}
REFERENCE = {
    "kind": "transformers",
    "score_scale": "cosine",
    "entry": "reference.py",
    "known_deviations": ["anchor_drop_over_cap"],
}
TOP = {
    "id": "qwen3-vl-embedding-2b",
    "model": MODEL,
    "revision": REVISION,
    "role": "embed",
    "input": ["text", "image", "video"],
    "licence": "apache-2.0",
}

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
    {"query": "what is a semaphore", "documents": ["A semaphore caps how many workers may hold a resource at once."]},
]


def _pairs(path: Path, rows: list[dict[str, Any]]) -> Path:
    path = path / "pairs.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The pinned revision's tokenizer.json, downloaded once for the module.

    Through the shared :func:`._served.fetch_tokenizer`: the file lands in
    ``$RCP_NDCG_VLLM_TOKENIZER_CACHE`` when set (the lane's scratch) else a pytest-managed
    directory, is hash-pinned, and an unreachable Hub skips with the reason.
    """
    url = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/tokenizer.json"
    return fetch_tokenizer(
        url,
        "qwen3-vl-embedding-2b/tokenizer.json",
        tmp_path_factory.mktemp("qwen3-vl-tokenizer"),
        sha256=TOKENIZER_SHA256,
    )


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


def test_recipe_contract_pins_every_field() -> None:
    """Every field of the resolved serve/client/reference blocks, plus the top-level facts, pinned exactly
    (the shared helper is exact in both directions: a drifted value and an unpinned field both fail)."""
    recipe = load_recipe(RECIPE_DIR)
    assert_recipe_contract(recipe, serve=SERVE, client=CLIENT, reference=REFERENCE, top=TOP)
    client = recipe.client
    template = client.template
    assert template is not None and template.shapes() == ("document",)
    assert template.adds_special_tokens("document") is True
    head = template.segments("document")[0]
    assert head.fixed is not None and DEFAULT_INSTRUCTION in head.fixed, "the pinned default instruction"
    assert (RECIPE_DIR / recipe.serve.chat_template).is_file()


def test_two_contract_mutants_are_red() -> None:
    """A drifted serve field and a drifted reference field each red the contract pin, naming the field
    (the sweep's finding-9 mutants: serve.max_model_len and reference.kind)."""
    recipe = load_recipe(RECIPE_DIR)
    serve_mutant = recipe.model_copy(update={"serve": recipe.serve.model_copy(update={"max_model_len": 16384})})
    with pytest.raises(AssertionError, match=r"serve\.max_model_len"):
        assert_recipe_contract(serve_mutant, serve=SERVE, client=CLIENT, reference=REFERENCE, top=TOP)
    reference_mutant = recipe.model_copy(
        update={"reference": recipe.reference.model_copy(update={"kind": "sentence_transformers"})}
    )
    with pytest.raises(AssertionError, match=r"reference\.kind"):
        assert_recipe_contract(reference_mutant, serve=SERVE, client=CLIENT, reference=REFERENCE, top=TOP)


def test_serve_argv_carries_the_pinned_flags() -> None:
    """The argv the wave runner renders: the nested images_kwargs pin, the pooler, the media limit and
    the video policy's --media-io-kwargs frame count."""
    recipe = load_recipe(RECIPE_DIR)
    argv = serve_argv(recipe, port=8100, served_model_name=recipe.id)
    assert "--chat-template" in argv
    assert argv[argv.index("--chat-template") + 1] == str(RECIPE_DIR / "template.jinja")
    assert json.loads(argv[argv.index("--pooler-config") + 1]) == {"seq_pooling_type": "LAST"}
    assert json.loads(argv[argv.index("--mm-processor-kwargs") + 1]) == {
        "images_kwargs": {"min_pixels": 4096, "max_pixels": 1843200}
    }
    assert json.loads(argv[argv.index("--limit-mm-per-prompt") + 1]) == {"image": 1, "video": 1}
    assert argv[argv.index("--media-io-kwargs") + 1] == '{"video": {"num_frames": 64}}'


def test_card_script_is_vendored_verbatim() -> None:
    """The vendored card script is byte-identical to the pinned revision's published script."""
    digest = hashlib.sha256((RECIPE_DIR / "qwen3_vl_embedding.py").read_bytes()).hexdigest()
    assert digest == CARD_SHA256


@pytest.mark.network
def test_stage1_on_cpu(recipe_cpu: Any, tokenizer: Path, tmp_path: Path) -> None:
    """Stage 1 with the real tokenizer: fit's renders, the anchor audit (21 sampled rows, 5 over cap),
    the reference render, the template file and the stub engine's /tokenize all agree."""
    recipe = recipe_cpu
    engine = start_stub("--tokenizer", str(tokenizer), "--max-model-len", "8192")
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
    # The cut facts come from the role client's own capture and census (R30: what the client sends).
    from rcp_ndcg_vllm.equivalence import fitting

    facts = stage1_facts(recipe, _PAIRS, fitting.tokenizer_of(recipe), OVER_LENGTH_PER_SHAPE)
    body = facts["per_shape"]["document"]
    assert len(body["texts"]) == len(_PAIRS) + OVER_LENGTH_PER_SHAPE
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

    from rcp_ndcg.data.preprocess import TextBudget, fit

    recipe = recipe_cpu
    tokenizer = fitting.tokenizer_of(recipe)
    budget = TextBudget(
        tokenizer=tokenizer.name,
        max_tokens=recipe.client.max_tokens,
        query_max_tokens=recipe.client.query_max_tokens,
        template=recipe.client.template,
        on_overflow=recipe.client.on_overflow,
        chunk=recipe.client.chunk,
        aggregation=recipe.client.aggregation,
    )
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
    """A declared anchor edge the rendered ids never carry turns the anchor check red, naming it.

    Dropping the trailing segment alone cannot red THIS recipe's audit: the anchor of ``anchor: last``
    with ``add_special_tokens: true`` is the post-processor's endoftext, which every render still
    carries (the audit computes its expected edge from the declaration). The mutation that breaks the
    anchor contract is a declared marker anchor whose marker is absent from every render -- the audit
    then reports the missing marker names and fails.
    """
    mutated_dir = tmp_path / "qwen3-vl-embedding-2b"  # the id must equal the directory name
    shutil.copytree(Path(str(recipe_cpu._dir)), mutated_dir)
    data = yaml.safe_load((mutated_dir / "recipe.yaml").read_text(encoding="utf-8"))
    template = data["client"]["template"]
    assert template["anchor"] == "last"
    template["anchor"] = "marker"
    template["anchor_markers"] = ["vision_start"]  # a real special this frame never contains
    (mutated_dir / "recipe.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    mutated = load_recipe(mutated_dir)

    rows = [_PAIRS[0]]
    control = stage1_prompts(recipe_cpu, _pairs(tmp_path, rows), None, over_length_per_shape=1)
    document = stage1_prompts(mutated, _pairs(tmp_path, rows), None, over_length_per_shape=1)
    assert control["anchor_check"]["passed"] is True
    assert document["anchor_check"]["passed"] is False, document["anchor_check"]
    failures = document["anchor_check"]["failures"]
    assert failures and failures[0]["check"] == "markers", "the red check names the missing markers"
    assert "vision_start" in failures[0]["missing_names"]
