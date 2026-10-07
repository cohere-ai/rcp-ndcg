"""The qwen3-vl-embedding-2b recipe: it validates and pins its declared contract, stage 1 passes on CPU.

The CPU checks run against the REAL tokenizer of the pinned revision (tokenizer.json, and the checkpoint's
chat_template.jinja, fetched through the shared :func:`._served.fetch_tokenizer` into
``RCP_NDCG_VLLM_TOKENIZER_CACHE`` (or ``tmp_path``) and hash-pinned, so a changed file fails here); offline
runs skip with a clear reason (the conftest's network gate: every test here needs
``RCP_NDCG_NETWORK_TESTS=1``). The contract test pins EVERY field of the resolved ``serve``, ``client`` and
``reference`` blocks through the shared :func:`._contract.assert_recipe_contract`, and two drift mutants
are shown red. The stage-1 run exercises the harness's own checks (fit renders, the anchor audit, the
engine /tokenize against the stub engine carrying the same tokenizer) plus the reference subprocess's
render mode. What the client ships is read from the product's own role client (``_served``), never
re-derived. The tests add what the harness cannot check for this recipe: both declared shapes render the
checkpoint's own chat template, the reference renders the card's own over-cap truncation (the declared
``anchor_drop_over_cap``: the client keeps the frame, the card does not), and an image item is refused on
the declared text route.
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
from ._served import fetch_tokenizer, served_texts, stage1_facts

RECIPE_DIR = Path(__file__).resolve().parents[2] / "recipes" / "qwen3-vl-embedding-2b"
REVISION = "9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda"
MODEL = "Qwen/Qwen3-VL-Embedding-2B"
TOKENIZER_SHA256 = "def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a"
CARD_SHA256 = "8ffa74a1a6bb759610c57865ea416fd4daf9936cb787520e1112a3e1d547f36a"
CHAT_TEMPLATE_SHA256 = "a47e6afb389f86f45be7810f17d2686fd42b2bec7ba6e6958abf85845af258c5"
DEFAULT_INSTRUCTION = "Represent the user's input."
OVER_LENGTH_PER_SHAPE = 5
#: The declared frame of both shapes (the card frames a query exactly like a document); the instruction
#: closes its own segment, a whole delimited unit of the query frame (the case loader's rule).
_SYSTEM = "{special:im_start}system\nRepresent the user's input."
_USER = "{special:im_end}\n{special:im_start}user\n"
_TAIL = "{special:im_end}\n{special:im_start}assistant\n"

#: The resolved blocks the contract pins (the product's ``model_dump(mode="json")`` shape): every
#: field of ``serve``, ``client`` (minus the runtime ``base_url``) and ``reference``, defaults
#: included, so a schema default that moves reds here and is re-pinned deliberately.
SERVE = {
    "runner": "pooling",
    "convert": "embed",
    "hf_overrides": {},
    "chat_template": None,
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
    "image_policy": {"min_px": 4096, "max_px": 1843200, "processor": None},
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
        "PoolerNormalize head; no served chat template (completion-style text route, the frame is the "
        "client's); media: nested images_kwargs min_pixels=4096 max_pixels=1843200 "
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
        "query": [
            {"fixed": _SYSTEM, "content": None},
            {"fixed": _USER, "content": None},
            {"fixed": None, "content": "query"},
            {"fixed": _TAIL, "content": None},
        ],
        "document": [
            {"fixed": _SYSTEM, "content": None},
            {"fixed": _USER, "content": None},
            {"fixed": None, "content": "document"},
            {"fixed": _TAIL, "content": None},
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


def _reference_module() -> Any:
    """The recipe's reference.py as a module (its top level imports only the standard library)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("qwen3_vl_embedding_reference", RECIPE_DIR / "reference.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
def chat_template(tmp_path_factory: pytest.TempPathFactory) -> str:
    """The checkpoint's own chat_template.jinja at the pinned revision (hash-pinned, shared cache)."""
    url = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/chat_template.jinja"
    path = fetch_tokenizer(
        url,
        "qwen3-vl-embedding-2b/chat_template.jinja",
        tmp_path_factory.mktemp("qwen3-vl-template"),
        sha256=CHAT_TEMPLATE_SHA256,
    )
    return path.read_text(encoding="utf-8")


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


def _reference_render(recipe: Any, rows: list[dict[str, Any]], work: Path) -> dict[tuple[int, str], str]:
    """The reference subprocess's render mode over ``rows``, keyed by ``(row index, shape)`` (its own
    python via run_reference; the render mode imports no torch, so the harness's interpreter drives it)."""
    out = work / "reference-render.json"
    run_reference(
        sys.executable,
        str(Path(str(recipe._dir)) / recipe.reference.entry),
        mode="render",
        pairs_path=_pairs(work, rows),
        out_path=out,
        tokenizer_spec=str(recipe.client.tokenizer),
    )
    rendered = json.loads(out.read_text(encoding="utf-8"))["rows"]
    return {(int(row["index"]), str(row["shape"])): str(row["text"]) for row in rendered}


def test_recipe_contract_pins_every_field() -> None:
    """Every field of the resolved serve/client/reference blocks, plus the top-level facts, pinned exactly
    (the shared helper is exact in both directions: a drifted value and an unpinned field both fail)."""
    recipe = load_recipe(RECIPE_DIR)
    assert_recipe_contract(recipe, serve=SERVE, client=CLIENT, reference=REFERENCE, top=TOP)
    template = recipe.client.template
    assert template is not None and template.shapes() == ("query", "document")
    for shape in ("query", "document"):
        assert template.adds_special_tokens(shape) is True
        head = template.segments(shape)[0]
        assert head.fixed is not None and DEFAULT_INSTRUCTION in head.fixed, "the card's default instruction"
    assert not (RECIPE_DIR / "template.jinja").exists(), "no template file ships (serve.chat_template null)"


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
    """The argv the wave runner renders: no template file, the nested images_kwargs pin, the pooler, the
    media limit and the video policy's --media-io-kwargs frame count."""
    recipe = load_recipe(RECIPE_DIR)
    argv = serve_argv(recipe, port=8100, served_model_name=recipe.id)
    assert "--chat-template" not in argv
    assert "--trust-remote-code" not in argv
    assert json.loads(argv[argv.index("--pooler-config") + 1]) == {"seq_pooling_type": "LAST"}
    assert json.loads(argv[argv.index("--mm-processor-kwargs") + 1]) == {
        "images_kwargs": {"min_pixels": 4096, "max_pixels": 1843200}
    }
    assert json.loads(argv[argv.index("--limit-mm-per-prompt") + 1]) == {"image": 1, "video": 1}
    assert argv[argv.index("--media-io-kwargs") + 1] == '{"video": {"num_frames": 64}}'


def test_card_script_is_vendored_verbatim_and_its_constants_bind() -> None:
    """The vendored card script is byte-identical to the pinned revision's published script, and the
    constants the reference reads from it are the recipe's own budget and frame instruction."""
    digest = hashlib.sha256((RECIPE_DIR / "qwen3_vl_embedding.py").read_bytes()).hexdigest()
    assert digest == CARD_SHA256
    constants = _reference_module().card_constants()
    assert constants == {"max_length": 8192, "default_instruction": DEFAULT_INSTRUCTION}
    recipe = load_recipe(RECIPE_DIR)
    assert recipe.client.max_tokens == constants["max_length"] == recipe.serve.max_model_len


@pytest.mark.network
def test_both_declared_shapes_render_the_checkpoint_chat_template(
    recipe_cpu: Any, chat_template: str, tokenizer: Path
) -> None:
    """The frame both shapes declare is the checkpoint's own: its chat_template.jinja (hash-pinned) renders
    the card's conversation -- the default instruction as the system turn, the text as the user turn, the
    generation prompt -- to exactly the declared render, for a query and a document alike; and so does the
    reference's prompt builder. (The harness's template check renders one string per row and cannot tell
    the two shapes apart, so this test stands in for it.)"""
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    from rcp_ndcg_vllm.equivalence import fitting

    environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    template = environment.from_string(chat_template)
    product_tokenizer = fitting.tokenizer_of(recipe_cpu)
    reference = _reference_module()
    for shape, text in (("query", "what is the capital of France"), ("document", "Paris is the capital.")):
        messages = [
            {"role": "system", "content": [{"type": "text", "text": DEFAULT_INSTRUCTION}]},
            {"role": "user", "content": [{"type": "text", "text": text}]},
        ]
        card = template.render(messages=messages, add_generation_prompt=True)
        declared = recipe_cpu.client.template.render(shape, product_tokenizer, query=text, document=text)
        assert declared == card, shape
        assert reference.card_prompt(text, DEFAULT_INSTRUCTION) == card, shape


@pytest.mark.network
def test_stage1_on_cpu(recipe_cpu: Any, tokenizer: Path, tmp_path: Path) -> None:
    """Stage 1 with the real tokenizer: fit's renders for both shapes, the anchor audit (5 over-length
    samples per shape), the reference render and the stub engine's /tokenize all agree."""
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
    assert document["anchor_check"]["checked"] >= 2 * (len(_PAIRS) + OVER_LENGTH_PER_SHAPE)
    # The cut facts come from the role client's own capture and census (R30: what the client sends).
    from rcp_ndcg_vllm.equivalence import fitting

    facts = stage1_facts(recipe, _PAIRS, fitting.tokenizer_of(recipe), OVER_LENGTH_PER_SHAPE)
    for shape in ("query", "document"):
        body = facts["per_shape"][shape]
        assert len(body["texts"]) >= len(_PAIRS) + OVER_LENGTH_PER_SHAPE, shape
        assert body["cut_rows"] >= OVER_LENGTH_PER_SHAPE, f"{shape}: the over-length rows are cut"
    render = document["render_check"]
    assert render["status"] == "run" and render["passed"] is True, render["failures"][:1]
    assert render["rows"] == 2 * len(_PAIRS), "one render per declared shape per pairs row"
    assert document["template_render_check"] is None, "no served template file (see the chat-template test)"
    engine_check = document["engine_tokenize_check"]
    assert engine_check["status"] == "run" and engine_check["passed"] is True, engine_check["failures"][:1]
    assert document["passed"] is True


@pytest.mark.network
def test_reference_renders_the_card_truncation_not_the_client_cut(recipe_cpu: Any, tmp_path: Path) -> None:
    """Decision 9 on this recipe: the reference renders the card's prompt and the card's own over-cap cut.

    Under the cap the reference's render is byte-identical with what the role client ships, on both
    shapes. Over the cap they differ exactly as ``anchor_drop_over_cap`` declares: the card's
    ``truncation=True`` right-cuts the whole prompt (8191 prompt tokens + the re-appended endoftext), so
    its render loses the frame's tail, while the client keeps the whole frame and cuts the content only.
    """
    from rcp_ndcg_vllm.equivalence import fitting

    recipe = recipe_cpu
    tokenizer = fitting.tokenizer_of(recipe)
    long_text = "Island biogeography studies the species richness of isolated habitats. " * 900
    rows = [
        {"query": "what is the capital of France", "documents": ["Paris is the capital of France."]},
        {"query": "a short one", "documents": ["  padded with spaces  "]},
        {"query": long_text, "documents": [long_text]},
    ]
    reference = _reference_render(recipe, rows, tmp_path)
    tail = "<|im_end|>\n<|im_start|>assistant\n"
    endoftext = tokenizer.special_id("endoftext")
    for shape in ("query", "document"):
        texts = [str(row["query"]) if shape == "query" else str(row["documents"][0]) for row in rows]
        shipped = served_texts(recipe, texts, shape)
        for index in (0, 1):
            assert reference[(index, shape)] == shipped[index], f"row {index} {shape}: under cap, byte-identical"
        over_reference, over_client = reference[(2, shape)], shipped[2]
        reference_ids = tokenizer.ids(over_reference, add_special_tokens=True)
        client_ids = tokenizer.ids(over_client, add_special_tokens=True)
        assert len(reference_ids) == recipe.client.max_tokens, "the card fills its cap exactly"
        assert len(client_ids) <= recipe.client.max_tokens
        assert reference_ids[-1] == client_ids[-1] == endoftext
        assert over_client.endswith(tail), "the client keeps the frame's tail"
        assert not over_reference.endswith(tail), "the card's right cut drops it (anchor_drop_over_cap)"
        assert over_reference.startswith(over_client[: over_client.index(long_text[:40])])


@pytest.mark.network
@pytest.mark.xfail(
    strict=True,
    reason="product gap: the embed client applies empty_doc after fit has rendered the frame "
    "(rcp_ndcg.inference.clients.embed._prepare -> _apply_empty_documents sees the rendered, non-empty "
    "text), so a templated role's empty_doc: send_text never fires and an empty document ships as an "
    "empty user turn. Strict: this goes red the day the product applies the policy before the render.",
)
def test_an_empty_document_renders_the_card_null(recipe_cpu: Any, tmp_path: Path) -> None:
    """The card renders an empty input as the literal NULL; the declared empty_doc: send_text "NULL" is the
    recipe's way to ship the same prompt."""
    reference = _reference_render(recipe_cpu, [{"query": "q", "documents": [""]}], tmp_path)
    assert "user\nNULL<|im_end|>" in reference[(0, "document")]
    assert served_texts(recipe_cpu, [""], "document") == [reference[(0, "document")]]


@pytest.mark.network
def test_an_image_item_is_refused_on_the_declared_text_route(recipe_cpu: Any, tmp_path: Path) -> None:
    """The recipe declares media capacity, but its text route carries no media parts: the product refuses
    an image item loudly (a typed CapabilityError naming the route), before anything is fetched."""
    from rcp_ndcg_core.content import Content
    from rcp_ndcg_vllm.equivalence.wire import role_client

    from rcp_ndcg.errors import CapabilityError
    from rcp_ndcg.inference.types import EncodeRole

    client, capture = role_client(recipe_cpu, None)
    with pytest.raises(CapabilityError, match="takes text only"):
        client.encode([Content.from_image(f"file://{tmp_path / 'never-read.png'}")], EncodeRole.DOCUMENT)
    assert not capture.exchanges, "nothing was sent"


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
