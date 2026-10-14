"""The ``qwen3-vl-embedding`` family: every variant validates, pins its declared contract, and passes
stage 1 on CPU with its own real tokenizer.

One module per family (decision 34), parametrized over the family's variants: the 2b and the 8b. The
CPU checks run against the REAL tokenizer of each pinned revision (tokenizer.json, and the checkpoint's
chat_template.jinja, fetched through the shared :func:`._served.fetch_tokenizer` into
``RCP_NDCG_VLLM_TOKENIZER_CACHE`` (or ``tmp_path``) and hash-pinned, so a changed file fails here);
offline runs skip with a clear reason (the conftest's network gate: every test here needs
``RCP_NDCG_NETWORK_TESTS=1``). The contract test pins EVERY field of the resolved ``serve``, ``client``
and ``reference`` blocks through the shared :func:`._contract.assert_recipe_contract`, and two drift
mutants are shown red. The stage-1 run exercises the harness's own checks (fit renders, the anchor audit,
the engine /tokenize against the stub engine carrying the same tokenizer) plus the reference
subprocess's render mode (which loads the variant's model and revision from ``--recipe``). What the
client ships is read from the product's own role client (``_served``), never re-derived. The tests add
what the harness cannot check for this family: both declared shapes render the checkpoint's own chat
template, the served render is that frame only with the declared generation prompt, the reference
renders the card's own over-cap truncation (the declared ``anchor_drop_over_cap``: the client keeps the
frame, the card does not), and the media stage holds every image of the media request set to the card's
resize -- and, against the stub engine, the engine's media count to the client's, pinned and not.
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
from rcp_ndcg_test.equivalence import stage1_prompts
from rcp_ndcg_test.equivalence.reference import run_reference
from rcp_ndcg_vllm import load_recipe, serve_argv
from rcp_ndcg_vllm.recipe import default_recipes_root

from tests.conftest import start_stub

from ._contract import assert_recipe_contract
from ._served import client_template, fetch_tokenizer, served_texts, stage1_facts

RECIPE_DIR = default_recipes_root() / "qwen3-vl-embedding"
#: The family's variants at their pinned revisions (re-checked against the Hub API; not gated). The
#: tokenizer.json and chat_template.jinja bytes are IDENTICAL at both pins (hash-pinned below), so the
#: same sha256 covers each variant's own download.
VARIANTS: dict[str, dict[str, Any]] = {
    "qwen3-vl-embedding-2b": {
        "model": "Qwen/Qwen3-VL-Embedding-2B",
        "revision": "9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda",
        "mrl_range": [64, 2048],
        # E2 round 7, stage 2: the measured floor over the 282 gated comparisons (94 vectors at full
        # width, k=64 and k=2048) and the measured count below the published 0.999.
        "vec_min_cosine": 0.9985,
        "measured": ("0.998541", "0.998906", "14 of 282"),
    },
    "qwen3-vl-embedding-8b": {
        "model": "Qwen/Qwen3-VL-Embedding-8B",
        "revision": "2c4565515e0f265c6511776e7193b22c0968ddc7",
        "mrl_range": [64, 4096],
        # E2 round 7, stage 2: the same shape at the 8b's MRL widths (full width and k=4096 share the
        # floor; k=64 is the worst k).
        "vec_min_cosine": 0.9983,
        "measured": ("0.998335", "0.998674", "15 of 282"),
    },
}
VARIANT_IDS = sorted(VARIANTS)
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
    "patches": [],
    "runner": "pooling",
    "convert": "embed",
    "hf_overrides": {"is_matryoshka": True},
    "chat_template": None,
    "pooler_config": {"seq_pooling_type": "LAST"},
    "trust_remote_code": False,
    "max_model_len": 8192,
    "dtype": "bfloat16",
    "plugin": None,
    "plugin_architectures": [],
    "io_processor_plugin": None,
    "mm_processor_kwargs": {
        "images_kwargs": {"min_pixels": 4096, "max_pixels": 1843200},
        "videos_kwargs": {"min_pixels": 4096, "max_pixels": 7864320},
    },
    "limit_mm_per_prompt": {"image": 1, "video": 1},
    "extra_args": ["--media-io-kwargs", '{"video": {"fps": 2}}'],
}
CLIENT = {
    "api": "openai_embeddings",
    "instruction": "none",
    "request_shape": "messages",
    "add_generation_prompt": True,
    "max_tokens": 8192,
    "image_processor": "qwen3_vl",
    "image_policy": {"min_px": 4096, "max_px": 1843200, "engine_pixel_pinning": True},
    "max_images": 1,
    "max_videos": 1,
    "video_policy": {
        "fps": 2,
        "wire": "video_url",
        "engine_video_pinning": True,
        "engine_video_min_pixels": 4096,
        "engine_video_max_pixels": 7864320,
    },
    "template": {
        "query": [
            {"fixed": "{special:im_start}system\nRepresent the user's input."},
            {"fixed": "{special:im_end}\n{special:im_start}user\n"},
            {"content": "query"},
            {"fixed": "{special:im_end}\n{special:im_start}assistant\n"},
        ],
        "document": [
            {"fixed": "{special:im_start}system\nRepresent the user's input."},
            {"fixed": "{special:im_end}\n{special:im_start}user\n"},
            {"content": "document"},
            {"fixed": "{special:im_end}\n{special:im_start}assistant\n"},
        ],
        "anchor": "last",
        "add_special_tokens": True,
    },
    "on_overflow": "cut",
    "empty_doc": "send_text",
    "empty_doc_text": "NULL",
    "normalize": True,
    "mrl_kind": "truncation",
}
REFERENCE = {
    "attn_implementation": None,
    "kind": "transformers",
    "score_scale": "cosine",
    "entry": "reference.py",
    "known_deviations": ["anchor_drop_over_cap"],
    "device": "cuda",
}
TOP = {
    "role": "embed",
    "input": ["text", "image", "video"],
    "licence": "apache-2.0",
}


def _expected_top(variant_id: str) -> dict[str, object]:
    facts = VARIANTS[variant_id]
    return {"id": variant_id, "model": facts["model"], "revision": facts["revision"], **TOP}


def _expected_client(variant_id: str) -> dict[str, object]:
    facts = VARIANTS[variant_id]
    return {
        **CLIENT,
        "tokenizer": f"{facts['model']}@{facts['revision']}",
        "mrl_range": facts["mrl_range"],
        "model": variant_id,
        "revision": facts["revision"],
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
    bytecode = sys.dont_write_bytecode  # exec_module must not drop a __pycache__ into the recipe dir
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = bytecode
    return module


def _tokenizer_file(variant_id: str, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The variant's pinned tokenizer.json, downloaded once.

    Through the shared :func:`._served.fetch_tokenizer`: the file lands in
    ``$RCP_NDCG_VLLM_TOKENIZER_CACHE`` when set (the lane's scratch) else a pytest-managed
    directory, is hash-pinned, and an unreachable Hub skips with the reason. The bytes are the same at
    both pins (the fixture's sha256 covers each variant's own download).
    """
    facts = VARIANTS[variant_id]
    url = f"https://huggingface.co/{facts['model']}/resolve/{facts['revision']}/tokenizer.json"
    return fetch_tokenizer(
        url,
        f"{variant_id}/tokenizer.json",
        tmp_path_factory.mktemp(f"{variant_id}-tokenizer"),
        sha256=TOKENIZER_SHA256,
    )


def _chat_template_path(variant_id: str, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The checkpoint's own chat_template.jinja at the variant's pinned revision (hash-pinned)."""
    facts = VARIANTS[variant_id]
    url = f"https://huggingface.co/{facts['model']}/resolve/{facts['revision']}/chat_template.jinja"
    return fetch_tokenizer(
        url,
        f"{variant_id}/chat_template.jinja",
        tmp_path_factory.mktemp(f"{variant_id}-template"),
        sha256=CHAT_TEMPLATE_SHA256,
    )


def _chat_template_text(variant_id: str, tmp_path_factory: pytest.TempPathFactory) -> str:
    """The checkpoint's template as text (the path is the one file, :func:`_chat_template_path`)."""
    return _chat_template_path(variant_id, tmp_path_factory).read_text(encoding="utf-8")


def _recipe_cpu(variant_id: str, tokenizer: Path, tmp_path: Path) -> Any:
    """The variant, loaded from a pytest-managed copy whose client.tokenizer names the downloaded
    tokenizer file (the recipe itself pins the Hub repository id and revision; the bytes are hash-equal)."""
    target = tmp_path / "qwen3-vl-embedding"  # the family directory name
    shutil.copytree(RECIPE_DIR, target)
    data = yaml.safe_load((target / "family.yaml").read_text(encoding="utf-8"))
    data["client"]["tokenizer"] = str(tokenizer)
    (target / "family.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return load_recipe(variant_id, root=tmp_path)


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
        tokenizer_spec=str(recipe.client.get("tokenizer")),
        recipe=recipe,
    )
    rendered = json.loads(out.read_text(encoding="utf-8"))["rows"]
    return {(int(row["index"]), str(row["shape"])): str(row["text"]) for row in rendered}


def _assert_contract(recipe: Any) -> None:
    """The variant's full resolved contract: every serve/client/reference field plus the top-level facts."""
    assert_recipe_contract(
        recipe,
        serve=SERVE,
        client=_expected_client(recipe.id),
        reference=REFERENCE,
        top=_expected_top(recipe.id),
    )


@pytest.mark.parametrize("variant_id", VARIANT_IDS)
def test_recipe_contract_pins_every_field(variant_id: str) -> None:
    """Every field of the resolved serve/client/reference blocks, plus the top-level facts, pinned exactly
    (the shared helper is exact in both directions: a drifted value and an unpinned field both fail)."""
    recipe = load_recipe(variant_id)
    _assert_contract(recipe)
    template = client_template(recipe)
    assert template is not None and template.shapes() == ("query", "document")
    for shape in ("query", "document"):
        assert template.adds_special_tokens(shape) is True
        head = template.segments(shape)[0]
        assert head.fixed is not None and DEFAULT_INSTRUCTION in head.fixed, "the card's default instruction"
    assert recipe.resources.gpus == 1  # bf16 weights + KV fit one 80 GB-class GPU at every size
    assert not (RECIPE_DIR / "template.jinja").exists(), "no template file ships (serve.chat_template null)"


@pytest.mark.parametrize("variant_id", VARIANT_IDS)
def test_the_variant_declares_its_measured_bf16_cosine_floor(variant_id: str) -> None:
    """E2 round 7: both sides are bf16 (no mismatch to fix), so each size declares the tight bound
    covering its measured stage-2 floor, with the per-MRL numbers in its notes.

    The 2b measured min cosine 0.998541 at full width and k=2048 and 0.998906 at k=64, with 14 of the
    282 gated comparisons below the published 0.999; the 8b measured 0.998335 at full width and
    k=4096 and 0.998674 at k=64, with 15 below. The declarations (2b 0.9985, 8b 0.9983) sit just
    below each measured floor, so the gate still catches a real regression; the published 0.999 is
    the bf16 kernel-stack residual, not a defect."""
    recipe = load_recipe(variant_id)
    facts = VARIANTS[variant_id]
    assert recipe.gates.vec_min_cosine == facts["vec_min_cosine"]
    for measured in facts["measured"]:
        assert measured in recipe.notes, measured
    # Each size carries its own bound: no family-wide gate is declared.
    for other in VARIANT_IDS:
        if other != variant_id:
            assert load_recipe(other).gates.vec_min_cosine == VARIANTS[other]["vec_min_cosine"]


@pytest.mark.parametrize("variant_id", VARIANT_IDS)
def test_two_contract_mutants_are_red(variant_id: str) -> None:
    """A drifted serve field and a drifted reference field each red the contract pin, naming the field
    (the contract mutants: serve.max_model_len and reference.kind), per variant."""
    recipe = load_recipe(variant_id)
    serve_mutant = recipe.model_copy(update={"serve": recipe.serve.model_copy(update={"max_model_len": 16384})})
    with pytest.raises(AssertionError, match=r"serve\.max_model_len"):
        _assert_contract(serve_mutant)
    reference_mutant = recipe.model_copy(
        update={"reference": recipe.reference.model_copy(update={"kind": "sentence_transformers"})}
    )
    with pytest.raises(AssertionError, match=r"reference\.kind"):
        _assert_contract(reference_mutant)


@pytest.mark.parametrize("variant_id", VARIANT_IDS)
def test_serve_argv_carries_the_pinned_flags(variant_id: str) -> None:
    """The argv the wave runner renders: no template file, the nested images_kwargs pin, the pooler, the
    media limit and the video policy's --media-io-kwargs sampling rate (the Qwen3-VL backend's fps rule), per
    variant."""
    recipe = load_recipe(variant_id)
    argv = serve_argv(recipe, port=8100, served_model_name=recipe.id)
    assert argv[:3] == ["vllm", "serve", VARIANTS[variant_id]["model"]]
    assert argv[argv.index("--revision") + 1] == VARIANTS[variant_id]["revision"]
    assert "--chat-template" not in argv
    assert "--trust-remote-code" not in argv
    assert json.loads(argv[argv.index("--pooler-config") + 1]) == {"seq_pooling_type": "LAST"}
    assert json.loads(argv[argv.index("--mm-processor-kwargs") + 1]) == {
        "images_kwargs": {"min_pixels": 4096, "max_pixels": 1843200},
        "videos_kwargs": {"min_pixels": 4096, "max_pixels": 7864320},
    }
    assert json.loads(argv[argv.index("--limit-mm-per-prompt") + 1]) == {"image": 1, "video": 1}
    assert argv[argv.index("--media-io-kwargs") + 1] == '{"video": {"fps": 2}}'


@pytest.mark.parametrize("variant_id", VARIANT_IDS)
def test_card_script_is_vendored_verbatim_and_its_constants_bind(variant_id: str) -> None:
    """The vendored card script is byte-identical to every pinned revision's published script, and the
    constants the reference reads from it are the family's budget and frame instruction."""
    digest = hashlib.sha256((RECIPE_DIR / "qwen3_vl_embedding.py").read_bytes()).hexdigest()
    assert digest == CARD_SHA256
    constants = _reference_module().card_constants()
    assert constants == {"max_length": 8192, "default_instruction": DEFAULT_INSTRUCTION}
    recipe = load_recipe(variant_id)
    assert recipe.client.get("max_tokens") == constants["max_length"] == recipe.serve.max_model_len


@pytest.mark.network
@pytest.mark.parametrize("variant_id", VARIANT_IDS)
def test_both_declared_shapes_render_the_checkpoint_chat_template(
    variant_id: str, tmp_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """The frame both shapes declare is the checkpoint's own: its chat_template.jinja (hash-pinned at the
    variant's revision) renders the card's conversation -- the default instruction as the system turn, the
    text as the user turn, the generation prompt -- to exactly the declared render, for a query and a
    document alike; and so does the reference's prompt builder. (The harness's template check renders one
    string per row and cannot tell the two shapes apart, so this test stands in for it.)"""
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    from rcp_ndcg_test.equivalence import fitting

    recipe_cpu = _recipe_cpu(variant_id, _tokenizer_file(variant_id, tmp_path_factory), tmp_path)
    chat_template = _chat_template_text(variant_id, tmp_path_factory)
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
        declared = client_template(recipe_cpu).render(shape, product_tokenizer, query=text, document=text)
        assert declared == card, shape
        assert reference.card_prompt(text, DEFAULT_INSTRUCTION) == card, shape


@pytest.mark.network
@pytest.mark.parametrize("variant_id", VARIANT_IDS)
def test_stage1_on_cpu(variant_id: str, tmp_path: Path, tmp_path_factory: pytest.TempPathFactory) -> None:
    """Stage 1 with the real tokenizer, per variant: fit's renders for both shapes, the anchor audit (5
    over-length samples per shape), the reference render, the stub engine's /tokenize and the stub's own
    ``usage.prompt_tokens`` of the declared render all agree.  The stub is served the checkpoint's chat
    template (``--model-chat-template``): the messages route's budget rests on that frame, so stage 1's
    prompt-token probe compares the engine's count of it with the client's declared render."""
    tokenizer = _tokenizer_file(variant_id, tmp_path_factory)
    recipe = _recipe_cpu(variant_id, tokenizer, tmp_path)
    engine = start_stub(
        "--tokenizer",
        str(tokenizer),
        "--max-model-len",
        "8192",
        "--model-chat-template",
        str(_chat_template_path(variant_id, tmp_path_factory)),
    )
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
    from rcp_ndcg_test.equivalence import fitting

    facts = stage1_facts(recipe, _PAIRS, fitting.tokenizer_of(recipe), OVER_LENGTH_PER_SHAPE)
    for shape in ("query", "document"):
        body = facts["per_shape"][shape]
        assert len(body["texts"]) >= len(_PAIRS) + OVER_LENGTH_PER_SHAPE, shape
        assert body["cut_rows"] >= OVER_LENGTH_PER_SHAPE, f"{shape}: the over-length rows are cut"
    render = document["render_check"]
    assert render["status"] == "run" and render["passed"] is True, render["failures"][:1]
    assert render["rows"] == 2 * len(_PAIRS), "one render per declared shape per pairs row"
    template = document["template_render_check"]
    assert template["status"] == "run" and template["passed"] is True, template["failures"][:1]
    assert template["template"].endswith(":chat_template.jinja") and template["template_sha256"] == CHAT_TEMPLATE_SHA256
    engine_check = document["engine_tokenize_check"]
    assert engine_check["status"] == "run" and engine_check["passed"] is True, engine_check["failures"][:1]
    prompt_tokens = document["engine_prompt_tokens_check"]
    assert prompt_tokens["status"] == "run" and prompt_tokens["passed"] is True, prompt_tokens["failures"][:1]
    assert prompt_tokens["checked"] > 0
    assert document["passed"] is True


@pytest.mark.network
@pytest.mark.parametrize("variant_id", VARIANT_IDS)
def test_reference_renders_the_card_truncation_not_the_client_cut(
    variant_id: str, tmp_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Decision 9 on this family: the reference renders the card's prompt and the card's own over-cap cut.

    Under the cap the reference's render is byte-identical with what the role client ships, on both
    shapes. Over the cap they differ exactly as ``anchor_drop_over_cap`` declares: the card's
    ``truncation=True`` right-cuts the whole prompt (8191 prompt tokens + the re-appended endoftext), so
    its render loses the frame's tail, while the client keeps the whole frame and cuts the content only.
    """
    from rcp_ndcg_test.equivalence import fitting

    recipe = _recipe_cpu(variant_id, _tokenizer_file(variant_id, tmp_path_factory), tmp_path)
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
        assert len(reference_ids) == recipe.client.get("max_tokens"), "the card fills its cap exactly"
        assert len(client_ids) <= recipe.client.get("max_tokens")
        assert reference_ids[-1] == client_ids[-1] == endoftext
        assert over_client.endswith(tail), "the client keeps the frame's tail"
        assert not over_reference.endswith(tail), "the card's right cut drops it (anchor_drop_over_cap)"
        assert over_reference.startswith(over_client[: over_client.index(long_text[:40])])


@pytest.mark.network
@pytest.mark.parametrize("variant_id", VARIANT_IDS)
def test_an_empty_document_renders_the_card_null(
    variant_id: str, tmp_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """The card renders an empty input as the literal NULL; the declared empty_doc: send_text "NULL" is the
    family's way to ship the same prompt."""
    recipe_cpu = _recipe_cpu(variant_id, _tokenizer_file(variant_id, tmp_path_factory), tmp_path)
    reference = _reference_render(recipe_cpu, [{"query": "q", "documents": [""]}], tmp_path)
    assert "user\nNULL<|im_end|>" in reference[(0, "document")]
    assert served_texts(recipe_cpu, [""], "document") == [reference[(0, "document")]]


@pytest.mark.network
@pytest.mark.parametrize("variant_id", VARIANT_IDS)
def test_the_served_render_is_the_declared_frame_only_with_the_generation_prompt(
    variant_id: str, tmp_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """The messages route ships the content; the engine frames it with the checkpoint's chat template. With
    the declared ``add_generation_prompt: true`` (sent) the render is the declared frame byte for byte; without
    it -- vLLM's chat default, false -- the assistant header is missing, and with it the pooled anchor."""
    from rcp_ndcg_core.content import Content
    from rcp_ndcg_test.equivalence import fitting, stages
    from rcp_ndcg_test.equivalence.wire import role_client

    from rcp_ndcg.inference.types import EncodeRole

    recipe_cpu = _recipe_cpu(variant_id, _tokenizer_file(variant_id, tmp_path_factory), tmp_path)
    chat_template = _chat_template_text(variant_id, tmp_path_factory)
    client, capture = role_client(recipe_cpu, None)
    tokenizer = fitting.tokenizer_of(recipe_cpu)
    for role, shape in ((EncodeRole.QUERY, "query"), (EncodeRole.DOCUMENT, "document")):
        client.encode([Content.from_text("Paris is the capital of France.")], role)
        body = capture.exchanges[-1]["request_body"]
        assert body["add_generation_prompt"] is True and body["add_special_tokens"] is True
        conversation = body["messages"]
        declared = client_template(recipe_cpu).render(
            shape, tokenizer, query="Paris is the capital of France.", document="Paris is the capital of France."
        )
        assert stages.render_chat(chat_template, conversation, add_generation_prompt=True) == declared
        without = stages.render_chat(chat_template, conversation, add_generation_prompt=False)
        assert declared.startswith(without) and declared[len(without) :] == "<|im_start|>assistant\n"


def _media_pairs(tmp_path: Path, recipe: Any, *, video: bool = False) -> Path:
    """One text row and the media request set's rows (the generator's synthetic image buckets).

    The video clips are included only when ``video``: a container's reference side loads the checkpoint's
    own ``Qwen3VLVideoProcessor``, which the recipe-tests environment (the dev extra: no transformers, no
    video decoder) does not carry, so the image-bucket stage runs without them and the wave's media stage
    -- which runs the family's reference python -- covers them.
    """
    from rcp_ndcg_test.observe.media_set import planned_media_rows

    rows, _ = planned_media_rows(recipe)
    if not video:
        rows = [row for row in rows if "media:video" not in (row.get("strata") or [])]
    return _pairs(tmp_path, [_PAIRS[0], *[{key: row[key] for key in ("query", "documents", "media")} for row in rows]])


@pytest.mark.network
@pytest.mark.parametrize("variant_id", VARIANT_IDS)
def test_the_media_stage_holds_the_client_to_the_card_on_every_image_bucket(
    variant_id: str, tmp_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Offline, the product's client and the card's reference agree on every image of the media request set:
    the placement (media before text), the prepared geometry under the pinned budget and the tokens. (The
    video rows' reference side loads the checkpoint's own video processor and runs in the wave's media
    stage; the processor's realisation is pinned offline by the stub tests below.)"""
    from rcp_ndcg_test.equivalence.media import stage_media
    from rcp_ndcg_test.observe.media_set import planned_media_rows

    recipe_cpu = _recipe_cpu(variant_id, _tokenizer_file(variant_id, tmp_path_factory), tmp_path)
    document = stage_media(recipe_cpu, _media_pairs(tmp_path, recipe_cpu), sys.executable)
    assert document is not None and document["passed"] is True, document["failures"][:3]
    # one media item per planned image row (the media-inputs set: the image buckets, the captioned page,
    # the multi-image and the query-image rows)
    planned, _ = planned_media_rows(recipe_cpu)
    images = [row for row in planned if "media:video" not in (row.get("strata") or [])]
    assert document["items"] == len(images) and document["refusals"] == []


def test_the_reference_reads_media_rows_into_the_cards_own_inputs() -> None:
    """The reference's media path decodes the harness's ``media`` entries into the card's own
    ``Qwen3VLEmbedder.process`` input keys: an inline image to a loaded PIL image, a container to the file
    the checkpoint's own video processor reads (the item naming the recipe's declared fps; the per-frame
    pixel bounds of the old card-loader route are gone); the text parts stand where the card can express
    them, and the forms the card cannot express are refused (never silently reordered). No model weights:
    the row reading is the card's ``format_model_input`` contract."""
    import base64
    import io
    import tempfile

    from PIL import Image
    from rcp_ndcg_test.observe.media_set import video_entry

    module = _reference_module()
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), (10, 20, 30)).save(buffer, format="PNG")
    image = {"kind": "image", "uri": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()}
    clip = video_entry("icon", 64, 64, 64)  # 64 frames at the media set's 8 fps
    with tempfile.TemporaryDirectory() as work:
        root = Path(work)
        payload = module.card_inputs("caption", [image], work=root, clip_name="query-0", declared_fps=2.0)
        assert payload["image"].size == (16, 16)
        assert payload["text"] == "caption"
        assert "video" not in payload
        video_payload = module.card_inputs("", [clip], work=root, clip_name="query-1", declared_fps=2.0)
        clip_path = Path(video_payload["video"])
        assert clip_path.is_file()
        assert clip_path.read_bytes() == base64.b64decode(str(clip["uri"]).split(",", 1)[1])
        assert video_payload["fps"] == 2.0
        # the card-loader route's per-frame bounds are gone: the processor realises the container itself
        assert "max_frames" not in video_payload and "min_pixels" not in video_payload
        # The side's text parts stand after the media (the card's order); a text part before or between
        # media parts, several images/videos, or an image before its video is refused loudly.
        trailing = module.card_inputs(
            "caption",
            [image, {"kind": "text", "text": "and "}],
            work=root,
            clip_name="query-2",
            declared_fps=2.0,
        )
        assert trailing["text"] == "and caption"
        for entries in (
            [{"kind": "text", "text": "lead"}, image],
            [image, image],
            [image, clip],
        ):
            with pytest.raises(SystemExit):
                module.side_parts("", entries)


def test_the_reference_processes_each_input_on_its_own(monkeypatch: pytest.MonkeyPatch) -> None:
    """The card's ``process`` batches every input into one processor call; transformers' processor
    replaces every multimodal placeholder spelling found in the batch's texts from ONE shared iterator
    per modality (``processing_utils.get_text_with_replacements``), so a text row carrying the
    tokenizer's literal image/video placeholder spellings -- the pairs' special-token-spellings
    stratum, which the served engine tokenizes as plain special ids -- consumes the batch's media
    replacements and the call dies with StopIteration (reproduced against the real processor and
    transformers 5.19 on the wave's pairs; 5.10.4 carries the same shared iterator). The reference
    therefore calls the card's ``process`` once per input -- the card pools per sequence, so each
    vector is unchanged -- and this pins that contract plus the positional pairing of the assembled
    rows. No model weights: the fake card records every call's input count."""
    import types

    module = _reference_module()
    calls: list[int] = []

    class _FakeModel:
        def to(self, device):
            return self

    class _FakeCard:
        """The card's ``Qwen3VLEmbedder`` surface: the conversation builder plus ``process``."""

        def __init__(self, model_name_or_path: str) -> None:
            self.model = _FakeModel()

        def format_model_input(self, **kwargs):
            return [{"role": "user", "content": [{"type": "text", "text": str(kwargs.get("text") or "NULL")}]}]

        def process(self, inputs, normalize=True):  # noqa: ARG002
            calls.append(len(inputs))
            return [[float(len(calls) - 1)] for _ in inputs]

    card = types.ModuleType("qwen3_vl_embedding")
    card.Qwen3VLEmbedder = _FakeCard  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "qwen3_vl_embedding", card)
    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = lambda *args, **kwargs: "/stub/checkpoint"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    recipe = load_recipe("qwen3-vl-embedding-2b").model_dump(mode="json")
    pairs = [{"query": "q0", "documents": ["d0", "d1"]}, {"query": "q1", "documents": ["d2"]}]
    document = module.mode_embed(recipe, pairs, "cpu")
    assert calls == [1, 1, 1, 1, 1], "one process call per input, never one call over the whole batch"
    rows = document["rows"]
    assert rows[0]["query_vectors"] == [[0.0]]
    assert rows[0]["document_vectors"] == [[2.0], [3.0]]
    assert rows[1]["query_vectors"] == [[1.0]]
    assert rows[1]["document_vectors"] == [[4.0]]


@pytest.mark.network
def test_the_engine_media_count_holds_under_the_pin_and_fails_unpinned(
    tmp_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Against the stub engine emulating the checkpoint (its chat template, the qwen3_vl patch factor, the
    checkpoint's own default budget 4096..1310720 px): served with the recipe's pin, the engine counts every
    image as the client does; served without it (negative control (f)), every page the client prepared above
    1310720 px is resized again and the engine's count differs. (The family's 2b row; the 8b declares the
    same media policy and the same processor family.)"""
    from rcp_ndcg_test.equivalence.media import stage_media
    from rcp_ndcg_test.observe.controls import control_variants

    variant_id = "qwen3-vl-embedding-2b"
    tokenizer = _tokenizer_file(variant_id, tmp_path_factory)
    recipe_cpu = _recipe_cpu(variant_id, tokenizer, tmp_path)
    chat_template = _chat_template_text(variant_id, tmp_path_factory)
    template_file = tmp_path / "chat_template.jinja"
    template_file.write_text(chat_template, encoding="utf-8")
    # the emulated checkpoint's own processor family (the video frames are sized by the family's
    # factor/budget, not by --model-image-factor): the recipe's declared image_processor
    model = [
        "--model-chat-template",
        str(template_file),
        "--model-processor",
        str(recipe_cpu.client["image_processor"]),
    ]
    model += ["--model-image-factor", "32"]
    model += ["--model-image-pixels", "4096,1310720", "--max-model-len", "8192"]
    pairs = _media_pairs(tmp_path, recipe_cpu)
    (unpinned,) = [v["recipe"] for v in control_variants(recipe_cpu) if v["control"] == "(f)"]
    results = {}
    for name, served in (("pinned", recipe_cpu), ("unpinned", unpinned)):
        argv = serve_argv(served, port=0, served_model_name=recipe_cpu.id)
        flags = [value for value in argv[argv.index(served.model) + 1 :] if value != "0.0.0.0"]
        flags = [flag for flag in flags if flag != "--host"]
        engine = start_stub("--tokenizer", str(tokenizer), *flags, *model)
        try:
            results[name] = stage_media(recipe_cpu, pairs, sys.executable, base_url=engine.base_url)
        finally:
            engine.stop()
    pinned, broken = results["pinned"], results["unpinned"]
    assert pinned is not None and pinned["passed"] is True, (pinned["failures"][:2], pinned["engine_check"])
    assert broken is not None and broken["passed"] is False
    assert {failure["row"] for failure in broken["engine_check"]["failures"]}, "the unpinned engine counts differently"


@pytest.mark.network
def test_stage1_anchor_mutation_is_red(tmp_path: Path, tmp_path_factory: pytest.TempPathFactory) -> None:
    """A declared anchor edge the rendered ids never carry turns the anchor check red, naming it.

    Dropping the trailing segment alone cannot red THIS family's audit: the anchor of ``anchor: last``
    with ``add_special_tokens: true`` is the post-processor's endoftext, which every render still
    carries (the audit computes its expected edge from the declaration). The mutation that breaks the
    anchor contract is a declared marker anchor whose marker is absent from every render -- the audit
    then reports the missing marker names and fails.
    """
    recipe_cpu = _recipe_cpu(
        "qwen3-vl-embedding-2b", _tokenizer_file("qwen3-vl-embedding-2b", tmp_path_factory), tmp_path
    )
    mutated_dir = tmp_path / "mutated" / "qwen3-vl-embedding"  # the family id must equal the directory name
    shutil.copytree(Path(str(recipe_cpu._dir)), mutated_dir)
    data = yaml.safe_load((mutated_dir / "family.yaml").read_text(encoding="utf-8"))
    template = data["client"]["template"]
    assert template["anchor"] == "last"
    template["anchor"] = "marker"
    template["anchor_markers"] = ["vision_start"]  # a real special this frame never contains
    (mutated_dir / "family.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    mutated = load_recipe("qwen3-vl-embedding-2b", root=tmp_path / "mutated")

    rows = [_PAIRS[0]]
    control = stage1_prompts(recipe_cpu, _pairs(tmp_path, rows), None, over_length_per_shape=1)
    document = stage1_prompts(mutated, _pairs(tmp_path, rows), None, over_length_per_shape=1)
    assert control["anchor_check"]["passed"] is True
    assert document["anchor_check"]["passed"] is False, document["anchor_check"]
    failures = document["anchor_check"]["failures"]
    assert failures and failures[0]["check"] == "markers", "the red check names the missing markers"
    assert "vision_start" in failures[0]["missing_names"]


#: This family's recipes (the internal-label scan below covers both).
RECIPE_IDS = ("qwen3-vl-embedding-2b", "qwen3-vl-reranker-2b")


def test_the_media_modes_refuse_the_retired_media_columns() -> None:
    """A row carrying the retired per-column media fields is refused loudly by the embed and media modes:
    the harness's media rows carry the ``media`` field, and a silently ignored column would compare a
    different prompt."""
    module = _reference_module()
    row = {"query": "q", "documents": ["d"], "query_image": "data:image/png;base64,AA=="}
    with pytest.raises(SystemExit, match="retired media columns"):
        module.mode_media([row], {})
    with pytest.raises(SystemExit, match="retired media columns"):
        module.mode_embed({}, [row], "cpu")


def _stub_video_processor(
    grid: tuple[int, int, int] = (2, 16, 28), frames_indices: tuple[int, ...] = (0, 1, 2, 3)
) -> tuple[Any, list[dict[str, Any]]]:
    """A stand-in for the checkpoint's ``Qwen3VLVideoProcessor``: records its calls, answers a chosen grid.

    The stub carries the surface the reference reads: ``__call__``, ``patch_size`` and ``merge_size`` (the
    grid's patch multiples and the merged patch count) -- the same attributes the real processor's outputs
    are read through. No decode, no weights.
    """
    from types import SimpleNamespace

    calls: list[dict[str, Any]] = []

    class _StubVideoProcessor:
        patch_size, merge_size, temporal_patch_size = 16, 2, 2

        def __call__(self, videos: list[str], **kwargs: Any) -> dict[str, Any]:
            calls.append({"videos": list(videos), "kwargs": dict(kwargs)})
            return {
                "video_grid_thw": [list(grid)],
                "video_metadata": [SimpleNamespace(frames_indices=list(frames_indices))],
            }

    return _StubVideoProcessor(), calls


def test_the_reference_video_preparation_reads_the_processors_own_realisation(tmp_path: Path) -> None:
    """The reference's container route hands the container to the checkpoint's own video processor at the
    recipe's declared fps and whole-clip pin, and reports what that processor's outputs say it realised:
    the frames it sampled, the per-frame size it resized to and its patch tokens -- never a re-derived
    count. The shipped video case's clip (448x256, 4 frames at 2 fps) is the synthetic one: the stub's
    grid is the real processor's for that clip (2 temporal groups of 16x28 patches), the engine's ported
    whole-clip rule sizes the same frame under the same pin, and the old card-loader route could not
    express the pin at all (its per-frame ceiling refused it)."""
    from rcp_ndcg_test.observe.media_set import video_entry

    from rcp_ndcg.data.resolution import PROCESSORS, _clip_frame_size

    module = _reference_module()
    recipe = load_recipe("qwen3-vl-embedding-2b").model_dump(mode="json")
    pin = module.video_pixel_pin(recipe)
    assert pin == (4096, 7864320)
    processor, calls = _stub_video_processor()
    facts = module.video_realisation(
        processor,
        video_entry("case", 448, 256, 4, fps=2.0),
        work=tmp_path,
        clip_name="document-0-0",
        declared_fps=2.0,
        pin=pin,
    )
    assert facts == {"frames": 4, "frame_width": 448, "frame_height": 256, "tokens": 224}
    (call,) = calls
    assert call["videos"] == [str(tmp_path / "document-0-0.avi")]
    assert call["kwargs"] == module.video_processor_kwargs(2.0, pin)
    assert call["kwargs"]["size"] == {"shortest_edge": 4096, "longest_edge": 7864320}
    assert call["kwargs"]["fps"] == 2.0 and call["kwargs"]["do_sample_frames"] is True
    assert call["kwargs"]["cap_pixels_per_frame"] is False
    assert _clip_frame_size(PROCESSORS["qwen3_vl"], 4, 256, 448, min_pixels=pin[0], max_pixels=pin[1]) == (256, 448)


def test_the_media_mode_reports_the_processors_own_realisation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The media mode's video item is the checkpoint processor's realisation (frames, per-frame pixels,
    patch tokens), not a re-derivation: the mode loads the processor from the recipe's pinned snapshot
    and reports exactly the grid and metadata the processor returns."""
    from rcp_ndcg_test.observe.media_set import video_entry

    module = _reference_module()
    recipe = load_recipe("qwen3-vl-embedding-2b").model_dump(mode="json")
    processor, calls = _stub_video_processor()
    monkeypatch.setattr(module, "load_video_processor", lambda _recipe: processor)
    row = {
        "query": "a bird",
        "documents": [""],
        "media": {"query": [], "documents": [[video_entry("case", 448, 256, 4, fps=2.0)]]},
    }
    document = module.mode_media([row], recipe)
    assert document["rows"] == [
        {
            "index": 0,
            "side": "document 0",
            "placement": ["video"],
            "media": [{"kind": "video", "frames": 4, "frame_width": 448, "frame_height": 256, "tokens": 224}],
        }
    ]
    assert calls and calls[0]["kwargs"]["size"] == {"shortest_edge": 4096, "longest_edge": 7864320}


def test_the_reference_refuses_a_video_without_a_declared_rate(tmp_path: Path) -> None:
    """A container needs the recipe's declared engine rate: without it the checkpoint's processor would
    sample at its own rate, a different instrument, so the reference refuses loudly."""
    from rcp_ndcg_test.observe.media_set import video_entry

    module = _reference_module()
    clip = video_entry("icon", 64, 64, 64)
    with pytest.raises(SystemExit, match="no video rate"):
        module.card_inputs("", [clip], work=tmp_path, clip_name="query-0", declared_fps=None)
    with pytest.raises(SystemExit, match="no video rate"):
        module.declared_video_rate(None)


def test_the_shipped_video_case_reaches_the_processors_route(tmp_path: Path) -> None:
    """The media translation, proven on the shipped video case's own container: the case fill's entry
    builder reads ``media/clip.mp4`` off the case file and the reference's container route accepts it --
    the bytes land in a scratch file whose suffix states the container (ISOBMFF), the item names the
    recipe's declared fps, and the whole-clip pin is handed to the checkpoint's processor as its own
    ``size`` (the per-frame ceiling that refused the old card-loader route does not exist there)."""
    import base64

    from rcp_ndcg_test.case_fill import _media_entry
    from rcp_ndcg_test.cases import load_case

    cases = Path(__file__).resolve().parents[2] / "cases"
    case = load_case(cases / "qwen3-vl-embedding-2b" / "video-short-single.yaml")
    entry = _media_entry(case, "media/clip.mp4")
    assert entry["kind"] == "video" and entry["mime"] == "video/mp4"
    assert (entry["width"], entry["height"], entry["num_frames"], entry["fps"]) == (448, 256, 4, 2.0)

    module = _reference_module()
    payload = module.card_inputs("", [entry], work=tmp_path, clip_name="document-0-0", declared_fps=2.0)
    assert payload["fps"] == 2.0
    assert "image" not in payload and "text" not in payload and "max_frames" not in payload
    clip = Path(payload["video"])
    assert clip.suffix == ".mp4" and clip.read_bytes() == base64.b64decode(str(entry["uri"]).split(",", 1)[1])

    recipe = load_recipe("qwen3-vl-embedding-2b").model_dump(mode="json")
    kwargs = module.video_processor_kwargs(2.0, module.video_pixel_pin(recipe))
    assert kwargs["size"] == {"shortest_edge": 4096, "longest_edge": 7864320}
    assert kwargs["fps"] == 2.0
