"""The ctxl-rerank-v2-instruct-multilingual-6b recipe: the full contract, and stage 1 on CPU.

The contract test freezes every ``serve``/``client``/``reference`` field through the shared
:func:`assert_recipe_contract` (nothing rides unpinned), and two mutants show it red on drift.  The
served semantics this family decided are pinned with failing-first tests: ``instruction: none``
(the paper configs' mode — the reference never folds or appends a pairs row's instruction) and the
merged rerank client's settle rule (the query ships at its declared share whenever it exceeds it —
the pair fit alone binds the share on overflow only).

Stage 1 is CPU-only: the recipe's ``tokenizer.json`` (the only Hub artifact it needs) downloads
through the shared ``fetch_tokenizer`` (``RCP_NDCG_VLLM_TOKENIZER_CACHE`` when set, else
``tmp_path``; sha256-pinned), and the reference subprocess (``--mode render``, pure string work)
fills the span format with the paper's own spans (the raw query and documents, uncut -- never a
port of the client's cut): equal to the wire's on every under-cap row, over-cap rows in the declared
``anchor_drop_over_cap`` table.  Those tests are network tests and skip offline
(``tests/recipes/conftest.py``).  Weights, score mode and the engine belong to the GPU wave (the
recipe ships ``status: unverified`` until it passes there).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from rcp_ndcg_test.equivalence import stage1_prompts
from rcp_ndcg_test.equivalence.reference import run_reference
from rcp_ndcg_vllm import load_recipe, serve_argv
from rcp_ndcg_vllm.recipe import default_recipes_root

from ._contract import assert_recipe_contract
from ._served import client_template, fetch_tokenizer, served_pair, served_rows, stage1_facts

RECIPE_ID = "ctxl-rerank-v2-instruct-multilingual-6b"
REPO_ID = "ContextualAI/ctxl-rerank-v2-instruct-multilingual-6b"
REVISION = "f14ca1a2fc204dc0934c84a3d2e278f8ff646b80"  # re-checked against the HF API on 2026-10-06

RECIPE_DIR = default_recipes_root() / RECIPE_ID
TOKENIZER_URL = f"https://huggingface.co/{REPO_ID}/resolve/{REVISION}/tokenizer.json"
TOKENIZER_SHA256 = "b0240ce510f08e6c2041724e9043e33be9d251d1e4a4d94eb68cd47b954b61d2"  # Hub LFS oid at REVISION

MAX_TOKENS = 8192
QUERY_MAX_TOKENS = 4096

FRAME_HEAD = "Check whether a given document contains information helpful to answer the query.\n<Document> "
FRAME_MID = "\n<Query> "
FRAME_TAIL = " ??"

EXPECTED_SERVE = {
    "chat_template": "template.jinja",  # the naming convention
    "convert": None,
    "dtype": "bfloat16",
    "extra_args": [],
    "hf_overrides": {
        "architectures": ["MistralForSequenceClassification"],
        "classifier_from_token": ["<unk>"],
        "method": "no_post_processing",
    },
    "io_processor_plugin": None,
    "limit_mm_per_prompt": None,
    "max_model_len": 32768,
    "mm_processor_kwargs": {},
    "plugin": None,
    "pooler_config": {"use_activation": False},
    "runner": "pooling",
    "trust_remote_code": False,
}

EXPECTED_CLIENT = {
    "api": "rerank",
    "tokenizer": "ContextualAI/ctxl-rerank-v2-instruct-multilingual-6b@f14ca1a2fc204dc0934c84a3d2e278f8ff646b80",
    "max_tokens": 8192,
    "query_max_tokens": 4096,
    "template": {
        "pair": [
            {"fixed": "Check whether a given document contains information helpful to answer the query.\n<Document> "},
            {"content": "document"},
            {"fixed": "\n<Query> "},
            {"content": "query"},
            {"fixed": " ??"},
        ],
        "anchor": "last",
        "add_special_tokens": True,
    },
    "instruction": "none",
    "use_activation": False,
    "on_overflow": "cut",
    "empty_doc": "send",
    "model": "ctxl-rerank-v2-instruct-multilingual-6b",
    "revision": "f14ca1a2fc204dc0934c84a3d2e278f8ff646b80",
}

EXPECTED_REFERENCE = {
    "entry": "reference.py",
    "kind": "transformers",
    "known_deviations": ["anchor_drop_over_cap"],
    "score_scale": "logit",
}

EXPECTED_TOP = {
    "id": RECIPE_ID,
    "licence": "CC-BY-NC-SA-4.0",
    "revision": REVISION,
    "role": "rerank",
    "scoring": "pointwise",
}

EXPECTED_ENGINE = {
    "image": "vllm/vllm-openai:v0.31.0",
    "min_version": "0.31.0",
    "name": "vllm",
    "startup_timeout_s": 1800,  # the schema default; the recipe no longer restates it
}

# 20 pairs (multilingual on purpose: the checkpoint is multilingual), all under the 8192-token
# budget and within the share so the gating comparison is the served-vs-reference one the recipe
# declares. One mid-file row carries an instruction on purpose: the served path and the reference
# both ignore it (instruction: none), so stage 1 must stay green with it present.
_PAIRS: list[dict] = [
    {"query": "What is the capital of France?", "documents": ["Paris is the capital and largest city of France."]},
    {"query": "法国的首都是什么？", "documents": ["巴黎是法国的首都，也是全国的政治、经济和文化中心。"]},
    {"query": "Was ist die Hauptstadt von Deutschland?", "documents": ["Berlin ist die Hauptstadt des Landes."]},
    {"query": "¿Cuál es la capital de España?", "documents": ["Madrid es la capital de España y su ciudad mayor."]},
    {"query": "日本の首都はどこですか", "documents": ["東京は日本の首都であり、人口は約1,400万人です。"]},
    {
        "query": "Who wrote Hamlet and when?",
        "documents": ["Hamlet is a tragedy written by William Shakespeare between 1599 and 1601."],
        "instruction": "Answer from the document only.",
    },
    {"query": "Quelle est la capitale du Portugal ?", "documents": ["Lisbonne est la capitale du Portugal."]},
    {"query": "What gas do plants absorb?", "documents": ["Plants absorb carbon dioxide during photosynthesis."]},
    {
        "query": "Wie funktioniert Photosynthese?",
        "documents": ["Bei der Photosynthese wandeln Pflanzen Lichtenergie in chemische Energie um."],
    },
    {"query": "What is the boiling point of water at sea level?", "documents": ["At sea level, water boils at 100 C."]},
    {
        "query": "Le loup-garou est-il un mythe européen ?",
        "documents": ["Le loup-garou est une créature légendaire du folklore européen."],
    },
    {"query": "What is the currency of Japan?", "documents": ["The yen is the official currency of Japan."]},
    {
        "query": "Chi ha dipinto la Monna Lisa?",
        "documents": ["La Monna Lisa è un dipinto a olio di Leonardo da Vinci."],
    },
    {"query": "What does DNA stand for?", "documents": ["DNA stands for deoxyribonucleic acid."]},
    {
        "query": "Wie hoch ist der Mount Everest?",
        "documents": ["Der Mount Everest ist mit 8.848 Metern der höchste Berg der Erde."],
    },
    {
        "query": "What is the largest planet in the solar system?",
        "documents": ["Jupiter is the largest planet in the solar system."],
    },
    {
        "query": "Где была основана древняя библиотека?",
        "documents": ["Знаменитая библиотека была основана в третьем веке до нашей эры в Александрии."],
    },
    {
        "query": "What is artificial photosynthesis?",
        "documents": ["Research on artificial photosynthesis aims to store solar energy in chemical fuels."],
    },
    {"query": "Quel est le plus long fleuve du monde ?", "documents": ["Le Nil et l'Amazone se disputent le titre."]},
    {"query": "What language is spoken in Brazil?", "documents": ["Portuguese is the language of Brazil."]},
]


def _tokenizer_file(tmp_path: Path) -> Path:
    """The recipe's ``tokenizer.json`` at the pinned revision, sha256-checked, through the shared
    tokenizer cache (``_served.fetch_tokenizer``); skips with the reason when offline."""
    return fetch_tokenizer(TOKENIZER_URL, f"{RECIPE_ID}@{REVISION}/tokenizer.json", tmp_path, sha256=TOKENIZER_SHA256)


def _local_recipe(tmp_path: Path):
    """The recipe copied into ``tmp_path`` with ``client.tokenizer`` on the downloaded tokenizer (the
    same bytes, a local spec), so the product and the reference subprocess tokenise offline."""
    tokenizer_file = _tokenizer_file(tmp_path)
    copied = tmp_path / RECIPE_ID
    shutil.copytree(RECIPE_DIR, copied)
    data = yaml.safe_load((copied / "recipe.yaml").read_text(encoding="utf-8"))
    data["client"]["tokenizer"] = str(tokenizer_file)
    (copied / "recipe.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return load_recipe(copied), tokenizer_file


def _long_text(tokenizer, tokens: int) -> str:
    """A whitespace-clean text of more than ``tokens`` tokens, sized from one measured unit (no
    re-count of a growing text)."""
    unit = " flibbertigibbet"
    per_unit = max(1, tokenizer.count(unit * 8) // 8)
    text = "flibbertigibbet" + unit * (tokens // per_unit + 50) + "."
    assert tokenizer.count(text) > tokens
    return text


def _pairs_path(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "pairs.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_recipe_loads_and_declares_the_full_contract() -> None:
    """Every serve, client and reference field (plus the pinned top-level and engine ones) is
    frozen: a value drift, an unpinned field or a vanished field all fail naming the exact path."""
    recipe = load_recipe(RECIPE_DIR)
    assert_recipe_contract(
        recipe,
        serve=EXPECTED_SERVE,
        client=EXPECTED_CLIENT,
        reference=EXPECTED_REFERENCE,
        top=EXPECTED_TOP,
    )
    assert recipe.engine.model_dump(mode="json") == EXPECTED_ENGINE
    assert recipe.input == ["text"] and recipe.status.state == "unverified"
    assert len(recipe.sources) >= 5 and recipe.notes


def _mutated_recipe(tmp_path: Path, mutate):
    """The shipped recipe copied under ``tmp_path/.../<id>`` (the loader pins id == directory name)
    and drifted once; used by both mutants."""
    copied = tmp_path / "mutant" / RECIPE_ID
    shutil.copytree(RECIPE_DIR, copied)
    data = yaml.safe_load((copied / "recipe.yaml").read_text(encoding="utf-8"))
    mutate(data)
    (copied / "recipe.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return load_recipe(copied)


def test_mutant_dropping_the_serve_max_model_len_reds_the_contract_naming_the_field(tmp_path: Path) -> None:
    """Mutant 1: ``serve.max_model_len`` 32768 -> 40960 must red, naming the field."""
    mutated = _mutated_recipe(tmp_path, lambda data: data["serve"].__setitem__("max_model_len", 40960))
    with pytest.raises(AssertionError, match=r"serve\.max_model_len"):
        assert_recipe_contract(
            mutated, serve=EXPECTED_SERVE, client=EXPECTED_CLIENT, reference=EXPECTED_REFERENCE, top=EXPECTED_TOP
        )


def test_mutant_changing_the_reference_kind_reds_the_contract_naming_the_field(tmp_path: Path) -> None:
    """Mutant 2: ``reference.kind`` transformers -> remote_code must red, naming the
    field."""
    mutated = _mutated_recipe(tmp_path, lambda data: data["reference"].__setitem__("kind", "remote_code"))
    with pytest.raises(AssertionError, match=r"reference\.kind"):
        assert_recipe_contract(
            mutated, serve=EXPECTED_SERVE, client=EXPECTED_CLIENT, reference=EXPECTED_REFERENCE, top=EXPECTED_TOP
        )


def test_serve_argv_renders_the_pinned_engine_invocation() -> None:
    """The argv pins the revision, the shipped template file and the raw-logit pooler config."""
    recipe = load_recipe(RECIPE_DIR)
    argv = serve_argv(recipe, port=8100, served_model_name=recipe.id)
    assert argv[:3] == ["vllm", "serve", recipe.model]
    assert argv[argv.index("--revision") + 1] == REVISION
    assert argv[argv.index("--served-model-name") + 1] == recipe.id
    assert argv[argv.index("--pooler-config") + 1] == '{"use_activation": false}'
    assert argv[argv.index("--chat-template") + 1] == str(RECIPE_DIR / "template.jinja")
    assert json.loads(argv[argv.index("--hf-overrides") + 1]) == EXPECTED_SERVE["hf_overrides"]
    assert argv[argv.index("--max-model-len") + 1] == "32768"


def test_the_served_and_reference_prompts_ignore_the_pairs_row_instruction(tmp_path: Path) -> None:
    """``instruction: none`` end to end (the family decision): a pairs row's instruction is ignored
    on the wire and by the reference — the bare query ships, paddings and all (the reference never
    folds ``Task: ...`` nor appends the model card's inline form)."""
    recipe, tokenizer_file = _local_recipe(tmp_path)
    query = "  padded query  "
    document = "A document."
    shipped = served_pair(recipe, query, [document], instruction="Answer from the document only.")
    assert shipped == {"query": query, "documents": [document]}
    rows = [
        {"query": query, "documents": [document], "instruction": "Answer from the document only."},
        {"query": query, "documents": [document], "instruction": "  "},
    ]
    pairs_path = _pairs_path(tmp_path, rows)
    reference = run_reference(
        sys.executable,
        str(RECIPE_DIR / "reference.py"),
        mode="render",
        pairs_path=pairs_path,
        out_path=tmp_path / "reference.json",
        tokenizer_spec=str(tokenizer_file),
    )
    for row in reference["rows"]:
        assert row["query"] == query, "the reference must render the bare query, never a fold"
        assert row["documents"] == [document]


def test_the_reference_renders_the_paper_spans_never_the_clients_cut(tmp_path: Path) -> None:
    """Decision 9: the reference's ``render`` fills the span format with the paper's own spans -- the
    raw query and documents its prompt builder receives, uncut.  Under the cap and within the share
    they equal the wire's spans byte for byte; on an over-share query and an over-budget document the
    wire ships the client's cut (the settle rule; the content-only cut) while the reference keeps the
    paper's uncut spans (the declared divergence row and the declared anchor_drop_over_cap row)."""
    from rcp_ndcg.data.tokenizer import load_tokenizer

    recipe, tokenizer_file = _local_recipe(tmp_path)
    tokenizer = load_tokenizer(str(tokenizer_file))
    long_query = _long_text(tokenizer, QUERY_MAX_TOKENS + 400)
    long_document = _long_text(tokenizer, MAX_TOKENS + 200)
    rows = [
        {"query": "short query", "documents": ["a short document.", "a second document."]},
        {"query": long_query, "documents": ["a short document under the settled share."]},
        {"query": "short query", "documents": [long_document]},
    ]
    pairs_path = _pairs_path(tmp_path, rows)
    wire = served_rows(recipe, rows, tokenizer)["per_shape"]["pair"]["spans"]
    reference = run_reference(
        sys.executable,
        str(RECIPE_DIR / "reference.py"),
        mode="render",
        pairs_path=pairs_path,
        out_path=tmp_path / "reference.json",
        tokenizer_spec=str(tokenizer_file),
    )
    spans = [{"query": row["query"], "documents": list(row["documents"])} for row in reference["rows"]]
    assert spans == [{"query": row["query"], "documents": row["documents"]} for row in rows]  # uncut
    assert spans[0] == wire[0]  # under the cap and the share: byte-identical with the wire
    # The wire settles the over-share query at its share (the merged client's settle rule).
    assert tokenizer.count(long_query) > QUERY_MAX_TOKENS >= tokenizer.count(wire[1]["query"])
    assert wire[1]["query"] == long_query[: len(wire[1]["query"])] != long_query
    # The over-budget pair: the wire cuts the document only, the " ??" anchor re-attached in budget.
    rendered = FRAME_HEAD + wire[2]["documents"][0] + FRAME_MID + wire[2]["query"] + FRAME_TAIL
    assert tokenizer.count(rendered, add_special_tokens=True) <= MAX_TOKENS
    assert wire[2]["query"] == "short query" and wire[2]["documents"][0] != long_document


def test_stage1_on_cpu_passes_token_ids_anchors_and_the_served_template(tmp_path: Path) -> None:
    """Stage 1 on CPU: the product's fit through the role client, the reference's spans, the anchor
    audit and the served template over 20 pairs plus 5 over-length samples."""
    from rcp_ndcg.data.tokenizer import load_tokenizer

    recipe, tokenizer_file = _local_recipe(tmp_path)
    tokenizer = load_tokenizer(str(tokenizer_file))
    rows = [*_PAIRS, {"query": "short query", "documents": [_long_text(tokenizer, MAX_TOKENS + 200)]}]
    pairs_path = _pairs_path(tmp_path, rows)
    document = stage1_prompts(recipe, pairs_path, str(sys.executable), over_length_per_shape=5)
    assert document["sampled"] == 26  # 20 pairs + 1 over-budget pair + 5 over-length pair samples
    assert document["passed"] is True, document
    anchor = document["anchor_check"]
    assert anchor["passed"] is True, anchor["failures"][:1]
    assert anchor["checked"] == 2 * document["sampled"]  # one settled query + one document span per row
    render = document["render_check"]
    assert render["status"] == "run" and render["passed"] is True, render["failures"][:1]
    assert render["rows"] == len(rows)
    # The over-budget pairs row differs by declaration (the paper's uncut spans vs the client's
    # cut): reported in the non-gating table, never gated; every other row compared exactly.
    over_cap = render["over_cap"]
    assert over_cap["known_deviation"] is True and over_cap["gating"] is False
    assert [entry["index"] for entry in over_cap["rows"]] == [len(_PAIRS)]
    assert document["template_render_check"]["passed"] is True, document["template_render_check"]["failures"][:1]
    engine = document["engine_tokenize_check"]
    assert engine["status"] == "not_run" and engine["passed"] is None  # no engine on CPU: neutral, never passed
    facts = stage1_facts(recipe, rows, tokenizer, 5)
    assert facts["per_shape"]["pair"]["cut_rows"] >= 5, facts["per_shape"]["pair"]["cut_rows"]
    template = client_template(recipe)
    assert template is not None
    # The fixed frame (including the post-processor's <s>) is reserved in the budget arithmetic.
    assert facts["per_shape"]["pair"]["overhead"] > 0
    assert facts["per_shape"]["pair"]["overhead"] == template.overhead("pair", tokenizer)


def test_the_reference_scores_the_paper_prompt_text(tmp_path: Path) -> None:
    """The paper's prompt construction (the reference's ``prompt_text``) is the two-line frame with
    the bare query — document first, the " ??" tail, no instruction slot."""
    import importlib.util

    module_spec = importlib.util.spec_from_file_location("ctxl_6b_reference", RECIPE_DIR / "reference.py")
    module = importlib.util.module_from_spec(module_spec)
    bytecode = sys.dont_write_bytecode  # exec_module must not drop a __pycache__ into the recipe dir
    sys.dont_write_bytecode = True
    try:
        module_spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = bytecode
    row = _PAIRS[0]
    assert module._prompt_text(row["query"], row["documents"][0]) == (
        FRAME_HEAD + row["documents"][0] + FRAME_MID + row["query"] + FRAME_TAIL
    )


def test_mutation_dropping_the_trailing_anchor_segment_reddens_the_template_check(tmp_path: Path) -> None:
    """Dropping the declared template's trailing anchor segment (" ??") must fail the file-vs-
    declaration check (the rerank wire carries spans; the frame contract is the template check)."""
    mutated = tmp_path / "mutant" / RECIPE_ID
    shutil.copytree(RECIPE_DIR, mutated)
    data = yaml.safe_load((mutated / "recipe.yaml").read_text(encoding="utf-8"))
    data["client"]["tokenizer"] = str(_tokenizer_file(tmp_path))
    assert data["client"]["template"]["pair"][-1]["fixed"] == FRAME_TAIL
    data["client"]["template"]["pair"] = data["client"]["template"]["pair"][:-1]
    (mutated / "recipe.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    recipe = load_recipe(mutated)
    pairs_path = _pairs_path(tmp_path, _PAIRS[:3])
    document = stage1_prompts(recipe, pairs_path, str(sys.executable), over_length_per_shape=1)
    assert document["template_render_check"]["passed"] is False
    assert document["anchor_check"]["passed"] is True  # the span audit does not read the frame


def test_mutation_stripping_the_anchor_from_the_served_template_file_fails_the_template_check(
    tmp_path: Path,
) -> None:
    """A served template file that lost the tail anchor renders different ids than the client."""
    mutated = tmp_path / "file-mutant" / RECIPE_ID
    shutil.copytree(RECIPE_DIR, mutated)
    data = yaml.safe_load((mutated / "recipe.yaml").read_text(encoding="utf-8"))
    data["client"]["tokenizer"] = str(_tokenizer_file(tmp_path))
    (mutated / "recipe.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    template_file = mutated / "template.jinja"
    text = template_file.read_text(encoding="utf-8")
    stripped = text.rstrip("\n")
    assert stripped.endswith(FRAME_TAIL)
    template_file.write_text(stripped.removesuffix(FRAME_TAIL) + ("\n" if text.endswith("\n") else ""))
    recipe = load_recipe(mutated)
    pairs_path = _pairs_path(tmp_path, _PAIRS[:3])
    document = stage1_prompts(recipe, pairs_path, str(sys.executable), over_length_per_shape=1)
    assert document["template_render_check"]["passed"] is False
    assert document["anchor_check"]["passed"] is True  # the declared shape still keeps its anchor


def test_reference_score_mode_needs_its_own_environment(tmp_path: Path) -> None:
    """The reference's score mode reports the missing reference environment instead of a traceback."""
    pairs_path = _pairs_path(tmp_path, _PAIRS[:1])
    completed = subprocess.run(
        [
            sys.executable,
            str(RECIPE_DIR / "reference.py"),
            "--mode",
            "score",
            "--pairs",
            str(pairs_path),
            "--out",
            str(tmp_path / "out.json"),
            "--tokenizer",
            f"{REPO_ID}@{REVISION}",
            "--device",
            "cpu",
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert completed.returncode != 0
    assert "transformers" in completed.stderr or "ModuleNotFoundError" in completed.stderr


def test_neither_the_template_nor_the_reference_appends_an_instruction() -> None:
    """The family decision (``instruction: none``, as 1b and 2b): an instruction kwarg or a system
    message never reaches the served prompt, and the reference's prompt builder takes no instruction
    -- the bare query is the only query text on both sides."""
    import importlib.util
    import inspect

    import jinja2

    environment = jinja2.Environment(keep_trailing_newline=False)
    template = environment.from_string((RECIPE_DIR / "template.jinja").read_text(encoding="utf-8"))
    expected = FRAME_HEAD + "a document" + FRAME_MID + "a query" + FRAME_TAIL
    assert template.render(query="a query", document="a document", instruction="Be strict.") == expected
    messages = [
        {"role": "system", "content": "Be strict."},
        {"role": "query", "content": "a query"},
        {"role": "document", "content": "a document"},
    ]
    assert template.render(messages=messages, instruction="Be strict.") == expected

    module_spec = importlib.util.spec_from_file_location("ctxl_6b_reference_none", RECIPE_DIR / "reference.py")
    module = importlib.util.module_from_spec(module_spec)
    bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        module_spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = bytecode
    for function in (module._prompt_text, module.CtxlRerank.score, module.CtxlRerank.render, module.score):
        assert "instruction" not in inspect.signature(function).parameters, function.__qualname__
    assert module._prompt_text("a query", "a document") == expected
