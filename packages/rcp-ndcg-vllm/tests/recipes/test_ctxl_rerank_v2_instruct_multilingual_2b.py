"""The ``ctxl-rerank-v2-instruct-multilingual-2b`` recipe: the full contract, and stage 1 on CPU.

The contract test freezes every ``serve``/``client``/``reference`` field through the shared
:func:`assert_recipe_contract` (nothing rides unpinned), and two mutants show it red on drift.  The
served semantics this family decided are pinned with failing-first tests: ``instruction: none``
(the paper configs' mode — the reference never folds an instruction, a pairs row's instruction is
ignored on both sides) and the merged rerank client's settle rule (the query ships at its declared
share whenever it exceeds it — the pair fit alone binds the share on overflow only).

Stage 1 runs without an engine: the product's role client ships the request bodies, and the
reference subprocess's ``render`` must reproduce the wire's content spans byte for byte (its own
port of the settle rule and the anchor-preserving cut), with the served template file rendering what
the declared template renders.  The ``tokenizer.json`` downloads into the shared tokeniser cache
(``RCP_NDCG_VLLM_TOKENIZER_CACHE`` when set, else ``tmp_path``); those tests are network tests and
skip offline (``tests/recipes/conftest.py``).
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest
import yaml
from rcp_ndcg_vllm import load_recipe, serve_argv
from rcp_ndcg_vllm.equivalence import stage1_prompts
from rcp_ndcg_vllm.equivalence.reference import run_reference

from ._contract import assert_recipe_contract
from ._served import served_pair, served_rows, stage1_facts, tokenizer_cache

RECIPE_ID = "ctxl-rerank-v2-instruct-multilingual-2b"
MODEL_ID = "ContextualAI/ctxl-rerank-v2-instruct-multilingual-2b"
REVISION = "6ffef5dc552583b8db58dc4a87f79f7aee78d2d9"
RECIPE_DIR = Path(__file__).resolve().parents[2] / "recipes" / RECIPE_ID

MAX_TOKENS = 8192
QUERY_MAX_TOKENS = 4096

FRAME_HEAD = "Check whether a given document contains information helpful to answer the query.\n<Document> "
FRAME_MID = "\n<Query> "
FRAME_TAIL = " ??"

EXPECTED_SERVE = {
    "chat_template": "template.jinja",
    "convert": None,
    "dtype": "bfloat16",
    "extra_args": [],
    "hf_overrides": {
        "architectures": ["Qwen3ForSequenceClassification"],
        "classifier_from_token": ["!"],
        "method": "no_post_processing",
    },
    "io_processor_plugin": None,
    "limit_mm_per_prompt": None,
    "max_model_len": 8192,  # the paper's MAX_SEQ_LENGTH as the engine window
    "mm_processor_kwargs": {},
    "plugin": None,
    "pooler_config": {"use_activation": False},
    "runner": "pooling",
    "trust_remote_code": False,
}

EXPECTED_CLIENT = {
    "aggregation": "max",
    "api": "rerank",
    "api_key_env": None,
    "batch_size": None,
    "chunk": None,
    "concurrency": 64,
    "connect_timeout_s": 5.0,
    "empty_doc": "send",
    "empty_doc_text": None,
    "empty_query": "refuse",
    "headers_env": {},
    "image_policy": None,
    "image_processor": None,
    "instruction": "none",  # the paper's served-path config: a bare query, never a fold
    "listwise": False,
    "max_images": 0,
    "max_retries": 2,
    "max_tokens": 8192,
    "max_videos": 0,
    "media_sides": ["query", "document"],
    "model": RECIPE_ID,
    "on_overflow": "cut",
    "query_max_tokens": 4096,
    # Unset in the recipe (the family convention of these three): `client_config` records the
    # recipe id; the serve block and `sources` keep the server-side settings (sweep finding 13).
    "recipe": None,
    "request_shape": "text",
    "revision": REVISION,
    "template": {
        "add_special_tokens": {"pair": True},
        "anchor": "last",
        "anchor_markers": [],
        "document": None,
        "normalize": [],
        "pair": [
            {"content": None, "fixed": FRAME_HEAD},
            {"content": "document", "fixed": None},
            {"content": None, "fixed": FRAME_MID},
            {"content": "query", "fixed": None},
            {"content": None, "fixed": FRAME_TAIL},
        ],
        "query": None,
    },
    "timeout_s": 600.0,
    "tokenizer": f"{MODEL_ID}@{REVISION}",
    "use_activation": False,
    "video_policy": None,
    "wait_on_outage_s": None,
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
    "startup_timeout_s": 1800,  # the schema default; the recipe no longer restates it (finding 15)
}


def _tokenizer_file(tmp_path: Path) -> Path:
    """The recipe's ``tokenizer.json`` at the pinned revision, in the shared tokeniser cache (the
    lane's scratch dir when ``RCP_NDCG_VLLM_TOKENIZER_CACHE`` is set, else ``tmp_path``)."""
    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as error:  # pragma: no cover - the [hf] extra
        pytest.skip(f"the Hub download needs huggingface_hub: {error}")
    try:
        target = tokenizer_cache(tmp_path / "tokenizer")
        return Path(hf_hub_download(MODEL_ID, "tokenizer.json", revision=REVISION, local_dir=str(target)))
    except Exception as error:  # noqa: BLE001 - any Hub failure means the same skip
        pytest.skip(f"offline: the {MODEL_ID} tokenizer.json is not downloadable ({error})")


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


def _over_share_query(tokenizer) -> str:
    """A whitespace-clean query over the 4096-token share but under the pair budget (the declared
    divergence row class)."""
    words: list[str] = []
    text = ""
    for _ in range(6000):
        words.append("flibbertigibbet")
        text = " ".join(words) + "."
        if tokenizer.count(text) > QUERY_MAX_TOKENS + 400:
            break
    return text


def write_pairs(path: Path, rows: list[dict]) -> Path:
    """The pairs JSONL file for a stage-1 or reference run."""
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


#: Fifteen realistic pairs (three with a run-level instruction): short on purpose — the gating
#: pairs keep queries within the share and documents within the budget (the recipe's declared
#: divergence policy) — plus one over-budget pair and one over-share query in the span tests.
PAIRS: list[dict] = [
    {
        "query": "What is the capital of France?",
        "documents": ["Paris is the capital and largest city of France."],
        "instruction": "Answer with the city name",
    },
    {"query": "Who wrote Pride and Prejudice?", "documents": ["Pride and Prejudice is an 1813 novel by Jane Austen."]},
    {
        "query": "capital of Japan",
        "documents": ["Tokyo is the capital of Japan and the seat of its government."],
        "instruction": "Find the city",
    },
    {
        "query": "definition of entropy",
        "documents": ["Entropy is a measurable property, most commonly associated with a state of disorder."],
    },
    {
        "query": "largest planet",
        "documents": ["Jupiter is the fifth planet from the Sun and the largest."],
        "instruction": "  ",
    },
    {"query": "speed of light", "documents": ["The speed of light in vacuum equals 299,792,458 metres per second."]},
    {
        "query": "who painted the Mona Lisa",
        "documents": ["The Mona Lisa was painted by the Italian Renaissance artist Leonardo da Vinci."],
    },
    {"query": "chemical symbol for gold", "documents": ["Gold is a chemical element with the symbol Au."]},
    {
        "query": "longest river in Africa",
        "documents": ["The Nile is a major north-flowing river in Africa, and the longest."],
    },
    {
        "query": "first person on the Moon",
        "documents": ["Neil Armstrong was an astronaut and the first person to walk on the Moon."],
    },
    {
        "query": "boiling point of water",
        "documents": ["The boiling point of water is 100 degrees Celsius at standard pressure."],
    },
    {"query": "currency of Japan", "documents": ["The yen is the official currency of Japan."]},
    {
        "query": "who developed relativity",
        "documents": ["Albert Einstein developed the theory of relativity, a pillar of modern physics."],
    },
    {
        "query": "  main language of Brazil  ",
        "documents": ["Portuguese is the official language of Brazil, spoken by nearly everyone."],
    },
    {
        "query": "smallest prime number",
        "documents": ["The number 2 is the smallest prime number and the only even prime."],
    },
]


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
    """Mutant 1 (sweep finding 9): ``serve.max_model_len`` 8192 -> 16384 must red, naming the field."""
    mutated = _mutated_recipe(tmp_path, lambda data: data["serve"].__setitem__("max_model_len", 16384))
    with pytest.raises(AssertionError, match=r"serve\.max_model_len"):
        assert_recipe_contract(
            mutated, serve=EXPECTED_SERVE, client=EXPECTED_CLIENT, reference=EXPECTED_REFERENCE, top=EXPECTED_TOP
        )


def test_mutant_changing_the_reference_kind_reds_the_contract_naming_the_field(tmp_path: Path) -> None:
    """Mutant 2 (sweep finding 9): ``reference.kind`` transformers -> remote_code must red, naming the
    field."""
    mutated = _mutated_recipe(tmp_path, lambda data: data["reference"].__setitem__("kind", "remote_code"))
    with pytest.raises(AssertionError, match=r"reference\.kind"):
        assert_recipe_contract(
            mutated, serve=EXPECTED_SERVE, client=EXPECTED_CLIENT, reference=EXPECTED_REFERENCE, top=EXPECTED_TOP
        )


def test_serve_argv_renders_the_golden_engine_command() -> None:
    """The recipe's ``vllm serve`` argv: the template file, the id-0 score head, the logit pooler."""
    recipe = load_recipe(RECIPE_DIR)
    argv = serve_argv(recipe, port=8100, served_model_name=recipe.id)
    assert argv[:3] == ["vllm", "serve", MODEL_ID]
    assert argv[argv.index("--revision") + 1] == REVISION
    assert argv[argv.index("--served-model-name") + 1] == recipe.id
    assert argv[argv.index("--chat-template") + 1] == str(RECIPE_DIR / "template.jinja")
    assert argv[argv.index("--pooler-config") + 1] == '{"use_activation": false}'
    assert argv[argv.index("--max-model-len") + 1] == str(MAX_TOKENS)
    assert json.loads(argv[argv.index("--hf-overrides") + 1]) == EXPECTED_SERVE["hf_overrides"]
    assert argv[argv.index("--runner") + 1] == "pooling"


def test_the_reference_environment_is_documented() -> None:
    """The reference declares the environment it needs, beside itself (the reference rule)."""
    text = (RECIPE_DIR / "requirements-reference.txt").read_text(encoding="utf-8")
    assert "torch==2.9.1" in text
    assert "transformers==4.57.6" in text


def test_the_reference_resolves_the_hub_tokenizer_spec_without_the_revision_suffix() -> None:
    """The recipe's ``client.tokenizer`` (repo@revision) reaches transformers as a bare repo id: the
    reference splits the spec and passes the revision separately (its pinned model and revision stay
    equal to the recipe's)."""
    import importlib.util

    module_spec = importlib.util.spec_from_file_location("ctxl_2b_reference", RECIPE_DIR / "reference.py")
    module = importlib.util.module_from_spec(module_spec)
    bytecode = sys.dont_write_bytecode  # exec_module must not drop a __pycache__ into the recipe dir
    sys.dont_write_bytecode = True
    try:
        module_spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = bytecode
    assert module.DEFAULT_REVISION == REVISION, "the reference's pinned revision drifted from the recipe"
    assert module.DEFAULT_MODEL == MODEL_ID
    assert module._tokenizer_dir(f"{MODEL_ID}@{REVISION}") == MODEL_ID
    assert module._tokenizer_dir(MODEL_ID) == MODEL_ID
    assert "@" not in module._tokenizer_dir(f"{MODEL_ID}@{REVISION}")


def test_the_served_and_reference_prompts_ignore_the_pairs_row_instruction(tmp_path: Path) -> None:
    """``instruction: none`` end to end (sweep items #1 and #2): a pairs row's instruction is ignored
    on the wire and by the reference — the bare query ships, paddings and all, and no side folds
    ``Task: ...\\nQuery: ...``."""
    recipe, tokenizer_file = _local_recipe(tmp_path)
    query = "  padded query  "
    document = "A document."
    shipped = served_pair(recipe, query, [document], instruction="Answer with the city name")
    assert shipped == {"query": query, "documents": [document]}
    rows = [
        {"query": query, "documents": [document], "instruction": "Answer with the city name"},
        {"query": query, "documents": [document], "instruction": "  "},
    ]
    pairs_path = write_pairs(tmp_path / "pairs.jsonl", rows)
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


def test_the_reference_render_spans_are_the_served_spans_including_over_share_and_over_budget(
    tmp_path: Path,
) -> None:
    """The reference's ``render`` is its own port of the wire's settle rule and cut: byte-equal to
    the client's captured spans (sweep item #5: the query settles at its share whenever it exceeds
    it — the share never binds on overflow only)."""
    from rcp_ndcg.data.tokenizer import load_tokenizer

    recipe, tokenizer_file = _local_recipe(tmp_path)
    tokenizer = load_tokenizer(str(tokenizer_file))
    long_query = _over_share_query(tokenizer)
    rows = [
        {"query": "short query", "documents": ["a short document.", "a second document."]},
        {"query": long_query, "documents": ["a short document under the settled share."]},
        {"query": "short query", "documents": ["parisisthecapitaloffranceandeurope" * 400]},
        {"query": "empty document ships", "documents": [""]},
    ]
    pairs_path = write_pairs(tmp_path / "pairs.jsonl", rows)
    wire = served_rows(recipe, rows, tokenizer)["per_shape"]["pair"]["spans"]
    reference = run_reference(
        sys.executable,
        str(RECIPE_DIR / "reference.py"),
        mode="render",
        pairs_path=pairs_path,
        out_path=tmp_path / "reference.json",
        tokenizer_spec=str(tokenizer_file),
    )
    assert [{"query": row["query"], "documents": list(row["documents"])} for row in reference["rows"]] == wire
    assert tokenizer.count(long_query) > QUERY_MAX_TOKENS
    assert tokenizer.count(wire[1]["query"]) <= QUERY_MAX_TOKENS
    assert wire[1]["query"] == long_query[: len(wire[1]["query"])]
    rendered = FRAME_HEAD + wire[2]["documents"][0] + FRAME_MID + wire[2]["query"] + FRAME_TAIL
    assert tokenizer.count(rendered, add_special_tokens=True) <= MAX_TOKENS
    assert rendered.endswith(FRAME_TAIL)
    assert wire[3]["documents"] == [""]  # empty_doc: send keeps the empty string


def test_stage1_on_cpu_passes_token_id_equality_and_the_anchor_check(tmp_path: Path) -> None:
    """Stage 1 on CPU: the reference's spans and the served template over the pairs file plus the
    harness's 5 over-length samples, with the anchor audit on every sampled row."""
    from rcp_ndcg.data.tokenizer import load_tokenizer

    recipe, tokenizer_file = _local_recipe(tmp_path)
    tokenizer = load_tokenizer(str(tokenizer_file))
    pairs_path = write_pairs(tmp_path / "pairs.jsonl", PAIRS)
    document = stage1_prompts(recipe, pairs_path, sys.executable, over_length_per_shape=5)

    assert document["sampled"] >= 20
    assert document["anchor_check"]["passed"] is True, document["anchor_check"]["failures"][:1]
    assert document["anchor_check"]["checked"] == 2 * document["sampled"]
    render = document["render_check"]
    assert render["status"] == "run" and render["passed"] is True, render["failures"][:1]
    assert render["rows"] == len(PAIRS)
    assert document["template_render_check"]["passed"] is True, document["template_render_check"]["failures"][:1]
    assert document["engine_tokenize_check"]["status"] == "not_run"  # no engine on CPU: never "passed"
    assert document["passed"] is True, document
    facts = stage1_facts(recipe, PAIRS, tokenizer, 5)
    assert facts["per_shape"]["pair"]["cut_rows"] >= 5, facts["per_shape"]["pair"]["cut_rows"]
    template = recipe.client.template
    assert template is not None
    assert facts["per_shape"]["pair"]["overhead"] == template.overhead("pair", tokenizer)


def test_dropping_the_trailing_anchor_segment_reddens_the_template_check(tmp_path: Path) -> None:
    """Mutation: drop the declared template's trailing anchor segment — the file-vs-declaration
    check goes red on the template the engine renders (the file still emits the " ??" the
    declaration lost)."""
    mutated = tmp_path / "mutant" / RECIPE_ID
    shutil.copytree(RECIPE_DIR, mutated)
    data = yaml.safe_load((mutated / "recipe.yaml").read_text(encoding="utf-8"))
    data["client"]["tokenizer"] = str(_tokenizer_file(tmp_path))
    assert data["client"]["template"]["pair"][-1]["fixed"] == FRAME_TAIL
    data["client"]["template"]["pair"] = data["client"]["template"]["pair"][:-1]
    (mutated / "recipe.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    recipe = load_recipe(mutated)
    pairs_path = write_pairs(tmp_path / "pairs.jsonl", PAIRS[:3])
    document = stage1_prompts(recipe, pairs_path, sys.executable, over_length_per_shape=1)
    assert document["template_render_check"]["passed"] is False
    assert document["anchor_check"]["passed"] is True  # the span audit does not read the frame
