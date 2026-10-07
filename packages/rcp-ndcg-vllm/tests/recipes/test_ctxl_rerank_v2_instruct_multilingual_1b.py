"""The recipe ``ctxl-rerank-v2-instruct-multilingual-1b``: the full contract, and stage 1 on CPU.

The contract test freezes every ``serve``/``client``/``reference`` field through the shared
:func:`assert_recipe_contract` (nothing rides unpinned), and two mutants show it red on drift.  The
served semantics this family decided are pinned with failing-first tests: ``instruction: none``
(the paper configs' mode — the reference never folds an instruction, a pairs row's instruction is
ignored on both sides) and the merged rerank client's settle rule (the query ships at its declared
share whenever it exceeds it — the pair fit alone binds the share on overflow only).

Stage 1 here runs without an engine: the product's role client ships the request bodies, the
reference subprocess's ``render`` must reproduce the wire's content spans byte for byte (its own
port of the settle rule and the anchor-preserving cut), and the served template file must render
what the declared template renders.  The tokenizer files (``tokenizer.json`` only) download into
the shared tokeniser cache (``RCP_NDCG_VLLM_TOKENIZER_CACHE`` when set, else ``tmp_path``); those
tests are network tests and skip offline (``tests/recipes/conftest.py``).
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

RECIPE_ID = "ctxl-rerank-v2-instruct-multilingual-1b"
REPO = "ContextualAI/ctxl-rerank-v2-instruct-multilingual-1b"
REVISION = "8fd1edf6a98564cb712064f884b8ef7df5c1b876"
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
    "max_model_len": 32768,
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
    "batch_size": 32,
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
    "recipe": None,
    "request_shape": "text",
    "revision": REVISION,
    "template": {
        "add_special_tokens": True,
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
    "tokenizer": f"{REPO}@{REVISION}",
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


def _tokenizer_file(tmp_path: Path) -> Path:
    """The recipe's ``tokenizer.json`` at the pinned revision, in the shared tokeniser cache.

    The download lands in ``RCP_NDCG_VLLM_TOKENIZER_CACHE`` (the lane's scratch dir) when set, else
    the test's ``tmp_path`` — never the checkout.  Skips with the reason when offline.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as error:  # pragma: no cover - the [hf] extra
        pytest.skip(f"the Hub download needs huggingface_hub: {error}")
    try:
        target = tokenizer_cache(tmp_path / "tokenizer")
        return Path(hf_hub_download(REPO, "tokenizer.json", revision=REVISION, local_dir=str(target)))
    except Exception as error:  # noqa: BLE001 - any Hub failure means the same skip
        pytest.skip(f"offline: the {REPO} tokenizer.json is not downloadable ({error})")


def _local_recipe(tmp_path: Path) -> tuple:
    """The recipe copied into ``tmp_path`` with ``client.tokenizer`` on the downloaded tokenizer (the
    same bytes, a local spec), so the product and the reference subprocess tokenise offline."""
    copied = tmp_path / RECIPE_ID
    shutil.copytree(RECIPE_DIR, copied)
    data = yaml.safe_load((copied / "recipe.yaml").read_text(encoding="utf-8"))
    data["client"]["tokenizer"] = str(_tokenizer_file(tmp_path))
    (copied / "recipe.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return load_recipe(copied), Path(data["client"]["tokenizer"])


def _pairs(tmp_path: Path) -> tuple[Path, list[dict]]:
    """15 in-budget pairs (four with a run-level instruction, one padded) plus one over-budget pair
    and one over-share query (the declared divergence row class)."""
    words = (
        "paris river model retrieval document query score ranking europe capital bank token context "
        "rerank multilingual evidence passage neural archive festival"
    ).split()
    rows: list[dict] = []
    for index in range(15):
        query = f"what does ranking {index} say about the {words[index]} of europe"
        document = " ".join(words[(index + offset) % len(words)] for offset in range(24))
        row: dict = {"query": query, "documents": [document]}
        if index % 4 == 0:
            row["instruction"] = f"Follow retrieval task {index}."
        if index == 3:  # padded query and a whitespace-only instruction: both stay raw (none mode)
            row["query"] = f"  {query}  "
            row["instruction"] = "  "
        rows.append(row)
    rows.append({"query": "short query", "documents": ["parisisthecapitaloffranceandeurope" * 380]})
    rows.append({"query": "over the share", "documents": ["a document that stays short on purpose."]})
    path = tmp_path / "pairs.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path, rows


def test_recipe_loads_and_declares_the_full_contract() -> None:
    """Every serve, client and reference field (plus the pinned top-level ones) is frozen: a value
    drift, an unpinned field or a vanished field all fail naming the exact path."""
    assert_recipe_contract(
        load_recipe(RECIPE_DIR),
        serve=EXPECTED_SERVE,
        client=EXPECTED_CLIENT,
        reference=EXPECTED_REFERENCE,
        top=EXPECTED_TOP,
    )


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
    """Mutant 1 (sweep finding 9): ``serve.max_model_len`` 32768 -> 16384 must red, naming the field."""
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
    """The recipe's ``vllm serve`` argv: the revision, the shipped template file and the raw-logit
    pooler default (sweep finding 9: every file keeps a golden-argv test)."""
    recipe = load_recipe(RECIPE_DIR)
    argv = serve_argv(recipe, port=8100, served_model_name=recipe.id)
    assert argv[:3] == ["vllm", "serve", REPO]
    assert argv[argv.index("--revision") + 1] == REVISION
    assert argv[argv.index("--served-model-name") + 1] == RECIPE_ID
    assert argv[argv.index("--chat-template") + 1] == str(RECIPE_DIR / "template.jinja")
    assert argv[argv.index("--pooler-config") + 1] == '{"use_activation": false}'
    assert argv[argv.index("--max-model-len") + 1] == "32768"
    assert argv[argv.index("--dtype") + 1] == "bfloat16"
    assert json.loads(argv[argv.index("--hf-overrides") + 1]) == EXPECTED_SERVE["hf_overrides"]
    assert argv[argv.index("--runner") + 1] == "pooling"


def test_the_reference_environment_is_documented() -> None:
    """The reference declares the environment it needs beside itself (the reference rule, sweep
    finding 10: one ``requirements-reference.txt`` per recipe)."""
    text = (RECIPE_DIR / "requirements-reference.txt").read_text(encoding="utf-8")
    assert "torch==2.9.1" in text
    assert "transformers==4.57.6" in text


def test_the_served_and_reference_prompts_ignore_the_pairs_row_instruction(tmp_path: Path) -> None:
    """``instruction: none`` end to end (sweep items #1 and #2): a pairs row's instruction is ignored
    on the wire and by the reference — the bare query ships, paddings and all, and no side folds
    ``Task: ...\\nQuery: ...``."""
    recipe, tokenizer_file = _local_recipe(tmp_path)
    query = "  padded query  "
    document = "A document."
    shipped = served_pair(recipe, query, [document], instruction="Follow retrieval task 3.")
    assert shipped == {"query": query, "documents": [document]}

    rows = [
        {"query": query, "documents": [document], "instruction": "Follow retrieval task 3."},
        {"query": query, "documents": [document], "instruction": "  "},  # whitespace-only: still ignored
    ]
    pairs_path = tmp_path / "pairs.jsonl"
    pairs_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    reference = run_reference(
        sys.executable,
        str(recipe._dir / recipe.reference.entry),
        mode="render",
        pairs_path=pairs_path,
        out_path=tmp_path / "reference.json",
        tokenizer_spec=str(tokenizer_file),
    )
    for row in reference["rows"]:
        assert row["query"] == query, "the reference must render the bare query, never a fold"
        assert row["documents"] == [document]
    assert REVISION == recipe.revision


def test_the_reference_render_spans_are_the_served_spans_including_over_share_and_over_budget(
    tmp_path: Path,
) -> None:
    """The reference's ``render`` is its own port of the wire's settle rule and cut: byte-equal to
    the client's captured spans (sweep item #5: the query settles at its share whenever it exceeds
    it — the share never binds on overflow only)."""
    from rcp_ndcg.data.tokenizer import load_tokenizer

    recipe, tokenizer_file = _local_recipe(tmp_path)
    tokenizer = load_tokenizer(str(tokenizer_file))
    words: list[str] = []
    long_query = ""
    for _ in range(6000):  # a whitespace-clean query over the share but under the pair budget
        words.append("flibbertigibbet")
        long_query = " ".join(words) + "."
        if tokenizer.count(long_query) > QUERY_MAX_TOKENS + 400:
            break
    rows = [
        {"query": "short query", "documents": ["a short document.", "a second document."]},
        {"query": long_query, "documents": ["a short document under the settled share."]},
        {"query": "short query", "documents": ["parisisthecapitaloffranceandeurope" * 400]},
    ]
    pairs_path = tmp_path / "pairs.jsonl"
    pairs_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    wire = served_rows(recipe, rows, tokenizer)["per_shape"]["pair"]["spans"]
    reference = run_reference(
        sys.executable,
        str(recipe._dir / recipe.reference.entry),
        mode="render",
        pairs_path=pairs_path,
        out_path=tmp_path / "reference.json",
        tokenizer_spec=str(tokenizer_file),
    )
    assert [{"query": row["query"], "documents": list(row["documents"])} for row in reference["rows"]] == wire
    # The settle rule, pinned as the wire carries it: an under-budget pair with an over-share query
    # ships the query settled at its share (the share binds whenever the query exceeds it).
    assert tokenizer.count(long_query) > QUERY_MAX_TOKENS
    assert tokenizer.count(wire[1]["query"]) <= QUERY_MAX_TOKENS
    assert wire[1]["query"] == long_query[: len(wire[1]["query"])]
    # An over-budget pair is cut content-only: the spans re-assemble within the budget, anchor on.
    rendered = FRAME_HEAD + wire[2]["documents"][0] + FRAME_MID + wire[2]["query"] + FRAME_TAIL
    assert tokenizer.count(rendered, add_special_tokens=True) <= MAX_TOKENS
    assert rendered.endswith(FRAME_TAIL)


def test_stage1_passes_on_cpu_with_the_anchor_audit_and_the_render_comparison(tmp_path: Path) -> None:
    """Stage 1 with no engine: the 17 pairs-file rows plus the harness's 5 over-length samples —
    the audit, the template file and the reference render all agree; /tokenize stays ``not_run``."""
    from rcp_ndcg.data.tokenizer import load_tokenizer

    recipe, tokenizer_file = _local_recipe(tmp_path)
    tokenizer = load_tokenizer(str(tokenizer_file))
    pairs_path, rows = _pairs(tmp_path)
    document = stage1_prompts(recipe, pairs_path, sys.executable, over_length_per_shape=5)
    assert document["sampled"] >= 20, document["sampled"]
    anchor = document["anchor_check"]
    assert anchor["passed"] is True, anchor["failures"][:1]
    assert anchor["checked"] == 2 * document["sampled"]  # one settled query + one document span per row
    render = document["render_check"]
    assert render["status"] == "run" and render["passed"] is True, render["failures"][:1]
    assert render["rows"] == len(rows)
    assert document["template_render_check"]["passed"] is True, document["template_render_check"]["failures"][:1]
    engine = document["engine_tokenize_check"]
    assert engine["status"] == "not_run" and engine["passed"] is None  # no engine on CPU: neutral
    assert document["passed"] is True, document
    # The cut facts come from the client's own census (R30): over-length samples, the over-budget
    # pair and the over-share query were cut; an in-budget pair's content never was.
    facts = stage1_facts(recipe, rows, tokenizer, 5)
    assert facts["per_shape"]["pair"]["cut_rows"] >= 5, facts["per_shape"]["pair"]["cut_rows"]
    template = recipe.client.template
    assert template is not None
    assert facts["per_shape"]["pair"]["overhead"] == template.overhead("pair", tokenizer)


def test_dropping_the_trailing_anchor_segment_reddens_the_template_check(tmp_path: Path) -> None:
    """Mutation: dropping the declared template's trailing `` ??`` segment must red the
    file-vs-declaration check (the rerank wire carries content spans only, so the frame contract is
    what ``template_render_check`` pins; the served file keeps emitting the anchor the declaration
    lost)."""
    mutated = tmp_path / "mutant" / RECIPE_ID
    shutil.copytree(RECIPE_DIR, mutated)
    data = yaml.safe_load((mutated / "recipe.yaml").read_text(encoding="utf-8"))
    data["client"]["tokenizer"] = str(_tokenizer_file(tmp_path))
    assert data["client"]["template"]["pair"][-1]["fixed"] == FRAME_TAIL
    data["client"]["template"]["pair"] = data["client"]["template"]["pair"][:-1]
    (mutated / "recipe.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    recipe = load_recipe(mutated)
    pairs_path, _ = _pairs(tmp_path)
    document = stage1_prompts(recipe, pairs_path, sys.executable, over_length_per_shape=1)
    assert document["template_render_check"]["passed"] is False  # the file still emits the " ??"
    assert document["anchor_check"]["passed"] is True  # the span audit does not read the frame
