"""The recipe ``ctxl-rerank-v2-instruct-multilingual-1b``: the full contract, and stage 1 on CPU.

The contract test freezes every ``serve``/``client``/``reference`` field through the shared
:func:`assert_recipe_contract` (nothing rides unpinned), and two mutants show it red on drift.  The
served semantics this family decided are pinned with failing-first tests: ``instruction: none``
(the paper configs' mode — the reference never folds an instruction, a pairs row's instruction is
ignored on both sides) and the merged rerank client's settle rule (the query ships at its declared
share whenever it exceeds it — the pair fit alone binds the share on overflow only).

Stage 1 here runs without an engine: the product's role client ships the request bodies, the
reference subprocess's ``render`` fills the span format with the paper's own spans (the raw query
and documents, uncut -- never a port of the client's cut), which must equal the wire's spans on
every under-cap row while over-cap rows ride the declared ``anchor_drop_over_cap`` table, and the
served template file must render what the declared template renders.  ``tokenizer.json`` downloads
through the shared ``fetch_tokenizer`` (``RCP_NDCG_VLLM_TOKENIZER_CACHE`` when set, else
``tmp_path``; sha256-pinned); those tests are network tests and skip offline
(``tests/recipes/conftest.py``).
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import pytest
import yaml
from rcp_ndcg_vllm import load_recipe, serve_argv
from rcp_ndcg_vllm.equivalence import stage1_prompts
from rcp_ndcg_vllm.equivalence.reference import run_reference

from ._contract import assert_recipe_contract
from ._served import fetch_tokenizer, served_pair, served_rows, stage1_facts

RECIPE_ID = "ctxl-rerank-v2-instruct-multilingual-1b"
REPO = "ContextualAI/ctxl-rerank-v2-instruct-multilingual-1b"
REVISION = "8fd1edf6a98564cb712064f884b8ef7df5c1b876"
RECIPE_DIR = Path(__file__).resolve().parents[2] / "recipes" / RECIPE_ID
TOKENIZER_URL = f"https://huggingface.co/{REPO}/resolve/{REVISION}/tokenizer.json"
TOKENIZER_SHA256 = "f253e845dff94cb1ac558f76905ea5fbe19c21ebf2d9b4e44f28ef0007968267"  # Hub LFS oid at REVISION

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
    """The recipe's ``tokenizer.json`` at the pinned revision, sha256-checked, through the shared
    tokenizer cache (``_served.fetch_tokenizer``); skips with the reason when offline."""
    return fetch_tokenizer(TOKENIZER_URL, f"{RECIPE_ID}@{REVISION}/tokenizer.json", tmp_path, sha256=TOKENIZER_SHA256)


def _long_text(tokenizer, tokens: int) -> str:
    """A whitespace-clean text of more than ``tokens`` tokens, sized from one measured unit (no
    re-count of a growing text)."""
    unit = " flibbertigibbet"
    per_unit = max(1, tokenizer.count(unit * 8) // 8)
    text = "flibbertigibbet" + unit * (tokens // per_unit + 50) + "."
    assert tokenizer.count(text) > tokens
    return text


def _local_recipe(tmp_path: Path) -> tuple:
    """The recipe copied into ``tmp_path`` with ``client.tokenizer`` on the downloaded tokenizer (the
    same bytes, a local spec), so the product and the reference subprocess tokenise offline."""
    copied = tmp_path / RECIPE_ID
    shutil.copytree(RECIPE_DIR, copied)
    data = yaml.safe_load((copied / "recipe.yaml").read_text(encoding="utf-8"))
    data["client"]["tokenizer"] = str(_tokenizer_file(tmp_path))
    (copied / "recipe.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return load_recipe(copied), Path(data["client"]["tokenizer"])


def _pairs(tmp_path: Path, tokenizer) -> tuple[Path, list[dict]]:
    """15 in-budget pairs (four with a run-level instruction, one padded) plus one over-budget pair
    (the declared anchor_drop_over_cap row class); no over-share query (the gating pairs keep queries
    within the share -- the recipe's pairs-file rule)."""
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
    rows.append({"query": "short query", "documents": [_long_text(tokenizer, MAX_TOKENS + 200)]})
    rows.append({"query": "a short query", "documents": ["a document that stays short on purpose."]})
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
    """Mutant 1: ``serve.max_model_len`` 32768 -> 16384 must red, naming the field."""
    mutated = _mutated_recipe(tmp_path, lambda data: data["serve"].__setitem__("max_model_len", 16384))
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


def test_serve_argv_renders_the_golden_engine_command() -> None:
    """The recipe's ``vllm serve`` argv: the revision, the shipped template file and the raw-logit
    pooler default (every recipe test keeps a golden-argv test)."""
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
    """The reference declares the environment it needs beside itself (the reference rule: one
    ``requirements-reference.txt`` per recipe)."""
    text = (RECIPE_DIR / "requirements-reference.txt").read_text(encoding="utf-8")
    assert "torch==2.9.1" in text
    assert "transformers==4.57.6" in text


def test_the_served_and_reference_prompts_ignore_the_pairs_row_instruction(tmp_path: Path) -> None:
    """``instruction: none`` end to end (the family decision): a pairs row's instruction is ignored
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


def test_stage1_passes_on_cpu_with_the_anchor_audit_and_the_render_comparison(tmp_path: Path) -> None:
    """Stage 1 with no engine: the 17 pairs-file rows plus the harness's 5 over-length samples —
    the audit, the template file and the reference render agree on every under-cap row (the
    over-budget row rides the non-gating table); /tokenize stays ``not_run``."""
    from rcp_ndcg.data.tokenizer import load_tokenizer

    recipe, tokenizer_file = _local_recipe(tmp_path)
    tokenizer = load_tokenizer(str(tokenizer_file))
    pairs_path, rows = _pairs(tmp_path, tokenizer)
    document = stage1_prompts(recipe, pairs_path, sys.executable, over_length_per_shape=5)
    assert document["sampled"] >= 20, document["sampled"]
    anchor = document["anchor_check"]
    assert anchor["passed"] is True, anchor["failures"][:1]
    assert anchor["checked"] == 2 * document["sampled"]  # one settled query + one document span per row
    render = document["render_check"]
    assert render["status"] == "run" and render["passed"] is True, render["failures"][:1]
    assert render["rows"] == len(rows)
    # The one over-budget pairs row differs by declaration (the paper's uncut spans vs the client's
    # cut): reported in the non-gating table, never gated; every other row compared exactly.
    over_cap = render["over_cap"]
    assert over_cap["known_deviation"] is True and over_cap["gating"] is False
    assert [entry["index"] for entry in over_cap["rows"]] == [15]
    assert document["template_render_check"]["passed"] is True, document["template_render_check"]["failures"][:1]
    engine = document["engine_tokenize_check"]
    assert engine["status"] == "not_run" and engine["passed"] is None  # no engine on CPU: neutral
    assert document["passed"] is True, document
    # The cut facts come from the client's own census (never re-derived by the test): the over-length samples and the
    # over-budget pair were cut; an in-budget pair's content never was.
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
    from rcp_ndcg.data.tokenizer import load_tokenizer

    mutated = tmp_path / "mutant" / RECIPE_ID
    shutil.copytree(RECIPE_DIR, mutated)
    data = yaml.safe_load((mutated / "recipe.yaml").read_text(encoding="utf-8"))
    data["client"]["tokenizer"] = str(_tokenizer_file(tmp_path))
    assert data["client"]["template"]["pair"][-1]["fixed"] == FRAME_TAIL
    data["client"]["template"]["pair"] = data["client"]["template"]["pair"][:-1]
    (mutated / "recipe.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    recipe = load_recipe(mutated)
    pairs_path, _ = _pairs(tmp_path, load_tokenizer(data["client"]["tokenizer"]))
    document = stage1_prompts(recipe, pairs_path, sys.executable, over_length_per_shape=1)
    assert document["template_render_check"]["passed"] is False  # the file still emits the " ??"
    assert document["anchor_check"]["passed"] is True  # the span audit does not read the frame


#: Internal process labels that must not ship in a recipe (review shorthand, private work
#: directories, rule ids no public document defines). Public rule ids (R29, documented in
#: docs/how-to/add-a-model.md) stay allowed.
INTERNAL_LABELS = re.compile(
    r"p1-tail|fam-(?:dense|ctxl)|\bsweep|lanes' base|audit-synth|\br-(?:ctxl|jina[35]|octen|zembed1|qwen3-emb)\b"
    r"|\bresearch\b|\blanes?\b|REVIEW-LOG|ANCHOR-FINDING|\bR(?!29\b)\d{1,2}\b|clients-final"
    r"|\boperator\b|\b09x\b|\.refs/|recipe-common|corrections table|\bfinding #?\d"
)


@pytest.mark.parametrize(
    "recipe_id",
    [
        "ctxl-rerank-v2-instruct-multilingual-1b",
        "ctxl-rerank-v2-instruct-multilingual-2b",
        "ctxl-rerank-v2-instruct-multilingual-6b",
    ],
)
def test_shipped_recipe_files_carry_no_internal_labels(recipe_id: str) -> None:
    """Every shipped file of the ctxl recipes reads as a self-contained public statement: no
    internal process shorthand, private work directory or undefined rule id."""
    hits = [
        f"{path.name}:{number}: {line.strip()[:120]}"
        for path in sorted((RECIPE_DIR.parent / recipe_id).iterdir())
        if path.is_file()
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if INTERNAL_LABELS.search(line)
    ]
    assert not hits, "\n".join(hits)
