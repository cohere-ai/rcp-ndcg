"""The zerank-1-reranker recipe: the declared contract, stage 1 on CPU, and the frame mutation.

The contract test pins every resolved ``serve``/``client``/``reference`` field through the recipe
lanes' shared helper (``_contract.assert_recipe_contract``), with a two-mutant negative control: a
serve or reference field that drifts reds, naming the field path. Stage 1 runs the harness's own
machinery on the real tokenizer (downloaded into ``RCP_NDCG_VLLM_TOKENIZER_CACHE`` when set, else
``tmp_path``, pinned by SHA-256; public Hub file, never a token file), and the reference's
tokenizer-only ``--mode render`` runs here: it writes the paper's own cut (the whole rendered prompt
right-cut at 8192 tokens) in the harness's span format -- never the client's cut (decision 9), so
under-cap rows equal the wire byte for byte and over-cap rows are the declared ``anchor_drop_over_cap``,
reported non-gating. The paper's ``query.strip()``/``doc.strip()`` is the recipe's declared
normalisation (``normalize: [strip]``), never a jinja trim filter. The paper-exact score mode needs
the reference environment and the GPU wave.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
from rcp_ndcg_vllm import RecipeError, load_recipe, serve_argv
from rcp_ndcg_vllm.equivalence import stage1_prompts

from ._contract import assert_recipe_contract
from ._served import fetch_tokenizer, served_pair, served_rows

RECIPE_DIR = Path(__file__).resolve().parents[2] / "recipes" / "zerank-1-reranker"
REVISION = "d03c467e29e29c0a16a130a86ce3b62d30116a2c"
REPO = "zeroentropy/zerank-1-reranker"
TOKENIZER_URL = f"https://huggingface.co/{REPO}/resolve/{REVISION}/tokenizer.json"
TOKENIZER_SHA256 = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"

_QUERY = "What is the capital of France?"
_DOCUMENT = "Paris is the capital and the largest city of France."

#: The recipe's full resolved contract: every field of every block, exactly as the product models
#: resolve it (authored values and schema defaults alike). Nothing may ride unpinned.
CONTRACT: dict[str, Any] = {
    "serve": {
        "runner": "pooling",
        "convert": None,
        "hf_overrides": {
            "architectures": ["Qwen3ForSequenceClassification"],
            "classifier_from_token": ["Yes"],
            "method": "no_post_processing",
        },
        "chat_template": "template.jinja",
        "pooler_config": {"logit_sigma": 5, "use_activation": True},
        "trust_remote_code": False,
        "max_model_len": 32768,
        "dtype": "bfloat16",
        "plugin": None,
        "io_processor_plugin": None,
        "mm_processor_kwargs": {},
        "limit_mm_per_prompt": None,
        "extra_args": [],
    },
    "client": {
        "api": "rerank",
        "model": "zerank-1-reranker",
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
        "video_policy": None,
        "max_images": 0,
        "max_videos": 0,
        "media_sides": ["query", "document"],
        "recipe": (
            "vllm v0.31.0: --runner pooling, hf_overrides Qwen3ForSequenceClassification + "
            "classifier_from_token [Yes] + method no_post_processing, --chat-template template.jinja, "
            "pooler logit_sigma 5 + use_activation true (sigmoid(l_Yes/5) at the last token, 1-label head)"
        ),
        "tokenizer": f"{REPO}@{REVISION}",
        "max_tokens": 8192,
        "instruction": "none",
        "use_activation": True,
        "query_max_tokens": 4096,
        "document_max_tokens": None,
        "template": {
            "query": None,
            "document": None,
            "pair": [
                {"fixed": "{special:im_start}system\n", "content": None},
                {"fixed": None, "content": "query"},
                {"fixed": "{special:im_end}\n{special:im_start}user\n", "content": None},
                {"fixed": None, "content": "document"},
                {"fixed": "{special:im_end}\n{special:im_start}assistant\n", "content": None},
            ],
            "anchor": "last",
            "anchor_markers": [],
            "add_special_tokens": True,
            "normalize": ["strip"],
        },
        "on_overflow": "cut",
        "chunk": None,
        "aggregation": "max",
        "empty_doc": "send",
        "empty_doc_text": None,
        "empty_query": "send",
        "request_shape": "text",
        "listwise": False,
        "batch_size": None,
    },
    "reference": {
        "kind": "transformers",
        "score_scale": "probability",
        "entry": "reference.py",
        "known_deviations": ["anchor_drop_over_cap"],
    },
}

TOP = {
    "id": "zerank-1-reranker",
    "model": REPO,
    "revision": REVISION,
    "role": "rerank",
    "input": ["text"],
    "scoring": "pointwise",
    "licence": "apache-2.0",
}


def _tokenizer_file(tmp_path: Path) -> Path:
    """The pinned tokenizer.json in the shared cache (SHA-256 checked; downloaded there, nowhere else)."""
    return fetch_tokenizer(TOKENIZER_URL, f"{RECIPE_DIR.name}-tokenizer.json", tmp_path, sha256=TOKENIZER_SHA256)


def _recipe_and_tokenizer(tmp_path: Path) -> tuple[Any, Any]:
    """The committed recipe bound to the downloaded tokenizer file, plus the product's loaded tokenizer.

    The recipe pins the Hub spec (what production resolves); the tests run on the same tokenizer.json
    over a local file, so no test reaches the Hub after the one pinned download (offline-safe).
    """
    from rcp_ndcg.data.tokenizer import load_tokenizer

    file = _tokenizer_file(tmp_path)
    recipe = load_recipe(RECIPE_DIR)
    client = recipe.client.model_copy(update={"tokenizer": str(file)})
    return recipe.model_copy(update={"client": client}), load_tokenizer(str(file))


def _render_engine_shape(template_text: str, query: str, document: str) -> str:
    """The template file rendered the engine's way: ``messages`` with the query/document roles.

    The engine's jinja environment is transformers': trim_blocks and lstrip_blocks on, no
    StrictUndefined (the conditional in the template file takes the messages branch there).
    """
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=False)
    return environment.from_string(template_text).render(
        messages=[{"role": "query", "content": query}, {"role": "document", "content": document}], tools=None
    )


def _render_harness_shape(template_text: str, query: str, document: str) -> str:
    """The template file rendered the harness's way (the stage-1 template check's settings)."""
    from jinja2 import StrictUndefined
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    environment = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=False, undefined=StrictUndefined
    )
    return environment.from_string(template_text).render(query=query, document=document, instruction="")


def _paper_render(tokenizer: Any, query: str, document: str) -> str:
    """The paper's prompt construction for one pair (``ZerankRerank._format_inputs``): the ChatML
    frame with the stripped query as the system turn and the stripped document as the user turn,
    ending with the assistant header whose newline is the last-token anchor. The specials come from
    the tokenizer by name - never typed literally here."""
    im_start = tokenizer.special_text("im_start")
    im_end = tokenizer.special_text("im_end")
    return (
        f"{im_start}system\n{query.strip()}{im_end}\n{im_start}user\n{document.strip()}{im_end}\n{im_start}assistant\n"
    )


def _sample_pairs() -> list[dict[str, Any]]:
    """The sampled pairs: 20 in-budget rows and 5 over-budget rows (a long document, queries past
    the declared share, unicode and CJK), plus one whitespace-padded row and one empty-document row
    for the declared normalisation and ``empty_doc: send``. An over-share query in an UNDER-budget
    pair gates red by design (the reference never ports the settle rule): it has its own test."""
    rows: list[dict[str, Any]] = [
        {
            "query": f"query {index}: what does the reranker read",
            "documents": [f"document {index} about retrieval, rerankers and the tokens they score"],
        }
        for index in range(20)
    ]
    rows[0] = {"query": _QUERY, "documents": [_DOCUMENT]}  # the card's own shape, natural text
    rows.extend(
        [
            # document over the pair budget: the document span is cut, the anchor kept
            {"query": _QUERY, "documents": ["parisisthecapitaloffranceandeurope" * 3100]},
            # query over the share AND the pair over budget even after the settle: the query is cut to its
            # share first, then the document to what remains (the harness's over-cap census needs that
            # document cut: a pair that only the settle brings under budget is classified under-cap)
            {"query": "alphagammaepsilon" * 1400, "documents": ["thequickbrownfoxjumpsover" * 800]},
            # unicode, emoji and CJK over the cap
            {"query": _QUERY, "documents": ["北京上海广州深圳🦜\U0001f600" * 2400]},
            # a second query-over-share row, over the cap: the document span gets what remains
            {"query": "thequickbrownfoxjumpsover" * 1400, "documents": [_DOCUMENT]},
            # a long unit far over the cap: the whole budget is spent on the cut document
            {"query": _QUERY, "documents": ["thetengreensfrogsjumpovertheriver" * 1500]},
            # whitespace-padded: the declared normalisation (strip) runs on both sides
            {"query": "  padded query \n\t", "documents": ["\n leading document "]},
            # an empty document as given (empty_doc: send)
            {"query": "empty document query", "documents": [""]},
        ]
    )
    return rows


def _write_pairs(tmp_path: Path, rows: list[dict[str, Any]]) -> Path:
    """The sampled rows as the pairs JSONL file."""
    pairs_path = tmp_path / "pairs.jsonl"
    pairs_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return pairs_path


@pytest.mark.network
def test_recipe_contract_pins_every_field() -> None:
    """Every resolved serve/client/reference field (and every top-level fact) is pinned exactly."""
    recipe = load_recipe(RECIPE_DIR)
    assert_recipe_contract(
        recipe, serve=CONTRACT["serve"], client=CONTRACT["client"], reference=CONTRACT["reference"], top=TOP
    )
    assert recipe.serve.max_model_len >= recipe.client.max_tokens
    assert (RECIPE_DIR / "template.jinja").is_file()
    assert (RECIPE_DIR / "requirements-reference.txt").is_file()
    assert recipe.sources


def test_the_contract_reds_on_two_mutants() -> None:
    """Two mutants of the declared contract must red the pin (the sweep's surviving mutants)."""
    recipe = load_recipe(RECIPE_DIR)
    serve_mutant = recipe.model_copy(update={"serve": recipe.serve.model_copy(update={"max_model_len": 40960})})
    with pytest.raises(AssertionError, match="max_model_len"):
        assert_recipe_contract(
            serve_mutant, serve=CONTRACT["serve"], client=CONTRACT["client"], reference=CONTRACT["reference"], top=TOP
        )
    reference_mutant = recipe.model_copy(
        update={"reference": recipe.reference.model_copy(update={"kind": "sentence_transformers"})}
    )
    with pytest.raises(AssertionError, match="kind"):
        assert_recipe_contract(
            reference_mutant,
            serve=CONTRACT["serve"],
            client=CONTRACT["client"],
            reference=CONTRACT["reference"],
            top=TOP,
        )


def test_serve_argv_matches_the_measured_vllm_invocation() -> None:
    """The rendered argv is the research's verified serve command (plus --revision, from the recipe)."""
    recipe = load_recipe(RECIPE_DIR)
    argv = serve_argv(recipe, port=8100, served_model_name=recipe.id)
    assert argv[:3] == ["vllm", "serve", f"{REPO}"]
    for flag, value in {
        "--revision": REVISION,
        "--runner": "pooling",
        "--dtype": "bfloat16",
        "--max-model-len": "32768",
        "--tensor-parallel-size": "1",
    }.items():
        assert argv[argv.index(flag) + 1] == value, flag
    assert "--convert" not in argv, "a rerank recipe never flags --convert"
    overrides = json.loads(argv[argv.index("--hf-overrides") + 1])
    assert overrides == {
        "architectures": ["Qwen3ForSequenceClassification"],
        "classifier_from_token": ["Yes"],
        "method": "no_post_processing",
    }
    assert json.loads(argv[argv.index("--pooler-config") + 1]) == {"logit_sigma": 5, "use_activation": True}
    assert argv[argv.index("--chat-template") + 1].endswith("template.jinja")


@pytest.mark.network
def test_template_file_renders_both_call_shapes_to_the_same_ids(tmp_path: Path) -> None:
    """The served template file renders the wire's spans: engine way and harness way, per sampled row.

    For every uncut row the fit render (the wire's stripped spans inside the frame) and the paper's
    construction are byte-identical. For cut rows the spans are the wire's own and the file renders
    them (the frame is the engine's). The file renders raw input raw: the strip is the declared
    normalisation on the wire side, never a jinja filter."""
    recipe, tokenizer = _recipe_and_tokenizer(tmp_path)
    template_text = (RECIPE_DIR / "template.jinja").read_text(encoding="utf-8")
    assert "| trim" not in template_text
    template = recipe.client.template
    assert template is not None and template.normalisers("pair") == ("strip",)
    anchor_tail = tokenizer.special_text("im_start") + "assistant\n"
    shipped_rows = served_rows(recipe, _sample_pairs(), tokenizer)["per_shape"]["pair"]["spans"]
    checked_in_budget = 0
    for row, shipped in zip(_sample_pairs(), shipped_rows, strict=True):
        fitted_query, fitted_document = shipped["query"], shipped["documents"][0]
        fit_text = template.render("pair", tokenizer, query=fitted_query, document=fitted_document)
        flag = template.adds_special_tokens("pair")
        assert tokenizer.count(fit_text, add_special_tokens=flag) <= recipe.client.max_tokens
        assert fit_text.endswith(anchor_tail)
        engine_render = _render_engine_shape(template_text, fitted_query, fitted_document)
        assert engine_render == fit_text, row["query"][:40]
        if not (fitted_query != row["query"].strip() or fitted_document != row["documents"][0].strip()):
            # Uncut: the paper's construction on the raw input equals the served render byte for byte
            # (the declared normalisation IS the paper's strip).
            paper = _paper_render(tokenizer, row["query"], row["documents"][0])
            assert paper == fit_text and paper == engine_render, row["query"][:40]
            checked_in_budget += 1
    assert checked_in_budget >= 20  # the brief's floor: at least 20 sampled pairs with full equality


@pytest.mark.network
def test_stage1_on_cpu_passes_the_anchor_template_and_render_checks(tmp_path: Path) -> None:
    """Stage 1 on CPU: the anchor audit, the served-template check and the reference render over
    >= 20 pairs (>= 5 over cap), all green; the /tokenize check is ``not_run`` without an engine."""
    recipe, _ = _recipe_and_tokenizer(tmp_path)
    rows = _sample_pairs()
    assert len(rows) >= 20
    max_tokens = recipe.client.max_tokens or 0
    assert max_tokens == 8192  # pinned here too: the over-cap floor below depends on it
    document = stage1_prompts(recipe, _write_pairs(tmp_path, rows), sys.executable, over_length_per_shape=5)
    assert document["sampled"] >= 25
    assert document["anchor_check"]["passed"] is True, document["anchor_check"]["failures"][:2]
    assert document["template_render_check"] is not None
    assert document["template_render_check"]["passed"] is True, document["template_render_check"]["failures"][:2]
    render_check = document["render_check"]
    assert render_check["status"] == "run" and render_check["passed"] is True, render_check["failures"][:2]
    assert render_check["rows"] == len(rows)
    assert document["engine_tokenize_check"]["status"] == "not_run"  # no engine on CPU; never reported passed


def _reference_render(tmp_path: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The reference subprocess's ``--mode render`` rows for ``rows`` (tokenizer only, no torch)."""
    out_path = tmp_path / "reference.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(RECIPE_DIR / "reference.py"),
            "--mode",
            "render",
            "--pairs",
            _write_pairs(tmp_path, rows),
            "--out",
            str(out_path),
            "--tokenizer",
            str(_tokenizer_file(tmp_path)),
            "--device",
            "cpu",
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stderr[-500:]
    return json.loads(out_path.read_text(encoding="utf-8"))["rows"]


@pytest.mark.network
def test_the_reference_renders_the_papers_cut_never_the_clients(tmp_path: Path) -> None:
    """Decision 9 on this reference: under the cap its spans are the wire's, byte for byte (the paper's
    strip is the declared normalisation); over the cap they are the paper's own cut -- the whole rendered
    prompt right-cut at 8192 tokens, the assistant header (the anchor) dropped -- never the client's
    anchor-preserving cut, and stage 1 reports those rows non-gating under ``anchor_drop_over_cap``."""
    recipe, tokenizer = _recipe_and_tokenizer(tmp_path)
    rows: list[dict[str, Any]] = [
        {"query": _QUERY, "documents": [_DOCUMENT, "  Lyon is a city in France.  "]},
        {"query": _QUERY, "documents": ["parisisthecapitaloffranceandeurope" * 3100]},
    ]
    reference = _reference_render(tmp_path, rows)
    served = served_rows(recipe, rows, tokenizer)["per_shape"]["pair"]["spans"]
    assert {"query": reference[0]["query"], "documents": reference[0]["documents"]} == served[0]
    assert served[0] == {"query": _QUERY, "documents": [_DOCUMENT, "Lyon is a city in France."]}
    # over the cap: the paper's prompt, right-cut at the budget -- the anchor tail is gone
    paper = _paper_render(tokenizer, rows[1]["query"], rows[1]["documents"][0])
    kept = _paper_render(tokenizer, reference[1]["query"], reference[1]["documents"][0])
    tail = tokenizer.special_text("im_end") + "\n" + tokenizer.special_text("im_start") + "assistant\n"
    assert kept.endswith(tail)  # the frame helper re-attaches it: strip it to compare with the cut
    assert tokenizer.ids(kept[: -len(tail)]) == tokenizer.ids(paper)[: recipe.client.max_tokens]
    # the client reserves the anchor and cuts the document shorter: the two cuts differ by declaration
    assert reference[1]["documents"][0].startswith(served[1]["documents"][0])
    assert len(served[1]["documents"][0]) < len(reference[1]["documents"][0])
    document = stage1_prompts(recipe, _write_pairs(tmp_path, rows), sys.executable, over_length_per_shape=1)
    assert document["render_check"]["passed"] is True, document["render_check"]["failures"][:1]
    assert document["render_check"]["over_cap"]["known_deviation"] is True
    assert {row["index"] for row in document["render_check"]["over_cap"]["rows"]} == {1}


@pytest.mark.network
def test_an_over_share_query_under_the_budget_is_reported(tmp_path: Path) -> None:
    """A query the client settles at its share in a pair the budget takes whole is a change the client made
    (decision 9): the paper keeps it whole and the reference never ports the settle rule, so under the declared
    over-cap deviation the row's query span is reported, not gated -- and the report names ``query_share``."""
    recipe, tokenizer = _recipe_and_tokenizer(tmp_path)
    rows = [{"query": "alphagammaepsilon" * 1200, "documents": [_DOCUMENT]}]
    query_tokens = tokenizer.count(rows[0]["query"])
    assert (recipe.client.query_max_tokens or 0) < query_tokens < (recipe.client.max_tokens or 0) - 100
    reference = _reference_render(tmp_path, rows)
    assert reference[0]["query"] == rows[0]["query"]  # the paper keeps it whole
    document = stage1_prompts(recipe, _write_pairs(tmp_path, rows), sys.executable, over_length_per_shape=1)
    render = document["render_check"]
    assert render["passed"] is True, render["failures"][:1]
    (reported,) = render["over_cap"]["rows"]
    assert reported["index"] == 0 and [m["span"] for m in reported["mismatches"]] == ["query"]
    assert [change["mechanisms"] for change in reported["changes"]] == [["query_share"]]


@pytest.mark.network
def test_padded_inputs_align_through_the_declared_normalisation(tmp_path: Path) -> None:
    """Whitespace-padded inputs: the paper's ``.strip()`` is the DECLARED normalisation
    (``normalize: [strip]``) for the whole family.  The wire's spans carry the stripped texts; the
    template file renders raw input raw (no jinja trim filter anywhere); and the paper's
    construction on the padded input equals the served render byte for byte."""
    recipe, tokenizer = _recipe_and_tokenizer(tmp_path)
    template_text = (RECIPE_DIR / "template.jinja").read_text(encoding="utf-8")
    padded_query, padded_document = "  padded query \n\t", "\n leading document "
    shipped = served_pair(recipe, padded_query, [padded_document])
    assert shipped == {"query": "padded query", "documents": ["leading document"]}
    template = recipe.client.template
    assert template is not None
    fit_text = template.render("pair", tokenizer, query="padded query", document="leading document")
    assert fit_text == _paper_render(tokenizer, padded_query, padded_document)
    # the template file carries no trim filter: it renders input raw, the declared render likewise
    assert "| trim" not in template_text
    assert _render_harness_shape(template_text, padded_query, padded_document) != fit_text


@pytest.mark.network
def test_dropping_the_trailing_anchor_segment_turns_the_template_check_red(tmp_path: Path) -> None:
    """The mutation the brief asks for: the trailing anchor segment (the assistant header the score
    is pooled from) dropped from the declared template - the file-vs-declaration check must red.

    The rerank wire carries the cut CONTENT spans (the frame is the engine's own template), so the
    anchor audit audits the settled query and the document spans and moves no frame; the frame
    contract is ``template_render_check`` (the declared shape must end with the header the file
    emits), which this mutation turns red."""
    recipe, tokenizer = _recipe_and_tokenizer(tmp_path)
    mutated = tmp_path / "zerank-1-reranker"
    mutated.mkdir()
    for name in ("recipe.yaml", "template.jinja", "reference.py", "requirements-reference.txt"):
        shutil.copy(RECIPE_DIR / name, mutated / name)
    data = yaml.safe_load((mutated / "recipe.yaml").read_text(encoding="utf-8"))
    data["client"]["tokenizer"] = str(_tokenizer_file(tmp_path))  # same pinned tokenizer, no Hub at run time
    segments = data["client"]["template"]["pair"]
    assert segments[-1]["fixed"].endswith("assistant\n")  # the anchor being dropped
    data["client"]["template"]["pair"] = segments[:-1]  # drop the trailing anchor segment entirely
    (mutated / "recipe.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    recipe = load_recipe(mutated)  # still loadable: add_special_tokens allows a content tail
    assert len(recipe.client.template.segments("pair")) == 4
    document = stage1_prompts(
        recipe, _write_pairs(tmp_path, _sample_pairs()[:3]), sys.executable, over_length_per_shape=1
    )
    assert document["template_render_check"]["passed"] is False, "the dropped anchor must fail the frame check"
    assert document["template_render_check"]["failures"], "the check must name the failing renders"
    assert document["anchor_check"]["passed"] is True  # the spans themselves are unchanged


def test_a_recipe_whose_template_file_is_missing_is_refused(tmp_path: Path) -> None:
    """R10: the template file must ship - the schema refuses a recipe naming a missing file, because
    vLLM's rerank path degrades silently (pair-encode, no warning for this architecture) without it."""
    broken = tmp_path / "zerank-1-reranker"
    broken.mkdir()
    for name in ("recipe.yaml", "reference.py"):
        shutil.copy(RECIPE_DIR / name, broken / name)
    with pytest.raises(RecipeError, match="chat_template"):
        load_recipe(broken)
