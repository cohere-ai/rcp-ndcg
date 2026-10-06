"""The zerank-1-reranker recipe: validation, stage 1 on CPU, and the anchor mutation.

Stage 1 on CPU needs only the recipe tokenizer's ``tokenizer.json`` (11 MB), downloaded from the Hub
at the revision the recipe pins, into the test's own tmp directory; when the Hub is unreachable
(offline CI) the tokenizer-backed tests skip with that reason. The paper-exact reference subprocess
(``--mode render``, stage 1's reference side) needs the recipe's reference environment
(``requirements-reference.txt`` in the recipe directory); the test runs it when
``RCP_ZERANK_REFERENCE_PYTHON`` names such an interpreter and reports the skip reason otherwise.

The template file is the engine's own jinja (vLLM ``--chat-template``): it must render the declared
pair shape's ids both the engine's way (``messages=[{role: query}, {role: document}]``, as
``CrossEncoderIOProcessor.get_score_prompt`` calls it) and the harness's way (the query/document
variables, StrictUndefined), and both must equal the paper's prompt construction (the checkpoint's
own chat template over ``[{system: query.strip()}, {user: doc.strip()}]`` with
``add_generation_prompt=True``) - the fidelity the r-zerank1 research measured at 21/21.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
from pathlib import Path
from typing import Any

import pytest
import yaml
from rcp_ndcg_vllm import RecipeError, load_recipe, serve_argv
from rcp_ndcg_vllm.equivalence import stage1_prompts

from rcp_ndcg.data.preprocess import TextBudget, fit
from rcp_ndcg.data.tokenizer import TextTokenizer, load_tokenizer

RECIPE_DIR = Path(__file__).resolve().parents[2] / "recipes" / "zerank-1-reranker"
REVISION = "d03c467e29e29c0a16a130a86ce3b62d30116a2c"
TOKENIZER_SPEC = f"zeroentropy/zerank-1-reranker@{REVISION}"

_QUERY = "What is the capital of France?"
_DOCUMENT = "Paris is the capital and the largest city of France."


def _tokenizer(monkeypatch: Any, tmp_path: Any) -> TextTokenizer:
    """The recipe tokenizer, loaded through the product's own loader at the recipe's declared spec
    (``repo@revision``), with the Hub cache pointed into the test's tmp directory so nothing lands
    outside it (the reference subprocess inherits the environment). Skips with a clear reason when
    the Hub is unreachable (offline CI) or ``huggingface_hub`` is missing."""
    try:
        import huggingface_hub  # noqa: F401 - the import is the availability probe
    except ModuleNotFoundError as error:
        pytest.skip(f"the Hub download needs huggingface_hub: {error}")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    old_timeout = socket.getdefaulttimeout()
    try:
        socket.setdefaulttimeout(15.0)
        return load_tokenizer(TOKENIZER_SPEC)
    except Exception as error:  # noqa: BLE001 - any Hub failure means the same skip
        pytest.skip(f"offline: downloading the recipe tokenizer failed ({type(error).__name__}: {error})")
    finally:
        socket.setdefaulttimeout(old_timeout)


def _budget(recipe: Any, tokenizer: TextTokenizer) -> TextBudget:
    """The recipe's declared pair budget, on the given tokenizer (the served client's fit call)."""
    client = recipe.client
    return TextBudget(
        tokenizer=tokenizer.name,
        max_tokens=client.max_tokens,
        query_max_tokens=client.query_max_tokens,
        template=client.template,
        on_overflow=client.on_overflow,
    )


def _fit_pair(recipe: Any, tokenizer: TextTokenizer, query: str, document: str) -> Any:
    """The product's fit of one pair (the served client's render, contents and census)."""
    return fit([(query, document)], "pair", _budget(recipe, tokenizer), tokenizer, ids=["0"])


def _render_engine_shape(template_text: str, query: str, document: str) -> str:
    """The template file rendered the engine's way: ``messages`` with the query/document roles.

    The engine's jinja environment is transformers': trim_blocks and lstrip_blocks on, no
    StrictUndefined (vLLM's rerank path passes no query/document variables - the conditional in the
    template file must take the messages branch there, and the query branch under the harness's
    StrictUndefined check).
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


def _paper_render(tokenizer: TextTokenizer, query: str, document: str) -> str:
    """The paper's prompt construction for one pair (``ZerankRerank._format_inputs``): the ChatML
    frame with the stripped query as the system turn and the stripped document as the user turn,
    ending with the assistant header whose newline is the last-token anchor. The specials come from
    the tokenizer by name - never typed literally here."""
    im_start = tokenizer.special_text("im_start")
    im_end = tokenizer.special_text("im_end")
    return (
        f"{im_start}system\n{query.strip()}{im_end}\n{im_start}user\n{document.strip()}{im_end}\n{im_start}assistant\n"
    )


def _sample_pairs(tokenizer: TextTokenizer) -> list[dict[str, Any]]:
    """The sampled pairs: 20 whitespace-clean in-budget rows and 5 over-budget rows (long documents,
    a long query past the declared share, unicode and CJK), sized against the tokenizer so they
    really cross the cap. Every over-cap row's text is one glued unit with no whitespace, so a
    token-boundary cut never leaves edge whitespace and the template file's trim stays an identity
    on the fitted contents."""
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
            # query over the declared share with a short document: the pair still fits the budget whole
            {"query": "alphagammaepsilon" * 1200, "documents": [_DOCUMENT]},
            # query over the share AND the pair over budget: the query is cut to its share first
            {"query": "alphagammaepsilon" * 1400, "documents": ["thequickbrownfoxjumpsover" * 400]},
            # unicode, emoji and CJK over the cap (glued, so cuts never leave edge whitespace)
            {"query": _QUERY, "documents": ["北京上海广州深圳🦜\U0001f600" * 2400]},
            # a second query-over-share row, over the cap: the document span gets what remains
            {"query": "thequickbrownfoxjumpsover" * 1400, "documents": [_DOCUMENT]},
            # a long unit far over the cap: the whole budget is spent on the cut document
            {"query": _QUERY, "documents": ["thetengreensfrogsjumpovertheriver" * 1500]},
        ]
    )
    return rows


def _write_pairs(tmp_path: Any, rows: list[dict[str, Any]]) -> Path:
    """The sampled rows as the pairs JSONL file."""
    pairs_path = tmp_path / "pairs.jsonl"
    pairs_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return pairs_path


def test_recipe_validates_and_pins_the_researched_serve_shape() -> None:
    """The recipe loads through the product's endpoint config and carries the measured serve shape."""
    recipe = load_recipe(RECIPE_DIR)
    assert recipe.id == "zerank-1-reranker" == RECIPE_DIR.name
    assert recipe.model == "zeroentropy/zerank-1-reranker"
    assert recipe.revision == REVISION
    assert recipe.role == "rerank" and recipe.scoring == "pointwise"
    assert recipe.licence == "apache-2.0"
    from rcp_ndcg.inference.config import RerankEndpoint

    assert isinstance(recipe.client, RerankEndpoint)
    assert recipe.client.tokenizer == TOKENIZER_SPEC
    assert recipe.client.max_tokens == 8192
    assert recipe.client.query_max_tokens == 4096
    assert recipe.client.on_overflow == "cut"
    assert recipe.client.empty_doc == "send"
    assert recipe.client.instruction == "none"
    assert recipe.client.use_activation is True  # probability-scale scores; explicit, never the default
    assert recipe.client.listwise is False
    template = recipe.client.template
    assert template.shapes() == ("pair",) and template.anchor == "last"
    assert template.adds_special_tokens("pair") is True
    assert recipe.serve.runner == "pooling" and recipe.serve.convert == "classify"
    assert recipe.serve.chat_template == "template.jinja"
    assert recipe.serve.pooler_config == {"logit_sigma": 5}
    assert recipe.serve.trust_remote_code is False
    assert recipe.serve.dtype == "bfloat16"
    assert recipe.serve.max_model_len == 32768 >= recipe.client.max_tokens
    assert recipe.serve.plugin is None  # the architecture resolves natively: no plugin wheel
    assert recipe.serve.mm_processor_kwargs == {}  # text-only model: no media policy to pin
    assert recipe.reference.known_deviations == ["anchor_drop_over_cap"]
    assert recipe.reference.score_scale == "probability"
    assert (RECIPE_DIR / "template.jinja").is_file()
    assert (RECIPE_DIR / "requirements-reference.txt").is_file()
    assert recipe.sources  # every URL and path:line the recipe rests on is listed


def test_serve_argv_matches_the_measured_vllm_invocation() -> None:
    """The rendered argv is the research's verified serve command (plus --revision, from the recipe)."""
    recipe = load_recipe(RECIPE_DIR)
    argv = serve_argv(recipe, port=8100, served_model_name=recipe.id)
    assert argv[:3] == ["vllm", "serve", "zeroentropy/zerank-1-reranker"]
    for flag, value in {
        "--revision": REVISION,
        "--runner": "pooling",
        "--convert": "classify",
        "--dtype": "bfloat16",
        "--max-model-len": "32768",
        "--tensor-parallel-size": "1",
    }.items():
        assert argv[argv.index(flag) + 1] == value, flag
    overrides = json.loads(argv[argv.index("--hf-overrides") + 1])
    assert overrides == {
        "architectures": ["Qwen3ForSequenceClassification"],
        "classifier_from_token": ["Yes"],
        "method": "no_post_processing",
    }
    assert json.loads(argv[argv.index("--pooler-config") + 1]) == {"logit_sigma": 5}
    assert argv[argv.index("--chat-template") + 1].endswith("template.jinja")


def test_template_file_renders_both_call_shapes_to_the_same_ids(tmp_path: Any, monkeypatch: Any) -> None:
    """The served template file renders the declared shape's ids, both call shapes, per sampled row.

    For in-budget rows the declared shape's fit render, the file's engine render, the file's harness
    render and the paper's construction must be byte-identical. For over-budget rows the client cuts
    the content spans first; the file render of the fitted contents must still equal the fit render
    (the anchor re-attached, the budget held), and the harness render and the paper construction are
    then only checked on the under-cap rows they are about."""
    recipe = load_recipe(RECIPE_DIR)
    tokenizer = _tokenizer(monkeypatch, tmp_path)
    template_text = (RECIPE_DIR / "template.jinja").read_text(encoding="utf-8")
    anchor_tail = tokenizer.special_text("im_start") + "assistant\n"
    checked_in_budget = 0
    for row in _sample_pairs(tokenizer):
        result = _fit_pair(recipe, tokenizer, row["query"], row["documents"][0])
        fitted_query, fitted_document = result.contents[0]
        fit_text = result.texts[0]
        flag = recipe.client.template.adds_special_tokens("pair")
        # The declared budget holds on the assembled render, and the anchor survived every cut.
        assert tokenizer.count(fit_text, add_special_tokens=flag) <= recipe.client.max_tokens
        assert fit_text.endswith(anchor_tail)
        engine_render = _render_engine_shape(template_text, fitted_query, fitted_document)
        assert engine_render == fit_text, row["query"][:40]
        content_tokens = tokenizer.count(fitted_query) + tokenizer.count(fitted_document)
        raw_tokens = tokenizer.count(row["query"]) + tokenizer.count(row["documents"][0])
        if content_tokens == raw_tokens and not result.cuts:
            # In budget and uncut, on whitespace-clean text: the harness render and the paper
            # construction agree too (the trim is an identity on clean content).
            harness_render = _render_harness_shape(template_text, row["query"], row["documents"][0])
            assert harness_render == fit_text, row["query"][:40]
            paper = _paper_render(tokenizer, row["query"], row["documents"][0])
            assert paper == fit_text and paper == engine_render
            checked_in_budget += 1
    assert checked_in_budget >= 20  # the brief's floor: at least 20 sampled pairs with full id equality


def test_stage1_on_cpu_passes_anchor_and_template_checks(tmp_path: Any, monkeypatch: Any) -> None:
    """Stage 1 on CPU: the anchor audit and the served-template check over >= 20 pairs, >= 5 over cap.

    ``stage1_prompts`` runs the product's fit over the pairs file and its own over-length samples,
    audits every anchor on the fitted renders, and compares the declared shape with the template
    file on the pairs file's first row. The reference subprocess's render comparison needs the
    recipe's reference environment; it is asserted by ``test_reference_render_matches`` when
    ``RCP_ZERANK_REFERENCE_PYTHON`` provides one, and reported ``not_run`` here (neutral, never
    passed)."""
    recipe = load_recipe(RECIPE_DIR)
    tokenizer = _tokenizer(monkeypatch, tmp_path)
    rows = _sample_pairs(tokenizer)
    assert len(rows) >= 20
    over_cap = [
        row
        for row in rows
        if tokenizer.count(row["query"])
        + tokenizer.count(row["documents"][0])
        + recipe.client.template.overhead("pair", tokenizer)
        > recipe.client.max_tokens
    ]
    assert len(over_cap) >= 5
    document = stage1_prompts(recipe, _write_pairs(tmp_path, rows), os.environ.get("RCP_ZERANK_REFERENCE_PYTHON"))
    assert document["sampled"] >= 25
    assert document["anchor_check"]["passed"] is True, document["anchor_check"]["failures"][:2]
    assert document["anchor_check"]["checked"] >= 25
    assert document["template_render_check"] is not None
    assert document["template_render_check"]["passed"] is True, document["template_render_check"]["failures"][:2]
    assert document["render_check"]["status"] in ("run", "not_run")
    assert document["engine_tokenize_check"]["status"] == "not_run"  # no engine on CPU; never reported passed


def test_reference_render_matches(tmp_path: Any, monkeypatch: Any) -> None:
    """With a reference environment (RCP_ZERANK_REFERENCE_PYTHON), stage 1's render check passes.

    The reference subprocess builds each in-budget pair's prompt the paper's way (the checkpoint's
    own chat template over the stripped texts) and must equal the product's fit render byte for byte;
    on over-cap pairs it renders the anchor-preserving shape through the product's fit."""
    reference_python = os.environ.get("RCP_ZERANK_REFERENCE_PYTHON")
    if not reference_python:
        pytest.skip(
            "no reference environment: set RCP_ZERANK_REFERENCE_PYTHON to a python with the recipe's "
            "requirements-reference.txt (transformers for the paper construction, rcp-ndcg for fit)"
        )
    recipe = load_recipe(RECIPE_DIR)
    tokenizer = _tokenizer(monkeypatch, tmp_path)
    rows = _sample_pairs(tokenizer)[:6]  # render fidelity: the first in-budget rows and the over-cap ones
    document = stage1_prompts(recipe, _write_pairs(tmp_path, rows), reference_python)
    assert document["render_check"]["status"] == "run"
    assert document["render_check"]["passed"] is True, document["render_check"]["failures"][:2]


def test_padded_inputs_align_through_the_templates_trim(tmp_path: Any, monkeypatch: Any) -> None:
    """Whitespace-padded inputs: the engine's render (the file's trim) equals the paper's ``.strip()``
    construction; the declared shape keeps the raw text (its token count never undershoots the
    engine's render). The declared shape and the file diverge on such rows by design - the pairs file
    for stage 1 must be whitespace-clean."""
    recipe = load_recipe(RECIPE_DIR)
    tokenizer = _tokenizer(monkeypatch, tmp_path)
    template_text = (RECIPE_DIR / "template.jinja").read_text(encoding="utf-8")
    padded_query, padded_document = "  padded query \n\t", "\n leading document "
    engine_render = _render_engine_shape(template_text, padded_query, padded_document)
    assert engine_render == _paper_render(tokenizer, padded_query, padded_document)
    result = _fit_pair(recipe, tokenizer, padded_query, padded_document)
    assert result.texts[0] != engine_render  # the declared shape fills content raw: the conservative model


def test_dropping_the_trailing_anchor_segment_turns_the_anchor_check_red(tmp_path: Any, monkeypatch: Any) -> None:
    """The mutation the brief asks for: the trailing anchor (the assistant header the score is pooled
    from) dropped from the declared template's tail segment - the anchor audit must fail, loudly."""
    mutated = tmp_path / "zerank-1-reranker"
    mutated.mkdir()
    for name in ("recipe.yaml", "template.jinja", "reference.py"):
        shutil.copy(RECIPE_DIR / name, mutated / name)
    data = yaml.safe_load((mutated / "recipe.yaml").read_text(encoding="utf-8"))
    segments = data["client"]["template"]["pair"]
    assert segments[-1]["fixed"].endswith("assistant\n")  # the anchor being dropped
    data["client"]["template"]["pair"] = segments[:-1]  # drop the trailing anchor segment entirely
    (mutated / "recipe.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    recipe = load_recipe(mutated)  # still loadable: the shape pins add_special_tokens, so a content tail is allowed
    tokenizer = _tokenizer(monkeypatch, tmp_path)
    rows = _sample_pairs(tokenizer)[:2]
    document = stage1_prompts(recipe, _write_pairs(tmp_path, rows), None, over_length_per_shape=3)
    assert document["anchor_check"]["passed"] is False, "the dropped anchor must fail the audit"
    assert document["anchor_check"]["failures"], "the audit must name the failing renders"
    assert all(failure["check"] == "tail" for failure in document["anchor_check"]["failures"])


def test_a_recipe_whose_template_file_is_missing_is_refused(tmp_path: Any) -> None:
    """R10: the template file must ship - the schema refuses a recipe naming a missing file, because
    vLLM's rerank path degrades silently (pair-encode, no warning for this architecture) without it."""
    broken = tmp_path / "zerank-1-reranker"
    broken.mkdir()
    for name in ("recipe.yaml", "reference.py"):
        shutil.copy(RECIPE_DIR / name, broken / name)
    with pytest.raises(RecipeError, match="chat_template"):
        load_recipe(broken)
