"""The zerank-1-small-reranker recipe: validation, CPU stage 1, and the anchor mutation.

The recipe validates against the product's RerankEndpoint at load (the schema does that); these tests
check the recipe's own declared facts and run stage 1 on the real tokenizer files (downloaded into the
lane's scratch directory, or tmp_path in CI; skipped with a clear reason when offline). The mutation
test drops the template's trailing anchor segment and shows the anchor check red.
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
from rcp_ndcg_vllm import load_recipe, serve_argv
from rcp_ndcg_vllm.equivalence import stage1_prompts
from rcp_ndcg_vllm.equivalence.fitting import tokenizer_of

from ._contract import assert_recipe_contract
from ._served import fetch_tokenizer, served_rows, stage1_facts

RECIPE_DIR = Path(__file__).resolve().parents[2] / "recipes" / "zerank-1-small-reranker"
REVISION = "a65fd51c450e9b47fdddab98e31166ecad21af8d"
REPO = "zeroentropy/zerank-1-small-reranker"
TOKENIZER_URL = f"https://huggingface.co/{REPO}/resolve/{REVISION}/tokenizer.json"
TOKENIZER_SHA256 = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"

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
        "chat_template": "zerank_score_template.jinja",
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
        "model": "zerank-1-small-reranker",
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
            "classifier_from_token [Yes] + method no_post_processing, --chat-template "
            "zerank_score_template.jinja, pooler logit_sigma 5 + use_activation true "
            "(sigmoid(l_Yes/5) at the last token, 1-label head)"
        ),
        "tokenizer": f"{REPO}@{REVISION}",
        "max_tokens": 8192,
        "instruction": "none",
        "use_activation": True,
        "query_max_tokens": 4096,
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
    "id": "zerank-1-small-reranker",
    "model": REPO,
    "revision": REVISION,
    "role": "rerank",
    "input": ["text"],
    "scoring": "pointwise",
    "licence": "apache-2.0",
}

_RECIPE_FILES = ("recipe.yaml", "zerank_score_template.jinja", "reference.py")


def _tokenizer_file(tmp_path: Path) -> Path:
    """The pinned tokenizer.json in the shared tokeniser cache (``RCP_NDCG_VLLM_TOKENIZER_CACHE``
    when set, the test's ``tmp_path`` otherwise): one download per recipe name, SHA-256 pinned."""
    return fetch_tokenizer(TOKENIZER_URL, f"{RECIPE_DIR.name}-tokenizer.json", tmp_path, sha256=TOKENIZER_SHA256)


#: Stage 1 on CPU needs the real tokenizer files: downloaded into the shared tokeniser cache
#: (network). The conftest marks and skips these tests without ``RCP_NDCG_NETWORK_TESTS=1``.
stage1_env = pytest.mark.network


def _resolved_recipe(tmp_path: Path) -> Any:
    """The recipe, copied into tmp_path with client.tokenizer pointing at the downloaded file.

    load_recipe checks the referenced files against the recipe's own directory, so the template and
    the reference come along; the copy lives in tmp_path (tests write nowhere else).
    """
    local_tokenizer = _tokenizer_file(tmp_path)
    copied = tmp_path / RECIPE_DIR.name
    copied.mkdir()
    for name in _RECIPE_FILES:
        shutil.copy(RECIPE_DIR / name, copied / name)
    data = yaml.safe_load((copied / "recipe.yaml").read_text(encoding="utf-8"))
    data["client"]["tokenizer"] = str(local_tokenizer)
    (copied / "recipe.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return load_recipe(copied)


def _pairs_path(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "pairs.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _sample_pairs() -> list[dict]:
    """Twenty clean pairs (no edge whitespace: the served frame renders the client's text as sent),
    varied in length, including an empty document (empty_doc: send)."""
    facts = [
        ("What is the capital of France?", "Paris is the capital and largest city of France."),
        ("Who wrote the Odyssey?", "The Odyssey is attributed to the ancient Greek poet Homer."),
        ("chemical symbol for gold", "Gold has the chemical symbol Au and atomic number 79."),
        ("longest river in the world", "The Nile and the Amazon compete for the title of longest river."),
        ("speed of light in vacuum", "Light travels at exactly 299,792,458 metres per second in vacuum."),
        ("Who painted the Mona Lisa?", "The Mona Lisa was painted by Leonardo da Vinci in the early 1500s."),
        ("definition of entropy", "Entropy measures the number of microstates consistent with a macrostate."),
        ("When did the Berlin Wall fall?", "The Berlin Wall fell on 9 November 1989."),
        ("largest planet in the solar system", "Jupiter is the largest planet in the solar system."),
        ("What is photosynthesis?", "Photosynthesis converts light energy into chemical energy in plants."),
        ("smallest prime number", "The smallest prime number is 2, which is also the only even prime."),
        ("currency of Japan", "The Japanese yen is the official currency of Japan."),
        ("Who developed the theory of relativity?", "Albert Einstein developed both special and general relativity."),
        ("boiling point of water at sea level", "Water boils at 100 degrees Celsius at standard pressure."),
        ("What does DNA stand for?", "DNA stands for deoxyribonucleic acid, the molecule of heredity."),
        ("first person on the Moon", "Neil Armstrong stepped onto the Moon on 20 July 1969."),
        ("main export of Brazil", "Brazil is a leading exporter of coffee, soybeans and iron ore."),
        ("What causes tides?", "Tides are caused mainly by the Moon's gravitational pull on the oceans."),
        ("How many continents are there?", "There are seven continents on Earth."),
        ("", "The Great Barrier Reef lies off the coast of Queensland, Australia."),
    ]
    return [{"query": query, "documents": [document]} for query, document in facts]


def test_recipe_validates_against_the_schema_and_the_product() -> None:
    """The recipe loads: role-aware client block, explicit budget, the template file, the deviation."""
    recipe = load_recipe(RECIPE_DIR)
    assert_recipe_contract(
        recipe, serve=CONTRACT["serve"], client=CONTRACT["client"], reference=CONTRACT["reference"], top=TOP
    )
    template = recipe.client.template
    assert template is not None
    assert template.shapes() == ("pair",) and template.anchor == "last"
    assert template.adds_special_tokens("pair") is True
    assert template.normalisers("pair") == ("strip",)  # the family's declared normalisation
    assert [segment.content for segment in template.segments("pair") if segment.content] == ["query", "document"]
    assert (RECIPE_DIR / recipe.serve.chat_template).is_file()
    assert recipe.status.state == "unverified"
    assert any("r-zerank1" in source for source in recipe.sources)


def test_the_contract_reds_on_two_mutants() -> None:
    """Two mutants of the declared contract must red the pin (the sweep's surviving mutants)."""
    recipe = load_recipe(RECIPE_DIR)
    serve_mutant = recipe.model_copy(update={"serve": recipe.serve.model_copy(update={"max_model_len": 40960})})
    with pytest.raises(AssertionError, match="max_model_len"):
        assert_recipe_contract(
            serve_mutant, serve=CONTRACT["serve"], client=CONTRACT["client"], reference=CONTRACT["reference"], top=TOP
        )
    reference_mutant = recipe.model_copy(
        update={"reference": recipe.reference.model_copy(update={"kind": "remote_code"})}
    )
    with pytest.raises(AssertionError, match="kind"):
        assert_recipe_contract(
            reference_mutant,
            serve=CONTRACT["serve"],
            client=CONTRACT["client"],
            reference=CONTRACT["reference"],
            top=TOP,
        )


def test_serve_argv_renders_the_pinned_serving_command() -> None:
    """The argv carries the revision, the template file, the classify overrides and the pooler scale."""
    recipe = load_recipe(RECIPE_DIR)
    argv = serve_argv(recipe, port=8100, served_model_name=recipe.id)
    assert argv[:3] == ["vllm", "serve", recipe.model]
    assert argv[argv.index("--revision") + 1] == REVISION
    assert argv[argv.index("--served-model-name") + 1] == recipe.id
    assert argv[argv.index("--pooler-config") + 1] == json.dumps(
        {"logit_sigma": 5, "use_activation": True}, sort_keys=True
    )
    assert argv[argv.index("--hf-overrides") + 1] == json.dumps(
        {
            "architectures": ["Qwen3ForSequenceClassification"],
            "classifier_from_token": ["Yes"],
            "method": "no_post_processing",
        },
        sort_keys=True,
    )
    assert argv[argv.index("--chat-template") + 1] == str(RECIPE_DIR / "zerank_score_template.jinja")
    assert argv[argv.index("--max-model-len") + 1] == "32768"
    assert argv[argv.index("--runner") + 1] == "pooling" and "--convert" not in argv


@stage1_env
@pytest.mark.network
def test_the_template_file_renders_the_declared_pair_frame(tmp_path: Path) -> None:
    """The served template file's plain-text branch is the declared pair shape, byte for byte
    (the engine branch is the same frame; stage 1's template_render_check asserts it on the recipe)."""
    from jinja2 import StrictUndefined
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    from rcp_ndcg.data.tokenizer import load_tokenizer

    recipe = load_recipe(RECIPE_DIR)
    tokenizer = load_tokenizer(str(_tokenizer_file(tmp_path)))
    environment = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=False, undefined=StrictUndefined
    )
    render = environment.from_string((RECIPE_DIR / recipe.serve.chat_template).read_text(encoding="utf-8"))
    im_start = tokenizer.special_text("im_start")
    im_end = tokenizer.special_text("im_end")
    for query, document in [("capital of france", "Paris is the capital of France."), ("", "")]:
        plain = render.render(query=query, document=document, instruction="")
        frame = recipe.client.template.render("pair", tokenizer, query=query, document=document)
        assert plain == frame
        # the declared frame, built independently from the tokenizer's added tokens:
        assert frame == (
            f"{im_start}system\n{query}{im_end}\n{im_start}user\n{document}{im_end}\n{im_start}assistant\n"
        )
        assert frame.endswith(f"{im_start}assistant\n")  # the anchor: the paper's last token is its newline
        assert frame.startswith(f"{im_start}system\n")


@stage1_env
@pytest.mark.network
def test_stage1_on_cpu_token_ids_anchor_check_and_over_length_pairs(tmp_path: Path) -> None:
    """Stage 1 on the real tokenizer: fit's renders, the template file, the reference subprocess and
    the anchors agree on 25 sampled pairs (20 from the pairs file, 5 over-length)."""
    recipe = _resolved_recipe(tmp_path)
    tokenizer = tokenizer_of(recipe)
    # The score head is the lm_head row of the token "Yes": a single token, id 9454, measured on
    # the pinned tokenizer.json (reference.py asserts it again when it loads, on the GPU wave).
    assert tokenizer.ids("Yes") == [9454]
    # The frame's fixed overhead, measured (add_special_tokens adds none for this tokenizer).
    assert recipe.client.template.overhead("pair", tokenizer) == 13
    document = stage1_prompts(recipe, _pairs_path(tmp_path, _sample_pairs()), sys.executable, over_length_per_shape=5)
    assert document["sampled"] == 25  # 20 pairs + 5 over-length
    assert document["passed"] is True, document
    # The cut facts come from the role client's own capture and census: exactly the over-length
    # samples were cut (one row each: the settled query span and the document span), and the
    # product measured the frame's fixed overhead.
    facts = stage1_facts(recipe, _sample_pairs(), tokenizer, 5)
    assert facts["per_shape"]["pair"]["overhead"] == 13
    assert facts["per_shape"]["pair"]["cut_rows"] == 5
    assert document["anchor_check"]["passed"] is True and document["anchor_check"]["checked"] == 49
    # audited: one settled query + one document span per 25 pairs (row 20's empty query is not
    # a span the audit counts), on the captured wire
    assert document["template_render_check"]["passed"] is True
    assert document["engine_tokenize_check"]["status"] == "not_run"  # CPU: never reported as passed
    render_check = document["render_check"]
    assert render_check["status"] == "run" and render_check["passed"] is True
    assert render_check["rows"] == 20  # the reference subprocess's render, byte-identical to fit's


@stage1_env
@pytest.mark.network
def test_reference_render_matches_fit_on_over_budget_pairs(tmp_path: Path) -> None:
    """The reference's anchor-preserving cut spans (its --mode render) are the client's own wire
    spans, byte for byte, including over-budget pairs: the query cut to its share, the document to
    the rest -- and the engine's frame around them keeps the assistant header (the anchor)."""
    recipe = _resolved_recipe(tmp_path)
    tokenizer = tokenizer_of(recipe)
    im_start = tokenizer.special_text("im_start")
    rows = _sample_pairs()[:2] + [
        # over budget, both spans: the query alone is past its 4096-token share, the document past
        # the 8192 budget ("alfa " re-tokenizes to 2 tokens and "bravo " to 3, so the pairs are
        # comfortably over budget)
        {"query": "alfa " * 6000, "documents": ["bravo " * 6000]},
        # the query under its share, the document alone past the budget
        {"query": "short query", "documents": ["bravo " * 9000]},
        # an empty document under budget (empty_doc: send)
        {"query": "empty document", "documents": [""]},
    ]
    fitted = served_rows(recipe, rows, tokenizer)["per_shape"]["pair"]["spans"]
    out_path = tmp_path / "reference.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(RECIPE_DIR / "reference.py"),
            "--mode",
            "render",
            "--pairs",
            _pairs_path(tmp_path, rows),
            "--out",
            str(out_path),
            "--tokenizer",
            str(recipe.client.tokenizer),
            "--device",
            "cpu",
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stderr[-500:]
    rows_out = json.loads(out_path.read_text(encoding="utf-8"))["rows"]
    # The reference's cut spans are the client's own, byte for byte (compared like the harness: on
    # the spans the wire carries).
    assert [{"query": row["query"], "documents": list(row["documents"])} for row in rows_out] == fitted
    header = f"{im_start}assistant\n"
    template = recipe.client.template
    # The frame the engine assembles around the two over-budget rows' spans keeps the anchor (the
    # paper's whole-prompt cut would drop it) and fits the declared budget.
    for span in fitted[2:4]:
        assert template is not None
        frame = template.render("pair", tokenizer, query=span["query"], document=span["documents"][0])
        assert frame.endswith(header), "the anchor must survive an over-budget cut"
        assert len(tokenizer.ids(frame, add_special_tokens=True)) <= 8192  # the budget held
    assert "alfa " * 5999 not in fitted[2]["query"]  # the query span was cut to its declared share
    assert "bravo " * 5999 not in fitted[2]["documents"][0]  # the document span was cut to the remainder
    assert fitted[4] == {"query": "empty document", "documents": [""]}  # empty_doc: send keeps the empty string


@stage1_env
@pytest.mark.network
def test_padded_inputs_strip_through_the_declared_normalisation(tmp_path: Path) -> None:
    """Whitespace-padded inputs: the wire ships the declared normalisation (``normalize: [strip]``)
    - the paper's ``query.strip()``/``doc.strip()`` - as the content spans, and the reference's spans
    agree byte for byte (one rule for the family: the strip is declared, never a template filter).
    """
    from rcp_ndcg_vllm.equivalence.fitting import tokenizer_of

    recipe = _resolved_recipe(tmp_path)
    tokenizer = tokenizer_of(recipe)
    rows = [{"query": "  padded query \n\t", "documents": ["\n leading document "]}]
    spans = served_rows(recipe, rows, tokenizer)["per_shape"]["pair"]["spans"]
    assert spans == [{"query": "padded query", "documents": ["leading document"]}]
    im_start = tokenizer.special_text("im_start")
    im_end = tokenizer.special_text("im_end")
    template = recipe.client.template
    assert template is not None
    built = template.render("pair", tokenizer, query=spans[0]["query"], document=spans[0]["documents"][0])
    assert built == (
        f"{im_start}system\npadded query{im_end}\n{im_start}user\nleading document{im_end}\n{im_start}assistant\n"
    )
    assert built.endswith(f"{im_end}\n{im_start}assistant\n")


@stage1_env
@pytest.mark.network
def test_mutation_dropping_the_anchor_segment_reddens_the_template_check(tmp_path: Path) -> None:
    """Drop the template's trailing anchor segment: the file-vs-declaration check reds.

    The rerank wire carries the cut CONTENT spans (the frame is the engine's own template), so
    stage 1's anchor audit audits the settled query and the document spans -- a frame change does
    not move it; the frame contract is pinned by ``template_render_check`` (the declared shape
    must end with the header the file emits), which this mutation turns red.
    """
    recipe = _resolved_recipe(tmp_path)
    template = recipe.client.template
    client = recipe.client.model_copy(update={"template": template.model_copy(update={"pair": template.pair[:-1]})})
    mutated: Any = recipe.model_copy(update={"client": client})
    assert len(mutated.client.template.segments("pair")) == 4
    document = stage1_prompts(
        mutated, _pairs_path(tmp_path, _sample_pairs()[:3]), sys.executable, over_length_per_shape=1
    )
    # The declared shape no longer ends with the header the file emits: the template check is red.
    assert document["template_render_check"]["passed"] is False
    # The spans themselves are unchanged (the query and document contents ship as before).
    assert document["anchor_check"]["passed"] is True
