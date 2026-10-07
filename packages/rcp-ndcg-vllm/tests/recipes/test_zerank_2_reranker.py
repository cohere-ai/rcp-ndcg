"""The zerank-2-reranker recipe: the declared contract, CPU stage 1, the reference spans, the mutation.

The contract test pins every resolved ``serve``/``client``/``reference`` field through the recipe
lanes' shared helper (``_contract.assert_recipe_contract``), with a two-mutant negative control.
Stage 1 runs the harness's own machinery on the real tokenizer (downloaded into
``RCP_NDCG_VLLM_TOKENIZER_CACHE`` when set, else ``tmp_path``, pinned by SHA-256; public Hub file,
never a token file). The reference's ``--mode render`` writes the paper's own cut (the whole rendered
prompt right-cut at 8192 tokens, after the paper's strip) in the harness's span format -- never the
client's cut (decision 9): under-cap rows equal the wire byte for byte, over-cap rows are the declared
``anchor_drop_over_cap`` (reported non-gating), and an over-share query under the budget gates red.
The reference's score mode needs torch and the checkpoint weights -- it runs on the GPU wave.
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
from rcp_ndcg_vllm.equivalence.fitting import tokenizer_of
from rcp_ndcg_vllm.recipe import Recipe

from ._contract import assert_recipe_contract
from ._served import fetch_tokenizer, served_rows, stage1_facts

TESTS = Path(__file__).resolve().parent  # packages/rcp-ndcg-vllm/tests/recipes
PACKAGE = TESTS.parent.parent  # packages/rcp-ndcg-vllm
RECIPES = PACKAGE / "recipes"
RECIPE_DIR = RECIPES / "zerank-2-reranker"
RECIPE_ID = "zerank-2-reranker"
MODEL = "zeroentropy/zerank-2-reranker"
REVISION = "5eae30d5ee3c6b2df2ef6d723bde45172d761c4c"
TEMPLATE = "template.jinja"
MAX_TOKENS = 8192
QUERY_MAX_TOKENS = 4096
TOKENIZER_URL = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/tokenizer.json"
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
        "chat_template": TEMPLATE,
        "pooler_config": {"logit_sigma": 5, "use_activation": True},
        "trust_remote_code": False,
        "max_model_len": 8192,
        "dtype": "bfloat16",
        "plugin": None,
        "io_processor_plugin": None,
        "mm_processor_kwargs": {},
        "limit_mm_per_prompt": None,
        "extra_args": [],
    },
    "client": {
        "api": "rerank",
        "model": "zerank-2-reranker",
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
            "template.jinja, pooler logit_sigma 5 + use_activation true "
            "(sigmoid(l_Yes/5) at the last token, 1-label head)"
        ),
        "tokenizer": f"{MODEL}@{REVISION}",
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
    "id": RECIPE_ID,
    "model": MODEL,
    "revision": REVISION,
    "role": "rerank",
    "input": ["text"],
    "scoring": "pointwise",
    "licence": "apache-2.0",
}


def committed() -> Recipe:
    """The committed recipe, loaded and validated as the harness loads it."""
    return load_recipe(RECIPE_DIR)


def with_local_tokenizer(tokenizer_path: Path) -> Recipe:
    """The committed recipe, reading its tokenizer from the downloaded file.

    The committed recipe names the Hub spec (what production resolves); the stage-1 checks run on
    the same tokenizer.json, downloaded into the shared cache, so they stay offline-capable after
    the one download.
    """
    recipe = committed()
    client = recipe.client.model_copy(update={"tokenizer": str(tokenizer_path)})
    return recipe.model_copy(update={"client": client})


@pytest.fixture(scope="session")
def zerank_tokenizer(tmp_path_factory: Any) -> Path:
    """The recipe tokenizer's file, pinned by SHA-256, in the shared tokeniser cache or ``tmp_path``.

    Skips with a clear reason when offline (CI): stage 1 on CPU is meaningless without the
    tokenizer the recipe declares.
    """
    tmp_path = Path(tmp_path_factory.mktemp("zerank-tokenizer"))
    return fetch_tokenizer(TOKENIZER_URL, f"{RECIPE_DIR.name}-tokenizer.json", tmp_path, sha256=TOKENIZER_SHA256)


def write_pairs(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def sample_pairs() -> list[dict]:
    """17 pairs rows: 13 of varied length, one instruction-bearing (the recipe folds none), one
    empty document, and two over the pair budget (both sides must cut the content and keep the
    anchor). Deterministic; the long rows are big on purpose (~21k tokens), so the render check
    exercises the reference's anchor-preserving cut against the product's fit."""
    import random

    random.seed(11)
    words = (
        "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi "
        "rho sigma tau upsilon phi chi psi omega"
    ).split()

    def document(words_target: int) -> str:
        pieces: list[str] = []
        total = 0
        while total < words_target:
            piece = " ".join(random.choices(words, k=6))
            pieces.append(piece)
            total += len(piece.split())
        return " ".join(pieces)

    rows = [
        {"query": f"what does the {words[index]} of row {index} mean", "documents": [document(random.randint(15, 150))]}
        for index in range(13)
    ]
    # The harness's over-length sampler pads rows[0]'s spans marker-wise, re-counting the growing
    # text per step (stages._over_length): a many-word seed makes that loop spin for minutes.
    # Keep row 0 short; the length variety lives in the rows behind it.
    rows[0] = {"query": "what does the alpha of row 0 mean", "documents": ["the short seed document"]}
    rows.append(
        {"query": "instruction-bearing query", "documents": [document(40)], "instruction": "Rank by relevance."}
    )
    rows.append({"query": "empty document query", "documents": [""]})
    long_document = document(21_000)
    rows.append({"query": "over the pair cap", "documents": [long_document]})
    rows.append({"query": "far over the pair cap", "documents": [f"{long_document} {long_document}"]})
    return rows


# -----------------------------------------------------------------------------------------------
# The recipe validates (offline).
# -----------------------------------------------------------------------------------------------


def test_recipe_contract_pins_every_field() -> None:
    """Every resolved serve/client/reference field (and every top-level fact) is pinned exactly."""
    recipe = committed()
    assert_recipe_contract(
        recipe, serve=CONTRACT["serve"], client=CONTRACT["client"], reference=CONTRACT["reference"], top=TOP
    )
    assert recipe.serve.max_model_len == recipe.client.max_tokens
    assert (RECIPE_DIR / TEMPLATE).is_file()  # R10: without the file vLLM warns and concatenates
    assert (RECIPE_DIR / "requirements-reference.txt").is_file()
    assert recipe.sources


def test_the_contract_reds_on_two_mutants() -> None:
    """Two mutants of the declared contract must red the pin (the sweep's surviving mutants)."""
    recipe = committed()
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


def test_template_declares_specials_by_name_and_the_anchor_tail() -> None:
    """The frame is data: specials by name (never a literal), the anchor is the assistant header."""
    template = committed().client.template
    assert template is not None
    assert template.shapes() == ("pair",)
    assert template.anchor == "last"
    assert template.adds_special_tokens("pair") is True
    assert template.normalisers("pair") == ("strip",)  # the family's declared normalisation
    fixed = [segment.fixed for segment in template.pair if segment.fixed is not None]
    assert len(fixed) == 3
    assert all("{special:" in segment for segment in fixed), fixed
    assert not any("<|" in segment for segment in fixed), "specials are declared by name, never typed literally"
    assert [segment.content for segment in template.pair if segment.content is not None] == ["query", "document"]
    assert template.pair[-1].fixed.endswith("{special:im_start}assistant\n")  # the trailing anchor segment


def test_the_template_file_ships_and_the_argv_carries_the_serving_facts() -> None:
    recipe = committed()
    argv = serve_argv(recipe, port=8100, served_model_name=recipe.id)
    joined = " ".join(argv)
    assert "--chat-template" in argv and str(RECIPE_DIR / TEMPLATE) in argv
    assert "--runner pooling" in joined
    assert "--convert" not in argv, "a rerank recipe never flags --convert"
    assert "--max-model-len 8192" in joined
    assert "--pooler-config" in joined and '"logit_sigma": 5' in joined and '"use_activation": true' in joined
    assert "--dtype bfloat16" in joined
    assert "--trust-remote-code" not in joined
    overrides = json.loads(argv[argv.index("--hf-overrides") + 1])
    assert overrides["architectures"] == ["Qwen3ForSequenceClassification"]
    assert overrides["classifier_from_token"] == ["Yes"]
    assert overrides["method"] == "no_post_processing"


def test_a_recipe_naming_a_missing_local_tokenizer_is_refused(tmp_path: Path) -> None:
    """A local tokenizer spec whose file is absent is refused with the missing-file message, not
    with a Hub download attempt of a path-shaped repo id."""
    missing = tmp_path / "absent.json"
    pairs_path = write_pairs(tmp_path / "pairs.jsonl", sample_pairs()[:1])
    completed = subprocess.run(
        [
            sys.executable,
            str(RECIPE_DIR / "reference.py"),
            "--mode",
            "render",
            "--pairs",
            str(pairs_path),
            "--out",
            str(tmp_path / "out.json"),
            "--tokenizer",
            str(missing),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode != 0
    assert f"no tokenizer file at {missing}" in completed.stderr + completed.stdout


# -----------------------------------------------------------------------------------------------
# Stage 1 on CPU: the product's client capture, the anchor audit, the served-template check and the
# reference subprocess's spans.
# -----------------------------------------------------------------------------------------------


def test_stage1_passes_on_cpu(tmp_path: Path, zerank_tokenizer: Path) -> None:
    recipe = with_local_tokenizer(zerank_tokenizer)
    pairs = write_pairs(tmp_path / "pairs.jsonl", sample_pairs())
    document = stage1_prompts(recipe, pairs, sys.executable, over_length_per_shape=5)
    assert document["passed"] is True, document
    # At least 20 sampled pairs incl. 5 over-length ones (the pair shape is the declared one).
    assert document["sampled"] == 22
    assert document["anchor_check"]["passed"] is True, document["anchor_check"]["failures"][:1]
    assert document["anchor_check"]["checked"] == 44  # a settled query + a document span per row
    assert document["render_check"]["status"] == "run"
    assert document["render_check"]["passed"] is True, document["render_check"]["failures"][:1]
    assert document["render_check"]["rows"] == 17
    assert document["template_render_check"]["passed"] is True, document["template_render_check"]["failures"][:1]
    assert document["engine_tokenize_check"]["status"] == "not_run"  # no engine on CPU; never reported passed
    # The declared budget: the frame overhead is 13 tokens, measured, not assumed; the over-budget
    # rows were cut (the product's own census), not sent whole.
    facts = stage1_facts(recipe, sample_pairs(), tokenizer_of(recipe), 5)
    assert facts["per_shape"]["pair"]["overhead"] == 13
    assert facts["per_shape"]["pair"]["cuts"] > 0


def test_the_reference_spans_the_fit_ids_and_the_served_template_agree(tmp_path: Path, zerank_tokenizer: Path) -> None:
    """Token-id equality: the reference subprocess's spans, the wire's captured spans and the
    served template file's render tokenize to the same ids, per sampled row."""
    import jinja2
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    recipe = with_local_tokenizer(zerank_tokenizer)
    tokenizer = tokenizer_of(recipe)
    rows = sample_pairs()
    template_text = (RECIPE_DIR / TEMPLATE).read_text(encoding="utf-8")
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, undefined=jinja2.StrictUndefined)
    served = served_rows(recipe, rows, tokenizer)["per_shape"]["pair"]["spans"]
    flag = recipe.client.template.adds_special_tokens("pair")
    template = recipe.client.template
    assert template is not None

    for index, row in enumerate(rows[:4]):
        inputs = [(row["query"], row["documents"][0])]
        del inputs  # the wire's spans are the fit's render; nothing is re-derived here (R30)
        span = served[index]
        fitted = template.render("pair", tokenizer, query=span["query"], document=span["documents"][0])
        # The served template file renders the same prompt from the wire's spans (the harness's check).
        jinja_text = env.from_string(template_text).render(
            query=span["query"], document=span["documents"][0], instruction=""
        )
        assert jinja_text == fitted, index
        ids = tokenizer.ids(fitted, add_special_tokens=flag)
        assert ids == tokenizer.ids(jinja_text, add_special_tokens=flag)
        # The anchor: the trailing fixed segment sits at the tail of every rendered id list.
        anchor = template.segments("pair")[-1].render(tokenizer)
        assert ids[-len(tokenizer.ids(anchor)) :] == tokenizer.ids(anchor), index


def test_stage1_anchor_check_survives_over_length_inputs(tmp_path: Path, zerank_tokenizer: Path) -> None:
    """The anchor audit samples over-length inputs on purpose (5 here) and every cut keeps the
    settled query within its share and the document within the budget."""
    recipe = with_local_tokenizer(zerank_tokenizer)
    pairs = write_pairs(tmp_path / "pairs.jsonl", sample_pairs()[:2])
    document = stage1_prompts(recipe, pairs, None, over_length_per_shape=5)
    assert document["sampled"] == 7  # 2 pairs rows + 5 over-length samples
    assert document["anchor_check"]["passed"] is True, document["anchor_check"]["failures"][:1]
    assert document["anchor_check"]["checked"] == 14  # a settled query + a document span per row


def test_the_served_template_renders_identically_for_the_engine_and_the_harness(
    tmp_path: Path, zerank_tokenizer: Path
) -> None:
    """The served file's two branches and the declared pair shape render byte-identically, per row.

    The engine's score route renders the file over query/document `messages` (tools=None) with
    transformers' serving environment (trim_blocks/lstrip_blocks, NOT StrictUndefined); the
    harness's stage-1 check renders it from the plain texts under StrictUndefined; the declared
    shape is what the product's template assembles around the wire's spans. All three must agree
    for every sampled row - the harness's own check sees only the plain-text branch on CPU (no
    engine), so this test pins the engine branch too.
    """
    import jinja2
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    recipe = with_local_tokenizer(zerank_tokenizer)
    tokenizer = tokenizer_of(recipe)
    template_text = (RECIPE_DIR / TEMPLATE).read_text(encoding="utf-8")
    engine_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    harness_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, undefined=jinja2.StrictUndefined)
    template = recipe.client.template
    assert template is not None
    flag = template.adds_special_tokens("pair")
    served = served_rows(recipe, sample_pairs(), tokenizer)["per_shape"]["pair"]["spans"]
    for index, span in enumerate(served):
        cut_query, cut_document = span["query"], span["documents"][0]  # what the client sends
        fitted = template.render("pair", tokenizer, query=cut_query, document=cut_document)
        plain = harness_env.from_string(template_text).render(query=cut_query, document=cut_document, instruction="")
        engine = engine_env.from_string(template_text).render(
            messages=[
                {"role": "query", "content": cut_query},
                {"role": "document", "content": cut_document},
            ],
            tools=None,
        )
        assert plain == engine == fitted, index
        assert tokenizer.ids(engine, add_special_tokens=flag) == tokenizer.ids(fitted, add_special_tokens=flag), index


def test_reference_cli_renders_the_papers_spans_and_refuses_embed(tmp_path: Path, zerank_tokenizer: Path) -> None:
    """The subprocess contract: exit 0, the JSON shape (the harness's rerank span format), and the
    over-cap row's spans are the PAPER's cut -- the whole prompt right-cut at the budget, the anchor
    dropped -- never the client's anchor-preserving cut, which keeps the anchor within the budget."""
    recipe = with_local_tokenizer(zerank_tokenizer)
    tokenizer = tokenizer_of(recipe)
    rows = sample_pairs()
    over_cap = rows[-1]  # the far-over-cap row: the pair overflows, both sides cut the document
    pairs_path = write_pairs(tmp_path / "pairs.jsonl", [over_cap])
    out_path = tmp_path / "reference.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(RECIPE_DIR / "reference.py"),
            "--mode",
            "render",
            "--pairs",
            str(pairs_path),
            "--out",
            str(out_path),
            "--tokenizer",
            str(zerank_tokenizer),
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stderr[-500:]
    document = json.loads(out_path.read_text(encoding="utf-8"))
    assert set(document["rows"][0]) == {"index", "shape", "query", "documents"}
    spans = served_rows(recipe, [over_cap], tokenizer)["per_shape"]["pair"]["spans"]
    paper_span = document["rows"][0]
    im_start, im_end = tokenizer.special_text("im_start"), tokenizer.special_text("im_end")
    head, mid = f"{im_start}system\n", f"{im_end}\n{im_start}user\n"
    whole = f"{head}{over_cap['query']}{mid}{over_cap['documents'][0].strip()}{im_end}\n{im_start}assistant\n"
    kept = f"{head}{paper_span['query']}{mid}{paper_span['documents'][0]}"
    assert tokenizer.ids(kept) == tokenizer.ids(whole)[:MAX_TOKENS]  # the paper's right cut, anchor dropped
    assert paper_span["documents"][0].startswith(spans[0]["documents"][0])  # the client cuts shorter
    assert paper_span["documents"][0] != spans[0]["documents"][0]
    template = recipe.client.template
    assert template is not None
    frame = template.render("pair", tokenizer, query=spans[0]["query"], document=spans[0]["documents"][0])
    assert frame.endswith(f"{tokenizer.special_text('im_start')}assistant\n"), "the anchor survives the cut"
    assert len(tokenizer.ids(frame, add_special_tokens=True)) <= MAX_TOKENS  # the budget held

    refused = subprocess.run(
        [
            sys.executable,
            str(RECIPE_DIR / "reference.py"),
            "--mode",
            "embed",
            "--pairs",
            str(pairs_path),
            "--out",
            str(tmp_path / "refused.json"),
            "--tokenizer",
            str(zerank_tokenizer),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert refused.returncode != 0
    assert "reranker" in (refused.stderr + refused.stdout)


def test_settle_rule_and_declared_normalisation_against_the_papers_spans(
    tmp_path: Path, zerank_tokenizer: Path
) -> None:
    """The wire's settle-once query and the declared normalisation, against the paper's own spans.

    (1) An over-share query (over ``query_max_tokens``) in an under-budget pair is settled at its share
    by the rerank client (settle-once) while the paper keeps it whole: an under-cap row, so stage 1's
    render check gates it red -- the reference never ports the settle rule (decision 9); (2)
    whitespace-padded inputs round through the declared normalisation (``normalize: [strip]``: the
    paper's strip) on both sides, byte for byte.
    """
    recipe = with_local_tokenizer(zerank_tokenizer)
    rows = [
        # over the 4096-token share, the pair itself under the 8192 budget
        {"query": "alphagammaepsilon" * 1200, "documents": ["short document"]},
        # whitespace-padded: both sides strip through the declared normalisation
        {"query": "  padded query \n\t", "documents": ["\n leading document "]},
    ]
    spans = served_rows(recipe, rows, tokenizer_of(recipe))["per_shape"]["pair"]["spans"]
    shares_ok = tokenizer_of(recipe).count(spans[0]["query"]) == QUERY_MAX_TOKENS
    assert shares_ok and spans[0]["query"] != rows[0]["query"], spans[0]["query"][:80]
    assert spans[0]["query"] == rows[0]["query"][: len(spans[0]["query"])]  # a verbatim prefix
    assert spans[0]["documents"] == ["short document"]
    assert spans[1] == {"query": "padded query", "documents": ["leading document"]}
    out_path = tmp_path / "reference.json"
    pairs = write_pairs(tmp_path / "pairs.jsonl", rows)
    completed = subprocess.run(
        [
            sys.executable,
            str(RECIPE_DIR / "reference.py"),
            "--mode",
            "render",
            "--pairs",
            str(pairs),
            "--out",
            str(out_path),
            "--tokenizer",
            str(zerank_tokenizer),
            "--device",
            "cpu",
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stderr[-500:]
    rows_out = json.loads(out_path.read_text(encoding="utf-8"))["rows"]
    assert rows_out[0]["query"] == rows[0]["query"]  # the paper keeps the whole query
    assert {"query": rows_out[1]["query"], "documents": rows_out[1]["documents"]} == spans[1]
    document = stage1_prompts(recipe, pairs, sys.executable, over_length_per_shape=1)
    failures = document["render_check"]["failures"]
    assert document["render_check"]["passed"] is False and {failure["index"] for failure in failures} == {0}


# -----------------------------------------------------------------------------------------------
# The mutation: dropping the template's trailing anchor segment turns the frame check red.
# -----------------------------------------------------------------------------------------------


def test_mutation_dropping_the_anchor_segment_reddens_the_template_check(
    tmp_path: Path, zerank_tokenizer: Path
) -> None:
    """Drop the template's trailing anchor segment: the file-vs-declaration check reds.

    The rerank wire carries the cut CONTENT spans (the frame is the engine's own template), so
    stage 1's anchor audit audits the settled query and the document spans -- a frame change does
    not move it; the frame contract is pinned by ``template_render_check`` (the declared shape
    must end with the header the file emits), which this mutation turns red.
    """
    mutated = tmp_path / "zerank-2-reranker"
    mutated.mkdir()
    for name in ("recipe.yaml", TEMPLATE, "reference.py", "requirements-reference.txt"):
        shutil.copy(RECIPE_DIR / name, mutated / name)
    data = yaml.safe_load((mutated / "recipe.yaml").read_text(encoding="utf-8"))
    data["client"]["tokenizer"] = str(zerank_tokenizer)  # same pinned tokenizer, no Hub at run time
    segments = data["client"]["template"]["pair"]
    assert segments[-1]["fixed"].endswith("assistant\n")  # the anchor being dropped
    data["client"]["template"]["pair"] = segments[:-1]  # drop the trailing anchor segment entirely
    (mutated / "recipe.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    recipe = load_recipe(mutated)  # still loadable: add_special_tokens allows a content tail
    pairs = write_pairs(tmp_path / "pairs.jsonl", sample_pairs()[:3])
    healthy = stage1_prompts(with_local_tokenizer(zerank_tokenizer), pairs, None, over_length_per_shape=2)
    assert healthy["template_render_check"]["passed"] is True
    document = stage1_prompts(recipe, pairs, None, over_length_per_shape=2)
    assert document["template_render_check"]["passed"] is False
    assert document["template_render_check"]["failures"], "the check must name the failing renders"
    assert document["anchor_check"]["passed"] is True  # the spans themselves are unchanged


def test_a_recipe_whose_template_file_is_missing_is_refused(tmp_path: Path) -> None:
    """R10: the template file must ship - the schema refuses a recipe naming a missing file."""
    broken = tmp_path / "zerank-2-reranker"
    broken.mkdir()
    for name in ("recipe.yaml", "reference.py"):
        shutil.copy(RECIPE_DIR / name, broken / name)
    with pytest.raises(RecipeError, match="chat_template"):
        load_recipe(broken)
