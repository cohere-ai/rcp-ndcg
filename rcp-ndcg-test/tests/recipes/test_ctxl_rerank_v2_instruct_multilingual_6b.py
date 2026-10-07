"""The ctxl-rerank-v2-instruct-multilingual-6b recipe: load, CPU stage 1, the anchor mutations.

Stage 1 here is CPU-only: the recipe's tokenizer.json (the only Hub artifact it needs, ~11 MB) is
downloaded once at the recipe's pinned revision into a cache directory -- the lane's scratch dir
(``rec-<recipe id>/scratch`` beside this checkout, so the download is fetched once per machine,
never into the checkout), else pytest's own temp dir; set ``RCP_NDCG_VLLM_TOKENIZER_CACHE`` to
override the location. The product's tokenizer load is then pinned to the fetched snapshot, so
stage 1 counts in the checkpoint's tokens without a second download. Tests that need the download
skip themselves with a clear reason when the Hub is unreachable (offline CI) or
``huggingface_hub`` is not installed. No weights, no engine: the score path belongs to the GPU
wave (the recipe ships ``status: unverified`` until it passes there).

The reference subprocess (``reference.py``, ``--mode render``) is stdlib-only, so the token-id
(render) equality runs everywhere the package's tests run.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from rcp_ndcg_test.equivalence import stage1_prompts
from rcp_ndcg_test.equivalence.reference import run_reference
from rcp_ndcg_vllm import client_config, load_recipe, serve_argv
from rcp_ndcg_vllm.recipe import default_recipes_root

from rcp_ndcg.data.templates import TemplateSpec
from rcp_ndcg.inference.config import RerankEndpoint

RECIPE_ID = "ctxl-rerank-v2-instruct-multilingual-6b"
REPO_ID = "ContextualAI/ctxl-rerank-v2-instruct-multilingual-6b"
REVISION = "f14ca1a2fc204dc0934c84a3d2e278f8ff646b80"  # re-checked against the HF API on 2026-10-06
_TOKENIZER_SPEC = f"{REPO_ID}@{REVISION}"

# 20 pairs (multilingual on purpose: the checkpoint is multilingual), all under the 8192-token
# budget so stage 1's render equality is the served-vs-reference comparison the recipe declares.
# The first row carries no instruction: the harness's served-template check renders the first row,
# whose instruction would reach the jinja file but not the client's render (instruction: none).
# One mid-file row carries an instruction on purpose: the served path folds none (the paper's
# served-path config) and the reference ignores it, so stage 1 must stay green with it present.
_PAIRS: list[dict] = [
    {
        "query": "What is the capital of France?",
        "documents": ["Paris is the capital and largest city of France."],
    },
    {
        "query": "法国的首都是什么？",
        "documents": ["巴黎是法国的首都，也是全国的政治、经济和文化中心。"],
    },
    {
        "query": "Was ist die Hauptstadt von Deutschland?",
        "documents": ["Berlin ist die Hauptstadt und der Sitz der Regierung Deutschlands."],
    },
    {
        "query": "¿Cuál es la capital de España?",
        "documents": ["Madrid es la capital de España y su municipio más poblado."],
    },
    {
        "query": "日本の首都はどこですか",
        "documents": ["東京は日本の首都であり、人口は約1,400万人です。"],
    },
    {
        "query": "Who wrote Hamlet and when?",
        "documents": [
            "Hamlet is a tragedy written by William Shakespeare between 1599 and 1601, first "
            "performed at the Globe Theatre."
        ],
        "instruction": "Answer from the document only.",
    },
    {
        "query": "Quelle est la capitale du Portugal ?",
        "documents": ["Lisbonne est la capitale et la plus grande ville du Portugal."],
    },
    {
        "query": "What gas do plants absorb from the atmosphere?",
        "documents": [
            "Plants absorb carbon dioxide from the atmosphere during photosynthesis and release oxygen as a by-product."
        ],
    },
    {
        "query": "Wie funktioniert Photosynthese?",
        "documents": [
            "Bei der Photosynthese wandeln Pflanzen Lichtenergie in chemische Energie um und "
            "bilden Glukose und Sauerstoff."
        ],
    },
    {
        "query": "What is the boiling point of water at sea level?",
        "documents": ["At sea level, water boils at 100 degrees Celsius, or 212 degrees Fahrenheit."],
    },
    {
        "query": "Le loup-garou est-il un mythe européen ?",
        "documents": [
            "Le loup-garou est une créature légendaire présente dans le folklore de nombreuses régions d'Europe."
        ],
    },
    {
        "query": "What is the currency of Japan?",
        "documents": [
            "The yen is the official currency of Japan and the third most traded currency in the "
            "foreign exchange market."
        ],
    },
    {
        "query": "Chi ha dipinto la Monna Lisa?",
        "documents": [
            "La Gioconda, o Monna Lisa, è un dipinto a olio di Leonardo da Vinci, realizzato tra il 1503 e il 1519."
        ],
    },
    {
        "query": "What does DNA stand for?",
        "documents": [
            "DNA stands for deoxyribonucleic acid, the molecule that carries the genetic "
            "instructions of living organisms."
        ],
    },
    {
        "query": "Wie hoch ist der Mount Everest?",
        "documents": ["Der Mount Everest ist mit 8.848 Metern über dem Meeresspiegel der höchste Berg der Erde."],
    },
    {
        "query": "What is the largest planet in the solar system?",
        "documents": [
            "Jupiter is the largest planet in the solar system, with a mass more than twice that "
            "of all the other planets combined."
        ],
    },
    {
        "query": "Где была основана древняя библиотека?",
        "documents": ["Знаменитая библиотека была основана в третьем веке до нашей эры в Александрии."],
    },
    {
        "query": "What is artificial photosynthesis?",
        "documents": [
            "Research on artificial photosynthesis aims to store solar energy in chemical fuels, "
            "mimicking the natural process."
        ],
    },
    {
        "query": "Quel est le plus long fleuve du monde ?",
        "documents": [
            "Le Nil et l'Amazone se disputent le titre de plus long fleuve du monde, selon les méthodes de mesure."
        ],
    },
    {
        "query": "What language is spoken in Brazil?",
        "documents": [
            "Portuguese is the official and most widely spoken language of Brazil, a former Portuguese colony."
        ],
    },
]


@pytest.fixture(scope="session")
def tokenizer_snapshot(tmp_path_factory) -> Path:
    """The recipe's tokenizer.json downloaded at the pinned revision (tokenizer files only).

    Skips with the reason when the Hub is unreachable (offline CI) or ``huggingface_hub`` (the
    product's ``[hf]`` extra) is not installed. Stage 1 fits and audits in the recipe tokenizer's
    tokens, so nothing else can stand in for the download.
    """
    override = os.environ.get("RCP_NDCG_VLLM_TOKENIZER_CACHE")
    if override:
        cache = Path(override)
        cache.mkdir(parents=True, exist_ok=True)
    else:
        lane_scratch = Path(__file__).resolve().parents[4].parent / f"rec-{RECIPE_ID}" / "scratch"
        cache = lane_scratch / "hf-home" if lane_scratch.is_dir() else tmp_path_factory.mktemp("tokenizer-cache")
    try:
        import huggingface_hub
    except ModuleNotFoundError as error:
        pytest.skip(f"huggingface_hub is not installed (pip install 'rcp-ndcg[hf]'): {error}")
    try:
        path = huggingface_hub.hf_hub_download(REPO_ID, "tokenizer.json", revision=REVISION, cache_dir=str(cache))
    except Exception as error:  # noqa: BLE001 - offline (CI), rate limit, or a Hub outage
        pytest.skip(f"the Hugging Face Hub is unreachable; stage 1 needs the recipe's tokenizer files: {error}")
    return Path(path)


def _pin_tokenizer(monkeypatch: pytest.MonkeyPatch, snapshot: Path) -> None:
    """Point the product's Hub download at the already-fetched snapshot (same bytes, no re-fetch)."""
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *args, **kwargs: str(snapshot))


@pytest.fixture(scope="session")
def pairs_file(tmp_path_factory) -> Path:
    """The 20 sampled pairs, written once."""
    path = tmp_path_factory.mktemp("ctxl-pairs") / "pairs.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in _PAIRS), encoding="utf-8")
    return path


def _recipe_dir() -> Path:
    """The recipe directory: the package's own ``recipes/<id>`` (default_recipes_root)."""
    return default_recipes_root() / RECIPE_ID


def test_recipe_loads_with_the_declared_serving_and_client_blocks() -> None:
    """The recipe validates through the product's endpoint config, with every binding field."""
    recipe = load_recipe(_recipe_dir())
    assert recipe.id == _recipe_dir().name == "ctxl-rerank-v2-instruct-multilingual-6b"
    assert recipe.model == "ContextualAI/ctxl-rerank-v2-instruct-multilingual-6b"
    assert recipe.revision == REVISION  # re-checked against the Hub API on 2026-10-06
    assert recipe.role == "rerank" and recipe.scoring == "pointwise" and recipe.input == ["text"]
    assert recipe.licence == "CC-BY-NC-SA-4.0"
    client = recipe.client
    assert isinstance(client, RerankEndpoint)
    assert client.model == recipe.id and client.revision == REVISION  # injected at load
    assert client.tokenizer == _TOKENIZER_SPEC
    assert client.max_tokens == 8192 and client.query_max_tokens == 4096
    assert client.on_overflow == "cut"
    assert client.instruction == "none"  # the paper's served-path config; the default fold is wrong here
    assert client.use_activation is False
    assert client.listwise is False
    template = client.template
    assert template is not None and template.shapes() == ("pair",)
    assert template.anchor == "last" and template.adds_special_tokens("pair") is True
    segments = template.segments("pair")
    assert segments[-1].fixed == " ??"  # the tail anchor the score is read from
    serve = recipe.serve
    assert serve.pooler_config == {"use_activation": False}  # raw logit, as a server-side default
    assert serve.hf_overrides["architectures"] == ["MistralForSequenceClassification"]
    assert serve.hf_overrides["classifier_from_token"] == ["<unk>"]  # this tokenizer's id 0
    assert serve.hf_overrides["method"] == "no_post_processing"
    assert serve.chat_template == "score-template-6b.jinja"
    assert serve.max_model_len == 32768 >= client.max_tokens
    assert serve.dtype == "bfloat16" and serve.trust_remote_code is False
    assert serve.plugin is None  # nothing is added to the stock engine image
    assert recipe.engine.image == "vllm/vllm-openai:v0.31.0" and recipe.engine.min_version == "0.31.0"
    assert recipe.resources.gpus == 1
    assert recipe.reference.kind == "transformers" and recipe.reference.score_scale == "logit"
    assert recipe.reference.entry == "reference.py"
    assert recipe.reference.known_deviations == ["anchor_drop_over_cap"]
    assert recipe.status.state == "unverified"
    assert len(recipe.sources) >= 5 and recipe.notes


def test_serve_argv_renders_the_pinned_engine_invocation() -> None:
    """The argv pins the revision, the shipped template file and the raw-logit pooler config."""
    recipe = load_recipe(_recipe_dir())
    argv = serve_argv(recipe, port=8100, served_model_name=recipe.id)
    assert argv[:3] == ["vllm", "serve", recipe.model]
    assert argv[argv.index("--revision") + 1] == REVISION
    assert argv[argv.index("--served-model-name") + 1] == recipe.id
    assert argv[argv.index("--pooler-config") + 1] == '{"use_activation": false}'
    assert argv[argv.index("--chat-template") + 1] == str(_recipe_dir() / "score-template-6b.jinja")
    assert argv[argv.index("--hf-overrides") + 1] == json.dumps(
        {
            "architectures": ["MistralForSequenceClassification"],
            "classifier_from_token": ["<unk>"],
            "method": "no_post_processing",
        },
        sort_keys=True,
    )
    assert argv[argv.index("--max-model-len") + 1] == "32768"


def test_client_block_round_trips_through_the_product_loader() -> None:
    """client_config() is the product's config: its dump constructs the product model unchanged."""
    recipe = load_recipe(_recipe_dir())
    config = client_config(recipe, base_url="http://127.0.0.1:8100/v1")
    assert config["model"] == recipe.id and config["recipe"] == recipe.id
    endpoint = RerankEndpoint(**config)
    assert endpoint.max_tokens == 8192 and endpoint.query_max_tokens == 4096
    assert RerankEndpoint.model_validate(config).model == recipe.id


def test_stage1_on_cpu_passes_token_ids_anchors_and_the_served_template(
    tokenizer_snapshot, monkeypatch, pairs_file
) -> None:
    """Stage 1 on CPU: the product's fit, the reference render, the anchor audit and the template.

    20 sampled pairs (one instruction-bearing, on purpose: the recipe declares instruction: none
    and the reference folds none) plus 5 over-length pairs the harness derives per declared shape:
    the anchor audit covers every sample, the token-id (render) equality every pairs-file row.
    """
    recipe = load_recipe(_recipe_dir())
    _pin_tokenizer(monkeypatch, tokenizer_snapshot)
    document = stage1_prompts(recipe, pairs_file, str(sys.executable), over_length_per_shape=5)
    assert document["sampled"] == 25  # 20 pairs + 5 over-length pair samples
    assert document["passed"] is True
    anchor = document["anchor_check"]
    assert anchor["passed"] is True, anchor["failures"][:1]
    assert anchor["checked"] == document["sampled"]  # every sample, in-budget and over
    render = document["render_check"]
    assert render["status"] == "run" and render["passed"] is True, render["failures"][:1]
    assert render["rows"] == len(_PAIRS)
    assert document["template_render_check"]["passed"] is True, document["template_render_check"]["failures"][:1]
    engine = document["engine_tokenize_check"]
    assert engine["status"] == "not_run" and engine["passed"] is None  # no engine on CPU: neutral, never passed
    assert document["fit"]["pair"]["cuts"] >= 5  # the over-length samples were cut, never sent over budget
    assert document["fit"]["pair"]["overhead"] > 0  # the fixed frame (incl. the post-processor's <s>) is reserved


def test_reference_render_subprocess_agrees_on_the_prompt_text(tmp_path) -> None:
    """The reference subprocess (stdlib render mode) writes the paper's exact prompt per row."""
    recipe = load_recipe(_recipe_dir())
    pairs_path = tmp_path / "pairs.jsonl"
    pairs_path.write_text("".join(json.dumps(row) + "\n" for row in _PAIRS[:4]), encoding="utf-8")
    reference = run_reference(
        sys.executable,
        str(_recipe_dir() / recipe.reference.entry),
        mode="render",
        pairs_path=pairs_path,
        out_path=tmp_path / "out.json",
        tokenizer_spec=_TOKENIZER_SPEC,
    )
    assert len(reference["rows"]) == 4
    first = reference["rows"][0]
    assert first["shape"] == "pair"
    assert first["text"] == (
        "Check whether a given document contains information helpful to answer the query.\n"
        f"<Document> {_PAIRS[0]['documents'][0]}\n<Query> {_PAIRS[0]['query']} ??"
    )


def test_mutation_dropping_the_trailing_anchor_segment_turns_the_anchor_check_red(
    tokenizer_snapshot, monkeypatch, pairs_file
) -> None:
    """Dropping the declared template's trailing anchor segment (" ??") must fail the anchor audit."""
    recipe = load_recipe(_recipe_dir())
    template = TemplateSpec.model_validate(recipe.client.get("template"))
    trimmed = template.model_copy(update={"pair": template.segments("pair")[:-1]})
    mutated = recipe.model_copy(update={"client": {**recipe.client, "template": trimmed}})
    assert mutated.client.get("template").segments("pair")[-1].content == "query"  # the tail is now content
    _pin_tokenizer(monkeypatch, tokenizer_snapshot)
    document = stage1_prompts(mutated, pairs_file, str(sys.executable), over_length_per_shape=2)
    assert document["anchor_check"]["passed"] is False
    assert document["anchor_check"]["failures"], "the anchor audit must record the dropped tail"
    assert all(failure["check"] == "tail" for failure in document["anchor_check"]["failures"])


def test_mutation_stripping_the_anchor_from_the_served_template_file_fails_the_template_check(
    tokenizer_snapshot, monkeypatch, tmp_path, pairs_file
) -> None:
    """A served template file that lost the tail anchor renders different ids than the client."""
    import shutil

    copied = tmp_path / "ctxl-rerank-v2-instruct-multilingual-6b"
    shutil.copytree(_recipe_dir(), copied)
    template_file = copied / "score-template-6b.jinja"
    text = template_file.read_text(encoding="utf-8")
    assert text.endswith(" ??")
    template_file.write_text(text.removesuffix(" ??"), encoding="utf-8")
    mutated = load_recipe(copied)
    _pin_tokenizer(monkeypatch, tokenizer_snapshot)
    document = stage1_prompts(mutated, pairs_file, str(sys.executable), over_length_per_shape=1)
    assert document["template_render_check"]["passed"] is False
    assert document["anchor_check"]["passed"] is True  # the declared shape still keeps its anchor


def test_reference_score_mode_needs_its_own_environment(tmp_path) -> None:
    """The reference's score mode reports the missing reference environment instead of a traceback."""
    recipe = load_recipe(_recipe_dir())
    pairs_path = tmp_path / "pairs.jsonl"
    pairs_path.write_text("".join(json.dumps(row) + "\n" for row in _PAIRS[:1]), encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(_recipe_dir() / recipe.reference.entry),
            "--mode",
            "score",
            "--pairs",
            str(pairs_path),
            "--out",
            str(tmp_path / "out.json"),
            "--tokenizer",
            _TOKENIZER_SPEC,
            "--device",
            "cpu",
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert completed.returncode != 0
    assert "transformers" in completed.stderr or "ModuleNotFoundError" in completed.stderr
