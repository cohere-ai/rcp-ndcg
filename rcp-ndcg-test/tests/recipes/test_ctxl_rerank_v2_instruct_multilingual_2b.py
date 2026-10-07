"""The ``ctxl-rerank-v2-instruct-multilingual-2b`` recipe: it validates, and stage 1 passes on CPU.

The CPU stage-1 runs need only the checkpoint's tokenizer files, downloaded into the test's own
temporary directory; when the environment is offline (CI), they skip with that reason instead of
failing. Everything else — the recipe's load through the product's endpoint config, the ``vllm
serve`` argv it renders to, the anchor mutation — runs offline.
"""

from __future__ import annotations

import json
import shutil
import sys
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from rcp_ndcg_test.equivalence.wire import role_client
from rcp_ndcg_vllm import client_config, load_recipe, serve_argv

from rcp_ndcg.data.templates import TemplateSpec

RECIPE_DIR = Path(__file__).resolve().parents[2] / "recipes" / "ctxl-rerank-v2-instruct-multilingual-2b"
MODEL_ID = "ContextualAI/ctxl-rerank-v2-instruct-multilingual-2b"
REVISION = "6ffef5dc552583b8db58dc4a87f79f7aee78d2d9"
MAX_TOKENS = 8192
QUERY_MAX_TOKENS = 4096


def _pair(query: str, document: str, instruction: str | None = None) -> dict:
    """One pairs-file row."""
    return {"query": query, "documents": [document], "instruction": instruction}


#: Fifteen realistic pairs (three with a run-level instruction): the stage-1 sample is these rows
#: plus the five over-length inputs the harness derives per shape, 20 in total.
PAIRS: list[dict] = [
    _pair(
        "What is the capital of France?",
        "Paris is the capital and largest city of France.",
        "Answer with the city name",
    ),
    _pair("Who wrote Pride and Prejudice?", "Pride and Prejudice is an 1813 novel by Jane Austen."),
    _pair(
        "capital of Japan", "Tokyo is the capital of Japan and the seat of the Japanese government.", "Find the city"
    ),
    _pair(
        "definition of entropy",
        "Entropy is a measurable physical property, most commonly associated with a state of disorder.",
    ),
    _pair("largest planet", "Jupiter is the fifth planet from the Sun and the largest in the Solar System."),
    _pair(
        "speed of light",
        "The speed of light in vacuum is a universal physical constant equal to 299,792,458 metres per second.",
    ),
    _pair(
        "who painted the Mona Lisa",
        "The Mona Lisa is a half-length portrait painting by the Italian Renaissance artist Leonardo da Vinci.",
    ),
    _pair("chemical symbol for gold", "Gold is a chemical element with the symbol Au and atomic number 79."),
    _pair(
        "longest river in Africa",
        "The Nile is a major north-flowing river in northeastern Africa, and the longest river in Africa.",
    ),
    _pair(
        "first person on the Moon",
        "Neil Armstrong was an American astronaut who became the first person to walk on the Moon.",
    ),
    _pair(
        "boiling point of water",
        "The boiling point of water is 100 degrees Celsius at standard atmospheric pressure.",
    ),
    _pair(
        "currency of Japan",
        "The yen is the official currency of Japan and the third most traded currency on the foreign exchange market.",
    ),
    _pair(
        "who developed the theory of relativity",
        "Albert Einstein developed the theory of relativity, one of the two pillars of modern physics.",
        "Name the scientist",
    ),
    _pair(
        "main language of Brazil ",
        "Portuguese is the official and national language of Brazil, spoken by nearly the entire population.",
    ),
    _pair("smallest prime number", "The number 2 is the smallest prime number and the only even prime."),
]


@pytest.fixture(scope="module")
def tokenizer_file(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """The checkpoint's ``tokenizer.json``, downloaded into the test's temporary directory.

    Skips with the reason when the Hub is unreachable (an offline CI run): stage 1 measures and
    cuts in the checkpoint tokenizer's tokens, and the product's loader refuses a missing file.
    """
    target = tmp_path_factory.mktemp("tokenizer") / "tokenizer.json"
    url = f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/tokenizer.json"
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=120.0) as response:
            if response.status_code != 200:
                pytest.skip(f"the Hub returned HTTP {response.status_code} for the checkpoint's tokenizer.json")
            with target.open("wb") as handle:
                for chunk in response.iter_bytes():
                    handle.write(chunk)
    except httpx.HTTPError as error:
        pytest.skip(
            f"offline (no route to the Hub): the CPU stage-1 check needs the checkpoint's tokenizer.json ({error})"
        )
    yield target


def local_recipe(tokenizer_file: Path):
    """The recipe with the tokenizer pointed at the downloaded file (the Hub spec is runtime)."""
    recipe = load_recipe(RECIPE_DIR)
    return recipe.model_copy(update={"client": {**recipe.client, "tokenizer": str(tokenizer_file)}})


def write_pairs(path: Path) -> Path:
    """The pairs JSONL file for the stage-1 runs."""
    path.write_text("".join(json.dumps(row) + "\n" for row in PAIRS), encoding="utf-8")
    return path


def test_recipe_loads_and_declares_the_served_contract() -> None:
    """The recipe validates against the product's endpoint config, with every binding field set."""
    from rcp_ndcg.inference.config import RerankEndpoint

    recipe = load_recipe(RECIPE_DIR)
    assert recipe.id == "ctxl-rerank-v2-instruct-multilingual-2b"
    assert recipe.model == MODEL_ID
    assert recipe.revision == REVISION
    assert recipe.role == "rerank" and recipe.scoring == "pointwise" and recipe.input == ["text"]
    assert isinstance(recipe.client, RerankEndpoint)
    # The explicit budget and the raw-logit activation, both declared.
    assert recipe.client.get("tokenizer") == f"{MODEL_ID}@{REVISION}"
    assert recipe.client.get("max_tokens") == MAX_TOKENS
    assert recipe.client.get("query_max_tokens") == QUERY_MAX_TOKENS
    assert recipe.client.get("use_activation") is False
    assert recipe.client.get("on_overflow") == "cut"
    assert recipe.client.get("instruction") == "fold"
    assert recipe.client.get("listwise") is False
    # The declared template: one pair shape, document before query, the trailing anchor.
    template = TemplateSpec.model_validate(recipe.client.get("template"))
    assert template is not None and template.shapes() == ("pair",)
    assert template.anchor == "last"
    assert template.segments("pair")[-1].fixed == " ??"
    assert template.adds_special_tokens("pair") is True
    # The served engine: the shipped template file and the raw-logit pooler default.
    assert recipe.serve.chat_template == "template.jinja"
    assert recipe.serve.pooler_config == {"use_activation": False}
    assert recipe.serve.max_model_len >= recipe.client.get("max_tokens")
    assert recipe.serve.dtype == "bfloat16"
    assert recipe.serve.plugin is None and recipe.serve.trust_remote_code is False
    assert recipe.serve.hf_overrides["architectures"] == ["Qwen3ForSequenceClassification"]
    assert recipe.serve.hf_overrides["classifier_from_token"] == ["!"]
    assert recipe.serve.hf_overrides["method"] == "no_post_processing"
    # The paper deviation is declared, not copied into the served path.
    assert recipe.reference.known_deviations == ["anchor_drop_over_cap"]
    assert recipe.reference.score_scale == "logit"
    assert recipe.status.state == "unverified"
    # The client block is the product's config, round-tripped.
    config = client_config(recipe, base_url="http://127.0.0.1:8100/v1")
    assert RerankEndpoint(**config).model == recipe.id


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
    overrides = json.loads(argv[argv.index("--hf-overrides") + 1])
    assert overrides["architectures"] == ["Qwen3ForSequenceClassification"]
    assert overrides["classifier_from_token"] == ["!"]
    assert overrides["method"] == "no_post_processing"
    assert argv[argv.index("--runner") + 1] == "pooling"


def test_reference_environment_is_documented() -> None:
    """The reference declares the environment it needs, beside itself (the reference rule)."""
    text = (RECIPE_DIR / "requirements-reference.txt").read_text(encoding="utf-8")
    assert "torch==2.9.1" in text
    assert "transformers==4.57.6" in text


def test_reference_resolves_the_hub_tokenizer_spec_without_the_revision_suffix() -> None:
    """The recipe's client.tokenizer (repo@revision) reaches transformers as a bare repo id.

    ``AutoTokenizer.from_pretrained`` rejects 'repo@revision' (the '@' is not a repo-id
    character), so the reference must split the spec and pass the revision separately — the
    harness hands the recipe's spec to the reference subprocess verbatim, and a 'repo@revision'
    repo id would fail every real reference run (verifier round 1, blocker). The reference's
    pinned revision and model id are also pinned here: a drift from the recipe's would only
    surface as a stage-2 score-gate failure on the GPU wave.
    """
    import importlib.util

    module_spec = importlib.util.spec_from_file_location("ctxl_reference", RECIPE_DIR / "reference.py")
    module = importlib.util.module_from_spec(module_spec)
    # exec_module would drop a __pycache__ into the recipe directory; tests write only to tmp_path.
    bytecode = sys.dont_write_bytecode
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


def test_stage1_on_cpu_passes_token_id_equality_and_the_anchor_check(tmp_path: Path, tokenizer_file: Path) -> None:
    """Stage 1 on CPU: token-id equality with the reference and the anchor audit over 20 sampled
    pairs, 5 of them over-length (the over-length inputs pad the pair's own content spans)."""
    from rcp_ndcg_test.equivalence import fitting, stage1_prompts
    from rcp_ndcg_test.equivalence.reference import run_reference

    from rcp_ndcg.data.preprocess import fit

    recipe = local_recipe(tokenizer_file)
    pairs_path = write_pairs(tmp_path / "pairs.jsonl")
    document = stage1_prompts(recipe, pairs_path, sys.executable, over_length_per_shape=5)

    assert document["sampled"] >= 20
    assert document["fit"]["pair"]["cuts"] >= 5, "the sample must include over-length pairs"
    assert document["anchor_check"]["passed"] is True, document["anchor_check"]["failures"][:1]
    assert document["anchor_check"]["checked"] == document["sampled"]
    assert document["render_check"]["status"] == "run"
    assert document["render_check"]["passed"] is True, document["render_check"]["failures"][:1]
    assert document["template_render_check"]["passed"] is True, document["template_render_check"]["failures"][:1]
    assert document["engine_tokenize_check"]["status"] == "not_run"  # no engine on CPU; never "passed"
    assert document["passed"] is True

    # Token-id equality, explicitly: the ids of fit's render equal the ids of the reference's
    # render for every pairs row (the stage-1 render check compares texts; this pins the ids).
    tokenizer = fitting.tokenizer_of(recipe)
    budget = role_client(recipe, None)[0]._resolve_budget()[0].model_copy(update={"tokenizer": tokenizer.name})
    rows = [json.loads(line) for line in pairs_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    reference = run_reference(
        sys.executable,
        str(RECIPE_DIR / recipe.reference.entry),
        mode="render",
        pairs_path=pairs_path,
        out_path=tmp_path / "reference.json",
        tokenizer_spec=str(tokenizer_file),
    )
    assert len(reference["rows"]) == len(rows)
    anchor_id = tokenizer.ids(" ??")[-1]
    for row in reference["rows"]:
        index = int(row["index"])
        query = fitting.fold_query(recipe, rows[index]["query"], rows[index].get("instruction"))
        fitted = fit([(query, rows[index]["documents"][0])], "pair", budget, tokenizer, ids=["0"])
        assert tokenizer.ids(fitted.texts[0], add_special_tokens=True) == tokenizer.ids(
            row["text"], add_special_tokens=True
        ), f"row {index}: the served ids differ from the reference ids"
        # The anchor sits at the scored position in every render.
        assert tokenizer.ids(row["text"])[-1] == anchor_id


def test_dropping_the_trailing_anchor_segment_turns_the_anchor_check_red(tmp_path: Path, tokenizer_file: Path) -> None:
    """The mutation: drop the template's trailing anchor segment and show the anchor check red."""
    import yaml
    from rcp_ndcg_test.equivalence import stage1_prompts

    copied = tmp_path / "ctxl-rerank-v2-instruct-multilingual-2b"
    copied.mkdir()
    for name in ("recipe.yaml", "template.jinja", "reference.py"):
        shutil.copy(RECIPE_DIR / name, copied / name)
    data = yaml.safe_load((copied / "recipe.yaml").read_text(encoding="utf-8"))

    # Drop the trailing anchor segment: the pair shape then ends with the query content. The
    # recipe still loads (add_special_tokens: {pair: true} declares the post-processor's tail as
    # the shape's closing tokens) — and stage 1's anchor check goes red: the rendered ids no
    # longer carry the " ??" tail the model reads its score from.
    del data["client"]["template"]["pair"][-1]
    (copied / "recipe.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    template_text = (copied / "template.jinja").read_text(encoding="utf-8")
    (copied / "template.jinja").write_text(
        template_text.replace("{{ query_text }} ??", "{{ query_text }}"), encoding="utf-8"
    )

    mutated = load_recipe(copied)
    local = mutated.model_copy(update={"client": {**mutated.client, "tokenizer": str(tokenizer_file)}})
    pairs_path = write_pairs(tmp_path / "pairs.jsonl")
    document = stage1_prompts(local, pairs_path, sys.executable, over_length_per_shape=3)
    assert document["anchor_check"]["passed"] is False
    assert document["anchor_check"]["failures"], "the anchor audit must name the failing renders"
    assert document["anchor_check"]["failures"][0]["check"] == "tail"
    assert document["template_render_check"]["passed"] is True  # file and declaration mutated alike
    assert document["render_check"]["passed"] is False  # and the reference still renders the tail
    assert document["passed"] is False
