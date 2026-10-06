"""The zerank-1-small-reranker recipe: validation, CPU stage 1, and the anchor mutation.

The recipe validates against the product's RerankEndpoint at load (the schema does that); these tests
check the recipe's own declared facts and run stage 1 on the real tokenizer files (downloaded into the
lane's scratch directory, or tmp_path in CI; skipped with a clear reason when offline). The mutation
test drops the template's trailing anchor segment and shows the anchor check red.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
from rcp_ndcg_vllm import load_recipe, serve_argv
from rcp_ndcg_vllm.equivalence import stage1_prompts
from rcp_ndcg_vllm.equivalence.fitting import fit_rows, tokenizer_of

RECIPE_DIR = Path(__file__).resolve().parents[2] / "recipes" / "zerank-1-small-reranker"
REVISION = "a65fd51c450e9b47fdddab98e31166ecad21af8d"
REPO = "zeroentropy/zerank-1-small-reranker"
#: The lane's scratch directory (COMMON.md); the download target when it exists, tmp_path otherwise.
SCRATCH_TOKENIZER = Path("/root/repos/rcp-ndcg-lanes/rec-zerank-1-small-reranker/scratch/tokenizer/tokenizer.json")

_RECIPE_FILES = ("recipe.yaml", "zerank_score_template.jinja", "reference.py")


def _tokenizer_cached() -> bool:
    """Whether the recipe tokenizer's file is already in the lane's scratch directory."""
    return SCRATCH_TOKENIZER.is_file()


#: Stage 1 on CPU needs the real tokenizer files: cached in the scratch dir, or downloaded from the
#: Hub (network). The marker names the network need; the skip covers the offline case with a reason.
stage1_env = pytest.mark.skipif(
    not os.environ.get("RCP_NDCG_NETWORK_TESTS") and not _tokenizer_cached(),
    reason="needs the zerank-1-small-reranker tokenizer files: cached in the lane scratch dir, or set "
    "RCP_NDCG_NETWORK_TESTS=1 to download them",
)


def _tokenizer_file(tmp_path: Path) -> Path:
    """The tokenizer.json the recipe names, from the scratch cache or downloaded into it (CI: tmp_path)."""
    if _tokenizer_cached():
        return SCRATCH_TOKENIZER
    target_dir = SCRATCH_TOKENIZER.parent if SCRATCH_TOKENIZER.parent.parent.is_dir() else tmp_path / "tokenizer"
    try:
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(REPO, "tokenizer.json", revision=REVISION, local_dir=str(target_dir)))
    except Exception as error:  # offline (CI) and not cached
        pytest.skip(
            f"offline: the zerank-1-small-reranker tokenizer.json is neither cached in {target_dir} "
            f"nor downloadable ({error})"
        )


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
    assert recipe.id == "zerank-1-small-reranker" == RECIPE_DIR.name
    assert recipe.model == "zeroentropy/zerank-1-small-reranker"
    assert recipe.revision == REVISION
    assert recipe.role == "rerank" and recipe.input == ["text"] and recipe.scoring == "pointwise"
    assert recipe.licence == "apache-2.0"
    client = recipe.client
    assert client.tokenizer == f"{REPO}@{REVISION}"
    assert client.max_tokens == 8192 and client.query_max_tokens == 4096
    assert client.on_overflow == "cut" and client.use_activation is True and client.listwise is False
    assert client.instruction == "none" and client.empty_doc == "send"
    template = client.template
    assert template.shapes() == ("pair",) and template.anchor == "last"
    assert template.adds_special_tokens("pair") is True
    assert [segment.content for segment in template.segments("pair") if segment.content] == ["query", "document"]
    assert recipe.serve.chat_template == "zerank_score_template.jinja"
    assert (RECIPE_DIR / recipe.serve.chat_template).is_file()
    assert recipe.serve.hf_overrides["classifier_from_token"] == ["Yes"]
    assert recipe.serve.pooler_config == {"logit_sigma": 5, "use_activation": True}
    assert recipe.serve.dtype == "bfloat16" and recipe.serve.trust_remote_code is False
    assert recipe.serve.max_model_len >= client.max_tokens
    assert recipe.reference.known_deviations == ["anchor_drop_over_cap"]
    assert recipe.reference.score_scale == "probability"
    assert recipe.status.state == "unverified"
    assert any("r-zerank1" in source for source in recipe.sources)


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
    assert document["fit"]["pair"]["overhead"] == 13
    assert document["fit"]["pair"]["cuts"] == 5  # exactly the over-length samples were cut
    assert document["anchor_check"]["passed"] is True and document["anchor_check"]["checked"] == 25
    assert document["template_render_check"]["passed"] is True
    assert document["engine_tokenize_check"]["status"] == "not_run"  # CPU: never reported as passed
    render_check = document["render_check"]
    assert render_check["status"] == "run" and render_check["passed"] is True
    assert render_check["rows"] == 20  # the reference subprocess's render, byte-identical to fit's


@stage1_env
@pytest.mark.network
def test_reference_render_matches_fit_on_over_budget_pairs(tmp_path: Path) -> None:
    """The reference's anchor-preserving render (its --mode render) is the product's fit, byte for
    byte, including over-budget pairs: the query cut to its share, the document to the rest, the
    assistant header (the anchor) always re-attached."""
    recipe = _resolved_recipe(tmp_path)
    tokenizer = tokenizer_of(recipe)
    im_start = tokenizer.special_text("im_start")
    im_end = tokenizer.special_text("im_end")
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
    fitted = fit_rows(recipe, rows, tokenizer)["per_shape"]["pair"]["texts"]
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
    assert [row["text"] for row in rows_out] == fitted
    header = f"{im_start}assistant\n"
    # The two over-budget renders keep the anchor (the paper's whole-prompt cut would drop it).
    for text in fitted[2:4]:
        assert text.endswith(header), "the anchor must survive an over-budget cut"
    assert len(tokenizer.ids(fitted[2], add_special_tokens=True)) <= 8192  # the budget held
    assert "alfa " * 5999 not in fitted[2]  # the query span was cut to its declared share
    assert "bravo " * 5999 not in fitted[2]  # the document span was cut to the remainder
    assert fitted[4] == (f"{im_start}system\nempty document{im_end}\n{im_start}user\n{im_end}\n{im_start}assistant\n")


@stage1_env
@pytest.mark.network
def test_mutation_dropping_the_anchor_segment_turns_the_anchor_check_red(tmp_path: Path) -> None:
    """Drop the template's trailing anchor segment: the anchor check fails (empty anchor edge), while
    the untouched recipe's anchor check passes (proven by test_stage1_on_cpu_...)."""
    recipe = _resolved_recipe(tmp_path)
    template = recipe.client.template
    client = recipe.client.model_copy(update={"template": template.model_copy(update={"pair": template.pair[:-1]})})
    mutated: Any = recipe.model_copy(update={"client": client})
    assert len(mutated.client.template.segments("pair")) == 4
    document = stage1_prompts(mutated, _pairs_path(tmp_path, _sample_pairs()[:3]), None, over_length_per_shape=1)
    assert document["anchor_check"]["passed"] is False
    assert document["anchor_check"]["failures"]
    assert all(failure["check"] == "tail" for failure in document["anchor_check"]["failures"])
    # The declared shape no longer ends with the header the file emits: the template check is red too.
    assert document["template_render_check"]["passed"] is False
