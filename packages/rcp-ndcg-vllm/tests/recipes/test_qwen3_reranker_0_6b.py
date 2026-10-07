"""The ``qwen3-reranker-0.6b`` recipe: the schema, stage 1 on CPU, and the anchor mutation.

Stage 1 runs on the real Qwen3-Reranker-0.6B tokenizer (``tokenizer.json`` and friends only — no
weights), downloaded into the lane's scratch directory and skipped with a clear reason when
offline. The reference runs as a subprocess (its ``render`` mode needs the tokenizer only; the
harness process never imports torch or transformers).
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

import pytest
import yaml
from rcp_ndcg_vllm import client_config, load_recipe, serve_argv
from rcp_ndcg_vllm.equivalence import stage1_prompts

from rcp_ndcg.data.tokenizer import load_tokenizer
from rcp_ndcg.inference.config import RerankEndpoint

RECIPE_DIR = Path(__file__).resolve().parents[2] / "recipes" / "qwen3-reranker-0.6b"
REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"
TOKENIZER_SHA256 = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
# The tokenizer file set (tokenizer.json plus its sidecars: the added-token names -- im_start,
# im_end, think, /think -- resolve from tokenizer_config.json, so a bare tokenizer.json would
# silently fail template rendering).
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")


def _reference_python() -> str:
    """The interpreter the reference subprocess runs in (a reference env may be named)."""
    return os.environ.get("RCP_NDCG_REFERENCE_PYTHON", sys.executable)


@pytest.fixture
def tokenizer_dir(tmp_path: Path) -> Path:
    """The pinned revision's tokenizer files in the shared download cache, or ``tmp_path``.

    Only tokenizer files are fetched (no weights). The downloads land in the cache the shared
    ``_served`` helper honours (``RCP_NDCG_VLLM_TOKENIZER_CACHE`` -- the lane's scratch dir --
    else ``tmp_path``) and are reused offline; with no cached copy and no network the test skips
    with a clear reason -- the CPU stage-1 check needs the real tokenizer, never a stand-in.
    """
    from ._served import fetch_tokenizer

    for name in TOKENIZER_FILES:
        fetch_tokenizer(
            f"https://huggingface.co/Qwen/Qwen3-Reranker-0.6B/resolve/{REVISION}/{name}",
            f"{RECIPE_DIR.name}/{name}",
            tmp_path,
            sha256=TOKENIZER_SHA256 if name == "tokenizer.json" else None,
        )
    return fetch_tokenizer(
        f"https://huggingface.co/Qwen/Qwen3-Reranker-0.6B/resolve/{REVISION}/tokenizer.json",
        f"{RECIPE_DIR.name}/tokenizer.json",
        tmp_path,
        sha256=TOKENIZER_SHA256,
    ).parent


def _reference_constants() -> dict[str, str]:
    """The reference module's prompt constants, imported from the recipe directory.

    The module's top level imports only the standard library (torch and transformers load lazily
    in ``load``/``score``), so importing it here never pulls weights into the harness process.
    """
    spec = importlib.util.spec_from_file_location("qwen3_reranker_reference", RECIPE_DIR / "reference.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {"prefix": module.PREFIX_TEXT, "suffix": module.SUFFIX_TEXT, "instruction": module.DEFAULT_INSTRUCTION}


def _local_recipe(tmp_path: Path, tokenizer_dir: Path):
    """The recipe with its declared Hub tokenizer spec pointed at the downloaded files.

    The copy loads through the same ``load_recipe`` validation; only the tokenizer spec changes
    (same file bytes, same SHA-256 identity), so stage 1 never touches the network.
    """
    target = tmp_path / RECIPE_DIR.name
    shutil.copytree(RECIPE_DIR, target)
    data = yaml.safe_load((target / "recipe.yaml").read_text(encoding="utf-8"))
    data["client"]["tokenizer"] = str(tokenizer_dir)
    (target / "recipe.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return load_recipe(target)


def _pairs(tmp_path: Path) -> tuple[Path, list[dict]]:
    """The stage-1 pairs: 20 in-budget rows plus 5 over-budget ones (no ``shape`` key, so the
    render check compares all 25 against the reference subprocess's paper render)."""
    rows = [
        {
            "query": f"query {index}: what is the capital of country {index}?",
            "documents": [
                f"document {index}: The capital of country {index} is city {index}, a place with "
                "rivers and history." * (1 + index % 3)
            ],
        }
        for index in range(20)
    ]
    # Over the 8192-token budget: english filler, CJK filler, mixed punctuation, a long query with
    # a long document, and a huge single-word run — the anchor must survive every cut.
    rows.append({"query": "q" * 5, "documents": ["word filler sentence number one two three four five six. " * 1500]})
    rows.append({"query": "short", "documents": ["汉字填充句子，用于测试分词边界与截断行为。" * 700]})
    rows.append(
        {
            "query": "a query with punctuation! and numbers 12345 plus UTF-8 emoji balloon",
            "documents": ["Mixed 汉字 and latin words, with punctuation; colons: and quotes - plus balloons. " * 500],
        }
    )
    rows.append(
        {"query": "longer query " * 200, "documents": ["filler text for the document side of the pair. " * 900]}
    )
    rows.append({"query": "q", "documents": ["x " * 12000]})
    # A non-NFC pair (decomposed accents): the engine tokenizes the NFC-normalized form, so the
    # ids match, but the reference's render must keep the raw characters for the render check.
    decomposed = "cafe" + chr(101) + chr(769)  # e + combining acute (not NFC)
    rows.append(
        {"query": f"what about {decomposed}?", "documents": [f"The {decomposed} is served over the river." * 3]}
    )
    path = tmp_path / "pairs.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path, rows


def test_recipe_loads_and_validates() -> None:
    """The recipe loads against the product's RerankEndpoint, with the paper's budgets."""
    recipe = load_recipe(RECIPE_DIR)
    assert recipe.id == "qwen3-reranker-0.6b" == RECIPE_DIR.name
    assert recipe.model == "Qwen/Qwen3-Reranker-0.6B"
    assert recipe.revision == REVISION and len(recipe.revision) == 40
    assert recipe.role == "rerank" and recipe.scoring == "pointwise" and recipe.input == ["text"]
    assert recipe.licence == "apache-2.0"
    assert recipe.engine.image == "vllm/vllm-openai:v0.31.0" and recipe.serve.dtype == "bfloat16"
    assert recipe.serve.max_model_len >= recipe.client.max_tokens  # the engine must not 400 the budget
    assert recipe.serve.chat_template == "template.jinja"  # R10: the template file ships
    client = recipe.client
    assert client.tokenizer == f"Qwen/Qwen3-Reranker-0.6B@{REVISION}"
    assert client.max_tokens == 8192 and client.query_max_tokens == 4096  # the paper's budgets
    assert client.instruction == "none"  # the paper's served path sends no instruction
    assert client.use_activation is True  # probability-scale head, matching score_scale
    assert client.on_overflow == "cut"
    assert client.template.anchor == "last"
    # One over-cap policy family-wide (the operator's 09:2x decision): the reference keeps every
    # anchor and cuts over-cap content the paper's way (the joint longest_first pair cut), never
    # copying the client's cut -- so over-cap rows are reported, not gated.
    assert recipe.reference.score_scale == "probability"
    assert recipe.reference.known_deviations == ["over_cap_cut_differs"]
    assert recipe.status.state == "unverified"


def test_client_config_round_trips_through_the_product() -> None:
    """The client block is the product's endpoint config: the dump constructs the model unchanged."""
    recipe = load_recipe(RECIPE_DIR)
    config = client_config(recipe, base_url="http://127.0.0.1:8100/v1")
    assert config["model"] == recipe.id
    # The authored client.recipe string is kept as declared (harness: never overwritten with the id).
    assert config["recipe"] == recipe.client.recipe
    endpoint = RerankEndpoint(**config)
    assert str(endpoint.base_url) == "http://127.0.0.1:8100/v1"
    again = RerankEndpoint.model_validate(config)
    assert again.model == recipe.id


def test_serve_argv_renders_the_paper_engine_command() -> None:
    """The argv a wave runs: pooling runner, the conversion overrides, the shipped template."""
    recipe = load_recipe(RECIPE_DIR)
    argv = serve_argv(recipe, port=8100, served_model_name=recipe.id)
    assert argv[:3] == ["vllm", "serve", "Qwen/Qwen3-Reranker-0.6B"]
    assert argv[argv.index("--revision") + 1] == REVISION
    assert argv[argv.index("--served-model-name") + 1] == recipe.id
    overrides = json.loads(argv[argv.index("--hf-overrides") + 1])
    assert overrides["architectures"] == ["Qwen3ForSequenceClassification"]
    assert overrides["classifier_from_token"] == ["no", "yes"]
    assert overrides["is_original_qwen3_reranker"] is True
    assert argv[argv.index("--chat-template") + 1] == str(RECIPE_DIR / "template.jinja")
    assert argv[argv.index("--max-model-len") + 1] == "10000"
    assert argv[argv.index("--dtype") + 1] == "bfloat16"
    assert "--trust-remote-code" not in argv and "--convert" not in argv
    # The server-side activation pin beside the wire's use_activation: true (one family rule).
    assert json.loads(argv[argv.index("--pooler-config") + 1]) == {"use_activation": True}


def test_served_template_renders_the_reference_frame_for_both_callers(tokenizer_dir: Path) -> None:
    """The shipped template is dual-mode: the check's query/document render and the engine's
    messages render both produce the paper prompt — byte-identical to the declared shape's render."""
    from jinja2 import StrictUndefined
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    reference = _reference_constants()
    recipe = load_recipe(RECIPE_DIR)
    tokenizer = load_tokenizer(str(tokenizer_dir / "tokenizer.json"))
    assert tokenizer.sha256 == TOKENIZER_SHA256
    query, document = "capital of france", "Paris is the capital of France."
    declared = recipe.client.template.render("pair", tokenizer, query=query, document=document)
    paper = (
        reference["prefix"]
        + f"<Instruct>: {reference['instruction']}\n<Query>: {query}\n<Document>: {document}"
        + reference["suffix"]
    )
    environment = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=False, undefined=StrictUndefined
    )
    template = environment.from_string((RECIPE_DIR / recipe.serve.chat_template).read_text(encoding="utf-8"))
    assert template.render(query=query, document=document, instruction="") == declared == paper
    # The engine branch: only `messages` in scope, exactly vLLM's safe_apply_chat_template call.
    assert (
        template.render(messages=[{"role": "query", "content": query}, {"role": "document", "content": document}])
        == paper
    )


def test_stage1_on_cpu_passes_with_the_reference_render(tmp_path: Path, tokenizer_dir: Path) -> None:
    """Stage 1 on CPU: 20 in-budget + 5 over-budget pairs, the span comparison and the anchor audit.

    The reference subprocess renders the paper's own cut content spans itself (tokenizer only,
    no weights): the anchor-preserving joint pair cut at raw character offsets, never the client's
    cut. The shipped spans must equal it byte for byte on every under-cap row; the 5 over-budget
    pairs are reported non-gating under the over_cap_cut_differs declaration. Every anchor must
    survive the over-length inputs (this recipe's 5, plus the 20 per shape stage 1 samples itself).
    """
    recipe = _local_recipe(tmp_path, tokenizer_dir)
    pairs, _ = _pairs(tmp_path)
    report = stage1_prompts(recipe, pairs, _reference_python(), over_length_per_shape=20)
    assert report["passed"] is True, json.dumps(report)[:2000]
    assert report["render_check"]["status"] == "run" and report["render_check"]["rows"] == 26
    assert report["render_check"]["passed"] is True, report["render_check"]["failures"][:1]
    assert report["anchor_check"]["passed"] is True, report["anchor_check"]["failures"][:1]
    assert report["anchor_check"]["checked"] == 92  # 26 rows + 20 over-length samples: one check per span
    assert report["template_render_check"]["passed"] is True
    assert report["engine_tokenize_check"]["status"] == "not_run"  # no engine on CPU: never "passed"


def test_stage1_pairs_carry_five_over_budget_rows(tmp_path: Path, tokenizer_dir: Path) -> None:
    """The 5 marked rows of the pairs file are genuinely over the 8192-token budget."""
    recipe = _local_recipe(tmp_path, tokenizer_dir)
    tokenizer = load_tokenizer(str(tokenizer_dir / "tokenizer.json"))
    _, rows = _pairs(tmp_path)
    overhead = recipe.client.template.overhead("pair", tokenizer)
    counts = [overhead + tokenizer.count(row["query"]) + tokenizer.count(row["documents"][0]) for row in rows]
    assert sum(count > recipe.client.max_tokens for count in counts) >= 5


def test_mutation_dropping_the_trailing_anchor_segment_reddens_the_template_check(
    tmp_path: Path, tokenizer_dir: Path
) -> None:
    """Mutation, on the wire's own contract: dropping the trailing anchor segment loses the
    assistant suffix — the scored position — and the file-vs-declaration check must go red.

    The rerank wire carries the cut content spans (the frame is the engine's own template), so
    stage 1's anchor audit audits the settled query and the document spans and does not move on a
    frame change; the frame contract is pinned by ``template_render_check``, which this mutation
    turns red (the rendered ids no longer end with the declared anchor), and the schema itself
    refuses the un-pinned shape.
    """
    from rcp_ndcg.data.templates import TemplateSpec

    reference = _reference_constants()
    recipe = _local_recipe(tmp_path, tokenizer_dir)
    tokenizer = load_tokenizer(str(tokenizer_dir / "tokenizer.json"))
    template = recipe.client.template
    assert template is not None

    # the real frame ends with the assistant suffix (the scored anchor), via the product's render
    golden = template.render("pair", tokenizer, query="capital of france", document="paris is the capital of France.")
    real_ids = tokenizer.ids(golden, add_special_tokens=True)
    suffix_ids = tokenizer.ids(reference["suffix"], add_special_tokens=True)
    assert real_ids[-len(suffix_ids) :] == suffix_ids

    # the mutation: drop the trailing anchor segment, keep the shape loadable by declaring the
    # tokenizer's post-processor as the anchor — which for Qwen adds nothing, so the anchor is
    # really gone (the product refuses the same shape without that declaration)
    anchorless = TemplateSpec(
        pair=tuple(segment for segment in template.pair[:-1]), anchor="last", add_special_tokens={"pair": True}
    )
    client = recipe.client.model_copy(update={"template": anchorless})
    mutated = recipe.model_copy(update={"client": client})
    assert mutated.client.template is not None
    no_tail_ids = tokenizer.ids(
        mutated.client.template.render(
            "pair", tokenizer, query="capital of france", document="paris is the capital of France."
        ),
        add_special_tokens=True,
    )
    assert no_tail_ids[-len(suffix_ids) :] != suffix_ids, "the mutation must really lose the anchor"

    rows = [
        {"query": "capital of france", "documents": ["paris is the capital of france."]},
        {"query": "who wrote hamlet", "documents": ["hamlet was written by william shakespeare around 1600."]},
        {"query": "speed of light", "documents": ["light travels at about 299792 kilometres per second."]},
    ]
    pairs = tmp_path / "mutation-pairs.jsonl"
    pairs.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    report = stage1_prompts(mutated, pairs, _reference_python(), over_length_per_shape=2)
    assert report["template_render_check"]["passed"] is False  # the file still emits the dropped suffix
    assert report["anchor_check"]["passed"] is True  # the wire's spans are unchanged by the frame drop

    # and the schema itself refuses an 'anchor: last' shape with neither a fixed tail nor the
    # post-processor declaration
    with pytest.raises(ValueError, match="anchor: last"):
        TemplateSpec(pair=tuple(segment for segment in template.pair[:-1]), anchor="last", add_special_tokens=False)


def test_non_nfc_rows_compare_byte_identical_spans(tmp_path: Path, tokenizer_dir: Path) -> None:
    """NFD (non-NFC) input: the render comparison byte-equals the RAW characters.

    ``decode(encode(x))`` is not the identity for this checkpoint (its normalizer maps non-NFC text
    to NFC, keeping the ids equal but not the characters): a decode round trip renders different
    BYTES for the same kept span. The reference cuts at raw character offsets (verbatim prefixes),
    so on decomposed-accent rows the shipped spans and the reference's match byte for byte — and
    this goes red the moment either side decodes instead of cutting.
    """
    decomposed = "cafe" + chr(101) + chr(769)  # e + combining acute: NFD, never NFC
    rows = [
        {"query": f"what about {decomposed}?", "documents": [f"The {decomposed} is served over the river."]},
        {
            "query": f"menu of the {decomposed} {decomposed} house",
            "documents": [f"{decomposed} soup and {decomposed} pie, with notes on the {decomposed} " * 12],
        },
    ]
    recipe = _local_recipe(tmp_path, tokenizer_dir)
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    report = stage1_prompts(recipe, pairs, _reference_python(), over_length_per_shape=1)
    assert report["render_check"]["status"] == "run" and report["render_check"]["rows"] == 2
    assert report["render_check"]["passed"] is True, report["render_check"]["failures"][:1]


def test_the_client_settles_an_over_share_query_once_at_its_share(tmp_path: Path, tokenizer_dir: Path) -> None:
    """The settle rule, pinned on the wire: an over-share query is NOT sent whole.

    The rerank client settles the shared query once per call and ships it at its declared
    ``query_max_tokens`` (4096) whenever the query exceeds it — the shipped span is a verbatim
    prefix of the raw query (a raw character cut, never a decode round trip), identical for every
    document of the request.
    """
    from ._served import served_pair

    recipe = _local_recipe(tmp_path, tokenizer_dir)
    tokenizer = load_tokenizer(str(tokenizer_dir / "tokenizer.json"))
    long_query = "over share query filler token " * 1400
    assert tokenizer.count(long_query) > 4096
    spans = served_pair(recipe, long_query, ["a short document.", "another short document."])
    settled = spans["query"]
    assert settled != long_query, "an over-share query must be cut at its share, never sent whole"
    assert long_query.startswith(settled), "the settled span must be a verbatim prefix of the raw query"
    assert 4080 <= tokenizer.count(settled) <= 4096


def test_the_short_query_ships_whole(tmp_path: Path, tokenizer_dir: Path) -> None:
    """The other half of the settle rule: an under-share query is sent uncut (nothing settles it)."""
    from ._served import served_pair

    recipe = _local_recipe(tmp_path, tokenizer_dir)
    query = "capital of france"
    spans = served_pair(recipe, query, ["paris is the capital of france."])
    assert spans["query"] == query
