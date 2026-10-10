"""The ``qwen3-reranker`` family: its contract per variant, stage 1 on CPU, the paper's own cut, and
the mutations (decision 34: one family module, parametrized over its variant ids; every field pinned
per variant; two mutants red per family).

Stage 1 runs on the real tokenizer files of the pinned revisions (no weights), fetched through the shared
``_served.fetch_tokenizer`` into ``RCP_NDCG_VLLM_TOKENIZER_CACHE`` (else ``tmp_path``) and skipped with a
clear reason offline. The reference runs as a subprocess (its ``render`` mode needs the tokenizer only; the
harness process never imports torch or transformers) and reads its variant from the resolved recipe the
tests pass as ``--recipe``.

The family's over-cap policy (owner decision 9): the reference is the paper's ``QwenOGRerank`` cut -- the
pair string right-cut at ``8192 - 39 - 9`` tokens, both anchors re-attached -- and never the client's, so
the recipe declares ``over_cap_cut_differs``: over-cap rows are reported, not gated, and under-cap rows gate
exactly (an over-share query under the budget included: the reference never ports the settle rule).
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
from rcp_ndcg_test.equivalence import stage1_prompts
from rcp_ndcg_vllm.recipe import (
    client_config,
    default_recipes_root,
    resolve_recipe,
    serve_argv,
)

from rcp_ndcg.data.tokenizer import load_tokenizer
from rcp_ndcg.inference.config import RerankEndpoint

from ._contract import assert_recipe_contract
from ._served import client_template, fetch_tokenizer, served_pair, served_rows

RECIPES = default_recipes_root()
FAMILY_DIR = RECIPES / "qwen3-reranker"

#: The family's variants (decision 34): the per-size facts the tests pin. One tokenizer.json is
#: byte-identical at all three pinned revisions (one SHA-256); the budgets and the frame are the
#: family's shared contract, the family client omits the endpoint defaults (``request_shape``,
#: ``listwise``) the old standalone recipes declared.
VARIANTS: dict[str, dict[str, Any]] = {
    "qwen3-reranker-0.6b": {
        "repo": "Qwen/Qwen3-Reranker-0.6B",
        "revision": "e61197ed45024b0ed8a2d74b80b4d909f1255473",
        "sha256": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    },
    "qwen3-reranker-4b": {
        "repo": "Qwen/Qwen3-Reranker-4B",
        "revision": "22e683669bc0f0bd69640a1354a6d0aebcfeede5",
        "sha256": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    },
    "qwen3-reranker-8b": {
        "repo": "Qwen/Qwen3-Reranker-8B",
        "revision": "77d193c791ed757ca307ee72715aa132723da912",
        "sha256": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    },
}
VARIANT_IDS = list(VARIANTS)

#: The tokenizer file set (tokenizer.json, SHA-256 pinned, plus its sidecars).
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")

MAX_TOKENS = 8192
QUERY_MAX_TOKENS = 4096
OVERHEAD = 73  # the frame's fixed tokens: 39-token prefix + 9-token suffix + 25 for the instruction and labels
PAIR_BUDGET = MAX_TOKENS - 39 - 9  # the paper's pair-string budget (QwenOGRerank._process_inputs)
#: The assistant suffix -- the scored anchor -- as ids of the pinned tokenizer.
SUFFIX_IDS = [151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271]

# chat-template markers, built without typing them literally
IM_START = chr(60) + "|im_start|>"
IM_END = chr(60) + "|im_end|>"
THINK_OPEN = chr(60) + "think" + chr(62)
THINK_CLOSE = chr(60) + "/" + "think" + chr(62)
THINK_NAME = chr(60) + "think" + chr(62)
THINK_CLOSE_NAME = chr(60) + "/" + "think" + chr(62)
INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
PREFIX = (
    f"{IM_START}system\nJudge whether the Document meets the requirements based on the Query "
    f'and the Instruct provided. Note that the answer can only be "yes" or "no".{IM_END}\n{IM_START}user\n'
)
HEADER = f"<Instruct>: {INSTRUCTION}\n<Query>: "
MID = "\n<Document>: "
SUFFIX = f"{IM_END}\n{IM_START}assistant\n{THINK_OPEN}\n\n{THINK_CLOSE}\n\n"


def _expected_contract(variant_id: str) -> dict[str, Any]:
    """The variant's full resolved contract: every field of every block, exactly as the product models
    resolve it (authored values and schema defaults alike). Nothing may ride unpinned."""
    variant = VARIANTS[variant_id]
    return {
        "serve": {
            "patches": [],
            "runner": "pooling",
            "convert": None,
            "hf_overrides": {
                "architectures": ["Qwen3ForSequenceClassification"],
                "classifier_from_token": ["no", "yes"],
                "is_original_qwen3_reranker": True,
                "head_dtype": "model",
            },
            "chat_template": "template.jinja",
            "pooler_config": {"use_activation": True},
            "trust_remote_code": False,
            "max_model_len": 10000,
            "dtype": "bfloat16",
            "plugin": None,
            "io_processor_plugin": None,
            "mm_processor_kwargs": {},
            "limit_mm_per_prompt": None,
            "extra_args": [],
        },
        "client": {
            "api": "rerank",
            "tokenizer": f"{variant['repo']}@{variant['revision']}",
            "max_tokens": 8192,
            "query_max_tokens": 4096,
            "template": {
                "pair": [
                    {
                        "fixed": "{special:im_start}system\n"
                        "Judge whether the Document meets the requirements based on the Query and "
                        'the Instruct provided. Note that the answer can only be "yes" or "no".'
                        "{special:im_end}\n"
                        "{special:im_start}user\n"
                        "<Instruct>: Given a web search query, retrieve relevant passages that "
                        "answer the query\n"
                        "<Query>: "
                    },
                    {"content": "query"},
                    {"fixed": "\n<Document>: "},
                    {"content": "document"},
                    {
                        "fixed": "{special:im_end}\n{special:im_start}assistant\n"
                        "{special:" + THINK_NAME + "}\n\n{special:" + THINK_CLOSE_NAME + "}\n\n"
                    },
                ],
                "anchor": "last",
                "add_special_tokens": True,
            },
            "instruction": "none",
            "use_activation": True,
            "on_overflow": "cut",
            "empty_doc": "send",
            "empty_query": "send",
            "model": variant_id,
            "revision": variant["revision"],
        },
        "reference": {
            "attn_implementation": "sdpa",
            "kind": "transformers",
            "score_scale": "probability",
            "entry": "reference.py",
            "known_deviations": ["over_cap_cut_differs"],
            "device": None,  # the schema default
        },
    }


def _expected_top(variant_id: str) -> dict[str, Any]:
    variant = VARIANTS[variant_id]
    return {
        "id": variant_id,
        "model": variant["repo"],
        "revision": variant["revision"],
        "role": "rerank",
        "input": ["text"],
        "scoring": "pointwise",
        "licence": "apache-2.0",
    }


def _reference_python() -> str:
    """The interpreter the reference subprocess runs in (a reference env may be named)."""
    return os.environ.get("RCP_NDCG_REFERENCE_PYTHON", sys.executable)


@pytest.fixture(params=VARIANT_IDS)
def variant_id(request: pytest.FixtureRequest) -> str:
    """One variant id of the family (every test runs per variant)."""
    return str(request.param)


@pytest.fixture
def tokenizer_dir(tmp_path: Path, variant_id: str) -> Path:
    """The pinned revision's tokenizer files in the shared download cache (one home: ``_served``)."""
    variant = VARIANTS[variant_id]
    for name in TOKENIZER_FILES:
        path = fetch_tokenizer(
            f"https://huggingface.co/{variant['repo']}/resolve/{variant['revision']}/{name}",
            f"{FAMILY_DIR.name}/{name}",
            tmp_path,
            sha256=variant["sha256"] if name == "tokenizer.json" else None,
        )
    return path.parent


def _local_recipe(tmp_path: Path, tokenizer_dir: Path, variant_id: str) -> Any:
    """The family copied into ``tmp_path`` with its shared tokenizer spec pointed at the downloaded
    files (same bytes, same SHA-256 identity), the variant resolved through the same loader."""
    target = tmp_path / FAMILY_DIR.name
    shutil.copytree(FAMILY_DIR, target)
    data = yaml.safe_load((target / "family.yaml").read_text(encoding="utf-8"))
    data["client"]["tokenizer"] = str(tokenizer_dir)
    (target / "family.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return resolve_recipe(variant_id, root=tmp_path)


def _write_pairs(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _resolved_recipe_file(tmp_path: Path, recipe: Any) -> Path:
    """The resolved recipe JSON, exactly what the harness's ``run_reference`` passes as ``--recipe``."""
    path = tmp_path / "reference-render.recipe.json"
    path.write_text(json.dumps(recipe.model_dump(mode="json"), sort_keys=True), encoding="utf-8")
    return path


def _reference_render(tmp_path: Path, recipe: Any, rows: list[dict]) -> list[dict]:
    """The reference subprocess's ``--mode render`` rows for ``rows``."""
    out = tmp_path / "reference-render.json"
    completed = subprocess.run(
        [
            _reference_python(),
            str(FAMILY_DIR / "reference.py"),
            "--mode",
            "render",
            "--pairs",
            str(_write_pairs(tmp_path / "reference-pairs.jsonl", rows)),
            "--out",
            str(out),
            "--tokenizer",
            str(recipe.client.get("tokenizer")),
            "--recipe",
            str(_resolved_recipe_file(tmp_path, recipe)),
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stderr[-800:]
    return json.loads(out.read_text(encoding="utf-8"))["rows"]


def _pairs() -> list[dict]:
    """The stage-1 pairs: 20 in-budget rows, 5 over-budget ones (queries within the share) and one
    non-NFC row; no ``shape`` key, so the render check compares every row."""
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
    # Over the 8192-token budget: english filler, CJK filler, mixed punctuation, a long query with a long
    # document, and a single-word run -- the anchor must survive every cut.
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
    # A non-NFC pair (decomposed accents): the engine tokenizes the NFC-normalized form, so the ids match,
    # but both sides must keep the raw characters for the render check.
    decomposed = "cafe" + chr(101) + chr(769)  # e + combining acute (not NFC)
    rows.append(
        {"query": f"what about {decomposed}?", "documents": [f"The {decomposed} is served over the river." * 3]}
    )
    return rows


def _contract(recipe: Any, variant_id: str) -> None:
    expected = _expected_contract(variant_id)
    assert_recipe_contract(
        recipe,
        serve=expected["serve"],
        client=expected["client"],
        reference=expected["reference"],
        top=_expected_top(variant_id),
    )


def test_recipe_contract_pins_every_field(variant_id: str) -> None:
    """The recipe loads against the product's RerankEndpoint and declares exactly its contract."""
    recipe = resolve_recipe(variant_id)
    _contract(recipe, variant_id)
    assert recipe.id == variant_id
    assert recipe.engine.image == "vllm/vllm-openai:v0.31.0" and recipe.engine.min_version == "0.31.0"
    assert recipe.serve.max_model_len >= recipe.client.get("max_tokens")  # the engine must not 400 the budget
    assert (FAMILY_DIR / recipe.serve.chat_template).is_file()  # R10: the template file ships
    assert (FAMILY_DIR / "requirements-reference.txt").is_file()  # the family's one convention
    assert recipe.status.state == "unverified"
    assert recipe.sources


def test_the_contract_reds_on_two_mutants(variant_id: str) -> None:
    """Two mutants of the declared contract red the pin: a served field and the declared deviation."""
    recipe = resolve_recipe(variant_id)
    serve_mutant = recipe.model_copy(update={"serve": recipe.serve.model_copy(update={"max_model_len": 16384})})
    with pytest.raises(AssertionError, match="max_model_len"):
        _contract(serve_mutant, variant_id)
    reference_mutant = recipe.model_copy(
        update={"reference": recipe.reference.model_copy(update={"known_deviations": []})}
    )
    with pytest.raises(AssertionError, match="known_deviations"):
        _contract(reference_mutant, variant_id)


def test_client_config_round_trips_through_the_product(variant_id: str) -> None:
    """The client block is the product's endpoint config: the dump constructs the model unchanged."""
    recipe = resolve_recipe(variant_id)
    config = client_config(recipe, base_url="http://127.0.0.1:8100/v1")
    assert config["model"] == recipe.id
    assert config["recipe"] == recipe.id  # the pointer is filled with the recipe's own id when undeclared
    endpoint = RerankEndpoint(**config)
    assert str(endpoint.base_url) == "http://127.0.0.1:8100/v1"


def test_serve_argv_renders_the_engine_command(variant_id: str) -> None:
    """The argv a wave runs: pooling runner, the conversion overrides, the shipped template, the pin."""
    recipe = resolve_recipe(variant_id)
    argv = serve_argv(recipe, port=8100, served_model_name=recipe.id)
    assert argv[:3] == ["vllm", "serve", VARIANTS[variant_id]["repo"]]
    assert argv[argv.index("--revision") + 1] == VARIANTS[variant_id]["revision"]
    assert argv[argv.index("--served-model-name") + 1] == recipe.id
    assert argv[argv.index("--runner") + 1] == "pooling"
    assert json.loads(argv[argv.index("--hf-overrides") + 1]) == _expected_contract(variant_id)["serve"]["hf_overrides"]
    assert argv[argv.index("--chat-template") + 1] == str(FAMILY_DIR / "template.jinja")
    assert argv[argv.index("--max-model-len") + 1] == "10000"
    assert argv[argv.index("--dtype") + 1] == "bfloat16"
    assert json.loads(argv[argv.index("--pooler-config") + 1]) == {"use_activation": True}
    assert "--trust-remote-code" not in argv and "--convert" not in argv


def test_the_template_file_renders_the_paper_prompt_for_both_callers(tokenizer_dir: Path, variant_id: str) -> None:
    """The family's one template file renders the paper prompt for the engine's ``messages`` call and the
    stage-1 check's variables alike -- byte-identical to the declared shape."""
    from jinja2 import StrictUndefined
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    recipe = resolve_recipe(variant_id)
    tokenizer = load_tokenizer(str(tokenizer_dir / "tokenizer.json"))
    assert tokenizer.sha256 == VARIANTS[variant_id]["sha256"]
    text = (FAMILY_DIR / recipe.serve.chat_template).read_text(encoding="utf-8")
    for query, document in [("capital of france", "Paris is the capital of France."), ("", "")]:
        paper = PREFIX + HEADER + query + MID + document + SUFFIX
        declared = client_template(recipe).render("pair", tokenizer, query=query, document=document)
        assert declared == paper
        messages = [{"role": "query", "content": query}, {"role": "document", "content": document}]
        # transformers' compile settings (the engine's safe_apply_chat_template) and the harness's strict ones
        for undefined in ({}, {"undefined": StrictUndefined}):
            environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, **undefined)
            template = environment.from_string(text)
            assert template.render(messages=messages, add_generation_prompt=False) == paper
            assert template.render(query=query, document=document, instruction="") == paper
    # a request instruction is not read: the declared frame bakes the paper's default in
    environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    messages = [{"role": "query", "content": "q"}, {"role": "document", "content": "d"}]
    assert environment.from_string(text).render(messages=messages, instruction="other task") == (
        PREFIX + HEADER + "q" + MID + "d" + SUFFIX
    )


def test_stage1_on_cpu_passes_with_the_reference_render(tmp_path: Path, tokenizer_dir: Path, variant_id: str) -> None:
    """Stage 1 on CPU: 21 in-budget + 5 over-budget rows, the span comparison and the anchor audit.

    The reference subprocess renders the paper's own cut spans (tokenizer only, no weights). The shipped
    spans equal them byte for byte on every under-cap row; over-cap rows whose cuts differ are reported
    in the non-gating table under the over_cap_cut_differs declaration. Every anchor survives the
    over-length inputs (this file's 5 plus the 20 stage 1 samples itself).
    """
    recipe = _local_recipe(tmp_path, tokenizer_dir, variant_id)
    rows = _pairs()
    tokenizer = load_tokenizer(str(tokenizer_dir / "tokenizer.json"))
    counts = [OVERHEAD + tokenizer.count(row["query"]) + tokenizer.count(row["documents"][0]) for row in rows]
    assert sum(count > MAX_TOKENS for count in counts) == 5
    report = stage1_prompts(
        recipe, _write_pairs(tmp_path / "pairs.jsonl", rows), _reference_python(), over_length_per_shape=20
    )
    assert report["passed"] is True, json.dumps(report)[:2000]
    render = report["render_check"]
    assert render["status"] == "run" and render["rows"] == 26 and render["passed"] is True, render["failures"][:1]
    assert report["anchor_check"]["passed"] is True, report["anchor_check"]["failures"][:1]
    assert report["template_render_check"]["passed"] is True
    assert report["engine_tokenize_check"]["status"] == "not_run"  # no engine on CPU: never "passed"


def test_the_reference_renders_the_papers_cut_never_the_clients(
    tmp_path: Path, tokenizer_dir: Path, variant_id: str
) -> None:
    """Decision 9 on this reference: under the cap the reference's spans are the client's, byte for byte;
    over it they are the paper's own pair cut (8144 pair tokens, the query never settled at a share), so an
    over-share query shows the two cuts apart -- and stage 1 reports that row, non-gating."""
    recipe = _local_recipe(tmp_path, tokenizer_dir, variant_id)
    tokenizer = load_tokenizer(str(tokenizer_dir / "tokenizer.json"))
    over_share_query = "alfa bravo charlie delta " * 700  # 4900 tokens: over the share, the client settles it
    long_document = "echo foxtrot golf hotel " * 1300
    rows = [
        {"query": "capital of france", "documents": ["Paris is the capital of France.", "Lyon is in France."]},
        {"query": "short query", "documents": ["india juliett kilo lima " * 2600]},  # over cap, query in share
        {"query": over_share_query, "documents": [long_document]},  # over cap, query over its share
    ]
    assert tokenizer.count(over_share_query) > QUERY_MAX_TOKENS
    reference = _reference_render(tmp_path, recipe, rows)
    served = served_rows(recipe, rows, tokenizer)["per_shape"]["pair"]["spans"]
    # under the cap: identical
    assert reference[0]["query"] == served[0]["query"] == rows[0]["query"]
    assert reference[0]["documents"] == served[0]["documents"] == rows[0]["documents"]
    # over the cap: the paper's pair cut, exactly -- the kept pair re-tokenizes to the budget's first ids
    for index in (1, 2):
        pair = HEADER + rows[index]["query"] + MID + rows[index]["documents"][0]
        kept = HEADER + reference[index]["query"] + MID + reference[index]["documents"][0]
        assert tokenizer.ids(kept) == tokenizer.ids(pair)[:PAIR_BUDGET]
    # the over-share query: the paper keeps it whole inside its pair cut, the client settles it at its share
    assert reference[2]["query"] == over_share_query
    assert served[2]["query"] != over_share_query and tokenizer.count(served[2]["query"]) <= QUERY_MAX_TOKENS
    report = stage1_prompts(
        recipe, _write_pairs(tmp_path / "pairs.jsonl", rows), _reference_python(), over_length_per_shape=1
    )
    assert report["render_check"]["passed"] is True, report["render_check"]["failures"][:1]
    assert report["render_check"]["over_cap"]["known_deviation"] is True
    assert 2 in {row["index"] for row in report["render_check"]["over_cap"]["rows"]}


def test_an_over_share_query_under_the_budget_is_reported(tmp_path: Path, tokenizer_dir: Path, variant_id: str) -> None:
    """A query the client settles at its share in a pair the budget takes whole is a change the client made
    (decision 9): the paper keeps it whole and the reference never ports the settle rule, so under the declared
    over-cap deviation the row's query span is reported, not gated -- and the report names ``query_share``."""
    recipe = _local_recipe(tmp_path, tokenizer_dir, variant_id)
    tokenizer = load_tokenizer(str(tokenizer_dir / "tokenizer.json"))
    query = "mike november oscar papa " * 1100
    assert QUERY_MAX_TOKENS < tokenizer.count(query) and OVERHEAD + tokenizer.count(query) + 20 < MAX_TOKENS
    rows = [{"query": query, "documents": ["a short document about the alphabet."]}]
    report = stage1_prompts(
        recipe, _write_pairs(tmp_path / "pairs.jsonl", rows), _reference_python(), over_length_per_shape=1
    )
    render = report["render_check"]
    assert render["passed"] is True, render["failures"][:1]
    (reported,) = render["over_cap"]["rows"]
    assert reported["index"] == 0 and [m["span"] for m in reported["mismatches"]] == ["query"]
    assert [change["mechanisms"] for change in reported["changes"]] == [["query_share"]]


def test_non_nfc_rows_compare_byte_identical_spans(tmp_path: Path, tokenizer_dir: Path, variant_id: str) -> None:
    """NFD (non-NFC) input: the render comparison byte-equals the RAW characters.

    ``decode(encode(x))`` is not the identity for this checkpoint (its normalizer maps non-NFC text to NFC,
    keeping the ids equal but not the characters). The reference cuts at raw character offsets, so on
    decomposed-accent rows -- under and over the cap -- the spans keep the raw bytes, and this goes red the
    moment either side decodes instead of cutting.
    """
    decomposed = "cafe" + chr(101) + chr(769)  # e + combining acute: NFD, never NFC
    rows = [
        {"query": f"what about {decomposed}?", "documents": [f"The {decomposed} is served over the river."]},
        {"query": f"menu of the {decomposed} house", "documents": [f"{decomposed} soup and {decomposed} pie. " * 1500]},
    ]
    recipe = _local_recipe(tmp_path, tokenizer_dir, variant_id)
    report = stage1_prompts(
        recipe, _write_pairs(tmp_path / "pairs.jsonl", rows), _reference_python(), over_length_per_shape=1
    )
    assert report["render_check"]["status"] == "run" and report["render_check"]["rows"] == 2
    assert report["render_check"]["passed"] is True, report["render_check"]["failures"][:1]
    reference = _reference_render(tmp_path, recipe, rows)
    assert reference[0] == {"index": 0, "shape": "pair", "query": rows[0]["query"], "documents": rows[0]["documents"]}
    assert rows[1]["documents"][0].startswith(reference[1]["documents"][0])  # a raw prefix, never NFC-mapped
    assert chr(769) in reference[1]["documents"][0]


def test_mutation_dropping_the_trailing_anchor_segment_reddens_the_template_check(
    tmp_path: Path, tokenizer_dir: Path, variant_id: str
) -> None:
    """Mutation: dropping the trailing anchor segment loses the assistant suffix -- the scored position --
    and the file-vs-declaration check goes red (the wire's spans, which the anchor audit reads, do not move
    on a frame change); the schema itself refuses the shape without the post-processor declaration."""
    from rcp_ndcg.data.templates import TemplateSpec

    recipe = _local_recipe(tmp_path, tokenizer_dir, variant_id)
    tokenizer = load_tokenizer(str(tokenizer_dir / "tokenizer.json"))
    template = client_template(recipe)
    golden = template.render("pair", tokenizer, query="capital of france", document="paris is the capital.")
    assert tokenizer.ids(golden, add_special_tokens=True)[-len(SUFFIX_IDS) :] == SUFFIX_IDS
    anchorless = TemplateSpec(pair=tuple(template.pair[:-1]), anchor="last", add_special_tokens=True)
    mutated = recipe.model_copy(update={"client": {**recipe.client, "template": anchorless}})
    rendered = client_template(mutated).render("pair", tokenizer, query="capital of france", document="paris.")
    assert tokenizer.ids(rendered, add_special_tokens=True)[-len(SUFFIX_IDS) :] != SUFFIX_IDS
    rows = [{"query": "capital of france", "documents": ["paris is the capital of france."]}]
    report = stage1_prompts(
        mutated, _write_pairs(tmp_path / "pairs.jsonl", rows), _reference_python(), over_length_per_shape=1
    )
    assert report["template_render_check"]["passed"] is False
    assert report["anchor_check"]["passed"] is True
    with pytest.raises(ValueError, match="anchor: last"):
        TemplateSpec(pair=tuple(template.pair[:-1]), anchor="last", add_special_tokens=False)


def test_mutation_template_file_without_its_trailing_newline_reddens_the_template_check(
    tmp_path: Path, tokenizer_dir: Path, variant_id: str
) -> None:
    """The template file without its added trailing newline (the stock vLLM example's defect) renders one
    newline short of the paper prompt at the scored position: the template check goes red."""
    recipe = _local_recipe(tmp_path, tokenizer_dir, variant_id)
    template_file = Path(recipe._dir) / "template.jinja"
    # the stock example ends with exactly two newlines
    template_file.write_text(template_file.read_text(encoding="utf-8").rstrip("\n") + "\n\n", encoding="utf-8")
    mutated = resolve_recipe(variant_id, root=tmp_path)
    rows = [{"query": "capital of france", "documents": ["paris is the capital of france."]}]
    report = stage1_prompts(
        mutated, _write_pairs(tmp_path / "pairs.jsonl", rows), _reference_python(), over_length_per_shape=1
    )
    assert report["template_render_check"]["passed"] is False
    assert report["passed"] is False


def test_the_client_settles_an_over_share_query_once_at_its_share(
    tmp_path: Path, tokenizer_dir: Path, variant_id: str
) -> None:
    """The settle rule, pinned on the wire: an over-share query ships as a verbatim prefix at its share."""
    recipe = _local_recipe(tmp_path, tokenizer_dir, variant_id)
    tokenizer = load_tokenizer(str(tokenizer_dir / "tokenizer.json"))
    long_query = "over share query filler token " * 1400
    assert tokenizer.count(long_query) > QUERY_MAX_TOKENS
    spans = served_pair(recipe, long_query, ["a short document.", "another short document."])
    assert spans["query"] != long_query and long_query.startswith(spans["query"])
    assert QUERY_MAX_TOKENS - 16 <= tokenizer.count(spans["query"]) <= QUERY_MAX_TOKENS
    assert served_pair(recipe, "capital of france", ["paris."])["query"] == "capital of france"


def test_score_mode_setup_parses_and_reaches_the_model_load(variant_id: str) -> None:
    """Score mode (stage 2) gets past its setup on CPU: with torch and transformers stubbed and an
    empty pairs list, ``score_rows`` returns without parsing anything as JSON. Failing test first:
    on the unfixed reference this died parsing the YAML recipe with json.loads, which stage 1's
    render-only suite never executes."""
    import importlib.util
    import types

    class _StubTokenizer:
        """The parts of the paper's AutoTokenizer the setup path calls before any row is scored."""

        def convert_tokens_to_ids(self, token: str) -> int:
            return {"no": 2152, "yes": 9693}[token]

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            return [1, 2]

    class _StubModel:
        def eval(self):
            return self

        def to(self, device):
            return self

    class _StubAutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return _StubTokenizer()

    class _StubAutoModel:
        loads: list[dict] = []

        @staticmethod
        def from_pretrained(*args, **kwargs):
            _StubAutoModel.loads.append(kwargs)
            return _StubModel()

    torch_stub = types.ModuleType("torch")
    torch_stub.bfloat16 = "bfloat16"
    cuda_stub = types.ModuleType("torch.cuda")

    class _OutOfMemory(RuntimeError):
        """The stubbed torch.cuda.OutOfMemoryError the score path catches."""

    class _NullContext:
        """The stubbed torch.no_grad()."""

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    cuda_stub.OutOfMemoryError = _OutOfMemory
    torch_stub.cuda = cuda_stub
    torch_stub.no_grad = _NullContext
    transformers_stub = types.ModuleType("transformers")
    transformers_stub.AutoTokenizer = _StubAutoTokenizer
    transformers_stub.AutoModelForCausalLM = _StubAutoModel
    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(sys, "dont_write_bytecode", True)  # tests write only to tmp_path, never a recipe dir
        monkey.setitem(sys.modules, "torch", torch_stub)
        monkey.setitem(sys.modules, "transformers", transformers_stub)
        spec = importlib.util.spec_from_file_location("qwen3_reranker_reference", FAMILY_DIR / "reference.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        variant = VARIANTS[variant_id]
        recipe = {
            "id": variant_id,
            "model": variant["repo"],
            "revision": variant["revision"],
            "reference": {"attn_implementation": "sdpa"},
        }
        assert module.score_rows([], recipe, f"{variant['repo']}@{variant['revision']}", "cpu") == []
        # The declared attention implementation reaches from_pretrained, never a torch.cuda.is_available()
        # choice (the stock reference environment carries no flash-attn).
        assert _StubAutoModel.loads and _StubAutoModel.loads[-1]["attn_implementation"] == "sdpa"
    finally:
        monkey.undo()
