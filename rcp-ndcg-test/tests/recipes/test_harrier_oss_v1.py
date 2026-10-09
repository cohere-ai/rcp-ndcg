"""The ``harrier-oss-v1`` family: its contract per variant, stage 1 on CPU, the card's own
sentence-transformers prompts, and the mutations (decision 34: one family module, parametrized over
its variant ids; every field pinned per variant; two mutants red per family).

Stage 1 runs on the real tokenizer files of the pinned revisions (no weights), fetched through the
shared ``_served.fetch_tokenizer`` into ``RCP_NDCG_VLLM_TOKENIZER_CACHE`` (else ``tmp_path``) and
skipped with a clear reason offline. The reference subprocess runs in ``render`` mode only (the
checkpoint's own prompt file plus the tokenizer: no torch, no weights); it reads its variant from the
resolved recipe the tests pass as ``--recipe``.

The family's over-cap policy (owner decision 9): the card's sentence-transformers path truncates ids
at the Transformer module's max_seq_length -- which sentence-transformers infers when the checkpoint
sets none as min(config.max_position_embeddings, tokenizer.model_max_length): 32768 for the 270m and
the 0.6b, 131072 for the 27b -- and the card's transformers snippet truncates ids at 32768 keeping
the anchor; the client cuts verbatim text with the anchors reserved. The reference renders the card's
own prompt, never the client's cut, so the recipe declares ``over_cap_cut_differs``: over-cap rows are
reported, not gated, and under-cap rows gate byte-exactly.

The 27b's 131072 max_position_embeddings is NOT served: the card declares "Max Tokens 32,768" for
every variant, and one forward of 131072 tokens would push the MLP activation (131072 x 21504 =
2.82e9 elements) over 2^31 -- the 32-bit element-index fault GPU-E1 established
(pplx-embed-v2-context-9b-preview died exactly there). That decision is pinned by a test.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
from rcp_ndcg_test.equivalence import stage1_prompts
from rcp_ndcg_test.equivalence.fitting import tokenizer_of
from rcp_ndcg_vllm.recipe import (
    client_config,
    default_recipes_root,
    resolve_recipe,
    serve_argv,
)

from rcp_ndcg.data.templates import Segment
from rcp_ndcg.inference.config import EmbeddingEndpoint

from ._contract import assert_recipe_contract
from ._served import client_template, fetch_tokenizer, served_texts, stage1_facts

RECIPES = default_recipes_root()
FAMILY_DIR = RECIPES / "harrier-oss-v1"

#: The family's variants (decision 34): the per-size facts the tests pin. The two Gemma backbones
#: share one byte-identical tokenizer.json (one SHA-256); the 0.6b's Qwen tokenizer is its own. The
#: appended anchor differs by backbone: the Gemma variants' post-processor appends the eos token
#: (id 1, bos id 2 leading), the Qwen tokenizer appends the endoftext (id 151643).
VARIANTS: dict[str, dict[str, Any]] = {
    "harrier-oss-v1-270m": {
        "repo": "microsoft/harrier-oss-v1-270m",
        "revision": "31de22b673913c7d658c0f03f792d77c2dcf8ebd",
        "tokenizer_sha256": "6852f8d561078cc0cebe70ca03c5bfdd0d60a45f9d2e0e1e4cc05b68e9ec329e",
        "anchor_id": 1,  # the Gemma <eos> the post-processor appends; the pooled token
        "card_document_ids": 78,
        "st_max_seq_length": 32768,  # min(config.max_position_embeddings, tokenizer.model_max_length)
        "dim": 640,  # 1_Pooling/config.json word_embedding_dimension
    },
    "harrier-oss-v1-0.6b": {
        "repo": "microsoft/harrier-oss-v1-0.6b",
        "revision": "f9b9dc8d367d443f2479d27aa5d8d2850c0774ee",
        "tokenizer_sha256": "def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a",
        "anchor_id": 151643,  # the Qwen endoftext the post-processor appends; the pooled token
        "card_document_ids": 74,
        "st_max_seq_length": 32768,
        "dim": 1024,
    },
    "harrier-oss-v1-27b": {
        "repo": "microsoft/harrier-oss-v1-27b",
        "revision": "0c0fc62f6d8af9e8604cb818c412301b103a0093",
        "tokenizer_sha256": "6852f8d561078cc0cebe70ca03c5bfdd0d60a45f9d2e0e1e4cc05b68e9ec329e",
        "anchor_id": 1,
        "card_document_ids": 78,
        "st_max_seq_length": 131072,
        "dim": 5376,
    },
}
VARIANT_IDS = list(VARIANTS)

#: The checkpoint's shared sentence-transformers files, byte-identical at all three pinned revisions
#: (the family's evidence): the ST prompts and the MTEB v2 per-task instructions.
ST_CONFIG_SHA256 = "ad2096929147368b5d0ba5322ea394d50911be4d348091c9f3b0ad06c3763d91"
MTEB_PROMPTS_SHA256 = "08aaf10dc3d61ac54af15027d3a491ea06b2ed3edcb06dc05584539f5555fe51"
MTEB_PROMPTS_TASKS = 131

#: The head pipeline vLLM's conversion assumes ("no extra layers", adapters.py:249-262): the ST module
#: chain and its per-module paths, byte-pinned across the family (modules.json sha256). A revision that
#: added a Dense/projection head or moved the pooling mode would serve a different model; this pin is
#: the red flag for it.
ST_MODULES_SHA256 = "84e40c8e006c9b1d6c122e02cba9b02458120b5fb0c87b746c41e0207cf642cf"
ST_MODULE_TYPES = [
    "sentence_transformers.models.Transformer",
    "sentence_transformers.models.Pooling",
    "sentence_transformers.models.Normalize",
]
ST_MODULE_PATHS = ["", "1_Pooling", "2_Normalize"]

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "config_sentence_transformers.json")

#: The card's own example (its printed queries and documents), and the measured id invariants at the
#: pinned revisions: the framed query renders to 27 ids per variant, ending on the appended anchor.
CARD_QUERY = "how much protein should a female eat"
CARD_DOCUMENT = (
    "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 "
    "grams per day. But, as you can see from this chart, you'll need to increase that if you're "
    "expecting or training for a marathon. Check out the chart below to see how much protein you "
    "should be eating each day."
)
CARD_QUERY_IDS = 27
MAX_TOKENS = 32768
N_PAIRS = 21
OVER_LENGTH_PER_SHAPE = 5


def _skip_unless_hub_reachable() -> None:
    """The recipe tokenizer is a Hub spec: the tests need huggingface_hub and the tokenizer files."""
    huggingface_hub = pytest.importorskip(
        "huggingface_hub",
        reason="huggingface_hub is not installed (the recipe tokenizer is a Hub spec; install rcp-ndcg[hf])",
    )
    if os.environ.get("HF_HUB_OFFLINE", "") not in ("", "0"):
        try:
            huggingface_hub.hf_hub_download(VARIANTS["harrier-oss-v1-270m"]["repo"], "tokenizer.json")
        except Exception as error:  # noqa: BLE001 - a cold cache offline skips; anything else fails later
            pytest.skip(f"HF_HUB_OFFLINE is set and the pinned tokenizer files are not cached: {error}")
        return
    try:
        socket.create_connection(("huggingface.co", 443), timeout=5).close()
    except OSError:
        pytest.skip("offline: the pinned tokenizer files must be downloaded from the Hugging Face Hub")


@pytest.fixture(params=VARIANT_IDS)
def variant_id(request: pytest.FixtureRequest) -> str:
    """One variant id of the family (every test runs per variant)."""
    return str(request.param)


@pytest.fixture
def tokenizer_dir(tmp_path: Path, variant_id: str) -> Path:
    """The pinned revision's tokenizer files in the shared download cache (one home: ``_served``).

    The two Gemma variants share one byte-identical tokenizer.json (one pinned SHA-256); each
    variant's directory also carries the checkpoint's own config_sentence_transformers.json, so the
    reference's render mode reads the prompt from the checkpoint files it was handed.
    """
    variant = VARIANTS[variant_id]
    path: Path | None = None
    for name in TOKENIZER_FILES:
        # Only tokenizer.json carries a pinned SHA-256 (one per backbone: the two Gemma variants share
        # theirs byte-identically); the sidecars are pinned by revision, not by hash.
        path = fetch_tokenizer(
            f"https://huggingface.co/{variant['repo']}/resolve/{variant['revision']}/{name}",
            f"harrier-oss-v1/{variant_id}/{name}",
            tmp_path,
            sha256=variant["tokenizer_sha256"] if name == "tokenizer.json" else None,
        )
    assert path is not None
    return path.parent


def _expected_contract(variant_id: str) -> dict[str, Any]:
    """The variant's full resolved contract: every field of every block, exactly as the product models
    resolve it (authored values and schema defaults alike). Nothing may ride unpinned."""
    variant = VARIANTS[variant_id]
    return {
        "serve": {
            "runner": "pooling",
            "convert": None,
            "hf_overrides": {},
            "chat_template": None,
            "pooler_config": {},
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
            "api": "openai_embeddings",
            "request_shape": "text",
            "tokenizer": f"{variant['repo']}@{variant['revision']}",
            "max_tokens": 32768,
            "template": {
                "query": [
                    {
                        "fixed": "Instruct: Given a web search query, retrieve relevant passages that "
                        "answer the query\nQuery: "
                    },
                    {"content": "query"},
                ],
                "document": [{"content": "document"}],
                "anchor": "last",
                "add_special_tokens": True,
            },
            "on_overflow": "cut",
            "empty_doc": "send",
            "normalize": True,
            "model": variant_id,
            "revision": variant["revision"],
        },
        "reference": {
            "entry": "reference.py",
            "kind": "sentence_transformers",
            "known_deviations": ["over_cap_cut_differs"],
            "device": None,  # the schema default
            "score_scale": "cosine",
        },
    }


def _expected_top(variant_id: str) -> dict[str, Any]:
    variant = VARIANTS[variant_id]
    return {
        "id": variant_id,
        "model": variant["repo"],
        "revision": variant["revision"],
        "role": "embed",
        "input": ["text"],
        "licence": "mit",
    }


def _contract(recipe: Any, variant_id: str) -> None:
    expected = _expected_contract(variant_id)
    assert_recipe_contract(
        recipe,
        serve=expected["serve"],
        client=expected["client"],
        reference=expected["reference"],
        top=_expected_top(variant_id),
    )


def _reference_python() -> str:
    """The interpreter the reference subprocess runs in (a reference env may be named)."""
    return os.environ.get("RCP_NDCG_REFERENCE_PYTHON", sys.executable)


def _local_recipe(tmp_path: Path, tokenizer_dir: Path, variant_id: str) -> Any:
    """The family copied into ``tmp_path`` with its shared tokenizer spec pointed at the downloaded
    files (same bytes, same SHA-256 identity), the variant resolved through the same loader."""
    target = tmp_path / FAMILY_DIR.name
    shutil.copytree(FAMILY_DIR, target)
    data = yaml.safe_load((target / "family.yaml").read_text(encoding="utf-8"))
    data["client"]["tokenizer"] = str(tokenizer_dir)
    (target / "family.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return resolve_recipe(variant_id, root=tmp_path)


def _pairs() -> list[dict[str, Any]]:
    """The stage-1 pairs: the card's own example first, then varied short rows, then five over-budget
    ones; no ``shape`` key, so the render check compares every row."""
    rows: list[dict[str, Any]] = [
        {"query": CARD_QUERY, "documents": [CARD_DOCUMENT]},
        {"query": "summit define", "documents": ["Definition of summit for English Language Learners."]},
        {"query": "capital of france", "documents": ["Paris is the capital and largest city of France."]},
    ]
    topics = [
        ("what is gravity", "Gravity is a fundamental interaction that attracts masses."),
        ("explain photosynthesis", "Photosynthesis converts light energy into chemical energy in plants."),
        ("who wrote hamlet", "Hamlet was written by William Shakespeare around 1600."),
        ("largest planet", "Jupiter is the largest planet in the solar system."),
        ("currency of japan", "The yen is the official currency of Japan."),
        ("what is an embedding", "An embedding maps text into a vector space where distance means similarity."),
        ("last token pooling", "Last-token pooling reads the hidden state of the final position."),
        ("semantic search", "Semantic search retrieves documents by meaning rather than keywords."),
        ("longest river", "The Nile and the Amazon compete for the title of longest river."),
        ("vllm serving", "vLLM serves models with paged attention and continuous batching."),
        ("sentence transformers", "sentence-transformers encodes texts with a pooling layer and a normaliser."),
        ("coldest place on earth", "Antarctica records the lowest temperatures measured on Earth."),
        ("how do transformers work", "Transformers process sequences with self-attention layers."),
        ("matryoshka embeddings", "Matryoshka embeddings stay usable when truncated to fewer dimensions."),
        ("definition of retrieval", "Retrieval is the task of finding relevant documents for a query."),
        ("speed of light", "The speed of light in vacuum is exactly 299792458 metres per second."),
        ("boiling point of water", "Water boils at 100 degrees Celsius at standard atmospheric pressure."),
        ("python list comprehension", "A list comprehension builds a list from an iterable in one expression."),
    ]
    for index, (query, document) in enumerate(topics):
        rows.append({"query": query, "documents": [document, f"Unrelated passage number {index} about oats."]})
    # Over the 32768-token budget: english filler, CJK filler, mixed punctuation, a long query with a
    # long document, and a single-word run -- the anchor must survive every cut.
    rows.append({"query": "q" * 5, "documents": ["word filler sentence number one two three four five six. " * 6000]})
    rows.append({"query": "short", "documents": ["汉字填充句子，用于测试分词边界与截断行为。" * 3000]})
    rows.append(
        {
            "query": "a query with punctuation! and numbers 12345 plus UTF-8 emoji balloon",
            "documents": ["Mixed 汉字 and latin words, with punctuation; colons: and quotes - plus balloons. " * 2000],
        }
    )
    rows.append(
        {"query": "longer query " * 900, "documents": ["filler text for the document side of the pair. " * 4000]}
    )
    rows.append({"query": "q", "documents": ["x " * 40000]})
    return rows


def _write_pairs(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# The contract (offline-capable facts; the conftest still gates the file as network).
# ---------------------------------------------------------------------------


def test_recipe_contract_pins_every_field(variant_id: str) -> None:
    """The recipe loads against the product's EmbeddingEndpoint and declares exactly its contract."""
    recipe = resolve_recipe(variant_id)
    _contract(recipe, variant_id)
    assert recipe.engine.image == "vllm/vllm-openai:v0.31.0" and recipe.engine.min_version == "0.31.0"
    max_tokens = recipe.client.get("max_tokens")
    assert isinstance(max_tokens, int)
    assert recipe.serve.max_model_len >= max_tokens  # the engine must not 400 the budget
    assert (FAMILY_DIR / "reference.py").is_file()
    assert (FAMILY_DIR / "requirements-reference.txt").is_file()
    assert recipe.status.state == "unverified"
    assert recipe.sources
    assert recipe.serve.runner == "pooling" and recipe.serve.convert is None
    assert recipe.reference.kind == "sentence_transformers" and recipe.reference.score_scale == "cosine"


def test_the_contract_reds_on_two_mutants(variant_id: str) -> None:
    """Two mutants of the declared contract red the pin: a served field and the declared deviation."""
    recipe = resolve_recipe(variant_id)
    _contract(recipe, variant_id)  # the pinned recipe itself is green
    serve_mutant = recipe.model_copy(update={"serve": recipe.serve.model_copy(update={"max_model_len": 131072})})
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
    assert config["recipe"] == recipe.id
    endpoint = EmbeddingEndpoint(**config)
    assert str(endpoint.base_url) == "http://127.0.0.1:8100/v1"
    assert endpoint.template is not None and endpoint.template.anchor == "last"


def test_serve_argv_renders_the_engine_command(variant_id: str) -> None:
    """The argv a wave runs: pooling runner, no template file, no overrides, the card's budget."""
    variant = VARIANTS[variant_id]
    recipe = resolve_recipe(variant_id)
    argv = serve_argv(recipe, port=8100, served_model_name=recipe.id)
    assert argv[:3] == ["vllm", "serve", variant["repo"]]
    assert argv[argv.index("--revision") + 1] == variant["revision"]
    assert argv[argv.index("--served-model-name") + 1] == recipe.id
    assert argv[argv.index("--runner") + 1] == "pooling"
    assert argv[argv.index("--dtype") + 1] == "bfloat16"
    assert argv[argv.index("--max-model-len") + 1] == "32768"
    assert json.loads(argv[argv.index("--hf-overrides") + 1]) == {}
    assert json.loads(argv[argv.index("--pooler-config") + 1]) == {}
    assert "--chat-template" not in argv and "--trust-remote-code" not in argv and "--convert" not in argv


def test_the_27b_does_not_serve_its_full_context() -> None:
    """The 131072 decision, pinned: the card's own Max Tokens (32,768) is served, and the notes carry
    the 32-bit arithmetic (131072 x 21504 > 2^31, the GPU-E1 element-index fault class)."""
    recipe = resolve_recipe("harrier-oss-v1-27b")
    assert recipe.serve.max_model_len == 32768 < 131072
    assert "2^31" in recipe.notes and "131072 x 21504" in recipe.notes
    assert "Max Tokens 32,768" in recipe.notes
    # and the decision is the family's own card fact, not a per-variant override
    for variant in VARIANT_IDS:
        assert resolve_recipe(variant).serve.max_model_len == 32768


# ---------------------------------------------------------------------------
# The checkpoint's own prompts (the card's usage), at the pinned revisions.
# ---------------------------------------------------------------------------


@pytest.mark.network
def test_the_checkpoint_prompt_files_are_identical_across_the_variants(tmp_path: Path) -> None:
    """The family's evidence, pinned: the checkpoints' sentence-transformers prompt config and the
    MTEB v2 per-task instructions are byte-identical at all three pinned revisions."""
    _skip_unless_hub_reachable()
    from huggingface_hub import hf_hub_download

    seen: dict[str, list[str]] = {"config_sentence_transformers.json": [], "mteb_v2_eval_prompts.json": []}
    for variant in VARIANTS.values():
        for name in seen:
            path = hf_hub_download(variant["repo"], name, revision=variant["revision"])
            data = Path(path).read_bytes()
            seen[name].append(hashlib.sha256(data).hexdigest())
    assert seen["config_sentence_transformers.json"] == [ST_CONFIG_SHA256] * 3
    assert seen["mteb_v2_eval_prompts.json"] == [MTEB_PROMPTS_SHA256] * 3
    prompts = json.loads(
        Path(
            hf_hub_download(
                VARIANTS["harrier-oss-v1-0.6b"]["repo"],
                "mteb_v2_eval_prompts.json",
                revision=VARIANTS["harrier-oss-v1-0.6b"]["revision"],
            )
        ).read_text(encoding="utf-8")
    )
    assert len(prompts) == MTEB_PROMPTS_TASKS and all(isinstance(value, str) for value in prompts.values())


def test_the_query_frame_is_the_checkpoint_sentence_transformers_prompt(
    tmp_path: Path, tokenizer_dir: Path, variant_id: str
) -> None:
    """The template's frame is byte-identical to the checkpoint's own prompts.web_search_query, with
    the trailing space that prompt carries; documents are bare; the appended post-processor token is
    the pooled anchor."""
    _skip_unless_hub_reachable()
    recipe = _local_recipe(tmp_path, tokenizer_dir, variant_id)
    template = client_template(recipe)
    assert template is not None
    prompt = json.loads((tokenizer_dir / "config_sentence_transformers.json").read_text(encoding="utf-8"))["prompts"][
        "web_search_query"
    ]
    query_segments = template.segments("query")
    assert all(isinstance(segment, Segment) for segment in query_segments)
    assert query_segments[0].fixed == prompt
    assert prompt.endswith("Query: ")  # the trailing space this checkpoint's prompt carries
    assert [segment.content for segment in query_segments if segment.content is not None] == ["query"]
    assert query_segments[-1].content == "query"  # the shape ends on its content (the anchor is the appended token)
    document_segments = template.segments("document")
    assert [segment.content for segment in document_segments if segment.content is not None] == ["document"]
    assert all(segment.fixed is None for segment in document_segments)  # bare documents
    assert template.adds_special_tokens("query") is True
    assert template.adds_special_tokens("document") is True


# ---------------------------------------------------------------------------
# Stage 1 on CPU: the reference render, the anchors, the over-cap table.
# ---------------------------------------------------------------------------


def test_stage1_on_cpu_passes_with_the_reference_render(tmp_path: Path, tokenizer_dir: Path, variant_id: str) -> None:
    """Stage 1 on CPU: 21 in-budget + 5 over-budget rows, byte-exact renders and the anchor audit.

    The reference subprocess renders the card's prompts (the checkpoint's prompt file plus the pinned
    tokenizer; no weights). The shipped prompts equal them byte for byte on every under-cap row;
    over-cap rows whose cuts differ are reported in the non-gating table under the over_cap_cut_differs
    declaration. Every anchor survives the over-length inputs (this file's 5 per shape plus stage 1's
    own samples).
    """
    _skip_unless_hub_reachable()
    recipe = _local_recipe(tmp_path, tokenizer_dir, variant_id)
    rows = _pairs()
    tokenizer = tokenizer_of(recipe)
    pairs_path = _write_pairs(tmp_path / "pairs.jsonl", rows)
    document = stage1_prompts(recipe, pairs_path, _reference_python(), over_length_per_shape=OVER_LENGTH_PER_SHAPE)
    assert document["passed"] is True, json.dumps(document)[:2000]
    render = document["render_check"]
    assert render["status"] == "run" and render["passed"] is True, render["failures"][:1]
    assert document["anchor_check"]["passed"] is True, document["anchor_check"]["failures"][:1]
    assert document["engine_tokenize_check"]["status"] == "not_run"  # no engine on CPU: never "passed"
    # Both declared shapes were sampled with over-length inputs, and every one overflowed (the role
    # client's own census, read through its capture). The census runs on the in-budget rows only: the
    # pairs file's own over-budget rows are the over-cap test's business, and a row probed under both
    # shapes counts its cuts under each.
    facts = stage1_facts(recipe, rows[:N_PAIRS], tokenizer, OVER_LENGTH_PER_SHAPE)
    assert facts["per_shape"]["query"]["cuts"] == OVER_LENGTH_PER_SHAPE
    assert facts["per_shape"]["document"]["cuts"] == OVER_LENGTH_PER_SHAPE


def test_the_card_example_renders_to_the_measured_ids(tmp_path: Path, tokenizer_dir: Path, variant_id: str) -> None:
    """Token equality against the reference subprocess, with the measured invariants: the card's own
    example query renders to 27 framed ids per variant, ending on the appended anchor."""
    _skip_unless_hub_reachable()
    variant = VARIANTS[variant_id]
    recipe = _local_recipe(tmp_path, tokenizer_dir, variant_id)
    tokenizer = tokenizer_of(recipe)
    rows = _pairs()[:1]
    pairs = _write_pairs(tmp_path / "pairs.jsonl", rows)
    out = tmp_path / "reference.json"
    recipe_file = tmp_path / "reference-render.recipe.json"
    recipe_file.write_text(json.dumps(recipe.model_dump(mode="json"), sort_keys=True), encoding="utf-8")
    completed = subprocess.run(
        [
            _reference_python(),
            str(FAMILY_DIR / "reference.py"),
            "--mode",
            "render",
            "--pairs",
            str(pairs),
            "--out",
            str(out),
            "--tokenizer",
            str(recipe.client.get("tokenizer")),
            "--recipe",
            str(recipe_file),
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stderr[-800:]
    rows_by_key = {(row["index"], row["shape"]): row for row in json.loads(out.read_text(encoding="utf-8"))["rows"]}
    template = client_template(recipe)
    assert template is not None
    frame = template.segments("query")[0].fixed
    query_row, document_row = rows_by_key[(0, "query")], rows_by_key[(0, "document")]
    assert query_row["text"] == frame + CARD_QUERY
    assert document_row["text"] == CARD_DOCUMENT
    # The served fit renders the same strings (what the engine receives), and the ids carry the
    # measured invariants: 27 framed-query ids, ending on the post-processor's appended anchor.
    served_query = served_texts(recipe, [CARD_QUERY], "query")[0]
    served_document = served_texts(recipe, [CARD_DOCUMENT], "document")[0]
    assert served_query == query_row["text"]
    assert served_document == document_row["text"]
    query_ids = tokenizer.ids(served_query, add_special_tokens=True)
    document_ids = tokenizer.ids(served_document, add_special_tokens=True)
    assert len(query_ids) == CARD_QUERY_IDS and query_ids[-1] == variant["anchor_id"]
    assert len(document_ids) == variant["card_document_ids"] and document_ids[-1] == variant["anchor_id"]


def test_an_over_cap_pairs_row_rides_the_declared_table_with_the_cards_own_prompt(
    tmp_path: Path, tokenizer_dir: Path, variant_id: str
) -> None:
    """Decision 9 on this reference: it renders the card's own prompt (the transformers snippet cuts
    ids at encode; the sentence-transformers path truncates at its inferred max_seq_length, 32768 for
    the 270m and the 0.6b and 131072 for the 27b, with the post-processor's anchor appended after the
    truncation) and never ports the client's cut. An over-budget pairs row therefore differs from the
    client's content-only cut by declaration (``over_cap_cut_differs``): stage 1 reports it in the
    non-gating table and still passes on every under-cap row."""
    _skip_unless_hub_reachable()
    recipe = _local_recipe(tmp_path, tokenizer_dir, variant_id)
    rows = _pairs()
    long_document = "long document about retrieval " * 9000
    rows.append({"query": CARD_QUERY, "documents": [long_document]})
    document = stage1_prompts(
        recipe, _write_pairs(tmp_path / "pairs.jsonl", rows), _reference_python(), over_length_per_shape=1
    )
    render = document["render_check"]
    assert render["passed"] is True, render["failures"][:1]
    over_cap = render["over_cap"]
    assert over_cap["known_deviation"] is True and over_cap["gating"] is False
    # The appended row and the pairs file's own five over-budget rows (indexes N_PAIRS..): all documents.
    assert {(entry["index"], entry["shape"]) for entry in over_cap["rows"]} == {
        (index, "document") for index in range(N_PAIRS, len(rows))
    }
    recipe_file = tmp_path / "over-cap.recipe.json"
    recipe_file.write_text(json.dumps(recipe.model_dump(mode="json"), sort_keys=True), encoding="utf-8")
    completed = subprocess.run(
        [
            _reference_python(),
            str(FAMILY_DIR / "reference.py"),
            "--mode",
            "render",
            "--pairs",
            str(_write_pairs(tmp_path / "over-cap-pairs.jsonl", rows)),
            "--out",
            str(tmp_path / "over-cap-reference.json"),
            "--tokenizer",
            str(recipe.client.get("tokenizer")),
            "--recipe",
            str(recipe_file),
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stderr[-800:]
    texts = {
        (row["index"], row["shape"]): row["text"]
        for row in json.loads((tmp_path / "over-cap-reference.json").read_text(encoding="utf-8"))["rows"]
    }
    assert texts[(len(rows) - 1, "document")] == long_document  # the card's own prompt, uncut
    assert document["passed"] is True


def test_the_st_truncation_cap_is_the_inferred_max_seq_length(
    tmp_path: Path, tokenizer_dir: Path, variant_id: str
) -> None:
    """The card's sentence-transformers path truncates at the Transformer module's max_seq_length,
    which sentence-transformers infers when the checkpoint sets none as
    ``min(config.max_position_embeddings, tokenizer.model_max_length)`` (``models/Transformer.py`` at
    >=3.0): 32768 / 32768 / 131072 for the three variants. The post-processor appends the pooled
    anchor after truncation, so an over-cap text cut at the cap keeps it (measured)."""
    _skip_unless_hub_reachable()
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    variant = VARIANTS[variant_id]
    config = json.loads(
        Path(hf_hub_download(variant["repo"], "config.json", revision=variant["revision"])).read_text(encoding="utf-8")
    )
    tokenizer_config = json.loads((tokenizer_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
    inferred = min(config["max_position_embeddings"], tokenizer_config["model_max_length"])
    assert inferred == variant["st_max_seq_length"]
    tokenizer = Tokenizer.from_file(str(tokenizer_dir / "tokenizer.json"))
    tokenizer.enable_truncation(max_length=inferred)
    filler = "word filler sentence about retrieval and rankings. " * (inferred // 4)
    encoded = tokenizer.encode(filler)
    assert len(encoded.ids) == inferred and encoded.ids[-1] == variant["anchor_id"]


def test_the_head_pipeline_is_pinned_at_every_revision(tmp_path: Path, tokenizer_dir: Path, variant_id: str) -> None:
    """The served path's load-bearing head facts, pinned at each revision: ``modules.json`` is exactly
    Transformer -> Pooling -> Normalize (one SHA-256 across the family) and ``1_Pooling/config.json``
    is lasttoken pooling with the variant's dim. vLLM's conversion assumes no extra layers
    (``adapters.py:249-262``), so a revision that added a Dense/projection head or changed the pooling
    mode would serve a different model with nothing else red; this test is that red."""
    _skip_unless_hub_reachable()
    from huggingface_hub import hf_hub_download

    variant = VARIANTS[variant_id]
    modules_raw = Path(hf_hub_download(variant["repo"], "modules.json", revision=variant["revision"])).read_bytes()
    assert hashlib.sha256(modules_raw).hexdigest() == ST_MODULES_SHA256
    modules = json.loads(modules_raw)
    assert [module["type"] for module in modules] == ST_MODULE_TYPES  # no Dense head, no extra layer
    assert [module["path"] for module in modules] == ST_MODULE_PATHS
    pooling = json.loads(
        Path(hf_hub_download(variant["repo"], "1_Pooling/config.json", revision=variant["revision"])).read_text(
            encoding="utf-8"
        )
    )
    assert pooling["pooling_mode_lasttoken"] is True
    assert all(
        pooling[mode] is False
        for mode in (
            "pooling_mode_cls_token",
            "pooling_mode_mean_tokens",
            "pooling_mode_max_tokens",
            "pooling_mode_mean_sqrt_len_tokens",
            "pooling_mode_weightedmean_tokens",
        )
    )
    assert pooling["word_embedding_dimension"] == variant["dim"]


@pytest.mark.network
def test_the_cards_declare_no_matryoshka_mechanism(variant_id: str) -> None:
    """The client declares no ``dimensions`` because the cards offer none: the pinned card text has no
    Matryoshka/MRL truncation mechanism, so a sub-dimension serving would be a new recipe with its own
    gates (the same rule as the qwen3-embedding family)."""
    _skip_unless_hub_reachable()
    from huggingface_hub import hf_hub_download

    variant = VARIANTS[variant_id]
    card = Path(hf_hub_download(variant["repo"], "README.md", revision=variant["revision"])).read_text(encoding="utf-8")
    lowered = card.lower()
    assert "matryoshka" not in lowered and "mrl" not in lowered, (
        "the card gained a Matryoshka/MRL mechanism: the client's omitted dimensions decision needs revisiting"
    )


def test_mutation_dropping_the_anchor_declaration_is_refused(
    tmp_path: Path, tokenizer_dir: Path, variant_id: str
) -> None:
    """The schema mutation: without add_special_tokens the content-final shapes have no anchor -- the
    product's TemplateSpec refuses the declaration at load (anchor: last must end on a fixed segment
    or declare the appended post-processor token), so the mutated recipe never serves."""
    _skip_unless_hub_reachable()
    copied = tmp_path / FAMILY_DIR.name
    shutil.copytree(FAMILY_DIR, copied)
    data = yaml.safe_load((copied / "family.yaml").read_text(encoding="utf-8"))
    data["client"]["tokenizer"] = str(tokenizer_dir)
    data["client"]["template"]["add_special_tokens"] = False
    (copied / "family.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    mutated = resolve_recipe(variant_id, root=tmp_path)
    with pytest.raises(ValueError, match="anchor: last"):
        EmbeddingEndpoint(**client_config(mutated, base_url=None))


def test_mutation_dropping_the_query_frame_reddens_the_render_check(
    tmp_path: Path, tokenizer_dir: Path, variant_id: str
) -> None:
    """The behavioural mutation: dropping the query frame's fixed segment (the checkpoint's own
    web_search_query prompt) changes what the client ships -- the served query loses the instruction
    the model was trained to read -- and the render comparison reds for the query shape while the
    (bare) document shape still gates."""
    _skip_unless_hub_reachable()
    copied = tmp_path / FAMILY_DIR.name
    shutil.copytree(FAMILY_DIR, copied)
    data = yaml.safe_load((copied / "family.yaml").read_text(encoding="utf-8"))
    data["client"]["tokenizer"] = str(tokenizer_dir)
    data["client"]["template"]["query"] = [{"content": "query"}]
    (copied / "family.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    mutated = resolve_recipe(variant_id, root=tmp_path)
    rows = _pairs()[:2]
    document = stage1_prompts(
        mutated, _write_pairs(tmp_path / "pairs.jsonl", rows), _reference_python(), over_length_per_shape=1
    )
    assert document["passed"] is False
    assert document["anchor_check"]["passed"] is True  # the anchor declaration did not move
    failures = document["render_check"]["failures"]
    assert failures and {failure["shape"] for failure in failures} == {"query"}


def test_mutation_max_tokens_drift_moves_the_wire_budget(tmp_path: Path, tokenizer_dir: Path, variant_id: str) -> None:
    """The second behavioural mutant: a drifted client.max_tokens changes what the client ships (a
    shorter budget cuts under-budget rows the pinned recipe sends whole) -- the served texts move with
    the declared budget, so the contract pin above is what holds the budget in place."""
    _skip_unless_hub_reachable()
    target = tmp_path / FAMILY_DIR.name
    shutil.copytree(FAMILY_DIR, target)
    data = yaml.safe_load((target / "family.yaml").read_text(encoding="utf-8"))
    data["client"]["tokenizer"] = str(tokenizer_dir)
    data["client"]["max_tokens"] = 512
    (target / "family.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    mutated = resolve_recipe(variant_id, root=tmp_path)
    long_document = "document filler about retrieval systems and rankings. " * 400
    served = served_texts(mutated, [CARD_QUERY], "query")[0]
    served_document = served_texts(mutated, [long_document], "document")[0]
    tokenizer = tokenizer_of(mutated)
    assert tokenizer.count(served) <= 512
    assert served_document != long_document  # the mutant cuts what the pinned recipe sends whole
    pinned = resolve_recipe(variant_id)
    assert served_texts(pinned, [long_document], "document")[0] == long_document


def test_reference_refuses_a_foreign_tokenizer_spec(tmp_path: Path, tokenizer_dir: Path, variant_id: str) -> None:
    """The reference pins its variant through --recipe: a --tokenizer spec naming another checkpoint
    (or another revision) is refused, never defaulted."""
    _skip_unless_hub_reachable()
    recipe = _local_recipe(tmp_path, tokenizer_dir, variant_id)
    recipe_file = tmp_path / "ref.recipe.json"
    recipe_file.write_text(json.dumps(recipe.model_dump(mode="json"), sort_keys=True), encoding="utf-8")
    pairs = _write_pairs(tmp_path / "pairs.jsonl", _pairs()[:1])
    foreign = "Qwen/Qwen3-Embedding-0.6B@97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
    completed = subprocess.run(
        [
            _reference_python(),
            str(FAMILY_DIR / "reference.py"),
            "--mode",
            "render",
            "--pairs",
            str(pairs),
            "--out",
            str(tmp_path / "out.json"),
            "--tokenizer",
            foreign,
            "--recipe",
            str(recipe_file),
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert completed.returncode != 0
    assert "does not pin" in completed.stderr
