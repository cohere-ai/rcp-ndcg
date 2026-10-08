"""The ``qwen3-embedding-0.6b`` recipe: the schema validates, and stage 1 passes on CPU.

Stage 1 runs the harness's own ``stage1_prompts`` with the recipe directory's ``reference.py`` as the
subprocess (its render mode needs only ``huggingface_hub`` -- no torch, no weights). The pinned
tokenizer files are downloaded into the shared tokenizer cache (``RCP_NDCG_VLLM_TOKENIZER_CACHE`` when
set, ``tmp_path`` otherwise; ``_served.tokenizer_cache``) and the tests skip with a clear reason
when the Hub is unreachable or ``huggingface_hub`` is absent (CI's slim venv).

The measured invariants (at the pinned revision): the card's example
query renders to 27 token ids and the example document to 8, each ending on the post-processor's
endoftext anchor (id 151643) that last-token pooling reads.
"""

from __future__ import annotations

import json
import os
import socket
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
from rcp_ndcg_test.equivalence import stage1_prompts
from rcp_ndcg_test.equivalence.fitting import tokenizer_of
from rcp_ndcg_test.equivalence.reference import run_reference
from rcp_ndcg_vllm import load_recipe, serve_argv
from rcp_ndcg_vllm.recipe import default_recipes_root

from rcp_ndcg.data.templates import Segment

from ._contract import assert_recipe_contract
from ._served import client_template, served_texts, stage1_facts, tokenizer_cache

REPO = "Qwen/Qwen3-Embedding-0.6B"
REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"  # re-checked against the Hub API; not gated
END_OF_TEXT_NAME = "endoftext"  # the appended anchor the last-token pooler reads; never typed out
END_OF_TEXT_ID = 151643
RECIPE_DIR = default_recipes_root() / "qwen3-embedding-0.6b"
CARD_QUERY = "What is the capital of China?"
CARD_DOCUMENT = "The capital of China is Beijing."
#: The measured invariant on the card's example (token equality at the pinned revision).
CARD_QUERY_IDS = 27
CARD_DOCUMENT_IDS = 8
N_PAIRS = 22
OVER_LENGTH_PER_SHAPE = 5


def _skip_unless_hub_reachable() -> None:
    """The recipe tokenizer is a Hub spec: the tests need huggingface_hub and the tokenizer files.

    Skips carry their reason in the message; anything else (a revision gone, a corrupt cache) fails.
    With ``HF_HUB_OFFLINE`` set, a warm cache still runs (the Hub client serves it locally); a cold
    one skips instead of erroring.
    """
    huggingface_hub = pytest.importorskip(
        "huggingface_hub",
        reason="huggingface_hub is not installed (the recipe tokenizer is a Hub spec; install rcp-ndcg[hf])",
    )
    if os.environ.get("HF_HUB_OFFLINE", "") not in ("", "0"):
        try:
            huggingface_hub.hf_hub_download(REPO, "tokenizer.json", revision=REVISION)
        except (huggingface_hub.errors.OfflineModeIsEnabled, huggingface_hub.errors.LocalEntryNotFoundError) as error:
            pytest.skip(f"HF_HUB_OFFLINE is set and the pinned tokenizer files are not cached: {error}")
        return
    try:
        socket.create_connection(("huggingface.co", 443), timeout=5).close()
    except OSError:
        pytest.skip("offline: the pinned tokenizer files must be downloaded from the Hugging Face Hub")


@pytest.fixture
def hub_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The Hub cache for the tokenizer downloads: the shared tokenizer cache (one home)."""
    cache = tokenizer_cache(tmp_path / "hf-cache")
    monkeypatch.setenv("HF_HOME", str(cache))
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    return cache


def pairs_rows() -> list[dict[str, Any]]:
    """Sampled pairs for stage 1: the card's own example first, then varied short rows."""
    rows: list[dict[str, Any]] = [
        {
            "query": CARD_QUERY,
            "documents": [
                CARD_DOCUMENT,
                "Gravity is a force that attracts two bodies towards each other. It gives weight to "
                "physical objects and is responsible for the movement of planets around the sun.",
            ],
        }
    ]
    topics = [
        ("capital of france", "Paris is the capital and largest city of France."),
        ("what is gravity", "Gravity is a fundamental interaction that attracts masses."),
        ("explain photosynthesis", "Photosynthesis converts light energy into chemical energy in plants."),
        ("who wrote hamlet", "Hamlet was written by William Shakespeare around 1600."),
        ("boiling point of water", "Water boils at 100 degrees Celsius at standard atmospheric pressure."),
        ("largest planet", "Jupiter is the largest planet in the solar system."),
        ("speed of light", "The speed of light in vacuum is exactly 299792458 metres per second."),
        ("currency of japan", "The yen is the official currency of Japan."),
        ("definition of retrieval", "Retrieval is the task of finding relevant documents for a query."),
        ("what is ndcg", "Normalized discounted cumulative gain measures ranking quality."),
        ("python list comprehension", "A list comprehension builds a list from an iterable in one expression."),
        ("how do transformers work", "Transformers process sequences with self-attention layers."),
        ("coldest place on earth", "Antarctica records the lowest temperatures measured on Earth."),
        ("what is an embedding", "An embedding maps text into a vector space where distance means similarity."),
        ("last token pooling", "Last-token pooling reads the hidden state of the final position."),
        ("matryoshka embeddings", "Matryoshka embeddings stay usable when truncated to fewer dimensions."),
        ("semantic search", "Semantic search retrieves documents by meaning rather than keywords."),
        ("density of gold", "Gold has a density of 19.3 grams per cubic centimetre."),
        ("longest river", "The Nile and the Amazon compete for the title of longest river."),
        ("vllm serving", "vLLM serves models with paged attention and continuous batching."),
        ("sentence transformers", "sentence-transformers encodes texts with a pooling layer and a normaliser."),
    ]
    for index, (query, document) in enumerate(topics):
        rows.append({"query": query, "documents": [document, f"Unrelated passage number {index} about oats."]})
    return rows


def write_pairs(path: Path, rows: list[dict[str, Any]]) -> Path:
    """One pairs JSONL file (the harness's load_pairs format)."""
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_the_recipe_loads_and_declares_the_served_path() -> None:
    """The schema validates and the rendered argv is the served path."""
    recipe = load_recipe(RECIPE_DIR)
    _assert_contract(recipe)  # every serve, client and reference field pinned, exactly
    assert recipe.id == "qwen3-embedding-0.6b"
    assert recipe.model == REPO and recipe.revision == REVISION
    assert recipe.role == "embed" and recipe.input == ["text"] and recipe.licence == "apache-2.0"
    assert recipe.client.get("api") == "openai_embeddings"
    assert recipe.client.get("tokenizer") == f"{REPO}@{REVISION}"
    assert recipe.client.get("max_tokens") == 8192 and recipe.serve.max_model_len == 32768
    assert recipe.client.get("on_overflow") == "cut"
    assert "query_prompt" not in recipe.client and "doc_prompt" not in recipe.client  # the frame is the template
    template = client_template(recipe)
    assert template is not None
    assert template.shapes() == ("query", "document")
    assert template.anchor == "last"
    assert recipe.serve.runner == "pooling" and recipe.serve.convert is None
    assert recipe.serve.dtype == "bfloat16" and recipe.serve.max_model_len == 32768
    assert recipe.serve.chat_template is None and recipe.serve.trust_remote_code is False
    assert recipe.serve.pooler_config == {} and recipe.serve.plugin is None
    assert recipe.reference.kind == "transformers" and recipe.reference.score_scale == "cosine"
    assert recipe.reference.known_deviations == ["over_cap_cut_differs"] and recipe.status.state == "unverified"
    argv = serve_argv(recipe, port=8100, served_model_name=recipe.id)
    assert argv[:3] == ["vllm", "serve", REPO]
    assert argv[argv.index("--revision") + 1] == REVISION
    assert argv[argv.index("--runner") + 1] == "pooling"
    assert argv[argv.index("--dtype") + 1] == "bfloat16"
    assert argv[argv.index("--max-model-len") + 1] == "32768"
    assert argv[argv.index("--pooler-config") + 1] == "{}"
    assert "--chat-template" not in argv and "--trust-remote-code" not in argv


def test_the_query_frame_is_the_checkpoint_sentence_transformers_prompt(hub_cache: Path) -> None:
    """The template's frame is byte-identical to the checkpoint's own prompts.query."""
    _skip_unless_hub_reachable()
    from huggingface_hub import hf_hub_download

    recipe = load_recipe(RECIPE_DIR)
    template = client_template(recipe)
    assert template is not None
    st_path = hf_hub_download(REPO, "config_sentence_transformers.json", revision=REVISION)
    prompts = json.loads(Path(st_path).read_text(encoding="utf-8"))["prompts"]
    query_segments = template.segments("query")
    assert all(isinstance(segment, Segment) for segment in query_segments)
    assert query_segments[0].fixed == prompts["query"]
    assert [segment.content for segment in query_segments if segment.content is not None] == ["query"]
    assert query_segments[-1].content == "query"  # the shape ends on its content (the anchor is the appended token)
    document_segments = template.segments("document")
    assert [segment.content for segment in document_segments if segment.content is not None] == ["document"]
    assert all(segment.fixed is None for segment in document_segments)  # bare documents
    assert template.adds_special_tokens("query") is True
    assert template.adds_special_tokens("document") is True


def test_stage1_token_ids_and_anchors_pass_on_cpu(tmp_path: Path, hub_cache: Path) -> None:
    """Stage 1 on CPU: the reference render agrees byte-exactly and every anchor survives every cut."""
    _skip_unless_hub_reachable()
    recipe = load_recipe(RECIPE_DIR)
    pairs = write_pairs(tmp_path / "pairs.jsonl", pairs_rows())
    document = stage1_prompts(recipe, str(pairs), sys.executable, over_length_per_shape=OVER_LENGTH_PER_SHAPE)
    assert document["passed"] is True, (document["anchor_check"], document["render_check"])
    assert document["sampled"] >= N_PAIRS
    # Both declared shapes were sampled on purpose with over-length inputs, and every one overflowed
    # (so the anchor audit really audited cut renders, not only whole ones). The cut counts are the
    # role client's own census, read through its capture.
    facts = stage1_facts(recipe, pairs_rows(), tokenizer_of(recipe), OVER_LENGTH_PER_SHAPE)
    assert facts["per_shape"]["query"]["cuts"] == OVER_LENGTH_PER_SHAPE
    assert facts["per_shape"]["document"]["cuts"] == OVER_LENGTH_PER_SHAPE
    assert document["anchor_check"]["passed"] is True
    assert document["anchor_check"]["checked"] >= N_PAIRS + 2 * OVER_LENGTH_PER_SHAPE
    assert document["render_check"]["status"] == "run"
    assert document["render_check"]["passed"] is True
    # Without an engine the /tokenize check is reported not_run, never passed (R29).
    assert document["engine_tokenize_check"]["status"] == "not_run"
    assert document["engine_tokenize_check"]["passed"] is None


def test_the_card_example_renders_to_the_measured_ids(tmp_path: Path, hub_cache: Path) -> None:
    """Token-id equality against the reference subprocess, with the measured invariants."""
    _skip_unless_hub_reachable()

    recipe = load_recipe(RECIPE_DIR)
    tokenizer = tokenizer_of(recipe)
    pairs = write_pairs(tmp_path / "pairs.jsonl", pairs_rows()[:1])
    out = tmp_path / "reference.json"
    reference = run_reference(
        sys.executable,
        str(RECIPE_DIR / recipe.reference.entry),
        mode="render",
        pairs_path=str(pairs),
        out_path=out,
        tokenizer_spec=f"{REPO}@{REVISION}",
    )
    rows = {(row["index"], row["shape"]): row for row in reference["rows"]}
    query_row, document_row = rows[(0, "query")], rows[(0, "document")]
    template = client_template(recipe)
    assert template is not None
    frame = template.segments("query")[0].fixed
    assert query_row["text"] == frame + CARD_QUERY
    assert document_row["text"] == CARD_DOCUMENT
    # The served fit renders the same strings (what the engine receives), and the ids carry the
    # measured invariants: 27 and 8 tokens, each ending on the endoftext anchor.
    served_query = served_texts(recipe, [CARD_QUERY], "query")[0]
    served_document = served_texts(recipe, [CARD_DOCUMENT], "document")[0]
    assert served_query == query_row["text"]
    assert served_document == document_row["text"]
    query_ids = tokenizer.ids(served_query, add_special_tokens=True)
    document_ids = tokenizer.ids(served_document, add_special_tokens=True)
    assert len(query_ids) == CARD_QUERY_IDS and query_ids[-1] == END_OF_TEXT_ID
    assert len(document_ids) == CARD_DOCUMENT_IDS
    assert tokenizer.special_id(END_OF_TEXT_NAME) == END_OF_TEXT_ID
    pinned = tokenizer_of(recipe)
    assert isinstance(pinned.sha256, str) and len(pinned.sha256) == 64


def test_dropping_the_trailing_anchor_position_declaration_turns_the_anchor_check_red(
    tmp_path: Path, hub_cache: Path
) -> None:
    """The mutation: a wrong anchor declaration reds the audit on the shape it misdescribes.

    The query shape ends on its content span and declares the appended endoftext anchor with
    add_special_tokens: true. Declaring ``anchor: first`` instead tells the audit the anchor is a
    head fixed segment -- the document shape has none, so its declared edge cannot hold and every
    document render reds.
    """
    _skip_unless_hub_reachable()
    copied = tmp_path / "qwen3-embedding-0.6b"
    copied.mkdir()
    for name in ("recipe.yaml", "reference.py"):
        (copied / name).write_bytes((RECIPE_DIR / name).read_bytes())
    data = yaml.safe_load((copied / "recipe.yaml").read_text(encoding="utf-8"))
    data["client"]["template"]["anchor"] = "first"
    (copied / "recipe.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    mutated = load_recipe(copied)
    pairs = write_pairs(tmp_path / "pairs.jsonl", pairs_rows())
    document = stage1_prompts(mutated, str(pairs), None, over_length_per_shape=OVER_LENGTH_PER_SHAPE)
    assert document["anchor_check"]["passed"] is False
    assert "document" in {failure["shape"] for failure in document["anchor_check"]["failures"]}


def test_an_over_cap_pairs_row_rides_the_declared_table_with_the_cards_uncut_prompt(
    tmp_path: Path, hub_cache: Path
) -> None:
    """Decision 9: the reference renders the card's prompt uncut (the card truncates ids at encode)
    and never ports the client's cut.  An over-budget pairs row therefore differs from the client's
    content-only cut by declaration (``over_cap_cut_differs``): stage 1 reports it in the non-gating
    table and still passes on every under-cap row."""
    _skip_unless_hub_reachable()
    recipe = load_recipe(RECIPE_DIR)
    rows = pairs_rows()
    long_document = "long document about retrieval " * 9000
    rows.append({"query": CARD_QUERY, "documents": [long_document]})
    pairs = write_pairs(tmp_path / "pairs.jsonl", rows)
    document = stage1_prompts(recipe, str(pairs), sys.executable, over_length_per_shape=OVER_LENGTH_PER_SHAPE)
    render = document["render_check"]
    assert render["passed"] is True, render["failures"][:1]
    over_cap = render["over_cap"]
    assert over_cap["known_deviation"] is True and over_cap["gating"] is False
    assert [(entry["index"], entry["shape"]) for entry in over_cap["rows"]] == [(len(rows) - 1, "document")]
    reference = run_reference(
        sys.executable,
        str(RECIPE_DIR / recipe.reference.entry),
        mode="render",
        pairs_path=str(pairs),
        out_path=tmp_path / "reference.json",
        tokenizer_spec=f"{REPO}@{REVISION}",
    )
    texts = {(row["index"], row["shape"]): row["text"] for row in reference["rows"]}
    assert texts[(len(rows) - 1, "document")] == long_document  # the card's uncut prompt
    assert document["passed"] is True


# ---------------------------------------------------------------------------
# The declared contract: every serve, client and reference field pinned.
# ---------------------------------------------------------------------------

EXPECTED_TOP = {
    "id": "qwen3-embedding-0.6b",
    "input": ["text"],
    "licence": "apache-2.0",
    "model": "Qwen/Qwen3-Embedding-0.6B",
    "revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
    "role": "embed",
}
EXPECTED_SERVE = {
    "chat_template": None,
    "convert": None,
    "dtype": "bfloat16",
    "extra_args": [],
    "hf_overrides": {},
    "io_processor_plugin": None,
    "limit_mm_per_prompt": None,
    "max_model_len": 32768,
    "mm_processor_kwargs": {},
    "plugin": None,
    "pooler_config": {},
    "runner": "pooling",
    "trust_remote_code": False,
}
EXPECTED_CLIENT = {
    "api": "openai_embeddings",
    "request_shape": "text",
    "tokenizer": "Qwen/Qwen3-Embedding-0.6B@97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
    "max_tokens": 8192,
    "template": {
        "query": [
            {"fixed": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"},
            {"content": "query"},
        ],
        "document": [{"content": "document"}],
        "anchor": "last",
        "add_special_tokens": True,
    },
    "on_overflow": "cut",
    "empty_doc": "send",
    "normalize": True,
    "dimensions": None,
    "model": "qwen3-embedding-0.6b",
    "revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
}
EXPECTED_REFERENCE = {
    "entry": "reference.py",
    "kind": "transformers",
    "known_deviations": ["over_cap_cut_differs"],
    "score_scale": "cosine",
}

# Two mutants per recipe against the contract pin above: each drift must fail, naming the field.
MUTANTS: list[tuple[str, tuple[str, ...], object, str]] = [
    ("serve.max_model_len drifts to 40960", ("serve", "max_model_len"), 40960, "max_model_len"),
    ("client.max_tokens drifts to 4096", ("client", "max_tokens"), 4096, "client.max_tokens"),
]


def _mutated_recipe(tmp_path: Path, path: tuple[str, ...], value: object) -> object:
    """The recipe directory copied into ``tmp_path`` with one YAML field set to ``value``."""
    import shutil

    import yaml

    target = tmp_path / RECIPE_DIR.name
    shutil.copytree(RECIPE_DIR, target)
    yaml_path = target / "recipe.yaml"
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    node = data
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return load_recipe(target)


def _assert_contract(recipe: object) -> None:
    assert_recipe_contract(
        recipe,
        serve=EXPECTED_SERVE,
        client=EXPECTED_CLIENT,
        reference=EXPECTED_REFERENCE,
        top=EXPECTED_TOP,
    )


@pytest.mark.parametrize(("label", "path", "value", "needle"), MUTANTS, ids=[m[0] for m in MUTANTS])
def test_two_contract_mutants_are_red(
    label: str, path: tuple[str, ...], value: object, needle: str, tmp_path: Path
) -> None:
    """A drifted field fails the contract assertion naming it (two mutants per recipe)."""
    _assert_contract(load_recipe(RECIPE_DIR))  # the pinned recipe itself is green
    with pytest.raises(AssertionError) as caught:
        _assert_contract(_mutated_recipe(tmp_path, path, value))
    assert needle in str(caught.value), f"{label}: the failure must name {needle}: {caught.value}"


def test_notes_state_the_merged_budget_wiring_and_the_query_cap_check() -> None:
    """The notes read the merged product: the budget is fitted on the
    wire, and the query_max_tokens check found no separate referent cap."""
    notes = load_recipe(RECIPE_DIR).notes
    assert "fitted to the declared budget on the wire" in notes
    assert "not wired yet" not in notes and "harness pre-fits" not in notes
    assert "no separate query cap exists in the referent" in notes
