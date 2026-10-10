"""The ``qwen3-embedding`` family: every variant validates, pins its declared contract, and passes
stage 1 on CPU with its own real tokenizer.

One module per family (decision 34), parametrized over the family's variants: the 0.6b, the 4b and the
8b. Stage 1 runs the harness's own ``stage1_prompts`` with the family directory's ``reference.py`` as the
subprocess (its render mode needs only ``huggingface_hub`` -- no torch, no weights; the reference reads
the variant's model and revision from ``--recipe``). The pinned tokenizer files are downloaded into the
shared tokenizer cache (``RCP_NDCG_VLLM_TOKENIZER_CACHE`` when set, ``tmp_path`` otherwise;
``_served.tokenizer_cache``) and the tests skip with a clear reason when the Hub is unreachable or
``huggingface_hub`` is absent (CI's slim venv).

The measured invariants (at every pinned revision): the card's example query renders to 27 token ids
and the example document to 8, each ending on the post-processor's endoftext anchor (id 151643) that
last-token pooling reads. The 4b and 8b tokenizer.json files are the earlier build (22 added tokens to
the 0.6b's 26), but their vocab.json/merges.txt bytes, special tokens and post-processor are identical,
so the rendered ids are the same.
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

#: The family's variants at their pinned revisions (re-checked against the Hub API; not gated), with the
#: per-size facts the resolved contract carries: the checkpoint's own max_position_embeddings (the
#: serve.max_model_len override) and the tokenizer.json the stage-1 checks download and hash-pin.
VARIANTS: dict[str, dict[str, Any]] = {
    "qwen3-embedding-0.6b": {
        "model": "Qwen/Qwen3-Embedding-0.6B",
        "revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        "max_model_len": 32768,
        "mrl_range": [32, 1024],
        "tokenizer_sha256": "def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a",
    },
    "qwen3-embedding-4b": {
        "model": "Qwen/Qwen3-Embedding-4B",
        "revision": "5cf2132abc99cad020ac570b19d031efec650f2b",
        "max_model_len": 40960,
        "mrl_range": [32, 2560],
        "tokenizer_sha256": "83cdf8c3a34f68862319cb1810ee7b1e2c0a44e0864ae930194ddb76bb7feb8d",
    },
    "qwen3-embedding-8b": {
        "model": "Qwen/Qwen3-Embedding-8B",
        "revision": "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af",
        "max_model_len": 40960,
        "mrl_range": [32, 4096],
        "tokenizer_sha256": "83cdf8c3a34f68862319cb1810ee7b1e2c0a44e0864ae930194ddb76bb7feb8d",
    },
}
VARIANT_IDS = sorted(VARIANTS)
END_OF_TEXT_NAME = "endoftext"  # the appended anchor the last-token pooler reads; never typed out
END_OF_TEXT_ID = 151643
RECIPE_DIR = default_recipes_root() / "qwen3-embedding"
CARD_QUERY = "What is the capital of China?"
CARD_DOCUMENT = "The capital of China is Beijing."
#: The measured invariant on the card's example (token equality at every pinned revision).
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
            huggingface_hub.hf_hub_download(
                VARIANTS[VARIANT_IDS[0]]["model"],
                "tokenizer.json",
                revision=VARIANTS[VARIANT_IDS[0]]["revision"],
            )
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


@pytest.mark.parametrize("variant_id", VARIANT_IDS)
def test_the_recipe_loads_and_declares_the_served_path(variant_id: str) -> None:
    """The schema validates and the rendered argv is the served path, per variant."""
    facts = VARIANTS[variant_id]
    recipe = load_recipe(variant_id)
    _assert_contract(recipe)  # every serve, client and reference field pinned, exactly
    assert recipe.id == variant_id
    assert recipe.model == facts["model"] and recipe.revision == facts["revision"]
    assert recipe.role == "embed" and recipe.input == ["text"] and recipe.licence == "apache-2.0"
    assert recipe.client.get("api") == "openai_embeddings"
    assert recipe.client.get("tokenizer") == f"{facts['model']}@{facts['revision']}"
    assert recipe.client.get("max_tokens") == 8192 and recipe.serve.max_model_len == facts["max_model_len"]
    assert recipe.client.get("on_overflow") == "cut"
    assert "query_prompt" not in recipe.client and "doc_prompt" not in recipe.client  # the frame is the template
    template = client_template(recipe)
    assert template is not None
    assert template.shapes() == ("query", "document")
    assert template.anchor == "last"
    assert recipe.serve.runner == "pooling" and recipe.serve.convert is None
    assert recipe.serve.dtype == "bfloat16"
    assert recipe.serve.chat_template is None and recipe.serve.trust_remote_code is False
    assert recipe.serve.pooler_config == {} and recipe.serve.plugin is None
    assert recipe.resources.gpus == 1  # bf16 weights + KV fit one 80 GB-class GPU at every size
    assert recipe.reference.kind == "transformers" and recipe.reference.score_scale == "cosine"
    assert recipe.reference.known_deviations == ["over_cap_cut_differs"] and recipe.status.state == "unverified"
    argv = serve_argv(recipe, port=8100, served_model_name=recipe.id)
    assert argv[:3] == ["vllm", "serve", facts["model"]]
    assert argv[argv.index("--revision") + 1] == facts["revision"]
    assert argv[argv.index("--runner") + 1] == "pooling"
    assert argv[argv.index("--dtype") + 1] == "bfloat16"
    assert argv[argv.index("--max-model-len") + 1] == str(facts["max_model_len"])
    assert argv[argv.index("--pooler-config") + 1] == "{}"
    assert "--chat-template" not in argv and "--trust-remote-code" not in argv


@pytest.mark.network
@pytest.mark.parametrize("variant_id", VARIANT_IDS)
def test_the_query_frame_is_the_checkpoint_sentence_transformers_prompt(variant_id: str, hub_cache: Path) -> None:
    """The template's frame is byte-identical to the checkpoint's own prompts.query, at every size."""
    _skip_unless_hub_reachable()
    from huggingface_hub import hf_hub_download

    facts = VARIANTS[variant_id]
    recipe = load_recipe(variant_id)
    template = client_template(recipe)
    assert template is not None
    st_path = hf_hub_download(facts["model"], "config_sentence_transformers.json", revision=facts["revision"])
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


@pytest.mark.network
@pytest.mark.parametrize("variant_id", VARIANT_IDS)
def test_stage1_token_ids_and_anchors_pass_on_cpu(variant_id: str, tmp_path: Path, hub_cache: Path) -> None:
    """Stage 1 on CPU, per variant: the reference render agrees byte-exactly and every anchor survives
    every cut (the reference subprocess loads this variant's checkpoint pin from --recipe)."""
    _skip_unless_hub_reachable()
    recipe = load_recipe(variant_id)
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


def test_embed_rows_takes_the_checkpoint_from_the_resolved_recipe(monkeypatch: pytest.MonkeyPatch) -> None:
    """The checkpoint identity comes from the resolved recipe (``--recipe``), not the module constants.

    A variant row pointing at another checkpoint must load THAT checkpoint (and the tokenizer spec must
    pin it); the stubbed heavy modules keep the test offline (the harness process imports no torch).
    """
    import importlib.util
    import sys as _sys
    import types

    calls: list[dict] = []

    class _FakeModel:
        def to(self, device):
            return self

        def eval(self):
            return self

    def fake_from_pretrained(name, **kwargs):
        calls.append({"name": name, **kwargs})
        return _FakeModel()

    monkeypatch.setitem(
        _sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoModel=types.SimpleNamespace(from_pretrained=fake_from_pretrained),
            AutoTokenizer=types.SimpleNamespace(from_pretrained=fake_from_pretrained),
        ),
    )
    torch_stub = types.ModuleType("torch")
    functional = types.ModuleType("torch.nn.functional")
    torch_stub.nn = types.SimpleNamespace(functional=functional)
    monkeypatch.setitem(_sys.modules, "torch", torch_stub)
    monkeypatch.setitem(_sys.modules, "torch.nn", types.ModuleType("torch.nn"))
    monkeypatch.setitem(_sys.modules, "torch.nn.functional", functional)
    spec = importlib.util.spec_from_file_location("qwen3_embedding_reference", RECIPE_DIR / "reference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.embed_rows(
        [],
        "example-org/other-checkpoint@" + "0" * 40,
        "cpu",
        {"model": "example-org/other-checkpoint", "revision": "0" * 40},
    )
    assert result == {"rows": []}
    assert [call["name"] for call in calls] == ["example-org/other-checkpoint", "example-org/other-checkpoint"]
    assert all(call["revision"] == "0" * 40 for call in calls)


@pytest.mark.network
@pytest.mark.parametrize("variant_id", VARIANT_IDS)
def test_the_card_example_renders_to_the_measured_ids(variant_id: str, tmp_path: Path, hub_cache: Path) -> None:
    """Token-id equality against the reference subprocess, with the measured invariants, per variant."""
    _skip_unless_hub_reachable()
    facts = VARIANTS[variant_id]
    recipe = load_recipe(variant_id)
    tokenizer = tokenizer_of(recipe)
    pairs = write_pairs(tmp_path / "pairs.jsonl", pairs_rows()[:1])
    out = tmp_path / "reference.json"
    reference = run_reference(
        sys.executable,
        str(RECIPE_DIR / recipe.reference.entry),
        mode="render",
        pairs_path=str(pairs),
        out_path=out,
        tokenizer_spec=f"{facts['model']}@{facts['revision']}",
        recipe=recipe,
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
    assert pinned.sha256 == facts["tokenizer_sha256"], "the variant's own tokenizer bytes, hash-pinned"


@pytest.mark.network
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
    copied = tmp_path / "qwen3-embedding"
    copied.mkdir()
    for name in ("family.yaml", "reference.py"):
        (copied / name).write_bytes((RECIPE_DIR / name).read_bytes())
    data = yaml.safe_load((copied / "family.yaml").read_text(encoding="utf-8"))
    data["client"]["template"]["anchor"] = "first"
    (copied / "family.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    mutated = load_recipe("qwen3-embedding-0.6b", root=tmp_path)
    pairs = write_pairs(tmp_path / "pairs.jsonl", pairs_rows())
    document = stage1_prompts(mutated, str(pairs), None, over_length_per_shape=OVER_LENGTH_PER_SHAPE)
    assert document["anchor_check"]["passed"] is False
    assert "document" in {failure["shape"] for failure in document["anchor_check"]["failures"]}


@pytest.mark.network
def test_an_over_cap_pairs_row_rides_the_declared_table_with_the_cards_uncut_prompt(
    tmp_path: Path, hub_cache: Path
) -> None:
    """Decision 9: the reference renders the card's prompt uncut (the card truncates ids at encode)
    and never ports the client's cut.  An over-budget pairs row therefore differs from the client's
    content-only cut by declaration (``over_cap_cut_differs``): stage 1 reports it in the non-gating
    table and still passes on every under-cap row."""
    _skip_unless_hub_reachable()
    recipe = load_recipe("qwen3-embedding-0.6b")
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
        tokenizer_spec=recipe.client["tokenizer"],
        recipe=recipe,
    )
    texts = {(row["index"], row["shape"]): row["text"] for row in reference["rows"]}
    assert texts[(len(rows) - 1, "document")] == long_document  # the card's uncut prompt
    assert document["passed"] is True


# ---------------------------------------------------------------------------
# The declared contract: every serve, client and reference field pinned.
# ---------------------------------------------------------------------------

EXPECTED_TOP = {
    "input": ["text"],
    "licence": "apache-2.0",
    "role": "embed",
}
EXPECTED_SERVE = {
    "patches": [],
    "chat_template": None,
    "convert": None,
    "dtype": "bfloat16",
    "extra_args": [],
    "hf_overrides": {"is_matryoshka": True},
    "io_processor_plugin": None,
    "limit_mm_per_prompt": None,
    "mm_processor_kwargs": {},
    "plugin": None,
    "patches": [],
    "plugin_architectures": [],
    "pooler_config": {},
    "runner": "pooling",
    "trust_remote_code": False,
}
EXPECTED_CLIENT = {
    "api": "openai_embeddings",
    "instruction": "none",
    "request_shape": "text",
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
    "mrl_kind": "truncation",
    "dimensions": None,
}
EXPECTED_REFERENCE = {
    "attn_implementation": None,
    "entry": "reference.py",
    "kind": "transformers",
    "known_deviations": ["over_cap_cut_differs"],
    "device": "cuda",
    "score_scale": "cosine",
}

# Two mutants per variant against the contract pin above: each drift must fail, naming the field.
# The 0.6b's drift moves its 32768 up; the 4b's and 8b's move theirs down -- the per-size
# max_model_len override is load-bearing, not a copy of the family's value.
MUTANTS: list[tuple[str, str, str, object, str]] = [
    ("qwen3-embedding-0.6b", "serve.max_model_len drifts to 40960", "serve", 40960, "max_model_len"),
    ("qwen3-embedding-0.6b", "client.max_tokens drifts to 4096", "client", 4096, "client.max_tokens"),
    ("qwen3-embedding-4b", "serve.max_model_len drifts to 32768", "serve", 32768, "max_model_len"),
    ("qwen3-embedding-4b", "client.max_tokens drifts to 4096", "client", 4096, "client.max_tokens"),
    ("qwen3-embedding-8b", "serve.max_model_len drifts to 32768", "serve", 32768, "max_model_len"),
    ("qwen3-embedding-8b", "client.max_tokens drifts to 4096", "client", 4096, "client.max_tokens"),
]


def _mutated_recipe(variant_id: str, block: str, value: object) -> object:
    """The loaded variant with one resolved field replaced (``serve`` or ``client``)."""
    recipe = load_recipe(variant_id)
    if block == "serve":
        mutated = recipe.serve.model_copy(update={"max_model_len": value})
    else:
        mutated = {**recipe.client, "max_tokens": value}
    return recipe.model_copy(update={block: mutated})


def _expected_top(variant_id: str) -> dict[str, object]:
    facts = VARIANTS[variant_id]
    return {"id": variant_id, "model": facts["model"], "revision": facts["revision"], **EXPECTED_TOP}


def _expected_serve(variant_id: str) -> dict[str, object]:
    return {**EXPECTED_SERVE, "max_model_len": VARIANTS[variant_id]["max_model_len"]}


def _expected_client(variant_id: str) -> dict[str, object]:
    facts = VARIANTS[variant_id]
    return {
        **EXPECTED_CLIENT,
        "tokenizer": f"{facts['model']}@{facts['revision']}",
        "mrl_range": facts["mrl_range"],
        "model": variant_id,
        "revision": facts["revision"],
    }


def _assert_contract(recipe: object) -> None:
    assert_recipe_contract(
        recipe,
        serve=_expected_serve(recipe.id),
        client=_expected_client(recipe.id),
        reference=EXPECTED_REFERENCE,
        top=_expected_top(recipe.id),
    )


@pytest.mark.parametrize(("variant_id", "label", "block", "value", "needle"), MUTANTS, ids=[m[1] for m in MUTANTS])
def test_two_contract_mutants_are_red(variant_id: str, label: str, block: str, value: object, needle: str) -> None:
    """A drifted field fails the contract assertion naming it (two mutants per variant)."""
    _assert_contract(load_recipe(variant_id))  # the pinned recipe itself is green
    with pytest.raises(AssertionError) as caught:
        _assert_contract(_mutated_recipe(variant_id, block, value))
    assert needle in str(caught.value), f"{label}: the failure must name {needle}: {caught.value}"


def test_notes_state_the_merged_budget_wiring_and_the_query_cap_check() -> None:
    """The notes read the merged product: the budget is fitted on the
    wire, and the query_max_tokens check found no separate referent cap."""
    notes = load_recipe("qwen3-embedding-0.6b").notes
    assert "fitted to the declared budget on the wire" in notes
    assert "not wired yet" not in notes and "harness pre-fits" not in notes
    assert "no separate query cap exists in the referent" in notes
