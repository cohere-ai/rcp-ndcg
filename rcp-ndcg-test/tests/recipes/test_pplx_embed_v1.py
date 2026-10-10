"""The ``pplx-embed-v1`` family: its contract per variant, stage 1 on CPU, the family proof and
the mutations (decision 34: one family module, parametrized over its variant ids; every field pinned
per variant; two mutants red per family).

Stage 1 runs on the real tokenizer files of the pinned revisions (no weights), fetched through the
shared ``_served.fetch_tokenizer`` into ``RCP_NDCG_VLLM_TOKENIZER_CACHE`` (else ``tmp_path``) and
skipped with a clear reason offline. The reference runs as a subprocess (its ``render`` mode needs
the tokenizer and the checkpoint's two small config files only; the harness process never imports
torch or transformers) and reads its variant from the resolved recipe the tests pass as ``--recipe``.

The family's over-cap policy (owner decision 9): the reference renders the card's own cut -- the
text right-cut at ``min(tokenizer model_max_length, config max_position_embeddings)`` = 32768, the
number sentence-transformers' ``Transformer`` caps its tokenizer to -- and never the client's
content-only cut, so the recipe declares ``over_cap_cut_differs``: the one corner where an id cut
splits a multi-token character is reported, not gated, and every text the client sent uncut gates
exactly.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from rcp_ndcg_test.equivalence import stage1_prompts
from rcp_ndcg_vllm.recipe import (
    client_config,
    default_recipes_root,
    load_family,
    resolve_recipe,
    serve_argv,
)

from rcp_ndcg.data.templates import TemplateSpec
from rcp_ndcg.data.tokenizer import load_tokenizer
from rcp_ndcg.inference.config import EmbeddingEndpoint

from ._contract import assert_recipe_contract
from ._served import client_template, fetch_tokenizer, served_texts
from .test_pplx_embed_v2_late import INTERNAL_LABELS

FAMILY_ID = "pplx-embed-v1"
FAMILY_DIR = default_recipes_root() / FAMILY_ID

#: The family's variants (decision 34): the per-size facts the tests pin. The two tokenizer.json
#: files are byte-identical at the pinned revisions (one SHA-256); the architecture, the pooling,
#: the (absent) prompts and the (absent) head are the family's shared contract.
VARIANTS: dict[str, dict[str, Any]] = {
    "pplx-embed-v1-0.6b": {
        "repo": "perplexity-ai/pplx-embed-v1-0.6b",
        "revision": "2c4d510dd4a732063c31a0f70193e35067b51fd8",
        "sha256": "c6fb5c5bbba5fa5f8332edfb6d8aa67bd7fb3d75365b1765f108201698eaebf5",
        "dim": 1024,
        "hidden_size": 1024,
        "layers": 28,
    },
    "pplx-embed-v1-4b": {
        "repo": "perplexity-ai/pplx-embed-v1-4b",
        "revision": "06456497a00540a582918fe8dcd3a5eabb207772",
        "sha256": "c6fb5c5bbba5fa5f8332edfb6d8aa67bd7fb3d75365b1765f108201698eaebf5",
        "dim": 2560,
        "hidden_size": 2560,
        "layers": 36,
    },
}
VARIANT_IDS = list(VARIANTS)

#: The model's own context limit (config.json max_position_embeddings; the card's 32K) and the
#: tokenizer's own, higher, declared maximum (tokenizer_config.json model_max_length).
CONTEXT_LIMIT = 32768
TOKENIZER_MAX_LENGTH = 131072
N_PAIRS = 20
OVER_LENGTH_PER_SHAPE = 5

#: The checkpoint's own storage quantiser: the sentence-transformers pipeline's last module
#: (modules.json) and its class in the repo's st_quantize.py. The reference stops before it (the
#: served engine returns the float mean-pooled vector); the tests pin the facts the decision rests on.
QUANTIZER_MODULE_TYPE = "st_quantize.FlexibleQuantizer"
QUANTIZER_CLASS_NAME = "FlexibleQuantizer"


def _sha256(path: Path) -> str:
    """The SHA-256 of one file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(params=VARIANT_IDS)
def variant_id(request: pytest.FixtureRequest) -> str:
    """One variant id of the family (every test runs per variant)."""
    return str(request.param)


@pytest.fixture
def tokenizer_dir(tmp_path: Path, variant_id: str) -> Path:
    """The pinned revision's tokenizer files in the shared download cache (one home: ``_served``)."""
    variant = VARIANTS[variant_id]
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        path = fetch_tokenizer(
            f"https://huggingface.co/{variant['repo']}/resolve/{variant['revision']}/{name}",
            f"{FAMILY_ID}/{name}",
            tmp_path,
            sha256=variant["sha256"] if name == "tokenizer.json" else None,
        )
    return path.parent


def _checkpoint_file(tokenizer_spec: str, filename: str) -> Any:
    """One small JSON file of the pinned checkpoint (``config.json``, ``tokenizer_config.json``,
    ``modules.json``), fetched at the spec's revision."""
    from huggingface_hub import hf_hub_download

    repo, _, revision = tokenizer_spec.partition("@")
    path = hf_hub_download(repo, filename, revision=revision)
    text = Path(path).read_text(encoding="utf-8")
    return json.loads(text) if filename.endswith(".json") else text


def _pairs() -> list[dict[str, Any]]:
    """Twenty realistic retrieval pairs (several documents carry punctuation and one an emoji)."""
    rows = [
        ("what statute governs limitations?", "A statute of limitations sets the deadline to file a suit."),
        ("capital of france", "Paris is the capital and largest city of France."),
        ("who wrote pride and prejudice", "Pride and Prejudice is an 1813 novel written by Jane Austen."),
        ("boiling point of water", "Water boils at 100 degrees Celsius at standard atmospheric pressure."),
        ("largest planet", "Jupiter is the largest planet in the solar system, more than twice as massive as Saturn."),
        ("speed of light", "The speed of light in a vacuum is exactly 299,792,458 metres per second."),
        ("first moon landing", "Apollo 11 landed on the Moon on July 20, 1969; Armstrong stepped out six hours later."),
        ("python list vs tuple", "A tuple is immutable while a list is mutable, so tuples can serve as keys."),
        ("how vaccines work", "Vaccines train the immune system to recognize pathogens without causing the disease."),
        ("the great wall length", "The Great Wall of China stretches over 21,000 kilometers across many dynasties."),
        ("causes of inflation", "Inflation rises when demand outpaces supply, or when production costs increase."),
        ("photosynthesis inputs", "Photosynthesis converts carbon dioxide and water into glucose using light energy."),
        ("black hole definition", "A black hole is a region where gravity is so strong that nothing can escape it."),
        ("eiffel tower height", "The Eiffel Tower stands about 330 metres tall and was completed in 1889."),
        ("what is dna", "DNA carries the genetic instructions used in the growth and development of organisms."),
        ("olympics origin", "The Olympic Games began in ancient Greece at Olympia in 776 BC."),
        ("restful api meaning", "A REST API exposes resources over HTTP using standard verbs and stateless requests."),
        ("avogadro number", "Avogadro's number, 6.02214076e23, counts particles in one mole of a substance."),
        (
            "why is the sky blue",
            "Air molecules scatter shorter blue wavelengths of sunlight more than longer red ones.",
        ),
        ("sql join types", "INNER JOIN keeps matches; LEFT JOIN keeps all rows of the left table with nulls."),
    ]
    return [{"query": query, "documents": [document]} for query, document in rows]


def _write_pairs(path: Path, rows: list[dict[str, Any]]) -> Path:
    """One pairs JSONL file (the harness's load_pairs format)."""
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# The declared contract: every serve, client and reference field pinned per variant.
# ---------------------------------------------------------------------------
def _expected_client(variant_id: str) -> dict[str, Any]:
    variant = VARIANTS[variant_id]
    return {
        "api": "openai_embeddings",
        "instruction": "none",
        "request_shape": "text",
        "recipe": (
            "vLLM v0.31.0 pooling runner; the stock Qwen3ForCausalLM converted to embed with "
            "is_causal false; the plugin-registered config class parses config.json locally; the "
            "checkpoint's own ST metadata resolves MEAN pooling and no activation; raw text on "
            "/v1/embeddings; the 32768-token right cut client-side"
        ),
        "max_tokens": CONTEXT_LIMIT,
        "template": {
            "query": [{"content": "query"}],
            "document": [{"content": "document"}],
            "anchor": "mean",
            "add_special_tokens": True,
        },
        "on_overflow": "cut",
        "empty_doc": "omit_zero",
        "normalize": True,
        "dimensions": None,
        "model": variant_id,
        "revision": variant["revision"],
        "tokenizer": f"{variant['repo']}@{variant['revision']}",
    }


EXPECTED_SERVE = {
    "patches": ["pooling-full-context"],
    "runner": "pooling",
    "convert": None,
    "hf_overrides": {"architectures": ["Qwen3ForCausalLM"], "is_causal": False},
    "chat_template": None,
    "pooler_config": {},
    "trust_remote_code": False,
    "max_model_len": CONTEXT_LIMIT,
    "dtype": "bfloat16",
    "plugin": "rcp-ndcg-vllm",
    "io_processor_plugin": None,
    "mm_processor_kwargs": {},
    "limit_mm_per_prompt": None,
    "extra_args": [],
}

EXPECTED_REFERENCE = {
    "attn_implementation": None,
    "kind": "sentence_transformers",
    "score_scale": "cosine",
    "entry": "reference.py",
    "known_deviations": ["over_cap_cut_differs"],
    "device": None,
}


def _expected_top(variant_id: str) -> dict[str, Any]:
    variant = VARIANTS[variant_id]
    return {
        "id": variant_id,
        "model": variant["repo"],
        "revision": variant["revision"],
        "role": "embed",
        "input": ["text"],
        "licence": "MIT",
    }


def _contract(recipe: Any, variant_id: str) -> None:
    assert_recipe_contract(
        recipe,
        serve=EXPECTED_SERVE,
        client=_expected_client(variant_id),
        reference=EXPECTED_REFERENCE,
        top=_expected_top(variant_id),
    )


def test_recipe_contract_pins_every_field(variant_id: str) -> None:
    """The recipe loads against the product's EmbeddingEndpoint and declares exactly its contract."""
    recipe = resolve_recipe(variant_id)
    _contract(recipe, variant_id)
    assert recipe.engine.image == "vllm/vllm-openai:v0.31.0" and recipe.engine.min_version == "0.31.0"
    assert recipe.resources.gpus == 1
    assert recipe.serve.max_model_len >= recipe.client.get("max_tokens")  # the engine must not 400 the budget
    assert (FAMILY_DIR / "requirements-reference.txt").is_file()  # the family's one convention
    assert (FAMILY_DIR / "reference.py").is_file()
    assert recipe.status.state == "unverified"
    assert recipe.sources


def test_the_contract_reds_on_two_mutants(variant_id: str) -> None:
    """Two mutants of the declared contract red the pin: a served field and the client's budget."""
    recipe = resolve_recipe(variant_id)
    serve_mutant = recipe.model_copy(update={"serve": recipe.serve.model_copy(update={"max_model_len": 33024})})
    with pytest.raises(AssertionError, match="max_model_len"):
        _contract(serve_mutant, variant_id)
    client_mutant = recipe.model_copy(update={"client": {**recipe.client, "max_tokens": 8192}})
    with pytest.raises(AssertionError, match="max_tokens"):
        _contract(client_mutant, variant_id)


def test_client_config_round_trips_through_the_product(variant_id: str) -> None:
    """The client block is the product's endpoint config: the dump constructs the model unchanged."""
    recipe = resolve_recipe(variant_id)
    config = client_config(recipe, base_url="http://127.0.0.1:8100/v1")
    assert config["model"] == recipe.id
    endpoint = EmbeddingEndpoint(**config)
    assert endpoint.empty_doc == "omit_zero"
    assert endpoint.max_tokens == CONTEXT_LIMIT
    assert endpoint.normalize is True
    template = client_template(recipe)
    assert template is not None
    assert template.shapes() == ("query", "document")
    assert template.anchor == "mean"
    assert template.adds_special_tokens("query") is True and template.adds_special_tokens("document") is True


def test_serve_argv_renders_the_engine_command(variant_id: str) -> None:
    """The argv a wave runs: the stock architecture override, the bidirectional contract, no convert,
    no remote code, no template and no plugin flag."""
    recipe = resolve_recipe(variant_id)
    argv = serve_argv(recipe, port=8100, served_model_name=recipe.id)
    assert argv[:3] == ["vllm", "serve", VARIANTS[variant_id]["repo"]]
    assert argv[argv.index("--revision") + 1] == VARIANTS[variant_id]["revision"]
    assert argv[argv.index("--served-model-name") + 1] == recipe.id
    assert argv[argv.index("--runner") + 1] == "pooling"
    assert json.loads(argv[argv.index("--hf-overrides") + 1]) == {
        "architectures": ["Qwen3ForCausalLM"],
        "is_causal": False,
    }
    assert argv[argv.index("--max-model-len") + 1] == str(CONTEXT_LIMIT)
    assert argv[argv.index("--dtype") + 1] == "bfloat16"
    assert argv[argv.index("--pooler-config") + 1] == "{}"
    assert "--convert" not in argv  # the pooling runner auto-resolves embed for this architecture
    assert "--trust-remote-code" not in argv  # the plugin registers the config class locally
    assert "--chat-template" not in argv
    assert not any("rcp-ndcg-vllm" in argument for argument in argv)  # the plugin never reaches the argv


def test_the_family_shares_every_behavioural_block() -> None:
    """Decision 34's family proof: the two variants differ only in their identity facts.

    The family file carries ONE serve/client/reference block; the variants' rows carry no
    ``overrides``; the resolved recipes' behavioural blocks are equal across the sizes (the
    per-size facts -- the model, the revision, the injected tokenizer spec and the notes -- are
    the only differences). A future size that needs another value is refused by the family schema,
    so this test pins the property the schema enforces.
    """
    family = load_family(FAMILY_DIR)
    assert [variant.id for variant in family.variants] == VARIANT_IDS
    assert all(not variant.overrides.serve and not variant.overrides.client for variant in family.variants)
    assert all(variant.overrides.resources is None for variant in family.variants)
    recipes = [resolve_recipe(variant_id) for variant_id in VARIANT_IDS]
    assert recipes[0].serve == recipes[1].serve
    assert recipes[0].client["template"] == recipes[1].client["template"]
    assert recipes[0].client["empty_doc"] == recipes[1].client["empty_doc"] == "omit_zero"
    assert recipes[0].reference == recipes[1].reference
    assert recipes[0].input == recipes[1].input == ["text"]
    assert recipes[0].role == recipes[1].role == "embed"
    assert recipes[0].model != recipes[1].model and recipes[0].revision != recipes[1].revision


def test_the_two_tokenizers_are_byte_identical(variant_id: str, tokenizer_dir: Path) -> None:
    """The family's tokenizer is one file: each pinned revision's tokenizer.json hashes to the same
    SHA-256 (the pin), and the tokenizer_config.json is identical too (measured).

    The tokenizer carries the Qwen specials and NO chat template; the ByteLevel post-processor has
    no BOS/EOS template, so ``add_special_tokens`` is a no-op -- the raw-text wire rests on that.
    """
    variant = VARIANTS[variant_id]
    assert _sha256(tokenizer_dir / "tokenizer.json") == variant["sha256"]
    config = _checkpoint_file(f"{variant['repo']}@{variant['revision']}", "tokenizer_config.json")
    assert "chat_template" not in config
    assert config["model_max_length"] == TOKENIZER_MAX_LENGTH
    assert config["bos_token"] is None
    tokenizer = load_tokenizer(str(tokenizer_dir / "tokenizer.json"))
    assert tokenizer.sha256 == variant["sha256"]
    assert tokenizer.special_id("endoftext") == 151643
    assert tokenizer.ids("hello world", add_special_tokens=True) == tokenizer.ids(
        "hello world", add_special_tokens=False
    )
    assert tokenizer.ids("", add_special_tokens=True) == []  # an empty text renders zero tokens


def test_the_reference_cap_is_the_models_context(variant_id: str, tokenizer_dir: Path) -> None:
    """The card's cut cap: min(tokenizer model_max_length 131072, config max_position_embeddings 32768).

    sentence-transformers' Transformer caps the tokenizer at the model's context and ships no
    sentence_bert_config.json here, so the capped value is also ``max_seq_length``. The reference
    derives exactly this number from the two pinned files (never a hardcoded constant).
    """
    variant = VARIANTS[variant_id]
    spec = f"{variant['repo']}@{variant['revision']}"
    config = _checkpoint_file(spec, "config.json")
    tokenizer_config = _checkpoint_file(spec, "tokenizer_config.json")
    assert config["max_position_embeddings"] == CONTEXT_LIMIT
    assert tokenizer_config["model_max_length"] == TOKENIZER_MAX_LENGTH
    assert min(tokenizer_config["model_max_length"], config["max_position_embeddings"]) == CONTEXT_LIMIT
    # The architecture and pooling facts the family proof rests on (the same at both revisions).
    assert config["architectures"] == ["PPLXQwen3Model"]
    assert config["model_type"] == "bidirectional_pplx_qwen3"
    assert config["hidden_size"] == VARIANTS[variant_id]["hidden_size"]
    assert config["num_hidden_layers"] == VARIANTS[variant_id]["layers"]
    assert config["dtype"] == "float32"
    assert config["use_bidirectional_attention"] is True
    assert config["tie_word_embeddings"] is True
    pooling = _checkpoint_file(spec, "1_Pooling/config.json")
    assert pooling["pooling_mode_mean_tokens"] is True
    assert pooling["word_embedding_dimension"] == VARIANTS[variant_id]["dim"]
    assert pooling["pooling_mode_lasttoken"] is False and pooling["pooling_mode_cls_token"] is False


@pytest.mark.network
def test_the_storage_quantizer_is_the_pipelines_last_module(variant_id: str) -> None:
    """The reference's float-encoder contract rests on the checkpoint's own pipeline: Transformer ->
    Pooling -> the int8 storage quantizer, whose class the repo ships in st_quantize.py.

    The reference loads the card's pipeline and stops before the quantizer (the served engine returns
    the float mean-pooled vector); if a revision changes the pipeline the reference fails loudly.
    """
    variant = VARIANTS[variant_id]
    spec = f"{variant['repo']}@{variant['revision']}"
    modules = _checkpoint_file(spec, "modules.json")
    assert [module["type"] for module in modules] == [
        "sentence_transformers.models.Transformer",
        "sentence_transformers.models.Pooling",
        QUANTIZER_MODULE_TYPE,
    ]
    source = _checkpoint_file(spec, "st_quantize.py")
    assert f"class {QUANTIZER_CLASS_NAME}" in source
    assert 'Literal["int8", "binary", "ubinary"] = "int8"' in source  # the default storage view
    import importlib.util

    module_spec = importlib.util.spec_from_file_location("pplx_embed_v1_reference", FAMILY_DIR / "reference.py")
    assert module_spec and module_spec.loader
    reference_module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(reference_module)
    assert reference_module.EXPECTED_MODULES == ("Transformer", "Pooling", QUANTIZER_CLASS_NAME)


def test_stage1_passes_on_cpu(tmp_path: Path, tokenizer_dir: Path, variant_id: str) -> None:
    """Stage 1 on CPU: the product's fit, the anchor audit and the reference render, all green.

    Twenty pairs rows plus the harness's own over-length inputs (five per declared shape, padded in
    that shape's own content span, cut by the product's budget with every fixed token reserved). The
    render check compares the reference subprocess's renders (the card's own cut) against the captured
    texts, zero tolerance; the engine /tokenize check reports not_run without an engine, never passed.
    """
    variant = VARIANTS[variant_id]
    recipe = resolve_recipe(variant_id)
    pairs = _write_pairs(tmp_path / "pairs.jsonl", _pairs())
    document = stage1_prompts(recipe, str(pairs), sys.executable, over_length_per_shape=OVER_LENGTH_PER_SHAPE)
    assert document["passed"] is True, (document["anchor_check"], document["render_check"])
    assert document["sampled"] == N_PAIRS + 2 * OVER_LENGTH_PER_SHAPE
    assert document["anchor_check"]["passed"] is True, document["anchor_check"]["failures"][:1]
    assert document["anchor_check"]["checked"] == 2 * N_PAIRS + 2 * OVER_LENGTH_PER_SHAPE
    assert document["render_check"]["status"] == "run"
    assert document["render_check"]["passed"] is True, document["render_check"]["failures"][:1]
    assert document["render_check"]["rows"] == 2 * N_PAIRS
    assert document["template_render_check"] is None  # no served chat template
    engine = document["engine_tokenize_check"]
    assert engine["status"] == "not_run" and engine["passed"] is None
    assert variant["sha256"] == load_tokenizer(str(tokenizer_dir / "tokenizer.json")).sha256


def test_the_reference_render_is_the_raw_text_under_the_cap(
    tmp_path: Path, tokenizer_dir: Path, variant_id: str
) -> None:
    """The card prepends nothing: under the cap the reference's render is the raw text, byte for byte,
    and the client ships the same string; over the cap it is the card's right cut (a prefix)."""
    from rcp_ndcg_test.equivalence.reference import run_reference

    variant = VARIANTS[variant_id]
    recipe = resolve_recipe(variant_id)
    rows = [
        {"query": "capital of france", "documents": ["Paris is the capital of France."]},
        {"query": "emoji \U0001f680 test " * 12000, "documents": ["plain document"]},
    ]
    pairs = _write_pairs(tmp_path / "pairs.jsonl", rows)
    out = tmp_path / "reference.json"
    reference = run_reference(
        sys.executable,
        str(FAMILY_DIR / "reference.py"),
        mode="render",
        pairs_path=str(pairs),
        out_path=out,
        tokenizer_spec=f"{variant['repo']}@{variant['revision']}",
        recipe=recipe,
    )
    texts = {(row["index"], row["shape"]): row["text"] for row in reference["rows"]}
    assert texts[(0, "query")] == rows[0]["query"]  # no prompt, under the cap
    assert texts[(0, "document")] == rows[0]["documents"][0]
    assert served_texts(recipe, [rows[0]["query"]], "query") == [rows[0]["query"]]
    over_cap_query = texts[(1, "query")]
    assert rows[1]["query"].startswith(over_cap_query) and len(over_cap_query) < len(rows[1]["query"])
    assert over_cap_query  # the cut kept text


def test_the_card_cut_differs_where_it_splits_a_character(tmp_path: Path, tokenizer_dir: Path, variant_id: str) -> None:
    """Why the recipe declares ``over_cap_cut_differs``: the card cuts ids, the client cuts text.

    An emoji spans several byte-level tokens; when the card's 32768th id falls inside one, the card's
    model reads that emoji's leading byte token, which no text carries. The reference's text keeps
    whole tokens of whole characters -- a strict prefix of the card's ids -- and the client ships the
    same text here: the render check sees equal texts, while the model inputs differ by that one id (a
    stage-2 difference on the vector, inside the declared deviation).
    """
    variant = VARIANTS[variant_id]
    recipe = resolve_recipe(variant_id)
    raw = "emoji \U0001f680 test " * 12000
    reference = run_reference_render(recipe, [{"query": raw, "documents": ["d"]}], tmp_path, variant)
    reference_text = reference[(0, "query")]
    shipped = served_texts(recipe, [raw], "query")[0]
    tokenizer = load_tokenizer(str(tokenizer_dir / "tokenizer.json"))
    backend = tokenizer.backend
    backend.no_truncation()
    if len(backend.encode(raw, add_special_tokens=True).ids) <= CONTEXT_LIMIT:
        pytest.fail("the sample is not over the cap; the corner cannot be measured")
    backend.enable_truncation(max_length=CONTEXT_LIMIT, strategy="longest_first", direction="right")
    try:
        card_ids = backend.encode(raw, add_special_tokens=True).ids
    finally:
        backend.no_truncation()
    reference_ids = tokenizer.ids(reference_text, add_special_tokens=True)
    assert raw.startswith(reference_text)
    assert len(card_ids) == CONTEXT_LIMIT and card_ids[: len(reference_ids)] == reference_ids
    assert len(reference_ids) < len(card_ids), "the card reads an id (the emoji's leading bytes) no text carries"
    assert shipped == reference_text, "the client's text cut keeps the same whole tokens here"
    assert load_family(FAMILY_DIR).reference.known_deviations == ["over_cap_cut_differs"]


def run_reference_render(
    recipe: Any, rows: list[dict[str, Any]], tmp_path: Path, variant: dict[str, Any]
) -> dict[tuple[int, str], str]:
    """The reference subprocess's render mode over ``rows``, keyed by ``(row index, shape)``."""
    from rcp_ndcg_test.equivalence.reference import run_reference

    pairs = _write_pairs(tmp_path / "pairs.jsonl", rows)
    out = tmp_path / "reference.json"
    reference = run_reference(
        sys.executable,
        str(FAMILY_DIR / "reference.py"),
        mode="render",
        pairs_path=str(pairs),
        out_path=out,
        tokenizer_spec=f"{variant['repo']}@{variant['revision']}",
        recipe=recipe,
    )
    return {(int(row["index"]), str(row["shape"])): str(row["text"]) for row in reference["rows"]}


def test_the_empty_document_policy_is_omit_zero(tokenizer_dir: Path, variant_id: str) -> None:
    """An empty document renders zero tokens for this tokenizer (no frame, no post-processor), and
    the engine's MEAN pooler would divide by zero; the declared policy is omit_zero.

    The card's own pipeline pools an empty input to a zero vector, so the omitted document's zero
    matches the card's value; the pairs generator leaves the empty-content row out and records the
    stratum absent, and nothing is silently sent.
    """
    variant = VARIANTS[variant_id]
    recipe = resolve_recipe(variant_id)
    tokenizer = load_tokenizer(str(tokenizer_dir / "tokenizer.json"))
    assert tokenizer.ids("", add_special_tokens=True) == []
    template = TemplateSpec.model_validate(recipe.client["template"])
    assert template.render("document", tokenizer, document="") == ""
    assert recipe.client["empty_doc"] == "omit_zero"
    assert recipe.client.get("empty_doc_text") is None
    assert variant["sha256"]  # the tokenizer pin is real, not a placeholder


def test_shipped_recipe_files_carry_no_internal_labels() -> None:
    """Every shipped file of this family reads as a self-contained public statement: no internal
    process shorthand, private work directory or undefined rule id (the families' scan, one family's
    own file set)."""
    hits = [
        f"{path.name}:{number}: {line.strip()[:120]}"
        for path in sorted(FAMILY_DIR.iterdir())
        if path.is_file()
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if INTERNAL_LABELS.search(line)
    ]
    assert not hits, "\n".join(hits)
