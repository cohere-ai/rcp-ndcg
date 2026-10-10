"""The ``ctxl-rerank-v2-instruct-multilingual`` family: the full contract per variant, and stage 1 on
CPU (decision 34: one family module, parametrized over its variant ids; every field pinned per variant;
two mutants red per family).

The contract test freezes every ``serve``/``client``/``reference`` field through the shared
:func:`assert_recipe_contract` (nothing rides unpinned), and two mutants show it red on drift.  The
served semantics this family decided are pinned with failing-first tests: ``instruction: none``
(the paper configs' mode — the reference never folds an instruction, a pairs row's instruction is
ignored on both sides) and the merged rerank client's settle rule (the query ships at its declared
share whenever it exceeds it — the pair fit alone binds the share on overflow only).

Stage 1 runs without an engine: the product's role client ships the request bodies, and the
reference subprocess's ``render`` fills the span format with the paper's own spans (the raw query
and documents, uncut -- never a port of the client's cut), equal to the wire's on every under-cap
row while over-cap rows ride the declared ``anchor_drop_over_cap`` table, with the served template
file rendering what the declared template renders.  ``tokenizer.json`` downloads through the shared
``fetch_tokenizer`` (``RCP_NDCG_VLLM_TOKENIZER_CACHE`` when set, else ``tmp_path``; sha256-pinned);
those tests are network tests and skip offline (``tests/recipes/conftest.py``).
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest
import yaml
from rcp_ndcg_test.equivalence import stage1_prompts
from rcp_ndcg_test.equivalence.reference import run_reference
from rcp_ndcg_vllm import resolve_recipe, serve_argv
from rcp_ndcg_vllm.recipe import default_recipes_root

from ._contract import assert_recipe_contract
from ._served import client_template, fetch_tokenizer, served_pair, served_rows, stage1_facts

RECIPE_ID = "ctxl-rerank-v2-instruct-multilingual"
RECIPE_DIR = default_recipes_root() / RECIPE_ID
FAMILY_DIR = RECIPE_DIR  # the family directory (the recipes root holds families, decision 34)
#: The family's variants (decision 34): the per-size facts the tests pin. The 1b/2b checkpoints are
#: Qwen3ForCausalLM (token id 0 "!" is the head's classifier token), the 6b is the vendor's
#: MistralForCausalLM ("<unk>" as id 0); the 1b serves at the 6b's 32768 window, the 2b at the paper's
#: 8192; the 1b carries the paper's batch_size 32 (the 2b/6b endpoints run the schema default). The
#: family client omits the endpoint defaults (``request_shape``, ``listwise``, ``add_special_tokens``)
#: the old standalone recipes declared.
VARIANTS: dict[str, dict[str, str | int]] = {
    "ctxl-rerank-v2-instruct-multilingual-1b": {
        "repo": "ContextualAI/ctxl-rerank-v2-instruct-multilingual-1b",
        "revision": "8fd1edf6a98564cb712064f884b8ef7df5c1b876",
        "sha256": "f253e845dff94cb1ac558f76905ea5fbe19c21ebf2d9b4e44f28ef0007968267",
        "max_model_len": 32768,
        "architecture": "Qwen3ForSequenceClassification",
        "classifier_token": "!",
        "batch_size": 32,
    },
    "ctxl-rerank-v2-instruct-multilingual-2b": {
        "repo": "ContextualAI/ctxl-rerank-v2-instruct-multilingual-2b",
        "revision": "6ffef5dc552583b8db58dc4a87f79f7aee78d2d9",
        "sha256": "f253e845dff94cb1ac558f76905ea5fbe19c21ebf2d9b4e44f28ef0007968267",
        "max_model_len": 8192,
        "architecture": "Qwen3ForSequenceClassification",
        "classifier_token": "!",
        "batch_size": 32,
    },
    "ctxl-rerank-v2-instruct-multilingual-6b": {
        "repo": "ContextualAI/ctxl-rerank-v2-instruct-multilingual-6b",
        "revision": "f14ca1a2fc204dc0934c84a3d2e278f8ff646b80",
        "sha256": "b0240ce510f08e6c2041724e9043e33be9d251d1e4a4d94eb68cd47b954b61d2",
        "max_model_len": 32768,
        "architecture": "MistralForSequenceClassification",
        "classifier_token": "<unk>",
        "batch_size": 32,
    },
}
VARIANT_IDS = list(VARIANTS)

MAX_TOKENS = 8192
QUERY_MAX_TOKENS = 4096

FRAME_HEAD = "Check whether a given document contains information helpful to answer the query.\n<Document> "
FRAME_MID = "\n<Query> "
FRAME_TAIL = " ??"


def _expected(variant_id: str) -> dict[str, dict[str, object]]:
    """The variant's full resolved contract: every field of every block, exactly as the product models
    resolve it (authored values and schema defaults alike). Nothing may ride unpinned."""
    variant = VARIANTS[variant_id]
    return {
        "serve": {
            "patches": [],
            "chat_template": "template.jinja",
            "convert": None,
            "dtype": "bfloat16",
            "extra_args": [],
            "hf_overrides": {
                "architectures": [variant["architecture"]],
                "classifier_from_token": [variant["classifier_token"]],
                "method": "no_post_processing",
                # no head_dtype: the engine's pooling head keeps vLLM's fp32 default, the engine side
                # of the like-for-like pair with the reference's fp32 score head (2026-10-10 decision)
            },
            "io_processor_plugin": None,
            "limit_mm_per_prompt": None,
            "max_model_len": variant["max_model_len"],
            "mm_processor_kwargs": {},
            "plugin": None,
            "plugin_architectures": [],
            "pooler_config": {"use_activation": False},
            "runner": "pooling",
            "trust_remote_code": False,
        },
        "client": {
            "api": "rerank",
            "tokenizer": f"{variant['repo']}@{variant['revision']}",
            "max_tokens": 8192,
            "query_max_tokens": 4096,
            "template": {
                "pair": [
                    {"fixed": FRAME_HEAD},
                    {"content": "document"},
                    {"fixed": FRAME_MID},
                    {"content": "query"},
                    {"fixed": FRAME_TAIL},
                ],
                "anchor": "last",
                "add_special_tokens": True,
            },
            "instruction": "none",
            "use_activation": False,
            "on_overflow": "cut",
            "empty_doc": "send",
            "empty_query": "send",
            "model": variant_id,
            "revision": variant["revision"],
        },
        "reference": {
            "attn_implementation": "sdpa",
            "entry": "reference.py",
            "kind": "transformers",
            "known_deviations": ["anchor_drop_over_cap"],
            "device": None,  # the schema default
            "score_scale": "logit",
        },
        "top": {
            "id": variant_id,
            "licence": "CC-BY-NC-SA-4.0",
            "revision": variant["revision"],
            "role": "rerank",
            "scoring": "pointwise",
        },
        "engine": {
            "image": "vllm/vllm-openai:v0.31.0",
            "min_version": "0.31.0",
            "name": "vllm",
            "startup_timeout_s": 1800,  # the schema default; the recipe no longer restates it
            "step_budget_s": None,  # the schema default; the recipe no longer restates it
        },
    }


@pytest.fixture(params=VARIANT_IDS)
def variant_id(request: pytest.FixtureRequest) -> str:
    """One variant id of the family (every test runs per variant)."""
    return str(request.param)


def _tokenizer_file(tmp_path: Path, variant_id: str) -> Path:
    """The variant's ``tokenizer.json`` at the pinned revision, sha256-checked, through the shared
    tokenizer cache (``_served.fetch_tokenizer``); skips with the reason when offline."""
    variant = VARIANTS[variant_id]
    url = f"https://huggingface.co/{variant['repo']}/resolve/{variant['revision']}/tokenizer.json"
    return fetch_tokenizer(url, f"{variant_id}/tokenizer.json", tmp_path, sha256=variant["sha256"])


def _local_recipe(tmp_path: Path, variant_id: str):
    """The family copied into ``tmp_path`` with its shared ``client.tokenizer`` on the downloaded
    tokenizer (the same bytes, a local spec), so the product and the reference subprocess tokenise
    offline; the variant resolves through the same loader."""
    tokenizer_file = _tokenizer_file(tmp_path, variant_id)
    copied = tmp_path / "ctxl-rerank-v2-instruct-multilingual"
    shutil.copytree(RECIPE_DIR, copied)
    data = yaml.safe_load((copied / "family.yaml").read_text(encoding="utf-8"))
    data["client"]["tokenizer"] = str(tokenizer_file)
    (copied / "family.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return resolve_recipe(variant_id, root=tmp_path), tokenizer_file


def _long_text(tokenizer, tokens: int) -> str:
    """A whitespace-clean text of more than ``tokens`` tokens, sized from one measured unit (no
    re-count of a growing text)."""
    unit = " flibbertigibbet"
    per_unit = max(1, tokenizer.count(unit * 8) // 8)
    text = "flibbertigibbet" + unit * (tokens // per_unit + 50) + "."
    assert tokenizer.count(text) > tokens
    return text


def write_pairs(path: Path, rows: list[dict]) -> Path:
    """The pairs JSONL file for a stage-1 or reference run."""
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


#: Fifteen realistic pairs (three with a run-level instruction): short on purpose — the gating
#: pairs keep queries within the share and documents within the budget (the recipe's declared
#: divergence policy) — plus one over-budget pair and one over-share query in the span tests.
PAIRS: list[dict] = [
    {
        "query": "What is the capital of France?",
        "documents": ["Paris is the capital and largest city of France."],
        "instruction": "Answer with the city name",
    },
    {"query": "Who wrote Pride and Prejudice?", "documents": ["Pride and Prejudice is an 1813 novel by Jane Austen."]},
    {
        "query": "capital of Japan",
        "documents": ["Tokyo is the capital of Japan and the seat of its government."],
        "instruction": "Find the city",
    },
    {
        "query": "definition of entropy",
        "documents": ["Entropy is a measurable property, most commonly associated with a state of disorder."],
    },
    {
        "query": "largest planet",
        "documents": ["Jupiter is the fifth planet from the Sun and the largest."],
        "instruction": "  ",
    },
    {"query": "speed of light", "documents": ["The speed of light in vacuum equals 299,792,458 metres per second."]},
    {
        "query": "who painted the Mona Lisa",
        "documents": ["The Mona Lisa was painted by the Italian Renaissance artist Leonardo da Vinci."],
    },
    {"query": "chemical symbol for gold", "documents": ["Gold is a chemical element with the symbol Au."]},
    {
        "query": "longest river in Africa",
        "documents": ["The Nile is a major north-flowing river in Africa, and the longest."],
    },
    {
        "query": "first person on the Moon",
        "documents": ["Neil Armstrong was an astronaut and the first person to walk on the Moon."],
    },
    {
        "query": "boiling point of water",
        "documents": ["The boiling point of water is 100 degrees Celsius at standard pressure."],
    },
    {"query": "currency of Japan", "documents": ["The yen is the official currency of Japan."]},
    {
        "query": "who developed relativity",
        "documents": ["Albert Einstein developed the theory of relativity, a pillar of modern physics."],
    },
    {
        "query": "  main language of Brazil  ",
        "documents": ["Portuguese is the official language of Brazil, spoken by nearly everyone."],
    },
    {
        "query": "smallest prime number",
        "documents": ["The number 2 is the smallest prime number and the only even prime."],
    },
]


def test_recipe_loads_and_declares_the_full_contract(variant_id: str) -> None:
    """Every serve, client and reference field (plus the pinned top-level and engine ones) is
    frozen: a value drift, an unpinned field or a vanished field all fail naming the exact path."""
    expected = _expected(variant_id)
    recipe = resolve_recipe(variant_id)
    assert_recipe_contract(
        recipe,
        serve=expected["serve"],
        client=expected["client"],
        reference=expected["reference"],
        top=expected["top"],
    )
    assert recipe.engine.model_dump(mode="json") == expected["engine"]
    assert recipe.input == ["text"] and recipe.status.state == "unverified"


def test_mutant_dropping_the_serve_max_model_len_reds_the_contract_naming_the_field(variant_id: str) -> None:
    """Mutant 1: ``serve.max_model_len`` -> 16384 must red, naming the field."""
    expected = _expected(variant_id)
    recipe = resolve_recipe(variant_id)
    mutated = recipe.model_copy(update={"serve": recipe.serve.model_copy(update={"max_model_len": 16384})})
    with pytest.raises(AssertionError, match=r"serve\.max_model_len"):
        assert_recipe_contract(
            mutated,
            serve=expected["serve"],
            client=expected["client"],
            reference=expected["reference"],
            top=expected["top"],
        )


def test_mutant_changing_the_reference_kind_reds_the_contract_naming_the_field(variant_id: str) -> None:
    """Mutant 2: ``reference.kind`` transformers -> remote_code must red, naming the field."""
    expected = _expected(variant_id)
    recipe = resolve_recipe(variant_id)
    mutated = recipe.model_copy(update={"reference": recipe.reference.model_copy(update={"kind": "remote_code"})})
    with pytest.raises(AssertionError, match=r"reference\.kind"):
        assert_recipe_contract(
            mutated,
            serve=expected["serve"],
            client=expected["client"],
            reference=expected["reference"],
            top=expected["top"],
        )


def test_serve_argv_renders_the_golden_engine_command(variant_id: str) -> None:
    """The variant's ``vllm serve`` argv: the template file, the id-0 score head, the logit pooler."""
    recipe = resolve_recipe(variant_id)
    argv = serve_argv(recipe, port=8100, served_model_name=recipe.id)
    assert argv[:3] == ["vllm", "serve", VARIANTS[variant_id]["repo"]]
    assert argv[argv.index("--revision") + 1] == VARIANTS[variant_id]["revision"]
    assert argv[argv.index("--served-model-name") + 1] == recipe.id
    assert argv[argv.index("--chat-template") + 1] == str(FAMILY_DIR / "template.jinja")
    assert argv[argv.index("--pooler-config") + 1] == '{"use_activation": false}'
    assert argv[argv.index("--max-model-len") + 1] == str(VARIANTS[variant_id]["max_model_len"])
    assert json.loads(argv[argv.index("--hf-overrides") + 1]) == _expected(variant_id)["serve"]["hf_overrides"]
    assert argv[argv.index("--runner") + 1] == "pooling"


def test_the_reference_environment_is_documented() -> None:
    """The reference declares the environment it needs, beside itself (the reference rule): the family's
    ``reference.in`` (the image's torch with the paper's transformers pin; flash-attn is dropped -- the
    reference declares sdpa) resolved to its ``reference.lock``."""
    text = (FAMILY_DIR / "reference.in").read_text(encoding="utf-8")
    assert "torch>=2.0" in text and "torch==2.9.1" not in text
    assert "transformers==4.57.6" in text
    assert not any(line.strip().startswith("flash-attn") for line in text.splitlines())
    assert (FAMILY_DIR / "reference.lock").is_file()


def test_the_reference_resolves_the_hub_tokenizer_spec_without_the_revision_suffix() -> None:
    """The family's ``client.tokenizer`` (repo@revision) reaches transformers as a bare repo id: the
    reference splits the spec and passes the revision separately (the model and revision come from
    the resolved recipe, the family's variants)."""
    import importlib.util

    module_spec = importlib.util.spec_from_file_location("ctxl_reference", RECIPE_DIR / "reference.py")
    module = importlib.util.module_from_spec(module_spec)
    bytecode = sys.dont_write_bytecode  # exec_module must not drop a __pycache__ into the recipe dir
    sys.dont_write_bytecode = True
    try:
        module_spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = bytecode
    assert set(module.BATCH_SIZES) == set(VARIANT_IDS), "the family's per-variant batch sizes drifted"
    repo = str(VARIANTS["ctxl-rerank-v2-instruct-multilingual-2b"]["repo"])
    assert module._tokenizer_dir(f"{repo}@rev") == repo
    assert module._tokenizer_dir(repo) == repo
    assert "@" not in module._tokenizer_dir(f"{repo}@rev")


def test_the_reference_loads_with_the_declared_attention_implementation() -> None:
    """The declared ``reference.attn_implementation`` reaches ``from_pretrained`` (sdpa here), never a
    silent ``torch.cuda.is_available()`` choice: the stock reference environment carries no flash-attn."""
    import importlib.util
    import types

    class _StubModel:
        def eval(self) -> object:
            return self

        def to(self, device: str) -> object:
            return self

    loads: list[dict] = []

    class _StubAutoModel:
        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> object:
            loads.append(kwargs)
            return _StubModel()

    torch_stub = types.ModuleType("torch")
    torch_stub.bfloat16 = "bfloat16"
    transformers_stub = types.ModuleType("transformers")
    transformers_stub.AutoModelForCausalLM = _StubAutoModel
    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(sys, "dont_write_bytecode", True)
        monkey.setitem(sys.modules, "torch", torch_stub)
        monkey.setitem(sys.modules, "transformers", transformers_stub)
        spec = importlib.util.spec_from_file_location("ctxl_reference_load", RECIPE_DIR / "reference.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        reference = object.__new__(module.CtxlRerankReference)
        reference.model = None
        reference.model_name = "ContextualAI/ctxl-rerank-v2-instruct-multilingual-1b"
        reference.revision = "0" * 40
        reference.dtype = "bfloat16"
        reference.attn_implementation = "sdpa"
        reference.load("cpu")
        assert loads and loads[-1]["attn_implementation"] == "sdpa"
        assert loads[-1]["revision"] == "0" * 40
    finally:
        monkey.undo()


def test_the_reference_scores_through_the_engines_fp32_head() -> None:
    """The score head is the engine's: the final hidden state cast to float32, projected onto the
    score row (``lm_head.weight[0]``) in float32 -- vLLM's fp32 pooling head (the recipes declare no
    ``head_dtype``). The paper's bf16 ``logits[:, -1, 0]`` is deliberately not what this reads (the
    stub's logits are a different value), and the hidden states are requested explicitly."""
    import importlib.util

    import torch

    hidden = torch.tensor([[[1.0, 2.0, 3.0]]], dtype=torch.bfloat16)  # (batch 1, position 1, hidden 3)
    score_row = torch.tensor([[0.5, -0.25, 1.0]], dtype=torch.bfloat16)  # (vocab 3, hidden 3); row 0 is the score

    class _LmHead:
        weight = score_row

    class _Out:
        hidden_states = (hidden,)
        logits = torch.tensor([[[99.0, 0.0, 0.0]]], dtype=torch.bfloat16)  # the bf16 head path this test must not read

    class _Model:
        lm_head = _LmHead()

        def __call__(self, **kwargs: object) -> object:
            assert kwargs.get("output_hidden_states") is True
            return _Out()

    class _Fast:
        def __call__(self, batch: object, **kwargs: object) -> dict[str, object]:
            return {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}

    class _Tokenizer:
        _fast = _Fast()

    spec = importlib.util.spec_from_file_location("ctxl_reference_score", RECIPE_DIR / "reference.py")
    module = importlib.util.module_from_spec(spec)
    bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = bytecode
    reference = object.__new__(module.CtxlRerankReference)
    reference.device = "cpu"
    reference.max_length = 8192
    reference.tokenizer = _Tokenizer()
    reference.model = _Model()
    scores = reference._forward_scores(["a prompt"])
    expected = float((hidden[:, -1].float() @ score_row[0].float())[0])
    assert scores == [expected]
    assert scores != [99.0]  # the bf16 logits path is not read


@pytest.mark.network
def test_the_served_and_reference_prompts_ignore_the_pairs_row_instruction(tmp_path: Path, variant_id: str) -> None:
    """``instruction: none`` end to end (the family decision): a pairs row's instruction is ignored
    on the wire and by the reference — the bare query ships, paddings and all, and no side folds
    ``Task: ...\\nQuery: ...``."""
    recipe, tokenizer_file = _local_recipe(tmp_path, variant_id)
    query = "  padded query  "
    document = "A document."
    shipped = served_pair(recipe, query, [document], instruction="Answer with the city name")
    assert shipped == {"query": query, "documents": [document]}
    rows = [
        {"query": query, "documents": [document], "instruction": "Answer with the city name"},
        {"query": query, "documents": [document], "instruction": "  "},
    ]
    pairs_path = write_pairs(tmp_path / "pairs.jsonl", rows)
    reference = run_reference(
        sys.executable,
        str(FAMILY_DIR / "reference.py"),
        mode="render",
        pairs_path=pairs_path,
        out_path=tmp_path / "reference.json",
        tokenizer_spec=str(tokenizer_file),
        recipe=recipe,
    )
    for row in reference["rows"]:
        assert row["query"] == query, "the reference must render the bare query, never a fold"
        assert row["documents"] == [document]


@pytest.mark.network
def test_the_reference_renders_the_paper_spans_never_the_clients_cut(tmp_path: Path, variant_id: str) -> None:
    """Decision 9: the reference's ``render`` fills the span format with the paper's own spans -- the
    raw query and documents its prompt builder receives, uncut.  Under the cap and within the share
    they equal the wire's spans byte for byte; on an over-share query and an over-budget document the
    wire ships the client's cut (the settle rule; the content-only cut) while the reference keeps the
    paper's uncut spans (the declared divergence row and the declared anchor_drop_over_cap row)."""
    from rcp_ndcg.data.tokenizer import load_tokenizer

    recipe, tokenizer_file = _local_recipe(tmp_path, variant_id)
    tokenizer = load_tokenizer(str(tokenizer_file))
    long_query = _long_text(tokenizer, QUERY_MAX_TOKENS + 400)
    long_document = _long_text(tokenizer, MAX_TOKENS + 200)
    rows = [
        {"query": "short query", "documents": ["a short document.", "a second document."]},
        {"query": long_query, "documents": ["a short document under the settled share."]},
        {"query": "short query", "documents": [long_document]},
    ]
    pairs_path = write_pairs(tmp_path / "pairs.jsonl", rows)
    wire = served_rows(recipe, rows, tokenizer)["per_shape"]["pair"]["spans"]
    reference = run_reference(
        sys.executable,
        str(FAMILY_DIR / "reference.py"),
        mode="render",
        pairs_path=pairs_path,
        out_path=tmp_path / "reference.json",
        tokenizer_spec=str(tokenizer_file),
        recipe=recipe,
    )
    spans = [{"query": row["query"], "documents": list(row["documents"])} for row in reference["rows"]]
    assert spans == [{"query": row["query"], "documents": row["documents"]} for row in rows]  # uncut
    assert spans[0] == wire[0]  # under the cap and the share: byte-identical with the wire
    # The wire settles the over-share query at its share (the merged client's settle rule).
    assert tokenizer.count(long_query) > QUERY_MAX_TOKENS >= tokenizer.count(wire[1]["query"])
    assert wire[1]["query"] == long_query[: len(wire[1]["query"])] != long_query
    # The over-budget pair: the wire cuts the document only, the " ??" anchor re-attached in budget.
    rendered = FRAME_HEAD + wire[2]["documents"][0] + FRAME_MID + wire[2]["query"] + FRAME_TAIL
    assert tokenizer.count(rendered, add_special_tokens=True) <= MAX_TOKENS
    assert wire[2]["query"] == "short query" and wire[2]["documents"][0] != long_document


@pytest.mark.network
def test_stage1_on_cpu_passes_token_id_equality_and_the_anchor_check(tmp_path: Path, variant_id: str) -> None:
    """Stage 1 on CPU: the reference's spans and the served template over the pairs file plus the
    harness's 5 over-length samples, with the anchor audit on every sampled row."""
    from rcp_ndcg.data.tokenizer import load_tokenizer

    recipe, tokenizer_file = _local_recipe(tmp_path, variant_id)
    tokenizer = load_tokenizer(str(tokenizer_file))
    rows = [*PAIRS, {"query": "short query", "documents": [_long_text(tokenizer, MAX_TOKENS + 200)]}]
    pairs_path = write_pairs(tmp_path / "pairs.jsonl", rows)
    document = stage1_prompts(recipe, pairs_path, sys.executable, over_length_per_shape=5)

    assert document["sampled"] >= 20
    assert document["anchor_check"]["passed"] is True, document["anchor_check"]["failures"][:1]
    assert document["anchor_check"]["checked"] == 2 * document["sampled"]
    render = document["render_check"]
    assert render["status"] == "run" and render["passed"] is True, render["failures"][:1]
    assert render["rows"] == len(rows)
    # The over-budget pairs row differs by declaration (the paper's uncut spans vs the client's
    # cut): reported in the non-gating table, never gated; every other row compared exactly.
    over_cap = render["over_cap"]
    assert over_cap["known_deviation"] is True and over_cap["gating"] is False
    assert [entry["index"] for entry in over_cap["rows"]] == [len(PAIRS)]
    assert document["template_render_check"]["passed"] is True, document["template_render_check"]["failures"][:1]
    assert document["engine_tokenize_check"]["status"] == "not_run"  # no engine on CPU: never "passed"
    assert document["passed"] is True, document
    facts = stage1_facts(recipe, rows, tokenizer, 5)
    assert facts["per_shape"]["pair"]["cut_rows"] >= 5, facts["per_shape"]["pair"]["cut_rows"]
    template = client_template(recipe)
    assert template is not None
    assert facts["per_shape"]["pair"]["overhead"] == template.overhead("pair", tokenizer)


@pytest.mark.network
def test_dropping_the_trailing_anchor_segment_reddens_the_template_check(tmp_path: Path, variant_id: str) -> None:
    """Mutation: drop the declared template's trailing anchor segment — the file-vs-declaration
    check goes red on the template the engine renders (the file still emits the " ??" the
    declaration lost)."""
    mutated = tmp_path / "mutant" / RECIPE_DIR.name
    shutil.copytree(RECIPE_DIR, mutated)
    data = yaml.safe_load((mutated / "family.yaml").read_text(encoding="utf-8"))
    data["client"]["tokenizer"] = str(_tokenizer_file(tmp_path, variant_id))
    assert data["client"]["template"]["pair"][-1]["fixed"] == FRAME_TAIL
    data["client"]["template"]["pair"] = data["client"]["template"]["pair"][:-1]
    (mutated / "family.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    recipe = resolve_recipe(variant_id, root=tmp_path / "mutant")
    pairs_path = write_pairs(tmp_path / "pairs.jsonl", PAIRS[:3])
    document = stage1_prompts(recipe, pairs_path, sys.executable, over_length_per_shape=1)
    assert document["template_render_check"]["passed"] is False
    assert document["anchor_check"]["passed"] is True  # the span audit does not read the frame
