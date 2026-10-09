"""The ``octen-embedding`` family: its contract per variant, stage 1 on CPU, and the mutations
(decision 34: one family module, parametrized over its variant ids; every field pinned per variant;
two mutants red per family).

Stage 1 needs only the checkpoint's tokenizer file (no weights, no GPU): it is downloaded into the
shared tokenizer cache (``_served.tokenizer_cache``: ``RCP_NDCG_VLLM_TOKENIZER_CACHE`` when set,
else the test's ``tmp_path``) and verified against its pinned sha256 -- the model weights are never
needed on CPU. The tests skip with a clear reason when offline and nothing is cached. The recipe's
own ``client.get("tokenizer")`` stays the pinned Hub spec; the stage-1 runs point a copy of the
recipe at the local tokenizer file.

What the tests pin:

- the recipe validates through the product's ``EmbeddingEndpoint`` (the client block IS the config);
- stage 1 on CPU: the product's ``fit`` renders every sampled prompt (at least 20 sampled inputs,
  10 of them over-length), the anchor audit passes, and the reference subprocess's renders are
  byte-identical (zero tolerance);
- token-id equality: the fitted render's ids equal the paper string's ids (``"- " + document``,
  queries as-is) and end with the appended end-of-text anchor (added-token name ``endoftext``, id
  151643) -- the CPU stand-in for the engine's ``/tokenize`` check (R29), which runs only against a
  live engine;
- over-length inputs keep every anchor: the ``"- "`` prefix at the head (string-level: fit re-attaches
  the fixed frame after the cut) and the appended anchor at the tail, within the 8192 budget;
- the variant notes carry the per-size facts (the served vector width and the architecture's
  context limit), so a size cannot ride with another size's numbers;
- mutations: dropping the template's trailing anchor segment reddens the anchor check, and a recipe
  that declares no appended anchor at all is refused by the product's template validator;
- the reference module's constants equal the paper code's (``hf_dense.MAX_LENGTH``, the
  ``"- "`` prefix of ``octen.yaml``/``torch_dense``), and its render mode emits the paper's strings
  for every variant.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from rcp_ndcg_test.equivalence import stage1_prompts
from rcp_ndcg_test.equivalence.fitting import tokenizer_of
from rcp_ndcg_vllm.recipe import client_config, default_recipes_root, resolve_recipe

from rcp_ndcg.inference.config import EmbeddingEndpoint

from ._contract import assert_recipe_contract
from ._served import client_template, fetch_tokenizer, served_texts, stage1_facts

FAMILY_ID = "octen-embedding"
RECIPE_DIR = default_recipes_root() / FAMILY_ID

#: The family's variants (decision 34): the per-size facts the tests pin. ``dim`` is the served
#: vector width (the checkpoint's ``1_Pooling/config.json`` word_embedding_dimension; this family's
#: client block does not declare a dimension) and ``context`` the architecture's
#: ``max_position_embeddings``; both are asserted present in the variant's notes below.
VARIANTS: dict[str, dict[str, Any]] = {
    "octen-embedding-0.6b": {
        "repo": "Octen/Octen-Embedding-0.6B",
        "revision": "d715b32ee68f057b54dff09fc93c23485bc403d3",
        "sha256": "def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a",
        "dim": 1024,
        "context": 32768,
    },
    "octen-embedding-4b": {
        "repo": "Octen/Octen-Embedding-4B",
        "revision": "fea468fae3f0caffbae8a12ba792d1c394b6277d",
        "sha256": "83cdf8c3a34f68862319cb1810ee7b1e2c0a44e0864ae930194ddb76bb7feb8d",
        "dim": 2560,
        "context": 40960,
    },
    "octen-embedding-8b": {
        "repo": "Octen/Octen-Embedding-8B",
        "revision": "5adcfa292e712091dfc30f0e97f0b2282e6cc66c",
        "sha256": "83cdf8c3a34f68862319cb1810ee7b1e2c0a44e0864ae930194ddb76bb7feb8d",
        "dim": 4096,
        "context": 40960,
    },
}
VARIANT_IDS = list(VARIANTS)

MAX_TOKENS = 8192
#: The id the tokenizer's post-processor appends to every sequence (added-token name: endoftext) --
#: the last token the model pools, and the anchor the budget reserves. Measured identical on all
#: three pinned tokenizers (the 4B and 8B share the tokenizer file; the 0.6B's differs only in
#: vocabulary size and the ChatML template, not in the post-processor).
APPENDED_ANCHOR_ID = 151643
DOCUMENT_PREFIX = "- "

PAIRS: list[dict[str, Any]] = [
    {
        "query": "What is the capital of France?",
        "documents": ["Paris is the capital and largest city of France."],
    },
    {
        "query": "Who wrote the play Hamlet?",
        "documents": ["Hamlet is a tragedy written by William Shakespeare."],
    },
    {
        "query": "How fast can a peregrine falcon dive?",
        "documents": ["A diving peregrine falcon has been measured above 300 kilometres per hour."],
    },
    {
        "query": "What causes the tides on Earth?",
        "documents": ["The moon's gravity pulls the oceans, raising two bulges of water."],
    },
    {
        "query": "When did the Berlin Wall fall?",
        "documents": ["The Berlin Wall was opened on the night of 9 November 1989."],
    },
    {
        "query": "What is the boiling point of water?",
        "documents": ["At sea level water boils at 100 degrees Celsius."],
    },
    {
        "query": "Who painted the Mona Lisa?",
        "documents": ["Leonardo da Vinci painted the Mona Lisa in the early sixteenth century."],
    },
    {
        "query": "What is the largest desert on Earth?",
        "documents": ["The Antarctic polar desert is the largest desert on Earth."],
    },
    {
        "query": "How does a transformer model attend?",
        "documents": ["Attention weights score every token against every other token in the sequence."],
    },
    {
        "query": "What metal is liquid at room temperature?",
        "documents": ["Mercury is the only metal that is liquid at room temperature."],
    },
    {
        "query": "Which river flows through Vienna?",
        "documents": ["The Danube flows through Vienna on its way to the Black Sea."],
    },
    {
        "query": "渋谷は何で有名ですか。",
        "documents": ["渋谷は東京の主要な買い物と娯楽の中心地として知られています。"],
    },
    {
        "query": "What did Marie Curie discover?",
        "documents": ["Marie Curie discovered polonium and radium and coined the term radioactivity."],
    },
    {
        "query": "Where is the Great Barrier Reef?",
        "documents": ["The Great Barrier Reef lies off the coast of Queensland in north-eastern Australia."],
    },
    {
        "query": "What powers the sun?",
        "documents": ["Hydrogen fusion in the sun's core converts mass into energy."],
    },
    {
        "query": "How many bones are in the human body?",
        "documents": ["An adult human skeleton has 206 bones; babies are born with more."],
    },
]


def _recipe(variant_id: str) -> Any:
    """One variant's real recipe, loaded and validated through the product's endpoint config."""
    return resolve_recipe(variant_id)


def _tokenizer_file(tmp_path: Path, variant_id: str) -> Path:
    """The checkpoint's ``tokenizer.json`` at the pinned revision, through the shared tokenizer cache
    and verified against its pinned sha256 (``_served.fetch_tokenizer``: one home for the download,
    the cache variable and the offline skip)."""
    variant = VARIANTS[variant_id]
    url = f"https://huggingface.co/{variant['repo']}/resolve/{variant['revision']}/tokenizer.json"
    name = f"{variant_id}@{variant['revision']}/tokenizer.json"
    return fetch_tokenizer(url, name, tmp_path, sha256=variant["sha256"])


def _local_recipe(recipe: Any, tokenizer_dir: Path) -> Any:
    """The recipe with its tokenizer pointed at the local files (the Hub spec stays in the family file)."""
    client = {**recipe.client, "tokenizer": str(tokenizer_dir)}
    return recipe.model_copy(update={"client": client})


def _write_pairs(tmp_path: Path) -> Path:
    path = tmp_path / "pairs.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in PAIRS), encoding="utf-8")
    return path


def _expected_contract(variant_id: str) -> dict[str, Any]:
    """The variant's full resolved contract: every field of every block, exactly as the product
    models resolve it (authored values and schema defaults alike). Nothing may ride unpinned."""
    variant = VARIANTS[variant_id]
    return {
        "top": {
            "id": variant_id,
            "input": ["text"],
            "licence": "apache-2.0",
            "model": variant["repo"],
            "revision": variant["revision"],
            "role": "embed",
        },
        "serve": {
            "chat_template": None,
            "convert": None,
            "dtype": "bfloat16",
            "extra_args": [],
            "hf_overrides": {},
            "io_processor_plugin": None,
            "limit_mm_per_prompt": None,
            "max_model_len": 8192,
            "mm_processor_kwargs": {},
            "plugin": None,
            "pooler_config": {},
            "runner": "pooling",
            "trust_remote_code": False,
        },
        "client": {
            "api": "openai_embeddings",
            "tokenizer": f"{variant['repo']}@{variant['revision']}",
            "max_tokens": 8192,
            "template": {
                "query": [{"content": "query"}],
                "document": [{"fixed": "- "}, {"content": "document"}],
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
            "kind": "transformers",
            "known_deviations": ["over_cap_cut_differs"],
            "device": None,  # the schema default
            "score_scale": "cosine",
        },
    }


def _assert_contract(recipe: Any, variant_id: str) -> None:
    expected = _expected_contract(variant_id)
    assert_recipe_contract(
        recipe,
        serve=expected["serve"],
        client=expected["client"],
        reference=expected["reference"],
        top=expected["top"],
    )


# -- the recipe validates ----------------------------------------------------------------------


@pytest.mark.parametrize("variant_id", VARIANT_IDS)
def test_recipe_validates_against_the_product_endpoints(variant_id: str) -> None:
    """The recipe loads; its client block is the product's EmbeddingEndpoint with the paper's facts."""
    recipe = _recipe(variant_id)
    _assert_contract(recipe, variant_id)  # every serve, client and reference field pinned, exactly
    assert recipe.id == variant_id
    assert recipe.model == VARIANTS[variant_id]["repo"]
    assert recipe.revision == VARIANTS[variant_id]["revision"]
    assert recipe.role == "embed" and recipe.input == ["text"]
    assert recipe.serve.runner == "pooling"
    assert recipe.serve.chat_template is None  # the ChatML template would change every prompt
    assert recipe.serve.max_model_len == MAX_TOKENS
    assert recipe.serve.dtype == "bfloat16"
    assert recipe.serve.hf_overrides == {} and recipe.serve.pooler_config == {}
    client = recipe.client
    assert client.get("tokenizer") == f"{recipe.model}@{recipe.revision}"
    assert client.get("max_tokens") == MAX_TOKENS  # the whole input sequence, anchors included
    assert client.get("on_overflow") == "cut" and client.get("empty_doc") == "send" and client.get("normalize") is True
    assert "query_prompt" not in client and "doc_prompt" not in client  # the prefix lives in the template only
    template = client_template(recipe)
    assert template is not None
    assert template.shapes() == ("query", "document")
    assert template.anchor == "last"
    assert template.adds_special_tokens("query") and template.adds_special_tokens("document")
    assert [segment.content for segment in template.query] == ["query"]
    assert [segment.fixed for segment in template.document] == [DOCUMENT_PREFIX, None]
    # The paper's encode-time cut keeps the anchor; its boundary can differ from fit's by a token.
    assert recipe.reference.known_deviations == ["over_cap_cut_differs"]
    assert recipe.status.state == "unverified"
    config = client_config(recipe, base_url="http://127.0.0.1:8100/v1")
    assert EmbeddingEndpoint(**config).model == recipe.id


@pytest.mark.parametrize("variant_id", VARIANT_IDS)
def test_variant_notes_carry_the_per_size_facts(variant_id: str) -> None:
    """The variant's notes state its served vector width and the architecture's context limit, so a
    size cannot ride with another size's numbers (the facts are not resolved recipe fields here)."""
    notes = _recipe(variant_id).notes
    variant = VARIANTS[variant_id]
    assert f"hidden_size {variant['dim']}" in notes, f"the {variant['dim']}-wide hidden size is not stated"
    assert f"max_position_embeddings {variant['context']}" in notes, (
        f"the {variant['context']}-token context limit is not stated"
    )


# -- stage 1 on CPU -----------------------------------------------------------------------------


@pytest.mark.network
@pytest.mark.parametrize("variant_id", VARIANT_IDS)
def test_stage1_on_cpu_passes_token_equality_and_the_anchor_check(tmp_path: Path, variant_id: str) -> None:
    """Stage 1 (CPU): fit's renders match the reference's, and the anchor audit passes.

    At least 20 sampled inputs with at least 5 over-length ones: the pairs file's 16 rows plus 5
    over-length samples per declared shape (query and document). The reference runs as a subprocess
    (its render mode needs no torch and no transformers); the engine /tokenize check is reported
    not_run without an engine, never passed.
    """
    recipe = _local_recipe(_recipe(variant_id), _tokenizer_file(tmp_path, variant_id))
    pairs = _write_pairs(tmp_path)
    document = stage1_prompts(recipe, pairs, sys.executable, over_length_per_shape=5)

    assert document["anchor_check"]["passed"] is True, document["anchor_check"]["failures"][:2]
    assert document["anchor_check"]["checked"] >= 20
    render_check = document["render_check"]
    assert render_check["status"] == "run"
    assert render_check["passed"] is True, render_check["failures"][:2]
    assert render_check["rows"] == 2 * len(PAIRS)  # every row rendered under both declared shapes
    assert document["template_render_check"] is None  # no served chat template to prove
    assert document["engine_tokenize_check"]["status"] == "not_run"
    assert document["passed"] is True
    # The budget reserves the fixed frame per shape: ids("- ") = 2 tokens + the appended anchor for
    # documents; the appended anchor alone for queries (the product's own measurement).
    facts = stage1_facts(recipe, PAIRS, tokenizer_of(recipe), 5)
    assert facts["per_shape"]["document"]["overhead"] == 3
    assert facts["per_shape"]["query"]["overhead"] == 1


@pytest.mark.network
@pytest.mark.parametrize("variant_id", VARIANT_IDS)
def test_fitted_render_token_ids_match_the_paper_string(tmp_path: Path, variant_id: str) -> None:
    """The declared shapes render to the paper's token ids (the CPU stand-in for /tokenize, R29).

    For in-budget rows the served render must be the paper string byte-identically -- the query as
    it is, the document as the ONE string "- " + text (never separately tokenised ids) -- and its
    ids, read as the engine reads them, must equal the ids of that string and end with the appended
    anchor.
    """
    recipe = _local_recipe(_recipe(variant_id), _tokenizer_file(tmp_path, variant_id))
    tokenizer = tokenizer_of(recipe)
    for row in PAIRS[:4]:
        query_text = served_texts(recipe, [row["query"]], "query")[0]
        assert query_text == row["query"]
        assert tokenizer.ids(query_text, add_special_tokens=True) == tokenizer.ids(
            row["query"], add_special_tokens=True
        )
        assert tokenizer.ids(query_text, add_special_tokens=True)[-1] == APPENDED_ANCHOR_ID

        document = row["documents"][0]
        doc_text = served_texts(recipe, [document], "document")[0]
        assert doc_text == DOCUMENT_PREFIX + document  # the paper string, byte-identical
        ids = tokenizer.ids(doc_text, add_special_tokens=True)
        assert ids == tokenizer.ids(DOCUMENT_PREFIX + document, add_special_tokens=True)
        assert ids[-1] == APPENDED_ANCHOR_ID


@pytest.mark.network
@pytest.mark.parametrize("variant_id", VARIANT_IDS)
def test_over_length_inputs_keep_every_anchor(tmp_path: Path, variant_id: str) -> None:
    """An over-length input is cut in the content span only: prefix and appended anchor survive.

    The harness's anchor audit already runs on every sampled render; this pins the two facts it
    cannot express for a head-prefix template: the head prefix is present (the audit's anchor edge
    is the tail) and the whole input stays within the budget the anchors reserved.
    """
    recipe = _local_recipe(_recipe(variant_id), _tokenizer_file(tmp_path, variant_id))
    tokenizer = tokenizer_of(recipe)
    unit = "the diesel locomotive hauled freight through the alpine tunnel and arrived late in the evening "
    long_document = unit * 500  # 500 repetitions tokenise well over the 8192-token budget
    long_query = unit * 500

    # The sample must actually overflow: the whole point is the cut, so pin it before asserting the anchors.
    raw_document_ids = tokenizer.ids(DOCUMENT_PREFIX + long_document, add_special_tokens=True)
    raw_query_ids = tokenizer.ids(long_query, add_special_tokens=True)
    assert len(raw_document_ids) > MAX_TOKENS, f"the sample document is under the budget ({len(raw_document_ids)})"
    assert len(raw_query_ids) > MAX_TOKENS, f"the sample never exceeds the budget ({len(raw_query_ids)})"

    doc_text = served_texts(recipe, [long_document], "document")[0]
    doc_ids = tokenizer.ids(doc_text, add_special_tokens=True)
    assert doc_text.startswith(DOCUMENT_PREFIX)  # the head prefix survived the cut
    assert doc_ids[-1] == APPENDED_ANCHOR_ID  # the pooled anchor survived the cut
    assert len(doc_ids) <= MAX_TOKENS < len(raw_document_ids)  # the render actually shrank to fit

    query_text = served_texts(recipe, [long_query], "query")[0]
    query_ids = tokenizer.ids(query_text, add_special_tokens=True)
    assert not query_text.startswith(DOCUMENT_PREFIX)  # queries carry no prefix
    assert query_ids[-1] == APPENDED_ANCHOR_ID
    assert len(query_ids) <= MAX_TOKENS < len(raw_query_ids)


# -- mutations: the anchor declaration is load-bearing ------------------------------------------


@pytest.mark.network
def test_mutation_declaring_the_wrong_anchor_position_reddens_the_anchor_check(tmp_path: Path) -> None:
    """A wrong anchor declaration turns the anchor check red.

    The appended end-of-text token is the anchor (declared by add_special_tokens: true on the
    content-final shapes). Declaring the head instead makes the query shape's declared edge
    unsatisfiable (it has no fixed head segment at all), so the audit reds on every render of it.
    """
    variant_id = "octen-embedding-8b"
    recipe = _local_recipe(_recipe(variant_id), _tokenizer_file(tmp_path, variant_id))
    template = client_template(recipe)
    assert template is not None
    mutated_template = {**recipe.client["template"], "anchor": "first"}  # the client block is plain data
    client = {**recipe.client, "template": mutated_template}
    mutated = recipe.model_copy(update={"client": client})

    pairs = _write_pairs(tmp_path)
    healthy = stage1_prompts(recipe, pairs, sys.executable, over_length_per_shape=2)
    assert healthy["anchor_check"]["passed"] is True

    mutated_document = stage1_prompts(mutated, pairs, sys.executable, over_length_per_shape=2)
    assert mutated_document["anchor_check"]["passed"] is False
    # The renders themselves are unchanged (the client ships the same texts): only the declared
    # anchor position is wrong.
    healthy_facts = stage1_facts(recipe, PAIRS, tokenizer_of(recipe), 2)
    mutated_facts = stage1_facts(mutated, PAIRS, tokenizer_of(mutated), 2)
    assert mutated_facts["per_shape"]["document"]["texts"] == healthy_facts["per_shape"]["document"]["texts"]
    assert mutated_facts["per_shape"]["query"]["texts"] == healthy_facts["per_shape"]["query"]["texts"]
    # The query shape has no fixed head segment: its declared edge cannot hold.
    shapes = {failure["shape"] for failure in mutated_document["anchor_check"]["failures"]}
    assert "query" in shapes


def test_mutation_without_the_appended_anchor_declaration_is_refused() -> None:
    """A recipe that stops declaring the post-processor's appended anchor is refused at load.

    With ``add_special_tokens: false`` and a shape that ends on its content span, an
    ``anchor: last`` template has no tail anchor at all: the product's validator refuses it rather
    than let the budget drop the token the model pools.
    """
    from rcp_ndcg.data.templates import TemplateSpec

    template = client_template(_recipe("octen-embedding-8b"))
    assert template is not None
    mutated = {
        "query": [{"content": "query"}],  # ends on content, declares no post-processor tokens
        "document": [{"fixed": DOCUMENT_PREFIX}, {"content": "document"}],
        "anchor": "last",
        "add_special_tokens": False,
    }
    with pytest.raises(ValueError, match="anchor: last"):
        TemplateSpec.model_validate(mutated)
    assert template.adds_special_tokens("document") is True  # and the shipped template declares it


# -- the reference module: constants anchored to the paper's code, render mode is the paper string


def _resolved_recipe_file(directory: Path, variant_id: str) -> Path:
    """The resolved recipe JSON, exactly what the harness's ``run_reference`` passes as ``--recipe``."""
    recipe = _recipe(variant_id)
    path = directory / f"{variant_id}.reference.recipe.json"
    path.write_text(json.dumps(recipe.model_dump(mode="json"), sort_keys=True), encoding="utf-8")
    return path


def _reference_module() -> Any:
    spec = importlib.util.spec_from_file_location("octen_reference", RECIPE_DIR / "reference.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _paper_module() -> Any:
    """The paper path's own octen encoder (experiments/paper/rerankers/reference/octen.py)."""
    paper_path = Path(__file__).resolve().parents[3] / "experiments" / "paper" / "rerankers" / "reference" / "octen.py"
    spec = importlib.util.spec_from_file_location("octen_paper_reference", paper_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reference_imports_no_torch_transformers_or_numpy_at_module_level() -> None:
    """Importing the reference module pulls in none of the heavy stacks (checked in a fresh python).

    The module runs as a subprocess in its own environment (requirements-reference.txt: torch and
    transformers), and its render mode is the harness's stage-1 side; a module-level numpy import
    would tie even the render mode to a numpy-capable python. torch and transformers are imported
    lazily inside the embed/score paths only.
    """
    reference_path = str(RECIPE_DIR / "reference.py")
    probe = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('octen_reference', {reference_path!r})\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "heavy = [name for name in ('torch', 'transformers', 'numpy') if name in sys.modules]\n"
        "print(','.join(heavy))\n"
    )
    completed = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stderr[-500:]
    imported = completed.stdout.strip().split(",") if completed.stdout.strip() else []
    assert imported == [], f"the reference module imported at import time: {imported}"


def test_load_takes_the_checkpoint_from_the_resolved_recipe(monkeypatch: pytest.MonkeyPatch) -> None:
    """The checkpoint identity comes from the resolved recipe (``--recipe``), not the module constants.

    A variant row pointing at another checkpoint must load THAT checkpoint; the stubbed heavy modules
    keep the test offline (the harness process imports no torch).
    """
    import sys
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
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoModel=types.SimpleNamespace(from_pretrained=fake_from_pretrained),
            AutoTokenizer=types.SimpleNamespace(from_pretrained=fake_from_pretrained),
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(bfloat16="bfloat16"))
    reference = _reference_module()
    reference.load("cpu", model="example-org/other-checkpoint", revision="0" * 40)
    assert [call["name"] for call in calls] == ["example-org/other-checkpoint", "example-org/other-checkpoint"]
    assert all(call["revision"] == "0" * 40 for call in calls)


def test_reference_constants_equal_the_paper_code() -> None:
    """The reference's constants are the paper path's (experiments/paper/rerankers/reference/octen.py),
    not the model card's; MODEL/REVISION name the paper's 8B checkpoint (the standalone defaults)."""
    from rcp_ndcg.retrieval import l2_normalize

    reference = _reference_module()
    paper = yaml.safe_load(
        (Path(__file__).resolve().parents[3] / "experiments/paper/retrieval/octen.yaml").read_text(encoding="utf-8")
    )["encoder"]
    paper_module = _paper_module()
    assert reference.MAX_LENGTH == paper_module.MAX_LENGTH == MAX_TOKENS
    # The document frame lives in the recipe's declared template (the paper config carries the
    # `recipe:` pointer since the mapping form, decision 17); the reference's constant is that frame.
    template = client_template(_recipe("octen-embedding-8b"))
    assert template is not None
    document_segments = template.segments("document")
    assert document_segments[0].fixed == reference.DOCUMENT_PREFIX == DOCUMENT_PREFIX
    assert reference.PAD_SIDE == "left" and reference.DTYPE == "bfloat16"
    assert (
        reference.BATCH_SIZE
        == paper["batch_size"]
        == paper_module.TorchDenseEncoder.__init__.__kwdefaults__["batch_size"]
    )
    assert reference.REVISION == VARIANTS["octen-embedding-8b"]["revision"]
    assert reference.MODEL == VARIANTS["octen-embedding-8b"]["repo"]
    probe = np.array([[3.0, 4.0], [0.0, 0.0]])
    assert reference._l2_normalize(probe).tolist() == l2_normalize(probe).tolist()  # noqa: SLF001
    with pytest.raises(ValueError, match="no instruction"):
        reference.render("x", "query", instruction="task text")
    with pytest.raises(ValueError, match="unknown role"):
        reference.render_prompt("x", "sentence")


@pytest.mark.parametrize("variant_id", VARIANT_IDS)
def test_reference_render_mode_emits_the_paper_strings(tmp_path: Path, variant_id: str) -> None:
    """The reference's render mode (a subprocess free of torch and transformers) emits the paper's strings."""
    reference = RECIPE_DIR / "reference.py"
    pairs = _write_pairs(tmp_path)
    out = tmp_path / f"{variant_id}.reference.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(reference),
            "--mode",
            "render",
            "--pairs",
            str(pairs),
            "--out",
            str(out),
            "--tokenizer",
            str(tmp_path),  # the render mode needs no tokenizer files
            "--recipe",
            str(_resolved_recipe_file(tmp_path, variant_id)),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr[-500:]
    rows = json.loads(out.read_text(encoding="utf-8"))["rows"]
    assert len(rows) == 2 * len(PAIRS)
    by_key = {(row["index"], row["shape"]): row["text"] for row in rows}
    for index, row in enumerate(PAIRS):
        assert by_key[(index, "query")] == row["query"]
        assert by_key[(index, "document")] == DOCUMENT_PREFIX + row["documents"][0]


# ---------------------------------------------------------------------------
# The declared contract: every serve, client and reference field pinned.
# ---------------------------------------------------------------------------

# Two mutants per recipe against the contract pin above: each drift must fail, naming the field.
MUTANTS: list[tuple[str, tuple[str, ...], object, str]] = [
    ("licence drifts to MIT", ("licence",), "MIT", "recipe.licence"),
    ("serve.max_model_len drifts to 16384", ("serve", "max_model_len"), 16384, "max_model_len"),
]


def _mutated_recipe(tmp_path: Path, variant_id: str, path: tuple[str, ...], value: object) -> object:
    """The family directory copied into ``tmp_path`` with one YAML field set to ``value``."""
    import shutil

    import yaml

    target = tmp_path / RECIPE_DIR.name
    shutil.copytree(RECIPE_DIR, target)
    yaml_path = target / "family.yaml"
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    node = data
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return resolve_recipe(variant_id, root=tmp_path)


@pytest.mark.parametrize("variant_id", VARIANT_IDS)
@pytest.mark.parametrize(("label", "path", "value", "needle"), MUTANTS, ids=[m[0] for m in MUTANTS])
def test_two_contract_mutants_are_red(
    label: str, path: tuple[str, ...], value: object, needle: str, tmp_path: Path, variant_id: str
) -> None:
    """A drifted field fails the contract assertion naming it (two mutants per family, per variant)."""
    _assert_contract(_recipe(variant_id), variant_id)  # the pinned recipe itself is green
    with pytest.raises(AssertionError) as caught:
        _assert_contract(_mutated_recipe(tmp_path, variant_id, path, value), variant_id)
    assert needle in str(caught.value), f"{label}: the failure must name {needle}: {caught.value}"


@pytest.mark.parametrize("variant_id", VARIANT_IDS)
def test_notes_state_the_merged_budget_wiring_and_the_query_cap_check(variant_id: str) -> None:
    """The notes read the merged product: the budget is fitted on the
    wire -- no stale pre-clients-merge status sentence -- and the query_max_tokens check found
    no separate referent cap."""
    notes = _recipe(variant_id).notes
    assert "fitted to the declared budget on the wire" in notes
    assert "refuse a\nbudget" not in notes  # the stale pre-wiring sentence (labels: the hygiene test)
    assert "no separate query cap exists in the referent" in notes
