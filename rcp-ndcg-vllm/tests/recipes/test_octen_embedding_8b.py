"""The ``octen-embedding-8b`` recipe: it validates, and stage 1 passes on CPU.

Stage 1 needs only the checkpoint's tokenizer files (no weights, no GPU): they are downloaded into
the test's ``tmp_path`` -- or taken from ``$RCP_NDCG_OCTEN_TOKENIZER_DIR`` when pre-seeded -- and the
tests skip with a clear reason when offline and nothing is cached. The recipe's own
``client.tokenizer`` stays the pinned Hub spec; the stage-1 runs point a copy of the recipe at the
local tokenizer directory.

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
- mutations: dropping the template's trailing anchor segment reddens the anchor check, and a recipe
  that declares no appended anchor at all is refused by the product's template validator;
- the reference module's constants equal the paper code's (``hf_dense.MAX_LENGTH``, the
  ``"- "`` prefix of ``octen.yaml``/``torch_dense``), and its render mode emits the paper's strings.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from rcp_ndcg_vllm import client_config, load_recipe
from rcp_ndcg_vllm.equivalence import stage1_prompts
from rcp_ndcg_vllm.equivalence.fitting import budget_of, tokenizer_of

from rcp_ndcg.inference.config import EmbeddingEndpoint

RECIPE_DIR = Path(__file__).resolve().parents[2] / "recipes" / "octen-embedding-8b"
REVISION = "5adcfa292e712091dfc30f0e97f0b2282e6cc66c"
MODEL = "Octen/Octen-Embedding-8B"
MAX_TOKENS = 8192
#: The id the tokenizer's post-processor appends to every sequence (added-token name: endoftext) --
#: the last token the model pools, and the anchor the budget reserves.
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


def _recipe() -> Any:
    """The real recipe, loaded and validated through the product's endpoint config."""
    return load_recipe(RECIPE_DIR)


def _tokenizer_dir(tmp_path: Path) -> Path:
    """The checkpoint's tokenizer files, locally: the env override, else a download into ``tmp_path``.

    Only tokenizer files (about 16 MB), never weights. Skips with a clear reason when offline and
    no pre-seeded copy is available (``$RCP_NDCG_OCTEN_TOKENIZER_DIR`` names one).
    """
    seeded = os.environ.get("RCP_NDCG_OCTEN_TOKENIZER_DIR")
    if seeded and (Path(seeded) / "tokenizer.json").is_file():
        return Path(seeded)
    target = tmp_path / "octen-tokenizer"
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            MODEL,
            revision=REVISION,
            local_dir=str(target),
            allow_patterns=[
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.json",
                "merges.txt",
                "special_tokens_map.json",
                "added_tokens.json",
            ],
        )
    except Exception as error:  # noqa: BLE001 -- any fetch failure means offline: skip, never fail
        pytest.skip(
            "offline and no pre-seeded tokenizer: the Octen tokenizer files are unavailable "
            f"({type(error).__name__}: {str(error)[:200]}); stage 1 needs the recipe's own "
            "tokenizer files only, never the weights"
        )
    return target


def _local_recipe(recipe: Any, tokenizer_dir: Path) -> Any:
    """The recipe with its tokenizer pointed at the local files (the Hub spec stays in recipe.yaml)."""
    client = recipe.client.model_copy(update={"tokenizer": str(tokenizer_dir)})
    return recipe.model_copy(update={"client": client})


def _write_pairs(tmp_path: Path) -> Path:
    path = tmp_path / "pairs.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in PAIRS), encoding="utf-8")
    return path


# -- the recipe validates ----------------------------------------------------------------------


def test_recipe_validates_against_the_product_endpoints() -> None:
    """The recipe loads; its client block is the product's EmbeddingEndpoint with the paper's facts."""
    recipe = _recipe()
    assert recipe.id == "octen-embedding-8b"
    assert recipe.model == MODEL
    assert recipe.revision == REVISION
    assert isinstance(recipe.client, EmbeddingEndpoint)
    assert recipe.role == "embed" and recipe.input == ["text"]
    assert recipe.serve.runner == "pooling"
    assert recipe.serve.chat_template is None  # the ChatML template would change every prompt
    assert recipe.serve.max_model_len == MAX_TOKENS
    assert recipe.serve.dtype == "bfloat16"
    assert recipe.serve.hf_overrides == {} and recipe.serve.pooler_config == {}
    client = recipe.client
    assert client.tokenizer == f"{MODEL}@{REVISION}"
    assert client.max_tokens == MAX_TOKENS  # the whole input sequence, anchors included
    assert client.on_overflow == "cut" and client.empty_doc == "send" and client.normalize is True
    assert client.query_prompt == "" and client.doc_prompt == ""  # the prefix lives in the template only
    template = client.template
    assert template is not None
    assert template.shapes() == ("query", "document")
    assert template.anchor == "last"
    assert template.adds_special_tokens("query") and template.adds_special_tokens("document")
    assert [segment.content for segment in template.query] == ["query"]
    assert [segment.fixed for segment in template.document] == [DOCUMENT_PREFIX, None, ""]
    assert recipe.reference.known_deviations == []  # the paper's cut keeps the appended anchor
    assert recipe.status.state == "unverified"
    config = client_config(recipe, base_url="http://127.0.0.1:8100/v1")
    assert EmbeddingEndpoint(**config).model == recipe.id


# -- stage 1 on CPU -----------------------------------------------------------------------------


def test_stage1_on_cpu_passes_token_equality_and_the_anchor_check(tmp_path: Path) -> None:
    """Stage 1 (CPU): fit's renders match the reference's, and the anchor audit passes.

    At least 20 sampled inputs with at least 5 over-length ones: the pairs file's 16 rows plus 5
    over-length samples per declared shape (query and document). The reference runs as a subprocess
    (its render mode needs no torch and no transformers); the engine /tokenize check is reported not_run without an
    engine, never passed.
    """
    recipe = _local_recipe(_recipe(), _tokenizer_dir(tmp_path))
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
    # documents; the appended anchor alone for queries.
    assert document["fit"]["document"]["overhead"] == 3
    assert document["fit"]["query"]["overhead"] == 1


def test_fitted_render_token_ids_match_the_paper_string(tmp_path: Path) -> None:
    """The declared shapes render to the paper's token ids (the CPU stand-in for /tokenize, R29).

    For in-budget rows the served render must be the paper string byte-identically -- the query as
    it is, the document as the ONE string "- " + text (never separately tokenised ids) -- and its
    ids, read as the engine reads them, must equal the ids of that string and end with the appended
    anchor.
    """
    from rcp_ndcg.data.preprocess import fit

    recipe = _local_recipe(_recipe(), _tokenizer_dir(tmp_path))
    tokenizer = tokenizer_of(recipe)
    budget = budget_of(recipe).model_copy(update={"tokenizer": tokenizer.name})
    for row in PAIRS[:4]:
        query_result = fit([row["query"]], "query", budget, tokenizer, ids=["0"])
        assert query_result.texts[0] == row["query"]
        assert tokenizer.ids(query_result.texts[0], add_special_tokens=True) == tokenizer.ids(
            row["query"], add_special_tokens=True
        )
        assert tokenizer.ids(query_result.texts[0], add_special_tokens=True)[-1] == APPENDED_ANCHOR_ID

        document = row["documents"][0]
        doc_result = fit([document], "document", budget, tokenizer, ids=["0"])
        assert doc_result.texts[0] == DOCUMENT_PREFIX + document  # the paper string, byte-identical
        ids = tokenizer.ids(doc_result.texts[0], add_special_tokens=True)
        assert ids == tokenizer.ids(DOCUMENT_PREFIX + document, add_special_tokens=True)
        assert ids[-1] == APPENDED_ANCHOR_ID


def test_over_length_inputs_keep_every_anchor(tmp_path: Path) -> None:
    """An over-length input is cut in the content span only: prefix and appended anchor survive.

    The harness's anchor audit already runs on every sampled render; this pins the two facts it
    cannot express for a head-prefix template: the head prefix is present (the audit's anchor edge
    is the tail) and the whole input stays within the budget the anchors reserved.
    """
    from rcp_ndcg.data.preprocess import fit

    recipe = _local_recipe(_recipe(), _tokenizer_dir(tmp_path))
    tokenizer = tokenizer_of(recipe)
    budget = budget_of(recipe).model_copy(update={"tokenizer": tokenizer.name})
    unit = "the diesel locomotive hauled freight through the alpine tunnel and arrived late in the evening "
    long_document = unit * 500  # 500 repetitions tokenise well over the 8192-token budget
    long_query = unit * 500

    # The sample must actually overflow: the whole point is the cut, so pin it before asserting the anchors.
    raw_document_ids = tokenizer.ids(DOCUMENT_PREFIX + long_document, add_special_tokens=True)
    raw_query_ids = tokenizer.ids(long_query, add_special_tokens=True)
    assert len(raw_document_ids) > MAX_TOKENS, f"the sample document is under the budget ({len(raw_document_ids)})"
    assert len(raw_query_ids) > MAX_TOKENS, f"the sample never exceeds the budget ({len(raw_query_ids)})"

    doc_result = fit([long_document], "document", budget, tokenizer, ids=["0"])
    doc_ids = tokenizer.ids(doc_result.texts[0], add_special_tokens=True)
    assert doc_result.texts[0].startswith(DOCUMENT_PREFIX)  # the head prefix survived the cut
    assert doc_ids[-1] == APPENDED_ANCHOR_ID  # the pooled anchor survived the cut
    assert len(doc_ids) <= MAX_TOKENS < len(raw_document_ids)  # the render actually shrank to fit

    query_result = fit([long_query], "query", budget, tokenizer, ids=["0"])
    query_ids = tokenizer.ids(query_result.texts[0], add_special_tokens=True)
    assert not query_result.texts[0].startswith(DOCUMENT_PREFIX)  # queries carry no prefix
    assert query_ids[-1] == APPENDED_ANCHOR_ID
    assert len(query_ids) <= MAX_TOKENS < len(raw_query_ids)


# -- mutations: the anchor declaration is load-bearing ------------------------------------------


def test_mutation_drop_the_trailing_anchor_segment_reddens_the_anchor_check(tmp_path: Path) -> None:
    """Dropping the template's trailing anchor segment turns the anchor check red.

    The trailing empty fixed segment is what the audit reads as the tail anchor carrier (the edge
    is the last fixed segment's ids plus the post-processor block). Without it the audit derives
    the edge from the head prefix instead, which BPE merging makes unsatisfiable for every real
    render -- so every document render fails the audit.
    """
    recipe = _local_recipe(_recipe(), _tokenizer_dir(tmp_path))
    template = recipe.client.template
    assert template is not None
    mutated_template = template.model_copy(update={"document": tuple(template.document)[:-1]})
    client = recipe.client.model_copy(update={"template": mutated_template})
    mutated = recipe.model_copy(update={"client": client})

    pairs = _write_pairs(tmp_path)
    healthy = stage1_prompts(recipe, pairs, sys.executable, over_length_per_shape=2)
    assert healthy["anchor_check"]["passed"] is True

    mutated_document = stage1_prompts(mutated, pairs, sys.executable, over_length_per_shape=2)
    assert mutated_document["anchor_check"]["passed"] is False
    assert mutated_document["fit"]["document"]["texts_head"] == healthy["fit"]["document"]["texts_head"]
    shapes = {failure["shape"] for failure in mutated_document["anchor_check"]["failures"]}
    assert shapes == {"document"}  # the query shape's declared anchor is unaffected


def test_mutation_without_the_appended_anchor_declaration_is_refused() -> None:
    """A recipe that stops declaring the post-processor's appended anchor is refused at load.

    With ``add_special_tokens: false`` and a shape that ends on its content span, an
    ``anchor: last`` template has no tail anchor at all: the product's validator refuses it rather
    than let the budget drop the token the model pools.
    """
    from rcp_ndcg.data.templates import TemplateSpec

    template = _recipe().client.template
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


def _reference_module() -> Any:
    spec = importlib.util.spec_from_file_location("octen_reference", RECIPE_DIR / "reference.py")
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


def test_reference_constants_equal_the_paper_code() -> None:
    """The reference's constants are the paper path's, not the model card's."""
    from rcp_ndcg.retrieval.encoder import l2_normalize
    from rcp_ndcg.retrieval.encoders.torch_dense import TorchDenseEncoder
    from rcp_ndcg.retrieval.hf_dense import MAX_LENGTH as paper_max_length

    reference = _reference_module()
    paper = yaml.safe_load(
        (Path(__file__).resolve().parents[4] / "experiments/paper/retrieval/octen.yaml").read_text(encoding="utf-8")
    )["encoder"]
    assert reference.MAX_LENGTH == paper_max_length == MAX_TOKENS
    assert reference.DOCUMENT_PREFIX == paper["doc_prompt"] == DOCUMENT_PREFIX
    assert reference.PAD_SIDE == "left" and reference.DTYPE == "bfloat16"
    assert reference.BATCH_SIZE == paper["batch_size"] == TorchDenseEncoder.__init__.__kwdefaults__["batch_size"]
    assert reference.REVISION == REVISION and reference.MODEL == MODEL
    probe = np.array([[3.0, 4.0], [0.0, 0.0]])
    assert reference._l2_normalize(probe).tolist() == l2_normalize(probe).tolist()  # noqa: SLF001
    with pytest.raises(ValueError, match="no instruction"):
        reference.render("x", "query", instruction="task text")
    with pytest.raises(ValueError, match="unknown role"):
        reference.render_prompt("x", "sentence")


def test_reference_render_mode_emits_the_paper_strings(tmp_path: Path) -> None:
    """The reference's render mode (a subprocess free of torch and transformers) emits the paper's strings."""
    reference = RECIPE_DIR / "reference.py"
    pairs = _write_pairs(tmp_path)
    out = tmp_path / "reference.json"
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
