"""The qwen3-reranker-4b recipe: schema, stage 1 on CPU with the real tokenizer, and the mutations.

The recipe validates through the product's endpoint config; stage 1 runs the product's ``fit``
with the model's Hub tokenizer (files only, downloaded into a scratch cache — skipped with a
clear reason when the Hub is unreachable, e.g. offline CI) and proves: the anchor-preserving
cut, token-id equality with the reference subprocess's render, and the served chat template
file rendering to the declared shapes' ids. The mutations show what stage 1 catches: a recipe
whose declared template drops the trailing anchor segment goes red on the anchor check, and
the served template file without its trailing newline (the stock vLLM example's defect) goes
red on the template check.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from rcp_ndcg_vllm import load_recipe

RECIPES = Path(__file__).resolve().parents[2] / "recipes"
RECIPE_DIR = RECIPES / "qwen3-reranker-4b"
REVISION = "22e683669bc0f0bd69640a1354a6d0aebcfeede5"

# The last fixed segment of the declared pair shape: the assistant suffix (the model's
# read-out anchor), as the paper's code appends it after the truncated pair.
SUFFIX_TOKENS = [151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271]


def _pairs(count: int) -> list[dict]:
    """In-budget pairs: short queries and documents, no instruction (the recipe drops them)."""
    return [
        {
            "query": f"capital of france sample {index}",
            "documents": [
                f"paris is the capital of france, document {index} about cities and rivers in europe {index}"
            ],
        }
        for index in range(count)
    ]


@pytest.fixture(scope="module")
def recipe():
    """The loaded recipe (its client block constructs the product's RerankEndpoint at load)."""
    return load_recipe(RECIPE_DIR)


@pytest.fixture(scope="module")
def qwen_tokenizer():
    """The recipe's Hub tokenizer, downloaded once into a scratch cache.

    ``RCP_VLLM_TOKENIZER_CACHE`` names the cache directory (the lane's scratch on a
    lane run); without it the tokenizer downloads into a pytest temp dir. Skips when
    the Hub is unreachable and the tokenizer is not cached — stage 1 needs the real
    tokenizer's ids, and there is no stand-in for them.
    """
    cache = os.environ.get("RCP_VLLM_TOKENIZER_CACHE")
    if cache:
        os.environ.setdefault("HF_HOME", cache)
    from rcp_ndcg.data.tokenizer import load_tokenizer

    try:
        tokenizer = load_tokenizer(f"Qwen/Qwen3-Reranker-4B@{REVISION}")
    except Exception as error:  # offline CI, or the Hub refused: a clear skip, never a fake pass
        pytest.skip(f"the Qwen/Qwen3-Reranker-4B tokenizer is unavailable (offline?): {error}")
    return tokenizer


def test_recipe_validates_against_the_product_schema(recipe) -> None:
    """The recipe loads; the client block is the product's RerankEndpoint with the paper's budgets."""
    from rcp_ndcg.inference.config import RerankEndpoint

    assert recipe.id == "qwen3-reranker-4b"
    assert recipe.model == "Qwen/Qwen3-Reranker-4B"
    assert recipe.revision == REVISION
    assert isinstance(recipe.client, RerankEndpoint)
    assert recipe.role == "rerank" and recipe.scoring == "pointwise"
    assert recipe.client.tokenizer == f"Qwen/Qwen3-Reranker-4B@{REVISION}"
    assert recipe.client.max_tokens == 8192  # MAX_SEQ_LENGTH
    assert recipe.client.query_max_tokens == 4096  # MAX_QUERY_LENGTH
    assert recipe.client.on_overflow == "cut"
    assert recipe.client.instruction == "none"
    assert recipe.client.use_activation is True  # probability-scale head
    assert recipe.client.template is not None and recipe.client.template.anchor == "last"
    # One over-cap policy family-wide (the operator's 09:2x decision): the reference keeps every
    # anchor and cuts over-cap content the paper's way (the joint longest_first pair cut), never
    # copying the client's cut -- so over-cap rows are reported, not gated. The old
    # anchor_drop_over_cap label was wrong: this reference never drops an anchor.
    assert recipe.reference.known_deviations == ["over_cap_cut_differs"]
    assert recipe.serve.chat_template == "template.jinja"
    assert recipe.serve.hf_overrides["architectures"] == ["Qwen3ForSequenceClassification"]
    assert recipe.serve.hf_overrides["classifier_from_token"] == ["no", "yes"]
    assert recipe.serve.hf_overrides["is_original_qwen3_reranker"] is True
    assert recipe.serve.max_model_len >= recipe.client.max_tokens
    assert recipe.status.state == "unverified"
    assert recipe.sources


def test_serve_argv_names_the_shipped_template(recipe) -> None:
    """The template file ships and the argv names it (REVIEW-LOG R10: without it the engine only
    warns, then concatenates the prompts); the served model name is the recipe id."""
    from rcp_ndcg_vllm.recipe import serve_argv

    argv = serve_argv(recipe, port=8100, served_model_name=recipe.id)
    assert "--chat-template" in argv
    assert str(RECIPE_DIR / "template.jinja") in argv
    assert argv[argv.index("--served-model-name") + 1] == "qwen3-reranker-4b"
    assert (RECIPE_DIR / "template.jinja").is_file()


def test_stage1_on_cpu_with_the_real_tokenizer(tmp_path: Path, recipe, qwen_tokenizer) -> None:
    """Stage 1 on CPU: 20 sampled pairs, 5 of them over-length, all green.

    The product's fit renders every sampled prompt; the anchor audit asserts the 9-token
    assistant suffix (the model's anchor) survives every cut; the reference subprocess's span
    render (the paper's anchor-preserving joint pair cut at raw character offsets, never the
    client's cut) equals the client's shipped spans byte for byte on every under-cap row; the 5
    over-length pairs are reported non-gating under the over_cap_cut_differs declaration. The
    served template file renders to the declared shape's text. Without an engine, the /tokenize
    check is reported not_run.
    """
    from rcp_ndcg_vllm.equivalence import stage1_prompts

    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text("".join(json.dumps(row) + "\n" for row in _pairs(15)), encoding="utf-8")
    document = stage1_prompts(recipe, pairs, sys.executable, over_length_per_shape=5)
    assert document["sampled"] == 20  # 15 in-budget pairs + 5 over-length samples
    assert document["anchor_check"]["passed"] is True, document["anchor_check"]["failures"][:2]
    assert document["anchor_check"]["checked"] >= 20
    assert document["render_check"]["passed"] is True, document["render_check"]["failures"][:2]
    assert document["render_check"]["rows"] == 15
    assert document["template_render_check"]["passed"] is True, document["template_render_check"]["failures"][:2]
    # R29 hygiene: without an engine the check is not_run, never passed.
    assert document["engine_tokenize_check"]["status"] == "not_run"
    assert document["engine_tokenize_check"]["passed"] is None


def test_over_length_pairs_keep_the_suffix_anchor_ids(tmp_path: Path, recipe, qwen_tokenizer) -> None:
    """The anchor check's substance, asserted directly: an over-cap pair's rendered ids still end
    with the 9 suffix tokens (im_end, newline, im_start, assistant, newline, think-open, blank,
    think-close, blank), and the render fits the 8192 budget exactly."""
    from rcp_ndcg.data.preprocess import TextBudget, fit

    client = recipe.client
    budget = TextBudget(
        tokenizer=qwen_tokenizer.name,
        max_tokens=client.max_tokens,
        query_max_tokens=client.query_max_tokens,
        template=client.template,
        on_overflow=client.on_overflow,
    )
    long_query = ("capital of france part 0 " * 1200).strip()
    long_document = ("the rivers and bridges of paris part 0 " * 1200).strip()
    result = fit([(long_query, long_document)], "pair", budget, qwen_tokenizer)
    ids = qwen_tokenizer.ids(result.texts[0], add_special_tokens=True)
    assert len(ids) <= client.max_tokens
    assert ids[-9:] == SUFFIX_TOKENS
    # and the prefix anchor: the render opens with the system-turn special
    assert qwen_tokenizer.ids(result.texts[0])[0] == 151644


def test_score_mode_setup_parses_and_reaches_the_model_load(recipe) -> None:
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
        @staticmethod
        def from_pretrained(*args, **kwargs):
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
        monkey.setitem(sys.modules, "torch", torch_stub)
        monkey.setitem(sys.modules, "transformers", transformers_stub)
        spec = importlib.util.spec_from_file_location("qwen3_reranker_4b_reference", RECIPE_DIR / "reference.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert module.score_rows([], f"Qwen/Qwen3-Reranker-4B@{REVISION}", "cpu") == []
    finally:
        monkey.undo()


def test_mutation_dropping_the_trailing_anchor_segment_reddens_the_template_check(
    tmp_path: Path, recipe, qwen_tokenizer
) -> None:
    """Mutation, on the wire's own contract: the declared template loses its trailing anchor
    segment (the assistant suffix) and the file-vs-declaration check must go red.

    The rerank wire carries the cut content spans (the frame is the engine's own template), so
    stage 1's anchor audit audits the settled query and the document spans and does not move on a
    frame change; the frame contract is pinned by ``template_render_check``, which this mutation
    turns red (the ids no longer end with the declared anchor), and the schema itself refuses the
    un-pinned shape.
    """
    from rcp_ndcg.data.templates import TemplateSpec

    template = recipe.client.template
    assert template is not None
    assert len(template.pair) == 5 and "fixed" in template.pair[-1]

    # the real frame ends with the 9-token assistant suffix (the scored anchor), via the product's
    # render
    golden = template.render(
        "pair", qwen_tokenizer, query="capital of france", document="paris is the capital of France."
    )
    suffix_ids = qwen_tokenizer.ids(golden, add_special_tokens=True)[-len(SUFFIX_TOKENS) :]
    assert suffix_ids == SUFFIX_TOKENS

    # the mutation: drop the trailing anchor segment, keep the shape loadable by declaring the
    # tokenizer's post-processor as the anchor (which adds nothing for Qwen: the anchor is gone)
    anchorless = TemplateSpec(
        pair=tuple(segment for segment in template.pair[:-1]), anchor="last", add_special_tokens={"pair": True}
    )
    mutated = recipe.model_copy(update={"client": recipe.client.model_copy(update={"template": anchorless})})
    assert mutated.client.template is not None
    rendered = mutated.client.template.render(
        "pair", qwen_tokenizer, query="capital of france", document="paris is the capital of France."
    )
    no_tail_ids = qwen_tokenizer.ids(rendered, add_special_tokens=True)
    assert no_tail_ids[-len(SUFFIX_TOKENS) :] != SUFFIX_TOKENS, "the mutation must really lose the anchor"

    from rcp_ndcg_vllm.equivalence import stage1_prompts

    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text("".join(json.dumps(row) + "\n" for row in _pairs(3)), encoding="utf-8")
    document = stage1_prompts(mutated, pairs, sys.executable, over_length_per_shape=2)
    assert document["template_render_check"]["passed"] is False  # the file still emits the dropped suffix
    assert document["anchor_check"]["passed"] is True  # the wire's spans are unchanged by the frame drop

    # and the schema itself refuses an 'anchor: last' shape with neither a fixed tail nor the
    # post-processor declaration
    with pytest.raises(ValueError, match="anchor: last"):
        TemplateSpec(pair=tuple(segment for segment in template.pair[:-1]), anchor="last", add_special_tokens=False)


def test_non_nfc_rows_compare_byte_identical_spans(tmp_path: Path, recipe, qwen_tokenizer) -> None:
    """NFD (non-NFC) input: the render comparison byte-equals the RAW characters.

    ``tok.decode(tok.encode(x))`` is not the identity for this checkpoint (its normalizer maps
    non-NFC text to NFC, keeping the ids equal but not the characters): the reference must cut at
    raw character offsets (verbatim prefixes), never decode the kept ids back. On decomposed-accent
    rows the shipped spans and the reference's must match byte for byte — this goes red the moment
    either side decodes instead of cutting (finding 4: ``bytes_equal=False, ids_equal=True``).
    """
    decomposed = "cafe" + chr(101) + chr(769)  # e + combining acute: NFD, never NFC
    rows = [
        {"query": f"what about {decomposed}?", "documents": [f"The {decomposed} is served over the river."]},
        {
            "query": f"menu of the {decomposed} {decomposed} house",
            "documents": [f"{decomposed} soup and {decomposed} pie, with notes on the {decomposed} " * 12],
        },
    ]
    from rcp_ndcg_vllm.equivalence import stage1_prompts

    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    document = stage1_prompts(recipe, pairs, sys.executable, over_length_per_shape=1)
    assert document["render_check"]["status"] == "run" and document["render_check"]["rows"] == 2
    assert document["render_check"]["passed"] is True, document["render_check"]["failures"][:1]


def test_the_client_settles_an_over_share_query_once_at_its_share(recipe, qwen_tokenizer) -> None:
    """The settle rule, pinned on the wire: an over-share query is NOT sent whole.

    The rerank client settles the shared query once per call and ships it at its declared
    ``query_max_tokens`` (4096) whenever the query exceeds it — the shipped span is a verbatim
    prefix of the raw query (a raw character cut, never a decode round trip), identical for every
    document of the request.
    """
    from ._served import served_pair

    long_query = "over share query filler token " * 1400
    assert qwen_tokenizer.count(long_query) > 4096
    spans = served_pair(recipe, long_query, ["a short document.", "another short document."])
    settled = spans["query"]
    assert settled != long_query, "an over-share query must be cut at its share, never sent whole"
    assert long_query.startswith(settled), "the settled span must be a verbatim prefix of the raw query"
    assert 4080 <= qwen_tokenizer.count(settled) <= 4096


def test_the_short_query_ships_whole(recipe) -> None:
    """The other half of the settle rule: an under-share query is sent uncut (nothing settles it)."""
    from ._served import served_pair

    query = "capital of france"
    spans = served_pair(recipe, query, ["paris is the capital of france."])
    assert spans["query"] == query


def test_mutation_template_file_without_its_trailing_newline_turns_the_template_check_red(
    tmp_path: Path, recipe, qwen_tokenizer
) -> None:
    """The served template file without the patch's trailing newline (the stock vLLM example
    file's defect, 685 bytes) renders one newline short of the declared shape: the token-id
    equality with the declared template fails, which is the paper-exactness check the recipe
    rests on."""
    mutated_dir = tmp_path / "qwen3-reranker-4b"
    mutated_dir.mkdir()
    for name in ("recipe.yaml", "reference.py", "template.jinja"):
        (mutated_dir / name).write_bytes((RECIPE_DIR / name).read_bytes())
    template = (mutated_dir / "template.jinja").read_text(encoding="utf-8")
    # Strips every trailing newline, one more than the stock-file defect: the renderer then
    # strips one more than the patch provides, and the render is a newline short of the paper
    # prompt's double-newline tail either way.
    (mutated_dir / "template.jinja").write_text(template.rstrip("\n"), encoding="utf-8")

    from rcp_ndcg_vllm.equivalence import stage1_prompts

    mutated = load_recipe(mutated_dir)
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text("".join(json.dumps(row) + "\n" for row in _pairs(3)), encoding="utf-8")
    document = stage1_prompts(mutated, pairs, sys.executable, over_length_per_shape=1)
    assert document["template_render_check"]["passed"] is False
    assert document["passed"] is False
