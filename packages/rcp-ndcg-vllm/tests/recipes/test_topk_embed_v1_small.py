"""The topk-embed-v1-small recipe: it validates, stage 1 passes on CPU (tokenizer files only), the
image wrapper and the keep-mask asymmetry are pinned, and the anchor checks have teeth.

The hub-dependent tests share one module fixture that fetches the pinned tokenizer (and the
checkpoint's small config files) once; offline or hub-less environments skip them with the reason.
Nothing downloads weights: stage 1 on CPU is a tokenizer-and-config check, never a model run.
"""

from __future__ import annotations

import json
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml
from rcp_ndcg_vllm import load_recipe
from rcp_ndcg_vllm.equivalence import stage1_prompts
from rcp_ndcg_vllm.equivalence.fitting import fit_rows, load_pairs
from rcp_ndcg_vllm.recipe import serve_argv

from rcp_ndcg.data.tokenizer import load_tokenizer

RECIPE_DIR = Path(__file__).resolve().parents[2] / "recipes" / "topk-embed-v1-small"
REVISION = "e54485ebab921f2c18c4d092b3f4c40dcca26781"
TOKENIZER_SPEC = f"topk-io/topk-embed-v1-small@{REVISION}"
MODEL = "topk-io/topk-embed-v1-small"

QUERY_HEAD = "Query: "
DOCUMENT_HEAD = "Document: "

# Twenty realistic retrieval pairs; several documents carry standalone punctuation, which is what
# makes the reference's document-side keep-mask load-bearing (see test_document_keep_mask).
_PAIRS: list[tuple[str, str]] = [
    ("What was Q3 revenue?", "Q3 revenue was $12 million, up 20% year over year."),
    ("capital of france", "Paris is the capital and largest city of France."),
    ("who wrote pride and prejudice", "Pride and Prejudice is an 1813 novel written by Jane Austen."),
    ("boiling point of water", "Water boils at 100 degrees Celsius at standard atmospheric pressure."),
    ("largest planet", "Jupiter is the largest planet in the solar system, more than twice as massive as Saturn."),
    ("speed of light", "The speed of light in a vacuum is exactly 299,792,458 metres per second."),
    ("first moon landing", "Apollo 11 landed on the Moon on July 20, 1969; Armstrong stepped out six hours later."),
    ("python list vs tuple", "A tuple is immutable while a list is mutable, so tuples can serve as dictionary keys."),
    ("how vaccines work", "Vaccines train the immune system to recognize pathogens without causing the disease."),
    ("the great wall length", "The Great Wall of China stretches over 21,000 kilometers, built across many dynasties."),
    ("causes of inflation", "Inflation rises when demand outpaces supply, or when production costs increase."),
    ("photosynthesis inputs", "Photosynthesis converts carbon dioxide and water into glucose using light energy."),
    ("black hole definition", "A black hole is a region where gravity is so strong that nothing can escape it."),
    ("eiffel tower height", "The Eiffel Tower stands about 330 metres tall and was completed in 1889."),
    ("what is dna", "DNA carries the genetic instructions used in the growth and development of organisms."),
    ("olympics origin", "The Olympic Games began in ancient Greece at Olympia in 776 BC."),
    ("restful api meaning", "A REST API exposes resources over HTTP using standard verbs and stateless requests."),
    ("avogadro number", "Avogadro's number, 6.02214076e23, counts particles in one mole of a substance."),
    ("why is the sky blue", "Air molecules scatter shorter blue wavelengths of sunlight more than longer red ones."),
    ("sql join types", "INNER JOIN keeps matches; LEFT JOIN keeps all rows of the left table with nulls."),
]


@pytest.fixture(scope="module")
def tokenizer():
    """The recipe's tokenizer (the product's loader, its Hub cache); skip when it cannot be fetched."""
    try:
        return load_tokenizer(TOKENIZER_SPEC)
    except Exception as error:  # offline CI, or the hub extra is not installed
        pytest.skip(f"cannot fetch the recipe tokenizer {TOKENIZER_SPEC!r} ({error}); stage 1 on CPU is skipped")


@pytest.fixture(scope="module")
def checkpoint(tokenizer) -> dict:
    """The pinned checkpoint's small config files (config.json, tokenizer_config.json, chat_template.jinja).

    Render mode needs only these (no weights, no torch); the tokenizer fixture has already proven the
    hub is reachable, so a failure here is a real fetch error, reported as such.
    """
    from huggingface_hub import hf_hub_download

    def fetch(name: str) -> dict | str:
        path = hf_hub_download(MODEL, name, revision=REVISION)
        text = Path(path).read_text(encoding="utf-8")
        return json.loads(text) if name.endswith(".json") else text

    return {
        "config": fetch("config.json"),
        "tokenizer_config": fetch("tokenizer_config.json"),
        "chat_template": fetch("chat_template.jinja"),
    }


def _pairs_file(tmp_path: Path) -> Path:
    """The twenty pairs as a JSONL file."""
    path = tmp_path / "pairs.jsonl"
    path.write_text(
        "".join(json.dumps({"query": query, "documents": [document]}) + "\n" for query, document in _PAIRS),
        encoding="utf-8",
    )
    return path


def _mutated_recipe(tmp_path: Path, change: Callable[[dict], dict]) -> Path:
    """A copy of the recipe directory with one YAML mutation applied (the mutations run as recipes)."""
    target = tmp_path / RECIPE_DIR.name
    target.mkdir()
    shutil.copy(RECIPE_DIR / "reference.py", target / "reference.py")
    data = yaml.safe_load((RECIPE_DIR / "recipe.yaml").read_text(encoding="utf-8"))
    (target / "recipe.yaml").write_text(yaml.safe_dump(change(data), sort_keys=False), encoding="utf-8")
    return target


def _write_reference_pairs(sampled: list[dict[str, Any]], work: Path) -> Path:
    """The pairs-file rows of a sampled set, as the pairs file the reference subprocess reads."""
    work.mkdir(parents=True, exist_ok=True)
    path = work / "pairs.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in sampled if "shape" not in row), encoding="utf-8")
    return path


def test_recipe_validates() -> None:
    """The recipe loads through the closed schema: multi_vector over vllm_pooling, plugin named."""
    recipe = load_recipe(RECIPE_DIR)
    assert recipe.id == "topk-embed-v1-small" == RECIPE_DIR.name
    assert recipe.model == MODEL and recipe.revision == REVISION
    assert recipe.role == "multi_vector" and recipe.scoring is None
    assert recipe.input == ["text", "image"]
    assert recipe.licence == "apache-2.0"
    assert recipe.client.api == "vllm_pooling"
    assert recipe.client.tokenizer == TOKENIZER_SPEC and recipe.client.max_tokens == 8192
    assert recipe.client.embed_dtype == "float16" and recipe.client.dim == 2048
    assert recipe.client.normalize is True and recipe.client.on_overflow == "cut"
    assert recipe.client.empty_doc == "omit_zero"
    assert recipe.serve.plugin == "topk-embed-vllm"
    assert recipe.serve.mm_processor_kwargs == {"min_pixels": 65536, "max_pixels": 1310720}
    assert recipe.serve.limit_mm_per_prompt == {"image": 1}
    assert recipe.serve.trust_remote_code is True and recipe.serve.dtype == "bfloat16"
    assert recipe.serve.max_model_len == 8448
    assert recipe.reference.kind == "sentence_transformers" and recipe.reference.score_scale == "cosine"
    assert recipe.status.state == "unverified"
    assert recipe.sources


def test_serve_argv_carries_the_serving_facts() -> None:
    """The rendered argv: the revision, the pixel cap, one image per prompt, no template file.

    R10 considered (recipe notes, note 5): the checkpoint ships its own chat template at the pinned
    revision and vLLM resolves it through AutoProcessor for the pooling-chat path, so no
    --chat-template flag is rendered here; the absence is recorded by this test.
    """
    argv = serve_argv(load_recipe(RECIPE_DIR), port=8100, served_model_name="topk-embed-v1-small")
    assert "--chat-template" not in argv
    assert "--trust-remote-code" in argv
    assert argv[argv.index("--mm-processor-kwargs") + 1] == json.dumps(
        {"max_pixels": 1310720, "min_pixels": 65536}, sort_keys=True
    )
    assert argv[argv.index("--limit-mm-per-prompt") + 1] == json.dumps({"image": 1}, sort_keys=True)
    assert argv[argv.index("--revision") + 1] == REVISION
    assert argv[argv.index("--max-model-len") + 1] == "8448"
    assert not any("topk-embed-vllm" in argument for argument in argv)  # the plugin never reaches the argv


def test_stage1_passes_on_cpu(tmp_path: Path, tokenizer) -> None:
    """The harness's stage 1 on CPU: fits, anchor audit and the reference render all agree."""
    recipe = load_recipe(RECIPE_DIR)
    document = stage1_prompts(
        recipe,
        _pairs_file(tmp_path),
        sys.executable,
        over_length_per_shape=5,
    )
    assert document["sampled"] >= 25, document["sampled"]  # 20 pairs + 5 over-length per shape
    assert document["anchor_check"]["passed"] is True, document["anchor_check"]["failures"][:1]
    assert document["anchor_check"]["checked"] == document["sampled"]
    assert document["render_check"]["status"] == "run"
    assert document["render_check"]["passed"] is True, document["render_check"]["failures"][:1]
    assert document["render_check"]["rows"] == 2 * 20  # one render per declared shape per pair
    assert document["template_render_check"] is None  # no served chat template (see test_serve_argv)
    assert document["engine_tokenize_check"]["status"] == "not_run"  # no engine on CPU; never "passed"
    assert document["passed"] is True


def test_over_length_fitted_render_is_a_prefix_of_the_reference_render(tmp_path: Path, tokenizer) -> None:
    """Token-id and text equality against the reference over the full sample, over-length included.

    In-budget renders are byte-identical (stage 1's render check compares the same thing). For the
    over-length samples the reference's render mode emits the uncut formatted prompt, so the fitted
    (served) text must be a verbatim prefix of it -- the cut is a text prefix of the content span,
    with the fixed head the model reads intact at the front.
    """
    from rcp_ndcg_vllm.equivalence.reference import run_reference
    from rcp_ndcg_vllm.equivalence.stages import _sampled_rows

    recipe = load_recipe(RECIPE_DIR)
    rows = load_pairs(_pairs_file(tmp_path))
    sampled = _sampled_rows(recipe, rows, tokenizer, 5)
    fitted = fit_rows(recipe, sampled, tokenizer)
    work = tmp_path / "ref"
    reference = run_reference(
        sys.executable,
        str(RECIPE_DIR / "reference.py"),
        mode="render",
        pairs_path=_write_reference_pairs(sampled, work),
        out_path=work / "reference.json",
        tokenizer_spec=TOKENIZER_SPEC,
    )
    reference_text = {(int(row["index"]), str(row["shape"])): str(row["text"]) for row in reference["rows"]}
    n_over_length = 0
    for shape, body in fitted["per_shape"].items():
        # The head's trailing space merges into the first content token (" What"), so the stable id
        # prefix of the head is the head without its join boundary.
        head_ids = tokenizer.ids((QUERY_HEAD if shape == "query" else DOCUMENT_HEAD).rstrip(), add_special_tokens=True)
        for position, text in zip(body["row_indexes"], body["texts"], strict=True):
            if position < len(rows):
                assert text == reference_text[(position, shape)]  # in-budget: byte-identical
                continue
            n_over_length += 1
            sampled_row = sampled[position]
            raw = str(sampled_row["query"]) if shape == "query" else str(sampled_row["documents"][0])
            uncut = format_uncut(raw, shape)  # the reference's uncut render of the padded sample
            assert uncut.startswith(text), (shape, text[:60])  # the cut is a verbatim prefix
            assert text.startswith(QUERY_HEAD if shape == "query" else DOCUMENT_HEAD)
            ids = tokenizer.ids(text, add_special_tokens=True)
            assert ids[: len(head_ids)] == head_ids  # the fixed head survived
            # The sample is genuinely over the declared budget as the ENGINE reads it (an uncapped
            # clone of the same tokenizer file; the shipped one caps the product's counting at 1024,
            # recipe notes G5).
            uncapped = type(tokenizer.backend).from_str(tokenizer.backend.to_str())
            uncapped.no_truncation()
            true_count = len(uncapped.encode(text).ids)
            assert true_count > recipe.client.max_tokens, (true_count, shape)
            if len(text) == len(uncut):
                # G5 as shipped: no cut fired, so the whole text goes out while the product's own id
                # view stops at the file's ceiling -- exactly the divergence the wave's /tokenize
                # check must surface.
                assert len(ids) < true_count
            else:
                assert len(text) < len(uncut)  # a ceiling-free counter cuts: a true prefix
    assert n_over_length == 10  # 5 per declared shape


def format_uncut(text: str, shape: str) -> str:
    """The reference's uncut formatted render of one over-length sample's padded text."""
    if shape == "query":
        return QUERY_HEAD + text.strip()
    return DOCUMENT_HEAD + text


def test_image_wrapper_is_pinned(tokenizer, checkpoint) -> None:
    """The chat template's image-only user message renders the exact wrapper ids; a text part diverges.

    The harness's stage-1 template check cannot render a messages-based template (it renders
    query/document variables), so the recipe's test pins the image wrapper here: the ids the client's
    media path must produce through the served chat template, at the pinned revision.
    """
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=False)
    environment.globals["raise_exception"] = _raise
    template = environment.from_string(checkpoint["chat_template"])
    rendered = template.render(messages=[{"role": "user", "content": [{"type": "image"}]}], add_generation_prompt=False)
    ids = tokenizer.ids(rendered, add_special_tokens=True)
    specials = {
        name: tokenizer.special_id(name) for name in ("im_start", "im_end", "vision_start", "vision_end", "image_pad")
    }
    newline = tokenizer.ids("\n", add_special_tokens=False)
    user = tokenizer.ids("user", add_special_tokens=False)
    expected = [specials["im_start"], *user, *newline, specials["vision_start"]]
    expected += [specials["image_pad"]]
    expected += [specials["vision_end"], specials["im_end"], *newline]
    assert ids == expected, [tokenizer.backend.decode([value]) for value in ids]
    assert ids.count(specials["image_pad"]) == 1  # exactly one image placeholder

    with_text = template.render(
        messages=[{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": DOCUMENT_HEAD + "x"}]}],
        add_generation_prompt=False,
    )
    assert tokenizer.ids(with_text, add_special_tokens=True) != ids  # the image-ONLY message is the contract


def _raise(message: str) -> None:
    """The chat templates' raise_exception helper, as the engine's jinja environment provides it."""
    raise ValueError(message)


def test_document_keep_mask_drops_skip_positions(tokenizer, checkpoint) -> None:
    """The keep-mask asymmetry at token level: documents drop skip positions, queries keep everything.

    The mask is the reference's own (topk_embed_st.py:81); the served client needs it and the schema
    has no field (recipe notes, G1) -- this pins the semantics the field must carry.
    """
    config = checkpoint["config"]
    skip = {int(value) for value in config["scoring_skip_ids"]}
    assert len(skip) == 41
    # Completeness: every single-char non-alnum ASCII token in the vocabulary is skipped...
    dropped = 0
    for token, value in tokenizer.backend.get_vocab().items():
        if len(token) == 1 and ord(token) < 128 and not token.isalnum():
            assert value in skip, (value, token)
            dropped += 1
    assert dropped == 32  # ...and they are exactly the 32 punctuation ids of the 41
    ascii_skip = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 25, 26, 27, 28, 29, 30, 31}
    ascii_skip |= {58, 59, 60, 61, 62, 63, 90, 91, 92, 93}
    assert {value for value in skip if value < 248044} == ascii_skip  # the other 9 are specials (>= 248044)
    # Asymmetry on a real pair: the document drops positions (its own colon included), the query
    # keeps every token of the same characters.
    document_text = DOCUMENT_HEAD + "Q3 revenue was $12 million, up 20% year over year."
    query_text = QUERY_HEAD + "What was Q3 revenue?"
    document_ids = tokenizer.ids(document_text, add_special_tokens=True)
    query_ids = tokenizer.ids(query_text, add_special_tokens=True)
    kept_document = [value for value in document_ids if value not in skip]
    kept_query = [value for value in query_ids if value not in skip]
    assert len(kept_document) < len(document_ids), (len(kept_document), len(document_ids))
    assert 25 in document_ids and 25 in skip  # the prompt's own colon is dropped on the document side
    # The asymmetry is real: the same doc-side mask applied to the QUERY would drop its punctuation
    # too -- the reference never does that (queries keep everything), so one shared mask is wrong in
    # both directions (the research's R10 risk).
    assert kept_query != query_ids
    assert [value for value in query_ids if value not in skip] == kept_query


def test_cut_preserves_the_frame_head(tmp_path: Path, tokenizer) -> None:
    """At a budget the counting can see, the cut hits the content span only and the head survives.

    The checkpoint's tokenizer file caps the product's counting at 1024 tokens (recipe notes, G5), so
    the declared 8192 budget cannot fire a cut today; this test shrinks the budget to 64 in a mutated
    copy so the cut path actually runs, and asserts the anchor rule on its output: the fixed head
    ("Query: " / "Document: ") opens every cut render, and the render stays within the budget.
    """
    from rcp_ndcg_vllm.equivalence.stages import _over_length

    recipe = load_recipe(
        _mutated_recipe(tmp_path, lambda data: {**data, "client": {**data["client"], "max_tokens": 64}})
    )
    budget = 64
    for shape in ("query", "document"):
        head = QUERY_HEAD if shape == "query" else DOCUMENT_HEAD
        head_ids = tokenizer.ids(head.rstrip(), add_special_tokens=True)  # the join boundary merges forward
        seed = "What was Q3 revenue?" if shape == "query" else "Q3 revenue was $12 million."
        padded = _over_length(seed, 8192, tokenizer, 0)
        row = {
            "query": padded if shape == "query" else "What was Q3 revenue?",
            "documents": [padded if shape == "document" else "Q3 revenue was $12 million."],
            "shape": shape,
        }
        fitted = fit_rows(recipe, [row], tokenizer)
        body = fitted["per_shape"][shape]
        assert body["cuts"] == 1, body  # the cut fired
        text = body["texts"][0]
        assert text.startswith(head)  # the frame's text survived the cut
        ids = tokenizer.ids(text, add_special_tokens=True)
        assert ids[: len(head_ids)] == head_ids, text[:60]  # the fixed head survived the cut
        assert len(ids) <= budget, len(ids)  # the render honours the budget


def test_mutation_anchor_to_last_makes_the_anchor_check_red(tmp_path: Path, tokenizer) -> None:
    """A wrong anchor declaration is red: `last` demands the fixed head at the tail of every render.

    The declared anchor is `mean` (per-token pooling reads every kept position, so no position is the
    anchor); flipping it to `last` must turn the audit red, proving the check asserts positions and
    does not pass vacuously.
    """

    def mutate(data: dict) -> dict:
        data["client"]["template"]["anchor"] = "last"
        return data

    recipe = load_recipe(_mutated_recipe(tmp_path, mutate))
    document = stage1_prompts(recipe, _pairs_file(tmp_path), None, over_length_per_shape=1)
    assert document["anchor_check"]["passed"] is False
    failures = document["anchor_check"]["failures"]
    assert failures and failures[0]["check"] == "tail"
    assert document["passed"] is False


def test_mutation_drop_frame_segments_makes_the_render_check_red(tmp_path: Path, tokenizer) -> None:
    """Dropping the fixed head segments loses the prompts the model reads: the render check fires.

    The fixed segments are the frame the budget reserves and the prompt the model was trained with;
    dropping them (the head analogue of dropping a trailing anchor segment) must be red, not silent.
    """

    def mutate(data: dict) -> dict:
        data["client"]["template"]["query"] = [{"content": "query"}]
        data["client"]["template"]["document"] = [{"content": "document"}]
        return data

    recipe = load_recipe(_mutated_recipe(tmp_path, mutate))
    document = stage1_prompts(recipe, _pairs_file(tmp_path), sys.executable, over_length_per_shape=1)
    assert document["render_check"]["status"] == "run"
    assert document["render_check"]["passed"] is False
    assert document["render_check"]["failures"], "the reference render must disagree with a frameless fit"
    assert document["passed"] is False
