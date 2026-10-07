"""The ``qwen3-reranker-8b`` recipe: schema validation, stage 1 on CPU, and the anchor mutation.

Stage 1 needs the model's real tokenizer (``tokenizer.json`` only -- no weights, no torch): it
comes from ``RCP_QWEN3_RERANKER_8B_TOKENIZER_DIR`` (a directory holding the pinned revision's
tokenizer.json, e.g. the lane's scratch copy) or is downloaded into ``tmp_path``.  Offline --
CI without either -- the tokenizer tests skip with a clear reason; the schema test always runs.

The reference subprocess is the recipe's own ``reference.py`` in render mode (tokenizers only,
no torch), run with the test interpreter: the harness process imports no torch.
"""

from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

import pytest
from rcp_ndcg_vllm import RecipeError, load_recipe, serve_argv
from rcp_ndcg_vllm.equivalence import stage1_prompts

from ._served import served_pair, stage1_facts

REPO = "Qwen/Qwen3-Reranker-8B"
REVISION = "77d193c791ed757ca307ee72715aa132723da912"
RECIPE_DIR = Path(__file__).resolve().parents[2] / "recipes" / "qwen3-reranker-8b"

MAX_TOKENS = 8192
QUERY_MAX_TOKENS = 4096
OVERHEAD = 73  # 39-token prefix + 9-token suffix + 25 for the instruction and the labels

# chat-template markers, built without typing them literally
IM_START = chr(60) + "|im_start|>"
IM_END = chr(60) + "|im_end|>"
THINK_OPEN = chr(60) + "think" + chr(62)
THINK_CLOSE = chr(60) + "/" + "think" + chr(62)
PREFIX = (
    f"{IM_START}system\nJudge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".'
    f"{IM_END}\n{IM_START}user\n<Instruct>: Given a web search query, retrieve relevant passages "
    "that answer the query\n<Query>: "
)
SUFFIX = f"{IM_END}\n{IM_START}assistant\n{THINK_OPEN}\n\n{THINK_CLOSE}\n\n"
PAIR_MID = "\n<Document>: "


def tokenizer_dir_fixture() -> Path | None:
    """The tokenizer directory: the env-named copy first, else a download into tmp_path."""
    env_dir = os.environ.get("RCP_QWEN3_RERANKER_8B_TOKENIZER_DIR")
    if env_dir and (Path(env_dir) / "tokenizer.json").is_file():
        return Path(env_dir)
    return None


@pytest.fixture(scope="module")
def tokenizer_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The pinned revision's tokenizer files: the lane scratch copy or a fresh download.

    Skips with a clear reason when offline and nothing is cached (CI): the tokenizer is the
    one network need of these tests, and no weights are ever fetched.
    """
    seeded = tokenizer_dir_fixture()
    if seeded is not None:
        return seeded
    target = tmp_path_factory.mktemp("qwen3-tokenizer")
    url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/tokenizer.json"
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            (target / "tokenizer.json").write_bytes(response.read())
    except OSError as error:
        pytest.skip(
            f"offline: the {REPO} tokenizer is unavailable (name a directory holding its "
            f"tokenizer.json in RCP_QWEN3_RERANKER_8B_TOKENIZER_DIR, or run with network access "
            f"so it downloads into tmp_path): {error}"
        )
    return target


def local_recipe(tokenizer_dir: Path):
    """The recipe with its tokenizer pointed at the local tokenizer.json directory."""
    recipe = load_recipe(RECIPE_DIR)
    client = recipe.client.model_copy(update={"tokenizer": str(tokenizer_dir)})
    return recipe.model_copy(update={"client": client})


def sample_pairs() -> list[dict]:
    """16 short pairs plus 5 over-length ones (long documents, short queries)."""
    short = [
        ("capital of france", "paris is the capital of france."),
        ("who wrote hamlet", "hamlet was written by william shakespeare around 1600."),
        ("speed of light", "light travels at about 299792 kilometres per second in vacuum."),
        ("largest ocean", "the pacific ocean is the largest ocean on earth."),
        ("python gil", "the gil serialises bytecode execution across threads in cpython."),
        ("boiling point of water", "water boils at 100 degrees celsius at sea level."),
        ("first moon landing", "apollo 11 landed on the moon on july 20 1969."),
        ("currency of japan", "the yen is the official currency of japan."),
        ("tallest mountain", "mount everest rises about 8849 metres above sea level."),
        ("inventor of telephone", "alexander graham bell patented the telephone in 1876."),
        ("longest river", "the nile and the amazon compete for the longest river title."),
        ("chemical symbol for gold", "the chemical symbol for gold is au."),
        ("great wall length", "the great wall of china spans over 21000 kilometres."),
        ("human bones count", "an adult human skeleton has 206 bones."),
        ("photosynthesis input", "photosynthesis turns carbon dioxide and water into glucose."),
        ("oldest university", "the university of bologna was founded in 1088."),
    ]
    pairs = [{"query": query, "documents": [document]} for query, document in short]
    long_document = "the quick brown fox jumps over the lazy dog. " * 950
    for index in range(5):
        pairs.append(
            {
                "query": f"retrieval probe {index}: what does the corpus say about topic {index}",
                "documents": [f"{long_document} closing note {index}"],
            }
        )
    return pairs


def write_pairs(path: Path, pairs: list[dict]) -> Path:
    import json

    path.write_text("".join(json.dumps(row) + "\n" for row in pairs), encoding="utf-8")
    return path


def test_recipe_validates() -> None:
    """The recipe loads; the client block is the product's RerankEndpoint; R10's file ships."""
    recipe = load_recipe(RECIPE_DIR)
    assert recipe.id == "qwen3-reranker-8b"
    assert recipe.model == REPO
    assert recipe.revision == REVISION
    assert recipe.role == "rerank" and recipe.scoring == "pointwise"
    assert recipe.serve.chat_template == "template.jinja"  # R10: the template file ships
    assert (RECIPE_DIR / recipe.serve.chat_template).is_file()
    assert recipe.serve.max_model_len >= MAX_TOKENS
    assert recipe.serve.dtype == "bfloat16"
    assert recipe.serve.hf_overrides == {
        "architectures": ["Qwen3ForSequenceClassification"],
        "classifier_from_token": ["no", "yes"],
        "is_original_qwen3_reranker": True,
    }
    client = recipe.client
    assert client.tokenizer == f"{REPO}@{REVISION}"
    assert client.max_tokens == MAX_TOKENS
    assert client.query_max_tokens == QUERY_MAX_TOKENS
    assert client.on_overflow == "cut"
    assert client.use_activation is True  # probability scale, matching the reference
    assert client.instruction == "none"  # the paper's fixed default instruction
    assert client.template.anchor == "last"
    assert client.template.shapes() == ("pair",)
    assert recipe.reference.known_deviations == []  # the paper code drops no anchor (measured)
    assert recipe.reference.score_scale == "probability"
    assert recipe.status.state == "unverified"
    assert recipe.sources, "the recipe lists its sources"
    argv = serve_argv(recipe, port=8100, served_model_name=recipe.id)
    assert "--chat-template" in argv and "--runner pooling".split()[0] in argv


def test_stage1_on_cpu(tmp_path: Path, tokenizer_dir: Path) -> None:
    """Stage 1 on CPU: token-id equality against the reference render and the template file,
    and every anchor intact on over-length inputs (21 pairs file rows, 5 of them over cap,
    plus the harness's own 5 over-length samples per shape)."""
    from rcp_ndcg.data.tokenizer import load_tokenizer

    recipe = local_recipe(tokenizer_dir)
    tokenizer = load_tokenizer(str(tokenizer_dir))
    pairs = sample_pairs()
    over_cap = 0
    for row in pairs:
        document = row["documents"][0]
        tokens = OVERHEAD + tokenizer.count(row["query"]) + tokenizer.count(document)
        over_cap += int(tokens > MAX_TOKENS)
    assert over_cap == 5, "the pairs file must carry 5 over-length rows"
    pairs_path = write_pairs(tmp_path / "pairs.jsonl", pairs)

    document = stage1_prompts(recipe, pairs_path, sys.executable, over_length_per_shape=5)
    assert document["passed"] is True, (
        document["anchor_check"]["failures"][:1],
        document["render_check"]["failures"][:1] if document["render_check"] else None,
        document["template_render_check"]["failures"][:1] if document["template_render_check"] else None,
    )
    assert document["anchor_check"]["passed"] is True
    assert document["anchor_check"]["checked"] >= 26
    assert document["render_check"]["status"] == "run" and document["render_check"]["passed"] is True
    assert document["template_render_check"]["passed"] is True  # the declared shapes == the file
    assert document["engine_tokenize_check"]["status"] == "not_run"  # no engine on CPU
    # The cut facts come from the role client's own capture and census (R30: what the client really
    # sent): the 5 over-length pairs-file rows and the harness's 5 padded samples were cut, and the
    # product measured the fixed frame's overhead.
    facts = stage1_facts(recipe, pairs, tokenizer, 5)
    assert facts["per_shape"]["pair"]["overhead"] == OVERHEAD
    assert facts["per_shape"]["pair"]["cut_rows"] == 10  # 5 pairs-file rows + 5 harness-padded samples

    # golden: one under-cap pair ships whole, and its spans re-assemble byte-identically to the
    # paper reference's prompt (the product's own template render).
    template = recipe.client.template
    assert template is not None
    query, doc = pairs[0]["query"], pairs[0]["documents"][0]
    spans = served_pair(recipe, query, [doc])
    assert spans == {"query": query, "documents": [doc]}
    expected = PREFIX + query + PAIR_MID + doc + SUFFIX
    assert template.render("pair", tokenizer, query=spans["query"], document=spans["documents"][0]) == expected

    # and one over-cap pair is cut to exactly the paper budget with the anchor intact: the spans
    # the client ships re-assemble to a render at the budget that ends on the scored suffix.
    long_row = next(row for row in pairs if OVERHEAD + tokenizer.count(row["documents"][0]) > MAX_TOKENS)
    spans = served_pair(recipe, long_row["query"], long_row["documents"])
    rendered = template.render("pair", tokenizer, query=spans["query"], document=spans["documents"][0])
    ids = tokenizer.ids(rendered, add_special_tokens=True)
    assert len(ids) == MAX_TOKENS
    assert ids[-len(tokenizer.ids(SUFFIX)) :] == tokenizer.ids(SUFFIX)


def test_mutation_dropping_the_trailing_anchor_segment_reddens_the_template_check(
    tmp_path: Path, tokenizer_dir: Path
) -> None:
    """Dropping the template's trailing anchor segment loses the assistant suffix -- the
    scored position -- and the file-vs-declaration check must go red.

    The rerank wire carries the cut content spans (the frame is the engine's own template), so
    stage 1's anchor audit audits the settled query and the document spans and does not move on a
    frame change; the frame contract is pinned by ``template_render_check``, which this mutation
    turns red (the proved frame loses the anchor above).
    """
    from rcp_ndcg.data.templates import TemplateSpec
    from rcp_ndcg.data.tokenizer import load_tokenizer

    recipe = local_recipe(tokenizer_dir)
    tokenizer = load_tokenizer(str(tokenizer_dir))
    template = recipe.client.template
    assert template is not None

    # the real frame ends with the 9-token assistant suffix (the scored anchor), built by the
    # product's own render
    golden = template.render("pair", tokenizer, query="capital of france", document="paris is the capital of france.")
    real_ids = tokenizer.ids(golden, add_special_tokens=True)
    suffix_ids = tokenizer.ids(SUFFIX, add_special_tokens=True)
    assert real_ids[-len(suffix_ids) :] == suffix_ids

    # the mutation: drop the trailing anchor segment, keep the shape loadable by declaring the
    # tokenizer's post-processor as the anchor -- which for Qwen adds nothing, so the anchor
    # is really gone (the product refuses the same shape without that declaration)
    anchorless = TemplateSpec(
        pair=tuple(segment for segment in template.pair[:-1]), anchor="last", add_special_tokens=True
    )
    client = recipe.client.model_copy(update={"template": anchorless})
    mutated = recipe.model_copy(update={"client": client})
    assert mutated.client.template is not None
    no_tail_ids = tokenizer.ids(
        mutated.client.template.render(
            "pair", tokenizer, query="capital of france", document="paris is the capital of france."
        ),
        add_special_tokens=True,
    )
    assert no_tail_ids[-len(suffix_ids) :] != suffix_ids, "the mutation must really lose the anchor"

    pairs_path = write_pairs(tmp_path / "pairs.jsonl", sample_pairs()[:3])
    document = stage1_prompts(mutated, pairs_path, sys.executable, over_length_per_shape=2)
    assert document["template_render_check"]["passed"] is False, "the file still emits the dropped suffix"
    assert document["anchor_check"]["passed"] is True  # the wire's spans are unchanged by the frame drop

    # and the schema itself refuses an 'anchor: last' shape with neither a fixed tail nor the
    # post-processor declaration
    with pytest.raises(ValueError, match="anchor: last"):
        TemplateSpec(pair=tuple(segment for segment in template.pair[:-1]), anchor="last", add_special_tokens=False)


def test_the_recipe_refuses_an_undeclared_query_share(tmp_path: Path) -> None:
    """A client block the product would refuse (a query share at or over the budget) is refused
    at load, with the product's message -- the recipe's client block IS the product's config."""
    data = (RECIPE_DIR / "recipe.yaml").read_text(encoding="utf-8")
    broken_dir = tmp_path / "qwen3-reranker-8b"
    broken_dir.mkdir()
    (broken_dir / "recipe.yaml").write_text(
        data.replace("query_max_tokens: 4096", "query_max_tokens: 8192"), encoding="utf-8"
    )
    (broken_dir / "template.jinja").write_text((RECIPE_DIR / "template.jinja").read_text(encoding="utf-8"))
    (broken_dir / "reference.py").write_text((RECIPE_DIR / "reference.py").read_text(encoding="utf-8"))
    with pytest.raises(RecipeError, match="query_max_tokens"):
        load_recipe(broken_dir)
