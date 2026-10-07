"""The pplx-embed-v2-context-9b-preview recipe: the schema, stage 1 on CPU, the anchor mutation.

The recipe validates through the product's ``PoolingEndpoint``; stage 1 runs the harness's
CPU checks against the model's real tokenizer (downloaded into the lane's scratch
directory, outside the checkout, and skipped with a clear reason when offline); the
tokenization facts the wire contract rests on are pinned against the downloaded
tokenizer; and dropping the template's anchor segment turns the anchor audit red.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import yaml
from rcp_ndcg_vllm import load_recipe, serve_argv
from rcp_ndcg_vllm.equivalence import stage1_prompts

from rcp_ndcg.data.tokenizer import load_tokenizer
from rcp_ndcg.inference.config import PoolingEndpoint

RECIPE_ID = "pplx-embed-v2-context-9b-preview"
REVISION = "b667039ee8b438a6350fbc91bbcecd86f9d363ba"
TOKENIZER_SPEC = f"perplexity-ai/{RECIPE_ID}@{REVISION}"
QUERY_PREFIX_ID = 248077  # the model prepends this one id before the query's own tokens
DOCUMENT_PREFIX_ID = 248078  # the special id a server-side parse of "[D] " keeps
DOCUMENT_PREFIX_LITERAL_IDS = (62724, 60)  # the two literal ids the reference's split renders
BOUNDARY_ID = 248079  # the chunk boundary marker, one added id in both renderings
N_PAIRS = 20

RECIPES = Path(__file__).resolve().parents[2] / "recipes" / RECIPE_ID

# The tokenizer cache: ``RCP_NDCG_VLLM_TOKENIZER_CACHE`` when set (the lane's scratch dir -- the
# marker's downloads land there), else the system temp directory. When the pinned snapshot is
# already cached the load runs offline; a fresh machine downloads on the first run and skips
# cleanly with no network.
CACHE = Path(
    os.environ.get("RCP_NDCG_VLLM_TOKENIZER_CACHE") or Path(tempfile.gettempdir()) / "rcp-ndcg-pplx-tokenizers"
)
HF_CACHE = CACHE / "hf-cache"
_CACHED_SNAPSHOT = HF_CACHE / f"models--perplexity-ai--{RECIPE_ID}" / "snapshots" / REVISION / "tokenizer.json"

# huggingface_hub reads its cache directory (and the offline flag) at import time; this
# module binds them before anything in the process imports it.
os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE))
if _CACHED_SNAPSHOT.is_file():
    os.environ["HF_HUB_OFFLINE"] = "1"


@pytest.fixture(scope="module")
def tokenizer():
    """The recipe's own tokenizer, from the scratch cache or the Hub; skipped when offline."""
    try:
        return load_tokenizer(TOKENIZER_SPEC)
    except Exception as error:  # noqa: BLE001  (any Hub/transport failure means offline)
        pytest.skip(
            f"offline: the tokenizer of {TOKENIZER_SPEC} is neither cached under {HF_CACHE} "
            f"nor downloadable ({type(error).__name__}: {error}); stage 1 needs the pinned tokenizer.json"
        )


def _pairs(tokenizer) -> list[dict]:
    """Twenty sampled pairs: single- and multi-chunk documents joined by the boundary marker.

    The document content is the model's chunk list joined by the marker, per the
    recipe's declared chunking; the marker is resolved from the tokenizer's added
    vocabulary, never typed. Chunk text carries no all-special token (the chunker
    guarantee the recipe declares). One row carries an empty document: the model's
    own empty-chunk branch pools it to the zero vector.
    """
    marker = tokenizer.special_text("chunk_sep")

    def document(*chunks: str) -> str:
        return marker.join(chunks)

    # The harness builds its over-length samples by repeating the first row's texts;
    # whether the repeated sample exceeds the 262,142-token cap depends on the seed
    # (the repetition joins merge differently per seed - measured both ways: a
    # letter-initial seed lands over, a long single-line seed landed under). The long
    # multi-chunk first row makes the document-shape samples exceed the cap robustly
    # and carries chunk markers into the padded content, so the anchor audit covers
    # the marker-bearing, over-cap form of the document leg.
    layers = [
        f"paris expanded outward from the cite island for two thousand years; every era left its layer of "
        f"walls, bridges and boulevards on the plan (layer {index})"
        for index in range(24)
    ]
    rows = [
        (
            "what is the capital of france",
            layers,
        ),
        (
            "how do solar panels turn sunlight into electricity",
            [
                "a solar cell converts sunlight to electricity through the photovoltaic effect in a thin silicon wafer",
                "photons striking the wafer free electrons, and the built-in field sweeps them to the contacts",
                "an inverter turns the panel's direct current into the alternating current a household uses",
            ],
        ),
        (
            "who wrote the novel pride and prejudice",
            [
                "jane austen published pride and prejudice in 1813, after a first draft titled first impressions",
            ],
        ),
        (
            "why does bread dough rise",
            [
                "yeast ferments the sugars in flour and releases carbon dioxide",
                "the gluten network traps the gas, so the dough expands and bakes into a light, open crumb",
            ],
        ),
        (
            "what causes the northern lights",
            [
                "charged particles of the solar wind are guided along earth's field lines toward the poles",
                "there they excite oxygen and nitrogen in the upper atmosphere, which releases green and red light",
            ],
        ),
        (
            "when did the berlin wall fall",
            [
                "the berlin wall was opened on the ninth of november 1989 after an improvised announcement",
            ],
        ),
        (
            "how does a refrigerator cool its interior",
            [
                "a compressor squeezes a refrigerant vapour, which sheds heat through the condenser coils at the back",
                "expanding through a valve the fluid turns cold and absorbs heat from the cabinet",
            ],
        ),
        (
            "which planet has the most moons",
            [
                "saturn overtook jupiter as the planet with the most confirmed moons after a new batch was announced",
            ],
        ),
        (
            "what is the difference between weather and climate",
            [
                "weather is the state of the atmosphere over hours or days",
                "climate is the statistics of weather over decades, so a cold winter does not refute a warming trend",
            ],
        ),
        (
            "how do vaccines train the immune system",
            [
                "a vaccine presents the immune system with a harmless fragment or blueprint of a pathogen",
                "the b cells and t cells that recognise the fragment multiply and leave memory behind",
                "on a real infection the memory response fires before the pathogen takes hold",
            ],
        ),
        (
            "what did the marshall programme fund",
            [
                "the marshall programme financed the rebuilding of western european industry, ports and housing",
            ],
        ),
        (
            "why do leaves change colour in autumn",
            [
                "cooling nights and shorter days make the leaf drop its chlorophyll",
                "the yellow and orange pigments already there become visible, and some species make fresh red ones",
            ],
        ),
        (
            "what is the boiling point of water at altitude",
            [
                "water boils at lower temperatures as pressure falls, so on a high pass it boils near ninety degrees",
            ],
        ),
        (
            "how does the internet route a packet",
            [
                "routers read each packet's destination and forward it along the next hop their tables choose",
                "no single router knows the whole path; each hop shortens the distance to the destination network",
            ],
        ),
        (
            "who first described natural selection",
            [
                "darwin and alfred russel wallace presented the idea of evolution by natural selection in 1858",
                "darwin's on the origin of species followed a year later and set out the evidence in detail",
            ],
        ),
        (
            "what keeps the earth's core hot",
            [
                "the core holds residual heat from the planet's formation plus radioactive decay",
                "heat flows outward through the mantle, driving the convection that moves the tectonic plates",
                "the inner core stays solid because pressure raises its melting point above its temperature",
            ],
        ),
        (
            "what is the corpus luteum",
            [
                "after an egg leaves the ovary the follicle becomes the corpus luteum, which secretes progesterone",
            ],
        ),
        (
            "how do noise cancelling headphones work",
            [
                "a microphone samples the ambient sound and a speaker plays its inverted waveform to cancel it",
            ],
        ),
        (
            "what is the oldest known written language",
            [
                "sumerian cuneiform tablets from the fourth millennium before our era are the oldest readable writing",
                "earlier symbols exist, but scholars cannot show they encode language, not numbers or marks",
            ],
        ),
        (
            "why is the sky blue at noon",
            [
                "",
                "air scatters short wavelengths far more than long ones, so blue light reaches the eye from everywhere",
                "at sunset the light crosses more air, the blue is scattered away, and the remainder reads red",
            ],
        ),
    ]
    return [{"query": query, "documents": [document(*chunks)]} for query, chunks in rows]


def test_recipe_validates_against_the_product_endpoint() -> None:
    """The recipe loads, and its client block is the product's PoolingEndpoint with the declared budget."""
    recipe = load_recipe(RECIPES)
    assert recipe.id == RECIPE_ID
    assert recipe.model == f"perplexity-ai/{RECIPE_ID}"
    assert recipe.revision == REVISION
    assert recipe.role == "multi_vector"
    assert recipe.input == ["text"]
    assert isinstance(recipe.client, PoolingEndpoint)
    assert recipe.client.api == "vllm_pooling"
    assert recipe.client.request_shape == "token_ids"
    assert recipe.client.tokenizer == TOKENIZER_SPEC
    assert recipe.client.max_tokens == 262142
    template = recipe.client.template
    assert template is not None and template.shapes() == ("query", "document")
    assert template is not None and template.anchor == "first"
    assert recipe.client.on_overflow == "cut"
    assert recipe.client.empty_doc == "send"
    assert recipe.client.normalize is True
    assert recipe.client.embed_dtype == "float16"
    assert recipe.client.dim == 2048
    assert recipe.serve.dtype == "float32"
    assert recipe.serve.max_model_len == 262144
    assert recipe.serve.runner == "pooling"
    assert recipe.serve.plugin == "rcp-ndcg-vllm-pplx"
    assert recipe.serve.chat_template is None
    assert recipe.serve.trust_remote_code is True
    assert recipe.reference.kind == "remote_code"
    assert recipe.status.state == "unverified"


def test_serve_argv_carries_the_plugin_and_the_pooling_flags() -> None:
    """The rendered argv serves the stock image with the plugin, fp32 and the token_embed pooler."""
    recipe = load_recipe(RECIPES)
    argv = serve_argv(recipe, port=8100, served_model_name=RECIPE_ID)
    assert argv[:3] == ["vllm", "serve", recipe.model]
    assert argv[argv.index("--revision") + 1] == REVISION
    assert argv[argv.index("--dtype") + 1] == "float32"
    assert argv[argv.index("--max-model-len") + 1] == "262144"
    assert argv[argv.index("--pooler-config") + 1] == '{"task": "token_embed"}'
    assert "--trust-remote-code" in argv
    assert "--chat-template" not in argv  # the plugin names the wheel; the template would be inert on this route


def test_wire_contract_tokenization_facts(tokenizer) -> None:
    """The measured tokenizer facts the wire contract rests on, at the pinned revision.

    The query's text render tokenizes to exactly the reference's ids (the [Q] prefix
    crosses as the one special id, the body unshifted). The document's text render
    keeps the added special id where the model's own split renders two literal
    tokens: the measured divergence that makes the wire token_ids.
    """
    query = "what drives scientific breakthroughs"
    query_ids = tokenizer.ids(f"[Q] {query}")
    assert query_ids[0] == QUERY_PREFIX_ID
    assert query_ids == [QUERY_PREFIX_ID, *tokenizer.ids(query)]
    document = tokenizer.special_text("chunk_sep").join(["first chunk text", "second chunk text"])
    document_ids = tokenizer.ids(f"[D] {document}")
    assert document_ids[0] == DOCUMENT_PREFIX_ID
    assert document_ids.count(BOUNDARY_ID) == 1
    # The reference's own parse (split_special_tokens=True, verified with transformers
    # 5.18.0 at this revision) renders the prefix as the two literal ids below, and one
    # token more in total for this content; the server-side parse keeps the special
    # id. The recipe's budget reserves the measured worst-case delta (two tokens).
    assert tokenizer.backend.token_to_id("[D") == DOCUMENT_PREFIX_LITERAL_IDS[0]
    assert tokenizer.backend.token_to_id("]") == DOCUMENT_PREFIX_LITERAL_IDS[1]
    assert document_ids[:2] != list(DOCUMENT_PREFIX_LITERAL_IDS)


def test_stage1_passes_on_cpu(tmp_path: Path, tokenizer) -> None:
    """Stage 1 on CPU: the product's fit, the anchor audit, the reference render - all green.

    Twenty pairs rows are sampled, plus the harness's own over-length inputs
    (five per declared shape, padded in that shape's own content span, cut by the
    product's budget mechanism with every anchor reserved).
    """
    recipe = load_recipe(RECIPES)
    pairs_path = tmp_path / "pairs.jsonl"
    pairs_path.write_text("".join(json.dumps(row) + "\n" for row in _pairs(tokenizer)), encoding="utf-8")
    document = stage1_prompts(recipe, pairs_path, sys.executable, over_length_per_shape=5)
    assert document["pairs"] == N_PAIRS
    assert document["sampled"] == N_PAIRS + 2 * 5
    assert document["anchor_check"]["passed"] is True, document["anchor_check"]["failures"][:1]
    assert document["render_check"]["status"] == "run"
    assert document["render_check"]["passed"] is True, document["render_check"]["failures"][:2]
    assert document["template_render_check"] is None  # no chat_template: the route applies none
    assert document["engine_tokenize_check"]["status"] == "not_run"
    assert document["passed"] is True


def test_dropping_the_anchor_segment_turns_the_anchor_check_red(tmp_path: Path, tokenizer) -> None:
    """The mutation: drop the template's anchor segment (the leading role prefix) and the audit goes red.

    The pplx anchor is the head fixed segment (anchor: first), not a trailing one: the
    query's pooled mean includes the prefix token and the plugin reads the leading id to
    tell the roles apart. Dropping it keeps the template loadable (add_special_tokens:
    true declares the route's post-processor, which appends nothing) and leaves every
    render without its anchor ids.
    """
    pairs_path = tmp_path / "pairs.jsonl"
    pairs_path.write_text("".join(json.dumps(row) + "\n" for row in _pairs(tokenizer)[:2]), encoding="utf-8")
    healthy = stage1_prompts(load_recipe(RECIPES), pairs_path, None, over_length_per_shape=1)
    assert healthy["anchor_check"]["passed"] is True  # positive control: the audit is not vacuously red

    mutated_dir = tmp_path / RECIPE_ID
    mutated_dir.mkdir()
    data = yaml.safe_load((RECIPES / "recipe.yaml").read_text(encoding="utf-8"))
    template = data["client"]["template"]
    for shape in ("query", "document"):
        template[shape] = [segment for segment in template[shape] if "content" in segment]
    shutil_reference = mutated_dir / "reference.py"
    shutil_reference.write_text((RECIPES / "reference.py").read_text(encoding="utf-8"), encoding="utf-8")
    (mutated_dir / "recipe.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    broken = load_recipe(mutated_dir)
    red = stage1_prompts(broken, pairs_path, None, over_length_per_shape=1)
    assert red["anchor_check"]["passed"] is False
    assert red["anchor_check"]["failures"], "the audit must name the shape it failed"


def test_reference_embed_refuses_cpu_before_any_download(tmp_path: Path) -> None:
    """The reference's GPU guard: on cpu it raises with the checkpoint's size and writes no output."""
    out_path = tmp_path / "reference.json"
    pairs_path = tmp_path / "pairs.jsonl"
    pairs_path.write_text('{"query": "q", "documents": ["d"]}\n', encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(RECIPES / "reference.py"),
            "--mode",
            "embed",
            "--pairs",
            str(pairs_path),
            "--out",
            str(out_path),
            "--tokenizer",
            TOKENIZER_SPEC,
            "--device",
            "cpu",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode != 0, completed.stdout
    assert "GPU host" in completed.stderr
    assert not out_path.exists()
