"""The pplx-embed-v2-context-9b-preview recipe: contract, stage 1 on CPU, the ids facts.

The recipe validates through the product's ``PoolingEndpoint`` and every resolved serve/client/
reference field is pinned through the shared helper (``tests/recipes/_contract.py``), with two
mutant tests showing a drifted recipe going red by name. Stage 1 runs the harness's CPU checks
against the model's real tokenizer (downloaded under ``RCP_NDCG_VLLM_TOKENIZER_CACHE``, else the
system temp directory -- never the checkout -- and skipped with a clear reason when offline); the
tokenization facts the wire contract rests on are pinned against the downloaded tokenizer, and the
declared document-ids product gap is pinned on both sides (flip that test to equality when the
product gains the split-parse id seam the notes name).
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
from rcp_ndcg_test.equivalence import stage1_prompts
from rcp_ndcg_vllm import load_recipe, serve_argv
from rcp_ndcg_vllm.recipe import default_recipes_root

from ._contract import assert_recipe_contract

RECIPE_ID = "pplx-embed-v2-context-9b-preview"
REVISION = "b667039ee8b438a6350fbc91bbcecd86f9d363ba"
TOKENIZER_SPEC = f"perplexity-ai/{RECIPE_ID}@{REVISION}"
QUERY_PREFIX_ID = 248077  # the model prepends this one id before the query's own tokens
DOCUMENT_PREFIX_ID = 248078  # the special id a server-side parse of "[D] " keeps
DOCUMENT_PREFIX_LITERAL_IDS = (62724, 60)  # the two literal ids the reference's split renders
BOUNDARY_ID = 248079  # the chunk boundary marker, one added id in both renderings
N_PAIRS = 20

FAMILY_ID = "pplx-embed-v2-context"
RECIPES = default_recipes_root() / FAMILY_ID

# The tokenizer cache: ``RCP_NDCG_VLLM_TOKENIZER_CACHE`` when set (the lane's scratch dir -- the
# marker's downloads land there), else the system temp directory. When the pinned snapshot is
# already cached the load runs offline; a fresh machine downloads on the first run and skips
# cleanly with no network.
CACHE = Path(
    os.environ.get("RCP_NDCG_VLLM_TOKENIZER_CACHE") or Path(tempfile.gettempdir()) / "rcp-ndcg-pplx-tokenizers"
)
HF_CACHE = CACHE / "hf-cache"

# huggingface_hub reads its cache directory at import time; this module binds it before anything in the
# process imports it. It does NOT set HF_HUB_OFFLINE: the flag is process-wide and a module-level write
# would leak the offline mode into every sibling test module in the same worker (the gate runs this whole
# directory with -n 4), turning their Hub reads into offline-mode failures. The tokenizer fixture below
# falls back to the cache and skips when neither the cache nor the Hub can answer.
os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE))

EXPECTED_SERVE = {
    "patches": ["pooling-full-context"],
    "runner": "pooling",
    "convert": None,
    "hf_overrides": {},
    "chat_template": None,
    "pooler_config": {
        "task": "token_embed",
    },
    "trust_remote_code": False,
    "max_model_len": 131072,
    "dtype": "bfloat16",
    "plugin": "rcp-ndcg-vllm",
    "io_processor_plugin": None,
    "mm_processor_kwargs": {},
    "limit_mm_per_prompt": None,
    "extra_args": [],
}

EXPECTED_CLIENT = {
    "api": "vllm_pooling",
    "instruction": "none",
    "request_shape": "token_ids",
    "tokenizer": "perplexity-ai/pplx-embed-v2-context-9b-preview@b667039ee8b438a6350fbc91bbcecd86f9d363ba",
    "max_tokens": 131070,
    "outputs": "per_chunk",
    "template": {
        "query": [{"fixed": "{special:[Q] }"}, {"content": "query"}],
        "document": [{"fixed": "{special:[D] }"}, {"content": "document"}],
        "anchor": "first",
        "add_special_tokens": True,
    },
    "on_overflow": "cut",
    "empty_doc": "send",
    "normalize": True,
    "embed_dtype": "float16",
    "dim": 2048,
    "model": "pplx-embed-v2-context-9b-preview",
    "revision": "b667039ee8b438a6350fbc91bbcecd86f9d363ba",
}

EXPECTED_REFERENCE = {
    "attn_implementation": None,
    "kind": "remote_code",
    "score_scale": "cosine",
    "entry": "reference.py",
    "known_deviations": ["over_cap_cut_differs"],
    "device": "cuda",
}

EXPECTED_TOP = {
    "id": "pplx-embed-v2-context-9b-preview",
    "licence": "MIT",
    "revision": "b667039ee8b438a6350fbc91bbcecd86f9d363ba",
    "role": "multi_vector",
    "input": ["text"],
    "model": "perplexity-ai/pplx-embed-v2-context-9b-preview",
}


@pytest.fixture(scope="module")
def tokenizer():
    """The recipe's own tokenizer, from the cache or the Hub; skipped when offline."""
    try:
        from rcp_ndcg.data.tokenizer import load_tokenizer

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


def test_recipe_contract() -> None:
    """Every resolved serve/client/reference field is pinned (the shared helper, both directions)."""
    recipe = load_recipe(RECIPES)
    assert_recipe_contract(
        recipe,
        serve=EXPECTED_SERVE,
        client=EXPECTED_CLIENT,
        reference=EXPECTED_REFERENCE,
        top=EXPECTED_TOP,
    )
    assert recipe.status.state == "unverified"
    assert recipe.sources


def test_contract_mutant_serve_max_model_len_is_red(tmp_path: Path) -> None:
    """Mutant 1: serve.max_model_len 131072 -> 262144 must red, naming the field.

    Drifted upward: a context below the client's 131,070-token budget is refused by the schema itself
    (client.max_tokens must not exceed engine.max_model_len) before the contract pin is reached."""

    def mutate(data: dict) -> dict:
        data["serve"]["max_model_len"] = 327680
        return data

    drifted = load_recipe(RECIPE_ID, root=_mutated_recipe(tmp_path / "mutant", mutate).parent)
    with pytest.raises(AssertionError, match=r"serve\.max_model_len"):
        assert_recipe_contract(
            drifted, serve=EXPECTED_SERVE, client=EXPECTED_CLIENT, reference=EXPECTED_REFERENCE, top=EXPECTED_TOP
        )


def test_contract_mutant_reference_kind_is_red(tmp_path: Path) -> None:
    """Mutant 2: reference.kind remote_code -> transformers must red."""

    def mutate(data: dict) -> dict:
        data["reference"]["kind"] = "transformers"
        return data

    drifted = load_recipe(RECIPE_ID, root=_mutated_recipe(tmp_path / "mutant", mutate).parent)
    with pytest.raises(AssertionError, match=r"reference\.kind"):
        assert_recipe_contract(
            drifted, serve=EXPECTED_SERVE, client=EXPECTED_CLIENT, reference=EXPECTED_REFERENCE, top=EXPECTED_TOP
        )


def _resolved_recipe_file(directory: Path) -> Path:
    """The resolved recipe JSON, exactly what the harness's ``run_reference`` passes as ``--recipe``."""
    import json

    recipe = load_recipe(RECIPE_ID)
    path = directory / "reference.recipe.json"
    path.write_text(json.dumps(recipe.model_dump(mode="json"), sort_keys=True), encoding="utf-8")
    return path


def _mutated_recipe(root: Path, change) -> Path:
    """A copy of the recipe directory with one YAML mutation applied (the contract mutants)."""
    root.mkdir(parents=True, exist_ok=True)
    target = root / FAMILY_ID  # the family directory name (the loader pins family id == directory name)
    target.mkdir()
    (target / "reference.py").write_text((RECIPES / "reference.py").read_text(encoding="utf-8"), encoding="utf-8")
    data = yaml.safe_load((RECIPES / "family.yaml").read_text(encoding="utf-8"))
    (target / "family.yaml").write_text(yaml.safe_dump(change(data), sort_keys=False), encoding="utf-8")
    return target


def _probe_recipe(root: Path, change=None) -> Path:
    """A probe copy for the offline fake: the answer bounded, the request mechanism unchanged.

    Stage 1 audits what the role client SENDS; the fake's answer is scaffolding. The product's offline fake
    answers one seeded vector per token of every item (one draw per vector, seeded by the item's whole
    body: for a token-ids body, its id list), so at the shipped budget the over-length samples -- padded to
    twice the ~262,142-token budget -- make every probed item half a million vector draws over a
    megabyte-long body, and a 2048-wide answer a multi-gigabyte matrix. The probe copy therefore declares
    ``dim: 8`` (the answer's width, reply-side only) and ``max_tokens: 2048`` (the over-length samples are
    padded past the DECLARED budget, so they shrink with it). Every check stage 1
    runs is budget-independent (the fit's content-only cut, the anchors, the render comparison of the pairs
    rows, all under 2048 tokens); the shipped 262,142 and 2048 are pinned by ``test_recipe_contract``, and the
    full-budget stage 1 runs against the engine on the GPU wave.
    """

    def bound(data: dict) -> dict:
        narrowed = {**data, "client": {**data["client"], "dim": 8, "max_tokens": 2048}}
        return change(narrowed) if change else narrowed

    return _mutated_recipe(root, bound)


def test_serve_argv_carries_the_plugin_and_the_pooling_flags() -> None:
    """The rendered argv serves the stock image with the plugin, bf16 and the token_embed pooler.

    No --trust-remote-code: the plugin's registered config class parses config.json locally (the
    AutoConfig registration the contract-core tests pin)."""
    recipe = load_recipe(RECIPES)
    argv = serve_argv(recipe, port=8100, served_model_name=RECIPE_ID)
    assert argv[:3] == ["vllm", "serve", recipe.model]
    assert argv[argv.index("--revision") + 1] == REVISION
    assert argv[argv.index("--dtype") + 1] == "bfloat16"
    assert argv[argv.index("--max-model-len") + 1] == "131072"
    assert argv[argv.index("--pooler-config") + 1] == '{"task": "token_embed"}'
    assert "--trust-remote-code" not in argv
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


def _split_ids(tokenizer, text: str) -> list[int]:
    """The reference's split-side ids of ``text``: transformers' ``split_special_tokens=True``.

    That flag is the raw ``tokenizers`` ``encode_special_tokens`` toggle
    (transformers 5.x tokenization_utils_tokenizers.py:466 and :1055-1060): added SPECIAL
    tokens textify (the role prefixes render as their literal tokens), non-special added
    tokens (the chunk_sep boundary marker) keep their one id. The backend object is shared,
    so the flag is restored after the call.
    """
    backend = tokenizer.backend
    backend.encode_special_tokens = True
    try:
        return list(backend.encode(text, add_special_tokens=False).ids)
    finally:
        backend.encode_special_tokens = False


def test_the_wire_ids_match_on_the_query_and_diverge_on_the_document(tokenizer) -> None:
    """The declared product gap (the notes' PRODUCT GAP paragraph), pinned on both sides.

    What request_shape: token_ids sends is the tokenizer's added-token parse of the fitted
    render; what the model's reference call runs is the split-side parse of the same text
    (modeling_pplx_contextual.py:46-113). The QUERY leg matches (the remote prepends the
    248077 id and split-tokenizes the body; the added-token parse of the render equals it).
    The DOCUMENT leg diverges at the prefix boundary (248078 vs (62724, 60), and the first
    content token behind it), so the plugin refuses document ids until the product's id
    derivation gains the split-parse seam the notes name. When that seam lands and the recipe
    adopts it, THIS TEST FLIPS: assert equality there and delete the divergence lines.
    """
    query = "what drives scientific breakthroughs"
    marker = tokenizer.special_text("chunk_sep")
    document = marker.join(["first chunk text", "second chunk text"])

    client_query_ids = tokenizer.ids(f"[Q] {query}")
    client_document_ids = tokenizer.ids(f"[D] {document}")
    reference_query_ids = [QUERY_PREFIX_ID, *_split_ids(tokenizer, query)]
    reference_document_ids = _split_ids(tokenizer, f"[D] {document}")

    # The query leg is the reference's id-level render TODAY (measured parity).
    assert client_query_ids == reference_query_ids
    # The document leg is not: both halves of the divergence are pinned here.
    assert reference_document_ids[:2] == list(DOCUMENT_PREFIX_LITERAL_IDS)
    assert client_document_ids[0] == DOCUMENT_PREFIX_ID
    assert client_document_ids != reference_document_ids
    # Beyond the prefix boundary the marker keeps its id in both parses (a non-special
    # added token survives the split toggle), which is why the plugin can segment the
    # reference's id space at all.
    assert reference_document_ids.count(BOUNDARY_ID) == 1


def test_stage1_passes_on_cpu(tmp_path: Path, tokenizer) -> None:
    """Stage 1 on CPU: the product's fit, the anchor audit, the reference render - all green.

    Twenty pairs rows are sampled, plus the harness's own over-length inputs
    (five per declared shape, padded in that shape's own content span, cut by the
    product's budget mechanism with every anchor reserved). The wire is ``token_ids``: the anchor
    audit reads the sent id lists as sent, the render check compares them with the product
    tokenizer's ids of the reference's text (the engine ``/tokenize`` check needs an engine).
    """
    recipe = load_recipe(RECIPE_ID, root=_probe_recipe(tmp_path / "probe").parent)
    pairs_path = tmp_path / "pairs.jsonl"
    pairs_path.write_text("".join(json.dumps(row) + "\n" for row in _pairs(tokenizer)), encoding="utf-8")
    document = stage1_prompts(recipe, pairs_path, sys.executable, over_length_per_shape=5)
    assert document["pairs"] == N_PAIRS
    assert document["sampled"] == N_PAIRS + 2 * 5
    assert document["anchor_check"]["passed"] is True, document["anchor_check"]["failures"][:1]
    # Every sampled row's query and document body audited on its sent ids: a query and a document per
    # pairs row, plus the five over-length samples of each shape.
    assert document["anchor_check"]["checked"] == 2 * N_PAIRS + 2 * 5
    assert document["render_check"]["status"] == "run"
    assert document["render_check"]["rows"] == 2 * N_PAIRS  # the pairs rows, compared on ids per shape
    assert document["render_check"]["passed"] is True, document["render_check"]["failures"][:2]
    assert document["template_render_check"] is None  # no chat_template: the route applies none
    engine = document["engine_tokenize_check"]
    assert engine["status"] == "not_run" and engine["passed"] is None
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
    healthy = stage1_prompts(
        load_recipe(RECIPE_ID, root=_probe_recipe(tmp_path / "probe").parent), pairs_path, None, over_length_per_shape=1
    )
    assert healthy["anchor_check"]["passed"] is True  # positive control: the audit is not vacuously red

    def mutate(data: dict) -> dict:
        template = data["client"]["template"]
        for shape in ("query", "document"):
            template[shape] = [segment for segment in template[shape] if "content" in segment]
        return data

    # the probe copy bounds the answer (dim 8, 2048-token budget): the full-budget probe would be
    # half a million fake draws; the shipped budget is pinned by test_recipe_contract
    broken = load_recipe(RECIPE_ID, root=_probe_recipe(tmp_path / "mutant", mutate).parent)
    red = stage1_prompts(broken, pairs_path, None, over_length_per_shape=1)
    assert red["anchor_check"]["passed"] is False
    failures = red["anchor_check"]["failures"]
    assert failures, "the audit must name the shape it failed"
    # The head audit read the sent ids and found the role prefix missing, on both shapes.
    assert {failure["check"] for failure in failures} == {"head"}
    assert {failure["shape"] for failure in failures} == {"query", "document"}


def test_embed_takes_the_checkpoint_from_the_resolved_recipe(monkeypatch: pytest.MonkeyPatch) -> None:
    """The checkpoint identity comes from the resolved recipe (``--recipe``), not the module constants.

    A variant row pointing at another checkpoint must load THAT checkpoint; the stubbed heavy modules
    keep the test offline (the harness process imports no torch).
    """
    import importlib.util
    import types

    import numpy as np

    calls: list[dict] = []

    class _FakeModel:
        def to(self, device):
            return self

        def eval(self):
            return self

        def encode_queries(self, texts, normalize_embeddings=False):  # noqa: ARG002
            return np.zeros((1, 2, 3), dtype=np.float32)

        def encode(self, texts, normalize_embeddings=False):  # noqa: ARG002
            return np.zeros((1, 2, 3), dtype=np.float32)

    def fake_from_pretrained(name, **kwargs):
        calls.append({"name": name, **kwargs})
        return _FakeModel()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(AutoModel=types.SimpleNamespace(from_pretrained=fake_from_pretrained)),
    )
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace())
    spec = importlib.util.spec_from_file_location("pplx_context_reference", RECIPES / "reference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.embed(
        [{"query": "q", "documents": ["d"]}], "cuda", repo="example-org/other-checkpoint", revision="0" * 40
    )
    assert result["rows"], result
    assert calls[0]["name"] == "example-org/other-checkpoint"
    assert calls[0]["revision"] == "0" * 40
    assert calls[0]["trust_remote_code"] is True


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
            "--recipe",
            str(_resolved_recipe_file(tmp_path)),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode != 0, completed.stdout
    assert "GPU host" in completed.stderr
    assert not out_path.exists()
