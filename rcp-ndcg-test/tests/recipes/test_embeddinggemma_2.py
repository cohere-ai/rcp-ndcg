"""The embeddinggemma-2 recipe: the contract pinned, stage 1 on CPU, and the media stage against the card.

Every test here is network-gated by the directory's conftest (the client's tokenizer is the Hub spec the
recipe pins). The contract test pins EVERY field of the resolved ``serve``, ``client`` and ``reference``
blocks through the shared :func:`._contract.assert_recipe_contract`, and two drift mutants are shown red.
Stage 1 runs the harness's own checks against the real tokenizer (the fit renders, the anchor audit, the
engine ``/tokenize`` against the stub) plus the reference subprocess's render mode. The media stage holds
every image of the media request set to the checkpoint's Gemma 4 resize and every video to the engine's
pinned frame count -- the client's own decisions, read through the product's role client, never re-derived.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import re
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Any

import pytest
import yaml
from rcp_ndcg_test.equivalence import stage1_prompts
from rcp_ndcg_vllm import load_recipe, serve_argv
from rcp_ndcg_vllm.recipe import default_recipes_root

from ._contract import assert_recipe_contract
from ._served import fetch_tokenizer

RECIPE_DIR = default_recipes_root() / "embeddinggemma-2"
RECIPE_ID = "embeddinggemma-2"
MODEL = "google/embeddinggemma-2"
REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
TOKENIZER_SHA256 = "4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4"
CHAT_TEMPLATE_SHA256 = "4b852efc0b9960283e735363331e6f325b33bc74bdbaa076f595bc4e9b94d85e"
OVER_LENGTH_PER_SHAPE = 5
QUERY_PROMPT = "task: search result | query: "
DOC_PROMPT = "title: none | text: "

#: The resolved blocks the contract pins (the product's ``model_dump(mode="json")`` shape): every field of
#: ``serve``, ``client`` (minus the runtime ``base_url``) and ``reference``, defaults included.
SERVE = {
    "patches": [],
    "runner": "pooling",
    "convert": None,
    "hf_overrides": {"is_matryoshka": True, "matryoshka_dimensions": [128, 256, 512, 768]},
    "chat_template": None,
    "pooler_config": {},
    "trust_remote_code": False,
    "max_model_len": 8192,
    "dtype": "bfloat16",
    "plugin": None,
    "plugin_architectures": [],
    "io_processor_plugin": None,
    "mm_processor_kwargs": {},
    "limit_mm_per_prompt": {"image": 1, "video": 1},
    "extra_args": ["--media-io-kwargs", '{"video": {"fps": 60, "max_frames": 32}}'],
}
CLIENT = {
    "api": "openai_embeddings",
    "instruction": "none",
    "request_shape": "messages",
    "max_tokens": 8192,
    "query_prompt": QUERY_PROMPT,
    "doc_prompt": DOC_PROMPT,
    "template": {
        "query": [{"content": "query"}],
        "document": [{"content": "document"}],
        "anchor": "mean",
        "add_special_tokens": True,
    },
    "image_processor": "gemma4",
    "image_policy": {"max_soft_tokens": 280},
    "max_images": 1,
    "max_videos": 1,
    "video_policy": {"num_frames": 32, "wire": "video_url", "engine_video_pinning": True},
    "on_overflow": "cut",
    "empty_doc": "send",
    "normalize": True,
    "mrl_kind": "truncation",
    "mrl_dims": [128, 256, 512, 768],
    "model": RECIPE_ID,
    "revision": REVISION,
    "tokenizer": f"{MODEL}@{REVISION}",
}
REFERENCE = {
    "attn_implementation": None,
    "kind": "sentence_transformers",
    "score_scale": "cosine",
    "entry": "reference.py",
    "known_deviations": ["over_cap_cut_differs"],
    "device": None,  # the schema default
}
TOP = {
    "id": RECIPE_ID,
    "model": MODEL,
    "revision": REVISION,
    "role": "embed",
    "input": ["text", "image", "video"],
    "licence": "apache-2.0",
}

_PAIRS: list[dict[str, Any]] = [
    {"query": "what causes the northern lights", "documents": ["Charged particles from the sun excite gases."]},
    {"query": "how do vaccines work", "documents": ["A vaccine trains the immune system to recognise a pathogen."]},
    {"query": "what is a semaphore", "documents": ["A semaphore caps how many workers hold a resource at once."]},
    {"query": "largest planet in the solar system", "documents": ["Jupiter is the largest planet."]},
    {"query": "first crewed lunar landing", "documents": ["The first crewed lunar landing happened in July 1969."]},
    {"query": "what does DNA encode", "documents": ["DNA encodes proteins as nucleotide base sequences."]},
    {"query": "why is the sky blue", "documents": ["Short wavelengths scatter more strongly in the atmosphere."]},
    {"query": "boiling point of water", "documents": ["Water boils at one hundred degrees Celsius."]},
]


def _pairs(path: Path, rows: list[dict[str, Any]]) -> Path:
    path = path / "pairs.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _reference_module() -> Any:
    """The recipe's reference.py as a module (its top level imports only the standard library)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("embeddinggemma2_reference", RECIPE_DIR / "reference.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The pinned revision's tokenizer.json, downloaded once for the module through the shared
    :func:`._served.fetch_tokenizer` (its own urllib download, so a worker's ``HF_HUB_OFFLINE`` from
    another module cannot reach it); hash-pinned, cached in ``$RCP_NDCG_VLLM_TOKENIZER_CACHE``."""
    url = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/tokenizer.json"
    return fetch_tokenizer(
        url,
        "embeddinggemma-2/tokenizer.json",
        tmp_path_factory.mktemp("embeddinggemma-tokenizer"),
        sha256=TOKENIZER_SHA256,
    )


@pytest.fixture(scope="module")
def chat_template(tmp_path_factory: pytest.TempPathFactory) -> str:
    """The checkpoint's own chat_template.jinja at the pinned revision (hash-pinned, shared cache)."""
    url = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/chat_template.jinja"
    path = fetch_tokenizer(
        url,
        "embeddinggemma-2/chat_template.jinja",
        tmp_path_factory.mktemp("embeddinggemma-template"),
        sha256=CHAT_TEMPLATE_SHA256,
    )
    return path.read_text(encoding="utf-8")


@pytest.fixture
def _seed_checkpoint_template(_tokenizer_cache: None, chat_template: str) -> None:
    """Seed the checkpoint's chat template into the Hub cache the harness reads it from.

    ``_messages_template_check`` reads the checkpoint's own template through ``hf_hub_download``, and a
    sibling recipe module's import can put a worker into Hub-offline mode; seeding the fetched bytes makes
    the check resolve offline.  Requested by the two tests that run the harness (stage 1 and the media
    stage), never autouse: the module's offline pins must not be gated on a Hub download.  The conftest's
    ``HF_HOME`` fixture runs first (a dependency), so the cache path is the test's own."""
    import huggingface_hub.constants as constants

    snapshot = Path(constants.HF_HUB_CACHE) / f"models--{MODEL.replace('/', '--')}" / "snapshots" / REVISION
    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot / "chat_template.jinja").write_text(chat_template, encoding="utf-8")


@pytest.fixture(scope="module")
def recipe_cpu(tmp_path_factory: pytest.TempPathFactory, tokenizer: Path) -> Any:
    """The shipped recipe, loaded from a pytest-managed copy whose client.tokenizer names the downloaded
    tokenizer file (the recipe itself pins the Hub repository id and revision; the bytes are hash-equal)."""
    target = tmp_path_factory.mktemp("embeddinggemma-recipe") / RECIPE_ID
    shutil.copytree(RECIPE_DIR, target)
    data = yaml.safe_load((target / "family.yaml").read_text(encoding="utf-8"))
    data["client"]["tokenizer"] = str(tokenizer)
    (target / "family.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return load_recipe(RECIPE_ID, root=target.parent)


def test_recipe_contract_pins_every_field() -> None:
    """Every field of the resolved serve/client/reference blocks, plus the top-level facts, pinned exactly
    (the shared helper is exact in both directions: a drifted value and an unpinned field both fail)."""
    recipe = load_recipe(RECIPE_DIR)
    assert_recipe_contract(recipe, serve=SERVE, client=CLIENT, reference=REFERENCE, top=TOP)
    assert recipe.engine.image.startswith("vllm/vllm-openai:nightly-8cbd5d03")
    assert "@sha256:" in recipe.engine.image, "a nightly image is pinned by digest (decision 38)"
    assert recipe.engine.min_version == "0.31.1.dev0"
    assert not (RECIPE_DIR / "template.jinja").exists(), "no template file ships (serve.chat_template null)"


def test_two_contract_mutants_are_red() -> None:
    """A drifted serve field and a drifted reference field each red the contract pin, naming the field."""
    recipe = load_recipe(RECIPE_DIR)
    serve_mutant = recipe.model_copy(update={"serve": recipe.serve.model_copy(update={"max_model_len": 16384})})
    with pytest.raises(AssertionError, match=r"serve\.max_model_len"):
        assert_recipe_contract(serve_mutant, serve=SERVE, client=CLIENT, reference=REFERENCE, top=TOP)
    reference_mutant = recipe.model_copy(
        update={"reference": recipe.reference.model_copy(update={"kind": "transformers"})}
    )
    with pytest.raises(AssertionError, match=r"reference\.kind"):
        assert_recipe_contract(reference_mutant, serve=SERVE, client=CLIENT, reference=REFERENCE, top=TOP)


def test_serve_argv_carries_the_pinned_flags() -> None:
    """The argv the wave runner renders: no template, the pooling runner, the 8K context, the media limit
    and the video policy's fps/max_frames pin."""
    recipe = load_recipe(RECIPE_DIR)
    argv = serve_argv(recipe, port=8100, served_model_name=recipe.id)
    assert "--chat-template" not in argv
    assert "--trust-remote-code" not in argv
    assert argv[argv.index("--runner") + 1] == "pooling"
    assert argv[argv.index("--max-model-len") + 1] == "8192"
    assert argv[argv.index("--dtype") + 1] == "bfloat16"
    assert json.loads(argv[argv.index("--limit-mm-per-prompt") + 1]) == {"image": 1, "video": 1}
    assert argv[argv.index("--media-io-kwargs") + 1] == '{"video": {"fps": 60, "max_frames": 32}}'


def test_the_client_block_builds_the_product_endpoint() -> None:
    """The client block is the product's own endpoint config: the product validates it (prompts beside a
    content-only template, the gemma4 soft-token policy, the pinned video policy)."""
    from rcp_ndcg_vllm.recipe import client_config

    from rcp_ndcg.inference.config import EmbeddingEndpoint

    recipe = load_recipe(RECIPE_DIR)
    endpoint = EmbeddingEndpoint(**client_config(recipe, base_url="http://127.0.0.1:8100/v1"))
    assert endpoint.query_prompt == QUERY_PROMPT and endpoint.doc_prompt == DOC_PROMPT
    assert endpoint.image_processor == "gemma4" and endpoint.image_policy is not None
    assert endpoint.image_policy.max_soft_tokens == 280 and endpoint.image_policy.is_native is False
    assert endpoint.video_policy is not None and endpoint.video_policy.num_frames == 32


def test_the_reference_geometry_and_frame_sampling_match_the_checkpoint() -> None:
    """The reference's Gemma 4 resize and frame sampling, pinned against the checkpoint's processor facts
    (an icon scales UP to 768x768; a 2480x3508 page scales down to 672x912; a 4 s clip at the pinned
    60 fps cap shows 32 frames)."""
    module = _reference_module()
    assert module.gemma4_resize(16, 16, max_soft_tokens=280) == (768, 768)
    assert module.gemma4_resize(3508, 2480, max_soft_tokens=280) == (912, 672)
    assert module.gemma4_resize(1080, 1920, max_soft_tokens=280) == (576, 1056)
    assert module.gemma4_fixed_point(4096, 576, max_soft_tokens=280) == (2160, 288)
    assert module.gemma4_fixed_point(20, 3000, max_soft_tokens=280) == (48, 13344)
    assert module._sampled_frames({"duration_s": 4.0, "num_frames": 32, "fps": 8.0}, 60.0, 32) == 32
    assert module._sampled_frames({"duration_s": 10.0, "num_frames": 300, "fps": 30.0}, 60.0, 32) == 32
    assert module._sampled_frames({"duration_s": 0.25, "num_frames": 8, "fps": 32.0}, 60.0, 32) == 15


def test_the_reference_geometry_matches_the_product_over_a_size_grid() -> None:
    """The reference's Gemma 4 resize and fixed point are faithful ports of the product's: over a size grid
    the two copies agree exactly. (The duplication is deliberate -- the reference environment never imports
    the product -- and this test is its cross-check.)"""
    from rcp_ndcg.data.resolution import gemma4_fixed_point, gemma4_resize

    module = _reference_module()
    for height in range(16, 2048, 97):
        for width in range(16, 2048, 89):
            assert module.gemma4_resize(height, width, max_soft_tokens=280) == gemma4_resize(
                height, width, max_soft_tokens=280
            ), f"resize diverged at {height}x{width}"
            assert module.gemma4_fixed_point(height, width, max_soft_tokens=280) == gemma4_fixed_point(
                height, width, max_soft_tokens=280
            ), f"fixed point diverged at {height}x{width}"


def test_the_tokenizer_is_the_pinned_revision_bytes() -> None:
    """The recipe's tokenizer spec is covered by the vendored store (the golden test's offline tokenizer
    source) and its bytes hash to the pinned SHA-256."""
    from rcp_ndcg_test.fingerprint import stored_tokenizer, use_tokenizer_store

    store = Path(__file__).resolve().parents[2] / "corpora" / "vllm-0.31.0" / "_tokenizers"
    use_tokenizer_store(store)
    found = stored_tokenizer(f"{MODEL}@{REVISION}")
    assert found is not None, f"the tokenizer store {store} does not cover {MODEL}@{REVISION}"
    data, sha = found
    assert sha == TOKENIZER_SHA256 and hashlib.sha256(data).hexdigest() == TOKENIZER_SHA256


def test_the_reference_media_side_is_the_card_geometry() -> None:
    """One 16x16 image: the checkpoint's processor scales it to 768x768 and the prompt gets 256 pooled
    patches plus the two vision markers; the placement mirrors the client's parts (the prompt text, the
    media, the body text -- every text part where it stands)."""
    from PIL import Image

    module = _reference_module()
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), (10, 20, 30)).save(buffer, format="PNG")
    entry = {"kind": "image", "uri": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()}
    side = module.media_side(
        "caption",
        [entry],
        prompt=DOC_PROMPT,
        image_budget=280,
        video_fps=60.0,
        video_max_frames=32,
    )
    assert side["placement"] == ["text", "image", "text"]
    assert side["media"] == [{"kind": "image", "width": 768, "height": 768, "tokens": 258}]


@pytest.mark.network
def test_the_declared_prompts_are_the_checkpoint_prompts() -> None:
    """The recipe's query/document prompts are the checkpoint's own config_sentence_transformers.json
    entries at the pinned revision (the card's SearchQuery/Document prompt names)."""
    url = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/config_sentence_transformers.json"
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 - the pinned public file
        prompts = json.loads(response.read())["prompts"]
    recipe = load_recipe(RECIPE_DIR)
    assert recipe.client["query_prompt"] == prompts["SearchQuery"] == prompts["query"]
    assert recipe.client["doc_prompt"] == prompts["Document"] == prompts["document"]


@pytest.mark.network
def test_the_checkpoint_chat_template_is_the_declared_frame(tmp_path: Path) -> None:
    """The checkpoint's chat_template.jinja (hash-pinned) renders a user turn to exactly its content: no
    role markers, no default system turn, so the declared content-only frame is the engine's own render."""
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    url = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/chat_template.jinja"
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 - the pinned public file
        template_text = response.read().decode("utf-8")
    assert hashlib.sha256(template_text.encode()).hexdigest() == CHAT_TEMPLATE_SHA256
    environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    rendered = environment.from_string(template_text).render(
        messages=[{"role": "user", "content": [{"type": "text", "text": "hello world"}]}],
        add_generation_prompt=False,
    )
    assert rendered == "hello world"


@pytest.mark.network
def test_stage1_on_cpu(recipe_cpu: Any, _seed_checkpoint_template: None, tmp_path: Path) -> None:
    """The harness's stage 1 over the real tokenizer: the client's renders equal the reference's, the
    anchor audit reads every sampled input (a mean-pooling shape has no anchor token), and the
    over-length samples' cuts are audited under the declared deviation."""
    document = stage1_prompts(
        recipe_cpu,
        _pairs(tmp_path, _PAIRS),
        sys.executable,
        over_length_per_shape=OVER_LENGTH_PER_SHAPE,
    )
    assert document["anchor_check"]["passed"] is True, document["anchor_check"]["failures"][:2]
    assert document["anchor_check"]["checked"] > 0, "the audit read no input"
    assert document["render_check"]["passed"] is True, document["render_check"]["failures"][:2]
    assert document["passed"] is True


@pytest.mark.network
def test_the_media_stage_holds_the_client_to_the_card(
    recipe_cpu: Any, _seed_checkpoint_template: None, tmp_path: Path
) -> None:
    """Offline (no engine): the product's client and the card's reference agree on every image of the media
    request set -- the placement, the prepared geometry under the 280-soft-token budget and the tokens --
    and on every video's declared frame count."""
    from rcp_ndcg_test.equivalence.media import stage_media
    from rcp_ndcg_test.observe.media_set import planned_media_rows

    rows, _ = planned_media_rows(recipe_cpu)
    media_pairs = _pairs(tmp_path, [{key: row[key] for key in ("query", "documents", "media")} for row in rows])
    document = stage_media(recipe_cpu, media_pairs, sys.executable)
    assert document is not None and document["passed"] is True, document["failures"][:3]
    assert document["items"] == 13, (
        "eight image buckets, the captioned page, the mixed batch, the query image and two clips"
    )
    assert document["refusals"] == []
    assert document["engine_check"]["status"] == "not_run"


def test_shipped_recipe_files_carry_no_internal_labels() -> None:
    """Every shipped file reads as a self-contained public statement: no internal process shorthand, private
    work directory or undefined rule id."""
    hits = [
        f"{path.name}:{number}: {line.strip()[:120]}"
        for path in sorted(RECIPE_DIR.iterdir())
        if path.is_file()
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if re.search(
            r"fam-(?:dense|ctxl|vl|late)|\bsweep|lanes' base|\bresearch\b|\blanes?\b|REVIEW-LOG|\boperator\b"
            r"|\.refs/|recipe-common",
            line,
        )
    ]
    assert not hits, "\n".join(hits)


def test_the_family_yaml_is_the_only_recipe_file() -> None:
    """The family directory's shape (decision 34): family.yaml, reference.py and requirements-reference.txt,
    no standalone recipe.yaml and no template file."""
    names = sorted(path.name for path in RECIPE_DIR.iterdir() if path.is_file())
    assert names == ["family.yaml", "reference.py", "requirements-reference.txt"]
