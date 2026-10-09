"""The pplx-embed-v2-late-0.6b recipe: contract, stage 1 on CPU, the skiplist and wrapper pins.

The hub-dependent tests share one module fixture that fetches the pinned tokenizer (and the
checkpoint's small config files) once; offline or hub-less environments skip them with the
reason. Nothing downloads weights: stage 1 on CPU is a tokenizer-and-config check, never a
model run.

The contract test pins every resolved ``serve``, ``client`` and ``reference`` field through
the shared helper (``tests/recipes/_contract.py``); its two mutant tests show a drifted
recipe going red by name. Stage 1 runs the harness's CPU checks against the model's real
tokenizer; the tokenization facts the text wire rests on (the added-token role prefixes,
the MultiVectorMask skip ids, the verbatim prompts) are pinned against the downloaded
tokenizer and the pinned config files.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml
from rcp_ndcg_test.equivalence import stage1_prompts
from rcp_ndcg_test.equivalence.fitting import load_pairs
from rcp_ndcg_vllm import load_recipe
from rcp_ndcg_vllm.recipe import serve_argv

from ._contract import assert_recipe_contract
from ._served import served_rows, served_texts, tokenizer_cache

RECIPE_ID = "pplx-embed-v2-late-0.6b"
REVISION = "8fc2de24534aa3610d85fa59c463313a5f096455"
TOKENIZER_SPEC = f"perplexity-ai/{RECIPE_ID}@{REVISION}"
MODEL = "perplexity-ai/pplx-embed-v2-late-0.6b"

RECIPES = Path(__file__).resolve().parents[3] / "src" / "rcp_ndcg_vllm" / "recipes" / RECIPE_ID
QUERY_HEAD = "[Q] "  # config_sentence_transformers.json prompts.query at the pinned revision
DOCUMENT_HEAD = "[D] "  # prompts.document
QUERY_PREFIX_ID = 248077  # the added special id "[Q] " renders as (tokenizer.json added_tokens)
DOCUMENT_PREFIX_ID = 248078  # the added special id "[D] " renders as
N_PAIRS = 20


@pytest.fixture
def hub_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The Hub cache for the downloads: the declared tokenizer cache, else this test's ``tmp_path``
    (the conftest's rule, with HF_HUB_CACHE bound beside the conftest's HF_HOME so
    ``huggingface_hub``'s cache resolves here too -- per test, never process-global)."""
    cache = tokenizer_cache(tmp_path / "tokenizer-cache") / "hf-cache"
    monkeypatch.setenv("HF_HOME", str(cache))
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    return cache


# Twenty realistic retrieval pairs; several documents carry standalone punctuation, which
# is what makes the MultiVectorMask's document-side skiplist load-bearing (see
# test_document_skip_ids_match_the_multivector_mask).
_PAIRS: list[tuple[str, str]] = [
    ("what statute governs limitations?", "A statute of limitations sets the deadline to file a suit."),
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


@pytest.fixture
def tokenizer(hub_cache):
    """The recipe's tokenizer (the product's loader, its Hub cache); skip when it cannot be fetched."""
    try:
        from rcp_ndcg.data.tokenizer import load_tokenizer

        return load_tokenizer(TOKENIZER_SPEC)
    except Exception as error:  # offline CI, or the hub extra is not installed
        pytest.skip(f"cannot fetch the recipe tokenizer {TOKENIZER_SPEC!r} ({error}); stage 1 on CPU is skipped")


@pytest.fixture
def checkpoint(tokenizer, hub_cache) -> dict:
    """The pinned checkpoint's small config files (config.json, config_sentence_transformers.json,
    sentence_bert_config.json, chat_template.jinja, tokenizer_config.json).

    Render mode needs only these (no weights, no torch); the tokenizer fixture has already
    proven the hub is reachable, so a failure here is a real fetch error, reported as such.
    """
    from huggingface_hub import hf_hub_download

    def fetch(name: str) -> dict | str:
        path = hf_hub_download(MODEL, name, revision=REVISION)
        text = Path(path).read_text(encoding="utf-8")
        return json.loads(text) if name.endswith(".json") else text

    return {
        "config": fetch("config.json"),
        "config_st": fetch("config_sentence_transformers.json"),
        "sentence_bert": fetch("sentence_bert_config.json"),
        "mask_config": fetch("2_MultiVectorMask/config.json"),
        "chat_template": fetch("chat_template.jinja"),
        "tokenizer_config": fetch("tokenizer_config.json"),
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
    tmp_path.mkdir(parents=True, exist_ok=True)
    target = tmp_path / RECIPE_ID
    target.mkdir()
    shutil.copy(RECIPES / "reference.py", target / "reference.py")
    data = yaml.safe_load((RECIPES / "recipe.yaml").read_text(encoding="utf-8"))
    (target / "recipe.yaml").write_text(yaml.safe_dump(change(data), sort_keys=False), encoding="utf-8")
    return target


def _probe_recipe(tmp_path: Path, change: Callable[[dict], dict] | None = None) -> Path:
    """A probe copy of the recipe for the offline fake: the reply-side fields bounded, the requests unchanged.

    Stage 1 audits what the role client SENDS; the offline fake's answer is scaffolding. Two
    reply-side fields of the shipped recipe make that answer heavy or unanswerable, so the
    probe copy changes exactly those two and nothing a request carries:

    - ``dim: 8`` -- the answer's size: the fake answers one ``dim``-wide vector per token (one
      seeded draw per vector), and the stage-1 samples run to 2 x 4096 tokens per probed text,
      so the shipped 128-wide answer is dead weight the audit never reads;
    - ``document_skip_token_ids: []`` -- the fake counts one token per whitespace word, not the
      recipe tokenizer's tokens, and the pooling client refuses a document answer whose vector
      count is not the count of the ids it sent (the skip ids would not align). The skip ids
      act on the reply only.

    Every assertion these tests make reads requests (texts, ids, cuts, anchors, the render
    comparison); the shipped values are pinned by ``test_recipe_contract``.
    """

    def bound(data: dict) -> dict:
        narrowed = {**data, "client": {**data["client"], "dim": 8, "document_skip_token_ids": []}}
        return change(narrowed) if change else narrowed

    return _mutated_recipe(tmp_path, bound)


EXPECTED_SERVE = {
    "runner": "pooling",
    "convert": None,
    "hf_overrides": {"embed_dim": 128},
    "chat_template": None,
    "pooler_config": {},
    "trust_remote_code": False,
    "max_model_len": 4352,
    "dtype": "bfloat16",
    "plugin": "rcp-ndcg-vllm",
    "io_processor_plugin": None,
    "mm_processor_kwargs": {"images_kwargs": {"min_pixels": 3136, "max_pixels": 1800964}},
    "limit_mm_per_prompt": {
        "image": 1,
    },
    "extra_args": [],
}

EXPECTED_CLIENT = {
    "api": "vllm_pooling",
    "model": "pplx-embed-v2-late-0.6b",
    "revision": "8fc2de24534aa3610d85fa59c463313a5f096455",
    "api_key_env": None,
    "headers_env": {},
    "concurrency": 64,
    "timeout_s": 600.0,
    "connect_timeout_s": 5.0,
    "max_retries": 2,
    "wait_on_outage_s": None,
    "image_processor": "qwen3_vl",
    "image_policy": {
        "min_px": 3136,
        "max_px": 1800964,
        "engine_pixel_pinning": True,
        "processor": None,
    },
    "video_policy": None,
    "max_images": 1,
    "max_videos": 0,
    "media_sides": ["document"],
    "recipe": (
        "vLLM v0.31.0 pooling runner; plugin-registered PplxLateMultiVectorModel "
        "(ColQwen3_5Model subclass: the 1_Dense head loaded into custom_text_proj., the zero "
        "bias marked loaded; the checkpoint's is_causal false read by vLLM); the role-prefix "
        "frame ([Q] / [D] by name in the template); raw-text wire; the 1024/4096 per-shape "
        "right cuts client-side; 32 document-side skip ids"
    ),
    "tokenizer": "perplexity-ai/pplx-embed-v2-late-0.6b@8fc2de24534aa3610d85fa59c463313a5f096455",
    "max_tokens": 4096,
    "query_max_tokens": 1024,
    "template": {
        "query": [
            {
                "fixed": "{special:[Q] }",
                "content": None,
            },
            {
                "fixed": None,
                "content": "query",
            },
        ],
        "document": [
            {
                "fixed": "{special:[D] }",
                "content": None,
            },
            {
                "fixed": None,
                "content": "document",
            },
        ],
        "pair": None,
        "anchor": "mean",
        "anchor_markers": [],
        "add_special_tokens": True,
        "normalize": [],
    },
    "on_overflow": "cut",
    "chunk": None,
    "aggregation": "max",
    "empty_doc": "send",
    "empty_doc_text": None,
    "request_shape": "text",
    "add_generation_prompt": None,
    "query_prompt": "",
    "doc_prompt": "",
    "normalize": True,
    "dimensions": None,
    "batch_size": 32,
    "embed_dtype": "float16",
    "dim": 128,
    "document_skip_token_ids": [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        58,
        59,
        60,
        61,
        62,
        63,
        90,
        91,
        92,
        93,
    ],
    "mrl_dim": None,
    "outputs": "per_token",
}

EXPECTED_REFERENCE = {
    "kind": "sentence_transformers",
    "score_scale": "cosine",
    "entry": "reference.py",
    "known_deviations": ["over_cap_cut_differs"],
    "device": None,
}

EXPECTED_ENGINE = {
    "name": "vllm",
    "image": "vllm/vllm-openai:v0.31.0",
    "min_version": "0.31.0",
    "startup_timeout_s": 1800,
}

EXPECTED_RESOURCES = {
    "gpus": 1,
}

EXPECTED_TOP = {
    "id": "pplx-embed-v2-late-0.6b",
    "licence": "MIT",
    "revision": "8fc2de24534aa3610d85fa59c463313a5f096455",
    "role": "multi_vector",
    "input": ["text", "image"],
    "model": "perplexity-ai/pplx-embed-v2-late-0.6b",
}


def test_recipe_contract() -> None:
    """Every resolved serve/client/reference field is pinned (the shared helper, both directions).

    The engine and resources blocks carry their own pin (the helper's top-level comparison is
    for scalars): min_version is the image actually verified, startup_timeout_s the schema
    default.
    """
    recipe = load_recipe(RECIPES)
    assert_recipe_contract(
        recipe,
        serve=EXPECTED_SERVE,
        client=EXPECTED_CLIENT,
        reference=EXPECTED_REFERENCE,
        top=EXPECTED_TOP,
    )
    assert recipe.engine.model_dump(mode="json") == EXPECTED_ENGINE
    assert recipe.resources.model_dump(mode="json") == EXPECTED_RESOURCES
    assert recipe.status.state == "unverified"
    assert recipe.sources


def test_contract_mutant_serve_max_model_len_is_red(tmp_path: Path) -> None:
    """Mutant 1: serve.max_model_len 4352 -> 8448 must red, naming the field (drifted upward:
    a context far above both declared budgets over-reserves KV for nothing)."""

    def mutate(data: dict) -> dict:
        data["serve"]["max_model_len"] = 8448
        return data

    drifted = load_recipe(_mutated_recipe(tmp_path / "mutant", mutate))
    with pytest.raises(AssertionError, match=r"serve\.max_model_len"):
        assert_recipe_contract(
            drifted, serve=EXPECTED_SERVE, client=EXPECTED_CLIENT, reference=EXPECTED_REFERENCE, top=EXPECTED_TOP
        )


def test_contract_mutant_reference_kind_is_red(tmp_path: Path) -> None:
    """Mutant 2: reference.kind sentence_transformers -> remote_code must red.

    remote_code would be doubly wrong: the checkpoint carries no custom code at all (the
    card: the export is native ST modules), and the reference runs the card's own path."""

    def mutate(data: dict) -> dict:
        data["reference"]["kind"] = "remote_code"
        return data

    drifted = load_recipe(_mutated_recipe(tmp_path / "mutant", mutate))
    with pytest.raises(AssertionError, match=r"reference\.kind"):
        assert_recipe_contract(
            drifted, serve=EXPECTED_SERVE, client=EXPECTED_CLIENT, reference=EXPECTED_REFERENCE, top=EXPECTED_TOP
        )


def test_serve_argv_carries_the_serving_facts() -> None:
    """The rendered argv: the revision, the declared embed_dim, one image per prompt, no
    template file, no --trust-remote-code (the config parses locally: no auto_map in the
    checkpoint's config.json), and the plugin named only through its entry point."""
    recipe = load_recipe(RECIPES)
    argv = serve_argv(recipe, port=8100, served_model_name=RECIPE_ID)
    assert argv[:3] == ["vllm", "serve", recipe.model]
    assert argv[argv.index("--revision") + 1] == REVISION
    assert argv[argv.index("--dtype") + 1] == "bfloat16"
    assert argv[argv.index("--max-model-len") + 1] == "4352"
    assert json.loads(argv[argv.index("--hf-overrides") + 1]) == {"embed_dim": 128}
    assert json.loads(argv[argv.index("--mm-processor-kwargs") + 1]) == {
        "images_kwargs": {"min_pixels": 3136, "max_pixels": 1800964}
    }
    assert argv[argv.index("--limit-mm-per-prompt") + 1] == json.dumps({"image": 1}, sort_keys=True)
    assert argv[argv.index("--pooler-config") + 1] == "{}"  # the model class builds its own pooler
    assert "--trust-remote-code" not in argv
    assert "--chat-template" not in argv  # the checkpoint's own template rides the revision
    assert not any("rcp-ndcg-vllm" in argument for argument in argv)  # the plugin never reaches the argv


def test_the_prompts_are_the_checkpoints_own(tokenizer, checkpoint) -> None:
    """The declared fixed segments resolve to the checkpoint's added tokens, byte for byte.

    The prompts come from config_sentence_transformers.json (never hardcoded); both are
    ADDED special tokens of this tokenizer, so the resolved text renders and tokenizes to
    exactly the reference's ids: [248077|248078] + body, the measured parse both sides run
    (the engine-side added-token parse and sentence-transformers' own -- no
    split_special_tokens anywhere in this checkpoint's path, unlike the contextual 9B's
    remote code, which is why THAT recipe rides token_ids and this one rides text).
    """
    assert checkpoint["config_st"]["prompts"] == {"query": QUERY_HEAD, "document": DOCUMENT_HEAD}
    assert tokenizer.special_id("[Q] ") == QUERY_PREFIX_ID
    assert tokenizer.special_id("[D] ") == DOCUMENT_PREFIX_ID
    # The wire: a text render of the fixed segment + content tokenizes to the prefix id + the
    # content's own tokens, with no post-processor additions and no re-tokenization boundary.
    query = "what drives scientific breakthroughs"
    query_ids = tokenizer.ids(f"{QUERY_HEAD}{query}", add_special_tokens=True)
    assert query_ids == [QUERY_PREFIX_ID, *tokenizer.ids(query, add_special_tokens=True)]
    document_ids = tokenizer.ids(f"{DOCUMENT_HEAD}{_PAIRS[0][1]}", add_special_tokens=True)
    assert document_ids[0] == DOCUMENT_PREFIX_ID
    assert tokenizer.ids(QUERY_HEAD, add_special_tokens=True) == [QUERY_PREFIX_ID]  # the bare prompt


def test_document_skip_ids_match_the_multivector_mask(tokenizer, checkpoint) -> None:
    """The 32 declared skip ids are exactly the checkpoint's own skiplist resolved with its
    tokenizer, and the mask asymmetry is the one the reference's MultiVectorMask applies.

    2_MultiVectorMask/config.json skiplist_words is the 32 ASCII punctuation characters;
    sentence-transformers resolves each with convert_tokens_to_ids (each is a single vocab
    token here -- an unresolved word would be dropped loudly, not silently). Documents drop
    those positions (skiplist_tasks ['document']); queries keep everything; keep_only is
    null, so no image-patch allowlist exists (unlike topk's image-side mask).
    """
    recipe = load_recipe(RECIPES)
    mask_config = checkpoint["mask_config"]
    words = mask_config["skiplist_words"]
    assert mask_config["skiplist_tasks"] == ["document"]
    assert mask_config["keep_only_token_ids"] is None
    assert len(words) == 32 and all(len(word) == 1 and ord(word) < 128 and not word.isalnum() for word in words)
    resolved = [_token_id(tokenizer, word) for word in words]
    assert sorted(resolved) == list(recipe.client.document_skip_token_ids)
    # Completeness: every single-char non-alnum ASCII token in the vocabulary is skipped...
    dropped = 0
    for token, value in tokenizer.backend.get_vocab().items():
        if len(token) == 1 and ord(token) < 128 and not token.isalnum():
            assert value in set(recipe.client.document_skip_token_ids), (value, token)
            dropped += 1
    assert dropped == 32  # ...and they are exactly the 32
    # The prefix ids are NOT skiplisted (added special tokens): both sides keep the prefix vector.
    skip = set(recipe.client.document_skip_token_ids)
    assert QUERY_PREFIX_ID not in skip and DOCUMENT_PREFIX_ID not in skip
    # Asymmetry on a real pair: the document drops positions (its own punctuation included),
    # the query keeps every token of the same characters.
    document_text = DOCUMENT_HEAD + "Q3 revenue was $12 million, up 20% year over year."
    query_text = QUERY_HEAD + "What was Q3 revenue?"
    document_ids = tokenizer.ids(document_text, add_special_tokens=True)
    query_ids = tokenizer.ids(query_text, add_special_tokens=True)
    kept_document = [value for value in document_ids if value not in skip]
    assert len(kept_document) < len(document_ids), (len(kept_document), len(document_ids))
    assert 11 in document_ids and 11 in skip  # the document's own comma is dropped document-side
    # The asymmetry is real: the same mask applied to the QUERY would drop its punctuation
    # too (the query's '?' is id 30, a skiplist word) -- the reference never does that
    # (skiplist_tasks is ['document']; queries keep everything), so one shared mask is
    # wrong in both directions, and the client ships no query-side mask at all.
    assert 30 in query_ids and 30 in skip
    assert [value for value in query_ids if value not in skip] != query_ids


def _token_id(tokenizer: Any, token_text: str) -> int:
    """The id of a single-vocab-token string, as sentence-transformers'
    ``convert_tokens_to_ids`` resolves skiplist words (the byte-level vocab key)."""
    vocab = tokenizer.backend.get_vocab()
    if token_text not in vocab:
        raise AssertionError(f"{token_text!r} is not a single vocab token (MultiVectorMask would drop it loudly)")
    return vocab[token_text]


def test_stage1_passes_on_cpu(tmp_path: Path, tokenizer) -> None:
    """Stage 1 on CPU: the product's fit, the anchor audit, the reference render - all green.

    Twenty pairs rows are sampled, plus the harness's own over-length inputs (five per
    declared shape, padded in that shape's own content span, cut by the product's budget
    mechanism with every anchor reserved). The render check compares the reference
    subprocess's renders (the card's prompts and cut) against the captured texts, zero
    tolerance; the engine /tokenize check reports not_run without an engine, never passed.
    """
    recipe = load_recipe(_probe_recipe(tmp_path))
    document = stage1_prompts(
        recipe,
        _pairs_file(tmp_path),
        sys.executable,
        over_length_per_shape=5,
    )
    assert document["sampled"] == N_PAIRS + 2 * 5, document["sampled"]
    assert document["anchor_check"]["passed"] is True, document["anchor_check"]["failures"][:1]
    assert document["anchor_check"]["checked"] == 2 * N_PAIRS + 2 * 5  # one text per shape per pairs row + samples
    assert document["render_check"]["status"] == "run"
    assert document["render_check"]["passed"] is True, document["render_check"]["failures"][:1]
    assert document["render_check"]["rows"] == 2 * N_PAIRS  # one render per declared shape per pair
    assert document["template_render_check"] is None  # no served chat template (see test_serve_argv)
    engine = document["engine_tokenize_check"]
    assert engine["status"] == "not_run" and engine["passed"] is None
    assert document["passed"] is True


def _reference_render(rows: list[dict[str, Any]], work: Path) -> dict[tuple[int, str], str]:
    """The reference subprocess's render mode over ``rows``, keyed by ``(row index, shape)``."""
    from rcp_ndcg_test.equivalence.reference import run_reference

    work.mkdir(parents=True, exist_ok=True)
    pairs = work / "pairs.jsonl"
    pairs.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    reference = run_reference(
        sys.executable,
        str(RECIPES / "reference.py"),
        mode="render",
        pairs_path=pairs,
        out_path=work / "reference.json",
        tokenizer_spec=TOKENIZER_SPEC,
    )
    return {(int(row["index"]), str(row["shape"])): str(row["text"]) for row in reference["rows"]}


def format_uncut(text: str, shape: str) -> str:
    """The fitted UNcut render of one sample's raw text (the frame + the verbatim content).

    The reference's own render of the same text is byte-identical here for EVERY input:
    sentence-transformers prepends the prompt verbatim (no strip, no normalisation corners
    -- the declared normalize is [] on both shapes).
    """
    return (QUERY_HEAD if shape == "query" else DOCUMENT_HEAD) + (text or "")


def test_over_length_fitted_render_is_a_prefix_of_the_reference_render(tmp_path: Path, tokenizer) -> None:
    """Token-id and text equality against the reference over the full sample, over-length included.

    In-budget renders are byte-identical (stage 1's render check compares the same thing).
    For the over-length samples the reference's render mode emits the uncut formatted
    prompt, so the fitted (served) text must be a verbatim prefix of it -- the cut is a text
    prefix of the content span, with the fixed head the model reads intact at the front.
    The samples are padded above the declared budgets and the fit cuts each shape at its
    own cap.
    """
    from rcp_ndcg_test.equivalence.reference import run_reference
    from rcp_ndcg_test.equivalence.stages import _sampled_rows

    recipe = load_recipe(_probe_recipe(tmp_path))
    rows = load_pairs(_pairs_file(tmp_path))
    sampled = _sampled_rows(recipe, rows, tokenizer, 5)
    fitted = served_rows(recipe, sampled, tokenizer)
    work = tmp_path / "ref"
    pairs = work / "pairs.jsonl"
    pairs.parent.mkdir(parents=True, exist_ok=True)
    pairs.write_text("".join(json.dumps(row) + "\n" for row in sampled if "shape" not in row), encoding="utf-8")
    reference = run_reference(
        sys.executable,
        str(RECIPES / "reference.py"),
        mode="render",
        pairs_path=pairs,
        out_path=work / "reference.json",
        tokenizer_spec=TOKENIZER_SPEC,
    )
    reference_text = {(int(row["index"]), str(row["shape"])): str(row["text"]) for row in reference["rows"]}
    n_over_length = 0
    for shape, body in fitted["per_shape"].items():
        head_ids = tokenizer.ids(QUERY_HEAD if shape == "query" else DOCUMENT_HEAD, add_special_tokens=True)
        budget = recipe.client.query_max_tokens if shape == "query" else recipe.client.max_tokens
        for position, text in zip(body["row_indexes"], body["texts"], strict=True):
            if position < len(rows):
                assert text == reference_text[(position, shape)]  # in-budget: byte-identical
                continue
            n_over_length += 1
            sampled_row = sampled[position]
            raw = str(sampled_row["query"]) if shape == "query" else str(sampled_row["documents"][0])
            uncut = format_uncut(raw, shape)  # the fitted uncut render of the padded sample
            assert uncut.startswith(text), (shape, text[:60])  # the cut is a verbatim prefix
            assert text.startswith(QUERY_HEAD if shape == "query" else DOCUMENT_HEAD)
            ids = tokenizer.ids(text, add_special_tokens=True)
            assert ids[: len(head_ids)] == head_ids  # the fixed head survived
            whole_count = len(tokenizer.backend.encode(uncut).ids)
            true_count = len(tokenizer.backend.encode(text).ids)
            assert whole_count > budget, (whole_count, budget, shape)  # the sample is over the budget
            assert true_count <= budget, (true_count, budget, shape)  # the fit honours it
            assert len(text) < len(uncut)  # the cut fired: a true prefix
    assert n_over_length == 10  # 5 per declared shape


def test_over_cap_pairs_rows_render_the_card_cut(tmp_path: Path, tokenizer) -> None:
    """Decision 9: the reference renders the card's own cut, and stage 1 passes on over-cap
    pairs rows.

    The card's path right-cuts the rendered prompt at query_length 1024 / document_length
    4096 (sentence-transformers' truncation='longest_first' with the per-task max_length).
    An over-cap query row and an over-cap document row: the reference's render is a verbatim
    prefix of the uncut prompt that fills the shape's cap (the fixed head kept), equal here
    to what the role client ships, and stage 1 is green.
    """
    rows = [
        {"query": "what is a lighthouse", "documents": ["A lighthouse is a tower."]},
        {"query": "lighthouse " * 1500, "documents": ["harbour lighthouse restored " * 1500]},
    ]
    reference = _reference_render(rows, tmp_path / "ref")
    recipe = load_recipe(_probe_recipe(tmp_path / "probe"))
    for shape, cap, raw in (
        ("query", 1024, rows[1]["query"]),
        ("document", 4096, rows[1]["documents"][0]),
    ):
        text = reference[(1, shape)]
        uncut = format_uncut(raw, shape)
        assert uncut.startswith(text) and len(text) < len(uncut), f"{shape}: a verbatim prefix, cut"
        assert len(tokenizer.ids(text, add_special_tokens=True)) == cap, f"{shape}: the card fills its cap"
        assert text.startswith(QUERY_HEAD if shape == "query" else DOCUMENT_HEAD), f"{shape}: the head is kept"
        assert served_texts(recipe, [raw], shape) == [text], f"{shape}: the client's cut is the card's here"
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    document = stage1_prompts(recipe, pairs, sys.executable, over_length_per_shape=1)
    assert document["render_check"]["passed"] is True, document["render_check"]["failures"][:1]
    assert document["passed"] is True


def test_the_card_cut_differs_where_it_splits_a_character(tmp_path: Path, tokenizer) -> None:
    """Why the recipe declares ``over_cap_cut_differs``: the card cuts ids, the client cuts text.

    An emoji spans several byte-level tokens; when the card's 1024th id falls inside one, the
    card's model reads that emoji's leading byte token, which no text carries. The reference's
    text keeps whole tokens of whole characters -- a strict prefix of the card's ids, the head
    kept -- and the client ships the same text here: the render check sees equal texts, while
    the model inputs differ by that one id (a stage-2 difference on the vector, inside the
    declared deviation).
    """
    raw = "emoji \U0001f680 test " * 1500
    reference = _reference_render([{"query": raw, "documents": ["d"]}], tmp_path / "ref")[(0, "query")]
    shipped = served_texts(load_recipe(_probe_recipe(tmp_path / "probe")), [raw], "query")[0]
    prompt = format_uncut(raw, "query")
    backend = tokenizer.backend
    backend.enable_truncation(max_length=1024, strategy="longest_first", direction="right")
    try:
        card_ids = backend.encode(prompt, add_special_tokens=True).ids
    finally:
        backend.no_truncation()
    reference_ids = tokenizer.ids(reference, add_special_tokens=True)
    assert prompt.startswith(reference) and reference.startswith(QUERY_HEAD)
    assert len(card_ids) == 1024 and card_ids[: len(reference_ids)] == reference_ids
    assert len(reference_ids) < len(card_ids), "the card reads an id (the emoji's leading bytes) no text carries"
    assert shipped == reference, "the client's text cut keeps the same whole tokens here"
    assert load_recipe(RECIPES).reference.known_deviations == ["over_cap_cut_differs"]


def test_empty_document_renders_the_bare_prompt_and_gates(tmp_path: Path, tokenizer) -> None:
    """The reference's empty document is the bare prompt render: one kept vector, gating exactly.

    sentence-transformers prepends the prompt verbatim, so an empty document renders "[D] "
    -- exactly one token (the prefix id 248078, which is NO skiplist word), one kept vector.
    The recipe declares empty_doc: send: the client renders the same frame around empty
    content, the counts agree on both sides, and the row gates (topk's omit_zero
    approximation has no counterpart here -- the corner simply does not exist).
    """
    recipe = load_recipe(_probe_recipe(tmp_path / "probe"))
    reference_rows = _reference_render_rows([{"query": "q", "documents": [""]}], tmp_path / "empty")
    document_text = next(row["text"] for row in reference_rows if row["shape"] == "document")
    assert document_text == DOCUMENT_HEAD  # the bare prompt, nothing else
    assert served_texts(recipe, [""], "document") == [DOCUMENT_HEAD]  # the client's send renders the same
    assert len(tokenizer.ids(document_text, add_special_tokens=True)) == 1  # exactly the prefix id
    skip = set(load_recipe(RECIPES).client.document_skip_token_ids)
    assert DOCUMENT_PREFIX_ID not in skip  # the prefix token keeps its vector on the document side


def _reference_rows(work: Path) -> list[dict[str, Any]]:
    """The reference render's raw rows from the work directory's output file."""
    return json.loads((work / "reference.json").read_text(encoding="utf-8"))["rows"]


def _reference_render_rows(rows: list[dict[str, Any]], work: Path) -> list[dict[str, Any]]:
    """Run the reference subprocess's render mode and return its rows verbatim."""
    from rcp_ndcg_test.equivalence.reference import run_reference

    work.mkdir(parents=True, exist_ok=True)
    pairs = work / "pairs.jsonl"
    pairs.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    run_reference(
        sys.executable,
        str(RECIPES / "reference.py"),
        mode="render",
        pairs_path=pairs,
        out_path=work / "reference.json",
        tokenizer_spec=TOKENIZER_SPEC,
    )
    return _reference_rows(work)


def test_image_wrapper_is_pinned(tokenizer, checkpoint) -> None:
    """The chat template's image-only user message renders the exact wrapper ids; a text part
    diverges.

    The harness's stage-1 template check cannot render a messages-based template (it renders
    query/document variables), so the recipe's test pins the image wrapper here: the ids the
    client's media path must produce through the served chat template, at the pinned
    revision. This checkpoint's template is a pass-through (system text first, then vision
    markers, then text parts; no im-start/im-end wrappers).
    """
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=False)
    environment.globals["raise_exception"] = _raise
    template = environment.from_string(checkpoint["chat_template"])
    rendered = template.render(messages=[{"role": "user", "content": [{"type": "image"}]}], add_generation_prompt=False)
    ids = tokenizer.ids(rendered, add_special_tokens=True)
    specials = {name: tokenizer.special_id(name) for name in ("vision_start", "vision_end", "image_pad")}
    expected = [specials["vision_start"], specials["image_pad"], specials["vision_end"]]
    assert ids == expected, [tokenizer.backend.decode([value]) for value in ids]
    assert ids.count(specials["image_pad"]) == 1  # exactly one image placeholder

    with_text = template.render(
        messages=[{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": DOCUMENT_HEAD + "x"}]}],
        add_generation_prompt=False,
    )
    assert tokenizer.ids(with_text, add_special_tokens=True) != ids  # the image-ONLY message is the contract

    # A system message renders first, verbatim: the reference's image-document render is the
    # [D] prompt followed by the vision markers (the ST prompt-as-system-message path).
    with_prompt = template.render(
        messages=[
            {"role": "system", "content": [{"type": "text", "text": DOCUMENT_HEAD}]},
            {"role": "user", "content": [{"type": "image"}]},
        ],
        add_generation_prompt=False,
    )
    assert with_prompt == DOCUMENT_HEAD + rendered  # the template emits the system text, then the image


def _raise(message: str) -> None:
    """The chat templates' raise_exception helper, as the engine's jinja environment provides it."""
    raise ValueError(message)


def test_cut_preserves_the_frame_head(tmp_path: Path, tokenizer) -> None:
    """At a budget the counting can see, the cut hits the content span only and the head survives.

    The test shrinks both shape budgets to 64 in a mutated copy so the cut path runs cheaply
    (the declared budgets fire too -- see the over-length test) and asserts the anchor rule on
    its output: the fixed head ("[Q] " / "[D] ") opens every cut render, and the render stays
    within the budget.
    """
    from rcp_ndcg_test.equivalence.stages import _over_length

    def small_budget(data: dict) -> dict:
        data["client"]["max_tokens"] = 64
        data["client"]["query_max_tokens"] = 64
        return data

    recipe = load_recipe(_probe_recipe(tmp_path, small_budget))
    budget = 64
    for shape in ("query", "document"):
        head = QUERY_HEAD if shape == "query" else DOCUMENT_HEAD
        head_ids = tokenizer.ids(head, add_special_tokens=True)  # the head is one added token
        seed = "What was Q3 revenue?" if shape == "query" else "Q3 revenue was $12 million."
        padded = _over_length(seed, 8192, tokenizer, 0)
        row = {
            "query": padded if shape == "query" else "What was Q3 revenue?",
            "documents": [padded if shape == "document" else "Q3 revenue was $12 million."],
            "shape": shape,
        }
        fitted = served_rows(recipe, [row], tokenizer)
        body = fitted["per_shape"][shape]
        assert body["cuts"] == 1, body  # the cut fired
        text = body["texts"][0]
        assert text.startswith(head)  # the frame's text survived the cut
        ids = tokenizer.ids(text, add_special_tokens=True)
        assert ids[: len(head_ids)] == head_ids, text[:60]  # the fixed head survived the cut
        assert len(ids) <= budget, len(ids)  # the render honours the budget


def test_mutation_anchor_to_last_makes_the_anchor_check_red(tmp_path: Path, tokenizer) -> None:
    """A wrong anchor declaration is red: `last` demands the fixed head at the tail of every render.

    The declared anchor is `mean` (per-token pooling reads every kept position, so no
    position is the anchor); flipping it to `last` must turn the audit red, proving the check
    asserts positions and does not pass vacuously.
    """

    def mutate(data: dict) -> dict:
        data["client"]["template"]["anchor"] = "last"
        return data

    recipe = load_recipe(_probe_recipe(tmp_path, mutate))
    document = stage1_prompts(recipe, _pairs_file(tmp_path), None, over_length_per_shape=1)
    assert document["anchor_check"]["passed"] is False
    failures = document["anchor_check"]["failures"]
    assert failures and failures[0]["check"] == "tail"
    assert document["passed"] is False


def test_mutation_drop_frame_segments_makes_the_render_check_red(tmp_path: Path, tokenizer) -> None:
    """Dropping the fixed head segments loses the prompts the model reads: the render check fires.

    The fixed segments are the frame the budget reserves and the prompt the model was trained
    with; dropping them (the head analogue of dropping a trailing anchor segment) must be red,
    not silent.
    """

    def mutate(data: dict) -> dict:
        data["client"]["template"]["query"] = [{"content": "query"}]
        data["client"]["template"]["document"] = [{"content": "document"}]
        return data

    recipe = load_recipe(_probe_recipe(tmp_path, mutate))
    document = stage1_prompts(recipe, _pairs_file(tmp_path), sys.executable, over_length_per_shape=1)
    assert document["render_check"]["status"] == "run"
    assert document["render_check"]["passed"] is False
    assert document["render_check"]["failures"], "the reference render must disagree with a frameless fit"
    assert document["passed"] is False


def test_shipped_recipe_files_carry_no_internal_labels() -> None:
    """Every shipped file of this recipe reads as a self-contained public statement: no
    internal process shorthand, private work directory or undefined rule id (the families'
    scan, one recipe's own file set)."""
    hits = [
        f"{path.name}:{number}: {line.strip()[:120]}"
        for path in sorted(RECIPES.iterdir())
        if path.is_file()
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if INTERNAL_LABELS.search(line)
    ]
    assert not hits, "\n".join(hits)


INTERNAL_LABELS = re.compile(
    r"p1-tail|fam-(?:dense|ctxl|vl|late)|\bsweep|lanes' base|audit-synth"
    r"|\br-(?:ctxl|jina[35]|octen|zembed1|qwen3-emb|qwen3vl-emb|qwen3vl-rer|topk|pplx)\b"
    r"|\bresearch\b|\blanes?\b|REVIEW-LOG|ANCHOR-FINDING|\bR(?!29\b)\d{1,2}\b|\bG[1-5]\b|clients-final"
    r"|\boperator\b|\b09x\b|\.refs/|recipe-common|corrections table|\bfinding #?\d|shake"
)
