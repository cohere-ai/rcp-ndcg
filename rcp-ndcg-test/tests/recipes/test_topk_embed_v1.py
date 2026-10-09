"""The topk-embed-v1-small recipe: contract, stage 1 on CPU, the keep-mask and wrapper pins.

The hub-dependent tests share one module fixture that fetches the pinned tokenizer (and the
checkpoint's small config files) once; offline or hub-less environments skip them with the reason.
Nothing downloads weights: stage 1 on CPU is a tokenizer-and-config check, never a model run.

The contract test pins every resolved ``serve``, ``client`` and ``reference`` field through the
shared helper (``tests/recipes/_contract.py``); its two mutant tests show a drifted recipe going
red by name. Stage 1 was re-run after the G5 tokenizer-loading fix with over-length inputs above
1024 tokens (queries past the cap, documents past the old counting ceiling to 8192).
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
from rcp_ndcg_vllm.recipe import default_recipes_root, resolve_recipe, serve_argv

from rcp_ndcg.data.tokenizer import load_tokenizer

from ._contract import assert_recipe_contract
from ._served import served_rows, served_texts

RECIPE_DIR = default_recipes_root() / "topk-embed-v1"
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
    tmp_path.mkdir(parents=True, exist_ok=True)
    target = tmp_path / RECIPE_DIR.name
    target.mkdir()
    shutil.copy(RECIPE_DIR / "reference.py", target / "reference.py")
    data = yaml.safe_load((RECIPE_DIR / "family.yaml").read_text(encoding="utf-8"))
    (target / "family.yaml").write_text(yaml.safe_dump(change(data), sort_keys=False), encoding="utf-8")
    return target


def _probe_recipe(tmp_path: Path, change: Callable[[dict], dict] | None = None) -> Path:
    """A probe copy of the recipe for the offline fake: the reply-side fields bounded, the requests unchanged.

    Stage 1 audits what the role client SENDS; the offline fake's answer is scaffolding. Two reply-side
    fields of the shipped recipe make that answer heavy or unanswerable, so the probe copy changes exactly
    those two and nothing a request carries:

    - ``dim: 8`` -- the answer's size: the fake answers one ``dim``-wide vector per token (one seeded
      draw per vector, so the shipped width costs seconds, not minutes), and the stage-1 samples run to
      2 x 8192 tokens per probed text, so the shipped 2048-wide answer is a 64 MiB float32 matrix per
      text for nothing the audit reads;
    - ``document_skip_token_ids: []`` -- the fake counts one token per whitespace word, not the recipe
      tokenizer's tokens, and the pooling client refuses a document answer whose vector count is not the
      count of the ids it sent (the skip ids would not align). The skip ids act on the reply only.

    Every assertion these tests make reads requests (texts, ids, cuts, anchors, the render comparison);
    the shipped values are pinned by ``test_recipe_contract``. ``test_cut_preserves_the_frame_head``
    additionally shrinks ``max_tokens`` (its point is the budget).

    Args:
        tmp_path: the test's temporary directory (the recipe copy lives there).
        change: an optional further YAML mutation, applied after the probe bounds.
    """

    def bound(data: dict) -> dict:
        narrowed = {**data, "client": {**data["client"], "dim": 8, "document_skip_token_ids": []}}
        return change(narrowed) if change else narrowed

    return _mutated_recipe(tmp_path, bound)


def _write_reference_pairs(sampled: list[dict[str, Any]], work: Path) -> Path:
    """The pairs-file rows of a sampled set, as the pairs file the reference subprocess reads."""
    work.mkdir(parents=True, exist_ok=True)
    path = work / "pairs.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in sampled if "shape" not in row), encoding="utf-8")
    return path


EXPECTED_SERVE = {
    "runner": "pooling",
    "convert": None,
    "hf_overrides": {},
    "chat_template": None,
    "pooler_config": {},
    "trust_remote_code": False,
    "max_model_len": 8448,
    "dtype": "bfloat16",
    "plugin": "rcp-ndcg-vllm",
    "io_processor_plugin": None,
    "mm_processor_kwargs": {"images_kwargs": {"min_pixels": 65536, "max_pixels": 1310720}},
    "limit_mm_per_prompt": {
        "image": 1,
    },
    "extra_args": [],
}

EXPECTED_CLIENT = {
    "api": "vllm_pooling",
    "tokenizer": "topk-io/topk-embed-v1-small@e54485ebab921f2c18c4d092b3f4c40dcca26781",
    "max_tokens": 8192,
    "query_max_tokens": 1024,
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
        248044,
        248046,
        248053,
        248054,
        248056,
        248057,
        248070,
        248071,
        248076,
    ],
    "media_sides": ["document"],
    "image_processor": "qwen3_vl",
    "image_policy": {"min_px": 65536, "max_px": 1310720},
    "max_images": 1,
    "max_videos": 0,
    "template": {
        "query": [{"fixed": "Query: "}, {"content": "query"}],
        "document": [{"fixed": "Document: "}, {"content": "document"}],
        "anchor": "mean",
        "add_special_tokens": True,
        "normalize": ["strip"],
    },
    "on_overflow": "cut",
    "empty_doc": "omit_zero",
    "normalize": True,
    "embed_dtype": "float16",
    "dim": 2048,
    "model": "topk-embed-v1-small",
    "revision": "e54485ebab921f2c18c4d092b3f4c40dcca26781",
}

EXPECTED_REFERENCE = {
    "kind": "sentence_transformers",
    "score_scale": "cosine",
    "entry": "reference.py",
    "known_deviations": ["over_cap_cut_differs"],
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
    "id": "topk-embed-v1-small",
    "licence": "apache-2.0",
    "revision": "e54485ebab921f2c18c4d092b3f4c40dcca26781",
    "role": "multi_vector",
    "input": ["text", "image"],
    "model": "topk-io/topk-embed-v1-small",
}


def test_the_recorder_records_topks_media_row_as_sent(tmp_path: Path, tokenizer) -> None:
    """The skip rule at image positions (workstream 09): an image document rides the messages route under
    document_skip_token_ids -- the media vectors are kept whole (the render's text positions cannot be
    located client-side; the deviation is the row's processing record) -- so the recorder's model layer
    records the media request set's first row as sent, never a refusal and never an exception that would
    end the corpus step and lose the text rows with it."""
    from rcp_ndcg_test.equivalence.fitting import tokenizer_of
    from rcp_ndcg_test.observe.media_set import planned_media_rows
    from rcp_ndcg_test.record import _Collector, _model_layer

    recipe = load_recipe(_mutated_recipe(tmp_path, lambda data: {**data, "client": {**data["client"], "dim": 8}}))
    assert recipe.client.get("document_skip_token_ids"), "the media row needs the shipped skip ids"
    rows, _ = planned_media_rows(recipe)
    row = {**{key: rows[0][key] for key in ("query", "documents", "media")}, "request_id": "pairs:21"}
    collected = _Collector(recipe, tokenizer_of(recipe))
    _model_layer(recipe, "", [row], collected, (1,))
    (record,) = collected.records
    assert record["inputs"]["probe"] == "ok" and record["inputs"]["request_id"] == "pairs:21"
    assert record["response"]["status"] == 200


def test_the_reference_refuses_a_resolved_recipe_of_another_checkpoint(tmp_path: Path) -> None:
    """The reference reads ``--recipe``: a resolved recipe naming another checkpoint is refused.

    The checkpoint loads from the tokenizer spec's repository (the variant's ``client.tokenizer``);
    the resolved recipe is the variant's identity, so a mismatch means the harness resolved a
    different variant than this reference would serve -- refused loudly, never served silently.
    """
    import json as _json
    import subprocess

    recipe = load_recipe(RECIPE_DIR)
    resolved = recipe.model_dump(mode="json")
    resolved["revision"] = "0" * 40
    resolved["client"]["revision"] = "0" * 40
    recipe_file = tmp_path / "reference.recipe.json"
    recipe_file.write_text(_json.dumps(resolved), encoding="utf-8")
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text('{"query": "q", "documents": ["d"]}\n', encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(RECIPE_DIR / "reference.py"),
            "--mode",
            "render",
            "--pairs",
            str(pairs),
            "--out",
            str(tmp_path / "out.json"),
            "--tokenizer",
            str(recipe.client["tokenizer"]),
            "--recipe",
            str(recipe_file),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode != 0
    assert "would load a different checkpoint" in completed.stderr + completed.stdout


def test_recipe_contract() -> None:
    """Every resolved serve/client/reference field is pinned (the shared helper, both directions).

    The engine and resources blocks carry their own pin (the helper's top-level comparison is for
    scalars): min_version is the image actually verified, startup_timeout_s the schema default.
    """
    recipe = load_recipe(RECIPE_DIR)
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
    """Mutant 1 (the sweep reviewer's): serve.max_model_len 8448 -> 16384 must red, naming the field."""

    def mutate(data: dict) -> dict:
        data["serve"]["max_model_len"] = 16384
        return data

    drifted = load_recipe(_mutated_recipe(tmp_path / "mutant", mutate))
    with pytest.raises(AssertionError, match=r"serve\.max_model_len"):
        assert_recipe_contract(
            drifted, serve=EXPECTED_SERVE, client=EXPECTED_CLIENT, reference=EXPECTED_REFERENCE, top=EXPECTED_TOP
        )


def test_contract_mutant_reference_kind_is_red(tmp_path: Path) -> None:
    """Mutant 2 (the sweep reviewer's): reference.kind sentence_transformers -> transformers must red."""

    def mutate(data: dict) -> dict:
        data["reference"]["kind"] = "transformers"
        return data

    drifted = load_recipe(_mutated_recipe(tmp_path / "mutant", mutate))
    with pytest.raises(AssertionError, match=r"reference\.kind"):
        assert_recipe_contract(
            drifted, serve=EXPECTED_SERVE, client=EXPECTED_CLIENT, reference=EXPECTED_REFERENCE, top=EXPECTED_TOP
        )


def test_serve_argv_carries_the_serving_facts() -> None:
    """The rendered argv: the revision, the pixel cap (the nested R20 shape), one image per prompt, no
    template file.

    R10 considered (recipe notes, note 5): the checkpoint ships its own chat template at the pinned
    revision and vLLM resolves it through AutoProcessor for the pooling-chat path, so no
    --chat-template flag is rendered here; the absence is recorded by this test. And no
    --trust-remote-code: the plugin's registered config class parses config.json locally (the wave's
    fla canary proves no remote code ran).
    """
    argv = serve_argv(load_recipe(RECIPE_DIR), port=8100, served_model_name="topk-embed-v1-small")
    assert "--chat-template" not in argv
    assert "--trust-remote-code" not in argv
    assert json.loads(argv[argv.index("--mm-processor-kwargs") + 1]) == {
        "images_kwargs": {"min_pixels": 65536, "max_pixels": 1310720}
    }
    assert argv[argv.index("--limit-mm-per-prompt") + 1] == json.dumps({"image": 1}, sort_keys=True)
    assert argv[argv.index("--revision") + 1] == REVISION
    assert argv[argv.index("--max-model-len") + 1] == "8448"
    assert not any("rcp-ndcg-vllm" in argument for argument in argv)  # the plugin never reaches the argv


def test_stage1_passes_on_cpu(tmp_path: Path, tokenizer) -> None:
    """The harness's stage 1 on CPU: fits, anchor audit and the reference render all agree.

    The G5 re-run: the over-length samples are padded ABOVE 1024 tokens (the old counting ceiling;
    query samples over the 1024-budget, document samples over the 8192-budget) and the anchor and
    render checks pass on them.
    """
    recipe = load_recipe(_probe_recipe(tmp_path))
    document = stage1_prompts(
        recipe,
        _pairs_file(tmp_path),
        sys.executable,
        over_length_per_shape=5,
    )
    assert document["sampled"] >= 25, document["sampled"]  # 20 pairs + 5 over-length per shape
    assert document["anchor_check"]["passed"] is True, document["anchor_check"]["failures"][:1]
    assert document["anchor_check"]["checked"] == 2 * 20 + 10  # one text per shape per pairs row + samples
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
    with the fixed head the model reads intact at the front. The samples are padded above 1024
    tokens (the G5 re-run's range) and the fit cuts each shape at its declared budget.
    """
    from rcp_ndcg_test.equivalence.reference import run_reference
    from rcp_ndcg_test.equivalence.stages import _sampled_rows

    recipe = load_recipe(_probe_recipe(tmp_path))
    rows = load_pairs(_pairs_file(tmp_path))
    sampled = _sampled_rows(recipe, rows, tokenizer, 5)
    fitted = served_rows(recipe, sampled, tokenizer)
    work = tmp_path / "ref"
    reference = run_reference(
        sys.executable,
        str(RECIPE_DIR / "reference.py"),
        mode="render",
        pairs_path=_write_reference_pairs(sampled, work),
        out_path=work / "reference.json",
        tokenizer_spec=TOKENIZER_SPEC,
        recipe=recipe,
    )
    reference_text = {(int(row["index"]), str(row["shape"])): str(row["text"]) for row in reference["rows"]}
    n_over_length = 0
    for shape, body in fitted["per_shape"].items():
        # The head's trailing space merges into the first content token (" What"), so the stable id
        # prefix of the head is the head without its join boundary.
        head_ids = tokenizer.ids((QUERY_HEAD if shape == "query" else DOCUMENT_HEAD).rstrip(), add_special_tokens=True)
        budget = recipe.client.get("query_max_tokens") if shape == "query" else recipe.client.get("max_tokens")
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
            # The sample is genuinely over the declared budget (and over the old 1024-token
            # counting ceiling) and the cut fired there: G5 is FIXED in the product's tokenizer
            # loading (the file's embedded truncation is reset at load), so the counter sees past
            # 1024 and the fit cuts at the shape's declared budget.
            whole_count = len(tokenizer.backend.encode(uncut).ids)
            true_count = len(tokenizer.backend.encode(text).ids)
            assert whole_count > 1024, (whole_count, shape)  # the G5 re-run's range
            assert whole_count > budget, (whole_count, budget, shape)
            assert true_count <= budget, (true_count, budget, shape)
            assert len(text) < len(uncut)  # the cut fired: a true prefix
    assert n_over_length == 10  # 5 per declared shape


def _reference_render(rows: list[dict[str, Any]], work: Path) -> dict[tuple[int, str], str]:
    """The reference subprocess's render mode over ``rows``, keyed by ``(row index, shape)``."""
    from rcp_ndcg_test.equivalence.reference import run_reference

    recipe = load_recipe(RECIPE_DIR)
    work.mkdir(parents=True, exist_ok=True)
    pairs = work / "pairs.jsonl"
    pairs.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    reference = run_reference(
        sys.executable,
        str(RECIPE_DIR / "reference.py"),
        mode="render",
        pairs_path=pairs,
        out_path=work / "reference.json",
        tokenizer_spec=TOKENIZER_SPEC,
        recipe=recipe,
    )
    return {(int(row["index"]), str(row["shape"])): str(row["text"]) for row in reference["rows"]}


def test_over_cap_pairs_rows_of_each_shape_render_the_card_cut(tmp_path: Path, tokenizer) -> None:
    """Decision 9: the reference renders the card's own cut, and stage 1 passes on over-cap pairs rows.

    The card's wrapper right-cuts the formatted prompt at query_length 1024 / document_length 8192
    (topk_embed_st.py:75-77, the processor's ``truncation=True``). An over-cap query row and an over-cap
    document row: the reference's render is a verbatim prefix of the uncut prompt that fills the shape's
    cap (the fixed head kept), equal here to what the role client ships, and stage 1 is green.
    """
    rows = [
        {"query": "what is a lighthouse", "documents": ["A lighthouse is a tower."]},
        {"query": "lighthouse " * 1500, "documents": ["harbour lighthouse restored " * 3000]},
    ]
    reference = _reference_render(rows, tmp_path / "ref")
    recipe = load_recipe(_probe_recipe(tmp_path / "probe"))
    for shape, cap, raw in (
        ("query", 1024, rows[1]["query"]),
        ("document", 8192, rows[1]["documents"][0]),
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

    An emoji spans several byte-level tokens; when the card's 1024th id falls inside one, the card's model
    reads that emoji's leading byte token, which no text carries. The reference's text keeps whole tokens of
    whole characters -- a strict prefix of the card's ids, the head kept -- and the client ships the same
    text here: the render check sees equal texts, while the model inputs differ by that one id (a stage-2
    difference on the vector, inside the declared deviation).
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
    assert load_recipe(RECIPE_DIR).reference.known_deviations == ["over_cap_cut_differs"]


def _card_ids(tokenizer: Any, prompt: str, cap: int) -> list[int]:
    """The card's ids for one prompt: the right cut at ``cap`` the processor's fast tokenizer runs."""
    backend = tokenizer.backend
    backend.enable_truncation(max_length=cap, strategy="longest_first", direction="right")
    try:
        return list(backend.encode(prompt, add_special_tokens=True).ids)
    finally:
        backend.no_truncation()


@pytest.mark.parametrize("repeats", [1030, 1031, 1100])
def test_a_decomposed_character_at_the_cut_keeps_the_card_ids(tmp_path: Path, tokenizer, repeats: int) -> None:
    """NFD content at the cut: the tokenizer's NFC normaliser reports a token's offsets without its combining
    mark (" cafe" + U+0301 -> offsets cover " cafe"), so the kept text must run to the next kept token's start,
    not the last kept token's end -- otherwise the mark is lost and the text re-tokenizes to different ids.
    Both shapes: the reference's ids equal the card's exactly."""
    raw = "caf" + " cafe\u0301" * repeats
    reference = _reference_render([{"query": raw, "documents": [raw * 8]}], tmp_path / "ref")
    for shape, cap in (("query", 1024), ("document", 8192)):
        text = reference[(0, shape)]
        prompt = format_uncut(raw if shape == "query" else raw * 8, shape)
        assert prompt.startswith(text), shape
        assert tokenizer.ids(text, add_special_tokens=True) == _card_ids(tokenizer, prompt, cap), shape


def format_uncut(text: str, shape: str) -> str:
    """The fitted UNcut render of one sample's raw text (the frame + the normalised content).

    The reference's own render of the same text is byte-identical here except at the two declared
    normalisation corners (see test_the_declared_normalisation_corners); the over-length samples
    carry no edge whitespace, so the two coincide on every sample.
    """
    if shape == "query":
        return QUERY_HEAD + text.strip()
    return DOCUMENT_HEAD + text.strip()


def test_the_declared_normalisation_corners(tmp_path: Path, tokenizer) -> None:
    """The [strip] normalisation: exact on the query and the document's trailing edge; two declared
    corners where the reference's whole-render strip cannot be a content-span strip.

    The reference formats query as head + text.strip() and document as (head + text).strip()
    (topk_embed_st.py:56-60). Content-span [strip] reproduces the query render for EVERY input and
    the document render except (a) leading-whitespace content (the wrapper keeps it: the frame's own
    leading text shields it) and (b) empty / whitespace-only content ("Document:", the
    empty-document note). Those two rows are the recipe's declared divergence rows (the notes,
    "Content normalisation"); pinning both sides here is what keeps the declaration true.
    """
    recipe = load_recipe(_probe_recipe(tmp_path))
    assert served_texts(recipe, ["  what was q3 revenue?  "], "query") == [QUERY_HEAD + "what was q3 revenue?"]
    assert served_texts(recipe, ["q3 revenue was $12 million  "], "document") == [
        DOCUMENT_HEAD + "q3 revenue was $12 million"
    ]
    # Declared corner (a): leading whitespace in document content keeps on the reference side,
    # goes through the content-span strip on the fitted side -- a divergence row by declaration.
    fitted = served_texts(recipe, ["  indented passage"], "document")[0]
    reference = format_uncut_reference("  indented passage", "document")
    assert fitted == DOCUMENT_HEAD + "indented passage"
    assert reference == DOCUMENT_HEAD + "  indented passage"
    assert fitted != reference


def format_uncut_reference(text: str, shape: str) -> str:
    """The reference's own format_query/format_document of the raw text (pinned in the over-length
    test's subprocess comparison: the render mode emits exactly these two functions' output)."""
    if shape == "query":
        return QUERY_HEAD + (text or "").strip()
    return (DOCUMENT_HEAD + (text or "")).strip() or "."


def test_reference_empty_document_keeps_one_token(tmp_path: Path, tokenizer, checkpoint) -> None:
    """The reference's empty document renders as 'Document:' -- one kept token, not the eos fallback.

    The wrapper's `or eos or "."` chain is dead code for this checkpoint (the stripped render of an
    empty document is non-empty), so the reference scores MaxSim against exactly one vector. The
    recipe declares empty_doc omit_zero (never sent, 0.0) as a recorded approximation of that edge,
    never reference equivalence; this pins the fact the declaration rests on.
    """
    from rcp_ndcg_test.equivalence.reference import run_reference

    recipe = load_recipe(RECIPE_DIR)
    work = tmp_path / "empty"
    pairs_path = work / "pairs.jsonl"
    pairs_path.parent.mkdir(parents=True)
    pairs_path.write_text(json.dumps({"query": "q", "documents": [""]}) + "\n", encoding="utf-8")
    reference = run_reference(
        sys.executable,
        str(RECIPE_DIR / "reference.py"),
        mode="render",
        pairs_path=pairs_path,
        out_path=work / "reference.json",
        tokenizer_spec=TOKENIZER_SPEC,
        recipe=recipe,
    )
    document_text = next(row["text"] for row in reference["rows"] if row["shape"] == "document")
    assert document_text == DOCUMENT_HEAD.rstrip()  # "Document:": the eos fallback never fired
    config = checkpoint["config"]
    skip = {int(value) for value in config["scoring_skip_ids"]}
    kept = [value for value in tokenizer.ids(document_text, add_special_tokens=True) if value not in skip]
    assert len(kept) == 1  # exactly one kept vector for an empty document in the reference


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

    The mask is the reference's own (topk_embed_st.py:81) and the recipe now DECLARES it
    (client.document_skip_token_ids == config.json scoring_skip_ids, pinned here and in the
    contract): the client drops the vectors at the ids it sent and cross-checks the counts.
    """
    config = checkpoint["config"]
    skip = {int(value) for value in config["scoring_skip_ids"]}
    assert len(skip) == 41
    assert tuple(sorted(skip)) == tuple(load_recipe(RECIPE_DIR).client.get("document_skip_token_ids"))
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

    The test shrinks both shape budgets to 64 in a mutated copy so the cut path runs cheaply (the
    declared budgets fire too -- see the over-length test) and asserts the anchor rule on its
    output: the fixed head ("Query: " / "Document: ") opens every cut render, and the render stays
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
        head_ids = tokenizer.ids(head.rstrip(), add_special_tokens=True)  # the join boundary merges forward
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

    The declared anchor is `mean` (per-token pooling reads every kept position, so no position is the
    anchor); flipping it to `last` must turn the audit red, proving the check asserts positions and
    does not pass vacuously.
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

    The fixed segments are the frame the budget reserves and the prompt the model was trained with;
    dropping them (the head analogue of dropping a trailing anchor segment) must be red, not silent.
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


#: This family's recipes (the internal-label scan below covers both).
RECIPE_IDS = ("topk-embed-v1-small", "pplx-embed-v2-context-9b-preview")

#: Internal process labels that must not ship in a recipe (review shorthand, private work directories,
#: rule ids no public document defines) -- the dense and ctxl families' pattern plus this family's own.
#: Public rule ids (R29, documented in docs/how-to/add-a-model.md) stay allowed.
INTERNAL_LABELS = re.compile(
    r"p1-tail|fam-(?:dense|ctxl|vl|late)|\bsweep|lanes' base|audit-synth"
    r"|\br-(?:ctxl|jina[35]|octen|zembed1|qwen3-emb|qwen3vl-emb|qwen3vl-rer|topk|pplx)\b"
    r"|\bresearch\b|\blanes?\b|REVIEW-LOG|ANCHOR-FINDING|\bR(?!29\b)\d{1,2}\b|\bG[1-5]\b|clients-final"
    r"|\boperator\b|\b09x\b|\.refs/|recipe-common|corrections table|\bfinding #?\d|shake"
)


@pytest.mark.parametrize("recipe_id", RECIPE_IDS)
def test_shipped_recipe_files_carry_no_internal_labels(recipe_id: str) -> None:
    """Every shipped file of this family's recipes reads as a self-contained public statement: no
    internal process shorthand, private work directory or undefined rule id."""
    family_dir = resolve_recipe(recipe_id)._dir  # the variant's family directory (decision 34)
    assert family_dir is not None
    hits = [
        f"{path.name}:{number}: {line.strip()[:120]}"
        for path in sorted(family_dir.iterdir())
        if path.is_file()
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if INTERNAL_LABELS.search(line)
    ]
    assert not hits, "\n".join(hits)
