"""The ``qwen3-vl-reranker`` family: every variant validates against the product schema, pins its
declared contract, passes stage 1 on CPU, and the anchor audit is red when the template's trailing anchor
segment is gone.

One module per family (decision 34), parametrized over the family's variants: the 2b and the 8b (whose
tokenizer files are byte-identical, downloaded and compared per variant). Every field of the resolved
``serve``, ``client`` and ``reference`` blocks is pinned exactly through the shared
:func:`._contract.assert_recipe_contract`, and two drift mutants per variant are shown red. Stage 1 here
runs with the model's tokenizer files only, downloaded once through the shared cache (the conftest's
network gate: every test here needs ``RCP_NDCG_NETWORK_TESTS=1``; downloads land under
``RCP_NDCG_VLLM_TOKENIZER_CACHE`` or ``tmp_path``) -- no weights, no GPU. The GPU wave (stages 2-3) runs
the harness's full sampling (>= 20 over-length inputs per shape) on the node.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
from jinja2 import StrictUndefined
from jinja2.sandbox import ImmutableSandboxedEnvironment
from rcp_ndcg_test.equivalence import stage1_prompts
from rcp_ndcg_test.equivalence.fitting import tokenizer_of
from rcp_ndcg_vllm import load_recipe, serve_argv
from rcp_ndcg_vllm.recipe import default_recipes_root

from rcp_ndcg.data.tokenizer import load_tokenizer
from tests.conftest import start_stub

from ._contract import assert_recipe_contract
from ._served import client_template, served_pair, stage1_facts

RECIPE_DIR = default_recipes_root() / "qwen3-vl-reranker"
#: The family's variants at their pinned revisions (re-checked against the Hub API; not gated). Every
#: tokenizer file (and chat_template.jinja) is byte-identical at both pins, so the same bytes serve both
#: downloads -- each variant's own revision is fetched and compared.
VARIANTS: dict[str, dict[str, str]] = {
    "qwen3-vl-reranker-2b": {
        "model": "Qwen/Qwen3-VL-Reranker-2B",
        "revision": "4bd860ac4f15ad1897a214615cccc700f8f71818",
    },
    "qwen3-vl-reranker-8b": {
        "model": "Qwen/Qwen3-VL-Reranker-8B",
        "revision": "b212dc8c91a8164aef1ea2de9c1a867611e75c04",
    },
}
VARIANT_IDS = sorted(VARIANTS)
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
)

#: The SHA-256 of each tokenizer file at both pins (the files are byte-identical between the variants;
#: the fixture asserts each variant's own download against this table).
FILE_SHA256: dict[str, str] = {
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "tokenizer_config.json": "81ec7bb9530159b326c0bef1d0b6c33d392090524014ea3f0123a3c1eb9c2af5",
    "special_tokens_map.json": "76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd",
    "added_tokens.json": "c0284b582e14987fbd3d5a2cb2bd139084371ed9acbae488829a1c900833c680",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    "chat_template.jinja": "3636d0f0bd6bef02654cdffdc447b79cb2cef8ab02cc75267345946291a489e4",
}

#: The resolved blocks the contract pins (the product's ``model_dump(mode="json")`` shape): every
#: field of ``serve``, ``client`` (minus the runtime ``base_url``) and ``reference``, defaults
#: included, so a schema default that moves reds here and is re-pinned deliberately.
SERVE = {
    "runner": "pooling",
    "convert": None,
    "hf_overrides": {
        "architectures": ["Qwen3VLForSequenceClassification"],
        "classifier_from_token": ["no", "yes"],
        "is_original_qwen3_reranker": True,
    },
    "chat_template": "template.jinja",
    "pooler_config": {"use_activation": True},
    "trust_remote_code": False,
    "max_model_len": 32768,
    "dtype": "bfloat16",
    "plugin": None,
    "io_processor_plugin": None,
    "mm_processor_kwargs": {"images_kwargs": {"min_pixels": 4096, "max_pixels": 1310720}},
    "limit_mm_per_prompt": {"image": 1},
    "extra_args": [],
}
CLIENT = {
    "api": "rerank",
    "request_shape": "text",
    "max_tokens": 8192,
    "query_max_tokens": 4096,
    "max_images": 1,
    "image_processor": "qwen3_vl",
    "image_policy": {"min_px": 4096, "max_px": 1310720, "engine_pixel_pinning": True},
    "template": {
        "pair": [
            {
                "fixed": "{special:im_start}system\n"
                "Judge whether the Document meets the requirements based on the Query and "
                'the Instruct provided. Note that the answer can only be "yes" or '
                '"no".{special:im_end}\n'
                "{special:im_start}user\n"
                "<Instruct>: Given a search query, retrieve relevant candidates that answer "
                "the query.<Query>:"
            },
            {"content": "query"},
            {"fixed": "\n<Document>:"},
            {"content": "document"},
            {"fixed": "{special:im_end}\n{special:im_start}assistant\n"},
        ],
        "anchor": "last",
        "add_special_tokens": True,
    },
    "instruction": "none",
    "use_activation": True,
    "on_overflow": "cut",
    "empty_doc": "send_text",
    "empty_doc_text": "NULL",
}
REFERENCE = {
    "kind": "transformers",
    "score_scale": "probability",
    "entry": "reference.py",
    "known_deviations": ["over_cap_cut_differs", "media_approximation"],
    "device": None,
}
TOP = {
    "role": "rerank",
    "scoring": "pointwise",
    "input": ["text", "image"],
    "licence": "apache-2.0",
}


def _expected_top(variant_id: str) -> dict[str, object]:
    facts = VARIANTS[variant_id]
    return {"id": variant_id, "model": facts["model"], "revision": facts["revision"], **TOP}


def _expected_client(variant_id: str) -> dict[str, object]:
    facts = VARIANTS[variant_id]
    return {
        **CLIENT,
        "tokenizer": f"{facts['model']}@{facts['revision']}",
        "model": variant_id,
        "revision": facts["revision"],
    }


def recipe(variant_id: str = "qwen3-vl-reranker-2b") -> object:
    """The variant as shipped, loaded and validated through the product's endpoint config."""
    return load_recipe(variant_id)


def _assert_contract(loaded: object) -> None:
    """The variant's full resolved contract: every serve/client/reference field plus the top-level facts."""
    assert_recipe_contract(
        loaded,
        serve=SERVE,
        client=_expected_client(loaded.id),
        reference=REFERENCE,
        top=_expected_top(loaded.id),
    )


def _snapshot(variant_id: str, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The variant's tokenizer files at its pinned revision, downloaded once into the shared cache.

    Every file goes through the shared :func:`._served.fetch_tokenizer` into
    ``$RCP_NDCG_VLLM_TOKENIZER_CACHE`` (the lane's scratch) when set, else a pytest-managed
    directory, and is hash-pinned, so a changed file at the pin fails here. Skips with the reason when
    the Hub is unreachable (offline CI): stage 1 needs the tokenizer files and nothing else -- no
    weights, no GPU. Every file is byte-identical at both variants' pins, and each variant's own
    revision is fetched here.
    """
    from ._served import fetch_tokenizer

    facts = VARIANTS[variant_id]
    fallback = tmp_path_factory.mktemp(f"{variant_id}-tokenizer-cache")
    paths = [
        fetch_tokenizer(
            f"https://huggingface.co/{facts['model']}/resolve/{facts['revision']}/{name}",
            f"{variant_id}/{name}",
            fallback,
            sha256=FILE_SHA256[name],
        )
        for name in TOKENIZER_FILES
    ]
    return paths[0].parent


# ---------------------------------------------------------------------------
# The recipe validates, without any network: the client block IS the product's endpoint config.
# ---------------------------------------------------------------------------


def test_the_reference_refuses_a_resolved_recipe_of_another_checkpoint(tmp_path: Path) -> None:
    """The reference reads ``--recipe``: a resolved recipe naming another checkpoint is refused.

    The checkpoint loads from the tokenizer spec's repository (the variant's ``client.tokenizer``);
    the resolved recipe is the variant's identity, so a mismatch means the harness resolved a
    different variant than this reference would serve -- refused loudly, never served silently.
    """
    import json as _json
    import subprocess

    recipe = load_recipe("qwen3-vl-reranker-2b")  # the family's first variant (a family id is never a recipe)
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


@pytest.mark.parametrize("variant_id", VARIANT_IDS)
def test_recipe_contract_pins_every_field(variant_id: str) -> None:
    """Every field of the resolved serve/client/reference blocks, plus the top-level facts, pinned exactly
    (the shared helper is exact in both directions: a drifted value and an unpinned field both fail)."""
    loaded = recipe(variant_id)
    _assert_contract(loaded)
    template = client_template(loaded)
    assert template is not None and template.shapes() == ("pair",)
    assert loaded.resources.gpus == 1  # bf16 weights + KV fit one 80 GB-class GPU at every size
    assert (RECIPE_DIR / loaded.serve.chat_template).is_file()


@pytest.mark.parametrize("variant_id", VARIANT_IDS)
def test_two_contract_mutants_are_red(variant_id: str) -> None:
    """A drifted serve field and a drifted reference field each red the contract pin, naming the field
    (the sweep's finding-9 mutants: serve.max_model_len and reference.kind), per variant."""
    loaded = recipe(variant_id)
    serve_mutant = loaded.model_copy(update={"serve": loaded.serve.model_copy(update={"max_model_len": 40960})})
    with pytest.raises(AssertionError, match=r"serve\.max_model_len"):
        _assert_contract(serve_mutant)
    reference_mutant = loaded.model_copy(
        update={"reference": loaded.reference.model_copy(update={"kind": "remote_code"})}
    )
    with pytest.raises(AssertionError, match=r"reference\.kind"):
        _assert_contract(reference_mutant)


@pytest.mark.parametrize("variant_id", VARIANT_IDS)
def test_serve_argv_carries_the_pinned_flags(variant_id: str) -> None:
    """The argv the wave runner renders: overrides, template, the nested media kwargs pin, the pinned
    pooler activation, the media limit, no extra flags, per variant."""
    loaded = recipe(variant_id)
    argv = serve_argv(loaded, port=8100, served_model_name=loaded.id)
    assert argv[:3] == ["vllm", "serve", VARIANTS[variant_id]["model"]]
    assert argv[argv.index("--revision") + 1] == VARIANTS[variant_id]["revision"]
    assert json.loads(argv[argv.index("--hf-overrides") + 1]) == {
        "architectures": ["Qwen3VLForSequenceClassification"],
        "classifier_from_token": ["no", "yes"],
        "is_original_qwen3_reranker": True,
    }
    assert json.loads(argv[argv.index("--mm-processor-kwargs") + 1]) == {
        "images_kwargs": {"min_pixels": 4096, "max_pixels": 1310720}
    }
    assert json.loads(argv[argv.index("--pooler-config") + 1]) == {"use_activation": True}
    assert json.loads(argv[argv.index("--limit-mm-per-prompt") + 1]) == {"image": 1}
    assert argv[argv.index("--chat-template") + 1] == str(RECIPE_DIR / "template.jinja")
    assert "--dtype" in argv and argv[argv.index("--dtype") + 1] == "bfloat16"
    assert argv[argv.index("--max-model-len") + 1] == "32768"


# ---------------------------------------------------------------------------
# Stage 1 on CPU: the declared shapes render to the served template file's ids, the anchors
# survive every over-length cut, and fit's render equals the reference subprocess's render.
# ---------------------------------------------------------------------------


def stage1_recipe(variant_id: str, snapshot: Path):
    """The variant with its tokenizer pointed at the downloaded snapshot (a test view).

    The shipped YAML keeps the Hub spec ``<repo>@<commit>``; the stage-1 copy reads the same
    bytes from the shared cache, so ``fit`` never needs the network.
    """
    loaded = recipe(variant_id)
    return loaded.model_copy(update={"client": {**loaded.client, "tokenizer": str(snapshot)}})


def write_pairs(path: Path) -> Path:
    """Twenty text pairs: sixteen short rows and four with documents of a few hundred tokens."""
    filler = "the retrieval pipeline scores this document against the query under the declared budget. "
    rows = []
    for index in range(16):
        rows.append(
            {
                "query": f"what is the capital of country {index}",
                "documents": [
                    f"country {index} sits on the coast and its capital hosts the main harbour {index}",
                    f"a short note about country {index}",
                ],
            }
        )
    for index in range(4):
        rows.append(
            {
                "query": f"long document stress test {index}",
                "documents": [f"{filler} passage {index} carries the evidence near the end of the text. " * 12],
            }
        )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


@pytest.mark.network
@pytest.mark.parametrize("variant_id", VARIANT_IDS)
def test_stage1_passes_on_cpu_token_ids_anchors_and_reference_render(
    variant_id: str, tmp_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """At least 20 sampled pairs including 5 over-length ones, with every check green, per variant."""
    snapshot = _snapshot(variant_id, tmp_path_factory)
    loaded = stage1_recipe(variant_id, snapshot)
    pairs = write_pairs(tmp_path / "pairs.jsonl")
    document = stage1_prompts(
        loaded, pairs, sys.executable, over_length_per_shape=5
    )  # fmt: skip
    assert document["passed"] is True, json.dumps(document["anchor_check"]["failures"][:1])
    # The cut facts come from the role client's own capture and census (R30: what the client sends):
    # 20 pairs-file rows plus the harness's 5 padded over-length samples, and the product measured
    # the fixed frame's overhead.
    rows = [json.loads(line) for line in pairs.read_text(encoding="utf-8").splitlines()]
    facts = stage1_facts(loaded, rows, tokenizer_of(loaded), 5)
    pair = facts["per_shape"]["pair"]
    assert len(pair["spans"]) == 20 + 5  # the pairs file's rows + the over-length samples
    assert pair["overhead"] == 66  # the empty render's fixed frame, measured, in tokens
    anchor = document["anchor_check"]
    # the rerank-side audit counts every sampled row's spans (query and document spans alike)
    assert anchor["passed"] is True and anchor["checked"] >= 40
    template_check = document["template_render_check"]
    assert template_check is not None and template_check["passed"] is True
    # The token-id form of the same proof: fit's rendered ids equal the served template file's ids.
    tokenizer = tokenizer_of(loaded)
    assert load_tokenizer(str(snapshot)).sha256 == tokenizer.sha256
    row = json.loads(pairs.read_text(encoding="utf-8").splitlines()[0])
    declared = client_template(loaded).render("pair", tokenizer, query=row["query"], document=row["documents"][0])
    env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=False, undefined=StrictUndefined
    )
    jinja_render = env.from_string((RECIPE_DIR / "template.jinja").read_text(encoding="utf-8")).render(
        query=row["query"], document=row["documents"][0], instruction=""
    )
    assert tokenizer.ids(declared) == tokenizer.ids(jinja_render)
    render = document["render_check"]
    assert render["status"] == "run" and render["passed"] is True and render["rows"] == 20
    # No engine on CPU: the /tokenize check is reported not_run, never as passed.
    assert document["engine_tokenize_check"]["status"] == "not_run"
    assert document["engine_tokenize_check"]["passed"] is None


@pytest.mark.network
def test_mutation_dropping_the_tail_from_the_declared_shape_reddens_the_template_check(
    tmp_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Drop the pair shape's trailing fixed segment (the assistant tail) and the declared frame no
    longer matches the file the engine renders: the stage-1 template check goes red.

    The mutated recipe still loads (its ``add_special_tokens: true`` lets a shape end with the
    document content) and the span audit stays green (it audits the wire's spans, not the frame) --
    which is exactly why ``template_render_check`` exists: the served template file still emits the
    dropped suffix and the engine would score a prompt the recipe no longer declares.
    """
    snapshot = _snapshot("qwen3-vl-reranker-2b", tmp_path_factory)
    mutated_dir = tmp_path / "qwen3-vl-reranker"  # the family id must equal the directory name
    mutated_dir.mkdir()
    for name in ("family.yaml", "template.jinja", "reference.py"):
        shutil.copyfile(RECIPE_DIR / name, mutated_dir / name)
    data = yaml.safe_load((mutated_dir / "family.yaml").read_text(encoding="utf-8"))
    data["client"]["tokenizer"] = str(snapshot)
    data["client"]["template"]["pair"] = data["client"]["template"]["pair"][:-1]
    (mutated_dir / "family.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    mutated = load_recipe("qwen3-vl-reranker-2b", root=tmp_path)
    assert client_template(mutated).segments("pair")[-1].content == "document"

    document = stage1_prompts(mutated, write_pairs(tmp_path / "pairs.jsonl"), None, over_length_per_shape=2)
    anchor = document["anchor_check"]
    assert anchor["passed"] is True, "the span audit reads the wire's spans, not the frame"
    template_check = document["template_render_check"]
    assert template_check["passed"] is False, "the file still emits the dropped suffix"
    assert template_check["failures"], "the red check names what moved"


@pytest.mark.network
@pytest.mark.parametrize("variant_id", VARIANT_IDS)
def test_reference_renders_the_card_cut_not_the_client_cut(
    variant_id: str, tmp_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Decision 9 on this family: the reference's spans are the card's (its prompt, its own cut).

    Under the cap and within the query share, the card's spans equal what the role client ships. The
    two cuts then differ exactly as ``over_cap_cut_differs`` declares: an over-share query ships settled
    at its 4096-token share while the card reads it whole (the pair still fits the card); an over-cap
    document is cut by both but to different lengths (the card keeps the specials, the first
    8192 - specials non-special tokens of all but the last 5 ids, and re-appends those 5 -- the
    assistant tail, the anchor, survives on both sides).
    """
    from rcp_ndcg_test.equivalence.reference import run_reference

    snapshot = _snapshot(variant_id, tmp_path_factory)
    loaded = stage1_recipe(variant_id, snapshot)
    tokenizer = tokenizer_of(loaded)
    long_query = "which catalogue entry describes the harbour lighthouse restoration project " * 420
    long_document = "the harbour lighthouse was restored with funds raised by the town council. " * 900
    rows = [
        {"query": "what is the capital of France", "documents": ["Paris is the capital of France.", ""]},
        {"query": long_query, "documents": ["a short note"]},
        {"query": "lighthouse restoration", "documents": [long_document]},
    ]
    assert tokenizer.count(long_query) > 4096, "the query is over its declared share"
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    out = tmp_path / "render.json"
    run_reference(
        sys.executable,
        str(RECIPE_DIR / "reference.py"),
        mode="render",
        pairs_path=pairs,
        out_path=out,
        tokenizer_spec=str(snapshot),
        recipe=loaded,
    )
    card = {int(row["index"]): row for row in json.loads(out.read_text(encoding="utf-8"))["rows"]}
    shipped = [served_pair(loaded, row["query"], row["documents"]) for row in rows]

    assert card[0]["query"] == shipped[0]["query"] and card[0]["documents"] == shipped[0]["documents"]
    assert card[0]["documents"][1] == "NULL", "the card's empty-side rule, which empty_doc send_text sends"
    # The over-share query: settled by the client, whole for the card.
    assert card[1]["query"] == long_query
    assert long_query.startswith(shipped[1]["query"]) and len(shipped[1]["query"]) < len(long_query)
    assert tokenizer.count(shipped[1]["query"]) <= 4096
    assert card[1]["documents"] == shipped[1]["documents"] == ["a short note"]
    # The over-cap document: both cut it, each its own way; both keep the query whole here.
    assert card[2]["query"] == shipped[2]["query"] == "lighthouse restoration"
    card_document, client_document = card[2]["documents"][0], shipped[2]["documents"][0]
    assert long_document.startswith(card_document) and long_document.startswith(client_document)
    assert len(card_document) < len(long_document) and len(client_document) < len(long_document)
    assert card_document != client_document, "the two cuts differ: over_cap_cut_differs"
    frame = client_template(loaded)
    card_ids = tokenizer.ids(
        frame.render("pair", tokenizer, query=card[2]["query"], document=card_document), add_special_tokens=True
    )
    client_ids = tokenizer.ids(
        frame.render("pair", tokenizer, query=shipped[2]["query"], document=client_document),
        add_special_tokens=True,
    )
    assert len(client_ids) <= loaded.client.get("max_tokens") < len(card_ids) <= loaded.client.get("max_tokens") + 5


@pytest.mark.network
def test_a_whitespace_only_query_is_the_cards_verbatim_text(
    tmp_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """The card's format_mm_content keeps any non-empty text verbatim, so a whitespace-only query is a query:
    the client sends it (only the empty string is refused, empty_query: refuse) and the reference renders it as
    the card does -- the spans are equal."""
    from rcp_ndcg_test.equivalence.reference import run_reference

    snapshot = _snapshot("qwen3-vl-reranker-2b", tmp_path_factory)
    loaded = stage1_recipe("qwen3-vl-reranker-2b", snapshot)
    rows = [{"query": "   ", "documents": ["Paris is the capital of France."]}]
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    out = tmp_path / "render.json"
    run_reference(
        sys.executable,
        str(RECIPE_DIR / "reference.py"),
        mode="render",
        pairs_path=pairs,
        out_path=out,
        tokenizer_spec=str(snapshot),
        recipe=loaded,
    )
    (card,) = json.loads(out.read_text(encoding="utf-8"))["rows"]
    shipped = served_pair(loaded, rows[0]["query"], rows[0]["documents"])
    assert card["query"] == shipped["query"] == "   " and card["documents"] == shipped["documents"]


def _media_pairs(tmp_path: Path, loaded: Any) -> Path:
    """One text row and the media request set's rows (the generator's synthetic image buckets)."""
    from rcp_ndcg_test.observe.media_set import planned_media_rows

    rows, _ = planned_media_rows(loaded)
    text = {"query": "what is the capital of France", "documents": ["Paris is the capital of France."]}
    lines = [text, *[{key: row[key] for key in ("query", "documents", "media")} for row in rows]]
    path = tmp_path / "pairs.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in lines), encoding="utf-8")
    return path


@pytest.mark.network
def test_the_media_stage_holds_the_client_to_the_card(tmp_path: Path, tmp_path_factory: pytest.TempPathFactory) -> None:
    """Offline, the product's rerank client and the card agree on every image of the media request set (one
    pair per request): the placement, the prepared geometry under the pinned budget, the tokens. Against the
    stub engine emulating the checkpoint (factor 32, its own preprocessor_config budget 4095..1310720 px),
    the engine's media count equals the client's under the pin -- and unpinned too: the pin restates the
    checkpoint's default, so negative control (f) does not apply (its row says so, read from the checkpoint's
    own preprocessor_config.json at the pinned revision); an engine pinned to other numbers fails. (The
    family's 2b row; the 8b declares the same media policy.)"""
    from rcp_ndcg_test.equivalence.media import stage_media
    from rcp_ndcg_test.observe.controls import control_variants
    from rcp_ndcg_test.observe.media_set import planned_media_rows

    snapshot = _snapshot("qwen3-vl-reranker-2b", tmp_path_factory)
    loaded = stage1_recipe("qwen3-vl-reranker-2b", snapshot)
    pairs = _media_pairs(tmp_path, loaded)
    document = stage_media(loaded, pairs, sys.executable)
    assert document is not None and document["passed"] is True, (document["failures"][:3], document["refusals"][:2])
    # one media item per planned row (the media-inputs set: the image buckets, the captioned page,
    # the multi-image and the query-image rows, and -- where the recipe takes video -- the clips)
    planned, _ = planned_media_rows(loaded)
    assert document["items"] == len(planned)
    # Control (f) needs the checkpoint's own preprocessor budget from the Hub (or its cache). A Hub
    # that cannot be asked leaves the control ``unresolved`` -- a blocker on the node, a skip here.
    control = None
    for _ in range(3):
        (control,) = [v for v in control_variants(loaded) if v["control"] == "(f)"]
        if control["kind"] != "unresolved":
            break
        import time

        time.sleep(1.0)
    assert control is not None
    if control["kind"] == "unresolved":
        pytest.skip(f"the checkpoint's own pixel budget is not readable: {control['reason']}")
    assert control["kind"] is None and "4095-1310720" in control["reason"], control
    unpinned = loaded.model_copy(update={"serve": loaded.serve.model_copy(update={"mm_processor_kwargs": {}})})
    moved = loaded.model_copy(
        update={
            "serve": loaded.serve.model_copy(update={"mm_processor_kwargs": {"images_kwargs": {"max_pixels": 655360}}})
        }
    )
    results = {}
    for name, served in (("pinned", loaded), ("unpinned", unpinned), ("moved", moved)):
        argv = serve_argv(served, port=0, served_model_name=loaded.id)
        flags = [value for value in argv[argv.index(served.model) + 1 :] if value not in ("0.0.0.0", "--host")]
        model = ["--model-image-factor", "32", "--model-image-pixels", "4095,1310720"]
        engine = start_stub("--tokenizer", str(snapshot / "tokenizer.json"), *flags, *model)
        try:
            results[name] = stage_media(loaded, pairs, sys.executable, base_url=engine.base_url)
        finally:
            engine.stop()
    assert results["pinned"]["passed"] is True, results["pinned"]["engine_check"]
    assert results["unpinned"]["passed"] is True, results["unpinned"]["engine_check"]
    assert results["moved"]["engine_check"]["passed"] is False
