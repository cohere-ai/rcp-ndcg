"""The recipe ``qwen3-vl-reranker-2b``: it validates against the product schema and pins its declared
contract, its stage 1 passes on CPU, and the anchor audit is red when the template's trailing anchor
segment is gone.

Every field of the resolved ``serve``, ``client`` and ``reference`` blocks is pinned exactly through
the shared :func:`._contract.assert_recipe_contract`, and two drift mutants are shown red. Stage 1
here runs with the model's tokenizer files only, downloaded once through the shared cache (the
conftest's network gate: every test here needs ``RCP_NDCG_NETWORK_TESTS=1``; downloads land under
``RCP_NDCG_VLLM_TOKENIZER_CACHE`` or ``tmp_path``) -- no weights, no GPU. The GPU wave (stages 2-3)
runs the harness's full sampling (>= 20 over-length inputs per shape) on the node.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest
import yaml
from jinja2 import StrictUndefined
from jinja2.sandbox import ImmutableSandboxedEnvironment
from rcp_ndcg_vllm import load_recipe, serve_argv
from rcp_ndcg_vllm.equivalence import stage1_prompts
from rcp_ndcg_vllm.equivalence.fitting import tokenizer_of

from rcp_ndcg.data.tokenizer import load_tokenizer
from rcp_ndcg.inference.config import RerankEndpoint

from ._contract import assert_recipe_contract
from ._served import served_pair, stage1_facts

RECIPE_DIR = Path(__file__).resolve().parents[2] / "recipes" / "qwen3-vl-reranker-2b"
REVISION = "4bd860ac4f15ad1897a214615cccc700f8f71818"
REPO = "Qwen/Qwen3-VL-Reranker-2B"
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
)

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
    "model": "qwen3-vl-reranker-2b",
    "revision": REVISION,
    "api_key_env": None,
    "headers_env": {},
    "concurrency": 64,
    "timeout_s": 600.0,
    "connect_timeout_s": 5.0,
    "max_retries": 2,
    "wait_on_outage_s": None,
    "image_processor": None,
    "image_policy": {"min_px": 4096, "max_px": 1310720, "processor": None},
    "video_policy": None,
    "max_images": 1,
    "max_videos": 0,
    "media_sides": ["query", "document"],
    "recipe": (
        "vllm v0.31.0 pooling/classify: Qwen3VLForSequenceClassification via as_seq_cls_model; "
        "hf_overrides {architectures, classifier_from_token [no, yes], "
        "is_original_qwen3_reranker}; served chat template template.jinja; LAST pooling with "
        "use_activation true pinned server-side and sent on the wire; mm_processor_kwargs "
        "nested images_kwargs min_pixels 4096 / max_pixels 1310720 (the one pixel-pin shape); one "
        "media item per request (limit_mm_per_prompt image=1 = max_images 1)"
    ),
    "tokenizer": f"{REPO}@{REVISION}",
    "max_tokens": 8192,
    "instruction": "none",
    "use_activation": True,
    "query_max_tokens": 4096,
    "template": {
        "query": None,
        "document": None,
        "pair": [
            {
                "fixed": (
                    "{special:im_start}system\nJudge whether the Document meets the requirements "
                    "based on the Query and the Instruct provided. Note that the answer can only be "
                    '"yes" or "no".{special:im_end}\n{special:im_start}user\n'
                    "<Instruct>: Given a search query, retrieve relevant candidates that answer "
                    "the query.<Query>:"
                ),
                "content": None,
            },
            {"fixed": None, "content": "query"},
            {"fixed": "\n<Document>:", "content": None},
            {"fixed": None, "content": "document"},
            {"fixed": "{special:im_end}\n{special:im_start}assistant\n", "content": None},
        ],
        "anchor": "last",
        "anchor_markers": [],
        "add_special_tokens": True,
        "normalize": [],
    },
    "on_overflow": "cut",
    "chunk": None,
    "aggregation": "max",
    "empty_doc": "send_text",
    "empty_doc_text": "NULL",
    "empty_query": "refuse",
    "request_shape": "text",
    "listwise": False,
    "batch_size": None,
}
REFERENCE = {
    "kind": "transformers",
    "score_scale": "probability",
    "entry": "reference.py",
    "known_deviations": ["over_cap_cut_differs"],
}
TOP = {
    "id": "qwen3-vl-reranker-2b",
    "model": REPO,
    "revision": REVISION,
    "role": "rerank",
    "scoring": "pointwise",
    "input": ["text", "image"],
    "licence": "apache-2.0",
}


def recipe() -> object:
    """The recipe as shipped, loaded and validated through the product's endpoint config."""
    return load_recipe(RECIPE_DIR)


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The model's tokenizer files at the pinned revision, downloaded once into the shared cache.

    Every file goes through the shared :func:`._served.fetch_tokenizer` into
    ``$RCP_NDCG_VLLM_TOKENIZER_CACHE`` (the lane's scratch) when set, else a pytest-managed
    directory. Skips with the reason when the Hub is unreachable (offline CI): stage 1 needs the
    tokenizer files and nothing else -- no weights, no GPU.
    """
    from ._served import fetch_tokenizer

    fallback = tmp_path_factory.mktemp("recipe-tokenizer-cache")
    paths = [
        fetch_tokenizer(
            f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}",
            f"qwen3-vl-reranker-2b/{name}",
            fallback,
        )
        for name in TOKENIZER_FILES
    ]
    return paths[0].parent


# ---------------------------------------------------------------------------
# The recipe validates, without any network: the client block IS the product's endpoint config.
# ---------------------------------------------------------------------------


def test_recipe_contract_pins_every_field() -> None:
    """Every field of the resolved serve/client/reference blocks, plus the top-level facts, pinned exactly
    (the shared helper is exact in both directions: a drifted value and an unpinned field both fail)."""
    loaded = recipe()
    assert_recipe_contract(loaded, serve=SERVE, client=CLIENT, reference=REFERENCE, top=TOP)
    assert isinstance(loaded.client, RerankEndpoint)
    template = loaded.client.template
    assert template is not None and template.shapes() == ("pair",)
    assert (RECIPE_DIR / loaded.serve.chat_template).is_file()


def test_two_contract_mutants_are_red() -> None:
    """A drifted serve field and a drifted reference field each red the contract pin, naming the field
    (the sweep's finding-9 mutants: serve.max_model_len and reference.kind)."""
    loaded = recipe()
    serve_mutant = loaded.model_copy(update={"serve": loaded.serve.model_copy(update={"max_model_len": 40960})})
    with pytest.raises(AssertionError, match=r"serve\.max_model_len"):
        assert_recipe_contract(serve_mutant, serve=SERVE, client=CLIENT, reference=REFERENCE, top=TOP)
    reference_mutant = loaded.model_copy(
        update={"reference": loaded.reference.model_copy(update={"kind": "remote_code"})}
    )
    with pytest.raises(AssertionError, match=r"reference\.kind"):
        assert_recipe_contract(reference_mutant, serve=SERVE, client=CLIENT, reference=REFERENCE, top=TOP)


def test_serve_argv_carries_the_pinned_flags() -> None:
    """The argv the wave runner renders: overrides, template, the nested media kwargs pin, the pinned
    pooler activation, the media limit, no extra flags."""
    loaded = recipe()
    argv = serve_argv(loaded, port=8100, served_model_name=loaded.id)
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


def stage1_recipe(snapshot: Path):
    """The shipped recipe with its tokenizer pointed at the downloaded snapshot (a test view).

    The shipped YAML keeps the Hub spec ``<repo>@<commit>``; the stage-1 copy reads the same
    bytes from the shared cache, so ``fit`` never needs the network.
    """
    loaded = recipe()
    return loaded.model_copy(update={"client": loaded.client.model_copy(update={"tokenizer": str(snapshot)})})


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
def test_stage1_passes_on_cpu_token_ids_anchors_and_reference_render(tmp_path: Path, snapshot: Path) -> None:
    """At least 20 sampled pairs including 5 over-length ones, with every check green."""
    loaded = stage1_recipe(snapshot)
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
    declared = loaded.client.template.render("pair", tokenizer, query=row["query"], document=row["documents"][0])
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
    tmp_path: Path, snapshot: Path
) -> None:
    """Drop the pair shape's trailing fixed segment (the assistant tail) and the declared frame no
    longer matches the file the engine renders: the stage-1 template check goes red.

    The mutated recipe still loads (its ``add_special_tokens: true`` lets a shape end with the
    document content) and the span audit stays green (it audits the wire's spans, not the frame) --
    which is exactly why ``template_render_check`` exists: the served template file still emits the
    dropped suffix and the engine would score a prompt the recipe no longer declares.
    """
    mutated_dir = tmp_path / "qwen3-vl-reranker-2b"
    mutated_dir.mkdir()
    for name in ("recipe.yaml", "template.jinja", "reference.py"):
        shutil.copyfile(RECIPE_DIR / name, mutated_dir / name)
    data = yaml.safe_load((mutated_dir / "recipe.yaml").read_text(encoding="utf-8"))
    data["client"]["tokenizer"] = str(snapshot)
    data["client"]["template"]["pair"] = data["client"]["template"]["pair"][:-1]
    (mutated_dir / "recipe.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    mutated = load_recipe(mutated_dir)
    assert mutated.client.template.segments("pair")[-1].content == "document"

    document = stage1_prompts(mutated, write_pairs(tmp_path / "pairs.jsonl"), None, over_length_per_shape=2)
    anchor = document["anchor_check"]
    assert anchor["passed"] is True, "the span audit reads the wire's spans, not the frame"
    template_check = document["template_render_check"]
    assert template_check["passed"] is False, "the file still emits the dropped suffix"
    assert template_check["failures"], "the red check names what moved"


@pytest.mark.network
def test_reference_renders_the_card_cut_not_the_client_cut(tmp_path: Path, snapshot: Path) -> None:
    """Decision 9 on this recipe: the reference's spans are the card's (its prompt, its own cut).

    Under the cap and within the query share, the card's spans equal what the role client ships. The
    two cuts then differ exactly as ``over_cap_cut_differs`` declares: an over-share query ships settled
    at its 4096-token share while the card reads it whole (the pair still fits the card); an over-cap
    document is cut by both but to different lengths (the card keeps the specials, the first
    8192 - specials non-special tokens of all but the last 5 ids, and re-appends those 5 -- the
    assistant tail, the anchor, survives on both sides).
    """
    from rcp_ndcg_vllm.equivalence.reference import run_reference

    loaded = stage1_recipe(snapshot)
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
    frame = loaded.client.template
    card_ids = tokenizer.ids(
        frame.render("pair", tokenizer, query=card[2]["query"], document=card_document), add_special_tokens=True
    )
    client_ids = tokenizer.ids(
        frame.render("pair", tokenizer, query=shipped[2]["query"], document=client_document),
        add_special_tokens=True,
    )
    assert len(client_ids) <= loaded.client.max_tokens < len(card_ids) <= loaded.client.max_tokens + 5
