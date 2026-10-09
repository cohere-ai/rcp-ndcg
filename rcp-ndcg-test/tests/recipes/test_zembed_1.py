"""Stage 1 on CPU for the zembed-1-embedding recipe (``recipes/zembed-1-embedding``).

What is checked:

- the recipe loads against the product's endpoint config (offline; no tokenizer needed);
- stage 1 on CPU -- the product's ``fit`` renders, the anchor audit, and the reference subprocess's
  render -- passes token-id equality and the anchor check on a pairs file of at least 20 pairs
  including at least 5 over-cap ones. The tokenizer files are downloaded once into the shared tokenizer
  cache (``RCP_NDCG_VLLM_TOKENIZER_CACHE``, a temp dir otherwise); the tests skip with a clear reason
  when the download cannot run (offline in CI);
- the reference's render ids equal the ids the checkpoint's own remote code produces
  (``modeling_zembed.ZembedTransformer.tokenize``, run in the reference environment via
  ``RCP_ZEMBED_1_EMBEDDING_REFERENCE_PYTHON``) -- the model's published code path, not a copy;
- the mutation: dropping the template's trailing anchor segment turns the anchor check red.

The reference python is optional for stage 1 itself (the render comparison reports ``not_run``
without one, and the anchor audit still gates); the remote-code and render-equality tests need it
and skip without it. No test imports torch or transformers: the model-side code runs only in the
reference subprocess.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from rcp_ndcg_test.equivalence import stage1_prompts
from rcp_ndcg_vllm import Recipe, load_recipe
from rcp_ndcg_vllm.recipe import default_recipes_root

from rcp_ndcg.data.preprocess import TextBudget, fit
from rcp_ndcg.data.templates import TemplateSpec
from rcp_ndcg.data.tokenizer import TextTokenizer, load_tokenizer

from ._contract import assert_recipe_contract
from ._served import client_template, stage1_facts, tokenizer_cache

RECIPE_DIR = default_recipes_root() / "zembed-1"
REPO = "zeroentropy/zembed-1-embedding"
REVISION = "cf13c81f3274394053d166740294f7eea4586f7a"

CHECKPOINT_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "config_sentence_transformers.json",
    "sentence_bert_config.json",
    "modeling_zembed.py",
)
"""The tokenizer files plus the two config files (and the remote module) the reference binds to."""

MIN_PAIRS = 20
MIN_OVER_LENGTH = 5
MAX_TOKENS = 32768

_REFERENCE_PYTHON_ENV = "RCP_ZEMBED_1_EMBEDDING_REFERENCE_PYTHON"


@pytest.fixture(scope="session")
def scratch(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The shared tokenizer cache (``_served.tokenizer_cache``: re-runs reuse the download), else a
    session temp dir."""
    return tokenizer_cache(tmp_path_factory.mktemp("zembed-1-embedding"))


@pytest.fixture(scope="session")
def tokenizer_dir(scratch: Path) -> Iterator[Path]:
    """The checkpoint's tokenizer and config files, downloaded once into the tokenizer cache.

    Skips with a clear reason when the files are absent and cannot be fetched (offline in CI);
    nothing is written into the checkout.
    """
    target = scratch / f"zembed-1-embedding@{REVISION}"
    if not (target / "tokenizer.json").is_file():
        try:
            from huggingface_hub import hf_hub_download

            for name in CHECKPOINT_FILES:
                hf_hub_download(REPO, name, revision=REVISION, local_dir=target)
        except Exception as error:  # noqa: BLE001 - any fetch failure means offline: skip, not fail
            pytest.skip(
                f"offline: the {REPO}@{REVISION} tokenizer files are not cached under {target} "
                f"and the download failed ({type(error).__name__}: {error})"
            )
    yield target


@pytest.fixture(scope="session")
def recipe(tokenizer_dir: Path) -> Recipe:
    """The loaded recipe, with client.tokenizer pointed at the cached download (the shipped recipe
    keeps the Hub spec ``<repo>@<revision>``; the local copy only fixes where the files come from)."""
    loaded = load_recipe(RECIPE_DIR)
    client = {**loaded.client, "tokenizer": str(tokenizer_dir)}
    return loaded.model_copy(update={"client": client})


@pytest.fixture(scope="session")
def tokenizer(tokenizer_dir: Path) -> TextTokenizer:
    """The product's tokenizer over the downloaded tokenizer.json."""
    return load_tokenizer(str(tokenizer_dir))


def _pair_rows(tokenizer: TextTokenizer) -> list[dict[str, object]]:
    """The pairs file's rows: 15 in-budget pairs (varied lengths, one empty document, two with several
    documents) plus 6 over-cap ones (5 long documents, 1 long query and document) -- at least 20 pairs
    including at least 5 over-length."""
    unit = "Retrieval models map a query and its documents into one shared vector space. "
    long_text = unit * 64
    while tokenizer.count(long_text) < 33100:  # over the 32768-token whole-prompt cap
        long_text += unit
    rows: list[dict[str, object]] = [
        {
            "query": "What is backpropagation?",
            "documents": [
                "Backpropagation computes gradients by applying the chain rule backwards through the "
                "computation graph of the network."
            ],
        },
        {
            "query": "capital of france",
            "documents": ["Paris is the capital and largest city of France.", "Lyon is a city in east-central France."],
        },
        {
            "query": "retrieval augmented generation",
            "documents": [
                "Retrieval augmented generation grounds a language model's answers in documents fetched at query time."
            ],
        },
        {
            "query": "sentence embeddings",
            "documents": [
                "A sentence embedding maps a sentence to a fixed-size vector so cosine similarity "
                "approximates semantic similarity.",
                "Dense retrieval compares the query vector with every document vector by inner product.",
            ],
        },
        {
            "query": "kubernetes scheduling",
            "documents": ["The scheduler binds pending pods to nodes that satisfy their resource requests."],
        },
        {
            "query": "matrix factorization",
            "documents": ["Matrix factorization approximates a rating matrix as the product of two low-rank matrices."],
        },
        {
            "query": "gradient descent variants",
            "documents": [
                "Stochastic gradient descent updates parameters on one sample at a time, adding noise but scaling."
            ],
        },
        {
            "query": "attention mechanism",
            "documents": [
                "Scaled dot-product attention compares every query with every key and mixes values by the softmax."
            ],
        },
        {"query": "tokenizer byte fallback", "documents": ["字节级分词把文本拆成字节，再由字节序列构建词表。"]},
        {
            "query": "auc metric",
            "documents": [
                "The area under the ROC curve is the probability that a random positive outranks a random negative."
            ],
        },
        {"query": "empty document edge", "documents": [""]},
        {
            "query": "unicode punctuation",
            "documents": ["Well-formed text with an em—dash, a tab\tcharacter, and an accentéd vowel."],
        },
        {
            "query": "first document only",
            "documents": [
                "Only the first document is rendered for the stage-1 comparison.",
                "The second document exists but the render comparison never sees it.",
            ],
        },
        {
            "query": "numbers and units",
            "documents": ["The model runs at 32768 tokens of context and 2560 output dimensions."],
        },
        {"query": "mirror symmetry", "documents": ["level", "level"]},
        {"query": "over cap one", "documents": [long_text]},
        {"query": "over cap two", "documents": [long_text + " variant two"]},
        {
            "query": "over cap three",
            "documents": [long_text + " variant three with a slightly different tail"],
        },
        {
            "query": "over cap four",
            "documents": [long_text + " variant four, again over the whole-prompt cap"],
        },
        {
            "query": "over cap five",
            "documents": [long_text + " variant five, the fifth over-length document"],
        },
        {"query": long_text, "documents": [long_text + " both sides long", "a short second document"]},
    ]
    assert len(rows) >= MIN_PAIRS
    assert (
        sum(
            1
            for row in rows
            if tokenizer.count(str(row["query"])) > tokenizer.count("x") * MAX_TOKENS
            or any(tokenizer.count(str(document)) > MAX_TOKENS for document in row["documents"])
        )
        >= MIN_OVER_LENGTH
    ), "the pairs file must carry at least 5 over-length rows (measured roughly by chars x tokens)"
    return rows


@pytest.fixture(scope="session")
def pairs_path(tmp_path_factory: pytest.TempPathFactory, tokenizer: TextTokenizer) -> Path:
    """The pairs file, written into a session temp dir once per session."""
    path = tmp_path_factory.mktemp("zembed-1-embedding-pairs") / "pairs.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in _pair_rows(tokenizer)), encoding="utf-8")
    return path


@pytest.fixture(scope="session")
def reference_python() -> str | None:
    """The reference interpreter, when the environment names one (its python carries torch,
    transformers and sentence-transformers)."""
    return os.environ.get(_REFERENCE_PYTHON_ENV) or None


# ---------------------------------------------------------------------------
# The recipe itself (offline)
# ---------------------------------------------------------------------------


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


def test_recipe_loads_and_declares_the_serving_shape() -> None:
    """The recipe validates against the product's endpoint config, with every serving decision explicit."""
    recipe = load_recipe(RECIPE_DIR)
    _assert_contract(recipe)  # every serve, client and reference field pinned, exactly
    assert recipe.id == "zembed-1-embedding"
    assert recipe.model == REPO
    assert recipe.revision == REVISION
    assert recipe.role == "embed"
    assert recipe.input == ["text"]
    assert recipe.licence == "apache-2.0"
    assert recipe.engine.image == "vllm/vllm-openai:v0.31.0"
    assert recipe.engine.min_version == "0.31.0"
    serve = recipe.serve
    assert (serve.runner, serve.convert) == ("pooling", "embed")
    assert serve.hf_overrides == {}
    assert serve.chat_template is None, "no chat template is applied anywhere on this serving path"
    assert serve.pooler_config == {}, "the engine reads the checkpoint's pooling config itself"
    assert serve.trust_remote_code is False
    assert serve.max_model_len == MAX_TOKENS
    assert serve.dtype == "bfloat16"
    assert serve.plugin is None
    client = recipe.client
    assert client.get("api") == "openai_embeddings"
    assert client.get("tokenizer") == f"{REPO}@{REVISION}"
    assert client.get("max_tokens") == MAX_TOKENS
    assert client.get("on_overflow") == "cut"
    assert client.get("empty_doc") == "send"
    assert client.get("normalize") is True
    assert client.get("dimensions") is None, "the projections are learned, not Matryoshka: dimensions is refused"
    assert recipe.reference.kind == "sentence_transformers"
    assert recipe.reference.score_scale == "cosine"
    assert recipe.reference.known_deviations == ["anchor_drop_over_cap"]  # the model's over-cap cut drops the anchor
    assert recipe.status.state == "unverified"
    assert recipe.sources, "the recipe lists the URLs and path:line references it rests on"


def test_recipe_template_declares_both_shapes_with_named_specials() -> None:
    """The template declares the query and document shapes as data, with specials by name and the
    trailing fixed anchor segment; the YAML never types a special token literally."""
    recipe = load_recipe(RECIPE_DIR)
    template = client_template(recipe)
    assert template is not None
    assert template.shapes() == ("query", "document")
    assert template.anchor == "last"
    assert template.adds_special_tokens("query") is True
    assert template.adds_special_tokens("document") is True
    for shape in ("query", "document"):
        segments = template.segments(shape)
        assert [segment.content for segment in segments] == [None, shape, None], "one content span, two fixed"
        assert "{special:im_start}" in segments[0].fixed
        assert "{special:im_end}" in segments[0].fixed
        assert segments[-1].fixed == "{special:im_end}\n"
        assert segments[0].content is None and segments[-1].content is None
    recipe_text = (RECIPE_DIR / "family.yaml").read_text(encoding="utf-8")
    literal = "<|" + "im_end" + "|>"
    assert literal not in recipe_text, "specials are written by name, never typed literally"


# ---------------------------------------------------------------------------
# Stage 1 on CPU
# ---------------------------------------------------------------------------


def test_stage1_on_cpu_passes_anchors_and_render(
    recipe: Recipe, tokenizer: TextTokenizer, pairs_path: Path, reference_python: str | None
) -> None:
    """Stage 1 on CPU: the anchor audit, the declared overhead, and the render comparison against the
    reference subprocess -- on >= 20 pairs including >= 5 over-cap."""
    # The render needs no environment of its own (string work over the checkpoint's config files),
    # so the harness's interpreter runs it when no reference python is named.
    document = stage1_prompts(recipe, pairs_path, reference_python or sys.executable)
    assert document["pairs"] >= MIN_PAIRS
    assert document["sampled"] >= MIN_PAIRS + 2 * 20, "every declared shape sampled with over-length inputs"
    anchor = document["anchor_check"]
    assert anchor["passed"] is True, anchor["failures"][:2]
    assert anchor["checked"] >= MIN_PAIRS
    # The declared overhead and the cut census come from the role client's own capture (the seam
    # stage 1 audits; the report's older fit section is gone since the harness rewired stage 1).
    rows = [json.loads(line) for line in pairs_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    shape_facts = stage1_facts(recipe, rows, tokenizer, 20)["per_shape"]["document"]
    assert shape_facts["overhead"] == client_template(recipe).overhead("document", tokenizer)
    # the over-cap rows really were cut: the census carried them and the anchors survived anyway
    assert shape_facts["cuts"] >= MIN_OVER_LENGTH + 20
    render_check = document["render_check"]
    assert render_check["status"] == "run"
    assert render_check["passed"] is True, render_check["failures"][:2]
    assert render_check["rows"] >= 2 * MIN_PAIRS, "every pair rendered under every declared shape"


def test_fitted_renders_carry_the_anchor_and_fit_the_budget(
    recipe: Recipe, tokenizer: TextTokenizer, pairs_path: Path
) -> None:
    """Token-level audit, independent of stage 1's own: every fitted render ends with the suffix ids
    (the anchor) and counts at most max_tokens as the engine reads it."""
    template = client_template(recipe)
    suffix_ids = tokenizer.ids(tokenizer.special_text("im_end") + "\n", add_special_tokens=False)
    rows = [json.loads(line) for line in pairs_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    budget = TextBudget(
        tokenizer=str(tokenizer.name),
        max_tokens=recipe.client.get("max_tokens"),
        template=template,
        on_overflow=recipe.client.get("on_overflow"),
    )
    for shape in ("query", "document"):
        inputs = [str(row["query"]) if shape == "query" else str(row["documents"][0]) for row in rows]
        result = fit(inputs, shape, budget, tokenizer, ids=[str(index) for index in range(len(rows))])
        for text in result.texts:
            ids = tokenizer.ids(text, add_special_tokens=True)
            assert ids[-len(suffix_ids) :] == suffix_ids, f"the {shape} render lost its anchor: {text[:80]!r}"
            assert len(ids) <= recipe.client.get("max_tokens")
        assert {cut.doc_id for cut in result.cuts}, "the over-cap rows were cut, not sent whole"


def test_reference_render_ids_match_the_model_own_remote_code(
    recipe: Recipe,
    tokenizer: TextTokenizer,
    tokenizer_dir: Path,
    pairs_path: Path,
    reference_python: str | None,
    tmp_path: Path,
) -> None:
    """The declared shapes' renders tokenize to the same ids the checkpoint's own remote code produces.

    The remote module runs in the reference environment (torch, transformers,
    sentence-transformers): ``ZembedTransformer.tokenize`` with no weights -- the model's published
    suffix append and whole-prompt truncation. Under cap the ids must equal the client's; over cap
    the remote code drops the suffix anchor (the defect the recipe's cut exists to prevent), and the
    test pins that divergence (the declared anchor_drop_over_cap rows; the reference never pre-cuts).
    """
    if reference_python is None:
        pytest.skip(f"no reference interpreter: set {_REFERENCE_PYTHON_ENV} to a python with torch, transformers")
    rows = [json.loads(line) for line in pairs_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    script = tmp_path / "remote_tokenize.py"
    script.write_text(
        "\n".join(
            [
                "import importlib.util, json, sys",
                "from pathlib import Path",
                "from transformers import AutoTokenizer",
                "directory = Path(sys.argv[1])",
                "tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True)",
                "config = json.loads((directory / 'config_sentence_transformers.json').read_text())",
                "st = json.loads((directory / 'sentence_bert_config.json').read_text())",
                "spec = importlib.util.spec_from_file_location('modeling_zembed', directory / 'modeling_zembed.py')",
                "module = importlib.util.module_from_spec(spec)",
                "spec.loader.exec_module(module)",
                "transformer = module.ZembedTransformer.__new__(module.ZembedTransformer)",
                "object.__setattr__(transformer, 'tokenizer', tokenizer)",
                "object.__setattr__(transformer, 'max_seq_length', int(st['max_seq_length']))",
                "out = []",
                "for entry in json.load(sys.stdin):",
                "    prompted = config['prompts'][entry['shape']] + entry['text']",
                "    ids = transformer.tokenize([prompted], padding=False)['input_ids'][0].tolist()",
                "    out.append({'index': entry['index'], 'shape': entry['shape'], 'ids': ids})",
                "json.dump(out, sys.stdout)",
            ]
        ),
        encoding="utf-8",
    )
    payload = [
        {"index": index, "shape": shape, "text": str(row["query"] if shape == "query" else row["documents"][0])}
        for index, row in enumerate(rows)
        for shape in ("query", "document")
    ]
    completed = subprocess.run(
        [reference_python, str(script), str(tokenizer_dir)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    remote = {(entry["index"], entry["shape"]): entry["ids"] for entry in json.loads(completed.stdout)}

    template = client_template(recipe)
    budget = TextBudget(
        tokenizer=str(tokenizer.name),
        max_tokens=recipe.client.get("max_tokens"),
        template=template,
        on_overflow=recipe.client.get("on_overflow"),
    )
    suffix_ids = tokenizer.ids(tokenizer.special_text("im_end") + "\n", add_special_tokens=False)
    # the pinned constants, re-read from the downloaded checkpoint file
    added = json.loads((tokenizer_dir / "added_tokens.json").read_text(encoding="utf-8"))
    assert int(added[tokenizer.special_text("im_start")]) == 151644
    assert int(added[tokenizer.special_text("im_end")]) == 151645
    for shape in ("query", "document"):
        inputs = [str(row["query"]) if shape == "query" else str(row["documents"][0]) for row in rows]
        result = fit(inputs, shape, budget, tokenizer, ids=[str(index) for index in range(len(rows))])
        over_cap = {cut.doc_id for cut in result.cuts}
        for index, text in enumerate(result.texts):
            served_ids = tokenizer.ids(text, add_special_tokens=True)
            assert served_ids == tokenizer.ids(text, add_special_tokens=False), "the post-processor adds no tokens"
            entry = remote[(index, shape)]
            if str(index) in over_cap:
                assert len(entry) == MAX_TOKENS, "the remote code truncated the raw over-cap prompt"
                assert entry[-len(suffix_ids) :] != suffix_ids, "the remote whole-prompt cut dropped the anchor"
                assert served_ids[-len(suffix_ids) :] == suffix_ids, "the recipe's cut kept it"
            else:
                assert entry == served_ids, (
                    f"row {index} {shape}: the remote tokenize ids differ from the declared render "
                    f"(remote {len(entry)} vs served {len(served_ids)})"
                )


# ---------------------------------------------------------------------------
# The mutation: dropping the trailing anchor segment turns the anchor check red
# ---------------------------------------------------------------------------


def test_mutation_dropping_the_trailing_anchor_segment_turns_the_anchor_check_red(
    recipe: Recipe, tokenizer_dir: Path, tmp_path: Path
) -> None:
    """Drop the template's trailing fixed segment (the suffix the pooler reads): the product still
    accepts the template (the post-processor flag declares an anchor), and stage 1's anchor check
    must go red -- the renders no longer end with the declared anchor ids."""
    template = client_template(recipe)
    assert template is not None
    mutated_template = TemplateSpec(
        query=template.query[:-1],
        document=template.document[:-1],
        anchor=template.anchor,
        add_special_tokens=template.add_special_tokens,
    )
    mutated_client = {**recipe.client, "template": mutated_template}
    mutated = recipe.model_copy(update={"client": mutated_client})
    pairs = tmp_path / "pairs-mutation.jsonl"
    rows = [
        {"query": "short query", "documents": ["a short document about retrieval models"]},
        {"query": "another query", "documents": ["another document, with a few more tokens than the first one"]},
        {"query": "third query", "documents": ["the third document, so the audit has several shapes to check"]},
    ]
    pairs.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    green = stage1_prompts(recipe, pairs, None, over_length_per_shape=2)
    assert green["anchor_check"]["passed"] is True
    red = stage1_prompts(mutated, pairs, None, over_length_per_shape=2)
    assert red["anchor_check"]["passed"] is False
    failures = red["anchor_check"]["failures"]
    assert failures, "the anchor check must report the renders whose tail is no longer the anchor"
    assert all(entry["check"] == "tail" for entry in failures)


# ---------------------------------------------------------------------------
# The declared contract: every serve, client and reference field pinned.
# ---------------------------------------------------------------------------

EXPECTED_TOP = {
    "id": "zembed-1-embedding",
    "input": ["text"],
    "licence": "apache-2.0",
    "model": "zeroentropy/zembed-1-embedding",
    "revision": "cf13c81f3274394053d166740294f7eea4586f7a",
    "role": "embed",
}
EXPECTED_SERVE = {
    "chat_template": None,
    "convert": "embed",
    "dtype": "bfloat16",
    "extra_args": [],
    "hf_overrides": {},
    "io_processor_plugin": None,
    "limit_mm_per_prompt": None,
    "max_model_len": 32768,
    "mm_processor_kwargs": {},
    "plugin": None,
    "patches": [],
    "plugin_architectures": [],
    "pooler_config": {},
    "runner": "pooling",
    "trust_remote_code": False,
}
EXPECTED_CLIENT = {
    "api": "openai_embeddings",
    "request_shape": "text",
    "tokenizer": "zeroentropy/zembed-1-embedding@cf13c81f3274394053d166740294f7eea4586f7a",
    "max_tokens": 32768,
    "template": {
        "query": [
            {"fixed": "{special:im_start}system\nquery{special:im_end}\n{special:im_start}user\n"},
            {"content": "query"},
            {"fixed": "{special:im_end}\n"},
        ],
        "document": [
            {"fixed": "{special:im_start}system\ndocument{special:im_end}\n{special:im_start}user\n"},
            {"content": "document"},
            {"fixed": "{special:im_end}\n"},
        ],
        "anchor": "last",
        "add_special_tokens": True,
    },
    "on_overflow": "cut",
    "empty_doc": "send",
    "normalize": True,
    "model": "zembed-1-embedding",
    "revision": "cf13c81f3274394053d166740294f7eea4586f7a",
}
EXPECTED_REFERENCE = {
    "entry": "reference.py",
    "kind": "sentence_transformers",
    "known_deviations": ["anchor_drop_over_cap"],
    "score_scale": "cosine",
}

# Two mutants per recipe against the contract pin above: each drift must fail, naming the field.
MUTANTS: list[tuple[str, tuple[str, ...], object, str]] = [
    ("serve.max_model_len drifts to 40960", ("serve", "max_model_len"), 40960, "max_model_len"),
    ("reference.kind drifts to transformers", ("reference", "kind"), "transformers", "reference.kind"),
]


def _mutated_recipe(tmp_path: Path, path: tuple[str, ...], value: object) -> object:
    """The recipe directory copied into ``tmp_path`` with one YAML field set to ``value``."""
    import shutil

    import yaml

    target = tmp_path / RECIPE_DIR.name
    shutil.copytree(RECIPE_DIR, target)
    yaml_path = target / "family.yaml"
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    node = data
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return load_recipe(target)


def _assert_contract(recipe: object) -> None:
    assert_recipe_contract(
        recipe,
        serve=EXPECTED_SERVE,
        client=EXPECTED_CLIENT,
        reference=EXPECTED_REFERENCE,
        top=EXPECTED_TOP,
    )


@pytest.mark.parametrize(("label", "path", "value", "needle"), MUTANTS, ids=[m[0] for m in MUTANTS])
def test_two_contract_mutants_are_red(
    label: str, path: tuple[str, ...], value: object, needle: str, tmp_path: Path
) -> None:
    """A drifted field fails the contract assertion naming it (two mutants per recipe)."""
    _assert_contract(load_recipe(RECIPE_DIR))  # the pinned recipe itself is green
    with pytest.raises(AssertionError) as caught:
        _assert_contract(_mutated_recipe(tmp_path, path, value))
    assert needle in str(caught.value), f"{label}: the failure must name {needle}: {caught.value}"


def test_requirements_reference_ships_the_documented_environment() -> None:
    """Finding #10: the note-7 referent exists -- requirements-reference.txt beside reference.py
    with the documented pins -- and no startup default is restated in the YAML."""
    path = RECIPE_DIR / "requirements-reference.txt"
    assert path.is_file(), "every recipe of this family ships its reference environment"
    text = path.read_text(encoding="utf-8")
    for pin in ("torch>=2.0", "transformers>=4.51", "sentence-transformers>=5.3,<5.4"):
        assert pin in text
    assert "startup_timeout_s" not in (RECIPE_DIR / "family.yaml").read_text(encoding="utf-8")
    notes = load_recipe(RECIPE_DIR).notes
    assert "no separate query cap exists in the referent" in notes
