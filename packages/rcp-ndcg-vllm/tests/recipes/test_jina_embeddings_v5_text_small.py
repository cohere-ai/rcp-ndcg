"""The ``jina-embeddings-v5-text-small`` recipe: schema validation and stage 1 on CPU.

The recipe validates offline (the client block constructs the product's
:class:`~rcp_ndcg.inference.config.EmbeddingEndpoint`). Stage 1 needs only the tokenizer file,
downloaded once into the lane's scratch directory (or ``tmp_path``) and verified against its
pinned sha256 — the model weights are never needed on CPU; the reference's ``render`` mode is
pure string work. When offline, the stage-1 tests skip with a clear reason; a cached copy with
the pinned hash keeps them runnable offline after the one download.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
import types
import urllib.request
from pathlib import Path
from typing import Any

import pytest
import yaml
from rcp_ndcg_vllm import Recipe, client_config, default_recipes_root, load_recipe
from rcp_ndcg_vllm.equivalence import stage1_prompts
from rcp_ndcg_vllm.equivalence.fitting import budget_of
from rcp_ndcg_vllm.equivalence.reference import run_reference

from rcp_ndcg.data.preprocess import fit
from rcp_ndcg.data.tokenizer import load_tokenizer
from rcp_ndcg.inference.config import EmbeddingEndpoint

RECIPE_ID = "jina-embeddings-v5-text-small"
MODEL = "jinaai/jina-embeddings-v5-text-small"
REVISION = "dd76d535f5447ca3897a9c893fb1e612ead98192"
TOKENIZER_URL = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/tokenizer.json"
TOKENIZER_SHA256 = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"  # r-jina5's pin

# The rendered prompts of the seed row, tokenized with the recipe's tokenizer at the pinned
# revision (ids measured 2026-10-05 from the tokenizer.json above). Pinned so a template or
# tokenizer drift moves the test, not the served numbers. Row 0 is also the seed the harness's
# over-length sampling repeats past the cap: both of its units overflow 32768 when joined (the
# junctions do not merge, unlike a seed ending in a bare sentence period).
QUERY_IDS = [2859, 25, 2585, 4937, 1558, 3100, 5821, 304, 264, 28202, 30]
DOCUMENT_IDS = [
    7524,
    25,
    758,
    264,
    28202,
    11,
    3100,
    34192,
    220,
    17,
    24,
    24,
    11,
    22,
    24,
    17,
    11,
    19,
    20,
    23,
    36256,
    1449,
    2086,
    481,
    264,
    20178,
    6783,
    315,
    6993,
    13,
]

#: The seed pair (whose over-length samples overflow the budget: measured 32784 and 32829 tokens),
#: the card's example pair set, and short retrieval rows; all far inside the 32768-token budget.
PAIRS: list[dict[str, Any]] = [
    {
        "query": "How fast does light travel in a vacuum?",
        "documents": ["In a vacuum, light travels 299,792,458 metres every second - a universal constant of nature."],
    },
    {
        "query": "Which planet is known as the Red Planet?",
        "documents": [
            "Venus is often called Earth's twin because of its similar size and proximity.",
            "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
            "Jupiter, the largest planet in our solar system, has a prominent red spot.",
            "Saturn, famous for its rings, is sometimes mistaken for the Red Planet.",
        ],
    },
    {"query": "What is the capital of France?", "documents": ["Paris is the capital and largest city of France."]},
    {"query": "Who wrote the theory of general relativity?", "documents": ["Albert Einstein published it in 1915."]},
    {
        "query": "What gas do plants absorb from the atmosphere?",
        "documents": ["Photosynthesis takes in carbon dioxide and releases oxygen."],
    },
    {
        "query": "Largest ocean on Earth",
        "documents": ["The Pacific Ocean is the largest and deepest of Earth's oceans."],
    },
    {
        "query": "Speed of light in vacuum",
        "documents": ["Light travels at about 299,792 kilometres per second in a vacuum."],
    },
    {
        "query": "What year did the Titanic sink?",
        "documents": ["The RMS Titanic sank on her maiden voyage in April 1912."],
    },
    {
        "query": "Define machine learning",
        "documents": ["Machine learning is the study of algorithms that improve through experience with data."],
    },
    {
        "query": "Boiling point of water at sea level",
        "documents": ["Water boils at 100 degrees Celsius at standard atmospheric pressure."],
    },
    {
        "query": "Which language has the most native speakers?",
        "documents": ["Mandarin Chinese has the most native speakers of any language."],
    },
    {
        "query": "What is the smallest prime number?",
        "documents": ["Two is the smallest prime number and the only even one."],
    },
    {"query": "Currency of Japan", "documents": ["The yen is the official currency of Japan."]},
]


def recipe_dir() -> Path:
    """The recipe directory this lane ships."""
    return default_recipes_root() / RECIPE_ID


def load() -> Recipe:
    """The loaded recipe (validates id == directory name and the product endpoint at load)."""
    return load_recipe(recipe_dir())


def _lane_scratch() -> Path | None:
    """The lane's scratch directory when this runs inside the lane's worktree (wt-<tag> naming)."""
    worktree = Path(__file__).resolve().parents[4]
    if worktree.name.startswith("wt-"):
        scratch = worktree.parent / worktree.name[3:] / "scratch"
        if scratch.is_dir():
            return scratch
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tokenizer_file(tmp_path: Path) -> Path:
    """tokenizer.json at the pinned revision, downloaded once and verified against its pinned sha256.

    The file lands in the lane's scratch directory when this runs in the lane worktree (shared by
    reruns and verifiers, never into the checkout), else in ``tmp_path``. Skips with a clear reason
    when offline; a cached copy with the pinned hash is reused offline.
    """
    scratch = _lane_scratch()
    target = (scratch / "tokenizer-cache" / "tokenizer.json") if scratch else (tmp_path / "tokenizer.json")
    if target.is_file() and _sha256(target) == TOKENIZER_SHA256:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(TOKENIZER_URL, timeout=120) as response:
            target.write_bytes(response.read())
    except OSError as error:
        pytest.skip(f"offline: cannot fetch {MODEL}@{REVISION} tokenizer.json ({error}); stage 1 on CPU needs it")
    assert _sha256(target) == TOKENIZER_SHA256, "the downloaded tokenizer.json does not match the pinned sha256"
    return target


def stage1_recipe(path: Path) -> Recipe:
    """The committed recipe reading its tokenizer from the downloaded file.

    The committed recipe names the Hub spec (what production resolves); the stage-1 checks run on
    the same tokenizer.json, downloaded into the scratch and verified against the pinned sha256,
    so they stay offline-capable after the one download.
    """
    recipe = load()
    client = recipe.client.model_copy(update={"tokenizer": str(path)})
    return recipe.model_copy(update={"client": client})


def _import_reference() -> Any:
    """The shipped reference.py imported as a module: its module level is pure stdlib, so the
    harness process can import it (torch and transformers load inside ``load`` only)."""

    spec = importlib.util.spec_from_file_location(
        "jina_embeddings_v5_text_small_reference", recipe_dir() / "reference.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_pairs(tmp_path: Path) -> Path:
    path = tmp_path / "pairs.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in PAIRS), encoding="utf-8")
    return path


def cast_shape(shape: str) -> Any:
    """A validated shape string as the product's RequestShape literal (the recipe validated it)."""
    return shape


def test_recipe_loads_with_the_product_endpoint_config() -> None:
    """The recipe validates and its client block IS the product's EmbeddingEndpoint."""
    recipe = load()
    assert recipe.id == RECIPE_ID == recipe_dir().name
    assert recipe.model == MODEL and recipe.revision == REVISION
    assert recipe.role == "embed" and recipe.input == ["text"] and recipe.licence == "cc-by-nc-4.0"
    assert isinstance(recipe.client, EmbeddingEndpoint)
    # The explicit budget: the Hub spec at the pinned revision, the whole-prompt cap, the policy.
    assert recipe.client.tokenizer == f"{MODEL}@{REVISION}"
    assert recipe.client.max_tokens == 32768 == recipe.serve.max_model_len
    assert recipe.client.on_overflow == "cut"
    assert recipe.client.request_shape == "text"
    assert recipe.client.empty_doc == "send"
    assert recipe.client.normalize is True
    assert recipe.client.dimensions is None
    assert recipe.client.query_prompt == "" and recipe.client.doc_prompt == ""  # the template owns the prefixes
    # The template as data: both shapes, marker + separator fixed segments, the content span,
    # the head anchor and the route's add_special_tokens.
    template = recipe.client.template
    assert template is not None and template.anchor == "first"
    assert template.adds_special_tokens("query") and template.adds_special_tokens("document")
    assert [segment.fixed for segment in template.segments("query")] == ["Query:", " ", None]
    assert [segment.fixed for segment in template.segments("document")] == ["Document:", " ", None]
    # The serve block: the engine image, the task-adapter merge, no served chat template.
    assert recipe.serve.runner == "pooling" and recipe.serve.dtype == "bfloat16"
    assert recipe.serve.trust_remote_code is True
    assert recipe.serve.chat_template is None  # raw text on /v1/embeddings; see the recipe notes
    assert recipe.serve.hf_overrides == {
        "jina_task": "retrieval",
        "is_matryoshka": True,
        "matryoshka_dimensions": [32, 64, 128, 256, 512, 768, 1024],
    }
    assert recipe.serve.plugin is None and recipe.serve.pooler_config == {}
    assert recipe.reference.kind == "remote_code" and recipe.reference.score_scale == "cosine"
    assert recipe.reference.known_deviations == []
    assert recipe.status.state == "unverified"
    # The client block is the product's config: the dump constructs the product model unchanged.
    config = client_config(recipe, base_url="http://127.0.0.1:8100/v1")
    endpoint = EmbeddingEndpoint.model_validate(config)
    assert endpoint.model == RECIPE_ID and endpoint.revision == recipe.revision
    assert endpoint.max_tokens == 32768 and endpoint.tokenizer == recipe.client.tokenizer


def test_stage1_on_cpu_token_id_equality_and_anchors(tmp_path: Path) -> None:
    """Stage 1 on CPU: token-id equality against the reference and the anchor check, over-length
    inputs included (at least 20 sampled pairs, 5 over-length per declared shape here)."""
    tokenizer_path = tokenizer_file(tmp_path)
    recipe = stage1_recipe(tokenizer_path)
    tokenizer = load_tokenizer(str(tokenizer_path))
    pairs_path = _write_pairs(tmp_path)
    document = stage1_prompts(recipe, pairs_path, sys.executable, over_length_per_shape=5)
    assert document["sampled"] == len(PAIRS) + 10 >= 20
    assert document["passed"] is True, (
        document["anchor_check"]["failures"][:1],
        document["render_check"]["failures"][:1],
    )
    # The anchor check covered every sampled input, over-length ones included.
    assert document["anchor_check"]["passed"] is True
    assert document["anchor_check"]["checked"] == document["sampled"]
    assert document["anchor_check"]["anchor"] == "first"
    # Both declared shapes carried their five over-length samples and cut them (the content span
    # only; the fixed frame is reserved, which the anchor check just asserted).
    for shape in ("query", "document"):
        assert document["fit"][shape]["overhead"] == 3  # "Query:"/"Document:" + the separator space
        assert document["fit"][shape]["cuts"] >= 5
    # The engine-side /tokenize check is reported not_run without an engine, never passed.
    assert document["engine_tokenize_check"]["status"] == "not_run"
    # Token-id equality, asserted on ids: the reference subprocess's rendered prompts, tokenized
    # by the product's tokenizer with the route's add_special_tokens, equal the product fit's
    # renders' ids, and fit's render is the card's prompt for the uncut input.
    reference = run_reference(
        sys.executable,
        str(recipe_dir() / recipe.reference.entry),
        mode="render",
        pairs_path=pairs_path,
        out_path=tmp_path / "reference-render.json",
        tokenizer_spec=str(tokenizer_path),
    )
    ref_by_key = {(row["index"], row["shape"]): row["text"] for row in reference["rows"]}
    expected_keys = {(index, shape) for index in range(len(PAIRS)) for shape in ("query", "document")}
    assert set(ref_by_key) == expected_keys
    budget = budget_of(recipe).model_copy(update={"tokenizer": tokenizer.name})
    for key, reference_text in ref_by_key.items():
        shape = cast_shape(key[1])
        raw = PAIRS[key[0]]["query"] if key[1] == "query" else PAIRS[key[0]]["documents"][0]
        fit_text = fit([raw], shape, budget, tokenizer, ids=["0"]).texts[0]
        assert fit_text == reference_text, key
        assert tokenizer.ids(fit_text, add_special_tokens=True) == tokenizer.ids(
            reference_text, add_special_tokens=True
        ), key
    # And the id anchor: the seed row's rendered prompts tokenize to the pinned ids.
    assert tokenizer.ids(ref_by_key[(0, "query")], add_special_tokens=True) == QUERY_IDS
    assert tokenizer.ids(ref_by_key[(0, "document")], add_special_tokens=True) == DOCUMENT_IDS


def test_reference_load_resolves_the_pinned_snapshot(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """load() must resolve the whole pinned snapshot and load everything from it.

    The checkpoint's remote code drops the revision kwarg for every inner load (config, base
    weights, adapters, tokenizer all resolve at Hub HEAD - modeling_jina_embeddings_v5.py:25-27,
    :28-32, :37-41, :57-60), so a repo-id load pins nothing. Behavioural and offline: the three
    heavy modules are stubbed, so the harness process imports no torch and touches no network;
    the full embed/score path still runs only in the reference environment (the GPU wave)."""
    reference = _import_reference()
    calls: list[dict[str, Any]] = []
    snapshot = tmp_path / "snapshot"

    def fake_snapshot_download(repo_id: str, **kwargs: Any) -> str:
        calls.append({"snapshot_download": {"repo_id": repo_id, **kwargs}})
        return str(snapshot)

    class _FakeModel:
        def to(self, device: str) -> _FakeModel:
            return self

        def eval(self) -> _FakeModel:
            return self

    def fake_from_pretrained(name: str, **kwargs: Any) -> _FakeModel:
        calls.append({"from_pretrained": name, **kwargs})
        return _FakeModel()

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=fake_snapshot_download))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoModel=types.SimpleNamespace(from_pretrained=fake_from_pretrained),
            AutoTokenizer=types.SimpleNamespace(from_pretrained=fake_from_pretrained),
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(bfloat16="bfloat16"))

    reference.load(device="cpu")
    assert calls[0] == {
        "snapshot_download": {
            "repo_id": MODEL,
            "revision": REVISION,
            "allow_patterns": ["*.json", "*.py", "*.txt", "*.jinja", "*.safetensors"],
        }
    }
    loads = [entry for entry in calls[1:] if "from_pretrained" in entry]
    assert len(loads) == 2, calls  # the model and the tokenizer, both from the snapshot
    assert all(entry["from_pretrained"] == str(snapshot) for entry in loads)
    assert all("revision" not in entry or entry["revision"] is None for entry in loads)

    # An explicit --model-path bypasses the Hub resolution entirely.
    calls.clear()
    reference.load(device="cpu", model_path="/local/snapshot")
    assert not any("snapshot_download" in entry for entry in calls)
    assert all(entry["from_pretrained"] == "/local/snapshot" for entry in calls if "from_pretrained" in entry)


def test_dropping_the_anchor_segment_turns_the_anchor_check_red(tmp_path: Path) -> None:
    """The mutation: drop the template's anchor segment (the leading fixed marker segment; this
    model's anchor is at the head) and the anchor check goes red on every render."""
    tokenizer_path = tokenizer_file(tmp_path)
    mutated_dir = tmp_path / RECIPE_ID
    mutated_dir.mkdir()
    for name in ("recipe.yaml", "reference.py"):
        shutil.copy(recipe_dir() / name, mutated_dir / name)
    data = yaml.safe_load((mutated_dir / "recipe.yaml").read_text(encoding="utf-8"))
    for shape in ("query", "document"):
        segments = data["client"]["template"][shape]
        assert segments[0]["fixed"] in ("Query:", "Document:")
        data["client"]["template"][shape] = segments[1:]  # the anchor segment is gone
    data["client"]["tokenizer"] = str(tokenizer_path)  # the downloaded file, as in the stage-1 test
    (mutated_dir / "recipe.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    recipe = load_recipe(mutated_dir)  # still a valid recipe: add_special_tokens covers the rule
    document = stage1_prompts(recipe, _write_pairs(tmp_path), sys.executable, over_length_per_shape=2)
    assert document["anchor_check"]["passed"] is False
    assert document["anchor_check"]["failures"], "the anchor check must report the dropped anchor"
    assert all(failure["check"] == "head" for failure in document["anchor_check"]["failures"])
    assert document["passed"] is False
