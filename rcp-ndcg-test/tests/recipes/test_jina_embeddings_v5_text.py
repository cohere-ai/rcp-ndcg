"""The ``jina-embeddings-v5-text`` family: schema validation and stage 1 on CPU per variant
(decision 34: one family module, parametrized over its variant ids; every field pinned per variant).

The recipe validates offline (the client block constructs the product's
:class:`~rcp_ndcg.inference.config.EmbeddingEndpoint`). Stage 1 needs only the tokenizer file,
downloaded once through the shared ``_served.fetch_tokenizer`` (into
``RCP_NDCG_VLLM_TOKENIZER_CACHE`` when set, else ``tmp_path``) and verified against its pinned
sha256 — the model weights are never needed on CPU; the reference's ``render`` mode is pure string
work. When offline, the stage-1 tests skip with a clear reason; a cached copy with the pinned hash
keeps them runnable offline after the one download.

The two sizes differ per the family's declared per-size fields: the -nano is EuroBERT-210m
(``is_decoder false``; vLLM dispatches ``JinaEmbeddingsV5Model`` to its encoder implementation) with
a 768-wide hidden size, an 8192-token context and its own Matryoshka list; the -small is Qwen3-0.6B
with 1024/32768 and the 1024-wide list. Both share the prompts, the task adapters and the pooling;
the -nano's tokenizer appends an end-of-text tail under ``add_special_tokens`` where the -small's
adds none, which is why the per-variant overheads differ.
"""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import yaml
from rcp_ndcg_test.equivalence import stage1_prompts
from rcp_ndcg_test.equivalence.reference import run_reference
from rcp_ndcg_vllm.recipe import Recipe, client_config, default_recipes_root, resolve_recipe

from rcp_ndcg.data.tokenizer import load_tokenizer
from rcp_ndcg.inference.config import EmbeddingEndpoint

from ._contract import assert_recipe_contract
from ._served import client_template, fetch_tokenizer, served_texts, stage1_facts

FAMILY_ID = "jina-embeddings-v5-text"  # the family directory (decision 34)
RECIPE_DIR = default_recipes_root() / FAMILY_ID

#: The family's variants (decision 34): the per-size facts the tests pin. ``sha256`` is the pinned
#: ``tokenizer.json``; ``overhead`` the declared template's fixed tokens per shape (the -nano's
#: tokenizer appends its end-of-text tail under add_special_tokens, the -small's appends nothing);
#: ``download_bytes`` the base ``model.safetensors`` size at the pinned revision, stated in the
#: variant's notes.
VARIANTS: dict[str, dict[str, Any]] = {
    "jina-embeddings-v5-text-nano": {
        "repo": "jinaai/jina-embeddings-v5-text-nano",
        "revision": "8a7f00aac812071b69403df470f1038ec85f8925",
        "sha256": "98d4a1d32152d6cedf85b5e88f3b205106dca1fe72aaab34e0ac13c238421069",
        "max_tokens": 8192,
        "max_model_len": 8192,
        "hf_overrides": {
            "jina_task": "retrieval",
            "is_matryoshka": True,
            "matryoshka_dimensions": [32, 64, 128, 256, 512, 768],
        },
        "mrl_dims": [32, 64, 128, 256, 512, 768],
        "overhead": 4,
        "download_bytes": "423,543,680",
        # The rendered prompts of the seed row, tokenized with the pinned tokenizer (the tail is the
        # appended end-of-text token, id 128001).
        "query_ids": [2929, 25, 2650, 5043, 1587, 3177, 5944, 304, 264, 29302, 30, 128001],
        "document_ids": [
            7676,
            25,
            763,
            264,
            29302,
            11,
            3177,
            35292,
            220,
            15531,
            11,
            24763,
            11,
            21209,
            37356,
            1475,
            2132,
            482,
            264,
            20789,
            6926,
            315,
            7138,
            13,
            128001,
        ],
    },
    "jina-embeddings-v5-text-small": {
        "repo": "jinaai/jina-embeddings-v5-text-small",
        "revision": "dd76d535f5447ca3897a9c893fb1e612ead98192",
        "sha256": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
        "max_tokens": 32768,
        "max_model_len": 32768,
        "hf_overrides": {
            "jina_task": "retrieval",
            "is_matryoshka": True,
            "matryoshka_dimensions": [32, 64, 128, 256, 512, 768, 1024],
        },
        "mrl_dims": [32, 64, 128, 256, 512, 768, 1024],
        "overhead": 3,
        "download_bytes": "1,192,133,208",
        "query_ids": [2859, 25, 2585, 4937, 1558, 3100, 5821, 304, 264, 28202, 30],
        "document_ids": [
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
        ],
    },
}
VARIANT_IDS = list(VARIANTS)

#: The seed pair (whose over-length samples overflow the budget), the card's example pair set, and
#: short retrieval rows; all far inside every variant's budget.
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
    """The family directory this file tests (file-relative layout: resolves against an installed wheel too)."""
    return RECIPE_DIR


def load(variant_id: str) -> Recipe:
    """The variant's loaded recipe (validates id == the variant row and the product endpoint at load)."""
    return resolve_recipe(variant_id)


def tokenizer_file(tmp_path: Path, variant_id: str) -> Path:
    """``tokenizer.json`` at the pinned revision, through the shared tokenizer cache and verified against
    its pinned sha256 (``_served.fetch_tokenizer``: one home for the download, the cache variable and
    the offline skip)."""
    variant = VARIANTS[variant_id]
    url = f"https://huggingface.co/{variant['repo']}/resolve/{variant['revision']}/tokenizer.json"
    name = f"{variant_id}@{variant['revision']}/tokenizer.json"
    return fetch_tokenizer(url, name, tmp_path, sha256=variant["sha256"])


def stage1_recipe(path: Path, variant_id: str) -> Recipe:
    """The committed recipe reading its tokenizer from the downloaded file.

    The committed recipe names the Hub spec (what production resolves); the stage-1 checks run on
    the same tokenizer.json, downloaded into the tokenizer cache and verified against the pinned
    sha256, so they stay offline-capable after the one download.
    """
    recipe = load(variant_id)
    client = {**recipe.client, "tokenizer": str(path)}
    return recipe.model_copy(update={"client": client})


def _import_reference() -> Any:
    """The shipped reference.py imported as a module: its module level is pure stdlib, so the
    harness process can import it (torch and transformers load inside ``load`` only)."""

    spec = importlib.util.spec_from_file_location("jina_embeddings_v5_text_reference", recipe_dir() / "reference.py")
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


def _expected_contract(variant_id: str) -> dict[str, Any]:
    """The variant's full resolved contract: every field of every block, exactly as the product
    models resolve it (authored values and schema defaults alike). Nothing may ride unpinned."""
    variant = VARIANTS[variant_id]
    return {
        "top": {
            "id": variant_id,
            "input": ["text"],
            "licence": "cc-by-nc-4.0",
            "model": variant["repo"],
            "revision": variant["revision"],
            "role": "embed",
        },
        "serve": {
            "chat_template": None,
            "convert": None,
            "dtype": "bfloat16",
            "extra_args": [],
            "hf_overrides": variant["hf_overrides"],
            "io_processor_plugin": None,
            "limit_mm_per_prompt": None,
            "max_model_len": variant["max_model_len"],
            "mm_processor_kwargs": {},
            "plugin": None,
            "pooler_config": {},
            "runner": "pooling",
            "trust_remote_code": True,
        },
        "client": {
            "api": "openai_embeddings",
            "tokenizer": f"{variant['repo']}@{variant['revision']}",
            "max_tokens": variant["max_tokens"],
            "template": {
                "query": [{"fixed": "Query:"}, {"fixed": " "}, {"content": "query"}],
                "document": [{"fixed": "Document:"}, {"fixed": " "}, {"content": "document"}],
                "anchor": "last_content",
                "add_special_tokens": True,
            },
            "on_overflow": "cut",
            "empty_doc": "send",
            "request_shape": "text",
            "normalize": True,
            "mrl_kind": "truncation",
            "mrl_dims": variant["mrl_dims"],
            "dimensions": None,
            "model": variant_id,
            "revision": variant["revision"],
        },
        "reference": {
            "entry": "reference.py",
            "kind": "remote_code",
            "known_deviations": ["over_cap_cut_differs"],
            "device": None,  # the schema default
            "score_scale": "cosine",
        },
    }


def _assert_contract(recipe: Any, variant_id: str) -> None:
    expected = _expected_contract(variant_id)
    assert_recipe_contract(
        recipe,
        serve=expected["serve"],
        client=expected["client"],
        reference=expected["reference"],
        top=expected["top"],
    )


@pytest.mark.parametrize("variant_id", VARIANT_IDS)
def test_recipe_loads_with_the_product_endpoint_config(variant_id: str) -> None:
    """The recipe validates and its client block IS the product's EmbeddingEndpoint."""
    recipe = load(variant_id)
    _assert_contract(recipe, variant_id)  # every serve, client and reference field pinned, exactly
    assert recipe.id == variant_id and recipe_dir().name == FAMILY_ID
    assert recipe.model == VARIANTS[variant_id]["repo"] and recipe.revision == VARIANTS[variant_id]["revision"]
    assert recipe.role == "embed" and recipe.input == ["text"] and recipe.licence == "cc-by-nc-4.0"
    # The explicit budget: the Hub spec at the pinned revision, the whole-prompt cap, the policy.
    assert recipe.client.get("tokenizer") == f"{recipe.model}@{recipe.revision}"
    assert recipe.client.get("max_tokens") == VARIANTS[variant_id]["max_tokens"]
    assert recipe.client.get("max_tokens") == recipe.serve.max_model_len
    assert recipe.client.get("on_overflow") == "cut"
    assert recipe.client.get("request_shape") == "text"
    assert recipe.client.get("empty_doc") == "send"
    assert recipe.client.get("normalize") is True
    assert recipe.client.get("dimensions") is None
    assert "query_prompt" not in recipe.client and "doc_prompt" not in recipe.client  # the template owns the prefixes
    # The template as data: both shapes, marker + separator fixed segments, the content span,
    # the pooled position's anchor (last_content: the mask's last real token -- the last kept
    # content token) and the route's add_special_tokens.
    template = client_template(recipe)
    assert template is not None and template.anchor == "last_content"
    assert template.adds_special_tokens("query") and template.adds_special_tokens("document")
    assert [segment.fixed for segment in template.segments("query")] == ["Query:", " ", None]
    assert [segment.fixed for segment in template.segments("document")] == ["Document:", " ", None]
    # The serve block: the engine image, the task-adapter merge, no served chat template.
    assert recipe.serve.runner == "pooling" and recipe.serve.dtype == "bfloat16"
    assert recipe.serve.trust_remote_code is True
    assert recipe.serve.chat_template is None  # raw text on /v1/embeddings; see the recipe notes
    assert recipe.serve.hf_overrides == VARIANTS[variant_id]["hf_overrides"]
    assert recipe.serve.plugin is None and recipe.serve.pooler_config == {}
    assert recipe.reference.kind == "remote_code" and recipe.reference.score_scale == "cosine"
    assert recipe.reference.known_deviations == ["over_cap_cut_differs"]
    assert recipe.status.state == "unverified"
    # The client block is the product's config: the dump constructs the product model unchanged.
    config = client_config(recipe, base_url="http://127.0.0.1:8100/v1")
    endpoint = EmbeddingEndpoint.model_validate(config)
    assert endpoint.model == variant_id and endpoint.revision == recipe.revision
    assert endpoint.max_tokens == VARIANTS[variant_id]["max_tokens"]
    assert endpoint.tokenizer == recipe.client.get("tokenizer")


@pytest.mark.network
@pytest.mark.parametrize("variant_id", VARIANT_IDS)
def test_stage1_on_cpu_token_id_equality_and_anchors(tmp_path: Path, variant_id: str) -> None:
    """Stage 1 on CPU: token-id equality against the reference and the anchor check, over-length
    inputs included (at least 20 sampled pairs, 5 over-length per declared shape here)."""
    tokenizer_path = tokenizer_file(tmp_path, variant_id)
    recipe = stage1_recipe(tokenizer_path, variant_id)
    tokenizer = load_tokenizer(str(tokenizer_path))
    pairs_path = _write_pairs(tmp_path)
    document = stage1_prompts(recipe, pairs_path, sys.executable, over_length_per_shape=5)
    assert document["sampled"] == len(PAIRS) + 10 >= 20
    # The render comparison gates (zero tolerance); over-cap rows the client cut ride the declared
    # over_cap_cut_differs table and do not gate.
    assert document["render_check"]["passed"] is True, document["render_check"]["failures"][:1]
    anchor = document["anchor_check"]
    # The anchor check covered every sampled input, over-length ones included (one text per
    # declared shape per pairs row -- the query, and EVERY document of the row -- one per sample).
    assert anchor["checked"] == len(PAIRS) + sum(len(row["documents"]) for row in PAIRS) + 10
    assert anchor["anchor"] == "last_content"
    # The harness audits last_content as the product defines it: the head markers open every render,
    # the post-processor's tail (when the tokenizer appends one, as the -nano's does) closes it, and
    # a content token sits between the two.
    assert anchor["passed"] is True, anchor["failures"][:1]
    # Both declared shapes carried their five over-length samples and cut them (the content span
    # only; the fixed frame is reserved, which the anchor check just asserted). The facts come from
    # the role client's own capture and census (what the served path really sent).
    facts = stage1_facts(recipe, PAIRS, tokenizer, 5)
    for shape in ("query", "document"):
        body = facts["per_shape"][shape]
        assert body["overhead"] == VARIANTS[variant_id]["overhead"]
        assert body["cuts"] >= 5
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
        recipe=recipe,
    )
    ref_by_key = {(row["index"], row["shape"]): row["text"] for row in reference["rows"]}
    expected_keys = {(index, shape) for index in range(len(PAIRS)) for shape in ("query", "document")}
    assert set(ref_by_key) == expected_keys
    for key, reference_text in ref_by_key.items():
        raw = PAIRS[key[0]]["query"] if key[1] == "query" else PAIRS[key[0]]["documents"][0]
        fit_text = served_texts(recipe, [raw], key[1])[0]
        assert fit_text == reference_text, key
        assert tokenizer.ids(fit_text, add_special_tokens=True) == tokenizer.ids(
            reference_text, add_special_tokens=True
        ), key
    # And the id anchor: the seed row's rendered prompts tokenize to the pinned ids.
    assert tokenizer.ids(ref_by_key[(0, "query")], add_special_tokens=True) == VARIANTS[variant_id]["query_ids"]
    assert tokenizer.ids(ref_by_key[(0, "document")], add_special_tokens=True) == VARIANTS[variant_id]["document_ids"]


@pytest.mark.parametrize("variant_id", VARIANT_IDS)
def test_reference_load_resolves_the_pinned_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, variant_id: str
) -> None:
    """load() must resolve the whole pinned snapshot and load everything from it.

    On a repo-id load the checkpoint's remote code leaves the adapters and the tokenizer at Hub
    HEAD and the base weights at HEAD on a cache miss (modeling_jina_embeddings_v5.py:37-41,
    :57-60; measured on transformers 5.17.0, cold cache) - only the config rides the caller's
    revision - so load() must resolve the whole snapshot itself. Behavioural and offline: the
    three heavy modules are stubbed, so the harness process imports no torch and touches no
    network; the full embed/score path still runs only in the reference environment (the GPU
    wave)."""
    reference = _import_reference()
    calls: list[dict[str, Any]] = []
    snapshot = tmp_path / "snapshot"
    variant = VARIANTS[variant_id]

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

    reference.load(device="cpu", repo=variant["repo"], revision=variant["revision"])
    assert calls[0] == {
        "snapshot_download": {
            "repo_id": variant["repo"],
            "revision": variant["revision"],
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

    # The checkpoint identity comes from the resolved recipe (--recipe), not the module constants:
    # a variant row pointing at another checkpoint must resolve THAT checkpoint's snapshot.
    calls.clear()
    reference.load(device="cpu", repo="example-org/other-checkpoint", revision="0" * 40)
    assert calls[0]["snapshot_download"]["repo_id"] == "example-org/other-checkpoint"
    assert calls[0]["snapshot_download"]["revision"] == "0" * 40


@pytest.mark.network
def test_dropping_the_anchor_segment_turns_the_render_check_red(tmp_path: Path) -> None:
    """The mutation: drop the template's leading fixed marker segment ("Query:" / "Document:").

    Under ``anchor: last_content`` the model pools the last kept content token (no positional
    requirement), so the marker's protection is the DECLARED FRAME being byte-identical to the
    checkpoint's own prompts -- and stage 1's render check (the reference's render against the
    fit's captured prompt, zero tolerance) is the gate: dropping the marker silently changes every
    prompt, and the render check reds naming the row.  (The anchor audit sees a dropped head
    marker too.)
    """
    variant_id = "jina-embeddings-v5-text-small"
    tokenizer_path = tokenizer_file(tmp_path, variant_id)
    mutated_dir = tmp_path / FAMILY_ID  # the family directory name (the loader pins family id == directory name)
    mutated_dir.mkdir()
    for name in ("family.yaml", "reference.py"):
        shutil.copy(recipe_dir() / name, mutated_dir / name)
    data = yaml.safe_load((mutated_dir / "family.yaml").read_text(encoding="utf-8"))
    for shape in ("query", "document"):
        segments = data["client"]["template"][shape]
        assert segments[0]["fixed"] in ("Query:", "Document:")
        data["client"]["template"][shape] = segments[1:]  # the anchor segment is gone
    data["client"]["tokenizer"] = str(tokenizer_path)  # the downloaded file, as in the stage-1 test
    (mutated_dir / "family.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    recipe = resolve_recipe(variant_id, root=tmp_path)  # still a valid recipe: add_special_tokens covers the rule
    document = stage1_prompts(recipe, _write_pairs(tmp_path), sys.executable, over_length_per_shape=2)
    render_check = document["render_check"]
    assert render_check["passed"] is False
    assert render_check["failures"], "the render check must report the dropped marker"
    assert document["passed"] is False


# ---------------------------------------------------------------------------
# The declared contract: every serve, client and reference field pinned.
# ---------------------------------------------------------------------------

# Two mutants per family against the contract pin above: each drift must fail, naming the field.
MUTANTS: list[tuple[str, tuple[str, ...], object, str]] = [
    ("client.template.anchor drifts to first", ("client", "template", "anchor"), "first", "client.template.anchor"),
    (
        "reference.kind drifts to sentence_transformers",
        ("reference", "kind"),
        "sentence_transformers",
        "reference.kind",
    ),
]


def _mutated_recipe(tmp_path: Path, variant_id: str, path: tuple[str, ...], value: object) -> object:
    """The family directory copied into ``tmp_path`` with one YAML field set to ``value``."""
    import shutil

    import yaml

    target = tmp_path / FAMILY_ID
    shutil.copytree(RECIPE_DIR, target)
    yaml_path = target / "family.yaml"
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    node = data
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return resolve_recipe(variant_id, root=tmp_path)


@pytest.mark.parametrize("variant_id", VARIANT_IDS)
@pytest.mark.parametrize(("label", "path", "value", "needle"), MUTANTS, ids=[m[0] for m in MUTANTS])
def test_two_contract_mutants_are_red(
    label: str, path: tuple[str, ...], value: object, needle: str, tmp_path: Path, variant_id: str
) -> None:
    """A drifted field fails the contract assertion naming it (two mutants per family, per variant)."""
    _assert_contract(load(variant_id), variant_id)  # the pinned recipe itself is green
    with pytest.raises(AssertionError) as caught:
        _assert_contract(_mutated_recipe(tmp_path, variant_id, path, value), variant_id)
    assert needle in str(caught.value), f"{label}: the failure must name {needle}: {caught.value}"


@pytest.mark.parametrize("variant_id", VARIANT_IDS)
def test_notes_pin_the_query_cap_check_the_feature_floor_and_the_download_figures(variant_id: str) -> None:
    """The notes' pins: the query_max_tokens check, the min_version rule with the feature
    floor in the notes, no restated startup default, and the re-derived download figures."""
    recipe = load(variant_id)
    notes = recipe.notes
    # Lane H's fixes are stated as today's behaviour: the last_content audit runs, over-cap texts are reported.
    assert "does not audit last_content" not in notes and "strict xfail" not in notes
    assert "Stage 1's anchor_check audits last_content" in notes
    assert "does not yet list peft" not in notes
    assert "no separate query cap exists in the referent" in notes
    assert recipe.engine.min_version == "0.31.0"  # the image verified -- not a measured feature floor
    assert recipe.engine.startup_timeout_s == 1800  # the schema default, not restated in the YAML
    assert "startup_timeout_s" not in (RECIPE_DIR / "family.yaml").read_text(encoding="utf-8")
    assert "first ships in v0.20.0" in notes  # the feature floor, named in the notes
    assert VARIANTS[variant_id]["download_bytes"] in notes  # the Hub tree API's size at the pinned revision


# ---------------------------------------------------------------------------
# The last_content anchor audit (the harness branch for the anchor kind this recipe declares).
# ---------------------------------------------------------------------------


@pytest.mark.network
@pytest.mark.parametrize("variant_id", VARIANT_IDS)
def test_stage1_anchor_check_knows_the_last_content_anchor(tmp_path: Path, variant_id: str) -> None:
    """The anchor gate is green: the harness audits `anchor: last_content` (the head markers open every render,
    the post-processor's tail closes it, and a content token sits between the two)."""
    tokenizer_path = tokenizer_file(tmp_path, variant_id)
    pairs_path = _write_pairs(tmp_path)
    recipe = stage1_recipe(tokenizer_path, variant_id)
    document = stage1_prompts(recipe, pairs_path, sys.executable, over_length_per_shape=2)
    assert document["anchor_check"]["passed"] is True


#: Internal process labels that must not ship in a recipe (review shorthand, private work
#: directories, rule ids no public document defines). Public rule ids (R29, documented in
#: docs/how-to/add-a-model.md) stay allowed.
INTERNAL_LABELS = re.compile(
    r"p1-tail|fam-(?:dense|ctxl)|\bsweep|lanes' base|audit-synth|\br-(?:ctxl|jina[35]|octen|zembed1|qwen3-emb)\b"
    r"|\bresearch\b|\blanes?\b|REVIEW-LOG|ANCHOR-FINDING|\bR(?!29\b)\d{1,2}\b|clients-final"
    r"|\boperator\b|\b09x\b|\.refs/|recipe-common|corrections table|\bfinding #?\d"
)


@pytest.mark.parametrize(
    "recipe_id",
    [
        "qwen3-embedding-0.6b",
        "octen-embedding-0.6b",
        "octen-embedding-4b",
        "octen-embedding-8b",
        "jina-embeddings-v5-text-nano",
        "jina-embeddings-v5-text-small",
        "zembed-1-embedding",
        "jina-reranker-v3",
    ],
)
def test_shipped_recipe_files_carry_no_internal_labels(recipe_id: str) -> None:
    """Every shipped file of the dense recipes reads as a self-contained public statement: no
    internal process shorthand, private work directory or undefined rule id."""
    # the variant's family directory (decision 34: the recipes root holds families)
    family_dir = resolve_recipe(recipe_id)._dir
    assert family_dir is not None
    hits = [
        f"{path.name}:{number}: {line.strip()[:120]}"
        for path in sorted(family_dir.iterdir())
        if path.is_file()
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if INTERNAL_LABELS.search(line)
    ]
    assert not hits, "\n".join(hits)
