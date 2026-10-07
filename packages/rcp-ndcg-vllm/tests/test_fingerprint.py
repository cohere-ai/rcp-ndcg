"""The recipe behaviour fingerprint: stable, named inputs, and nothing that cannot change a model output.

``rcp_ndcg_vllm.fingerprint.behaviour_fingerprint`` keys the model layer of the observation corpora
(GPU-VALIDATION.md item 8): it is the SHA-256 of everything that can change what the model returns -- model id
and revision, the ``serve`` block, the template file's bytes, the tokenizer's ``tokenizer.json`` SHA-256 and
the client fields that shape the request. Recipe identity and harness metadata (``id``, ``notes``,
``status``, ``reference``, ``gates``) are out, and so are rcp-ndcg's internals: refactoring the package never
needs a re-recording. :func:`fingerprint_inputs` returns the named inputs and their values, so a staleness
failure can name what changed.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import pytest
import yaml
from rcp_ndcg_vllm.errors import HarnessError
from rcp_ndcg_vllm.fingerprint import behaviour_fingerprint, fingerprint_inputs
from rcp_ndcg_vllm.recipe import load_recipe

RECIPES = Path(__file__).resolve().parent / "fixtures" / "recipes"
EMBED = RECIPES / "fixture-embed"
#: The fixture tokenizer the recipes name (``../../tokenizer.json`` relative to the recipe directory).
TOKENIZER = (RECIPES / ".." / "tokenizer.json").resolve()


def _load():
    return load_recipe(EMBED)


def _copy(tmp_path: Path, rewrite=None, *, rename: str | None = None, template: str | None = None) -> Path:
    """A copy of the fixture recipe under ``tmp_path``, optionally rewritten, renamed and given a template file."""
    name = rename or EMBED.name
    directory = tmp_path / name
    shutil.copytree(EMBED, directory)
    text = (directory / "recipe.yaml").read_text(encoding="utf-8").replace("../../tokenizer.json", str(TOKENIZER))
    if rename:
        text = text.replace(f"id: {EMBED.name}", f"id: {rename}")
    if template is not None:
        (directory / "template.jinja").write_text(template, encoding="utf-8")
        text = text.replace("chat_template: null", "chat_template: template.jinja")
    if rewrite is not None:
        data = rewrite(yaml.safe_load(text))
        text = yaml.safe_dump(data, sort_keys=False)
    (directory / "recipe.yaml").write_text(text, encoding="utf-8")
    return directory


def _changed(before: dict[str, str], after: dict[str, str]) -> set[str]:
    """The input names whose values differ between two fingerprint-input maps (added or removed included)."""
    return {name for name in set(before) | set(after) if before.get(name) != after.get(name)}


def test_the_fingerprint_is_a_stable_sha256_of_the_named_inputs() -> None:
    from rcp_ndcg_vllm.fingerprint import FINGERPRINT_SCHEMA

    recipe = _load()
    assert re.fullmatch(r"[0-9a-f]{64}", behaviour_fingerprint(recipe))
    assert behaviour_fingerprint(_load()) == behaviour_fingerprint(recipe)
    inputs = fingerprint_inputs(recipe)
    assert all(isinstance(value, str) for value in inputs.values())
    payload = FINGERPRINT_SCHEMA + "\n" + json.dumps(inputs, sort_keys=True, separators=(",", ":"))
    assert behaviour_fingerprint(recipe) == hashlib.sha256(payload.encode("utf-8")).hexdigest()
    assert inputs["fingerprint_schema"] == FINGERPRINT_SCHEMA
    assert sys.getdefaultencoding() == "utf-8"


def test_every_input_gpu_validation_item_8_names_is_present() -> None:
    inputs = fingerprint_inputs(_load())
    assert inputs["model"] == "fixtures/DenseEmbedder"
    assert inputs["revision"] == "0123456789abcdef0123456789abcdef01234567"
    assert "serve.max_model_len" in inputs and "serve.dtype" in inputs
    assert "serve.pooler_config" in inputs and "serve.hf_overrides" in inputs
    assert inputs["template_file"] == "absent"  # fixture-embed ships no template file
    assert inputs["tokenizer_sha256"] == hashlib.sha256(TOKENIZER.read_bytes()).hexdigest()
    assert "client.template" in inputs and "client.max_tokens" in inputs
    assert "client.api" in inputs


def test_a_serve_change_is_named(tmp_path: Path) -> None:
    before = fingerprint_inputs(_load())

    def rewrite(data: dict) -> dict:
        data["serve"]["max_model_len"] = 256
        return data

    edited = _copy(tmp_path, rewrite=rewrite)
    after = fingerprint_inputs(load_recipe(edited))
    assert behaviour_fingerprint(load_recipe(edited)) != behaviour_fingerprint(_load())
    assert _changed(before, after) == {"serve.max_model_len"}


def test_a_template_change_is_named(tmp_path: Path) -> None:
    before = fingerprint_inputs(_load())
    edited = _copy(tmp_path, template="before: {{ query }}")
    changed = _changed(before, fingerprint_inputs(load_recipe(edited)))
    assert "template_file" in changed  # the file's bytes
    assert "serve.chat_template" in changed  # the serve block now names a file
    assert behaviour_fingerprint(load_recipe(edited)) != behaviour_fingerprint(_load())


def test_the_same_template_file_name_with_different_bytes_names_the_template_alone(tmp_path: Path) -> None:
    one = _copy(tmp_path / "one", template="one: {{ query }}", rename="fixture-embed-a")
    two = _copy(tmp_path / "two", template="two: {{ query }}", rename="fixture-embed-b")
    before, after = fingerprint_inputs(load_recipe(one)), fingerprint_inputs(load_recipe(two))
    assert _changed(before, after) == {"template_file"}


def test_a_client_template_change_is_named(tmp_path: Path) -> None:
    def rewrite(data: dict) -> dict:
        data["client"]["template"]["document"] = [{"fixed": "doc: "}, {"content": "document"}]
        return data

    before = fingerprint_inputs(_load())
    after = fingerprint_inputs(load_recipe(_copy(tmp_path, rewrite=rewrite)))
    assert _changed(before, after) == {"client.template"}


def test_a_tokenizer_change_is_named(tmp_path: Path) -> None:
    edited = _copy(tmp_path)
    other = tmp_path / "other-tokenizer.json"
    other.write_bytes(TOKENIZER.read_bytes() + b"\n")
    text = (edited / "recipe.yaml").read_text(encoding="utf-8").replace(str(TOKENIZER), str(other))
    (edited / "recipe.yaml").write_text(text, encoding="utf-8")
    before, after = fingerprint_inputs(_load()), fingerprint_inputs(load_recipe(edited))
    assert _changed(before, after) == {"tokenizer_sha256"}


def test_harness_metadata_and_recipe_identity_are_out(tmp_path: Path) -> None:
    def rewrite(data: dict) -> dict:
        data["notes"] = "another note"
        data["status"] = {"state": "verified", "image": "vllm/vllm-openai:v0.31.0", "date": "2026-02-02", "report": "x"}
        data["reference"] = {"kind": "transformers", "score_scale": "logit", "entry": "reference.py"}
        data["gates"] = {"tau_min": 0.5}
        return data

    before = fingerprint_inputs(_load())
    after = fingerprint_inputs(load_recipe(_copy(tmp_path, rewrite=rewrite)))
    assert after == before
    # A renamed recipe (new id and directory, same behaviour) keys the same fingerprint.
    renamed = _copy(tmp_path, rewrite=rewrite, rename="fixture-embed-renamed")
    assert fingerprint_inputs(load_recipe(renamed))["model"] == "fixtures/DenseEmbedder"


def test_a_missing_tokenizer_raises_with_a_hint(tmp_path: Path) -> None:
    edited = _copy(tmp_path)
    text = (edited / "recipe.yaml").read_text(encoding="utf-8").replace(str(TOKENIZER), str(tmp_path / "missing.json"))
    (edited / "recipe.yaml").write_text(text, encoding="utf-8")
    with pytest.raises(HarnessError) as error:
        fingerprint_inputs(load_recipe(edited))
    assert "tokenizer" in str(error.value).lower()
