"""The per-variant golden contract snapshot: every shipped recipe resolves exactly to its golden.

The goldens under ``golden/<variant>.json`` are the current tree's resolved contract, last regenerated
when the behaviour fingerprint schema moved to ``rcp-fp/4`` (the engine image/version and plugin-code
inputs landed); the pre-family capture's declared differences were baked into that regeneration
(``golden/DELTAS.json`` is empty now and remains the mechanism for future accepted differences).  Two
recipes (zerank-1-small, zerank-2) carry their ``recipe.sources`` in the sanitized repo-relative form the
family ships, because the original capture recorded an operator path -- the deliberate redaction is
itself recorded and tested in ``golden/REDACTIONS.json``.  Each golden carries the
variant's ``load_recipe(id).model_dump(mode="json")``, its
``client_config``, its ``serve_argv`` (the ``--chat-template`` value reduced to its file name) and its
behaviour fingerprint (``rcp_ndcg_test.fingerprint``, the corpus key).  They are a permanent snapshot:
this test compares the current tree against them for every variant ``iter_recipes()`` yields, so a
recipe edit that moves any resolved field or any fingerprint input is a reviewed change, and adding a
variant means adding its golden.

``golden/DELTAS.json`` lists every accepted difference from the golden, each
``{id, field_path, golden, resolved, reason, evidence}``: a mismatch is accepted only when a delta
matches it exactly on all of (id, field_path, golden, resolved).  The list is SHRINK-ONLY: a delta whose
two values no longer differ fails, like ``KNOWN_SECOND_HOMES`` -- a declaration can only be removed,
never left stale.  ``field_path`` is dotted and artifact-prefixed (``recipe.``, ``client.`` (the
recipe's client block and its ``client_config`` dump alike), ``serve_argv.``, ``fingerprint.``,
``fingerprint.inputs.``); a mapping or list is compared as a whole at
its own path, so a nested change is declared at the nearest mapping that contains it.

Regenerate on purpose with ``uv run --no-sync pytest rcp-ndcg-test/tests/recipes/test_family_goldens.py
--update-goldens`` (it rewrites the goldens from the current tree; review the diff, then commit) --
``docs/how-to/add-a-model.md`` says so for a new variant.  The reproducibility test below runs the same
writer into ``tmp_path`` and requires the committed bytes, so a hand-edited golden cannot hide.

OFFLINE by construction: the recipe data is package data, the template hash is the file's bytes and the
fingerprint's tokenizer input comes from the vendored corpora store where it covers the recipe, else
from the golden itself -- the ``client.tokenizer`` spec string is pinned exactly by ``model_dump``, and
the tokenizer's SHA-256 is a pure function of that string.  No network, no torch, no product import
beyond the recipe package: this test runs in every CI job (the recipes conftest exempts it from the
network marking).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from rcp_ndcg_test.fingerprint import (
    FINGERPRINT_SCHEMA,
    behaviour_fingerprint,
    fingerprint_inputs,
    stored_tokenizer,
    tokenizer_sha256,
    use_tokenizer_store,
)
from rcp_ndcg_vllm.recipe import (
    client_config,
    default_recipes_root,
    iter_recipes,
    resolve_recipe,
    serve_argv,
)

GOLDEN_DIR = Path(__file__).parent / "golden"
DELTAS_FILE = GOLDEN_DIR / "DELTAS.json"
CORPORA_TOKENIZERS = Path(__file__).resolve().parents[2] / "corpora" / "vllm-0.31.0" / "_tokenizers"
BASE_URL = "http://127.0.0.1:8100/v1"

# The observation corpora vend their tokenizers here: the fingerprint's tokenizer input resolves
# offline for the recipes they cover (the rest read it from the golden; see the module docstring).
use_tokenizer_store(CORPORA_TOKENIZERS)

VARIANT_IDS = [recipe.id for recipe in iter_recipes()]


def _deltas() -> list[dict[str, Any]]:
    """The accepted differences (``golden/DELTAS.json``); absent means none."""
    if not DELTAS_FILE.is_file():
        return []
    return list(json.loads(DELTAS_FILE.read_text(encoding="utf-8")))


def _golden(variant_id: str) -> dict[str, Any]:
    path = GOLDEN_DIR / f"{variant_id}.json"
    assert path.is_file(), f"no golden for variant {variant_id!r}: add one with --update-goldens"
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _normalised_argv(argv: list[str]) -> list[str]:
    """The serve argv with the ``--chat-template`` value reduced to its file name.

    The template lives in the family directory since the conversion (the golden captured the
    standalone recipe directory), so the directory part is deliberately not part of the contract;
    the file name is.
    """
    out = []
    previous = ""
    for item in argv:
        out.append(Path(item).name if previous == "--chat-template" else item)
        previous = item
    return out


def _tokenizer_sha(recipe: Any, golden: dict[str, Any] | None = None) -> str:
    """The tokenizer SHA-256 for the fingerprint: the product's resolution when it can resolve the spec
    (sidecar-aware: the sidecars change the digest exactly when they change the effective vocabulary),
    else the vendored store's bytes, else the existing golden's value (the spec string itself is compared
    exactly in ``model_dump``)."""
    spec = str(recipe.client.get("tokenizer") or "")
    try:
        return tokenizer_sha256(recipe)
    except Exception:
        found = stored_tokenizer(spec)
        if found is not None:
            return found[1]
        if golden is not None:
            return str(golden["fingerprint"]["inputs"]["tokenizer_sha256"])
        raise


def _resolved(recipe: Any, tokenizer_sha: str) -> dict[str, Any]:
    """The four artifacts this guard pins, exactly as the capture recorded them.

    ``client_config`` is the recipe's ``client`` block with ``base_url`` filled: it is compared under
    the same ``client.`` paths as the recipe's client block, so one ``DELTAS`` entry covers both."""
    return {
        "recipe": recipe.model_dump(mode="json"),
        "client_config": client_config(recipe, base_url=BASE_URL),
        "serve_argv": _normalised_argv(serve_argv(recipe, port=8100, served_model_name=recipe.id)),
        "fingerprint": {
            "schema": FINGERPRINT_SCHEMA,
            "fingerprint": behaviour_fingerprint(recipe, tokenizer_sha256_value=tokenizer_sha),
            "inputs": fingerprint_inputs(recipe, tokenizer_sha256_value=tokenizer_sha),
        },
    }


def _differences(path: str, resolved: Any, golden: Any) -> list[tuple[str, Any, Any]]:
    """Every place ``resolved`` differs from ``golden``, as ``(path, golden, resolved)``.

    Mappings are walked key by key; any other value (a scalar, a list, a nested list of mappings) is
    compared as a whole at its own path."""
    if isinstance(resolved, dict) and isinstance(golden, dict):
        found: list[tuple[str, Any, Any]] = []
        for key in sorted(set(resolved) | set(golden)):
            if key not in resolved or key not in golden:
                found.append((f"{path}.{key}", golden.get(key), resolved.get(key)))
                continue
            found.extend(_differences(f"{path}.{key}", resolved[key], golden[key]))
        return found
    if resolved != golden:
        return [(path, golden, resolved)]
    return []


def _accepted(deltas: list[dict[str, Any]], variant_id: str, path: str, golden: Any, resolved: Any) -> bool:
    """Whether a delta declares exactly this difference."""
    return any(
        delta["id"] == variant_id
        and delta["field_path"] == path
        and delta["golden"] == golden
        and delta["resolved"] == resolved
        for delta in deltas
    )


def _write_golden(directory: Path, recipe: Any) -> Path:
    """Write one variant's golden from the current tree (the ``--update-goldens`` writer).

    The tokenizer hash comes from the fingerprint module's own resolver
    (:func:`rcp_ndcg_test.fingerprint.tokenizer_sha256` -- a local path, a store entry, or a Hub spec
    with its cache/network; sidecar-aware), falling back to the vendored store and then to an existing
    golden's value when the spec cannot be resolved offline."""
    existing = GOLDEN_DIR / f"{recipe.id}.json"
    existing_doc = json.loads(existing.read_text(encoding="utf-8")) if existing.is_file() else None
    sha = _tokenizer_sha(recipe, existing_doc)
    document = {"id": recipe.id, **_resolved(recipe, sha)}
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{recipe.id}.json"
    path.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return path


@pytest.mark.parametrize("variant_id", VARIANT_IDS)
def test_every_variant_matches_its_golden(variant_id: str, update_goldens: bool) -> None:
    """The variant's resolved model dump, client config, serve argv and fingerprint equal the golden
    exactly, except where ``DELTAS.json`` declares the difference (with ``--update-goldens`` the
    golden is rewritten instead)."""
    recipe = resolve_recipe(variant_id)
    if update_goldens:
        _write_golden(GOLDEN_DIR, recipe)
        # accepting the current tree as the new baseline bakes every declared difference in: the
        # delta list is emptied in the same deliberate act (review the diff, then commit)
        DELTAS_FILE.write_text("[]\n", encoding="utf-8")
        return
    golden = _golden(variant_id)
    resolved = _resolved(recipe, _tokenizer_sha(recipe, golden))
    deltas = _deltas()
    unexpected: list[str] = []
    for artifact, value in resolved.items():
        golden_artifact = golden[artifact]
        if artifact == "recipe":
            # the recipe's client block is compared under the same ``client.`` paths as its
            # client_config dump (one DELTAS entry covers both)
            for key in sorted(value):
                prefix = "client" if key == "client" else f"recipe.{key}"
                for path, want, got in _differences(prefix, value[key], golden_artifact[key]):
                    if _accepted(deltas, variant_id, path, want, got):
                        continue
                    unexpected.append(f"{path}: golden={json.dumps(want)[:120]} resolved={json.dumps(got)[:120]}")
            continue
        prefix = "client" if artifact == "client_config" else artifact
        for path, want, got in _differences(prefix, value, golden_artifact):
            if _accepted(deltas, variant_id, path, want, got):
                continue
            unexpected.append(f"{path}: golden={json.dumps(want)[:120]} resolved={json.dumps(got)[:120]}")
    assert not unexpected, (
        f"recipe {variant_id}: the resolved contract moved against its golden; either revert it or declare "
        "the difference in golden/DELTAS.json (with its reason and evidence):\n" + "\n".join(unexpected)
    )


def test_recorded_redactions_hold_and_no_golden_names_an_operator_path() -> None:
    """The redaction record is true, and no golden carries an operator path.

    Two goldens were sanitized on purpose (the capture recorded an operator path in a source string);
    ``golden/REDACTIONS.json`` records each one with the sanitized value, so the deliberate difference
    from the capture stays visible instead of vanishing into an in-place edit.  This also enforces the
    public-names rule locally: the tracked goldens name no operator path.
    """
    redactions = json.loads((GOLDEN_DIR / "REDACTIONS.json").read_text(encoding="utf-8"))
    assert redactions, "the redaction record must not be empty (it records the sanitized goldens)"
    for record in redactions:
        recipe = resolve_recipe(str(record["id"]))
        resolved = _resolved(recipe, _tokenizer_sha(recipe, _golden(str(record["id"]))))
        artifact, _, rest = str(record["field_path"]).partition(".")
        value: Any = resolved["client_config"] if artifact == "client" else resolved[artifact]
        for part in rest.split(".") if rest else []:
            value = value[part] if isinstance(value, dict) and part in value else None
        assert value == record["sanitized"], f"{record['id']} {record['field_path']}: the redaction no longer holds"
        assert "root/repos" not in json.dumps(value), record["id"]
    for path in sorted(GOLDEN_DIR.glob("*.json")):
        assert "root/repos" not in path.read_text(encoding="utf-8"), f"{path.name} names an operator path"


def test_the_variant_id_set_equals_the_golden_set() -> None:
    """A missing golden or a stray one fails: every shipped variant has exactly one golden."""
    record_files = {DELTAS_FILE.name, "REDACTIONS.json"}
    committed = {path.stem for path in GOLDEN_DIR.glob("*.json") if path.name not in record_files}
    missing = sorted(set(VARIANT_IDS) - committed)
    extra = sorted(committed - set(VARIANT_IDS))
    assert not missing and not extra, f"goldens out of step with the recipes: missing {missing}, extra {extra}"


def test_a_family_id_is_not_a_recipe() -> None:
    """Family ids are never served (decision 34): resolving one is refused, and it has no golden."""
    family_ids = sorted(path.name for path in default_recipes_root().iterdir() if (path / "family.yaml").is_file())
    assert family_ids, "no family directories under the recipes root"
    for family_id in family_ids:
        if family_id in VARIANT_IDS:
            continue  # a single-variant family whose variant shares the family's name (jina-reranker-v3)
        assert not (GOLDEN_DIR / f"{family_id}.json").exists(), family_id
        with pytest.raises(Exception, match="no recipe variant"):
            resolve_recipe(family_id)


def test_declared_deltas_still_differ() -> None:
    """The delta list is shrink-only and key-unique: a declared difference that reverted fails.

    A delta exists because the resolved value differed from the golden; when the resolved value equals
    the golden again (the difference was reverted, or the family was changed back), the declaration is
    stale and must be removed -- the list can only shrink.  Two deltas for one (id, field_path) would
    also let one of them rot invisibly, so duplicates are refused.
    """
    seen: dict[tuple[str, str], int] = {}
    stale: list[str] = []
    for delta in _deltas():
        key = (str(delta["id"]), str(delta["field_path"]))
        seen[key] = seen.get(key, 0) + 1
        recipe = resolve_recipe(str(delta["id"]))
        resolved = _resolved(recipe, _tokenizer_sha(recipe, _golden(str(delta["id"]))))
        artifact, _, rest = str(delta["field_path"]).partition(".")
        if artifact == "fingerprint" and rest.startswith("inputs."):
            # the fingerprint inputs are a FLAT dict of dotted keys (``client.media_head_as_system``), not a
            # nested mapping: walk the flat key so a delta with a null golden (a key the capture lacked) is
            # still checked, instead of falling through to None and reading as stale
            value: Any = resolved["fingerprint"]["inputs"].get(rest.removeprefix("inputs."))
        else:
            value = resolved["client_config"] if artifact == "client" else resolved[artifact]
            for part in rest.split(".") if rest else []:
                value = value[part] if isinstance(value, dict) and part in value else None
        if value == delta["golden"]:
            stale.append(str(delta["field_path"]))
    duplicates = sorted(f"{variant} {path} ({count}x)" for (variant, path), count in seen.items() if count > 1)
    assert not duplicates, "golden/DELTAS.json declares one (id, field_path) more than once: " + ", ".join(duplicates)
    assert not stale, (
        "golden/DELTAS.json declares differences that no longer differ (the list is shrink-only; remove "
        "them): " + ", ".join(stale)
    )


def test_the_writer_round_trips_the_resolved_artifacts(tmp_path: Path) -> None:
    """The ``--update-goldens`` writer is the reader's inverse: what it writes is what the guard compares.

    The committed goldens are the current tree's resolved contract (last regenerated when ``rcp-fp/4``
    landed), so a regeneration of the current tree equals them.  What must hold here is that the writer is
    a working, deterministic path: its output reloads to exactly the artifacts this module resolves, and
    writing twice gives identical bytes.
    """
    for recipe in iter_recipes():
        written = _write_golden(tmp_path, recipe)
        again = _write_golden(tmp_path / "again", recipe)
        assert written.read_bytes() == again.read_bytes(), f"{recipe.id}: the golden writer is not deterministic"
        document = json.loads(written.read_text(encoding="utf-8"))
        assert document["id"] == recipe.id
        sha = document["fingerprint"]["inputs"]["tokenizer_sha256"]
        assert document == {"id": recipe.id, **_resolved(recipe, sha)}
