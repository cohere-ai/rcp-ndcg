"""The exported JSON Schemas: ids, the schema tag of command outputs, and the export."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rcp import schemas
from rcp.errors import UsageError
from tests._jsonschema import problems


def test_every_schema_carries_its_id_and_url() -> None:
    assert schemas.SCHEMA_ID_BASE.endswith("/rcp-ndcg/blob/main/schemas")
    for entry in schemas.entries():
        schema = schemas.show(entry.name)

        assert schema["$id"] == f"{schemas.SCHEMA_ID_BASE}/{entry.name}.v1.json"
        assert schema["x-rcp-schema"] == f"rcp.{entry.name}.v1"
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"


def test_a_command_output_schema_requires_its_tag() -> None:
    schema = schemas.show("rcp.run-list.v1")

    assert schema["properties"]["schema"] == {"const": "rcp.run-list.v1", "type": "string"}
    assert schema["required"][0] == "schema"


def test_the_kinds_cover_configs_artifacts_and_outputs() -> None:
    kinds = {entry.name: entry.kind for entry in schemas.entries()}

    assert kinds["run-config"] == "config"
    assert kinds["run-manifest"] == "artifact"
    assert kinds["cli"] == kinds["commands"] == kinds["dataset-summary"] == "cli-output"
    assert set(kinds.values()) == {"config", "artifact", "cli-output"}


def test_a_record_that_names_its_schema_names_its_entry() -> None:
    # A judgement, a manifest, an estimate and an extension carry their schema id; it must be the exported one.
    tagged = 0
    for entry in schemas.SCHEMAS:
        field = schemas.show(entry.name).get("properties", {}).get("schema", {})
        current = field.get("const") or (field.get("enum") or [None])[0]
        if current is not None:
            assert current == entry.schema_id, entry.name
            tagged += 1
    assert tagged >= 4
    assert schemas.show("judge-config")["x-rcp-schema"] == "rcp.judge-config.v1"


def test_a_record_admits_both_spellings_and_writes_the_new_one() -> None:
    """An artifact written before the rename carries ``rcp-ndcg.<name>.v1``: every exported record admits both
    spellings, and its own default is the new ``rcp.<name>.v1``."""
    records = 0
    for entry in schemas.SCHEMAS:
        field = schemas.show(entry.name)["properties"].get("schema", {})
        accepted = field.get("enum") or [field.get("const")]
        if None in accepted:
            continue
        assert accepted == [f"rcp.{entry.name}.v1", f"rcp-ndcg.{entry.name}.v1"], entry.name
        assert field["default"] == f"rcp.{entry.name}.v1", entry.name
        records += 1
    assert records >= 4


def test_an_unknown_name_is_a_usage_error_listing_the_known_ones() -> None:
    with pytest.raises(UsageError) as caught:
        schemas.show("nope")

    assert "run-manifest" in caught.value.details["known"]


def test_a_schema_id_from_before_the_rename_names_the_same_schema() -> None:
    """``rcp-ndcg.<name>.v1`` is the id an artifact written before the rename carries: ``show`` takes it (``_key``
    strips either prefix), so a reader can paste the tag it found in an old artifact."""
    assert schemas.show("rcp-ndcg.run-manifest.v1") == schemas.show("rcp.run-manifest.v1")


def test_export_writes_one_file_per_schema(tmp_path: Path) -> None:
    written = schemas.export(tmp_path / "out")

    assert sorted(path.name for path in written) == sorted(f"{name}.v1.json" for name in schemas.names())
    assert all(json.loads(path.read_text(encoding="utf-8"))["$id"].endswith(path.name) for path in written)


def test_the_shipped_run_configs_have_the_types_the_run_config_schema_declares() -> None:
    import yaml

    schema = schemas.show("run-config")
    from rcp.examples import run_config_names, run_config_path

    paths = [run_config_path(name) for name in run_config_names()]
    assert len(paths) > 1
    for path in paths:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        for key, value in config.items():
            assert key in schema["properties"], (path.name, key)
            assert problems(schema["properties"][key], value, root=schema) == [], (path.name, key, value)


def test_every_model_that_tags_its_payload_has_an_exported_schema() -> None:
    """A payload that names a schema id promises a schema of that name (the manifest, a calibration's coverage
    and identity, an index record, ...). Every spelling a model admits is checked: since the rename each payload
    tag admits both ``rcp.<name>.v1`` and the pre-rename ``rcp-ndcg.<name>.v1``."""
    import importlib
    import pkgutil
    import typing

    from pydantic import BaseModel
    from rcp_core.schemas import accepted_schema_ids

    import rcp

    names = {entry.name for entry in schemas.entries()}
    exported = {spelling: name for name in names for spelling in accepted_schema_ids(name)}
    tagged: dict[str, str] = {}
    for info in pkgutil.walk_packages(rcp.__path__, "rcp."):
        if info.name.endswith("__main__"):
            continue
        try:
            module = importlib.import_module(info.name)
        except ImportError:  # an optional extra
            continue
        for obj in vars(module).values():
            if isinstance(obj, type) and issubclass(obj, BaseModel) and "schema_name" in obj.model_fields:
                annotation = obj.model_fields["schema_name"].annotation
                if typing.get_origin(annotation) is not typing.Literal:
                    # Not a payload tag: an artifact's own schema id is a plain ``str | None``.
                    continue
                for tag in typing.get_args(annotation):
                    tagged[tag] = f"{obj.__module__}.{obj.__qualname__}"
    missing = {tag: where for tag, where in tagged.items() if tag not in exported}
    assert not missing, missing
    assert len({exported[tag] for tag in tagged}) >= 4, tagged  # the walk found the payload records


def _runner_sections_of_the_docs() -> list[tuple[str, dict]]:
    """Every YAML block of the docs that holds a ``runner:`` section, as ``(where, mapping)``."""
    import re

    import yaml

    repo = Path(__file__).resolve().parents[1]
    blocks = []
    for page in sorted((repo / "docs").rglob("*.md")):
        for index, block in enumerate(re.findall(r"```yaml\n(.*?)```", page.read_text(encoding="utf-8"), re.S)):
            for data in yaml.safe_load_all(block):  # a Kubernetes example holds several documents
                if isinstance(data, dict) and ("runner" in data or {"dataset", "steps"} <= data.keys()):
                    blocks.append((f"{page.name}#{index}", data))
    return blocks


def test_a_written_run_config_and_the_docs_runner_examples_validate_against_the_run_config_schema() -> None:
    """Every runner section matched two branches of the schema's oneOf (a public runner's and the plugin's), so no
    written run.yaml validated; and a public runner's option typo matched the plugin's branch, so it did."""
    from rcp.runs import RunConfig

    schema = schemas.show("run-config")
    runners = [
        None,
        {"name": "slurm", "options": {"partition": "gpu", "resources": {"gpus": 1}, "env": {"HF_HOME": "/hf"}}},
        {"name": "kubernetes", "options": {"namespace": "eval"}},
        {"name": "my-scheduler", "options": {"queue": "a"}},
    ]
    for runner in runners:
        fields = {"dataset": "jsonl:rows.jsonl", "judge": "fake", **({"runner": runner} if runner else {})}
        written = RunConfig.model_validate(fields).resolved()  # what run.yaml and the manifest record
        assert problems(schema, written) == [], (runner, problems(schema, written))
    documented = _runner_sections_of_the_docs()
    assert len(documented) >= 2
    for where, data in documented:
        for key, value in data.items():
            assert problems(schema["properties"][key], value, root=schema) == [], (where, key)
        if "dataset" in data and "steps" in data:  # a complete run config, not a runner fragment
            RunConfig.model_validate(data)  # the loader `run start` uses: a documented example must load today
    typo = {"name": "slurm", "options": {"partitoin": "gpu"}}
    assert problems(schema["properties"]["runner"], typo, root=schema) != []


def test_the_eval_score_schema_describes_the_results_fields() -> None:
    """``EvalScoreResult``'s wrap serializer returns ``dict[str, Any]``, and pydantic takes that return type
    for the serialization-mode schema: the exported ``rcp.eval-score.v1`` (and the ``eval_score`` MCP
    tool's outputSchema) described one property, ``schema``, with ``additionalProperties: true`` -- an empty
    contract for the output of the main ``--json`` command."""
    from rcp_core.protocol import Protocol

    from rcp.cli.eval import EvalScoreResult

    schema = schemas.show("eval-score")
    assert {
        "protocol",
        "gains_source",
        "metrics",
        "k",
        "summary",
        "per_dataset",
        "per_query",
        "warnings",
        "inputs",
        "out",
    } <= set(schema["properties"])
    data = {
        "schema": "rcp.eval-score.v1",
        **EvalScoreResult(protocol=Protocol(name="plain", ties="doc_id_desc"), k=[10]).model_dump(mode="json"),
    }
    assert problems({**schema, "additionalProperties": False}, data) == []
