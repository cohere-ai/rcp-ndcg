"""The exported JSON Schemas: ids, the schema tag of command outputs, and the export."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rcp_ndcg import schemas
from rcp_ndcg.errors import UsageError
from tests._jsonschema import problems


def test_every_schema_carries_its_id_and_url() -> None:
    assert schemas.SCHEMA_ID_BASE.endswith("/rcp-ndcg/blob/main/schemas")
    for entry in schemas.entries():
        schema = schemas.show(entry.name)

        assert schema["$id"] == f"{schemas.SCHEMA_ID_BASE}/{entry.name}.v1.json"
        assert schema["x-rcp-ndcg-schema"] == f"rcp-ndcg.{entry.name}.v1"
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"


def test_a_command_output_schema_requires_its_tag() -> None:
    schema = schemas.show("rcp-ndcg.run-list.v1")

    assert schema["properties"]["schema"] == {"const": "rcp-ndcg.run-list.v1", "type": "string"}
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
        const = schemas.show(entry.name).get("properties", {}).get("schema", {}).get("const")
        if const is not None:
            assert const == entry.schema_id, entry.name
            tagged += 1
    assert tagged >= 4
    assert schemas.show("judge-config")["x-rcp-ndcg-schema"] == "rcp-ndcg.judge-config.v1"


def test_an_unknown_name_is_a_usage_error_listing_the_known_ones() -> None:
    with pytest.raises(UsageError) as caught:
        schemas.show("nope")

    assert "run-manifest" in caught.value.details["known"]


def test_export_writes_one_file_per_schema(tmp_path: Path) -> None:
    written = schemas.export(tmp_path / "out")

    assert sorted(path.name for path in written) == sorted(f"{name}.v1.json" for name in schemas.names())
    assert all(json.loads(path.read_text(encoding="utf-8"))["$id"].endswith(path.name) for path in written)


def test_the_shipped_run_configs_have_the_types_the_run_config_schema_declares() -> None:
    import yaml

    schema = schemas.show("run-config")
    from rcp_ndcg.examples import run_config_names, run_config_path

    paths = [run_config_path(name) for name in run_config_names()]
    assert len(paths) > 1
    for path in paths:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        for key, value in config.items():
            assert key in schema["properties"], (path.name, key)
            assert problems(schema["properties"][key], value, root=schema) == [], (path.name, key, value)


def test_every_model_that_tags_its_payload_has_an_exported_schema() -> None:
    """A payload that names ``rcp-ndcg.<name>.v1`` promises a schema of that name (the manifest, a calibration's
    coverage and identity, an index record, ...)."""
    import importlib
    import pkgutil
    import typing

    from pydantic import BaseModel

    import rcp_ndcg

    exported = {entry.schema_id for entry in schemas.entries()}
    tagged: dict[str, str] = {}
    for info in pkgutil.walk_packages(rcp_ndcg.__path__, "rcp_ndcg."):
        if info.name.endswith("__main__"):
            continue
        try:
            module = importlib.import_module(info.name)
        except ImportError:  # an optional extra
            continue
        for obj in vars(module).values():
            if isinstance(obj, type) and issubclass(obj, BaseModel) and "schema_name" in obj.model_fields:
                (tag,) = typing.get_args(obj.model_fields["schema_name"].annotation)
                tagged[tag] = f"{obj.__module__}.{obj.__qualname__}"
    missing = {tag: where for tag, where in tagged.items() if tag not in exported}
    assert not missing, missing


def _runner_sections_of_the_docs() -> list[tuple[str, dict]]:
    """Every YAML block of the docs that holds a ``runner:`` section, as ``(where, mapping)``."""
    import re

    import yaml

    repo = Path(__file__).resolve().parents[1]
    blocks = []
    for page in sorted((repo / "docs").rglob("*.md")):
        for index, block in enumerate(re.findall(r"```yaml\n(.*?)```", page.read_text(encoding="utf-8"), re.S)):
            for data in yaml.safe_load_all(block):  # a Kubernetes example holds several documents
                if isinstance(data, dict) and "runner" in data:
                    blocks.append((f"{page.name}#{index}", data))
    return blocks


def test_a_written_run_config_and_the_docs_runner_examples_validate_against_the_run_config_schema() -> None:
    """Every runner section matched two branches of the schema's oneOf (a public runner's and the plugin's), so no
    written run.yaml validated; and a public runner's option typo matched the plugin's branch, so it did."""
    from rcp_ndcg.runs import RunConfig

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
    typo = {"name": "slurm", "options": {"partitoin": "gpu"}}
    assert problems(schema["properties"]["runner"], typo, root=schema) != []
