"""The files that carry the calibration and judgement-store tags match their exported JSON Schemas."""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path

import pytest

from rcp_ndcg import schemas
from rcp_ndcg.calibration.fit import Calibration
from rcp_ndcg.errors import DataError
from rcp_ndcg.judging.client import EngineInfo
from rcp_ndcg.judging.store import JudgementStore
from rcp_ndcg.testing import TinyWorld
from tests._jsonschema import problems


def test_a_saved_calibration_items_file_matches_the_calibration_schema(world: TinyWorld) -> None:
    schema = schemas.show("calibration")
    items = json.loads((world.calibration / "items.json").read_text(encoding="utf-8"))

    assert schema["x-rcp-ndcg-schema"] == items["schema"] == "rcp-ndcg.calibration.v1"
    assert problems(schema, items) == []
    assert problems(schema, {**items, "gamma": "1.0"}) != []  # the check is not vacuous


def test_a_judgement_store_identity_matches_the_judgement_store_schema(world: TinyWorld, tmp_path: Path) -> None:
    schema = schemas.show("judgement-store")
    store = JudgementStore(tmp_path / "store")
    store.root.mkdir()
    shutil.copyfile(world.judgements / "identity.json", store.identity_path)
    store.note_engines("rubric", [EngineInfo(url="http://127.0.0.1:8000/v1", model="fake", owned_by="vllm")])
    identity = json.loads(store.identity_path.read_text(encoding="utf-8"))

    assert schema["x-rcp-ndcg-schema"] == identity["schema"] == "rcp-ndcg.judgement-store.v1"
    assert set(identity["stages"]) == {"tournament", "rubric"}
    assert identity["stages"]["rubric"]["engines"]
    assert problems(schema, identity) == []
    assert problems(schema, {**identity, "stages": {"rubric": {}}}) != []


def test_a_store_identity_that_breaks_its_schema_is_a_data_error(world: TinyWorld, tmp_path: Path) -> None:
    store = JudgementStore(tmp_path / "store")
    store.root.mkdir()
    identity = json.loads((world.judgements / "identity.json").read_text(encoding="utf-8"))
    del identity["stages"]["rubric"]["family"]
    store.identity_path.write_text(json.dumps(identity), encoding="utf-8")

    with pytest.raises(DataError, match="judgement-store.v1") as caught:
        store.identities()

    assert caught.value.details["errors"][0]["loc"] == ("stages", "rubric", "family")


def test_a_calibration_items_file_that_breaks_its_schema_is_a_data_error(world: TinyWorld, tmp_path: Path) -> None:
    target = tmp_path / "calibration"
    shutil.copytree(world.calibration, target)
    items = json.loads((target / "items.json").read_text(encoding="utf-8"))
    del items["gamma"]
    (target / "items.json").write_text(json.dumps(items), encoding="utf-8")

    with pytest.raises(DataError, match="calibration.v1") as caught:
        Calibration.load(target)

    assert caught.value.details["errors"][0]["loc"] == ("gamma",)


def _corrupt_thetas(world: TinyWorld, tmp_path: Path, **updates) -> Path:
    import pandas as pd

    target = tmp_path / "calibration"
    shutil.copytree(world.calibration, target)
    frame = pd.read_parquet(target / "thetas.parquet")
    for column, value in updates.items():
        frame.loc[0, column] = value
    frame.to_parquet(target / "thetas.parquet", index=False)
    return target


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        ({"source": "guessed"}, "source 'guessed'"),
        ({"theta": float("nan")}, "theta of"),
        ({"theta": float("inf")}, "theta of"),
        ({"theta_se": float("inf")}, "standard error of"),
    ],
)
def test_a_corrupt_thetas_row_is_a_data_error(world: TinyWorld, tmp_path: Path, updates: dict, match: str) -> None:
    """thetas.parquet rows are read straight into ThetaRow: an unknown source or a non-finite number used to
    load silently (a NaN theta then calibrated NaN gains), unlike every JSON file of the layout."""
    target = _corrupt_thetas(world, tmp_path, **updates)

    with pytest.raises(DataError, match=match) as caught:
        Calibration.load(target)

    assert "thetas.parquet" in caught.value.message
    assert caught.value.hint


def test_a_valid_row_with_a_missing_se_still_loads(world: TinyWorld, tmp_path: Path) -> None:
    target = _corrupt_thetas(world, tmp_path, theta_se=float("nan"))

    loaded = Calibration.load(target)

    assert loaded.thetas[0].theta_se is None, "a NaN SE is the missing one, not a zero"


def test_saving_a_non_finite_number_is_a_data_error_not_a_bare_value_error(world: TinyWorld, tmp_path: Path) -> None:
    """json.dumps(allow_nan=False) raises a bare ValueError: a NaN in an artifact names the file instead."""
    calibration = Calibration.load(world.calibration)
    reliability = calibration.diagnostics.reliability
    broken = dataclasses.replace(
        calibration,
        diagnostics=calibration.diagnostics.model_copy(
            update={
                "reliability": reliability.model_copy(
                    update={"overall": reliability.overall.model_copy(update={"ece": float("nan")})}
                )
            }
        ),
    )

    with pytest.raises(DataError, match="diagnostics.json") as caught:
        broken.save(tmp_path / "broken")

    assert "JSON" in caught.value.message and caught.value.hint


def test_every_line_of_extensions_jsonl_matches_the_extension_record_schema(world: TinyWorld, tmp_path: Path) -> None:
    from rcp_ndcg.calibration import read_judgements, score_documents

    calibration = Calibration.load(world.calibration)
    extended = calibration.extended(score_documents(calibration, read_judgements(world.judgements)))
    extended.save(tmp_path / "extended")
    lines = (tmp_path / "extended" / "extensions.jsonl").read_text(encoding="utf-8").splitlines()
    schema = schemas.show("extension-record")

    assert lines and schema["x-rcp-ndcg-schema"] == "rcp-ndcg.extension-record.v1"
    for line in lines:
        record = json.loads(line)
        assert record["schema"] == "rcp-ndcg.extension-record.v1"
        assert problems(schema, record) == []
    assert problems(schema, {**json.loads(lines[0]), "source": "guessed"}) != []
