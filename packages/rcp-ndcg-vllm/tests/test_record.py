"""The recorder: one JSON file per exchange, no hostnames, bytes bodies base64 with their framing headers."""

from __future__ import annotations

import base64
import json
from typing import Any

import numpy as np
from rcp_ndcg_vllm import load_recipe
from rcp_ndcg_vllm.record import record

from tests.conftest import RECIPES, start_stub


def _load(recipe_id: str):
    return load_recipe(RECIPES / recipe_id)


def test_record_writes_one_file_per_exchange(tmp_path: Any) -> None:
    """The fixed request set: models, embeddings float/base64, pooling float/base64/bytes, rerank, score, errors."""
    recipe = _load("fixture-multi-vector")
    engine = start_stub()
    try:
        written = record(recipe, engine.base_url, tmp_path)
    finally:
        engine.stop()
    names = [path.name for path in written]
    assert len(names) == 10
    for expected in (
        "get-v1-models.json",
        "post-v1-embeddings-float.json",
        "post-v1-embeddings-base64.json",
        "post-pooling-float.json",
        "post-pooling-base64.json",
        "post-pooling-bytes.json",
        "post-rerank.json",
        "post-score.json",
        "post-v1-embeddings-overlength.json",
        "post-v1-embeddings-unknown-field.json",
    ):
        assert expected in names, expected
    assert all(str(path).startswith(str(tmp_path / "vllm-0.31.0" / "fixture-multi-vector")) for path in written)


def test_recorded_exchange_shape(tmp_path: Any) -> None:
    recipe = _load("fixture-embed")
    engine = start_stub("--served-model-name", recipe.id)
    try:
        written = record(recipe, engine.base_url, tmp_path)
    finally:
        engine.stop()
    by_name = {path.name: json.loads(path.read_text(encoding="utf-8")) for path in written}
    models = by_name["get-v1-models.json"]
    assert models["route"] == "/v1/models"
    assert models["request"]["url"] == "http://engine/v1/models"
    assert models["status"] == 200
    assert set(models["headers"]) == {"content-type", "server"}
    assert models["body"]["data"][0]["id"] == recipe.id
    embeddings = by_name["post-v1-embeddings-float.json"]
    assert embeddings["body"]["data"][0]["embedding"]
    assert embeddings["request"]["body"]["model"] == recipe.id


def test_bytes_bodies_are_base64_with_framing_headers(tmp_path: Any) -> None:
    recipe = _load("fixture-multi-vector")
    engine = start_stub()
    try:
        written = record(recipe, engine.base_url, tmp_path)
    finally:
        engine.stop()
    by_name = {path.name: json.loads(path.read_text(encoding="utf-8")) for path in written}
    raw = by_name["post-pooling-bytes.json"]
    body = raw["body"]
    assert raw["status"] == 200
    payload = base64.b64decode(body["base64"])
    assert body["framing_headers"]["content-type"] == "application/octet-stream"
    assert len(payload) % 8 == 0  # whole float16 rows of DIM=8
    vectors = np.frombuffer(payload, dtype="<f2")
    assert vectors.size > 0


def test_error_bodies_are_recorded(tmp_path: Any) -> None:
    """The two error bodies the adapters map: the over-length 400 and the unknown field."""
    recipe = _load("fixture-rerank-pointwise")
    engine = start_stub()
    try:
        written = record(recipe, engine.base_url, tmp_path)
    finally:
        engine.stop()
    by_name = {path.name: json.loads(path.read_text(encoding="utf-8")) for path in written}
    overlength = by_name["post-v1-embeddings-overlength.json"]
    assert overlength["status"] == 400
    assert "maximum context length" in overlength["body"]["error"]["message"]
    unknown = by_name["post-v1-embeddings-unknown-field.json"]
    assert unknown["status"] == 400
    assert "unknown field" in unknown["body"]["error"]["message"]


def test_no_hostname_or_secret_in_the_files(tmp_path: Any) -> None:
    """The base URL is replaced by the http://engine placeholder; nothing else identifies the host."""
    recipe = _load("fixture-embed")
    engine = start_stub()
    try:
        written = record(recipe, engine.base_url, tmp_path)
    finally:
        engine.stop()
    for path in written:
        text = path.read_text(encoding="utf-8")
        assert f"127.0.0.1:{engine.port}" not in text
        assert "localhost" not in text
