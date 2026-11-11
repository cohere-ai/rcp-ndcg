"""``embedding_types`` is a content declaration: it moves an identity and the index cache key.

The field names the quantized representations a run uses. A run with ``embedding_types: [int8]`` computes
different vectors from the same float reply, so it must never share an identity or a cache entry with a run
that declares none.
"""

from __future__ import annotations

from pathlib import Path

from rcp.inference.config import EmbeddingEndpoint
from rcp.retrieval._api import _identity
from rcp.retrieval.config import CohereEmbedding, DenseConfig
from rcp.support.identity import FieldRole, check_declarations, identity_payload
from rcp.support.paths import cache_dir


def test_embedding_types_is_undeclared_by_default() -> None:
    assert EmbeddingEndpoint(api="cohere", model="embed-v4.0").embedding_types is None


def test_embedding_types_is_a_content_field() -> None:
    assert EmbeddingEndpoint.IDENTITY_ROLES["embedding_types"] is FieldRole.CONTENT
    check_declarations(EmbeddingEndpoint)


def test_an_empty_declaration_is_the_unset_one() -> None:
    plain = CohereEmbedding(model="embed-v4.0")
    empty = CohereEmbedding(model="embed-v4.0", embedding_types=[])

    assert empty.embedding_types is None
    assert identity_payload(empty) == identity_payload(plain)


def test_a_declared_type_enters_the_identity_and_an_unset_one_does_not() -> None:
    plain = CohereEmbedding(model="embed-v4.0")
    quantized = CohereEmbedding(model="embed-v4.0", embedding_types=["int8"])

    assert plain.embedding_types is None
    assert quantized.embedding_types == ("int8",)
    assert "embedding_types" not in identity_payload(plain)
    assert identity_payload(quantized)["embedding_types"] == ["int8"]
    assert identity_payload(plain) != identity_payload(quantized)


def test_the_exported_run_config_schema_declares_the_field() -> None:
    from rcp.schemas import show

    defs = show("run-config")["$defs"]
    for config_name in ("CohereEmbedding", "ServedEmbedding", "VoyageEmbedding"):
        field = defs[config_name]["properties"]["embedding_types"]
        assert field["anyOf"][0]["items"]["enum"] == ["int8", "uint8", "binary"]


def _index_cache_key(identity: str) -> Path:
    """The default index directory ``retrieve`` derives from an identity (``cache_dir()/indexes/<prefix>``)."""
    return cache_dir() / "indexes" / identity[:16]


def test_the_two_configs_produce_different_identities_and_cache_keys() -> None:
    plain = DenseConfig(encoder=CohereEmbedding(model="embed-v4.0"))
    quantized = DenseConfig(encoder=CohereEmbedding(model="embed-v4.0", embedding_types=["int8"]))

    plain_identity = _identity(plain, ["d1"], [])
    quantized_identity = _identity(quantized, ["d1"], [])

    assert plain_identity != quantized_identity
    assert _index_cache_key(plain_identity) != _index_cache_key(quantized_identity)
