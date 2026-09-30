"""The canonical serialiser: what it refuses, and what it must not conflate.

Every identity in this project is a digest of these bytes, so two properties
matter and the tests here are about nothing else:

* **determinism** -- the same value gives the same bytes, in this process and the
  next. A key that changes between processes reads as "nothing is cached" every
  time, which is a silent full recompute;
* **injectivity where it counts** -- two values that mean different things must
  not encode identically, or two different configs share one key (false sharing:
  wrong numbers, no error).

The refusals are the interesting part. ``json.dumps(..., default=str)`` accepts
anything and encodes an unknown object as its ``repr``, so ``<Foo object at 0x7f3c…>`` becomes part of a key and the key
moves when the allocator does.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import UTC, date, datetime, timedelta, timezone
from enum import Enum, IntEnum, StrEnum
from pathlib import Path, PurePosixPath

import pytest
from pydantic import BaseModel
from rcp_ndcg_core._hashing import (
    UnhashableValueError,
    canonical,
    canonical_json,
    hash_payload,
    hash_text,
    short,
)


class TestDeterminism:
    def test_key_order_does_not_matter(self) -> None:
        assert canonical_json({"a": 1, "b": 2}) == canonical_json({"b": 2, "a": 1})

    def test_nested_key_order_does_not_matter(self) -> None:
        assert canonical_json({"x": {"a": 1, "b": 2}}) == canonical_json({"x": {"b": 2, "a": 1}})

    def test_stable_across_processes(self) -> None:
        # PYTHONHASHSEED randomises dict/str hashing per process; a serialiser that
        # leaked iteration order would differ between runs and no in-process test
        # could see it.
        program = (
            "from rcp_ndcg.support.identity import hash_payload;"
            "print(hash_payload({'b': 2, 'a': 1, 'c': {'z': [1, 2], 'y': 'x'}}))"
        )
        digests = {
            subprocess.run(  # noqa: S603 - fixed argv, no shell
                [sys.executable, "-c", program],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            for _ in range(3)
        }
        assert len(digests) == 1
        assert digests.pop() == hash_payload({"a": 1, "b": 2, "c": {"y": "x", "z": [1, 2]}})

    def test_text_is_utf8_not_escaped(self) -> None:
        # ``ensure_ascii=False``: the bytes are the text. A query in Chinese hashes
        # its own bytes rather than an escaped transliteration.
        assert canonical_json({"q": "\u6f22"}) == '{"q":"漢"}'.encode()


class TestRefusals:
    def test_arbitrary_object(self) -> None:
        class Opaque:
            def __str__(self) -> str:  # the trap: it looks serialisable
                return "opaque"

        with pytest.raises(UnhashableValueError, match="no declared canonical form"):
            canonical_json({"cfg": Opaque()})

    def test_the_refusal_names_the_path(self) -> None:
        class Opaque:
            pass

        with pytest.raises(UnhashableValueError, match=r"payload\.outer\.inner\[1\]"):
            canonical_json({"outer": {"inner": [1, Opaque()]}})

    def test_set(self) -> None:
        # A set has no order, so its JSON array has no canonical form.
        with pytest.raises(UnhashableValueError, match="no canonical order"):
            canonical_json({"tags": {"b", "a"}})

    def test_bytes(self) -> None:
        with pytest.raises(UnhashableValueError, match="no canonical JSON form"):
            canonical_json({"blob": b"\x00\x01"})

    def test_nan_and_infinity(self) -> None:
        # NaN != NaN, so an identity containing one never matches itself.
        for value in (float("nan"), float("inf"), float("-inf")):
            with pytest.raises(UnhashableValueError, match="non-finite"):
                canonical_json({"x": value})

    def test_naive_datetime(self) -> None:
        with pytest.raises(UnhashableValueError, match="names no instant"):
            canonical_json({"at": datetime(2026, 1, 1, 12, 0, 0)})  # noqa: DTZ001 - the point of the test

    def test_non_string_mapping_key(self) -> None:
        with pytest.raises(UnhashableValueError, match="must be str"):
            canonical_json({1: "one"})


class TestEncodings:
    def test_pydantic_model(self) -> None:
        class Cfg(BaseModel):
            b: int = 2
            a: int = 1

        assert canonical_json(Cfg()) == b'{"a":1,"b":2}'

    def test_path(self) -> None:
        assert canonical_json({"p": Path("a/b.jsonl")}) == b'{"p":"a/b.jsonl"}'

    def test_path_equals_its_string(self) -> None:
        assert canonical_json(Path("a/b")) == canonical_json("a/b")
        assert canonical_json(PurePosixPath("a/b")) == canonical_json("a/b")

    def test_str_enum_equals_its_value(self) -> None:
        class Mode(StrEnum):
            KEEP = "keep_full"

        assert canonical_json(Mode.KEEP) == canonical_json("keep_full")

    def test_int_enum_equals_its_value(self) -> None:
        class Level(IntEnum):
            ONE = 1

        assert canonical_json(Level.ONE) == canonical_json(1)

    def test_plain_enum_equals_its_value(self) -> None:
        class Kind(Enum):
            A = "a"

        assert canonical_json(Kind.A) == canonical_json("a")

    def test_tuple_equals_list(self) -> None:
        # JSON has no tuple; a config read from YAML gives lists and the same config
        # built in Python may give tuples. They must not be two identities.
        assert canonical_json({"k": (1, 2)}) == canonical_json({"k": [1, 2]})

    def test_aware_datetime(self) -> None:
        assert canonical_json(datetime(2026, 1, 1, tzinfo=UTC)) == b'"2026-01-01T00:00:00+00:00"'

    def test_two_offsets_for_one_instant_differ(self) -> None:
        # Recorded honestly rather than normalised: these are PROVENANCE fields, and
        # collapsing them would claim the two records were written identically.
        utc = datetime(2026, 1, 1, 12, tzinfo=UTC)
        plus_one = datetime(2026, 1, 1, 13, tzinfo=timezone(timedelta(hours=1)))
        assert utc == plus_one
        assert canonical_json(utc) != canonical_json(plus_one)

    def test_date(self) -> None:
        assert canonical_json(date(2026, 1, 1)) == b'"2026-01-01"'


class TestValuesThatMustNotBeConflated:
    def test_int_and_float_stay_apart(self) -> None:
        # A count of 1 and a fraction of 1.0 are different settings; normalising them
        # would make a one-query smoke test share a key with a full run.
        assert canonical_json({"num_samples": 1}) != canonical_json({"num_samples": 1.0})

    def test_bool_and_int_stay_apart(self) -> None:
        assert canonical_json({"x": True}) != canonical_json({"x": 1})

    def test_none_and_missing_stay_apart_in_the_serialiser(self) -> None:
        # The serialiser distinguishes them; omitting ``None`` is a decision
        # ``identity_payload`` makes, deliberately, one level up.
        assert canonical_json({"x": None}) != canonical_json({})

    def test_empty_string_and_none_stay_apart(self) -> None:
        assert canonical_json({"x": ""}) != canonical_json({"x": None})

    def test_nested_and_flattened_stay_apart(self) -> None:
        assert canonical_json({"a": {"b": 1}}) != canonical_json({"a.b": 1})


class TestValuesThatMustBeConflated:
    def test_negative_zero_is_zero(self) -> None:
        # ``-0.0 == 0.0``, so they must not hash apart: a config that arrived at a
        # threshold by subtraction would otherwise get its own identity.
        assert canonical_json({"x": -0.0}) == canonical_json({"x": 0.0})

    def test_negative_zero_inside_a_list(self) -> None:
        assert canonical_json([-0.0]) == canonical_json([0.0])


class TestDigestFunctions:
    def test_hash_payload_is_a_full_sha256(self) -> None:
        digest = hash_payload({"a": 1})
        assert len(digest) == 64
        assert all(character in "0123456789abcdef" for character in digest)

    def test_hash_text_is_the_text_not_its_json(self) -> None:
        # So a prompt digest can be reproduced with ``sha256sum`` on the file.
        import hashlib

        assert hash_text("hello") == hashlib.sha256(b"hello").hexdigest()
        assert hash_text("hello") != hash_payload("hello")

    def test_short_refuses_a_short_input(self) -> None:
        # Otherwise a value already shortened to 16 could be shortened again and
        # silently become a weaker identity than its name claims.
        with pytest.raises(ValueError, match="full sha256"):
            short("abc123", 4)

    def test_short_refuses_a_non_hex_input(self) -> None:
        with pytest.raises(ValueError, match="full sha256"):
            short("z" * 64, 4)

    def test_short_refuses_an_out_of_range_length(self) -> None:
        digest = hash_payload({"a": 1})
        for length in (0, 65):
            with pytest.raises(ValueError, match="length must be"):
                short(digest, length)

    def test_short_is_a_prefix(self) -> None:
        digest = hash_payload({"a": 1})
        assert short(digest, 16) == digest[:16]


class TestThereIsOneRecipe:
    def test_the_preprocessing_key_uses_it(self) -> None:
        from rcp_ndcg.data.preprocess import Preprocessing, TextPolicy

        policy = Preprocessing(text=TextPolicy.resolve("truncate"))
        assert policy.key == short(hash_payload(policy.model_dump(mode="json")), 16)

    def test_the_mirrored_copy_is_gone(self) -> None:
        # ``shared.preprocessing`` held a verbatim copy of the manifest's function.
        from rcp_ndcg.data import preprocess as preprocessing

        assert not hasattr(preprocessing, "hash_payload")

    def test_the_deleted_recipes_are_gone(self) -> None:
        from rcp_ndcg.support import identity as hashing

        for name in ("compute_hash", "compute_hash_long", "save_cache_metadata", "verify_cache_metadata"):
            assert not hasattr(hashing, name), f"{name} is still importable"


class TestCanonicalReturnsPlainData:
    def test_canonical_is_json_native(self) -> None:
        import json

        class Cfg(BaseModel):
            mode: str = "keep_full"
            cap: int | None = None

        value = canonical({"cfg": Cfg(), "path": Path("a/b"), "items": (1, 2)})
        assert json.dumps(value)  # no encoder needed
        assert value == {"cfg": {"mode": "keep_full", "cap": None}, "path": "a/b", "items": [1, 2]}
