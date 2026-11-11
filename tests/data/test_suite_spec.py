"""The suite file format (RFC-0002 §4.3): the four shipped suites, a suite by path, and a revision pin.

The four suites ship as data (``rcp/data/suites/<name>.yaml``) and are read through the one suite reader.
The gate is byte identity: :data:`FROZEN_DIGESTS` were taken from the four suites as the literal ``SUITES``
dict loaded them before the data files existed, over the fake Hub below.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from rcp.data import SUITES, VIDORE_NATIVE_LANGUAGE, Dataset, load_dataset
from rcp.data.io.suite import SuiteReader, SuiteSpec, shipped_suite_path
from rcp.errors import ConfigError, DataError

SHA = "b" * 40

#: The four suites as the literal dict held them: name -> (repo, subsets). Frozen on purpose.
FROZEN_SUITES: dict[str, tuple[str, tuple[str, ...]]] = {
    "nanobeir": (
        "fabianschmidt-cohere/rcp-ndcg-nanobeir",
        (
            "NanoArguAnaRetrieval",
            "NanoClimateFeverRetrieval",
            "NanoDBPediaRetrieval",
            "NanoFEVERRetrieval",
            "NanoFiQA2018Retrieval",
            "NanoHotpotQARetrieval",
            "NanoMSMARCORetrieval",
            "NanoNFCorpusRetrieval",
            "NanoNQRetrieval",
            "NanoQuoraRetrieval",
            "NanoSCIDOCSRetrieval",
            "NanoSciFactRetrieval",
            "NanoTouche2020Retrieval",
        ),
    ),
    "bright": (
        "fabianschmidt-cohere/rcp-ndcg-bright",
        (
            "aops",
            "biology",
            "earth_science",
            "economics",
            "leetcode",
            "pony",
            "psychology",
            "robotics",
            "stackoverflow",
            "sustainable_living",
            "theoremqa_questions",
            "theoremqa_theorems",
        ),
    ),
    "vidore": (
        "fabianschmidt-cohere/rcp-ndcg-vidore-v3",
        (
            "computer_science__english",
            "energy__french",
            "finance_en__english",
            "finance_fr__french",
            "hr__english",
            "industrial__english",
            "pharmaceuticals__english",
            "physics__french",
        ),
    ),
    "trecdl": (
        "fabianschmidt-cohere/rcp-ndcg-trecdl",
        (
            "trec_dl_2019",
            "trec_dl_2020",
        ),
    ),
}

#: SHA-256 of each suite Dataset's ``model_dump_json()`` over the fake Hub below, taken on the base commit.
FROZEN_DIGESTS: dict[str, str] = {
    "nanobeir": "3cca86f1e1232efe1d06067ddd0b1dc603ebb4e72d1a9fcf3643cbd7632876ac",
    "bright": "8c4cd2232e18347c656a5d80d417b6bf8022ff98cdc7a8f09b2c447ecba07409",
    "vidore": "4bab64fd7dbe8a2f4ef0c1808e330addf8107e68b254bdf0e90d0aaed7278ee7",
    "trecdl": "d9b8f64293fcc6302493d5a10af753de472cb8d5ca44ad837aeb1d2ddcde1334",
}


@pytest.fixture
def hub(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """A stand-in for the Hub: every suite subset has a one-row qrels table, served at one commit."""
    from rcp.data.io import hub as hub_module
    from rcp.data.revisions import resolve_revision

    root = tmp_path / "hub"
    for suite in SUITES.values():
        for index, subset in enumerate(suite.subsets):
            target = root / suite.repo.replace("/", "--") / subset / "qrels.parquet"
            target.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {
                    "query-id": [f"q-{index}"],
                    "corpus-id": [f"d-{index}"],
                    "score": [1.0],
                    "gain": [0.5],
                    "theta": [0.0],
                }
            ).to_parquet(target)

    def read_file(repo: str, path: str, revision: str | None = None, **_: object):
        target = root / repo.replace("/", "--") / path
        return target if target.is_file() else None

    monkeypatch.setattr(hub_module, "_hub_file", read_file)
    monkeypatch.setattr(hub_module, "_hub_listing", lambda repo, revision: [])

    from huggingface_hub import constants as hub_constants

    cache = tmp_path / "hub-cache"
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    monkeypatch.setattr(hub_constants, "HF_HUB_CACHE", str(cache))
    for suite in SUITES.values():
        ref = cache / f"datasets--{suite.repo.replace('/', '--')}" / "refs" / "main"
        ref.parent.mkdir(parents=True, exist_ok=True)
        ref.write_text(SHA)
    resolve_revision.cache_clear()
    return root


def _write_suite(directory, **overrides: object) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    spec: dict[str, object] = {
        "kind": "suite",
        "name": "my-suite",
        "repo": FROZEN_SUITES["nanobeir"][0],
        "subsets": [FROZEN_SUITES["nanobeir"][1][0]],
        "protocol": "plain",
        "revision": None,
        "gains": False,
    }
    spec.update(overrides)
    (directory / "suite.yaml").write_text(yaml.safe_dump(spec), encoding="utf-8")


def test_the_four_shipped_suites_are_the_frozen_literal() -> None:
    assert {name: (suite.repo, suite.subsets) for name, suite in SUITES.items()} == FROZEN_SUITES
    assert {f"{domain}__{language}" for domain, language in VIDORE_NATIVE_LANGUAGE.items()} == set(
        SUITES["vidore"].subsets
    ), "the bare-domain aliases name the vidore suite's own subsets"


@pytest.mark.parametrize("name", sorted(FROZEN_SUITES))
def test_a_shipped_suite_reads_through_the_reader(name: str) -> None:
    spec = SuiteReader(str(shipped_suite_path(name))).spec

    assert spec.kind == "suite"
    assert (spec.name, spec.repo, spec.subsets) == (name, *FROZEN_SUITES[name])
    assert spec.protocol == name and spec.revision is None and spec.gains is True
    assert (SUITES[name].repo, SUITES[name].subsets) == (spec.repo, spec.subsets)


@pytest.mark.parametrize("name", sorted(FROZEN_DIGESTS))
def test_a_shipped_suite_loads_byte_identically(hub: object, name: str) -> None:
    payload = load_dataset(f"suite:{name}").model_dump_json()

    assert hashlib.sha256(payload.encode()).hexdigest() == FROZEN_DIGESTS[name], payload


def test_a_suite_repository_loads_by_its_hf_uri(hub: object) -> None:
    dataset = load_dataset(f"hf://{FROZEN_SUITES['trecdl'][0]}")

    assert dataset.name == "trecdl"
    assert [part.name for part in dataset.parts] == list(FROZEN_SUITES["trecdl"][1])


def test_a_shipped_suite_resolves_from_its_data_file_without_the_view(
    hub: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rcp.data import dataset as dataset_module

    monkeypatch.setattr(dataset_module, "SUITES", {})

    dataset = load_dataset("suite:trecdl")

    assert dataset.name == "trecdl" and len(dataset.parts) == 2


def test_a_revision_option_pins_every_part(hub: object) -> None:
    commit = "c" * 40

    dataset = load_dataset("suite:trecdl", revision=commit)

    assert dataset.revision == commit
    assert all(part.revision == commit for part in dataset.parts)


def test_a_suite_directory_outside_the_package_loads_by_path(tmp_path, hub: object) -> None:
    directory = tmp_path / "my-suite"
    _write_suite(directory)

    dataset = load_dataset(f"suite:{directory}")

    assert dataset.name == "my-suite"
    assert dataset.protocol == "plain", "the protocol comes from the file, not the directory name"
    assert [part.name for part in dataset.parts] == [FROZEN_SUITES["nanobeir"][1][0]]
    assert dataset.parts[0].protocol == "plain"


def test_a_suite_file_loads_by_path_too(tmp_path, hub: object) -> None:
    directory = tmp_path / "my-suite"
    _write_suite(directory)

    dataset = load_dataset(f"suite:{directory / 'suite.yaml'}")

    assert dataset.name == "my-suite"


def test_a_suite_whose_revision_does_not_match_its_data_is_refused(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from rcp.data import dataset as dataset_module

    pin, data = "a" * 40, "c" * 40
    directory = tmp_path / "pinned"
    _write_suite(directory, revision=pin)
    monkeypatch.setattr(
        dataset_module,
        "_from_reader",
        lambda reader, *, uri, protocol=None: Dataset(name="part", revision=data, subset="part"),
    )

    with pytest.raises(DataError) as caught:
        load_dataset(f"suite:{directory}")

    assert pin in str(caught.value) and data in str(caught.value), "both revisions are named"
    assert caught.value.details["revision"] == pin
    assert caught.value.details["data_revision"] == data


def test_a_suite_whose_revision_matches_its_data_loads(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from rcp.data import dataset as dataset_module

    pin = "a" * 40
    directory = tmp_path / "pinned"
    _write_suite(directory, revision=pin)
    monkeypatch.setattr(
        dataset_module,
        "_from_reader",
        lambda reader, *, uri, protocol=None: Dataset(name="part", revision=pin, subset="part"),
    )

    assert load_dataset(f"suite:{directory}").revision == pin


def test_an_unknown_suite_name_is_still_refused() -> None:
    with pytest.raises(ConfigError, match="unknown suite"):
        load_dataset("suite:not-a-suite")


class TestTheSuiteIdentity:
    """A ``suite:`` URI is named by the definition it reads: the commit its repository resolves to, and -- when
    the file pins nothing -- its own bytes, so an edited downstream suite never reuses stale work."""

    def test_editing_a_downstream_suite_moves_the_identity(self, tmp_path, hub: object) -> None:
        from rcp.runs.config import DatasetSource

        directory = tmp_path / "my-suite"
        _write_suite(directory)
        source = DatasetSource(uri=f"suite:{directory}")
        before = source.identity()

        _write_suite(directory, subsets=[FROZEN_SUITES["nanobeir"][1][0], FROZEN_SUITES["nanobeir"][1][1]])
        after = source.identity()

        assert after != before, "an edited suite.yaml is another dataset"
        assert after["resolved"]["repo"] == FROZEN_SUITES["nanobeir"][0], "the file's repository is resolved"
        assert after["content"]["dataset"] != before["content"]["dataset"]

    def test_a_suite_pinned_to_a_commit_is_named_by_its_repository_and_revision(self, tmp_path) -> None:
        from rcp.runs.config import DatasetSource

        pin = "a" * 40
        directory = tmp_path / "pinned"
        _write_suite(directory, revision=pin)
        source = DatasetSource(uri=f"suite:{directory}")

        identity = source.identity()

        assert identity["resolved"] == {
            "repo": FROZEN_SUITES["nanobeir"][0],
            "commit": pin,
            "verified": True,
        }
        assert "content" not in identity, "a full commit pins the data: the file's own bytes are not in the identity"

        # The same definition, written another way: a whitespace edit is not a new dataset.
        (directory / "suite.yaml").write_text(
            f"# the same suite, written another way\n"
            f"name: my-suite\n"
            f"repo: {FROZEN_SUITES['nanobeir'][0]}\n"
            f"protocol: plain\n"
            f"revision: {pin}\n"
            f"gains: false\n"
            f"subsets:\n  - {FROZEN_SUITES['nanobeir'][1][0]}\n",
            encoding="utf-8",
        )
        assert source.identity() == identity

    def test_a_shipped_suite_is_named_by_its_repository_and_commit(self, hub: object) -> None:
        from rcp.runs.config import DatasetSource

        identity = DatasetSource(uri="suite:trecdl").identity()

        assert identity == {
            "uri": "suite:trecdl",
            "resolved": {"repo": FROZEN_SUITES["trecdl"][0], "commit": SHA, "verified": True},
        }, "the four shipped suites are package data: their identity is the repository at its commit"


def test_a_suite_file_with_a_wrong_kind_is_refused(tmp_path) -> None:
    directory = tmp_path / "not-a-suite"
    _write_suite(directory, kind="dataset")

    with pytest.raises(ConfigError, match="kind"):
        _ = SuiteReader(str(directory)).spec


def test_a_suite_file_with_an_unknown_key_is_refused(tmp_path) -> None:
    directory = tmp_path / "typo"
    _write_suite(directory, subset=["one"])

    with pytest.raises(ConfigError, match="subset"):
        _ = SuiteReader(str(directory)).spec


def test_a_suite_without_subsets_is_refused(tmp_path) -> None:
    directory = tmp_path / "empty"
    _write_suite(directory, subsets=[])

    with pytest.raises(ConfigError, match="subsets"):
        _ = SuiteReader(str(directory)).spec


def test_the_suite_spec_schema_is_the_committed_schema() -> None:
    committed = json.loads((Path(__file__).parents[2] / "schemas" / "suite.v1.json").read_text(encoding="utf-8"))
    assert committed["properties"]["kind"]["const"] == "suite"
    assert set(committed["properties"]) == {"kind", "name", "repo", "subsets", "protocol", "revision", "gains"}
    assert SuiteSpec.model_json_schema()["properties"].keys() == committed["properties"].keys()
