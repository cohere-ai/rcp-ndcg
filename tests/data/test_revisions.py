"""Resolved HuggingFace commits in the dataset identity and in the run steps' identities.

The Hub is faked at the boundary production calls: the ``huggingface_hub`` module
(``HfApi().dataset_info``) and the hub cache's
``refs/<revision>`` files. Nothing here reaches the network: the suite's conftest
keeps ``HF_HUB_OFFLINE=1`` and an empty cache unless a test sets up otherwise.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from rcp_ndcg.data.revisions import resolve_revision
from rcp_ndcg.errors import RcpNdcgWarning
from rcp_ndcg.runs import Pipeline, RunConfig
from rcp_ndcg.support.identity import hash_payload

SHA_A = "a" * 40
SHA_B = "b" * 40


class FakeHub:
    """A stand-in ``huggingface_hub`` whose refs can move, counting every lookup."""

    def __init__(self) -> None:
        self.refs: dict[tuple[str, str, str], str] = {}
        self.calls: list[tuple[str, str, str]] = []

    def module(self) -> types.ModuleType:
        hub = self

        class HfApi:
            def dataset_info(self, repo_id: str, revision: str | None = None):
                return hub._info("dataset", repo_id, revision)

        module = types.ModuleType("huggingface_hub")
        module.HfApi = HfApi  # type: ignore[attr-defined]
        return module

    def _info(self, repo_type: str, repo_id: str, revision: str | None):
        key = (repo_type, repo_id, revision or "main")
        self.calls.append(key)
        if key not in self.refs:
            raise OSError(f"no such revision {key}")
        return types.SimpleNamespace(sha=self.refs[key])

    def move(self, repo_type: str, repo_id: str, sha: str, revision: str = "main") -> None:
        """Point a ref at a new commit, as a push to the Hub would, and forget this process's lookups."""
        self.refs[(repo_type, repo_id, revision)] = sha
        resolve_revision.cache_clear()


@pytest.fixture
def hub(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeHub:
    fake = FakeHub()
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake.module())
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    # An online resolution records refs/<ref> in the hub cache; keep that out of the shared empty cache.
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    return fake


# ---------------------------------------------------------------------------
# The resolver
# ---------------------------------------------------------------------------


class TestResolveRevision:
    def test_a_full_commit_is_its_own_answer_without_a_lookup(self, hub: FakeHub) -> None:
        resolved = resolve_revision("org/data", SHA_A)
        assert (resolved.commit, resolved.verified) == (SHA_A, True)
        assert hub.calls == []

    def test_a_branch_resolves_on_the_hub_once_per_process(self, hub: FakeHub) -> None:
        hub.move("dataset", "org/data", SHA_A)
        for _ in range(5):
            assert resolve_revision("org/data", None).commit == SHA_A
        assert hub.calls == [("dataset", "org/data", "main")]

    def test_offline_reads_the_cache_ref(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
        ref = tmp_path / "datasets--BeIR--fiqa" / "refs" / "main"
        ref.parent.mkdir(parents=True)
        ref.write_text(SHA_B)
        assert resolve_revision("BeIR/fiqa", None).commit == SHA_B

    def test_an_unreachable_hub_falls_back_to_the_cache(
        self, hub: FakeHub, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
        ref = tmp_path / "datasets--org--data" / "refs" / "main"
        ref.parent.mkdir(parents=True)
        ref.write_text(SHA_A)
        resolved = resolve_revision("org/data", None)  # the fake Hub has no such ref: it raises
        assert resolved.commit == SHA_A
        assert hub.calls == [("dataset", "org/data", "main")]

    def test_unresolvable_is_unverified_and_warns_with_the_code(self) -> None:
        """Offline with nothing to resolve, the run warns UNPINNED_REVISION naming the revision fix."""
        with pytest.warns(RcpNdcgWarning) as seen:
            resolved = resolve_revision("org/data", "main")
        assert (resolved.commit, resolved.verified) == (None, False)
        (record,) = seen
        warning = record.message
        assert isinstance(warning, RcpNdcgWarning) and warning.code == "UNPINNED_REVISION"
        assert "--revision" in str(warning) and "full sha" in str(warning)

    def test_a_corrupt_cache_ref_is_sanitized_not_fatal(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """A ref file that is not valid UTF-8 is removed before resolution, never crashes it."""
        import os

        monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
        ref = Path(os.environ["HF_HUB_CACHE"]) / "datasets--BeIR--fiqa" / "refs" / "main"
        ref.parent.mkdir(parents=True)
        ref.write_bytes(b"\xff\xfe not utf-8")
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")

        with pytest.warns(RcpNdcgWarning):  # offline with the corrupt ref gone, nothing is resolved: it warns
            resolved = resolve_revision("BeIR/fiqa", None)

        assert (resolved.commit, resolved.verified) == (None, False)
        assert not ref.exists(), "the corrupt ref is gone; the next online resolution rewrites it"

    def test_a_corrupt_cache_ref_offline_warns_unpinned(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """The sanitized-offline resolution still warns UNPINNED_REVISION: nothing is resolved."""
        import os

        monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
        ref = Path(os.environ["HF_HUB_CACHE"]) / "datasets--BeIR--fiqa" / "refs" / "main"
        ref.parent.mkdir(parents=True)
        ref.write_bytes(b"\xff\xfe not utf-8")
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")

        with pytest.warns(RcpNdcgWarning) as seen:
            resolve_revision("BeIR/fiqa", None)

        assert [record.message.code for record in seen] == ["UNPINNED_REVISION"]

    def test_a_corrupt_cache_ref_is_rewritten_online_and_absent_offline(
        self, hub: FakeHub, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A ref file that is not valid text never breaks a resolution: the online resolve writes over it."""
        import os

        ref = Path(os.environ["HF_HUB_CACHE"]) / "datasets--BeIR--fiqa" / "refs" / "main"
        ref.parent.mkdir(parents=True)
        ref.write_bytes(b"\xff\xfe not utf-8")
        hub.move("dataset", "BeIR/fiqa", SHA_A)

        assert resolve_revision("BeIR/fiqa", None).commit == SHA_A
        assert ref.read_text() == SHA_A

        resolve_revision.cache_clear()  # the offline run is another process
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        assert resolve_revision("BeIR/fiqa", None).commit == SHA_A


# ---------------------------------------------------------------------------
# The corpus: dataset identity
# ---------------------------------------------------------------------------


class TestDatasetRevision:
    def test_a_moved_upstream_makes_a_finished_run_step_stale(self, hub: FakeHub, tmp_path: Path) -> None:
        pipeline = _pipeline(tmp_path)
        hub.move("dataset", "BeIR/fiqa", SHA_A)
        before = pipeline._identity("tournament")
        assert before["dataset"]["resolved"] == {"repo": "BeIR/fiqa", "commit": SHA_A, "verified": True}
        hub.move("dataset", "BeIR/fiqa", SHA_B)
        assert hash_payload(pipeline._identity("tournament")) != hash_payload(before)

    def test_a_moved_upstream_is_refused_by_the_judgement_store(self, hub: FakeHub, tmp_path: Path) -> None:
        from rcp_ndcg_core._records import Document, Query

        from rcp_ndcg.data import Dataset
        from rcp_ndcg.errors import IdentityError
        from rcp_ndcg.llm import RubricSchedule, judge
        from rcp_ndcg.testing import FakeJudge

        loaded = Dataset(
            name="fiqa", uri="hf://BeIR/fiqa/fiqa", qrels={"q0": {"d1": 1.0}}, candidates={"q0": ["d1", "d2"]}
        )
        loaded._load_queries = lambda: [Query(query_id="q0", query="q")]
        loaded._load_corpus = lambda: [Document(doc_id="d1", text="one d1"), Document(doc_id="d2", text="two d2")]
        schedule = RubricSchedule(window=2, placements_per_doc=1.0, random_share=1.0)
        hub.move("dataset", "BeIR/fiqa", SHA_A)
        judge(loaded, None, FakeJudge(), stage="rubric", out=tmp_path, schedule=schedule)
        hub.move("dataset", "BeIR/fiqa", SHA_B)
        with pytest.raises(IdentityError, match="dataset.revision.commit"):
            judge(loaded, None, FakeJudge(), stage="rubric", out=tmp_path, schedule=schedule)


def _pipeline(tmp_path: Path) -> Pipeline:
    rankings = tmp_path / "rankings.jsonl"
    rankings.write_text(json.dumps({"query_id": "q0", "query": "q", "doc_ids": ["d1"]}))
    return Pipeline(
        RunConfig.model_validate(
            {
                "dataset": "hf://BeIR/fiqa/fiqa",
                "candidates": {"from": "rankings", "rankings": str(rankings)},
                "judge": "fake",
                "steps": ["retrieve", "tournament", "rubric"],
            }
        ),
        runs_dir=str(tmp_path / "runs"),
    )


def test_resume_checks_cost_one_lookup_per_repository(hub: FakeHub, tmp_path: Path) -> None:
    """A pipeline computes each step's identity on every resume check; the resolution
    behind it is one metadata call per repository per process."""
    hub.move("dataset", "BeIR/fiqa", SHA_B)
    pipeline = _pipeline(tmp_path)
    for _ in range(3):
        for step in ("retrieve", "tournament", "rubric"):
            pipeline._identity(step)
    assert hub.calls == [("dataset", "BeIR/fiqa", "main")]


def test_is_commit_is_public_and_the_private_pattern_stays_home() -> None:
    """The commit-shape check is public API (`is_commit`); no other module imports the private `_COMMIT`."""
    import inspect

    import rcp_ndcg.data.dataset as dataset_module
    from rcp_ndcg.data.revisions import is_commit

    sha = "b" * 40
    assert is_commit(sha)
    assert not is_commit("main"), "a branch is not a commit"
    assert not is_commit(sha.upper()), "a sha is lowercase hex"
    assert not is_commit(sha[:39]) and not is_commit(sha + "0"), "exactly 40 characters"
    assert not is_commit(None)
    assert "_COMMIT" not in inspect.getsource(dataset_module), "dataset.py reads revisions through the public name"
