"""Serving the fixture repositories as a Hub: the two boundaries the reader talks to are faked.

``_hub_file`` reads from the fixture directory (absent = the Hub's 404), ``_hub_listing`` lists it, and the
tests pin the revision to a full commit, which :func:`~rcp_ndcg.data.revisions.resolve_revision` accepts
without any lookup -- so the reader's layout logic runs offline, against the real cards of the canonical
repositories (their rows are sampled from the repositories' own files; media bytes and a corpus's placeholder OCR text
are synthesized where the sampled row carries none -- the layout is what is under test, not the pixels).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from rcp_ndcg.data.io import hub as hub_module
from rcp_ndcg.errors import MissingInputError

FIXTURES = Path(__file__).parent / "fixtures"
SHA = "c" * 40


class _Absent(Exception):
    """The fake of the Hub's 404, the shape ``_hub_file`` translates to ``None``."""


def _fake_hub_file(root: Path):
    def download(repo: str, path: str, revision: str | None = None, **_: Any) -> Any:
        target = root / path
        if not target.is_file():
            return None  # the Hub's 404, the shape ``_hub_file`` translates to ``None``
        return target

    return download


def _fake_listing(root: Path):
    files = sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file())

    def listing(repo: str, revision: str | None) -> list[str]:
        return files

    return listing


@contextmanager
def hub_fixture(name: str, tmp_path: Path | None = None) -> Iterator[Path]:
    """Serve ``tests/data/fixtures/<name>`` as a Hub repository (no network, real cards, sampled rows).

    With *tmp_path*, the fixture is copied there first, so a test that rewrites a table (a duplicates case, an
    instruction the reader must refuse) never writes into the checkout.
    """
    import shutil
    from unittest import mock

    source = FIXTURES / name
    if not (source / "README.md").is_file():
        raise MissingInputError(f"the fixture repository {name!r} has no card")
    if tmp_path is None:
        root = source
    else:
        root = tmp_path / name
        if root.exists():
            shutil.rmtree(root)
        shutil.copytree(source, root)
    with (
        mock.patch.object(hub_module, "_hub_file", _fake_hub_file(root)),
        mock.patch.object(hub_module, "_hub_listing", _fake_listing(root)),
    ):
        yield root


def fixture_path(name: str, *parts: str) -> Path:
    return FIXTURES.joinpath(name, *parts)
