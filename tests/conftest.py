"""Shared fixtures and factories for the test suite."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from pathlib import Path

# Keep a test run out of the user's cache: caches (media, remote-object and calibration
# caches) go to a per-session temporary directory.
_SESSION_SCRATCH = Path(tempfile.mkdtemp(prefix="rcp-ndcg-tests-"))
os.environ.setdefault("RCP_NDCG_CACHE_DIR", str(_SESSION_SCRATCH / "cache"))

import pytest  # noqa: E402
from rcp_ndcg_core._records import ID, RankingExample  # noqa: E402


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--update-snapshots",
        action="store_true",
        default=False,
        help="Rewrite tests/contract/snapshots/ and schemas/ from the current tree (see tests/contract).",
    )


@pytest.fixture(autouse=True)
def _hub_is_offline_and_empty(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch):
    """No test reaches the HuggingFace Hub or reads the developer's hub cache.

    Revision resolution (``rcp_ndcg.data.revisions``) runs whenever an identity
    names a Hub model or dataset. Offline with an empty cache it records the
    revision as unverified; tests that need a resolved commit install a fake Hub
    or write a cache ref themselves.
    """
    from rcp_ndcg.data.revisions import resolve_revision

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path_factory.getbasetemp() / "empty_hub_cache"))
    resolve_revision.cache_clear()
    yield
    resolve_revision.cache_clear()


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_docs() -> list[str]:
    return [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models require large datasets.",
        "Python is a versatile programming language.",
        "Climate change affects global weather patterns.",
        "Neural networks are inspired by biological neurons.",
    ]


@pytest.fixture()
def sample_doc_ids() -> list[ID]:
    return ["d1", "d2", "d3", "d4", "d5"]


@pytest.fixture()
def sample_scores() -> list[float]:
    return [5.0, 4.0, 3.0, 2.0, 1.0]


@pytest.fixture()
def sample_qrels() -> dict[ID, int]:
    return {"d1": 2, "d2": 1, "d3": 0, "d4": 1, "d5": 0}


# ---------------------------------------------------------------------------
# Composite fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def ranking_example(
    sample_docs: list[str],
    sample_doc_ids: list[ID],
    sample_scores: list[float],
    sample_qrels: dict[ID, int],
) -> RankingExample:
    return RankingExample(
        query="test query",
        query_id="q1",
        docs=sample_docs,
        doc_ids=sample_doc_ids,
        scores=sample_scores,
        qrels=sample_qrels,
    )


@pytest.fixture()
def ranking_example_no_docs(
    sample_doc_ids: list[ID],
    sample_scores: list[float],
    sample_qrels: dict[ID, int],
) -> RankingExample:
    return RankingExample(
        query="test query",
        query_id="q1",
        docs=None,
        doc_ids=sample_doc_ids,
        scores=sample_scores,
        qrels=sample_qrels,
    )


@pytest.fixture()
def tmp_jsonl(tmp_path: Path) -> Callable[[list[RankingExample]], Path]:
    def _write(examples: list[RankingExample]) -> Path:
        path = tmp_path / "examples.jsonl"
        with path.open("w") as f:
            for ex in examples:
                f.write(ex.model_dump_json(exclude_none=True) + "\n")
        return path

    return _write


# ---------------------------------------------------------------------------
# Synthetic video: built in-test, never downloaded
# ---------------------------------------------------------------------------


def _chunk(fourcc: bytes, body: bytes) -> bytes:
    import struct

    return fourcc + struct.pack("<I", len(body)) + body + (b"\0" if len(body) % 2 else b"")


def _riff_list(kind: bytes, body: bytes) -> bytes:
    return _chunk(b"LIST", kind + body)


def write_mjpeg_avi(path: Path, *, frames: int = 12, fps: int = 4, size: tuple[int, int] = (64, 48)) -> Path:
    """A real, decodable Motion-JPEG AVI: ``frames`` frames of ``size`` (w, h) at ``fps``.

    Each frame is a flat colour that changes with its index, so a decoder that
    samples frame *i* can be checked against it. Pillow is the only dependency.
    """
    import io
    import struct

    from PIL import Image

    width, height = size
    jpegs = []
    for index in range(frames):
        buffer = io.BytesIO()
        Image.new("RGB", size, color=(index * 20 % 256, 64, 255 - index * 20 % 256)).save(buffer, format="JPEG")
        jpegs.append(buffer.getvalue())

    avih = struct.pack(
        "<14I", 1_000_000 // fps, 0, 0, 0x10, frames, 0, 1, max(map(len, jpegs)), width, height, 0, 0, 0, 0
    )
    strh = b"vidsMJPG" + struct.pack(
        "<IHHIIIIIIiI4h", 0, 0, 0, 0, 1, fps, 0, frames, max(map(len, jpegs)), -1, 0, 0, 0, width, height
    )
    strf = struct.pack("<IiiHH4sIiiII", 40, width, height, 1, 24, b"MJPG", width * height * 3, 0, 0, 0, 0)
    hdrl = _riff_list(
        b"hdrl", _chunk(b"avih", avih) + _riff_list(b"strl", _chunk(b"strh", strh) + _chunk(b"strf", strf))
    )

    movi_body = b""
    index_entries = b""
    for jpeg in jpegs:
        index_entries += b"00dc" + struct.pack("<III", 0x10, 4 + len(movi_body), len(jpeg))
        movi_body += _chunk(b"00dc", jpeg)
    body = b"AVI " + hdrl + _riff_list(b"movi", movi_body) + _chunk(b"idx1", index_entries)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)
    return path


def write_mp4_header(path: Path, *, frames: int, timescale: int, duration: int, size: tuple[int, int]) -> Path:
    """An ISO BMFF file carrying only the boxes a header probe reads (no media data).

    Enough to pin the MP4/MOV header parser without an encoder; the AVI fixture is
    the decodable one.
    """
    import struct

    def box(kind: bytes, body: bytes) -> bytes:
        return struct.pack(">I", 8 + len(body)) + kind + body

    width, height = size
    tkhd = box(b"tkhd", b"\0\0\0\x03" + b"\0" * 72 + struct.pack(">II", width << 16, height << 16))
    mdhd = box(b"mdhd", b"\0\0\0\0" + struct.pack(">IIII", 0, 0, timescale, duration) + b"\0" * 4)
    hdlr = box(b"hdlr", b"\0" * 8 + b"vide" + b"\0" * 12 + b"video\0")
    stsz = box(b"stsz", b"\0" * 4 + struct.pack(">II", 0, frames) + b"\0\0\0\x01" * frames)
    minf = box(b"minf", box(b"stbl", stsz))
    trak = box(b"trak", tkhd + box(b"mdia", mdhd + hdlr + minf))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(box(b"ftyp", b"isom\0\0\0\0isom") + box(b"moov", trak))
    return path


@pytest.fixture()
def video_clip(tmp_path: Path) -> Callable[..., Path]:
    """Factory: ``video_clip("a.avi", frames=12, fps=4, size=(64, 48))`` -> a decodable MJPEG AVI."""

    def _make(name: str = "clip.avi", **kwargs) -> Path:
        return write_mjpeg_avi(tmp_path / name, **kwargs)

    return _make


@pytest.fixture(scope="session")
def word_tokenizer_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A ``tokenizer.json`` of the in-memory word-level test tokenizer (:mod:`tests._tokenizers`)."""
    from tests._tokenizers import save, word_tokenizer

    return save(word_tokenizer(), tmp_path_factory.mktemp("tokenizer"))
