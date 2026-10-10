"""The media stage: what the served client sends for image inputs against what the reference consumes, and the
engine's own media count -- so a vision-language recipe's media equivalence can fail (GPU-VALIDATION.md item 5:
control (f), an unpinned pixel budget, is caught here).

``fixture-vl-embed`` is a vision embedder on the ``messages`` route whose pixel budget (784..200704 px) is pinned
below the Qwen2-VL family's stock floor; its reference sizes images with its own card rule, and the stub engine
resizes them as the engine's processor does under the served pin (else the emulated checkpoint's default).
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from rcp_ndcg_test.equivalence import run as run_equivalence
from rcp_ndcg_test.equivalence.media import stage_media
from rcp_ndcg_vllm import load_recipe
from rcp_ndcg_vllm.recipe import serve_argv

from tests.conftest import RECIPES, TOKENIZER, sample_pairs, start_stub, write_pairs

REFERENCE_PYTHON = sys.executable
SIZES = ((16, 16), (300, 200), (1200, 900), (60, 2000))
"""Image sizes (width, height): under the pinned floor, inside the budget, over it, a tall strip."""


def png_entry(width: int, height: int) -> dict[str, Any]:
    """One pairs media entry: an inline PNG of ``width`` x ``height`` (a two-colour pattern)."""
    from PIL import Image

    image = Image.new("RGB", (width, height), (200, 30, 30))
    image.paste((20, 20, 220), (0, 0, max(1, width // 2), max(1, height // 2)))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    payload = buffer.getvalue()
    return {
        "kind": "image",
        "uri": "data:image/png;base64," + base64.b64encode(payload).decode("ascii"),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "mime": "image/png",
        "width": width,
        "height": height,
        "num_bytes": len(payload),
    }


def media_pairs(path: Path) -> Path:
    """Three text rows (one long) and media rows: a captioned page, an image-only page per size."""
    rows: list[dict[str, Any]] = list(sample_pairs()[:2])
    for row in rows:
        row.pop("instruction", None)
    # A long text row inside the budget, as the length ladder's rows are: an engine-side cut bites on it.
    rows.append({"query": "a long one", "documents": [" ".join(["cities and rivers in europe"] * 12)]})
    rows.append(
        {
            "query": "the red page",
            "documents": ["a caption under the page", ""],
            "media": {"query": [], "documents": [[png_entry(300, 200)], [png_entry(16, 16)]]},
        }
    )
    for width, height in SIZES:
        rows.append({"query": "a page", "documents": [""], "media": {"documents": [[png_entry(width, height)]]}})
    return write_pairs(path, rows)


def stub_for(recipe: Any, *flags: str) -> Any:
    """The stub engine started with the recipe's own serve argv (the served chat template and pixel pin)."""
    argv = serve_argv(recipe, port=0, served_model_name=recipe.id)
    argv = ["127.0.0.1" if value == "0.0.0.0" else value for value in argv[argv.index(recipe.model) + 1 :]]
    return start_stub("--tokenizer", str(TOKENIZER), *argv, *flags)


@pytest.fixture(scope="module")
def recipe() -> Any:
    return load_recipe(RECIPES / "fixture-vl-embed")


def test_the_media_stage_compares_what_the_client_sends_with_what_the_reference_consumes(
    recipe: Any, tmp_path: Path
) -> None:
    """Offline (the product's fake answers): every media item's count, placement, prepared geometry and tokens
    equal the reference's; the engine's count is not run without an engine (never passed)."""
    document = stage_media(recipe, media_pairs(tmp_path / "pairs.jsonl"), REFERENCE_PYTHON)
    assert document is not None
    assert document["passed"] is True, document["failures"][:3]
    assert document["items"] == 6 and document["rows"] == 1 + len(SIZES)
    assert document["engine_check"]["status"] == "not_run" and document["engine_check"]["passed"] is None


def test_a_text_first_placement_fails_the_stage(recipe: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The mutation: the client puts the fitted text before the media (the placement it had before the fit kept
    it in place) -- the captioned page's placement differs from the card's and the stage fails on it."""
    from rcp_ndcg_core.content import Content, TextPart

    from rcp_ndcg.inference.clients._base import RoleClient

    def text_first(content: Content, text: str) -> Content:
        parts: list[Any] = [TextPart(text=text)] if text else []
        return Content.from_parts(parts + [part for part in content.parts if not isinstance(part, TextPart)])

    monkeypatch.setattr(RoleClient, "_with_text", staticmethod(text_first))
    document = stage_media(recipe, media_pairs(tmp_path / "pairs.jsonl"), REFERENCE_PYTHON)
    assert document is not None and document["passed"] is False
    assert {failure["check"] for failure in document["failures"]} == {"placement"}


def test_a_client_policy_other_than_the_cards_fails_on_geometry_and_tokens(recipe: Any, tmp_path: Path) -> None:
    """The client prepares images under another budget than the card resizes them to: the geometry and the
    token count of every resized image differ, and the stage names both."""
    policy = {**recipe.client["image_policy"], "max_px": 100352}
    other = recipe.model_copy(update={"client": {**recipe.client, "image_policy": policy}})
    document = stage_media(other, media_pairs(tmp_path / "pairs.jsonl"), REFERENCE_PYTHON)
    assert document is not None and document["passed"] is False
    assert {"width", "height", "tokens"} <= {failure["check"] for failure in document["failures"]}


def test_the_engine_counts_what_the_client_counts_and_an_unpinned_engine_does_not(recipe: Any, tmp_path: Path) -> None:
    """With an engine: each media request again without its media gives the engine's media count. The stub
    served with the recipe's pin counts exactly the client's tokens; served without it (control (f)), the engine
    re-resizes the image under the pinned floor to its stock budget and the stage fails on the engine check."""
    pairs = media_pairs(tmp_path / "pairs.jsonl")
    engine = stub_for(recipe)
    try:
        document = stage_media(recipe, pairs, REFERENCE_PYTHON, base_url=engine.base_url)
    finally:
        engine.stop()
    assert document is not None and document["passed"] is True, (document["failures"], document["engine_check"])
    assert document["engine_check"]["checked"] == 6 and document["engine_check"]["passed"] is True
    unpinned = recipe.model_copy(update={"serve": recipe.serve.model_copy(update={"mm_processor_kwargs": {}})})
    engine = stub_for(unpinned)
    try:
        document = stage_media(recipe, pairs, REFERENCE_PYTHON, base_url=engine.base_url)
    finally:
        engine.stop()
    assert document is not None and document["passed"] is False
    assert document["failures"] == [] and document["engine_check"]["passed"] is False
    failures = document["engine_check"]["failures"]
    # the two 16 x 16 images: prepared at the pinned 784 px floor, re-resized by the engine to its stock 3136
    assert len(failures) == 2
    assert all(failure["engine_media_tokens"] > failure["client_media_tokens"] for failure in failures)


def test_the_harness_runs_the_media_stage_beside_stages_1_and_2(recipe: Any, tmp_path: Path) -> None:
    """``run`` adds the media stage for a recipe with media input; stage 2 compares the text rows and
    reports the fixture's declared media approximation non-gating (decision 35)."""
    pairs = media_pairs(tmp_path / "pairs.jsonl")
    engine = stub_for(recipe)
    try:
        document = run_equivalence(
            recipe,
            base_url=engine.base_url,
            pairs_path=str(pairs),
            out_dir=str(tmp_path / "out"),
            stages=[1, 2],
            reference_python=REFERENCE_PYTHON,
        )
    finally:
        engine.stop()
    assert document["stage1"]["media_rows"] == 1 + len(SIZES) and document["stage1"]["passed"] is True
    assert document["stage2"]["passed"] is True and document["media"]["passed"] is True
    assert document["passed"] is True
    assert "## Media" in (tmp_path / "out" / "EQUIVALENCE.md").read_text(encoding="utf-8")


def test_a_media_recipe_without_media_rows_fails_and_a_text_recipe_has_no_media_stage(
    recipe: Any, tmp_path: Path
) -> None:
    pairs = write_pairs(tmp_path / "pairs.jsonl", sample_pairs()[:1])
    document = stage_media(recipe, pairs, REFERENCE_PYTHON)
    assert document is not None and document["status"] == "no_media_rows" and document["passed"] is False
    assert stage_media(load_recipe(RECIPES / "fixture-embed"), pairs, REFERENCE_PYTHON) is None


def test_a_side_the_reference_refuses_fails_as_refused_even_when_the_client_refused_it_too(
    recipe: Any, tmp_path: Path
) -> None:
    """Two images on one document: the client refuses the request (max_images 1) and the card does not define
    it. The stage names the card's refusal -- what the generator prunes -- not merely a side the client did not
    send."""
    rows = [{"query": "two pages", "documents": [""], "media": {"documents": [[png_entry(64, 64), png_entry(32, 32)]]}}]
    document = stage_media(recipe, write_pairs(tmp_path / "pairs.jsonl", rows), REFERENCE_PYTHON)
    assert document is not None and document["passed"] is False
    assert [failure["check"] for failure in document["failures"]] == ["reference_refused"]
    assert document["refusals"] and "CapabilityError" in document["refusals"][0]["error"]


def test_an_image_whose_tokens_are_not_counted_fails(
    recipe: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A token count neither side produced is never an equality: uncounted (None) on both sides fails the item."""
    from rcp_ndcg_test.equivalence import media

    from rcp_ndcg.data import resolution

    def uncountable(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("no count")

    real = media._reference_facts

    def uncounted_reference(*args: Any, **kwargs: Any) -> Any:
        facts = real(*args, **kwargs)
        for side in facts.values():
            for item in side["media"]:
                item["tokens"] = None
        return facts

    monkeypatch.setattr(resolution, "content_media_tokens", uncountable)
    monkeypatch.setattr(media, "_reference_facts", uncounted_reference)
    document = stage_media(recipe, media_pairs(tmp_path / "pairs.jsonl"), REFERENCE_PYTHON)
    assert document is not None and document["passed"] is False
    assert {failure["check"] for failure in document["failures"]} == {"tokens"}


# ---------------------------------------------------------------------------
# Video and interleaved rows (the media gate's video half).
# ---------------------------------------------------------------------------


def video_entry(width: int, height: int, num_frames: int) -> dict[str, Any]:
    """One pairs video entry: an MJPEG AVI clip of ``num_frames`` frames at ``width`` x ``height``."""
    from rcp_ndcg_test.observe.media_set import video_entry as media_set_video_entry

    return media_set_video_entry("icon", width, height, num_frames)


def video_recipe() -> Any:
    """The video fixture (fixture-vl-video): a video policy of 4 pinned frames, two images, one video."""
    from rcp_ndcg_vllm import load_recipe

    from tests.conftest import RECIPES

    return load_recipe(RECIPES / "fixture-vl-video")


@pytest.fixture(scope="module")
def vl_recipe() -> Any:
    return video_recipe()


def video_pairs(path: Path, *, clip_frames: int = 12) -> Path:
    """Image rows, an interleaved row (text-image-text-image), and a video row per size, alone and with
    text: the containers carry more frames than the fixture's declared sampling (4), so a pinned engine
    samples exactly the policy's frames and an unpinned one does not."""
    rows: list[dict[str, Any]] = [
        {"query": "the clip", "documents": [""], "media": {"documents": [[video_entry(64, 64, clip_frames)]]}},
        {
            "query": "the moving clip",
            "documents": ["a caption under the clip"],
            "media": {"query": [], "documents": [[video_entry(224, 224, clip_frames)]]},
        },
        {
            "query": "two pages and their captions",
            "documents": [""],
            "media": {
                "query": [],
                "documents": [
                    [
                        {"kind": "text", "text": "the chart opens the page, "},
                        png_entry(64, 64),
                        {"kind": "text", "text": " and the table closes it"},
                        png_entry(120, 90),
                    ]
                ],
            },
        },
        {
            "query": "a plain batch",
            "documents": ["a plain text document", ""],
            "media": {"query": [], "documents": [[], [png_entry(300, 200)]]},
        },
        {
            "query": "whose page is this",
            "documents": ["a plain text document"],
            "media": {"query": [png_entry(64, 64)], "documents": [[]]},
        },
    ]
    return write_pairs(path, rows)


def test_video_and_interleaved_rows_gate_against_the_stub(vl_recipe: Any, tmp_path: Path) -> None:
    """Offline: every video item's declared frame count equals the reference's, every interleaved row's
    placement (every text part standing where it stands around the media) and every image's facts match,
    and the client counted every side; the engine's count is not run without an engine."""
    document = stage_media(vl_recipe, video_pairs(tmp_path / "pairs.jsonl"), REFERENCE_PYTHON)
    assert document is not None, document
    assert document["passed"] is True, document["failures"][:3]
    assert document["items"] == 6  # two clips, two interleaved images, the mixed batch's image, the query's
    assert document["engine_check"]["status"] == "not_run" and document["engine_check"]["passed"] is None


def test_the_stub_counts_a_clip_and_an_unpinned_engine_does_not(vl_recipe: Any, tmp_path: Path) -> None:
    """With an engine: the stub samples the container under its served pin (--media-io-kwargs) and counts
    it as the client counted it; served without the pin, the engine samples its own default frame count and
    the stage fails on the video rows' engine count (the images are unaffected)."""
    pairs = video_pairs(tmp_path / "pairs.jsonl")
    engine = stub_for(vl_recipe)
    try:
        document = stage_media(vl_recipe, pairs, REFERENCE_PYTHON, base_url=engine.base_url)
    finally:
        engine.stop()
    assert document is not None and document["passed"] is True, (document["failures"], document["engine_check"])
    assert document["engine_check"]["checked"] == 5 and document["engine_check"]["passed"] is True
    unpinned = vl_recipe.model_copy(update={"serve": vl_recipe.serve.model_copy(update={"extra_args": []})})
    engine = stub_for(unpinned)
    try:
        document = stage_media(vl_recipe, pairs, REFERENCE_PYTHON, base_url=engine.base_url)
    finally:
        engine.stop()
    assert document is not None and document["passed"] is False
    assert document["failures"] == [] and document["engine_check"]["passed"] is False
    assert len(document["engine_check"]["failures"]) == 2  # the two clip rows, not the image rows
    assert all(
        failure["engine_media_tokens"] > failure["client_media_tokens"]
        for failure in document["engine_check"]["failures"]
    )


def test_a_clip_shorter_than_the_declared_sampling_is_refused(vl_recipe: Any, tmp_path: Path) -> None:
    """A container with fewer frames than the policy's num_frames is refused by the client's own video
    policy (a short clip is not shown whole: vLLM would resample it at its processor's own rate) -- the stage
    names the refusal."""
    rows = [
        {"query": "short clip", "documents": [""], "media": {"documents": [[video_entry(64, 64, 2)]]}},
    ]
    document = stage_media(vl_recipe, write_pairs(tmp_path / "pairs.jsonl", rows), REFERENCE_PYTHON)
    assert document is not None and document["passed"] is False
    assert document["refusals"] and "VideoPolicyError" in document["refusals"][0]["error"]
    assert "frames" in document["refusals"][0]["error"]


def test_an_interleaved_row_pins_the_given_part_order(
    vl_recipe: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mutation: the client hoists the media before the text (the lowering it had before it kept the
    given part order) -- the interleaved row's placement differs from the card's and the stage fails on it."""
    from rcp_ndcg_core.content import Content, TextPart

    from rcp_ndcg.inference.clients._base import RoleClient

    def media_first(content: Content, text: str) -> Content:
        parts: list[Any] = [part for part in content.parts if not isinstance(part, TextPart)]
        if text or not content.has_media:
            parts.append(TextPart(text=text))
        return Content.from_parts(parts)

    monkeypatch.setattr(RoleClient, "_with_text", staticmethod(media_first))
    document = stage_media(vl_recipe, video_pairs(tmp_path / "pairs.jsonl"), REFERENCE_PYTHON)
    assert document is not None and document["passed"] is False
    assert {failure["check"] for failure in document["failures"]} == {"placement"}


# --- stage 2 over the media rows (owner decision 35 addendum) ----------------------------------


class _FakeEmbeddings:
    """The client seam's minimal embeddings result: one constant vector per text."""

    def __init__(self, vectors: list[list[float]]) -> None:
        self.vectors = vectors
        self.offsets = None


class _FakeMediaClient:
    """A fake role client that records whether each call carried media and returns constant vectors."""

    def __init__(self) -> None:
        self.processing: list[Any] = []
        self.media_calls: list[bool] = []

    def encode(self, contents: list[Any], role: Any) -> _FakeEmbeddings:
        self.media_calls.append(bool(contents[0].has_media))
        return _FakeEmbeddings([[1.0, 0.0]])


class _FakeCapture:
    def __init__(self) -> None:
        self.exchanges: list[dict[str, Any]] = []


def _constant_reference(rows_seen: list[list[dict[str, Any]]]) -> Any:
    """A fake ``run_reference``: records the rows it receives and returns constant vectors for each."""

    def run(reference_python: str, entry: str, *, mode: str, pairs_path: Path, **kwargs: Any) -> dict[str, Any]:
        rows = [json.loads(line) for line in Path(pairs_path).read_text(encoding="utf-8").splitlines() if line.strip()]
        rows_seen.append(rows)
        return {
            "rows": [
                {
                    "index": index,
                    "query_vectors": [[[1.0, 0.0]]],
                    "document_vectors": [[[1.0, 0.0]] for _ in row["documents"]],
                }
                for index, row in enumerate(rows)
            ]
        }

    return run


def _media_stage2_pairs(tmp_path: Path) -> Path:
    rows = [
        {"query": "a text query", "documents": ["a text document"]},
        {"query": "the red page", "documents": [""], "media": {"documents": [[png_entry(300, 200)]]}},
    ]
    return write_pairs(tmp_path / "pairs.jsonl", rows)


def test_stage2_compares_media_rows_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Decision 35: stage 2 runs over the media rows too -- the reference receives them and the client
    sends the product's Content (the media stage's path), so an image's vector gates like a text's."""
    from rcp_ndcg_test.equivalence import stages as stages_module

    recipe = load_recipe(RECIPES / "fixture-vl-embed")
    # The fixture declares the approximation; this test proves the default (no declaration) gates.
    reference = {**recipe.reference.model_dump(mode="json"), "known_deviations": []}
    recipe = recipe.model_copy(update={"reference": recipe.reference.model_validate(reference)})
    rows_seen: list[list[dict[str, Any]]] = []
    fake_client = _FakeMediaClient()
    monkeypatch.setattr(stages_module, "run_reference", _constant_reference(rows_seen))
    monkeypatch.setattr(stages_module, "role_client", lambda recipe, base_url, **kwargs: (fake_client, _FakeCapture()))
    summary = stages_module.stage2_scores(
        recipe, _media_stage2_pairs(tmp_path), "fake-python", base_url="http://engine"
    )
    assert summary["passed"] is True
    assert summary["media_rows"]["n_rows"] == 1
    assert summary["media_rows"]["gating"] is True
    assert len(rows_seen) == 1 and len(rows_seen[0]) == 2  # both rows reached the reference
    assert rows_seen[0][1]["media"]["documents"]  # the media row kept its bytes
    assert fake_client.media_calls == [False, True]  # the second call carried the image


def test_stage2_reports_a_declared_media_approximation_non_gating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A recipe declaring ``media_approximation``: its media rows are reported non-gating with the
    declaration named, never silently dropped or compared."""
    from rcp_ndcg_test.equivalence import stages as stages_module

    recipe = load_recipe(RECIPES / "fixture-vl-embed")
    reference = {**recipe.reference.model_dump(mode="json"), "known_deviations": ["media_approximation"]}
    recipe = recipe.model_copy(update={"reference": recipe.reference.model_validate(reference)})
    rows_seen: list[list[dict[str, Any]]] = []
    monkeypatch.setattr(stages_module, "run_reference", _constant_reference(rows_seen))
    monkeypatch.setattr(
        stages_module, "role_client", lambda recipe, base_url, **kwargs: (_FakeMediaClient(), _FakeCapture())
    )
    summary = stages_module.stage2_scores(
        recipe, _media_stage2_pairs(tmp_path), "fake-python", base_url="http://engine"
    )
    assert summary["passed"] is True
    assert summary["media_rows"]["known_approximation"] is True
    assert summary["media_rows"]["gating"] is False
    assert summary["media_rows"]["n_rows"] == 1
    assert "media_approximation" in summary["media_rows"]["reason"]
    assert len(rows_seen[0]) == 1  # only the text row reached the reference
