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
from types import SimpleNamespace
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
    # The gate declares its scope: it is an INPUT gate (no media vector or score is compared here).
    assert document["scope"] == "input"
    assert "no media vector or score is compared" in document["scope_note"]


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
    report = (tmp_path / "out" / "EQUIVALENCE.md").read_text(encoding="utf-8")
    assert "## Media" in report
    assert "scope: **input**" in report, "the report must label the media gate's input-only scope"


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
        # The MRL gate reads the declaration from the client's config; the fixture declares no head.
        self.config = SimpleNamespace(mrl_kind="none")

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


# ---------------------------------------------------------------------------
# A3: the frame probes -- the media row reaches the template check, and the engine's
# own prompt-token report is compared with the render the client budgeted against.
# ---------------------------------------------------------------------------


MEDIA_HEAD_TEMPLATE = (
    "{%- for message in messages -%}{% for part in message.content %}"
    "{% if part.type == 'image' %}XX{% endif %}{% endfor %}"
    "doc: {% for part in message.content -%}"
    "{%- if part.type == 'text' %}{{ part.text }}{% endif -%}"
    "{%- endfor %}{%- endfor -%}{% if add_generation_prompt %} [END]{% endif %}"
)
"""The fixture's served template, with a marker the engine emits only when a conversation carries media."""


def stage1_media_pairs(path: Path) -> Path:
    """One text row and one media row (an image-only document): the text row's conversation and the media
    row's, for the messages template check."""
    rows: list[dict[str, Any]] = [dict(sample_pairs()[0])]
    rows[0].pop("instruction", None)
    rows.append(
        {"query": "the red page", "documents": [""], "media": {"query": [], "documents": [[png_entry(300, 200)]]}}
    )
    return write_pairs(path, rows)


def test_stage1_renders_one_media_row_per_shape_in_the_template_check(recipe: Any, tmp_path: Path) -> None:
    """A3(b): the messages template check renders a media row's conversation, not only the text rows -- the
    served frame around the media item's content must equal the declared frame, and the render WITH the
    media must still open and close with the declared frame's fixed edges."""
    from rcp_ndcg_test.equivalence.stages import stage1_prompts

    document = stage1_prompts(recipe, stage1_media_pairs(tmp_path / "pairs.jsonl"), None, over_length_per_shape=1)
    check = document["template_render_check"]
    assert check["status"] == "run" and check["passed"] is True, check["failures"][:2]
    # every conversation the client sent is rendered: the text row's documents, the over-length sample's and
    # the media row's (which the text-only check never saw)
    assert check["checked"] == len(sample_pairs()[0]["documents"]) + 2


def test_the_media_template_probe_picks_the_row_that_carries_that_shapes_media(vl_recipe: Any, tmp_path: Path) -> None:
    """A3(b): the probe picks, per shape, a media row that carries media ON THAT SHAPE'S SIDE.  The pairs
    file's first media row is document-only; rendering it for the query shape would skip the query side's
    media, so the query frame would never be checked (the shipped qwen3-vl/embeddinggemma/pplx pairs files
    carry their query-media row after document-only rows)."""
    from rcp_ndcg_test.equivalence.fitting import load_pairs
    from rcp_ndcg_test.equivalence.media import media_rows
    from rcp_ndcg_test.equivalence.stages import _media_template_rows, stage1_prompts

    rows = [
        {"query": "a document page", "documents": [""], "media": {"documents": [[png_entry(64, 64)]]}},
        {
            "query": "a query page",
            "documents": ["a caption"],
            "media": {"query": [png_entry(64, 64)], "documents": [[]]},
        },
    ]
    pairs = write_pairs(tmp_path / "pairs.jsonl", rows)
    chosen = {row["shape"]: row for row in _media_template_rows(vl_recipe, media_rows(load_pairs(pairs)))}
    assert chosen["query"]["media"]["query"], "the query shape must get the row that carries query media"
    assert chosen["document"]["media"]["documents"][0], "the document shape must get a document-media row"
    document = stage1_prompts(vl_recipe, pairs, REFERENCE_PYTHON, over_length_per_shape=1)
    check = document["template_render_check"]
    assert check["passed"] is True, check["failures"][:2]
    # With a reference: the fixture's render mode emits BOTH declared shapes, so the every-text render
    # comparison holds the query and the document side to it (it emitted only the document shape before).
    assert document["render_check"]["passed"] is True, document["render_check"]["failures"][:2]
    # every captured conversation is rendered: the text row under both declared shapes, plus one media row
    # per shape (the query's and the document's)
    assert check["checked"] == 4, check


def test_a_media_frame_that_moves_the_declared_head_fails_the_template_check(recipe: Any, tmp_path: Path) -> None:
    """The mutation: a served template whose media render emits a marker before the declared head is caught
    by the media frame check (the text-only rows never exercise that branch)."""
    import shutil

    from rcp_ndcg_test.equivalence.stages import stage1_prompts

    directory = tmp_path / "recipes" / "fixture-vl-embed"
    shutil.copytree(RECIPES / "fixture-vl-embed", directory)
    shutil.copy(RECIPES.parent / "tokenizer.json", tmp_path / "tokenizer.json")  # ../../tokenizer.json
    (directory / "chat.jinja").write_text(MEDIA_HEAD_TEMPLATE, encoding="utf-8")
    mutated = load_recipe(directory)
    document = stage1_prompts(mutated, stage1_media_pairs(tmp_path / "pairs.jsonl"), None, over_length_per_shape=1)
    check = document["template_render_check"]
    assert check["passed"] is False
    assert {failure["check"] for failure in check["failures"]} == {"media_head"}, check["failures"][:2]


def test_the_engine_prompt_tokens_probe_compares_the_engine_report_with_the_declared_render(
    recipe: Any, tmp_path: Path
) -> None:
    """A3(a): the engine's own ``usage.prompt_tokens`` of one captured request per shape must equal the count
    of the render the client budgeted against -- the probe that the declared frame (the ``messages`` budget's
    premise) is the engine's frame.  Run and passed against the stub; ``not_run`` without an engine."""
    from rcp_ndcg_test.equivalence.stages import stage1_prompts

    pairs = stage1_media_pairs(tmp_path / "pairs.jsonl")
    engine = stub_for(recipe)
    try:
        document = stage1_prompts(recipe, pairs, None, base_url=engine.base_url, over_length_per_shape=1)
    finally:
        engine.stop()
    check = document["engine_prompt_tokens_check"]
    assert check["status"] == "run" and check["passed"] is True, check["failures"][:2]
    assert check["checked"] > 0
    offline = stage1_prompts(recipe, pairs, None, over_length_per_shape=1)
    assert offline["engine_prompt_tokens_check"]["status"] == "not_run"
    assert offline["engine_prompt_tokens_check"]["passed"] is None  # not_run is neutral, never passed


def test_the_engine_prompt_tokens_probe_fails_on_a_drifted_report(recipe: Any) -> None:
    """A mutant engine whose report is one token off fails the probe with both numbers named (the driver is
    the check itself, so no live engine is needed)."""
    from rcp_ndcg_test.equivalence import stages as stages_module
    from rcp_ndcg_test.equivalence.fitting import tokenizer_of

    tokenizer = tokenizer_of(recipe)
    text = "doc: the page [END]"
    count = tokenizer.count(text, add_special_tokens=True)
    probe = {"rows": [{"shapes": {"document": {"texts": [text], "usages": [count]}}}]}
    check = stages_module._engine_prompt_tokens_check(recipe, probe, tokenizer, "http://engine")
    assert check["status"] == "run" and check["passed"] is True and check["checked"] == 1
    drifted = {"rows": [{"shapes": {"document": {"texts": [text], "usages": [count + 1]}}}]}
    check = stages_module._engine_prompt_tokens_check(recipe, drifted, tokenizer, "http://engine")
    assert check["passed"] is False
    assert check["failures"][0]["engine_prompt_tokens"] == count + 1
    assert check["failures"][0]["declared_render_tokens"] == count


# ---------------------------------------------------------------------------
# The fps arm: a clip's realised frame count follows the engine's fps rule, on every side.
# ---------------------------------------------------------------------------


def fps_recipe(tmp_path: Path) -> Any:
    """A copy of the video fixture that declares the engine's fps rule instead of a pinned frame count: the
    Qwen3-VL backend samples by fps and ignores ``num_frames``, so the declared policy is the rate and the
    engine's argv pins it.  The reference computes the realised count from the clip's own facts."""
    import shutil

    directory = tmp_path / "recipes" / "fixture-vl-video-fps"
    shutil.copytree(RECIPES / "fixture-vl-video", directory)
    shutil.copy(RECIPES.parent / "tokenizer.json", tmp_path / "tokenizer.json")  # ../../tokenizer.json
    shutil.copy(RECIPES.parent / "deterministic.py", tmp_path / "deterministic.py")  # the reference's import
    manifest = (directory / "family.yaml").read_text(encoding="utf-8")
    manifest = manifest.replace("id: fixture-vl-video", "id: fixture-vl-video-fps")
    manifest = manifest.replace("image_processor: qwen2_vl", "image_processor: qwen3_vl")
    manifest = manifest.replace(
        """extra_args: ["--media-io-kwargs", '{"video": {"num_frames": 4}}']""",
        """extra_args: ["--media-io-kwargs", '{"video": {"fps": 2}}']""",
    )
    manifest = manifest.replace(
        "video_policy: {num_frames: 4, wire: video_url, engine_video_pinning: true}",
        "video_policy: {fps: 2, wire: video_url, engine_video_pinning: true}",
    )
    (directory / "family.yaml").write_text(manifest, encoding="utf-8")
    return load_recipe(directory)


def fps_pairs(path: Path) -> Path:
    """Two clip rows (64 frames at 8 fps): the fps rule realises 16 frames -- neither the clip's 64 nor the
    loader's default 32."""
    rows = [
        {"query": "the clip", "documents": [""], "media": {"documents": [[video_entry(64, 64, 64)]]}},
        {
            "query": "the moving clip",
            "documents": ["a caption under the clip"],
            "media": {"query": [], "documents": [[video_entry(64, 64, 64)]]},
        },
    ]
    return write_pairs(path, rows)


def test_an_fps_clip_gates_on_the_rules_realised_frame_count(tmp_path: Path) -> None:
    """The client counts the fps rule's realised frames (16, not the clip's 64 or the loader's 32), the
    reference computes the same count from the clip's facts, and the engine -- served with the declared pin
    -- counts exactly the client's tokens; served without the pin it samples its own default and the media
    gate's engine check fails."""
    from rcp_ndcg_test.equivalence import media as media_module
    from rcp_ndcg_test.equivalence.fitting import load_pairs
    from rcp_ndcg_test.equivalence.media import media_rows

    recipe = fps_recipe(tmp_path)
    pairs = fps_pairs(tmp_path / "pairs.jsonl")
    facts, _, _ = media_module._client_facts(recipe, media_rows(load_pairs(pairs)), None)
    frames = [item["frames"] for side in facts.values() for item in side["media"]]
    assert frames == [16, 16], f"the fps rule realises 16 of the clip's 64 frames at 8 fps, got {frames}"
    document = stage_media(recipe, pairs, REFERENCE_PYTHON)
    assert document is not None and document["passed"] is True, document["failures"][:3]
    engine = stub_for(recipe, "--model-processor", "qwen3_vl")
    try:
        pinned = stage_media(recipe, pairs, REFERENCE_PYTHON, base_url=engine.base_url)
    finally:
        engine.stop()
    assert pinned is not None and pinned["passed"] is True, (pinned["failures"], pinned["engine_check"])
    assert pinned["engine_check"]["checked"] == 2
    # An engine pinned to the WRONG rate (4 fps: 32 frames, not the declared 2 fps / 16): a pinned
    # num_frames cannot express the drift -- the product refuses that policy on this backend at count time.
    engine = stub_for(recipe, "--model-processor", "qwen3_vl", "--media-io-kwargs", '{"video": {"fps": 4}}')
    try:
        drifted = stage_media(recipe, pairs, REFERENCE_PYTHON, base_url=engine.base_url)
    finally:
        engine.stop()
    assert drifted is not None and drifted["passed"] is False
    assert drifted["failures"] == [] and drifted["engine_check"]["passed"] is False
    assert len(drifted["engine_check"]["failures"]) == 2
    assert all(
        failure["engine_media_tokens"] > failure["client_media_tokens"]
        for failure in drifted["engine_check"]["failures"]
    )
