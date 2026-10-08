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
import sys
from pathlib import Path
from typing import Any

import pytest
from rcp_ndcg_vllm import load_recipe
from rcp_ndcg_vllm.equivalence import run as run_equivalence
from rcp_ndcg_vllm.equivalence.media import stage_media
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
    """``run`` adds the media stage for a recipe with media input; stages 1 and 2 compare the text rows only."""
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
    from rcp_ndcg_vllm.equivalence import media

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
