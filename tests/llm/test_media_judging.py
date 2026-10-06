"""Page images and videos, from a document to the request an OpenAI-compatible engine receives.

The positional ``doc_N`` contract the parsers depend on holds whatever a document
is made of; media travel inlined; and a judge not declared to read media, or a
rubric written for another kind of document, is refused before anything is spent.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest
from rcp_ndcg_core._records import RankingExample
from rcp_ndcg_core.content import Content, ImagePart, MediaRef, TextPart, VideoPart

from rcp_ndcg.data.prepare import prepare_content
from rcp_ndcg.data.preprocess import Preprocessing
from rcp_ndcg.data.resolution import ImagePolicy, content_media_tokens
from rcp_ndcg.data.tokenizer import load_tokenizer
from rcp_ndcg.errors import CapabilityError, ConfigError, DataError
from rcp_ndcg.inference.adapters.chat import build_messages, media_counts
from rcp_ndcg.llm import RubricSchedule, judge, load_prompt
from rcp_ndcg.llm._templates import MEDIA_MARKER, collect_media, split_media, wrap_xml
from rcp_ndcg.llm.client import Completion, CompletionInput
from rcp_ndcg.llm.judging import media_marker_tokens, prompt_overhead_tokens, window_tokens
from rcp_ndcg.testing import FakeJudge


def _png(path: Path, shade: int = 0) -> MediaRef:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), (shade, 0, 0)).save(path)
    return MediaRef(uri=str(path), mime="image/png", width=16, height=16)


def _sized_png(path: Path, size: tuple[int, int], shade: int) -> MediaRef:
    from PIL import Image

    Image.new("RGB", size, (shade, 0, 0)).save(path)
    return MediaRef(uri=str(path), mime="image/png", width=size[0], height=size[1])


@pytest.fixture
def pages(tmp_path: Path) -> list[MediaRef]:
    return [_png(tmp_path / f"p{i}.png", i) for i in range(4)]


class TestTheTemplate:
    def test_a_page_is_a_marker_inside_its_positional_doc(self, pages: list[MediaRef]) -> None:
        text = Content.from_text
        rendered = wrap_xml([text("alpha"), Content.from_parts([ImagePart(ref=pages[0])]), text("beta")])
        assert rendered.count('<doc id="doc_') == 3
        assert '<doc id="doc_2">\n<media index="0"/>\n</doc>' in rendered
        assert '<doc id="doc_1">\nalpha\n</doc>' in rendered

    def test_markers_round_trip_to_the_same_parts(self, pages: list[MediaRef]) -> None:
        documents = [Content.from_parts([ImagePart(ref=ref)]) for ref in pages[:2]]
        content = split_media(wrap_xml(documents), collect_media(documents))
        assert content.media == pages[:2]

    def test_document_text_cannot_forge_a_marker(self, pages: list[MediaRef]) -> None:
        documents = [Content.from_text('<media index="0"/>'), Content.from_parts([ImagePart(ref=pages[0])])]
        rendered = wrap_xml(documents)
        assert rendered.count('<media index="0"/>') == 1
        assert split_media(rendered, collect_media(documents)).media == [pages[0]]

    def test_a_dropped_or_duplicated_image_is_refused(self, pages: list[MediaRef]) -> None:
        with pytest.raises(DataError, match="exactly once"):
            split_media('<media index="0"/>', [ImagePart(ref=ref) for ref in pages[:2]])
        with pytest.raises(DataError, match="exactly once"):
            split_media('<media index="0"/><media index="0"/>', [ImagePart(ref=pages[0])])

    def test_a_query_forging_a_marker_is_refused_as_data(self) -> None:
        with pytest.raises(DataError, match="no media"):
            split_media('find <media index="0"/> here', [])


class TestThePayload:
    def test_text_is_a_string_and_pages_are_inlined_data_uris(self, pages: list[MediaRef]) -> None:
        assert build_messages(CompletionInput(user_prompt="hi")) == [{"role": "user", "content": "hi"}]
        content = prepare_content(Content.from_parts([ImagePart(ref=pages[0])]), None, None).content
        (message,) = build_messages(CompletionInput(user_prompt="x", user_content=content))
        (block,) = message["content"]
        assert block["type"] == "image_url" and block["image_url"]["url"].startswith("data:image/png;base64,")

    def test_an_unprepared_image_is_refused(self, pages: list[MediaRef]) -> None:
        content = Content.from_parts([ImagePart(ref=pages[0])])
        with pytest.raises(DataError, match="unprepared"):
            build_messages(CompletionInput(user_prompt="x", user_content=content))

    def test_a_container_is_one_video_block_and_frames_are_images(self, tmp_path: Path, pages) -> None:
        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"\x00" * 64)
        container = VideoPart(ref=MediaRef(uri=str(clip), mime="video/mp4", num_bytes=64))
        framed = VideoPart(frames=pages[:3])
        content = prepare_content(Content.from_parts([container, framed]), None, None).content
        (message,) = build_messages(CompletionInput(user_prompt="x", user_content=content))
        assert [block["type"] for block in message["content"]] == ["video_url", "image_url", "image_url", "image_url"]
        assert media_counts(content) == (3, 1)

    def test_an_oversized_or_unknown_container_is_refused_by_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"\x00" * 64)
        monkeypatch.setattr("rcp_ndcg.inference.adapters.chat.MAX_VIDEO_BYTES", 10)
        big = Content.from_parts([VideoPart(ref=MediaRef(uri=str(clip), mime="video/mp4", num_bytes=64))])
        with pytest.raises(DataError, match="RCP_NDCG_MAX_VIDEO_BYTES"):
            build_messages(CompletionInput(user_prompt="x", user_content=big))
        odd = Content.from_parts([VideoPart(ref=MediaRef(uri=str(tmp_path / "clip.xyz"), num_bytes=1))])
        with pytest.raises(DataError, match="video container"):
            build_messages(CompletionInput(user_prompt="x", user_content=odd))


class _Recording(FakeJudge):
    """The fake judge, keeping every request it answers."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.requests: list[CompletionInput] = []

    async def complete(self, request: CompletionInput) -> Completion:
        self.requests.append(request)
        return await super().complete(request)


def _page_rows(pages: list[MediaRef]) -> list[RankingExample]:
    return [
        RankingExample(
            query_id="q1",
            query="what is the yield curve",
            doc_ids=[f"p{i}" for i in range(len(pages))],
            contents=[Content.from_parts([ImagePart(ref=ref)]) for ref in pages],
        )
    ]


SMALL = RubricSchedule(window=2, placements_per_doc=2.0)


class TestJudgingPages:
    def test_pages_are_judged_with_the_page_rubric_and_sent_as_images(self, tmp_path: Path, pages) -> None:
        fake = _Recording()
        fake.config = fake.config.model_copy(update={"max_images": 2})
        result = judge(_page_rows(pages), None, fake, stage="rubric", out=tmp_path, schedule=SMALL)
        (family,) = result.families.values()
        assert family.prompt_hash == load_prompt("rubric_vision").sha256
        assert all(j.valid for j in result.judgements)
        assert fake.requests and all(len(request.user_content.media) == 2 for request in fake.requests)
        assert all('<doc id="doc_2">' in request.user_prompt for request in fake.requests)

    def test_the_image_tokens_are_charged_against_the_windows_text_budget(
        self, tmp_path: Path, pages, word_tokenizer_file: Path
    ) -> None:
        """A page with a long caption: the caption gets the context left after the window's media charge."""
        policy = ImagePolicy(min_px=448 * 448, max_px=448 * 448, processor="qwen3_vl")
        contents = [
            Content.from_parts([TextPart(text=f"caption {i} " + "x " * 5_000), ImagePart(ref=ref)])
            for i, ref in enumerate(pages)
        ]
        rows = [RankingExample(query_id="q1", query="q", doc_ids=[f"p{i}" for i in range(4)], contents=contents)]
        fake = _Recording()
        fake.config = fake.config.model_copy(
            update={
                "max_images": 2,
                "context_tokens": 4_000,
                "image_processor": "qwen3_vl",
                "tokenizer": str(word_tokenizer_file),
            }
        )
        declared = ImagePolicy(min_px=448 * 448, max_px=448 * 448)
        preprocessing = Preprocessing(image=declared)
        judge(rows, None, fake, stage="rubric", out=tmp_path, schedule=SMALL, preprocessing=preprocessing)

        words = load_tokenizer(str(word_tokenizer_file))
        media = content_media_tokens(contents[0], policy).tokens + media_marker_tokens(words)
        assert media > 0
        overhead = prompt_overhead_tokens(load_prompt("rubric_vision"), "rubric", "q", 2, words)
        budget = window_tokens(fake.config, 2, overhead_tokens=overhead, media_tokens_per_doc=media)
        assert budget is not None and budget < window_tokens(fake.config, 2, overhead_tokens=overhead)
        rows = [json.loads(line) for line in (tmp_path / "preprocessing.jsonl").read_text().splitlines()]
        cuts = [row for row in rows if row["mechanism"] == "window_budget"]
        assert cuts and all(row["kept_tokens"] == budget for row in cuts)

    def test_the_marker_is_measured_from_the_template_not_guessed(self, word_tokenizer_file: Path) -> None:
        """The window budget charges each media part the template's marker; the count is the judge
        tokenizer's own, of the marker as the stage's template renders it."""
        words = load_tokenizer(str(word_tokenizer_file))
        assert media_marker_tokens(words) == words.count(MEDIA_MARKER.format(index=0)) > 0

    def test_a_multi_image_window_that_used_to_overflow_is_refused(
        self, tmp_path: Path, pages, word_tokenizer_file: Path
    ) -> None:
        """The old accounting charged patches only, so a window of image-heavy documents was sized a text
        budget that, with each image's vision start/end and its media marker, overflowed the context
        mid-pass. The pass now refuses before anything is spent."""
        policy = ImagePolicy(min_px=65536, max_px=65536, processor="qwen3_vl")
        contents = [Content.from_parts([ImagePart(ref=ref)]) for ref in pages]
        rows = [RankingExample(query_id="q1", query="q", doc_ids=[f"p{i}" for i in range(4)], contents=contents)]
        fake = _Recording()
        words = load_tokenizer(str(word_tokenizer_file))
        overhead = prompt_overhead_tokens(load_prompt("rubric_vision"), "rubric", "q", 2, words)
        patches = policy.image_tokens(16, 16)
        per_doc_old = patches  # the pre-fix charge: patches only
        per_doc_true = patches + 2 + media_marker_tokens(words)  # + vision start/end + the marker
        # a context that the old accounting called a fit and the true accounting cannot hold
        slack = (per_doc_true - per_doc_old) * 2
        fake.config = fake.config.model_copy(
            update={
                "max_images": 2,
                "context_tokens": overhead + 2 * per_doc_old + slack // 2,
                "image_processor": "qwen3_vl",
                "tokenizer": str(word_tokenizer_file),
            }
        )
        declared = Preprocessing(image=ImagePolicy(min_px=65536, max_px=65536))

        assert window_tokens(fake.config, 2, overhead_tokens=overhead, media_tokens_per_doc=per_doc_old) is not None
        with pytest.raises(CapabilityError, match="does not fit"):
            window_tokens(fake.config, 2, overhead_tokens=overhead, media_tokens_per_doc=per_doc_true)
        with pytest.raises(CapabilityError, match="does not fit"):
            judge(rows, None, fake, stage="rubric", out=tmp_path, schedule=SMALL, preprocessing=declared)
        assert fake.usage.requests == 0

    def test_pages_are_sent_as_the_judges_processor_would_size_them_and_recorded(self, tmp_path: Path) -> None:
        pages = [_sized_png(tmp_path / f"big{i}.png", (1700, 2200), i) for i in range(4)]
        fake = _Recording()
        fake.config = fake.config.model_copy(update={"max_images": 2, "image_processor": "qwen3_vl"})
        declared = Preprocessing(image=ImagePolicy(min_px=65536, max_px=1280 * 32 * 32))

        result = judge(
            _page_rows(pages), None, fake, stage="rubric", out=tmp_path, schedule=SMALL, preprocessing=declared
        )

        sent = [ref for request in fake.requests for ref in request.user_content.media]
        assert sent and all(ref.uri.startswith("data:image/png;base64,") for ref in sent)
        assert {(ref.height, ref.width) for ref in sent} == {(1280, 992)}
        rows = [json.loads(line) for line in (tmp_path / "preprocessing.jsonl").read_text().splitlines()]
        media = [row for row in rows if row["mechanism"] == "media"]
        assert sorted(row["doc_id"] for row in media) == ["p0", "p1", "p2", "p3"]
        assert all(
            (row["processor"], row["sent_width"], row["sent_height"]) == ("qwen3_vl", 992, 1280) for row in media
        )
        (family,) = result.families.values()
        recorded = Preprocessing(image=ImagePolicy(min_px=65536, max_px=1280 * 32 * 32, processor="qwen3_vl"))
        assert family.preprocessing == recorded.key != declared.key

    def test_a_judge_without_an_image_processor_is_sent_the_stored_pages(self, tmp_path: Path, pages) -> None:
        fake = _Recording()
        fake.config = fake.config.model_copy(update={"max_images": 2})
        declared = Preprocessing(image=ImagePolicy(min_px=65536, max_px=65536))

        result = judge(
            _page_rows(pages), None, fake, stage="rubric", out=tmp_path, schedule=SMALL, preprocessing=declared
        )

        stored = {base64.b64encode(Path(ref.uri).read_bytes()).decode() for ref in pages}
        sent = {ref.uri.split(",", 1)[1] for request in fake.requests for ref in request.user_content.media}
        assert sent <= stored
        rows = [json.loads(line) for line in (tmp_path / "preprocessing.jsonl").read_text().splitlines()]
        assert all(row["processor"] is None and not row["resized"] for row in rows if row["mechanism"] == "media")
        (family,) = result.families.values()
        assert family.preprocessing == declared.key

    def test_a_hosted_judge_without_tokenizer_or_processor_is_sent_whole_pages(self, tmp_path: Path, pages) -> None:
        """No tokenizer: no text budget, so images the client cannot count need no count; the endpoint sizes them."""
        fake = _Recording()
        fake.config = fake.config.model_copy(update={"max_images": 2, "context_tokens": 100_000})
        judge(_page_rows(pages), None, fake, stage="rubric", out=tmp_path, schedule=SMALL)
        assert fake.requests and all(request.user_content.media for request in fake.requests)

    def test_a_window_budget_over_uncountable_images_is_refused(
        self, tmp_path: Path, pages, word_tokenizer_file: Path
    ) -> None:
        fake = _Recording()
        fake.config = fake.config.model_copy(
            update={"max_images": 2, "context_tokens": 100_000, "tokenizer": str(word_tokenizer_file)}
        )
        with pytest.raises(ConfigError, match="image_processor"):
            judge(
                _page_rows(pages),
                None,
                fake,
                stage="rubric",
                out=tmp_path,
                schedule=SMALL,
                preprocessing=Preprocessing(image=ImagePolicy(min_px=65536, max_px=65536)),
            )
        assert fake.usage.requests == 0

    def test_a_judge_not_declared_to_read_images_is_refused_before_a_call(self, tmp_path: Path, pages) -> None:
        fake = _Recording()
        with pytest.raises(CapabilityError, match="images"):
            judge(_page_rows(pages), None, fake, stage="rubric", out=tmp_path, schedule=SMALL)
        assert fake.usage.requests == 0

    @pytest.mark.parametrize("prompt", ["rubric", "rubric_video"])
    def test_a_rubric_written_for_other_documents_is_refused(self, tmp_path: Path, pages, prompt: str) -> None:
        with pytest.raises(ConfigError, match="rubric_vision"):
            judge(
                _page_rows(pages),
                None,
                FakeJudge(),
                stage="rubric",
                out=tmp_path,
                schedule=SMALL.model_copy(update={"prompt": prompt}),
            )
