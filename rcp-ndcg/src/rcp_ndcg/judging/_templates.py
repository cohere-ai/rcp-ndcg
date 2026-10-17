"""Rendering a judging prompt: the query and a window of documents into the template's placeholders.

Documents are rendered as ``<doc id="doc_N">`` blocks (the positional ids every
parser relies on); media parts become markers in the text and are carried
alongside it, so the text can be parsed and measured while the payload carries the
images.
"""

from __future__ import annotations

import html
import re
from collections.abc import Sequence
from dataclasses import dataclass

from rcp_ndcg_core.content import Content, ImagePart, Part, TextPart, VideoPart

from rcp_ndcg.errors import DataError

#: Stands in for one media part -- an image, or a whole video -- inside a rendered prompt.
#:
#: Cannot be forged by document text: every body that goes through
#: :func:`wrap_xml` is ``html.escape``d first, so a document containing this
#: literally arrives as ``&lt;media ...``.  :func:`split_media` additionally
#: requires that the markers found are exactly the media collected, each once, so
#: an injection through an *unescaped* field (a query, an instruction) is a loud
#: mismatch rather than a duplicated or dropped image.
MEDIA_MARKER = '<media index="{index}"/>'
_MEDIA_MARKER_RE = re.compile(r'<media index="(\d+)"/>')

#: Framing markup a query must not be able to forge: the document block, the documents wrapper and the media
#: markers. The ``<`` is escaped (``&lt;doc``), which is how the documents' own text is neutralised too, so the
#: positional ``doc_N`` contract and the marker count stay the template's.
_QUERY_MARKUP = re.compile(r"<(?=/?(?:doc|documents|media)\b)", flags=re.IGNORECASE)


def neutralise_query(query: str) -> str:
    """``query`` with its framing markup escaped: a query cannot forge a ``<doc>`` block, the ``<documents>``
    wrapper or a media marker. Ordinary text -- and a placeholder tag the query happens to contain -- is left
    as it is (the template substitutes in one pass, so an inserted value is never rescanned)."""
    return _QUERY_MARKUP.sub("&lt;", query)


def wrap_xml(documents: Sequence[Content]) -> str:
    """Render documents as the ``<documents><doc id="doc_N">`` block.

    Media parts become :data:`MEDIA_MARKER` placeholders at their position in the
    document, one per part, which :func:`split_media` later turns back into the
    same parts. A video is one marker whether it is sent as a container or as
    frames: what it lowers to on the wire is the payload's decision, not the
    template's.
    The marker sits *inside* the ``<doc>`` element, so the positional ``doc_N``
    contract every parser depends on is unchanged whether a document is text,
    pixels, or both.
    """
    media_index = 0
    xml_output = "<documents>\n"
    for i, doc in enumerate(documents):
        # Positional ids (doc_1, doc_2, ...) are the contract the parsers rely on.
        body, media_index = _render_body(doc, media_index)
        xml_output += f'<doc id="doc_{i + 1}">\n{body}\n</doc>\n'

    xml_output += "</documents>"
    return xml_output


def rendered_text(text: str) -> str:
    """A document's text as the prompt carries it: stripped and XML-escaped. The window budget counts this."""
    return html.escape(text.strip())


def _render_body(content: Content, media_index: int) -> tuple[str, int]:
    """One document's inner text, with media replaced by markers.

    Returns the body and the next free media index, so numbering runs across the
    whole window rather than restarting per document.
    """
    if not content.has_media:
        return rendered_text(content.text), media_index

    chunks: list[str] = []
    for part in content.parts:
        if isinstance(part, TextPart):
            if part.text.strip():
                chunks.append(rendered_text(part.text))
            continue
        chunks.append(MEDIA_MARKER.format(index=media_index))
        media_index += 1
    return "\n".join(chunks), media_index


MediaPart = ImagePart | VideoPart


def collect_media(documents: Sequence[Content]) -> list[MediaPart]:
    """The media parts :func:`wrap_xml` emitted markers for, in marker order.

    Kept separate from rendering so the text can be measured, logged, truncated
    and cached as text, with the assets resolved only at send time. Parts, not
    bare references: a video must come back out of :func:`split_media` as a video.
    """
    return [part for doc in documents for part in doc.parts if isinstance(part, ImagePart | VideoPart)]


def split_media(prompt: str, media: Sequence[MediaPart]) -> Content:
    """Turn a rendered prompt plus its media back into interleaved parts.

    The inverse of :func:`wrap_xml`'s marker substitution, and the last step
    before a prompt becomes a provider payload. Each marker becomes the part it
    stood for, type included.

    Raises:
        DataError: if the markers are not exactly ``media``, each once. A missing
            marker means a template dropped an image; a duplicate or an
            out-of-range one means something injected a marker through an
            unescaped field. Either way the prompt no longer says what the caller
            thinks it says, so it must not be sent.
    """
    if not media:
        if _MEDIA_MARKER_RE.search(prompt):
            raise DataError("prompt contains media markers but no media was collected for it")
        return Content.from_text(prompt)

    parts: list[Part] = []
    seen: list[int] = []
    cursor = 0
    for match in _MEDIA_MARKER_RE.finditer(prompt):
        index = int(match.group(1))
        if index >= len(media):
            raise DataError(f"media marker {index} is out of range for {len(media)} collected assets")
        if prompt[cursor : match.start()]:
            parts.append(TextPart(text=prompt[cursor : match.start()]))
        parts.append(media[index])
        seen.append(index)
        cursor = match.end()
    if prompt[cursor:]:
        parts.append(TextPart(text=prompt[cursor:]))

    if seen != list(range(len(media))):
        raise DataError(
            f"prompt markers {seen} do not match the {len(media)} collected assets exactly once each; "
            "a template dropped an image, or a marker was injected through an unescaped field"
        )
    return Content.from_parts(parts)


@dataclass(frozen=True)
class ResolvedPrompt:
    """One rendered prompt: the text, and the media its markers stand for.

    Both, together, because they have to agree. The text is what gets parsed and
    measured; the content is what gets sent. Deriving one from the other later is
    how they drift.
    """

    text: str
    content: Content | None
    """``None`` for a text-only prompt -- the marker-free case, kept distinct so
    the wire payload for a text run is unchanged."""


class Placeholder:
    """A ``{name}`` slot of a template, filled from the runtime values."""

    name: str

    def resolve(self, query: str, documents: Sequence[Content]) -> str:
        raise NotImplementedError


class QueryPlaceholder(Placeholder):
    """The query, with its framing markup escaped (:func:`neutralise_query`)."""

    name = "query_placeholder"

    def resolve(self, query: str, documents: Sequence[Content]) -> str:
        return neutralise_query(query)


class NumDocumentsPlaceholder(Placeholder):
    """The number of documents in the window."""

    name = "num_documents_placeholder"

    def resolve(self, query: str, documents: Sequence[Content]) -> str:
        return str(len(documents))


class PassagesPlaceholder(Placeholder):
    """The window's documents."""

    name = "passages_placeholder"

    def resolve(self, query: str, documents: Sequence[Content]) -> str:
        return wrap_xml(documents)


@dataclass
class Template:
    """A prompt text with placeholders."""

    template_str: str
    placeholders: Sequence[Placeholder]

    def resolve(self, *, query: str, documents: Sequence[Content]) -> str:
        """Replace each ``{placeholder}`` with its value for the query and the window's documents.

        The substitution is one pass over the template: a value that itself holds a placeholder tag (a query
        about ``{passages_placeholder}``) is inserted literally and never rescanned into another value.
        """
        values = {placeholder.name: placeholder.resolve(query, documents) for placeholder in self.placeholders}
        pattern = re.compile("|".join(re.escape("{" + name + "}") for name in values))
        return pattern.sub(lambda match: values[match.group(0)[1:-1]], self.template_str)

    def resolve_prompt(self, *, query: str, documents: Sequence[Content]) -> ResolvedPrompt:
        """:meth:`resolve`, plus the media its markers stand for (``content=None`` for a text prompt).

        A marker in the rendered text with no media collected is refused here (the one production call site),
        so an injected marker is loud even when the window carries no media at all.
        """
        text = self.resolve(query=query, documents=documents)
        media = collect_media(documents)
        if media or _MEDIA_MARKER_RE.search(text):
            return ResolvedPrompt(text=text, content=split_media(text, media))
        return ResolvedPrompt(text=text, content=None)
