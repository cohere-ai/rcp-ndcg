"""The reference implementation for google/embeddinggemma-2: the equivalence harness's subprocess reference.

Runs in its own reference environment, never inside the harness process:

    <reference-python> reference.py --mode render|embed|media --pairs <file> --out <file> \
        --tokenizer <spec> --recipe <resolved-recipe.json> [--device cpu|cuda:0]

The embed mode is the model card's own usage (README.md:101-119): a sentence-transformers
``SentenceTransformer("google/embeddinggemma-2")`` with ``prompt_name="SearchQuery"`` for queries and
``"Document"`` for documents -- the checkpoint's pipeline is Transformer -> Pooling (mean,
``include_prompt: true``) -> Normalize, so the vectors come back L2-normalised 768-d. The declared prompts
are read from the resolved recipe's ``client.query_prompt``/``client.doc_prompt`` and cross-checked against
the checkpoint's own ``config_sentence_transformers.json`` prompt table at run time, so a checkpoint whose
prompt strings changed is a loud error, never a silent drift.

- ``--mode render`` (stage 1's reference side): ``{"rows": [{"index", "shape", "text"}]}`` -- per pairs-file
  row and per declared shape (``query``, ``document``), the prompt the card's model reads: the task prefix
  plus the raw text, UNCUT. The card's ST encode truncates the assembled prompt right at its
  ``max_seq_length``; the recipe declares ``over_cap_cut_differs`` so over-cap rows are reported, not gated,
  and this mode never ports the client's content-boundary cut.
- ``--mode embed`` (stage 2's reference side): ``{"rows": [{"index", "query_vectors", "document_vectors"}]}``
  -- one 768-d L2-normalised vector per text through the card's ``encode`` (one query per row, one vector per
  document). A media-bearing row is refused loudly (:func:`_refuse_media_rows`): the media stage is its own
  mode and the embed mode compares text rows.
- ``--mode media`` (the media stage's reference side): for every pairs row carrying ``media``, per side, what
  the card's model consumes: the client's parts in order (the task prompt text, the media, the body text),
  each image's size after the checkpoint's
  ``Gemma4ImageProcessor`` resize (transformers' ``get_aspect_ratio_preserving_size``: the aspect ratio is
  scaled toward ``max_soft_tokens x pooling_kernel_size^2`` patches, both edges floored to
  ``patch_size x pooling_kernel_size`` = 48) and its prompt tokens (the pooled patches plus the two vision
  markers), and each video's declared frame count (the engine's pinned sampling, read from the resolved
  recipe's ``serve.extra_args``). Needs PIL only.

Reference environment (``requirements-reference.txt`` in this directory): transformers 5.19.0 (the first
release with ``EmbeddingGemma2Model``, ``EmbeddingGemma2Processor`` and the Gemma 4 image/video processors;
the checkpoint was saved with 5.18.0.dev0), sentence-transformers >=6.1.0 (the checkpoint's own requirement),
torch (the image's own), PIL, huggingface_hub, pyyaml. The render and media modes import none of them.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import sys
from pathlib import Path
from typing import Any

#: The checkpoint's own task prompts (config_sentence_transformers.json), the fallback when the resolved
#: recipe declares none. The card's Quick Start uses these names: SearchQuery / Document.
DEFAULT_QUERY_PROMPT = "task: search result | query: "
DEFAULT_DOC_PROMPT = "title: none | text: "

#: The shapes the render mode reports per pairs row.
SHAPES = ("query", "document")

#: The pairs-file media columns a text-only mode refuses (the media stage is its own mode).
_MEDIA_COLUMNS = (
    "image",
    "images",
    "video",
    "videos",
    "query_image",
    "query_video",
    "documents_images",
    "documents_videos",
)

#: The Gemma 4 processor's patch and pooling geometry (transformers 5.19.0
#: ``models/gemma4/image_processing_gemma4.py``).
_GEMMA4_PATCH = 16
_GEMMA4_POOLING = 3


def _refuse_media_rows(pairs: list[dict[str, Any]]) -> None:
    """A media-bearing row is refused, never silently dropped: the text modes take the text fields only, so
    a swallowed image or video column would compare the wrong content."""
    hits = sorted({column for row in pairs for column in _MEDIA_COLUMNS if row.get(column)})
    if hits:
        raise SystemExit(f"this mode is text-only, but the pairs carry media columns: {hits}")


def _load_recipe(path: str | None) -> dict[str, Any]:
    """The resolved recipe the harness wrote beside the output (a family reference is parameterised by its
    variant, so the recipe travels with every invocation)."""
    if not path:
        raise SystemExit("--recipe is required (the resolved recipe the harness passes)")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _prompts(recipe: dict[str, Any]) -> tuple[str, str]:
    """The declared task prompts (the client's ``query_prompt``/``doc_prompt``), with the checkpoint's own
    strings as the fallback."""
    client = recipe.get("client") or {}
    return (
        str(client.get("query_prompt") or DEFAULT_QUERY_PROMPT),
        str(client.get("doc_prompt") or DEFAULT_DOC_PROMPT),
    )


def _video_pin(recipe: dict[str, Any]) -> tuple[float, int]:
    """The engine's declared video sampling, ``(fps, max_frames)``, read from the resolved recipe's
    ``serve.extra_args`` ``--media-io-kwargs`` (the same numbers the recipe pins the engine to)."""
    args = [str(value) for value in (recipe.get("serve") or {}).get("extra_args") or []]
    if "--media-io-kwargs" not in args:
        raise SystemExit("the recipe declares no --media-io-kwargs video pin; the reference cannot sample")
    spec = json.loads(args[args.index("--media-io-kwargs") + 1])
    video = spec.get("video") or {}
    if "fps" not in video or "max_frames" not in video:
        raise SystemExit(f"the recipe's video pin declares no fps/max_frames: {video}")
    return float(video["fps"]), int(video["max_frames"])


def _sampled_frames(entry: dict[str, Any], fps: float, max_frames: int) -> int:
    """The frames the engine's pinned sampling shows for one container: ``max(1, int(duration x fps))``
    capped at ``max_frames`` (vLLM's ``EmbeddingGemma2VideoBackend.compute_frames_index_to_sample``,
    ``num_sampled`` then a uniform re-sample to the cap)."""
    duration = entry.get("duration_s")
    if duration is None:
        frames, source_fps = entry.get("num_frames"), entry.get("fps")
        if frames and source_fps:
            duration = float(frames) / float(source_fps)
    if duration is None:
        raise SystemExit(f"the video {entry.get('uri')!r} records no duration or frame rate; cannot sample")
    return min(max(1, int(float(duration) * fps)), max_frames)


def _image_size(entry: dict[str, Any]) -> tuple[int, int]:
    """An entry's image as the checkpoint's processor loads it (an inline ``data:`` URI's bytes)."""
    from PIL import Image

    uri = str(entry.get("uri", ""))
    if not uri.startswith("data:"):
        raise SystemExit(f"the media stage sends inline images; got {uri[:48]!r}")
    with Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1]))) as handle:
        return handle.size


def gemma4_resize(height: int, width: int, *, max_soft_tokens: int) -> tuple[int, int]:
    """The checkpoint's Gemma 4 image resize (a faithful port of transformers 5.19.0
    ``get_aspect_ratio_preserving_size``): scale both edges by ``sqrt(max_patches x patch^2 / area)`` with
    ``max_patches = max_soft_tokens x pooling^2`` and floor each to ``patch x pooling`` (48)."""
    max_patches = max_soft_tokens * _GEMMA4_POOLING**2
    target_px = max_patches * _GEMMA4_PATCH**2
    scale = math.sqrt(target_px / (height * width))
    side_mult = _GEMMA4_POOLING * _GEMMA4_PATCH
    target_height = int(math.floor(scale * height / side_mult)) * side_mult
    target_width = int(math.floor(scale * width / side_mult)) * side_mult
    if target_height == 0 and target_width == 0:
        raise SystemExit(f"a {height}x{width} image resizes to 0x0 under {max_soft_tokens} soft tokens")
    max_side = (max_patches // _GEMMA4_POOLING**2) * side_mult
    if target_height == 0:
        target_height = side_mult
        target_width = min(int(math.floor(width / height)) * side_mult, max_side)
    elif target_width == 0:
        target_width = side_mult
        target_height = min(int(math.floor(height / width)) * side_mult, max_side)
    return target_height, target_width


def gemma4_fixed_point(height: int, width: int, *, max_soft_tokens: int) -> tuple[int, int]:
    """The size the checkpoint's processor KEEPS: :func:`gemma4_resize` iterated to its fixed point.

    The engine runs the processor on the prepared bytes, and the Gemma 4 resize is not idempotent (the
    floor to 48 can move an edge on a second pass), so the client prepares the fixed point and the model
    consumes it; this mode reports that same geometry."""
    current = (height, width)
    for _ in range(16):
        following = gemma4_resize(*current, max_soft_tokens=max_soft_tokens)
        if following == current:
            return current
        current = following
    raise SystemExit(f"the Gemma 4 resize of {height}x{width} did not settle within 16 passes")


def media_side(
    text: str,
    entries: list[dict[str, Any]],
    *,
    prompt: str,
    image_budget: int,
    video_fps: float,
    video_max_frames: int,
) -> dict[str, Any]:
    """One side as the card's model consumes it: the client's parts in order -- the task prompt and the body
    text joined into ONE leading text part (the product's ``_with_text``: the fitted text stands where the
    content's first text part stood, which the prompt makes the front), the media entries in order, each
    image resized by the checkpoint's Gemma 4 processor and counted as pooled patches plus the two vision
    markers, each video as the engine's pinned frame count (its tokens are the engine's to count)."""
    placement: list[str] = []
    media: list[dict[str, Any]] = []
    if prompt or text:
        placement.append("text")
    for entry in entries:
        kind = str(entry.get("kind", "image"))
        if kind == "video":
            placement.append("video")
            media.append({"kind": "video", "frames": _sampled_frames(entry, video_fps, video_max_frames)})
            continue
        placement.append("image")
        width, height = _image_size(entry)
        target_height, target_width = gemma4_fixed_point(height, width, max_soft_tokens=image_budget)
        patches = (target_height // _GEMMA4_PATCH) * (target_width // _GEMMA4_PATCH)
        tokens = patches // _GEMMA4_POOLING**2 + 2
        media.append({"kind": "image", "width": target_width, "height": target_height, "tokens": tokens})
    return {"placement": placement, "media": media}


def mode_media(recipe: dict[str, Any], pairs: list[dict[str, Any]]) -> dict[str, Any]:
    """The media stage's reference side (:func:`media_side` per row and side that carries media)."""
    client = recipe.get("client") or {}
    image_budget = int((client.get("image_policy") or {}).get("max_soft_tokens") or 280)
    query_prompt, doc_prompt = _prompts(recipe)
    video_fps, video_max_frames = _video_pin(recipe)
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(pairs):
        media = row.get("media") or {}
        if media.get("query"):
            rows.append(
                {
                    "index": index,
                    "side": "query",
                    **media_side(
                        str(row["query"]),
                        list(media["query"]),
                        prompt=query_prompt,
                        image_budget=image_budget,
                        video_fps=video_fps,
                        video_max_frames=video_max_frames,
                    ),
                }
            )
        for position, entries in enumerate(media.get("documents") or []):
            if entries:
                rows.append(
                    {
                        "index": index,
                        "side": f"document {position}",
                        **media_side(
                            str(row["documents"][position]),
                            list(entries),
                            prompt=doc_prompt,
                            image_budget=image_budget,
                            video_fps=video_fps,
                            video_max_frames=video_max_frames,
                        ),
                    }
                )
    return {"rows": rows}


def mode_render(recipe: dict[str, Any], pairs: list[dict[str, Any]]) -> dict[str, Any]:
    """Stage 1's reference side: the card's prompt per row and shape, UNCUT (the recipe declares
    ``over_cap_cut_differs``; the card's own truncation happens inside encode, not in this render)."""
    _refuse_media_rows(pairs)
    query_prompt, doc_prompt = _prompts(recipe)
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(pairs):
        rows.append({"index": index, "shape": "query", "text": query_prompt + str(row["query"])})
        rows.append({"index": index, "shape": "document", "text": doc_prompt + str(row["documents"][0])})
    return {"rows": rows}


def mode_embed(recipe: dict[str, Any], pairs: list[dict[str, Any]], device: str) -> dict[str, Any]:
    """Stage 2's reference side: the card's own sentence-transformers path -- ``encode`` with
    ``prompt_name="SearchQuery"`` / ``"Document"``, one L2-normalised 768-d vector per text."""
    _refuse_media_rows(pairs)
    from sentence_transformers import SentenceTransformer

    query_prompt, doc_prompt = _prompts(recipe)
    model = SentenceTransformer(
        str(recipe["model"]),
        revision=str(recipe.get("revision") or "") or None,
        device=device if device not in ("", "auto") else None,
    )
    prompts = dict(getattr(model, "prompts", {}) or {})
    for name, declared in (("SearchQuery", query_prompt), ("Document", doc_prompt)):
        if name in prompts and prompts[name] != declared:
            raise SystemExit(
                f"the checkpoint's {name!r} prompt {prompts[name]!r} differs from the recipe's {declared!r}: "
                "a checkpoint whose prompt table changed is a new instrument"
            )

    def vectors(texts: list[str], prompt_name: str) -> list[list[float]]:
        if not texts:
            return []
        matrix = model.encode(texts, prompt_name=prompt_name, normalize_embeddings=True)
        return [[float(value) for value in vector] for vector in matrix]

    rows: list[dict[str, Any]] = []
    for index, row in enumerate(pairs):
        query_vectors = vectors([str(row["query"])], "SearchQuery")
        document_vectors = vectors([str(document) for document in row["documents"]], "Document")
        rows.append({"index": index, "query_vectors": query_vectors, "document_vectors": document_vectors})
    return {"rows": rows}


def main(argv: list[str] | None = None) -> int:
    """The subprocess CLI: run the mode and write the JSON."""
    parser = argparse.ArgumentParser(description="the embeddinggemma-2 reference (the card's ST path)")
    parser.add_argument("--mode", required=True, choices=["render", "embed", "media"])
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tokenizer", default="")
    parser.add_argument("--recipe", default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    pairs = [json.loads(line) for line in Path(args.pairs).read_text(encoding="utf-8").splitlines() if line.strip()]
    recipe = _load_recipe(args.recipe)
    if args.mode == "render":
        output = mode_render(recipe, pairs)
    elif args.mode == "media":
        output = mode_media(recipe, pairs)
    else:
        output = mode_embed(recipe, pairs, args.device)
    Path(args.out).write_text(json.dumps(output) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
