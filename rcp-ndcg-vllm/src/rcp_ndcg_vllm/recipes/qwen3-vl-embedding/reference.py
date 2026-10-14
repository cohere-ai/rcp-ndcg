"""The reference implementation for the ``qwen3-vl-embedding`` family: the equivalence harness's subprocess reference.

Runs in its own reference environment, never inside the harness process:

    <reference-python> reference.py --mode render|embed|media --pairs <file> --out <file> \
        --tokenizer <repo>@<revision>|<path/to/tokenizer.json> [--device cpu|cuda:0] --recipe <resolved-recipe.json>

ONE reference for every size: the embed mode loads the model and revision the resolved
recipe names (``--recipe``), and the tokenizer spec the harness passes is that variant's own. Both
modes follow the model card's own code path, ``scripts/qwen3_vl_embedding.py`` (``Qwen3VLEmbedder``,
transformers), vendored VERBATIM beside this file as ``qwen3_vl_embedding.py``; the file is
byte-identical at every variant's pinned revision (2b:
https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B/blob/9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda/scripts/qwen3_vl_embedding.py,
8b: https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B/blob/2c4565515e0f265c6511776e7193b22c0968ddc7/scripts/qwen3_vl_embedding.py),
sha256 8ffa74a1a6bb759610c57865ea416fd4daf9936cb787520e1112a3e1d547f36a (pinned here and by the
recipe's test). The card's constants this file needs (``MAX_LENGTH`` and the default instruction) are
read from the vendored script's source, never restated.

- ``--mode render`` (stage 1's reference side): ``{"rows": [{"index", "shape", "text"}]}`` -- the prompt the
  card's model reads, per pairs-file row and per declared shape (the row's query for ``query``, its first
  document for ``document``): the card's ``format_model_input`` conversation (the default instruction as
  the system turn, the text -- or "NULL" for an empty input -- as the user turn) rendered as the
  checkpoint's chat template renders it with ``add_generation_prompt=True``, then the card's own
  truncation: ``_preprocess_inputs`` tokenizes with ``truncation=True, max_length=MAX_LENGTH``, a right
  cut of the whole prompt that reserves only the post-processor's appended endoftext. An over-cap input
  therefore loses the frame's tail (the recipe's declared ``anchor_drop_over_cap``); the cut is located
  by the kept tokens' character offsets on the rendered prompt (the ``tokenizers`` library's own
  truncation, as the processor runs it), never a decode. The post-processor's endoftext is the engine's
  and is not part of the text. Nothing here follows the product client's cut.
- ``--mode embed`` (stage 2's reference side): ``{"rows": [{"index", "query_vectors", "document_vectors"}]}``
  -- one L2-normalised vector per text or media side (2048-d at the 2b size, 4096-d at the 8b), through
  ``Qwen3VLEmbedder.process``. One declared deviation from the card's script: the reference calls
  ``process`` once per input instead of once per batch (:func:`_one_vector_per_input`) -- transformers'
  processor replaces every multimodal placeholder spelling found in the batch's texts from one shared
  iterator per modality, so a text row carrying the tokenizer's literal placeholder spellings makes the
  batched call raise ``StopIteration``; the card pools per sequence, so each input's vector is unchanged.
  The checkpoint is resolved with ``huggingface_hub.snapshot_download`` at the
  revision the resolved recipe names, so model and processor load the same pinned snapshot. A pairs row's
  ``media`` field is read (:func:`card_inputs`): an image entry becomes a loaded PIL image, a video entry is
  written to a scratch file and handed to the checkpoint's own ``Qwen3VLVideoProcessor`` at the recipe's
  declared fps (``client.video_policy.fps``) and whole-clip pixel pin (:func:`video_processor_kwargs`) -- the
  engine's own sampling and resize, not the card script's ``qwen_vl_utils`` route -- the text parts join the
  side's text, and the card's order (video, image, text) is the only order it can express -- an interleaved
  part sequence, several images/videos or an image before its video is refused loudly by :func:`side_parts`,
  never silently reordered.

- ``--mode media`` (the media stage's reference side): for every pairs row carrying ``media``, per side, what
  the card's model consumes -- the user turn's parts in the card's order (video, image, text; an interleaved
  text part where it stands), each image's size after the card's ``fetch_image`` resize (qwen-vl-utils'
  ``smart_resize`` under the card's MIN/MAX_PIXELS, read from the vendored script) and its tokens (merged
  patches plus the two vision markers), each container as the checkpoint's own ``Qwen3VLVideoProcessor``
  realises it (:func:`video_realisation`: the frames it sampled, the per-frame size it resized to and the
  patch tokens it produced). The image rows need PIL only; a video row loads the checkpoint's processor
  (transformers and the checkpoint's own video decoder). The render mode compares text rows (a media column
  there is refused loudly, :func:`_refuse_media_rows`). The recipe's ONE video sampling policy is the engine's
  fps rule (fps 2, the checkpoint video processor's rate), applied by the processor itself.

Which over-cap rows the harness reports rather than gates is the recipe's notes' ("Budgets").

Reference environment (``reference.in``/``reference.lock`` in this directory, installed into the reference
python): torch (the card pins 2.8.0), transformers>=4.57 (Qwen3VL), qwen-vl-utils>=0.0.14, pyyaml,
tokenizers, numpy, huggingface_hub. The render mode needs only tokenizers (plus huggingface_hub for a
``repo@revision`` tokenizer spec); the card module is imported lazily, so stage 1 runs the render mode
without torch or transformers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

#: The card script vendored verbatim beside this file; the hash pins its provenance.
CARD_SCRIPT = "qwen3_vl_embedding.py"
CARD_SCRIPT_SHA256 = "8ffa74a1a6bb759610c57865ea416fd4daf9936cb787520e1112a3e1d547f36a"

#: The shapes the render mode reports per pairs row: the card encodes queries and documents alike.
SHAPES = ("query", "document")

#: The pairs-file media columns the harness knows; a row carrying one is refused (see below).
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


def card_constants() -> dict[str, Any]:
    """The card script's own constants, read from the vendored file's source (never restated here).

    Returns:
        ``{"max_length": int, "default_instruction": str}``: the module-level ``MAX_LENGTH`` and the
        default of ``Qwen3VLEmbedder.__init__``'s ``default_instruction`` -- the cap and the system text
        the card applies to every input that carries no instruction of its own.
    """
    import ast

    tree = ast.parse((Path(__file__).resolve().parent / CARD_SCRIPT).read_text(encoding="utf-8"))
    found: dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "MAX_LENGTH" for target in node.targets
        ):
            found["max_length"] = ast.literal_eval(node.value)
        if isinstance(node, ast.ClassDef) and node.name == "Qwen3VLEmbedder":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    arguments = item.args.args[len(item.args.args) - len(item.args.defaults) :]
                    for argument, default in zip(arguments, item.args.defaults, strict=True):
                        if argument.arg == "default_instruction":
                            found["default_instruction"] = ast.literal_eval(default)
    if set(found) != {"max_length", "default_instruction"}:
        raise SystemExit(f"{CARD_SCRIPT} no longer defines MAX_LENGTH and the default instruction: {found}")
    return found


def _refuse_media_rows(pairs: list[dict[str, Any]]) -> None:
    """This reference's pairs contract is text, and a media-bearing row is refused, never silently dropped.

    The builders below take the row's text fields only, so ``{"text": ...}`` would swallow an image or
    video column and stage 2 would compare the wrong content. The recipe's ONE declared video sampling
    policy (the engine's fps rule at the checkpoint's fps 2) governs a future media wave: a container
    sampled at any other rule (the card script's fps 1 / max 64 default among them) is a different
    instrument and would be quietly missed here otherwise. Loud refusal, as for instruction rows.
    """
    for index, row in enumerate(pairs):
        carried = sorted(set(row) & set(_MEDIA_COLUMNS))
        if carried:
            raise SystemExit(
                f"pairs row {index} carries media columns {carried}, and this reference's pairs "
                "contract is text (the recipe's stages compare text): drop the media, or hold the "
                "row for the media wave under the recipe's declared video policy (the engine's fps "
                "rule at the checkpoint's fps 2)"
            )


def _refuse_instruction_rows(pairs: list[dict[str, Any]]) -> None:
    """The recipe serves the card's default instruction on both sides (its template pins it as fixed frame
    text): a row that carries its own instruction would diverge from the served frame -- loudly refused."""
    for index, row in enumerate(pairs):
        if row.get("instruction"):
            raise SystemExit(
                f"pairs row {index} carries an instruction, and this recipe serves the model's default "
                "instruction on both sides: drop the instruction field from the pairs file"
            )


def _special(name: str) -> str:
    """The literal form of one of the tokenizer's added specials, built from its name."""
    return "<|" + name + "|>"


def card_prompt(text: str, instruction: str) -> str:
    """The prompt the card's model reads for one text input, before truncation.

    The card's ``format_model_input`` builds a system turn (the instruction) and a user turn (the text, or
    the literal "NULL" when the input is empty), and ``_preprocess_inputs`` renders it through the
    checkpoint's chat template with ``add_generation_prompt=True``. For a text-only conversation that
    template (chat_template.jinja at the pinned revision) renders exactly this string; the recipe's test
    pins it against the checkpoint's file.
    """
    content = text if text else "NULL"
    return (
        _special("im_start")
        + "system\n"
        + instruction
        + _special("im_end")
        + "\n"
        + _special("im_start")
        + "user\n"
        + content
        + _special("im_end")
        + "\n"
        + _special("im_start")
        + "assistant\n"
    )


def _backend(spec: str) -> Any:
    """The ``tokenizers`` library over the spec's ``tokenizer.json`` (a local file or directory, else the
    Hub file at the spec's revision) -- the Rust tokenizer the card's processor runs."""
    from tokenizers import Tokenizer

    path = Path(spec).expanduser()
    if path.is_dir():
        path = path / "tokenizer.json"
    if not path.is_file():
        from huggingface_hub import hf_hub_download

        repo, _, revision = spec.partition("@")
        path = Path(hf_hub_download(repo, "tokenizer.json", revision=revision or None))
    return Tokenizer.from_file(str(path))


def card_truncation(prompt: str, backend: Any, max_length: int) -> str:
    """The card's truncation of one rendered prompt, as the text the model reads.

    ``_preprocess_inputs`` calls the processor with ``truncation=True, max_length=max_length``: the
    tokenizer right-cuts the prompt's ids so that they, plus the post-processor's appended endoftext, fit
    ``max_length``. The ``tokenizers`` library runs exactly that here (``enable_truncation``, the same
    call the processor's fast tokenizer makes), and the kept text is the rendered prompt up to the last
    kept token's end offset -- a verbatim prefix, never a decode. Under the cap the prompt is returned
    whole.
    """
    if len(backend.encode(prompt, add_special_tokens=True).ids) <= max_length:
        return prompt
    backend.enable_truncation(max_length=max_length, strategy="longest_first", direction="right")
    try:
        kept = backend.encode(prompt, add_special_tokens=True)
    finally:
        backend.no_truncation()
    return prompt[: max(end for _, end in kept.offsets)]


def mode_render(pairs: list[dict[str, Any]], tokenizer_spec: str) -> dict[str, Any]:
    """Stage 1's reference side: the card's prompt per row and shape, with the card's own truncation."""
    _refuse_instruction_rows(pairs)
    _refuse_media_rows(pairs)
    constants = card_constants()
    backend = _backend(tokenizer_spec)
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(pairs):
        for shape in SHAPES:
            text = str(row["query"]) if shape == "query" else str(row["documents"][0])
            prompt = card_prompt(text, constants["default_instruction"])
            rows.append(
                {"index": index, "shape": shape, "text": card_truncation(prompt, backend, constants["max_length"])}
            )
    return {"rows": rows}


def _refuse_old_media_columns(pairs: list[dict[str, Any]]) -> None:
    """The retired per-column media fields are refused loudly: the harness's media rows carry ``media``.

    The old columns (``query_image``/``documents_images`` and the video siblings) would otherwise be
    silently ignored and the row encoded as text -- a different prompt, never compared.
    """
    for index, row in enumerate(pairs):
        carried = sorted(
            key for key in ("query_image", "query_video", "documents_images", "documents_videos") if row.get(key)
        )
        if carried:
            raise SystemExit(
                f"pairs row {index} carries the retired media columns {carried}: the harness's media rows "
                "carry the `media` field; move the bytes there"
            )


def card_media_constants() -> dict[str, int]:
    """The card script's image constants, evaluated from the vendored file's module-level assignments (never
    restated): ``IMAGE_FACTOR`` (16 x 2: the patch the card hands ``process_vision_info`` as
    ``image_patch_size=16``, times the spatial merge), ``MIN_PIXELS``, ``MAX_PIXELS`` and ``MAX_FRAMES``."""
    import ast

    tree = ast.parse((Path(__file__).resolve().parent / CARD_SCRIPT).read_text(encoding="utf-8"))
    env: dict[str, Any] = {}
    names = ("IMAGE_BASE_FACTOR", "IMAGE_FACTOR", "MIN_PIXELS", "MAX_PIXELS", "MAX_FRAMES")
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in names:
                expression = ast.Expression(body=node.value)
                env[node.targets[0].id] = eval(
                    compile(expression, CARD_SCRIPT, "eval"), {"__builtins__": {}}, dict(env)
                )  # noqa: S307
    if set(env) != set(names):
        raise SystemExit(f"{CARD_SCRIPT} no longer defines {sorted(set(names) - set(env))}")
    return env


def card_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    """``qwen_vl_utils.vision_process.smart_resize`` (qwen-vl-utils 0.0.14, the card's preprocessing), as
    ``fetch_image`` calls it with the card's per-item ``min_pixels``/``max_pixels``: both edges rounded to the
    factor (at least one factor), the area floored into ``max_pixels`` or ceiled up to ``min_pixels``; an
    aspect ratio over 200 is refused."""
    import math

    if max(height, width) / min(height, width) > 200:
        raise SystemExit(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def _image_size(entry: dict[str, Any]) -> tuple[int, int]:
    """An entry's image as the card loads it (``fetch_image``: a ``data:image`` URI's bytes, or a file)."""
    import base64
    import io

    from PIL import Image

    uri = str(entry.get("uri", ""))
    if uri.startswith("data:"):
        source: Any = io.BytesIO(base64.b64decode(uri.split(",", 1)[1]))
    else:
        source = uri.removeprefix("file://")
    with Image.open(source) as handle:
        return handle.size


def declared_video_rate(policy: dict[str, Any]) -> float:
    """The recipe's declared engine rate (``client.video_policy.fps``), refused loudly when absent.

    The Qwen3-VL backend samples a container by fps and IGNORES a pinned ``num_frames``, so the recipe's
    fps is the ONE rate every side applies; without it the checkpoint's processor would sample at its own
    default rate, a different instrument.
    """
    fps = policy.get("fps")
    if fps is None:
        raise SystemExit(
            "the recipe declares no video rate (client.video_policy.fps): the reference cannot know the "
            "frames the engine shows, and the checkpoint's processor would sample at its own rate"
        )
    return float(fps)


def video_pixel_pin(recipe: dict[str, Any]) -> tuple[int, int]:
    """The recipe's whole-clip video pixel pin, ``serve.mm_processor_kwargs.videos_kwargs``.

    The checkpoint's Qwen3-VL video processor resizes a container's frames under these numbers, with the
    ceiling constraining the whole clip together; the reference hands them to the processor itself
    (:func:`video_processor_kwargs`), so a recipe that declares no pin is refused loudly (never silently
    resized by the processor's own defaults).
    """
    mm_processor_kwargs = (recipe.get("serve") or {}).get("mm_processor_kwargs") or {}
    videos = mm_processor_kwargs.get("videos_kwargs") or {}
    if "min_pixels" not in videos or "max_pixels" not in videos:
        raise SystemExit(
            f"the recipe declares no serve.mm_processor_kwargs.videos_kwargs pixel pin: {videos}; the "
            "reference's container route cannot reproduce the engine's video geometry"
        )
    return int(videos["min_pixels"]), int(videos["max_pixels"])


def video_processor_kwargs(declared_fps: float, pin: tuple[int, int]) -> dict[str, Any]:
    """The kwargs the checkpoint's own ``Qwen3VLVideoProcessor`` gets for one container.

    The engine's sampling and the recipe's pin, in the processor's own vocabulary:

    * ``do_sample_frames`` true with ``fps`` the declared rate: the processor decodes the container and
      samples ``int(total_frames / original_fps * fps)`` frames, clamped to its own 4..768 bounds and
      total, at ``np.linspace`` indices -- the rule vLLM v0.31.0's ``Qwen3VLVideoBackend``
      (``compute_frames_index_to_sample``) implements, which the engine reports as already sampled.
    * ``size`` the recipe's whole-clip pin: the video processor's clip-level budget, which is what
      ``serve.mm_processor_kwargs.videos_kwargs``'s ``min_pixels``/``max_pixels`` name (vLLM's own
      ``_get_vision_info`` folds them into ``size["shortest_edge"]``/``size["longest_edge"]``).
    * ``cap_pixels_per_frame`` false: the uncapped resize the engine's own transformers (5.17) applies;
      transformers 5.19 keeps that default and warns until v5.22, so the reference pins it explicitly.
    * ``return_metadata`` true: the processor reports the frames it sampled, which the media mode reads
      back (:func:`realised_video`).
    """
    return {
        "do_sample_frames": True,
        "fps": float(declared_fps),
        "size": {"shortest_edge": pin[0], "longest_edge": pin[1]},
        "cap_pixels_per_frame": False,
        "return_metadata": True,
    }


def load_video_processor(recipe: dict[str, Any]) -> Any:
    """The checkpoint's own ``Qwen3VLVideoProcessor`` at the recipe's pinned revision.

    Loaded from the pinned snapshot (``snapshot_download`` at the recipe's revision), so the reference
    runs the same processor class and the same ``video_preprocessor_config.json`` the engine does; a
    recipe that names no model is refused loudly.
    """
    from huggingface_hub import snapshot_download
    from transformers import Qwen3VLVideoProcessor

    model = str(recipe.get("model") or "")
    if not model:
        raise SystemExit("the resolved recipe names no model: the checkpoint's video processor cannot be loaded")
    path = snapshot_download(model, revision=str(recipe.get("revision")) or None)
    return Qwen3VLVideoProcessor.from_pretrained(path)


def realised_video(outputs: dict[str, Any], processor: Any) -> dict[str, Any]:
    """What the checkpoint's own video processor realised for one container, read from its own outputs.

    ``video_grid_thw`` is the patch grid the processor patchified the container into (temporal, height,
    width, each in patches) and ``video_metadata`` carries the frames it sampled (``frames_indices``).

    Returns:
        ``{"frames", "frame_width", "frame_height", "tokens"}``: the sampled frames, the per-frame size
        in pixels (the grid's patch multiples) and the container's merged patch tokens
        (``grid_t x grid_h x grid_w / merge_size**2``). Nothing is re-derived from the pairs entry: the
        processor's own outputs are the report. (``frame_width``/``frame_height``, not ``width``/
        ``height``: the media stage's per-item comparison reads ``width``/``height`` as the image's
        prepared size, which a container side never carries.)
    """
    grid = outputs["video_grid_thw"][0]
    grid_t, grid_h, grid_w = (int(value) for value in grid)
    patch = int(processor.patch_size)
    merge = int(processor.merge_size)
    metadata = outputs["video_metadata"][0]
    return {
        "frames": len(metadata.frames_indices),
        "frame_width": grid_w * patch,
        "frame_height": grid_h * patch,
        "tokens": grid_t * grid_h * grid_w // merge**2,
    }


def _video_suffix(payload: bytes) -> str:
    """The scratch file's suffix, from the container's own magic (RIFF AVI or ISOBMFF), never the URI's name."""
    if payload[:4] == b"RIFF":
        return ".avi"
    if payload[4:8] == b"ftyp":
        return ".mp4"
    raise SystemExit("the video bytes state neither a RIFF nor an ISOBMFF container; the decoder needs a container")


def _write_container(entry: dict[str, Any], *, work: Path, clip_name: str) -> Path:
    """One pairs video entry's inline bytes on disk, at a path the checkpoint's own decoder reads."""
    payload = _entry_bytes(entry)
    path = work / f"{clip_name}{_video_suffix(payload)}"
    path.write_bytes(payload)
    return path


def video_realisation(
    processor: Any, entry: dict[str, Any], *, work: Path, clip_name: str, declared_fps: float, pin: tuple[int, int]
) -> dict[str, Any]:
    """One container as the checkpoint's own video processor realises it: write it, call the processor, read
    the processor's own outputs back (:func:`realised_video`)."""
    path = _write_container(entry, work=work, clip_name=clip_name)
    outputs = processor(videos=[str(path)], **video_processor_kwargs(declared_fps, pin))
    return realised_video(outputs, processor)


def side_parts(text: str, entries: list[dict[str, Any]]) -> tuple[list[str], str]:
    """One side's media kinds and body text, refusing what the card's ``format_model_input`` cannot express.

    The card builds the user turn video, image, text; it takes at most one image and one video. A part
    sequence whose text stands before or between media parts, several images or videos, or an image before
    its video (the card renders the video first) is refused loudly: the engine's part order is the client's,
    and a reference that silently reordered it would compare a different prompt.

    Returns:
        ``(media_kinds, body)``: the media kinds in the card's order and the side's text (its text parts,
        in order, then its own text).
    """
    kinds = [str(entry.get("kind", "image")) for entry in entries]
    unknown = sorted({kind for kind in kinds if kind not in ("image", "video", "text")})
    if unknown:
        raise SystemExit(f"a media entry's kind must be image, video or text; got {unknown}")
    media_positions = [position for position, kind in enumerate(kinds) if kind != "text"]
    if media_positions:
        last_media = media_positions[-1]
        if any(kind == "text" for kind in kinds[:last_media]):
            raise SystemExit(
                "the side interleaves a text part before or between its media parts, and the card's "
                "format_model_input puts the text after the media: the card cannot express this input"
            )
    media = [kind for kind in kinds if kind != "text"]
    if media.count("image") > 1 or media.count("video") > 1:
        raise SystemExit("the card's format_model_input takes one image and one video per input")
    if "video" in media and "image" in media and media.index("video") > media.index("image"):
        raise SystemExit(
            "the side carries an image before its video, and the card's format_model_input renders the "
            "video first: the card cannot express this input"
        )
    body = "".join(str(entry.get("text", "")) for entry, kind in zip(entries, kinds, strict=True) if kind == "text")
    return media, body + (text or "")


def _entry_bytes(entry: dict[str, Any]) -> bytes:
    """An entry's inline bytes (the harness's ``data:`` URI), decoded."""
    import base64

    uri = str(entry.get("uri", ""))
    if not uri.startswith("data:"):
        raise SystemExit(f"the media stage sends inline media; got {uri[:48]!r}")
    return base64.b64decode(uri.split(",", 1)[1])


def _decode_image(entry: dict[str, Any]) -> Any:
    """An entry's inline image as a loaded PIL image (the card's ``fetch_image`` input)."""
    import io

    from PIL import Image

    with Image.open(io.BytesIO(_entry_bytes(entry))) as handle:
        return handle.convert("RGB")


def card_inputs(
    text: str,
    entries: list[dict[str, Any]],
    *,
    work: Path,
    clip_name: str,
    declared_fps: float | None,
) -> dict[str, Any]:
    """One side as the card's own ``Qwen3VLEmbedder.process`` input (its ``format_model_input`` keys).

    An image entry becomes a loaded PIL image; a video entry is written to a scratch file and its item
    names the recipe's declared rate (``fps``) -- the checkpoint's own video processor samples and resizes
    it (:func:`video_processor_kwargs`, applied by the reference's embedder), never the card script's
    ``qwen_vl_utils`` route. The card renders video, then image, then text (:func:`side_parts` refuses the
    forms it cannot express); an empty side is the card's own "NULL" rule (no key at all, which the card
    renders as NULL only when the side carries nothing).
    """
    _media, body = side_parts(text, entries)
    payload: dict[str, Any] = {}
    for entry in entries:
        kind = str(entry.get("kind", "image"))
        if kind == "video":
            if declared_fps is None:
                raise SystemExit(
                    "the recipe declares no video rate (client.video_policy.fps): the reference cannot know "
                    "the frames the engine shows, and the checkpoint's processor would sample at its own rate"
                )
            payload["video"] = str(_write_container(entry, work=work, clip_name=clip_name))
            payload["fps"] = float(declared_fps)
        elif kind == "image":
            payload["image"] = _decode_image(entry)
    if body:
        payload["text"] = body
    return payload


def _carries_video(pairs: list[dict[str, Any]]) -> bool:
    """Whether any pairs row carries a video entry (the media mode's container route is needed then)."""
    for row in pairs:
        media = row.get("media") or {}
        # the media field's shapes: ``query`` is a list of entries, ``documents`` a list of those lists
        for entries in [media.get("query") or [], *(media.get("documents") or [])]:
            if any(str(entry.get("kind", "image")) == "video" for entry in entries or []):
                return True
    return False


def media_side(
    text: str,
    entries: list[dict[str, Any]],
    constants: dict[str, int],
    *,
    processor: Any = None,
    pin: tuple[int, int] | None = None,
    declared_fps: float | None = None,
    work: Path | None = None,
    clip_name: str = "clip",
) -> dict[str, Any]:
    """One side as the card's model consumes it: ``format_model_input`` builds the user turn video first,
    then the image, then the text (one image and one video per input); each image is resized by
    ``fetch_image`` under the card's MIN/MAX_PIXELS and costs its merged patches ((h/32) x (w/32) image
    pads, the processor's do_resize being off) plus its vision start and end markers; each container is the
    checkpoint's own ``Qwen3VLVideoProcessor``'s realisation (:func:`video_realisation`): the frames it
    sampled, the per-frame size it resized to and its patch tokens. An interleaved text part stands where
    it stands.
    """
    side_parts(text, entries)
    factor = constants["IMAGE_FACTOR"]
    media: list[dict[str, Any]] = []
    placement: list[str] = []
    for entry in entries:
        kind = str(entry.get("kind", "image"))
        if kind == "text":
            if str(entry.get("text", "")):
                placement.append("text")
        elif kind == "video":
            if processor is None or pin is None or declared_fps is None or work is None:
                raise SystemExit(
                    "a video row needs the checkpoint's own video processor, the recipe's declared fps and "
                    "whole-clip pin, and a scratch file: the media mode's container route reports the "
                    "processor's own realisation, never a re-derived one"
                )
            placement.append("video")
            media.append(
                {
                    "kind": "video",
                    **video_realisation(
                        processor,
                        entry,
                        work=work,
                        clip_name=clip_name,
                        declared_fps=declared_fps,
                        pin=pin,
                    ),
                }
            )
        else:
            width, height = _image_size(entry)
            resized_h, resized_w = card_resize(height, width, factor, constants["MIN_PIXELS"], constants["MAX_PIXELS"])
            placement.append("image")
            tokens = (resized_h // factor) * (resized_w // factor) + 2
            media.append({"kind": "image", "width": resized_w, "height": resized_h, "tokens": tokens})
    if text:
        placement.append("text")
    return {"placement": placement, "media": media}


def mode_media(pairs: list[dict[str, Any]], recipe: dict[str, Any] | None = None) -> dict[str, Any]:
    """The media stage's reference side: per row and side that carries media, what the card's model consumes
    (:func:`media_side`). A container's facts are the checkpoint's own ``Qwen3VLVideoProcessor``'s
    (:func:`video_realisation`), loaded from the recipe's pinned snapshot only when a row carries one."""
    _refuse_old_media_columns(pairs)
    constants = card_media_constants()
    has_video = _carries_video(pairs)
    policy = ((recipe or {}).get("client") or {}).get("video_policy") or {}
    pin = video_pixel_pin(recipe or {}) if has_video else None
    declared_fps = declared_video_rate(policy) if has_video else None
    processor = load_video_processor(recipe or {}) if has_video else None
    rows: list[dict[str, Any]] = []
    # ignore_cleanup_errors: a network-backed tempdir can turn an entry visible after the cleanup's scan;
    # a scratch cleanup race must never fail a reference run (the harness's own tempdirs say the same).
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as work:
        root = Path(work)
        for index, row in enumerate(pairs):
            media = row.get("media") or {}
            if media.get("query"):
                rows.append(
                    {
                        "index": index,
                        "side": "query",
                        **media_side(
                            str(row["query"]),
                            media["query"],
                            constants,
                            processor=processor,
                            pin=pin,
                            declared_fps=declared_fps,
                            work=root,
                            clip_name=f"query-{index}",
                        ),
                    }
                )
            for position, entries in enumerate(media.get("documents") or []):
                if entries:
                    side = media_side(
                        str(row["documents"][position]),
                        entries,
                        constants,
                        processor=processor,
                        pin=pin,
                        declared_fps=declared_fps,
                        work=root,
                        clip_name=f"document-{index}-{position}",
                    )
                    rows.append({"index": index, "side": f"document {position}", **side})
    return {"rows": rows}


def _load_recipe() -> dict[str, Any]:
    """The resolved recipe the harness passed (the model and revision the embed mode loads)."""
    import sys

    recipe_file = next((sys.argv[i + 1] for i, arg in enumerate(sys.argv) if arg == "--recipe"), None)
    if recipe_file is None:
        raise SystemExit("--recipe is required: the harness passes the resolved recipe JSON")
    return json.loads(Path(recipe_file).read_text(encoding="utf-8"))


def _conversation_video(conversation: list[dict[str, Any]]) -> str | None:
    """A conversation's container path (the card's ``file://`` form stripped), or ``None`` without a video."""
    for message in conversation:
        for item in message.get("content") or []:
            if item.get("type") != "video":
                continue
            video = item.get("video")
            if not isinstance(video, str):
                raise SystemExit("the reference's container route needs the container's path, not a frame list")
            return video.removeprefix("file://")
    return None


def _without_video(conversation: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A conversation with its video items dropped, for the card's own image route."""
    return [
        {**message, "content": [item for item in message.get("content") or [] if item.get("type") != "video"]}
        for message in conversation
    ]


def video_processor_inputs(
    processor: Any,
    conversations: list[list[dict[str, Any]]],
    paths: list[str],
    *,
    declared_fps: float,
    pin: tuple[int, int],
    max_length: int,
) -> dict[str, Any]:
    """The checkpoint processor's call for conversations that carry a container.

    The card's text render and its image route stay exactly as the card's ``_preprocess_inputs`` has them:
    ``apply_chat_template`` with the generation prompt, and ``process_vision_info`` under the card's pixel
    bounds, whose images reach the processor with ``do_resize`` off. The containers go to the checkpoint's
    own video processor instead (:func:`video_processor_kwargs`), which decodes, samples at the recipe's
    declared rate and resizes under the recipe's whole-clip pin -- the engine's own realisation.
    """
    from qwen_vl_utils.vision_process import process_vision_info

    text = processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
    images, _ = process_vision_info(
        [_without_video(conversation) for conversation in conversations], image_patch_size=16
    )
    return processor(
        text=text,
        images=images,
        videos=paths,
        images_kwargs={"do_resize": False},  # the card's images are already resized by qwen-vl-utils
        videos_kwargs={
            **video_processor_kwargs(declared_fps, pin),
            "do_resize": True,  # the checkpoint's processor resizes the container under the pin
        },
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )


def mode_embed(recipe: dict[str, Any], pairs: list[dict[str, Any]], device: str) -> dict[str, Any]:
    """Stage 2's reference side: the card's own embeddings — chat template + processor, last-token
    pooling, L2 — one vector per side, queries and documents alike (the card encodes both sides alike).

    The card's ``process`` is called once per input (:func:`_one_vector_per_input`): the batched call
    dies with ``StopIteration`` when any text in the batch carries the tokenizer's literal multimodal
    placeholder spellings (see the helper's docstring), and the card's per-sequence pooling makes the
    per-input calls answer the same vectors.

    A media row's ``media`` field rides the card's own input keys (:func:`card_inputs`): the inline image
    bytes as a PIL image, a container as the file the checkpoint's own ``Qwen3VLVideoProcessor`` reads at
    the recipe's declared fps and under its whole-clip pixel pin (:func:`video_processor_kwargs`) -- the
    engine's own sampling and resize. The card renders video, image, text; a side it cannot express is
    refused loudly.
    """
    _refuse_old_media_columns(pairs)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from huggingface_hub import snapshot_download
    from qwen3_vl_embedding import Qwen3VLEmbedder

    pin = video_pixel_pin(recipe)
    policy = (recipe.get("client") or {}).get("video_policy") or {}
    declared_fps = policy.get("fps")

    class _ProcessorVideoEmbedder(Qwen3VLEmbedder):
        """The card's embedder with the container route on the checkpoint's own Qwen3VLVideoProcessor.

        The card's ``_preprocess_inputs`` hands every video to ``qwen_vl_utils`` (its own loader, its own
        per-frame pixel bounds, which cannot express the recipe's whole-clip pin); the override keeps the
        card's text render and its image route -- the same ``apply_chat_template`` call and the same
        ``process_vision_info`` images under the card's pixel bounds -- and sends a container through the
        checkpoint's own video processor instead (:func:`video_processor_kwargs`: the recipe's declared
        fps and whole-clip pin), so the reference realises the engine's frames and per-frame pixels."""

        def __init__(self, *args: Any, video_fps: float | None, video_pin: tuple[int, int], **kwargs: Any) -> None:
            self._video_fps = video_fps
            self._video_pin = video_pin
            super().__init__(*args, **kwargs)

        def _preprocess_inputs(self, conversations: list[list[dict[str, Any]]]) -> dict[str, Any]:
            paths = [path for path in (_conversation_video(conversation) for conversation in conversations) if path]
            if not paths:
                return super()._preprocess_inputs(conversations)
            if self._video_fps is None:
                raise SystemExit(
                    "the recipe declares no video rate (client.video_policy.fps): the checkpoint's "
                    "processor would sample the container at its own rate, a different instrument"
                )
            return video_processor_inputs(
                self.processor,
                conversations,
                paths,
                declared_fps=self._video_fps,
                pin=self._video_pin,
                max_length=self.max_length,
            )

    path = snapshot_download(str(recipe.get("model")), revision=str(recipe.get("revision")) or None)
    model = _ProcessorVideoEmbedder(model_name_or_path=str(path), video_fps=declared_fps, video_pin=pin)
    if device and device != "auto":
        model.model = model.model.to(device)

    _refuse_instruction_rows(pairs)  # the card then applies its own default: the pinned frame text

    # ignore_cleanup_errors: a network-backed tempdir can turn an entry visible after the cleanup's scan;
    # a scratch cleanup race must never fail a reference run (the harness's own tempdirs say the same).
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as work:
        root = Path(work)
        query_inputs = [
            card_inputs(
                str(row["query"]),
                list((row.get("media") or {}).get("query") or []),
                work=root,
                clip_name=f"query-{index}",
                declared_fps=declared_fps,
            )
            for index, row in enumerate(pairs)
        ]
        document_inputs: list[dict[str, Any]] = []
        for index, row in enumerate(pairs):
            documents_media = list((row.get("media") or {}).get("documents") or [])
            for position, document in enumerate(row["documents"]):
                entries = documents_media[position] if position < len(documents_media) else []
                document_inputs.append(
                    card_inputs(
                        str(document),
                        list(entries or []),
                        work=root,
                        clip_name=f"document-{index}-{position}",
                        declared_fps=declared_fps,
                    )
                )
        query_vectors = _one_vector_per_input(model, query_inputs)
        document_vectors = _one_vector_per_input(model, document_inputs)
    rows: list[dict[str, Any]] = []
    cursor = 0
    for index, row in enumerate(pairs):
        width = len(row["documents"])
        documents = document_vectors[cursor : cursor + width]
        cursor += width
        rows.append(
            {
                "index": index,
                "query_vectors": [[float(value) for value in query_vectors[index]]],
                "document_vectors": [[float(value) for value in document] for document in documents],
            }
        )
    return {"rows": rows}


def _one_vector_per_input(model: Any, inputs: list[dict[str, Any]]) -> list[Any]:
    """One vector per input: the card's ``process`` called once per input, never over the whole batch.

    The card batches every input into one processor call. transformers' processor replaces every
    multimodal placeholder spelling found in the batch's texts from ONE shared iterator per modality
    (``processing_utils.get_text_with_replacements``: ``next(replacements_iters[mm_type])``), so a text
    row carrying the tokenizer's literal image/video placeholder spellings -- the pairs'
    special-token-spellings stratum, which the served engine tokenizes as plain special ids -- consumes
    the batch's media replacements and the whole call dies with StopIteration (measured on the wave's
    pairs against the checkpoint's own processor: the query batch, 38 inputs with one image row, raises;
    each input on its own answers; transformers 5.10.4 carries the same shared iterator). The card pools
    per sequence and its batch is padding-only, so one call per input is the same vector per input
    without the cross-row contamination.
    """
    return [model.process([item], normalize=True)[0] for item in inputs]


def main() -> int:
    """The subprocess CLI: verify the vendored card script's hash, run the mode, write the JSON."""
    parser = argparse.ArgumentParser(
        description="the qwen3-vl-embedding family reference (the card's Qwen3VLEmbedder path)"
    )
    parser.add_argument("--mode", required=True, choices=["render", "embed", "media"])
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument(
        "--recipe",
        required=True,
        help="the resolved recipe JSON the harness passed (the variant's id, model and revision)",
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    vendored = Path(__file__).resolve().parent / CARD_SCRIPT
    digest = hashlib.sha256(vendored.read_bytes()).hexdigest()
    if digest != CARD_SCRIPT_SHA256:
        raise SystemExit(f"{CARD_SCRIPT} is not the pinned card script: sha256 {digest} != {CARD_SCRIPT_SHA256}")

    pairs = [json.loads(line) for line in Path(args.pairs).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.mode == "render":
        output = mode_render(pairs, args.tokenizer)
    elif args.mode == "media":
        output = mode_media(pairs, _load_recipe())
    else:
        output = mode_embed(_load_recipe(), pairs, args.device)
    Path(args.out).write_text(json.dumps(output) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
