"""The reference implementation for the ``qwen3-vl-embedding`` family: the equivalence harness's subprocess reference.

Runs in its own reference environment, never inside the harness process:

    <reference-python> reference.py --mode render|embed|media --pairs <file> --out <file> \
        --tokenizer <repo>@<revision>|<path/to/tokenizer.json> [--device cpu|cuda:0] --recipe <resolved-recipe.json>

ONE reference for every size (decision 34): the embed mode loads the model and revision the resolved
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
  ``Qwen3VLEmbedder.process``. The checkpoint is resolved with ``huggingface_hub.snapshot_download`` at the
  revision the resolved recipe names, so model and processor load the same pinned snapshot. A pairs row's
  ``media`` field is read (:func:`card_inputs`): an image entry becomes a loaded PIL image, a video entry is
  written to a scratch file and handed to the card's own loader at the recipe's declared fps
  (``client.video_policy.fps``) capped at the engine's realised frame count (:func:`realised_video_frames`),
  the text parts join the side's text, and the card's order (video, image, text) is the only order it can
  express -- an interleaved part sequence, several images/videos or an image before its video is refused
  loudly by :func:`side_parts`, never silently reordered.

- ``--mode media`` (the media stage's reference side): for every pairs row carrying ``media``, per side, what
  the card's model consumes -- the user turn's parts in the card's order (video, image, text; an interleaved
  text part where it stands), each image's size after the card's ``fetch_image`` resize (qwen-vl-utils'
  ``smart_resize`` under the card's MIN/MAX_PIXELS, read from the vendored script) and its tokens (merged
  patches plus the two vision markers). Needs PIL only. The render mode compares text rows (a media column
  there is refused loudly, :func:`_refuse_media_rows`). The recipe's ONE video sampling policy is the engine's
  fps rule (fps 2, the checkpoint video processor's rate): the card reads the container through its own
  loader with ``fps``/``max_frames``, and the media mode reports the realised frame count from each pairs
  entry's own frame count and rate.

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


def realised_video_frames(entry: dict[str, Any], declared_fps: float | None, constants: dict[str, int]) -> int:
    """The frames the served engine shows for one pairs video entry.

    vLLM v0.31.0's ``Qwen3VLVideoBackend.compute_frames_index_to_sample`` (multimodal/video.py:360-400)
    samples ``int(total_frames / original_fps * fps)`` frames, clamped to ``[min_frames=4, max_frames=768,
    total_frames]``; the Qwen3-VL backend IGNORES ``num_frames``. The recipe declares the engine's rate
    (``client.video_policy.fps``), and the pairs entry records the clip's own frame count and rate, so the
    realised count is computable here. Without a declared rate the card's frame-list route applies
    (``MAX_FRAMES`` segments).
    """
    total = entry.get("num_frames")
    original = entry.get("fps")
    if declared_fps is None or not total or not original:
        return constants["MAX_FRAMES"]
    target = min(float(declared_fps), 30.0)
    frames = int(int(total) / float(original) * target)
    return min(max(frames, 4), 768, int(total))


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
    constants: dict[str, int],
) -> dict[str, Any]:
    """One side as the card's own ``Qwen3VLEmbedder.process`` input (its ``format_model_input`` keys).

    An image entry becomes a loaded PIL image; a video entry is written to ``work/<clip_name>.avi`` and the
    card's container loader samples it at ``declared_fps`` (the recipe's engine pin) capped at the engine's
    realised frame count. The card renders video, then image, then text (:func:`side_parts` refuses the
    forms it cannot express); an empty side is the card's own "NULL" rule (no key at all, which the card
    renders as NULL only when the side carries nothing).
    """
    _media, body = side_parts(text, entries)
    payload: dict[str, Any] = {}
    for entry in entries:
        kind = str(entry.get("kind", "image"))
        if kind == "video":
            path = work / f"{clip_name}.avi"
            path.write_bytes(_entry_bytes(entry))
            payload["video"] = str(path)
            payload["max_frames"] = realised_video_frames(entry, declared_fps, constants)
            if declared_fps is not None:
                payload["fps"] = float(declared_fps)
        elif kind == "image":
            payload["image"] = _decode_image(entry)
    if body:
        payload["text"] = body
    return payload


def media_side(
    text: str, entries: list[dict[str, Any]], constants: dict[str, int], declared_fps: float | None = None
) -> dict[str, Any]:
    """One side as the card's model consumes it: ``format_model_input`` builds the user turn video first, then
    the image, then the text (one image and one video per input); each image is resized by ``fetch_image``
    under the card's MIN/MAX_PIXELS and costs its merged patches ((h/32) x (w/32) image pads, the
    processor's do_resize being off) plus its vision start and end markers; a video is the engine's own
    fps sample (:func:`realised_video_frames`; the card's container route samples it the same way) -- its
    tokens are the processor's and are not counted here. An interleaved text part stands where it stands.
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
            placement.append("video")
            media.append(
                {"kind": "video", "frames": realised_video_frames(entry, declared_fps, constants), "tokens": None}
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
    (:func:`media_side`), with a video's realised frame count following the engine's declared fps rule
    (:func:`realised_video_frames`)."""
    constants = card_media_constants()
    policy = (recipe or {}).get("client", {}).get("video_policy") or {}
    declared_fps = policy.get("fps")
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(pairs):
        media = row.get("media") or {}
        if media.get("query"):
            rows.append(
                {
                    "index": index,
                    "side": "query",
                    **media_side(str(row["query"]), media["query"], constants, declared_fps),
                }
            )
        for position, entries in enumerate(media.get("documents") or []):
            if entries:
                side = media_side(str(row["documents"][position]), entries, constants, declared_fps)
                rows.append({"index": index, "side": f"document {position}", **side})
    return {"rows": rows}


def _load_recipe() -> dict[str, Any]:
    """The resolved recipe the harness passed (the model and revision the embed mode loads)."""
    import sys

    recipe_file = next((sys.argv[i + 1] for i, arg in enumerate(sys.argv) if arg == "--recipe"), None)
    if recipe_file is None:
        raise SystemExit("--recipe is required: the harness passes the resolved recipe JSON")
    return json.loads(Path(recipe_file).read_text(encoding="utf-8"))


def mode_embed(recipe: dict[str, Any], pairs: list[dict[str, Any]], device: str) -> dict[str, Any]:
    """Stage 2's reference side: the card's own embeddings — chat template + processor, last-token
    pooling, L2 — one vector per side, queries and documents alike (the card encodes both sides alike).

    A media row's ``media`` field rides the card's own input keys (:func:`card_inputs`): the inline image
    bytes as a PIL image, a container through the card's loader at the recipe's declared fps. The card
    renders video, image, text; a side it cannot express is refused loudly.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from huggingface_hub import snapshot_download
    from qwen3_vl_embedding import Qwen3VLEmbedder

    path = snapshot_download(str(recipe.get("model")), revision=str(recipe.get("revision")) or None)
    model = Qwen3VLEmbedder(model_name_or_path=str(path))
    if device and device != "auto":
        model.model = model.model.to(device)

    _refuse_instruction_rows(pairs)  # the card then applies its own default: the pinned frame text
    constants = card_media_constants()
    policy = (recipe.get("client") or {}).get("video_policy") or {}
    declared_fps = policy.get("fps")
    import numpy as np

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
                constants=constants,
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
                        constants=constants,
                    )
                )
        query_matrix = model.process(query_inputs, normalize=True)
        document_matrix = (
            model.process(document_inputs, normalize=True) if document_inputs else np.zeros((0, 1), dtype=np.float32)
        )
    rows: list[dict[str, Any]] = []
    cursor = 0
    for index, row in enumerate(pairs):
        width = len(row["documents"])
        document_vectors = [[float(value) for value in vector] for vector in document_matrix[cursor : cursor + width]]
        cursor += width
        rows.append(
            {
                "index": index,
                "query_vectors": [[float(value) for value in query_matrix[index]]],
                "document_vectors": document_vectors,
            }
        )
    return {"rows": rows}


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
