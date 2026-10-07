"""The reference implementation for Qwen/Qwen3-VL-Embedding-2B: the equivalence harness's subprocess reference.

Runs in its own reference environment, never inside the harness process:

    <reference-python> reference.py --mode render|embed --pairs <file> --out <file> \
        --tokenizer <repo>@<revision>|<path/to/tokenizer.json> [--device cpu|cuda:0]

Both modes follow the model card's own code path, ``scripts/qwen3_vl_embedding.py`` (``Qwen3VLEmbedder``,
transformers), vendored VERBATIM beside this file as ``qwen3_vl_embedding.py``:
https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B/blob/9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda/scripts/qwen3_vl_embedding.py,
sha256 8ffa74a1a6bb759610c57865ea416fd4daf9936cb787520e1112a3e1d547f36a (pinned here and by the recipe's
test). The card's constants this file needs (``MAX_LENGTH`` and the default instruction) are read from the
vendored script's source, never restated.

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
  -- one L2-normalised 2048-d vector per text, through ``Qwen3VLEmbedder.process``. The checkpoint is
  resolved with ``huggingface_hub.snapshot_download`` at the recipe's revision, so model and processor load
  the same pinned snapshot.

Media is not exercised here (the media checks are token-level CPU checks and the GPU wave's probe-image
check): a pairs row carrying media columns is refused loudly (see :func:`_refuse_media_rows`).
The recipe's ONE video sampling policy (64 uniformly spaced frames per clip) governs a future media wave:
a video row would arrive as its 64 pre-sampled frames and pass through the card's frame-list route at
``num_segments`` 64 unchanged. Which over-cap rows the harness reports rather than gates is the
recipe's notes' ("Budgets"); the pairs files keep every text within the budget minus the frame.

Reference environment (``requirements-reference.txt`` in this directory, installed into the reference
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
    policy (64 uniformly spaced frames per clip) governs a future media wave: a container sampled at any
    other rule (the card's fps 1 / max 64 default among them) is a different instrument and would be
    quietly missed here otherwise. Loud refusal, as for instruction rows.
    """
    for index, row in enumerate(pairs):
        carried = sorted(set(row) & set(_MEDIA_COLUMNS))
        if carried:
            raise SystemExit(
                f"pairs row {index} carries media columns {carried}, and this reference's pairs "
                "contract is text (the recipe's stages compare text): drop the media, or hold the "
                "row for the media wave under the recipe's declared video policy (64 uniformly "
                "spaced frames per clip, pre-extracted at ingest)"
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


def _load_recipe() -> dict[str, Any]:
    """This recipe's YAML (the model and revision the embed mode loads), read from beside this file."""
    import yaml

    return yaml.safe_load((Path(__file__).resolve().parent / "recipe.yaml").read_text(encoding="utf-8"))


def mode_embed(recipe: dict[str, Any], pairs: list[dict[str, Any]], device: str) -> dict[str, Any]:
    """Stage 2's reference side: the card's own embeddings — chat template + processor, last-token
    pooling, L2 — one vector per text, queries and documents alike (the card encodes both sides alike)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from huggingface_hub import snapshot_download
    from qwen3_vl_embedding import Qwen3VLEmbedder

    path = snapshot_download(str(recipe.get("model")), revision=str(recipe.get("revision")) or None)
    model = Qwen3VLEmbedder(model_name_or_path=str(path))
    if device and device != "auto":
        model.model = model.model.to(device)

    _refuse_instruction_rows(pairs)  # the card then applies its own default: the pinned frame text
    _refuse_media_rows(pairs)
    query_inputs = [{"text": str(row["query"])} for row in pairs]
    document_inputs = [{"text": str(document)} for row in pairs for document in row["documents"]]
    import numpy as np

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
        description="the qwen3-vl-embedding-2b reference (the card's Qwen3VLEmbedder path)"
    )
    parser.add_argument("--mode", required=True, choices=["render", "embed"])
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    vendored = Path(__file__).resolve().parent / CARD_SCRIPT
    digest = hashlib.sha256(vendored.read_bytes()).hexdigest()
    if digest != CARD_SCRIPT_SHA256:
        raise SystemExit(f"{CARD_SCRIPT} is not the pinned card script: sha256 {digest} != {CARD_SCRIPT_SHA256}")

    pairs = [json.loads(line) for line in Path(args.pairs).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.mode == "render":
        output = mode_render(pairs, args.tokenizer)
    else:
        output = mode_embed(_load_recipe(), pairs, args.device)
    Path(args.out).write_text(json.dumps(output) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
