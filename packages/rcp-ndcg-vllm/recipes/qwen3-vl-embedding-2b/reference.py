"""The reference implementation for Qwen/Qwen3-VL-Embedding-2B: the equivalence harness's subprocess reference.

Runs in its own reference environment, never inside the harness process:

    <reference-python> reference.py --mode render|embed --pairs <file> --out <file> \
        --tokenizer <repo>@<revision>|<path/to/tokenizer.json> [--device cpu|cuda:0]

- ``--mode render`` (stage 1's reference side): ``{"rows": [{"index", "shape", "text"}]}`` — the
  anchor-preserving render per pairs-file row and per shape this recipe's template declares: the frame
  assembled with every fixed token reserved, the content span cut at a token boundary of the assembled
  render, the frame re-attached. The card's own truncation (processor ``truncation=True`` at
  MAX_LENGTH=8192) right-cuts the whole prompt and drops that anchor on over-cap inputs — the recipe's
  ``anchor_drop_over_cap`` known deviation — so the render mode mirrors the product's token-prefix cut
  instead. The mirror is verified byte-identical with the product's ``fit``, under and over the cap, in
  ``tests/recipes/test_qwen3_vl_embedding_2b.py``.
- ``--mode embed`` (stage 2's reference side): ``{"rows": [{"index", "query_vectors", "document_vectors"}]}``
  — one L2-normalised 2048-d vector per text, through the model's own published code path: the model
  card's ``scripts/qwen3_vl_embedding.py`` (``Qwen3VLEmbedder``, transformers), vendored VERBATIM beside
  this file as ``qwen3_vl_embedding.py``:
  https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B/blob/9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda/scripts/qwen3_vl_embedding.py,
  sha256 8ffa74a1a6bb759610c57865ea416fd4daf9936cb787520e1112a3e1d547f36a (pinned here and by the
  recipe's test). The checkpoint is resolved with ``huggingface_hub.snapshot_download`` at the recipe's
  revision, so model and processor load the same pinned snapshot. Media equivalence is not exercised
  here: the pairs file carries text (the media checks are the research's token-level checks and the GPU
  wave's probe-image check).

Reference environment (``requirements-reference.txt`` in this directory, installed into the reference
python): torch (the card pins 2.8.0), transformers>=4.57 (Qwen3VL), qwen-vl-utils>=0.0.14, pyyaml,
tokenizers, numpy, huggingface_hub. The render mode needs only pyyaml, tokenizers and numpy; the card
module is imported lazily, so stage 1 runs the render mode without torch or transformers.
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

#: The card's built-in instruction (``format_model_input``), pinned by the recipe's template as fixed
#: frame text (see the recipe's notes); the reference refuses pairs rows that carry one of their own.
DEFAULT_INSTRUCTION = "Represent the user's input."

_SHAPES = ("query", "document", "pair")


def _refuse_instruction_rows(pairs: list[dict[str, Any]]) -> None:
    """This recipe's template pins the model's default instruction as fixed frame text (see the recipe's
    notes): a row that carries its own instruction would diverge from the served frame — loudly refused."""
    for index, row in enumerate(pairs):
        if row.get("instruction"):
            raise SystemExit(
                f"pairs row {index} carries an instruction, and this recipe's template pins the model's "
                "default instruction as fixed frame text: drop the instruction field from the pairs file"
            )


class _Tok:
    """The tokenizer the render mode counts and cuts with: the ``tokenizers`` library over the spec's
    ``tokenizer.json`` (a local file or directory, else the Hub file at the spec's revision)."""

    def __init__(self, spec: str) -> None:
        from tokenizers import Tokenizer

        path = Path(spec).expanduser()
        if path.is_dir():
            path = path / "tokenizer.json"
        if not path.is_file():
            from huggingface_hub import hf_hub_download

            repo, _, revision = spec.partition("@")
            path = Path(hf_hub_download(repo, "tokenizer.json", revision=revision or None))
        self.backend = Tokenizer.from_file(str(path))
        self.name = spec
        self._added: dict[str, str] = {}
        for token in self.backend.get_added_tokens_decoder().values():
            content = token.content
            wrapped = content.startswith("<|") and content.endswith("|>")
            name = content[2:-2] if wrapped else content
            self._added.setdefault(name, content)
            self._added.setdefault(content, content)

    def ids(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        """The token ids of ``text``, optionally with the post-processor's (as the engine reads them)."""
        return list(self.backend.encode(text, add_special_tokens=add_special_tokens).ids)

    def count(self, text: str, *, add_special_tokens: bool = False) -> int:
        """The number of tokens of ``text``; with ``add_special_tokens``, as the engine counts it."""
        return len(self.backend.encode(text, add_special_tokens=add_special_tokens).ids)

    def offsets(self, text: str) -> list[tuple[int, int]]:
        """``(start, end)`` character offsets in ``text`` of each of its tokens, in order."""
        return [tuple(offset) for offset in self.backend.encode(text, add_special_tokens=False).offsets]  # type: ignore[misc]

    def special_text(self, name: str) -> str:
        """The literal text of the added token named ``name`` (the bare name or the wrapped form)."""
        try:
            return self._added[name]
        except KeyError as error:
            known = sorted({key for key in self._added if not key.startswith("<|")})
            raise SystemExit(f"the tokenizer has no special token named {name!r} (its names: {known})") from error


def _load_recipe() -> dict[str, Any]:
    """This recipe's YAML, read from the recipe.yaml beside this file (the fixture references read
    theirs the same way). Only the anchor-preserving ``cut`` policy is implemented here, and the frame
    must still pin the card's default instruction (the drift check keeps the three copies tied)."""
    import yaml

    recipe: dict[str, Any] = yaml.safe_load(
        (Path(__file__).resolve().parent / "recipe.yaml").read_text(encoding="utf-8")
    )
    client = recipe.get("client") or {}
    if client.get("on_overflow") != "cut" or client.get("max_tokens") is None:
        raise SystemExit(
            "reference.py implements on_overflow: cut with a max_tokens budget; the recipe declares something else"
        )
    head = ((client.get("template") or {}).get("document") or [{}])[0].get("fixed") or ""
    if DEFAULT_INSTRUCTION not in head:
        raise SystemExit(
            "the recipe's declared frame no longer pins the card's default instruction "
            f"({DEFAULT_INSTRUCTION!r}) as fixed text; the reference refuses to drift from it"
        )
    return recipe


def _resolve_fixed(text: str, tokenizer: _Tok) -> str:
    """A fixed segment's text with every ``{special:name}`` placeholder resolved to the tokenizer's literal."""
    while True:
        start = text.find("{special:")
        if start < 0:
            return text
        end = text.find("}", start)
        if end < 0:
            raise SystemExit(f"a template fixed segment names an unterminated special: {text!r}")
        name = text[start + len("{special:") : end].strip()
        text = text[:start] + tokenizer.special_text(name) + text[end + 1 :]


def _assemble(segments: list[dict[str, Any]], tokenizer: _Tok, *, content: str, cuttable: str) -> str:
    """The full rendered request: fixed segments with their specials resolved (the model's default
    instruction is part of the fixed frame), the cuttable span filled with ``content``."""
    parts: list[str] = []
    for segment in segments:
        if segment.get("fixed") is not None:
            parts.append(_resolve_fixed(str(segment["fixed"]), tokenizer))
            continue
        if str(segment.get("content")) == cuttable:
            parts.append(content)
        # the shape's other content span renders empty (this recipe declares one cuttable span)
    return "".join(parts)


def _token_prefix(text: str, max_tokens: int, tokenizer: _Tok, *, assemble: Any) -> str:
    """The longest token-boundary prefix of ``text`` whose assembled render counts at most ``max_tokens``
    tokens as the engine reads them (the post-processor included): the product's
    ``rcp_ndcg.data.preprocess.token_prefix`` — offsets on the original text, candidates counted as
    assembled renders, a galloping then binary step-back. The mirror the recipe's test pins."""
    offsets = tokenizer.offsets(text)

    def prefix(tokens: int) -> str:
        return text[: offsets[tokens - 1][1]] if tokens > 0 else ""

    def fits(tokens: int) -> bool:
        return tokenizer.count(assemble(prefix(tokens)), add_special_tokens=True) <= max_tokens

    if fits(min(max_tokens, len(offsets))):
        return prefix(min(max_tokens, len(offsets)))
    over, fitting, step = min(max_tokens, len(offsets)), min(max_tokens, len(offsets)) - 1, 1
    while fitting > 0 and not fits(fitting):
        over, fitting, step = fitting, max(fitting - step, 0), step * 2
    while over - fitting > 1:
        middle = (over + fitting) // 2
        fitting, over = (middle, over) if fits(middle) else (fitting, middle)
    return prefix(fitting)


def _render_shape(segments: list[dict[str, Any]], tokenizer: _Tok, *, shape: str, content: str, max_tokens: int) -> str:
    """The anchor-preserving render of one shape: the content cut so the assembled render (counted with
    the shape's post-processor, as the engine reads it) fits ``max_tokens``, the frame re-attached."""
    cuttable = "query" if shape == "query" else "document"

    def assemble(piece: str) -> str:
        return _assemble(segments, tokenizer, content=piece, cuttable=cuttable)

    if tokenizer.count(assemble(content), add_special_tokens=True) > max_tokens:
        content = _token_prefix(content, max_tokens, tokenizer, assemble=assemble)
    return assemble(content)


def _declared_shapes(recipe: dict[str, Any]) -> list[str]:
    """The request shapes the recipe's template declares, in canonical order."""
    template = (recipe.get("client") or {}).get("template") or {}
    shapes = [shape for shape in _SHAPES if template.get(shape)]
    if not shapes:
        raise SystemExit("the recipe's template declares no shapes; the reference cannot render")
    return shapes


def mode_render(recipe: dict[str, Any], pairs: list[dict[str, Any]], tokenizer: _Tok) -> dict[str, Any]:
    """Stage 1's reference side: the anchor-preserving render per row, per declared shape. The frame is
    the recipe's declared one, with the model's default instruction pinned as fixed text; the pairs file
    carries no instruction field."""
    _refuse_instruction_rows(pairs)
    client = recipe.get("client") or {}
    template = client.get("template") or {}
    max_tokens = int(client["max_tokens"])
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(pairs):
        for shape in _declared_shapes(recipe):
            segments = list(template.get(shape) or [])
            text = str(row["query"]) if shape == "query" else str(row["documents"][0])
            rendered = _render_shape(segments, tokenizer, shape=shape, content=text, max_tokens=max_tokens)
            rows.append({"index": index, "shape": shape, "text": rendered})
    return {"rows": rows}


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

    recipe = _load_recipe()
    pairs = [json.loads(line) for line in Path(args.pairs).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.mode == "render":
        output = mode_render(recipe, pairs, _Tok(args.tokenizer))
    else:
        output = mode_embed(recipe, pairs, args.device)
    Path(args.out).write_text(json.dumps(output) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
