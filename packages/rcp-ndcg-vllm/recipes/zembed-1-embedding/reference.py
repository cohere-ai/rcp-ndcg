"""The reference implementation for the zembed-1-embedding recipe (zeroentropy/zembed-1-embedding).

Code path (the model's own published usage, the model card's snippet): sentence-transformers
``SentenceTransformer`` with ``trust_remote_code=True``, which loads the checkpoint's remote module
(``modeling_zembed.ZembedTransformer``, a ``sentence_transformers.models.Transformer`` subclass)
over transformers' Qwen3 backbone. The pipeline is: sentence-transformers prepends the role prompt
(``config_sentence_transformers.json``), the remote ``tokenize`` appends the suffix (the im_end
marker plus a newline; the token whose hidden state the last-token pooler reads) and right-truncates
the whole prompt at ``max_seq_length`` (``sentence_bert_config.json``: 32768), ``Pooling`` pools the
last token, ``Normalize`` L2-normalizes.

The reference is the model's published path verbatim and never ports the client's cut: it encodes
the raw texts and lets the remote tokenize's whole-prompt right cut do its own work at
``max_seq_length``.  That cut drops the pooled suffix token on over-cap inputs (the anchor defect the
served path must never have), while the SERVED side keeps the anchor by the recipe's declared client
cut (``on_overflow: cut``, content only, frame re-attached).  The reference's over-cap cut therefore
differs from the client's AND drops the anchor, so the recipe declares
``reference.known_deviations: [anchor_drop_over_cap]``: over-cap rows ride the non-gating table and
only under-cap rows gate.

Every constant here is read from the checkpoint's own files at run time (bound at
startup, never transcribed), and the suffix is checked against the literal the remote module
appends whenever ``modeling_zembed.py`` is resolvable beside the config.

Reference environment (its own python, never the harness's process): torch>=2.0,
transformers>=4.40, numpy, and sentence-transformers>=5.1,<5.2 (5.1.x measured). sentence-transformers
6.x must not be used: its preprocess-first pipeline bypasses tokenize-only remote modules and
silently drops the suffix. ``--mode render`` is string work over the checkpoint's config files
(stdlib; ``huggingface_hub`` for a Hub spec); ``--mode embed`` downloads the ~8 GB checkpoint and
wants a GPU (the wave passes ``--device``).

CLI (the harness's subprocess contract, enforced by
``rcp_ndcg_vllm.equivalence.reference.run_reference``)::

    reference.py --mode <render|embed> --pairs <file> --out <file> --tokenizer <spec> [--device <d>]

``--mode render`` writes ``{"rows": [{"index", "shape", "text"}]}`` -- per pair, the assembled
prompt for every declared shape (``query``, ``document``): prefix + content + suffix, uncut (the
model's own truncation is the remote tokenize's whole-prompt right cut, inside encode; over-cap
rows differ from the client's cut and ride the declared ``anchor_drop_over_cap`` table). ``--mode embed`` writes
``{"rows": [{"index", "query_vectors": [[...]], "document_vectors": [[...]]}]}`` -- one 2560-dim
L2-normalized vector per query and per document, on the model's published path.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = "zeroentropy/zembed-1-embedding"
REVISION = "cf13c81f3274394053d166740294f7eea4586f7a"

SHAPES: tuple[str, ...] = ("query", "document")
"""The declared request shapes; each maps to the checkpoint's prompt of the same name."""

_ADD_SPECIAL_TOKENS = True
"""The embeddings route's default (the tokenizer's post-processor); this checkpoint's
ByteLevel post-processor adds no tokens, which the recipe's stage 1 pins."""

_SUFFIX_APPEND = re.compile(r"text\s*\+\s*\"((?:[^\"\\]|\\.)*)\"")
"""How ``modeling_zembed.ZembedTransformer.tokenize`` writes the suffix it appends
(``texts = [text + "<literal>" for ...]``): the first ``text + "..."`` in the module source."""


def _local_dir(spec: str) -> Path | None:
    """The checkpoint directory ``spec`` names, or None for a Hub repository id (with optional @revision)."""
    candidate = Path(spec)
    if spec.startswith(("/", "./", "../", "~")) or candidate.is_dir():
        return candidate.expanduser()
    return None


def _read_json(path: Path) -> dict[str, object]:
    """One checkpoint JSON file, decoded."""
    data: dict[str, object] = json.loads(path.read_text(encoding="utf-8"))
    return data


@dataclass(frozen=True)
class Checkpoint:
    """The checkpoint's own constants, read from the files the model ships -- never transcribed.

    Attributes:
        prompts: The role prompts (``config_sentence_transformers.json``), keyed ``query``/``document``.
        suffix: The suffix the remote code appends to every text (the same file's ``suffix`` key,
            cross-checked against ``modeling_zembed.py``'s literal when that file is resolvable).
        max_seq_length: The model's whole-prompt cap in tokens (``sentence_bert_config.json``).
        default_prompt_name: The prompt a role-less encode uses (``config_sentence_transformers.json``).
        source: Where the constants were read from (a directory or ``repo@revision``), for the report.
    """

    prompts: dict[str, str]
    suffix: str
    max_seq_length: int
    default_prompt_name: str
    source: str

    @classmethod
    def load(cls, spec: str) -> Checkpoint:
        """Read the constants from the checkpoint ``spec`` names: a local directory or a Hub repo id."""
        directory = _local_dir(spec)
        if directory is not None:
            checkpoint = cls._from_dir(directory, source=str(directory))
        else:
            from huggingface_hub import hf_hub_download

            repo, _, revision = spec.partition("@")
            snapshot = Path(
                hf_hub_download(repo, "config_sentence_transformers.json", revision=revision or None)
            ).parent
            hf_hub_download(repo, "sentence_bert_config.json", revision=revision or None)
            hf_hub_download(repo, "modeling_zembed.py", revision=revision or None)
            checkpoint = cls._from_dir(snapshot, source=f"{repo}@{revision}" if revision else repo)
        missing = sorted(set(SHAPES) - set(checkpoint.prompts))
        if missing:
            raise RuntimeError(
                f"the checkpoint's config_sentence_transformers.json has no prompts for {missing}: "
                f"the declared shapes {list(SHAPES)} cannot be rendered"
            )
        return checkpoint

    @classmethod
    def _from_dir(cls, directory: Path, *, source: str) -> Checkpoint:
        """The constants from one checkpoint directory on disk; the suffix checked against the module."""
        config = _read_json(directory / "config_sentence_transformers.json")
        st_config = _read_json(directory / "sentence_bert_config.json")
        checkpoint = cls(
            prompts={str(name): str(text) for name, text in dict(config["prompts"]).items()},
            suffix=str(config["suffix"]),
            max_seq_length=int(st_config["max_seq_length"]),
            default_prompt_name=str(config.get("default_prompt_name", "document")),
            source=source,
        )
        module = directory / "modeling_zembed.py"
        if module.is_file():
            _check_suffix_literal(module, checkpoint.suffix)
        return checkpoint

    def bindings(self) -> dict[str, object]:
        """What the render was bound to, recorded beside the reference's output (provenance, not ids)."""
        return {
            "source": self.source,
            "max_seq_length": self.max_seq_length,
            "prompt_names": sorted(self.prompts),
            "default_prompt_name": self.default_prompt_name,
            "suffix": "config_sentence_transformers.json's suffix key, verified equal to the literal "
            "modeling_zembed.py's ZembedTransformer.tokenize appends",
        }


def _check_suffix_literal(module_path: Path, suffix: str) -> None:
    """The suffix must be the literal the checkpoint's remote module appends; drift here is the
    silent-anchor defect, so a mismatch is fatal rather than ignored."""
    source = module_path.read_text(encoding="utf-8")
    match = _SUFFIX_APPEND.search(source)
    if match is None:
        raise RuntimeError(
            f"{module_path}: no 'text + \"...\"' append found in ZembedTransformer.tokenize; the module's "
            "shape changed and this recipe's suffix binding must be re-derived"
        )
    literal = ast.literal_eval(f'"{match.group(1)}"')
    if literal != suffix:
        raise RuntimeError(
            f"{module_path}: ZembedTransformer.tokenize appends {literal!r} but the checkpoint's "
            f"config_sentence_transformers.json declares suffix {suffix!r}; the two must be the same string"
        )


class Renderer:
    """The reference's render of one text: the model's frame around the content, uncut.

    The prompt is prefix + content + suffix exactly as the model's own path assembles it; the
    over-cap truncation is the remote tokenize's whole-prompt right cut INSIDE the model (which
    over-cap drops the pooled suffix token -- the faithful behaviour the recipe declares as
    ``anchor_drop_over_cap``), never a port of the client's cut.
    """

    def __init__(self, checkpoint: Checkpoint) -> None:
        self.checkpoint = checkpoint

    def prompt_of(self, shape: str) -> str:
        """The checkpoint's prompt for one declared shape (the frame's head)."""
        if shape not in self.checkpoint.prompts:
            raise ValueError(f"shape must be one of {sorted(self.checkpoint.prompts)}, got {shape!r}")
        return self.checkpoint.prompts[shape]

    def render(self, text: str, shape: str) -> str:
        """The assembled prompt string for one text and shape: prefix + content + suffix, uncut."""
        return self.prompt_of(shape) + text + self.checkpoint.suffix


def _rows_of(pairs_path: str) -> list[dict[str, object]]:
    """The pairs file (the harness's pairs format) as rows."""
    rows: list[dict[str, object]] = []
    for number, line in enumerate(Path(pairs_path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{pairs_path}:{number} is not a JSON object")
        rows.append(row)
    if not rows:
        raise ValueError(f"{pairs_path} holds no pairs")
    return rows


def render_rows(pairs_path: str, tokenizer_spec: str) -> dict[str, object]:
    """Stage 1's reference side: per pair and declared shape, the rendered prompt text (CPU; no weights)."""
    renderer = Renderer(Checkpoint.load(tokenizer_spec))
    rows = []
    for index, row in enumerate(_rows_of(pairs_path)):
        for shape in SHAPES:
            text = str(row["query"]) if shape == "query" else str(row["documents"][0])
            rows.append({"index": index, "shape": shape, "text": renderer.render(text, shape)})
    return {"rows": rows, "bindings": renderer.checkpoint.bindings()}


def embed_rows(pairs_path: str, tokenizer_spec: str, device: str) -> dict[str, object]:
    """Stage 2's reference side: the model's published path over the RAW texts.

    ``encode_query``/``encode_document`` apply the role prompt, the remote tokenize appends the
    suffix and right-truncates the whole prompt at ``max_seq_length`` (its own rule -- over-cap
    drops the pooled suffix, the declared ``anchor_drop_over_cap`` behaviour), Pooling reads the
    last token, Normalize L2-normalizes. Vectors come back float32, one per query and one per
    document, 2560 dims.  Never a port of the client's cut.
    """
    import numpy as np

    renderer = Renderer(Checkpoint.load(tokenizer_spec))
    try:  # the model stack imports lazily: render mode runs without torch or sentence-transformers
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as error:  # pragma: no cover - the reference env ships it
        raise RuntimeError(
            "embed mode needs the reference environment (torch, transformers, sentence-transformers>=5.1,<5.2, "
            "numpy): install requirements-reference.txt into the --reference-python"
        ) from error
    model = SentenceTransformer(
        REPO,
        revision=REVISION,
        trust_remote_code=True,  # loads the checkpoint's modeling_zembed.ZembedTransformer
        model_kwargs={"torch_dtype": "bfloat16"},
        device=device,
    )
    rows = []
    for index, row in enumerate(_rows_of(pairs_path)):
        query_text = str(row["query"])
        query_vectors = [[float(value) for value in vector] for vector in np.asarray(model.encode_query([query_text]))]
        documents = [str(document) for document in row["documents"]]
        if documents:
            document_vectors = [
                [float(value) for value in vector] for vector in np.asarray(model.encode_document(documents))
            ]
        else:
            document_vectors = []
        rows.append({"index": index, "query_vectors": query_vectors, "document_vectors": document_vectors})
    return {"rows": rows, "bindings": renderer.checkpoint.bindings()}


def main() -> int:
    """The CLI the harness invokes (one mode, one pairs file, one output JSON)."""
    parser = argparse.ArgumentParser(description="the zembed-1-embedding reference implementation")
    parser.add_argument("--mode", required=True, choices=["render", "embed"])
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tokenizer", required=True, help="the checkpoint dir or repo id@revision to read")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.mode == "render":
        output = render_rows(args.pairs, args.tokenizer)
    else:
        output = embed_rows(args.pairs, args.tokenizer, args.device)
    Path(args.out).write_text(json.dumps(output) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
