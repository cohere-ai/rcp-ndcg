"""The reference implementation for the zembed-1-embedding recipe (zeroentropy/zembed-1-embedding).

Code path (the model's own published usage, the model card's snippet): sentence-transformers
``SentenceTransformer`` with ``trust_remote_code=True``, which loads the checkpoint's remote module
(``modeling_zembed.ZembedTransformer``, a ``sentence_transformers.models.Transformer`` subclass)
over transformers' Qwen3 backbone. The pipeline is: sentence-transformers prepends the role prompt
(``config_sentence_transformers.json``), the remote ``tokenize`` appends the suffix (the im_end
marker plus a newline; the token whose hidden state the last-token pooler reads) and right-truncates
the whole prompt at ``max_seq_length`` (``sentence_bert_config.json``: 32768), ``Pooling`` pools the
last token, ``Normalize`` L2-normalizes.

Before encoding, the reference applies the recipe's own anchor-preserving cut: the content is cut to
the budget that remains after reserving every fixed template token, and the frame is re-attached --
the same policy the served client declares (``on_overflow: cut``). Feeding the model raw over-cap
text would instead trigger the remote tokenize's whole-prompt right cut, which can drop the pooled
suffix token -- the exact anchor defect research/ANCHOR-FINDING.md bans (the mmmv commit 302b1c9d
class) -- so the reference does not reproduce it, and the recipe declares no
``reference.known_deviations``: stage 2 compares served against reference on identical renders at
every length. Every constant here is read from the checkpoint's own files at run time (bound at
startup, never transcribed), and the suffix is checked against the literal the remote module
appends whenever ``modeling_zembed.py`` is resolvable beside the config.

Reference environment (its own python, never the harness's process): torch>=2.0,
transformers>=4.40, numpy, and sentence-transformers>=3.0,<6 (5.1.x measured). sentence-transformers
6.x must not be used: its preprocess-first pipeline bypasses tokenize-only remote modules and
silently drops the suffix. ``--mode render`` needs only transformers (tokenizer files); ``--mode
embed`` downloads the ~8 GB checkpoint and wants a GPU (the wave passes ``--device``).

CLI (the harness's subprocess contract, enforced by
``rcp_ndcg_vllm.equivalence.reference.run_reference``)::

    reference.py --mode <render|embed> --pairs <file> --out <file> --tokenizer <spec> [--device <d>]

``--mode render`` writes ``{"rows": [{"index", "shape", "text"}]}`` -- per pair, the rendered prompt
for every declared shape (``query``, ``document``): prefix + content (cut to the budget) + suffix,
the exact string ``--mode embed`` then encodes. ``--mode embed`` writes
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


def _tokenizer(spec: str):
    """The checkpoint's tokenizer (transformers, the model's own stack), loaded once.

    A local directory loads with ``local_files_only`` so a stage-1 run never touches the network;
    a Hub spec downloads (or reuses the cache) at the pinned revision.
    """
    from transformers import AutoTokenizer

    directory = _local_dir(spec)
    if directory is not None:
        tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True)
    else:
        repo, _, revision = spec.partition("@")
        tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision or None)
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError(
            "the reference needs the fast tokenizer (tokenizer.json) for offset-based cutting; "
            f"the checkpoint at {spec} loaded {type(tokenizer).__name__}"
        )
    return tokenizer


def _count(tokenizer: object, text: str, *, add_special_tokens: bool) -> int:
    """The token count of ``text`` as the engine reads it (the post-processor's tokens included when
    ``add_special_tokens``)."""
    encoded = tokenizer(text, add_special_tokens=add_special_tokens)  # type: ignore[attr-defined]
    return len(encoded["input_ids"])


def token_prefix(
    text: str,
    max_tokens: int,
    tokenizer: object,
    *,
    rendered,  # noqa: ANN001 - a (str -> str) frame closure; typed loosely on purpose
    add_special_tokens: bool,
) -> str:
    """A prefix of ``text`` that ends at one of its first ``max_tokens`` token boundaries and whose
    assembled render counts at most ``max_tokens``: the longest such prefix the search finds.

    The reference's independent implementation of the recipe's declared cut (the product implements
    the same search in ``rcp_ndcg.data.preprocess.token_prefix``; stage 1 proves the two agree byte
    for byte). The cut is located with the tokenizer's offset mapping on the original text, so the
    result is a verbatim prefix; a candidate is counted as the engine reads it -- ``rendered(prefix)``,
    the full frame around the piece -- because a cut word can re-tokenize differently in place.
    """
    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]  # type: ignore[attr-defined]

    def count(piece: str) -> int:
        return _count(tokenizer, rendered(piece), add_special_tokens=add_special_tokens)

    def prefix(tokens: int) -> str:
        return text[: offsets[tokens - 1][1]] if tokens > 0 else ""

    def fits(tokens: int) -> bool:
        return count(prefix(tokens)) <= max_tokens

    over = min(max_tokens, len(offsets))
    if fits(over):
        return prefix(over)
    fitting, step = over - 1, 1
    while fitting > 0 and not fits(fitting):
        over, fitting, step = fitting, max(fitting - step, 0), step * 2
    while over - fitting > 1:
        middle = (over + fitting) // 2
        fitting, over = (middle, over) if fits(middle) else (fitting, middle)
    return prefix(fitting)


class Renderer:
    """The reference's render of one text: the model's frame, the content cut to the model's cap.

    The cut reserves the frame: the content piece is searched so that ``prompt + piece + suffix``
    counts at most ``max_seq_length`` tokens, then the frame is re-attached -- the anchor (the
    suffix, the pooled token) survives every cut.
    """

    def __init__(self, checkpoint: Checkpoint, tokenizer_spec: str) -> None:
        self.checkpoint = checkpoint
        self.tokenizer = _tokenizer(tokenizer_spec)

    def prompt_of(self, shape: str) -> str:
        """The checkpoint's prompt for one declared shape (the frame's head)."""
        if shape not in self.checkpoint.prompts:
            raise ValueError(f"shape must be one of {sorted(self.checkpoint.prompts)}, got {shape!r}")
        return self.checkpoint.prompts[shape]

    def content_piece(self, text: str, shape: str) -> str:
        """The content span as the reference fits it: cut to the budget the frame leaves, else whole."""
        prompt = self.prompt_of(shape)
        suffix = self.checkpoint.suffix
        cap = self.checkpoint.max_seq_length
        if _count(self.tokenizer, prompt + text + suffix, add_special_tokens=_ADD_SPECIAL_TOKENS) <= cap:
            return text
        return token_prefix(
            text,
            cap,
            self.tokenizer,
            rendered=lambda piece: prompt + piece + suffix,
            add_special_tokens=_ADD_SPECIAL_TOKENS,
        )

    def render(self, text: str, shape: str) -> str:
        """The exact prompt string the recipe sends the engine for one text and shape."""
        return self.prompt_of(shape) + self.content_piece(text, shape) + self.checkpoint.suffix


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
    renderer = Renderer(Checkpoint.load(tokenizer_spec), tokenizer_spec)
    rows = []
    for index, row in enumerate(_rows_of(pairs_path)):
        for shape in SHAPES:
            text = str(row["query"]) if shape == "query" else str(row["documents"][0])
            rows.append({"index": index, "shape": shape, "text": renderer.render(text, shape)})
    return {"rows": rows, "bindings": renderer.checkpoint.bindings()}


def embed_rows(pairs_path: str, tokenizer_spec: str, device: str) -> dict[str, object]:
    """Stage 2's reference side: the model's published path over the recipe's fitted renders.

    The content piece is cut exactly as in ``--mode render`` (same constants, same search), then the
    model card's usage encodes it: ``encode_query``/``encode_document`` apply the role prompt, the
    remote tokenize appends the suffix (a no-op truncation at this length, since the piece was cut
    to leave the frame room), Pooling reads the last token, Normalize L2-normalizes. Vectors come
    back float32, one per query and one per document, 2560 dims.
    """
    import numpy as np

    renderer = Renderer(Checkpoint.load(tokenizer_spec), tokenizer_spec)
    try:  # the model stack imports lazily: render mode runs without torch or sentence-transformers
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as error:  # pragma: no cover - the reference env ships it
        raise RuntimeError(
            "embed mode needs the reference environment (torch, transformers, sentence-transformers>=3.0,<6, "
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
        query_piece = renderer.content_piece(str(row["query"]), "query")
        query_vectors = [[float(value) for value in vector] for vector in np.asarray(model.encode_query([query_piece]))]
        documents = [str(document) for document in row["documents"]]
        if documents:
            pieces = [renderer.content_piece(document, "document") for document in documents]
            document_vectors = [
                [float(value) for value in vector] for vector in np.asarray(model.encode_document(pieces))
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
