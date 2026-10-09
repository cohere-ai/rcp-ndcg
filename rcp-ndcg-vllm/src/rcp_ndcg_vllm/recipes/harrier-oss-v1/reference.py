"""The harrier-oss-v1 family's one reference (decision 34): the model card's own
sentence-transformers usage, faithful, parameterised by the variant.

The card's usage (identical for all three variants except the model name)::

    model = SentenceTransformer("microsoft/harrier-oss-v1-<size>", model_kwargs={"dtype": "auto"})
    query_embeddings = model.encode(queries, prompt_name="web_search_query")
    document_embeddings = model.encode(documents)

Queries carry the checkpoint's own ``web_search_query`` prompt (config_sentence_transformers.json,
byte-identical at all three pinned revisions); documents are embedded bare ("No need to add
instruction for retrieval documents", card); the pipeline is the checkpoint's own
Transformer -> Pooling(lasttoken) -> Normalize, so ``encode`` returns L2-normalised vectors. Any
``instruction`` a pairs row carries is ignored on both sides of the equivalence check: the card's fold
is the only instruction this path applies, and a task-specialised variant is a new recipe (the family
template's frame), never a per-row field.

One file serves every variant of the family: the variant travels with the invocation, in the resolved
recipe the harness passes. Run as the harness's reference subprocess (never imported by the harness,
which holds no torch)::

    reference.py --mode <render|embed> --pairs <file> --out <file> --tokenizer <spec> \
        --recipe <resolved-recipe.json> [--device <d>]

``--recipe`` is the resolved recipe the harness loaded (a JSON dump of the family-expanded
``Recipe``): its ``model``/``revision`` pin the checkpoint every mode loads, and a ``--tokenizer``
spec that names another checkpoint (or another revision) is refused, never defaulted.

``render`` mode emits the prompt TEXT the served client must render -- the harness compares the texts
byte-exactly and tokenises both sides with the same tokenizer; the GPU wave's stage-1 ``/tokenize``
check is the authoritative engine-side cross-check. The render is the prompt the card's code hands its
tokenizer, UNCUT: the card's transformers snippet truncates the ids inside
``tokenizer(..., truncation=True, max_length=32768)`` and its sentence-transformers path truncates at
the Transformer module's max_seq_length, which sentence-transformers infers when the checkpoint sets
none (no sentence_bert_config.json, no ``max_seq_length`` in modules.json) as
``min(config.max_position_embeddings, tokenizer.model_max_length)`` (``models/Transformer.py`` at
>=3.0) -- 32768 for the 270m and the 0.6b, 131072 for the 27b -- and this reference never reproduces
the product client's cut. Under the budget the render equals the served prompt byte for byte; over
the budget the two cuts can differ at the content boundary, so the recipe declares
``over_cap_cut_differs`` and over-cap rows ride the harness's non-gating table.

The anchor: the served route (``/v1/embeddings``, string inputs) tokenizes with the tokenizer's
default ``add_special_tokens=True`` (vllm/renderers/params.py:183 at tag v0.31.0), appending one
post-processor token that last-token pooling reads -- the Gemma variants' ``<eos>`` (id 1, bos id 2
leading) for the 270m and the 27b, the Qwen endoftext (id 151643) for the 0.6b (measured at the
pinned revisions). Both survive the client's anchor-preserving cut, so the declared deviation is
``over_cap_cut_differs``, never ``anchor_drop_over_cap``.

Reference environment: ``requirements-reference.txt`` beside this file. ``render`` needs only
``huggingface-hub`` (the pinned tokenizer.json and the checkpoint's own prompt file), so stage 1 runs
on CPU without torch or weights; ``embed`` is the card's sentence-transformers path (torch,
transformers, sentence-transformers).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PROMPT_NAME = "web_search_query"
"""The checkpoint's preconfigured retrieval prompt (config_sentence_transformers.json prompts), the
card's ``encode(..., prompt_name=...)``; byte-identical at all three pinned revisions."""

BATCH_SIZE = 8
"""Texts per forward pass in embed mode (the card encodes both lists outright; batching is
behaviour-neutral for last-token pooling with the attention mask, so this is throughput only)."""


def _load_recipe(path: str) -> dict[str, Any]:
    """The resolved recipe the harness passed (``--recipe``): the variant's identity."""
    recipe = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("model", "revision", "id"):
        if not recipe.get(key):
            raise SystemExit(f"--recipe is missing {key!r}: this reference pins its variant through it")
    return recipe


def _checkpoint_files(tokenizer_spec: str, model: str, revision: str) -> Path:
    """The directory holding the checkpoint's tokenizer.json (and its own prompt file).

    A local spec (a recipe-relative tokenizer path the harness resolved, or a directory) is used as
    it stands; a Hub spec must name THIS recipe's ``model@revision`` -- drift is refused loudly, the
    reference tokenises and reads the prompts with the checkpoint's own files, never whatever a newer
    HEAD carries.
    """
    candidate = Path(tokenizer_spec).expanduser()
    if candidate.suffix == ".json" or candidate.exists():
        return candidate if candidate.is_dir() else candidate.parent
    repo, _, spec_revision = tokenizer_spec.partition("@")
    if (repo and repo != model) or (spec_revision and spec_revision != revision):
        raise SystemExit(f"the tokenizer spec {tokenizer_spec!r} does not pin this recipe's {model}@{revision}")
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            model,
            revision=revision,
            allow_patterns=["tokenizer.json", "tokenizer_config.json", "config_sentence_transformers.json"],
        )
    )


def _web_search_prompt(directory: Path) -> str:
    """The checkpoint's own ``web_search_query`` prompt, byte-exact from its config file."""
    path = directory / "config_sentence_transformers.json"
    prompts = json.loads(path.read_text(encoding="utf-8"))["prompts"]
    prompt = prompts[PROMPT_NAME]
    if not isinstance(prompt, str) or not prompt:
        raise SystemExit(f"{path}: prompts.{PROMPT_NAME} is not a non-empty string")
    return prompt


def render_rows(pairs: list[dict[str, Any]], recipe: dict[str, Any], tokenizer_spec: str) -> dict[str, Any]:
    """``--mode render``: the prompt text per row per declared shape, as the card's code builds it.

    The query shape carries the checkpoint's own ``web_search_query`` prompt (read from the
    checkpoint's config file, not restated here); the document shape is the bare text. Uncut: the
    card truncates the ids at encode, never the text, and the reference never ports the client's cut.
    """
    directory = _checkpoint_files(tokenizer_spec, recipe["model"], recipe["revision"])
    prompt = _web_search_prompt(directory)
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(pairs):
        rows.append({"index": index, "shape": "query", "text": prompt + str(row["query"])})
        rows.append({"index": index, "shape": "document", "text": str(row["documents"][0])})
    return {"rows": rows}


def embed_rows(pairs: list[dict[str, Any]], recipe: dict[str, Any], tokenizer_spec: str, device: str) -> dict[str, Any]:
    """``--mode embed``: the card's sentence-transformers path -- L2-normalised float32 vectors.

    One vector for the row's query (the checkpoint's own prompt applied through
    ``prompt_name=web_search_query``) and one per document (bare), from the checkpoint's own
    Transformer -> Pooling(lasttoken) -> Normalize pipeline at the pinned revision.
    """
    model_id, revision = recipe["model"], recipe["revision"]
    _checkpoint_files(tokenizer_spec, model_id, revision)  # the spec pin is checked in render mode too

    import numpy as np
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        model_id,
        revision=revision,
        device=device,
        model_kwargs={"dtype": "auto"},
    )

    def embed(texts: list[str], *, prompt_name: str | None) -> list[list[float]]:
        """One L2-normalised float32 vector per text (the model's Normalize module), in batches."""
        out: list[Any] = []
        for offset in range(0, len(texts), BATCH_SIZE):
            embeddings = model.encode(
                texts[offset : offset + BATCH_SIZE],
                prompt_name=prompt_name,
                convert_to_numpy=True,
            )
            out.append(np.asarray(embeddings, dtype=np.float32))
        return np.concatenate(out, axis=0).tolist()

    rows: list[dict[str, Any]] = []
    for index, row in enumerate(pairs):
        query_vectors = embed([str(row["query"])], prompt_name=PROMPT_NAME)
        document_vectors = embed([str(document) for document in row["documents"]], prompt_name=None)
        rows.append(
            {
                "index": index,
                "query_vectors": query_vectors,
                "document_vectors": document_vectors,
            }
        )
    return {"rows": rows}


def main() -> int:
    """The reference CLI the harness's runner invokes (see ``equivalence/reference.py``)."""
    parser = argparse.ArgumentParser(description="the microsoft/harrier-oss-v1 family reference")
    parser.add_argument("--mode", required=True, choices=["render", "embed"])
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="the recipe client's tokenizer spec: the variant's model@revision, or a local path "
        "to the same checkpoint's tokenizer files",
    )
    parser.add_argument("--recipe", required=True, help="the resolved recipe JSON the harness writes beside --out")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    recipe = _load_recipe(args.recipe)
    pairs = [json.loads(line) for line in Path(args.pairs).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.mode == "render":
        result = render_rows(pairs, recipe, args.tokenizer)
    else:
        result = embed_rows(pairs, recipe, args.tokenizer, args.device)
    Path(args.out).write_text(json.dumps(result) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    # The card's own example (its printed queries and documents) reproduces the card's printed
    # cosines through --mode embed: the vectors are the checkpoint's own lasttoken+Normalize
    # pipeline at the pinned revision, and the harness gates vectors per vector (cosine >= 1-1e-3
    # after the float16 cast), not any single run's printed number.
    raise SystemExit(main())
