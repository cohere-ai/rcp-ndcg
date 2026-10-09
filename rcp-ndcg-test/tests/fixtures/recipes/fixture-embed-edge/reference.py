"""The fixture reference, run as a subprocess in its own environment (never imported by the harness).

The reference is deliberately independent: it renders the prompt from its own constants -- never through the
product's :func:`rcp_ndcg.data.preprocess.fit`, so stage 1's render check compares two implementations, not a
function with itself.  The vectors come from the same deterministic numbers as the stub engine
(``tests/fixtures/deterministic.py``).  It reads the pairs file and writes the mode's JSON to ``--out``.

Render contract (embed roles): per pairs-file row and declared shape, a row of ``index``, ``shape`` and
``text`` -- the exact prompt the engine reads (the fixed frame around the content; the post-processor's
tokens are the engine's and are not part of the client's render).  One input per shape is rendered (the
row's query for the query shape, its first document for the document shape).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PREFIX = "doc: "
"""The fixed frame before the document span (specials written ``{special:name}``, resolved from the tokenizer)."""

SUFFIX = ""
"""The fixed frame after the document (empty when the shape ends in content)."""

QUERY_PREFIX = "query: "
"""The query side's frame (only compared when the recipe declares the query shape)."""

EMBED_TAG = "embed"
"""The deterministic vectors' tag: the stub engine's own tag for the recipe's role."""


def main() -> int:
    from rcp_ndcg.data.tokenizer import load_tokenizer

    parser = argparse.ArgumentParser(description="the fixture embed reference")
    parser.add_argument("--mode", required=True, choices=["render", "embed"])
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--recipe", required=True, help="the resolved recipe JSON the harness passed")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from deterministic import token_vectors, vector

    tokenizer = load_tokenizer(args.tokenizer)
    prefix = PREFIX.replace("{special:cls}", tokenizer.special_text("cls"))
    suffix = SUFFIX.replace("{special:sep}", tokenizer.special_text("sep"))

    def assemble(document: str) -> str:
        """The reference's own render of one document-side input."""
        return prefix + document + suffix

    # The variant's resolved recipe travels with the invocation (decision 34: one family reference per
    # family, parameterised by the variant): the declared shapes and the role come from it.
    recipe = json.loads(Path(args.recipe).read_text(encoding="utf-8"))
    template = recipe["client"].get("template") or {}
    shapes = [
        shape for shape in ("query", "document", "pair") if isinstance(template.get(shape), list)
    ] or ["document"]
    ragged = recipe["role"] == "multi_vector"
    pairs = [json.loads(line) for line in Path(args.pairs).read_text(encoding="utf-8").splitlines() if line.strip()]

    if args.mode == "render":
        rows = []
        for index, row in enumerate(pairs):
            for shape in shapes:
                rendered = QUERY_PREFIX + row["query"] if shape == "query" else assemble(row["documents"][0])
                rows.append({"index": index, "shape": shape, "text": rendered})
        output_result = {"rows": rows}
    else:
        rows = []
        for index, row in enumerate(pairs):
            row_result: dict = {"index": index}
            encode = token_vectors if ragged else vector
            if "query" in shapes:
                row_result["query_vectors"] = [[float(x) for x in encode(QUERY_PREFIX + row["query"], EMBED_TAG)]]
            row_result["document_vectors"] = [
                [float(x) for x in encode(assemble(document), EMBED_TAG)] for document in row["documents"]
            ]
            rows.append(row_result)
        output_result = {"rows": rows}
    Path(args.out).write_text(json.dumps(output_result, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
