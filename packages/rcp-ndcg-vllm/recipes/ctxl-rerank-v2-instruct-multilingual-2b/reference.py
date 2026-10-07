"""Reference implementation for the served recipe ``ctxl-rerank-v2-instruct-multilingual-2b``
(ContextualAI/ctxl-rerank-v2-instruct-multilingual-2b).

Derived with unchanged behaviour from the paper's exact in-process implementation,
``experiments/paper/rerankers/reference/contextual.py`` (``ContextualRerank``, itself moved
unchanged from the former ``src/rcp_ndcg/retrieval/external_rerankers.py`` when the package
stopped carrying in-process models; the paper's numbers rest on it). The paper pipeline loaded it
with ``max_seq_len=8192`` (``MAX_SEQ_LENGTH``, ``rcp_ndcg.retrieval.cross_encoder``), the config's
``batch_size`` (16 for this model, ``experiments/paper/rerankers/ctxl_rerank_2b.yaml``), bfloat16
(the pipeline's ``DTYPE``) and the config's revision (the paper configs pin none; the recipe's
commit is pinned here, so a moved default checkpoint cannot silently break equivalence).

The prompt is the paper's two-line frame (document before query, the " ??" tail the score reads
out from)::

    Check whether a given document contains information helpful to answer the query.
    <Document> {document}
    <Query> {query} ??

The score is the raw logit of vocabulary position 0 at the final position (left-padded, the last
real token's): ``logits[:, -1, 0]``, no sigmoid, no softmax. Whole-prompt right truncation at
8192 tokens is the paper's own behaviour (``truncation=True, max_length=8192`` on the whole
prompt): over the cap it drops the trailing " ??" anchor and the query behind it — the recipe
declares that as ``reference.known_deviations: [anchor_drop_over_cap]`` instead of copying it
into the served path, whose client-side cut reserves every fixed frame token and cuts only the
content spans (the product's ``fit``). Under the cap AND within the declared query share the two
paths render byte-identical prompts (a query over the share is a declared divergence row: the wire
settles it at the share once per call, the paper keeps it whole).

Declared interface behaviours, neither a scoring change:

- the pairs file's ``instruction`` is ignored: the recipe declares ``instruction: none`` (the
  paper's served-path mode) and the paper's in-process path itself never set an instruction (its
  ``instruction=None`` behaviour) — no side folds ``Task: ...\nQuery: ...``, and the bare query is
  what both sides use;
- ``--mode render`` fills the harness's rerank span format (``{"index", "shape": "pair", "query",
  "documents"}``) with the spans the paper's prompt builder receives: the raw query and the raw
  documents, uncut -- the paper cuts only at encode (the whole-prompt right truncation in
  ``_forward_scores``).  Under the cap and within the query share these are the spans the served
  wire carries, byte for byte; over the cap they differ by declaration (``anchor_drop_over_cap``,
  stage 1's non-gating table), and a query over the 4096-token share is a declared divergence row
  (the merged rerank client settles the shared query once per call and ships it at the share
  whenever it exceeds it; the paper has no query share).  The reference never reproduces the
  product client's cut.

Run as the harness's reference subprocess (never imported by the harness, which holds no torch)::

    reference.py --mode <render|score> --pairs <file> --out <file> --tokenizer <spec> [--device <d>]

``render`` writes ``{"rows": [{"index", "shape", "query", "documents"}]}``; ``score`` writes
``{"rows": [{"index", "scores": [...]}]}`` on the recipe's ``score_scale: logit`` (one raw logit
per document).

Reference environment (``requirements-reference.txt`` beside this file, documented not installed):
torch 2.9.1, transformers 4.57.6, accelerate, flash-attn 2.8.3 (the paper's former ``[local]``
extra pins, from ``experiments/paper/rerankers/reference/requirements.txt``). ``render`` is pure
string work (no tokenizer, no weights); ``score`` needs the weights (about 5.97 GB stored,
5.19 GB in bf16, from the Hub's safetensors metadata), the transformers pin and the device the harness
passes. No rcp-ndcg import: the reference environment is the paper's, not the harness's.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

#: The recipe's checkpoint (the recipe ``id``/``revision``; the served model resolves this commit).
DEFAULT_MODEL = "ContextualAI/ctxl-rerank-v2-instruct-multilingual-2b"
DEFAULT_REVISION = "6ffef5dc552583b8db58dc4a87f79f7aee78d2d9"

#: The paper budget: ``MAX_SEQ_LENGTH``, the whole prompt's right-truncation length at encode (the
#: recipe's ``client.max_tokens``; the recipe's ``query_max_tokens`` 4096 is the served path's share,
#: which the paper's in-process path never applied).
MAX_SEQ_LEN = 8192

#: The paper's per-model batch size and padded-area budget (``ContextualRerank.__init__``;
#: the area budget counts *characters*, exactly as the paper code did).
BATCH_SIZE = 16
BATCH_SIZE_TOKENS = 15_000

#: The score's vocabulary position at the final position (``logits[:, -1, 0]``: vocabulary
#: position 0 is the checkpoint's relevance logit, ``1_LogitScore/config.json`` true_token_id 0).
VOCAB_POSITION = 0

#: The fixed frame, exactly the served template's fixed segments (and the paper's
#: ``_format_prompts``). The tail is the anchor: the model reads its score at the final
#: position, so the " ??" tokens must survive every cut (the served cut reserves them;
#: the paper's whole-prompt truncation does not — the declared deviation).
HEAD = "Check whether a given document contains information helpful to answer the query.\n<Document> "
MID = "\n<Query> "
TAIL = " ??"


def _tokenizer_dir(spec: str) -> str:
    """The tokenizer location the ``--tokenizer`` spec names, in the form transformers loads.

    The spec is the recipe's ``client.tokenizer``: a Hub id with an optional ``@revision``, a
    directory of tokenizer files, or a ``tokenizer.json`` path (its directory is used). A Hub id
    is returned WITHOUT its ``@revision`` — the revision travels separately (the constructor's
    ``revision`` argument); ``repo@revision`` is not a repo id transformers can resolve.
    """
    candidate = Path(spec).expanduser()
    if candidate.is_file():
        return str(candidate.parent)
    if candidate.is_dir():
        return str(candidate)
    repo, _, _ = spec.partition("@")
    return repo


class PairTokenizer:
    """The paper's tokenizer for ``score``: transformers' ``AutoTokenizer`` at the pinned revision.

    Loaded with ``padding_side="left"`` exactly as the paper class's ``__init__``; padding applies to
    ``score`` batching only (``_forward_scores``).  Needs the reference environment (transformers).
    """

    def __init__(self, spec: str, *, revision: str) -> None:
        from transformers import AutoTokenizer

        self._fast = AutoTokenizer.from_pretrained(_tokenizer_dir(spec), padding_side="left", revision=revision)
        # The paper's fallback (ContextualRerank.__init__); inert for this checkpoint, whose
        # tokenizer_config.json already carries pad_token "+".
        if self._fast.pad_token is None:
            self._fast.pad_token = self._fast.eos_token


class CtxlRerankReference:
    """The paper's in-process scoring recipe for ctxl-rerank-v2, as ``load``/``render``/``score``.

    Behaviour is the paper's (``experiments/paper/rerankers/reference/contextual.py``,
    ``ContextualRerank``): the same prompt frame, padding side (left), whole-prompt right
    truncation at ``MAX_SEQ_LEN``, dtype (bfloat16, the pipeline's), the length-descending
    character-batch permutation with the 16-document cap and the 15,000-character padded-area
    budget, and the OOM back-off that splits a batch in half. The only changes from that module
    are the harness's subprocess interface (``--mode`` CLI, pairs file in, JSON out) and the pinned
    revision; the row's instruction is ignored (``instruction: none``, the paper's own path).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        tokenizer_spec: str,
        max_seq_len: int = MAX_SEQ_LEN,
        batch_size: int = BATCH_SIZE,
        batch_size_tokens: int = BATCH_SIZE_TOKENS,
        dtype: str = "bfloat16",
        revision: str = DEFAULT_REVISION,
    ) -> None:
        # padding_side="left", exactly the paper class's __init__ (see PairTokenizer).
        self.tokenizer = PairTokenizer(tokenizer_spec, revision=revision)
        self.model_name = model_name
        self.revision = revision
        self.max_length = max_seq_len
        self.batch_size = batch_size
        self.batch_size_tokens = batch_size_tokens
        self.dtype = dtype
        self.model: Any = None
        self.device = "cpu"

    # -- the harness's interface ------------------------------------------------------------
    def load(self, device: str) -> CtxlRerankReference:
        """Load the checkpoint (first call) and move it to ``device`` (e.g. ``cuda:0`` or ``cpu``).

        The weights load here, in the reference process: the harness imports no torch. dtype is
        the paper pipeline's bfloat16; the attention implementation is the paper class's default
        (flash-attention-2) where it exists — the paper's runs were single-GPU — and the plain
        implementation on a CPU diagnostic run, where flash-attention-2 does not exist.
        """
        import torch
        from transformers import AutoModelForCausalLM

        if self.model is None:
            model_kwargs: dict[str, Any] = {
                "dtype": torch.bfloat16 if self.dtype == "bfloat16" else None,
                "revision": self.revision,
            }
            device = device_name(device)
            if device.startswith("cuda"):
                model_kwargs["attn_implementation"] = "flash_attention_2"
            else:
                model_kwargs["attn_implementation"] = None
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
            self.model.eval()
        self.model.to(device)
        self.device = device
        return self

    def prompt_text(self, query: str, doc: str) -> str:
        """The exact prompt string for one pair (the paper's ``_format_prompts`` row).

        The paper's own instruction slot is empty (``ContextualRerank.instruction`` is None on the
        paper path, and the recipe declares ``instruction: none``): the bare query rides here.
        """
        return f"{HEAD}{doc}{MID}{query}{TAIL}"

    def score(self, query: str, docs: list[str], instruction: str | None = None) -> list[float]:
        """The raw relevance logit per document, aligned with ``docs`` (the paper's ``predict``).

        The instruction is ignored (the recipe declares instruction: none and the paper path
        received none), then the paper's batching runs unchanged: length-descending permutation,
        the padded-area character budget with the 16-document cap, and the OOM back-off. Batching
        affects throughput only, never the per-document score.
        """
        del instruction
        prompts = [self.prompt_text(query, doc) for doc in docs]

        permutation = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
        sorted_prompts = [prompts[i] for i in permutation]

        batches: list[list[str]] = []
        max_len = 0
        for text in sorted_prompts:
            text_len = len(text)
            over_tokens = bool(batches) and (len(batches[-1]) + 1) * max(max_len, text_len) > self.batch_size_tokens
            over_count = bool(batches) and len(batches[-1]) >= self.batch_size
            if not batches or over_tokens or over_count:
                batches.append([])
                max_len = 0
            batches[-1].append(text)
            max_len = max(max_len, text_len)

        sorted_scores: list[float] = []
        for batch in batches:
            sorted_scores.extend(self._forward_scores(batch))

        result = [0.0] * len(docs)
        for orig_idx, score in zip(permutation, sorted_scores, strict=True):
            result[orig_idx] = score
        return result

    # -- the paper's mechanics (unchanged) ----------------------------------------------------
    def _forward_scores(self, batch: list[str]) -> list[float]:
        """One batch's raw logits at the final position, vocabulary position 0 (the paper's code).

        Whole-prompt right truncation at ``MAX_SEQ_LEN`` (the paper's ``truncation=True,
        max_length=8192``: over the cap the tail — the " ??" anchor and the query behind it — is
        dropped; the recipe declares this as ``anchor_drop_over_cap``), left padding to the
        batch's longest sequence, and the float32 cast of the gathered logits.
        """
        import torch

        try:
            with torch.no_grad():
                enc = self.tokenizer._fast(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                )
                input_ids = enc["input_ids"].to(self.device)
                attention_mask = enc["attention_mask"].to(self.device)
                out = self.model(input_ids=input_ids, attention_mask=attention_mask)
                # Left-padded, so the final position is the last real token for every row;
                # vocab index 0 is ctxl's relevance logit (the paper's logits[:, -1, 0]).
                return out.logits[:, -1, VOCAB_POSITION].float().tolist()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(batch) == 1:
                raise
            mid = len(batch) // 2
            return self._forward_scores(batch[:mid]) + self._forward_scores(batch[mid:])


def paper_spans(row: dict[str, Any]) -> dict[str, Any]:
    """One pairs row in the harness's rerank span format: the spans the paper's prompt builder gets.

    The raw query and the raw documents, uncut (the paper cuts only at encode); the row's instruction
    is ignored (``instruction: none``)."""
    return {"query": str(row["query"]), "documents": [str(document) for document in row["documents"]]}


def device_name(device: str) -> str:
    """The device string the model is moved to: the harness's ``--device`` (``cpu`` default)."""
    return device or "cpu"


def main() -> int:
    """The harness's reference CLI: ``--mode render|score`` over a pairs JSONL file."""
    parser = argparse.ArgumentParser(description="the ctxl-rerank-v2-instruct-multilingual-2b paper reference")
    parser.add_argument("--mode", required=True, choices=["render", "score"])
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    rows_in = [json.loads(line) for line in Path(args.pairs).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.mode == "render":
        # The harness's rerank span format, filled with the paper's own spans (uncut; see paper_spans).
        rows = [
            {"index": index, "shape": str(row.get("shape") or "pair"), **paper_spans(row)}
            for index, row in enumerate(rows_in)
        ]
    else:
        reference = CtxlRerankReference(tokenizer_spec=args.tokenizer).load(device_name(args.device))
        rows = [
            {
                "index": index,
                "scores": reference.score(str(row["query"]), list(row["documents"])),
            }
            for index, row in enumerate(rows_in)
        ]
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump({"rows": rows}, handle, indent=1)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
