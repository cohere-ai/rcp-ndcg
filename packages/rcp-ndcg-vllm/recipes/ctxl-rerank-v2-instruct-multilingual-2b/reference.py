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
- ``--mode render`` implements the harness's rerank reference contract: the cut CONTENT spans the
  wire carries per pairs row (the query settled at its 4096-token share whenever it exceeds it —
  the merged rerank client's settle rule; a bare pair fit binds the share on overflow only — then
  through fit's probe pair, each document cut to what remains, the " ??" anchor re-attached by the
  engine around these spans).  It is an independent port of the product's mechanism (offset-based
  verbatim prefixes), checked byte-for-byte against the client's captured spans by stage 1's
  render comparison; ``score`` keeps the paper's whole-prompt truncation (the declared deviation
  lives only in the score path: over-cap pairs and the declared over-share divergence rows).

Run as the harness's reference subprocess (never imported by the harness, which holds no torch)::

    reference.py --mode <render|score> --pairs <file> --out <file> --tokenizer <spec> [--device <d>]

``render`` writes ``{"rows": [{"index", "shape", "text"}]}``; ``score`` writes
``{"rows": [{"index", "scores": [...]}]}`` on the recipe's ``score_scale: logit`` (one raw logit
per document).

Reference environment (``requirements-reference.txt`` beside this file, documented not installed):
torch 2.9.1, transformers 4.57.6, accelerate, flash-attn 2.8.3 (the paper's former ``[local]``
extra pins, from ``experiments/paper/rerankers/reference/requirements.txt``). ``render`` needs the
tokenizer only (transformers, or the ``tokenizers`` library over the same ``tokenizer.json`` when
transformers is absent — stage 1 on CPU); ``score`` needs the weights (about 5.97 GB stored,
5.19 GB in bf16 — the Hub's safetensors metadata, corrected by the audit-synth lane from the
research draft's stale per-precision figures), the transformers pin and the device the harness
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

#: The paper budget: ``MAX_SEQ_LENGTH`` (the whole prompt), and ``MAX_QUERY_LENGTH``, the query
#: share the paper's own served path sent as ``max_tokens_per_query``; the recipe declares both as
#: ``client.max_tokens``/``client.query_max_tokens``.
MAX_SEQ_LEN = 8192
QUERY_MAX_TOKENS = 4096

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


def served_spans(tokenizer: PairTokenizer, query: str, document: str) -> tuple[str, str]:
    """The cut content spans (query, document) the served wire carries for one pair.

    Port of the product's role client for this recipe's declaration (the settle rule and ``fit``'s
    pair path): the query's span settles once per request — to its declared share
    (``QUERY_MAX_TOKENS``) whenever it exceeds it — then through fit's probe pair (the query with
    an empty document) that guarantees the frame and its anchor fit even alone, and the document
    gets what remains.  Every cut is a verbatim token-boundary prefix of the original text (never a
    decode round trip).  A query that fills the budget and leaves the document nothing is refused,
    never cut undeclared.
    """

    def assemble(q: str, d: str) -> str:
        return f"{HEAD}{d}{MID}{q}{TAIL}"

    cap = MAX_SEQ_LEN
    # 1. settle at the declared share whenever the query exceeds it (the wire's settle rule):
    if tokenizer.count(query) > QUERY_MAX_TOKENS:
        q_final = token_prefix(query, QUERY_MAX_TOKENS, tokenizer)
    else:
        q_final = query
    # 2. fit's probe pair (the query with an empty document): the query keeps the frame room.
    q_final = token_prefix(q_final, cap, tokenizer, rendered=lambda piece: assemble(piece, ""), add_special_tokens=True)
    q_min = tokenizer.count(assemble(q_final, ""), add_special_tokens=True)
    if q_min >= cap and tokenizer.count(document) > 0:
        raise RuntimeError(
            f"the query fills the pair budget of {cap} tokens and leaves the document nothing: "
            "lower query_max_tokens (or raise max_tokens), so the document keeps a share"
        )
    # 3. the document gets what remains after the settled query and the frame:
    d_final = token_prefix(
        document, cap, tokenizer, rendered=lambda piece: assemble(q_final, piece), add_special_tokens=True
    )
    return q_final, d_final


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
    """The tokenizer surface the reference needs, over the recipe's ``tokenizer.json``.

    The paper path is transformers' ``AutoTokenizer`` (the reference environment of
    ``requirements-reference.txt``, where ``score`` runs). Stage 1 on CPU runs in the harness's
    environment, which carries no transformers; there the same ``tokenizer.json`` loads through
    the ``tokenizers`` library directly — the library transformers' fast tokenizers wrap — giving
    the same ids, offsets and counts for this checkpoint (its ByteLevel post-processor adds no
    tokens; measured on the checkpoint's tokenizer.json).

    The ``padding_side`` the paper code sets is ``left``; padding applies to ``score`` batching
    only (``_forward_scores``), never to the rendered prompt text.
    """

    def __init__(self, spec: str, *, revision: str) -> None:
        path = Path(spec).expanduser()
        file = path / "tokenizer.json" if path.is_dir() else path
        try:
            from transformers import AutoTokenizer
        except ModuleNotFoundError:
            if not file.is_file():
                raise SystemExit(
                    f"the tokenizer spec {spec!r} names no local tokenizer.json and transformers is "
                    "not installed: pass a local tokenizer.json path, or run in the reference "
                    "environment of requirements-reference.txt"
                ) from None
            from tokenizers import Tokenizer

            self._fast = Tokenizer.from_file(str(file))
            self._hub = False
        else:
            self._fast = AutoTokenizer.from_pretrained(_tokenizer_dir(spec), padding_side="left", revision=revision)
            # The paper's fallback (ContextualRerank.__init__); inert for this checkpoint, whose
            # tokenizer_config.json already carries pad_token "+".
            if self._fast.pad_token is None:
                self._fast.pad_token = self._fast.eos_token
            self._hub = True

    def count(self, text: str, *, add_special_tokens: bool = False) -> int:
        """The number of tokens of ``text``, optionally as the engine counts it."""
        return len(self.ids(text, add_special_tokens=add_special_tokens))

    def ids(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        """The token ids of ``text``, optionally with the post-processor's (none for this model)."""
        if self._hub:
            return list(self._fast(text, add_special_tokens=add_special_tokens)["input_ids"])
        return list(self._fast.encode(text, add_special_tokens=add_special_tokens).ids)

    def offsets(self, text: str) -> list[tuple[int, int]]:
        """``(start, end)`` character offsets in ``text`` of each of its tokens, in order."""
        if self._hub:
            encoding = self._fast(text, add_special_tokens=False, return_offsets_mapping=True)
            return [tuple(pair) for pair in encoding["offset_mapping"]]
        return [tuple(pair) for pair in self._fast.encode(text, add_special_tokens=False).offsets]


def token_prefix(
    text: str,
    max_tokens: int,
    tokenizer: PairTokenizer,
    *,
    rendered: Any = None,
    add_special_tokens: bool = False,
) -> str:
    """The longest prefix of ``text`` ending at one of its first ``max_tokens`` token boundaries.

    The recipe budget's cut, over the recipe tokenizer's offset mapping: a verbatim prefix of
    ``text`` (never tokens decoded back to text), counted as the engine reads it —
    ``rendered(piece)`` when given (the assembled render, so a byte-level merge across a span
    join cannot push the request over the budget) — a galloping, then binary search over token
    boundaries. The mirror of the product's ``rcp_ndcg.data.preprocess.token_prefix``, which the
    reference environment does not carry; stage 1's render comparison holds the two equal.
    """

    def count(piece: str) -> int:
        return tokenizer.count(
            rendered(piece) if rendered is not None else piece, add_special_tokens=add_special_tokens
        )

    if count(text) <= max_tokens:
        return text
    offsets = tokenizer.offsets(text)

    def prefix(tokens: int) -> str:
        return text[: offsets[tokens - 1][1]] if tokens > 0 else ""

    def fits(tokens: int) -> bool:
        return count(prefix(tokens)) <= max_tokens

    # ``fitting`` fits (the empty prefix always does); ``over`` does not, or is past the budget.
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


class CtxlRerankReference:
    """The paper's in-process scoring recipe for ctxl-rerank-v2, as ``load``/``render``/``score``.

    Behaviour is the paper's (``experiments/paper/rerankers/reference/contextual.py``,
    ``ContextualRerank``): the same prompt frame, padding side (left), whole-prompt right
    truncation at ``MAX_SEQ_LEN``, dtype (bfloat16, the pipeline's), the length-descending
    character-batch permutation with the 16-document cap and the 15,000-character padded-area
    budget, and the OOM back-off that splits a batch in half. The only changes from that module
    are the harness's subprocess interface (``--mode`` CLI, pairs file in, JSON out), the pinned
    revision and the instruction fold the served client applies.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        tokenizer_spec: str,
        max_seq_len: int = MAX_SEQ_LEN,
        query_max_tokens: int = QUERY_MAX_TOKENS,
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
        self.query_max_tokens = query_max_tokens
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

    def render_spans(self, query: str, documents: list[str]) -> dict[str, Any]:
        """The harness's rerank render row body: the wire's settled query span and document spans."""
        query_span, _ = served_spans(self.tokenizer, query, documents[0] if documents else "")
        return {
            "query": query_span,
            "documents": [served_spans(self.tokenizer, query, str(document))[1] for document in documents],
        }

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
        # The harness's rerank reference contract: the wire's content spans per pairs row (the
        # query settled once, each document cut to what remains). The row's instruction is ignored
        # (the recipe declares instruction: none).
        reference = CtxlRerankReference(tokenizer_spec=args.tokenizer)
        rows = []
        for index, row in enumerate(rows_in):
            documents = [str(document) for document in row["documents"]]
            spans = reference.render_spans(str(row["query"]), documents)
            rows.append(
                {
                    "index": index,
                    "shape": str(row.get("shape") or "pair"),
                    "query": spans["query"],
                    "documents": spans["documents"],
                }
            )
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
