"""Reference implementation for the served recipe ``ctxl-rerank-v2-instruct-multilingual-2b``
(ContextualAI/ctxl-rerank-v2-instruct-multilingual-2b).

Derived with unchanged behaviour from the paper's exact in-process implementation,
``experiments/paper/rerankers/reference/contextual.py`` (``ContextualRerank``, itself moved
unchanged from the former ``rcp-ndcg/src/rcp_ndcg/retrieval/external_rerankers.py`` when the package
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
content spans (the product's ``fit``). Under the cap the two paths render byte-identical prompts.

Two deliberate interface-level additions, neither a scoring change:

- the pairs file's ``instruction`` is folded into the query text exactly as the served client
  folds it (``Task: {instruction}\\nQuery: {text}``, ``rcp_ndcg_core._records``), so an
  instruction-bearing row renders and scores on the same prompt on both sides; the paper path
  itself never sets an instruction (its ``instruction=None`` behaviour is the no-instruction
  case here);
- ``render`` implements the recipe's declared anchor-preserving render (the fixed frame
  reserved, the query cut to its 4096-token share on overflow, the document cut to the rest,
  the " ??" tail re-attached) — the render-mode contract of the harness's stage 1, so the
  reference subprocess agrees with the served ``fit`` on every row the pairs file carries. The
  mirror is checked byte-for-byte against the product's ``fit`` by stage 1's zero-tolerance
  render comparison; ``score`` keeps the paper's whole-prompt truncation (the declared
  deviation lives only in the score path, on over-cap pairs).

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

#: The paper budget: ``MAX_SEQ_LENGTH`` (``rcp_ndcg.retrieval.cross_encoder``), the whole prompt,
#: and ``MAX_QUERY_LENGTH``, the query share the paper's own served path sent as
#: ``max_tokens_per_query`` and the recipe declares as ``client.query_max_tokens``.
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


def fold_instruction(query: str, instruction: str | None) -> str:
    """The served client's fold (``instruction: fold``): ``Task: {instruction}\\nQuery: {text}``.

    The product's own fold render (``rcp_ndcg.inference.clients``' fold via
    ``rcp_ndcg_core._records``, ``Query.format_query``): with an instruction, both sides are
    stripped and the instruction leads (``Task: ...\nQuery: ...``); without one, the query is
    passed through exactly as the pairs file carries it. The reference folds the pairs row's
    instruction the same way, so both sides prompt identically.
    """
    named = (instruction or "").strip()
    if named:
        return f"Task: {named}\nQuery: {query.strip()}"
    return query


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

        The instruction (already folded into ``query`` by the served client, and by :meth:`render`
        and :meth:`score`) rides inside the query text; the paper's own instruction slot is empty
        (``ContextualRerank.instruction`` is None on the paper path).
        """
        return f"{HEAD}{doc}{MID}{query}{TAIL}"

    def render(self, query: str, doc: str, instruction: str | None = None) -> str:
        """The recipe's declared anchor-preserving render for one pair.

        The fixed frame is reserved, the query is cut to its 4096-token share first on overflow
        (the declared ``query_max_tokens``), the document is cut to the remaining room, and the
        " ??" tail is re-attached — the mirror of the served ``fit`` that stage 1's render
        comparison holds byte-equal. (The paper's own score path truncates the whole prompt from
        the right instead — the declared ``anchor_drop_over_cap`` deviation, which lives in the
        score path, never here.)
        """
        folded = fold_instruction(query, instruction)
        return self._fit_pair(folded, doc)

    def score(self, query: str, docs: list[str], instruction: str | None = None) -> list[float]:
        """The raw relevance logit per document, aligned with ``docs`` (the paper's ``predict``).

        The instruction is folded into the query exactly as the served client folds it, then the
        paper's batching runs unchanged: length-descending permutation, the padded-area character
        budget with the 16-document cap, and the OOM back-off. Batching affects throughput only,
        never the per-document score.
        """
        folded = fold_instruction(query, instruction)
        prompts = [self.prompt_text(folded, doc) for doc in docs]

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

    # -- the declared render (the fit mirror) -------------------------------------------------
    def assemble(self, query: str, document: str) -> str:
        """The full rendered prompt around whatever the content spans now hold."""
        return f"{HEAD}{document}{MID}{query}{TAIL}"

    def _fit_pair(self, query: str, document: str) -> str:
        """The pair fit of one (folded) query and its document, as the recipe's budget declares.

        The mirror of the product's ``fit`` for the recipe's declaration (``max_tokens`` 8192,
        ``query_max_tokens`` 4096, ``on_overflow: cut``, the pair template, post-processor tokens
        added): an under-budget pair is returned byte-identical to the uncut render; over the
        budget the query is cut to its share first, then the document to the remaining room, both
        at token boundaries of the original text, with the frame (and the " ??" anchor) always
        re-attached.
        """
        tokenizer = self.tokenizer
        flag = True  # add_special_tokens for the pair shape (the engine's route behaviour)
        cap = self.max_length

        if tokenizer.count(self.assemble(query, document), add_special_tokens=flag) <= cap:
            return self.assemble(query, document)
        # Over budget: the query's span is settled first, to its declared share.
        share = self.query_max_tokens
        q_final = query if tokenizer.count(query) <= share else token_prefix(query, share, tokenizer)
        # The query must leave room for the frame even with an empty document.
        q_final = token_prefix(
            q_final, cap, tokenizer, rendered=lambda piece: self.assemble(piece, ""), add_special_tokens=flag
        )
        q_min = tokenizer.count(self.assemble(q_final, ""), add_special_tokens=flag)
        if q_min >= cap and tokenizer.count(document) > 0:
            raise RuntimeError(
                f"the query fills the pair budget of {cap} tokens and leaves the document nothing: "
                "lower query_max_tokens (or raise max_tokens), so the document keeps a share"
            )
        d_final = token_prefix(
            document,
            cap,
            tokenizer,
            rendered=lambda piece: self.assemble(q_final, piece),
            add_special_tokens=flag,
        )
        return self.assemble(q_final, d_final)

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
        reference = CtxlRerankReference(tokenizer_spec=args.tokenizer)
        rows = [
            {
                "index": index,
                "shape": str(row.get("shape") or "pair"),
                "text": reference.render(
                    row["query"], row["documents"][0] if row["documents"] else "", row.get("instruction")
                ),
            }
            for index, row in enumerate(rows_in)
        ]
    else:
        reference = CtxlRerankReference(tokenizer_spec=args.tokenizer).load(device_name(args.device))
        rows = [
            {
                "index": index,
                "scores": reference.score(row["query"], list(row["documents"]), row.get("instruction")),
            }
            for index, row in enumerate(rows_in)
        ]
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump({"rows": rows}, handle, indent=1)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
