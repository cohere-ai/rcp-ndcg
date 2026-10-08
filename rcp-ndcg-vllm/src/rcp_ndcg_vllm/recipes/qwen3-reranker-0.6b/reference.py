"""Reference implementation for the served recipe ``qwen3-reranker-0.6b`` (Qwen/Qwen3-Reranker-0.6B).

Derived with unchanged behaviour from the paper's exact in-process implementation,
``experiments/paper/rerankers/reference/qwen3.py`` (``QwenOGRerank``, itself moved unchanged from
the former ``rcp-ndcg/src/rcp_ndcg/retrieval/external_rerankers.py`` of the PRE-UNIFIED-INFERENCE tree
(the in-package copy it was extracted from; both moved unchanged when the package stopped carrying
in-process models and the unified-inference merge deleted the old retrieval tree; the paper's
numbers rest on the file at HEAD). The paper pipeline loaded it with
``max_seq_len=8192`` (``MAX_SEQ_LENGTH``, the pre-unified-inference ``retrieval/cross_encoder.py``
constant), the config's
``batch_size`` (16 for this model, the pre-unified-inference ``experiments/paper/rerankers/qwen3_reranker_0_6b.yaml``),
bfloat16 (the pipeline's ``DTYPE``; the class default would be float16) and the config's revision
(pinned here to the recipe's commit, so a moved default checkpoint cannot silently break
equivalence).

The prompt is the model card's chat frame:

- prefix = IM_START + ``system\\n`` + the judge sentence + IM_END + ``\\n`` + IM_START + ``user\\n``
  (39 tokens with the recipe tokenizer);
- pair = ``<Instruct>: {instruction}\\n<Query>: {query}\\n<Document>: {doc}`` (single newlines);
- suffix = IM_END + ``\\n`` + IM_START + ``assistant\\n`` + think-open + ``\\n\\n`` + think-close
  + ``\\n\\n`` (9 tokens).

The instruction is the class's fixed default (the model card's
``config_sentence_transformers.json`` prompt; the paper path never passed a per-row instruction),
and the pair string alone is truncated (``longest_first``) to
``max_length - len(prefix) - len(suffix)`` = 8144 tokens; the prefix and suffix are always
re-attached, so the anchor the model scores at is never dropped. This is the paper's own cut,
NEVER the client's (the rerank client settles an over-share query at its share where the paper
keeps it whole inside the pair cut), so ``reference.known_deviations`` declares only
``over_cap_cut_differs``: the harness reports over-cap pairs non-gating and gates under-cap rows
exactly. The score is ``softmax([no_logit, yes_logit])[yes]`` at the last position
(``true_token_id`` 9693, ``false_token_id`` 2152), a probability in [0, 1].

Run as the harness's reference subprocess (never imported by the harness, which holds no torch)::

    reference.py --mode <render|score> --pairs <file> --out <file> --tokenizer <spec> [--device <d>]

``render`` writes ``{"rows": [{"index", "shape": "pair", "query": str, "documents": [str, ...]}]}``
(the wire's cut content spans under the paper's own cut: the pair truncated at its token budget at
raw character offsets, both anchors reserved so the frame re-assembles unchanged); ``score``
writes ``{"rows": [{"index", "scores": [...]}]}`` on the recipe's ``score_scale: probability``.

Reference environment (``requirements-reference.txt`` beside this file, documented not installed):
torch 2.9.1, transformers 4.57.6, accelerate, flash-attn 2.8.3 (the paper's former ``[local]``
extra pins, from ``experiments/paper/rerankers/reference/requirements.txt`` at HEAD). ``render`` needs the
tokenizer only (transformers, or the ``tokenizers`` library over the same ``tokenizer.json`` when
transformers is absent — stage 1 on CPU); ``score`` needs the weights, the transformers pin and the
device the harness passes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# --- chat-template markers, built without typing them literally -----------------------------
IM_START = chr(60) + "|im_start|>"
IM_END = chr(60) + "|im_end|>"
THINK_OPEN = chr(60) + "think" + chr(62)
THINK_CLOSE = chr(60) + "/" + "think" + chr(62)

JUDGE_TEXT = (
    "Judge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".'
)

#: The fixed frame, built from the markers above: the system turn with the judge sentence, the
#: user turn, and — after the pair — the assistant turn with the empty think block the paper
#: prompt ends with (39 prefix tokens and 9 suffix tokens with the recipe tokenizer).
PREFIX_TEXT = f"{IM_START}system\n{JUDGE_TEXT}{IM_END}\n{IM_START}user\n"
SUFFIX_TEXT = f"{IM_END}\n{IM_START}assistant\n{THINK_OPEN}\n\n{THINK_CLOSE}\n\n"

#: The default instruction (the model card's ``config_sentence_transformers.json`` prompt;
#: identical string in the paper's ``QwenOGRerank.__init__`` and in the served template's default).
DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"

#: The recipe's checkpoint (the recipe ``id``/``revision``; the served model resolves this commit).
DEFAULT_MODEL = "Qwen/Qwen3-Reranker-0.6B"
DEFAULT_REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"

#: The paper budget: ``MAX_SEQ_LENGTH`` (the pre-unified-inference ``retrieval/cross_encoder.py``), the whole prompt.
MAX_SEQ_LENGTH = 8192

#: The paper's per-model batch size (the pre-unified-inference ``qwen3_reranker_0_6b.yaml``).
BATCH_SIZE = 16


def _tokenizer_dir(spec: str) -> str:
    """The tokenizer location the ``--tokenizer`` spec names, in the form transformers loads.

    The spec is the recipe's ``client.tokenizer``: a Hub id with an optional ``@revision`` (the
    revision is passed separately to ``from_pretrained``), a directory of tokenizer files, or a
    ``tokenizer.json`` path (its directory is used).
    """
    candidate = Path(spec).expanduser()
    if candidate.is_file():
        return str(candidate.parent)
    if candidate.is_dir():
        return str(candidate)
    repo, _, _revision = spec.partition("@")
    return repo


class PairTokenizer:
    """The tokenizer surface the reference needs, over the recipe's ``tokenizer.json``.

    The paper path is transformers' ``AutoTokenizer`` (the reference environment of
    ``requirements-reference.txt``, where ``score`` runs). Stage 1 on CPU runs in the harness's
    environment, which carries no transformers; there the same ``tokenizer.json`` loads through the
    ``tokenizers`` library directly — the library transformers' fast tokenizers wrap — giving the
    same ids and truncation for this checkpoint (its post-processor adds no tokens; the research
    lane measured the parts-concatenated and whole-prompt ids equal).

    The ``padding_side`` the paper code sets is ``left``; padding applies to ``score`` batching
    only (``_process_inputs``), never to the rendered prompt text.
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
            self._hub = True

    def convert_tokens_to_ids(self, token: str) -> int:
        """The id of one vocabulary token (``no`` -> 2152, ``yes`` -> 9693)."""
        if self._hub:
            return int(self._fast.convert_tokens_to_ids(token))
        return int(self._fast.token_to_id(token))

    def encode_prefix(self, text: str) -> list[int]:
        """The ids of ``text`` with no post-processor tokens (the frame parts)."""
        if self._hub:
            return list(self._fast.encode(text, add_special_tokens=False))
        return list(self._fast.encode(text, add_special_tokens=False).ids)

    def encode_with_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        """The ids of ``text`` (with the tokenizer's default post-processor) and their character
        offsets into the raw string. The tokenizer's normalizer changes text forms (e.g. NFC), but
        the offsets map back to the characters that were sent."""
        if self._hub:
            encoded = self._fast(text, return_offsets_mapping=True)
            ids = encoded["input_ids"]
            offsets = encoded["offset_mapping"]
            if ids and isinstance(ids[0], list):  # a single string still comes back batch-wrapped
                ids, offsets = ids[0], offsets[0]
            return list(ids), [tuple(pair) for pair in offsets]
        encoding = self._fast.encode(text, add_special_tokens=True)
        return list(encoding.ids), [tuple(pair) for pair in encoding.offsets]

    def encode_truncated(self, texts: list[str], max_length: int) -> list[list[int]]:
        """The ids of every text, truncated ``longest_first`` at ``max_length`` (the paper's call)."""
        if self._hub:
            encoded = self._fast(
                texts,
                padding=False,
                truncation="longest_first",
                return_attention_mask=False,
                max_length=max_length,
            )
            return [list(ids) for ids in encoded["input_ids"]]
        self._fast.no_truncation()
        self._fast.enable_truncation(max_length=max_length, strategy="longest_first")
        try:
            return [list(self._fast.encode(text, add_special_tokens=True).ids) for text in texts]
        finally:
            self._fast.no_truncation()

    def pad(self, inputs: dict[str, Any], *, max_length: int) -> dict[str, Any]:
        """Left-pad a batch to its longest sequence, capped at ``max_length`` (the paper's call)."""
        assert self._hub, "padding is a score-mode operation and runs in the transformers environment"
        return self._fast.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)


class Qwen3RerankerReference:
    """The paper's in-process scoring recipe for Qwen3-Reranker-0.6B, as ``load``/``render``/``score``.

    Behaviour is the paper's (``experiments/paper/rerankers/reference/qwen3.py``, ``QwenOGRerank``):
    the same prompt frame, padding side (left), pair-only ``longest_first`` truncation at
    ``MAX_SEQ_LENGTH`` minus the frame, dtype (bfloat16, the pipeline's), attention implementation
    (flash-attention-2, the class default) and the OOM back-off that splits a batch in half. The
    only changes from that module are the harness's subprocess interface (``load``/``render``/
    ``score`` with the ``--mode`` CLI) and the pinned revision.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        tokenizer_spec: str,
        max_seq_len: int = MAX_SEQ_LENGTH,
        batch_size: int = BATCH_SIZE,
        instruction: str = DEFAULT_INSTRUCTION,
        dtype: str = "bfloat16",
        attn_implementation: str | None = "flash_attention_2",
        revision: str = DEFAULT_REVISION,
    ) -> None:
        # padding_side="left", exactly the paper class's __init__ (see PairTokenizer).
        self.tokenizer = PairTokenizer(tokenizer_spec, revision=revision)
        self.model_name = model_name
        self.revision = revision
        self.max_length = max_seq_len
        self.batch_size = batch_size
        self.instruction = instruction
        self.dtype = dtype
        self.attn_implementation = attn_implementation

        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")  # 2152
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")  # 9693

        self.prefix = PREFIX_TEXT
        self.suffix = SUFFIX_TEXT
        self.prefix_tokens = self.tokenizer.encode_prefix(PREFIX_TEXT)
        self.suffix_tokens = self.tokenizer.encode_prefix(SUFFIX_TEXT)
        self.model: Any = None
        self.device = "cpu"

    # -- the harness's interface ------------------------------------------------------------
    def load(self, device: str) -> Qwen3RerankerReference:
        """Load the checkpoint (first call) and move it to ``device`` (e.g. ``cuda:0`` or ``cpu``).

        The weights load here, in the reference process: the harness imports no torch. dtype is the
        paper pipeline's bfloat16; the attention implementation is the paper class's default.
        """
        from transformers import AutoModelForCausalLM

        if self.model is None:
            model_kwargs: dict[str, Any] = {"dtype": self.dtype, "revision": self.revision}
            if self.attn_implementation is not None:
                model_kwargs["attn_implementation"] = self.attn_implementation
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
            self.model.eval()
        self.model.to(device)
        self.device = device
        return self

    def kept_pair_spans(self, query: str, doc: str, instruction: str | None = None) -> tuple[str, str]:
        """The paper's kept ``(query, document)`` content spans for one pair.

        The pair string is truncated ``longest_first`` at ``max_length - len(prefix) -
        len(suffix)`` tokens (the paper's ``_process_inputs``) at the RAW character offsets of the
        kept tokens — never a ``decode(encode())`` round trip: this checkpoint's normalizer maps
        non-NFC text to NFC, keeping the token ids equal but not the characters. The prefix and
        suffix are never cut (the caller renders them around the spans); the kept spans are the
        verbatim prefixes of the query and document pieces of the kept pair text — the instruction
        and the label text live in the served template's fixed frame and are not part of the spans.
        """
        if instruction is None:
            instruction = self.instruction
        header = f"<Instruct>: {instruction}\n<Query>: "
        mid = "\n<Document>: "
        pair = header + query + mid + doc
        budget = self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        ids, offsets = self.tokenizer.encode_with_offsets(pair)
        kept = pair if len(ids) <= budget else pair[: offsets[budget - 1][1]]
        query_start, query_end = len(header), len(header) + len(query)
        document_start = query_end + len(mid)
        query_span = kept[query_start : min(query_end, len(kept))] if len(kept) > query_start else ""
        document_span = kept[document_start:] if len(kept) > document_start else ""
        return query_span, document_span

    def render_spans(self, query: str, docs: list[str], instruction: str | None = None) -> tuple[str, list[str]]:
        """The wire's cut content spans for one pairs row: the query span and one span per document.

        The cut is :meth:`kept_pair_spans` per pair; the kept query span is the same for every
        document of the row (the paper's right cut lands at a fixed pair-token frontier), and a
        disagreement raises rather than papering over it — stage 1 renders one query span per row.
        """
        if not docs:
            query_span, _ = self.kept_pair_spans(query, "", instruction)
            return query_span, []
        pairs = [self.kept_pair_spans(query, document, instruction) for document in docs]
        query_span = pairs[0][0]
        if any(span != query_span for span, _ in pairs):
            raise RuntimeError(
                "the paper's pair cut kept different query spans for one shared query; "
                "stage 1 renders one query span per row"
            )
        return query_span, [document for _, document in pairs]

    def render(self, query: str, doc: str, instruction: str | None = None) -> str:
        """The exact prompt text for one pair, with the paper truncation applied.

        The pair string is truncated ``longest_first`` at ``max_length - len(prefix) - len(suffix)``
        (the prefix and suffix are never cut, so the anchor always survives) and re-attached. Under
        the budget the raw strings are returned: ``decode(encode(x))`` is not the identity for this
        checkpoint (its normalizer maps non-NFC text to NFC, keeping the token ids equal but not
        the characters), so the kept text is cut at the raw character offset of the last kept token
        instead of decoded from ids.
        """
        if instruction is None:
            instruction = self.instruction
        pair = f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"
        budget = self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        ids, offsets = self.tokenizer.encode_with_offsets(pair)
        if len(ids) <= budget:
            return self.prefix + pair + self.suffix
        return self.prefix + pair[: offsets[budget - 1][1]] + self.suffix

    def score(self, query: str, docs: list[str], instruction: str | None = None) -> list[float]:
        """The probability of "yes" per document, aligned with ``docs`` (the paper's ``predict``)."""
        if instruction is not None and instruction != self.instruction:
            raise ValueError(
                "this reference pins the paper's fixed default instruction (the client's instruction "
                f"mode is none); got {instruction!r}"
            )
        pairs = [f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {doc}" for doc in docs]
        scores: list[float] = []
        import torch

        with torch.no_grad():
            for i in range(0, len(pairs), self.batch_size):
                scores.extend(self._score_pairs(pairs[i : i + self.batch_size]))
        if len(scores) != len(docs):
            raise RuntimeError(f"Qwen3RerankerReference: expected {len(docs)} scores, got {len(scores)}")
        return scores

    # -- the paper's mechanics (unchanged) ---------------------------------------------------
    def _process_inputs(self, pairs: list[str]) -> dict[str, Any]:
        """Tokenize the pairs, truncate each at the pair budget and re-attach the frame.

        Mirrors the paper's ``_process_inputs``: ``padding=False``,
        ``truncation="longest_first"``, ``max_length = max_length - len(prefix) - len(suffix)`` (the
        tokenizer's default ``add_special_tokens``, which adds none for this tokenizer), then the
        prefix and suffix ids around every pair, then left-padding to the batch's longest sequence
        (capped at ``max_length``).
        """
        inputs: dict[str, Any] = {
            "input_ids": self.tokenizer.encode_truncated(
                pairs, self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
            )
        }
        for i, ele in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = self.prefix_tokens + ele + self.suffix_tokens
        inputs = self.tokenizer.pad(inputs, max_length=self.max_length)
        return {k: v.to(self.device) for k, v in inputs.items()}

    def _compute_scores(self, inputs: dict[str, Any]) -> list[float]:
        """``softmax([no, yes])[yes]`` at the last position, as float32 probabilities."""
        import torch

        logits = self.model(**inputs).logits[:, -1, :]
        true_vector = logits[:, self.token_true_id]
        false_vector = logits[:, self.token_false_id]
        stacked = torch.stack([false_vector, true_vector], dim=1)
        scores = torch.nn.functional.log_softmax(stacked, dim=1)[:, 1].exp()
        return scores.float().tolist()

    def _score_pairs(self, pairs: list[str]) -> list[float]:
        """Score one batch, halving it on CUDA OOM (the paper code's back-off)."""
        import torch

        try:
            return self._compute_scores(self._process_inputs(pairs))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(pairs) == 1:
                raise
            mid = len(pairs) // 2
            return self._score_pairs(pairs[:mid]) + self._score_pairs(pairs[mid:])


def main() -> int:
    """The harness's reference CLI: ``--mode render|score`` over a pairs JSONL file."""
    parser = argparse.ArgumentParser(description="the qwen3-reranker-0.6b paper reference")
    parser.add_argument("--mode", required=True, choices=["render", "score"])
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    rows_in = [json.loads(line) for line in Path(args.pairs).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.mode == "render":
        reference = Qwen3RerankerReference(tokenizer_spec=args.tokenizer)
        rows: list[dict[str, Any]] = []
        for index, row in enumerate(rows_in):
            query_span, document_spans = reference.render_spans(
                str(row.get("query", "")), [str(document) for document in row.get("documents", [])]
            )
            rows.append(
                {
                    "index": index,
                    "shape": str(row.get("shape") or "pair"),
                    "query": query_span,
                    "documents": document_spans,
                }
            )
    else:
        reference = Qwen3RerankerReference(tokenizer_spec=args.tokenizer).load(args.device)
        rows = [
            {
                "index": index,
                "scores": reference.score(row["query"], list(row["documents"])),
            }
            for index, row in enumerate(rows_in)
        ]
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump({"rows": rows}, handle, indent=1)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
