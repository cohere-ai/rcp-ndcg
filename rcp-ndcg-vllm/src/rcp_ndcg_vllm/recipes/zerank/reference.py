"""The zerank family's one reference (the paper's ``ZerankRerank``, parameterised by the variant).

Ported unchanged in behaviour from the paper's in-process code,
``experiments/paper/rerankers/reference/zerank.py`` (``ZerankRerank``, adapted from the vendor's
zerank-1 code, Apache-2.0), as the paper config instantiated it: max_seq_len=8192, bfloat16, right
padding. Where the model card and the paper code disagree, the paper code wins:

* score scale: ``sigmoid(yes_logit / 5)`` (probability in [0, 1]).
* context budget: the whole rendered prompt truncates at 8192 tokens from the right; the cards
  advertise 32768. 8192 wins (it is what the paper ran). NOTE: this whole-prompt right cut
  drops the assistant header -- the anchor the score head reads -- on over-cap pairs; the SERVED
  path reserves the anchor and cuts the content instead. The recipe declares this as
  ``reference.known_deviations: [anchor_drop_over_cap]``; stage 2 reports over-cap pairs outside
  the gates. It is deliberately NOT copied here from the served path: this file is the paper's
  scoring, and the deviation is the paper's.
* padding side: right (set explicitly; also the tokenizer default).
* trust_remote_code: not needed.

One file for all three sizes (decision 34): the checkpoint (``model``, ``revision``) comes from the
resolved recipe the harness passes as ``--recipe``; the reference imports nothing of the product.

Subprocess contract (the harness invokes)::

    reference.py --mode render|score --pairs <file> --out <file> --tokenizer <spec> \
        --recipe <resolved-recipe.json> [--device <d>]

- ``--mode score``   -> ``{"rows": [{"index", "scores": [float, ...]}]}`` on the probability
  scale (stage 2; needs torch and the checkpoint weights).
- ``--mode render``  -> ``{"rows": [{"index", "shape": "pair", "query", "documents"}]}`` under the
  paper's own cut (:func:`paper_spans`; the ``tokenizers`` library over the recipe's tokenizer.json,
  the same file the harness and the engine read; no torch, so stage 1 runs on CPU).
- ``--mode embed``   -> refused: a reranker has no embed mode.

The frame is built from the tokenizer's added tokens by name (``im_start``, ``im_end``), never
typed literally.  Reference environment: ``reference.in``/``reference.lock`` beside this file, in ITS
OWN python (never the harness's process, never the engine image).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MAX_SEQ_LEN = 8192  # the paper's MAX_SEQ_LENGTH: the whole-pair budget, the recipes' client.max_tokens
BATCH_SIZE_TOKENS = 15_000  # the paper's char-length batching budget (class default)
DTYPE = "bfloat16"  # the paper's dtype; the checkpoint configs agree
YES_TOKEN = "Yes"
YES_TOKEN_ID = 9454  # the checkpoints' "Yes" token id (1_LogitScore/config.json true_token_id)

TOKENIZER_FILE = "tokenizer.json"
#: Whether the scoring route counts the tokenizer's post-processor tokens. The engine adds them
#: (add_special_tokens=True); for these tokenizers they are measured to add none (plain strings and
#: the empty render alike), so the frame overhead below is the whole fixed cost either way.
#: Declared, not assumed.
ADD_SPECIAL_TOKENS = True


def _recipe_dir() -> Path:
    """The directory this reference and its ``family.yaml`` live in."""
    return Path(__file__).resolve().parent


def _split_spec(spec: str) -> tuple[str, str | None, bool]:
    """A tokenizer spec into (repo-or-path, revision, is_path).

    The same resolution rule as the product's ``rcp_ndcg.data.tokenizer._local_path``: a path is
    anything that exists, is absolute or relative, or ends in ``.json``."""
    if spec.startswith(("/", "./", "../", "~")) or spec.endswith(".json") or Path(spec).exists():
        return spec, None, True
    repo, _, revision = spec.partition("@")
    return repo, revision or None, False


def _tokenizer_file(spec: str) -> Path:
    """The tokenizer.json file ``spec`` names: a local path as given, else downloaded from the Hub.

    A spec that names a local file (the product's path rule) but exists nowhere is refused with the
    missing-file message -- never a Hub download attempt of a path-shaped repository id."""
    path, revision, is_path = _split_spec(spec)
    candidate = Path(path).expanduser()
    file = candidate / TOKENIZER_FILE if candidate.is_dir() else candidate
    if file.is_file():
        return file
    if is_path:
        raise SystemExit(f"no tokenizer file at {file}")
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(path, TOKENIZER_FILE, revision=revision))


def _load_backend_tokenizer(spec: str):
    """The reference's counting tokenizer: the `tokenizers` library over the recipe's tokenizer.json."""
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(_tokenizer_file(spec)))


def _added_tokens(tokenizer) -> dict[str, str]:
    """The added vocabulary: the special's bare name (and its literal form) to its literal text."""
    tokens: dict[str, str] = {}
    for token in tokenizer.get_added_tokens_decoder().values():
        content = token.content
        name = content[2:-2] if content.startswith("<|") and content.endswith("|>") else content
        tokens.setdefault(name, content)
        tokens.setdefault(content, content)
    return tokens


def _special_text(tokenizer, name: str) -> str:
    """The literal text of the added token named ``name`` (as the recipe's ``{special:<name>}`` resolves)."""
    tokens = _added_tokens(tokenizer)
    try:
        return tokens[name]
    except KeyError:
        known = sorted({key for key in tokens if not key.startswith("<|")})
        raise SystemExit(f"the tokenizer has no added token named {name!r}; its added tokens are {known}") from None


def _frame(tokenizer) -> tuple[str, str, str]:
    """The pair frame: (head, middle, tail) around the query and document spans -- the paper's prompt
    (the checkpoint's chat template with the generation prompt, measured equal), the recipes'
    declared pair shape and the template file's zerank branch. Specials resolved from the tokenizer."""
    im_start = _special_text(tokenizer, "im_start")
    im_end = _special_text(tokenizer, "im_end")
    head = f"{im_start}system\n"
    mid = f"{im_end}\n{im_start}user\n"
    tail = f"{im_end}\n{im_start}assistant\n"
    return head, mid, tail


# -------------------------------------------------------------------------------------------
# render mode (stage 1): the paper's own cut, written as the content spans the harness compares.
# The format is the harness's rerank render contract; the cut is the paper's ZerankRerank, never
# the client's (no query share, no settle rule, no anchor reservation).
# -------------------------------------------------------------------------------------------


def paper_spans(tokenizer, query: str, document: str) -> tuple[str, str]:
    """The paper's kept ``(query, document)`` content spans for one pair.

    The paper's ``ZerankRerank._format_inputs`` renders ``[{system: query.strip()}, {user:
    doc.strip()}]`` through the checkpoint's chat template with the generation prompt (that render
    is exactly this recipe's frame, ``head + query + mid + document + tail``: measured on the
    pinned checkpoints' own templates), and ``_batch_logits`` tokenizes the WHOLE prompt with
    ``truncation=True, max_length=MAX_SEQ_LEN`` -- a right cut.  The kept text is located at the raw
    character offset of the last kept token (never a ``decode(encode())`` round trip); the spans
    are the kept parts of the stripped query and document.  Under the cap they are the stripped
    texts, byte for byte; over it the cut drops the tail first -- the assistant header, the anchor
    the score is read from -- then the document's end (and, for a query that fills the cap alone,
    the query's): the recipe's declared ``anchor_drop_over_cap``.
    """
    head, mid, tail = _frame(tokenizer)
    query, document = query.strip(), document.strip()
    prompt = head + query + mid + document + tail
    encoded = tokenizer.encode(prompt, add_special_tokens=ADD_SPECIAL_TOKENS)
    if len(encoded.ids) <= MAX_SEQ_LEN:
        return query, document
    kept = prompt[: encoded.offsets[MAX_SEQ_LEN - 1][1]]
    query_start = len(head)
    document_start = query_start + len(query) + len(mid)
    return kept[query_start : query_start + len(query)], kept[document_start : document_start + len(document)]


def render_rows(tokenizer, pairs: list[dict]) -> list[dict]:
    """``--mode render``: per pairs row, ``{"index", "shape": "pair", "query", "documents"}`` under
    the paper's cut (:func:`paper_spans`).  The recipe declares ``instruction: none``: a row's
    instruction is not part of this model's input.  The paper's right cut lands at one token
    frontier of the shared query, so the kept query span is the same for every document of a row;
    a disagreement is refused rather than papered over."""
    rows = []
    for index, row in enumerate(pairs):
        shape = str(row.get("shape") or "pair")
        if shape != "pair":
            raise SystemExit(f"row {index}: shape {shape!r} is not declared; this recipe declares the pair shape")
        query = str(row["query"])
        spans = [paper_spans(tokenizer, query, str(document)) for document in row["documents"]]
        query_span = spans[0][0] if spans else paper_spans(tokenizer, query, "")[0]
        if any(span != query_span for span, _ in spans):
            raise SystemExit(f"row {index}: the paper's cut kept different query spans for one shared query")
        rows.append({"index": index, "shape": shape, "query": query_span, "documents": [d for _, d in spans]})
    return rows


# -------------------------------------------------------------------------------------------
# The paper's in-process reference (experiments/paper/rerankers/reference/zerank.py, unchanged
# in behaviour; the interface the harness documents: load(device), render(...), score(...)).
# -------------------------------------------------------------------------------------------


class ZerankReference:
    """``load(device)`` / ``render(query, doc, instruction)`` / ``score(query, docs, instruction)``.

    The paper's ``ZerankRerank``: chat template with system=query and user=document (both
    stripped), the assistant generation prompt appended, the "Yes"-token logit read at the last
    non-pad position, divided by 5 and sigmoided. The whole rendered prompt truncates at
    ``MAX_SEQ_LEN`` from the right at forward time -- the known ``anchor_drop_over_cap`` deviation
    the recipe declares, never copied into the served path.
    """

    def __init__(self) -> None:
        self.tokenizer = None
        self.model = None
        self.device = None
        self.yes_token_id: int | None = None

    # ------------------------------------------------------------------ load
    def load(self, device: str | None = None, *, tokenizer_spec: str, model: str, revision: str) -> ZerankReference:
        """Load the checkpoint the way the paper's ``ZerankRerank.__init__`` does (torch + transformers).

        The checkpoint is the resolved recipe's ``model`` at its ``revision`` (the variant the
        harness named); the tokenizer is the recipe's own spec."""
        torch, automodel, autotokenizer = _reference_stack()
        repo, revision_spec, _ = _split_spec(tokenizer_spec)
        self.tokenizer = autotokenizer.from_pretrained(repo, padding_side="right", revision=revision_spec)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = automodel.from_pretrained(model, dtype=torch.bfloat16, revision=revision)
        self.model.eval()

        self.yes_token_id = self.tokenizer.encode(YES_TOKEN, add_special_tokens=False)[0]
        assert self.yes_token_id == YES_TOKEN_ID, self.yes_token_id

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        return self

    # ----------------------------------------------------------------- render
    def render(self, query: str, doc: str, instruction: str | None = None) -> str:
        """The paper's prompt TEXT for one pair: the checkpoint chat template over
        [system=query.strip(), user=document.strip()] with the generation prompt appended.

        The family takes no instruction parameter (the caller folds any into *query*); this
        reference ignores one, exactly like the served client with ``instruction: none``.
        Truncation is NOT applied here -- the paper truncates the whole rendered prompt at forward
        time (``_batch_logits``, right side, ``MAX_SEQ_LEN``); ``score`` reproduces that.
        """
        if self.tokenizer is None:
            raise RuntimeError("call load() before render()")
        return self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": query.strip()},
                {"role": "user", "content": doc.strip()},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    # ------------------------------------------------------------------ score
    def score(self, query: str, docs: list[str], instruction: str | None = None) -> list[float]:
        """``sigmoid(yes_logit / 5)`` per document, aligned with *docs*.

        Port of the paper's ``predict`` + ``_batch_logits``: length-descending permutation for
        tight token buckets, right padding, the score read at the last non-pad position, the whole
        rendered prompt truncated at ``MAX_SEQ_LEN`` from the right.
        """
        import numpy as np
        from scipy.special import expit

        if self.tokenizer is None or self.model is None or self.yes_token_id is None:  # pragma: no cover - load() ran
            raise RuntimeError("call load() before score()")
        if instruction is not None:
            # The family takes no instruction; the paper pipeline never passed one. Dropping it is
            # what the served client (instruction: none) does too.
            instruction = None
        texts = [self.render(query, doc, instruction) for doc in docs]

        # Length-descending permutation for tight token buckets (paper, char-length budget).
        permutation = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
        sorted_texts = [texts[i] for i in permutation]

        batches: list[list[str]] = []
        max_length = 0
        for text in sorted_texts:
            text_len = len(text)
            if not batches or (len(batches[-1]) + 1) * max(max_length, text_len) > BATCH_SIZE_TOKENS:
                batches.append([])
                max_length = 0
            batches[-1].append(text)
            max_length = max(max_length, text_len)

        all_logits: list[float] = []
        for batch in batches:
            all_logits.extend(self._batch_logits(batch))

        scores = expit(np.asarray(all_logits)).tolist()
        result = [0.0] * len(docs)
        for orig_idx, score in zip(permutation, scores, strict=True):
            result[orig_idx] = score
        return result

    def _batch_logits(self, batch: list[str]) -> list[float]:
        """One padded forward; on CUDA OOM, halve and recurse (the paper's backoff)."""
        import torch

        try:
            with torch.no_grad():
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    return_tensors="pt",
                    truncation=True,
                    max_length=MAX_SEQ_LEN,
                )
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                outputs = self.model(**inputs, use_cache=False)
                attention_mask = inputs["attention_mask"]
                last_positions = attention_mask.sum(dim=1) - 1
                batch_indices = torch.arange(outputs.logits.shape[0], device=self.device)
                last_logits = outputs.logits[batch_indices, last_positions]
                yes_logits = last_logits[:, self.yes_token_id]
                return [float(value) / 5.0 for value in yes_logits]
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(batch) == 1:
                raise
            mid = len(batch) // 2
            return self._batch_logits(batch[:mid]) + self._batch_logits(batch[mid:])

    def embed(self, texts: list[str], role: str) -> None:  # pragma: no cover - not applicable
        """A reranker has no embed mode."""
        raise NotImplementedError("zerank is a reranker, not an embedder")


def _reference_stack():
    """torch and transformers, imported only where they are needed (the reference environment)."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as error:
        raise SystemExit(
            "score mode needs the reference environment (torch + transformers + the checkpoint "
            "weights); stage 2 runs it on the GPU wave. Pin recipes/zerank/reference.lock "
            "in that environment."
        ) from error
    return torch, AutoModelForCausalLM, AutoTokenizer


# -------------------------------------------------------------------------------------------
# The subprocess CLI the harness invokes.
# -------------------------------------------------------------------------------------------


def main() -> int:
    """The reference CLI: ``--mode render|score`` over the pairs file, the mode's JSON to ``--out``."""
    parser = argparse.ArgumentParser(description="the zerank family reference (the paper's ZerankRerank)")
    parser.add_argument("--mode", required=True, choices=["render", "score", "embed"])
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tokenizer", required=True, help="the recipe's tokenizer spec (repo@revision or a path)")
    parser.add_argument(
        "--recipe",
        required=True,
        help="the resolved recipe JSON the harness passed (the variant's model and revision)",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.mode == "embed":
        raise SystemExit("zerank is a reranker: the reference speaks render and score, not embed")

    model, revision = _model_spec_from_file(args.recipe)
    pairs = [json.loads(line) for line in Path(args.pairs).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.mode == "render":
        tokenizer = _load_backend_tokenizer(args.tokenizer)
        rows = render_rows(tokenizer, pairs)
        output = {"rows": rows}
    else:
        reference = ZerankReference().load(args.device, tokenizer_spec=args.tokenizer, model=model, revision=revision)
        rows = []
        for index, row in enumerate(pairs):
            scores = reference.score(row["query"], list(row["documents"]), row.get("instruction"))
            rows.append({"index": index, "scores": scores})
        output = {"rows": rows}
    Path(args.out).write_text(json.dumps(output, indent=1) + "\n", encoding="utf-8")
    return 0


def _model_spec_from_file(recipe_path: str) -> tuple[str, str]:
    """The resolved recipe's ``(model, revision)`` -- the checkpoint the score mode loads."""
    recipe = json.loads(Path(recipe_path).read_text(encoding="utf-8"))
    return str(recipe["model"]), str(recipe["revision"])


if __name__ == "__main__":
    raise SystemExit(main())
