# Report RF-research: the recipe root causes (CPU research for the recipe-fix lane)

**Status:** DONE. Deliverable: `handover/specs/recipe-root-causes.md` (403 lines), plus the reproduction
scripts in a scratch directory outside the repository. The lane changes no product or recipe file.

## Commits

- `24cf0abe` handover: recipe root causes -- the six GPU-E1 items researched on CPU (pooling paths, the tokenizer
  sidecar seam, the blank-document policy, the score-head dtype, the pplx warm-up input and the 2^31 bound)
- `ff3b76c7` handover: the root-cause spec's verifier round -- the ctxl-1b prompt is not the tokenizer's, the
  merged gate_up bound, the ST 6.1.0 reproduction, full recipe paths and the citation/test touch-points
- `a0f53297` Merge rfc-0001 into lane/rf-research (the current tip, so the spec cites the tree it will be read
  against)
- `d1a676a1` handover: the root-cause spec's round-2 sweep -- the qwen3 test filename, the noise-probe
  attribution, the evidence-file and line citations, and no scratch paths in the committed text

## What changed

One spec file, per brief item:

1. **zembed-1-embedding.** The engine's pooling path is correct: vLLM v0.31.0 resolves `LAST` + the fp32 head +
   L2 from the checkpoint's sentence-transformers metadata, and the card's own pipeline matches it — the
   recorded GPU engine vector equals the card's `encode_query` vector at cosine 0.999884, and all 24 pairs rows
   / 68 vectors sit at min 0.999203 on CPU. The E1 0.08-0.4 range is reproduced by the *reference* environment:
   the node's shared reference venv installs an unbounded `sentence-transformers`, and 6.1.0 (suffix dropped)
   gives min 0.081024 / max 0.833723 over the same 68 vectors. Fix: the per-recipe reference requirements, not
   the recipe's pooling.
2. **octen-embedding-8b.** Input ids, the pooled token and the appended anchor 151643 are identical on both
   sides on every pairs row (136 text/mode checks, 0 divergent, 0 wrong tails), so the E1 template/EOS/pooling
   hypothesis is refuted; the residual is engine-vs-reference numerics that needs a like-for-like GPU run.
3. **ctxl-rerank-v2-instruct-multilingual-1b.** `tokenizer_config.json`'s `pad_token: "+"` makes HF
   AutoTokenizer add the single character `+` as an added token; the product's bare-`tokenizers` load does not,
   so `" +"` is one token client-side and two engine-side. Reproduced exactly (563/565, 179/184, 676/686);
   ctxl-2b is latent (same files, no `" +"` in its pairs), ctxl-6b and every other recipe are clean. Fix: the
   product tokenizer load applies the sidecars; the tokenizer identity must cover them.
4. **jina-reranker-v3.** The paper drops a document whose `text.strip()` is empty; the product's `omit_zero`
   drops only `text == prefix`, so the whitespace-only row is sent (served 0.099 vs reference 0.0). Fix: one
   declared policy value for blank documents, declared by jina only (topk's referent is different).
5. **Reranker score precision.** vLLM's pooling/classify head is fp32 (`head_dtype` default for
   `runner_type == "pooling"`), while every one of the seven references computes the score in the model's bf16
   (the card's/paper's path). Fix: `serve.hf_overrides: {head_dtype: model}` per family, references unchanged;
   the recipe contract tests re-pin with it.
6. **pplx-embed-v2-context-9b-preview.** The engine's pooling warm-up sends a real prefill request with
   `prompt_token_ids = list(range(2))` = `[0, 1]`, which the plugin's all-zero-only dummy check misses. The
   merged `gate_up` tensor is 24576 wide: if its GEMM output is int32-indexed the `max_model_len` bound is
   87381 elements / 43690 bytes, not the post-activation half's 174762 — E2 decides; the declared 131072 sits
   between them. Fix: the plugin's warm-up rule (first id 0) plus the GPU confirmation.

## Verification

Round 1 — two fresh-context verifiers on `24cf0abe`, lens A (correctness) and lens B (regressions and
hygiene), on DeepSeek-V4.1-flash (xhigh), each told the other exists. Both reproduced the load-bearing numbers
with their own scripts:

- **A: FAIL.** One major: the spec's claim that ctxl-1b's 0.249 score gap came from a tokenizer-changed prompt
  is false — the served template and the reference render the same string on all 70 pairs, so the gap is the
  same numerics class as the 2b/6b. Minors: the summary's pplx bound stated as fact; an unnecessary
  second-defect hypothesis for zembed (ST 6.1.0 alone reproduces the range); three line drifts; a missing
  octen script and an over-generalised bf16-trend sentence; the wrong docs page; "staged but never installed";
  an incomplete sidecar field list; the tokenizer-identity invariant left false.
- **B: FAIL.** One blocker: eight recipe citations written without the `rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/`
  prefix (the drift-catalog trap). One major: the branch was behind the current `rfc-0001`. Minors: the
  contract tests that re-pin with the head_dtype change, the pplx recipe test pins, the CHANGELOG/docstring
  obligations, the `load_tokenizer` touch-point and the network gate, the topk "Document:" wording, provenance
  references, the pplx deviation kind, and the pplx tokenizer check (which B then ran under transformers 5.17:
  0/64 and 0/87 divergent).
- **What I did:** fixed every finding in `ff3b76c7` and merged the current `rfc-0001` in `a0f53297` (clean, no
  conflicts; every cited path and line re-checked after the merge).

Round 2 — one fresh confirmation verifier (lens A+B) on `a0f53297`: **PASS**, with eight minor sweep items
(wrong qwen3 test filename, noise-probe attribution, an evidence-file slip, an ST/transformers version
mismatch, scratch paths in the committed text, two line nits, a 2e-4 rounding). All eight fixed in `d1a676a1`;
the verifier independently re-derived the ctxl divergence, the octen id equality, the pplx tokenizer result and
the zembed recorded-vector cosine, and confirmed every round-1 fix and the scope (`git diff` shows only
`handover/`).

## Checks

The gate on `d1a676a1` (`bin/gate lane/rf-research`, slot 2):

```
ruff-check exit=0 All checks passed!
ruff-format exit=0 538 files already formatted
basedpyright exit=0 0 errors, 0 warnings, 0 notes
pytest exit=0 3269 passed, 93 skipped in 55.66s
contract-docs exit=0 287 passed, 52 skipped in 42.42s
mkdocs exit=0 Documentation built in 1.43 seconds
test-pkg exit=0 570 passed, 225 skipped in 349.63s
recipes exit=0 recipes: no failure outside the baseline (34 baseline failures remain, 0 fixed; pytest exit 1)
vllm-pkg exit=0 1 passed in 1.56s
vllm-models exit=0 70 passed, 7 skipped in 87.07s
run_all exit=0 leaderboards: 1022 checks, 987 match, 35 known deviations, 0 failed
                human study: 67 checks, 67 match, 0 failed; external LLM judges: 82 checks, 82 match, 0 failed
public-names exit=0 public-names: clean (2 baselined hits remain)
clean exit=0 clean
GATE: PASS
```

Reproduction (CPU, no GPU): the spec's six measurements, run and re-run by both verifier lenses — the zembed
recorded-vector cosine, the zembed stage-2 68-vector comparison under 5.3.0 and 6.1.0, the octen id/anchor
check, the ctxl-1b divergence and its plus-sign mechanism, the blast-radius scan over all 19 recipes, the pplx
tokenizer check under transformers 5.17, the pplx warm-up `[0, 1]` reproduction, and the vLLM tag line and
arithmetic checks.

## Open questions

- **The tokenizer identity** (item 3): `TextTokenizer.sha256` is the SHA-256 of `tokenizer.json` alone. Once the
  sidecars decide the effective vocabulary, the module docstring's "two passes whose judges tokenize differently
  never pool" invariant needs the sidecar content in the identity (or the fields recorded) — an owner call.
- **The zembed reference's robustness**: keep it dependent on the remote `tokenize` (pinned ST 5.3.x) or
  assemble the card's own prompt in the reference so no ST version can silently change it. Decision 9 allows
  either; the first needs the node to honour per-recipe requirements.
- **The pplx `max_model_len`**: the declared 131072 is below the post-activation bound (174762) but above the
  merged gate_up bound (87381 elements / 43690 bytes) under the strict interpretation; E2's real-request run
  decides, and the deviation kind (`over_cap_cut_differs` for a cap-only difference) should be confirmed
  against the harness.
- **The octen like-for-like decision**: the spec assigns the fp32-engine diagnostic to E2; if the residual
  survives it, the bound is declared with the measured evidence.
- **The head_dtype override per family**: if E2 shows the p99 still failing for one size (the 8b was 90.9 %), the
  alternative is a declared bound; the spec keeps the references faithful and the decision in E2.

## CHANGELOG entry

None: the lane changes no product or recipe file (handover scaffolding only).

## Public surface changes

None.

## Files outside scope

None.

## For the next lanes

- **recipe-fix**: the six sections of the spec are the fix directions; items 1 and 2 must *not* be "fixed" on
  the served side (the spec explains why), item 3's failing test must use a cached fixture or the network gate,
  and items 5/6 re-pin the recipe contract tests named in the spec.
- **ref-envs**: the shared reference requirements (`sentence-transformers>=3.0`) versus the per-recipe pins is
  the zembed root cause; the fix belongs there.
- **E2**: the four GPU confirmations the spec names — the zembed reference vectors under a pinned ST, the octen
  like-for-like run, the seven reranker gaps under `head_dtype: model`, and the pplx 131072 real-request run
  (plus the plugin's `[0, 1]` warm-up).
