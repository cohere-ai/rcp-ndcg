# Recipe root causes: the CPU research behind the recipe-fix lane

**Scope.** Six findings from the first GPU wave (`handover/reports/GPU-E1.md` and the E1 findings kept outside
the repository) researched on
CPU against the current tree (`rfc-0001` tip), the vLLM **v0.31.0** tag (`db9527a46873454610df6dbedf79a36d6bf1a7f6`),
the checkpoints at their pinned revisions and the product client. Every mechanism carries a `file:line`; every
number is either read from the checkpoint or measured in a scratch directory outside the repository, and the
script that measured it is named. This document is the input to the recipe-fix lane; it changes no product or
recipe file.

**What this document is not.** It is not a diff. Items are marked **fix** (a change the recipe-fix lane should
make), **refuted** (the E1 hypothesis does not hold; nothing to fix on the served side) or **open** (a
measurement the GPU wave must take). Where the research refutes an E1 diagnosis, the evidence is given in full
so the lane does not "fix" a correct path.

## 0. Summary

| # | Item | Root cause | Verdict |
|---|---|---|---|
| 1 | zembed-1-embedding vectors | The engine's pooling path is correct (LAST + fp32 head + L2 on the last hidden state); the E1 reference did not run the card's pipeline — the node's shared reference venv installs an **unbounded** `sentence-transformers`, and ST 6.1.0 (which bypasses the remote `tokenize` that appends the pooled suffix) reproduces E1's whole 0.08-0.4 range over the same 68 vectors | **refuted** (engine) + **fix** (reference environment) |
| 2 | octen-embedding-8b min cosine 0.9936 | Input ids, the pooled token and the appended anchor 151643 are **identical** on both sides on every pairs row (tiny, special-token spellings, base64, at_budget); the residual is engine (vLLM bf16 kernels, fp32 score path unused for embed) vs reference (HF bf16) numerics, with a shape the same-path bf16 noise does not show | **refuted** (template/EOS/pooling) + **open** (like-for-like GPU comparison) |
| 3 | ctxl-1b `/tokenize` +2..10 tokens | `tokenizer_config.json`'s `pad_token: "+"` makes HF AutoTokenizer add the single character `+` as an added token; the product's bare-`tokenizers` load of `tokenizer.json` does not, so `" +"` is one BPE token client-side and two tokens engine-side. Reproduced exactly (563/565, 179/184, 676/686) | **fix** (product tokenizer load) |
| 4 | jina-reranker-v3 whitespace-only document | The paper drops a document whose `text.strip()` is empty (`jina.py:66`); the product's `omit_zero` drops only `text == prefix`, so a whitespace-only document is sent and scored | **fix** (one declared policy) |
| 5 | Reranker score gaps | vLLM's pooling/classify head is **fp32** (`head_dtype` default for `runner_type == "pooling"`); every reference computes the score in the model's **bf16**. `serve.hf_overrides: {head_dtype: model}` makes the engine like-for-like; the references stay faithful (card/paper) | **fix** (serve override per family) |
| 6 | pplx-embed-v2-context-9b-preview | The engine's pooling warm-up sends a real prefill request with `prompt_token_ids = list(range(2))` = `[0, 1]` (`gpu/warmup.py:255-257`), which the plugin's all-zero-only dummy check misses; the merged `gate_up` tensor is 24576 wide, so **if** its GEMM output is int32-indexed the bound is 87381 elements / 43690 bytes, not the post-activation half's 174762 — E2 decides | **fix** (plugin warm-up rule) + **open** (131072 confirmation) |

## 1. zembed-1-embedding: the served vector is the card's; the reference was not

### 1.1 The two paths, from source

**Engine (vLLM v0.31.0, `--runner pooling --convert embed`, `pooler_config {}`).** The checkpoint's
sentence-transformers metadata is read before the architecture registry: `get_pooling_config`
(`vllm/transformers_utils/config.py:982-1063`) reads `modules.json`, finds the `Pooling` module
(`_ST_POOLING_MODULE_TYPES`, `:57-60`), reads `1_Pooling/config.json`, maps `pooling_mode_lasttoken: true` through
`parse_pooling_type` (`:1066-1077`, `"lasttoken" -> "LAST"`) to `seq_pooling_type = "LAST"` and sets
`use_activation` from the `Normalize` module's presence. `ModelConfig` merges that under the user's
`pooler_config` (`vllm/config/model.py:740-757`), so the E1 log's
`seq_pooling_type='LAST', use_activation=True` is this resolution. The embed conversion builds
`EmbeddingPoolerHead(head_dtype=model_config.head_dtype, projector=_load_st_projector(...), activation=PoolerNormalize())`
(`vllm/model_executor/layers/pooler/seqwise/poolers.py:97-108`, the head at `:102-106`); `_load_st_projector`
(`vllm/model_executor/models/adapters.py:40-68`) returns `None` here because `modules.json` carries no `Dense`
module, so **`projections.safetensors` is never loaded** — it is not referenced by `modules.json`, and no vLLM
path reads it. The head casts the pooled row to fp32 and L2-normalises it (`seqwise/heads.py:75-83` +
`PoolerNormalize`). `LastPool` takes `hidden_states[pooling_cursor.last_token_indices_gpu]`
(`seqwise/methods.py:50-57`), and the cursor's index is `cumsum[1:] - 1` over the scheduled tokens
(`vllm/v1/pool/metadata.py:168-176`) — the last token of each sequence. `as_embedding_model`'s own docstring
states it: "the embeddings of the whole prompt are extracted from the normalized hidden state corresponding to
the last token" (`adapters.py:250-255`).

**Card (the checkpoint's published pipeline).** `modules.json` = `ZembedTransformer` + `Pooling` + `Normalize`;
`1_Pooling/config.json` = `pooling_mode_lasttoken: true, include_prompt: true`;
`config_sentence_transformers.json` = the query/document prompts plus `suffix: "<|im_end|>\n"`;
`modeling_zembed.py` appends that suffix in `ZembedTransformer.tokenize`; sentence-transformers' lasttoken
branch gathers the last position with `attention_mask == 1` (`sentence_transformers/models/Pooling.py:210-236`)
and `Normalize` L2-normalises. With `include_prompt: true` the prompt mask is not applied
(`Pooling.py:142-152`), so the pooled position is the suffix's trailing newline — the same position vLLM pools.

So on paper the two paths agree on: the layer (the final normalised hidden state), the position (the last
token), the suffix being in the engine's input (the client renders it; stage 1's token-id check passes), the
normalisation (L2), and the projections file (unused by both at dim 2560).

### 1.2 CPU reproduction (the real 4B checkpoint at its pinned revision)

The checkpoint (`zeroentropy/zembed-1-embedding@cf13c81f3274394053d166740294f7eea4586f7a`, ~8 GB) was downloaded
into the machine's Hugging Face cache. Scripts (a scratch directory outside the repository):
`zembed_probe.py`, `zembed_st_probe.py`, `zembed_st53_probe.py`, `zembed_stage2_cpu.py`.

1. **The recorded engine vector against the card.** The observation corpus holds one real engine response from
   vLLM v0.31.0 on GPU (`rcp-ndcg-test/corpora/vllm-0.31.0/zembed-1-embedding/adbcf96f…/records.jsonl.gz`), for
   the rendered query prompt. Against it:
   - the direct transformers path (base model, last hidden state, fp32 cast, L2): cosine **0.999890**;
   - the card's `SentenceTransformer.encode_query` under the node's reference environment
     (sentence-transformers 5.3.0 + transformers 5.17.0, CPU, bf16): cosine **0.999884**;
   - mean pooling 0.606; the second-to-last position 0.697; the raw text 0.666; the document prompt 0.695.
2. **All 24 pairs rows, engine analogue vs card** (the stage-2 CPU script): min cosine **0.999203**
   (row 23 `length:at_budget`, document), every row ≥ 0.999. So the served vector is the card's on the current
   pairs file within bf16 noise.
3. **What does *not* match.** `encode_query` under sentence-transformers **6.1.0** (CPU)
   gives cosine **0.636370** — the suffix is dropped (the 5.4+ preprocess-first pipeline bypasses the remote
   `tokenize`; the recipe's own `requirements-reference.txt` documents this), and over the 24 pairs rows its
   68-vector comparison measured min 0.081024 / p10 0.262788 / max 0.833723 — E1's 0.08-0.4 class exactly. For
   reference, mean pooling of the *raw* text measured 0.0890 against the recorded engine vector.

### 1.3 Root cause and fix

The node builds **one** shared reference venv from `rcp-ndcg-vllm/requirements-reference.txt`
(`rcp-ndcg-test/src/rcp_ndcg_test/jobs/bootstrap.sh:564-566`: `${REFERENCE_REQUIREMENTS:-$STAGE_DIR/requirements-reference.txt}`),
and that file pins `sentence-transformers>=3.0` with **no upper bound**. The per-recipe file
(`rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/zembed-1-embedding/requirements-reference.txt`, `>=5.3,<5.4`) is never
read by the job — `rcp-ndcg-test/src/rcp_ndcg_test/jobs/rc_build.sh:134` stages the shared file only, and
`run_wave.py` takes a single `--reference-python`. E1 therefore ran the zembed reference on an ST whose pipeline
silently changes: ≥ 5.4 drops the suffix, and **that alone reproduces E1's whole class** — ST 6.1.0 over the
current 24 pairs rows / 68 vectors measured min cosine **0.081024**, p10 0.262788, max 0.833723, against min
0.999203 under ST 5.3.0 (the round-1 verifier's stage-2 reproductions: `0.999203` under 5.3.0 and
`0.081024` under 6.1.0). No second defect
(lost prompt or pooling) is needed. Either way the reference was **not** the card's pipeline; the recipe's pooling
path is correct.

**Fix (owned by recipe-fix + ref-envs):**
- make the node's reference environment follow the recipe's own `requirements-reference.txt` (the per-recipe
  reference environments the ref-envs lane is building), or pin the shared file to the intersection every
  reference accepts — for zembed that means `sentence-transformers>=5.3,<5.4`;
- optionally make the zembed reference robust to the ST version by assembling the card's own prompt
  (`config_sentence_transformers.json`'s prompt + text + suffix) and running the model directly — the remote
  module's only job is that suffix append and the whole-prompt truncation, so this stays faithful; owner call;
- record the E1 outcome and the reference-environment requirement in the recipe notes (recipe-fix item 14); the
  note's "~1e-3" claim is consistent with the engine-vs-card measurement but was never what E1 compared.

**What the GPU wave must confirm:** stage 2 with the current pairs file, with the reference venv's ST version
recorded (must be 5.3.x for the published path), and the per-row vectors of both sides dumped (the harness's
per-vector cosines plus the reference's `--mode embed` JSON) — a min cosine ≥ 0.999 is the pass; anything lower
must name which side moved. No recipe pooling change is to be made on the current evidence.

## 2. octen-embedding-8b: ids, anchor and pooled token are identical; the residual is numerics

### 2.1 Mechanism check, per the brief

The recipe renders documents as the one string `"- " + text` and queries as they are; the client declares
`anchor: last` with `add_special_tokens: true`
(`rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/octen-embedding-8b/recipe.yaml`, the client block); the engine's
`/v1/embeddings` tokenises the render with the checkpoint's post-processor
(`vllm/renderers/params.py:183` in the tag: `add_special_tokens` defaults true). The reference is the paper's path
(`rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/octen-embedding-8b/reference.py`, `embed()`: left padding, right
truncation at 8192,
`last_hidden_state[:, -1, :].float()`, L2).

The checkpoint's `tokenizer.json` post-processor is
`Sequence[ByteLevel, TemplateProcessing(A + <|endoftext|>)]` — it appends the endoftext anchor **151643** to every
sequence, on both the bare-`tokenizers` and the HF path. Measured on CPU (the octen id-check script):
- the product's bare load and `AutoTokenizer` produce **byte-identical ids** on every query and document of all
  24 pairs rows, with and without specials; the appended id is 151643 in every row (e.g. the tiny query: 5 ids,
  tail `[11016, 15, 151643]`; the special-token-spellings row: 44 ids, tail `[198, 151652, 151643]`);
- the pooled token is therefore the anchor on both sides, at the same position.

So E1's hypothesis ("likely a template, EOS or pooling-token difference") is **refuted**: there is no id, anchor
or pooled-token difference to fix.

### 2.2 What is left, and why it needs the GPU

The E1 numbers are min cosine 0.99362 at `length:tiny` (a 4-token query), 0.99795 (special-token spellings),
0.99862 (long base64), 0.99884 (at_budget document) — unchanged with the reference on GPU (0.99372), so not a
CPU-reference artefact. Two facts bound the explanation:

- same-path bf16 noise does **not** show this shape: a Qwen3-4B analogue (zembed-1-embedding) measured
  bf16-vs-fp32 last-token cosines of 0.999841 (4 tokens), 0.999794 (8), 0.999765 (7), 0.998790 (201) and
  0.999819 (32,764) — flat around 0.9998, with no tiny-input-worst pattern (script: `zembed_bf16_noise.py`,
  plus the round-1 verifier's 32,764-token probe);
- the engine is not simply "fp32 where the reference is bf16": for an **embed** recipe vLLM's fp32 `head_dtype`
  only affects the cast/normalise of the pooled row (`seqwise/poolers.py:97-108`), not the hidden states, so the
  engine and reference are both bf16 through the transformer.

The remaining candidates are kernel-level (vLLM's fused RMSNorm/GEMM/attention vs HF's bf16) or a
length-dependent one. They cannot be separated on CPU.

**What the GPU wave must confirm (like-for-like, decision 9):** run the reference **and** a second engine at
`dtype: float32` for the same rows (the paper loads bf16, so the fp32 engine is a diagnostic, not a recipe
change); and compare vLLM's last hidden state with HF's for the same ids on one tiny row (a two-line script on the
node). If the gap is vLLM-vs-HF bf16, declare the measured bound with its evidence (recipe-fix item 5's decision
table); if it is a real id/position difference, this document's id check must be repeated against the engine's own
`/tokenize` on the node.

## 3. ctxl-rerank-v2-instruct-multilingual-1b: the `+` pad token is an added token to AutoTokenizer only

### 3.1 Mechanism

The product loads a tokenizer as `tokenizer.json` alone with the bare `tokenizers` library:
`TextTokenizer.from_json` (`rcp-ndcg/src/rcp_ndcg/data/tokenizer.py:67-80`, `_backend_class().from_str(...)`) fed
by `hf_hub_download(repo, TOKENIZER_FILE, ...)` (`:217`). HF `AutoTokenizer` additionally honours
`tokenizer_config.json` and `added_tokens.json` (the "sidecars").

For ctxl-1b, `tokenizer_config.json` declares `pad_token: "+"` (a single character that is *already* in the BPE
vocab as id 10 but is **not** in `tokenizer.json`'s `added_tokens`). AutoTokenizer therefore adds `+` to its added
vocabulary (`get_added_vocab()["+"] == 10`, `"+" in all_special_tokens`), and the added-token matcher splits
every `" +"` into `" "` (220) + `"+"` (10). The bare load does not, so BPE keeps `" +"` as one token (488):

| text | bare `tokenizers` | AutoTokenizer |
|---|---|---|
| `"H + ion"` | `[39, 488, 27672]` | `[39, 220, 10, 27672]` |
| `"2 + 2"` | `[17, 488, 220, 17]` | `[17, 220, 10, 220, 17]` |

The engine is `AutoTokenizer` (vLLM, `tokenizer_mode=auto`), so it counts **more** tokens on any text with a
space before a plus.

### 3.2 Reproduction (exactly E1)

On the ctxl-1b pairs file, the three texts containing `" +"` diverge and no others:

```
row 3  document  fit 563  engine 565
row 3  document  fit 179  engine 184
row 4  document  fit 676  engine 686
```

These are E1's numbers (the E1 run's equivalence output for ctxl-1b, outside the repository:
`fit_len 563 / engine_len 565`, `179 / 184`, `676 / 686` with the same first ids). The scan over all 19 shipped
recipes (the blast-radius scan) gives: ctxl-1b **3/70 texts divergent**; ctxl-2b 0/70 (the same
byte-identical tokenizer files — its pairs simply carry no `" +"`, so the gap is latent); ctxl-6b 0/70 (its pad is
`<pad>`, already an added token); every other recipe 0. The two pplx checkpoints declare
`tokenizer_class: TokenizersBackend`, which transformers 4.57 cannot load; under transformers 5.17.0 (the
engine's version) both load and diverge on 0 texts (`pplx-embed-v2-context-9b-preview` 0/64,
`pplx-embed-v2-late-0.6b` 0/87, both declaring `extra_special_tokens: ['[Q] ', '[D] ']`), so the item-3 seam is
ctxl-1b (and latently -2b) only.

### 3.3 Fix

**Product** (`rcp_ndcg.data.tokenizer`): after loading `tokenizer.json`, apply the sidecars the way
`AutoTokenizer` does — read `added_tokens.json` and `tokenizer_config.json`'s `added_tokens_decoder`,
`additional_special_tokens`, `extra_special_tokens` and the single-token fields (`pad_token`, `eos_token`,
`bos_token`, `unk_token`, `sep_token`, `cls_token`, `mask_token`), adding each as a special/added token on the
bare backend (the verifier simulated exactly this: bare + `add_special_tokens(["+"] )` equals AutoTokenizer on
0/70 ctxl-1b texts). The engine is the tokenization truth, so the client must match it. Touch-points:
`load_tokenizer` (`rcp-ndcg/src/rcp_ndcg/data/tokenizer.py:186-217`) is where the sidecar bytes are fetched and
read (for a local path: beside `tokenizer.json`; for a Hub id: the same revision), while `TextTokenizer.from_json`
(`:67-80`) takes the bytes and stays the one construction site; the `added_tokens()`/`special_text()` surface
(which today reads only `tokenizer.json`'s added tokens); a failing test first that reproduces 563 vs 565 on the
ctxl-1b render — network-gated or against a cached fixture, since tests never fetch — plus the ctxl-2b's latent
case (a synthetic `" +"` text); and the contract snapshots + CHANGELOG for any new public behaviour. The
tokenizer identity needs a decision: `tokenizer_identity`/`TextTokenizer.sha256` is the SHA-256 of
`tokenizer.json` alone, and the module docstring's invariant "two passes whose judges tokenize differently never
pool" becomes false once the sidecars decide the effective vocabulary — hash the applied sidecar content into
the identity (or record the sidecar fields read) rather than leaving the invariant stated and untrue. A test-only
alternative (recipe declares a wider budget) is not acceptable: the client would still cut at different
boundaries than the engine.

**Recipe-fix**: no recipe change; after the fix, re-measure ctxl-1b's stage 1 and stage 2 (the 0.249 score gap
was measured against a prompt the reference tokenised differently — see item 5).

## 4. jina-reranker-v3: a blank document is empty to the paper and sent by the product

### 4.1 Mechanism

- The paper's `JinaRerank.predict` (`experiments/paper/rerankers/reference/jina.py:66`) filters documents with
  `if d.strip()` — a **whitespace-only** document is dropped and scored 0.0, and the scores are re-aligned to the
  input order. The recipe's reference ports exactly this
  (`rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/jina-reranker-v3/reference.py:221-225`).
- The product's `omit_zero` decides emptiness on the content as given, by exact string:
  `if content.text != prefix or content.has_media: kept` (`rcp-ndcg/src/rcp_ndcg/inference/clients/_base.py:840`);
  the rerank client calls it with the default prefix `""` (`inference/clients/rerank.py:437`). A whitespace-only
  document therefore has `text != ""` and is sent and scored.
- E1 measured the consequence on the pairs row `content:whitespace_only`
  (`rcp-ndcg-test/pairs/jina-reranker-v3.jsonl:15`): served 0.099 vs reference 0.0. The recipe's note
  ("`empty_doc: omit_zero` declares the paper's rule",
  `rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/jina-reranker-v3/recipe.yaml:128`) is false for this row.

### 4.2 The minimal declared policy and its touch-points

The policy must be declared, not implied, and must not silently change the other `omit_zero` recipe.
`topk-embed-v1-small` also declares `omit_zero`
(`rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/topk-embed-v1-small/recipe.yaml:121`), but its referent renders an empty
document as `"Document:"` (one kept token, a positive MaxSim; `empty_doc: send` would render `"Document: "`, two
tokens) — a strip-based rule would *create* a divergence there. So:

1. **Add one policy value** — `omit_zero_blank` (name for the owner) — to both endpoint Literals
   (`rcp-ndcg/src/rcp_ndcg/inference/config.py:337` for `EmbeddingEndpoint`, `:588` for `RerankEndpoint`), and
   implement it in `_apply_empty_documents`: empty means `content.text.strip() == ""` (and no media), while
   `omit_zero` keeps its exact rule. Docstrings at `:284` and `:531` get one sentence each.
2. **Declare it in `jina-reranker-v3`**
   (`rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/jina-reranker-v3/recipe.yaml:61`) and correct the note at `:128`;
   the family's `empty_query` declaration (owner decision 25's wording, the independent recipe review) is
   separate.
3. **Tests**: a product test with a whitespace-only document (`omit_zero_blank` omits and the caller places 0.0,
   `omit_zero` still sends), a recipe contract pin for the new value, and a stage-2 row from the existing
   `content:whitespace_only` pair. The recipe-fix lane's failing-test-first rule applies.
4. **Surface**: a new value of a public `Literal` is a public-surface change — regenerate
   `tests/contract/snapshots/`, add a CHANGELOG `### Public surface` entry, and update the pages that describe
   `empty_doc`: `docs/concepts/text-budgets.md:162-163,188` and `docs/how-to/add-a-model.md:115`.
5. **Alternative (owner decision)**: make `omit_zero` strip-based and give topk `send`; that changes one shipped
   recipe's declared approximation and its corpus key, so it is the larger change, not the minimal one.

**What the GPU wave must confirm:** the whitespace-only row scores exactly 0.0 served and reference, and the
Kendall tau over the remaining documents is 1.0.

## 5. Reranker score precision: vLLM's score head is fp32, every reference's is bf16

### 5.1 The engine's dtype, from source

- `ModelConfig.head_dtype` (`vllm/config/model.py:2047-2073`) resolves through `_get_head_dtype`
  (`:2447-2465`): with no `head_dtype` key it returns **`torch.float32` when `runner_type == "pooling"`** — and
  its docstring names the switch: "Pooling models default to an fp32 head; use
  `--hf-overrides '{"head_dtype": "model"}'` to disable it."
- The classify conversion builds the score layer in that dtype:
  `self.score = ReplicatedLinear(hidden, num_labels, bias=False, params_dtype=model_config.head_dtype, ...)`
  (`vllm/model_executor/models/adapters.py:365-373`), and `ClassifierPoolerHead.forward` casts the pooled hidden
  state to `head_dtype` before the classifier (`vllm/model_executor/layers/pooler/seqwise/heads.py:170-174`).
- `load_weights_using_from_2_way_softmax` (qwen3-reranker, qwen3-vl-reranker) computes the head from the LM head
  in **fp32**: `lm_head.weight[[true_id]].to(torch.float32) - lm_head.weight[[false_id]].to(torch.float32)`
  (`adapters.py:566-571`). `load_weights_no_post_processing` (ctxl) loads the bf16 row into the fp32 parameter
  (`adapters.py:637-642`). Either way the score is computed in fp32.

### 5.2 The references' dtype, from source

| Recipe | Reference | Score dtype |
|---|---|---|
| qwen3-reranker-0.6b | `reference.py:358-363` `self.model(**inputs).logits[:, -1, :]` | bf16 (model dtype) |
| qwen3-reranker-4b | `reference.py:258-263` same | bf16 |
| qwen3-reranker-8b | `reference.py:235-240` same | bf16 |
| ctxl-…-1b | `reference.py:188-189` `out.logits[:, -1, VOCAB_POSITION]` | bf16 (paper: `experiments/paper/rerankers/reference/contextual.py:113`) |
| ctxl-…-2b | `reference.py:261-262` same | bf16 |
| ctxl-…-6b | `reference.py:208` same | bf16 |
| qwen3-vl-reranker-2b | `reference.py:275-282`: `weight_yes - weight_no` then `.to(self.model.dtype)` | bf16 — and this is the **card script's** own path (the checkpoint's `scripts/qwen3_vl_reranker.py:95-101` subtracts in bf16 and `:89` casts the head to the model dtype) |

So the engine is *more precise* than the card, not less, and the gate compares the two. The measured gaps are the
bf16-input quantisation of the head: max |Δ| 0.0215/0.0389/0.0408 (qwen3-reranker 0.6b/4b/8b) with p99
97.7 %/97.7 %/90.9 % inside 0.02; ctxl max relative 0.0307/0.0872/0.249 (6b/2b/1b) against the 0.05 bound;
qwen3-vl-reranker-2b max |Δ| 0.0289 with p99 97.7 % (93 % in the first run).

### 5.3 Fix: one declared serve override per family

`serve.hf_overrides: {..., head_dtype: model}` for `qwen3-reranker-{0.6b,4b,8b}`, `qwen3-vl-reranker-2b` and
`ctxl-rerank-v2-instruct-multilingual-{1b,2b,6b}`. The references stay as they are: they are the card's/paper's
bf16 path, and decision 9 forbids bending them toward the product. The change re-pins every recipe's serve block:
the contract tests pin it (`EXPECTED_SERVE` in the ctxl tests, `CONTRACT["serve"]` in the qwen3 tests, the inline
serve block in `test_qwen3_vl_reranker_2b.py`):
`rcp-ndcg-test/tests/recipes/test_ctxl_rerank_v2_instruct_multilingual_{1b,2b,6b}.py`,
`test_qwen3_reranker_{0_6b,4b,8b}.py`, `test_qwen3_vl_reranker_2b.py` — so each must be updated with the change
and a CHANGELOG bullet folds into the release entry (AGENTS: a recipe's CHANGELOG bullets fold into the one
release entry). With `head_dtype: model`:

- the score layer's parameter becomes bf16; for `from_2_way_softmax` the fp32 difference is cast to bf16 — which
  equals the card's bf16 subtraction (the difference of two bf16 rows is exact in fp32, so both are the
  correctly-rounded bf16 of the same value); for `no_post_processing` the row is already bf16;
- the hidden state is no longer upcast, so the logit is the same bf16 matmul the reference computes.

**The ctxl-1b's 0.249 is larger than the 2b's 0.0872 and the 6b's 0.0307 but is the same class.** Its stage-1
`/tokenize` gap (item 3) does not change the scored prompt: the client sends text, the served template and the
reference's render produce the same string on all 70 pairs (the tokenizer gap changes the client's *count* and
its over-cap cut boundaries, not the under-cap text), and both the engine and the reference tokenize with
AutoTokenizer. So E2 must re-measure all three sizes under the override — 0.249 is the number to close or
declare, not a tokenizer artefact. **What the GPU wave must confirm:** the same E2 stage-2 rows with the
override, expecting the p99/max bounds to pass on all seven; and a like-for-like control (one recipe served fp32
vs the same reference) so the bound decision has its evidence.

## 6. pplx-embed-v2-context-9b-preview: the warm-up input and the `max_model_len` arithmetic

### 6.1 The plugin's dummy-run inputs at the tag

vLLM v0.31.0 warms a pooling engine with a **real prefill request** before it serves:
`vllm/v1/worker/gpu/warmup.py:255-257` builds `prompt_len = decode_query_len + 1` and
`prompt_token_ids = list(range(prompt_len))`; for a pooling model `decode_query_len` is 1 (no spec decode), so
the ids are `[0, 1]` — exactly what E1 measured. `:308-317` builds the pooling params and `:355` executes the
prefill through the normal worker path, so the ids reach the model's pooler. The pooler's *own* dummy grid is a
separate path and **is** all zeros (`vllm/v1/worker/gpu_model_runner.py:6322-6370`, and the pooling runner's
`_dummy_pooler_run_task`, `vllm/v1/worker/gpu/pool/pooling_runner.py:161-205`).

The plugin tolerates only the all-zero grid: `_is_warmup_dummy` returns `not any(token_ids)`
(`rcp-ndcg-vllm/src/rcp_ndcg_vllm/models/pplx/pooling_core.py:62-70`), so `[0, 1]` falls through to the
role-prefix check (`:152-159`) and raises. A real request can never start with id 0 — the query prefix is 248077
and the document prefix `[62724, 60]` (`pooling_core.py:45-56`) — so the minimal rule is **"the first id is 0 ⇒
the engine's warm-up"**, covering both the arange row and the zero grid. That is the owner-approved direction
("validate real requests only"). A CPU test with `[0, 1]` first (owner-approved) and one with the zero grid keeps
both warm-up shapes pinned.

### 6.2 The `max_model_len` bound from the config

The checkpoint's `config.json` (`perplexity-ai/pplx-embed-v2-context-9b-preview@b667039e…`, `text_config`):
`hidden_size 4096`, `num_attention_heads 16`, `head_dim 256`, `num_key_value_heads 4`, `attn_output_gate true`,
`intermediate_size 12288`, `num_hidden_layers 32` (24 `linear_attention` + 8 `full_attention`),
`linear_num_key_heads 16`, `linear_num_value_heads 32`, `linear_key_head_dim 128`,
`linear_value_head_dim 128`, `max_position_embeddings 262144`, `is_causal false`.

Per forward, one sequence of `L` tokens, served bf16 (2 B/element); `2^31 = 2,147,483,648`:

| tensor (source) | elements/token | bytes/token | largest L, elements | largest L, bytes |
|---|---|---|---|---|
| attention `q` (`q_size`, `qwen3_next.py:301`) | 4,096 | 8,192 | 524,287 | **262,143** (E1 faulted at 262,144) |
| merged `qkv_proj` output (`:309-317`, `16·2·256 + 2·4·256`) | 10,240 | 20,480 | 209,715 | 104,857 |
| MLP `gate_up_proj` output (`qwen2_moe.py:91-98`, `2·12288`) | 24,576 | 49,152 | **87,381** | **43,690** |
| MLP activation (`SiluAndMul`, `12288`) | 12,288 | 24,576 | 174,762 | 87,381 |
| GDN `in_proj_qkvz` output (`key·2 + value·2 = 12288`) | 12,288 | 24,576 | 174,762 | 87,381 |
| GDN `conv1d` output (`key·2 + value = 8192`) | 8,192 | 16,384 | 262,143 | 131,071 |
| hidden state | 4,096 | 8,192 | 524,287 | 262,143 |
| GDN recurrent state (`32·128·128`, per layer, length-independent, fp32) | — | — | 524,288 (no L bound) | — |

So:
- the operator's working bound `174,762 = 2^31 / 12,288` is the **post-activation** tensor's element bound; the
  brief's declared `131,072` is below it;
- the **merged `gate_up_proj` output is 24,576 wide** — twice the intermediate size, as
  `MergedColumnParallelLinear(hidden, [intermediate] * 2)` — so its element bound is **87,381** and its byte bound
  **43,690**; at `L = 131,072` it holds 3.2e9 elements / 6.4e9 bytes. Whether that faults depends on the kernel's
  indexing (the E1 fault was FlashAttention's int32 **byte** offsets on `q`; cuBLAS/cutlass GEMMs use 64-bit
  global offsets), and it cannot be decided on CPU.

**What the GPU wave must confirm (E2, one B200):** that a **real** request of exactly the declared
`max_model_len` completes end-to-end (not only the warm-up), watching for the `cudaErrorIllegalAddress` class at
warm-up and at serve; if the merged gate_up tensor faults, step the declaration down to **87,381** (elements) or
**43,690** (bytes). The recipe-fix lane declares the chosen `max_model_len` as an over-cap deviation with the
client cap to match — the pplx recipe anchors `first`, so a cap-only difference is the harness's
`over_cap_cut_differs` kind rather than `anchor_drop_over_cap`; the lane confirms the kind against the harness —
and records this arithmetic in the recipe notes. The change re-pins
`rcp-ndcg-test/tests/recipes/test_pplx_embed_v2_context_9b_preview.py` (`max_model_len` at `:66`, `max_tokens` at
`:79`, the serve-argv assertion and the mutant at `:306-317`), and the plugin fix updates the
`_is_warmup_dummy` docstring (`pooling_core.py:62-70`) that documents the all-zero-only exception.

## 7. Reproducing these measurements

The scripts live in the lane's scratch directory (outside the repository): `zembed_probe.py`,
`zembed_st_probe.py`, `zembed_st53_probe.py`, `zembed_stage2_cpu.py`, `zembed_bf16_noise.py`,
`octen_id_check.py`, `ctxl_tokenizer_compare.py`, `tokenizer_blast_radius.py`. They need the checkpoints at
their pinned revisions (Hugging Face cache), the vLLM v0.31.0 tag clone, and — for the sentence-transformers
runs — the node's reference environment (`sentence-transformers==5.3.0` with `transformers==5.17.0`). The zembed
and octen scripts take minutes on CPU; the ctxl scan is seconds per recipe. No script writes into the
repository, and none needs a GPU. The independent verifiers reproduced the load-bearing numbers with their own
scripts in the same scratch directory.
