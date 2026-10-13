# Adopt a model: the four classes, the pitfalls and the parity bar

[Add a serving recipe](add-a-model.md) is the mechanics: the family directory, the schema, the reference
interface, the three equivalence stages, the wave. This page is the doctrine above it — what "parity with
the reference" means for each of the four classes of model this package serves, what to look out for when
you bring a new checkpoint, and what a recipe must prove before its `status` may say `verified`.

The one-sentence rule: **the served engine must be indistinguishable from the model's own published code
path — or, where the paper's numbers rest on it, the paper's exact scorer — on the inputs the product
sends, within bounds that are declared, measured, and never loosened silently.**

## 1. What "verified" means

A recipe is not verified by a smoke call. It is verified when its served engine reproduces the reference
on the **observation corpus**: generated strata (over-length inputs per declared shape, the spelling
stratum, media rows, empty documents, ties) sent through the product's own role clients, compared per
vector, per document and per query, with the gates below. A single pass/fail hides the evidence; every
number in `equivalence.json` carries its referent.

| Comparison | Default gate |
|---|---|
| Vector, per vector (per token for multi-vector) | cosine ≥ 1 − 1e-3, after the same float16 cast on both sides |
| Probability score, per document | \|Δ\| ≤ 0.02 for 99% of documents, and ≤ 0.05 for all |
| Logit score, per document | \|Δ\| ≤ 0.05 · (1 + \|s\|) |
| Cosine score, per document | \|Δ\| ≤ 0.01 |
| Ranking, per query | median Kendall τ ≥ 0.98 |
| Stage 3, per subset | mean \|Δ nDCG@10\| ≤ 2e-3 |

Two things may legitimately stop a gate from passing, and both are declared, never silent:

- **A measured bound.** Where the residual is a genuine numeric floor — both sides run bf16 kernels, no
  ranking changes, τ = 1.0 — the family, or the size whose floor differs, declares the measured bound in
  `gates` (`vec_min_cosine`, `prob_p99_abs`, …) and names the evidence (per-MRL, per-document) in its
  notes and the CHANGELOG. A size may tighten a family bound, never loosen one; an unset field keeps the
  family's value. The bound still gates: a regression beyond it fails.
- **A declared deviation.** The reference's own behaviour over a cap, or a card pipeline that cannot run
  an input, is declared under `reference.known_deviations` (`anchor_drop_over_cap`,
  `over_cap_cut_differs`, `media_approximation`). Those inputs are reported in a separate non-gating
  table with their cause. The reference never copies the client's cut to make a pair pass.

Cuts are the client's, applied to content spans only, with the anchors reserved; the schema has no
engine-truncation field at all. Judges are the exception to score parity: no HF reference exists for a
rubric prompt, so a judge recipe declares no reference and its equivalence/record steps are reported as
skipped (counted as satisfied); a judge's parity is the instrument's own contracts, below.

## 2. The four classes

### Dense embedding — `role: embed`

One vector per text: a pooler (last token, mean, CLS) plus normalization, sometimes a Matryoshka head.
Served as `runner: pooling` with `api: vllm_pooling` or `openai_embeddings`; the wire is `messages`
(the engine's template frames the content) or `token_ids` (the client tokenizes and sends ids).

What to look out for:

- **The role prompt.** Query and document sides may carry different prefixes (`prompt_name`,
  `config_sentence_transformers.json`). A dataset with a task instruction is refused unless the recipe
  declares `instruction: none | fold` — never silently changed.
- **The token the pooler reads.** A last-token pooler's trailing end token is part of the anchor: the
  cut reserves it, re-attaches the template, and stage 1 asserts it survived. A mean pooler has no anchor
  token, so the audit asserts the fixed edges and at least one content token instead.
- **A suffix marker.** Some cards append their own suffix in code (an end marker plus newline, a
  separator); that suffix is the model's computation and belongs in the reference and in the render.
- **Matryoshka.** `mrl_kind` (`truncation`, `projection`, `none`) with the card's `mrl_dims`/`mrl_range`;
  the serve gate and the client declaration must agree, and the comparison is per k as well as at full
  width.
- **The engine is the tokenization truth.** With an engine URL, stage 1 requires the engine's
  `/tokenize` ids and counts to equal the client's render, and the engine's own `usage.prompt_tokens` to
  equal the client's budgeted count; without an engine both are `not_run`, never passed.
- **The bf16 floor.** A minimum cosine a little under 1 − 1e-3 with τ = 1.0 is the kernel stack, not the
  model: declare the measured per-variant bound (and per MRL width) rather than chasing the last 1e-4.

### Cross-encoder reranker — `role: rerank`

Pointwise or listwise; a score per document on a declared scale (`probability`, `logit`, `cosine`) with
`use_activation` set for a served wire. The engine's classifier head reads a token pair
(`classifier_from_token`), or a listwise prompt is generated and the answer parsed.

What to look out for:

- **The scale and the activation.** `sigmoid(yes_logit / 5)`, a raw logit and a cosine are three
  different gates. Declare `score_scale` from the reference; a wrong scale fails every document, not
  subtly.
- **The settled query.** A reranker settles one query span per call and shares it across its pointwise
  requests; the query and document shares are declared (`query_max_tokens`, `document_max_tokens`) and
  stage 1 asserts the settlement.
- **The instruction and the assistant anchor.** `instruction: none | field | fold`, and the trailing
  assistant suffix the model reads its score from — a `last` anchor, reserved by every cut.
- **Listwise prompts are the reference's builder.** The client cannot re-derive it, so the engine's
  prompt-token check reports `not_run` with that reason; the prompt itself is compared through the
  reference's `render` mode.
- **The reference's over-cap behaviour.** A joint `longest_first` truncation that drops anchors is
  declared (`anchor_drop_over_cap`); one that keeps anchors but cuts its own way is declared
  (`over_cap_cut_differs`). The reference stays the card's or the paper's.
- **Remote-code quirks.** A rope path that expects 3-D position ids, a missing `layer_type`, a nested
  `hf_overrides` the engine reads differently: fix with a narrow, declared shim (with a CPU test) or pin
  the transformers version the checkpoint needs — never by relaxing a gate.
- **The bf16 probability floor.** τ = 1.0 with a p99 just under 99% at the default tolerance is a kernel
  floor; declare the measured |Δ| tolerance per variant.

### Late interaction (multi-vector) — `role: multi_vector`

One L2-normalized vector per kept token; the retrieval score is fp32 MaxSim. The keep-rule is the
checkpoint's own mask, not a generic one. Served with the multi-vector plugin; the wire is `token_ids`
(the client tokenizes; the engine tokenizes nothing).

What to look out for:

- **The prefix tokens are tokens too.** The client must reproduce the checkpoint's own construction,
  including the prefix's own tokens and any standalone space token inside it. A plugin that pools a
  prefix token the reference never pools produces a real, reproducible mismatch: fix the plugin, not the
  gate.
- **The nesting of the reply.** One vector for a dense embedder, one matrix per text for a multi-vector
  model; the reference and the client agree on the nesting per side (query and document), and the
  comparison walks the same nesting.
- **All-masked documents.** Two zero vectors are agreement (cosine 1.0); a one-sided zero is a failure
  (0.0) — a document that keeps no token on one side only is a defect, and the convention is documented
  and tested in the harness.
- **Media.** A multi-vector recipe with image input runs the media stage as an *input* gate: the parts,
  their prepared sizes and the token counts must match the card, and the engine's own media count is an
  independent check. It proves the model is shown the same media, never that it returns the same numbers
  — that half is stage 2.

### LLM / judge — `role: judge`

A rubric or tournament judge, served by any OpenAI-compatible endpoint (one URL, replicas of the same
model allowed). A judge recipe ships `family.yaml` only: there is no HF score reference for a rubric
prompt, so no `reference.py` and no stage-2 gates.

What to look out for:

- **Sampling is declared, not inherited.** The self-hosted recipes send the engine's HF defaults
  (temperature 1.0, top_p 1.0, no top_k) rather than a vendor's defaults; what the recipe declares is
  what crosses the wire.
- **The prompt is the family.** The tournament and rubric prompts define the judgement family, with a
  parse version; a changed prompt is a new family and never pools with the shipped one. Every criterion
  is fixed — the rubric has exactly its five, and they are not reworded.
- **Budgets.** A self-hosted role config declares `tokenizer` + `max_tokens`; a hosted vendor profile
  without a tokenizer declares `budget_source: vendor` and sends content uncut. Media are counted in
  tokens, never money.
- **The smoke route first.** A judge's first end-to-end check is its smoke call; then the schedules, the
  judgement store, the cost estimate and the calibration are the instrument's own tested contracts.

## 3. The adoption workflow

1. **Pin the checkpoint.** The lowercased canonical Hub repo name (never a redirecting short name), the
   revision, the licence, and the card's own usage with its `README` lines — the reference follows the
   card, and where the paper's numbers rest on it, the paper's exact module wins and the divergence is
   noted.
2. **Write the reference.** One `reference.py` per family: the card's code path (or the paper's module)
   with the harness's subprocess CLI added and the provenance in the docstring. Justify every pin in
   `reference.in` with a `file:line` comment and generate `reference.lock`.
3. **Choose the serving path.** Flags first, then the checkpoint's chat template, then pooler config,
   then a plugin/patch (a vendored model backport) — with `plugin_architectures` declared when the plugin
   registers architectures.
4. **Declare the request as data.** Shapes as `fixed`/`content` segments, the anchor kind, explicit
   budgets, `on_overflow`, the empty-document policy, the title policy. Nothing is implicit.
5. **Stage 1 on CPU.** The anchor audit, the render check and the template check run without an engine;
   the engine checks report `not_run` — never "passed".
6. **The family's tests.** The family test module pins every field per variant (with mutants red), the
   recipe goldens are regenerated the documented way, and the contract snapshots are updated.
7. **The GPU wave.** Stage 2 (plus the media stage for media input), then read `equivalence.json` per
   referent. Fix or declare — in that order; a declared bound carries its evidence.
8. **Then, and only then, `verified`.** The status flips from the wave's own artifacts, the report is
   written, and the release gate (wave 0 and the end-to-end scenarios) follows.

## 4. What the repository keeps per model

Every retrieval family keeps its reference, runnable, in the tree:

```text
recipes/<family>/
  family.yaml      the contract: the shared blocks + the variants table
  reference.py     the family's ONE reference — the card's own code path or the paper's exact scorer,
                   with the harness's subprocess CLI added and its provenance in the docstring
  reference.in     the pins, each justified from the card or the code with a file:line comment
  reference.lock   the generated, hashed lock the node installs into the family's own venv
  template.jinja   only when the model needs a chat template
  <card>.py        a card script the reference runs verbatim (e.g. a remote-code file), hash-pinned
```

The harness never imports the reference — it holds no torch. Stage 2 launches it as a subprocess in the
family's own environment (`--reference-python`), built from the lock over the engine image's torch/CUDA,
so one recipe's transformers pin cannot break a wave. The original computation is therefore not only
kept: it is the thing the engine is measured against, and its environment is reproducible per family.

| Family | Class | The reference's provenance |
|---|---|---|
| `ctxl-rerank-v2-instruct-multilingual` | rerank | the paper's exact in-process scorer |
| `embeddinggemma-2` | embed (text, image, video) | the model card's own usage (sentence-transformers pipeline) |
| `harrier-oss-v1` | embed | the model card's own |
| `jina-embeddings-v5-text` | embed | the model card's own (remote code) |
| `jina-reranker-v3` | rerank (listwise) | the paper's exact scoring path + the card's remote code |
| `octen-embedding` | embed | the paper's in-process encoder path |
| `pplx-embed-v1` | embed | the model card's own code path |
| `pplx-embed-v2-context` | multi-vector | the checkpoint's own remote code at the pinned revision |
| `pplx-embed-v2-late` | multi-vector | the model card's own code path (native sentence-transformers modules) |
| `qwen3-embedding` | embed | the model card's own |
| `qwen3-reranker` | rerank (pointwise) | the paper's exact in-process scorer |
| `qwen3-vl-embedding` | embed (text, image, video) | the model card's own + its remote code vendored |
| `qwen3-vl-reranker` | rerank (pointwise) | the model card's own |
| `topk-embed-v1` | multi-vector | the model card's own code path |
| `zembed-1` | embed | the model card's snippet (remote model, the suffix its pooler reads) |
| `zerank` | rerank (pointwise) | the paper's in-process code (where card and paper disagree, the paper wins) |

The judge families are the exception by design: a rubric prompt has no HF score reference, so they ship
`family.yaml` alone and their parity is the judge instrument's own contracts (prompts, parse versions,
schedules, calibration) plus the smoke route.

## 5. The pitfall catalogue

Every row below is a failure this repository actually hit and resolved; the disposition column is the
pattern to reach for first.

| Symptom | What it was | Disposition |
|---|---|---|
| CUDA device-side assert on a real request, even eager | the tokenizer declares a special token at id = `vocab_size`; the corpus's spelling stratum gathered an out-of-range embedding | bound the spelling set by the checkpoint's declared vocab; record the checkpoint defect |
| min cosine just under the gate, τ = 1.0, no rank change | the two bf16 attention-kernel stacks | declare the measured per-variant (per-MRL) bound with the evidence |
| p99 within the gate is just under 99%, while the maximum delta is above the default | the same bf16 floor, probability side | declare the measured per-variant delta tolerance |
| engine returns one matrix where the reference returns rows | the query-vector nesting of the reply | fix the reference's nesting per the wire contract |
| `StopIteration` in the processor at a newer transformers | the card's processor call moved to per-input processing | call the card's processor the way the card's own code does |
| remote-code rope shape error (`TypeError`/`IndexError`, 2-D vs 3-D position ids) | a transformers API change under the checkpoint's remote code | a narrow, tested shim that targets the class the remote code calls, or the checkpoint's required transformers |
| engine dies at init with a Triton shared-memory limit, or FA4 rejects the head dim | a compile-time kernel demand that eager/batch flags cannot move; the attention backend was overridden | pin a backend whose kernels accept the model's head dims -- the image's own hook choice is not automatically one |
| the engine registry lacks the architecture | the image is a released vLLM, not a nightly | vendor the model into the plugin (backport) and declare `plugin_architectures` |
| the vendored model imports an absent upstream module path | the fold registered classes but not the modules the model imports | install the modules under the upstream paths at the vendored model's own import |
| cosine 0.0 on some vectors only | one side pooled a prefix's standalone space token; the other pooled nothing | fix the plugin; and two zero vectors are agreement, one-sided zero is not |
| a spelling-row query scores far below 1 | the client sent one parse where the checkpoint's code split-tokenizes the body and prepends a prefix id | the client's `token_ids` leg sends the checkpoint's own construction (a per-side declaration) |
| the reference dies on a text-only row carrying the checkpoint's placeholder spelling | ST's `encode` routes a text-only input through the checkpoint's multimodal processor, whose `validate_inputs` refuses the unbalanced spelling; the engine's tokenizer just matches it as the checkpoint's added token | give the reference the card's own tokenizer path for text-only sides (embeddinggemma-2's `reference.py::text_vectors`: the checkpoint's tokenizer over the prompt-prefixed text, then the card's module stack), never a spelling rule of its own |
| an engine whose media pins are missing passes an input check | the client resized or sampled the media its own way | the media stage compares prepared sizes and counts, and the engine's own count is the independent check |
| judge smoke fails before anything else | a route or payload the endpoint does not accept | the smoke route is the first check; then the instrument's contracts |

## 6. The parity bar, per class

| | Dense embedding | Cross-encoder reranker | Late interaction | Judge |
|---|---|---|---|---|
| What is compared | per-vector cosine (and stage-3 nDCG) | per-document score and per-query τ | per-token vectors and MaxSim | no score reference; the smoke route and the instrument |
| The reference | the card's pipeline or the paper's encoder | the card's scorer or the paper's scorer | the card's own code path | none (by design) |
| A measured numeric floor | declared per variant and per MRL width | declared per variant | declared per variant | not applicable |
| Declared deviations | over-cap cut, media approximation | over-cap cut, media approximation | over-cap cut, media approximation | none |
| Never | engine-side cuts; a silent bound; a reference reimplemented from the engine | the same | the same | a changed prompt pooled with the shipped family |
