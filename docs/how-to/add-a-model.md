# Add a model: serving recipes for vLLM

A recipe is one declarative description of how a model is served with vLLM and how `rcp-ndcg` reads it back.
Recipes are grouped into **families** (owner decision 34): one directory per model family holds the shared
serving contract and a `variants` table with only the per-size facts, so adding a size is adding a variant row,
while every size stays its own tested recipe id (served, contract-tested, stage-1-tested and GPU-validated on
its own; a family id is never served). This guide shows the format, how to check a served recipe against its
reference implementation, and how a wave of recipes is submitted to the GPU host. The package lives at
`rcp-ndcg-vllm/` (installed into the engine image, which carries its own vLLM and torch). A family directory of
your own needs none of that packaging to be *used*: `rcp-ndcg-vllm serve ./my-family/ [--variant <id>]` and
`recipe:./my-family/` load it through this same schema, marked unshipped and `unverified`
([serve a retrieval model](serve-a-model.md)).

## The family directory

One directory per model family, `rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/<family>/`, with these four files (a
family may also ship a vendored card script that its reference runs verbatim, byte-identical to the Hub file
and hash-pinned by the family's test):

```text
recipes/<family>/
  family.yaml                  # the family: the shared blocks + the variants table (the Family schema;
                               # every field is listed in schema/family.schema.json)
  template.jinja               # the family's ONE chat template for vllm serve --chat-template (only when the model needs one)
  reference.py                 # the family's ONE reference implementation, run as a subprocess per variant
                               # (see the reference interface)
  requirements-reference.txt   # the family's reference environment (installed when REFERENCE_REQUIREMENTS names it)
```

`family.id` equals the directory name, matches `^[a-z0-9][a-z0-9.-]*$`, and is never served. Each row of the
`variants` table is a full recipe id (`^[a-z0-9][a-z0-9.-]*$`, the lowercased canonical Hub repo name), the
`--served-model-name` the engine serves, and what `recipe: <variant id>` in `rcp-ndcg` resolves. A single-size
model is a family with one variant (one loader path for both). The schema is closed (`extra="forbid"`) and
role-aware: a field that only makes sense for one role is refused for the others, so a typo cannot silently
change what is served.

**Adding a size to an existing family** is one row plus its pins plus its tests:

1. Append a `variants` row: `id`, `model`, `revision`, and the whitelisted per-size `overrides` only
   (`resources`, `serve.max_model_len`/`hf_overrides`/`mm_processor_kwargs`/`limit_mm_per_prompt`,
   `client.max_tokens`/`query_max_tokens`/`document_max_tokens`/`dim`/`dimensions`/`batch_size`/`max_images`/
   `max_videos`; plus the per-size `notes`, `sources` and `status`). Anything else that differs is refused with a
   typed error naming the field: the shared blocks are the family's contract, so a size that behaves differently
   is its own family.
2. Pin the variant's fields in the family's test module (one module per family, parametrized over its
   variants; every field pinned per variant, two mutants red per family) and run the family's stage-1 network
   tests for the new variant.
3. Add the variant's golden: `uv run --no-sync pytest rcp-ndcg-test/tests/recipes/test_family_goldens.py
   --update-goldens` rewrites `tests/recipes/golden/` from the current tree (review the diff, then commit; the
   guard compares every variant's resolved contract and fingerprint against its golden in every CI job).
4. A new family needs its `family.yaml` (below), its reference, its template when it needs one, its family
   test module and its variant goldens.

Validate a family/variant without an engine (run from the package directory, so the recipes root resolves):

```bash
cd rcp-ndcg-vllm
python -m rcp_ndcg_test.equivalence --recipe <variant-id> --pairs pairs.jsonl --out /tmp/equiv --stages 1
```

## The template block, the anchors and the explicit budget

Three research findings shape the `serve` and `client` blocks, and the schema enforces them:

- **Anchors.** A model reads its output from fixed positions of its template — the *anchors* (a last-token
  pooler's end token, a CLS head's leading special, a marker id, a reranker's assistant suffix). Every cut
  applies to the content spans only, inside a budget computed after reserving every fixed template token, and
  the template is re-attached after the cut. Engine-side truncation of a rendered prompt cannot honour this in
  either direction, so the schema has no engine-truncation field at all: a recipe whose reference deliberately
  drops anchors declares `reference.known_deviations: [anchor_drop_over_cap]`, and one whose reference keeps the
  anchors but cuts over-cap content its own way (a joint `longest_first` truncation, say, where the client settles
  the query at its share) declares `[over_cap_cut_differs]`; either way the harness reports those pairs
  separately, outside the gates. The reference stays the paper's or the model card's: it never copies the
  client's cut to make an over-cap pair pass. The declared shape's `anchor` is `last`, `first`, `last_content` (a
  model that pools the last real token of raw text: the shape ends on its content span, and the audit asserts
  the head marker opening the render and a content token closing it, before the post-processor's tail), `mean`
  or `marker` (with `anchor_markers`), and stage 1 samples over-length inputs (at least 20 per shape) and asserts every
  anchor survived — reported as `anchor_check`, separately from token-id mismatches.
- **Segments: `fixed` and `content`, nothing else.** A shape is an ordered list of segments with exactly two
  kinds: `fixed` — ordinary template text, which may name special tokens as `{special:<name>}` placeholders
  that the recipe tokenizer's added tokens resolve at run time — and `content`, the cuttable span
  (`query` or `document`). The schema is closed, so `{"special": ...}`, `{"text": ...}` or `{"ids": ...}`
  segments are refused. Every declared shape needs at least one content span; an `anchor: last` shape must end
  with a fixed segment (or pin `add_special_tokens: true`, declaring the tokenizer's end token as the anchor).
  When `serve.chat_template` is set, stage 1 also proves the declared shapes render to the same token ids as
  the template file.
- **Explicit budgets.** Every recipe declares `client.tokenizer` and `client.max_tokens` — there is no implicit
  default. `on_overflow` is `cut` by default (content spans only, at token boundaries, anchors reserved;
  `chunk` and `fail` are the alternatives) and `aggregation: max` is the only chunk aggregation. The engine's
  `max_model_len` must fit the client's budget. `serve.pooler_config` keys are validated against the pinned
  engine's `PoolerConfig` fields, because the engine rejects unknown keys.
- **The engine is the tokenization truth (R29).** With an engine URL, stage 1 sends `fit`'s rendered prompts
  to the engine's `/tokenize` (same `add_special_tokens` as the route) and requires the ids and counts to
  equal `fit`'s; mismatches are reported per shape in `equivalence.json` (`engine_tokenize_check`) and fail
  the stage. Without an engine (CPU), the check is reported as `not_run`, never as passed.
- **The reference runs as a subprocess.** Stage 2 runs the recipe's `reference.py` as a subprocess
  (`--reference-python <path>`, required when stage 2 runs; no default) that reads the pairs file and writes
  scores or vectors to a file the harness compares. The harness process imports no torch or transformers; the
  reference environment is documented in `rcp-ndcg-vllm/requirements-reference.txt`. The engine comes
  up on the slot's GPUs first; the reference subprocess runs against the pairs file while the engine is up and
  releases its memory when it exits.

Family YAML at a glance (a complete, loadable family — `tests/docs` runs `load_family` and
`resolve_recipe` on it; the schema's docstrings define every field):

```yaml
id: example-reranker               # the family id; never served
schema_version: "1"              # the family/recipe file format's version (decision 18)
role: rerank                     # embed | multi_vector | rerank
input: [text]                    # subset of [text, image, video]
scoring: pointwise               # rerank only: pointwise | listwise
licence: apache-2.0              # SPDX id or see-model-card
engine: {name: vllm, image: "vllm/vllm-openai:v0.31.0", min_version: "0.31.0"}
resources: {gpus: 1}             # tensor_parallel_size = gpus
serve:                           # everything rendered into `vllm serve` argv; nothing implicit
  runner: pooling
  convert: null
  hf_overrides: {"architectures": ["ExampleForSequenceClassification"], "classifier_from_token": ["no", "yes"]}
                                 # always quote strings YAML reads as booleans ("no", "yes", "on")!
  chat_template: template.jinja  # a file in this directory, or null
  pooler_config: {use_activation: true}   # keys must be PoolerConfig fields at the pinned engine
  trust_remote_code: false
  max_model_len: 8192
  dtype: bfloat16
  plugin: null
  plugin_architectures: []       # when plugin is set: the architectures its engine registers (the behaviour fingerprint keys their modules)
  patches: []                    # engine patch names this recipe opts into; serve renders them into RCP_NDCG_VLLM_PATCHES
  extra_args: []                 # further flags, verbatim (one argv element per item)
client:                          # the product's endpoint config for the role; the product validates it at load
  api: rerank                    # the role's wire: openai_embeddings | vllm_pooling | rerank
  # client.model, client.revision and client.tokenizer are injected per variant
  # (model@revision); never declare them here
  max_tokens: 8192               # explicit; there is no implicit budget
  query_max_tokens: 1024         # the query's share of the pair budget; the document span gets the rest
  template:                      # the request shapes as data: fixed and content segments only
    pair:
      - {fixed: "SYSTEM: Judge whether the Document answers the Query. USER: Query: "}
      - {content: query}         # the cuttable span; fold mode puts the instruction text here
      - {fixed: " Document: "}
      - {content: document}
      - {fixed: "{special:im_end} ASSISTANT"}   # specials by name, resolved from the tokenizer
    anchor: last
  instruction: fold              # rerank only: none | field | fold | system (fold is the default)
  use_activation: true           # a served rerank wire must set it: the score's scale is content
  on_overflow: cut               # cut (default) | chunk | fail; cuts apply to content spans only
  empty_doc: send                # omit_zero | send | send_text
reference:
  kind: transformers             # transformers | sentence_transformers | remote_code | stored_scores
  score_scale: probability       # probability | logit | cosine; vectors compare per vector
  entry: reference.py
  known_deviations: []           # or [over_cap_cut_differs] etc.: over-cap pairs reported non-gating
gates: {}                        # overrides of the stage-2 defaults for this score_scale
status: {state: unverified, image: null, date: null, report: null}   # the family default
sources: []                      # shared URLs and path:line references (a variant adds its own)
notes: ""                        # shared notes (a variant adds its own)
variants:
  - id: example-reranker-0-6b    # the recipe id: served, `recipe:`-resolvable, contract-tested
    model: example-org/example-reranker
    revision: "0123456789abcdef0123456789abcdef01234567"   # quoted: a bare commit can read as a number
    notes: >-                    # the variant's own notes (appended to the family's)
      The per-size facts: this size's paper batch size, its measured tokenizer facts, ...
    sources:
      - "https://huggingface.co/example-org/example-reranker (the card at the pinned revision)"
    # overrides:                 # only the declared per-size fields; everything else is refused
    #   serve: {max_model_len: 16384}
    #   client: {max_tokens: 4096}
```

Two YAML footguns, both caught in review and by the golden tests: always quote string tokens that YAML reads as
booleans (`"no"`, `"yes"`, `"on"`), and always quote the 40-hex revision.

## The reference interface

`reference.py` runs as a subprocess CLI, not an import: stage 2 launches it with the recipe's
`--reference-python` and a mode, and reads back a JSON file. It may import torch, transformers or the
checkpoint's remote code — the harness process never does. The contract (enforced by
`equivalence/reference.py`'s runner, which all fixture references implement):

```text
reference.py --mode <render|score|embed|media> --pairs <file> --out <file> \
             --tokenizer "<repo>@<revision>|path/to/tokenizer.json" \
             --recipe <resolved-recipe.json> --device <cpu|cuda:0>
```

`--recipe` is the resolved recipe the harness loaded (`Recipe.model_dump(mode="json")` written beside
`--out`): one reference runs every variant of the family, so the variant's facts (its `id`, `model`,
`revision` and its resolved `client` block) travel with the invocation. A reference that needs a per-size fact
its code does not carry (a paper batch size, a dimension) reads it from there, never from a sibling file.

- `--mode render` — stage 1's reference side. Embedding roles: `{"rows": [{"index", "shape", "text": str}]}` —
  the exact prompt the engine reads for that pair's declared shape (the fixed frame around the content; the
  post-processor's tokens are the engine's). Rerank: `{"rows": [{"index", "shape": "pair", "query": str,
  "documents": [str, ...]}]}` — the spans the client ships (the query as the recipe's instruction mode folds
  it, the documents as shipped; the frame is the engine's own template). One input per shape is rendered (the
  row's query for the query shape, its first document for the document shape).
- `--mode score` — rerank: `{"rows": [{"index", "scores": [float, ...]}]}` on the recipe's
  `reference.score_scale` (probability | logit | cosine), one score per document in the order the pairs file
  gives them.
- `--mode embed` — embedding roles: `{"rows": [{"index", "query_vectors": [...], "document_vectors": [...]}]}` —
  per text: one vector for a dense embedder, one per-token matrix for a late-interaction model (the same
  nesting for query and document sides, for every text of the row).
- `--mode media` — a recipe with image or video input: for every pairs row carrying `media`, per side
  (`query`, `document <i>`) that carries media, what the card's model consumes:
  `{"rows": [{"index", "side", "placement": ["image", "text"], "media": [{"kind": "image", "width", "height",
  "tokens"}]}]}` — the parts in the card's order, each image's size after the card's own resize and its
  prompt tokens (vision markers included); a video is the card's declared frame count (`{"kind": "video",
  "frames": N}` — its tokens are the engine's to count, so they are not compared here); a side the card
  cannot consume is `{"index", "side", "refused": str}`.
- The reference environment: the node's bootstrap installs the staged `requirements-reference.txt` (the
  package-level file, `--no-deps` over the image's freeze); a family may ship its own
  `recipes/<family>/requirements-reference.txt` beside its reference, and the bootstrap reads it only when
  `REFERENCE_REQUIREMENTS` names it (it never picks a recipe directory on its own). The harness documents
  the files and installs neither.

## Choosing how vLLM serves a model

The serving path is chosen per model — flags alone, a chat template, pooler settings, or a plugin — and the
decision tree lives in this section once the survey of model families lands; for now, a recipe's `serve` section
renders verbatim into `vllm serve` argv, and `serve.plugin` is reserved for a `vllm.general_plugins` package when
no flag can express the model's scoring (both ship in rcp-ndcg-vllm's folded models, which register
perplexity-ai/pplx-embed-v2-context-9b-preview's per-chunk pooling head and its late-interaction sibling
pplx-embed-v2-late-0.6b on the stock image; its README carries the client contracts). A recipe that names a
plugin also declares `plugin_architectures` — the architectures its engine registers — because the behaviour
fingerprint keys the plugin code by hashing exactly those modules (`plugin_sha256.<module>`, plus the shared
entry modules); `patches` opts into an engine patch by name and is rendered into the engine's
`RCP_NDCG_VLLM_PATCHES`, and the fingerprint hashes every opted-in patch's module too. A plugin spec whose
modules the harness cannot resolve (a third party's wheel) is refused at fingerprint time, by name.

## Worked example: a last-token-pooling embedder (CPU)

A last-token-pooling embedder's anchor is its trailing end token: the wrapper renders `prefix + text + suffix`
and the naive fix — truncating the whole string on the right — drops the token the model was trained to read
out of. The correct cut reserves the suffix, cuts only the text, and re-attaches the template. The mechanism in
ten dependency-free lines (the real implementation is the product's `fit` in `rcp_ndcg.data.preprocess` — the
same call the role clients make, with the recipe's real budget — audited against the captured wire in stage 1):

```python
# The template block of a last-token-pooling embedder, as segments:
#   [{text: "doc: "}, {content: document}, {text: "<end>"}], anchor: last.
prefix_ids = [1, 2]          # token ids of the "doc: " fixed head, measured with the recipe tokenizer
suffix_ids = [9]             # the end token: the anchor the model pools from
content_ids = [3, 4, 5, 6, 7, 8]  # the document, already tokenised

max_tokens = 5
budget = max_tokens - len(prefix_ids) - len(suffix_ids)
cut = content_ids[:budget]   # cut the content span only ...
prompt_ids = prefix_ids + cut + suffix_ids   # ... and re-attach the template
assert prompt_ids == [1, 2, 3, 4, 9]
assert prompt_ids[-len(suffix_ids):] == suffix_ids  # the anchor survived the cut
assert len(prompt_ids) == max_tokens
```

The same assertion in code lives in the package's tests (`test_stage1_passes_for_every_anchor_kind...`): for an
over-length input, every declared anchor id sits at its declared position in the assembled ids.

## The three equivalence stages

Stage 1 probes the recipe's role client for every sampled input through the product's injection point (the
engine when `--base-url` is given, the product's offline fake otherwise) and audits what the client actually
sends — the harness re-derives no render, no cut and no settlement. Each declared shape is sampled on its own
(at least 20 over-length inputs per shape, padded in that shape's own content span), and:

- `anchor_check` asserts every anchor survived the client's cut, on the captured requests of every shape —
  the rendered prompt's edge for the embed roles, and for a reranker the settle-once query (one settled span
  per row, identical across the row's pointwise requests, within its declared `query_max_tokens`, every
  document span within its declared `document_max_tokens`, and no cut on an in-budget pair). Every wire shape is read: a `token_ids` body is audited on the ids it sends, a
  `messages` body as the declared frame around its conversation's text parts (joined with `"\n"`, as the
  engine joins them; its media parts are listed as placeholders beside them): the client sends the content
  and the engine's chat template frames it. An audit that read no input fails. An `anchor: first` head is asserted as the engine reads it: where the head
  meets the content a byte-level BPE re-tokenizes across the join (`"doc: "` then reads `Ġdocument`; under a
  Qwen-style pre-tokenizer `"Query:"` reads `:Paris`), so the edge is the post-processor's prefix and the
  head's own tokens in the assembled render: a text body must start with the head's characters, and its
  tokens lying wholly inside them are the edge; a `token_ids` body (no text on the wire) must open with the
  head tokens that lie wholly inside the head whatever content follows (measured on the head joined to a set
  of letters, digits, punctuation, spaces, newlines and other scripts). A head character that merges into
  the content is then not asserted on a `token_ids` body; the render check compares those rows' ids whole;
- `render_check` compares the reference subprocess's `render` output against the captured texts, zero
  tolerance (a `token_ids` body on ids: the ids it sent against the reference text's ids under the shape's
  `add_special_tokens` flag) — every declared shape of every pairs-file row (a row carrying the per-row `shape` field is
  compared too; the injected over-length samples are audited, not compared). Under a declared over-cap
  deviation, the rows the client changed are reported in a separate non-gating table here as well (the
  reference renders them its own way by declaration): every row the client's census records a cut for, with its
  `cause` -- the budget counted with the frame (a content under `max_tokens` whose framed request is over it),
  the reranker's query share (also inside a pair the budget takes whole) or a declared per-document cap. The
  decision is per text: a reranker's settled query changes every pair of its row, a cut document only its
  own pair, and every text the client sent uncut gates exactly, also beside a changed sibling;
- `engine_tokenize_check` (R29, needs the engine) requires the engine's `/tokenize` ids and counts of every
  captured text to equal the recipe tokenizer's; reported `not_run` without an engine, never as passed (and
  for a `token_ids` client, which sends ids and leaves the engine nothing to tokenize);
- `template_render_check`, when `serve.chat_template` is set: the template file's jinja2 render (the engine's
  settings) against the declared template's render, for every declared shape. On the `messages` route the file
  is the engine's chat template: it is rendered over every conversation the client sent, with the
  `add_generation_prompt` flag that request carried, and must equal the declared frame around that content,
  once. Without `serve.chat_template` the engine renders the checkpoint's own chat template, and the check
  reads it at the pinned revision (`chat_template.jinja`, else `chat_template.json`, else
  `tokenizer_config.json`, from the Hub cache or the Hub) and renders that; a template it cannot read fails
  the check (`unresolved`), never passes.

Stage 2 sends the same pairs through the product's role clients (`EmbeddingClient`, `PoolingClient`,
`RerankClient`) built from the recipe's real budget — the client prompts, fits and settles exactly as the
served path does (for a reranker, the shared query span settles once per call) — and applies the gates:
probability |Δ| ≤ 0.02 for 99% of documents and ≤ 0.05 for all; logit |Δ| ≤ 0.05·(1 + |s|); cosine scores
|Δ| ≤ 0.01; vectors cosine ≥ 1 − 1e-3 per vector (per token, after the same float16 cast); median per-query
Kendall τ ≥ 0.98. A recipe's `gates` section overrides any of these. With
a declared over-cap deviation, the inputs the client changed (decided on the client's own census rows and
their `cause`; a reranker's query settlement changes every pair of its call) are reported in a separate,
non-gating table — in stage 2 for every role — and the gates run on the inputs sent uncut only.

Stage 3 (optional) scores rankings per subset with `rcp-ndcg eval score` as a subprocess (the package depends
on `rcp-ndcg`, so the command is always available) and requires the mean |Δ nDCG@10| over subsets ≤ 2e-3.

```bash
python -m rcp_ndcg_test.equivalence --recipe <variant-id> --base-url http://127.0.0.1:8100 \
    --pairs pairs.jsonl --out /tmp/equiv            # stages 1 and 2 against a running engine
```

A recipe with image or video input also runs the **media stage** beside stages 1 and 2 (stages 1 and 2
compare the pairs file's text rows; its media rows are this stage's). A media row carries `media: {"query":
[...], "documents": [[...], ...]}`, each entry a `MediaRef` object (the bytes inline as a `data:` URI) plus
its `kind` — `image`, `video`, or, in a part sequence, `text` (an interleaved row's text segments, standing
where they stand); a side's content is its entries in order, then its text. The stage sends each media side
through the role client and reads what crossed the wire — the parts in order, each image's prepared size
decoded from the sent bytes, the tokens the client counted — against the reference's `--mode media`; with an
engine, it sends each media request again without its media parts, and the difference of the two
`usage.prompt_tokens` is the engine's own media count, which must equal the client's (an engine whose pixel
pin is missing re-resizes a prepared image and fails it; an engine not pinned to the declared video sampling
decodes other frames and fails it the same way). On CPU the test stub engine resizes with the product's own
`smart_resize` and counts a container through the product's own `content_media_tokens`, so there the engine
count catches a pin that is missing or different, never a bug in the product's resize or video accounting
itself: on CPU only the comparison with the reference's card resize can catch
that, and the engine count is an independent check only against a real engine. Every image gates exactly, a
video's declared frame count gates against the reference and its container's count against the engine, and an
interleaved row gates the given part order (the fitted text stands where the side's first text part stood); a
media recipe whose
pairs carry no media row fails. The pairs generator plans the media rows (one image per size bucket, a
captioned page, a batch mixing a text-only and an image document, a query image where the recipe allows query
media, interleaved and several-image documents where its `max_images` admits them, and an MJPEG AVI clip per
size, alone and with text, at the recipe's declared video sampling — `rcp_ndcg_test.observe.media_set`).

The exit code is 0 only when every gate passes; `equivalence.json` carries every number with its referent
(per document, per query, per subset) and `EQUIVALENCE.md` is the short section for the recipe's report.

## Submitting a wave

A wave runs on a node against a staged release candidate: [rc_build.sh](release-candidates.md#build-a-release-candidate)
stages the wheels, the wheelhouse, the recipes and the wave lists; `submit.sh` submits the wave's job against
that stage, and the node's `bootstrap.sh` builds the three environments and runs the wave runner:

```bash
export RCP_KJOBS_CONFIG=/path/to/jobs-config.yaml    # the job CLI's -f config (required, no default)
export RCP_GCS_AUTH_FILE=/path/to/gcs_auth.sh        # mounted at /etc/rcp/gcs_auth.sh; named, never read
export RCP_HF_TOKEN_FILE=/path/to/token              # passed as a kjobs secret, never read or echoed
rcp-ndcg-test/src/rcp_ndcg_test/jobs/submit.sh gs://YOUR-BUCKET/stage/rc0 gs://YOUR-BUCKET/waves <wave-name>
```

The three variables are required — the script refuses to run without them, because no tracked file may name a
machine's paths or a token. The recipe list is the staged `<RC>/wave-lists/<wave>.txt`; `--priority dev-high`,
`--max-jobs N`, `--script wave0` (the node test), `--script e2e` (the T4 run scenarios, whose wave list names
scenario ids) and `KJOBS=echo` (print the plan instead of submitting) are
optional. The full procedure — the candidate, the wheelhouse, the three environments and wave 0 — is in
[Release candidates and the GPU waves](release-candidates.md). The wave packs recipes onto the node's GPUs,
serves one engine per slot with its own GPU slice, port, `VLLM_PORT` and `TMPDIR`, checks the free disk and
evicts each model's weights after its last use, runs smoke, equivalence and the recorder, and writes
`<out>/<id>/{serve.log, equivalence.json, status.json}` plus a summary.

## Reference cases and the conformance suite

A recipe's behaviour is pinned by reference cases in the unpublished `rcp-ndcg-test` package
(`rcp-ndcg-test/`, a workspace member; never on PyPI): one case per file under
`rcp-ndcg-test/cases/<recipe-id>/`, each carrying the card's verbatim example or a generated
stratum, an `expected` block with its tolerance, and the strata cell it covers. The cases are validated on
every push (a malformed case or an incomplete strata grid fails CI), and the conformance runner sends each
case through the product's role clients — against a live engine (`target="engine"`, the engine's
`base_url` passed to `run_suite`) or a recipe-level fake engine (`target="fake"`, resolved through the
registry by recipe id) — comparing with `expected` under its tolerance. `expected.values: null` (a
generated case before its GPU wave) is a skip, never a pass. See `rcp-ndcg-test/README.md` for
the case format, how to add a case, and the fake-engine seam the verified emulators are later built from.
