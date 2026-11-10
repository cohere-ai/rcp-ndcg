# Add a model: serving recipes for vLLM

A recipe is one declarative description of how a model is served with vLLM and how `rcp-ndcg` reads it back.
Recipes are grouped into **families**: one directory per model family. It holds the shared serving contract and a `variants` table with only the per-size facts. Thus adding a size is adding a variant row. Every size stays its
own tested recipe id: served, contract-tested, stage-1-tested and GPU-validated on its own. A family id is
never served. This guide shows the format. It shows how to check a served recipe against its reference
implementation. It also shows how a wave of recipes is submitted to the GPU host. The package lives at
`rcp-ndcg-vllm/` and is installed into the engine image, which carries its own vLLM and torch. A family directory of your own needs none of that packaging to be *used*. `rcp-ndcg-vllm serve ./my-family/ [--variant <id>]` and `recipe:./my-family/` load it through this same schema. The loader marks it unshipped and
`unverified` ([serve a retrieval model](serve-a-model.md)). The class-specific doctrine is [adopt a
model](adopt-a-model.md). It states what parity means for an embedding, a reranker, a late-interaction model and a judge. It also lists what to look out for.

## The family directory

One directory per model family: `rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/<family>/`. It is package data, shipped
inside the wheel. It has these four files. A family may also ship a vendored card script that its reference runs
verbatim. That script is byte-identical to the Hub file and hash-pinned by the family's test.

```text
recipes/<family>/
  family.yaml                  # the family: the shared blocks + the variants table (the RecipeFamily schema;
                               # every field is listed in schema/family.schema.json)
  template.jinja               # the family's ONE chat template for vllm serve --chat-template (only when the model needs one)
  reference.py                 # the family's ONE reference implementation, run as a subprocess per variant
                               # (see the reference interface)
  reference.in                 # the family's reference pins, each justified from the card or the reference code
  reference.lock               # the generated, hashed lock the node installs into the family's reference venv
```

`family.id` equals the directory name and matches `^[a-z0-9][a-z0-9.-]*$`. It is never served. Each row of the `variants` table is a full recipe id (`^[a-z0-9][a-z0-9.-]*$`, the lowercased canonical Hub repo name). It is also the `--served-model-name` that the engine serves, and what `recipe: <variant id>` in `rcp-ndcg` resolves. A
single-size model is a family with one variant; one loader path serves both. The schema is closed
(`extra="forbid"`) and role-aware. A field that only makes sense for one role is refused for the others. Thus
a typo cannot silently change what is served.

**Adding a size to an existing family** is one row plus its pins plus its tests:

1. Append a `variants` row: `id`, `model`, `revision`, and the whitelisted per-size `overrides` only. These
   are `resources`, `serve.max_model_len`/`dtype`/`hf_overrides`/`mm_processor_kwargs`/
   `limit_mm_per_prompt`/`extra_args`/`patches`, `client.max_tokens`/`query_max_tokens`/`document_max_tokens`/
   `dim`/`dimensions`/`mrl_kind`/`mrl_dims`/`mrl_range`/`mrl_projection`/`batch_size`/`max_images`/
   `max_videos`, and `gates`. That value is the size's measured stage-2 bound where it differs from the family's. Also
   give the per-size `notes`, `sources`, `status` and `licence` (when the family's checkpoints are licensed
   differently). Anything else that differs is refused with a typed error naming the field. The shared blocks
   are the family's contract. Thus a size that behaves differently is its own family. A size's Matryoshka head
   is a per-size fact, because the card's set changes with the width. Declare `mrl_kind` (`truncation`,
   `projection` or `none`) with the card's `mrl_dims`/`mrl_range`. Where the engine's per-request `dimensions`
   path exists, declare the same set in `serve.hf_overrides` (`is_matryoshka`/`matryoshka_dimensions`). The
   loader refuses a serve gate and a client declaration that disagree. The recipes ship the checkpoint's full
   width. A run selects `k` from the declared set.
2. Pin the variant's fields in the family's test module. There is one module per family, parametrized over its
   variants. Pin every field per variant, with two mutants red per family. Run the family's stage-1 network
   tests for the new variant.
3. Add the variant's golden. `uv run --no-sync pytest rcp-ndcg-test/tests/recipes/test_family_goldens.py
   --update-goldens` rewrites `tests/recipes/golden/` from the current tree. Review the diff, then commit. The
   guard compares every variant's resolved contract and fingerprint against its golden in every CI job.
4. A new family needs its `family.yaml` (below), its reference and its family test module. It also needs its template when it needs one, and its variant goldens.

Validate a family/variant without an engine:

```bash
python -m rcp_ndcg_test.equivalence --recipe <variant-id> --pairs pairs.jsonl --out /tmp/equiv --stages 1
```

## The template block, the anchors and the explicit budget

Three research findings shape the `serve` and `client` blocks. The schema enforces them:

- **Anchors.** A model reads its output from fixed positions of its template. These positions are the *anchors*. Examples are a last-token pooler's end token, a CLS head's leading special, a marker id and a reranker's assistant suffix. Every cut
  applies to the content spans only, inside a budget computed after reserving every fixed template token. The
  template is re-attached after the cut. Engine-side truncation of a rendered prompt cannot honour this in
  either direction. Thus the schema has no engine-truncation field at all. A recipe whose reference
  deliberately drops anchors declares `reference.known_deviations: [anchor_drop_over_cap]`. A recipe's reference can keep the anchors but cut over-cap content its own way. An example is a joint `longest_first` truncation, where the client settles the query at its share. Such a recipe declares `[over_cap_cut_differs]`. Either way the harness
  reports those pairs separately, outside the gates. The reference stays the paper's or the model card's. It
  never copies the client's cut to make an over-cap pair pass. The declared shape's `anchor` is one of these values: `last`, `first`, `last_content`, `mean` or `marker` (with `anchor_markers`). The last-content form suits a model that pools the last real token of raw text. The shape ends on its content span. The audit asserts the head marker opening the render and a content token closing it, before the post-processor's tail. The mean form has no anchor token. The audit asserts what a cut must keep: the declared fixed edges survive. At least one content token sits between them and the post-processor's tokens. Stage 1 samples over-length inputs (at least 20 per shape) and
  asserts that every anchor survived. It reports this as `anchor_check`, separately from token-id mismatches.
- **Segments: `fixed` and `content`, nothing else.** A shape is an ordered list of segments with exactly two
  kinds. `fixed` is ordinary template text. It may name special tokens as `{special:<name>}` placeholders that the recipe tokenizer's added tokens resolve at run time. `content` is the cuttable span (`query` or
  `document`). The schema is closed, so `{"special": ...}`, `{"text": ...}` or `{"ids": ...}` segments are
  refused. Every declared shape needs at least one content span. An `anchor: last` shape must end with a fixed
  segment, or pin `add_special_tokens: true`, declaring the tokenizer's end token as the anchor. When `serve.chat_template` is set, stage 1 also proves one more thing. The declared shapes render to the same token ids as the template file.
- **Explicit budgets.** Every recipe declares `client.tokenizer` and `client.max_tokens`. There is no implicit
  default. `on_overflow` is `cut` by default: content spans only, at token boundaries, anchors reserved.
  `chunk` and `fail` are the alternatives. `aggregation: max` is the only chunk aggregation. The engine's
  `max_model_len` must fit the client's budget. `serve.pooler_config` keys are validated against the pinned
  engine's `PoolerConfig` fields, because the engine rejects unknown keys.
- **The engine is the tokenization truth (R29).** With an engine URL, stage 1 takes `fit`'s rendered prompts and sends them to the engine's `/tokenize`. It uses the same `add_special_tokens` as the route. It requires the ids and counts to
  equal `fit`'s. Mismatches are reported per shape in `equivalence.json` (`engine_tokenize_check`) and fail
  the stage. With an engine URL it also reads the engine's own `usage.prompt_tokens` of one captured request
  per shape. It requires that count to equal the token count of the render that the client budgeted against
  (`engine_prompt_tokens_check`). This check proves that the declared frame the `messages` budget rests on is
  the frame the engine renders. A listwise reranker's prompt is its reference's builder, so that recipe reports
  `not_run` with the reason. Without an engine (CPU), both checks are reported as `not_run`, never as passed.
- **The reference runs as a subprocess.** Stage 2 runs the recipe's `reference.py` as a subprocess. `--reference-python <path>` is required when stage 2 runs, with no default. The subprocess reads the pairs file
  and writes scores or vectors to a file that the harness compares. The harness process imports no torch or
  transformers. The reference environment is the family's `reference.in`/`reference.lock` (below). The engine
  comes up on the slot's GPUs first. The reference subprocess runs against the pairs file while the engine is
  up and releases its memory when it exits.

RecipeFamily YAML at a glance. This is a complete, loadable family: `tests/docs` runs `load_family` and `resolve_recipe` on it. The schema's docstrings define every field:

```yaml
id: example-reranker               # the family id; never served
schema_version: "1"              # the family/recipe file format's version
role: rerank                     # embed | multi_vector | rerank
input: [text]                    # subset of [text, image, video]
scoring: pointwise               # rerank only: pointwise | listwise
licence: apache-2.0              # SPDX id or see-model-card
engine: {name: vllm, image: "vllm/vllm-openai:v0.33.0", min_version: "0.33.0"}
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
  plugin_architectures: []       # optional: the plugin's architectures this recipe's engine registers (the
                                 # behaviour fingerprint keys their modules); a plugin with none is a
                                 # patch-only carrier, keyed by its patch names
  patches: []                    # engine patch names this recipe opts into; serve renders them into RCP_NDCG_VLLM_PATCHES
  extra_args: []                 # further flags, verbatim (one argv element per item); never a flag
                                 # serve_argv renders (--revision, --max-model-len, ...): the engine keeps the
                                 # last occurrence, so a duplicate would silently override the rendered value
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
  instruction: fold              # rerank: none | field | fold (system is refused); embed/pool: none | fold.
                                 # fold = the generic task-instruction default Task: <instruction>\nQuery: <text>;
                                 # a template {content: instruction} span places it instead (the engine renders
                                 # the span on a rerank wire, so the request carries the field; a wire without
                                 # the field refuses the combination). For embed/pool, leaving it unset means
                                 # UNDECLARED: a dataset with a task instruction is refused, never silently
                                 # changed -- declare fold or none.
  title: join                    # join (default) = MTEB's (title + " " + body).strip(); separate = the title
                                 # as its own leading text part. The sparse (BM25) path follows mteb's own
                                 # BM25 instead: title + "\n" + body, no task instruction
  use_activation: true           # a served rerank wire must set it: the score's scale is content
  on_overflow: cut               # cut (default) | chunk | fail; cuts apply to content spans only
  empty_doc: send                # omit_zero | omit_zero_blank | send | send_text
reference:
  kind: transformers             # transformers | sentence_transformers | remote_code | stored_scores
  score_scale: probability       # probability | logit | cosine; vectors compare per vector
  entry: reference.py
  device: cpu                    # cpu | cuda; cuda only where a CPU reference is impossible or the gate moves
  attn_implementation: sdpa      # sdpa | flash_attention_2 | eager; declared, never chosen by
                                 # torch.cuda.is_available() (the stock reference environment has no
                                 # compiled extras); the reranker families declare sdpa
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
    #   gates: {vec_min_cosine: 0.9987, media_vec_min_cosine: 0.9975}   # this size's measured bounds; unset fields keep the family's
```

Two YAML pitfalls are caught in review and by the golden tests. Always quote string tokens that YAML reads as booleans (`"no"`, `"yes"`, `"on"`). Always quote the 40-hex revision.

## The reference interface

`reference.py` runs as a subprocess CLI, not an import. Stage 2 launches it with the recipe's
`--reference-python` and a mode, and reads back a JSON file. It may import torch, transformers or the
checkpoint's remote code. The harness process never does. The contract is enforced by `equivalence/reference.py`'s
runner, which all fixture references implement:

```text
reference.py --mode <render|score|embed|media> --pairs <file> --out <file> \
             --tokenizer "<repo>@<revision>|path/to/tokenizer.json" \
             --recipe <resolved-recipe.json> --device <cpu|cuda:0>
```

`--recipe` is the resolved recipe that the harness loaded (`Recipe.model_dump(mode="json")` written beside
`--out`). One reference runs every variant of the family. Thus the variant's facts (its `id`, `model`,
`revision` and its resolved `client` block) travel with the invocation. A reference may need a per-size fact that its code does not carry. An example is a paper batch size or a dimension. It reads that fact from there, never from a sibling file.

- `--mode render` — stage 1's reference side. Embedding roles: `{"rows": [{"index", "shape", "text": str}]}`.
  This is the exact prompt that the engine reads for that pair's declared shape: the fixed frame around the content. The post-processor's tokens are the engine's. Rerank: `{"rows": [{"index", "shape": "pair",
  "query": str, "documents": [str, ...]}]}`. These are the spans that the client ships. The recipe's instruction mode folds the query. The documents are sent as shipped. The frame is the engine's own template. One
  input per shape is rendered: the row's query for the query shape, its first document for the document shape.
- `--mode score` — rerank: `{"rows": [{"index", "scores": [float, ...]}]}` on the recipe's `reference.score_scale` (probability | logit | cosine). Give one score per document in the order the pairs file gives them.
- `--mode embed` — embedding roles: `{"rows": [{"index", "query_vectors": [...], "document_vectors": [...]}]}`.
  Per text: one vector for a dense embedder, one per-token matrix for a late-interaction model. The nesting is the same for query and document sides, for every text of the row.
- `--mode media` — a recipe with image or video input. A pairs row can carry `media`. For each side (`query`, `document <i>`) that carries media, the output is what the card's model consumes: `{"rows":
  [{"index", "side", "placement": ["image", "text"], "media": [{"kind": "image", "width", "height",
  "tokens"}]}]}`. The parts are in the card's order. Each image's size is after the card's own resize, and its
  prompt tokens include the vision markers. A video is the card's declared frame count (`{"kind": "video", "frames": N}`). Its tokens are the engine's to count, so they are not compared here. A side that the card
  cannot consume is `{"index", "side", "refused": str}`.
- The reference environment: each family ships a short `reference.in` and the generated, hashed `reference.lock`. Each pin is justified from the card or the reference code with a `file:line` comment. The node builds ONE venv per family from the lock. It uses `--system-site-packages` over the engine image's torch/CUDA. The family's own pins are installed into the venv and take precedence over the image's copies. Thus one recipe's
  transformers pin cannot break the wave. The tool constrains torch and the CUDA stack to the image's freeze
  and nothing else. A family whose reference genuinely needs another torch declares `# own-torch: true` with its evidence in `reference.in`. It gets a venv of its own.

  Generate or regenerate the lock (from the checkout; the tool needs the index only here, never on the node):

  ```bash
  uv run --no-sync python -m rcp_ndcg_test.jobs.reference_lock build --family <family> \
      --in rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/<family>/reference.in \
      --out rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/<family>/reference.lock \
      --image-freeze rcp-ndcg-vllm/reference-image-v0.33.0.txt \
      --image <the family's engine.image from family.yaml>
  ```

  A digest-pinned nightly may have a torch/CUDA stack that is not committed. For it, add `--image-freeze-source vllm/vllm-openai:v0.33.0`, the released image the floors were checked against. The lock then records `image-freeze-sha256: uncommitted` with that source, never a false hash.

  `... check --family <family> --in ... --lock ... --image-freeze ...` validates a committed lock offline (the header hashes, every pin, the own-torch declaration). `rcp-ndcg-test/tests/test_reference_lock.py` runs the check over every family. A family pin with no index
  wheel (e.g. `flash-attn`, a GitHub-release wheel) must be staged in an `EXTRA_DIRS` wheelhouse, which the
  node's install searches. The harness documents the files and installs neither.

## Choosing how vLLM serves a model

The serving path is chosen per model: flags alone, a chat template, pooler settings, or a plugin. A recipe's
`serve` section renders verbatim into `vllm serve` argv. `serve.plugin` is reserved for a `vllm.general_plugins`
package when no flag can express the model's scoring. The wheel's folded models register every shipped plugin on the stock image. They are `topk-embed-v1-small`/`-xsmall` (`TopkEmbedModel`) and pplx-embed-v1's config-only `PplxV1Config`. They also include the pplx-embed-v2-context-9b-preview per-chunk pooling head (`PplxContextualModel`) and its late-interaction sibling pplx-embed-v2-late-0.6b (`Qwen3_5Model`). The last one is embeddinggemma-2's `EmbeddingGemma2Config`/`EmbeddingGemma2Model` backport. The README carries the client contracts. A recipe that names a plugin also declares `plugin_architectures`: the architectures that its engine registers. The list is empty for a patch-only carrier, whose patch modules are keyed anyway. The behaviour fingerprint keys the plugin code by
hashing exactly those modules (`plugin_sha256.<module>`, plus the shared entry modules). `patches` opts into an
engine patch by name and is rendered into the engine's `RCP_NDCG_VLLM_PATCHES`. The fingerprint hashes every
opted-in patch's module too. A plugin spec whose modules the harness cannot resolve (a third party's wheel) is
refused at fingerprint time, by name.

## Worked example: a last-token-pooling embedder (CPU)

A last-token-pooling embedder's anchor is its trailing end token. The wrapper renders `prefix + text + suffix`.
The naive fix truncates the whole string on the right. It drops the token that the model was trained to read out of. The correct cut reserves the suffix, cuts only the text, and re-attaches the template. The mechanism is
ten dependency-free lines. The real implementation is the product's `fit` in `rcp_ndcg.data.preprocess`. It is the same call that the role clients make, with the recipe's real budget. Stage 1 audits it against the captured
wire:

```python
# The template block of a last-token-pooling embedder, as segments:
#   [{fixed: "doc: "}, {content: document}, {fixed: "<end>"}], anchor: last.
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

The same assertion in code lives in the package's tests (`test_stage1_passes_for_every_anchor_kind...`). For an
over-length input, every declared anchor id sits at its declared position in the assembled ids.

## The three equivalence stages

Stage 1 probes the recipe's role client for every sampled input through the product's injection point. The engine is used when `--base-url` is given, and the product's offline fake otherwise. It audits what the client actually
sends. The harness re-derives no render, no cut and no settlement. Each declared shape is sampled on its own, with at least 20 over-length inputs per shape. The inputs are padded in that shape's own content span. The checks are:

- `anchor_check` asserts that every anchor survived the client's cut, on the captured requests of every shape.
  For the embed roles this is the rendered prompt's edge. For a reranker it is the settle-once query. There is one settled span per row, identical across the row's pointwise requests and within its declared `query_max_tokens`. Every document span stays within its declared `document_max_tokens`, and an in-budget pair has no cut. Every wire
  shape is read. A `token_ids` body is audited on the ids it sends. A `messages` body is audited as the declared frame around its conversation's text parts. The parts are flattened with `"\n"`, vLLM's *string* content format's join, vllm/entrypoints/chat_utils.py's `_get_full_multimodal_text_prompt`. Its media parts are listed as placeholders beside them. Thus the client sends the content and the engine's chat template frames it. An
  audit that read no input fails. An `anchor: first` head is asserted as the engine reads it. Where the head meets the content, a byte-level BPE re-tokenizes across the join. For example, `"doc: "` then reads `Ġdocument`; under a Qwen-style pre-tokenizer `"Query:"` reads `:Paris`. Thus the edge is the post-processor's prefix and the
  head's own tokens in the assembled render. A text body must start with the head's characters, and its tokens
  lying wholly inside them are the edge. A `token_ids` body has no text on the wire. It must open with the head tokens that lie wholly inside the head whatever content follows. The measurement uses the head joined to a set of letters, digits, punctuation, spaces, newlines and other scripts. A head character that merges into the
  content is then not asserted on a `token_ids` body. The render check compares those rows' ids whole.
- `render_check` compares the reference subprocess's `render` output against the captured texts, with zero
  tolerance. A `token_ids` body is compared on ids. The ids it sent are compared against the reference text's ids under the shape's `add_special_tokens` flag. The check covers every declared shape of every pairs-file row and **every
  text the client shipped**. The reference is asked to render each document, so a row's second and later
  documents are compared too. A row carrying the per-row `shape` field is compared too. The injected
  over-length samples are audited, not compared. Under a declared over-cap deviation, the rows the client changed are reported in a separate non-gating table here as well. The reference renders them its own way by declaration. The table lists every row that the client's census records a cut for, with its `cause`. The cause can be one of three things. It can be the budget counted with the frame (a content under `max_tokens` whose framed request is over it). It can be the reranker's query share (also inside a pair the budget takes whole). Or it can be a declared per-document cap. The decision is per text. A
  reranker's settled query changes every pair of its row; a cut document changes only its own pair. Every text
  that the client sent uncut gates exactly, also beside a changed sibling.
- `engine_tokenize_check` (R29, needs the engine) requires the engine's `/tokenize` ids and counts of every captured text. They must equal the recipe tokenizer's. It is reported `not_run` without an engine, never as passed.
  It is also not run for a `token_ids` client, which sends ids and leaves the engine nothing to tokenize.
- `template_render_check`, when `serve.chat_template` is set: the template file's jinja2 render (the engine's settings) against the declared template's render. There is one row per declared shape. A media row is included where the pairs file carries one. Thus the frame around a media item's content is checked too. On the `messages`
  route the file is the engine's chat template. It is rendered over every conversation the client sent, one media row's conversation per declared shape included. It uses the `add_generation_prompt` flag that request carried. It must equal the declared frame around that content, once. For a media conversation the comparison is on the conversation's text. The media placeholder is the engine's own expansion, counted by the media stage. The render WITH the media must still open and close with the declared frame's fixed edges. A media side's text is read as the engine's `openai` content format hands it to the template. The parts are in order, back to back. The served template defines the join between them. The client's `TEXT_JOIN` flattening is the string content format's. Thus a template that inserts a separator between a message's text parts fails the check. It does not pass on the flattened string. Without `serve.chat_template` the engine
  renders the checkpoint's own chat template. The check reads it at the pinned revision (`chat_template.jinja`, else `chat_template.json`, else `tokenizer_config.json`, from the Hub cache or the Hub). Then it renders that. A
  template it cannot read fails the check (`unresolved`), never passes.

Stage 2 sends the same pairs through the product's role clients (`EmbeddingClient`, `PoolingClient`,
`RerankClient`), built from the recipe's real budget. The client prompts, fits and settles exactly as the served path does. For a reranker, the shared query span settles once per call. Then the gates apply. The probability gate is |Δ| ≤ 0.02 for 99% of documents and ≤ 0.05 for all. The logit gate is |Δ| ≤ 0.05·(1 + |s|). The cosine gate is |Δ| ≤ 0.01. The vector gate is cosine ≥ 1 − 1e-3 per vector (per token, after the same float16 cast). The ranking gate is median per-query Kendall τ ≥ 0.98. A recipe's `gates` section overrides any of these. A family declares the shared overrides,
while a size whose measured precision floor differs overrides individual fields in its `overrides.gates`. A
set field replaces the family's value. An unset field keeps it. A tighter family bound is never loosened
silently. With a declared over-cap deviation, the inputs that the client changed are reported in a separate, non-gating table. Stage 2 does this for every role. The decision uses the client's own census rows and their `cause`. A reranker's query settlement changes every pair of its call. The gates run on the inputs sent uncut only.

Stage 3 (optional) scores rankings per subset with `rcp-ndcg eval score` as a subprocess. The package depends
on `rcp-ndcg`, so the command is always available. The stage requires the mean |Δ nDCG@10| over subsets ≤ 2e-3.

```bash
python -m rcp_ndcg_test.equivalence --recipe <variant-id> --base-url http://127.0.0.1:8100 \
    --pairs pairs.jsonl --out /tmp/equiv --reference-python <reference-venv-python>   # stages 1 and 2
```

A recipe with image or video input also runs the **media stage** beside stages 1 and 2. Stage 2 now compares
the media rows too. The reference reads their `media` field. An inline image is a loaded image prepared as the served path sends it. It uses the card's own resize under the declared budget, at the size that the engine's processor keeps. Thus the two sides see the same pixels. An image that the engine would resize again is a different input. The media rows' cosines move by exactly that much. A container is at the recipe's declared
video pin, and the parts stay in order. The reference computes the card's own vectors or scores for them, so
the same gates apply. A recipe whose card's own pipeline cannot run an input declares `reference.known_deviations: [media_approximation]`. That declaration reports its media rows non-gating with the reason in its notes.

**The media stage itself is an INPUT gate.** It compares what the client *sends* with what the reference consumes. With an engine, it also compares what the engine *counts*. The media stage never compares vectors or scores; that is
stage 2's half. Thus a passing media stage proves that the served path shows the model the same media. It never
proves that the model returns the same numbers. Its document carries `scope: input` and the `scope_note` saying
so, and `EQUIVALENCE.md` prints the scope.

A media row carries `media: {"query": [...], "documents": [[...], ...]}`. Each entry is a `MediaRef` object (the bytes inline as a `data:` URI) plus its `kind`. The kind is `image`, `video`, or, in a part sequence, `text`. That kind is an interleaved row's text segments, standing where they stand. A side's content is its entries in order, then its
text. The stage sends each media side through the role client. It reads what crossed the wire: the parts in order and the tokens that the client counted. It also reads each image's prepared size, decoded from the sent bytes. It compares
them against the reference's `--mode media`. With an engine, it sends each media request again without its
media parts. The difference of the two `usage.prompt_tokens`, less the engine's own render merge, is the engine's own media count. It must equal the client's. An engine whose pixel pin is missing re-resizes a
prepared image and fails it. An engine not pinned to the declared video sampling decodes other frames and
fails it the same way.

The render merge is the boundary token that a media placeholder moves when it stands between a message's text
parts. The engine's prompt is its render (the served template with the placeholder) minus the placeholder plus
the expansion. A tokenizer is not additive, so the word after the placeholder can lose the token that carried
the space. The check reads the merge from the served template's own render
(`rcp_ndcg_test.equivalence.stages.engine_render_merge`), never guesses it. Thus a genuine engine difference
still fails. On CPU the test stub engine resizes with the product's own `smart_resize`. It counts a container through the product's own `content_media_tokens`. There the engine count catches a pin that is missing or different. It never catches a bug in the product's resize or video accounting itself. On CPU, only the comparison with the reference's card resize can catch that. The engine count is an independent check only against a real engine. Every image gates exactly. A video's declared frame count gates against the reference, and its
container's count gates against the engine. An interleaved row gates the given part order; every text part
stands where it stands around the media. A media recipe whose pairs carry no media row fails. The pairs generator plans the media rows. There is one image per size bucket, a captioned page, and a batch. The batch mixes a text-only and an image document. It adds a query image where the recipe allows query media. It also adds interleaved and several-image documents where its `max_images` admits them. It also adds an MJPEG AVI clip per size, alone and with text, at the recipe's declared video sampling (`rcp_ndcg_test.observe.media_set`).

The exit code is 0 only when every gate passes. `equivalence.json` carries every number with its referent (per document, per query, per subset). `EQUIVALENCE.md` is the short section for the recipe's report.

## Submitting a wave

A wave runs on a node against a staged release candidate.
[rc_build.sh](release-candidates.md#build-a-release-candidate) stages the wheels, the wheelhouse, the recipes
and the wave lists. `submit.sh` submits the wave's job against that stage. The node's `bootstrap.sh` builds the
three environments and runs the wave runner:

```bash
export RCP_JOBS_CONFIG=/path/to/jobs-config.yaml     # the job CLI's -f config (required, no default)
export RCP_GCS_AUTH_FILE=/path/to/gcs_auth.sh        # mounted at /etc/rcp/gcs_auth.sh; named, never read
export RCP_HF_TOKEN_FILE=/path/to/token              # mounted into the job and read there, never passed in an argv
rcp-ndcg-test/src/rcp_ndcg_test/jobs/submit.sh gs://YOUR-BUCKET/stage/rc0 gs://YOUR-BUCKET/waves <wave-name>
```

The three variables are required. The script refuses to run without them, because no tracked file may name a
machine's paths or a token. The recipe list is the staged `<RC>/wave-lists/<wave>.txt`. These options are optional: `--priority <class>`, `--max-jobs N` and `--script wave0` (the node test). Another option is `--script e2e` for the T4 run scenarios, whose wave list names scenario ids. `RCP_JOBS_CLI=echo` prints the plan instead of submitting.
The full procedure is in [Release candidates and the GPU waves](release-candidates.md). It covers the candidate, the wheelhouse, the three environments and wave 0. The wave packs recipes onto the node's GPUs. It serves one
engine per slot with its own GPU slice, port, `VLLM_PORT` and `TMPDIR`. It checks the free disk and evicts each
model's weights after its last use. It runs smoke, equivalence and the recorder. It writes
`<out>/<id>/{serve.log, equivalence.json, status.json}` plus a summary.

## Reference cases and the conformance suite

A recipe's behaviour is pinned by reference cases in the unpublished `rcp-ndcg-test` package (`rcp-ndcg-test/`,
a workspace member; never on PyPI). There is one case per file under `rcp-ndcg-test/cases/<recipe-id>/`. Each case carries the card's verbatim example or a generated stratum. It also carries an `expected` block with its tolerance, and the strata cell it covers. The cases are validated on every push: a malformed case or an incomplete strata grid
fails CI. The conformance runner sends each case through the product's role clients. The target is a live engine (`target="engine"`, the engine's `base_url` passed to `run_suite`) or a recipe-level fake engine. The fake engine is resolved through the registry by recipe id (`target="fake"`). It compares with `expected` under its
tolerance. `expected.values: null` (a generated case before its GPU wave) is a skip, never a pass. See `rcp-ndcg-test/README.md` for the case format and how to add a case. It also describes the fake-engine extension point that the verified emulators are later built from.
