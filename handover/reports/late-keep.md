# Report `late-keep`: the exact late-interaction keep-rules, applied in the engine plugin

**Status:** DONE. The rule is declared once in the recipe, rendered for the engine in `serve.hf_overrides`
with its gate, applied engine-side by the plugin's own pooler, and checked client-side against the declared
kept count. Two families ship it: **pplx-embed-v2-late** (the checkpoint's own document mask: 32 punctuation
ids, gated by the document role prefix, plus the trained `[D] ` head as a system message) and
**topk-embed-v1** (the reference's image keep-mask as a media allowlist: `[248056]`, the image-patch token).
Base: `743d6a63` (rfc-0001 after media-rules and the recipe line); the branch merged the moved `rfc-0001`
twice: `01f4b9be` (l10c and qa-prep) at `eb167382`, and `45b66e1b` (fp-v4/`rcp-fp/4` and the judge recipes)
at `1025d0dd`.

## Commits

| hash | subject |
|---|---|
| `4af9e9e2` | The client's declared kept count: the engine-side document skip rule's product half |
| `10217866` | The pplx-late plugin's pooler applies the declared keep-rule engine-side |
| `36d1b5cb` | The pplx-embed-v2-late recipe declares the engine-side rule; the loader cross-checks it |
| `6fa8def3` | Docs, CHANGELOG and the regenerated public surface for the engine-side keep-rule |
| `a48ea45c` | The keep pooler joins the registry-lazy model modules (the no-torch walk skips it) |
| `eb167382` | Merge `rfc-0001` (`01f4b9be`: l10c and qa-prep) |
| `a7914342` | The engine-side keep-rule spares a query: a document role gate, and one head construction |
| `1eb87864` | The recipe declares the rule's document gate; the loader requires it |
| `7e7bf63b` | Per-item kept counts, the per-chunk refusal and the zero-kept item |
| `d5cc839f` | Docs, CHANGELOG and the regenerated schemas for the document gate and the per-item check |
| `c01cb24c` | The family sends the media render's trained head; the notes and docs say so |
| `db356346` | The media system head crosses as a structured text part |
| `1fc23b4f`, `9c991a51` | handover: the late-keep report (and both gated heads named) |
| `1025d0dd` | Merge `rfc-0001` (`45b66e1b`: fp-v4 and the judge recipes) |
| `b9a02f33` | The keep machinery serves both late-interaction families: a media allowlist and the topk pooler |
| `043c1bc3` | The client's media allowlist: the media count is the patch run, and no `skip_unapplied` |
| `742f5b38` | topk-embed-v1 declares its image keep-rule; the loader cross-checks it |
| `f94465a3` | Docs, CHANGELOG, schemas and MASTER section 9 for the media allowlist |
| `01087f84` | The allowlist's count is per media item; a colliding text render is refused before sending |
| `183c2ddf` | The topk notes say what the code does: the shipped recipe declares the image allowlist only |

## What changed (per brief item, and the operator's follow-up)

1. **The rule, applied engine-side in the plugin (the brief's core).** vLLM v0.31.0's pooling route cannot
   return the engine's per-position token ids (`PoolingParams.returned_token_ids` is a step pooler's output
   slice, `requires_token_ids` is the worker's internal CPU copy, the HTTP reply carries vectors only), so
   the plugin builds its own pooler: `KeepPool` (an `AllPool` subclass) reads the prompt token ids
   (`PoolingParamsUpdate(requires_token_ids=True)`) and indexes each sequence's hidden states by
   `kept_positions` -- the one pure function, tested with fake token-id rows. The rules are read from the
   served model's HF config (`serve.hf_overrides` merged into it); the head is the plugin's one construction
   (`models/token_pooler.token_embed_pooler`). With no rule declared the inherited stock pooler stands
   unchanged.
2. **The image's special/structural positions.** The plugin applies a rule to the token ids it sees, a media
   document's chat-template render included: it drops exactly the positions the declared rule names. The
   pplx-late rule is the checkpoint's own `MultiVectorMask` (the 32 ASCII punctuation ids), which names no
   structural id -- so the render's head and vision markers are kept, exactly as the reference keeps them.
   The topk rule is the reference's image keep-mask (`keep = ids == image_token_id`): the allowlist keeps
   only the image-patch positions, so the chat template's wrapper and a caption drop -- exactly the
   reference's kept set. The brief's parenthetical is read as the mechanism (the plugin sees the render's own
   ids and applies the rule to them by id), and the owner confirmed the checkpoint's own mask as the rule
   (no deviation).
3. **The client's declared count, checked against the reply.** `PoolingEndpoint.document_skip_engine_side`
   declares that the served plugin applies the text rule; `PoolingEndpoint.media_keep_token_ids` declares the
   media allowlist. The client then computes the declared kept count per item (`kept_vector_count`: the sent
   render's ids outside the text rule; a media document's sent head plus its caption plus the prepared media
   block under the text rule; the media's patch run under the allowlist), carries it on
   `PoolRequest.kept_counts`, and the pooling adapter checks each decoded reply item against its own count --
   a reply that ignored a rule is a typed `ProviderError`, never silently sliced, and the engine's
   `usage.prompt_tokens` (which counts the prompt) is no longer the check under a rule. A media document
   under a rule writes no `skip_unapplied` record; a recipe without one keeps the client-side rule and its
   record unchanged.
4. **Declared once, read by both, covered by the fingerprint.** Each rule's semantic home is the client
   block (`document_skip_token_ids`, `media_keep_token_ids`); the engine half is `serve.hf_overrides`
   (`document_skip_token_ids` + its gate `document_skip_prefix_token_id`; `document_keep_token_ids`), and the
   recipe loader (`_document_skip_agrees`, `_media_keep_agrees`) refuses a mismatch, either half alone, an
   empty or malformed engine half, or a text rule without the gate. The engine halves are CONTENT fields, so
   they are fingerprint inputs (`serve.hf_overrides`) and serve-time overrides of them are refused; the
   plugin modules that implement the rules are keyed for both architectures (`ARCHITECTURE_MODULES`), so a
   change to the rule moves both families' corpus keys (`rcp-fp/4`).
5. **Document-side and media-side gates, not query-side.** The checkpoint's mask declares
   `skiplist_tasks: ["document"]`, so the text rule spares a row that does not open with the declared
   document role prefix: a query prompt keeps every position, as the reference's `encode_query` does. The
   media allowlist is its own gate: a row carrying one of its ids is a media document (topk's image render
   carries the image-patch token). A TEXT render that carries an allowlist id (the literal special token)
   would be misread as media, so the client refuses that collision by name before sending.
6. **Minimal and isolated.** One function per side: `kept_positions` (engine) and `kept_vector_count` (the
   client's count) over the existing `skip_keep_mask`; the engine image has no `rcp_ndcg` (the plugin wheel
   is installed `--no-deps`), which is why the mask rule has one home per process.
7. **The reference applies the same rule.** The pplx-late checkpoint's `MultiVectorMask` drops exactly the
   ids the recipe declares (pinned by the recipe test against the checkpoint tokenizer), and its media render
   now opens with the trained `[D] ` head (`media_head_as_system: true`), so the reference's image render and
   the engine's agree. The topk reference's image keep-mask is the declared allowlist
   (`topk_embed_st.py:_image_row`), pinned to the checkpoint's `image_token_id`, so the served image
   documents are like-for-like with the reference instead of the client's kept-whole superset.
8. **CPU tests with fake token-id rows.** `rcp-ndcg-vllm/tests/models/test_keep.py` pins the pure rules
   (empty rows, repeated ids, a row fully inside a rule, the query gate, the media allowlist, the malformed
   declarations) and exercises the pooler's application against a stubbed vLLM surface with real torch
   tensors, so a pooler that skips the application goes red.
9. **Docs, snapshots, CHANGELOG.** See below.

## Verification

Four rounds of independent verifiers (fresh context, DeepSeek-V4.1-flash at xhigh): round 1 (two lenses)
FAIL with a blocker and majors; round 2 (one confirmation) PASS with three minors; round 3 (one
confirmation) FAIL with one blocker; round 4 (two lenses, on the topk follow-up) PASS with minors. All
findings were fixed.

| round | verdict | findings and disposition |
|---|---|---|
| 1 | FAIL | **Blocker (A)**: the engine-side rule was applied to QUERY prompts too (the pooler had no role signal), so a query containing a declared punctuation id was refused and its vectors would have dropped punctuation. Fixed: the document role gate (`kept_positions(..., document_prefix_id=...)`, `build_keep_pooler` and the loader refuse a rule without it; the recipe declares `document_skip_prefix_token_id: 248078`). **Major (B)**: a multi-item text batch checked only the SUM of the declared kept counts. Fixed: per-item counts + a compensating-pair test. **Major (A)**: the media render carried no trained head while the notes implied it; fixed by declaring `media_head_as_system: true` and qualifying the wording. Minors fixed: `document_skip_engine_side` + `outputs: per_chunk` silently skipped the check; the loader accepted an empty engine rule + the flag; zero-kept items decoded only in base64; the plugin's application had no red test; stale code docs; the test helper's docstring; two head builders. |
| 2 | PASS | Residual minors: the stale "the client drops the vectors at the ids it sent" sentence in the recipe notes -- fixed; the document gate did not recognize a media render (no head sent) so a captioned media document was refused -- fixed by declaring `media_head_as_system: true`; the docs/CHANGELOG implied the served media render carries a trained head -- qualified, now true of the shipped recipe. |
| 3 | FAIL | **Blocker**: the client sent the media system head as a bare string; the checkpoint's chat template iterates `message['content']` and reads `content['text']`, and the project's own model of the engine keeps a string content as it is -- so the head rendered empty there, the media render did not open with `248078`, and the media documents were refused. Fixed: the head crosses as a structured text part (the form vLLM's own chat parsing produces from a string), with a recipe test pinning the structured form and showing the bare string rendering empty; the verifier's own reproductions now report `ids[0] = 248078` and declared == kept. |
| 4 (topk follow-up) | PASS | Both lenses confirmed the operator's ask (the reference rule, the declared count over a 17-size grid, the gates, the loader, the no-`skip_unapplied` path, the fingerprint coverage of both families). Minors fixed: the allowlist's count assumed one media item (now per image/sampled frame, a video container refused) and the mixed-batch refusal missed the allowlist-alone case; a text render carrying an allowlist id is now refused by name before sending; the topk model docstring and the recipe comments/notes named the pplx-late pooler and claimed the engine applies the 41-id text rule (it is the client's); the report itself (this file) was stale. Observations recorded: a captioned image is accepted and its caption drops (the allowlist's declared behaviour; the reference's media mode refuses such rows, so only the harness media stage gates it), and `_no_inert_overflow_policies` need not list the allowlist (a self-hosted `PoolingEndpoint` must declare a tokenizer, so the tokenizer-less case is unreachable). |

Every fix landed test-first where a test could show it red (the red runs are quoted in the fix commits and
the verifiers' own reproductions).

## Checks

Last commands and results (on the final head `183c2ddf`, gate `bin/gate lane/late-keep`):

- `bin/gate lane/late-keep` -> **GATE: PASS**: ruff-check/format clean; basedpyright 0 errors;
  `pytest tests/` 3668 passed, 102 skipped; contract+docs 301 passed, 55 skipped; mkdocs `--strict` ok;
  `rcp-ndcg-test` 944 passed, 226 skipped; the recipes step "no failure outside the baseline (0 baseline
  failures remain, 0 fixed)"; vllm-pkg 49 passed and vllm-models 92 passed; run_all 1022 checks / 987 match /
  35 known deviations / 0 failed; human study 67/67; external judges 82/82; public-names clean (0 baselined
  hits); clean tree. Earlier gates: PASS on `a48ea45c`, `c01cb24c`, `db356346`, `1fc23b4f` and `f94465a3`.
- The lane's own runs: `heavy uv run --no-sync pytest tests/ -q -n 4 -p no:cacheprovider` 3668 passed;
  `uv run --no-sync pytest rcp-ndcg-vllm/tests` 132 passed, 11 skipped; `env
  RCP_NDCG_VLLM_TOKENIZER_CACHE=... uv run --no-sync pytest rcp-ndcg-test/tests -q -n 4` 944 passed;
  the network-gated recipe files (`test_pplx_embed_v2_late.py` 42, `test_topk_embed_v1.py` 47) passed;
  `test_family_goldens.py` 49 passed; `ruff format --check`/`ruff check` clean; `basedpyright` 0 errors;
  `mkdocs build --strict` ok.
- The verifiers' reproductions: the real checkpoint tokenizer keeps every query position and applies the
  rule to every `[D] ` render; the topk engine render equals the reference's `_image_row` ids and keep sets
  (including a captioned image and a non-media row carrying the literal token); the client's declared count
  equals the reference's `card_resize` patch count over a 17-size grid; the compensating pair, the
  `per_chunk` combination, the empty engine rule and the zero-kept decodings are refused/accepted as
  declared; the loader refuses every mismatched/half/malformed declaration; mutating the rule or the pooler
  moves both families' fingerprints.
- Failing-test-first evidence: the loader cross-check (`4 failed` before `_document_skip_agrees` was
  called), the query gate, the per-item compensating pair, the `per_chunk` refusal, the zero-kept decodings,
  the pooler's application (mutation), and the topk allowlist (mutations a-d in round 4: the rule, the
  client count, the loader and the recipe's engine half each go red).

## Docs updated

- `docs/concepts/late-interaction.md` -- the `document_skip_engine_side` bullet (the engine-side rule, its
  document role gate, the declared kept count and the per-item check, the media head), the
  `media_keep_token_ids` bullet (the media allowlist, its own gate, the patch-run count, the collision
  refusal) and the usage paragraph (`kept_counts`).
- `docs/concepts/text-budgets.md` -- the `skip_unapplied` mechanism is not written when the engine applied a
  rule.
- `rcp-ndcg-vllm/README.md` -- the model-plugin section states both engine-side rules, their gates and the
  client's count check.
- `handover/00-MASTER.md` section 9 -- the topk image-document gap is resolved (with what shipped).
- `CHANGELOG.md` -- the public-surface entry (below).
- The recipes themselves (`rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/pplx-embed-v2-late/family.yaml`,
  `topk-embed-v1/family.yaml`) are the rules' declarations and carry the notes.
- Greps run: `git grep -n "document_skip_token_ids\|media_keep_token_ids\|skip_unapplied\|skip_keep_mask\|kept_vector_count" -- docs README.md REPRODUCIBILITY.md skills examples experiments mkdocs.yml`;
  `git grep -n -i "keep-rule\|keep rule\|image position\|engine-side rule" -- docs`;
  `git grep -n "pplx-embed-v2-late\|topk-embed-v1" -- docs`; `git grep -n "system_head\|\"role\": \"system\"" -- rcp-ndcg/src tests rcp-ndcg-test/tests`;
  `git grep -n "late_keep\|late_pooler"` (only the historical report references remain).

## Open questions

1. **The topk image rule's detector is token presence.** A TEXT render carrying the literal
   `<|image_pad|>` string is misread as a media document engine-side; the client refuses that collision by
   name when it tracks the ids (a document under the text rule or the allowlist, or either side under
   `request_shape: token_ids`). A QUERY prompt under the default text wire is not guarded client-side (its
   ids are not tracked), but the engine's reply then carries one vector and the adapter's usage check
   refuses it -- loud, never silent. If the owner wants the guard before the request on the query side too,
   the client would tokenise every query (a cost the fit already pays).
2. **A captioned image is accepted** (the card says "Mixed text+image inputs are not supported"): the
   allowlist keeps only the patch positions, so the caption drops with no `skip_unapplied` record -- the
   allowlist's declared behaviour, and the harness's media stage is what catches the reference's refusal.
   Refusing a text part beside media for a recipe whose reference refuses it would need a new declaration.
3. **The media head's declaration was taken here** (the media-rules report had assigned
   `media_head_as_system: true` to recipe-fix): the engine-side rule's document gate needs the media render
   to open with the trained head, and it makes the media render match the reference. The recipe-fix lane
   should keep/verify it rather than re-declare it.
4. **The media path's score comparison stays the harness's named gap.** The media stage gates geometry and
   tokens only; both families' served media renders now equal their references' kept sets, but no CPU test
   scores an image pair (review M5/A2, harness-fix). The wave's probe-image check and the stage-2 media
   comparison are the first real evidence.
5. **The harness's engine model normalizes a string content.** vLLM v0.31.0's
   `chat_utils._parse_chat_message_content` wraps a string content into a text part before the template,
   while `rcp_ndcg_test.equivalence.stages.engine_conversation` keeps a string as it is (its docstring cites
   the part-level parser only). The structured head this lane sends is correct under both readings; the
   harness's model could still diverge for another string-content request (a harness-fix item, not this
   lane's file).

## CHANGELOG entry

Added under `## Unreleased`, `### Public surface` (the exact text is in `CHANGELOG.md`):

```markdown
- **The late-interaction image keep-rule, applied engine-side** (operator decision):
  `PoolingEndpoint.document_skip_engine_side` declares that the served plugin applies
  `document_skip_token_ids` engine-side. vLLM v0.31.0's pooling route cannot return the engine's
  per-position token ids, so the pplx-late plugin's pooler drops the rule's positions from the token ids it
  sees -- a text document's punctuation positions and a media document's chat-template render alike (the
  plugin sees the render's own ids; a rule that names a structural id drops exactly that position). The rule
  is document-side (the checkpoint's mask declares `skiplist_tasks: ["document"]`), so the engine half also
  declares the document role gate `serve.hf_overrides.document_skip_prefix_token_id` (the leading token id a
  document prompt opens with): a row that does not open with it is a query prompt and keeps every position,
  exactly as the reference's `encode_query` does. The recipe declares the rule once and renders it for the
  engine in `serve.hf_overrides.document_skip_token_ids` (a CONTENT field: it changes the engine's output, so
  it is a fingerprint input and a serve-time override of it is refused), and the recipe loader cross-checks
  the two halves and refuses either declared alone or a rule without the gate. The client then does not
  slice: it counts the declared kept vectors (`rcp_ndcg.data.postprocess.kept_vector_count`: the sent
  render's ids outside the rule, or a media document's sent head plus its prepared media block) and refuses a
  reply whose per-item count disagrees -- a reply that ignored the rule carries the prompt's count, which
  `usage.prompt_tokens` cannot distinguish, so the check is the declared count (`PoolRequest.kept_counts` on
  the request object) instead of the usage line. A media document under the rule writes no `skip_unapplied`
  record (the engine applied it); a recipe without the flag keeps the client-side rule and its record
  unchanged. The pplx-embed-v2-late family declares the rule (`document_skip_engine_side: true`; the 32
  punctuation ids of the checkpoint's `MultiVectorMask`, which keeps a media render's head and vision markers
  -- the rule names punctuation only) and sends the media render's trained `[D] ` head as a system message
  (`media_head_as_system: true`, the card's own sentence-transformers prompt): the pass-through engine chat
  template injects no frame of its own, so without it an image render would lose the trained prefix -- and
  would not open with the document role prefix the engine-side rule gates on. `PoolingEndpoint` also gains
  **`media_keep_token_ids`**, the media allowlist for a checkpoint whose image documents keep only a subset
  of the render's positions: the plugin applies it engine-side through the same path (the recipe renders it
  in `serve.hf_overrides.document_keep_token_ids`, which the loader cross-checks) and the allowlist is its
  own gate (a row carrying one of its ids is a media document, and only those positions are kept); the
  client counts the media block's patch run and refuses a reply that disagrees, so no `skip_unapplied`
  record is written for a media item. The topk-embed-v1 family declares it (`media_keep_token_ids: [248056]`
  beside the engine half): its reference keeps only the image-patch positions for an image document
  (`topk_embed_st.py:_image_row`: `keep = ids == image_token_id`), so the served image documents are now
  like-for-like with the reference instead of the client's kept-whole superset -- the family's named
  no-verify gap for image documents is gone (MASTER section 9).
```

## Public surface changes

- `PoolingEndpoint.document_skip_engine_side` and `PoolingEndpoint.media_keep_token_ids` (new fields;
  CONTENT) with their refusals (without the rule; beside `outputs: per_chunk`).
- `PoolRequest.kept_counts` (new field; the per-item declared kept counts).
- `rcp_ndcg.data.postprocess.kept_vector_count` (new function; the declared count of kept vectors; the module
  is internal, so the contract snapshot does not list it).
- The pooling adapter sends the media system head as a structured text part (the wire body's shape).
- Schemas regenerated: `schemas/index.v1.json`, `schemas/run-config.v1.json`; the contract snapshot
  `tests/contract/snapshots/python_api.json`. No CLI command, flag or exit code changed.

## Files outside scope

- `rcp-ndcg-vllm/src/rcp_ndcg_vllm/models/` -- the shared keep machinery (`keep_rule.py`, `keep_pooler.py`,
  `token_pooler.py`), the two model classes' pooler wiring, and the `ARCHITECTURE_MODULES`/lazy-list
  declarations the fingerprint keys.
- `rcp-ndcg/src/rcp_ndcg/inference/adapters/pooling.py` -- the media system head's structured content (the
  round-3 blocker's fix; the field is the media-rules lane's, the shape is the wire's).
- `rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/pplx-embed-v2-late/family.yaml` -- `media_head_as_system: true`
  (the recipe-fix lane's item; see Open questions 3); `topk-embed-v1/family.yaml` -- the allowlist
  declaration.
- `rcp-ndcg-test/src/rcp_ndcg_test/fingerprint.py`, the per-variant goldens and `handover/00-MASTER.md`
  (the two client fields' classification, the re-captured goldens under `rcp-fp/4`, the section-9 update).

## For the next lanes

- **recipe-fix**: keep `media_head_as_system: true` (now declared) and the topk allowlist (now declared) and
  verify both on the wave.
- **harness-fix**: the media score comparison (Open questions 4) and the harness's engine model for a string
  content (Open questions 5).
- **06/07**: the late-interaction page and the text-budgets page carry both engine-side rules; the wave's
  probe-image check and the stage-2 media comparison are the first real evidence for the media renders.
