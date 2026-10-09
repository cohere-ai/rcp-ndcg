# Report `late-keep`: the exact late-interaction image keep-rule, applied in the engine plugin

**Status:** DONE. The rule is declared once in the recipe (the client's `document_skip_token_ids`), rendered for
the engine in `serve.hf_overrides.document_skip_token_ids` with its document role gate
(`document_skip_prefix_token_id`), applied engine-side by the pplx-late plugin's own pooler, and checked
client-side against the declared kept count. The pplx-embed-v2-late family also sends the media render's
trained `[D] ` head (`media_head_as_system: true`), which the rule's document gate needs and which the
media-rules report had assigned to the recipe-fix lane. Base: `743d6a63` (rfc-0001 after media-rules and the
recipe line); the branch merged the moved `rfc-0001` (`01f4b9be`: l10c and qa-prep) at `eb167382`.

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

## What changed (per brief item)

1. **The rule, applied engine-side in the plugin (the brief's core).** vLLM v0.31.0's pooling route cannot
   return the engine's per-position token ids (`PoolingParams.returned_token_ids` is a step pooler's output
   slice, `requires_token_ids` is the worker's internal CPU copy, the HTTP reply carries vectors only), so
   the pplx-late plugin builds its own pooler: `PplxLateKeepPool` (an `AllPool` subclass) reads the prompt
   token ids (`PoolingParamsUpdate(requires_token_ids=True)`) and indexes each sequence's hidden states by
   `kept_positions` -- the one pure function, tested with fake token-id rows. The rule is read from the
   served model's HF config (`serve.hf_overrides` merged into it); the head is the plugin's one construction
   (`pooler.token_embed_pooler`, shared with the contextual pooler). With no rule declared the inherited
   stock pooler stands unchanged.
2. **The image's special/structural positions.** The plugin applies the rule to the token ids it sees, a
   media document's chat-template render included: it drops exactly the positions the declared rule names.
   The shipped rule is the checkpoint's own `MultiVectorMask` (the 32 ASCII punctuation ids), which names no
   structural id -- so the render's head and vision markers are kept, exactly as the reference's
   `MultiVectorMask` keeps them. A rule that named a structural id would drop exactly that position. The
   brief's parenthetical is read as the mechanism (the plugin sees the render's own ids and applies the rule
   to them by id), not as a claim that the checkpoint's mask excludes the structural positions; the report's
   Open questions state this reading for the owner.
3. **The client's declared count, checked against the reply.** `PoolingEndpoint.document_skip_engine_side`
   declares that the served plugin applies the rule. The client then computes the declared kept count per
   item (`kept_vector_count`: the sent render's ids outside the rule; a media document's sent head plus its
   caption plus the prepared media block), carries it on `PoolRequest.kept_counts`, and the pooling adapter
   checks each decoded reply item against its own count -- a reply that ignored the rule (the full prompt
   count) is a typed `ProviderError`, never silently sliced, and the engine's `usage.prompt_tokens` (which
   counts the prompt) is no longer the check under this rule. A media document under the rule writes no
   `skip_unapplied` record; a recipe without the flag keeps the client-side rule and its record unchanged.
4. **Declared once, read by both, covered by the fingerprint.** The rule's semantic home is the client block
   (`document_skip_token_ids`); the engine half is `serve.hf_overrides.document_skip_token_ids` with its
   document role gate `document_skip_prefix_token_id`, and the recipe loader (`_document_skip_agrees`)
   refuses a mismatch, either half alone, an empty engine list, a malformed half, or a rule without the gate.
   The engine half is a CONTENT field, so it is a fingerprint input (`serve.hf_overrides`), a serve-time
   override of it is refused, and the recipe's corpus key moves; the client flag itself is
   `post_processing` (it changes only the client's check).
5. **Document-side, not query-side.** The checkpoint's mask declares `skiplist_tasks: ["document"]`, so the
   plugin spares a row that does not open with the declared document role prefix: a query prompt keeps every
   position, as the reference's `encode_query` does. `build_late_pooler` and the loader both refuse a rule
   without the gate.
6. **Minimal and isolated.** One function per side: `kept_positions` (engine) and `kept_vector_count` (the
   client's count) over the existing `skip_keep_mask`; the engine image has no `rcp_ndcg` (the plugin wheel
   is installed `--no-deps`), which is why the mask rule has one home per process.
7. **The reference applies the same rule.** The checkpoint's `MultiVectorMask` drops exactly the ids the
   recipe declares (pinned by the recipe test against the checkpoint tokenizer), so stage 2's text
   comparison is like with like; the media render now opens with the trained `[D] ` head
   (`media_head_as_system: true`), so the reference's image render and the engine's agree too.
8. **CPU tests with fake token-id rows.** `rcp-ndcg-vllm/tests/models/pplx/test_late_keep.py` pins the pure
   rule (empty rows, repeated ids, a row fully inside the rule, the query gate, the malformed declarations)
   and exercises the pooler's application against a stubbed vLLM surface with real torch tensors (the plugin
   venv carries no vLLM), so a pooler that skips the application goes red.
9. **Docs, snapshots, CHANGELOG.** See below.

## Verification

Round 1 (two independent verifiers, fresh context, lens A correctness and lens B regressions, model
DeepSeek-V4.1-flash at xhigh) returned **FAIL** with a blocker and majors; round 2 (one fresh confirmation,
lens A+B) returned **PASS** with three minors; round 3 (one fresh confirmation on the round-2 fixes)
returned **FAIL** with one blocker (the media system head's wire shape), now fixed and reproduced green with
the verifier's own scripts.

| round | verdict | findings and disposition |
|---|---|---|
| 1 | FAIL | **Blocker (A)**: the engine-side rule was applied to QUERY prompts too (the pooler had no role signal), so a query containing a declared punctuation id was refused and its vectors would have dropped punctuation, diverging from the checkpoint's `skiplist_tasks: ["document"]` and the reference's `encode_query`. Fixed: the document role gate (`kept_positions(..., document_prefix_id=...)`, `build_late_pooler` and the loader refuse a rule without it; the recipe declares `document_skip_prefix_token_id: 248078`, pinned to the checkpoint tokenizer's `[D] ` id). **Major (B)**: a multi-item text batch checked only the SUM of the declared kept counts, so a compensating per-item mismatch was accepted. Fixed: per-item counts in `_kept_per_reply`/`_check_usage` + a compensating-pair test. **Major (A)**: the media render carried no trained head while the notes/CHANGELOG implied it; fixed by declaring `media_head_as_system: true` (the media-rules report's recipe-fix item) and qualifying the wording. Minors fixed: `document_skip_engine_side` + `outputs: per_chunk` silently skipped the check (now refused at the config); the loader accepted an empty engine rule + the flag (now read as absent); zero-kept items decoded only in base64 (now in all three encodings); the plugin's application had no red test (now a behavioural stub-vLLM test); stale code docs; the test helper's docstring; two head builders (now one). |
| 2 | PASS | No blocker. Residual minors: the stale "the client drops the vectors at the ids it sent" sentence in the recipe notes -- fixed; the document gate did not recognize a media render (no head sent) so a captioned media document was refused -- fixed by declaring `media_head_as_system: true`; the docs/CHANGELOG still implied the served media render carries a trained head -- qualified, and now true of the shipped recipe. |
| 3 | FAIL | **Blocker**: the client sent the media system head as a bare string; the checkpoint's chat template iterates `message['content']` and reads `content['text']`, and the project's own model of the engine (`stages.engine_conversation`) keeps a string content as it is -- so the head rendered empty there, the media render did not open with `248078`, and the shipped recipe's media documents were refused by the count check (reproduced end-to-end against the stub). Fixed: the head crosses as a structured text part (`[{"type": "text", "text": head}]`, the form vLLM's own chat parsing produces from a string), with a recipe test that pins the structured form against the checkpoint template and shows the bare string rendering empty; the verifier's `repro_media_head.py` and `repro_stub.py` now report `ids[0] = 248078`, declared == kept (7 = 7 image-only, 10 = 10 captioned) and both stub sends accepted. Also fixed: the render test's hand-built list could not catch the mismatch (the new pins do). |

Every fix landed test-first where a test could show it red (the round-1 findings' red runs are quoted in the
fix commits; the round-3 blocker's red is the verifier's own repro, now green).

## Checks

Last commands and results (on the code head `db356346`, gate `bin/gate lane/late-keep`; the gate was re-run
on the report commit `1fc23b4f` afterwards -- the same results, this file being the only difference):

- `bin/gate lane/late-keep` -> **GATE: PASS**: ruff-check/format clean; basedpyright 0 errors;
  `pytest tests/` 3647 passed, 102 skipped; contract+docs 304 passed, 55 skipped; mkdocs `--strict` ok;
  `rcp-ndcg-test` 642 passed, 398 skipped; the recipes step "no failure outside the baseline (0 baseline
  failures remain, 0 fixed)"; vllm-pkg 40 passed and vllm-models 87 passed; run_all 1022 checks / 987 match /
  35 known deviations / 0 failed; human study 67/67; external judges 82/82; public-names clean (0 baselined
  hits); clean tree. The pre-fix gate on `a48ea45c` and the round-2 gate on `c01cb24c` were also PASS.
- The lane's own runs: `heavy uv run --no-sync pytest tests/ -q -n 4 -p no:cacheprovider` 3647 passed;
  `uv run --no-sync pytest rcp-ndcg-vllm/tests` 118 passed, 11 skipped; `env
  RCP_NDCG_VLLM_TOKENIZER_CACHE=... uv run --no-sync pytest rcp-ndcg-test/tests -q -n 4` 642 passed;
  the network-gated `test_pplx_embed_v2_late.py` 42 passed; `test_family_goldens.py` 35 passed;
  `ruff format --check`/`ruff check` clean; `basedpyright` 0 errors; `mkdocs build --strict` ok.
- The verifiers' reproductions: the real checkpoint tokenizer keeps every query position and applies the
  rule to every `[D] ` render (round 2); the compensating pair is refused, `per_chunk` is refused, the empty
  engine rule is refused, zero-kept items decode in three encodings (round 2); the client's captured media
  request renders through the checkpoint template with `ids[0] = 248078` and the declared count equal to the
  engine-faithful kept count, and the stub accepts both media shapes (round 3, re-run by the builder after
  the fix).
- Failing-test-first evidence: the loader cross-check (`4 failed` before `_document_skip_agrees` was
  called), the query gate, the per-item compensating pair, the `per_chunk` refusal, the zero-kept decodings,
  and the pooler's application (mutation: skipping it fails the stub-vLLM test).

## Docs updated

- `docs/concepts/late-interaction.md` -- the `document_skip_engine_side` bullet (the engine-side rule, its
  document role gate, the declared kept count and the per-item check, the media head) and the usage
  paragraph (`kept_counts`).
- `docs/concepts/text-budgets.md` -- the `skip_unapplied` mechanism is not written when the engine applied
  the rule.
- `rcp-ndcg-vllm/README.md` -- the model-plugin section states the engine-side rule, its gate and the
  client's count check.
- `CHANGELOG.md` -- the public-surface entry (below).
- The recipe itself (`rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/pplx-embed-v2-late/family.yaml`) is the rule's
  declaration and carries the notes.
- Greps run: `git grep -n "document_skip_token_ids\|skip_unapplied\|skip_keep_mask\|kept_vector_count" --
  docs README.md REPRODUCIBILITY.md skills examples experiments mkdocs.yml`;
  `git grep -n -i "keep-rule\|keep rule\|image position\|engine-side rule" -- docs`;
  `git grep -n "pplx-embed-v2-late" -- docs`; `git grep -n "system_head\|\"role\": \"system\"" -- rcp-ndcg/src
  tests rcp-ndcg-test/tests`.

## Open questions

1. **The brief's parenthetical reading.** The brief says the plugin "drops the per-token vectors the recipe's
   declared keep-rule excludes (the image's special/structural positions, by the token ids the plugin sees)".
   The shipped rule (the checkpoint's own `MultiVectorMask`, 32 punctuation ids) names no structural id, so
   the render's head and vision markers are kept -- which is exactly what the reference keeps. The mechanism
   is by token id (a rule naming a structural id would drop exactly that position), but the shipped recipe's
   rule does not exclude those positions. If the owner intended the pplx-late image side to keep ONLY the
   patch positions (topk's own image mask), that is a deliberate deviation from the checkpoint's mask and
   needs the reference changed with it (the reference's `MultiVectorMask` keeps the head and markers); this
   lane did not take that step.
2. **The topk image keep-rule is untouched.** `topk-embed-v1`'s image side keeps only the image-patch
   positions in its reference (`topk_embed_st.py:_image_row`), which a skip-list declaration cannot express
   (the image token id is IN its 41-id text skip list); the family's image documents therefore stay on the
   client-side keep-whole path (`skip_unapplied`) and its named no-verify gap (MASTER section 9). The
   mechanism added here (a rule, a document gate, a declared kept count) is the shape a topk fix would use,
   but it needs its own media allowlist declaration and the topk model wired to the same pooler.
3. **`media_head_as_system: true` was declared here.** The media-rules report assigned it to the recipe-fix
   lane; this lane took it for `pplx-embed-v2-late` because the engine-side rule's document gate needs the
   media render to open with the trained head (and it makes the media render match the reference). The
   recipe-fix lane should keep/verify it rather than re-declare it.
4. **The media path's score comparison stays the harness's named gap.** The media stage gates geometry and
   tokens only; the engine-side rule and the head now make the served media render equal the reference's, but
   no CPU test scores an image pair (review M5/A2, harness-fix). The wave's probe-image check and the
   stage-2 media comparison are the first real evidence.
5. **The harness's engine model normalizes a string content.** vLLM v0.31.0's
   `chat_utils._parse_chat_message_content` wraps a string content into a text part before the template,
   while `rcp_ndcg_test.equivalence.stages.engine_conversation` keeps a string as it is (its docstring cites
   the part-level parser only). The structured head this lane sends is correct under both readings; the
   harness's model could still diverge for another string-content request (a harness-fix item, not this
   lane's file).

## CHANGELOG entry

Added under `## Unreleased`, `### Public surface`:

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
  would not open with the document role prefix the engine-side rule gates on.
```

## Public surface changes

- `PoolingEndpoint.document_skip_engine_side` (new field; CONTENT) and its refusals (without the rule, and
  beside `outputs: per_chunk`).
- `PoolRequest.kept_counts` (new field; the per-item declared kept counts).
- `rcp_ndcg.data.postprocess.kept_vector_count` (new function; the declared count of kept vectors; the module
  is internal, so the contract snapshot does not list it).
- The pooling adapter sends the media system head as a structured text part (the wire body's shape).
- Schemas regenerated: `schemas/index.v1.json`, `schemas/run-config.v1.json`; the contract snapshot
  `tests/contract/snapshots/python_api.json`. No CLI command, flag or exit code changed.

## Files outside scope

- `rcp-ndcg/src/rcp_ndcg/inference/adapters/pooling.py` -- the media system head's structured content (the
  round-3 blocker's fix; the field is the media-rules lane's, the shape is the wire's).
- `rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/pplx-embed-v2-late/family.yaml` -- `media_head_as_system: true`
  (the recipe-fix lane's item; see Open questions 3).
- `rcp-ndcg-vllm/src/rcp_ndcg_vllm/models/__init__.py` -- the registry-lazy module list gains
  `late_pooler` (the no-torch walk skips the modules that import vLLM by design).
- `rcp-ndcg-test/tests/recipes/golden/DELTAS.json`, `test_pplx_embed_v2_late.py`, `test_recipe.py` -- the
  recipe's contract pins and the accepted golden differences.

## For the next lanes

- **recipe-fix**: keep `media_head_as_system: true` (now declared) and verify it on the wave; the topk image
  rule (Open questions 2) needs its own declaration if the owner wants it in 0.0.1.
- **harness-fix**: the media score comparison (Open questions 4) and the harness's engine model for a string
  content (Open questions 5).
- **06/07**: the late-interaction page and the text-budgets page carry the engine-side rule; the wave's
  probe-image check and the stage-2 media comparison are the first real evidence for the media render.
