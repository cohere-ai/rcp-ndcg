# Lane fix11ctx — the token-ids recipe's bare probes carry the recipe's own render ids

**Lane:** `lane/fix11ctx` (worktree outside the operator's checkout). **Base:** the `rfc-0001` tip
`9224b700` (merged first; a fast-forward, so no merge commit exists in this lane). **Gate:**
`timeout 7200 bin/gate lane/fix11ctx` on `3bd22c08` (slot 2) — **GATE: PASS**, every step exit 0. No GPU
work in this lane; nothing pushed; `rfc-0001` was not merged into.

## 1. Status

Done, one item (the brief's), plus the merged tip's pre-existing red tests (three, not this lane's — see
§6) and one adjacent honesty fix found while testing it.

**The fix.** For a `client.request_shape: token_ids` recipe, every request the *harness* builds itself now
carries the recipe's own render ids — the construction the recipe's client sends — instead of a bare text:

- the corpus step's **standing probes** (`record.py::_bare_probes` through `_role_body`: the unknown field,
  the wrong model, the empty input, the over-length body);
- the corpus plan's **bare rows** (`observe/requests.py::_bare_body`: the uncut ladder, the long content
  kinds, the wire variants — the document side, or the query side when no document rides the row);
- the **smoke** request (`jobs/run_wave.py::_smoke`), which had its own copy of the derivation since
  `0388da2f`;
- the **recorder's error probes** (`record.py::_record_errors` / `_role_request`: the over-length body and
  the unknown field), which were the same latent shape and would have killed the EngineCore in any wave
  that passes `--record` (the submitted wave does; the recording pass is `--record --record-corpus`).

**The mechanism.** The checkpoint's plugin refuses a `/pooling` input whose first id is neither role prefix
by name, and vLLM answers an exception in the pooler by killing the EngineCore — the round-2 mechanism,
fixed for the smoke only. A text body tokenizes server-side, so its first id is the text's first token, not
a prefix: the request is one the recipe's client never sends.

**One home.** `rcp_ndcg_test.equivalence.fitting.sent_ids(recipe, shape, text)` — the harness's one
construction of the client's own ids: the recipe's template renders the text
(`TemplateSpec.render`, the string the client's fit ships), and the recipe's tokenizer tokenises that render
under the shape's declared parse, exactly the rule `PoolingClient._sent_ids` sends with (R30):
`TemplateSpec.split_ids` for a query under `query_split_special_tokens` (the reference's
prefix-plus-body construction), `TextTokenizer.ids(..., split_special_tokens=True)` for a document under
`document_split_special_tokens`, the file's own parse otherwise — the shape's declared `add_special_tokens`
flag throughout. The product keeps the one home of the parse (`TemplateSpec.split_ids`,
`TextTokenizer.ids`); the harness keeps the one home of the *construction* and every caller reads it.

**Adjacent fix.** A token-ids wire's model layer has no text on the wire: `_model_layer` no longer
stringifies the id lists into the `/tokenize` probe list, so the corpus records no `/tokenize` prompt for
them (the probe's contract is the engine's tokenization of exactly what the client sent, R29 — and the
record carries the ids as sent in its body, which `_attach_sent_token_ids` already reads).

**The stub.** The harness's CPU stub engine now serves vLLM's token-id prompts on `/pooling` (one vector
per id, the over-length rule counting the ids), so a token-ids recipe's ids are accepted end to end on CPU
— the product's offline fake already read them.

## 2. Commits

| commit | what |
|---|---|
| `1c4a6b15` | the render-ids construction and every harness caller (fitting.py, record.py, observe/requests.py, run_wave.py, the stub, four test modules, the corpus schema doc) |
| `15bee8ff` | the CHANGELOG entry |
| `3bd22c08` | the merged tip's three red case-fill tests and the one stale note (not this lane's; §6) |
| (report) | this report, `handover/reports/fix11ctx.md`, and the lane copy |

No co-author lines, no attribution, nothing pushed. The gated revision is `3bd22c08`; the report commit
follows it and changes no code.

## 3. Red-first evidence

Written as tests first and shown red on the unfixed tree (all of my lane's changes stashed, tests kept;
`uv run --no-sync pytest rcp-ndcg-test/tests/test_observe_requests.py rcp-ndcg-test/tests/test_observe_corpus.py
rcp-ndcg-test/tests/test_record_and_wave.py -q -k "token_ids or text_shaped or own_render_ids or
recorder_error"` — **3 failed, 2 passed**):

```
FAILED rcp-ndcg-test/tests/test_observe_requests.py::test_the_corpus_plan_sends_a_token_ids_recipe_the_document_render_ids
E       AssertionError: assert ['<|cls|>\n<|end|>\n<|sep|>'] == [[278, 28, 30, ...]]
E         At index 0 diff: '<|cls|>\n<|end|>\n<|sep|>' != [278, 28, 30, 94, 69, 78, 85, 94, 32, ...]
FAILED rcp-ndcg-test/tests/test_observe_corpus.py::test_the_standing_probes_carry_the_recipes_own_render_ids
E           AssertionError: unknown_field
E           assert ['a query'] == [[278, 28, 30, 94, 69, 78, 85, 94, 32, ...]]
FAILED rcp-ndcg-test/tests/test_record_and_wave.py::test_the_recorder_error_probes_carry_a_token_ids_recipes_ids
E           AssertionError: {'encoding_format': 'float', 'input': ['a a a a ... '], 'model': 'fixture-multi-vector-token-ids', 'task': 'token_embed'}
E           assert (True and False)
```

The adjacent fix's test was red on the unfixed `_model_layer` (only `record.py` stashed):

```
FAILED rcp-ndcg-test/tests/test_observe_corpus.py::test_a_token_ids_wires_model_layer_has_no_text_to_tokenize
E       AssertionError: a token-ids wire carries no text; the ids ride the record's body
E       assert [({'request_i... 80, 86, 2]')] == []
E         Left contains 2 more items, first extra item: ({...}, 'query:0', 'query', '[51, 28, 67, 282, 2]')
```

## 4. What the tests pin

- `test_the_corpus_plan_sends_a_token_ids_recipe_the_document_render_ids`: the plan's bare rows carry the
  document-side render ids under the declared split parse, computed in the test from the product's template
  and tokenizer (the spellings row's split parse differs from the file's own parse, so the declaration is
  what ships), plus the uncut ladder rung and a wire variant.
- `test_a_text_shaped_recipe_carries_its_bare_rows_as_text`: a recipe without `request_shape: token_ids`
  (an embedder and a multi-vector fixture) keeps its bare rows' texts.
- `test_the_standing_probes_carry_the_recipes_own_render_ids`: the probes' ids equal the ids the product's
  client put on the wire for the same document (captured through the client's own transport), the empty
  probe carries the empty render's ids, the plan's wire row carries the same ids, and a stub engine that
  takes vLLM's token-id prompts answers the probes 200 (the `unknown_field` body is refused 400 by the
  stub's own strict field model, as it is for text recipes).
- `test_the_recorder_error_probes_carry_a_token_ids_recipes_ids`: the recorder's three `/pooling` bodies
  (the client's own request and the two error probes) all carry id lists; one 200 and two 400s.
- `test_a_token_ids_wires_model_layer_has_no_text_to_tokenize`: the model layer's sent-text list is empty
  and the records carry id lists.

## 5. Verification

- Red-first runs above; then the four test modules green (210 passed with the case-fill and equivalence
  modules).
- `uv run --no-sync ruff format --check . && uv run --no-sync ruff check .` — clean; `basedpyright` — 0
  errors.
- The whole harness suite through the slot wrapper: `heavy uv run --no-sync pytest rcp-ndcg-test/tests -q
  -n 4` — 1189 passed, 229 skipped **before** the tip-drift fix (3 failed, §6), 1192 passed, 229 skipped
  after.
- The gate, `timeout 7200 bin/gate lane/fix11ctx` on `3bd22c08` (slot 2), every step exit 0:
  ruff-check 0, ruff-format 0, basedpyright 0, pytest (the product tree) 4145 passed / 155 skipped,
  contract-docs 413 passed / 107 skipped, mkdocs built, test-pkg 1192 passed / 229 skipped, recipes clean,
  vllm-pkg 50 passed, vllm-models 110 passed / 8 skipped, run_all 0 failed (35 known deviations),
  public-names clean, tree clean — **GATE: PASS**.

## 6. The merged tip's red tests (Files outside scope)

The `rfc-0001` tip `9224b700` ("The cuda reference-case values are filled...") left three tests in
`rcp-ndcg-test/tests/test_case_fill.py` red on the merged tree. They fail identically with this lane's
changes stashed, so they are the tip's, not this lane's, and the gate's `test-pkg` step runs exactly them.
The supervisor channel did not answer two `need_decision` calls; the parent's later steering confirmed the
chosen path (fix them in this lane and report them here).

- `test_a_media_case_is_no_longer_skipped_and_a_kind_none_case_still_is` and
  `test_an_embed_media_case_fills_and_carries_its_own_tolerance` copied the shipped cases and expected them
  pending (`pending_before` 10 and 9); the shipped files are the cuda fill's outputs now, so only the video
  case (the recorded exception) was still pending (`1`). The copies are now rewritten to the pending state
  the cases lanes' generator writes — values null, `status: pending_gpu`, the fixture's pending note — while
  every other field (the media refs, the declared tolerances, the inputs) stays, so the tests exercise the
  fill over the real cases' shape exactly as before.
- `test_the_retirement_vocabulary_covers_every_shipped_generated_note` refused the video case's note, which
  still carried "The case stays pending_gpu until ..." from the fill's own exception sentence
  (`case_fill.py::_EXCEPTION_SENTENCE`). The sentence now says "stays pending" (the case format's status
  token is not prose) and the one shipped note written with it is reworded; `_update_note` then appends the
  fill's provenance sentence as the test expects.

All 13 case-fill tests pass. The case-fill workstream should review these two files: they substitute for its
own update of the tests it filled the cases under.

## 7. Docs updated

- `rcp-ndcg-test/schema/observation-corpus.md` — the corpus request plan's paragraph: a `request_shape:
  token_ids` recipe's bare rows and standing probes carry the recipe's own render ids, and a token-ids
  wire's model layer probes no `/tokenize`.
- Docstrings that state the changed behaviour: `equivalence/fitting.py` (module docstring and `sent_ids`),
  `record.py` (`_model_layer`, `_bare_probes`, `_role_body`, `_record_errors`, `_role_request`),
  `observe/requests.py` (`CorpusPlan.bare`, `_bare_body`), `jobs/run_wave.py` (`_smoke`),
  `case_fill.py` (`_EXCEPTION_SENTENCE`).
- Greps run: `git grep -n "_smoke_query_ids\|sent_ids\|bare probe\|token-ids" -- docs README.md
  REPRODUCIBILITY.md skills examples experiments mkdocs.yml rcp-ndcg-test/README.md` — the remaining hits
  describe the adapters and the offline fake generically and stay true; `git grep -n -i "uncut\|bare probe\|
  bare row\|standing probe" -- docs rcp-ndcg-test/schema` — only the updated schema paragraph and the
  timeout sentence in `docs/how-to/validate-a-recipe.md` (unchanged behaviour).

## 8. Open questions

- None blocking. The tip drift (§6) is a substitution for another workstream's update; its review is the
  one open item.

## CHANGELOG entry

Verbatim, under the 0.0.1 `### Fixed` section (commit `15bee8ff`):

- **The harness's own requests carry a `request_shape: token_ids` recipe its own render ids, not bare
  texts** (the E2 round-10 r2 recording failure: pplx-embed-v2-context's corpus step still killed the
  EngineCore): the corpus's standing probes and its plan's bare rows (the uncut ladder, the long content
  kinds, the wire variants) built their role bodies from raw texts, which such a checkpoint's plugin
  refuses by name -- the refusal kills the EngineCore -- while its client sends pre-tokenised ids.  One
  helper (`rcp_ndcg_test.equivalence.fitting.sent_ids`: the product's template and tokenizer with the
  declared split parses, the construction the pooling client's `_sent_ids` sends) now builds the ids for
  the standing probes, the plan's bare rows and the smoke request (which had its own copy), the document
  side under the declared `document_split_special_tokens` parse; the recorder's own error probes (the
  over-length body and the unknown field) carry the same ids, so a `--record --record-corpus` wave no
  longer kills the engine before the corpus step; and a token-ids model layer probes no `/tokenize` (its
  input is an id list: there is no prompt text to tokenize, and the record carries the ids as sent).  A
  CPU test pins the plan's bare rows against the product's construction, one pins the probes against the
  client's own captured request for the same text, and the stub engine now serves vLLM's token-id prompts
  so the ids are accepted end to end.

## Public surface changes

None in the product: no `__all__`, CLI command/flag, exit code, schema or contract snapshot changed
(`tests/contract` and `tests/docs` pass unchanged). The unpublished `rcp-ndcg-test` package gains one
public name, `rcp_ndcg_test.equivalence.fitting.sent_ids` (in that module's `__all__`, documented), and the
stub engine's `/pooling` now serves vLLM's token-id prompts (a test-side stub, never shipped).

## Files outside scope

- `rcp-ndcg-test/tests/test_case_fill.py` — the two fill tests' copied cases are rewritten to pending
  (§6).
- `rcp-ndcg-test/src/rcp_ndcg_test/case_fill.py` — `_EXCEPTION_SENTENCE` drops the `pending_gpu` token.
- `rcp-ndcg-test/cases/qwen3-vl-embedding-2b/video-short-single.yaml` — the one shipped note written with
  the old sentence is reworded.

## For the next lanes

- **The recording pass for pplx-embed-v2-context can run**: its smoke, standing probes, plan bare rows and
  the recorder's error probes all carry the recipe's own document-side render ids now, and its token-ids
  model layer probes no `/tokenize`.  The submitted wave (`--record`) and the T1 observations pass
  (`--record-corpus`) both stop killing the EngineCore in the harness's own requests.
- **A token-ids recipe's corpus records no `/tokenize` probe**: its `inputs.sent_token_ids` comes from the
  record's own body (the ids as sent), which the kept-count resolver already reads.
- **The case-fill workstream** should review §6's two files; they substitute for its own test update.
