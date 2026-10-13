# Lane fix10corpus — the recording pass's corpus failures on rc0-202610130020

**Lane:** `lane/fix10corpus` (worktree outside the operator's checkout). **Base:** `rfc-0001` at `58847318`
(the mediafill merge). **Merged:** nothing — the tip was already the base (an ancestor of every commit here),
so no merge commit exists in this lane. **Gate:** `timeout 7200 bin/gate lane/fix10corpus` on `6df52983`
(slot 4) — **GATE: PASS**, every step exit 0. No GPU work in this lane; nothing pushed;
`rfc-0001` was not merged into.

## 1. Status

Done, all three items, plus one adjacent fix found while testing item 2:

1. **jina-reranker-v3's `completeness` missing `content:whitespace_only`.** The row is dropped by the
   *product's own declared policy*: the recipe declares `empty_doc: omit_zero_blank` (the paper's
   `if d.strip()` rule), so the client omits a whitespace-only document and scores it 0.0 without a model
   call; the plan's model-layer row for that kind (whose pairs row the stage-1 render check pruned) produced
   no exchange, while the bare uncut row did. The omission is now **recorded like a client refusal**
   (`omission_exchange`: no status, a `client_omission` body naming the row and the declared policy), so the
   plan's request id stays accounted for and the stratum is recorded — the same treatment the media
   refusals already get, and honest because the corpus then says *why* nothing crossed the wire instead of
   the plan quietly dropping a row.
2. **pplx-embed-v2-late `-0.6b`/`-9b` `scores_finite_and_vector_widths`** ("the /pooling reply reports 165
   prompt token(s) but its vectors decode to 152"). The check now expects the **kept** count: the manifest's
   recipe block records the declared `document_skip_engine_side`/`document_skip_token_ids`, each client-sent
   model-layer record carries the engine's own `/tokenize` ids of the render it sent
   (`inputs.sent_token_ids`), and the check derives each document item's kept count through the product's
   `kept_vector_count` (`observe.corpus.declared_kept_counts`) and hands it to the adapter. A reply at the
   kept count passes; one at the full count (the engine ignored the rule) and one at a count that is neither
   fail. The non-determinism measurement uses the same resolver (those recipes' vector drift was silently
   dropped before), and `equivalence_consistent` decodes both sides with the request's declared counts (the
   equivalence exchange has no `/tokenize` ids of its own — without this a consistent corpus read as a
   mismatch).
3. **pplx-embed-v2-context's "the engine exited (code 0) while its steps ran".** The corpus step's body
   stops the engine and starts a fresh one between the in-process passes and the after-restart pass; the
   step machinery abandons a body whose engine died (or whose budget ran out) and ends the recipe, whose
   GPUs the wave then releases. The restart is now **gated on the step still being the one in flight** (the
   worker's step token, cleared when `_step` returns, including on abandonment): a late restart from an
   abandoned body is refused with the reason instead of stopping or starting an engine the wave no longer
   tracks. A reproduction test kills the engine inside a stand-in corpus body, waits for the runner to
   abandon the step, calls the restart and pins the refusal; without the guard the body starts a second
   engine (a second `serve start` on the pod log).

**Per-recipe disposition.** `jina-reranker-v3` and `pplx-embed-v2-late` ×2: **harness-side fixed,
CPU-verified**; their `status` stays whatever the wave ledger says until the next recording run (this lane
runs no GPU). `pplx-embed-v2-context`: **the restart race is fixed**, but the same step will still fail for a
second, independent reason the brief did not name — see §8, the residual root cause.

## 2. Commits

| commit | what |
|---|---|
| `01600079` | the omission record + the kept-count check (record.py, observe/corpus.py, observe/provenance.py, the schema doc, two tests) |
| `79c1d50a` | the abandoned corpus body's restart guard (run_wave.py, the reproduction test) |
| `336fe291` | the round-10 CHANGELOG entries |
| `6df52983` | the equivalence consistency decodes both sides with the request's declared kept counts |
| (report) | this report, `handover/reports/fix10corpus.md`, and the lane copy |

No co-author lines, no attribution, nothing pushed. The gated revision is `6df52983`; the report commit
follows it and changes no code.

## 3. Item 1 — where the row is dropped, and why the omission record is the honest fix

The chain, verified in the code:

1. `pairs/jina-reranker-v3.jsonl` carries no `content:whitespace_only` row: the stage-1 render check pruned
   it (the recorded reason: `{"index": 14, "reference_text_head": "   \t\t\n    ", "served_text_head": "",
   "shape": "pair", "span": "query"}`), because the recipe's template strips the content span and the
   reference does not.
2. `observe/requests.corpus_plan` therefore plans the kind's model-layer row (`content:whitespace_only`) and
   its bare uncut row (`content:whitespace_only:uncut`) — the design's "a kind too long for a pairs row"
   path.
3. The product's rerank client omits the row: `RerankEndpoint.empty_doc: omit_zero_blank` +
   `clients/_base._apply_empty_documents` (`content.text.strip() == prefix.strip()`) → `rerank()` returns
   `RerankResult(scores=(0.0,))` with **no request**. The engine is never asked.
4. `record_corpus`'s model layer captured no exchange for the row, so the manifest's planned
   `content:whitespace_only` id was missing and `completeness` failed.

The fix records the omission (`record.py::omission_exchange`, one per planned row per pass) exactly as the
media path already records a client refusal — "so each refusal is its own record and the plan's request id
is accounted for". The alternative (declare the stratum absent in the plan) is honest too, but weaker: it
would need the plan to *predict* the product's empty rule (a second home of `omit_zero_blank`) and would
leave no corpus record of what the client actually did. The engine's own reply to the uncut bare row is
still recorded, so the stratum's engine-side evidence is unchanged.

The recording is in the rerank path, where a row's every call can send nothing: its two orders (given and
reversed) are the row's only sends, and no batch context exists to account for it. The embed/pool path
sends each item in batches of 1, 2, 8 and 32, so a row omitted from one batch is still named in the batch
contexts of the others; its only hole is a fully-omitted single-item group, which no shipped recipe reaches
today (the plan skips the empty string under an omitting policy, and the other kinds are not stripped
before the empty stage). If a recipe ever reaches it, the same `omission_exchange` is the fix.

## 4. Item 2 — the kept count, one home with the product's rule

- `observe/provenance.recipe_facts` records the declared cut (`document_skip_engine_side`,
  `document_skip_token_ids`), beside the `declared_dim` the width check already reads.
- `record.py::_tokenize_probes` now returns `{sent text: engine ids}` and `_attach_sent_token_ids` lands
  the ids on the client-sent model-layer records (`inputs.sent_token_ids`, one entry per item; a
  `token_ids`-wire item is its own id list). The ids are the engine's own `/tokenize` ids of exactly what
  the client sent (R29), and they live on the record so a repository subset — a sample that may drop the
  probe — stays checkable.
- `observe/corpus.declared_kept_counts(record, recipe_block)` derives each document item's kept count with
  the product's `kept_vector_count(ids, skip)` (the one home of the rule); `derived_vectors` gained
  `kept=`, passed to the adapter's `PoolRequest.kept_counts`, so the adapter checks each item against its
  declared count instead of the full `usage.prompt_tokens`. When the rule or the ids are absent the
  resolver returns `None` and the adapter's full usage cross-check stands — never a silently accepted
  count.
- `summarise_nondeterminism(..., kept_of=)` uses the same resolver, so an engine-side-rule recipe's vector
  drift is measured instead of silently dropped (the `_comparable` `except HarnessError` used to swallow
  the decode failure).
- `_consistency` decodes the stored record and the equivalence candidate with the stored record's declared
  counts (the same `exchange_id` is the same request), so the candidate's missing `/tokenize` ids cannot
  turn a consistent corpus into a mismatch.

## 5. Item 3 — the serialization

`_Worker` keeps `_step_token` (a fresh object per `_step` call, set before the helper starts, cleared in a
`finally` when `_step` returns — including when it abandons the body). `_Worker._observe_corpus` passes
`in_flight=lambda: self._step_token is token` into the module `_observe_corpus`, whose `restart()` closure
refuses with `HarnessError` when the step is no longer in flight. That is the smallest serialization that
binds: a previous step's body *cannot* be in flight when a later step runs (the runner ends the recipe on
every abandonment — `_stop_after_failure` returns True on an engine death and on a budget overrun), so the
only reachable interleaving is the corpus step's own abandoned body reaching its restart, which the guard
now refuses. The reproduction test shows the interleaving and the refusal; the mutation run (guard removed)
shows the second `serve start`.

## 6. Red-first evidence

Every fix was written as a test first and shown red on the unfixed tree (quoted failure lines):

1. `uv run --no-sync pytest rcp-ndcg-test/tests/test_observe_corpus.py -q -k "omission or kept_rule or
   engine_ids_of_each_sent"` — 3 failed:
   - `test_a_row_the_empty_policy_omits_is_recorded_as_its_omission` →
     `AssertionError: one omission per pass (two in-process and one after the restart)` / `assert 0 == 3`;
   - `test_a_kept_rule_reply_is_checked_against_the_declared_kept_count` →
     `assert _checks(verify_corpus(kept))["scores_finite_and_vector_widths"]["passed"] is True` →
     `assert False is True` (the decode refused the kept-count reply);
   - `test_the_recorder_records_the_engine_ids_of_each_sent_render` →
     `assert isinstance(ids, list) and len(ids) == len(texts)` → `assert (False) where False =
     isinstance(None, list)`.
2. `uv run --no-sync pytest rcp-ndcg-test/tests/test_wave_corpus.py -q -k abandoned_corpus_body` →
   `AssertionError: []` / `assert (0 == 1)` (the fake body never reached its restart because the wave ended
   first; after the fix the refusal is pinned). Mutation (guard removed, test kept):
   `assert len(outcomes) == 1 ...` failed with the second `run_wave: fixture-embed serve start` on the log.
3. The equivalence-consistency half of the kept-rule test: mutation (the per-record resolver restored) →
   `AssertionError: {'check': 'equivalence_consistent', 'passed': False, ...}` / `assert False is True`.

## 7. Verification

All commands run in the lane worktree with `uv run --no-sync`.

| command | result |
|---|---|
| `ruff format --check .` / `ruff check .` | 627 files formatted / all checks passed |
| `basedpyright` | 0 errors, 0 warnings, 0 notes |
| `pytest tests/ -q -n 8` (through the heavy slot wrapper) | 4145 passed, 155 skipped |
| `pytest tests/contract tests/docs -q` | 413 passed, 107 skipped |
| `pytest rcp-ndcg-test/tests -q` (faulthandler, 300 s per-test guard) | 1184 passed, 229 skipped |
| `mkdocs build --strict` | built |
| `timeout 7200 bin/gate lane/fix10corpus` on `6df52983` | **GATE: PASS** — every step exit 0 (see §7.1) |

### 7.1 The gate's per-step SUMMARY

```
rev lane/fix10corpus = 6df52983 (slot 4)
ruff-check exit=0 All checks passed!
ruff-format exit=0 627 files already formatted
basedpyright exit=0 0 errors, 0 warnings, 0 notes
pytest exit=0 4145 passed, 155 skipped in 51.78s
contract-docs exit=0 413 passed, 107 skipped in 43.84s
mkdocs exit=0 INFO    -  Documentation built in 2.37 seconds
test-pkg exit=0 ================ 1184 passed, 229 skipped in 815.28s (0:13:35) =================
recipes exit=0 recipes: no failure outside the baseline (0 baseline failures remain, 0 fixed; pytest exit 0)
vllm-pkg exit=0 50 passed in 4.67s
vllm-models exit=0 102 passed, 8 skipped in 61.83s (0:01:01)
run_all exit=0   external_judges  ok
leaderboards: 1022 checks, 987 match, 35 known deviations, 0 failed
human study: 67 checks, 67 match, 0 known deviations, 0 failed
external LLM judges: 82 checks, 82 match, 0 known deviations, 0 failed
public-names exit=0 public-names: clean (0 baselined hits remain)
clean exit=0 clean
GATE: PASS
```

## 8. The residual root cause for pplx-embed-v2-context (not fixed here: outside the brief's item 3)

The restart race is fixed, but the same corpus step will still fail: the corpus step's **bare probes are
requests the recipe's client never sends**, and for this checkpoint a prefix-less input kills the
EngineCore.

The evidence chain:

- the checkpoint's plugin pooler raises for any `/pooling` input whose first id is neither role prefix:
  `rcp_ndcg_vllm/models/pplx/pooling_core.py` `pool_sequence` ("an input's first token id must be the query
  prefix … or the document prefix …"); vLLM answers that exception by killing the EngineCore and the server
  exits (the round-2 failure's mechanism, recorded verbatim in commit `0388da2f`: "its fixed bare-text body
  for a multi_vector recipe is a request the recipe's client never sends, and the plugin's role check
  refuses a prefix-less input by name -- the refusal kills the EngineCore");
- that commit fixed only the **smoke** (`_smoke_query_ids`); the corpus step's bare probes were not touched:
  `record.py::_bare_probes` sends the standing probes (`unknown_field`, `malformed_json`, `wrong_model`,
  `empty_input`, `over_length`) through `_role_body`, whose multi_vector body is `{"input": [<text>]}` —
  tokenized server-side, so its first id is the text's first token;
- `observe/requests.py::_bare_body` builds the plan's bare rows (the uncut ladder, the long content kinds,
  the wire variants, the edges) the same way, as text;
- the recipe declares `request_shape: token_ids`, so every one of those probes is a request its client
  never sends.

The smallest fix (for the next lane, and worth its own failing test): for a `request_shape: token_ids`
recipe, the bare probes must carry the recipe's own render ids — the smoke's pattern extended to the
document side (the `[D] ` prefix under `document_split_special_tokens`) and to the plan's uncut rows — with
one shared helper (the recipe's template + tokenizer + the declared split flags) so the harness keeps one
derivation of the client's render. The alternative (declare the engine's own refusal of an uncut *text*
unobservable for such a recipe, with the reason) is weaker but honest; the ids path keeps the stratum.

## 9. Docs updated

- `rcp-ndcg-test/schema/observation-corpus.md` — the record's `inputs` block (the engine's sent ids, the
  omission record) and the `scores_finite_and_vector_widths` description (the kept-count rule).
- `observe/corpus.py` (`derived_vectors`, `summarise_nondeterminism`, `declared_kept_counts`, the checks'
  referents, the module docstring's public surface), `observe/provenance.py::recipe_facts`,
  `record.py` (`_model_layer`, `_tokenize_probes`, `omission_exchange`), `run_wave.py`
  (`_observe_corpus`'s `in_flight`, `_step`) — the docstrings that state the changed behaviour.
- `CHANGELOG.md` — three `### Fixed` bullets (the exact text below).
- `git grep -n -i "observation corpus"` over `docs/`, `README.md`, `REPRODUCIBILITY.md`, `skills/`,
  `examples/`, `experiments/`, `mkdocs.yml` and `rcp-ndcg-test/README.md`: the remaining hits describe the
  corpus format and its reader generically (the format is unchanged; the record schema stays 1 and the new
  `inputs` key is additive), and the schema doc above is the detailed home — no live page states the old
  behaviour.

## CHANGELOG entry

The three `### Fixed` bullets (verbatim, under the 0.0.1 section):

- **The observation corpus records a row the client's declared empty policy omits** (the E2 round-10 r2
  recording failure: jina-reranker-v3's corpus failed `completeness` with `content:whitespace_only`
  missing): the recipe declares `empty_doc: omit_zero_blank` -- the paper's own `if d.strip()` rule -- so a
  whitespace-only document is omitted and scored 0.0 without a model call, and the plan's model-layer row
  for that kind (whose pairs row the stage-1 render check pruned) produced no exchange while the bare
  uncut row did.  The omission is now recorded like a client refusal (`omission_exchange`): no status, a
  `client_omission` body naming the row and the declared policy, the request id accounted for, so
  completeness holds and the stratum is recorded.  A CPU test records a fixture with the rule end to end
  and pins one omission per pass beside the kept rows' own exchanges.
- **The corpus check expects the kept-vector count for a recipe with a served engine-side document keep
  rule** (the E2 round-10 r3 recording failure: `scores_finite_and_vector_widths` was False with "the
  /pooling reply reports 165 prompt token(s) but its vectors decode to 152" for pplx-embed-v2-late
  `-0.6b`/`-9b`): the plugin drops the rule's positions from the reply, so the vectors are the KEPT ones
  while `usage.prompt_tokens` counts the whole prompt, and the corpus check decoded the reply without the
  declared counts.  The manifest's recipe block now records the declared
  `document_skip_engine_side`/`document_skip_token_ids`, each model-layer record carries the engine's own
  `/tokenize` ids of the render it sent (`inputs.sent_token_ids`, so a repository subset stays checkable),
  and the check derives each document item's kept count through the product's `kept_vector_count` and
  hands it to the adapter: a reply at the kept count passes, one at the full count (the engine ignored
  the rule) and one at a count that is neither fail.  The non-determinism measurement uses the same
  resolver, so those recipes' vector drift is no longer silently dropped.  Two CPU tests pin the check
  and the recorded ids.
- **An abandoned observation-corpus step body can no longer restart the engine** (the E2 round-10 r2
  recording failure: pplx-embed-v2-context's "the engine exited (code 0) while its steps ran"): the
  corpus step's body stops the engine and starts a fresh one between the in-process passes and the
  after-restart pass, and the step machinery abandons a body whose engine died (or whose budget ran out)
  and ends the recipe -- its GPUs are released, and the wave may hand them to another recipe.  The
  restart is now gated on the step still being the one in flight (the worker's step token): a late
  restart from an abandoned body is refused with the reason instead of stopping or starting an engine the
  wave no longer tracks.  A CPU test reproduces the interleaving (the body kills its engine, waits for
  the runner to abandon it, then calls the restart) and pins the refusal; without the guard the body
  starts a second engine.

## Public surface changes

None. No `__all__`, CLI command/flag, exit code, schema or contract snapshot of the *product* changed
(`tests/contract` and `tests/docs` pass unchanged); the `rcp-ndcg-test` package (unpublished) gained two
public names — `record.omission_exchange` and `observe.corpus.declared_kept_counts` — both documented in
their modules' docstrings; the record schema stays `RECORD_SCHEMA = 1` and the new `inputs` key is
additive.

## Files outside scope

None. The touched files are the recorder, the acceptance checks and the provenance block, the wave runner,
their two test modules, the corpus schema doc and the CHANGELOG.

## For the next lanes

- **The recording pass must re-run** for `jina-reranker-v3` (item 1) and `pplx-embed-v2-late` ×2 (item 2);
  both are harness-side fixes and need no recipe change.
- **`pplx-embed-v2-context` needs §8's fix** (the bare probes on a `token_ids` wire) before its corpus step
  can pass; the restart guard alone leaves it failing cleanly (no abandoned restart, but the engine still
  dies on the first prefix-less bare probe).
- **The first full recording after these fixes** will also exercise the new `inputs.sent_token_ids`
  annotation and the kept-count check on the real engine replies: a reply at neither the kept nor the full
  count fails by name, so a plugin that stops applying the rule is caught, not absorbed.
