# Lane fix12ctx — pplx-embed-v2-context's per-chunk corpus check

**Lane:** `lane/fix12ctx` (worktree outside the operator's checkout). **Base / merged:** the current `rfc-0001`
tip is `c668c439` ("Merge lane fix11topk"), and it is already the merge base of every commit here (an ancestor,
not a merge commit): nothing to merge. **Gate:** `timeout 7200 bin/gate lane/fix12ctx` on `411b5bdf` (slot 3) —
**GATE: PASS**, every step exit 0 (ruff, format, basedpyright 0 errors, root suite 4145 passed / 155 skipped,
contract+docs 413/107, mkdocs strict, test-pkg 1194/229, recipes, vllm-pkg, vllm-models, run_all, public-names,
clean). No GPU work in this lane; nothing pushed; `rfc-0001` was not merged into.

## 1. Status

Done. The one item: `scores_finite_and_vector_widths` no longer decodes a per-chunk recipe's reply as a
token-per-position one. The check now expects the **declared per-chunk count** per item — the served plugin's
own segmentation over the ids the client sent — for a recipe whose client declares `outputs: per_chunk`; a
reply at the chunk count passes, one at the prompt-token count (the engine ignored the per-chunk head) and one
at a count that is neither fail, naming the expected count; a token-per-position recipe's rows are unchanged.

The recording re-run on the node is the wave's step (this lane runs no GPU): the corpus step's failure class is
reproduced on CPU and the fix is pinned by a test (below).

## 2. Commits

| commit | what |
|---|---|
| `bb85134d` | the id-level contract and the per-chunk segmentation move to the plugin's pure-Python `chunk_rule` (the pooling core reads the one home); keyed into the behaviour fingerprint; goldens regenerated; NOTICE + the derived-files audit updated |
| `411b5bdf` | the corpus check expects the declared per-chunk count (the recipe block records `outputs`; the resolver + check + the non-determinism/equivalence decode; the test; the schema doc; the CHANGELOG) |
| (report) | this report, `handover/reports/fix12ctx.md`, and the lane copy |

No co-author lines, no attribution, nothing pushed. The gated revision is `411b5bdf`; the report commit
follows it and changes no code.

## 3. The red-first evidence

The test (`rcp-ndcg-test/tests/test_observe_corpus.py::test_a_per_chunk_reply_is_checked_against_the_declared_chunk_count`)
was written first and run against the unfixed check (the plugin's `chunk_rule` module already existed as a pure
move, so the red is the check's own failure, not an import error):

```
$ uv run --no-sync pytest rcp-ndcg-test/tests/test_observe_corpus.py::test_a_per_chunk_reply_is_checked_against_the_declared_chunk_count -q
FAILED ...::test_a_per_chunk_reply_is_checked_against_the_declared_chunk_count
>       assert _checks(verify_corpus(chunks))["scores_finite_and_vector_widths"]["passed"] is True
E       assert False is True
```

and the check's own mismatch on the same synthetic corpus, printed by a scratch reproduction (the exact class
of the rc0 failure):

```
exchange 0067c078...: the vector reply does not decode: the /pooling reply reports 6 prompt token(s) but its
vectors decode to 2; a token_embed answer has one vector per prompt token, so the declared dim disagrees with
the served checkpoint (or the server ignored the token_embed task)
```

After the fix the same test passes (33 passed in the module), including the two failure arms:

```
item 0 decodes to 5 vector(s), not the declared per-chunk count 2 the served plugin's segmentation gives its sent ids
item 0 decodes to 3 vector(s), not the declared per-chunk count 2 the served plugin's segmentation gives its sent ids
```

## 4. The one home of the per-chunk count

The rule that says how many vectors a per-chunk reply carries per item is the plugin's own segmentation: one
span mean per chunk for a document (the segments between the boundary markers of the `[D] `-prefixed render),
one vector for a query. It lived inside `pooling_core.py` (pure torch, so the harness — which holds no torch
and no transformers — cannot import it). It now lives in
**`rcp_ndcg_vllm.models.pplx.chunk_rule`**: the id-level contract (`QUERY_PREFIX_TOKEN_ID`,
`DOCUMENT_PREFIX_TOKEN_IDS`, `PREFIX_SPACE_TOKEN_ID`, `BOUNDARY_TOKEN_ID`), `is_engine_dummy`,
`document_body`, `document_spans` and `chunk_count`. `pooling_core.pool_sequence` imports and uses the same
functions and re-exports the constants (the pooler's behaviour is byte-identical; its contract tests pass), and
the corpus check imports `chunk_count` from the same module — one home, no second rule.

The harness side mirrors the late-interaction kept-count fix: the manifest's recipe block records the declared
`outputs` head (`observe.provenance.recipe_facts`), and `observe.corpus.declared_chunk_counts(record,
recipe_block)` derives each item's count through `chunk_count` over the ids the client sent
(`inputs.sent_token_ids`, or the request body's own id list on a `token_ids` wire), exactly as
`declared_kept_counts` derives the kept counts. `derived_vectors(..., outputs=...)` hands the declared head to
the product's adapter (the per-chunk opt-out of the usage cross-check), and `_numbers` compares each decoded
item against its declared count; `_comparable`/`summarise_nondeterminism`/`_consistency` decode with the same
declared head, so a per-chunk recipe's vectors are measured for non-determinism instead of silently dropped.

The rule is keyed into the behaviour fingerprint (`ARCHITECTURE_MODULES["PplxContextualModel"]`), so a change
to the segmentation moves the corpus key. The keyed-list change moved the shared `rcp_ndcg_vllm.models` module,
which every plugin recipe's fingerprint keys: 14 goldens were regenerated with
`pytest rcp-ndcg-test/tests/recipes/test_family_goldens.py --update-goldens` (the context recipe's also gains
`plugin_sha256.rcp_ndcg_vllm.models.pplx.chunk_rule`), and the wave re-records those recipes' corpora.

## 5. Tests

- `rcp-ndcg-test/tests/test_observe_corpus.py::test_a_per_chunk_reply_is_checked_against_the_declared_chunk_count`
  (new, red first): a two-chunk document's ids (the real plugin prefixes + one boundary marker) with a reply at
  2 rows passes; a reply at the prompt-token count (5) and one at 3 rows fail, naming the expected count; the
  same reply at the prompt-token count passes under a recipe block with no per-chunk head.
- The plugin's own tests (`rcp-ndcg-vllm/tests/models/pplx/test_contract_core.py`, `test_serving.py`,
  `test_no_torch.py`, `test_plugin_modules.py`) pin that the move changed no pooling behaviour and that
  `chunk_rule` imports neither vLLM nor torch.
- The gate ran the whole bar: root suite, contract+docs, the test package (1194 passed), the recipe goldens,
  the vLLM package and model tests, `experiments/run_all`, public-names, clean.

## 6. Docs updated

- `rcp-ndcg-test/schema/observation-corpus.md` — the manifest's recipe block now names the declared `outputs`
  head, and the `scores_finite_and_vector_widths` description states the per-chunk expectation (the plugin's
  `chunk_rule.chunk_count` over the sent ids).
- `rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/pplx-embed-v2-context/family.yaml` — the prefix-space note now names
  `chunk_rule.PREFIX_SPACE_TOKEN_ID` (the constant's new home).
- `rcp-ndcg-vllm/src/rcp_ndcg_vllm/models/pplx/pooler.py` and `pooling_core.py` — the module docstrings point
  the segmentation at `chunk_rule` (the pooling core no longer restates the id contract).
- `NOTICE` (×5, byte-identical) and `tests/docs/test_packaging.py` — the new derived plugin file is attributed.
- Greps run: `git grep -n "pooling_core" -- .` (every hit reviewed; the NOTICE copies, the audit list, the
  family note, the pooler docstring and the plugin tests' imports updated or still true);
  `git grep -n "chunk_rule\|pooling_core" -- docs skills README.md REPRODUCIBILITY.md examples experiments`
  (no hits); `git grep -n "per_chunk" -- docs rcp-ndcg-test/schema rcp-ndcg-vllm/README.md` (the concept page's
  adapter description stays true — the corpus check is the harness side and is documented in the schema doc).

## 7. CHANGELOG entry

Under `## 0.0.1` → `### Fixed`, appended (verbatim in the CHANGELOG):

> **The corpus check expects the declared per-chunk count for a per-chunk recipe** (the final RC's recording
> failure: `scores_finite_and_vector_widths` was False with "the /pooling reply reports 9 prompt token(s) but
> its vectors decode to 1" for pplx-embed-v2-context-9b-preview): the served plugin pools one vector per
> document chunk, so a one-chunk document legitimately answers one vector while `usage.prompt_tokens` reports
> the whole prompt, and the corpus check decoded the reply as a token-per-position one.  The manifest's recipe
> block now records the declared `outputs` head; the check reads it and, for `outputs: per_chunk`, expects the
> DECLARED chunk count per item -- the plugin's own segmentation over the ids the client sent, one vector per
> document chunk and one for a query -- naming the expected count on a mismatch; a reply at the prompt-token
> count (the engine ignored the per-chunk head) and one at a count that is neither fail.  The segmentation and
> the id-level contract (the role prefixes, the boundary id, the prefix-space rule) moved to the plugin's new
> pure-Python `rcp_ndcg_vllm.models.pplx.chunk_rule` (no vLLM, no torch, so the harness imports it; the
> pooling core reads the same one home), which is keyed into the behaviour fingerprint: the shared
> `rcp_ndcg_vllm.models` module moved every plugin recipe's fingerprint, so the wave re-records those
> recipes' corpora and the goldens are regenerated.  The non-determinism measurement decodes per-chunk replies
> with the declared head too, so their vector drift is no longer silently dropped.  A CPU test pins the check
> (a reply at the chunk count passes; one at the prompt-token count and one that is neither fail, naming the
> expected count; a token-per-position recipe is unchanged).

## 8. Public surface changes

- `rcp_ndcg_vllm.models.pplx.chunk_rule` is a new module with `__all__` (the plugin's internal modules are not
  part of the frozen public surface, which is `rcp_ndcg_vllm.recipe`; the module is a keyed engine module).
- The observation corpus's manifest recipe block gains `outputs` (`rcp-ndcg-test`, unpublished; the schema doc
  updated).  No `schemas/`, CLI, exit-code or snapshot change; `tests/contract/snapshots/` is untouched.
- `PoolRequest` is unchanged (the adapter's existing `outputs` field carries the declared head).

## 9. Files outside scope

- `NOTICE` (×5) and `tests/docs/test_packaging.py`: the attribution/audit lists must name the new derived
  plugin file (the audit test fails otherwise) — the minimal necessary edits.
- The 14 recipe goldens: regenerated, not hand-edited (the fingerprint key moved).

## 10. For the next lanes

- The wave's recording re-run is the node-side confirmation: `pplx-embed-v2-context-9b-preview`'s corpus step
  should now pass, and every plugin recipe's corpus key moved (the shared `rcp_ndcg_vllm.models` module), so
  their corpora are re-recorded (the goldens already carry the new fingerprints).
- The equivalence stage's exchanges of the same requests are decoded with the recipe's declared head; if a
  future per-chunk plugin's segmentation differs, `chunk_rule` is the one home to extend (the check calls it
  through the recipe's declared `outputs`).
