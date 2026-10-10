# Lane `mrl-recipes`: every variant declares its MRL (owner decision 39)

## 1. Status

**DONE.** The gate passes on the final tip `19b6527a` (the lane merged `rfc-0001` at `26da5852` -- round
8: l08-judges and fp-v4, then content-wire and core-records -- and re-gated). Every embedding and
multi-vector variant in the merged catalog (24 families, 44 recipes: 16 retrieval families/34 variants and
8 judge families/10 recipes) declares its MRL kind and the model card's set; `recipe:<id>` selects `k`
from that declaration, and a `k` equal to the full width is the identity selection (no head, no record);
the goldens, deltas and corpora moved through their documented paths, each change listed; the docs catalog
carries the MRL column. Every recipe stays `status: unverified` (the GPU waves are the owner's).

## 2. Commits

| Commit | Subject |
|---|---|
| `7eb9210d` | The MRL selection is not a content disagreement: `recipe:<id>` plus `mrl_dim`/`dimensions` selects `k` from the declared set (decision 39) |
| `636105c2` | The recipe schema declares the MRL kind and set per size, and the serve-side engine gate is one checked rule (decision 39) |
| `5def2433` | Every embedding and multi-vector recipe declares its MRL head: the card's kind and set per variant, the engine gate where it serves one, and zembed's learned projection chain (decision 39) |
| `99031b84` | The MRL declarations' goldens, deltas and corpora: every resolved-contract change is declared, the qwen3-embedding corpus is re-keyed on metadata only, and the e2e goldens regenerate |
| `ffb091b3` | Docs and the changelog: the MRL catalog column, the recipe selection, the per-size declaration and the one serve/client rule |
| `92e74645` | The zembed projection-chain test clears a sibling module's forced-offline Hub for its read and skips when the Hub is unreachable |
| `958d01f9` | Verifier round 1: a recipe's own declared selection stays CONTENT, and the selection rule reads the declaration through the product's `MrlHead` |
| `4c938dcb` | Verifier round 1 minors: the cross-family MRL-kind guard, the zembed projection digest pin, the corpus index formatting and the matryoshka page's client-cut reason |
| `9e673083` | Merge `rfc-0001` (`01f4b9be`: qa-prep, l10c/l10d, sz-pplx, media-rules, export-seam) into the lane: the declarations port onto the 30-variant catalog, pplx-embed-v1 declares `mrl_kind: none` (its card's MRL row has no set), the goldens/DELTAS reconcile |
| `ba10b01a` | Final verification minors: a malformed operator MRL declaration is a typed refusal (no raw `OverflowError`/`TypeError`), and the merge-added deltas name their change |
| `6828cbaf` | Merge `rfc-0001` (`d630e4a6`: the remaining Qwen3 sizes): qwen3-embedding 4b/8b and qwen3-vl-embedding 8b declare their card's `mrl_range`, the family tests carry the per-size range |
| `76b45a82` | Merge `rfc-0001` (`afecce00`: harness-fix + ci-recipes) into the lane |
| `dff1edcc` | The full-width selection is the identity selection: `k == dim` applies no head and writes no `mrl_cut` record, `k > dim` stays refused, on both MRL routes and the sweep (owner decision, 2026-10-09) |
| `2168c783` | Lane report update: the operator's identity-selection decision and the pplx-embed-v1 `none` decision, with the amendment's tests and gate |
| `ecfaa38d` | Merge `rfc-0001` (`45b66e1b`: l08-judges, fp-v4, the integration commits) into the lane: the declarations port onto the 24-family/44-recipe catalog, the qwen3-embedding-0.6b corpus is re-keyed once more from the `rcp-fp/4` state to cover `serve.hf_overrides`, the `rcp-fp/4` goldens keep their captures with the MRL changes declared as DELTAS, and the dated provenance is dropped from the product docstrings |
| `19b6527a` | Merge `rfc-0001` (`26da5852`: content-wire and core-records) into the lane |

## 3. What changed

**1. The product selection rule (`rcp-ndcg/src/rcp_ndcg/inference/recipes.py`).** `expand_role_recipe`
treats a config's `mrl_dim`/`dimensions` as a *selection* when the recipe itself declares none
(`client.get(key) is None`): a `k` inside the recipe's `mrl_dims`/`mrl_range` is merged (no longer a CONTENT
disagreement with the recipe's declared `null`), and one outside is refused naming the set. A recipe that
DOES pin a selection keeps it: an in-set config override is refused naming both values (CONTENT equal or
refused). The declared set is read through the product's one MRL head home (`rcp_ndcg.data.mrl.MrlHead`:
`declaration` text and `supports` membership), with a tolerant fallback to the ordinary merge for a
malformed operator declaration (a raw `OverflowError`/`TypeError` never escapes; the endpoint's own
validation refuses the block).

**2. The recipe schema's one rule (`rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipe.py`).** `PER_VARIANT_CLIENT_FIELDS`
gains `mrl_kind`/`mrl_dims`/`mrl_range`/`mrl_projection` (a size's card set changes with its width), and
`_mrl_declarations_agree` checks the serve-side gate against the client declaration: a discrete engine
`matryoshka_dimensions` and the client's `mrl_dims` are the same set; an open `is_matryoshka` gate needs a
bounded client `mrl_dims`/`mrl_range`; a gate beside a projection kind, an explicit `false` beside a list,
an empty list and a non-numeric list are refused with the rule's own message. The rule runs on every
resolved variant (per-size overrides included).

**3. The recipe declarations (every embed/multi-vector family).** From `handover/specs/mrl-cards.md` at the
pinned revisions: qwen3-embedding 0.6b/4b/8b and qwen3-vl-embedding 2b/8b declare `mrl_kind: truncation`
with their card's prose `mrl_range` ([32, 1024], [32, 2560], [32, 4096], [64, 2048], [64, 4096]) and
`serve.hf_overrides: {is_matryoshka: true}`; jina small/nano and embeddinggemma-2 declare the card's
discrete `mrl_dims` (32..1024 / 32..768 / 128/256/512/768) mirrored exactly in
`serve.hf_overrides.matryoshka_dimensions`; topk small/xsmall and pplx-embed-v2-context declare their
card's sets (`mrl_kind: truncation`, client-side head; `/pooling` refuses the per-request `dimensions`);
zembed declares `mrl_kind: projection` with `mrl_projection.source`
(`hf://zeroentropy/zembed-1-embedding@cf13c81f.../projections.safetensors`) and the six projected sizes;
octen, harrier, pplx-embed-v2-late (0.6b and 9b) and pplx-embed-v1 (0.6b, 4b) declare `mrl_kind: none`
(the operator confirmed pplx-embed-v1 stays `none`: its card's `MRL: Yes` row gives no set, and no set is
invented). No recipe defaults the engine-side `dimensions` or `mrl_dim`; the recipes ship the checkpoint's
full width.

**4. Goldens, deltas and corpora.** The golden guard's accepted differences moved only through the
documented mechanisms: existing variants' changes are declared in `golden/DELTAS.json` (each entry with a
reason and evidence; 137 entries after the final merge, key-unique, one per current difference, no stale
delta), the e2e goldens regenerated through `RCP_UPDATE_GOLDENS=1` and the documented candidate-copy path,
and the qwen3-embedding-0.6b corpus was **re-keyed on metadata only** (the documented procedure in
`docs/how-to/use-verified-fake-engines.md`: its recorded exchanges carry no `dimensions`, so the new
serve-side gate shapes none of them; manifest fingerprint/inputs + `recipe.rekeyed`, subset and engine
indexes, `git mv`, and the suite's verification record appended). The qwen3-vl-embedding-2b corpus's stale
declaration gains `serve.hf_overrides` for the RC0 re-record. The goldens themselves were not hand-edited.

**5. Docs and the catalog.** `rcp-ndcg-vllm/README.md`'s catalog table gains the `MRL` column (the declared
kind and set per variant, `none` where the card declares no head); `docs/reference/recipes.md` documents the
column and the `recipe:` selection exception; `docs/how-to/add-a-model.md` lists the MRL per-size fields and
the one-rule declaration; `docs/concepts/matryoshka.md` gains "The recipes' declarations" (the one
serve/client rule and the reasons a card ships without a serve gate) and the identity-selection paragraph;
`CHANGELOG.md` carries the entry below.

**6. The identity selection (operator decision, 2026-10-09).** A `k` equal to the checkpoint's own width is
the identity selection on every MRL route: `MrlHead.apply` returns the vectors unchanged (no slice, no
renormalisation), the dense and pooling clients skip the head and write no `mrl_cut` record, the pooling
config refuses only `mrl_dim > dim` (not `>=`), and the ex-post sweep's `k == full_width` artifact is the
stored full-width vectors. The selection still enters the config's identity; the card's full-width member
(topk 2048/1024) stays declared and selectable. Red-first tests: the head's identity, both clients'
no-head/no-record paths, the pooling config's `==` allowed / `>` refused, the topk full-width selection
end to end, and the sweep's identity k against a direct run. `schemas/run-config.v1.json` and
`schemas/index.v1.json` regenerated; the matryoshka page and the CHANGELOG updated. The operator later
withdrew the dated provenance from the product docstrings and the CHANGELOG wording (the rule stands on
its own).

**7. The port onto `rfc-0001` round 8 (`ecfaa38d`, `19b6527a`).** `rfc-0001` moved to `45b66e1b` (l08-judges:
ten judge recipes as families; fp-v4: `rcp-fp/4`, every golden re-captured, the corpora re-keyed, DELTAS
emptied) and then `26da5852` (content-wire, core-records). The merge ported the MRL declarations onto the
24-family/44-recipe catalog (all 22 embed/multi-vector variants still declare their kind; the 10 judge
recipes are neither embed nor multi-vector and carry no `reference`); kept both cross-family assertions in
`test_recipes_root.py`; took fp-v4's CONTENT `batch_size` docstring; dropped the dated provenance from the
product docstrings, the matryoshka page and the CHANGELOG; re-keyed the qwen3-embedding-0.6b corpus once
more from the `rcp-fp/4` state (`4a6afbe...` -> `7e14af04...`, metadata-only: `serve.hf_overrides` is the
only moved input, no recorded request carries `dimensions`) and appended its verification record; left
`octen-embedding-8b` at fp-v4's key (its merged recipe's fingerprint is `unchanged`: `mrl_kind: none` is a
post-processing field, so there was nothing to re-key); declared the MRL resolved-contract changes as 85
DELTAS entries against fp-v4's re-captured goldens (each with a reason and evidence; the golden guard
passes, no stale delta); regenerated the e2e golden replay pins and the four-phase script golden through
their documented paths; and regenerated the two schemas for the docstring changes.

## 4. Verification

**Round 1 (two fresh-context verifiers, lenses A/B, DeepSeek-V4.1-flash at `xhigh`, on `92e74645`).**

- **Lens A: PASS**, one minor: a recipe that declares its own `dimensions`/`mrl_dim` was silently overridden
  by an in-set config selection. Fixed in `958d01f9` (the selection branch fires only when the recipe
  declares none; a pinned selection is CONTENT and a differing in-set override is refused naming both),
  with a red-first test (`test_a_recipe_that_declares_its_own_selection_keeps_it`).
- **Lens B: PASS**, five minors: the declaration reader duplicated `rcp_ndcg.data.mrl` (fixed by routing it
  through `MrlHead`); the zembed projection test skipped on any fetch error (fixed with the pinned file's
  SHA-256, and the source string is pinned by the contract test); no cross-family test required an
  embed/multi-vector variant's `mrl_kind` (fixed in `test_recipes_root.py`, mutation shown red); the corpus
  index files were reindented 2->1 space and the engine entry appended out of order (fixed to canonical
  2-space sorted JSON); the matryoshka page omitted the `/pooling` refusal as a client-cut reason (fixed).
  All in `958d01f9`/`4c938dcb`.

**Final pair (two fresh-context verifiers, lenses A/B, on the merged `9e673083`; no round 2 fix round was
required because round 1 found no blocker or major, per COMMON).**

- **Lens A: PASS**, two minors: topk's declared full-width member is unselectable on `/pooling`
  (`mrl_dim >= dim` is refused) -- kept and reported in Open questions (the card's set includes the full
  width; the recipes ship it unselected); six merge-added deltas had generic provenance -- fixed in
  `ba10b01a` (the entries now name the change).
- **Lens B: PASS**, two minors: `OverflowError` escaped the malformed-declaration fallback, and a
  non-numeric engine list leaked a raw `TypeError` -- both fixed in `ba10b01a` with red-first tests
  (`test_a_malformed_mrl_declaration_falls_through_to_a_typed_refusal`, the new loader-rule parametrized
  case).

The verifiers re-derived the declarations against `handover/specs/mrl-cards.md` and the live pinned cards,
mutation-tested the fixes (six mutations, each red), independently parsed the README MRL column against all
variants, verified the corpus re-key byte-for-byte and the stale entry's inputs, and ran the focused, full
and network-gated suites. No blocker or major was found in either round; the two open items the pair left
(the topk full-width member and the pplx-embed-v1 set) were decided by the operator on 2026-10-09 and are
recorded below.

**Operator amendment (2026-10-09, after the report was accepted).** The operator allowed `k == dim` as the
identity selection on every MRL route and withdrew the pplx-embed-v1 set request. Implemented in
`dff1edcc`, tests first: the head's identity test, both clients' no-head/no-record tests, the pooling
config's `==` allowed / `>` refused tests, the topk full-width selection end to end, and the sweep's
identity k against a direct run; schemas regenerated, the matryoshka page and the CHANGELOG updated; the
full gate passes on `dff1edcc`. No further verifier round was run: the operator's note named the process
(tests first, schemas/docs, gate) and the change is covered by the existing suites plus the new tests.

**Merges.** `rfc-0001` was merged six times as it advanced: `01f4b9be` (the port onto the 30-variant
catalog: pplx-embed-v1 declares `mrl_kind: none`, the deltas reconcile), `d630e4a6` (the remaining Qwen3
sizes declare their `mrl_range`; the family tests carry the per-size range), `afecce00` (harness-fix +
ci-recipes; no conflict), `45b66e1b` (l08-judges + fp-v4: the port onto the 24-family/44-recipe catalog,
the corpus re-key, the DELTAS against fp-v4's goldens), and `26da5852` (content-wire + core-records; no
conflict). Each merge was re-gated.

**Port verification (operator-directed, after the report was accepted).** The port onto round 8 was
implemented under the operator's process (merge, resolve, regenerate the documented way, tests, gate); no
verifier round was run because the operator's note named the process and the port changes no product
behaviour beyond the already-verified MRL rules. The evidence: the golden guard passes (49 tests, 85
declared MRL deltas, no stale delta); the conformance suite passes (the re-keyed qwen3-embedding corpus
replays green; the 7 stale corpora match their recomputed inputs exactly); the network-gated family tests
all pass one file at a time; the full root and test-package suites pass; and `bin/gate lane/mrl-recipes`
passes on `19b6527a`.

## 5. Checks

Last runs on the final tip `19b6527a` (gate log `gates/19b6527a/SUMMARY`):

```text
bin/gate lane/mrl-recipes                     -> GATE: PASS
ruff-check exit=0 / ruff-format exit=0 (586 files) / basedpyright exit=0 (0 errors)
pytest exit=0                                 -> 3686 passed, 102 skipped
contract-docs exit=0                          -> 301 passed, 55 skipped
mkdocs exit=0 (strict) / test-pkg exit=0      -> 938 passed, 223 skipped
recipes exit=0 (no failure outside the baseline)
vllm-pkg exit=0 (49 passed) / vllm-models exit=0 (72 passed, 7 skipped)
run_all exit=0 -> leaderboards 1022 checks, 987 match, 35 known deviations, 0 failed;
                  human study 67/67; external judges 82/82
public-names exit=0 / clean exit=0 (clean tree)
```

Focused evidence (final): `tests/inference/test_recipe_reference.py` 23 passed; `test_recipe.py` 47 passed;
`test_family_goldens.py` 49 passed (85 declared MRL deltas, no stale delta); `tests/conformance` 63
passed/1 skipped; the network-gated MRL recipe files one at a time (qwen3-embedding, qwen3-vl-embedding,
jina, embeddinggemma-2, topk, pplx-context, pplx-embed-v1, pplx-embed-v2-late, octen, harrier, zembed,
recipes-root) all green; `test_zembed_1.py`'s projection-chain test asserts the real file's SHA-256
(`c2857f09a857d564c78224cdc7baa763773fa96475dd7b9b29d598756b61d083`), its six F32 tensor shapes and the
head's chain for one `k`; the qwen3-embedding-0.6b corpus's `integrity_mismatches` is empty and
`recipe_state` is `unchanged` at the new key `7e14af04...` (two `rekeyed` entries: fp-v4's schema/engine
re-key and the MRL gate re-key).

## 6. Open questions

- **Both verifier-left items were decided by the operator on 2026-10-09.** The topk full-width member is
  now selectable as the identity selection (`k == dim`: no head, no record; `k > dim` refused), and
  pplx-embed-v1 stays `mrl_kind: none` (the later "declare its set" bullet was withdrawn; no set is
  invented). No open question remains from the verifier rounds.
- **A recipe's own selection is CONTENT.** A recipe that pins `dimensions`/`mrl_dim` keeps it; an in-set
  config override is refused naming both. No shipped recipe pins a selection (the operator's rule is that
  recipes ship the full width), but if a future recipe ships a default `k` that a run should be able to
  override, the rule changes.
- **The qwen3-embedding corpus is re-keyed, not re-recorded.** The documented metadata-only re-key keeps
  the conformance suite replaying it; the RC0 wave should still re-record it with the rest of the
  provisional corpora, and qwen3-vl-embedding-2b is declared stale for its own re-record. The
  octen-embedding-8b corpus needed no second re-key after the port: its merged recipe's fingerprint is
  `unchanged` from fp-v4's key (`mrl_kind: none` is a post-processing field and never moves the request
  bytes), so fp-v4's key already covers it.
- **The zembed projection chain is CPU-tested, not GPU-tested.** The real-file test pins the source's
  digest, tensors and one chain application; the per-`k` vector gates stay in the GPU wave (no GPU pytests,
  per the repo rule).

## CHANGELOG entry

Under `## Unreleased` / `### Public surface`:

```markdown
- **Every recipe declares its MRL head, and `recipe: <id>` selects from it** (owner decision 39): every
  shipped embedding and multi-vector variant declares its kind and the model card's set once in
  `family.yaml` (`client.mrl_kind` with `mrl_dims`/`mrl_range`; `mrl_projection` for the projection kind)
  and, where the card supports a cut and the engine serves it, the same set in `serve.hf_overrides`
  (`is_matryoshka`/`matryoshka_dimensions`), which the loader checks as one rule -- a discrete engine list
  and the client's `mrl_dims` are the same set, an open gate still needs a bounded client declaration, and
  a serve gate beside a projection kind is refused. The recipes ship the checkpoint's full width, and a
  `recipe: <id>` config's `mrl_dim`/`dimensions` is a *selection*: accepted when `k` is in the declared set
  and refused naming the set otherwise (no longer a CONTENT disagreement with the recipe's declared
  `null`). The per-variant client whitelist (`PER_VARIANT_CLIENT_FIELDS`) gains the MRL fields (a size's
  card set changes with its width), and the `MRL` column of the `rcp-ndcg-vllm` catalog names every
  variant's set.
```

and the `### Changed` bullet:

```markdown
- **The full-width selection is the identity selection**: a `k` equal to the
  checkpoint's own width (`mrl_dim` on either route, `dimensions` on the dense route) applies no head and
  writes no `mrl_cut` `ProcessingRecord`, so the card's full-width member stays selectable (topk's 2048 /
  1024, a range's ceiling); the selection still enters the config's identity, the ex-post sweep's
  `k == full_width` artifact is the stored full-width vectors, and a `k` wider than the vectors is still
  refused.
```

## Public surface changes

- **Behaviour**: `rcp_ndcg.inference.recipes.expand_role_recipe` accepts a config's `mrl_dim`/`dimensions`
  inside the recipe's declared set and refuses one outside it naming the set (previously refused as a
  CONTENT disagreement whenever the recipe declared `dimensions: null`); a recipe's own pinned selection
  stays CONTENT. A `k` equal to the full width is the identity selection (`MrlHead.apply` returns the
  vectors unchanged; the clients write no `mrl_cut` record; `PoolingEndpoint` refuses only `k > dim`).
- **Recipe schema**: `rcp_ndcg_vllm.recipe.PER_VARIANT_CLIENT_FIELDS` gains `mrl_kind`, `mrl_dims`,
  `mrl_range`, `mrl_projection`; the loader gains the serve/client MRL consistency refusal.
- **Schemas**: `schemas/run-config.v1.json` and `schemas/index.v1.json` regenerated for the `mrl_dim`
  descriptions. No new Python names, CLI commands/flags or exit codes from this lane; the merged
  `recipe.schema.json`/`family.schema.json` changes are `rfc-0001`'s harness-fix fields.

## Files outside scope

- `rcp-ndcg-test/corpora/vllm-0.31.0/` (the qwen3-embedding-0.6b metadata-only re-key: the manifest, its
  subset index, the engine index, and the suite's appended verification record) -- required to keep the
  conformance replays green under the new serve-side gate.
- `rcp-ndcg-test/tests/e2e/golden/goldens.json` and
  `rcp-ndcg-test/tests/fixtures/golden/job-text-four-phases.sh` (regenerated for the re-keyed fingerprint
  and the serve argv's new `--hf-overrides`).
- `rcp-ndcg-test/tests/conformance/stale.json` (the qwen3-vl-embedding-2b entry gains
  `serve.hf_overrides`).
- Tests updated for the declarations: `test_recipes_root.py`, `test_zembed_1.py`, `test_pplx_embed_v1.py`,
  `test_pplx_embed_v2_late.py`, `test_qwen3_embedding.py`, `test_qwen3_vl_embedding.py`, the other
  per-family contract modules, `test_recipe.py`, `test_recipe_reference.py`.
- Tests for the identity selection: `tests/data/test_mrl.py`, `tests/inference/test_mrl.py`,
  `tests/inference/test_pool_client.py`, `tests/retrieval/test_store.py`, and the regenerated
  `schemas/run-config.v1.json`/`schemas/index.v1.json`.
- Port-only resolutions (round 8): the `CHANGELOG.md` union, the `config.py` `batch_size` docstring (taken
  from fp-v4), the dropped dated provenance in `rcp-ndcg/src/rcp_ndcg/data/mrl.py` and
  `docs/concepts/matryoshka.md`, and the regenerated `rcp-fp/4`-based DELTAS/e2e goldens/schemas.

## For the next lanes

- **The RC0 wave** re-records qwen3-vl-embedding-2b (declared stale) and re-checks the re-keyed
  qwen3-embedding-0.6b corpus; the release checklist wants `stale.json` empty.
- **The GPU waves** should exercise at least one engine-side `dimensions` selection, one client-side
  `mrl_dim` selection and one identity selection (`k == dim`) per family (the gates are declared but no
  shipped recipe selects a `k`).
- **The harness (study items 5-8) remains open**: the fakes' `/pooling` should refuse `dimensions`,
  `/embeddings` should validate the `is_matryoshka` gate/set, and stage 2 should gate each declared `k`
  ex-post from one full-width reference run. The declared sets are now there for it to read.
- **pplx-embed-v1** stays `mrl_kind: none` per the operator (2026-10-09); a future set would be a new
  declaration with its own goldens/deltas.
- **`rcp-fp/4` and the corpora**: the qwen3-embedding-0.6b corpus carries two `rekeyed` entries (fp-v4's
  schema/engine re-key and the MRL gate re-key); a future fingerprint-schema move re-keys it again, never
  hand-merges.
