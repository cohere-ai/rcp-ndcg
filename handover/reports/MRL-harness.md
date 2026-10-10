# Lane `mrl-harness`: MRL in the validation harness (owner decision 39)

## 1. Status

DONE. The verified fake engine mirrors vLLM v0.31.0's Matryoshka rules on both routes, stage 2 gates
every declared `k` ex-post from one full-width run, the observation request set records an MRL stratum
read from the declaration, and the negative controls gained `(g)` an undeclared cut. `bin/gate
lane/mrl-harness` passes on the code tip (`f8e70cea`; the gate SUMMARY is under the revision's short
sha). The lane is based on `rfc-0001` @ `afecce00` (harness-fix and mrl-core merged), which is already an
ancestor of the tip: there was nothing new to merge at report time.

## 2. Commits

| Commit | Subject |
|---|---|
| `286ff071` | The verified fake engine mirrors vLLM's Matryoshka gates on both routes: `/pooling` refuses the per-request `dimensions`, `/embeddings` validates `is_matryoshka`, the range and the declared set, then slices the full-width vector before the L2 (one draw, so a wrong cut order or set is caught), and the recipe's `serve.hf_overrides` supply the gate facts |
| `bc41fd38` | Stage 2 gates every declared `k` ex-post from one full-width run: the served client's selection is stripped, the product's `MrlHead` cuts both sides per declared `k` (a set's members, or a range's endpoints and the run's selection) and one gate row per `k` lands in `equivalence.json` and the report, with `MRL_GATE_VERSION` naming the derivation (`k` never enters a stored-output key) |
| `b8535898` | The observation request set gains an MRL stratum read from the declaration (one bare `dimensions=k` probe per declared `k`, or a range's endpoints and the run's selection; `CORPUS_PLAN_VERSION` 2), the negative controls gain `(g)` an undeclared cut the engine's own gate must refuse, and the stub engine mirrors the same gates |
| `e47d2771` | Docs: the equivalence harness's per-`k` ex-post gate (the matryoshka page and the T2 bullet) and the corpus plan's MRL stratum |
| `1afffef0` | Round-1 verifier fixes: a range declaration gates the run's selection read from the recipe before the served client strips it (the major finding), the stage-2 summary keeps `n_vectors` as the base vectors and adds `n_comparisons`, the corpus doc lists control `(g)`, the plan version is pinned, the projection branch has a unit test, and the stub's width gate is documented as the emulated model's `DIM` |
| `f8e70cea` | Round-2 verifier minors: the range fix's client-side `mrl_dim` half is pinned by a multi-vector range test (stage 2 and the corpus plan), `n_vectors` counts only comparisons that produced a cosine, and the controls docstring says seven |
| `<report>` | handover: the MRL-harness report |

## 3. What changed

**1. Stage 2 gates every declared `k` ex-post (brief item 1).**
`rcp-ndcg-test/src/rcp_ndcg_test/equivalence/stages.py` builds the served client with its MRL selection
stripped (`wire.role_client(..., full_width=True)` pops `dimensions`/`mrl_dim` before the endpoint is
constructed), so the served engine and the reference subprocess both answer at full width. The run's
selection is read from the recipe's client block **before** the strip (round-1 fix), and `_mrl_gate(config,
selection)` builds the product's one head (`rcp_ndcg.data.mrl.MrlHead`) and the gated `k` values: every
`mrl_dims` member in declaration order, or a `mrl_range`'s two endpoints plus the run's selection when it
lies outside them. `_head_matrix` applies that head to both sides per `k`; `_compare_shape` compares each
cut with the ordinary per-vector/per-token cosine gate; `_vector_summary` emits one gate row per `k` beside
the full-width row, and `report._markdown` prints the `k`. The summary records `mrl_gate_version`
(`MRL_GATE_VERSION = 1`), the seam for the gating code's version; `k` is not part of any stored-output key
(there is no stored-output store in this tree yet; the ref-envs lane adds one for full-width outputs).
`n_vectors` stays the base vectors compared and `n_comparisons` counts every per-`k` comparison.
Fixtures: `fixture-embed-mrl` (engine-side `dimensions: 4`) and `fixture-multi-vector-mrl` (client-side
`mrl_dim: 4`), both declaring `mrl_dims: [2, 4, 8]`.

**2. The fake engine mirrors vLLM on both routes (brief item 2).**
`rcp-ndcg-test/src/rcp_ndcg_test/engines.py`: `EngineFacts` gains `is_matryoshka`,
`matryoshka_dimensions` and `embedding_size`; `/v1/embeddings` applies vLLM's three gates in order
(`is_matryoshka`, `1 <= k <= embedding_size`, membership in the declared set) with the engine's own
messages, then slices the full-width vector before the L2 (`_slice_normalised`); the surrogate's
full-width draw is keyed by the model input (`PromptSet.model_key`, the context without `dimensions`), so a
`k` reply is `normalize(full[:k])` of the same draw and a wrong order or set is caught; a replayed
observation stays verbatim. `/pooling` refuses the per-request `dimensions` with vLLM's message and
`param`. The in-tree wiring reads the gate facts from the recipe's `serve.hf_overrides`, and a corpus that
observed several widths takes the widest as the surrogate width.

**3. The request set's MRL stratum and the `(g)` control (brief item 3).**
`observe/requests.py` gains `_mrl_probe_dims`/`_mrl_absent_reason`: a declared head records one bare
`dimensions=k` probe per declared `k` (`mrl:dimensions=k`; every set member, or a range's endpoints and the
run's selection), and a recipe with no head records the `mrl` stratum absent with the reason and keeps the
undeclared-cut `wire:dimensions` probe. `CORPUS_PLAN_VERSION` is 2. `observe/controls.py` gains `(g)
mrl-undeclared`: a wire patch that sends an undeclared `k` the engine's own gate must refuse (a `k` outside
`serve.hf_overrides.matryoshka_dimensions`; the `/pooling` field refusal for a multi-vector recipe; any cut
on a checkpoint without the gate). A range card whose engine declares `is_matryoshka` without a set is
inapplicable, said why: vLLM admits every integer in `1..width` there and the card's floor is client-side.
The test stub engine mirrors the same gates from `--hf-overrides`, so `(g)` is shown caught through the
wave.

## 4. Verification

**Round 1 (two independent fresh-context verifiers, lenses A correctness and B regressions/hygiene, on
`e47d2771`; DeepSeek-V4.1-flash, xhigh; both told the other lens exists).**

- **Lens A: FAIL.** One major: a `mrl_range` recipe's run selection was never gated, because the selection
  was read from the already-stripped client config (dead branch); the observation plan probed `[2, 8, 4]`
  while stage 2 gated `[None, 2, 8]`. One minor: `stage2.n_vectors` counted per-`k` comparisons while the
  over-cap rows counted vectors. Everything else verified (cut-then-L2, unit-norm cut, the three gates, the
  `/pooling` refusal, the full-width strip, the MRL stratum, control `(g)`, the product head's `ConfigError`
  for a `k` wider than the vectors).
- **Lens B: PASS**, with minors: the corpus doc's control list still `(a)-(f)`; the same `n_vectors`
  finding; the stub's range gate uses the fixture width (argued not a defect: the stub's emulated
  checkpoint is `DIM`-wide, exactly vLLM's `embedding_size`); two test gaps (`_mrl_gate`'s range branch,
  `CORPUS_PLAN_VERSION` unpinned). The full suites, lint, types and docs were green, and three mutations
  (head no-op, L2-then-slice, no full-width strip) turned the matching tests red.
- **Fixes** (`1afffef0`): the selection is read from `recipe.client` before the strip and passed to
  `_mrl_gate(config, selection)` (failing-first evidence: the lens-A scratch reproduction showed gate rows
  `[None, 2, 8]` for `mrl_range [2, 8]` + `dimensions: 4`); `n_vectors`/`n_comparisons`; the doc control
  list; the plan-version pin; the projection-branch unit test; the stub width comment.

**Round 2 (one fresh confirmation verifier, lens A+B, on `1afffef0`; DeepSeek-V4.1-flash, xhigh).**

- **PASS**, with three minors, all fixed in `f8e70cea`: the controls docstring still said "six"; the
  `mrl_dim` half of the range fix was unpinned (no multi-vector range test); `n_vectors` counted
  count-mismatch error rows. The verifier independently reproduced the range fix (its own scratch recipe
  gives `[None, 2, 8, 4]`; passing `selection=None` turns the new test red), re-ran the focused files
  (74 passed), the full test package under xdist (925 passed) and lint/types/docs, and confirmed the gate
  for `1afffef0` was green except an intermittent `test-pkg` SIGSEGV (a different test than the earlier
  flake, passing in isolation and under xdist; the same crash class hit unrelated lanes' gates).

## 5. Checks

Last runs on the final tip (`f8e70cea`), via `bin/gate lane/mrl-harness` (slot 2):

```text
ruff-check exit=0   All checks passed
ruff-format exit=0  586 files already formatted
basedpyright exit=0 0 errors, 0 warnings, 0 notes
pytest exit=0       3627 passed, 102 skipped
contract-docs exit=0 304 passed, 55 skipped
mkdocs exit=0       Documentation built
test-pkg exit=0     928 passed, 222 skipped
recipes exit=0      no failure outside the baseline
vllm-pkg exit=0     40 passed
vllm-models exit=0  72 passed, 7 skipped
run_all exit=0      leaderboards 1022/987/35/0; human study 67/67; external judges 82/82
public-names exit=0 clean
clean exit=0        clean
GATE: PASS
```

The earlier gate runs on `e47d2771` and `1afffef0` failed only on an intermittent `test-pkg` SIGSEGV (two
different tests across runs, both passing in isolation and under xdist; the same class was observed on
unrelated lanes' gates); the final gate ran the step green.

## 6. Open questions

- **Range gating is a declared sample.** A `mrl_dims` set gates every member; a `mrl_range` gates its two
  endpoints plus the run's selection (a range cannot be enumerated, and the interior is not silently
  claimed). The observation stratum records exactly the same `k` values. If the owner wants a different
  range sample (e.g. a fixed grid or explicit `--mrl-dims`), that is a small change to `_mrl_gate` and
  `_mrl_probe_dims`.
- **`EngineFacts.embedding_size` is not wired from the checkpoint.** The in-tree test wiring leaves it
  unset, so the emulator's range gate uses the widest width its corpus observed; correct for every current
  corpus, latent for a future MRL corpus that never records the full width. The recipe schema could carry
  the checkpoint's width if the wave needs it.
- **The serve-time `pooler_config.dimensions` path is not modelled** by either fake (the schema allows it;
  no recipe uses it). If a recipe adopts it, the served pass is no longer full width and stage 2's
  full-width comparison would break; out of this lane's brief.
- **Corpora recorded with `CORPUS_PLAN_VERSION` 1** lack the MRL stratum. The staleness gate keys on the
  behaviour fingerprint, not the plan version, so a re-record decision for MRL recipes belongs to the
  mrl-recipes lane's wave.
- **`MRL_GATE_VERSION` is the seam for the ref-envs lane**: the stored reference outputs must stay
  full-width and must not be keyed by `k`; the head's derivation version is what changes the gate's
  meaning.

## CHANGELOG entry

None. No published package's public surface, contract snapshot or JSON Schema changed (the contract
surface explicitly treats `rcp_ndcg_test` as internal); the harness's new `MRL_GATE_VERSION` and the
observation-corpus plan version 2 live in the unpublished package.

## Public surface changes

None to a published surface. Internal to `rcp-ndcg-test`: the new `rcp_ndcg_test.equivalence.MRL_GATE_VERSION`
(and the stage-2 summary's `mrl_gate_version`, `mrl_dim` gate-row field, `n_comparisons`), the new
`mrl:dimensions=<k>` request-set strata, the `(g)` control, the `EngineFacts` MRL fields and the corpus
plan version 2. No CLI, flag, exit code or schema JSON changed.

## Files outside scope

- `docs/concepts/matryoshka.md`, `docs/how-to/validate-a-recipe.md` (the documentation duty; the harness's
  per-`k` gate and the T2 bullet).
- `rcp-ndcg-test/schema/observation-corpus.md` (the package's own corpus-plan/control doc).
- `rcp-ndcg-test/src/rcp_ndcg_test/jobs/run_wave.py` (the `(a)-(g)` docstrings/help text; no behaviour).
- Tests/helpers: `rcp-ndcg-test/tests/stub_engine.py`, `tests/_engines.py`,
  `tests/conformance/test_replay_key.py`, `tests/conformance/test_mutations.py`, `tests/test_recipe.py`.

No product file (`rcp-ndcg`, `rcp-ndcg-core`, `rcp-ndcg-vllm`) changed.

## Docs updated

- `docs/concepts/matryoshka.md` — a new "Gating every declared k" section (the full-width strip, the
  product head per `k`, `MRL_GATE_VERSION`, the range sample, the observation stratum).
- `docs/how-to/validate-a-recipe.md` — the T2 bullet links the per-`k` gate.
- `rcp-ndcg-test/schema/observation-corpus.md` — the corpus plan's MRL stratum and control `(g)`.

Grep commands run over the tree (tracked files): `git grep -n -i "mrl" -- docs` (the matryoshka and
embeddings/late-interaction pages reviewed); `git grep -n "(a)-(f)" -- rcp-ndcg-test docs` (none left);
`git grep -n "six controls" -- rcp-ndcg-test docs` (none); `git grep -n "dimensions=32" -- rcp-ndcg-test
docs` (only the intended non-MRL fallback and its tests); `git grep -n "mrl_gate_version\|MRL_GATE_VERSION"
-- rcp-ndcg-test docs`. `tests/docs` and `mkdocs build --strict` pass (the gate's `contract-docs` and
`mkdocs` steps).

## For the next lanes

- **mrl-recipes**: the declarations will activate the per-`k` gate in the wave; the two fixture recipes
  exercise it on CPU until then. The corpus request plan is version 2 — the MRL stratum records one probe
  per declared `k`, so MRL corpora should be re-recorded (the staleness gate will not see a plan-only
  change).
- **ref-envs**: store full-width reference outputs; never key them by `k`; include `MRL_GATE_VERSION` (with
  the product head's semantics it names) in the gating code's version.
- **The mrl-cards spec** (`handover/specs/mrl-cards.md`, temporary scaffolding) still lists the fake-engine
  and request-set gaps (G3/G4/G8) as open; this lane closes them.
