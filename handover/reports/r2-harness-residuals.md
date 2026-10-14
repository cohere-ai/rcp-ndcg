# Lane r2-harness-residuals — the round-2 harness residuals

**Lane:** `lane/r2-harness-residuals` (worktree ).
**Base:** `rfc-0001` at `afc76650`. **HEAD:** `4267d932` (5 commits, each red-first).
**Scope:** `release-code-fixes-ds41.md`'s harness residuals (items 1, 2, 3, 4 of the lane brief) and the
fleet-verdict gap check against the HV lane's merges (`c807701b` + its commits) — item 5. The topk gate,
the dtype and every bound are untouched; no flip was run.

## Per-item disposition

### 1. `engine_versions` published from failed rows — FIXED (`f242e769`)

The D1 fix (`ad5c9d…`, `ad5c1dc4`) gated `fingerprints` on the corpus step passing but built
`engine_versions` from every row (`run_wave.py:2060`), so a failed row's version entered the
`--changed-since` index and `protocol_due` stopped naming the engine version the release never recorded
(the ds41 residual's exact wording). `_wave_document` now builds both index maps from the recorded rows;
the values stay on the row for reporting.

- Red-first: extended
  `test_a_failed_corpus_step_is_not_published_to_the_changed_since_index` with
  `first["engine_versions"] == {}`, the recovery wave's
  `changes == {"fixture-embed": ["behaviour_fingerprint", "engine_version"]}` and
  `protocol_due == ["test-stub"]`; failed on the unfixed code at the empty-map assertion
  (`{'fixture-embed': 'test-stub'} == {}`).
- Note: the old test's docstring already claimed "or its engine version" without pinning it — the pin is
  now real.

### 2. The index ignored a per-recipe UPLOAD failure — FIXED (`53ab80de`)

A recipe whose corpus step passed but whose per-recipe upload failed (`run_wave.py:422`, folded at
`:443-450`) kept its key in the index: the destination never received the corpus, so a
`--changed-since` recovery from that (local) `wave.json` skipped it as recorded and the recovery wave
could report PASS over a corpus that never left the node. B1 already failed the wave; now the upload
failure also disqualifies the row from both index maps (`recorded` requires the corpus step passed AND
the row's upload record, when present, `ok`), so a failed upload behaves like a failed corpus step.

- Red-first: new `test_a_failed_recipe_upload_is_not_published_to_the_changed_since_index` — wave 1
  records with a failing object store (corpus step `passed`, `upload_failures` named, verdict `failed`),
  asserts `fingerprints == {}` and `engine_versions == {}`, then the recovery wave (upload working)
  re-records instead of skipping; failed on the unfixed code at the fingerprints assertion.
- Rows without an upload record (no `--upload`) index exactly as before.

### 3. Harness D8's stale comment — FIXED, comment side (`456b6b39`)

`stages.py`'s width-mismatch comment claimed "a named gate failure with both widths, never a crash",
while a served matrix narrower than a declared `k` raises the product head's
`ConfigError("mrl_dim … is wider than the vectors")` out of `_head_matrix` before the row check. The
behaviour is deliberate and already pinned (`test_stage2_uses_the_product_head_for_a_k_wider_than_the_vectors`),
so the comment was the wrong side; it now states both truths: a mismatch that survives the head is a
named gate row, and an unsupportable declared `k` is a typed configuration refusal, not a gate row.

- Red-first doc-contract test: `test_the_width_mismatch_comment_states_the_refusal_that_precedes_it`
  (the old "never a crash" claim must be gone and the refusal named); failed while the old comment stood.

### 4. The upload verification's storage fallback — TEST ADDED (`ad0bd0ce`)

The ds41 reviewer cleared the code (`_verify_upload` reads back through the product's storage, the same
module `_upload_storage` writes with) but the corruption tests all drove the CLI transfer; the python
fallback path had no test. New
`test_the_hash_read_back_covers_the_python_fallback`: `_upload_cli` unavailable, the fallback transfer
lands same-size corrupted bytes → the attempt fails with `content mismatch`; the clean fallback upload
verifies by reading the bytes back.

- Sensitivity (the reviewer's own method): removing the destination read-back from `_verify_upload` in
  vitro turns the test red; the shipped code is untouched.

### 5. The fleet-vs-HV-lane gap — one item closed, two design debts reported

Audit of `final-harness-ds41.md` and `final-harness-glm53.md` against the HV lane's 17 fixes
(`c807701b`): ds41 D1-D9 and glm53 D1-D4 are closed by `ad5c1dc4`, `930b5e4a`, `c1fcd193`, `9f356e33`,
`c272e161`, `6b852165`, `408c24c1`, `9ff9e807`, `628595f4`, `e5f5a99c`, `e4a1ff58`; D8 and the D1
residuals are this lane's items 1-3. What remained:

- **ds41 D10 (INFO) — FIXED (`4267d932`).** `fingerprint.py` classified `max_images`/`max_videos`
  `request` while `JudgeConfig.IDENTITY_ROLES` declares both RUNTIME; the classification comment now says
  why the over-keying is deliberate and conservative (an extra input forces a needless re-record, never a
  stale skip; judges record no corpus). Red-first:
  `test_the_media_caps_over_key_the_judge_role_deliberately` pins both sides of the disagreement and the
  comment's why. (The existing `test_every_client_field_role_agrees_with_the_identity_and_the_fingerprint`
  never saw this because the retrieval configs' CONTENT declaration shadows the judge's RUNTIME one.)
- **glm53 risk 2 (neutral `not_run` checks with an engine up but an empty measurement) — NOT closed.**
  The three sites are real (`_gate_pass`'s `passed=None` neutrality,
  `_engine_prompt_tokens_check`'s with-engine `not_run`, `stage_media`'s `passed is not False`), but the
  neutrality is declared in the code's own docstrings ("``not_run`` is neutral (it does not gate the
  stage)"), the reviewer framed it as a recommendation, and the named mitigations exist (`_consistency`
  and `_numbers` re-derive the prompt-token facts from the same replies). Closing it changes gate
  semantics — including the legitimate listwise-skip neutrality — which is outside this lane's named
  defect lists and needs an owner decision.
- **glm53 risk 3 (the version-locked status table) — NOT closed.** `expected_status` answers `None` for
  any version but 0.31.0 by declared design (the table is the *measured* shakedown result and
  `protocol_due` flags new versions for re-recording); the reviewer's "per-version measured: false row"
  is a reporting improvement, not a named defect.

## Discovery: the B5 corpus-key tests rode the D1 hole (repaired in `53ab80de`)

With the pod version pinned to `0.31.0`, `test_the_corpus_is_keyed_by_the_engine_version_the_pod_reports`
and `test_a_digest_pinned_recipe_records_the_pod_version_not_the_digest` drive a corpus step that
**fails**: the stub refuses unknown fields with 400 where the measured 0.31.0 table expects 200, so
`statuses_as_expected` fails and the step fails. Both tests were green anyway because the ungated
`engine_versions` map published the failed step's version — a live instance of the D1 hole. They now read
the row's `engine_version` (the reporting value), assert the empty index map, and assert the step state.
The stub's stricter refusal itself is untouched (deliberate, and the test comment says so).

## Validation (all `uv run --no-sync` on the branch head)

- `ruff format --check .` + `ruff check .` — clean; `basedpyright` — 0 errors.
- `tests/contract` — 17 passed (no public-surface movement; `rcp-ndcg-test` is never published and the
  wave document is not snapshotted, so no snapshot/CHANGELOG obligation, matching the HV lane's merge).
- Wave/index/upload: `test_wave_corpus` (15), `test_wave_upload` (8), `test_wavegroups`,
  `test_reference_store`, `test_wave_plugin_wheel`, `test_record_and_wave`, `test_upload_records` — green.
- Harness gates: `test_mrl_gate`, `test_fingerprint`, `test_equivalence`, `test_over_cap`,
  `test_observe_corpus`, `test_media`, `test_controls`, `test_conformance`, `test_changes`,
  `test_jobs_scripts`, `test_quality` — green (352 targeted tests total across the runs above).
- Red-first evidence per fix is in the commits; both index reds and both comment reds reproduced.

## What I could not reproduce / residuals

- Nothing in my items failed to reproduce; both review residuals reproduced exactly as written.
- Pre-existing, not mine: a plain (non-record) wave's `wave.json` now publishes empty index maps (no
  consumer in-tree reads them outside the record flow; the direction is fail-safe — an index without keys
  re-records everything).
- glm53 risks 2 and 3 remain open design debts (see item 5) — surfaced for an owner decision, not decided
  here.
- The GPU-only paths (real vLLM corpora, the real status-table passes) are out of reach offline; the
  changed-since gating is pinned against the stub corpus flow.
