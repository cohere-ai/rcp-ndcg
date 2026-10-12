# Lane mediafill — the media cases' expected values (the pairs `media` translation, the recorded outcomes)

Owner decision 2026-10-12: the 14 media cases are **in scope for 0.0.1**. The blocker was a tool gap in
`rcp_ndcg_test.case_fill`: it translated only the text of a case's inputs, so the cases'
`documents[].image` / `.video` refs never reached the pairs row's `media` field, and every media-bearing
case was skipped. This lane closes it without a new reference mode (the references' existing
`embed`/`score` modes read `media` — the stage-2 path), retires the pending notes of the shipped cases
(none of which the fill's vocabulary knew), and makes one case's refusal a recorded exception instead of
the recipe's loss.

**Gate:** `timeout 7200 bin/gate lane/mediafill` on the merged tip `ceb12da2` (slot 2): attempt 1 hit
the documented test-package SIGSEGV environment flake
(`test_wave_corpus.py::test_the_corpus_is_keyed_by_the_engine_version_the_pod_reports`, exit 139 --
`handover/reports/07a-qa-prep.md` section 2), attempt 2 **GATE: PASS** (every step exit=0; the pre-merge
commit `96046f4f` also passed on its first attempt).

## 1. Status — per-item disposition

| Item (brief) | Disposition |
|---|---|
| 1. The pairs `media` translation | **Done.** `_plan_rows` builds each document's entries from the case's own `media/<file>` bytes and adds `media: {query: [], documents: [[entry, ...], ...]}` when the case carries media; `_media_entry` reads the file, the product's suffix→MIME table, `image_dimensions` (images) and `probe_video_header` (containers). The entry shape is the media set's own (`image_entry`/`video_entry`, extended with `payload`/`mime`), so one encoder serves the synthetic media set and the cases. |
| 2. No new reference mode | **Done.** The fill runs the references' existing `embed`/`score` modes; the rows are the shape the media stage sends (pinned by the construction test). No reference file changed. |
| 3. Stop skipping media cases | **Done.** The media refusal is removed from `_unfillable_reason`; the per-case skip stays for `kind: none` and a wrong `expected.kind`. |
| 4. The notes gap | **Done, and wider than the brief assumed.** The QA correction was right: **0 of the 41 pending notes matched the three retirement phrases**, so every fill (tonight's 26 included) left a note contradicting the values. The vocabulary now covers the shipped phrasings (`expected.values stays null until …`, `GPU wave fills values;`, `expected.values are filled by the GPU wave (…)`, `The wave fills values from the reference …`, `origin reference, filled by the GPU wave.`), and a note carrying none gets the fill's provenance appended; a corpus-wide test runs every generated case's note through the update and refuses a surviving pending claim. |
| 5. Kinds and tolerance | **Done.** Each case's own tolerance is carried unchanged (the fill edits `values`, `status` and the note only; tests pin 0.002 for the qwen3-vl-embedding media cases and 0.01 for the qwen3-vl-reranker ones). Kinds unchanged (`similarity_matrix` for the embedders, `scores` for the reranker). |
| QA (a) per-case isolation | **Done.** One subprocess over every fillable case's rows (one model load) is the fast path; when that run fails, every case is retried on its own, so the refusal lands on the case whose rows carry it. A per-case failure is an **exception**: the reference's own reason in the report and in the case's note, the case left `pending_gpu`; the recipe's other cases fill. A budget overrun is not retried (the recipe's fill fails loudly). |
| QA (b) the video case | **Recorded exception — the frame-bounds limit is real.** `qwen3-vl-embedding-2b/video-short-single`'s clip is 448x256, 4 frames, 2 fps; the recipe's whole-clip pin is `videos_kwargs {min_pixels: 4096, max_pixels: 7864320}`, so the per-frame share at 4 frames is `7864320 // 4 = 1966080` px — above qwen-vl-utils' per-frame ceiling `768 * 32 * 32 = 786432`. `frame_bounds` refuses it (`SystemExit: the pin's per-frame share 1966080 exceeds the card loader's per-frame ceiling 786432`); a clip of **≥10 frames** at the pin would fit. The lane records it (the design doc's own fallback: "if the reference refuses the declared pin, the case is recorded with its reason"); regenerating the clip is a case-content change (new asset hash, new notes, media-stage re-checks) that belongs to the cases lanes, not here. The reference's own input path accepts the container (`card_inputs`: fps 2.0, max_frames 4, the bytes written for the card's loader) — the refusal is the pin, not the container. |
| QA (c) `query-cap-1024` | **The one structural non-fill, recorded.** It is `kind: none` (`values: null` + `tolerance: null`); the case format requires `status: pending_gpu` with null values, so it can never be filled. The fill skips it with the reason `kind: none carries no values by design`, its file untouched, in the report's `skipped` list — never a failure. |

## 2. Commits

| commit | what |
|---|---|
| `96046f4f` | `case_fill` reads a media case's `media/` files into the pairs row's `media` field (one encoder: `media_set.image_entry`/`video_entry` take a payload and a MIME type); the retirement vocabulary covers the shipped pending notes; a refused case is retried per case and recorded as an exception (report + note); the casefill job uploads exception YAMLs; tests, docs and CHANGELOG |

Base: `00038a8b`.  The current `rfc-0001` tip `4eb2d742` (lane r9: the eager backend, the topk reference
pin and the corpus MRL width check) is merged at `fa97034c` before the final gate; the merge was clean
(one auto-merged `CHANGELOG.md`, no semantic drift: r9 touched the recipes' pins and the corpus width
check, this lane the case fill).

## 3. Red-first evidence (quoted)

The three tests written before the implementation (3 failed, 6 passed):

```
E  rcp_ndcg_test.errors.CaseError: case qwen3-vl-embedding-2b/image-short-single: a document carries no text
E  AssertionError: assert ['gen-image-l...xed-modality'] == ['gen-kind-none']
E    At index 0 diff: 'gen-image-long-over' != 'gen-kind-none'
E  assert False
E   +  where False = all(<generator object test_a_filled_case_records_the_fill_when_its_note_has_no_pending_sentence...)
FAILED .../test_a_media_case_plans_the_case_files_into_the_pairs_media_field
FAILED .../test_a_media_case_is_no_longer_skipped_and_a_kind_none_case_still_is
FAILED .../test_a_filled_case_records_the_fill_when_its_note_has_no_pending_sentence
```

The retirement-vocabulary test, before the shipped phrasings were added:

```
E  AssertionError: ('image-budget-max.yaml', 'pending_gpu', 'Generated by make_long_inputs.py (...) ...
   status pending_gpu). tolerance abs 0.02 is the initial gate: ...')
```

The per-case isolation, against the pre-change `fill_cases` (one refused row raised out of the fill and
the recipe lost every case):

```
RAISED HarnessError: the reference subprocess (score) failed with exit code 1: the reference refuses
    this row: POISON is not an input this reference can score
pending files still: ['gen-long-over.yaml', 'gen-long-under.yaml', 'gen-mixed-lengths.yaml', 'gen-short.yaml']
```

With the fix, `test_one_cases_refusal_is_recorded_and_leaves_the_other_cases_filling` shows three cases
filling, the refused one recorded as an exception (reason in the report and the note), and a second run
recording the same exception without appending the reason twice.

The video case's refusal, proven on the shipped container (no model weights; the reference's pure
functions):

```
test_the_shipped_video_case_reaches_the_card_route_and_its_pin_refuses_it
  entry: 448x256, 4 frames, 2 fps (off media/clip.mp4's header)
  card_inputs: fps 2.0, max_frames 4, the container bytes written for the card's loader
  frame_bounds(4, (4096, 7864320)): SystemExit "the pin's per-frame share 1966080 exceeds the card
  loader's per-frame ceiling 786432"
```

## 4. The exact node command and the acceptance counts

The RC must be **rebuilt** (`rc_build.sh` stages `rcp-ndcg-test/cases` as `<RC>/cases/`; the wheel drops
`cases/`), then the casefill job runs the four cuda recipes:

```bash
bin/e2-run build <worktree> <rc>                       # or rc_build.sh <rc>
RCP_SUBMIT_DIR=<dir> submit.sh --script casefill --priority dev-high \
    gs://YOUR-BUCKET/<rc> gs://YOUR-BUCKET/waves-<rc> casefill
```

The node command `submit.sh` renders:

```
/bin/bash /etc/rcp/files/hftoken/hf_token_env.sh /bin/bash \
    /etc/rcp/files/casefill/casefill.sh <RC_STAGE_URI> <OUT_PREFIX>/casefill
```

which runs, in order:

```bash
bash /etc/rcp/files/bootstrap/bootstrap.sh envs <RC_STAGE_URI> --state "$STATE" \
    --recipes "$STAGE/recipes" --wave-list "$STATE/case-fill-wave.txt"   # the four ids
"$STATE/client" python -m rcp_ndcg_test.case_fill --recipe <id> \
    --reference-python "$STATE/reference/<family>/bin/python" --device cuda \
    --filled-at <date> --budget-s 7200 --cases-root "$STATE/cases"       # per recipe, continue-on-error
```

and uploads `<OUT_PREFIX>/casefill/cases/<recipe>/*.yaml` (the filled cases **and** the refused case's
recorded exception) and `<OUT_PREFIX>/casefill/casefill-report.json`.

**Expected on the committed corpus** (measured offline: `load_cases` + the fill's own partition over the
four recipes):

| Recipe | Pending | Fillable (media) | Expected outcome |
|---|---|---|---|
| topk-embed-v1-small | 15 | 14 (6) | 14 fills; `query-cap-1024` skipped (`kind: none`) |
| qwen3-embedding-0.6b | 8 | 8 (0) | 8 fills |
| qwen3-vl-embedding-2b | 9 | 9 (4) | 8 fills + `video-short-single` a recorded exception |
| qwen3-vl-reranker-2b | 9 | 9 (4) | 9 fills |
| **total** | **41** | **40 (14)** | **39 fills, 1 recorded exception, 1 structural skip** |

Acceptance: `casefill-report.json` `passed: true` (no recipe failed), the four per-recipe reports carry
39 `filled`, the video case's `exceptions[0].reason` names the per-frame ceiling, and
`grep -rl pending_gpu rcp-ndcg-test/cases` shows **exactly two** files under the four recipes — the video
case (exception, reason in its note) and `query-cap-1024` (structural non-fill, reason in the report) —
with **zero of the 14 media cases silently skipped**. The notes of all 39 filled cases carry no
`pending_gpu`, no `stays null` and no `GPU wave fills` phrasing (the corpus test pins the vocabulary).

## 5. Verification

- The failing tests first (section 3), then green: `rcp-ndcg-test/tests/test_case_fill.py` (12 passed),
  `test_jobs_scripts.py` (88 passed with `test_case_fill.py`), the family test
  `tests/recipes/test_qwen3_vl_embedding.py`.
- The whole test package through `heavy`: **1167 passed, 229 skipped** (pre-merge; the merged tree's
gate step: **1170 passed, 229 skipped**, the r9 merge adding three tests).
- The root suite through `heavy`: **4145 passed, 155 skipped**.
- `tests/test_cases_package.py tests/contract tests/docs`: **416 passed, 108 skipped**.
- `ruff format --check` clean, `ruff check` clean, `basedpyright` 0 errors.
- `bin/gate lane/mediafill`: `96046f4f` (pre-merge) passed on its first attempt; the merged tip `ceb12da2`
  passed on attempt 2 after attempt 1 hit the documented test-package SIGSEGV flake (both SUMMARYs quoted
  in the lane workspace's gate logs).

## 6. Open questions / residual risks

1. **The video case stays a recorded exception.** The reference's video route cannot express a container
   whose realised frame count is below 10 at the recipe's whole-clip pin; the case's clip is 4 frames.
   Regenerating the clip (`media/make_media.py`, ≥10 frames) would make it fillable but changes a
   committed asset and the case's own geometry notes — a cases-lane decision, recorded here rather than
   taken. The video parity stays covered by the media stage and stage 2's text rows.
2. **Tonight's 26 uploaded YAMLs are superseded.** Their notes carry the stale pending phrasings (the
   fill's vocabulary did not know them), so the acceptance run must re-fill from the committed pending
   tree and the operator must commit *that* upload, not tonight's.
3. **A reference-wide environment failure reads as per-case exceptions.** With per-case isolation, an
   out-of-memory or a network failure of every run marks every case an exception (each with the same
   reason) instead of failing the recipe; the report makes it visible (every case listed) and the
   operator reviews the uploads before committing. A budget overrun is the exception to this: it is
   raised as the recipe's failure.
4. **The per-case retry costs reference runs.** When the batch run fails, every case is retried on its
   own (one model load each): the acceptance run pays ~10 loads for qwen3-vl-embedding instead of 1.
   Within the job's `--budget-s 7200` per recipe (the budget is shared across the recipe's runs now).
5. `qwen3-vl-embedding-2b`'s reference loads the model before it reads the rows, so the video case's
   refusal wastes one load; a pre-check in the harness would duplicate the reference's rule (R30), so it
   is not done.

## Docs updated

- `rcp-ndcg-test/README.md` — the fill paragraph (media rows, the note update, exceptions).
- `docs/reference/rcp-ndcg-test.md` — the case-fill bullet.
- `docs/how-to/release-candidates.md` — the casefill-job paragraph (skips, media, exceptions).
- `CHANGELOG.md` — the case-fill bullet rewritten (media fillable, note vocabulary, exceptions) and a new
  bullet for the media set's `payload`/`mime` builders.
- Greps run: `git grep -n "media-value fill\|text-only reference\|case_fill\|pending_gpu\|media_set"` over
  `docs/ README.md REPRODUCIBILITY.md skills/ examples/ experiments/ handover/ CHANGELOG.md
  rcp-ndcg-test/`; every remaining hit describes current behaviour.

## CHANGELOG entry

Under `### Changed` (0.0.1): the existing `**The pending reference cases are filled by
rcp_ndcg_test.case_fill**` bullet now reads: one reference subprocess over every pending case's inputs;
the outputs shaped into `expected.kind`; a line-wise YAML edit with the fill recorded in the note (a known
pending sentence retired, the fill's provenance appended where the note carries none); a media-bearing
case is fillable (its `media/` files ride the pairs row's `media` field in the media set's own entry
shape, which the references' `embed`/`score` modes read exactly as stage 2's media rows); a case the
reference refuses (a container it cannot sample at the recipe's declared video pin) is retried on its own
and recorded as an exception (reason in the report and in the case's note, the case left `pending_gpu`);
a pending case the mode cannot fill is skipped per case with its reason (`kind: none`, another stage's
kind). Plus: `**The media set's entry builders encode a caller's own bytes**` — `image_entry` and
`video_entry` take a `payload` and a MIME type, one `data:`-URI encoder for the synthetic media set and
the cases.

## Public surface changes

None. `rcp_ndcg_test` is unpublished and internal; no `__all__` of a public module changed, no CLI
command or flag of the product, no exit code, no schema. The casefill job's embedded fill report gains an
`exceptions` list (an internal job artifact, `rcp-ndcg.casefill-report.v1` is unchanged as a schema: the
per-recipe fill report is merged, not schema-validated).

## Files outside scope

None outside the report files below.  Two historical lane reports describe the media cases as permanent
skips and are now false in the release state; each got a one-line **superseded by lane mediafill** note
(no history rewritten): `handover/reports/runplumb.md` (the case-fill job item and open question 1) and
`handover/reports/confprep.md` (section 4.3's remaining-fills paragraph).  (The design note in the lanes
workspace is amended with the video pin's limit; no other repository file changed.)

## For the next lanes

- **Operator:** rebuild the RC, run the casefill job (section 4), review the uploaded YAMLs (39 filled +
  the video case's exception note), and commit them. After the commit `grep -rl pending_gpu` over the four
  recipes shows exactly `video-short-single.yaml` and `query-cap-1024.yaml`, both with recorded reasons.
- The conformance runner then compares the engine against the 39 new values (`origin: reference`); the
  video case and `query-cap-1024` stay skips (null values), never passes.
- `test_casefill_pins_the_four_recipes_to_the_catalog_and_the_cases` now counts the four recipes'
  **generated corpus** (41 cases: 40 values-bearing, 14 media-bearing, 1 `kind: none`) instead of the
  pending set, so committing the filled YAMLs does not break it.
- A future cases lane that regenerates the video clip at ≥10 frames should flip that case from a recorded
  exception to a fill and re-run the node job for it.
