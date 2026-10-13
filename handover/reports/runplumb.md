# Lane runplumb — the plumbing the final parity re-run needs

**Lane:** `lane/runplumb` (worktree outside the operator's checkout). **Base:** `rfc-0001` at `615cd225` (the
branch point *is* the current `rfc-0001` tip: it is an ancestor of every commit here, so there was nothing to
merge). **Gate:** `timeout 7200 bin/gate lane/runplumb` on `de281253` (slot 2), first attempt — **GATE:
PASS**, every step exit 0. No GPU work; nothing pushed; `rfc-0001` was not merged into. **Read first:** the
lane brief (plan v2 plus the two adversarial reviews).

## 1. Status — per-item disposition

1. **`RCP_RECORD_CORPUS` → `--record-corpus` (done).** `bootstrap.sh`'s wave mode now renders its `run_wave`
   argv through one testable helper (`wave_command`): `RCP_RECORD_CORPUS=1` appends `--record-corpus`; unset,
   the argv is byte-identical to before (asserted word for word). `submit.sh` forwards
   `env.RCP_RECORD_CORPUS` into the job spec only when the operator set it, the way `env.RCP_IMAGE` is
   forwarded, and its header documents the variable. Wave mode's argv had no test before this.
2. **The case-fill job (done, with one supervisor-approved correction).** `rc_build.sh` stages
   `rcp-ndcg-test/cases` as `<RC>/cases/` (one home, stray-root refusal, none-is-a-warning, all tested);
   `jobs/casefill.sh` builds the four cuda families' reference environments through `bootstrap.sh envs`
   (a wave list of the four ids it writes), runs `rcp_ndcg_test.case_fill --device cuda` per recipe over a
   copy of the staged cases, continues past a recipe's failure, and uploads the edited YAMLs
   (`<OUT>/cases/<recipe>/<case>.yaml`) plus `casefill-report.json` for a reviewed commit; `submit.sh
   --script casefill` mounts it beside `bootstrap.sh`. The correction the brief's 40-value arithmetic
   needed, approved by the owner during the lane: `case_fill` used to refuse a **whole recipe** when any
   pending case carried media (measured: 3 of the 4 recipes refused before the reference ran), so the fill
   now **skips an un-fillable pending case per case** with the reason recorded in the report's new
   `skipped` list, and the recipe's text cases still fill. With the committed cases that is **26 fills**
   (topk 8, qwen3-embedding 8, qwen3-vl-embedding 5, qwen3-vl-reranker 5) and **15 recorded skips** (14
   media-bearing + topk's `query-cap-1024`, `kind: none`) — the 14 media cases stay `pending_gpu` with the
   recorded reason (the reference's `media` mode reports input facts, not model outputs; a media-value fill
   is a follow-up, **not** built here).  **Superseded by lane mediafill**: the media cases fill now (the
   pairs row carries the case's `media/` files, built by the media set's own entry builders; the
   qwen3-vl-embedding video case is a recorded exception at the recipe's video pin), and only topk's
   `kind: none` case stays the structural non-fill — see `handover/reports/mediafill.md`.  The mislabelled
   `qwen3-vl-reranker-2b/gen-text-short-uniform` case
   (`kind: similarity_matrix` on a rerank recipe) is corrected to `scores`, and a test pins every shipped
   case's `expected.kind` to its role's stage shape.
3. **The changed-only recording index (verified; documentation, no code change).** The committed repository
   corpus index is **not** the shape `run_wave --changed-since` reads. `rcp-ndcg.corpus-index/1` carries
   `corpora` (keyed `<recipe>/<fingerprint>`), `engine` and `schema`; `observe.corpus.changed_since` reads
   top-level `fingerprints` and `engine_versions` maps (a previous **recording** wave's `wave.json`). Fed the
   committed index, the selection reports **every** recipe changed (probe: `octen-embedding-8b` — which is
   unchanged — and `jina-embeddings-v5-text-small` both `["behaviour_fingerprint", "engine_version"]`);
   fed a `wave.json`-shaped index it reports the right pair (`octen` unchanged, `jina` changed on
   `behaviour_fingerprint`). Consequence for the recording pass: passing the committed index does not
   select changed-only — it re-records every recipe in the wave's list (safe, not silent: nothing is
   skipped). The exact working invocations are in section 4; the false "or corpus index" phrasing is fixed
   in `run_wave`'s docstring and `--changed-since` help, and the consequence is in
   `handover/reports/confprep.md`.
4. **The two doc corrections (done).** `run_wave`'s docstring and `--changed-since` help now state the real
   corpus path (`<out>/observations/vllm-<version>/<recipe>/<fingerprint>/<stamp>/`, `observe/corpus.py`'s
   `corpus_path`) and the index the flag reads. `handover/reports/confprep.md` steps 1–2 use the
   `observations/...` tree (and its section 3.1, step 0, 4.3 and open questions 6.1/6.3 were updated where
   this lane's changes made them false: the recording switch exists now, the fill skips per case, the
   reranker's kind is fixed).

## 2. Commits

| commit | what |
|---|---|
| `3eeb1299` | `case_fill` skips an un-fillable pending case per case (the media reason recorded); the reranker's case kind corrected; the role↔kind test; the docs and CHANGELOG |
| `24e2fe35` | `RCP_RECORD_CORPUS=1` → `--record-corpus` (the `wave_command` helper, submit forwarding, the wave-mode argv test) |
| `28ad5fbc` | the case-fill job: `cases/` staging, `casefill.sh`, `submit.sh --script casefill`, the tests, the release-candidates page |
| `463c39e8` | the corpus-path and changed-only-index corrections (`run_wave`, `confprep.md`) |
| `de281253` | the casefill end-to-end test fails a middle recipe (a mutation check found the first version could not catch a break-on-failure) |
| the report commit | this report |

No co-author lines, no attribution, nothing pushed.

## 3. Red-first evidence (quoted)

- **`case_fill` per-case skip** — `uv run --no-sync pytest rcp-ndcg-test/tests/test_case_fill.py -q`
  before the change: `3 failed, 3 passed`; the lines:
  - `E   rcp_ndcg_test.errors.CaseError: case qwen3-vl-reranker-2b/gen-image-long-over: document(s) d1 carry media, and the reference's score mode reads text only; a media case's values come from the media stage, not the fill`
  - `E   rcp_ndcg_test.errors.CaseError: case fixture-rerank-pointwise/gen-similarity: expected.kind 'similarity_matrix' is not what the score mode produces ('scores'); the fill does not guess another stage's values`
  - `E   KeyError: 'skipped'`
- **The reranker's case kind** — `pytest .../test_cases.py::test_every_shipped_case_kind_matches_its_roles_stage_shape -q`:
  `E   AssertionError: [('qwen3-vl-reranker-2b/gen-text-short-uniform', 'rerank', 'similarity_matrix')]`.
- **The wave-mode argv** — `pytest .../test_jobs_scripts.py::test_bootstrap_wave_command_adds_record_corpus_only_when_asked -q`:
  `E   AssertionError: bash: line 1: wave_command: command not found` / `assert 127 == 0`.
- **submit/casefill/rc_build** — the five new tests before their changes: `submit.sh: --script must be
  bootstrap, wave0 or e2e, got casefill` (exit 2); the record-corpus argv missing
  `env.RCP_RECORD_CORPUS=1`; `casefill.sh: No such file or directory` (exit 127); `stage_cases: command not
  found` (exit 127); `FileNotFoundError: .../stage/cases`.

**Mutation evidence** (each on the committed tree, then reverted): returning `None` from the media branch of
`_unfillable_reason` fails the skip test with
`CaseError: case qwen3-vl-reranker-2b/gen-image-short-single: a document carries no text`; deleting the
`RCP_RECORD_CORPUS` block fails the argv test at `At index 15 diff: '--pairs-dir' != '--record-corpus'`;
`break`-ing after a failed fill fails the end-to-end test at `assert 3 == 4` (the fourth recipe did not
run) — this last mutation is why the failing stub recipe is the third, not the last (`de281253`).

## 4. The exact node commands the passes will use

**Verification pass (all 44; plan v2 A)** — operator side, from the checkout whose scripts are mounted:

```bash
RCP_MAX_JOBS=8 bin/e2-run waves <worktree> <rc> r1 r2 r3 r4 r5 judges-j1 judges-j2
# (r1 groups into the v0.31.0 job and the nightly job; the judges ride their own chain)
```

The node command `submit.sh` renders per job (KJOBS=echo prints it), with `RCP_RECORD_CORPUS` unset so the
argv is the one every wave has always run:

```
/bin/bash /etc/rcp/files/hftoken/hf_token_env.sh /bin/bash /etc/rcp/files/bootstrap/bootstrap.sh \
    wave <RC_STAGE_URI> <OUT_URI> --wave <wave> [--wave-list /etc/rcp/files/wavelist/<wave>.<image-slug>.txt]
```

which `bootstrap.sh` execs as (the `wave_command` helper's output):

```
"$STATE/client" python -m rcp_ndcg_test.jobs.run_wave \
    --recipes "@$WAVE_LIST_FILE" --recipes-root "$RECIPES_ROOT" --gpus "$gpus" \
    --out "$STATE/wave" --upload "$OUT_URI" --record \
    --failed-plugins "$STATE/plugin-failures.txt" --reference-root "$STATE/reference" \
    [--plugin-wheel "$STAGE_DIR/wheelhouse/rcp_ndcg_vllm-<version>-*.whl"] \
    [--pairs-dir "$STAGE_DIR/pairs"]
```

**Recording pass (plan v2 B)** — the same submission with the switch set; the node argv gains exactly
`--record-corpus` after `--reference-root ...`:

```bash
RCP_RECORD_CORPUS=1 RCP_MAX_JOBS=<N> bin/e2-run waves <worktree> <rc> <the same wave names>
```

- *Changed-only, through the job path:* no `--changed-since` forwarding exists today, so the wave **list**
  is the selection: run the comparison against the committed corpora and give its changed ids as the list —

  ```bash
  uv run --no-sync python -m rcp_ndcg_test.changes changed \
      --recipes-root rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes \
      --corpora-root rcp-ndcg-test/corpora/vllm-0.31.0
  # keep every id whose "state" is not "unchanged"; stage that list as the wave's list and submit it with
  # RCP_RECORD_CORPUS=1.  The same lists re-record everything they name (the committed index cannot select).
  ```
- *Changed-only, by hand (the docs' T1 path), when a previous recording wave's `wave.json` exists:*

  ```bash
  "$STATE/client" python -m rcp_ndcg_test.jobs.run_wave \
      --recipes "@$WAVE_LIST_FILE" --recipes-root "$RECIPES_ROOT" --gpus "$gpus" \
      --out "$STATE/wave" --upload "$OUT_URI" --record --record-corpus \
      --changed-since "<a previous recording wave's wave.json>" \
      --failed-plugins "$STATE/plugin-failures.txt" --reference-root "$STATE/reference" \
      --pairs-dir "$STAGE_DIR/pairs"
  # --changed-since needs the pod's engine version before engines start: RCP_ENGINE_PYTHON (bootstrap exports it)
  ```

  Corpora land at `<out>/observations/vllm-<version>/<recipe>/<fingerprint>/<stamp>/` and upload with the
  wave's final whole-tree copy (`<OUT_URI>/observations/...`).

**Case-fill pass (plan v2 C)** — one job; the RC must be **rebuilt** (the cases staging is new):

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

and uploads `<OUT_PREFIX>/casefill/cases/<recipe>/*.yaml` and `<OUT_PREFIX>/casefill/casefill-report.json`.
Expected: 26 filled YAMLs, 15 recorded skips (14 media + 1 `kind: none`); the operator reviews the diff and
copies the YAMLs into the repository.

## 5. Verification

- The failing tests first, per item (section 3), then green: `rcp-ndcg-test/tests/test_case_fill.py`,
  `test_cases.py`, `test_jobs_scripts.py` — 158 passed.
- The whole test package through `heavy`: `1160 passed, 229 skipped`; the root suite through `heavy`:
  `4145 passed, 155 skipped`; `tests/contract tests/docs`: `413 passed, 107 skipped`; `mkdocs build
  --strict` clean; `ruff format --check`/`ruff check` clean; `basedpyright` 0 errors.
- The changed-only probe (section 1 item 3) and the per-recipe skip counts probe: topk 15 pending / 8
  fillable / 6 media + 1 `kind: none` skipped; qwen3-embedding 8/8/0; qwen3-vl-embedding 9/5/4;
  qwen3-vl-reranker 9/5/4.
- **Gate:** `timeout 7200 bin/gate lane/runplumb` on `de281253`, slot 2, first attempt — `GATE: PASS`,
  every step exit 0:

```
rev lane/runplumb = de281253 (slot 2)
ruff-check exit=0  basedpyright exit=0  pytest exit=0 (4145 passed, 155 skipped)
contract-docs exit=0 (413 passed, 107 skipped)  mkdocs exit=0
test-pkg exit=0 (1160 passed, 229 skipped)  recipes exit=0
vllm-pkg exit=0 (50 passed)  vllm-models exit=0 (102 passed, 8 skipped)
run_all exit=0  public-names exit=0  clean exit=0
GATE: PASS
```

## 6. Open questions / residual risks

1. **The 14 media-bearing pending cases stay `pending_gpu`** with the recorded reason (the reference's
   `media` mode reports input facts, not model outputs). This is the owner-visible residual of the case
   fills: 26 of the 40 cuda values land; a media-value fill needs a reference mode that returns model
   outputs for media, which this lane did not build. The conformance runner treats `values: null` as a skip,
   never a pass.  **Superseded by lane mediafill** (owner decision 2026-10-12: the media cases are in
   scope): the references' existing `embed`/`score` modes read the pairs row's `media` field, the fill
   builds it from the case's `media/` files, and the media cases fill — the qwen3-vl-embedding video case
   excepted, recorded with the recipe's video pin's per-frame-ceiling reason.  See
   `handover/reports/mediafill.md`.
2. **`--changed-since` still has no job-spec forwarding.** The recording pass re-records what its wave list
   names; a changed-only submitted pass needs the changed list *as* the wave list (section 4), or a
   `RCP_CHANGED_SINCE`-style forwarding added later. The committed repository corpus index cannot drive the
   selection (section 1 item 3); this is the plan's weakest link, now documented rather than assumed.
3. **The case-fill job's RC needs a rebuild** (the cases staging is new); the verification and recording
   passes need none (their scripts are mounted from the checkout at submit time), so an already-staged RC
   stays valid for them.
4. `qwen3-vl-reranker-2b` declares no `reference.device`; the job passes `--device cuda` for it as the brief
   specifies (its reference accepts cuda). Confprep's open question 6.2 (cpu-fill instead) is now moot for
   the text cases.

## Docs updated

- `docs/how-to/release-candidates.md` — the staged tree's `cases/` row, the recording pass's
  `RCP_RECORD_CORPUS` switch, the corrected T1 bullet (the corpus path, the index the flag reads), the
  `--script casefill` job, the `--script` option list.
- `rcp-ndcg-test/README.md` — step 4 (the per-case skip and the casefill job) and the cases-staging sentence.
- `docs/reference/rcp-ndcg-test.md` — the "Filling the cases" bullet.
- `handover/reports/confprep.md` — steps 1–2, section 3.1, step 0, 4.3, open questions 6.1/6.3.
- Greps run: `git grep -n 'record-corpus\|RCP_RECORD_CORPUS' -- docs README.md REPRODUCIBILITY.md skills
  examples experiments mkdocs.yml handover rcp-ndcg-test/README.md`; `git grep -n 'case_fill\|pending_gpu\|
  cases/' -- docs README.md rcp-ndcg-test/README.md`; `git grep -n 'changed_since\|changed-since' --
  rcp-ndcg-test/tests rcp-ndcg-test/src docs`.

## CHANGELOG entry

Under `## 0.0.1 — <date at tag time>` / `### Changed`: the existing `case_fill` bullet's refusal sentence is
amended, and one bullet is added (the exact text committed):

- (amended) "A pending case the mode cannot fill is **skipped per case**, its file left untouched and the
  reason recorded in the report: a media-bearing case's values cannot come from the text-only reference (its
  `media` mode reports input facts, not model outputs), so a media-value fill is a follow-up, and a
  `kind: none` case carries no values by design."
- (added) "**The recording pass and the cuda case fills are submittable jobs** (the unpublished
  `rcp-ndcg-test`): `RCP_RECORD_CORPUS=1` is forwarded by `submit.sh` into the job spec and makes
  `bootstrap.sh`'s wave mode pass `--record-corpus` (unset, the wave argv is byte-identical), so the
  observation corpora are re-recorded as their own wave — a corpus failure is a corpus re-run, not a recipe
  failure. A new `submit.sh --script casefill` job stages `rcp-ndcg-test/cases` into the RC (the wheel drops
  `cases/`), builds the four cuda families' reference environments through `bootstrap.sh envs`, runs
  `python -m rcp_ndcg_test.case_fill --device cuda` per recipe and uploads the edited case YAMLs for a
  reviewed commit. `qwen3-vl-reranker-2b/gen-text-short-uniform`'s `expected.kind` is corrected from
  `similarity_matrix` to `scores` (a rerank case's shape), with a test pinning every shipped case's kind to
  its role's stage shape."

## Public surface changes

`rcp-ndcg-test` only (unpublished, never on PyPI): `rcp_ndcg_test.case_fill` gains `SkippedCase`,
`FillReport.skipped` and the CLI report's `skipped` list; its documented refusal behaviour becomes per-case
skipping. No published distribution's surface moves; no contract snapshot or JSON Schema changes (the job
scripts, the wave lists and the cases are repository tooling/data, not the frozen product surface).

## Files outside scope

`handover/reports/confprep.md` — the brief's item 4 assigns its steps 1–2; the section 3.1, step 0, 4.3 and
open-question sentences were updated in the same edit because this lane's changes made them false
(`RCP_RECORD_CORPUS` now exists, the fill skips per case, the reranker's kind is fixed).

## For the next lanes

1. The verification and recording passes can be submitted with an already-staged RC (their scripts mount
   from the checkout); the case-fill pass needs a rebuild for `cases/`.
2. Record first, then replace the eight corpora with confprep's ten-step procedure, empty `stale.json` and
   re-verify the emulators; the recording pass's changed-only selection is section 4's list route.
3. After the case-fill job returns, review and commit the 26 filled YAMLs; the 14 media cases are the
   recorded residual (open question 1).
4. Flip the statuses from the wave evidence and write `GPU-E2.md` from it; the T4 e2e is its own job
   (`submit.sh --script e2e`).
