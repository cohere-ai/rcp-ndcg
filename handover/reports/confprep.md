# Lane `confprep` — the stale corpora, the emulators and the pending case values

## 1. Status

**DONE (CPU half of both deliverables).** The eight stale corpora have not been re-recorded yet (the GPU
waves are still running equivalence, not `--record-corpus`), so the corpus replacement itself is prepared,
tested and left to the follow-up turn with the exact command sequence (section 3.4). The reference-case
fill is executed: **37 of the 78 `pending_gpu` cases now carry the reference implementation's own values**
(qwen3-reranker-0.6b/-4b/-8b and zembed-1-embedding); the remaining 41 are cuda-reference families with
their exact node commands (section 4.3). The listwise replay coverage is restored on CPU (section 3.3).

The branch's base is the current `rfc-0001` tip (`5c2316a2`, the MTEB export merge), merged in `cce899fb`;
`rfc-0001` is an ancestor of the lane HEAD. Lane HEAD: **`43c9de5d`** (the report commit follows).

**Gate.** `bin/gate lane/confprep` at `43c9de5d`, five attempts. Every step is green on every attempt
(ruff, ruff-format, basedpyright, the root suite `4136 passed, 155 skipped`, contract+docs `413 passed,
107 skipped`, mkdocs `--strict`, the recipe step `0 baseline failures remain`, the vLLM package and model
suites, `run_all` `1022/987/35/0` + `67/67` + `82/82`, public-names, clean) **except the unpublished test
package's step, which exits 139 (SIGSEGV) at a random test**:

| Attempt | Failing test of the test-package step |
|---|---|
| 1 | `test_reference_env.py::test_family_rows_groups_variants_and_reports_load_failures` |
| 2 | `test_record_and_wave.py::test_wave_verifies_a_recipe_whose_controls_all_pass` |
| 3 | `test_quality.py::test_the_wave_runs_the_quality_stage_and_fails_a_recipe_on_it` |
| 4 | `test_wave_corpus.py::test_the_corpus_step_carries_its_budget_and_log_lines` |
| 5 | `test_wave_corpus.py::test_the_corpus_is_keyed_by_the_engine_version_the_pod_reports` |

This is the documented environment flake, not a test failure: the repo's own QA report
(`handover/reports/07a-qa-prep.md`, section 2) records the first post-merge `rcp-ndcg-test` single-process
run exiting 139 inside pytest's own tmpdir cleanup under 13 concurrent pytest processes, with the identical
rerun green; the verifier lens saw it twice more under load average ~53. In this lane the same five tests
pass in isolation, and the gate's exact test-package command ran green twice in this worktree under
comparable load: `env PYTHONFAULTHANDLER=1 RCP_NDCG_TEST_TIMEOUT=300 uv run --no-sync python -X faulthandler
-m pytest rcp-ndcg-test/tests -q -p no:cacheprovider` -> `1146 passed, 229 skipped` (12m20s), and the same
suite through the `heavy` slot wrapper -> `1146 passed, 229 skipped` (18m25s). The orchestrator will run the
final full gate on the merged `rfc-0001` tip in a quiet window; no code was changed for the flake.

## 2. Commits

| Commit | Subject |
|---|---|
| `8a55df77` | The case fill, the corpus-index regeneration and the stale listwise replay |
| `1af1d960` | The qwen3-reranker-0.6b pending reference cases are filled from its reference implementation |
| `4ff59ab4` | The qwen3-reranker-4b pending reference cases are filled from its reference implementation |
| `e21cbc9f` | The qwen3-reranker-8b pending reference cases are filled from its reference implementation |
| `eb927f5f` | The zembed-1-embedding pending reference cases are filled from its reference implementation |
| `cce899fb` | Merge branch 'rfc-0001' into lane/confprep (the MTEB export tip `5c2316a2`) |
| `43c9de5d` | The embed fill test builds its own pending grid |
| (this report) | The lane report |

## 3. Deliverable 1: the eight stale corpora, the emulator re-verification, the listwise replay

### 3.1 State: no re-recording has landed

`python -m rcp_ndcg_test.changes changed --recipes-root rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes
--corpora-root rcp-ndcg-test/corpora/vllm-0.31.0` reports all eight declared-stale recipes still
`changed`, each with the exact fingerprint inputs that moved (the staleness test pins the same lists):

| Recipe | State | Moved inputs | Recorded fingerprint |
|---|---|---|---|
| jina-embeddings-v5-text-small | changed | 18 | `31afc0cb…` |
| jina-reranker-v3 | changed | 9 | `279e67fb…` |
| qwen3-reranker-0.6b | changed | 11 | `9884cc4e…` |
| qwen3-reranker-4b | changed | 10 | `0a92497f…` |
| qwen3-reranker-8b | changed | 1 (`serve.hf_overrides`) | `ece867dc…` |
| qwen3-vl-embedding-2b | changed | 17 | `9f4f62aa…` |
| qwen3-vl-reranker-2b | changed | 1 (`serve.hf_overrides`) | `32420aee…` |
| zembed-1-embedding | changed | 13 | `80bba022…` |

GCS holds no observation corpus for any of them: no `observations/vllm-<version>/<recipe>/...` tree
exists under any `waves-*` prefix (checked `rc0-202610101230` through `rc0-202610120530`, the
`e1*`/`shake` prefixes, and -- the final re-check at the end of the lane -- the newest RC
`rc0-202610120815`, 08:15 UTC). That RC re-ran r2/r3/r4 for the eight: each recipe's `status.json` lists
the steps `serve`, `smoke`, `equivalence`, `record` and reports `verified`, but there is no
`observation_corpus` step and no corpus tree -- the wave runner's `--record-corpus` step (the
observation corpus: the plan's rows twice in one process, once after a restart, plus the probes and
`/tokenize`) has not run. `tests/conformance/stale.json` therefore keeps all eight entries, and the
release-flag test (`RCP_NDCG_RELEASE=1`) still fails until they are replaced.

### 3.2 Emulator re-verification

The conformance suite replays every committed corpus that is **not** declared stale, and
`RCP_APPEND_VERIFICATION=1` appends the computed verification record (which emulator verified the corpus,
against which engine version and recipe revision, with which tolerances, and the result). The eight stale
corpora are excluded from that replay set by design and each still fails the staleness gate by name,
exactly as declared (`test_a_declared_stale_corpus_fails_the_staleness_gate_by_name` pins the message and
the moved-input list). They are re-verified the moment their re-recording replaces the corpus
(section 3.4, step 8). Current state of the suite: `26 passed, 1 skipped` (the skip is the release flag).

### 3.3 The listwise replay coverage — genuinely missing, restored on CPU

The package's only listwise recipe is `jina-reranker-v3`, and its only committed corpus is the one declared
stale, so the standard replay set (`current_corpora`, which the staleness gate filters) contains no listwise
prompt path at all. The corpus's fingerprint inputs moved, but its recorded exchanges do **not** depend on
them: replaying the committed corpus against the current recipe's emulator (its real tokenizer, its own
`format_docs_prompt` N-passage builder) reproduces all four records exactly, the `POST /rerank` probe
included. A new conformance test
(`test_the_declared_stale_listwise_corpus_still_replays_its_role_requests`) exercises that replay, asserts
the corpus really carries a successful role request (no vacuous pass), and skips itself once the
re-recording removes the stale declaration (the standard replay then covers the listwise path). The corpus
stays non-gating release evidence; the test never relaxes the stale declaration. Test-side helper:
`tests/_engines.emulator_over(recipe_id, corpus)` builds the emulator over a caller-resolved corpus.

### 3.4 The CPU-side procedure, complete and tested

The re-record procedure is `verify` -> `changes diff` -> `subset` -> replace -> `index` -> re-verify. The
`verify`, `diff` and `subset` commands already existed and were exercised on the committed corpora in this
lane; the **index step had no tool** (the repository corpus index was hand-maintained), so
`rcp_ndcg_test.corpus.write_corpus_index` and the `observe.corpus index` subcommand are new: they regenerate
`corpora/<engine>/index.json` from the committed corpora (the facts the integrity test reads: behaviour
fingerprint, the SHA-256 of the committed manifest, the model revision). Verified on the real corpora: the
regenerated index has the same keys and byte-identical entries as the committed one.

Exact command sequence, once the recordings land (paths are relative to the repository root; `<work>` is a
scratch directory outside it):

```bash
# 0. the recording wave (on the node, in the bootstrap's client environment).  The wave bootstrap now
#    carries the switch: submit a recording pass with RCP_RECORD_CORPUS=1 (submit.sh forwards it into
#    the job spec; bootstrap.sh's wave mode then passes --record-corpus).  --changed-since takes a
#    PREVIOUS RECORDING wave's wave.json (its fingerprints and engine_versions maps) -- the committed
#    repository corpus index is a different schema and would re-record everything; to select the
#    recipes whose corpora are stale against the committed ones, use the changes command above and
#    make that list the wave list.
#
#    The equivalent direct command, run in the client environment the bootstrap built:
"$STATE/client" python -m rcp_ndcg_test.jobs.run_wave \
    --recipes "@$WAVE_LIST_FILE" --recipes-root "$RECIPES_ROOT" --gpus "$gpus" \
    --out "$STATE/wave" --upload "$OUT_URI" --record --record-corpus \
    --changed-since "<a previous recording wave's wave.json>" \
    --failed-plugins "$STATE/plugin-failures.txt" --reference-root "$STATE/reference" \
    --pairs-dir "$STAGE_DIR/pairs"
#    (bootstrap.sh's wave mode passes --record and, with RCP_RECORD_CORPUS=1, --record-corpus; the
#    changed-only index has no job-spec forwarding today -- see Open questions 6.1)

# 1. locate the recording (the wave runner writes
#    <out>/observations/vllm-<version>/<recipe>/<fingerprint>/<stamp>/ and uploads it with the wave)
gcloud storage ls "gs://cohere-dev-central-2/fabian/rcp-ndcg-rfc/waves-<rc>/<wave>/<engine-dir>/observations/vllm-<version>/<recipe>/<fingerprint>/"

# 2. download the full corpus (the stamped directory of the recording)
mkdir -p "<work>/<recipe>"
gcloud storage rsync --recursive "gs://…/<recipe>/<fingerprint>/<stamp>" "<work>/<recipe>/corpus"

# 3. the acceptance checks on the full corpus (completeness, statuses, vector widths, credentials, hashes)
uv run --no-sync python -m rcp_ndcg_test.observe.corpus verify "<work>/<recipe>/corpus"

# 4. the behaviour diff against the committed corpus, for review before replacing it
uv run --no-sync python -m rcp_ndcg_test.changes diff \
    --before "rcp-ndcg-test/corpora/vllm-0.31.0/<recipe>/<recorded-fingerprint>" \
    --after  "<work>/<recipe>/corpus"

# 5. the repository subset at the new fingerprint (stratified, <= 2 MB, hashes its files)
uv run --no-sync python -m rcp_ndcg_test.observe.corpus subset "<work>/<recipe>/corpus" "<work>/subset-<recipe>" \
    --gcs-path "gs://…/<recipe>/corpus"

# 6. replace the committed corpus (the new fingerprint is the subset manifest's behaviour_fingerprint)
rm -rf "rcp-ndcg-test/corpora/vllm-0.31.0/<recipe>/<recorded-fingerprint>"
mv "<work>/subset-<recipe>" "rcp-ndcg-test/corpora/vllm-0.31.0/<recipe>/<new-fingerprint>"

# 7. regenerate the repository corpus index from the committed corpora (never hand-edit it)
uv run --no-sync python -m rcp_ndcg_test.observe.corpus index rcp-ndcg-test/corpora/vllm-0.31.0

# 8. re-verify the emulator at the new fingerprint (appends the verification record, never rewrites it)
RCP_APPEND_VERIFICATION=1 uv run --no-sync pytest rcp-ndcg-test/tests/conformance/test_conformance.py -q

# 9. drop each recipe's entry from tests/conformance/stale.json once every one of the eight matches its
#    new recording; the release rule fails until the file is empty:
RCP_NDCG_RELEASE=1 uv run --no-sync pytest rcp-ndcg-test/tests/conformance/test_conformance.py -q

# 10. the whole test package again
uv run --no-sync pytest rcp-ndcg-test/tests -q
```

## 4. Deliverable 2: the `pending_gpu` expected values

### 4.1 The fill mechanism

The conformance package had no fill runner, so this lane adds one: `rcp_ndcg_test.case_fill` (CLI:
`python -m rcp_ndcg_test.case_fill --recipe <dir-or-id> --reference-python <the family's reference venv>`).
It loads the recipe's cases, runs the recipe's reference **once** over every pending case's inputs (one
model load; the pairs rows are the reference subprocess's own input format), shapes the outputs into the
case's `expected.kind` -- a scores matrix for a rerank recipe, the query x document similarity matrix for an
embed or multi_vector one (the cosine of the L2-normalized single vectors, the product's MaxSim for a
late-interaction reference) -- and edits each case YAML line-wise: `values` replaced, `status: pending_gpu`
-> `reproduced`, the pending note's own sentence retired. A media-bearing case and a kind the mode cannot
produce are refused by name **before** the reference runs (nothing is half-filled); a `kind: none` case
carries no values by design and is left alone. The written file is loaded again through the case format;
a file that no longer loads is restored and the error re-raised. The reference python's path never reaches
the repository: the note carries the device and the date only.

Failing first: `rcp-ndcg-test/tests/test_case_fill.py` was run before the module existed (4 tests, all
`ModuleNotFoundError: No module named 'rcp_ndcg_test.case_fill'`), and the corpus-index test failed with
`ImportError: cannot import name 'write_corpus_index'` before the function landed. Five tests now cover the
rerank and embed paths end to end (a fake reference subprocess), the media refusal, the CLI report, the
pending-note retirement for both note vocabularies, the no-op second run, and the product's MaxSim for a
ragged reference.

### 4.2 Filled: 37 of 78 cases (all CPU-reference families)

| Recipe | Pending before | Filled | Reference environment |
|---|---|---|---|
| qwen3-reranker-0.6b | 10 | **10** | CPU: torch 2.9.1+cpu, transformers 4.57.6 (the family's lock pins) |
| qwen3-reranker-4b | 10 | **10** | same |
| qwen3-reranker-8b | 10 | **10** | same |
| zembed-1-embedding | 7 | **7** | CPU: torch 2.9.1+cpu, transformers 5.19.0, sentence-transformers 5.3.0 (the family's lock pins) |
| **total** | **37** | **37** | |

The values are the reference implementation's own outputs for the case inputs (`origin: reference`), never
the engine's and never a guess. Sanity-checked after the runs: every qwen3-reranker filled cell is a
probability in [0, 1] (25 cells per size), every zembed cell a cosine (23 cells, min 0.0003, max 0.642).
Commands (the reference venvs live in this lane's scratch, built from PyPI and the PyTorch CPU index with
the families' own lock pins):

```bash
uv run --no-sync python -m rcp_ndcg_test.case_fill --recipe qwen3-reranker-0.6b \
    --reference-python <ref-qwen3-reranker>/bin/python --filled-at 2026-10-12
# … the same for qwen3-reranker-4b and -8b (--budget-s 3300 / 7200 for the larger CPU runs)
uv run --no-sync python -m rcp_ndcg_test.case_fill --recipe zembed-1-embedding \
    --reference-python <ref-zembed-1>/bin/python --filled-at 2026-10-12 --budget-s 10500
```

### 4.3 Remaining: 41 cases, cuda-reference families (exact node commands)

`topk-embed-v1-small` (15; `reference.device: cuda`), `qwen3-embedding-0.6b` (8; `device: cuda`),
`qwen3-vl-embedding-2b` (9; `device: cuda`) and `qwen3-vl-reranker-2b` (9; the family declares no device,
and the brief classes qwen3-vl cuda-only -- see Open questions 6.2). One of topk's 15 is
`query-cap-1024`, a `kind: none` case that carries no values by design; the fill leaves it pending.

On the node, in the client environment, with the wave bootstrap's per-family reference venvs
(`<reference-root>` is the bootstrap's `$STATE/reference`; the family directory names are the venv keys).
The fill now ships as the `--script casefill` job (the runplumb lane: `handover/reports/runplumb.md`),
which builds those venvs through `bootstrap.sh envs` and passes `--cases-root <stage>/cases` (the wheel
drops `cases/`, so the RC stages them); the same commands by hand:

```bash
python -m rcp_ndcg_test.case_fill --recipe topk-embed-v1-small \
    --reference-python "<reference-root>/topk-embed-v1/bin/python" --device cuda --budget-s 7200
python -m rcp_ndcg_test.case_fill --recipe qwen3-embedding-0.6b \
    --reference-python "<reference-root>/qwen3-embedding/bin/python" --device cuda --budget-s 7200
python -m rcp_ndcg_test.case_fill --recipe qwen3-vl-embedding-2b \
    --reference-python "<reference-root>/qwen3-vl-embedding/bin/python" --device cuda --budget-s 7200
python -m rcp_ndcg_test.case_fill --recipe qwen3-vl-reranker-2b \
    --reference-python "<reference-root>/qwen3-vl-reranker/bin/python" --device cuda --budget-s 7200
```

(`--device cuda` is what the wave runner passes when a reference GPU is reserved; qwen3-vl-reranker's
reference accepts `cpu` too, since its family declares no device.  A pending case the fill cannot produce
is skipped per case with its reason recorded, its file untouched: the 14 media-bearing ones (the
text-only reference cannot read them; the reference's `media` mode reports input facts, not model
outputs, so a media-value fill is a follow-up) and topk's `kind: none` one -- 26 of the 40 values fill.)

**Superseded by lane mediafill** (owner decision 2026-10-12: the media cases are in scope): the media
cases fill from the references' existing `embed`/`score` modes over the pairs row's `media` field, so 39
of the 40 values-bearing cases fill (the qwen3-vl-embedding video case is a recorded exception at the
recipe's video pin) and only topk's `kind: none` case is the structural non-fill.  See
`handover/reports/mediafill.md`.

## 5. Verification

- `bin/gate lane/confprep` at `43c9de5d`, five attempts: all steps exit=0 except the test-package SIGSEGV
  flake (section 1). The gate SUMMARY of the last attempt is quoted there.
- The test-package suite, single process, exactly the gate's environment:
  `1146 passed, 229 skipped` (twice: plain and through `heavy`).
- The root suite: `4136 passed, 155 skipped`; contract+docs: `413 passed, 107 skipped`; mkdocs `--strict`
  clean; `tests/test_cases_package.py` loads every case after the fills (`3 passed, 1 skipped`).
- ruff format/check clean, basedpyright 0 errors.
- The conformance suite after the listwise test: `26 passed, 1 skipped`.
- The corpus tools on real data: `observe.corpus verify` passes on the committed corpora; `changes diff`
  runs; `observe.corpus index` reproduces the committed index (same keys, identical entries).
- The reference runs themselves: 0.6b 2m05s (13 pairs rows), 4b 16m00s, 8b 16m (CPU, bf16); zembed 24m
  (7 cases, up to 32k-token documents).
- No staged files, the tree clean after the commits (tests write only to `tmp_path`).

## 6. Open questions

1. **The wave bootstrap carries the corpus re-recording switch now, but not `--changed-since`.**
   `bootstrap.sh`'s wave mode execs `run_wave … --record` plus `--record-corpus` when
   `RCP_RECORD_CORPUS=1`, and `submit.sh` forwards the variable into the job spec (the runplumb lane's
   change: see `handover/reports/runplumb.md`).  What is still unplumbed is the changed-only index:
   `--changed-since` takes a previous RECORDING wave's `wave.json`, and no env variable forwards it
   into the job spec -- a changed-only pass either runs `run_wave` directly in the client environment
   (the command in section 3.4 step 0) or makes the changed list the wave list (the `changes changed`
   selection above, which is the comparison against the committed corpora).
2. **qwen3-vl-reranker-2b declares no `reference.device`** while the brief classes the qwen3-vl family
   cuda-only. Its reference's score mode accepts `cpu`; the next lane decides whether to CPU-fill its nine
   cases here (its media cases still need the media path) or leave them to the node.
3. **`qwen3-vl-reranker-2b/gen-text-short-uniform` declares `kind: similarity_matrix` for a rerank
   recipe** (the loader accepts it; the fill skips it as another stage's shape, and the conformance
   runner would fail it on the shape). The cases lane's slip; a rerank case's kind should be `scores` --
   fixed in the runplumb lane, with a test pinning every shipped case's kind to its role's stage shape.
4. The eight corpora's re-recording is the release blocker for `stale.json`; every other stale-related
   check is in place (the declarations match the recomputed inputs exactly, the waivers file is empty, the
   release-flag test enforces the empty list).

## Docs updated

- `rcp-ndcg-test/README.md` -- "How to add one" step 4: the fill command, what it writes, the refusals.
- `docs/reference/rcp-ndcg-test.md` -- a "Filling the cases" bullet in "What it holds".
- `CHANGELOG.md` -- the two entries below, under `## 0.0.1` / `### Changed`.
- Greps run: `git grep -n "pending_gpu" -- docs README.md REPRODUCIBILITY.md skills examples experiments
  mkdocs.yml` (no hits), `git grep -rn "case_fill\|fill every\|pending" -- docs/reference/rcp-ndcg-test.md`,
  `git grep -n "record-corpus" -- rcp-ndcg-test/src rcp-ndcg-test/tests`.

## CHANGELOG entry

Under `## 0.0.1 — <date at tag time>` / `### Changed` (the exact text committed):

- **The pending reference cases are filled by `rcp_ndcg_test.case_fill`** (the unpublished `rcp-ndcg-test`):
  the release checklist's "fill every `pending_gpu` expected value" is now a tool -- one reference
  subprocess per recipe over every pending case's inputs (the pairs rows the reference already reads), the
  outputs shaped into the case's `expected.kind` (a scores matrix for a rerank recipe; the query x document
  similarity matrix for an embed or multi_vector one -- the cosine of the L2-normalized single vectors, the
  product's MaxSim for a late-interaction reference), and a line-wise YAML edit (`values`,
  `status: reproduced`, the pending note's own sentence retired) that leaves every other byte of the
  hand-written case in place.  A media-bearing case and a kind the mode cannot produce are refused by name
  before the reference runs; a `kind: none` case carries no values by design and is left alone.  The
  reference entry path has one home now (`rcp_ndcg_test.equivalence.reference.reference_entry`, used by
  stages 1-2, the media stage and the fill).
- **The listwise replay stays covered while jina-reranker-v3's corpus is declared stale**: the staleness
  gate keeps that corpus out of the standard conformance replay set, and with it the package's only
  listwise prompt path; a conformance test replays the corpus's recorded exchanges (its `/rerank` probes
  included) against the current recipe's emulator, which they reproduce exactly, and skips once the
  re-recording removes the declaration.

The corpus-index regeneration (`write_corpus_index`, the `observe.corpus index` subcommand) is part of the
first entry's tooling; no separate bullet was added for it.

## Public surface changes

`rcp-ndcg-test` only (unpublished, never on PyPI): new module `rcp_ndcg_test.case_fill`
(`FillReport`, `FilledCase`, `expected_values`, `fill_cases`, `main`); new
`rcp_ndcg_test.corpus.write_corpus_index`; new `rcp_ndcg_test.equivalence.reference.reference_entry`; the
`python -m rcp_ndcg_test.observe.corpus index <engine-dir>` subcommand; the test-side
`tests._engines.emulator_over`. No published distribution's surface moves; no contract snapshot or JSON
Schema changes (the case files are repository data, not a schema).

## Files outside scope

none. The changes are in `rcp-ndcg-test/` (source, tests, cases), its README, `docs/reference/rcp-ndcg-test.md`
and `CHANGELOG.md`.

## For the next lanes

1. Run the re-record wave with `--record-corpus` (section 3.4 step 0; `RCP_RECORD_CORPUS=1` reaches the
   wave through `submit.sh`), then the ten-step replacement sequence; empty `tests/conformance/stale.json`
   only when all eight match, and re-verify every emulator with `RCP_APPEND_VERIFICATION=1`.
2. Fill the four cuda-reference families' 41 cases on the node with the commands in section 4.3, then run
   the conformance suite against each family's engine (the cases are the reference side; the engine side is
   the wave's conformance target).
3. Decide the qwen3-vl-reranker device question (6.2) and fix the mislabelled
   `qwen3-vl-reranker-2b/gen-text-short-uniform` kind (6.3).
4. The lane's repo changes are on `lane/confprep`; the merge into `rfc-0001` is the orchestrator's.
