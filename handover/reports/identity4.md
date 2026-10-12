# Lane identity4 — the local-dataset content identity (the pre-tag review fix)

**Lane:** `lane/identity4` (worktree outside the operator's checkout). **Base:** `rfc-0001` at `3217b5ed`,
merged with its tip `9d119c5e` as `1dbfdec6`, then with the then-tip `0d613818` as `0b62931c`, then with the
then-tip `9c3adf0b` as `aaf60c36`. **Gate:** `timeout 7200 bin/gate lane/identity4` on `aaf60c36` (slot 3) —
**GATE: PASS**, every step exit 0 (the SUMMARY is below; the two earlier gates at `1dbfdec6` and `0b62931c`
passed too, and two attempts on heavily loaded machines failed a single per-test timeout each — the recipe
stage-1 and topk tests, proven passing alone — see the Gate section). No GPU work; nothing pushed;
`rfc-0001` was not merged into. **Integrated:** `rfc-0001` merged this lane's gated tree as `909ddca5`
(parents `9c3adf0b` + `aaf60c36`, no conflicts); its later commits (`d2e1a70d` r8, `97f4b64b` confprep) are
other lanes' work merged on top, so this branch is not re-merged with them.

## 1. Status

Done: the one finding the deep audit recommends fixing before the tag (its rows retrieval A4 + judge A4) is
closed. One correction to the audit's re-check: the **run** half was already fixed on the tip by lane
retrieval-fixes (`ebd64457`: `DatasetSource.identity()` folds a local source's content digest, with the
`test_a_local_datasets_content_names_it` and `test_an_edited_local_corpus_makes_the_retrieve_step_stale`
tests), so the audit's "a local `jsonl:` has no commit and no content digest" is stale for `runs`. The open
half was the **judgement store**: `_dataset_identity` keyed a loaded local dataset by name, absolute path and
revision only — and a local source has no revision — so editing `rows.jsonl` in place re-ran the judging step
(the step identity did change) and then silently reused every stored answer over the old texts (measured: the
second pass asked the judge **0 requests** and returned the 28 stored records).

What this lane did:

1. **The digest got one home in the data layer.** `_local_source_digest`, the scheme table and the digest
   cache moved out of `runs/config.py` into `rcp_ndcg.data.revisions`, which already owns what names a dataset
   source (the resolved commit). Two public functions: `dataset_content_digest(uri, options=None)` (the digest
   of the files a local source's reader reads) and `dataset_local_locations(uri, options=None)` (the
   locations, listed without being read). `DatasetSource.content_digest` now delegates; `RunConfig.local_inputs`
   and `_relative_to` read the same scheme table instead of their own copy.
2. **The store consumes it.** `judging._dataset_identity` adds `content` (the same `{name: sha256}` payload
   the run steps carry) for a local URI source; a remote source is byte-identical in behaviour (a resolved
   commit is the content; no digest is computed, no download is paid). An edited file is another dataset:
   `store.claim` refuses with `IdentityError` naming `dataset.content`, `force=True` supersedes; nothing is
   pooled or reused. The `dataset_key` (and therefore every record id) moves with the content, as it should.
3. **Tests first, per home** (quoted below): a judgement store whose local file is edited must not reuse the
   records; a run whose local dataset is edited must not resume/reuse the store.
4. **Docs, snapshots, CHANGELOG**: the two pages that describe step and store identity, the generated API
   page, `tests/contract/snapshots/python_api.json`, `schemas/judgement-store.v1.json` (the `identity`
   description) and the CHANGELOG.

## 2. Commits

| commit | what |
|---|---|
| `1b7f4e63` | the failing tests first (judge home, run home, the data-layer digest home, the beir store identity) |
| `3e4b5584` | the one home in `rcp_ndcg.data.revisions`, consumed by the run and the store |
| `6486dcc5` | the docs, the regenerated surface (snapshot, schema, API page) and the CHANGELOG entries |
| `1dbfdec6` | merge of `rfc-0001` (`9d119c5e`) into the lane, CHANGELOG conflict resolved (both bullets kept) |
| `b1990d9b` | this report (first version) |
| `0b62931c` | merge of `rfc-0001`'s then-tip (`0d613818`, lanes names and mteb) into the lane, clean |
| `b88f1d11` | the report carries the `0b62931c` merge and gate |
| `aaf60c36` | merge of `rfc-0001`'s then-tip (`9c3adf0b`, lane judge8) into the lane, clean (judging.py, its tests, judges.md and the CHANGELOG auto-merged; the judging suites re-run green) |
| `cf4d80fe` | the report carries the judge8 merge and the `aaf60c36` gate |
| `909ddca5` (on `rfc-0001`) | the integrator's merge of this lane's gated tree (`aaf60c36`); no conflicts |
| the report commit | this report |

No co-author lines, no attribution, nothing pushed.

## 3. What changed (files)

- `rcp-ndcg/src/rcp_ndcg/data/revisions.py`: the digest's one home (`_LOCAL_SCHEMES`, `_LOCAL_DIGEST_CACHE`,
  `_local_source_digest`, `dataset_local_locations`, `dataset_content_digest`), the module docstring, `__all__`.
- `rcp-ndcg/src/rcp_ndcg/runs/config.py`: the moved code removed; `DatasetSource.content_digest` delegates;
  `local_inputs`/`_relative_to` use the data-layer rule.
- `rcp-ndcg/src/rcp_ndcg/judging/judging.py`: `_dataset_identity` adds the content digest for a local URI
  source (the brief names `judging/store.py`; the identity is built here, and the store's `StageEntry`
  docstring now states the field).
- `rcp-ndcg/src/rcp_ndcg/runs/pipeline.py`: the `_dataset_identity` docstring states the local rule and its
  one home.
- `tests/data/test_revisions.py`, `tests/judging/test_judging.py`, `tests/runs/test_pipeline.py`,
  `tests/judging/test_document_text.py` (the beir store-identity assertion gains `content`).
- `docs/concepts/runs.md`, `docs/concepts/judges.md`, `docs/reference/api/rcp_ndcg.data.revisions.md`
  (regenerated), `CHANGELOG.md`, `tests/contract/snapshots/python_api.json`,
  `schemas/judgement-store.v1.json` (regenerated), this report.

## 4. Verification

### The failing tests first, per home (quoted)

Judge home, before the fix (`tests/judging/test_judging.py::TestIdentities::test_a_local_dataset_is_named_by_its_content_and_an_edited_file_is_refused`):

```
>       assert entry["identity"]["dataset"]["content"]["dataset"], "a local source is named by its content"
E       KeyError: 'content'
tests/judging/test_judging.py:1059: KeyError
FAILED ...::test_a_local_dataset_is_named_by_its_content_and_an_edited_file_is_refused
```

and the silent-reuse measurement at the tip (a scratch script outside the repository, same steps as the test):

```
second pass requests: 0 records returned: 28
store dataset identity: {"name": "rows", "revision": null, "uri": "jsonl:/tmp/.../rows.jsonl"}
```

Run home, before the fix (`tests/runs/test_pipeline.py::TestTheLocalDatasetContentAndTheBehaviourVersions::test_an_edited_local_dataset_refuses_the_store_instead_of_reusing_it`):

```
>       with pytest.raises(IdentityError, match="dataset.content") as refused:
E       Failed: DID NOT RAISE IdentityError
tests/runs/test_pipeline.py:1454: Failed
```

After the fix both pass, and the run-home test also pins that the edited corpus makes the judging step stale
(`plan()["tournament"] == "would run"`).

### Mutation evidence for the run home

The run half already landed in `ebd64457`; to show the run-home tests guard it through the shared home,
`dataset_content_digest` was temporarily made to return `{}`:

```
tests/runs/test_pipeline.py:1409: KeyError: 'content'                                  # test_a_local_datasets_content_names_it
tests/runs/test_pipeline.py:1437: AssertionError: the edited corpus makes the retrieve step stale
                                  assert 'would skip' == 'would run'                     # test_an_edited_local_corpus_makes_the_retrieve_step_stale
tests/runs/test_pipeline.py:     Failed: DID NOT RAISE IdentityError                     # the run/store test above
tests/judging/test_judging.py:1059: KeyError: 'content'                                  # the judge home
```

Restored afterwards (`grep -c MUTATION` = 0); the four test classes pass (`29 passed`).

### Suites run on the merged tree

- `uv run --no-sync pytest tests/runs tests/judging tests/data tests/retrieval tests/contract tests/docs -q -n 4`
  — **2046 passed, 138 skipped**.
- `heavy uv run --no-sync pytest rcp-ndcg-test/tests -q` — **1139 passed, 229 skipped** (`1140 passed` on the
  final gate's `test-pkg` step, after the mteb lane's test landed in the second merge).
- `ruff format --check`, `ruff check` — clean; `basedpyright` — **0 errors, 0 warnings, 0 notes**;
  `mkdocs build --strict` — built.
- The gate's recipe step env, the file that timed out on the loaded first attempt:
  `RCP_NDCG_NETWORK_TESTS=1 RCP_NDCG_VLLM_TOKENIZER_CACHE=... uv run --no-sync pytest
  rcp-ndcg-test/tests/recipes/test_zembed_1.py -q` — **10 passed, 1 skipped** on the quiet machine.

### Gate

First attempt, revision `1dbfdec6` (slot 3, machine load ~157/96 with three other gates and the GPU
waves): every step exit 0 except `recipes`, whose log shows a per-test timeout in a tokenizer encode —
`TimeoutError: the test exceeded its 300s per-test timeout (a hang fails fast)` at
`rcp-ndcg-test/tests/conftest.py:133`, inside
`test_zembed_1.py::test_stage1_on_cpu_passes_anchors_and_render`. Nothing in that stack touches this lane's
change (recipe stage-1 probe → embed client → text-budget fit → `tokenizer.count`), and the file passes alone
under the gate's own env, so the timeout is environmental (CPU starvation), not a regression. Retry on the
quieter machine (load ~29/96):

```
rev lane/identity4 = 1dbfdec6 (slot 5)
ruff-check exit=0 All checks passed!
ruff-format exit=0 624 files already formatted
basedpyright exit=0 0 errors, 0 warnings, 0 notes
pytest exit=0 4141 passed, 155 skipped in 54.74s
contract-docs exit=0 413 passed, 107 skipped in 41.38s
mkdocs exit=0 INFO    -  Documentation built in 3.06 seconds
test-pkg exit=0 ================ 1139 passed, 229 skipped in 770.66s (0:12:50) =================
recipes exit=0 recipes: no failure outside the baseline (0 baseline failures remain, 0 fixed; pytest exit 0)
vllm-pkg exit=0 50 passed in 5.17s
vllm-models exit=0 102 passed, 8 skipped in 61.54s (0:01:01)
run_all exit=0   external_judges  ok
leaderboards: 1022 checks, 987 match, 35 known deviations, 0 failed
human study: 67 checks, 67 match, 0 known deviations, 0 failed
external LLM judges: 82 checks, 82 match, 0 known deviations, 0 failed
public-names exit=0 public-names: clean (0 baselined hits remain)
clean exit=0 clean
GATE: PASS
```

After the merge with `rfc-0001`'s then-tip (`0d613818`):

```
rev lane/identity4 = 0b62931c (slot 3)
ruff-check exit=0 All checks passed!
ruff-format exit=0 624 files already formatted
basedpyright exit=0 0 errors, 0 warnings, 0 notes
pytest exit=0 4141 passed, 155 skipped in 56.07s
contract-docs exit=0 413 passed, 107 skipped in 45.78s
mkdocs exit=0 INFO    -  Documentation built in 2.21 seconds
test-pkg exit=0 ================ 1140 passed, 229 skipped in 928.49s (0:15:28) =================
recipes exit=0 recipes: no failure outside the baseline (0 baseline failures remain, 0 fixed; pytest exit 0)
vllm-pkg exit=0 50 passed in 9.30s
vllm-models exit=0 102 passed, 8 skipped in 72.59s (0:01:12)
run_all exit=0   external_judges  ok
leaderboards: 1022 checks, 987 match, 35 known deviations, 0 failed
human study: 67 checks, 67 match, 0 known deviations, 0 failed
external LLM judges: 82 checks, 82 match, 0 known deviations, 0 failed
public-names exit=0 public-names: clean (0 baselined hits remain)
clean exit=0 clean
GATE: PASS
```

After the merge with `rfc-0001`'s then-tip (`9c3adf0b`, lane judge8) — the authoritative gate on the final
tree — an attempt at load ~150/96 failed one root test (`tests/retrieval/test_topk.py::
TestTheSelectedSetIsAFunctionOfTheInputsOnly::test_a_huge_orthogonal_document_does_not_hide_the_true_winner`,
`TimeoutError: the test exceeded its 60s per-test timeout` in a 100,000-query numpy call, untouched by this
lane; it passes alone in 8.5s at load ~27), so it was retried when the load dropped:

```
rev lane/identity4 = aaf60c36 (slot 3)
ruff-check exit=0 All checks passed!
ruff-format exit=0 624 files already formatted
basedpyright exit=0 0 errors, 0 warnings, 0 notes
pytest exit=0 4145 passed, 155 skipped in 50.78s
contract-docs exit=0 413 passed, 107 skipped in 40.61s
mkdocs exit=0 INFO    -  Documentation built in 2.68 seconds
test-pkg exit=0 ================ 1140 passed, 229 skipped in 884.61s (0:14:44) =================
recipes exit=0 recipes: no failure outside the baseline (0 baseline failures remain, 0 fixed; pytest exit 0)
vllm-pkg exit=0 50 passed in 8.45s
vllm-models exit=0 102 passed, 8 skipped in 79.36s (0:01:19)
run_all exit=0   external_judges  ok
leaderboards: 1022 checks, 987 match, 35 known deviations, 0 failed
human study: 67 checks, 67 match, 0 known deviations, 0 failed
external LLM judges: 82 checks, 82 match, 0 known deviations, 0 failed
public-names exit=0 public-names: clean (0 baselined hits remain)
clean exit=0 clean
GATE: PASS
```

The commit after the gate changes only this report (handover Markdown); `public-names` and `mkdocs --strict`
were re-run on the final tip and stay clean.

## 5. Open questions

1. **The deep-audit closure table's retrieval A4 row is stale.** The run half landed in `ebd64457` before the
   audit's re-check (the tests and the code are on `3217b5ed`); only the judge A4 half was open. The closure
   table should say so, or a later reader will chase a fixed finding.
2. **The directory rule is the cheap pre-check, deliberately.** A directory source (`beir:`, `images:`,
   `videos:`, `frames:`) digests its sorted listing of relative names, sizes and `mtime_ns` — not the bytes
   of every image — and a plain file digests as its bytes. The residual below names what that cannot see; if
   the owner judges it too weak for image corpora, a `hash_media: true`-style opt-in is the next step.
3. **`dataset_local_locations` is public** because `runs/config.py` consumes it; if the owner prefers it
   private, the run config takes a private cross-layer import instead (precedent exists).

## Residual

What the digest does not cover, and why that is acceptable:

- **A same-size, same-`mtime_ns` rewrite of a plain file inside the process that cached it.** The digest is
  the file's bytes, but a resume check computes it many times, so it is cached per `(resolved path, size,
  mtime_ns)`; an edit that preserves both is invisible until the next process. This is the size+mtime
  pre-check the identity rules already allow, and `docs/concepts/retrieval.md` documents it.
- **A same-size, same-`mtime_ns` edit of a file inside a directory source** — invisible even across
  processes, because the directory digests the listing, not the bytes. A full content hash of a multi-GB
  image tree per resume check is what the rule avoids; `hash_media: true` hashes media at ingest for the
  index identity, and a text corpus whose bytes matter is a `jsonl:` file (hashed byte-for-byte).
- **A symlink retargeted to a file with the same size and mtime.** `Path.is_file()`/`is_dir()` follow the
  link, so a retarget to a different size or mtime moves the digest; a same-size/mtime target does not. A
  broken link yields no digest and the reader reports the missing input.
- **A reader that resolves remote content lazily.** Hub, `suite:` and `mteb:` sources keep the resolved-commit
  rule and pay no digest; a plugin reader scheme is not in the built-in local list, so it keeps the
  metadata-only rule. Remote behaviour is unchanged by design, which is the brief's requirement.
- **The store digests the URI's own location, not a local reader option.** A `Dataset` does not carry the
  options it was loaded with, so a local `qrels_uri` (labels, not the texts the judge reads) enters the run
  step identity but not the store identity. The corpus location — what the judge reads — does.

## Docs updated

- `docs/concepts/runs.md`: the resume paragraph gains the local-source sentence (a local dataset has no
  commit; it carries the digest of the files its reader reads, the same one the store carries).
- `docs/concepts/judges.md`: the store-identity bullet replaces "Local files have no such pin: editing … is
  your responsibility" with the new rule (a local source enters by the digest; an edited file is refused as
  another dataset).
- `docs/reference/api/rcp_ndcg.data.revisions.md`: regenerated (`rcp-ndcg docs api --out docs/reference/api`)
  for the two new public names and the module summary.
- `docs/concepts/retrieval.md` already documented the local content rule and its residuals (from the
  retrieval-fixes lane) and stays true; not changed.
- Greps run before the last commit: `git grep -n -i "local dataset\|local source\|local file" -- docs skills`;
  `git grep -n "content_digest\|dataset_content_digest\|dataset_local_locations" -- docs README.md skills
  examples experiments`; `git grep -n -i "resolved commit" -- docs`.

## CHANGELOG entry

Added inside the one `## 0.0.1` section:

- `### Public surface` — **A local dataset source's content digest has one home**: `rcp_ndcg.data.revisions`
  gains `dataset_content_digest(uri, options=None)` — the digest of the files a local source's reader reads
  (a file's bytes; a directory's sorted listing of names, sizes and mtimes), empty for a remote source — and
  `dataset_local_locations(uri, options=None)`, the locations a source's reader reads, listed without reading
  them. The run steps' identity and the judgement store's dataset identity both consume them, so one rule
  names a local source everywhere. The generated API page and the exported `judgement-store.v1.json` (the
  `identity` description) record them (regenerate as below).
- `### Fixed` — **A local dataset's content names it in the judgement store, not only in the run**: the run
  steps' identity already carried the digest of a local source's files (a `jsonl:` file's bytes; a `beir:` or
  `images:` directory's sorted listing), but the store's dataset identity keyed a local source by its path and
  revision only — and a local source has no revision — so editing `rows.jsonl` in place resumed the judging
  step and silently reused judgements computed over the old texts. The digest now enters the store's dataset
  identity too (one home: `rcp_ndcg.data.revisions.dataset_content_digest`): an edited file is another
  dataset, refused with `IdentityError` naming `dataset.content` (or superseded with `force=True`), never
  pooled or reused. Remote sources are unchanged: a resolved commit is the content, and no download is paid
  for the digest.

## Public surface changes

- `rcp_ndcg.data.revisions.dataset_content_digest(uri, options=None) -> dict[str, str]` and
  `rcp_ndcg.data.revisions.dataset_local_locations(uri, options=None) -> dict[str, str]` (new public
  functions); `tests/contract/snapshots/python_api.json` and `docs/reference/api/rcp_ndcg.data.revisions.md`
  regenerated.
- `schemas/judgement-store.v1.json` regenerated: the `identity` property's description records the local
  content digest (no structural change; `identity` is already an open object).
- No CLI command, flag, exit code or other schema changed. `DatasetSource.content_digest()` keeps its name
  and shape; inside the identity payload a local reader option now appears under `dataset.options.<key>`
  (was `options.<key>`), which no schema or snapshot pins.

## Files outside scope

- None. `judging/judging.py` (not `judging/store.py`) is where the store's dataset identity is built; the
  store's `StageEntry` docstring was updated beside it. `runs/config.py` and the data layer are the brief's
  "find the one home" instruction.

## For the next lanes

- A local source's content digest has exactly one home: `rcp_ndcg.data.revisions.dataset_content_digest`.
  A new built-in local reader scheme must join `_LOCAL_SCHEMES` there, or its content is not named.
- A local dataset edited in place now refuses a run's judging store (`IdentityError` naming
  `dataset.content`); the run's hint names the way out (a new run, or move the store aside). A re-judge with
  `force=True` supersedes the old records.
- The run identity payload's key for a local reader option moved to `dataset.options.<key>`; a stored
  manifest from before this lane re-keys on resume (nothing is released yet).
